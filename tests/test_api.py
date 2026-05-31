"""End-to-end router tests via TestClient with dependency overrides.

No Typesense, no embedding model — the service layer is replaced with fakes so
these tests exercise routing, validation, status codes, and error mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
from fastapi.testclient import TestClient

import main
from core.config import Settings, get_settings
from core.dependencies import (
    get_ingestion_service,
    get_ollama_client,
    get_rag_service,
    get_search_service,
    get_typesense_repository,
)
from models.schemas import (
    BulkIngestResponse,
    ChatResponse,
    ChunkHit,
    Citation,
    IngestResponse,
    SearchResponse,
)


@pytest.fixture
def client():
    # Plain TestClient (no `with`) → lifespan not run → no Typesense connection.
    c = TestClient(main.app, raise_server_exceptions=False)
    yield c
    main.app.dependency_overrides.clear()


# --- Fakes -----------------------------------------------------------------

class FakeIngestionService:
    def ingest_bytes(self, *, pdf_bytes, filename):
        return IngestResponse(
            pdf_id="sha123", filename=filename, sha256="sha123",
            num_pages=2, num_chunks=5, status="indexed",
        )

    def ingest_directory(self, directory):
        return (
            [IngestResponse(pdf_id="a", filename="a.pdf", sha256="a",
                            num_pages=1, num_chunks=1)],
            [],
        )


class FakeSearchService:
    def search(self, query, *, top_k=5, mode="hybrid", filter_by=None):
        return SearchResponse(
            query=query, mode=mode, count=1,
            hits=[ChunkHit(id="a_chunk_0", pdf_id="a", pdf_name="x.pdf",
                           page_start=1, page_end=1, chunk_index=0,
                           content="hit", score=0.9)],
        )


class FakeReadyRepo:
    def __init__(self, ready):
        self._ready = ready

    def is_ready(self):
        return self._ready


class FakePingOllama:
    def __init__(self, ok):
        self._ok = ok

    async def ping(self):
        return self._ok


class FakeRagService:
    async def answer(self, query, *, top_k=5):
        return ChatResponse(
            query=query,
            answer="Grounded answer about contracts.",
            citations=[Citation(pdf_id="a", pdf_name="contract.pdf",
                                page_start=2, page_end=2, chunk_index=0)],
            chunks_used=[ChunkHit(id="a_chunk_0", pdf_id="a", pdf_name="contract.pdf",
                                  page_start=2, page_end=2, chunk_index=0, content="...")],
        )

    async def stream_answer(self, query, *, top_k=5):
        for tok in ["Grounded ", "answer."]:
            yield {"type": "token", "data": tok}
        yield {"type": "citations", "data": [{"pdf_name": "contract.pdf"}]}


class FailingRagService:
    """Streams one token then fails — exercises the SSE error path (B3)."""

    async def stream_answer(self, query, *, top_k=5):
        from core.exceptions import LLMError

        yield {"type": "token", "data": "partial "}
        raise LLMError("LLM died mid-stream.")


class CountingIngestionService:
    """Records whether ingest_bytes was reached (to prove early rejection)."""

    def __init__(self):
        self.called = False

    def ingest_bytes(self, *, pdf_bytes, filename):
        self.called = True
        return IngestResponse(
            pdf_id="x", filename=filename, sha256="x", num_pages=1, num_chunks=1
        )


# --- Health ----------------------------------------------------------------

def test_health_liveness(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness_ok(client):
    main.app.dependency_overrides[get_typesense_repository] = lambda: FakeReadyRepo(True)
    main.app.dependency_overrides[get_ollama_client] = lambda: FakePingOllama(True)
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["checks"] == {"typesense": True, "ollama": True}


def test_readiness_degraded_when_typesense_down(client):
    main.app.dependency_overrides[get_typesense_repository] = lambda: FakeReadyRepo(False)
    main.app.dependency_overrides[get_ollama_client] = lambda: FakePingOllama(True)
    r = client.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"
    assert r.json()["checks"]["typesense"] is False


def test_readiness_degraded_when_ollama_down(client):
    main.app.dependency_overrides[get_typesense_repository] = lambda: FakeReadyRepo(True)
    main.app.dependency_overrides[get_ollama_client] = lambda: FakePingOllama(False)
    r = client.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["ollama"] is False


# --- Ingest ----------------------------------------------------------------

def test_ingest_success(client):
    main.app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService()
    r = client.post(
        "/ingest",
        files={"file": ("contract.pdf", b"%PDF-1.4 ...", "application/pdf")},
    )
    assert r.status_code == 201
    body = r.json()
    assert body["num_chunks"] == 5
    assert body["filename"] == "contract.pdf"


def test_ingest_rejects_non_pdf(client):
    main.app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService()
    r = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 422
    assert r.json()["error"] == "ingestion_error"


def test_ingest_rejects_oversize(client):
    spy = CountingIngestionService()
    main.app.dependency_overrides[get_ingestion_service] = lambda: spy

    def tiny_settings() -> Settings:
        s = Settings()
        s.MAX_UPLOAD_BYTES = 4
        return s

    main.app.dependency_overrides[get_settings] = tiny_settings
    r = client.post(
        "/ingest",
        files={"file": ("big.pdf", b"way more than four bytes", "application/pdf")},
    )
    assert r.status_code == 422
    assert r.json()["error"] == "ingestion_error"
    # B4: rejected before the body reached the ingestion service.
    assert spy.called is False


def test_ingest_bulk(client):
    main.app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService()
    r = client.post("/ingest/bulk")
    assert r.status_code == 200
    body = r.json()
    assert body["total_ingested"] == 1
    assert body["total_failed"] == 0


# --- Search ----------------------------------------------------------------

def test_search_post(client):
    main.app.dependency_overrides[get_search_service] = lambda: FakeSearchService()
    r = client.post("/search", json={"query": "breach of contract", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["hits"][0]["pdf_name"] == "x.pdf"


def test_search_get(client):
    main.app.dependency_overrides[get_search_service] = lambda: FakeSearchService()
    r = client.get("/search", params={"q": "duty of care", "top_k": 3, "mode": "hybrid"})
    assert r.status_code == 200
    assert r.json()["query"] == "duty of care"


def test_search_empty_query_rejected(client):
    main.app.dependency_overrides[get_search_service] = lambda: FakeSearchService()
    r = client.post("/search", json={"query": "", "top_k": 5})
    assert r.status_code == 422  # pydantic validation


def test_search_top_k_out_of_range_rejected(client):
    main.app.dependency_overrides[get_search_service] = lambda: FakeSearchService()
    r = client.post("/search", json={"query": "x", "top_k": 999})
    assert r.status_code == 422


# --- Chat ------------------------------------------------------------------

def test_chat_returns_cited_answer(client):
    main.app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
    r = client.post("/chat", json={"query": "What is a contract?", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Grounded answer about contracts."
    assert body["citations"][0]["pdf_name"] == "contract.pdf"
    assert body["chunks_used"][0]["chunk_index"] == 0


def test_chat_validation_rejects_empty_query(client):
    main.app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
    r = client.post("/chat", json={"query": "", "top_k": 5})
    assert r.status_code == 422


def test_chat_stream_emits_sse_events(client):
    main.app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
    r = client.post("/chat/stream", json={"query": "What is a contract?", "top_k": 5})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "event: token" in body
    assert "event: citations" in body
    assert "event: done" in body


def test_chat_stream_emits_error_event_on_midstream_failure(client):
    """B3 regression: a failure after streaming starts → terminal error+done,
    never a silently truncated stream."""
    main.app.dependency_overrides[get_rag_service] = lambda: FailingRagService()
    r = client.post("/chat/stream", json={"query": "boom", "top_k": 5})
    assert r.status_code == 200
    body = r.text
    assert "event: token" in body          # the partial token made it out
    assert "event: error" in body          # failure surfaced as an event
    assert "llm_error" in body             # carries the domain error code
    assert "event: done" in body           # stream still terminates cleanly


def test_chat_stream_unexpected_error_event(client):
    """Non-domain (unexpected) errors during streaming → generic error event."""

    class Boom:
        async def stream_answer(self, query, *, top_k=5):
            yield {"type": "token", "data": "x"}
            raise RuntimeError("not a domain error")

    main.app.dependency_overrides[get_rag_service] = lambda: Boom()
    r = client.post("/chat/stream", json={"query": "x", "top_k": 5})
    body = r.text
    assert "event: error" in body
    assert "internal_error" in body
    assert "event: done" in body


def test_ingest_fallback_size_check_when_size_unknown():
    """B4: when file.size is unknown (chunked upload), the post-read length
    check still enforces the limit before the service is invoked."""
    import asyncio

    from routers.ingest import ingest_pdf

    class NoSizeUpload:
        content_type = "application/pdf"
        size = None
        filename = "x.pdf"

        async def read(self):
            return b"x" * 100

    s = Settings()
    s.MAX_UPLOAD_BYTES = 10
    spy = CountingIngestionService()

    with pytest.raises(Exception) as exc_info:
        asyncio.run(ingest_pdf(file=NoSizeUpload(), settings=s, service=spy))
    assert "exceeds" in str(exc_info.value)
    assert spy.called is False


# --- Auth seam on the write path -------------------------------------------

def _auth_settings():
    s = Settings()
    s.API_AUTH_TOKEN = "secret"
    return s


def test_ingest_requires_key_when_auth_enabled(client):
    main.app.dependency_overrides[get_settings] = _auth_settings
    main.app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService()
    r = client.post(
        "/ingest", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}
    )
    assert r.status_code == 401
    assert r.json()["error"] == "unauthorized"


def test_ingest_accepts_valid_api_key(client):
    main.app.dependency_overrides[get_settings] = _auth_settings
    main.app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService()
    r = client.post(
        "/ingest",
        files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")},
        headers={"X-API-Key": "secret"},
    )
    assert r.status_code == 201


def test_ingest_accepts_bearer_token(client):
    main.app.dependency_overrides[get_settings] = _auth_settings
    main.app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService()
    r = client.post(
        "/ingest",
        files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")},
        headers={"Authorization": "Bearer secret"},
    )
    assert r.status_code == 201


def test_search_open_when_auth_enabled(client):
    # Auth guards the write path only; read stays open.
    main.app.dependency_overrides[get_settings] = _auth_settings
    main.app.dependency_overrides[get_search_service] = lambda: FakeSearchService()
    r = client.post("/search", json={"query": "contract", "top_k": 5})
    assert r.status_code == 200
