"""RagService — retrieval+generation orchestration (fakes for search + Ollama)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.config import Settings, get_settings
from models.schemas import ChunkHit, SearchResponse
from services import prompts
from services.rag_service import RagService


# --- Fakes -----------------------------------------------------------------

class FakeSearch:
    def __init__(self, hits):
        self._hits = hits
        self.calls = 0

    def search(self, query, *, top_k=5, mode="hybrid", filter_by=None):
        self.calls += 1
        return SearchResponse(query=query, mode=mode, count=len(self._hits), hits=self._hits)


class FakeOllama:
    def __init__(self, answer="Grounded answer.", tokens=None):
        self._answer = answer
        self._tokens = tokens or ["Grounded ", "answer."]
        self.chat_calls = 0

    async def chat(self, messages, *, temperature=0.2):
        self.chat_calls += 1
        self._last_messages = messages
        self.last_temperature = temperature
        return self._answer

    async def stream_chat(self, messages, *, temperature=0.2):
        self.last_temperature = temperature
        for t in self._tokens:
            yield t


def _hit(i=0, content="legal content", pages=(1, 1)):
    return ChunkHit(
        id=f"c{i}", pdf_id=f"p{i}", pdf_name=f"doc{i}.pdf",
        page_start=pages[0], page_end=pages[1], chunk_index=i, content=content,
        file_url=f"/pdfs/p{i}/doc{i}.pdf",
    )


def _service(hits, ollama=None):
    search = FakeSearch(hits)
    ollama = ollama or FakeOllama()
    svc = RagService(
        search_service=search, llm_client=ollama, settings=get_settings()
    )
    return svc, search, ollama


# --- answer ----------------------------------------------------------------

def test_answer_with_hits_returns_cited_response():
    hits = [_hit(0), _hit(1)]
    svc, search, ollama = _service(hits)
    resp = asyncio.run(svc.answer("What is negligence?"))

    assert resp.answer == "Grounded answer."
    assert len(resp.citations) == 2
    assert len(resp.chunks_used) == 2
    assert resp.citations[0].pdf_name == "doc0.pdf"
    assert ollama.chat_calls == 1
    assert search.calls == 1


def test_answer_no_hits_declines_without_calling_llm():
    svc, search, ollama = _service([])
    resp = asyncio.run(svc.answer("Unknown topic?"))

    assert resp.answer == prompts.NO_CONTEXT_ANSWER
    assert resp.citations == []
    assert resp.chunks_used == []
    assert ollama.chat_calls == 0  # no hallucination path


def test_answer_forwards_configured_temperature():
    # Deterministic answers: RagService must pass settings.LLM_TEMPERATURE
    # (not the client's own 0.2 default) to the LLM.
    settings = Settings()
    settings.LLM_TEMPERATURE = 0.0
    ollama = FakeOllama()
    svc = RagService(
        search_service=FakeSearch([_hit(0)]), llm_client=ollama, settings=settings
    )
    asyncio.run(svc.answer("q"))
    assert ollama.last_temperature == 0.0


def test_stream_forwards_configured_temperature():
    settings = Settings()
    settings.LLM_TEMPERATURE = 0.0
    ollama = FakeOllama()
    svc = RagService(
        search_service=FakeSearch([_hit(0)]), llm_client=ollama, settings=settings
    )
    _collect(svc.stream_answer("q"))
    assert ollama.last_temperature == 0.0


def test_answer_passes_context_to_llm():
    hits = [_hit(0, content="duty of care principle")]
    svc, _, ollama = _service(hits)
    asyncio.run(svc.answer("q"))
    system, user = ollama._last_messages
    assert "duty of care principle" in user["content"]


def test_answer_context_truncated_to_budget():
    # Many oversized chunks; only those fitting RAG_MAX_CONTEXT_CHARS are cited.
    big = "x" * 5000
    hits = [_hit(i, content=big) for i in range(10)]
    svc, _, _ = _service(hits)
    resp = asyncio.run(svc.answer("q"))
    assert len(resp.chunks_used) < 10
    assert len(resp.citations) == len(resp.chunks_used)


# --- stream_answer ---------------------------------------------------------

def _collect(aiter):
    async def run():
        return [e async for e in aiter]
    return asyncio.run(run())


def test_stream_yields_tokens_then_citations():
    hits = [_hit(0)]
    svc, _, _ = _service(hits, ollama=FakeOllama(tokens=["Hello", " there"]))
    events = _collect(svc.stream_answer("q"))

    token_events = [e for e in events if e["type"] == "token"]
    citation_events = [e for e in events if e["type"] == "citations"]
    assert [e["data"] for e in token_events] == ["Hello", " there"]
    assert len(citation_events) == 1
    assert citation_events[-1]["data"][0]["pdf_name"] == "doc0.pdf"


def test_stream_no_hits_emits_decline_and_empty_citations():
    svc, _, ollama = _service([])
    events = _collect(svc.stream_answer("q"))
    tokens = [e["data"] for e in events if e["type"] == "token"]
    citations = [e for e in events if e["type"] == "citations"][0]
    assert tokens == [prompts.NO_CONTEXT_ANSWER]
    assert citations["data"] == []


# --- RETRIEVAL_VIA_MCP flag ------------------------------------------------

class FakeMcp:
    def __init__(self, hits):
        self._hits = hits
        self.calls = 0

    async def search(self, query, *, top_k=5, mode="hybrid", filter_by=None):
        self.calls += 1
        return self._hits


def _settings_with_mcp(enabled: bool) -> Settings:
    s = Settings()
    s.RETRIEVAL_VIA_MCP = enabled
    return s


def test_retrieval_via_mcp_uses_mcp_client_not_search():
    hits = [_hit(0)]
    search = FakeSearch([])          # would yield nothing if (wrongly) used
    mcp = FakeMcp(hits)
    svc = RagService(
        search_service=search, llm_client=FakeOllama(),
        settings=_settings_with_mcp(True), mcp_client=mcp,
    )
    resp = asyncio.run(svc.answer("q"))
    assert mcp.calls == 1
    assert search.calls == 0
    assert len(resp.chunks_used) == 1


def test_retrieval_direct_when_flag_disabled():
    hits = [_hit(0)]
    search = FakeSearch(hits)
    mcp = FakeMcp([])
    svc = RagService(
        search_service=search, llm_client=FakeOllama(),
        settings=_settings_with_mcp(False), mcp_client=mcp,
    )
    resp = asyncio.run(svc.answer("q"))
    assert search.calls == 1
    assert mcp.calls == 0
    assert len(resp.chunks_used) == 1


def test_flag_enabled_but_no_mcp_client_falls_back_to_search():
    hits = [_hit(0)]
    search = FakeSearch(hits)
    svc = RagService(
        search_service=search, llm_client=FakeOllama(),
        settings=_settings_with_mcp(True), mcp_client=None,
    )
    resp = asyncio.run(svc.answer("q"))
    assert search.calls == 1
    assert len(resp.chunks_used) == 1
