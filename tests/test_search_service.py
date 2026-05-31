"""SearchService — query embedding, vector_query build, hit mapping."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.config import get_settings
from services.search_service import SearchService


class FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        return [0.1, 0.2, 0.3, 0.4]


def _raw_response():
    return {
        "found": 2,
        "hits": [
            {
                "document": {
                    "id": "a_chunk_0", "pdf_id": "a", "pdf_name": "contract.pdf",
                    "page": 3, "page_start": 3, "page_end": 4, "chunk_index": 0,
                    "content": "offer and acceptance", "file_url": "/pdfs/a/contract.pdf",
                },
                "text_match": 100, "vector_distance": 0.2,
            },
            {
                "document": {
                    "id": "b_chunk_5", "pdf_id": "b", "pdf_name": "tort.pdf",
                    "page": 9, "chunk_index": 5, "content": "duty of care",
                },
                "text_match": 50,
            },
        ],
    }


def _service(repo):
    return SearchService(
        embedding_service=FakeEmbeddings(),
        typesense_repo=repo,
        settings=get_settings(),
    )


def test_hybrid_search_maps_hits():
    repo = MagicMock()
    repo.search.return_value = _raw_response()
    resp = _service(repo).search("breach", mode="hybrid", top_k=5)

    assert resp.count == 2
    assert resp.mode == "hybrid"
    h0 = resp.hits[0]
    assert h0.id == "a_chunk_0"
    assert h0.page_start == 3 and h0.page_end == 4
    assert h0.score == round(1.0 - 0.2, 6)
    assert h0.file_url == "/pdfs/a/contract.pdf"


def test_hit_without_vector_distance_uses_text_match_score():
    repo = MagicMock()
    repo.search.return_value = _raw_response()
    resp = _service(repo).search("breach", mode="hybrid")
    assert resp.hits[1].score == 50.0
    # page_start/page_end fall back to `page`
    assert resp.hits[1].page_start == 9 and resp.hits[1].page_end == 9


def test_hybrid_builds_vector_query():
    repo = MagicMock()
    repo.search.return_value = {"hits": []}
    _service(repo).search("breach", mode="hybrid", top_k=7)
    vq = repo.search.call_args.kwargs["vector_query"]
    assert vq.startswith("embedding:([")
    assert "k:7" in vq


def test_keyword_mode_skips_embedding_and_vector_query():
    repo = MagicMock()
    repo.search.return_value = {"hits": []}
    embeddings = FakeEmbeddings()
    svc = SearchService(
        embedding_service=embeddings, typesense_repo=repo, settings=get_settings()
    )
    svc.search("breach", mode="keyword")
    assert embeddings.calls == 0
    assert repo.search.call_args.kwargs["vector_query"] is None


def test_vector_mode_embeds_query():
    repo = MagicMock()
    repo.search.return_value = {"hits": []}
    embeddings = FakeEmbeddings()
    svc = SearchService(
        embedding_service=embeddings, typesense_repo=repo, settings=get_settings()
    )
    svc.search("breach", mode="vector")
    assert embeddings.calls == 1


def test_top_k_defaults_to_settings():
    repo = MagicMock()
    repo.search.return_value = {"hits": []}
    _service(repo).search("breach", mode="keyword")
    assert repo.search.call_args.kwargs["top_k"] == get_settings().RAG_TOP_K


def test_empty_hits():
    repo = MagicMock()
    repo.search.return_value = {"hits": []}
    resp = _service(repo).search("nothing", mode="hybrid")
    assert resp.count == 0 and resp.hits == []


def test_hit_without_any_score_has_none_score():
    # Neither text_match nor vector_distance present → score is None.
    repo = MagicMock()
    repo.search.return_value = {
        "hits": [
            {"document": {"id": "a", "pdf_id": "p", "pdf_name": "x.pdf",
                          "page": 2, "chunk_index": 0, "content": "c"}}
        ]
    }
    resp = _service(repo).search("q", mode="keyword")
    assert resp.hits[0].score is None
    assert resp.hits[0].page_start == 2 and resp.hits[0].page_end == 2
