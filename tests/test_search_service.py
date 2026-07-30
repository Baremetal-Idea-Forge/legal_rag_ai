"""SearchService — query embedding, vector_query build, hit mapping."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.config import Settings, get_settings
from models.schemas import ChunkHit
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


def test_hit_maps_char_span_with_fallback():
    repo = MagicMock()
    repo.search.return_value = {
        "hits": [
            {"document": {"id": "a", "pdf_id": "p", "pdf_name": "x.pdf",
                          "page": 1, "chunk_index": 0, "content": "c",
                          "start_char": 120, "end_char": 480}},
            {"document": {"id": "b", "pdf_id": "p", "pdf_name": "x.pdf",
                          "page": 1, "chunk_index": 1, "content": "c"}},
        ]
    }
    resp = _service(repo).search("q", mode="keyword")
    assert resp.hits[0].start_char == 120 and resp.hits[0].end_char == 480
    # Pre-span documents fall back to the -1 sentinel.
    assert resp.hits[1].start_char == -1 and resp.hits[1].end_char == -1


# ===========================================================================
# Two-stage retrieval (document scoping)
# ===========================================================================

def _chunk_hit(pdf_id, score, i=0):
    return ChunkHit(
        id=f"{pdf_id}_c{i}", pdf_id=pdf_id, pdf_name=f"{pdf_id}.pdf",
        page_start=1, page_end=1, chunk_index=i, content="c", score=score,
    )


def _raw_hit(pdf_id, vector_distance, i=0):
    return {
        "document": {
            "id": f"{pdf_id}_chunk_{i}", "pdf_id": pdf_id,
            "pdf_name": f"{pdf_id}.pdf", "page": 1, "chunk_index": i,
            "content": f"content {pdf_id} {i}",
        },
        "vector_distance": vector_distance,
    }


def _two_stage_settings(**overrides) -> Settings:
    s = Settings()
    s.TWO_STAGE_ENABLED = True
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


class TestAggregateDocuments:
    def test_max_strategy_ranks_by_best_chunk(self):
        hits = [
            _chunk_hit("a", 0.9), _chunk_hit("b", 0.8, 1),
            _chunk_hit("b", 0.7, 2), _chunk_hit("b", 0.6, 3),
        ]
        scopes = SearchService._aggregate_documents(hits, strategy="max")
        assert [(s.pdf_id, s.score) for s in scopes] == [("a", 0.9), ("b", 0.8)]
        assert scopes[1].chunk_count == 3

    def test_sum_top3_rewards_breadth(self):
        # One brilliant chunk vs. three good ones: sum_top3 flips the ranking.
        hits = [
            _chunk_hit("a", 0.9), _chunk_hit("b", 0.8, 1),
            _chunk_hit("b", 0.7, 2), _chunk_hit("b", 0.6, 3),
            _chunk_hit("b", 0.5, 4),  # 4th-best is excluded from the sum
        ]
        scopes = SearchService._aggregate_documents(hits, strategy="sum_top3")
        assert scopes[0].pdf_id == "b"
        assert scopes[0].score == 0.8 + 0.7 + 0.6

    def test_scoreless_hits_are_skipped(self):
        hits = [_chunk_hit("a", None), _chunk_hit("b", 0.5, 1)]
        scopes = SearchService._aggregate_documents(hits)
        assert [s.pdf_id for s in scopes] == ["b"]

    def test_empty(self):
        assert SearchService._aggregate_documents([]) == []


def test_scope_filter_format():
    assert SearchService._scope_filter(["a", "b"]) == "pdf_id:=[a,b]"


class TestSearchTwoStage:
    def test_scopes_stage2_to_top_documents(self):
        repo = MagicMock()
        stage1 = {"hits": [_raw_hit("a", 0.1), _raw_hit("b", 0.4, 1),
                           _raw_hit("c", 0.2, 2)]}
        stage2 = {"hits": [_raw_hit("a", 0.1), _raw_hit("c", 0.2, 2)]}
        repo.search.side_effect = [stage1, stage2]

        svc = SearchService(
            embedding_service=FakeEmbeddings(), typesense_repo=repo,
            settings=_two_stage_settings(TWO_STAGE_DOC_COUNT=2),
        )
        resp, scopes = svc.search_two_stage("q", top_k=5)

        assert repo.search.call_count == 2
        # Stage 1 casts the wide candidate net…
        assert repo.search.call_args_list[0].kwargs["top_k"] == \
            Settings().TWO_STAGE_CANDIDATE_K
        # …stage 2 is restricted to the winning documents (a: 0.9, c: 0.8).
        assert repo.search.call_args_list[1].kwargs["filter_by"] == "pdf_id:=[a,c]"
        assert repo.search.call_args_list[1].kwargs["top_k"] == 5
        assert [s.pdf_id for s in scopes] == ["a", "c", "b"]
        assert resp.count == 2

    def test_existing_filter_is_combined_with_scope(self):
        repo = MagicMock()
        repo.search.side_effect = [
            {"hits": [_raw_hit("a", 0.1)]}, {"hits": []},
        ]
        svc = SearchService(
            embedding_service=FakeEmbeddings(), typesense_repo=repo,
            settings=_two_stage_settings(),
        )
        svc.search_two_stage("q", filter_by="page:>3")
        combined = repo.search.call_args_list[1].kwargs["filter_by"]
        assert combined == "(page:>3) && pdf_id:=[a]"

    def test_empty_stage1_returns_no_scopes_and_no_second_search(self):
        repo = MagicMock()
        repo.search.return_value = {"hits": []}
        svc = SearchService(
            embedding_service=FakeEmbeddings(), typesense_repo=repo,
            settings=_two_stage_settings(),
        )
        resp, scopes = svc.search_two_stage("q")
        assert resp.hits == [] and scopes == []
        assert repo.search.call_count == 1
