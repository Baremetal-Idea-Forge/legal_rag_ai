"""TypesenseRepository — delegation + error translation (mocked ts module)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
from typesense.exceptions import ObjectNotFound, TypesenseClientError

import repositories.typesense_repository as repo_mod
from core.exceptions import (
    DocumentNotFoundError,
    IngestionError,
    SearchError,
    UpstreamUnavailableError,
)
from repositories.typesense_repository import TypesenseRepository


@pytest.fixture
def repo():
    return TypesenseRepository()


class TestEnsureCollection:
    def test_ok(self, repo):
        with patch.object(repo_mod.ts, "ensure_tasks_collection") as m:
            repo.ensure_collection()
            m.assert_called_once()

    def test_upstream_down(self, repo):
        with patch.object(repo_mod.ts, "ensure_tasks_collection",
                          side_effect=TypesenseClientError("down")):
            with pytest.raises(UpstreamUnavailableError):
                repo.ensure_collection()


class TestIsReady:
    def test_true(self, repo):
        with patch.object(repo_mod.ts, "collection_exists", return_value=True):
            assert repo.is_ready() is True

    def test_false(self, repo):
        with patch.object(repo_mod.ts, "collection_exists", return_value=False):
            assert repo.is_ready() is False

    def test_never_raises(self, repo):
        with patch.object(repo_mod.ts, "collection_exists",
                          side_effect=TypesenseClientError("boom")):
            assert repo.is_ready() is False


class TestIndexChunks:
    def test_empty_returns_empty(self, repo):
        assert repo.index_chunks([]) == []

    def test_success(self, repo):
        with patch.object(repo_mod.ts, "bulk_index_pdf_chunks",
                          return_value=[{"success": True}, {"success": True}]) as m:
            out = repo.index_chunks([{"id": "1"}, {"id": "2"}])
            assert len(out) == 2
            m.assert_called_once()

    def test_partial_failure_raises_ingestion_error(self, repo):
        with patch.object(repo_mod.ts, "bulk_index_pdf_chunks",
                          return_value=[{"success": True}, {"success": False, "error": "x"}]):
            with pytest.raises(IngestionError):
                repo.index_chunks([{"id": "1"}, {"id": "2"}])

    def test_upstream_down(self, repo):
        with patch.object(repo_mod.ts, "bulk_index_pdf_chunks",
                          side_effect=TypesenseClientError("down")):
            with pytest.raises(UpstreamUnavailableError):
                repo.index_chunks([{"id": "1"}])


class TestGetDocument:
    def test_found(self, repo):
        with patch.object(repo_mod.ts, "get_pdf_chunk_document",
                          return_value={"id": "c1"}):
            assert repo.get_document("c1")["id"] == "c1"

    def test_not_found(self, repo):
        with patch.object(repo_mod.ts, "get_pdf_chunk_document",
                          side_effect=ObjectNotFound("nope")):
            with pytest.raises(DocumentNotFoundError):
                repo.get_document("missing")

    def test_upstream_down(self, repo):
        with patch.object(repo_mod.ts, "get_pdf_chunk_document",
                          side_effect=TypesenseClientError("down")):
            with pytest.raises(UpstreamUnavailableError):
                repo.get_document("c1")


class TestSearch:
    def test_passes_params_through(self, repo):
        with patch.object(repo_mod.ts, "search_pdf_chunks",
                          return_value={"hits": []}) as m:
            repo.search("contract", mode="hybrid", top_k=7,
                        filter_by="pdf_id:=x", vector_query="embedding:([0.1], k:7)")
            kwargs = m.call_args.kwargs
            assert kwargs["mode"] == "hybrid"
            assert kwargs["per_page"] == 7
            assert kwargs["filter_by"] == "pdf_id:=x"
            assert kwargs["vector_query"] == "embedding:([0.1], k:7)"

    def test_value_error_becomes_search_error(self, repo):
        with patch.object(repo_mod.ts, "search_pdf_chunks",
                          side_effect=ValueError("bad mode")):
            with pytest.raises(SearchError):
                repo.search("q", mode="nope")

    def test_upstream_down(self, repo):
        with patch.object(repo_mod.ts, "search_pdf_chunks",
                          side_effect=TypesenseClientError("down")):
            with pytest.raises(UpstreamUnavailableError):
                repo.search("q")
