"""EmbeddingService — lazy load, delegation, error wrapping (mocked helper)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest

from core.exceptions import IngestionError, SearchError
from services.embedding_service import EmbeddingService


class FakeHelper:
    embedding_dimension = 4

    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_pdf_chunks(self, docs, text_key="content"):
        return [{**d, "embedding": [0.1, 0.2, 0.3, 0.4]} for d in docs]


def test_embed_query_returns_vector():
    assert EmbeddingService(helper=FakeHelper()).embed_query("contract") == [0.1, 0.2, 0.3, 0.4]


def test_dimension():
    assert EmbeddingService(helper=FakeHelper()).dimension == 4


def test_embed_documents():
    out = EmbeddingService(helper=FakeHelper()).embed_documents(["a", "b"])
    assert len(out) == 2 and len(out[0]) == 4


def test_embed_chunk_documents_adds_embedding():
    out = EmbeddingService(helper=FakeHelper()).embed_chunk_documents([{"content": "x"}])
    assert out[0]["embedding"] == [0.1, 0.2, 0.3, 0.4]


def test_lazy_factory_invoked_once():
    calls = []

    def factory():
        calls.append(1)
        return FakeHelper()

    svc = EmbeddingService(helper_factory=factory)
    assert not calls  # not loaded until first use
    svc.embed_query("a")
    svc.embed_query("b")
    assert len(calls) == 1


def test_embed_query_error_wrapped_as_search_error():
    class Boom:
        embedding_dimension = 4

        def embed_query(self, t):
            raise RuntimeError("model boom")

    with pytest.raises(SearchError):
        EmbeddingService(helper=Boom()).embed_query("q")


def test_embed_documents_error_wrapped_as_ingestion_error():
    class Boom:
        def embed_documents(self, ts):
            raise RuntimeError("model boom")

    with pytest.raises(IngestionError):
        EmbeddingService(helper=Boom()).embed_documents(["a"])


def test_embed_chunk_documents_error_wrapped_as_ingestion_error():
    class Boom:
        def embed_pdf_chunks(self, docs, text_key="content"):
            raise RuntimeError("model boom")

    with pytest.raises(IngestionError):
        EmbeddingService(helper=Boom()).embed_chunk_documents([{"content": "x"}])
