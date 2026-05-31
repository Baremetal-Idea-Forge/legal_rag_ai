"""
Comprehensive test suite for backend/models/embeddings_model.py.

All tests mock SentenceTransformer to avoid downloading the ~1 GB model.
Integration tests verify the full pipeline from real PDFs through the helper
using a mocked encoder.

Covers:
  - _clean_text
  - _encode_texts: happy path, 1D fix (B2), custom encode_query fix (B1), empty list
  - embed_query / embed_document: result type, shape, delegation
  - embed_documents: batching, empty list, all-empty texts
  - embed_pdf_chunks: full pipeline, missing key warning (B5), custom text_key
  - get_embedding_helper: factory, lru_cache maxsize (B4)
  - Integration: real PDFs → build docs → embed_pdf_chunks → check embedding field
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from models.embeddings_model import (
    LegalBGEEmbeddingHelper,
    _STANDARD_ENCODE_KWARGS,
    get_embedding_helper,
)

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

DIM = 8  # small dimension for fast tests


def _make_2d(n: int = 1, dim: int = DIM) -> np.ndarray:
    return np.random.rand(n, dim).astype(np.float32)


def _make_1d(dim: int = DIM) -> np.ndarray:
    return np.random.rand(dim).astype(np.float32)


def _build_helper(
    encode_return=None,
    *,
    dim: int = DIM,
    has_encode_query: bool = False,
    encode_query_fn=None,
    normalize: bool = True,
) -> tuple[LegalBGEEmbeddingHelper, MagicMock]:
    """
    Create a LegalBGEEmbeddingHelper with a mocked SentenceTransformer.

    encode_return: what mock.encode() returns (default: zeros 2D)
    has_encode_query: if True, adds encode_query / encode_document to the mock
    encode_query_fn: custom callable to use as encode_query (implies has_encode_query)
    """
    if encode_return is None:
        encode_return = _make_2d(1, dim)

    mock_model = MagicMock()
    mock_model.get_embedding_dimension.return_value = dim
    mock_model.encode.return_value = encode_return

    # By default, remove custom encode_query/document so hasattr returns False
    if not has_encode_query and encode_query_fn is None:
        del mock_model.encode_query
        del mock_model.encode_document
    elif encode_query_fn is not None:
        mock_model.encode_query = encode_query_fn
        mock_model.encode_document = encode_query_fn
    else:
        mock_model.encode_query.return_value = encode_return
        mock_model.encode_document.return_value = encode_return

    with patch("models.embeddings_model.SentenceTransformer", return_value=mock_model):
        helper = LegalBGEEmbeddingHelper("fake/model", normalize_embeddings=normalize)

    return helper, mock_model


# ===========================================================================
# 1. _clean_text
# ===========================================================================

class TestCleanText:
    def test_collapses_multiple_spaces(self):
        assert LegalBGEEmbeddingHelper._clean_text("a   b  c") == "a b c"

    def test_strips_leading_trailing(self):
        assert LegalBGEEmbeddingHelper._clean_text("  hello  ") == "hello"

    def test_replaces_newlines_with_space(self):
        result = LegalBGEEmbeddingHelper._clean_text("line1\nline2")
        assert "\n" not in result
        assert "line1" in result and "line2" in result

    def test_tabs_collapsed(self):
        assert LegalBGEEmbeddingHelper._clean_text("a\tb") == "a b"

    def test_empty_string_returns_empty(self):
        assert LegalBGEEmbeddingHelper._clean_text("") == ""

    def test_none_input_returns_empty(self):
        assert LegalBGEEmbeddingHelper._clean_text(None) == ""  # type: ignore[arg-type]

    def test_whitespace_only_returns_empty(self):
        assert LegalBGEEmbeddingHelper._clean_text("   \t\n  ") == ""

    def test_normal_text_unchanged(self):
        assert LegalBGEEmbeddingHelper._clean_text("hello world") == "hello world"


# ===========================================================================
# 2. _encode_texts
# ===========================================================================

class TestEncodeTexts:
    def test_empty_list_returns_empty(self):
        h, _ = _build_helper()
        assert h._encode_texts([], task="document") == []

    def test_returns_list_of_lists(self):
        h, _ = _build_helper(_make_2d(2))
        result = h._encode_texts(["a", "b"], task="document")
        assert isinstance(result, list)
        assert all(isinstance(v, list) for v in result)

    def test_correct_count_returned(self):
        n = 5
        h, _ = _build_helper(_make_2d(n))
        result = h._encode_texts(["text"] * n, task="document")
        assert len(result) == n

    def test_inner_vectors_are_floats(self):
        h, _ = _build_helper(_make_2d(1))
        result = h._encode_texts(["test"], task="document")
        assert all(isinstance(f, float) for f in result[0])

    def test_inner_vector_dimension_matches_model(self):
        h, _ = _build_helper(_make_2d(1, DIM))
        result = h._encode_texts(["test"], task="document")
        assert len(result[0]) == DIM

    # FIX B2 regression: 1D array must be reshaped to 2D
    def test_1d_ndarray_reshaped_to_2d(self):
        h, _ = _build_helper(_make_1d(DIM))
        result = h._encode_texts(["test"], task="document")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == DIM

    def test_1d_array_embed_query_returns_list_not_float(self):
        """B2 regression: embed_query must never return a float."""
        h, _ = _build_helper(_make_1d(DIM))
        result = h.embed_query("legal text")
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) == DIM

    # FIX B1 regression: custom encode_query with strict signature
    def test_strict_encode_query_no_extra_kwargs(self):
        """B1 regression: TypeError must not be raised when encode_query has a minimal signature."""
        def strict_eq(sentences):  # accepts NO extra kwargs
            return _make_2d(len(sentences), DIM)

        h, _ = _build_helper(encode_query_fn=strict_eq)
        result = h.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == DIM

    def test_encode_query_with_var_kwargs_gets_all_params(self):
        """encode_query accepting **kwargs should receive all standard kwargs."""
        captured: dict = {}

        def flexible_eq(sentences, **kwargs):
            captured.update(kwargs)
            return _make_2d(len(sentences), DIM)

        h, _ = _build_helper(encode_query_fn=flexible_eq)
        h.embed_query("test")
        assert "batch_size" in captured
        assert "normalize_embeddings" in captured

    def test_fallback_to_encode_when_no_custom_method(self):
        h, mock = _build_helper(_make_2d(1))
        h._encode_texts(["test"], task="query")
        mock.encode.assert_called_once()

    def test_uses_encode_query_for_query_task(self):
        h, mock = _build_helper(_make_2d(1), has_encode_query=True)
        h._encode_texts(["test"], task="query")
        mock.encode_query.assert_called_once()
        mock.encode.assert_not_called()

    def test_uses_encode_document_for_document_task(self):
        h, mock = _build_helper(_make_2d(1), has_encode_query=True)
        h._encode_texts(["test"], task="document")
        mock.encode_document.assert_called_once()
        mock.encode.assert_not_called()

    def test_normalize_embeddings_forwarded_to_encode(self):
        h, mock = _build_helper(_make_2d(1), normalize=False)
        h._encode_texts(["test"], task="document")
        call_kwargs = mock.encode.call_args[1]
        assert call_kwargs["normalize_embeddings"] is False

    def test_batch_size_forwarded(self):
        h, mock = _build_helper(_make_2d(1))
        h._encode_texts(["test"], task="document", batch_size=64)
        assert mock.encode.call_args[1]["batch_size"] == 64

    def test_non_ndarray_fallback_path(self):
        """When encode returns a list of arrays (non-ndarray), the else branch handles it."""
        rows = [np.array([0.1] * DIM) for _ in range(2)]
        h, mock = _build_helper(rows)  # returns list, not ndarray
        result = h._encode_texts(["a", "b"], task="document")
        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)

    def test_uninspectable_encode_query_falls_back_to_empty_kwargs(self):
        """
        B1 edge case (lines 104-106): if inspect.signature raises ValueError/TypeError
        (e.g. the callable is a C built-in), _encode_texts falls back to passing NO
        extra kwargs and just calls encode_query(texts).
        """
        import inspect as _inspect

        call_log: list = []

        def inspectable_eq(sentences):
            # This function IS inspectable (no **kwargs) so normally only
            # the listed params would be forwarded.  Here we patch
            # inspect.signature to simulate an un-inspectable callable.
            call_log.append(sentences)
            return _make_2d(len(sentences), DIM)

        # Make inspect.signature raise ValueError for this specific callable
        original_sig = _inspect.signature

        def patched_sig(obj, **kw):
            if obj is inspectable_eq:
                raise ValueError("simulated uninspectable function")
            return original_sig(obj, **kw)

        h, _ = _build_helper(encode_query_fn=inspectable_eq)
        with patch("models.embeddings_model.inspect.signature", side_effect=patched_sig):
            result = h._encode_texts(["test"], task="query")

        # Must succeed and return a valid embedding despite the inspect failure
        assert len(call_log) == 1
        assert isinstance(result, list)
        assert isinstance(result[0], list)


# ===========================================================================
# 3. embed_query
# ===========================================================================

class TestEmbedQuery:
    def test_returns_list_of_floats(self):
        h, _ = _build_helper(_make_2d(1))
        result = h.embed_query("legal contract")
        assert isinstance(result, list)
        assert all(isinstance(f, float) for f in result)

    def test_length_equals_embedding_dimension(self):
        h, _ = _build_helper(_make_2d(1, DIM))
        assert len(h.embed_query("test")) == DIM

    def test_cleans_text_before_encoding(self):
        h, mock = _build_helper(_make_2d(1))
        h.embed_query("  hello   world  ")
        encoded_text = mock.encode.call_args[0][0][0]
        assert encoded_text == "hello world"

    def test_empty_string_does_not_crash(self):
        h, _ = _build_helper(_make_2d(1))
        result = h.embed_query("")
        assert isinstance(result, list)

    def test_delegates_to_encode_texts_with_query_task(self):
        h, mock = _build_helper(_make_2d(1), has_encode_query=True)
        h.embed_query("test")
        mock.encode_query.assert_called_once()


# ===========================================================================
# 4. embed_document
# ===========================================================================

class TestEmbedDocument:
    def test_returns_list_of_floats(self):
        h, _ = _build_helper(_make_2d(1))
        result = h.embed_document("legal provision")
        assert isinstance(result, list)
        assert all(isinstance(f, float) for f in result)

    def test_length_equals_embedding_dimension(self):
        h, _ = _build_helper(_make_2d(1, DIM))
        assert len(h.embed_document("test")) == DIM

    def test_cleans_text_before_encoding(self):
        h, mock = _build_helper(_make_2d(1))
        h.embed_document("  contract\nclause  ")
        encoded_text = mock.encode.call_args[0][0][0]
        assert "\n" not in encoded_text

    def test_delegates_to_encode_texts_with_document_task(self):
        h, mock = _build_helper(_make_2d(1), has_encode_query=True)
        h.embed_document("test")
        mock.encode_document.assert_called_once()


# ===========================================================================
# 5. embed_documents
# ===========================================================================

class TestEmbedDocuments:
    def test_empty_list_returns_empty(self):
        h, _ = _build_helper()
        assert h.embed_documents([]) == []

    def test_returns_one_vector_per_text(self):
        n = 4
        h, _ = _build_helper(_make_2d(n))
        result = h.embed_documents(["text"] * n)
        assert len(result) == n

    def test_each_vector_is_list_of_floats(self):
        h, _ = _build_helper(_make_2d(3))
        for vec in h.embed_documents(["a", "b", "c"]):
            assert isinstance(vec, list)
            assert all(isinstance(f, float) for f in vec)

    def test_cleans_all_texts(self):
        h, mock = _build_helper(_make_2d(2))
        h.embed_documents(["  a  b  ", "c\nd"])
        texts_sent = mock.encode.call_args[0][0]
        assert texts_sent[0] == "a b"
        assert "\n" not in texts_sent[1]

    def test_batch_size_forwarded(self):
        h, mock = _build_helper(_make_2d(1))
        h.embed_documents(["test"], batch_size=16)
        assert mock.encode.call_args[1]["batch_size"] == 16

    def test_all_whitespace_texts_produces_vectors(self):
        h, _ = _build_helper(_make_2d(2))
        result = h.embed_documents(["   ", "\t\n"])
        assert len(result) == 2


# ===========================================================================
# 6. embed_pdf_chunks
# ===========================================================================

class TestEmbedPdfChunks:
    def test_empty_chunks_returns_empty(self):
        h, _ = _build_helper()
        assert h.embed_pdf_chunks([]) == []

    def test_adds_embedding_field(self):
        n = 3
        h, _ = _build_helper(_make_2d(n))
        chunks = [{"content": f"text {i}", "page": i} for i in range(n)]
        result = h.embed_pdf_chunks(chunks)
        for r in result:
            assert "embedding" in r

    def test_preserves_original_fields(self):
        h, _ = _build_helper(_make_2d(1))
        chunk = {"content": "legal text", "page": 5, "chunk_index": 2}
        result = h.embed_pdf_chunks([chunk])
        assert result[0]["page"] == 5
        assert result[0]["chunk_index"] == 2
        assert result[0]["content"] == "legal text"

    def test_does_not_mutate_original_chunk(self):
        h, _ = _build_helper(_make_2d(1))
        chunk = {"content": "legal text", "page": 1}
        h.embed_pdf_chunks([chunk])
        assert "embedding" not in chunk

    def test_embedding_is_list_of_floats(self):
        h, _ = _build_helper(_make_2d(2))
        chunks = [{"content": "a"}, {"content": "b"}]
        result = h.embed_pdf_chunks(chunks)
        for r in result:
            assert isinstance(r["embedding"], list)
            assert all(isinstance(f, float) for f in r["embedding"])

    def test_count_matches_input(self):
        n = 5
        h, _ = _build_helper(_make_2d(n))
        chunks = [{"content": f"text {i}"} for i in range(n)]
        assert len(h.embed_pdf_chunks(chunks)) == n

    def test_custom_text_key(self):
        h, mock = _build_helper(_make_2d(1))
        chunks = [{"body": "alternative field text"}]
        h.embed_pdf_chunks(chunks, text_key="body")
        texts_sent = mock.encode.call_args[0][0]
        assert texts_sent[0] == "alternative field text"

    def test_default_text_key_is_content(self):
        h, mock = _build_helper(_make_2d(1))
        chunks = [{"content": "contract text"}]
        h.embed_pdf_chunks(chunks)
        assert mock.encode.call_args[0][0][0] == "contract text"

    # FIX B5 regression: warning for missing / empty text
    def test_missing_text_key_emits_warning(self, caplog):
        h, _ = _build_helper(_make_2d(1))
        with caplog.at_level(logging.WARNING, logger="models.embeddings_model"):
            h.embed_pdf_chunks([{"page": 1}])
        assert any("empty text" in m for m in caplog.messages)

    def test_empty_content_emits_warning(self, caplog):
        h, _ = _build_helper(_make_2d(1))
        with caplog.at_level(logging.WARNING, logger="models.embeddings_model"):
            h.embed_pdf_chunks([{"content": ""}])
        assert any("empty text" in m for m in caplog.messages)

    def test_whitespace_only_content_emits_warning(self, caplog):
        h, _ = _build_helper(_make_2d(1))
        with caplog.at_level(logging.WARNING, logger="models.embeddings_model"):
            h.embed_pdf_chunks([{"content": "   "}])
        assert any("empty text" in m for m in caplog.messages)

    def test_good_content_no_warning(self, caplog):
        h, _ = _build_helper(_make_2d(1))
        with caplog.at_level(logging.WARNING, logger="models.embeddings_model"):
            h.embed_pdf_chunks([{"content": "valid text"}])
        assert not any("empty text" in m for m in caplog.messages)

    def test_batch_size_forwarded(self):
        h, mock = _build_helper(_make_2d(2))
        h.embed_pdf_chunks([{"content": "a"}, {"content": "b"}], batch_size=64)
        assert mock.encode.call_args[1]["batch_size"] == 64

    def test_cleans_text_before_encoding(self):
        h, mock = _build_helper(_make_2d(1))
        h.embed_pdf_chunks([{"content": "  noisy  \n text  "}])
        assert mock.encode.call_args[0][0][0] == "noisy text"

    def test_mixed_missing_and_present_keys(self, caplog):
        n = 3
        h, _ = _build_helper(_make_2d(n))
        chunks = [{"content": "ok"}, {"page": 2}, {"content": "also ok"}]
        with caplog.at_level(logging.WARNING, logger="models.embeddings_model"):
            result = h.embed_pdf_chunks(chunks)
        assert len(result) == n
        # Only middle chunk should warn
        assert sum(1 for m in caplog.messages if "empty text" in m) == 1


# ===========================================================================
# 7. __init__ / constructor
# ===========================================================================

class TestInit:
    def test_model_loaded_with_given_name(self):
        with patch("models.embeddings_model.SentenceTransformer") as mock_st:
            mock_st.return_value.get_embedding_dimension.return_value = DIM
            LegalBGEEmbeddingHelper("custom/model")
            mock_st.assert_called_once_with("custom/model")

    def test_default_model_name_from_env(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL_NAME", "env/model")
        with patch("models.embeddings_model.SentenceTransformer") as mock_st:
            mock_st.return_value.get_embedding_dimension.return_value = DIM
            h = LegalBGEEmbeddingHelper()
            assert h.model_name == "env/model"

    def test_default_model_name_fallback(self, monkeypatch):
        monkeypatch.delenv("EMBEDDING_MODEL_NAME", raising=False)
        with patch("models.embeddings_model.SentenceTransformer") as mock_st:
            mock_st.return_value.get_embedding_dimension.return_value = DIM
            h = LegalBGEEmbeddingHelper()
            assert h.model_name == "yuriyvnv/legal-bge-m3"

    def test_device_passed_to_sentence_transformer(self):
        with patch("models.embeddings_model.SentenceTransformer") as mock_st:
            mock_st.return_value.get_embedding_dimension.return_value = DIM
            LegalBGEEmbeddingHelper("model", device="cpu")
            assert mock_st.call_args[1]["device"] == "cpu"

    def test_no_device_kwarg_when_none(self):
        with patch("models.embeddings_model.SentenceTransformer") as mock_st:
            mock_st.return_value.get_embedding_dimension.return_value = DIM
            LegalBGEEmbeddingHelper("model", device=None)
            assert "device" not in mock_st.call_args[1]

    def test_embedding_dimension_stored(self):
        with patch("models.embeddings_model.SentenceTransformer") as mock_st:
            mock_st.return_value.get_embedding_dimension.return_value = 768
            h = LegalBGEEmbeddingHelper("model")
            assert h.embedding_dimension == 768

    def test_device_from_env(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")
        with patch("models.embeddings_model.SentenceTransformer") as mock_st:
            mock_st.return_value.get_embedding_dimension.return_value = DIM
            h = LegalBGEEmbeddingHelper("model")
            assert h.device == "cuda"


# ===========================================================================
# 8. get_embedding_helper factory
# ===========================================================================

class TestGetEmbeddingHelper:
    # FIX B4 regression: lru_cache must be unbounded
    def test_lru_cache_maxsize_is_none(self):
        assert get_embedding_helper.cache_info().maxsize is None

    def test_same_args_returns_same_object(self):
        get_embedding_helper.cache_clear()
        with patch("models.embeddings_model.SentenceTransformer") as mock_st:
            mock_st.return_value.get_embedding_dimension.return_value = DIM
            h1 = get_embedding_helper("model-a", None, True)
            h2 = get_embedding_helper("model-a", None, True)
            assert h1 is h2
        get_embedding_helper.cache_clear()

    def test_different_args_returns_different_objects(self):
        get_embedding_helper.cache_clear()
        with patch("models.embeddings_model.SentenceTransformer") as mock_st:
            mock_st.return_value.get_embedding_dimension.return_value = DIM
            h1 = get_embedding_helper("model-a", None, True)
            h2 = get_embedding_helper("model-b", None, True)
            assert h1 is not h2
        get_embedding_helper.cache_clear()

    def test_both_configs_cached_simultaneously(self):
        """B4: with maxsize=None both configs coexist in cache."""
        get_embedding_helper.cache_clear()
        with patch("models.embeddings_model.SentenceTransformer") as mock_st:
            mock_st.return_value.get_embedding_dimension.return_value = DIM
            get_embedding_helper("m1", None, True)
            get_embedding_helper("m2", None, True)
            # Calling m1 again should NOT re-instantiate (it's still in cache)
            call_count_before = mock_st.call_count
            get_embedding_helper("m1", None, True)
            assert mock_st.call_count == call_count_before
        get_embedding_helper.cache_clear()


# ===========================================================================
# 9. _STANDARD_ENCODE_KWARGS constant
# ===========================================================================

def test_standard_encode_kwargs_contains_expected():
    assert _STANDARD_ENCODE_KWARGS == {
        "batch_size",
        "show_progress_bar",
        "convert_to_numpy",
        "normalize_embeddings",
    }


# ===========================================================================
# 10. Integration — real PDFs → build docs → embed_pdf_chunks
# ===========================================================================

class TestRealPDFEmbeddingIntegration:
    """
    Process real PDFs through pdf_helper, then pass the resulting doc dicts
    through embed_pdf_chunks (with a mocked model) to verify the full pipeline
    produces correctly-shaped embedding fields.
    """

    from conftest import READABLE_PDFS as _readable_pdfs

    MAX_CHUNKS = 50   # cap to avoid slow tests
    EMBED_DIM = 1024  # matches schema

    def _get_docs(self, pdf_path: str) -> list[dict]:
        from helpers.pdf_helper import PDFHelper, StoredPDF
        h = PDFHelper()
        pages = h.extract_pages(pdf_path)
        chunks = h.chunk_pages(pages, max_chars=3500, overlap_chars=400)
        stored = StoredPDF(
            "int-id", Path(pdf_path).name, Path(pdf_path).name,
            pdf_path, "/url", "sha", 0
        )
        return h.build_typesense_chunk_documents(pdf=stored, chunks=chunks)

    @pytest.fixture(
        params=[str(p) for p in _readable_pdfs[:5]],
        ids=[p.name for p in _readable_pdfs[:5]],
    )
    def pdf_docs(self, request):
        return self._get_docs(request.param)

    def _embed_with_mock(self, docs: list[dict]) -> list[dict]:
        n = min(len(docs), self.MAX_CHUNKS)
        docs_slice = docs[:n]
        fake_vectors = _make_2d(n, self.EMBED_DIM)
        emb_helper, _ = _build_helper(fake_vectors, dim=self.EMBED_DIM)
        return emb_helper.embed_pdf_chunks(docs_slice)

    def test_all_output_docs_have_embedding(self, pdf_docs):
        result = self._embed_with_mock(pdf_docs)
        for doc in result:
            assert "embedding" in doc

    def test_embedding_is_list_of_floats(self, pdf_docs):
        result = self._embed_with_mock(pdf_docs)
        for doc in result:
            assert isinstance(doc["embedding"], list)
            assert all(isinstance(f, float) for f in doc["embedding"])

    def test_embedding_length_matches_schema(self, pdf_docs):
        result = self._embed_with_mock(pdf_docs)
        for doc in result:
            assert len(doc["embedding"]) == self.EMBED_DIM

    def test_original_fields_preserved(self, pdf_docs):
        result = self._embed_with_mock(pdf_docs)
        for doc in result:
            for key in ("id", "pdf_id", "content", "page"):
                assert key in doc

    def test_no_original_doc_mutated(self, pdf_docs):
        docs_slice = pdf_docs[: self.MAX_CHUNKS]
        for d in docs_slice:
            assert "embedding" not in d, "doc already has embedding before embedding step"
        self._embed_with_mock(pdf_docs)
        for d in docs_slice:
            assert "embedding" not in d, "original doc was mutated"

    def test_all_readable_pdfs_produce_valid_embeddings(self):
        for pdf_path in self._readable_pdfs:
            docs = self._get_docs(str(pdf_path))
            result = self._embed_with_mock(docs)
            assert len(result) > 0
            for doc in result:
                assert isinstance(doc.get("embedding"), list)
