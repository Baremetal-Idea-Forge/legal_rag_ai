"""
Comprehensive test suite for backend/helpers/typesense_helper.py.

Covers:
  - Config: TYPESENSE_PDF_CHUNKS_COLLECTION key present
  - Schema: PDF_CHUNKS_SCHEMA structure, field completeness, field types
  - _to_typesense_doc: validation logic, require_embedding flag
  - Collection management: create / get / drop / exists / ensure (mocked client)
  - Document CRUD: index / get / update / delete / bulk (mocked client)
  - search_pdf_chunks: all modes, param construction, validation errors (mocked)
  - Integration: process real PDFs → verify docs pass schema and validation
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import helpers.typesense_helper as ts
from helpers.typesense_helper import (
    PDF_CHUNKS_SCHEMA,
    TASKS_SCHEMA,
    _VALID_SEARCH_MODES,
    _to_typesense_doc,
    bulk_index_pdf_chunks,
    collection_exists,
    create_tasks_collection,
    delete_pdf_chunk_document,
    drop_tasks_collection,
    ensure_tasks_collection,
    get_pdf_chunk_document,
    get_tasks_collection,
    get_typesense_client,
    index_pdf_chunk,
    search_pdf_chunks,
    update_pdf_chunk_document,
)
from core.config import get_settings
from helpers.pdf_helper import PDFHelper, PDFChunk, StoredPDF

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"


def _mock_client() -> MagicMock:
    """Return a MagicMock that structurally mirrors the typesense.Client API."""
    m = MagicMock()
    # collections["name"] returns a collection resource
    collection_resource = MagicMock()
    m.collections.__getitem__.return_value = collection_resource
    m.collections.create.return_value = {"name": "pdf_chunks"}
    collection_resource.retrieve.return_value = {"name": "pdf_chunks"}
    collection_resource.delete.return_value = {"name": "pdf_chunks"}
    collection_resource.documents.upsert.return_value = {"id": "test"}
    collection_resource.documents.__getitem__.return_value.retrieve.return_value = {"id": "test"}
    collection_resource.documents.__getitem__.return_value.update.return_value = {"id": "test"}
    collection_resource.documents.__getitem__.return_value.delete.return_value = {"id": "test"}
    collection_resource.documents.import_.return_value = [{"success": True}]
    collection_resource.documents.search.return_value = {"hits": []}
    return m


def _valid_chunk_doc(chunk_id: str = "pdf1_chunk_0") -> dict:
    return {
        "id": chunk_id,
        "pdf_id": "pdf1",
        "pdf_name": "contract.pdf",
        "content": "This agreement is entered into...",
        "page": 1,
        "page_start": 1,
        "page_end": 1,
        "chunk_index": 0,
        "file_url": "/pdfs/pdf1/contract.pdf",
        "local_path": "/storage/pdf1/contract.pdf",
        "embedding": [0.1] * 1024,
    }


# ===========================================================================
# 1. Config key existence (B1 regression)
# ===========================================================================

class TestConfig:
    def test_pdf_chunks_collection_key_exists(self):
        """B1: TYPESENSE_PDF_CHUNKS_COLLECTION must exist in Settings."""
        s = get_settings()
        assert hasattr(s, "TYPESENSE_PDF_CHUNKS_COLLECTION")

    def test_pdf_chunks_collection_default_value(self):
        s = get_settings()
        assert s.TYPESENSE_PDF_CHUNKS_COLLECTION == "pdf_chunks"

    def test_tasks_collection_still_present_for_backward_compat(self):
        s = get_settings()
        assert hasattr(s, "TYPESENSE_TASKS_COLLECTION")


# ===========================================================================
# 2. Schema validation (B2 / B3 regression)
# ===========================================================================

class TestSchema:
    def _schema_fields(self) -> dict[str, dict]:
        return {f["name"]: f for f in PDF_CHUNKS_SCHEMA["fields"]}

    def test_backward_compat_alias(self):
        """TASKS_SCHEMA alias must point to the same object as PDF_CHUNKS_SCHEMA."""
        assert TASKS_SCHEMA is PDF_CHUNKS_SCHEMA

    def test_schema_name_matches_config(self):
        """B2: schema 'name' must equal TYPESENSE_PDF_CHUNKS_COLLECTION."""
        assert PDF_CHUNKS_SCHEMA["name"] == get_settings().TYPESENSE_PDF_CHUNKS_COLLECTION

    def test_schema_has_default_sorting_field(self):
        assert "default_sorting_field" in PDF_CHUNKS_SCHEMA

    def test_default_sorting_field_exists_in_fields(self):
        sort_field = PDF_CHUNKS_SCHEMA["default_sorting_field"]
        names = {f["name"] for f in PDF_CHUNKS_SCHEMA["fields"]}
        assert sort_field in names

    # Required fields
    def test_required_field_id(self):
        f = self._schema_fields()
        assert "id" in f
        assert f["id"]["type"] == "string"

    def test_required_field_pdf_id(self):
        f = self._schema_fields()
        assert "pdf_id" in f
        assert f["pdf_id"]["type"] == "string"

    def test_required_field_pdf_name(self):
        f = self._schema_fields()
        assert "pdf_name" in f

    def test_required_field_page(self):
        f = self._schema_fields()
        assert "page" in f
        assert f["page"]["type"] == "int32"

    def test_required_field_chunk_index(self):
        f = self._schema_fields()
        assert "chunk_index" in f
        assert f["chunk_index"]["type"] == "int32"

    def test_required_field_content(self):
        f = self._schema_fields()
        assert "content" in f
        assert f["content"]["type"] == "string"

    def test_required_field_embedding(self):
        f = self._schema_fields()
        assert "embedding" in f
        assert f["embedding"]["type"] == "float[]"
        assert f["embedding"]["num_dim"] == 1024

    # FIX B3 regression: previously missing fields
    def test_field_file_url_present(self):
        assert "file_url" in self._schema_fields()

    def test_field_local_path_present(self):
        assert "local_path" in self._schema_fields()

    def test_field_page_start_present(self):
        assert "page_start" in self._schema_fields()

    def test_field_page_end_present(self):
        assert "page_end" in self._schema_fields()

    def test_page_start_type_int32(self):
        f = self._schema_fields()
        assert f["page_start"]["type"] == "int32"

    def test_page_end_type_int32(self):
        f = self._schema_fields()
        assert f["page_end"]["type"] == "int32"

    def test_embedding_is_optional(self):
        """Embedding marked optional so docs can be stored before embedding."""
        f = self._schema_fields()
        assert f["embedding"].get("optional") is True

    def test_facet_fields_are_facetable(self):
        f = self._schema_fields()
        assert f["pdf_id"].get("facet") is True
        assert f["page"].get("facet") is True

    def test_schema_covers_all_doc_builder_fields(self):
        """All keys from build_typesense_chunk_documents must exist in schema."""
        stored = StoredPDF("id1", "f.pdf", "f.pdf", "/tmp", "/url", "sha", 100)
        chunks = [PDFChunk(0, 1, 2, "text")]
        docs = PDFHelper().build_typesense_chunk_documents(pdf=stored, chunks=chunks)
        doc_fields = set(docs[0].keys())
        schema_fields = {f["name"] for f in PDF_CHUNKS_SCHEMA["fields"]}
        missing = doc_fields - schema_fields
        assert not missing, f"Fields in doc but absent from schema: {missing}"


# ===========================================================================
# 3. _to_typesense_doc validation (B4 regression)
# ===========================================================================

class TestToTypesenseDoc:
    def test_valid_full_doc_passes(self):
        doc = _valid_chunk_doc()
        result = _to_typesense_doc(doc)
        assert result["id"] == doc["id"]

    def test_returns_copy_not_same_object(self):
        doc = _valid_chunk_doc()
        result = _to_typesense_doc(doc)
        assert result is not doc

    def test_mutation_of_result_does_not_affect_original(self):
        doc = _valid_chunk_doc()
        result = _to_typesense_doc(doc)
        result["extra"] = "injected"
        assert "extra" not in doc

    def test_missing_id_raises(self):
        d = _valid_chunk_doc()
        del d["id"]
        with pytest.raises(ValueError, match="'id'"):
            _to_typesense_doc(d)

    def test_missing_content_raises(self):
        d = _valid_chunk_doc()
        del d["content"]
        with pytest.raises(ValueError, match="'content'"):
            _to_typesense_doc(d)

    def test_missing_embedding_raises_when_required(self):
        """Default require_embedding=True: must raise if embedding absent."""
        d = _valid_chunk_doc()
        del d["embedding"]
        with pytest.raises(ValueError, match="'embedding'"):
            _to_typesense_doc(d, require_embedding=True)

    def test_missing_embedding_allowed_when_not_required(self):
        """B4 fix: require_embedding=False must not raise for partial updates."""
        d = _valid_chunk_doc()
        del d["embedding"]
        result = _to_typesense_doc(d, require_embedding=False)
        assert "embedding" not in result

    def test_extra_fields_preserved(self):
        d = _valid_chunk_doc()
        d["custom_meta"] = "value"
        result = _to_typesense_doc(d)
        assert result["custom_meta"] == "value"

    def test_default_require_embedding_is_true(self):
        d = {k: v for k, v in _valid_chunk_doc().items() if k != "embedding"}
        with pytest.raises(ValueError):
            _to_typesense_doc(d)


# ===========================================================================
# 4. Collection management (all mocked)
# ===========================================================================

class TestCollectionManagement:
    @pytest.fixture(autouse=True)
    def mock_client(self):
        m = _mock_client()
        with patch.object(ts, "client", m):
            yield m

    def test_create_calls_collections_create(self, mock_client):
        create_tasks_collection()
        mock_client.collections.create.assert_called_once()

    def test_create_injects_collection_name_from_config(self, mock_client):
        """B2 fix: schema passed to create must use config name, not hardcoded."""
        create_tasks_collection()
        schema_passed = mock_client.collections.create.call_args[0][0]
        assert schema_passed["name"] == get_settings().TYPESENSE_PDF_CHUNKS_COLLECTION

    def test_create_uses_pdf_chunks_collection_name(self, mock_client):
        """B1 fix: collection_name read from TYPESENSE_PDF_CHUNKS_COLLECTION."""
        create_tasks_collection()
        schema_passed = mock_client.collections.create.call_args[0][0]
        assert schema_passed["name"] == "pdf_chunks"

    def test_create_force_drops_first(self, mock_client):
        from typesense.exceptions import ObjectNotFound
        mock_client.collections.__getitem__.return_value.delete.return_value = {}
        create_tasks_collection(force=True)
        mock_client.collections.__getitem__.return_value.delete.assert_called_once()
        mock_client.collections.create.assert_called_once()

    def test_create_handles_already_exists(self, mock_client):
        from typesense.exceptions import ObjectAlreadyExists
        mock_client.collections.create.side_effect = ObjectAlreadyExists
        result = create_tasks_collection()
        # Should fall back to get_tasks_collection
        mock_client.collections.__getitem__.return_value.retrieve.assert_called_once()

    def test_get_tasks_collection_uses_pdf_chunks_collection(self, mock_client):
        """B1 fix: get uses TYPESENSE_PDF_CHUNKS_COLLECTION."""
        get_tasks_collection()
        key_used = mock_client.collections.__getitem__.call_args[0][0]
        assert key_used == get_settings().TYPESENSE_PDF_CHUNKS_COLLECTION

    def test_drop_tasks_collection_uses_correct_name(self, mock_client):
        drop_tasks_collection()
        key_used = mock_client.collections.__getitem__.call_args[0][0]
        assert key_used == get_settings().TYPESENSE_PDF_CHUNKS_COLLECTION

    def test_drop_handles_not_found(self, mock_client):
        from typesense.exceptions import ObjectNotFound
        mock_client.collections.__getitem__.return_value.delete.side_effect = ObjectNotFound
        result = drop_tasks_collection()
        assert result == {}

    def test_collection_exists_true(self, mock_client):
        assert collection_exists() is True

    def test_collection_exists_false_on_not_found(self, mock_client):
        from typesense.exceptions import ObjectNotFound
        mock_client.collections.__getitem__.return_value.retrieve.side_effect = ObjectNotFound
        assert collection_exists() is False

    def test_ensure_creates_when_not_exists(self, mock_client):
        from typesense.exceptions import ObjectNotFound
        mock_client.collections.__getitem__.return_value.retrieve.side_effect = ObjectNotFound
        mock_client.collections.__getitem__.return_value.retrieve.side_effect = [ObjectNotFound, {}]
        mock_client.collections.create.return_value = {"name": "pdf_chunks"}
        ensure_tasks_collection()
        mock_client.collections.create.assert_called_once()

    def test_ensure_skips_when_exists(self, mock_client):
        ensure_tasks_collection()
        mock_client.collections.create.assert_not_called()


# ===========================================================================
# 5. Document CRUD helpers (all mocked)
# ===========================================================================

class TestDocumentCRUD:
    @pytest.fixture(autouse=True)
    def mock_client(self):
        m = _mock_client()
        with patch.object(ts, "client", m):
            yield m

    def test_index_pdf_chunk_upserts(self, mock_client):
        doc = _valid_chunk_doc()
        index_pdf_chunk(doc)
        mock_client.collections.__getitem__.return_value.documents.upsert.assert_called_once_with(doc)

    def test_index_pdf_chunk_uses_correct_collection(self, mock_client):
        index_pdf_chunk(_valid_chunk_doc())
        key_used = mock_client.collections.__getitem__.call_args[0][0]
        assert key_used == "pdf_chunks"

    def test_index_pdf_chunk_raises_without_embedding(self):
        d = _valid_chunk_doc()
        del d["embedding"]
        with pytest.raises(ValueError, match="embedding"):
            index_pdf_chunk(d)

    def test_index_pdf_chunk_raises_without_id(self):
        d = _valid_chunk_doc()
        del d["id"]
        with pytest.raises(ValueError, match="'id'"):
            index_pdf_chunk(d)

    def test_get_pdf_chunk_document(self, mock_client):
        get_pdf_chunk_document("chunk_001")
        mock_client.collections.__getitem__.return_value.documents.__getitem__.assert_called_with("chunk_001")

    def test_get_pdf_chunk_uses_correct_collection(self, mock_client):
        get_pdf_chunk_document("chunk_001")
        key_used = mock_client.collections.__getitem__.call_args[0][0]
        assert key_used == "pdf_chunks"

    def test_update_pdf_chunk_does_not_require_embedding(self, mock_client):
        """B4 fix: partial update without embedding must succeed."""
        update_pdf_chunk_document("chunk_001", {"content": "updated text"})
        mock_client.collections.__getitem__.return_value.documents.__getitem__.return_value.update.assert_called_once()

    def test_update_pdf_chunk_uses_correct_collection(self, mock_client):
        update_pdf_chunk_document("chunk_001", {"content": "updated"})
        key_used = mock_client.collections.__getitem__.call_args[0][0]
        assert key_used == "pdf_chunks"

    def test_update_injects_id_into_doc(self, mock_client):
        update_pdf_chunk_document("chunk_999", {"content": "text"})
        doc_sent = mock_client.collections.__getitem__.return_value.documents.__getitem__.return_value.update.call_args[0][0]
        assert doc_sent["id"] == "chunk_999"

    def test_update_raises_without_content(self, mock_client):
        with pytest.raises(ValueError, match="'content'"):
            update_pdf_chunk_document("chunk_001", {})

    def test_delete_pdf_chunk_document(self, mock_client):
        delete_pdf_chunk_document("chunk_001")
        mock_client.collections.__getitem__.return_value.documents.__getitem__.return_value.delete.assert_called_once()

    def test_delete_uses_correct_collection(self, mock_client):
        delete_pdf_chunk_document("chunk_001")
        key_used = mock_client.collections.__getitem__.call_args[0][0]
        assert key_used == "pdf_chunks"

    def test_bulk_index_calls_import(self, mock_client):
        docs = [_valid_chunk_doc(f"pdf1_chunk_{i}") for i in range(3)]
        bulk_index_pdf_chunks(docs)
        mock_client.collections.__getitem__.return_value.documents.import_.assert_called_once()

    def test_bulk_index_passes_upsert_action(self, mock_client):
        bulk_index_pdf_chunks([_valid_chunk_doc()])
        _, kwargs_or_args = mock_client.collections.__getitem__.return_value.documents.import_.call_args
        call_args = mock_client.collections.__getitem__.return_value.documents.import_.call_args
        action_arg = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})
        # Verify action is "upsert"
        args_list = list(call_args[0])
        assert args_list[1] == {"action": "upsert"}

    def test_bulk_index_validates_each_doc(self, mock_client):
        docs = [_valid_chunk_doc("pdf1_chunk_0"), {"id": "pdf1_chunk_1", "content": "text"}]
        with pytest.raises(ValueError, match="embedding"):
            bulk_index_pdf_chunks(docs)

    def test_bulk_index_uses_correct_collection(self, mock_client):
        bulk_index_pdf_chunks([_valid_chunk_doc()])
        key_used = mock_client.collections.__getitem__.call_args[0][0]
        assert key_used == "pdf_chunks"

    def test_bulk_index_empty_list(self, mock_client):
        result = bulk_index_pdf_chunks([])
        mock_client.collections.__getitem__.return_value.documents.import_.assert_called_once()


# ===========================================================================
# 6. search_pdf_chunks (all mocked)
# ===========================================================================

class TestSearchPdfChunks:
    @pytest.fixture(autouse=True)
    def mock_client(self):
        m = _mock_client()
        with patch.object(ts, "client", m):
            yield m

    def _last_search_params(self, mock_client) -> dict:
        return mock_client.collections.__getitem__.return_value.documents.search.call_args[0][0]

    # --- B6 regression: mode validation ---
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid search mode"):
            search_pdf_chunks("query", mode="fuzzy")

    def test_invalid_mode_message_lists_valid_modes(self):
        with pytest.raises(ValueError) as exc_info:
            search_pdf_chunks("query", mode="bad")
        msg = str(exc_info.value)
        for valid in ("hybrid", "keyword", "vector"):
            assert valid in msg

    # --- B7 regression: vector mode validation ---
    def test_vector_mode_without_vector_query_raises(self):
        with pytest.raises(ValueError, match="vector_query"):
            search_pdf_chunks("query", mode="vector", vector_query=None)

    def test_vector_mode_with_vector_query_succeeds(self, mock_client):
        vq = "embedding:(1024 vectors...)"
        search_pdf_chunks("query", mode="vector", vector_query=vq)
        params = self._last_search_params(mock_client)
        assert params["vector_query"] == vq

    # --- Mode-specific param construction ---
    def test_keyword_mode_sets_query_by(self, mock_client):
        search_pdf_chunks("contract breach", mode="keyword")
        params = self._last_search_params(mock_client)
        assert params["query_by"] == "content"
        assert params["q"] == "contract breach"
        assert "vector_query" not in params

    def test_keyword_mode_custom_query_by(self, mock_client):
        search_pdf_chunks("test", mode="keyword", query_by="pdf_name")
        params = self._last_search_params(mock_client)
        assert params["query_by"] == "pdf_name"

    def test_hybrid_mode_sets_query_and_query_by(self, mock_client):
        search_pdf_chunks("test", mode="hybrid")
        params = self._last_search_params(mock_client)
        assert params["q"] == "test"
        assert "query_by" in params

    def test_hybrid_with_vector_query_includes_it(self, mock_client):
        vq = "embedding:([0.1, 0.2, ...], k:10)"
        search_pdf_chunks("test", mode="hybrid", vector_query=vq)
        params = self._last_search_params(mock_client)
        assert params["vector_query"] == vq

    def test_vector_mode_sets_q_star(self, mock_client):
        vq = "embedding:([0.1]*1024, k:10)"
        search_pdf_chunks("ignored", mode="vector", vector_query=vq)
        params = self._last_search_params(mock_client)
        assert params["q"] == "*"
        assert "query_by" not in params

    # --- Optional params ---
    def test_filter_by_included_when_provided(self, mock_client):
        search_pdf_chunks("test", filter_by="pdf_id:=abc123")
        params = self._last_search_params(mock_client)
        assert params["filter_by"] == "pdf_id:=abc123"

    def test_filter_by_absent_when_none(self, mock_client):
        search_pdf_chunks("test", filter_by=None)
        params = self._last_search_params(mock_client)
        assert "filter_by" not in params

    def test_sort_by_included_when_provided(self, mock_client):
        search_pdf_chunks("test", sort_by="page:asc")
        params = self._last_search_params(mock_client)
        assert params["sort_by"] == "page:asc"

    def test_sort_by_absent_when_none(self, mock_client):
        search_pdf_chunks("test", sort_by=None)
        params = self._last_search_params(mock_client)
        assert "sort_by" not in params

    def test_facet_by_included_when_provided(self, mock_client):
        search_pdf_chunks("test", facet_by="pdf_id")
        params = self._last_search_params(mock_client)
        assert params["facet_by"] == "pdf_id"

    def test_facet_by_absent_when_none(self, mock_client):
        search_pdf_chunks("test", facet_by=None)
        params = self._last_search_params(mock_client)
        assert "facet_by" not in params

    # --- B5 regression: default facets must NOT include 'section' ---
    def test_default_facet_by_does_not_include_section(self, mock_client):
        search_pdf_chunks("test")
        params = self._last_search_params(mock_client)
        facet_str = params.get("facet_by", "")
        assert "section" not in (facet_str or "")

    def test_default_facet_by_includes_pdf_id_and_page(self, mock_client):
        search_pdf_chunks("test")
        params = self._last_search_params(mock_client)
        facet_str = params.get("facet_by", "")
        assert "pdf_id" in facet_str
        assert "page" in facet_str

    # --- Pagination ---
    def test_page_and_per_page_defaults(self, mock_client):
        search_pdf_chunks("test")
        params = self._last_search_params(mock_client)
        assert params["page"] == 1
        assert params["per_page"] == 20

    def test_custom_pagination(self, mock_client):
        search_pdf_chunks("test", page=3, per_page=50)
        params = self._last_search_params(mock_client)
        assert params["page"] == 3
        assert params["per_page"] == 50

    def test_uses_pdf_chunks_collection(self, mock_client):
        search_pdf_chunks("test")
        key_used = mock_client.collections.__getitem__.call_args[0][0]
        assert key_used == "pdf_chunks"

    def test_valid_modes_accepted(self, mock_client):
        for mode in ("keyword", "hybrid"):
            search_pdf_chunks("test", mode=mode)
        vq = "embedding:([0.1]*1024, k:5)"
        search_pdf_chunks("test", mode="vector", vector_query=vq)

    def test_highlight_fields_in_params(self, mock_client):
        search_pdf_chunks("test", highlight_fields="pdf_name")
        params = self._last_search_params(mock_client)
        assert params["highlight_full_fields"] == "pdf_name"


# ===========================================================================
# 7. get_typesense_client factory
# ===========================================================================

def test_get_typesense_client_returns_client():
    import typesense
    c = get_typesense_client()
    assert isinstance(c, typesense.Client)


# ===========================================================================
# 8. _VALID_SEARCH_MODES constant
# ===========================================================================

def test_valid_search_modes_contains_all_expected():
    assert _VALID_SEARCH_MODES == {"keyword", "vector", "hybrid"}


# ===========================================================================
# 9. Integration — real PDFs → verify docs satisfy schema and validation
# ===========================================================================

class TestRealPDFIntegration:
    """
    No Typesense server needed: we process PDFs through pdf_helper, build docs,
    and verify that every produced doc:
      - passes _to_typesense_doc (sans embedding which is added later)
      - has all non-optional schema fields
      - has no extra fields not in schema
      - has consistent id format
    """
    from conftest import READABLE_PDFS
    _readable = READABLE_PDFS

    MAX_CHARS = 3500
    OVERLAP = 400
    SCHEMA_FIELD_NAMES = {f["name"] for f in PDF_CHUNKS_SCHEMA["fields"]}
    # Fields that are optional in schema — absence is allowed
    OPTIONAL_FIELDS = {
        f["name"]
        for f in PDF_CHUNKS_SCHEMA["fields"]
        if f.get("optional")
    }

    def _build_docs(self, pdf_path: str) -> list[dict]:
        h = PDFHelper()
        pages = h.extract_pages(pdf_path)
        chunks = h.chunk_pages(pages, max_chars=self.MAX_CHARS, overlap_chars=self.OVERLAP)
        stored = StoredPDF(
            pdf_id="int-test-id",
            original_filename=Path(pdf_path).name,
            stored_filename=Path(pdf_path).name,
            local_path=pdf_path,
            file_url=f"/pdfs/int-test-id/{Path(pdf_path).name}",
            sha256="sha",
            size_bytes=0,
        )
        return h.build_typesense_chunk_documents(pdf=stored, chunks=chunks)

    @pytest.fixture(
        params=[str(p) for p in READABLE_PDFS[:5]],
        ids=[p.name for p in READABLE_PDFS[:5]],
    )
    def sample_docs(self, request):
        return self._build_docs(request.param)

    def test_at_least_one_doc_produced(self, sample_docs):
        assert len(sample_docs) > 0

    def test_all_docs_pass_to_typesense_doc_without_embedding(self, sample_docs):
        """Each doc must pass validation with require_embedding=False."""
        for doc in sample_docs:
            result = _to_typesense_doc(doc, require_embedding=False)
            assert result is not None

    def test_all_docs_have_id_field(self, sample_docs):
        for doc in sample_docs:
            assert "id" in doc and doc["id"]

    def test_all_docs_have_content_field(self, sample_docs):
        for doc in sample_docs:
            assert "content" in doc and doc["content"].strip()

    def test_all_docs_have_pdf_id(self, sample_docs):
        for doc in sample_docs:
            assert "pdf_id" in doc

    def test_all_docs_have_page_field(self, sample_docs):
        for doc in sample_docs:
            assert "page" in doc
            assert isinstance(doc["page"], int)

    def test_page_field_equals_page_start(self, sample_docs):
        for doc in sample_docs:
            assert doc["page"] == doc["page_start"]

    def test_id_format_consistent(self, sample_docs):
        """ID must follow pattern {pdf_id}_chunk_{index}."""
        for doc in sample_docs:
            parts = doc["id"].rsplit("_chunk_", 1)
            assert len(parts) == 2, f"Unexpected ID format: {doc['id']}"
            assert parts[1].isdigit(), f"chunk index not numeric in: {doc['id']}"

    def test_no_field_absent_from_schema(self, sample_docs):
        """No doc field should exist outside the schema definition."""
        schema_fields = self.SCHEMA_FIELD_NAMES
        for doc in sample_docs:
            extra = set(doc.keys()) - schema_fields - {"embedding"}
            assert not extra, f"Doc has fields not in schema: {extra}"

    def test_doc_has_no_empty_string_values_for_required_fields(self, sample_docs):
        for doc in sample_docs:
            for key in ("id", "pdf_id", "pdf_name", "content"):
                assert str(doc[key]).strip(), f"Empty required field '{key}' in doc {doc['id']}"

    def test_chunk_indices_sequential_per_pdf(self, sample_docs):
        indices = [doc["chunk_index"] for doc in sample_docs]
        assert indices == list(range(len(sample_docs)))

    def test_all_readable_pdfs_produce_valid_docs(self):
        """Run the full pipeline on ALL readable PDFs."""
        for pdf_path in self._readable:
            docs = self._build_docs(str(pdf_path))
            assert len(docs) > 0, f"No docs for {pdf_path.name}"
            for doc in docs:
                _to_typesense_doc(doc, require_embedding=False)
