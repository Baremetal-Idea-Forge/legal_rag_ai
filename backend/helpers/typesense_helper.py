"""
Typesense helper methods — client setup, collection management,
document CRUD, and search utilities for the PDF chunks collection.
"""

import typesense
from typesense.exceptions import ObjectNotFound, ObjectAlreadyExists, TypesenseClientError
from typing import Any
import logging

from core.config import get_settings

logger = logging.getLogger(__name__)

# Valid search modes accepted by search_pdf_chunks
_VALID_SEARCH_MODES = frozenset({"keyword", "vector", "hybrid"})

# ---------------------------------------------------------------------------
# Schema definition
# FIX B2/B3: renamed to PDF_CHUNKS_SCHEMA; added file_url, local_path,
#             page_start, page_end fields that build_typesense_chunk_documents
#             emits; embedding made optional so non-embedded docs can be stored.
# ---------------------------------------------------------------------------

PDF_CHUNKS_SCHEMA: dict[str, Any] = {
    # FIX B2: "name" is injected from config at creation time so that
    # overriding TYPESENSE_PDF_CHUNKS_COLLECTION in .env always takes effect.
    # The literal here is just the default / documentation value.
    "name": "pdf_chunks",
    "fields": [
        {"name": "id",            "type": "string"},
        {"name": "pdf_id",        "type": "string",  "facet": True},
        {"name": "pdf_name",      "type": "string"},
        # FIX B3: fields produced by build_typesense_chunk_documents but absent
        # from the original schema.
        {"name": "file_url",      "type": "string",  "optional": True},
        {"name": "local_path",    "type": "string",  "optional": True},
        {"name": "page",          "type": "int32",   "facet": True},
        {"name": "page_start",    "type": "int32",   "optional": True},
        {"name": "page_end",      "type": "int32",   "optional": True},
        {"name": "chunk_index",   "type": "int32"},
        {"name": "content",       "type": "string"},
        # Citation anchors: the chunk's absolute character span in
        # PDFHelper.build_document_text(). Optional because chunks indexed
        # before spans existed carry no offsets (they store -1 once re-ingested).
        {"name": "start_char",    "type": "int32",   "optional": True},
        {"name": "end_char",      "type": "int32",   "optional": True},
        # Document-level summary used to build the embedded retrieval_text.
        # Stored, never merged into `content`, and never cited.
        {"name": "summary",       "type": "string",  "optional": True},
        # Optional so documents can be stored before embeddings are computed.
        {"name": "embedding",     "type": "float[]", "num_dim": 1024, "optional": True},
    ],
    "default_sorting_field": "page",
}

# Backward-compatible alias (old name kept so existing imports don't break)
TASKS_SCHEMA = PDF_CHUNKS_SCHEMA

# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_typesense_client() -> typesense.Client:
    """Return a configured Typesense client (call once and reuse)."""
    settings = get_settings()
    return typesense.Client({
        "nodes": [{
            "host":     settings.TYPESENSE_HOST,
            "port":     settings.TYPESENSE_PORT,
            "protocol": settings.TYPESENSE_PROTOCOL,
        }],
        "api_key":                   settings.TYPESENSE_API_KEY,
        "connection_timeout_seconds": settings.TYPESENSE_CONNECTION_TIMEOUT,
    })


# Module-level singleton — importable across the app.
# Re-create by calling get_typesense_client() if settings change at runtime.
client: typesense.Client = get_typesense_client()

# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def create_tasks_collection(force: bool = False) -> dict:
    """
    Create the pdf_chunks collection.

    Args:
        force: If True, drop and recreate if it already exists.

    Returns:
        The collection schema dict returned by Typesense.
    """
    settings = get_settings()
    # FIX B1: use TYPESENSE_PDF_CHUNKS_COLLECTION (was TYPESENSE_TASKS_COLLECTION)
    collection_name = settings.TYPESENSE_PDF_CHUNKS_COLLECTION

    if force:
        drop_tasks_collection()

    # FIX B2: inject collection name from config so it always matches CRUD ops
    schema = {**PDF_CHUNKS_SCHEMA, "name": collection_name}
    try:
        result = client.collections.create(schema)
        logger.info("Collection '%s' created.", collection_name)
        return result
    except ObjectAlreadyExists:
        logger.info("Collection '%s' already exists — skipping.", collection_name)
        return get_tasks_collection()


def get_tasks_collection() -> dict:
    """Retrieve the pdf_chunks collection metadata."""
    settings = get_settings()
    # FIX B1: use TYPESENSE_PDF_CHUNKS_COLLECTION
    return client.collections[settings.TYPESENSE_PDF_CHUNKS_COLLECTION].retrieve()


def drop_tasks_collection() -> dict:
    """Delete the pdf_chunks collection entirely (destructive!)."""
    settings = get_settings()
    # FIX B1: use TYPESENSE_PDF_CHUNKS_COLLECTION
    collection_name = settings.TYPESENSE_PDF_CHUNKS_COLLECTION
    try:
        result = client.collections[collection_name].delete()
        logger.warning("Collection '%s' dropped.", collection_name)
        return result
    except ObjectNotFound:
        logger.info("Collection '%s' not found — nothing to drop.", collection_name)
        return {}


def collection_exists() -> bool:
    """Return True if the pdf_chunks collection exists in Typesense."""
    try:
        get_tasks_collection()
        return True
    except ObjectNotFound:
        return False


def ensure_tasks_collection() -> None:
    """Create the collection if it does not exist yet."""
    if not collection_exists():
        create_tasks_collection()


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def _to_typesense_doc(chunk_dict: dict, *, require_embedding: bool = True) -> dict:
    """
    Validate and copy a chunk dict for indexing or updating.

    Args:
        chunk_dict:        The source document fields.
        require_embedding: If True (default for index/bulk), embedding must be
                           present.  Pass False for partial updates where only
                           some fields change.
    """
    doc = chunk_dict.copy()

    if "id" not in doc:
        raise ValueError("chunk_dict must contain 'id'")
    if "content" not in doc:
        raise ValueError("chunk_dict must contain 'content'")
    # FIX B4: embedding check is now conditional so partial updates work
    if require_embedding and "embedding" not in doc:
        raise ValueError("chunk_dict must contain 'embedding'")

    return doc


def index_pdf_chunk(chunk_dict: dict) -> dict:
    settings = get_settings()
    doc = _to_typesense_doc(chunk_dict, require_embedding=True)
    # FIX B1: use TYPESENSE_PDF_CHUNKS_COLLECTION
    return client.collections[settings.TYPESENSE_PDF_CHUNKS_COLLECTION].documents.upsert(doc)


def get_pdf_chunk_document(chunk_id: str) -> dict:
    settings = get_settings()
    # FIX B1: use TYPESENSE_PDF_CHUNKS_COLLECTION
    return client.collections[settings.TYPESENSE_PDF_CHUNKS_COLLECTION].documents[chunk_id].retrieve()


def update_pdf_chunk_document(chunk_id: str, partial: dict) -> dict:
    settings = get_settings()
    # FIX B4: partial update must NOT require embedding
    doc = _to_typesense_doc({**partial, "id": chunk_id}, require_embedding=False)
    # FIX B1: use TYPESENSE_PDF_CHUNKS_COLLECTION
    return client.collections[settings.TYPESENSE_PDF_CHUNKS_COLLECTION].documents[chunk_id].update(doc)


def delete_pdf_chunk_document(chunk_id: str) -> dict:
    settings = get_settings()
    # FIX B1: use TYPESENSE_PDF_CHUNKS_COLLECTION
    return client.collections[settings.TYPESENSE_PDF_CHUNKS_COLLECTION].documents[chunk_id].delete()


def bulk_index_pdf_chunks(chunk_dicts: list[dict]) -> list[dict]:
    settings = get_settings()
    docs = [_to_typesense_doc(chunk, require_embedding=True) for chunk in chunk_dicts]
    # FIX B1: use TYPESENSE_PDF_CHUNKS_COLLECTION
    return client.collections[settings.TYPESENSE_PDF_CHUNKS_COLLECTION].documents.import_(
        docs, {"action": "upsert"}
    )


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def search_pdf_chunks(
    query: str,
    *,
    mode: str = "hybrid",   # "keyword" | "vector" | "hybrid"
    query_by: str = "content",
    filter_by: str | None = None,
    sort_by: str | None = None,
    # FIX B5: removed non-existent 'section' field from default facets
    facet_by: str | None = "pdf_id,page",
    page: int = 1,
    per_page: int = 20,
    highlight_fields: str = "content",
    vector_query: str | None = None,
) -> dict:
    # FIX B6: validate mode before building params
    if mode not in _VALID_SEARCH_MODES:
        raise ValueError(
            f"Invalid search mode '{mode}'. Must be one of: {sorted(_VALID_SEARCH_MODES)}"
        )
    # FIX B7: vector search without a vector_query produces a broken request
    if mode == "vector" and not vector_query:
        raise ValueError("vector_query is required when mode='vector'")

    settings = get_settings()
    params: dict[str, Any] = {
        "q":                    query if mode in ("keyword", "hybrid") else "*",
        "page":                 page,
        "per_page":             per_page,
        "highlight_full_fields": highlight_fields,
    }

    if mode in ("keyword", "hybrid"):
        params["query_by"] = query_by

    if filter_by:
        params["filter_by"] = filter_by
    if sort_by:
        params["sort_by"] = sort_by
    if facet_by:
        params["facet_by"] = facet_by

    if mode in ("vector", "hybrid") and vector_query:
        params["vector_query"] = vector_query

    # FIX B1: use TYPESENSE_PDF_CHUNKS_COLLECTION
    # FIX B8: Typesense rejects GET searches whose query string exceeds 4000
    # chars. A 1024-dim vector_query (~7 KB) blows past that limit, so route
    # searches through POST /multi_search which carries the payload in the body.
    search_requests = {
        "searches": [
            {"collection": settings.TYPESENSE_PDF_CHUNKS_COLLECTION, **params}
        ]
    }
    result = client.multi_search.perform(search_requests, {})["results"][0]
    # multi_search returns HTTP 200 even for per-search failures; surface them
    # as a client error so the repository maps them to our domain exceptions.
    if "error" in result:
        raise TypesenseClientError(result["error"])
    return result
