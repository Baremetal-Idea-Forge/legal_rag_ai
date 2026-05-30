"""
Typesense helper methods — client setup, collection management,
document CRUD, and search utilities for the Task collection.
"""

import typesense
from typesense.exceptions import ObjectNotFound, ObjectAlreadyExists
from typing import Any
import logging

from core.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

TASKS_SCHEMA: dict[str, Any] = {
  "name": "pdf_chunks",
  "fields": [
    {"name": "id", "type": "string"},
    {"name": "pdf_id", "type": "string", "facet": True},
    {"name": "pdf_name", "type": "string"},
    {"name": "page", "type": "int32", "facet": True},
    {"name": "chunk_index", "type": "int32"},
    {"name": "content", "type": "string"},
    {"name": "embedding", "type": "float[]", "num_dim": 1024}
  ],
  "default_sorting_field": "page"
}

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
        "api_key":            settings.TYPESENSE_API_KEY,
        "connection_timeout_seconds": settings.TYPESENSE_CONNECTION_TIMEOUT,
    })


# Module-level singleton — importable across the app
client: typesense.Client = get_typesense_client()

# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def create_tasks_collection(force: bool = False) -> dict:
    """
    Create the tasks collection.

    Args:
        force: If True, drop and recreate if it already exists.

    Returns:
        The collection schema dict returned by Typesense.
    """
    settings = get_settings()
    collection_name = settings.TYPESENSE_TASKS_COLLECTION

    if force:
        drop_tasks_collection()

    try:
        result = client.collections.create(TASKS_SCHEMA)
        logger.info("Collection '%s' created.", collection_name)
        return result
    except ObjectAlreadyExists:
        logger.info("Collection '%s' already exists — skipping.", collection_name)
        return get_tasks_collection()


def get_tasks_collection() -> dict:
    """Retrieve the tasks collection metadata."""
    settings = get_settings()
    return client.collections[settings.TYPESENSE_TASKS_COLLECTION].retrieve()


def drop_tasks_collection() -> dict:
    """Delete the tasks collection entirely (destructive!)."""
    settings = get_settings()
    try:
        result = client.collections[settings.TYPESENSE_TASKS_COLLECTION].delete()
        logger.warning("Collection '%s' dropped.", settings.TYPESENSE_TASKS_COLLECTION)
        return result
    except ObjectNotFound:
        logger.info("Collection '%s' not found — nothing to drop.", settings.TYPESENSE_TASKS_COLLECTION)
        return {}


def collection_exists() -> bool:
    """Return True if the tasks collection exists in Typesense."""
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

def _to_typesense_doc(chunk_dict: dict) -> dict:
    doc = chunk_dict.copy()

    # Required fields for PDF chunk indexing
    if "id" not in doc:
        raise ValueError("chunk_dict must contain 'id'")
    if "content" not in doc:
        raise ValueError("chunk_dict must contain 'content'")
    if "embedding" not in doc:
        raise ValueError("chunk_dict must contain 'embedding'")

    return doc


def index_pdf_chunk(chunk_dict: dict) -> dict:
    settings = get_settings()
    doc = _to_typesense_doc(chunk_dict)
    return client.collections[settings.TYPESENSE_PDF_CHUNKS_COLLECTION].documents.upsert(doc)


def get_pdf_chunk_document(chunk_id: str) -> dict:
    settings = get_settings()
    return client.collections[settings.TYPESENSE_PDF_CHUNKS_COLLECTION].documents[chunk_id].retrieve()


def update_pdf_chunk_document(chunk_id: str, partial: dict) -> dict:
    settings = get_settings()
    doc = _to_typesense_doc({**partial, "id": chunk_id})
    return client.collections[settings.TYPESENSE_PDF_CHUNKS_COLLECTION].documents[chunk_id].update(doc)


def delete_pdf_chunk_document(chunk_id: str) -> dict:
    settings = get_settings()
    return client.collections[settings.TYPESENSE_PDF_CHUNKS_COLLECTION].documents[chunk_id].delete()


def bulk_index_pdf_chunks(chunk_dicts: list[dict]) -> list[dict]:
    settings = get_settings()
    docs = [_to_typesense_doc(chunk) for chunk in chunk_dicts]
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
    facet_by: str | None = "pdf_id,page,section",
    page: int = 1,
    per_page: int = 20,
    highlight_fields: str = "content",
    vector_query: str | None = None,
) -> dict:
    settings = get_settings()
    params: dict[str, Any] = {
        "q": query if mode in ("keyword", "hybrid") else "*",
        "page": page,
        "per_page": per_page,
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

    return client.collections[settings.TYPESENSE_PDF_CHUNKS_COLLECTION].documents.search(params)

