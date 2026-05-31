"""
Legal RAG — MCP retrieval server.

Exposes read-side retrieval as MCP tools (`Model ⟷ MCP ⟷ Embedding/Typesense`).
To stay DRY it imports the backend's retrieval primitives directly, so there is
exactly one SearchService, one embedding model, and one Typesense config across
the whole system.

Run:
    uv run python mcp-server/server.py
The server listens on MCP_SERVER_URL (default http://localhost:9000) at /mcp
using the streamable-HTTP transport.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Reuse the backend's retrieval stack (single source of truth).
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.dependencies import (  # noqa: E402
    get_embedding_service,
    get_typesense_repository,
)
from services.search_service import SearchService  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("mcp-server")

_settings = get_settings()
_search: SearchService | None = None


def _get_search() -> SearchService:
    """Lazily build (and cache) the SearchService — model loads on first call."""
    global _search
    if _search is None:
        _search = SearchService(
            embedding_service=get_embedding_service(),
            typesense_repo=get_typesense_repository(),
            settings=_settings,
        )
    return _search


# -- Tool implementations (pure-ish; unit-testable) -------------------------

def search_legal_chunks_impl(
    query: str, top_k: int = 5, mode: str = "hybrid", filter_by: str | None = None
) -> dict[str, Any]:
    resp = _get_search().search(query, top_k=top_k, mode=mode, filter_by=filter_by)
    return {"hits": [hit.model_dump() for hit in resp.hits]}


def get_document_impl(chunk_id: str) -> dict[str, Any]:
    return {"document": get_typesense_repository().get_document(chunk_id)}


# -- MCP server + tool registration -----------------------------------------

_parsed = urlparse(_settings.MCP_SERVER_URL)
mcp = FastMCP(
    "legal-rag-retrieval",
    host=_parsed.hostname or "127.0.0.1",
    port=_parsed.port or 9000,
)


@mcp.tool()
def search_legal_chunks(
    query: str, top_k: int = 5, mode: str = "hybrid", filter_by: str | None = None
) -> dict[str, Any]:
    """Hybrid (keyword + vector) search over indexed legal chunks.

    Returns {"hits": [ChunkHit, ...]} ordered by relevance.
    """
    return search_legal_chunks_impl(query, top_k, mode, filter_by)


@mcp.tool()
def get_document(chunk_id: str) -> dict[str, Any]:
    """Fetch a single indexed chunk document by its id. Returns {"document": {...}}."""
    return get_document_impl(chunk_id)


if __name__ == "__main__":
    logger.info("Starting Legal RAG MCP server at %s/mcp", _settings.MCP_SERVER_URL)
    mcp.run(transport="streamable-http")
