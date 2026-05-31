"""
Backend-side MCP client.

Connects to the mcp-server over streamable HTTP and invokes its retrieval tools.
All connection/transport failures surface as UpstreamUnavailableError (503) so
the RAG loop degrades cleanly when the MCP server is down.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from core.exceptions import UpstreamUnavailableError
from models.schemas import ChunkHit

logger = logging.getLogger(__name__)

# Test seam: (tool_name, arguments) -> already-extracted payload dict.
SessionFactory = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class McpClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = 30.0,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/mcp"
        self._timeout = timeout
        self._session_factory = session_factory

    # -- public API ---------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        mode: str = "hybrid",
        filter_by: str | None = None,
    ) -> list[ChunkHit]:
        payload = await self._call_tool(
            "search_legal_chunks",
            {"query": query, "top_k": top_k, "mode": mode, "filter_by": filter_by},
        )
        # FIX B2: a malformed payload (version skew, partial response) must not
        # leak a raw pydantic ValidationError — translate to the domain error.
        try:
            return self._hits_from_payload(payload)
        except Exception as exc:
            logger.error("Malformed MCP search payload: %s", exc)
            raise UpstreamUnavailableError(
                "Malformed response from the retrieval (MCP) server.", detail=str(exc)
            ) from exc

    async def get_document(self, chunk_id: str) -> dict[str, Any]:
        payload = await self._call_tool("get_document", {"chunk_id": chunk_id})
        return payload.get("document", {})

    # -- internals ----------------------------------------------------------

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if self._session_factory is not None:
                return await self._session_factory(name, arguments)
            return await self._call_remote(name, arguments)
        except UpstreamUnavailableError:
            raise
        except Exception as exc:
            logger.error("MCP tool '%s' failed: %s", name, exc)
            raise UpstreamUnavailableError(
                "The retrieval (MCP) server is unavailable.", detail=str(exc)
            ) from exc

    async def _call_remote(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        # Imported lazily so the module loads without the mcp client deps present.
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(self._url, timeout=self._timeout) as (
            read, write, _,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return self._extract(result)

    @staticmethod
    def _extract(result: Any) -> dict[str, Any]:
        """Pull the JSON payload out of an MCP CallToolResult."""
        if getattr(result, "isError", False):
            text = McpClient._first_text(result) or "unknown error"
            raise UpstreamUnavailableError("MCP tool returned an error.", detail=text)

        structured = getattr(result, "structuredContent", None)
        if structured:
            return structured

        text = McpClient._first_text(result)
        if text:
            return json.loads(text)
        return {}

    @staticmethod
    def _first_text(result: Any) -> str | None:
        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            if text:
                return text
        return None

    @staticmethod
    def _hits_from_payload(payload: dict[str, Any]) -> list[ChunkHit]:
        return [ChunkHit(**hit) for hit in payload.get("hits", [])]
