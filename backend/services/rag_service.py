"""
RAG service — the orchestration of retrieval + generation.

    retrieve (hybrid) → select context → grounded prompt → Ollama → cited answer

Retrieval runs in a threadpool (it embeds + hits Typesense synchronously) so the
async event loop is never blocked. When retrieval yields nothing, the LLM is not
called at all — we return a fixed "not found" answer (no hallucination).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from starlette.concurrency import run_in_threadpool

from clients.mcp_client import McpClient
from clients.ollama_client import OllamaClient
from core.config import Settings
from models.schemas import ChatResponse, Citation, ChunkHit
from services import prompts
from services.search_service import SearchService

logger = logging.getLogger(__name__)


class RagService:
    def __init__(
        self,
        *,
        search_service: SearchService,
        ollama_client: OllamaClient,
        settings: Settings,
        mcp_client: McpClient | None = None,
    ) -> None:
        self._search = search_service
        self._ollama = ollama_client
        self._settings = settings
        self._mcp = mcp_client

    # -- non-streaming ------------------------------------------------------

    async def answer(self, query: str, *, top_k: int | None = None) -> ChatResponse:
        used = await self._retrieve_context(query, top_k)
        if not used:
            logger.info("No context for query '%s' — declining.", query[:60])
            return ChatResponse(
                query=query, answer=prompts.NO_CONTEXT_ANSWER, citations=[], chunks_used=[]
            )

        messages = prompts.build_messages(query, prompts.format_context(used))
        answer_text = await self._ollama.chat(messages)

        return ChatResponse(
            query=query,
            answer=answer_text,
            citations=[self._to_citation(h) for h in used],
            chunks_used=used,
        )

    # -- streaming ----------------------------------------------------------

    async def stream_answer(
        self, query: str, *, top_k: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Yields event dicts: {"type": "token", "data": str} for each token, then
        a terminal {"type": "citations", "data": [...]}.
        """
        used = await self._retrieve_context(query, top_k)
        if not used:
            yield {"type": "token", "data": prompts.NO_CONTEXT_ANSWER}
            yield {"type": "citations", "data": []}
            return

        messages = prompts.build_messages(query, prompts.format_context(used))
        async for token in self._ollama.stream_chat(messages):
            yield {"type": "token", "data": token}
        yield {
            "type": "citations",
            "data": [self._to_citation(h).model_dump() for h in used],
        }

    # -- internals ----------------------------------------------------------

    async def _retrieve_context(
        self, query: str, top_k: int | None
    ) -> list[ChunkHit]:
        top_k = top_k or self._settings.RAG_TOP_K

        if self._settings.RETRIEVAL_VIA_MCP and self._mcp is not None:
            # Read path routed through the MCP server (Step 3 topology).
            hits = await self._mcp.search(query, top_k=top_k, mode="hybrid")
        else:
            # Direct retrieval; sync call offloaded so the event loop is free.
            result = await run_in_threadpool(
                self._search.search, query, top_k=top_k, mode="hybrid"
            )
            hits = result.hits

        return prompts.select_hits_within(
            hits, max_chars=self._settings.RAG_MAX_CONTEXT_CHARS
        )

    @staticmethod
    def _to_citation(hit: ChunkHit) -> Citation:
        return Citation(
            pdf_id=hit.pdf_id,
            pdf_name=hit.pdf_name,
            page_start=hit.page_start,
            page_end=hit.page_end,
            chunk_index=hit.chunk_index,
            file_url=hit.file_url,
        )
