"""
RAG service — the orchestration of retrieval + generation.

    retrieve → select context → grounded prompt → LLM → verify → cited answer

Retrieval runs in a threadpool (it embeds + hits Typesense synchronously) so the
async event loop is never blocked. When retrieval yields nothing, the LLM is not
called at all — we return a fixed "not found" answer (no hallucination).

Feature-flagged stages (all off by default, see core.config):

  - TWO_STAGE_ENABLED — document-scoped retrieval with a stage-1 abstention
    floor (ABSTAIN_MIN_DOC_SCORE).
  - VERIFY_ENABLED — triad verification gated on min(scores); on failure the
    query is rewritten and retried, capped at VERIFY_MAX_RETRIES, then the
    service abstains rather than emit an unverified answer. Streaming skips
    this gate: its tokens are already with the client before a judge could run.
  - AUDIT_LOG_ENABLED — one JSONL record per response, written before the
    response is returned.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from starlette.concurrency import run_in_threadpool

from clients.mcp_client import McpClient
from core.config import Settings
from models.schemas import ChatResponse, ChunkHit, DocumentScope, VerificationScores
from services import prompts
from services.audit import AuditLog
from services.cite import bind_citations
from services.search_service import SearchService
from services.verify import AnswerVerifier

logger = logging.getLogger(__name__)


class RagService:
    def __init__(
        self,
        *,
        search_service: SearchService,
        llm_client: Any,
        settings: Settings,
        mcp_client: McpClient | None = None,
        verifier: AnswerVerifier | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self._search = search_service
        self._llm = llm_client
        self._settings = settings
        self._mcp = mcp_client
        self._verifier = verifier
        self._audit = audit_log

    # -- non-streaming ------------------------------------------------------

    async def answer(self, query: str, *, top_k: int | None = None) -> ChatResponse:
        used, scopes, abstain_reason = await self._retrieve_context(query, top_k)
        if abstain_reason or not used:
            reason = abstain_reason or "retrieval returned no usable context"
            logger.info("Abstaining on '%s': %s", query[:60], reason)
            return self._respond(
                ChatResponse(
                    query=query,
                    answer=prompts.NO_CONTEXT_ANSWER,
                    citations=[],
                    chunks_used=[],
                    abstained=True,
                    abstain_reason=reason,
                    scoped_documents=scopes,
                )
            )

        current_query = query
        attempts = 0
        while True:
            context_text = prompts.format_context(used)
            messages = prompts.build_messages(current_query, context_text)
            answer_text = await self._llm.chat(
                messages, temperature=self._settings.LLM_TEMPERATURE
            )

            verification = await self._verify(query, context_text, answer_text)
            gate = self._settings.VERIFY_MIN_TRIAD_SCORE
            if verification is None or verification.minimum >= gate:
                citations, coverage = bind_citations(answer_text, used)
                return self._respond(
                    ChatResponse(
                        query=query,
                        answer=answer_text,
                        citations=citations,
                        chunks_used=used,
                        citation_coverage=coverage,
                        verification=verification,
                        scoped_documents=scopes,
                    )
                )

            if attempts >= self._settings.VERIFY_MAX_RETRIES:
                # Abstain, but attach the best-effort spans so a reviewer can
                # see what the system looked at before declining.
                citations, _ = bind_citations(answer_text, used)
                return self._respond(
                    ChatResponse(
                        query=query,
                        answer=prompts.UNVERIFIED_ANSWER,
                        citations=citations,
                        chunks_used=used,
                        citation_coverage=0.0,
                        abstained=True,
                        abstain_reason=(
                            f"verification minimum {verification.minimum:.2f} "
                            f"below gate {gate} after {attempts + 1} attempts"
                        ),
                        verification=verification,
                        scoped_documents=scopes,
                    )
                )

            attempts += 1
            logger.info(
                "Verification gate failed (min %.2f < %s); attempt %d/%d with "
                "a rewritten query.",
                verification.minimum, gate, attempts,
                self._settings.VERIFY_MAX_RETRIES,
            )
            current_query = await self._rewrite_query(current_query)
            refreshed, new_scopes, retry_abstain = await self._retrieve_context(
                current_query, top_k
            )
            # A rewrite that retrieves nothing keeps the previous context —
            # regenerating against it can still clear the gate.
            if refreshed and not retry_abstain:
                used, scopes = refreshed, new_scopes

    # -- streaming ----------------------------------------------------------

    async def stream_answer(
        self, query: str, *, top_k: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Yields event dicts: {"type": "token", "data": str} for each token, then
        a terminal {"type": "citations", "data": [...]}.

        Verification does not run here — tokens are already with the client
        before any judge could score them. Use `answer` where the gate matters.
        """
        used, scopes, abstain_reason = await self._retrieve_context(query, top_k)
        if abstain_reason or not used:
            reason = abstain_reason or "retrieval returned no usable context"
            logger.info("Abstaining on streaming '%s': %s", query[:60], reason)
            yield {"type": "token", "data": prompts.NO_CONTEXT_ANSWER}
            yield {"type": "citations", "data": []}
            self._respond(
                ChatResponse(
                    query=query,
                    answer=prompts.NO_CONTEXT_ANSWER,
                    citations=[],
                    chunks_used=[],
                    abstained=True,
                    abstain_reason=reason,
                    scoped_documents=scopes,
                )
            )
            return

        context_text = prompts.format_context(used)
        messages = prompts.build_messages(query, context_text)

        parts: list[str] = []
        async for token in self._llm.stream_chat(
            messages, temperature=self._settings.LLM_TEMPERATURE
        ):
            parts.append(token)
            yield {"type": "token", "data": token}

        answer_text = "".join(parts)
        citations, coverage = bind_citations(answer_text, used)
        yield {
            "type": "citations",
            "data": [c.model_dump() for c in citations],
        }
        self._respond(
            ChatResponse(
                query=query,
                answer=answer_text,
                citations=citations,
                chunks_used=used,
                citation_coverage=coverage,
                scoped_documents=scopes,
            )
        )

    # -- internals ----------------------------------------------------------

    async def _retrieve_context(
        self, query: str, top_k: int | None
    ) -> tuple[list[ChunkHit], list[DocumentScope], str | None]:
        """
        Retrieve and budget-select context.

        Returns (selected hits, stage-1 document scopes, abstain reason).
        A non-None reason means the two-stage path declined before span
        retrieval; scopes may still carry what stage 1 saw, for the audit
        record.
        """
        settings = self._settings
        top_k = top_k or settings.RAG_TOP_K
        scopes: list[DocumentScope] = []

        if settings.RETRIEVAL_VIA_MCP and self._mcp is not None:
            # Read path routed through the MCP server (Step 3 topology).
            hits = await self._mcp.search(query, top_k=top_k, mode="hybrid")
        elif settings.TWO_STAGE_ENABLED:
            result, scopes = await run_in_threadpool(
                self._search.search_two_stage, query, top_k=top_k, mode="hybrid"
            )
            if not scopes:
                return [], [], "stage 1 retrieval found no candidate documents"
            floor = settings.ABSTAIN_MIN_DOC_SCORE
            if floor > 0 and scopes[0].score < floor:
                return [], scopes, (
                    f"top document score {scopes[0].score:.4f} below the "
                    f"abstention floor {floor}"
                )
            hits = result.hits
        else:
            # Direct retrieval; sync call offloaded so the event loop is free.
            result = await run_in_threadpool(
                self._search.search, query, top_k=top_k, mode="hybrid"
            )
            hits = result.hits

        selected = prompts.select_hits_within(
            hits, max_chars=settings.RAG_MAX_CONTEXT_CHARS
        )
        logger.info(
            "Retrieved %d → selected %d (max_chars=%s) for query '%s'",
            len(hits),
            len(selected),
            settings.RAG_MAX_CONTEXT_CHARS,
            query[:60],
        )
        return selected, scopes, None

    async def _verify(
        self, query: str, context_text: str, answer_text: str
    ) -> VerificationScores | None:
        """Triad scores, or None when verification is off or the judge failed."""
        if not self._settings.VERIFY_ENABLED or self._verifier is None:
            return None
        return await self._verifier.verify(
            query=query, context=context_text, answer=answer_text
        )

    async def _rewrite_query(self, query: str) -> str:
        try:
            rewritten = await self._llm.chat(
                prompts.build_rewrite_messages(query), temperature=0.0
            )
        except Exception as exc:  # noqa: BLE001 — loop machinery, never fatal
            logger.warning("Query rewrite failed (%s); keeping the original.", exc)
            return query
        rewritten = " ".join((rewritten or "").split())
        return rewritten or query

    def _respond(self, response: ChatResponse) -> ChatResponse:
        """Audit (when enabled) and hand back the response."""
        if self._settings.AUDIT_LOG_ENABLED and self._audit is not None:
            self._audit.record(self._audit_entry(response))
        return response

    def _audit_entry(self, response: ChatResponse) -> dict[str, Any]:
        settings = self._settings
        return {
            "query": response.query,
            "answer": response.answer,
            "abstained": response.abstained,
            "abstain_reason": response.abstain_reason,
            "citation_coverage": response.citation_coverage,
            "verification": (
                response.verification.model_dump() if response.verification else None
            ),
            "scoped_documents": [d.model_dump() for d in response.scoped_documents],
            "chunks": [
                {
                    "id": hit.id,
                    "pdf_id": hit.pdf_id,
                    "chunk_index": hit.chunk_index,
                    "start_char": hit.start_char,
                    "end_char": hit.end_char,
                    "score": hit.score,
                }
                for hit in response.chunks_used
            ],
            "plan": {
                "two_stage": settings.TWO_STAGE_ENABLED,
                "verify": settings.VERIFY_ENABLED,
                "top_k": settings.RAG_TOP_K,
            },
            "model": {
                "provider": settings.LLM_PROVIDER,
                "name": (
                    settings.GEMINI_MODEL
                    if settings.LLM_PROVIDER == "gemini"
                    else settings.OLLAMA_MODEL
                ),
            },
        }
