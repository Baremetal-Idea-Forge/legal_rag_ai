"""
RAG service — the orchestration of retrieval + generation.

    retrieve → select context → grounded prompt → LLM → verify → cited answer

Retrieval runs in a threadpool (it embeds + hits Typesense synchronously) so the
async event loop is never blocked. When retrieval yields nothing, the LLM is not
called at all — we return a fixed "not found" answer (no hallucination).

Feature-flagged stages (all off by default, see core.config):

  - EXACT_TERM_ROUTING_ENABLED — L6 routing: dense-only retrieval by default,
    the lexical (hybrid) channel only when the exact-term detector flags the
    query. Off = always hybrid (pre-v2 behavior).
  - GATE1_MIN_SCORE — Gate 1: abstain before any generation call when the best
    dense score is below the floor, optionally after one rewrite-and-retry
    (GATE1_REWRITE_RETRY) — the only place query rewriting exists in v2.
  - TWO_STAGE_ENABLED — document-scoped retrieval with a stage-1 abstention
    floor (ABSTAIN_MIN_DOC_SCORE).
  - VERIFY_ENABLED — triad verification gated on min(scores); on failure the
    service abstains with best-effort sources (Gate 2, abstain-first — v2
    removed the regenerate loop, R8). Streaming skips this gate: its tokens
    are already with the client before a judge could run.
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
from services.exact_terms import detect_exact_terms
from services.search_service import SearchService
from services.verify import AnswerVerifier

logger = logging.getLogger(__name__)


def _best_dense_score(hits: list[ChunkHit]) -> float | None:
    """
    Best dense similarity (1 - vector_distance) among the hits, or None when
    no hit carries one. Gate 1 floors on this plane only — keyword text_match
    scores are unbounded integers and cannot share a floor with it.
    """
    scores = [
        h.score
        for h in hits
        if h.vector_distance is not None and h.score is not None
    ]
    return max(scores) if scores else None


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
        used, scopes, abstain_reason, mode = await self._retrieve_context(
            query, top_k
        )
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
                    retrieval_mode=mode,
                )
            )

        context_text = prompts.format_context(used)
        messages = prompts.build_messages(query, context_text)
        answer_text = await self._llm.chat(
            messages, temperature=self._settings.LLM_TEMPERATURE
        )

        verification = await self._verify(query, context_text, answer_text)
        gate = self._settings.VERIFY_MIN_TRIAD_SCORE
        if verification is not None and verification.minimum < gate:
            # Gate 2, abstain-first: an answer that failed the triad is
            # declined, never regenerated (v2 non-goal R8). The best-effort
            # spans stay attached so a reviewer sees what was looked at.
            logger.info(
                "Verification gate failed (min %.2f < %s); abstaining.",
                verification.minimum, gate,
            )
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
                        f"below gate {gate}"
                    ),
                    verification=verification,
                    scoped_documents=scopes,
                    retrieval_mode=mode,
                )
            )

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
                retrieval_mode=mode,
            )
        )

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
        used, scopes, abstain_reason, mode = await self._retrieve_context(
            query, top_k
        )
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
                    retrieval_mode=mode,
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
                retrieval_mode=mode,
            )
        )

    # -- internals ----------------------------------------------------------

    async def _retrieve_context(
        self, query: str, top_k: int | None
    ) -> tuple[list[ChunkHit], list[DocumentScope], str | None, str]:
        """
        Route, retrieve, gate, and budget-select context.

        Returns (selected hits, document scopes, abstain reason, search mode).
        A non-None reason means retrieval declined before generation — either
        the two-stage stage-1 floor or Gate 1; scopes then carry what was seen
        (stage-1 documents or the Gate-1 nearest documents) for the audit
        record.
        """
        settings = self._settings
        top_k = top_k or settings.RAG_TOP_K
        scopes: list[DocumentScope] = []
        mode = self._route_mode(query)

        if settings.TWO_STAGE_ENABLED and not (
            settings.RETRIEVAL_VIA_MCP and self._mcp is not None
        ):
            result, scopes = await run_in_threadpool(
                self._search.search_two_stage, query, top_k=top_k, mode=mode
            )
            if not scopes:
                return [], [], "stage 1 retrieval found no candidate documents", mode
            floor = settings.ABSTAIN_MIN_DOC_SCORE
            if floor > 0 and scopes[0].score < floor:
                return [], scopes, (
                    f"top document score {scopes[0].score:.4f} below the "
                    f"abstention floor {floor}"
                ), mode
            hits = result.hits
        else:
            hits = await self._search_once(query, top_k=top_k, mode=mode)
            hits, gate_reason = await self._gate1(query, hits, top_k=top_k, mode=mode)
            if gate_reason:
                # Abstain with the nearest documents attached — the review
                # trail Gate 1 owes for a zero-generation decline.
                nearest = SearchService._aggregate_documents(hits, strategy="max")
                return [], nearest, gate_reason, mode

        selected = prompts.select_hits_within(
            hits, max_chars=settings.RAG_MAX_CONTEXT_CHARS
        )
        logger.info(
            "Retrieved %d → selected %d (mode=%s, max_chars=%s) for query '%s'",
            len(hits),
            len(selected),
            mode,
            settings.RAG_MAX_CONTEXT_CHARS,
            query[:60],
        )
        return selected, scopes, None, mode

    def _route_mode(self, query: str) -> str:
        """
        L6 routing: dense-only by default, the lexical (hybrid) channel only
        when the exact-term detector fires. With the flag off, every query
        stays on the pre-v2 hybrid path.
        """
        if not self._settings.EXACT_TERM_ROUTING_ENABLED:
            return "hybrid"
        flag = detect_exact_terms(query)
        if flag.flagged:
            logger.info(
                "Exact-term route fired (%s) for '%s'",
                ",".join(flag.reasons), query[:60],
            )
            return "hybrid"
        return "vector"

    async def _search_once(
        self, query: str, *, top_k: int, mode: str
    ) -> list[ChunkHit]:
        """One retrieval pass over whichever read path is configured."""
        if self._settings.RETRIEVAL_VIA_MCP and self._mcp is not None:
            # Read path routed through the MCP server (Step 3 topology).
            return await self._mcp.search(query, top_k=top_k, mode=mode)
        # Direct retrieval; sync call offloaded so the event loop is free.
        result = await run_in_threadpool(
            self._search.search, query, top_k=top_k, mode=mode
        )
        return result.hits

    async def _gate1(
        self, query: str, hits: list[ChunkHit], *, top_k: int, mode: str
    ) -> tuple[list[ChunkHit], str | None]:
        """
        Gate 1 — retrieval-score floor (ACTION_PLAN v2 §2.3 [Q6]).

        Compares the best dense chunk score (1 - vector_distance) against
        GATE1_MIN_SCORE; keyword text_match scores are unbounded integers and
        cannot share the floor, so hit sets without a dense score skip the
        gate. Below the floor, one rewrite-and-re-retrieve attempt runs
        (GATE1_REWRITE_RETRY) before abstention. Returns (hits, abstain
        reason); a non-None reason means abstain without calling the LLM.
        """
        floor = self._settings.GATE1_MIN_SCORE
        if floor <= 0:
            return hits, None
        best = _best_dense_score(hits)
        if best is None:
            # Nothing to floor: empty retrieval (the caller's no-context
            # abstention covers it) or keyword-only hits without a dense score.
            return hits, None
        if best >= floor:
            return hits, None

        if self._settings.GATE1_REWRITE_RETRY:
            rewritten = await self._rewrite_query(query)
            if rewritten != query:
                retry_hits = await self._search_once(
                    rewritten, top_k=top_k, mode=mode
                )
                retry_best = _best_dense_score(retry_hits)
                if retry_best is not None and retry_best >= floor:
                    logger.info(
                        "Gate 1 cleared on rewrite (%.4f → %.4f) for '%s'",
                        best, retry_best, query[:60],
                    )
                    return retry_hits, None

        return hits, (
            f"best retrieval score {best:.4f} below the Gate-1 floor {floor}"
        )

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
            "retrieval_mode": response.retrieval_mode,
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
                "exact_term_routing": settings.EXACT_TERM_ROUTING_ENABLED,
                "gate1_min_score": settings.GATE1_MIN_SCORE,
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
