"""
Search service — the read path.

    embed query → hybrid Typesense search → ranked ChunkHit[]

For vector/hybrid modes the query embedding is formatted into a Typesense
`vector_query` string (`embedding:([...], k:N)`).

`search_two_stage` adds document scoping on top: retrieve a wide band of
candidates, aggregate their scores per parent document, then re-retrieve spans
restricted to the winning documents. Single-stage retrieval ranks every chunk in
the corpus against the query, so a provision that is textually similar but sits
in the wrong statute outranks the right one — document-level retrieval mismatch.
Scoping stage 2 removes those chunks from contention entirely.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import Settings
from models.schemas import ChunkHit, DocumentScope, SearchResponse
from repositories.typesense_repository import TypesenseRepository
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

_VECTOR_MODES = {"vector", "hybrid"}


class SearchService:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        typesense_repo: TypesenseRepository,
        settings: Settings,
    ) -> None:
        self._embeddings = embedding_service
        self._typesense = typesense_repo
        self._settings = settings

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: str = "hybrid",
        filter_by: str | None = None,
    ) -> SearchResponse:
        top_k = top_k or self._settings.RAG_TOP_K

        vector_query: str | None = None
        if mode in _VECTOR_MODES:
            embedding = self._embeddings.embed_query(query)
            # Log embedding size for debugging
            logger.debug("Query embedding: %d dims, first 3 values: %.6f, %.6f, %.6f",
                        len(embedding), embedding[0], embedding[1], embedding[2])
            vector_query = self._build_vector_query(embedding, top_k)

        raw = self._typesense.search(
            query,
            mode=mode,
            top_k=top_k,
            filter_by=filter_by,
            vector_query=vector_query,
        )

        hits = [self._to_hit(h) for h in raw.get("hits", [])]
        logger.info("Search '%s' (%s) → %d hits", query[:60], mode, len(hits))

        # Log detailed scores for each hit
        for i, hit in enumerate(hits[:5]):
            logger.debug(
                "  Hit %d: %s (chunk %d, score=%.4f, text_match=%s, vector_dist=%s)",
                i + 1, hit.pdf_name, hit.chunk_index, hit.score or 0.0,
                hit.text_match, hit.vector_distance,
            )

        return SearchResponse(query=query, mode=mode, count=len(hits), hits=hits)

    def search_two_stage(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: str = "hybrid",
        filter_by: str | None = None,
    ) -> tuple[SearchResponse, list[DocumentScope]]:
        """
        Document-scoped retrieval. Returns the stage-2 spans plus the ranked
        document scopes that produced them (needed for the abstention decision
        and for the audit record).

        An empty scope list means stage 1 found nothing; the caller decides
        whether that is an abstention.
        """
        settings = self._settings
        top_k = top_k or settings.RAG_TOP_K

        stage1 = self.search(
            query,
            top_k=settings.TWO_STAGE_CANDIDATE_K,
            mode=mode,
            filter_by=filter_by,
        )
        scopes = self._aggregate_documents(
            stage1.hits, strategy=settings.TWO_STAGE_DOC_SCORE
        )
        if not scopes:
            logger.info("Two-stage: stage 1 returned no candidates for '%s'", query[:60])
            return SearchResponse(query=query, mode=mode, count=0, hits=[]), []

        selected = scopes[: max(1, settings.TWO_STAGE_DOC_COUNT)]
        logger.info(
            "Two-stage: %d candidates → %d docs scoped %s",
            len(stage1.hits),
            len(selected),
            [(s.pdf_name, round(s.score, 4)) for s in selected],
        )

        scope_filter = self._scope_filter([s.pdf_id for s in selected])
        combined = f"({filter_by}) && {scope_filter}" if filter_by else scope_filter
        stage2 = self.search(query, top_k=top_k, mode=mode, filter_by=combined)
        return stage2, scopes

    @staticmethod
    def _aggregate_documents(
        hits: list[ChunkHit], *, strategy: str = "max"
    ) -> list[DocumentScope]:
        """
        Collapse chunk hits into per-document scores, best document first.

        "max" takes the single strongest chunk — it rewards one precise hit.
        "sum_top3" adds the best three, rewarding a document that is relevant
        throughout rather than at one coincidental phrase. Which wins is
        corpus-dependent; benchmark both before choosing.
        """
        grouped: dict[str, list[ChunkHit]] = {}
        for hit in hits:
            if hit.score is None:
                continue
            grouped.setdefault(hit.pdf_id, []).append(hit)

        scopes: list[DocumentScope] = []
        for pdf_id, doc_hits in grouped.items():
            scores = sorted((h.score for h in doc_hits), reverse=True)  # type: ignore[misc]
            score = sum(scores[:3]) if strategy == "sum_top3" else scores[0]
            scopes.append(
                DocumentScope(
                    pdf_id=pdf_id,
                    pdf_name=doc_hits[0].pdf_name,
                    score=float(score),
                    chunk_count=len(doc_hits),
                )
            )

        scopes.sort(key=lambda s: s.score, reverse=True)
        return scopes

    @staticmethod
    def _scope_filter(pdf_ids: list[str]) -> str:
        """Typesense filter restricting results to the scoped documents."""
        joined = ",".join(pdf_ids)
        return f"pdf_id:=[{joined}]"

    @staticmethod
    def _build_vector_query(embedding: list[float], top_k: int) -> str:
        vector = ",".join(repr(float(x)) for x in embedding)
        return f"embedding:([{vector}], k:{top_k})"

    @staticmethod
    def _to_hit(hit: dict[str, Any]) -> ChunkHit:
        doc = hit.get("document", {})
        text_match = hit.get("text_match")
        vector_distance = hit.get("vector_distance")

        if vector_distance is not None:
            score: float | None = round(1.0 - float(vector_distance), 6)
        elif text_match is not None:
            score = float(text_match)
        else:
            score = None

        # page_start/page_end are optional in the schema; fall back to `page`.
        page = doc.get("page")
        page_start = doc.get("page_start", page)
        page_end = doc.get("page_end", page)

        return ChunkHit(
            id=doc.get("id", ""),
            pdf_id=doc.get("pdf_id", ""),
            pdf_name=doc.get("pdf_name", ""),
            page_start=page_start if page_start is not None else 0,
            page_end=page_end if page_end is not None else 0,
            chunk_index=doc.get("chunk_index", 0),
            content=doc.get("content", ""),
            file_url=doc.get("file_url"),
            score=score,
            text_match=text_match,
            vector_distance=vector_distance,
            start_char=doc.get("start_char", -1),
            end_char=doc.get("end_char", -1),
        )
