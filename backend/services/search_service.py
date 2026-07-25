"""
Search service — the read path.

    embed query → hybrid Typesense search → ranked ChunkHit[]

For vector/hybrid modes the query embedding is formatted into a Typesense
`vector_query` string (`embedding:([...], k:N)`).
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import Settings
from models.schemas import ChunkHit, SearchResponse
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
        )
