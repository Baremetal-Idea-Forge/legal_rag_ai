"""
Data-access layer for Typesense.

Thin wrapper over helpers/typesense_helper that owns NO business logic — it
only translates calls + Typesense exceptions into our domain vocabulary.
"""

from __future__ import annotations

import logging
from typing import Any

from typesense.exceptions import ObjectNotFound, TypesenseClientError

import helpers.typesense_helper as ts
from core.exceptions import (
    DocumentNotFoundError,
    IngestionError,
    SearchError,
    UpstreamUnavailableError,
)

logger = logging.getLogger(__name__)


class TypesenseRepository:
    """Persistence + retrieval of PDF chunk documents."""

    def ensure_collection(self) -> None:
        try:
            ts.ensure_tasks_collection()
        except TypesenseClientError as exc:
            raise UpstreamUnavailableError(
                "Typesense is unavailable while ensuring the collection.",
                detail=str(exc),
            ) from exc

    def is_ready(self) -> bool:
        """Best-effort readiness probe — never raises."""
        try:
            return ts.collection_exists()
        except Exception as exc:  # readiness must not propagate
            logger.warning("Typesense readiness check failed: %s", exc)
            return False

    def index_chunks(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not docs:
            return []
        try:
            results = ts.bulk_index_pdf_chunks(docs)
        except TypesenseClientError as exc:
            raise UpstreamUnavailableError(
                "Typesense is unavailable while indexing chunks.", detail=str(exc)
            ) from exc

        failures = [r for r in results if not r.get("success", True)]
        if failures:
            raise IngestionError(
                f"{len(failures)} of {len(docs)} chunks failed to index.",
                detail=failures[:5],
            )
        return results

    def get_document(self, chunk_id: str) -> dict[str, Any]:
        try:
            return ts.get_pdf_chunk_document(chunk_id)
        except ObjectNotFound as exc:
            raise DocumentNotFoundError(
                f"Chunk '{chunk_id}' not found.", detail=chunk_id
            ) from exc
        except TypesenseClientError as exc:
            raise UpstreamUnavailableError(
                "Typesense is unavailable while fetching a document.", detail=str(exc)
            ) from exc

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid",
        top_k: int = 5,
        filter_by: str | None = None,
        vector_query: str | None = None,
        page: int = 1,
    ) -> dict[str, Any]:
        try:
            return ts.search_pdf_chunks(
                query,
                mode=mode,
                per_page=top_k,
                page=page,
                filter_by=filter_by,
                vector_query=vector_query,
            )
        except ValueError as exc:
            # Bad mode / missing vector_query — a caller error, not upstream.
            raise SearchError(str(exc)) from exc
        except TypesenseClientError as exc:
            raise UpstreamUnavailableError(
                "Typesense is unavailable while searching.", detail=str(exc)
            ) from exc
