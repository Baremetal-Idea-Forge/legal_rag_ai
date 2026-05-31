"""
Embedding service — wraps models/embeddings_model.LegalBGEEmbeddingHelper.

The underlying SentenceTransformer model is large and slow to load, so the
helper is constructed lazily on first use (never at import or DI-wiring time).
A pre-built helper may be injected (tests, custom configs).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from core.exceptions import IngestionError, SearchError

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(
        self,
        helper: Any | None = None,
        *,
        helper_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._helper = helper
        self._factory = helper_factory

    def _get_helper(self) -> Any:
        if self._helper is None:
            if self._factory is None:
                # Imported lazily so importing this module never pulls torch.
                from models.embeddings_model import get_embedding_helper

                self._factory = get_embedding_helper
            logger.info("Loading embedding model (first use)…")
            self._helper = self._factory()
        return self._helper

    @property
    def dimension(self) -> int:
        return self._get_helper().embedding_dimension

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._get_helper().embed_query(text)
        except Exception as exc:
            raise SearchError("Failed to embed query.", detail=str(exc)) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._get_helper().embed_documents(texts)
        except Exception as exc:
            raise IngestionError("Failed to embed documents.", detail=str(exc)) from exc

    def embed_chunk_documents(
        self, docs: list[dict[str, Any]], *, text_key: str = "content"
    ) -> list[dict[str, Any]]:
        """Return new docs each with an added 'embedding' field."""
        try:
            return self._get_helper().embed_pdf_chunks(docs, text_key=text_key)
        except Exception as exc:
            raise IngestionError(
                "Failed to embed chunk documents.", detail=str(exc)
            ) from exc
