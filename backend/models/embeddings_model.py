from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LegalBGEEmbeddingHelper:
    """
    Helper around a SentenceTransformer-compatible legal embedding model.

    Default model is configurable via EMBEDDING_MODEL_NAME.
    This helper:
      - loads the model once
      - encodes queries and documents
      - returns plain Python lists for Typesense
      - can embed a list of PDF chunks in one pass
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_NAME",
            "yuriyvnv/legal-bge-m3",
        )
        self.device = device or os.getenv("EMBEDDING_DEVICE")
        self.normalize_embeddings = normalize_embeddings

        logger.info("Loading embedding model: %s", self.model_name)
        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device

        self.model = SentenceTransformer(self.model_name, **kwargs)
        self.embedding_dimension = self.model.get_embedding_dimension()

        logger.info(
            "Embedding model loaded: %s (dim=%s)",
            self.model_name,
            self.embedding_dimension,
        )

    def _encode_texts(
        self,
        texts: list[str],
        *,
        task: str,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        if not texts:
            return []

        encode_fn = None
        if task == "query" and hasattr(self.model, "encode_query"):
            encode_fn = self.model.encode_query
        elif task == "document" and hasattr(self.model, "encode_document"):
            encode_fn = self.model.encode_document

        if encode_fn is None:
            encode_fn = self.model.encode

        embeddings = encode_fn(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        if isinstance(embeddings, np.ndarray):
            return embeddings.astype(float).tolist()

        return [np.asarray(vec, dtype=float).tolist() for vec in embeddings]

    def embed_query(self, text: str) -> list[float]:
        text = self._clean_text(text)
        return self._encode_texts([text], task="query")[0]

    def embed_document(self, text: str) -> list[float]:
        text = self._clean_text(text)
        return self._encode_texts([text], task="document")[0]

    def embed_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        cleaned = [self._clean_text(t) for t in texts]
        return self._encode_texts(
            cleaned,
            task="document",
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

    def embed_pdf_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        text_key: str = "content",
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Takes chunk dictionaries, embeds their text, and returns new dicts
        with an 'embedding' field added.
        """
        if not chunks:
            return []

        texts = [self._clean_text(chunk.get(text_key, "")) for chunk in chunks]
        vectors = self.embed_documents(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        enriched: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors):
            new_chunk = dict(chunk)
            new_chunk["embedding"] = vector
            enriched.append(new_chunk)

        return enriched

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join((text or "").split())


@lru_cache(maxsize=1)
def get_embedding_helper(model_name: str | None = None,
        device: str | None = None,
        normalize_embeddings: bool = True) -> LegalBGEEmbeddingHelper:
    return LegalBGEEmbeddingHelper(model_name, device, normalize_embeddings)