from __future__ import annotations

import inspect
import logging
import os
from functools import lru_cache
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# kwargs that SentenceTransformer.encode accepts and we want to forward
_STANDARD_ENCODE_KWARGS = frozenset(
    {"batch_size", "show_progress_bar", "convert_to_numpy", "normalize_embeddings"}
)


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

        # Full set of kwargs we want to forward
        all_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "show_progress_bar": show_progress_bar,
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize_embeddings,
        }

        if encode_fn is None:
            # Standard SentenceTransformer.encode accepts all our kwargs
            encode_fn = self.model.encode
            kwargs = all_kwargs
        else:
            # FIX B1: custom encode_query/encode_document may have a different
            # signature (e.g. no batch_size, normalize_embeddings).  Inspect the
            # callable and only forward kwargs it actually accepts.
            try:
                sig = inspect.signature(encode_fn)
                accepts_var_keyword = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if accepts_var_keyword:
                    kwargs = all_kwargs
                else:
                    kwargs = {
                        k: v for k, v in all_kwargs.items() if k in sig.parameters
                    }
            except (ValueError, TypeError):
                # Cannot introspect (e.g. built-in); pass no extra kwargs
                kwargs = {}

        embeddings = encode_fn(texts, **kwargs)

        # FIX B2: normalise the return shape.  Some models / custom methods
        # return a 1-D array of shape (dim,) for a single input rather than
        # the expected (1, dim).  That makes tolist() produce a flat
        # [float, ...] instead of [[float, ...]], which breaks embed_query /
        # embed_document (they'd get a float back instead of a list).
        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
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

        FIX B5: warn when a chunk's text field is missing or empty after
        cleaning so callers can detect data quality issues early.
        """
        if not chunks:
            return []

        texts: list[str] = []
        for i, chunk in enumerate(chunks):
            raw = chunk.get(text_key, "")
            cleaned = self._clean_text(raw)
            if not cleaned:
                logger.warning(
                    "embed_pdf_chunks: chunk %d has empty text for key '%s'; "
                    "embedding will be computed for an empty string.",
                    i,
                    text_key,
                )
            texts.append(cleaned)

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


# FIX B4: maxsize=1 means a second call with different args evicts the first
# cached model, forcing an expensive reload.  Use maxsize=None (unbounded) so
# every distinct configuration is cached for the lifetime of the process.
@lru_cache(maxsize=None)
def get_embedding_helper(
    model_name: str | None = None,
    device: str | None = None,
    normalize_embeddings: bool = True,
) -> LegalBGEEmbeddingHelper:
    return LegalBGEEmbeddingHelper(model_name, device, normalize_embeddings)
