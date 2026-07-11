"""
Ingestion service — orchestrates the write path:

    store → extract → chunk → build docs → embed → index

Idempotency: the PDF's SHA-256 is used as the pdf_id, so re-ingesting identical
bytes produces identical chunk ids and upserts in place (no duplicates).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from core.config import Settings
from core.exceptions import IngestionError
from models.schemas import IngestFailure, IngestResponse
from repositories.pdf_storage_repository import PdfStorageRepository
from repositories.typesense_repository import TypesenseRepository
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        *,
        pdf_repo: PdfStorageRepository,
        embedding_service: EmbeddingService,
        typesense_repo: TypesenseRepository,
        settings: Settings,
    ) -> None:
        self._pdf_repo = pdf_repo
        self._embeddings = embedding_service
        self._typesense = typesense_repo
        self._settings = settings

    def ingest_bytes(self, *, pdf_bytes: bytes, filename: str) -> IngestResponse:
        if not pdf_bytes:
            raise IngestionError("Uploaded file is empty.", detail=filename)

        # Content-addressed id → deterministic, idempotent re-ingestion.
        sha = hashlib.sha256(pdf_bytes).hexdigest()
        logger.info("Ingesting '%s' (sha=%s…, %d bytes)", filename, sha[:12], len(pdf_bytes))

        stored = self._pdf_repo.save(
            pdf_bytes=pdf_bytes, original_filename=filename, pdf_id=sha
        )

        pages = self._pdf_repo.extract_pages(
            stored.local_path,
            ocr_enabled=self._settings.OCR_ENABLED,
            ocr_min_text_length=self._settings.OCR_MIN_TEXT_LENGTH,
        )
        chunks = self._pdf_repo.chunk(
            pages,
            max_chars=self._settings.CHUNK_MAX_CHARS,
            overlap_chars=self._settings.CHUNK_OVERLAP_CHARS,
        )
        if not chunks:
            raise IngestionError(
                f"No extractable text in '{filename}'.", detail=filename
            )

        docs = self._pdf_repo.build_documents(pdf=stored, chunks=chunks)
        embedded = self._embeddings.embed_chunk_documents(docs)
        self._typesense.index_chunks(embedded)

        logger.info(
            "Ingested '%s': %d pages, %d chunks indexed.",
            filename,
            len(pages),
            len(chunks),
        )
        return IngestResponse(
            pdf_id=stored.pdf_id,
            filename=stored.original_filename,
            sha256=stored.sha256,
            num_pages=len(pages),
            num_chunks=len(chunks),
            status="indexed",
        )

    def ingest_directory(
        self, directory: str | Path
    ) -> tuple[list[IngestResponse], list[IngestFailure]]:
        """
        Ingest every *.pdf in a directory. Corrupted/unreadable PDFs are
        reported as failures without aborting the batch.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise IngestionError(f"Not a directory: {directory}", detail=str(directory))

        ingested: list[IngestResponse] = []
        failed: list[IngestFailure] = []

        for pdf_path in sorted(directory.glob("*.pdf")):
            try:
                result = self.ingest_bytes(
                    pdf_bytes=pdf_path.read_bytes(), filename=pdf_path.name
                )
                ingested.append(result)
            except IngestionError as exc:
                logger.warning("Skipping '%s': %s", pdf_path.name, exc.message)
                failed.append(IngestFailure(filename=pdf_path.name, error=exc.message))
            except OSError as exc:
                # FIX B1: a glob match may be unreadable (a directory named
                # *.pdf, a permission error, or a file removed mid-scan).
                # Report it as a failure instead of aborting the whole batch.
                logger.warning("Skipping unreadable '%s': %s", pdf_path.name, exc)
                failed.append(IngestFailure(filename=pdf_path.name, error=str(exc)))

        return ingested, failed
