"""
Data-access layer for PDF storage + parsing.

Thin wrapper over helpers/pdf_helper.PDFHelper exposing the granular steps the
ingestion service orchestrates (save → extract → chunk → build docs).
"""

from __future__ import annotations

import logging
from typing import Any

from helpers.pdf_helper import PDFChunk, PDFHelper, PDFPageText, StoredPDF
from core.exceptions import IngestionError

logger = logging.getLogger(__name__)


class PdfStorageRepository:
    def __init__(self, helper: PDFHelper) -> None:
        self._helper = helper

    def save(
        self, *, pdf_bytes: bytes, original_filename: str, pdf_id: str | None = None
    ) -> StoredPDF:
        try:
            return self._helper.save_pdf_bytes(
                pdf_bytes=pdf_bytes,
                original_filename=original_filename,
                pdf_id=pdf_id,
            )
        except OSError as exc:
            raise IngestionError(
                f"Failed to store PDF '{original_filename}'.", detail=str(exc)
            ) from exc

    def extract_pages(self, local_path: str) -> list[PDFPageText]:
        # extract_pages raises ValueError for corrupted/unsupported PDFs.
        try:
            return self._helper.extract_pages(local_path)
        except (ValueError, FileNotFoundError) as exc:
            raise IngestionError(
                f"Failed to read PDF at '{local_path}'.", detail=str(exc)
            ) from exc

    def chunk(
        self,
        pages: list[PDFPageText],
        *,
        max_chars: int,
        overlap_chars: int,
    ) -> list[PDFChunk]:
        return self._helper.chunk_pages(
            pages, max_chars=max_chars, overlap_chars=overlap_chars
        )

    def build_documents(
        self, *, pdf: StoredPDF, chunks: list[PDFChunk]
    ) -> list[dict[str, Any]]:
        return self._helper.build_typesense_chunk_documents(pdf=pdf, chunks=chunks)
