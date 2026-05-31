"""Ingestion endpoints — upload a PDF or bulk-ingest the data/ corpus."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, UploadFile

from core.config import Settings, get_settings
from core.dependencies import get_ingestion_service
from core.exceptions import IngestionError
from core.security import rate_limiter, require_api_key
from models.schemas import BulkIngestResponse, IngestResponse
from services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

# Write path: optional API-key auth + rate limiting (both no-ops unless enabled).
router = APIRouter(
    prefix="/ingest",
    tags=["Ingest"],
    dependencies=[Depends(require_api_key), Depends(rate_limiter)],
)


@router.post("", response_model=IngestResponse, status_code=201)
async def ingest_pdf(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """Upload a single PDF → store, chunk, embed, index."""
    if file.content_type != settings.ALLOWED_UPLOAD_CONTENT_TYPE:
        raise IngestionError(
            f"Unsupported content type '{file.content_type}'. "
            f"Expected '{settings.ALLOWED_UPLOAD_CONTENT_TYPE}'.",
            detail=file.content_type,
        )

    # FIX B4: reject oversized uploads by declared size BEFORE buffering the
    # whole body into memory (DoS guard). Falls back to the post-read check
    # when the size is not known (chunked / no Content-Length).
    if file.size is not None and file.size > settings.MAX_UPLOAD_BYTES:
        raise IngestionError(
            f"File exceeds the {settings.MAX_UPLOAD_BYTES}-byte limit.",
            detail=file.size,
        )

    pdf_bytes = await file.read()
    if len(pdf_bytes) > settings.MAX_UPLOAD_BYTES:
        raise IngestionError(
            f"File exceeds the {settings.MAX_UPLOAD_BYTES}-byte limit.",
            detail=len(pdf_bytes),
        )

    return service.ingest_bytes(
        pdf_bytes=pdf_bytes, filename=file.filename or "document.pdf"
    )


@router.post("/bulk", response_model=BulkIngestResponse)
def ingest_bulk(
    service: IngestionService = Depends(get_ingestion_service),
) -> BulkIngestResponse:
    """Ingest every PDF in the configured data/ directory (admin operation)."""
    ingested, failed = service.ingest_directory("data")
    return BulkIngestResponse(
        ingested=ingested,
        failed=failed,
        total_ingested=len(ingested),
        total_failed=len(failed),
    )
