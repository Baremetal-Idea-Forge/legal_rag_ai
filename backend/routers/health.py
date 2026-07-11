"""Liveness + (deep) readiness endpoints."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, Response

from core.config import Settings, get_settings
from core.dependencies import get_llm_client, get_typesense_repository
from models.schemas import HealthResponse, PdfStorageInfo, ReadinessResponse, StorageResponse
from repositories.typesense_repository import TypesenseRepository

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness — the process is up. Always 200."""
    return HealthResponse(status="ok", version=settings.APP_VERSION)


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(
    response: Response,
    settings: Settings = Depends(get_settings),
    typesense_repo: TypesenseRepository = Depends(get_typesense_repository),
    llm_client=Depends(get_llm_client),
) -> ReadinessResponse:
    """Readiness — required dependencies reachable. 503 if any is down."""
    checks = {
        "typesense": typesense_repo.is_ready(),
        "llm": await llm_client.ping(),
    }
    all_ok = all(checks.values())
    response.status_code = 200 if all_ok else 503
    return ReadinessResponse(
        status="ready" if all_ok else "degraded",
        version=settings.APP_VERSION,
        checks=checks,
    )


def _human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.2f} PB"


@router.get("/health/storage", response_model=StorageResponse)
def storage(settings: Settings = Depends(get_settings)) -> StorageResponse:
    """Disk usage for Typesense data dir + PDF storage stats."""
    data_dir = Path(settings.TYPESENSE_DATA_DIR)
    quota_bytes = settings.DISK_QUOTA_GB * 1024 * 1024 * 1024

    if data_dir.exists():
        usage = shutil.disk_usage(data_dir)
        disk_used = usage.used
    else:
        disk_used = 0

    pdf_dir = Path(settings.PDF_STORAGE_DIR)
    num_files = 0
    total_pdf_bytes = 0
    if pdf_dir.exists():
        for root, _dirs, files in os.walk(pdf_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    num_files += 1
                    total_pdf_bytes += os.path.getsize(os.path.join(root, f))

    used_pct = (disk_used / quota_bytes * 100) if quota_bytes > 0 else 0.0

    return StorageResponse(
        typesense_data_dir=str(data_dir),
        disk_used_bytes=disk_used,
        disk_used_human=_human_size(disk_used),
        disk_total_bytes=quota_bytes,
        disk_total_human=_human_size(quota_bytes),
        disk_used_pct=round(used_pct, 2),
        pdf_storage=PdfStorageInfo(
            num_files=num_files,
            total_size_bytes=total_pdf_bytes,
            total_size_human=_human_size(total_pdf_bytes),
        ),
    )
