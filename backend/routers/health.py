"""Liveness + (deep) readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from clients.ollama_client import OllamaClient
from core.config import Settings, get_settings
from core.dependencies import get_ollama_client, get_typesense_repository
from models.schemas import HealthResponse, ReadinessResponse
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
    ollama: OllamaClient = Depends(get_ollama_client),
) -> ReadinessResponse:
    """Readiness — required dependencies reachable. 503 if any is down."""
    checks = {
        "typesense": typesense_repo.is_ready(),
        "ollama": await ollama.ping(),
    }
    all_ok = all(checks.values())
    response.status_code = 200 if all_ok else 503
    return ReadinessResponse(
        status="ready" if all_ok else "degraded",
        version=settings.APP_VERSION,
        checks=checks,
    )
