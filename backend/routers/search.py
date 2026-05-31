"""Search endpoints — hybrid retrieval over indexed legal chunks."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from core.dependencies import get_search_service
from core.security import rate_limiter
from models.schemas import SearchRequest, SearchResponse
from services.search_service import SearchService

router = APIRouter(
    prefix="/search", tags=["Search"], dependencies=[Depends(rate_limiter)]
)


@router.post("", response_model=SearchResponse)
def search_post(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return service.search(
        request.query,
        top_k=request.top_k,
        mode=request.mode,
        filter_by=request.filter_by,
    )


@router.get("", response_model=SearchResponse)
def search_get(
    q: str = Query(..., min_length=1, max_length=2000, description="Search text"),
    top_k: int = Query(5, ge=1, le=50),
    mode: str = Query("hybrid"),
    filter_by: str | None = Query(None),
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return service.search(q, top_k=top_k, mode=mode, filter_by=filter_by)
