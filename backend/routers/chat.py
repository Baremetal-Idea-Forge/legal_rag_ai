"""Chat endpoint — grounded, cited RAG answers (sync + SSE streaming)."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from core.dependencies import get_rag_service
from core.exceptions import LegalRagError
from core.security import rate_limiter
from models.schemas import ChatRequest, ChatResponse
from services.rag_service import RagService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"], dependencies=[Depends(rate_limiter)])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: RagService = Depends(get_rag_service),
) -> ChatResponse:
    """Answer a question grounded in the indexed legal corpus, with citations."""
    return await service.answer(request.query, top_k=request.top_k)


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: RagService = Depends(get_rag_service),
) -> StreamingResponse:
    """
    Server-Sent Events stream. Emits `event: token` lines as the answer is
    generated, then a terminal `event: citations` carrying the source list.
    """

    async def event_source():
        # FIX B3: HTTP 200 + headers are already sent once streaming starts, so
        # a mid-stream failure can't be turned into an error response by the
        # exception handlers. Emit a terminal `error` event instead of letting
        # the stream truncate silently.
        try:
            async for event in service.stream_answer(request.query, top_k=request.top_k):
                payload = json.dumps(event["data"])
                yield f"event: {event['type']}\ndata: {payload}\n\n"
        except LegalRagError as exc:
            logger.warning("Chat stream failed: %s", exc.message)
            body = json.dumps({"error": exc.error_code, "message": exc.message})
            yield f"event: error\ndata: {body}\n\n"
        except Exception:
            logger.exception("Unexpected chat stream failure")
            body = json.dumps({"error": "internal_error", "message": "Streaming failed."})
            yield f"event: error\ndata: {body}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
