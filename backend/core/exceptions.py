"""
Domain exception hierarchy and the FastAPI handlers that map them to HTTP.

Services and repositories raise these typed errors; routers stay clean of
try/except. A single set of handlers converts them to consistent JSON bodies.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class LegalRagError(Exception):
    """Base class for all application-domain errors."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, detail: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class AuthError(LegalRagError):
    """Raised when a request is missing or presents an invalid credential."""

    status_code = 401
    error_code = "unauthorized"


class RateLimitError(LegalRagError):
    """Raised when a client exceeds the configured request rate."""

    status_code = 429
    error_code = "rate_limited"


class DocumentNotFoundError(LegalRagError):
    status_code = 404
    error_code = "document_not_found"


class IngestionError(LegalRagError):
    """Raised when a document cannot be stored, parsed, embedded, or indexed."""

    status_code = 422
    error_code = "ingestion_error"


class SearchError(LegalRagError):
    """Raised when a search request cannot be executed."""

    status_code = 502
    error_code = "search_error"


class LLMError(LegalRagError):
    """Raised when the LLM (Ollama) fails to produce a response."""

    status_code = 503
    error_code = "llm_error"


class UpstreamUnavailableError(LegalRagError):
    """Raised when a required upstream (Typesense, MCP, Ollama) is unreachable."""

    status_code = 503
    error_code = "upstream_unavailable"


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _error_body(error_code: str, message: str, detail: object | None = None) -> dict:
    body: dict[str, object] = {"error": error_code, "message": message}
    if detail is not None:
        body["detail"] = detail
    return body


async def _handle_domain_error(request: Request, exc: LegalRagError) -> JSONResponse:
    # 4xx are client problems (info); 5xx are our/upstream problems (error).
    log = logger.warning if exc.status_code < 500 else logger.error
    log("%s: %s", exc.error_code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.error_code, exc.message, exc.detail),
    )


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=_error_body("internal_error", "An unexpected error occurred."),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire domain + catch-all handlers onto the app."""
    app.add_exception_handler(LegalRagError, _handle_domain_error)
    app.add_exception_handler(Exception, _handle_unexpected_error)
