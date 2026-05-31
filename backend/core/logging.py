"""
Logging configuration and request-correlation middleware.

Every log line carries a request_id (when emitted inside a request) so logs
for a single HTTP call can be traced end-to-end. The id is stored in a
ContextVar and injected into every LogRecord via a logging.Filter.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Holds the current request id for the active async context.
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

REQUEST_ID_HEADER = "X-Request-ID"


def get_request_id() -> str:
    """Return the request id bound to the current context (or '-')."""
    return _request_id_ctx.get()


class _RequestIDFilter(logging.Filter):
    """Inject the contextual request_id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def configure_logging(level: int | str = logging.INFO) -> None:
    """
    Idempotent logging setup. Safe to call from app startup and tests.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Replace handlers so repeated calls don't stack duplicate output.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  [%(request_id)s]  %(name)s — %(message)s"
        )
    )
    handler.addFilter(_RequestIDFilter())
    root.addHandler(handler)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Assign/propagate a request id and log request start/finish with timing.
    """

    def __init__(self, app, logger_name: str = "request") -> None:
        super().__init__(app)
        self._logger = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming or uuid.uuid4().hex
        token = _request_id_ctx.set(request_id)
        start = time.perf_counter()

        self._logger.info("→ %s %s", request.method, request.url.path)
        try:
            response = await call_next(request)
            # FIX B5: log the completion line *inside* the try so the request_id
            # ContextVar is still set (the `finally` reset must run last, or the
            # "←" log loses its correlation id).
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.headers[REQUEST_ID_HEADER] = request_id
            self._logger.info(
                "← %s %s %s (%.1fms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
            return response
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            # Exception handlers turn this into a response; log the failure here.
            self._logger.exception(
                "✗ %s %s failed after %.1fms",
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise
        finally:
            _request_id_ctx.reset(token)
