"""RequestIDMiddleware — id propagation, response header, error path."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.logging import (
    REQUEST_ID_HEADER,
    RequestIDMiddleware,
    _RequestIDFilter,
    configure_logging,
    get_request_id,
)


def _app() -> FastAPI:
    configure_logging()
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ok")
    def ok():
        return {"request_id": get_request_id()}

    @app.get("/explode")
    def explode():
        raise RuntimeError("boom")

    return app


client = TestClient(_app(), raise_server_exceptions=False)


def test_response_carries_request_id_header():
    r = client.get("/ok")
    assert r.status_code == 200
    assert r.headers[REQUEST_ID_HEADER]


def test_incoming_request_id_is_propagated():
    r = client.get("/ok", headers={REQUEST_ID_HEADER: "trace-123"})
    assert r.headers[REQUEST_ID_HEADER] == "trace-123"
    # And the same id is visible inside the handler via the ContextVar.
    assert r.json()["request_id"] == "trace-123"


def test_request_id_generated_when_absent():
    r = client.get("/ok")
    assert r.json()["request_id"] != "-"


def test_error_path_still_logs_and_propagates():
    # Exercises the middleware's except/raise branch.
    r = client.get("/explode")
    assert r.status_code == 500


def test_request_id_resets_between_requests():
    a = client.get("/ok", headers={REQUEST_ID_HEADER: "first"}).json()["request_id"]
    b = client.get("/ok", headers={REQUEST_ID_HEADER: "second"}).json()["request_id"]
    assert a == "first" and b == "second"
    # Outside any request the context is back to default.
    assert get_request_id() == "-"


def test_completion_log_keeps_request_id():
    """B5 regression: the '←' completion line must carry the request_id.

    A capture handler with the real filter records request_id at *emit* time —
    so it reflects whether the ContextVar was still set when the line was logged.
    """
    captured: list[tuple[str, str | None]] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append((record.getMessage(), getattr(record, "request_id", None)))

    handler = Capture()
    handler.addFilter(_RequestIDFilter())  # populates request_id live, at emit
    req_logger = logging.getLogger("request")
    req_logger.addHandler(handler)
    try:
        client.get("/ok", headers={REQUEST_ID_HEADER: "corr-42"})
    finally:
        req_logger.removeHandler(handler)

    completion = [rid for msg, rid in captured if msg.startswith("←")]
    assert completion, "no completion log captured"
    assert completion[0] == "corr-42"
