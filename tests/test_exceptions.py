"""Exception-handler mapping: domain errors → consistent JSON + status."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.exceptions import (
    DocumentNotFoundError,
    IngestionError,
    LLMError,
    SearchError,
    UpstreamUnavailableError,
    register_exception_handlers,
)


def _build_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise/{kind}")
    def _raise(kind: str):
        errors = {
            "notfound": DocumentNotFoundError("missing", detail="abc"),
            "ingest": IngestionError("bad pdf"),
            "search": SearchError("search boom"),
            "llm": LLMError("llm down"),
            "upstream": UpstreamUnavailableError("typesense down"),
        }
        raise errors[kind]

    @app.get("/boom")
    def _boom():
        raise RuntimeError("totally unexpected")

    return app


client = TestClient(_build_app(), raise_server_exceptions=False)


@pytest.mark.parametrize(
    "kind,status,code",
    [
        ("notfound", 404, "document_not_found"),
        ("ingest", 422, "ingestion_error"),
        ("search", 502, "search_error"),
        ("llm", 503, "llm_error"),
        ("upstream", 503, "upstream_unavailable"),
    ],
)
def test_domain_error_mapping(kind, status, code):
    r = client.get(f"/raise/{kind}")
    assert r.status_code == status
    body = r.json()
    assert body["error"] == code
    assert body["message"]


def test_detail_passed_through():
    r = client.get("/raise/notfound")
    assert r.json()["detail"] == "abc"


def test_unexpected_error_becomes_500():
    r = client.get("/boom")
    assert r.status_code == 500
    assert r.json()["error"] == "internal_error"
    # Internal details must not leak in the body.
    assert "totally unexpected" not in r.text
