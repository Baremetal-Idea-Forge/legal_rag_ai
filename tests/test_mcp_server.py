"""MCP server tools — impl functions + in-process tool dispatch (mocked search)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent / "mcp-server"))

import pytest

import server
from models.schemas import ChunkHit, SearchResponse


class FakeSearch:
    def __init__(self):
        self.last = None

    def search(self, query, *, top_k=5, mode="hybrid", filter_by=None):
        self.last = {"query": query, "top_k": top_k, "mode": mode, "filter_by": filter_by}
        return SearchResponse(
            query=query, mode=mode, count=1,
            hits=[ChunkHit(id="a_chunk_0", pdf_id="a", pdf_name="contract.pdf",
                           page_start=1, page_end=1, chunk_index=0, content="offer")],
        )


class FakeTs:
    def get_document(self, chunk_id):
        return {"id": chunk_id, "content": "stored chunk"}


@pytest.fixture(autouse=True)
def reset_search():
    server._search = None
    yield
    server._search = None


def test_search_impl_returns_hits(monkeypatch):
    fake = FakeSearch()
    monkeypatch.setattr(server, "_get_search", lambda: fake)
    out = server.search_legal_chunks_impl("negligence", top_k=3, mode="hybrid")
    assert out["hits"][0]["id"] == "a_chunk_0"
    assert fake.last["top_k"] == 3


def test_get_document_impl_returns_document(monkeypatch):
    monkeypatch.setattr(server, "get_typesense_repository", lambda: FakeTs())
    out = server.get_document_impl("a_chunk_0")
    assert out["document"]["id"] == "a_chunk_0"


def _result_text(result):
    # FastMCP.call_tool returns Sequence[ContentBlock] or (content, structured).
    content = result[0] if isinstance(result, tuple) else result
    return "".join(getattr(b, "text", "") for b in content)


def test_registered_search_tool_dispatch(monkeypatch):
    monkeypatch.setattr(server, "_get_search", lambda: FakeSearch())
    result = asyncio.run(
        server.mcp.call_tool("search_legal_chunks", {"query": "tort", "top_k": 2})
    )
    payload = json.loads(_result_text(result))
    assert payload["hits"][0]["pdf_name"] == "contract.pdf"


def test_registered_get_document_tool_dispatch(monkeypatch):
    monkeypatch.setattr(server, "get_typesense_repository", lambda: FakeTs())
    result = asyncio.run(
        server.mcp.call_tool("get_document", {"chunk_id": "a_chunk_0"})
    )
    payload = json.loads(_result_text(result))
    assert payload["document"]["id"] == "a_chunk_0"


def test_lazy_search_singleton(monkeypatch):
    built = []

    class OneEmbed:
        pass

    monkeypatch.setattr(server, "get_embedding_service", lambda: built.append(1) or OneEmbed())
    monkeypatch.setattr(server, "get_typesense_repository", lambda: FakeTs())
    # First call builds, second reuses.
    s1 = server._get_search()
    s2 = server._get_search()
    assert s1 is s2
    assert len(built) == 1
