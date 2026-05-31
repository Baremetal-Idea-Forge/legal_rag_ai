"""McpClient — payload parsing, result extraction, error mapping (no live server)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest

from clients.mcp_client import McpClient
from core.exceptions import UpstreamUnavailableError
from models.schemas import ChunkHit


def _hit_dict(i=0):
    return ChunkHit(
        id=f"c{i}", pdf_id="p", pdf_name="x.pdf", page_start=1, page_end=1,
        chunk_index=i, content="c",
    ).model_dump()


# --- pure helpers ----------------------------------------------------------

def test_hits_from_payload():
    hits = McpClient._hits_from_payload({"hits": [_hit_dict(0), _hit_dict(1)]})
    assert len(hits) == 2
    assert all(isinstance(h, ChunkHit) for h in hits)


def test_hits_from_empty_payload():
    assert McpClient._hits_from_payload({}) == []


class _Block:
    def __init__(self, text):
        self.text = text


class _Result:
    def __init__(self, *, structured=None, content=None, is_error=False):
        self.structuredContent = structured
        self.content = content or []
        self.isError = is_error


def test_extract_prefers_structured_content():
    r = _Result(structured={"hits": [_hit_dict(0)]})
    assert McpClient._extract(r)["hits"][0]["id"] == "c0"


def test_extract_falls_back_to_text_json():
    r = _Result(content=[_Block('{"hits": [{"id": "a"}]}')])
    assert McpClient._extract(r)["hits"][0]["id"] == "a"


def test_extract_iserror_raises_upstream():
    r = _Result(content=[_Block("tool exploded")], is_error=True)
    with pytest.raises(UpstreamUnavailableError):
        McpClient._extract(r)


def test_extract_empty_returns_empty_dict():
    assert McpClient._extract(_Result()) == {}


# --- search / get_document via injected session seam -----------------------

def test_search_returns_chunk_hits():
    async def factory(name, args):
        assert name == "search_legal_chunks"
        assert args["query"] == "negligence"
        return {"hits": [_hit_dict(0), _hit_dict(1)]}

    client = McpClient(base_url="http://mcp:9000", session_factory=factory)
    hits = asyncio.run(client.search("negligence", top_k=3))
    assert len(hits) == 2
    assert isinstance(hits[0], ChunkHit)


def test_get_document_returns_doc():
    async def factory(name, args):
        assert name == "get_document"
        return {"document": {"id": args["chunk_id"]}}

    client = McpClient(base_url="http://mcp:9000", session_factory=factory)
    doc = asyncio.run(client.get_document("a_chunk_0"))
    assert doc["id"] == "a_chunk_0"


def test_failure_wrapped_as_upstream_unavailable():
    async def factory(name, args):
        raise RuntimeError("connection refused")

    client = McpClient(base_url="http://mcp:9000", session_factory=factory)
    with pytest.raises(UpstreamUnavailableError):
        asyncio.run(client.search("q"))


def test_upstream_error_from_seam_not_double_wrapped():
    async def factory(name, args):
        raise UpstreamUnavailableError("already typed")

    client = McpClient(base_url="http://mcp:9000", session_factory=factory)
    with pytest.raises(UpstreamUnavailableError, match="already typed"):
        asyncio.run(client.search("q"))


def test_base_url_appends_mcp_path():
    client = McpClient(base_url="http://mcp:9000/")
    assert client._url == "http://mcp:9000/mcp"


def test_malformed_search_payload_wrapped_as_upstream():
    """B2 regression: hits missing required fields must not leak a raw
    pydantic ValidationError — translate to UpstreamUnavailableError."""
    async def factory(name, args):
        return {"hits": [{"totally": "wrong shape"}]}

    client = McpClient(base_url="http://mcp:9000", session_factory=factory)
    with pytest.raises(UpstreamUnavailableError):
        asyncio.run(client.search("q"))


def test_hits_not_a_list_wrapped_as_upstream():
    async def factory(name, args):
        return {"hits": "not-a-list"}

    client = McpClient(base_url="http://mcp:9000", session_factory=factory)
    with pytest.raises(UpstreamUnavailableError):
        asyncio.run(client.search("q"))
