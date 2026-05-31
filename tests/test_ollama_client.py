"""OllamaClient — success, retries, error mapping, streaming (httpx MockTransport)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import httpx
import pytest

from clients.ollama_client import OllamaClient
from core.exceptions import LLMError


def _client(handler, **kwargs) -> OllamaClient:
    return OllamaClient(
        base_url="http://ollama:11434",
        model="gemma3:4b",
        timeout_seconds=5,
        max_retries=kwargs.pop("max_retries", 2),
        transport=httpx.MockTransport(handler),
    )


def _messages():
    return [{"role": "user", "content": "hi"}]


# --- chat (non-streaming) --------------------------------------------------

def test_chat_success():
    def handler(request):
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Hello!"}})

    out = asyncio.run(_client(handler).chat(_messages()))
    assert out == "Hello!"


def test_chat_sends_model_and_messages():
    seen = {}

    def handler(request):
        import json
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    asyncio.run(_client(handler).chat(_messages()))
    assert seen["model"] == "gemma3:4b"
    assert seen["stream"] is False
    assert seen["messages"] == _messages()


def test_chat_malformed_response_raises_llm_error():
    def handler(request):
        return httpx.Response(200, json={"unexpected": "shape"})

    with pytest.raises(LLMError):
        asyncio.run(_client(handler).chat(_messages()))


def test_chat_4xx_raises_immediately_no_retry():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    with pytest.raises(LLMError):
        asyncio.run(_client(handler).chat(_messages()))
    assert calls["n"] == 1  # no retries on client error


def test_chat_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"message": {"content": "recovered"}})

    with patch("clients.ollama_client.asyncio.sleep", new=AsyncMock()):
        out = asyncio.run(_client(handler, max_retries=2).chat(_messages()))
    assert out == "recovered"
    assert calls["n"] == 3


def test_chat_exhausts_retries_raises_llm_error():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with patch("clients.ollama_client.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(LLMError):
            asyncio.run(_client(handler, max_retries=2).chat(_messages()))


def test_chat_5xx_retried_then_fails():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    with patch("clients.ollama_client.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(LLMError):
            asyncio.run(_client(handler, max_retries=1).chat(_messages()))
    assert calls["n"] == 2  # initial + 1 retry


# --- stream_chat -----------------------------------------------------------

def _collect(aiter):
    async def run():
        return [tok async for tok in aiter]
    return asyncio.run(run())


def test_stream_yields_tokens():
    body = (
        b'{"message":{"content":"Hello"},"done":false}\n'
        b'{"message":{"content":" world"},"done":true}\n'
    )

    def handler(request):
        return httpx.Response(200, content=body)

    tokens = _collect(_client(handler).stream_chat(_messages()))
    assert tokens == ["Hello", " world"]


def test_stream_error_raises_llm_error():
    def handler(request):
        return httpx.Response(500, text="boom")

    with pytest.raises(LLMError):
        _collect(_client(handler).stream_chat(_messages()))


# --- ping (readiness) ------------------------------------------------------

def test_ping_true_when_reachable():
    def handler(request):
        return httpx.Response(200, json={"models": []})

    assert asyncio.run(_client(handler).ping()) is True


def test_ping_false_on_server_error():
    def handler(request):
        return httpx.Response(500, text="down")

    assert asyncio.run(_client(handler).ping()) is False


def test_ping_false_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    assert asyncio.run(_client(handler).ping()) is False
