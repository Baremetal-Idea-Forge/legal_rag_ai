"""
Async client for a local Ollama server (Gemma 3 4B).

Uses Ollama's native chat API (`POST /api/chat`). All failure modes — server
down, timeout, bad status, malformed payload — surface as a single typed
LLMError (HTTP 503) so callers handle the LLM subsystem uniformly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import httpx

from core.exceptions import LLMError

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: int = 120,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._transport = transport  # injectable for tests

    # -- public API ---------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> str:
        """Non-streaming chat completion → assistant message content."""
        payload = self._payload(messages, temperature, stream=False)
        logger.debug(
            "Ollama chat request - model=%s, temp=%.2f, msg_count=%d, msg_sizes=%s",
            self._model, temperature, len(messages),
            [len(m.get("content", "")) for m in messages],
        )
        data = await self._post_with_retries("/api/chat", payload)
        try:
            response_text = data["message"]["content"]
            logger.debug("Ollama response: %d chars, first 100: %s",
                        len(response_text), response_text[:100])
            return response_text
        except (KeyError, TypeError) as exc:
            raise LLMError("Malformed response from Ollama.", detail=str(exc)) from exc

    async def ping(self) -> bool:
        """Best-effort readiness probe — never raises. Short timeout."""
        try:
            async with httpx.AsyncClient(timeout=2.0, transport=self._transport) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code < 500
        except Exception:
            return False

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        """Streaming chat completion → yields content tokens as they arrive."""
        payload = self._payload(messages, temperature, stream=True)
        url = f"{self._base_url}/api/chat"
        logger.debug(
            "Ollama stream request - model=%s, temp=%.2f, msg_count=%d",
            self._model, temperature, len(messages),
        )
        token_count = 0
        try:
            async with self._client() as client:
                async with client.stream("POST", url, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content")
                        if token:
                            token_count += len(token)
                            yield token
                        if chunk.get("done"):
                            logger.debug("Ollama stream complete - %d chars received", token_count)
                            break
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.error("Ollama stream failed: %s", exc)
            raise LLMError("LLM streaming failed.", detail=str(exc)) from exc

    # -- internals ----------------------------------------------------------

    def _payload(
        self, messages: list[dict[str, str]], temperature: float, *, stream: bool
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout, transport=self._transport)

    async def _post_with_retries(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                async with self._client() as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as exc:
                # 4xx is our fault (bad request) — do not retry.
                if exc.response.status_code < 500:
                    raise LLMError(
                        f"Ollama rejected the request ({exc.response.status_code}).",
                        detail=exc.response.text[:500],
                    ) from exc
                last_exc = exc
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                    httpx.RemoteProtocolError) as exc:
                last_exc = exc

            if attempt < self._max_retries:
                backoff = 0.5 * (2 ** attempt)
                logger.warning(
                    "Ollama call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self._max_retries + 1, last_exc, backoff,
                )
                await asyncio.sleep(backoff)

        logger.error("Ollama unavailable after %d attempts: %s",
                     self._max_retries + 1, last_exc)
        raise LLMError(
            "The language model is unavailable. Is Ollama running?",
            detail=str(last_exc),
        ) from last_exc
