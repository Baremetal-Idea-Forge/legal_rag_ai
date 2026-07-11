"""
Async client for Google Gemini API (gemini-2.5-flash).

Mirrors the OllamaClient interface (chat, stream_chat, ping) so both
providers are interchangeable via the LLM_PROVIDER switch.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from google import genai
from google.genai import types

from core.exceptions import LLMError

logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout_seconds: int = 120,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        # Lazy: a missing/invalid key fails on first use (caught by ping →
        # readiness reports degraded), not at construction time (which would
        # crash /health/ready with a 500).
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> str:
        contents = self._to_contents(messages)
        config = types.GenerateContentConfig(temperature=temperature)
        try:
            response = self._get_client().models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            logger.error("Gemini chat failed: %s", exc)
            raise LLMError("Gemini API request failed.", detail=str(exc)) from exc

    async def ping(self) -> bool:
        try:
            self._get_client().models.get(model=self._model)
            return True
        except Exception:
            return False

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        contents = self._to_contents(messages)
        config = types.GenerateContentConfig(temperature=temperature)
        try:
            for chunk in self._get_client().models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.error("Gemini stream failed: %s", exc)
            raise LLMError("Gemini streaming failed.", detail=str(exc)) from exc

    @staticmethod
    def _to_contents(messages: list[dict[str, str]]) -> list[types.Content]:
        """Convert OpenAI-style messages to Gemini Content objects."""
        contents: list[types.Content] = []
        system_parts: list[str] = []

        for msg in messages:
            role = msg["role"]
            text = msg["content"]

            if role == "system":
                system_parts.append(text)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part(text=text)],
                )
            )

        # Prepend system instructions as a user message if present
        if system_parts and contents:
            system_content = types.Content(
                role="user",
                parts=[types.Part(text="\n\n".join(system_parts))],
            )
            contents.insert(0, system_content)

        return contents
