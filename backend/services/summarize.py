"""
Document summarizer for summary-augmented chunking (SAC).

One LLM call per document produces a short summary that ingestion prepends to
every chunk of that document *for embedding only*. The point is document
scoping: a chunk reading "No person shall be deprived of his life…" carries no
signal about which statute it came from, so a query naming the document
retrieves chunks from the wrong one. The summary puts that signal into every
chunk's vector.

Two deliberate constraints:

  - The prompt is GENERIC, not expert/role-played. An expert-framed summary
    prompt retrieves the right document just as well but pushes the generator
    toward boilerplate, so the answer quality drops while retrieval looks fine.
  - The summary is capped hard (~150 chars). A longer summary starts to dominate
    the chunk's embedding and washes out the provision-level signal that the
    structural chunker exists to preserve.

Summaries are cached by document sha256, so re-ingesting the same bytes costs no
tokens.
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

# Generic on purpose — see the module docstring.
_SUMMARY_PROMPT = (
    "Summarise what this legal document is, in one sentence of at most "
    "{max_chars} characters. Name the document and its subject matter. "
    "Do not add commentary, headings, or quotation marks.\n\n"
    "DOCUMENT EXCERPT:\n{excerpt}"
)

# How much of the document is shown to the summarizer. The opening pages carry
# the title and scope; sending the whole corpus text would cost tokens without
# improving a 150-char summary.
_EXCERPT_CHARS = 4000


class DocumentSummarizer:
    """
    Produces and caches ~150-char generic document summaries.

    Failure is non-fatal: a document that cannot be summarised is indexed
    without one (chunks embed their body alone, i.e. pre-SAC behaviour). Losing
    a retrieval aid must never cost us the document itself.
    """

    def __init__(
        self,
        *,
        llm_client: object,
        max_chars: int = 150,
        tolerance_chars: int = 20,
        cache_path: str | Path | None = None,
    ) -> None:
        self._llm = llm_client
        self._max_chars = max_chars
        self._tolerance = tolerance_chars
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, str] | None = None

    # -- cache --------------------------------------------------------------

    def _load_cache(self) -> dict[str, str]:
        if self._cache is not None:
            return self._cache
        self._cache = {}
        if self._cache_path and self._cache_path.exists():
            try:
                data = json.loads(self._cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._cache = {
                        str(k): str(v) for k, v in data.items() if isinstance(v, str)
                    }
            except (OSError, ValueError) as exc:
                logger.warning("Summary cache unreadable (%s); starting empty.", exc)
        return self._cache

    def _store(self, doc_hash: str, summary: str) -> None:
        cache = self._load_cache()
        cache[doc_hash] = summary
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("Could not persist summary cache: %s", exc)

    # -- summarization ------------------------------------------------------

    async def summarize(self, *, document_text: str, doc_hash: str) -> str:
        """
        Return a cached or freshly generated summary, or "" if none could be
        produced. Never raises.
        """
        cached = self._load_cache().get(doc_hash)
        if cached is not None:
            logger.debug("Summary cache hit for %s…", doc_hash[:12])
            return cached

        excerpt = document_text[:_EXCERPT_CHARS].strip()
        if not excerpt:
            return ""

        summary = await self._generate(excerpt, self._max_chars)
        # One retry at a reduced target: models overshoot a char budget far more
        # often than they undershoot, and asking for less is the cheapest fix.
        if summary and len(summary) > self._max_chars + self._tolerance:
            logger.info(
                "Summary overran (%d > %d+%d); regenerating at a lower target.",
                len(summary), self._max_chars, self._tolerance,
            )
            retried = await self._generate(excerpt, max(40, self._max_chars // 2))
            if retried:
                summary = retried

        if not summary:
            logger.warning(
                "No summary for %s… — indexing without one (body-only embeddings).",
                doc_hash[:12],
            )
            return ""

        # Hard cap: the embedding contract depends on the summary staying small.
        summary = summary[: self._max_chars + self._tolerance].strip()
        self._store(doc_hash, summary)
        logger.info("Summary for %s… (%d chars): %s", doc_hash[:12], len(summary), summary)
        return summary

    def summarize_sync(self, *, document_text: str, doc_hash: str) -> str:
        """
        Blocking `summarize` for the synchronous ingestion path.

        Ingestion is sync end to end (PyMuPDF, sentence-transformers, Typesense);
        only the LLM clients are async, so the coroutine is driven here instead
        of colouring the whole write path async. Inside a running loop the work
        goes to a private loop on a worker thread — that loop is never the
        caller's, and both LLM clients build their HTTP client per call, so
        neither is bound to the wrong loop.
        """
        coro = self.summarize(document_text=document_text, doc_hash=doc_hash)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    async def _generate(self, excerpt: str, max_chars: int) -> str:
        prompt = _SUMMARY_PROMPT.format(max_chars=max_chars, excerpt=excerpt)
        try:
            raw = await self._llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.0
            )
        except Exception as exc:  # noqa: BLE001 — a retrieval aid, never fatal
            logger.warning("Summarizer LLM call failed: %s", exc)
            return ""
        return _clean(raw)


def _clean(raw: str) -> str:
    """Collapse a model reply to a single unquoted line."""
    text = " ".join((raw or "").split())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text
