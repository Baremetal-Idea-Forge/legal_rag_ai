"""services/summarize.py — SAC document summaries: caching, overrun, failure."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.summarize import DocumentSummarizer, _clean


class FakeLLM:
    """Returns queued replies; records prompts. Raises when a reply is an Exception."""

    def __init__(self, *replies: object):
        self._replies = list(replies)
        self.prompts: list[str] = []

    async def chat(self, messages, *, temperature=0.2):
        self.prompts.append(messages[0]["content"])
        reply = self._replies.pop(0) if self._replies else "A generic summary."
        if isinstance(reply, Exception):
            raise reply
        return reply


def _summarizer(llm=None, tmp_path=None, **kw):
    return DocumentSummarizer(
        llm_client=llm or FakeLLM(),
        cache_path=(tmp_path / "summaries.json") if tmp_path else None,
        **kw,
    )


def _run(coro):
    return asyncio.run(coro)


# --- summarize -------------------------------------------------------------

def test_returns_cleaned_summary():
    llm = FakeLLM('  "The Sale of Goods Act, 1930 — transfer of goods."  ')
    s = _summarizer(llm)
    out = _run(s.summarize(document_text="Sale of Goods Act…", doc_hash="a" * 64))
    assert out == "The Sale of Goods Act, 1930 — transfer of goods."


def test_empty_document_returns_empty_without_llm_call():
    llm = FakeLLM()
    s = _summarizer(llm)
    out = _run(s.summarize(document_text="   ", doc_hash="a" * 64))
    assert out == ""
    assert llm.prompts == []


def test_llm_failure_returns_empty_never_raises():
    llm = FakeLLM(RuntimeError("ollama down"))
    s = _summarizer(llm)
    out = _run(s.summarize(document_text="text", doc_hash="a" * 64))
    assert out == ""


def test_overrun_regenerates_once_at_reduced_target():
    long_reply = "x" * 400
    llm = FakeLLM(long_reply, "Short corrected summary.")
    s = _summarizer(llm, max_chars=150, tolerance_chars=20)
    out = _run(s.summarize(document_text="text", doc_hash="a" * 64))
    assert out == "Short corrected summary."
    assert len(llm.prompts) == 2
    # The retry asks for half the original budget.
    assert "75 characters" in llm.prompts[1]


def test_overrun_retry_still_long_is_hard_capped():
    llm = FakeLLM("x" * 400, "y" * 400)
    s = _summarizer(llm, max_chars=150, tolerance_chars=20)
    out = _run(s.summarize(document_text="text", doc_hash="a" * 64))
    assert len(out) <= 170  # max + tolerance


# --- cache -----------------------------------------------------------------

def test_cache_hit_skips_llm(tmp_path):
    llm = FakeLLM("A summary.")
    s = _summarizer(llm, tmp_path=tmp_path)
    first = _run(s.summarize(document_text="text", doc_hash="a" * 64))
    second = _run(s.summarize(document_text="text", doc_hash="a" * 64))
    assert first == second == "A summary."
    assert len(llm.prompts) == 1


def test_cache_persists_across_instances(tmp_path):
    _run(
        _summarizer(FakeLLM("Persisted."), tmp_path=tmp_path).summarize(
            document_text="text", doc_hash="b" * 64
        )
    )
    llm2 = FakeLLM("Should not be called.")
    out = _run(
        _summarizer(llm2, tmp_path=tmp_path).summarize(
            document_text="text", doc_hash="b" * 64
        )
    )
    assert out == "Persisted."
    assert llm2.prompts == []


def test_failed_summary_is_not_cached(tmp_path):
    llm = FakeLLM(RuntimeError("down"), "Recovered summary.")
    s = _summarizer(llm, tmp_path=tmp_path)
    assert _run(s.summarize(document_text="t", doc_hash="c" * 64)) == ""
    assert _run(s.summarize(document_text="t", doc_hash="c" * 64)) == "Recovered summary."


def test_corrupt_cache_file_starts_empty(tmp_path):
    cache = tmp_path / "summaries.json"
    cache.write_text("{not json", encoding="utf-8")
    s = DocumentSummarizer(llm_client=FakeLLM("Fresh."), cache_path=cache)
    out = _run(s.summarize(document_text="t", doc_hash="d" * 64))
    assert out == "Fresh."
    # And the fresh summary was persisted over the corrupt file.
    assert json.loads(cache.read_text(encoding="utf-8"))["d" * 64] == "Fresh."


# --- summarize_sync --------------------------------------------------------

def test_summarize_sync_outside_event_loop():
    s = _summarizer(FakeLLM("Sync summary."))
    assert s.summarize_sync(document_text="t", doc_hash="e" * 64) == "Sync summary."


def test_summarize_sync_inside_running_loop():
    # The ingestion path can be reached from async request handlers; the sync
    # bridge must not try to re-enter the caller's running loop.
    s = _summarizer(FakeLLM("Nested summary."))

    async def caller():
        return s.summarize_sync(document_text="t", doc_hash="f" * 64)

    assert asyncio.run(caller()) == "Nested summary."


# --- _clean ----------------------------------------------------------------

def test_clean_collapses_whitespace_and_strips_quotes():
    assert _clean('  "A  spaced\n summary"  ') == "A spaced summary"
    assert _clean("'quoted'") == "quoted"
    assert _clean("") == ""
    assert _clean(None) == ""  # defensive: models can return None content
