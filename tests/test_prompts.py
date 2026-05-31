"""Prompt construction — context selection, formatting, message assembly."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from models.schemas import ChunkHit
from services import prompts


def _hit(pdf_name="act.pdf", page_start=1, page_end=1, content="text", **kw):
    return ChunkHit(
        id=kw.get("id", "c0"),
        pdf_id=kw.get("pdf_id", "p0"),
        pdf_name=pdf_name,
        page_start=page_start,
        page_end=page_end,
        chunk_index=kw.get("chunk_index", 0),
        content=content,
    )


class TestSelectHitsWithin:
    def test_keeps_all_when_within_budget(self):
        hits = [_hit(content="a" * 50) for _ in range(3)]
        selected = prompts.select_hits_within(hits, max_chars=10_000)
        assert len(selected) == 3

    def test_truncates_to_budget(self):
        hits = [_hit(content="a" * 500) for _ in range(10)]
        selected = prompts.select_hits_within(hits, max_chars=1200)
        assert 1 <= len(selected) < 10

    def test_always_keeps_first_even_if_oversized(self):
        hits = [_hit(content="a" * 5000)]
        selected = prompts.select_hits_within(hits, max_chars=100)
        assert len(selected) == 1

    def test_empty(self):
        assert prompts.select_hits_within([], max_chars=1000) == []


class TestFormatContext:
    def test_single_page_header(self):
        ctx = prompts.format_context([_hit(pdf_name="ipc.pdf", page_start=7, page_end=7)])
        assert "[ipc.pdf p.7]" in ctx

    def test_page_range_header(self):
        ctx = prompts.format_context([_hit(pdf_name="ipc.pdf", page_start=7, page_end=9)])
        assert "[ipc.pdf pp.7-9]" in ctx

    def test_includes_content(self):
        ctx = prompts.format_context([_hit(content="offer and acceptance")])
        assert "offer and acceptance" in ctx

    def test_separator_between_chunks(self):
        ctx = prompts.format_context([_hit(content="a"), _hit(content="b")])
        assert "---" in ctx


class TestBuildMessages:
    def test_has_system_and_user(self):
        msgs = prompts.build_messages("What is tort?", "CTX")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_contains_query_and_context(self):
        msgs = prompts.build_messages("What is tort?", "some context here")
        assert "What is tort?" in msgs[1]["content"]
        assert "some context here" in msgs[1]["content"]

    def test_system_prompt_enforces_grounding(self):
        msgs = prompts.build_messages("q", "c")
        assert "ONLY" in msgs[0]["content"]


def test_no_context_answer_constant():
    assert "could not find" in prompts.NO_CONTEXT_ANSWER.lower()
