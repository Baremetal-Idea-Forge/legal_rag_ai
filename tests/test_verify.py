"""services/verify.py — triad parsing, gating semantics, judge failure."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from models.schemas import VerificationScores
from services.verify import AnswerVerifier, _parse_scores


class FakeLLM:
    def __init__(self, reply: object):
        self._reply = reply
        self.calls = 0

    async def chat(self, messages, *, temperature=0.2):
        self.calls += 1
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


def _verify(reply):
    v = AnswerVerifier(llm_client=FakeLLM(reply))
    return asyncio.run(v.verify(query="q", context="c", answer="a"))


# --- VerificationScores.minimum -------------------------------------------

def test_minimum_is_the_gate_not_the_mean():
    # The Wahidur failure shape: high answer relevance, low groundedness.
    scores = VerificationScores(
        context_relevance=0.9, groundedness=0.26, answer_relevance=0.88
    )
    assert scores.minimum == 0.26


# --- _parse_scores ---------------------------------------------------------

class TestParseScores:
    def test_strict_json(self):
        s = _parse_scores(
            '{"context_relevance": 0.8, "groundedness": 0.7, "answer_relevance": 0.9}'
        )
        assert s == VerificationScores(
            context_relevance=0.8, groundedness=0.7, answer_relevance=0.9
        )

    def test_json_embedded_in_prose(self):
        s = _parse_scores(
            'Here are the scores:\n{"context_relevance": 1, "groundedness": 0.5, '
            '"answer_relevance": 1}\nHope that helps!'
        )
        assert s is not None and s.groundedness == 0.5

    def test_out_of_range_values_are_clamped(self):
        s = _parse_scores(
            '{"context_relevance": 1.7, "groundedness": -0.2, "answer_relevance": 0.5}'
        )
        assert s is not None
        assert s.context_relevance == 1.0
        assert s.groundedness == 0.0

    def test_missing_key_returns_none(self):
        assert _parse_scores('{"groundedness": 0.9}') is None

    def test_non_numeric_returns_none(self):
        assert (
            _parse_scores(
                '{"context_relevance": "high", "groundedness": 1, "answer_relevance": 1}'
            )
            is None
        )

    def test_no_json_returns_none(self):
        assert _parse_scores("The answer looks fine to me.") is None
        assert _parse_scores("") is None


# --- AnswerVerifier.verify -------------------------------------------------

def test_verify_returns_scores():
    s = _verify('{"context_relevance": 0.9, "groundedness": 0.8, "answer_relevance": 0.7}')
    assert s is not None and s.minimum == 0.7


def test_verify_llm_failure_returns_none_never_raises():
    assert _verify(RuntimeError("judge down")) is None


def test_verify_unparseable_reply_returns_none():
    assert _verify("I refuse to grade this.") is None


def test_verify_sends_query_context_answer_to_judge():
    llm = FakeLLM('{"context_relevance": 1, "groundedness": 1, "answer_relevance": 1}')
    v = AnswerVerifier(llm_client=llm)

    captured = {}

    async def chat(messages, *, temperature=0.2):
        captured["prompt"] = messages[0]["content"]
        captured["temperature"] = temperature
        return '{"context_relevance": 1, "groundedness": 1, "answer_relevance": 1}'

    llm.chat = chat
    asyncio.run(v.verify(query="THE QUERY", context="THE CONTEXT", answer="THE ANSWER"))
    assert "THE QUERY" in captured["prompt"]
    assert "THE CONTEXT" in captured["prompt"]
    assert "THE ANSWER" in captured["prompt"]
    assert captured["temperature"] == 0.0  # judging must be deterministic
