"""
Answer verification — the RAG triad.

Three scores in 0..1, each judging a different failure:

  - context relevance:  did retrieval bring back material that addresses the
                        query at all?
  - groundedness:       is each claim in the answer supported by the retrieved
                        text, rather than by the model's own knowledge?
  - answer relevance:   does the answer address the question that was asked?

The caller gates on min(triad), never the mean. The triad exists precisely
because the three fail independently — a fluent answer can be highly relevant
to the question while barely grounded in the sources, and averaging hides
exactly that case.

The judge is one LLM call returning strict JSON. When the judge is unavailable
or its reply cannot be parsed, `verify` returns None and the caller decides;
here the answer path treats None as "gate skipped" — verification is a quality
gate, not a dependency the whole answer path should die on.
"""

from __future__ import annotations

import json
import logging
import re

from models.schemas import VerificationScores

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """You are grading one answer produced by a legal document \
Q&A system. Score each dimension independently from 0.0 (complete failure) \
to 1.0 (perfect), using only the material given below.

- context_relevance: the CONTEXT contains material that addresses the QUESTION.
- groundedness: every factual claim in the ANSWER is supported by the CONTEXT. \
Penalise any claim that goes beyond it, however plausible.
- answer_relevance: the ANSWER addresses the QUESTION that was asked.

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
{answer}

Reply with ONLY a JSON object, no prose:
{{"context_relevance": <float>, "groundedness": <float>, "answer_relevance": <float>}}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class AnswerVerifier:
    """Scores (query, context, answer) triads with an LLM judge."""

    def __init__(self, *, llm_client: object) -> None:
        self._llm = llm_client

    async def verify(
        self, *, query: str, context: str, answer: str
    ) -> VerificationScores | None:
        """Triad scores, or None when the judge failed (caller skips the gate)."""
        prompt = _JUDGE_PROMPT.format(query=query, context=context, answer=answer)
        try:
            raw = await self._llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.0
            )
        except Exception as exc:  # noqa: BLE001 — a gate, never fatal
            logger.warning("Verifier LLM call failed: %s", exc)
            return None
        scores = _parse_scores(raw)
        if scores is None:
            logger.warning("Verifier reply unparseable: %s", (raw or "")[:200])
        return scores


def _parse_scores(raw: str) -> VerificationScores | None:
    """Extract the triad from a judge reply; None if it isn't there."""
    match = _JSON_RE.search(raw or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return VerificationScores(
            context_relevance=_clamp(data["context_relevance"]),
            groundedness=_clamp(data["groundedness"]),
            answer_relevance=_clamp(data["answer_relevance"]),
        )
    except (ValueError, KeyError, TypeError):
        return None


def _clamp(value: object) -> float:
    return min(1.0, max(0.0, float(value)))  # type: ignore[arg-type]
