"""
Claim → span citation binding.

The generator is told to label every factual claim with a span id ([S1], [S2], …)
matching the retrieved excerpts it was given. This module reads those labels back
out and binds each to a concrete `(document_id, start_char, end_char)`, which is
what makes an answer auditable: a reader can go to the exact source characters
rather than to a page number they have to scan.

Two outputs matter:

  - `citations` — the bound spans, in the order first cited.
  - `citation_coverage` — the fraction of the answer's factual sentences that
    carry a resolvable span label. A confident, fluent, uncited answer is the
    dangerous failure mode in this domain, so it gets a number rather than a
    vibe.

A label the model invents ([S9] when only 3 spans were supplied) is dropped, not
guessed at — binding a claim to the wrong source is worse than leaving it
unbound, and the coverage ratio records the loss either way.
"""

from __future__ import annotations

import logging
import re

from models.schemas import ChunkHit, Citation

logger = logging.getLogger(__name__)

# Span labels as emitted by the prompt: [S1], [S2, S3], [S1][S4].
_SPAN_LABEL_RE = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]", re.IGNORECASE)
_SPAN_ID_RE = re.compile(r"S\d+", re.IGNORECASE)

# Sentence-ish split. Deliberately crude: legal text is full of "s. 5(1)(a)" and
# "No. 3 of 1956", and a real sentence tokenizer buys accuracy we do not need to
# compute a ratio.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")

# A fragment shorter than this is a heading, list marker, or stub — not a claim
# that needs its own citation.
_MIN_CLAIM_CHARS = 25


def span_id_for(index: int) -> str:
    """Label for the nth retrieved excerpt (0-based)."""
    return f"S{index + 1}"


def extract_span_ids(text: str) -> list[str]:
    """Span labels in `text`, uppercased, in order of first appearance."""
    found: list[str] = []
    for match in _SPAN_LABEL_RE.finditer(text):
        for raw in _SPAN_ID_RE.findall(match.group(1)):
            span_id = raw.upper()
            if span_id not in found:
                found.append(span_id)
    return found


def to_citation(hit: ChunkHit, span_id: str | None = None) -> Citation:
    """Bind one retrieved chunk to its citable source span."""
    return Citation(
        pdf_id=hit.pdf_id,
        pdf_name=hit.pdf_name,
        page_start=hit.page_start,
        page_end=hit.page_end,
        chunk_index=hit.chunk_index,
        file_url=hit.file_url,
        start_char=hit.start_char,
        end_char=hit.end_char,
        span_id=span_id,
    )


def bind_citations(
    answer: str, hits: list[ChunkHit]
) -> tuple[list[Citation], float]:
    """
    Bind the answer's span labels to retrieved chunks.

    Returns the citations actually referenced and the citation coverage ratio.
    When the answer carries no labels at all — an older prompt, or a model that
    ignored the instruction — every retrieved chunk is returned as an unlabelled
    citation with coverage 0.0, so the gap is visible instead of being silently
    reported as fully cited.
    """
    by_span = {span_id_for(i): hit for i, hit in enumerate(hits)}
    cited_ids = extract_span_ids(answer)

    unknown = [s for s in cited_ids if s not in by_span]
    if unknown:
        logger.warning(
            "Answer cited %s but only %d spans were supplied — dropping.",
            unknown, len(hits),
        )

    resolved = [s for s in cited_ids if s in by_span]
    if not resolved:
        return [to_citation(hit) for hit in hits], 0.0

    citations = [to_citation(by_span[s], span_id=s) for s in resolved]
    return citations, citation_coverage(answer)


def citation_coverage(answer: str) -> float:
    """
    Fraction of the answer's factual sentences carrying a span label.

    1.0 when there is nothing substantive to cite (an empty answer or a bare
    refusal), so an abstention is never penalised as uncited.
    """
    claims = [
        s for s in _SENTENCE_RE.split(answer.strip()) if len(s.strip()) >= _MIN_CLAIM_CHARS
    ]
    if not claims:
        return 1.0
    cited = sum(1 for claim in claims if _SPAN_LABEL_RE.search(claim))
    return cited / len(claims)
