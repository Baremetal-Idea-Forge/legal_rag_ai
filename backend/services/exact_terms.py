"""
Exact-term query detector — the L6 routing signal (ACTION_PLAN v2 §2.3).

Dense-only retrieval is the v2 default: BM25 helps bind *documents* but costs
span-level precision (Reuter App. B), so the lexical (hybrid) channel fires
only when exact-term matching is the point of the query. This detector is the
only thing that can activate that route. It is rule-based and CPU-cheap by
design — three signals, no learned model:

  - quoted_phrase:  "consequential damages" (straight or curly quotes)
  - section_number: § 12, Section 302, Art. 21, Order XXI, Rule 4(2)
  - defined_term:   a run of 2+ Capitalized Words past the start of the query
                    ("...under the Sale of Goods Act"). Runs at position 0 are
                    ignored — that is ordinary sentence capitalization.

False positives are cheap (the query is searched in hybrid mode, the pre-v2
default); false negatives lose only the lexical boost. Bias toward precision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "..." or “...” with at least two characters inside.
_QUOTED_PHRASE_RE = re.compile(r'"[^"]{2,}"|“[^”]{2,}”')

# § / Section / Article / Rule / Order / Clause / Regulation followed by an
# arabic number (with optional suffixes: 12A, 4(2), 300.1) or an UPPERCASE
# Roman numeral. The keyword is case-insensitive; the Roman numeral is not
# ("order i placed" must not fire).
_SECTION_NUMBER_RE = re.compile(
    r"""
    (?: § | \b(?:sections?|sec|ss?|articles?|arts?|rules?|orders?|clauses?|cl|regulations?|regs?)\.? )
    \s*
    (?: \d+[0-9A-Za-z().\-]* | (?-i:[IVXLCDM]+)\b )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Two or more consecutive Capitalized Words, optionally joined by the small
# connectives common in statute names ("Sale of Goods Act").
_CAP_WORD = r"[A-Z][\w'&-]*"
_DEFINED_TERM_RE = re.compile(
    rf"\b{_CAP_WORD}(?:\s+(?:of|the|and|for|in)\s+{_CAP_WORD}|\s+{_CAP_WORD})+"
)


@dataclass(frozen=True)
class ExactTermFlag:
    """Detector outcome; `reasons` names the signals that fired (for audit)."""

    flagged: bool
    reasons: tuple[str, ...]


def detect_exact_terms(query: str) -> ExactTermFlag:
    """Flag queries where exact-term (lexical) matching is the point."""
    reasons: list[str] = []
    if _QUOTED_PHRASE_RE.search(query):
        reasons.append("quoted_phrase")
    if _SECTION_NUMBER_RE.search(query):
        reasons.append("section_number")
    match = _DEFINED_TERM_RE.search(query)
    if match and match.start() > 0:
        reasons.append("defined_term")
    return ExactTermFlag(flagged=bool(reasons), reasons=tuple(reasons))
