"""
Prompt construction for grounded legal Q&A.

The system prompt is deliberately strict: answer ONLY from the supplied context,
cite every claim, and decline when the answer is not present. This is the
primary defence against hallucination in a high-stakes legal domain.
"""

from __future__ import annotations

import logging

from models.schemas import ChunkHit
from services.cite import span_id_for

logger = logging.getLogger(__name__)

# Returned verbatim when retrieval yields no usable context.
NO_CONTEXT_ANSWER = (
    "I could not find information about that in the available legal documents."
)

# Returned when retrieval found context but verification could not clear the
# gate within its retry budget. Distinct from NO_CONTEXT_ANSWER: there *was*
# material, we just could not stand behind an answer drawn from it.
UNVERIFIED_ANSWER = (
    "I found related material but could not verify an answer against it with "
    "enough confidence to report one. The sources retrieved are listed below."
)

SYSTEM_PROMPT = """You are a precise legal research assistant. You answer \
questions strictly using the excerpts from legal documents provided in the \
CONTEXT below.

Rules:
1. Use ONLY the information in the CONTEXT. Do not rely on outside knowledge.
2. Every context excerpt is labelled with a span id like [S1]. Cite the span \
id(s) supporting each factual claim inline, e.g. "...within 30 days [S2]." \
Cite only span ids that appear in the CONTEXT.
3. If the CONTEXT contains material relevant to the question, answer from it — \
even when it only partially addresses the question. Reply "I could not find \
information about that in the available legal documents." ONLY when the CONTEXT \
has nothing at all on the subject. Do not guess or fabricate, but do not refuse \
when relevant text is present.
4. Quote statutory language precisely; do not paraphrase section numbers.
5. Be concise and neutral. Do not give legal advice — report what the \
documents say."""


def _page_ref(hit: ChunkHit) -> str:
    return (
        f"p.{hit.page_start}"
        if hit.page_start == hit.page_end
        else f"pp.{hit.page_start}-{hit.page_end}"
    )


def _block(hit: ChunkHit, index: int) -> str:
    """One context excerpt: span label + human-readable source header + body."""
    return f"[{span_id_for(index)}] [{hit.pdf_name} {_page_ref(hit)}]\n{hit.content}"


def select_hits_within(hits: list[ChunkHit], *, max_chars: int) -> list[ChunkHit]:
    """
    Return the leading sublist of hits whose rendered size fits in max_chars.
    Always keeps at least the first hit (so a single huge chunk still answers).
    """
    selected: list[ChunkHit] = []
    used = 0
    for i, hit in enumerate(hits):
        block_len = len(_block(hit, i))
        if selected and used + block_len > max_chars:
            logger.debug(
                "Context limit hit at chunk %d (%s): %d + %d > %d (max_chars)",
                i, hit.pdf_name, used, block_len, max_chars,
            )
            break
        selected.append(hit)
        used += block_len
        logger.debug(
            "Selected chunk %d (%s p.%s, idx=%d): %d chars, cumulative=%d/%d",
            i, hit.pdf_name, _page_ref(hit), hit.chunk_index, block_len, used, max_chars,
        )

    if len(selected) < len(hits):
        dropped = len(hits) - len(selected)
        logger.info(
            "Context selection: %d/%d chunks kept (dropped %d due to char limit)",
            len(selected), len(hits), dropped,
        )

    return selected


def format_context(hits: list[ChunkHit]) -> str:
    """
    Render the given chunks into one citable context block.

    Span labels are positional ([S1] = hits[0]), matching services.cite's
    binding of the answer's labels back to these same hits.
    """
    blocks = [_block(hit, i) for i, hit in enumerate(hits)]
    return "\n\n---\n\n".join(blocks)


def build_messages(query: str, context: str) -> list[dict[str, str]]:
    user = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the CONTEXT above, citing the supporting span id "
        "inline after each factual claim, e.g. [S1]."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_rewrite_messages(query: str) -> list[dict[str, str]]:
    """One-shot query rewrite for the Gate-1 weak-retrieval retry (L11)."""
    user = (
        "Rewrite this legal research question to be clearer and more specific, "
        "so a document search is more likely to find its answer. Preserve its "
        "meaning. Reply with ONLY the rewritten question.\n\n"
        f"QUESTION: {query}"
    )
    return [{"role": "user", "content": user}]
