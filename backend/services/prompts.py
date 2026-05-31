"""
Prompt construction for grounded legal Q&A.

The system prompt is deliberately strict: answer ONLY from the supplied context,
cite every claim, and decline when the answer is not present. This is the
primary defence against hallucination in a high-stakes legal domain.
"""

from __future__ import annotations

from models.schemas import ChunkHit

# Returned verbatim when retrieval yields no usable context.
NO_CONTEXT_ANSWER = (
    "I could not find information about that in the available legal documents."
)

SYSTEM_PROMPT = """You are a precise legal research assistant. You answer \
questions strictly using the excerpts from legal documents provided in the \
CONTEXT below.

Rules:
1. Use ONLY the information in the CONTEXT. Do not rely on outside knowledge.
2. Cite the source for every claim using the inline form [pdf_name p.N] that \
appears in the context headers.
3. If the CONTEXT does not contain the answer, reply exactly: \
"I could not find information about that in the available legal documents." \
Do not guess or fabricate.
4. Quote statutory language precisely; do not paraphrase section numbers.
5. Be concise and neutral. Do not give legal advice — report what the \
documents say."""


def _page_ref(hit: ChunkHit) -> str:
    return (
        f"p.{hit.page_start}"
        if hit.page_start == hit.page_end
        else f"pp.{hit.page_start}-{hit.page_end}"
    )


def select_hits_within(hits: list[ChunkHit], *, max_chars: int) -> list[ChunkHit]:
    """
    Return the leading sublist of hits whose rendered size fits in max_chars.
    Always keeps at least the first hit (so a single huge chunk still answers).
    """
    selected: list[ChunkHit] = []
    used = 0
    for hit in hits:
        block_len = len(f"[{hit.pdf_name} {_page_ref(hit)}]\n{hit.content}")
        if selected and used + block_len > max_chars:
            break
        selected.append(hit)
        used += block_len
    return selected


def format_context(hits: list[ChunkHit]) -> str:
    """Render the given chunks into one citable context block."""
    blocks = [
        f"[{hit.pdf_name} {_page_ref(hit)}]\n{hit.content}" for hit in hits
    ]
    return "\n\n---\n\n".join(blocks)


def build_messages(query: str, context: str) -> list[dict[str, str]]:
    user = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the CONTEXT above, with inline [source p.N] citations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
