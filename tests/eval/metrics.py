"""
Retrieval eval metrics — dependency-free, binary substring relevance.

A retrieved chunk is "relevant" to a query if any of the query's expected
substrings appears in it (case-insensitive). Substrings survive re-chunking
(chunk ids do not), so a golden set stays valid across ingestion/chunking
changes — exactly what we need to measure the structural-chunking fix.
"""

from __future__ import annotations


def _is_relevant(chunk_text: str, expected: list[str]) -> bool:
    low = chunk_text.lower()
    return any(sub.lower() in low for sub in expected)


def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """1.0 if any of the top-k chunks is relevant, else 0.0."""
    return 1.0 if any(_is_relevant(t, expected) for t in retrieved[:k]) else 0.0


def mrr_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Reciprocal rank of the first relevant chunk within top-k (0.0 if none)."""
    for rank, text in enumerate(retrieved[:k], start=1):
        if _is_relevant(text, expected):
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of the query's expected substrings found across the top-k chunks."""
    if not expected:
        return 0.0
    blob = "\n".join(retrieved[:k]).lower()
    found = sum(1 for sub in expected if sub.lower() in blob)
    return found / len(expected)


def aggregate(rows: list[tuple[list[str], list[str]]], k: int) -> dict:
    """Mean hit@k / mrr@k / recall@k over rows of (retrieved, expected)."""
    n = len(rows)
    if n == 0:
        return {"queries": 0, "k": k, "hit@k": 0.0, "mrr@k": 0.0, "recall@k": 0.0}
    return {
        "queries": n,
        "k": k,
        "hit@k": sum(hit_at_k(r, e, k) for r, e in rows) / n,
        "mrr@k": sum(mrr_at_k(r, e, k) for r, e in rows) / n,
        "recall@k": sum(recall_at_k(r, e, k) for r, e in rows) / n,
    }


# --- document-level metrics (ACTION_PLAN v2 §9) ----------------------------
# A chunk "comes from" the expected document if the expected name is a
# case-insensitive substring of the chunk's document name — same convention
# as chunk relevance above, robust to extension/prefix variants.

def _from_doc(doc_name: str, expected_doc: str) -> bool:
    return expected_doc.lower() in doc_name.lower()


def dochit_at_k(retrieved_docs: list[str], expected_doc: str, k: int) -> float:
    """1.0 if any top-k chunk comes from the expected document, else 0.0."""
    return 1.0 if any(_from_doc(d, expected_doc) for d in retrieved_docs[:k]) else 0.0


def drm_at_k(retrieved_docs: list[str], expected_doc: str, k: int) -> float:
    """
    Document-retrieval mismatch: fraction of the top-k chunks drawn from the
    WRONG document (Reuter's headline failure metric — lower is better).
    Empty retrieval counts as full mismatch.
    """
    top = retrieved_docs[:k]
    if not top:
        return 1.0
    return sum(1 for d in top if not _from_doc(d, expected_doc)) / len(top)


def aggregate_docs(rows: list[tuple[list[str], str]], k: int) -> dict:
    """Mean dochit@k / drm@k over rows of (retrieved doc names, expected doc)."""
    n = len(rows)
    if n == 0:
        return {"queries": 0, "k": k, "dochit@k": 0.0, "drm@k": 0.0}
    return {
        "queries": n,
        "k": k,
        "dochit@k": sum(dochit_at_k(r, e, k) for r, e in rows) / n,
        "drm@k": sum(drm_at_k(r, e, k) for r, e in rows) / n,
    }
