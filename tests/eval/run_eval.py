"""
Retrieval eval harness — scores the LIVE search stack against a golden set.

Needs a running Typesense with your corpus indexed (same as the app). Run:

    python tests/eval/run_eval.py
    python tests/eval/run_eval.py --k 8 --mode vector
    python tests/eval/run_eval.py --golden path/to/your_golden.json
    python tests/eval/run_eval.py --two-stage   # document-scoped retrieval
    python tests/eval/run_eval.py --routed      # L6 per-query routing

Reports mean Hit@k, MRR@k and Recall@k plus per-query detail, so every
retrieval change (chunking, hybrid alpha, a reranker) gets a NUMBER instead of
a vibe. Re-run before and after each change and compare.

Golden queries may carry an optional "expected_doc" (substring of the source
document's name); those queries additionally report DocHit@k and DRM@k —
document-retrieval mismatch, the failure mode ACTION_PLAN v2 is built around.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))                              # local `import metrics`
sys.path.insert(0, str(HERE.parent.parent / "backend"))   # backend package

import metrics  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.dependencies import (  # noqa: E402
    get_embedding_service,
    get_typesense_repository,
)
from services.exact_terms import detect_exact_terms  # noqa: E402
from services.search_service import SearchService  # noqa: E402


def load_golden(path: str) -> tuple[int, list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return int(data.get("k", 5)), data["queries"]


def build_search_service() -> SearchService:
    return SearchService(
        embedding_service=get_embedding_service(),
        typesense_repo=get_typesense_repository(),
        settings=get_settings(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=str(HERE / "golden_set.json"))
    parser.add_argument("--k", type=int, default=None, help="override golden-set k")
    parser.add_argument(
        "--mode", default="hybrid", choices=["keyword", "vector", "hybrid"]
    )
    parser.add_argument(
        "--two-stage", action="store_true",
        help="route through document-scoped two-stage retrieval "
             "(uses TWO_STAGE_* settings regardless of TWO_STAGE_ENABLED)",
    )
    parser.add_argument(
        "--routed", action="store_true",
        help="L6 per-query routing: hybrid when the exact-term detector "
             "fires, vector (dense-only) otherwise — overrides --mode; "
             "benchmark this before enabling EXACT_TERM_ROUTING_ENABLED",
    )
    args = parser.parse_args()

    k_default, queries = load_golden(args.golden)
    k = args.k or k_default
    search = build_search_service()

    print(
        f"\nRetrieval eval — {len(queries)} queries, "
        + ("mode=routed" if args.routed else f"mode={args.mode}")
        + f", k={k}"
        + (", two-stage" if args.two_stage else "")
    )
    print("-" * 72)

    rows: list[tuple[list[str], list[str]]] = []
    doc_rows: list[tuple[list[str], str]] = []
    for q in queries:
        if args.routed:
            mode = "hybrid" if detect_exact_terms(q["query"]).flagged else "vector"
        else:
            mode = args.mode
        try:
            if args.two_stage:
                resp, _ = search.search_two_stage(q["query"], top_k=k, mode=mode)
            else:
                resp = search.search(q["query"], top_k=k, mode=mode)
            retrieved = [h.content for h in resp.hits]
            retrieved_docs = [h.pdf_name for h in resp.hits]
        except Exception as exc:  # noqa: BLE001 — harness: report and abort
            print(f"  ! {q['query'][:50]!r}: search failed: {exc}")
            print("\nIs Typesense running and the corpus indexed? Aborting.")
            return 1

        expected = q["expected"]
        rows.append((retrieved, expected))
        doc_detail = ""
        if q.get("expected_doc"):
            doc_rows.append((retrieved_docs, q["expected_doc"]))
            doc_detail = (
                f"doc={metrics.dochit_at_k(retrieved_docs, q['expected_doc'], k):.0f} "
                f"drm={metrics.drm_at_k(retrieved_docs, q['expected_doc'], k):.2f} "
            )
        print(
            f"  hit={metrics.hit_at_k(retrieved, expected, k):.0f} "
            f"rr={metrics.mrr_at_k(retrieved, expected, k):.2f} "
            f"rec={metrics.recall_at_k(retrieved, expected, k):.2f}  "
            f"{doc_detail}"
            f"{q['query'][:52]}"
        )

    agg = metrics.aggregate(rows, k)
    print("-" * 72)
    print(
        f"  MEAN  Hit@{k}={agg['hit@k']:.3f}  MRR@{k}={agg['mrr@k']:.3f}  "
        f"Recall@{k}={agg['recall@k']:.3f}  (n={agg['queries']})"
    )
    if doc_rows:
        doc_agg = metrics.aggregate_docs(doc_rows, k)
        print(
            f"  MEAN  DocHit@{k}={doc_agg['dochit@k']:.3f}  "
            f"DRM@{k}={doc_agg['drm@k']:.3f}  (n={doc_agg['queries']} with expected_doc)"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
