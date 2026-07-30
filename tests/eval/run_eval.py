"""
Retrieval eval harness — scores the LIVE search stack against a golden set.

Needs a running Typesense with your corpus indexed (same as the app). Run:

    python tests/eval/run_eval.py
    python tests/eval/run_eval.py --k 8 --mode vector
    python tests/eval/run_eval.py --golden path/to/your_golden.json
    python tests/eval/run_eval.py --two-stage   # document-scoped retrieval

Reports mean Hit@k, MRR@k and Recall@k plus per-query detail, so every
retrieval change (chunking, hybrid alpha, a reranker) gets a NUMBER instead of
a vibe. Re-run before and after each change and compare.
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
    args = parser.parse_args()

    k_default, queries = load_golden(args.golden)
    k = args.k or k_default
    search = build_search_service()

    print(
        f"\nRetrieval eval — {len(queries)} queries, mode={args.mode}, k={k}"
        + (", two-stage" if args.two_stage else "")
    )
    print("-" * 72)

    rows: list[tuple[list[str], list[str]]] = []
    for q in queries:
        try:
            if args.two_stage:
                resp, _ = search.search_two_stage(
                    q["query"], top_k=k, mode=args.mode
                )
            else:
                resp = search.search(q["query"], top_k=k, mode=args.mode)
            retrieved = [h.content for h in resp.hits]
        except Exception as exc:  # noqa: BLE001 — harness: report and abort
            print(f"  ! {q['query'][:50]!r}: search failed: {exc}")
            print("\nIs Typesense running and the corpus indexed? Aborting.")
            return 1

        expected = q["expected"]
        rows.append((retrieved, expected))
        print(
            f"  hit={metrics.hit_at_k(retrieved, expected, k):.0f} "
            f"rr={metrics.mrr_at_k(retrieved, expected, k):.2f} "
            f"rec={metrics.recall_at_k(retrieved, expected, k):.2f}  "
            f"{q['query'][:52]}"
        )

    agg = metrics.aggregate(rows, k)
    print("-" * 72)
    print(
        f"  MEAN  Hit@{k}={agg['hit@k']:.3f}  MRR@{k}={agg['mrr@k']:.3f}  "
        f"Recall@{k}={agg['recall@k']:.3f}  (n={agg['queries']})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
