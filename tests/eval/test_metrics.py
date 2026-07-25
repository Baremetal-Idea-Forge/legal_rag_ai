"""Unit tests for the retrieval eval metrics (pure — no live stack)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import metrics


class TestHitAtK:
    def test_relevant_in_top_k(self):
        retrieved = ["nope", "Article 21 protects life", "other"]
        assert metrics.hit_at_k(retrieved, ["Article 21"], k=3) == 1.0

    def test_relevant_beyond_k_is_miss(self):
        retrieved = ["nope", "nope", "Article 21 protects life"]
        assert metrics.hit_at_k(retrieved, ["Article 21"], k=2) == 0.0

    def test_no_relevant(self):
        assert metrics.hit_at_k(["a", "b"], ["Article 21"], k=2) == 0.0


class TestMrrAtK:
    def test_first_relevant_at_rank_2(self):
        retrieved = ["irrelevant", "Article 21 here", "Article 21 again"]
        assert metrics.mrr_at_k(retrieved, ["Article 21"], k=3) == 0.5

    def test_first_relevant_at_rank_1(self):
        assert metrics.mrr_at_k(["Article 21"], ["Article 21"], k=1) == 1.0

    def test_none_relevant(self):
        assert metrics.mrr_at_k(["x", "y"], ["Article 21"], k=2) == 0.0


class TestRecallAtK:
    def test_partial_recall(self):
        retrieved = ["mentions Article 21 clearly"]
        expected = ["Article 21", "life or personal liberty", "procedure"]
        assert metrics.recall_at_k(retrieved, expected, k=1) == 1 / 3

    def test_full_recall(self):
        retrieved = ["Article 21 — life or personal liberty by procedure"]
        expected = ["Article 21", "life or personal liberty", "procedure"]
        assert metrics.recall_at_k(retrieved, expected, k=1) == 1.0

    def test_empty_expected(self):
        assert metrics.recall_at_k(["anything"], [], k=1) == 0.0


def test_case_insensitive_matching():
    assert metrics.hit_at_k(["ARTICLE 21 in caps"], ["article 21"], k=1) == 1.0


class TestAggregate:
    def test_averages_over_rows(self):
        rows = [
            (["Article 21 hit"], ["Article 21"]),   # hit=1, rr=1, rec=1
            (["miss", "miss"], ["Article 21"]),      # hit=0, rr=0, rec=0
        ]
        agg = metrics.aggregate(rows, k=2)
        assert agg["queries"] == 2
        assert agg["hit@k"] == 0.5
        assert agg["mrr@k"] == 0.5
        assert agg["recall@k"] == 0.5

    def test_empty_rows(self):
        agg = metrics.aggregate([], k=5)
        assert agg == {"queries": 0, "k": 5, "hit@k": 0.0, "mrr@k": 0.0, "recall@k": 0.0}
