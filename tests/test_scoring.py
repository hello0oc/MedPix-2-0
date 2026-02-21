"""Unit tests for trial_matching_agent.scoring module.

Tests the TrialGPT scoring formulas, IR metrics, and edge cases.
All tests are offline — no API calls, no network.
"""
from __future__ import annotations

import math
import pytest

from trial_matching_agent.scoring import (
    classification_metrics,
    compute_ndcg,
    compute_precision_at_k,
    compute_recall_at_k,
    get_agg_score,
    get_matching_score,
    get_trial_score,
    rank_trials,
)


# ══════════════════════════════════════════════════════════════════════
# get_matching_score
# ══════════════════════════════════════════════════════════════════════

class TestGetMatchingScore:
    """Test the TrialGPT matching score formula."""

    def test_all_included_not_excluded(self):
        """Perfect match: all inclusion met, nothing excluded."""
        output = {
            "inclusion": {
                "0": ["reasoning", [0], "included"],
                "1": ["reasoning", [1], "included"],
            },
            "exclusion": {
                "0": ["reasoning", [0], "not excluded"],
                "1": ["reasoning", [0], "not excluded"],
            },
        }
        score = get_matching_score(output)
        # included=2, not_inc=0, no_info_inc=0
        # => 2/(2+0+0+0.1) = 2/2.1 ≈ 0.952
        assert score > 0.9

    def test_all_not_enough_info(self):
        """All criteria have insufficient information."""
        output = {
            "inclusion": {
                "0": ["reasoning", [], "not enough information"],
            },
            "exclusion": {
                "0": ["reasoning", [], "not enough information"],
            },
        }
        score = get_matching_score(output)
        # included=0, not_inc=0, no_info_inc=1
        # => 0/(0+0+1+0.1) = 0.0
        assert score == 0.0

    def test_excluded_penalty(self):
        """Having any exclusion match should reduce score."""
        base = {
            "inclusion": {"0": ["r", [0], "included"]},
            "exclusion": {"0": ["r", [0], "not excluded"]},
        }
        score_base = get_matching_score(base)

        with_exclusion = {
            "inclusion": {"0": ["r", [0], "included"]},
            "exclusion": {"0": ["r", [0], "excluded"]},
        }
        score_exc = get_matching_score(with_exclusion)
        assert score_exc < score_base

    def test_not_included_penalty(self):
        """Having 'not included' criteria should reduce score."""
        full = {
            "inclusion": {
                "0": ["r", [0], "included"],
                "1": ["r", [0], "included"],
            },
            "exclusion": {},
        }
        partial = {
            "inclusion": {
                "0": ["r", [0], "included"],
                "1": ["r", [0], "not included"],
            },
            "exclusion": {},
        }
        assert get_matching_score(partial) < get_matching_score(full)

    def test_empty_criteria(self):
        """No criteria at all."""
        score = get_matching_score({"inclusion": {}, "exclusion": {}})
        assert score == 0.0

    def test_not_applicable_ignored(self):
        """'not applicable' criteria should not count against score."""
        output = {
            "inclusion": {
                "0": ["r", [0], "included"],
                "1": ["r", [0], "not applicable"],
            },
            "exclusion": {
                "0": ["r", [0], "not applicable"],
            },
        }
        score = get_matching_score(output)
        # Only 1 included counted, 0 negatives
        assert score > 0.8


# ══════════════════════════════════════════════════════════════════════
# get_agg_score
# ══════════════════════════════════════════════════════════════════════

class TestGetAggScore:
    """Test the aggregation score computation."""

    def test_perfect_scores(self):
        agg = {"relevance_score_R": 100, "eligibility_score_E": 100}
        assert get_agg_score(agg) == pytest.approx(2.0)

    def test_zero_scores(self):
        agg = {"relevance_score_R": 0, "eligibility_score_E": 0}
        assert get_agg_score(agg) == pytest.approx(0.0)

    def test_mixed_scores(self):
        agg = {"relevance_score_R": 80, "eligibility_score_E": 60}
        assert get_agg_score(agg) == pytest.approx(1.4)


# ══════════════════════════════════════════════════════════════════════
# get_trial_score
# ══════════════════════════════════════════════════════════════════════

class TestGetTrialScore:
    def test_additive(self):
        m = 0.8
        a = 1.0
        score = get_trial_score(m, a)
        # Score is matching_score + agg_score (additive, not averaged)
        assert score == pytest.approx(m + a)

    def test_zero_both(self):
        assert get_trial_score(0.0, 0.0) == 0.0


# ══════════════════════════════════════════════════════════════════════
# NDCG
# ══════════════════════════════════════════════════════════════════════

class TestNDCG:
    def test_perfect_ranking(self):
        """If predicted ranking matches truth, NDCG = 1.0."""
        ranked = ["A", "B", "C"]
        true = {"A": 2, "B": 1, "C": 0}
        ndcg = compute_ndcg(ranked, true, k=3)
        assert ndcg == pytest.approx(1.0)

    def test_worst_ranking(self):
        """Reversed ranking should have lower NDCG."""
        ranked = ["C", "B", "A"]  # worst first
        true = {"A": 2, "B": 1, "C": 0}
        ndcg = compute_ndcg(ranked, true, k=3)
        assert ndcg < 1.0

    def test_single_item(self):
        ranked = ["A"]
        true = {"A": 2}
        assert compute_ndcg(ranked, true, k=1) == pytest.approx(1.0)

    def test_empty_input(self):
        assert compute_ndcg([], {}, k=5) == 0.0


# ══════════════════════════════════════════════════════════════════════
# Precision / Recall
# ══════════════════════════════════════════════════════════════════════

class TestPrecisionRecall:
    def test_precision_perfect(self):
        ranked = ["A", "B", "C"]
        true = {"A": 2, "B": 1, "C": 0}
        p = compute_precision_at_k(ranked, true, k=2, threshold=1)
        assert p == pytest.approx(1.0)

    def test_precision_none_relevant(self):
        ranked = ["A", "B"]
        true = {"A": 0, "B": 0}
        p = compute_precision_at_k(ranked, true, k=2, threshold=1)
        assert p == pytest.approx(0.0)

    def test_recall_partial(self):
        ranked = ["A", "B", "C"]
        true = {"A": 2, "B": 1, "C": 1}
        r = compute_recall_at_k(ranked, true, k=2, threshold=1)
        # Top-2 are A, B (both relevant), total relevant = 3
        assert r == pytest.approx(2 / 3)


# ══════════════════════════════════════════════════════════════════════
# Classification metrics
# ══════════════════════════════════════════════════════════════════════

class TestClassificationMetrics:
    def test_perfect_classification(self):
        true = [0, 1, 2, 0, 1, 2]
        pred = [0, 1, 2, 0, 1, 2]
        m = classification_metrics(true, pred)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["macro_f1"] == pytest.approx(1.0)

    def test_all_wrong(self):
        true = [0, 1, 2]
        pred = [2, 0, 1]
        m = classification_metrics(true, pred)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_empty_lists(self):
        m = classification_metrics([], [])
        assert m["accuracy"] == 0.0


# ══════════════════════════════════════════════════════════════════════
# rank_trials
# ══════════════════════════════════════════════════════════════════════

class TestRankTrials:
    def test_ordering(self):
        matching = {
            "A": {"inclusion": {"0": ["r", [0], "included"]}, "exclusion": {}},
            "B": {"inclusion": {"0": ["r", [0], "included"], "1": ["r", [0], "included"]}, "exclusion": {}},
            "C": {"inclusion": {"0": ["r", [0], "not included"]}, "exclusion": {}},
        }
        aggregation = {
            "A": {"relevance_score_R": 50, "eligibility_score_E": 50},
            "B": {"relevance_score_R": 90, "eligibility_score_E": 80},
            "C": {"relevance_score_R": 10, "eligibility_score_E": 10},
        }
        ranked = rank_trials(matching, aggregation)
        assert ranked[0][0] == "B"  # highest combined score
        assert ranked[-1][0] == "C"  # lowest

    def test_empty(self):
        assert rank_trials({}, {}) == []
