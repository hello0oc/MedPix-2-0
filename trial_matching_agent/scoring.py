"""Scoring functions — exact ports of TrialGPT formulas.

This module reproduces the scoring logic from TrialGPT's
``trialgpt_ranking/rank_results.py`` for direct apples-to-apples comparison,
plus standard IR metrics (NDCG, precision@k).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

EPS = 1e-9


# ──────────────────────────────────────────────────────────────────────────
# TrialGPT criterion-level matching score
# ──────────────────────────────────────────────────────────────────────────
def get_matching_score(matching: Dict[str, Any]) -> float:
    """Compute the TrialGPT matching score from criterion-level predictions.

    Exact port of ``rank_results.py :: get_matching_score()``.

    Parameters
    ----------
    matching : dict
        ``{"inclusion": {"0": [reasoning, [sent_ids], label], ...},
           "exclusion": {"0": [reasoning, [sent_ids], label], ...}}``

    Returns
    -------
    float  — matching score (higher = more eligible)
    """
    included = 0
    not_inc = 0
    na_inc = 0
    no_info_inc = 0

    excluded = 0
    not_exc = 0
    na_exc = 0
    no_info_exc = 0

    # Count inclusion labels
    for _criterion_idx, info in (matching.get("inclusion") or {}).items():
        if not isinstance(info, list) or len(info) != 3:
            continue
        label = info[2]
        if label == "included":
            included += 1
        elif label == "not included":
            not_inc += 1
        elif label == "not applicable":
            na_inc += 1
        elif label == "not enough information":
            no_info_inc += 1

    # Count exclusion labels
    for _criterion_idx, info in (matching.get("exclusion") or {}).items():
        if not isinstance(info, list) or len(info) != 3:
            continue
        label = info[2]
        if label == "excluded":
            excluded += 1
        elif label == "not excluded":
            not_exc += 1
        elif label == "not applicable":
            na_exc += 1
        elif label == "not enough information":
            no_info_exc += 1

    # Compute score — exact TrialGPT formula
    score = 0.0
    score += included / (included + not_inc + no_info_inc + EPS)

    if not_inc > 0:
        score -= 1

    if excluded > 0:
        score -= 1

    return score


# ──────────────────────────────────────────────────────────────────────────
# TrialGPT aggregation score
# ──────────────────────────────────────────────────────────────────────────
def get_agg_score(assessment: Dict[str, Any]) -> float:
    """Compute the aggregation score from R and E values.

    Exact port of ``rank_results.py :: get_agg_score()``.
    """
    try:
        rel_score = float(assessment.get("relevance_score_R", 0))
        eli_score = float(assessment.get("eligibility_score_E", 0))
    except (TypeError, ValueError):
        rel_score = 0.0
        eli_score = 0.0

    return (rel_score + eli_score) / 100.0


# ──────────────────────────────────────────────────────────────────────────
# Combined trial score
# ──────────────────────────────────────────────────────────────────────────
def get_trial_score(matching_score: float, agg_score: float) -> float:
    """Combined ranking score = matching_score + agg_score."""
    return matching_score + agg_score


# ──────────────────────────────────────────────────────────────────────────
# Standard IR metrics
# ──────────────────────────────────────────────────────────────────────────
def _dcg(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because rank is 1-indexed
    return dcg


def compute_ndcg(ranked_nct_ids: List[str],
                 qrels: Dict[str, int],
                 k: int) -> float:
    """Compute NDCG@k for a ranked list of NCT IDs against ground truth.

    Parameters
    ----------
    ranked_nct_ids : list[str]
        Trial IDs in ranked order (best first).
    qrels : dict[str, int]
        Ground truth: {nct_id: relevance_label}.
    k : int
        Rank cutoff.

    Returns
    -------
    float — NDCG@k in [0, 1].
    """
    if not qrels or not ranked_nct_ids:
        return 0.0

    # Actual relevances in ranked order
    relevances = [float(qrels.get(nct_id, 0)) for nct_id in ranked_nct_ids[:k]]
    dcg = _dcg(relevances, k)

    # Ideal relevances
    ideal = sorted(qrels.values(), reverse=True)
    ideal_rels = [float(r) for r in ideal[:k]]
    idcg = _dcg(ideal_rels, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_precision_at_k(ranked_nct_ids: List[str],
                           qrels: Dict[str, int],
                           k: int,
                           threshold: int = 1) -> float:
    """Precision@k: fraction of top-k results with relevance >= threshold."""
    if not ranked_nct_ids:
        return 0.0
    relevant = sum(
        1 for nct_id in ranked_nct_ids[:k]
        if qrels.get(nct_id, 0) >= threshold
    )
    return relevant / min(k, len(ranked_nct_ids))


def compute_recall_at_k(ranked_nct_ids: List[str],
                        qrels: Dict[str, int],
                        k: int,
                        threshold: int = 1) -> float:
    """Recall@k: fraction of relevant items that appear in top-k."""
    total_relevant = sum(1 for v in qrels.values() if v >= threshold)
    if total_relevant == 0:
        return 0.0
    recalled = sum(
        1 for nct_id in ranked_nct_ids[:k]
        if qrels.get(nct_id, 0) >= threshold
    )
    return recalled / total_relevant


# ──────────────────────────────────────────────────────────────────────────
# Classification metrics (3-class: eligible / excluded / not-relevant)
# ──────────────────────────────────────────────────────────────────────────
def classification_metrics(
    predicted: List[int],
    actual: List[int],
    labels: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute accuracy, macro-F1, and Cohen's kappa for 3-class classification.

    Parameters
    ----------
    predicted, actual : list[int]
        Predicted and ground-truth class labels (0, 1, 2).
    labels : list[int], optional
        The set of labels. Defaults to [0, 1, 2].

    Returns
    -------
    dict with keys: accuracy, macro_f1, cohens_kappa
    """
    if labels is None:
        labels = [0, 1, 2]

    n = len(predicted)
    if n == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "cohens_kappa": 0.0}

    # Accuracy
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    accuracy = correct / n

    # Per-class precision, recall, F1
    f1s = []
    for lbl in labels:
        tp = sum(1 for p, a in zip(predicted, actual) if p == lbl and a == lbl)
        fp = sum(1 for p, a in zip(predicted, actual) if p == lbl and a != lbl)
        fn = sum(1 for p, a in zip(predicted, actual) if p != lbl and a == lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    # Cohen's kappa
    # p_o = observed agreement, p_e = expected agreement by chance
    p_o = accuracy
    p_e = 0.0
    for lbl in labels:
        p_pred = sum(1 for p in predicted if p == lbl) / n
        p_actual = sum(1 for a in actual if a == lbl) / n
        p_e += p_pred * p_actual
    kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "cohens_kappa": kappa,
    }


# ──────────────────────────────────────────────────────────────────────────
# Convenience: rank trials from matching + aggregation results
# ──────────────────────────────────────────────────────────────────────────
def rank_trials(
    matching_results: Dict[str, Dict[str, Any]],
    aggregation_results: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, float]]:
    """Rank trials by combined matching + aggregation score.

    Returns list of (nct_id, total_score) sorted descending.
    """
    trial2score: Dict[str, float] = {}

    for nct_id, matching in matching_results.items():
        m_score = get_matching_score(matching)
        a_score = 0.0
        if nct_id in aggregation_results:
            a_score = get_agg_score(aggregation_results[nct_id])
        trial2score[nct_id] = get_trial_score(m_score, a_score)

    return sorted(trial2score.items(), key=lambda x: -x[1])
