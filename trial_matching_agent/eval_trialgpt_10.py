#!/usr/bin/env python3
"""Evaluate the agentic trial-matching pipeline on 10 TrialGPT patients.

For each patient we:
  1. Extract a structured profile  (Gemini 2.5 Pro)
  2. Generate search keywords       (Gemini 2.5 Pro)
  3. Score up to 5 candidate trials  (criterion matching + aggregation)
  4. Rank the trials
  5. Identify missing information (feedback loop)
  6. Compare predicted ranking & labels to TrialGPT ground truth

Aggregate metrics across all 10 patients are reported at the end.

Usage::

    python trial_matching_agent/eval_trialgpt_10.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

# ── Path setup ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trial_matching_agent.config import AgentConfig, DATA_DIR
from trial_matching_agent.schemas import (
    AgentState,
    AggregationResult,
    CriterionMatch,
    FeedbackRequest,
    MatchingResult,
    PatientProfile,
    RankedTrial,
    StepRecord,
    TrialInfo,
)
from trial_matching_agent.scoring import (
    get_matching_score,
    get_agg_score,
    get_trial_score,
    compute_ndcg,
    compute_precision_at_k,
    classification_metrics,
)
from trial_matching_agent.tools import (
    _call_gemini,
    _parse_json_from_text,
    extract_patient_profile,
    generate_search_keywords,
    match_criteria,
    aggregate_and_score,
    identify_missing_info,
    rank_and_build_results,
)
from trial_matching_agent.data_loader import (
    load_trialgpt_queries,
    load_trialgpt_qrels,
    load_trialgpt_retrieved,
)

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval10")

OUTPUT_DIR = REPO_ROOT / "trial_matching_agent" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRIALS_PER_PATIENT = 5   # keep cost reasonable
NUM_PATIENTS = 10


# ══════════════════════════════════════════════════════════════════════
# Patient selection — pick 10 diverse patients
# ══════════════════════════════════════════════════════════════════════

def select_patients(
    retrieved_all: List[Dict[str, Any]],
    n: int = NUM_PATIENTS,
) -> List[Dict[str, Any]]:
    """Select *n* patients with diverse label distributions.

    Strategy: score each patient by how balanced its label mix is and
    how many total trials it has.  Pick top-n by balance, ensuring we
    cover patients with 0-heavy, 2-heavy, and mixed distributions.
    """
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for entry in retrieved_all:
        n2 = len(entry.get("2", []))
        n1 = len(entry.get("1", []))
        n0 = len(entry.get("0", []))
        total = n2 + n1 + n0
        if total < 6:
            continue  # skip patients with too few trials
        # Favour patients with all three labels present and a reasonable
        # count so we can sample a balanced set of 5
        has_all = int(n2 >= 1 and n1 >= 1 and n0 >= 1)
        # Shannon entropy of label proportions (higher → more balanced)
        probs = [c / total for c in [n2, n1, n0] if c > 0]
        entropy = -sum(p * math.log2(p) for p in probs)
        score = has_all * 10 + entropy + min(total, 30) / 30
        scored.append((score, entry))

    scored.sort(key=lambda x: -x[0])
    # Take top-n but also ensure at least a couple of "hard" patients
    # with 0-dominant distributions
    selected = [entry for _, entry in scored[:n]]
    return selected


# ══════════════════════════════════════════════════════════════════════
# Build TrialInfo from retrieved-trial dict
# ══════════════════════════════════════════════════════════════════════

def _trial_dict_to_info(trial_dict: Dict[str, Any]) -> TrialInfo:
    diseases = trial_dict.get("diseases_list", trial_dict.get("diseases", []))
    if isinstance(diseases, str):
        diseases = [diseases]
    drugs = trial_dict.get("drugs_list", trial_dict.get("drugs", []))
    if isinstance(drugs, str):
        drugs = [drugs]
    return TrialInfo(
        nct_id=trial_dict.get("NCTID", ""),
        brief_title=trial_dict.get("brief_title", ""),
        phase=trial_dict.get("phase", ""),
        drugs_list=drugs,
        diseases_list=diseases,
        enrollment=trial_dict.get("enrollment", ""),
        inclusion_criteria=trial_dict.get("inclusion_criteria", ""),
        exclusion_criteria=trial_dict.get("exclusion_criteria", ""),
        brief_summary=trial_dict.get("brief_summary", ""),
    )


# ══════════════════════════════════════════════════════════════════════
# Build balanced candidate set for one patient
# ══════════════════════════════════════════════════════════════════════

def _build_candidate_set(
    entry: Dict[str, Any],
    max_trials: int = MAX_TRIALS_PER_PATIENT,
) -> Tuple[List[TrialInfo], Dict[str, int]]:
    """Pick a balanced mix of label-2, label-1, label-0 trials.

    Returns (trial_info_list, true_labels_dict).
    Strategy: sample ceil(max/3) from each label, then trim to max_trials.
    When a label bucket has fewer, redistribute to the others.
    """
    per_label = max(1, math.ceil(max_trials / 3))
    trials: List[TrialInfo] = []
    labels: Dict[str, int] = {}
    remaining = max_trials

    for label_str in ["2", "1", "0"]:
        label_int = int(label_str)
        bucket = entry.get(label_str, [])
        take = min(per_label, len(bucket), remaining)
        for td in bucket[:take]:
            ti = _trial_dict_to_info(td)
            if ti.nct_id and ti.nct_id not in labels:
                trials.append(ti)
                labels[ti.nct_id] = label_int
                remaining -= 1
        if remaining <= 0:
            break

    # If we have remaining slots, fill from largest bucket
    if remaining > 0:
        for label_str in ["2", "1", "0"]:
            label_int = int(label_str)
            bucket = entry.get(label_str, [])
            for td in bucket:
                ti = _trial_dict_to_info(td)
                if ti.nct_id and ti.nct_id not in labels and remaining > 0:
                    trials.append(ti)
                    labels[ti.nct_id] = label_int
                    remaining -= 1

    return trials, labels


# ══════════════════════════════════════════════════════════════════════
# Score-to-label conversion (same thresholds as agent.py)
# ══════════════════════════════════════════════════════════════════════

def _score_to_3class(total_score: float, agg: Optional[AggregationResult]) -> int:
    if agg and total_score > 0.65:
        if agg.eligibility_score_E >= 50:
            return 2
    if total_score > 0.30:
        return 1
    return 0


# ══════════════════════════════════════════════════════════════════════
# Run pipeline for one patient
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(
    patient_id: str,
    patient_text: str,
    candidate_trials: List[TrialInfo],
    config: AgentConfig,
) -> AgentState:
    """Full agentic pipeline for one patient (text-only)."""
    state = AgentState(patient_id=patient_id, patient_text=patient_text)

    # 1. Profile extraction
    log.info(f"    [1/6] Profile extraction …")
    t0 = time.time()
    state.patient_profile = extract_patient_profile(
        ehr_text=patient_text,
        patient_id=patient_id,
        config=config,
    )
    nf = len(state.patient_profile.key_facts)
    log.info(f"          {nf} key facts ({time.time()-t0:.1f}s)")

    display_text = state.patient_profile.profile_text or patient_text

    # 2. Keyword generation
    log.info(f"    [2/6] Keyword generation …")
    t0 = time.time()
    kw = generate_search_keywords(patient_text=display_text[:3000], config=config)
    state.search_conditions = kw.get("conditions", [])
    log.info(f"          conditions={state.search_conditions[:3]} ({time.time()-t0:.1f}s)")

    # 3. Use pre-retrieved candidates (substituting for BM25 search)
    state.retrieved_trials = candidate_trials[:config.max_candidate_trials]
    log.info(f"    [3/6] {len(state.retrieved_trials)} candidate trials")

    if not state.retrieved_trials:
        return state

    # 4. Criterion-level matching
    log.info(f"    [4/6] Criterion matching ({len(state.retrieved_trials)} trials) …")
    for trial in state.retrieved_trials:
        t0 = time.time()
        mr = match_criteria(
            patient_text=display_text,
            trial_info=trial,
            config=config,
        )
        state.matching_results[trial.nct_id] = mr
        elapsed = time.time() - t0
        n_inc = len(mr.inclusion)
        n_exc = len(mr.exclusion)
        log.info(f"          {trial.nct_id}: {n_inc}inc+{n_exc}exc ({elapsed:.1f}s)")

    # 5. Aggregation
    log.info(f"    [5/6] Aggregation …")
    for trial in state.retrieved_trials:
        mr = state.matching_results.get(trial.nct_id)
        if not mr:
            continue
        t0 = time.time()
        agg = aggregate_and_score(
            patient_text=display_text,
            trial_info=trial,
            matching_result=mr,
            config=config,
        )
        state.aggregation_results[trial.nct_id] = agg
        log.info(f"          {trial.nct_id}: R={agg.relevance_score_R:.0f} "
                 f"E={agg.eligibility_score_E:.0f} ({time.time()-t0:.1f}s)")

    # 6. Rank
    state.ranked_trials = rank_and_build_results(
        matching_results=state.matching_results,
        aggregation_results=state.aggregation_results,
        trial_infos={t.nct_id: t for t in state.retrieved_trials},
    )
    log.info(f"    [6/6] Ranked → top={state.ranked_trials[0].nct_id if state.ranked_trials else '—'}")

    # 7. Feedback (optional — useful for the report but not scored)
    t0 = time.time()
    state.feedback_requests = identify_missing_info(
        patient_summary=display_text[:1500],
        matching_results=state.matching_results,
        trial_infos={t.nct_id: t for t in state.retrieved_trials},
        config=config,
    )
    log.info(f"          feedback: {len(state.feedback_requests)} items ({time.time()-t0:.1f}s)")

    return state


# ══════════════════════════════════════════════════════════════════════
# Evaluate one patient
# ══════════════════════════════════════════════════════════════════════

def evaluate_patient(
    patient_idx: int,
    entry: Dict[str, Any],
    config: AgentConfig,
) -> Dict[str, Any]:
    """Run pipeline + compute per-patient metrics. Returns result dict."""
    patient_id = entry["patient_id"]
    patient_text = entry["patient"]
    trials, true_labels = _build_candidate_set(entry, MAX_TRIALS_PER_PATIENT)

    label_dist = Counter(true_labels.values())
    log.info(f"  Patient {patient_idx+1}/{NUM_PATIENTS}: {patient_id}")
    log.info(f"    Candidates: {len(trials)} "
             f"(2={label_dist.get(2,0)}, 1={label_dist.get(1,0)}, 0={label_dist.get(0,0)})")
    log.info(f"    Text: {patient_text[:120]}…")

    t_start = time.time()
    state = run_pipeline(
        patient_id=patient_id,
        patient_text=patient_text,
        candidate_trials=trials,
        config=config,
    )
    elapsed = time.time() - t_start

    # ── Compute per-patient metrics ──────────────────────────────────
    pred_labels: List[int] = []
    gt_labels: List[int] = []
    per_trial: List[Dict[str, Any]] = []

    for rt in state.ranked_trials:
        gt = true_labels.get(rt.nct_id, 0)
        pred = _score_to_3class(rt.total_score, rt.aggregation_result)
        pred_labels.append(pred)
        gt_labels.append(gt)
        agg = rt.aggregation_result
        per_trial.append({
            "nct_id": rt.nct_id,
            "title": rt.title[:80],
            "total_score": round(rt.total_score, 3),
            "R": round(agg.relevance_score_R, 1) if agg else None,
            "E": round(agg.eligibility_score_E, 1) if agg else None,
            "predicted": pred,
            "true": gt,
            "correct": pred == gt,
        })

    ranked_ids = [rt.nct_id for rt in state.ranked_trials]
    ndcg5 = compute_ndcg(ranked_ids, true_labels, k=5)
    p5 = compute_precision_at_k(ranked_ids, true_labels, k=5, threshold=1)
    cls = classification_metrics(gt_labels, pred_labels) if gt_labels else {
        "accuracy": 0, "macro_f1": 0, "cohens_kappa": 0}

    log.info(f"    ── Results ──")
    for t in per_trial:
        mark = "✓" if t["correct"] else "✗"
        log.info(f"      {t['nct_id']}: pred={t['predicted']} true={t['true']} "
                 f"score={t['total_score']:.3f}  R={t['R']}  E={t['E']}  {mark}")
    log.info(f"    NDCG@5={ndcg5:.4f}  P@5={p5:.4f}  "
             f"Acc={cls['accuracy']:.3f}  F1={cls['macro_f1']:.3f}  "
             f"κ={cls['cohens_kappa']:.3f}  time={elapsed:.0f}s")

    return {
        "patient_id": patient_id,
        "n_candidates": len(trials),
        "label_distribution": {str(k): v for k, v in label_dist.items()},
        "trials": per_trial,
        "metrics": {
            "ndcg_at_5": round(ndcg5, 4),
            "precision_at_5": round(p5, 4),
            "accuracy": round(cls["accuracy"], 4),
            "macro_f1": round(cls["macro_f1"], 4),
            "cohens_kappa": round(cls["cohens_kappa"], 4),
        },
        "n_feedback": len(state.feedback_requests),
        "time_sec": round(elapsed, 1),
    }


# ══════════════════════════════════════════════════════════════════════
# Aggregate metrics across all patients
# ══════════════════════════════════════════════════════════════════════

def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute macro-averaged and micro-averaged evaluation metrics."""

    # ── Macro averages (mean of per-patient metrics) ─────────────────
    metric_keys = ["ndcg_at_5", "precision_at_5", "accuracy", "macro_f1", "cohens_kappa"]
    macro = {}
    for k in metric_keys:
        vals = [r["metrics"][k] for r in results if r["metrics"].get(k) is not None]
        macro[k] = round(sum(vals) / len(vals), 4) if vals else 0.0

    # ── Micro averages (pool all trial-level predictions) ────────────
    all_pred: List[int] = []
    all_true: List[int] = []
    all_ranked_ids: List[List[str]] = []
    all_true_labels: List[Dict[str, int]] = []

    for r in results:
        preds = [t["predicted"] for t in r["trials"]]
        trues = [t["true"] for t in r["trials"]]
        all_pred.extend(preds)
        all_true.extend(trues)
        ranked_ids = [t["nct_id"] for t in r["trials"]]
        true_map = {t["nct_id"]: t["true"] for t in r["trials"]}
        all_ranked_ids.append(ranked_ids)
        all_true_labels.append(true_map)

    micro_cls = classification_metrics(all_true, all_pred) if all_true else {
        "accuracy": 0, "macro_f1": 0, "cohens_kappa": 0}

    # ── Per-class metrics ────────────────────────────────────────────
    class_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for p, t in zip(all_pred, all_true):
        if p == t:
            class_counts[t]["tp"] += 1
        else:
            class_counts[p]["fp"] += 1
            class_counts[t]["fn"] += 1

    per_class = {}
    for cls_label in [0, 1, 2]:
        c = class_counts[cls_label]
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0
        rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_class[str(cls_label)] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in all_true if t == cls_label),
        }

    # ── Confusion matrix ─────────────────────────────────────────────
    conf = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for p, t in zip(all_pred, all_true):
        conf[t][p] += 1

    # ── Ranking: mean NDCG@5 and P@5 across patients ────────────────
    ndcg_values = [r["metrics"]["ndcg_at_5"] for r in results]
    p5_values = [r["metrics"]["precision_at_5"] for r in results]

    return {
        "n_patients": len(results),
        "n_total_trials": len(all_true),
        "macro_avg": macro,
        "micro": {
            "accuracy": round(micro_cls["accuracy"], 4),
            "macro_f1": round(micro_cls["macro_f1"], 4),
            "cohens_kappa": round(micro_cls["cohens_kappa"], 4),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "rows_are_true_cols_are_predicted": conf,
            "labels": [0, 1, 2],
        },
        "ranking": {
            "mean_ndcg_at_5": round(sum(ndcg_values) / len(ndcg_values), 4) if ndcg_values else 0,
            "mean_precision_at_5": round(sum(p5_values) / len(p5_values), 4) if p5_values else 0,
            "ndcg_at_5_per_patient": [round(v, 4) for v in ndcg_values],
            "precision_at_5_per_patient": [round(v, 4) for v in p5_values],
        },
        "total_time_sec": round(sum(r["time_sec"] for r in results), 1),
    }


# ══════════════════════════════════════════════════════════════════════
# Pretty-print final report
# ══════════════════════════════════════════════════════════════════════

def print_report(agg: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    log.info("")
    log.info("=" * 72)
    log.info("  EVALUATION REPORT — 10 TrialGPT Patients")
    log.info("=" * 72)

    log.info("")
    log.info("  ── Per-Patient Summary ──")
    log.info(f"  {'Patient':<18} {'Trials':>6} {'NDCG@5':>8} {'P@5':>6} "
             f"{'Acc':>6} {'F1':>6} {'κ':>6} {'Time':>6}")
    log.info(f"  {'─'*18} {'─'*6} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for r in results:
        m = r["metrics"]
        log.info(
            f"  {r['patient_id']:<18} {r['n_candidates']:>6} "
            f"{m['ndcg_at_5']:>8.4f} {m['precision_at_5']:>6.2f} "
            f"{m['accuracy']:>6.3f} {m['macro_f1']:>6.3f} "
            f"{m['cohens_kappa']:>6.3f} {r['time_sec']:>5.0f}s"
        )

    log.info("")
    log.info("  ── Aggregate Metrics ──")
    ma = agg["macro_avg"]
    mi = agg["micro"]
    log.info(f"  Macro-avg NDCG@5:       {agg['ranking']['mean_ndcg_at_5']:.4f}")
    log.info(f"  Macro-avg Precision@5:  {agg['ranking']['mean_precision_at_5']:.4f}")
    log.info(f"  Macro-avg Accuracy:     {ma['accuracy']:.4f}")
    log.info(f"  Macro-avg F1:           {ma['macro_f1']:.4f}")
    log.info(f"  Macro-avg κ:            {ma['cohens_kappa']:.4f}")
    log.info(f"  Micro Accuracy:         {mi['accuracy']:.4f}")
    log.info(f"  Micro Macro-F1:         {mi['macro_f1']:.4f}")
    log.info(f"  Micro κ:                {mi['cohens_kappa']:.4f}")

    log.info("")
    log.info("  ── Per-Class Metrics ──")
    log.info(f"  {'Label':>6} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    for cls_label in ["0", "1", "2"]:
        c = agg["per_class"][cls_label]
        label_name = {
            "0": "excl",
            "1": "partial",
            "2": "elig",
        }[cls_label]
        log.info(
            f"  {cls_label:>3}({label_name:<7}) {c['precision']:>10.4f} "
            f"{c['recall']:>8.4f} {c['f1']:>8.4f} {c['support']:>8}"
        )

    log.info("")
    log.info("  ── Confusion Matrix (rows=true, cols=predicted) ──")
    log.info(f"  {'':>12} pred=0  pred=1  pred=2")
    for i, label in enumerate(["true=0", "true=1", "true=2"]):
        row = agg["confusion_matrix"]["rows_are_true_cols_are_predicted"][i]
        log.info(f"  {label:>12}  {row[0]:>5}   {row[1]:>5}   {row[2]:>5}")

    log.info("")
    total_min = agg["total_time_sec"] / 60
    log.info(f"  Total evaluation time: {agg['total_time_sec']:.0f}s ({total_min:.1f}min)")
    log.info(f"  Total trials evaluated: {agg['n_total_trials']}")
    log.info("=" * 72)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    config = AgentConfig.from_env()
    if not config.gemini_api_key:
        log.error("GEMINI_API_KEY not found.")
        sys.exit(1)
    config.max_candidate_trials = MAX_TRIALS_PER_PATIENT

    log.info("=" * 72)
    log.info("  TrialGPT 10-Patient Evaluation")
    log.info(f"  Model: {config.gemini_model}")
    log.info(f"  Candidates per patient: {MAX_TRIALS_PER_PATIENT}")
    log.info(f"  Timestamp: {datetime.now(timezone.utc).isoformat()}")
    log.info("=" * 72)

    # ── Load data ────────────────────────────────────────────────────
    log.info("Loading TrialGPT data …")
    retrieved_all = load_trialgpt_retrieved()
    log.info(f"  {len(retrieved_all)} patients available")

    # ── Select 10 patients ───────────────────────────────────────────
    patients = select_patients(retrieved_all, NUM_PATIENTS)
    log.info(f"  Selected {len(patients)} patients:")
    for i, p in enumerate(patients):
        pid = p["patient_id"]
        n2 = len(p.get("2", []))
        n1 = len(p.get("1", []))
        n0 = len(p.get("0", []))
        log.info(f"    {i+1}. {pid} (elig={n2} partial={n1} excl={n0})")

    # ── Run evaluation ───────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    start_time = time.time()

    for i, entry in enumerate(patients):
        log.info("")
        log.info(f"{'─'*72}")
        result = evaluate_patient(i, entry, config)
        results.append(result)

        # Save incremental results after each patient (crash-safe)
        _save_incremental(results)

    total_elapsed = time.time() - start_time

    # ── Aggregate ────────────────────────────────────────────────────
    agg = aggregate_metrics(results)

    # ── Report ───────────────────────────────────────────────────────
    print_report(agg, results)

    # ── Save final output ────────────────────────────────────────────
    final_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": config.gemini_model,
        "candidates_per_patient": MAX_TRIALS_PER_PATIENT,
        "total_time_sec": round(total_elapsed, 1),
        "aggregate": agg,
        "per_patient": results,
    }
    out_path = OUTPUT_DIR / "eval_trialgpt_10_results.json"
    out_path.write_text(json.dumps(final_output, indent=2, default=str), encoding="utf-8")
    log.info(f"\n  Results saved to {out_path}")


def _save_incremental(results: List[Dict[str, Any]]) -> None:
    """Save partial results after each patient for crash recovery."""
    path = OUTPUT_DIR / "eval_trialgpt_10_partial.json"
    path.write_text(json.dumps({
        "completed": len(results),
        "per_patient": results,
    }, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
