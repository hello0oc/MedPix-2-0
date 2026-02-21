#!/usr/bin/env python3
"""Demonstration runner — executes the agentic trial-matching pipeline on:

  1. A TrialGPT patient (text-only, with ground truth)
  2. An NSCLC patient with pure EHR text (no images)
  3. The same NSCLC patient with EHR text + linked CT images

Outputs structured JSON results to trial_matching_agent/output/.

Usage::

    python trial_matching_agent/run_three_demos.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Setup path ──────────────────────────────────────────────────────────
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
    analyze_images,
    identify_missing_info,
    rank_and_build_results,
)
from trial_matching_agent.data_loader import (
    load_trialgpt_queries,
    load_trialgpt_qrels,
    load_trialgpt_retrieved,
    load_nsclc_cases,
    load_prebuilt_profiles,
    build_ehr_text,
    get_resolved_images,
    BM25Index,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")

OUTPUT_DIR = REPO_ROOT / "trial_matching_agent" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Helper: build TrialInfo from retrieved_trials entry
# ══════════════════════════════════════════════════════════════════════

def _trial_dict_to_info(trial_dict: Dict[str, Any]) -> TrialInfo:
    """Convert a TrialGPT retrieved-trial dict to our TrialInfo schema."""
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
# Helper: pretty-print results
# ══════════════════════════════════════════════════════════════════════

def print_banner(title: str) -> None:
    log.info("")
    log.info("=" * 72)
    log.info(f"  {title}")
    log.info("=" * 72)


def print_ranked_trials(ranked: List[RankedTrial], limit: int = 5) -> None:
    for i, rt in enumerate(ranked[:limit], 1):
        agg = rt.aggregation_result
        r_score = agg.relevance_score_R if agg else 0
        e_score = agg.eligibility_score_E if agg else 0
        log.info(
            f"  {i}. {rt.nct_id}  score={rt.total_score:.3f}  "
            f"R={r_score:.0f}  E={e_score:.0f}"
        )
        log.info(f"     {rt.title[:75]}")

        # Show matching summary
        mr = rt.matching_result
        if mr:
            n_inc = len(mr.inclusion)
            n_exc = len(mr.exclusion)
            inc_labels = [cm.label for cm in mr.inclusion.values()]
            n_met = sum(1 for l in inc_labels if l == "included")
            n_not = sum(1 for l in inc_labels if l == "not included")
            n_nei = sum(1 for l in inc_labels if l == "not enough information")
            exc_labels = [cm.label for cm in mr.exclusion.values()]
            n_excl = sum(1 for l in exc_labels if l == "excluded")
            log.info(
                f"     Inclusion ({n_inc} criteria): "
                f"{n_met} met, {n_not} not met, {n_nei} unknown"
            )
            log.info(
                f"     Exclusion ({n_exc} criteria): "
                f"{n_excl} excluded"
            )


def print_feedback(feedback: List[FeedbackRequest]) -> None:
    if not feedback:
        log.info("  No missing information identified.")
        return
    log.info(f"  {len(feedback)} items of missing information:")
    for fb in feedback:
        prio = fb.priority.upper() if isinstance(fb.priority, str) else fb.priority
        log.info(f"    [{prio}] {fb.question}")
        if fb.reason:
            log.info(f"      → {fb.reason[:100]}")


# ══════════════════════════════════════════════════════════════════════
# Helper: run one patient through the pipeline
# ══════════════════════════════════════════════════════════════════════

def run_pipeline_for_patient(
    patient_id: str,
    patient_text: str,
    candidate_trials: List[TrialInfo],
    config: AgentConfig,
    image_paths: Optional[List[Path]] = None,
    prebuilt_profile: Optional[PatientProfile] = None,
    label: str = "",
) -> AgentState:
    """Run the full matching pipeline for one patient against given trials."""
    state = AgentState(patient_id=patient_id, patient_text=patient_text)
    image_paths = image_paths or []
    step = 0

    # ── Step 1: Profile extraction ───────────────────────────────────
    step += 1
    log.info(f"  [{step}] Extracting patient profile …")
    t0 = time.time()
    if prebuilt_profile:
        state.patient_profile = prebuilt_profile
        log.info(f"      Using prebuilt profile ({len(prebuilt_profile.key_facts)} facts)")
    else:
        state.patient_profile = extract_patient_profile(
            ehr_text=patient_text,
            patient_id=patient_id,
            config=config,
        )
        log.info(f"      Extracted {len(state.patient_profile.key_facts)} key facts "
                 f"({time.time()-t0:.1f}s)")
    state.step_history.append(StepRecord(step=step, tool="extract_patient_profile",
        input_summary=f"text_len={len(patient_text)}",
        output_summary=f"facts={len(state.patient_profile.key_facts)}"))

    patient_display = state.patient_profile.profile_text or patient_text

    # ── Step 2: Image analysis (if images provided) ──────────────────
    if image_paths:
        step += 1
        log.info(f"  [{step}] Analyzing {len(image_paths)} medical images …")
        t0 = time.time()
        state.image_analysis = analyze_images(
            image_paths=image_paths,
            clinical_context=patient_display[:500],
            config=config,
        )
        findings = state.image_analysis.findings if state.image_analysis else []
        log.info(f"      {len(findings)} findings ({time.time()-t0:.1f}s)")
        if state.image_analysis and state.image_analysis.raw_text:
            log.info(f"      Preview: {state.image_analysis.raw_text[:150]}")
        state.step_history.append(StepRecord(step=step, tool="analyze_images",
            input_summary=f"n_images={len(image_paths)}",
            output_summary=f"findings={len(findings)}"))

    # ── Step 3: Keyword generation ───────────────────────────────────
    step += 1
    log.info(f"  [{step}] Generating search keywords …")
    t0 = time.time()
    kw_result = generate_search_keywords(
        patient_text=patient_display[:3000],
        config=config,
    )
    state.keyword_summary = kw_result.get("summary", "")
    state.search_conditions = kw_result.get("conditions", [])
    log.info(f"      Conditions: {state.search_conditions[:5]} ({time.time()-t0:.1f}s)")
    state.step_history.append(StepRecord(step=step, tool="generate_search_keywords",
        output_summary=f"conditions={state.search_conditions[:3]}"))

    # ── Step 4: Use pre-retrieved trial candidates ───────────────────
    step += 1
    state.retrieved_trials = candidate_trials[:config.max_candidate_trials]
    log.info(f"  [{step}] Using {len(state.retrieved_trials)} candidate trials")
    state.step_history.append(StepRecord(step=step, tool="search_trials",
        output_summary=f"n_trials={len(state.retrieved_trials)}"))

    if not state.retrieved_trials:
        log.warning("  No candidate trials — pipeline ends early.")
        return state

    # ── Step 5: Criterion-level matching ─────────────────────────────
    step += 1
    log.info(f"  [{step}] Matching criteria for {len(state.retrieved_trials)} trials …")
    for trial in state.retrieved_trials:
        t0 = time.time()
        log.info(f"      → {trial.nct_id}: {trial.brief_title[:55]} …")
        mr = match_criteria(
            patient_text=patient_display,
            trial_info=trial,
            config=config,
        )
        state.matching_results[trial.nct_id] = mr
        n_inc = len(mr.inclusion)
        n_exc = len(mr.exclusion)
        n_img = sum(1 for cm in list(mr.inclusion.values()) + list(mr.exclusion.values()) if cm.imaging_relevant)
        log.info(f"        {n_inc} inclusion + {n_exc} exclusion criteria, "
                 f"{n_img} imaging-relevant ({time.time()-t0:.1f}s)")
    state.step_history.append(StepRecord(step=step, tool="match_criteria",
        output_summary=f"matched {len(state.matching_results)} trials"))

    # ── Step 6: Aggregate & score ────────────────────────────────────
    step += 1
    log.info(f"  [{step}] Aggregating scores …")
    img_summary = ""
    if state.image_analysis:
        img_summary = state.image_analysis.raw_text[:500] if state.image_analysis.raw_text else ""
    for trial in state.retrieved_trials:
        mr = state.matching_results.get(trial.nct_id)
        if mr:
            t0 = time.time()
            agg = aggregate_and_score(
                patient_text=patient_display,
                trial_info=trial,
                matching_result=mr,
                config=config,
                image_analysis_summary=img_summary,
            )
            state.aggregation_results[trial.nct_id] = agg
            log.info(f"      {trial.nct_id}: R={agg.relevance_score_R:.0f} E={agg.eligibility_score_E:.0f} ({time.time()-t0:.1f}s)")
    state.step_history.append(StepRecord(step=step, tool="aggregate_and_score",
        output_summary=f"aggregated {len(state.aggregation_results)}"))

    # ── Step 7: Rank ─────────────────────────────────────────────────
    step += 1
    state.ranked_trials = rank_and_build_results(
        matching_results=state.matching_results,
        aggregation_results=state.aggregation_results,
        trial_infos={t.nct_id: t for t in state.retrieved_trials},
    )
    log.info(f"  [{step}] Ranked {len(state.ranked_trials)} trials")
    state.step_history.append(StepRecord(step=step, tool="rank_and_build_results",
        output_summary=f"top={state.ranked_trials[0].nct_id if state.ranked_trials else 'none'}"))

    # ── Step 8: Feedback / missing info ──────────────────────────────
    step += 1
    log.info(f"  [{step}] Identifying missing information …")
    t0 = time.time()
    state.feedback_requests = identify_missing_info(
        patient_summary=patient_display[:1500],
        matching_results=state.matching_results,
        trial_infos={t.nct_id: t for t in state.retrieved_trials},
        config=config,
    )
    log.info(f"      {len(state.feedback_requests)} feedback items ({time.time()-t0:.1f}s)")
    state.step_history.append(StepRecord(step=step, tool="identify_missing_info",
        output_summary=f"feedback={len(state.feedback_requests)}"))

    return state


# ══════════════════════════════════════════════════════════════════════
# DEMO 1: TrialGPT patient (text only, with ground truth)
# ══════════════════════════════════════════════════════════════════════

def demo_trialgpt_patient(config: AgentConfig) -> Dict[str, Any]:
    """Run pipeline on one TrialGPT patient and compare to ground truth."""
    print_banner("DEMO 1: TrialGPT Patient (Text-Only, Ground Truth)")

    # Load data
    queries = load_trialgpt_queries()
    qrels = load_trialgpt_qrels()
    retrieved_all = load_trialgpt_retrieved()

    # Pick a patient with a mix of eligible/ineligible trials
    patient_entry = None
    for entry in retrieved_all:
        pid = entry["patient_id"]
        n2 = len(entry.get("2", []))  # eligible
        n1 = len(entry.get("1", []))  # partial
        n0 = len(entry.get("0", []))  # not relevant
        if n2 >= 2 and n1 >= 2 and n0 >= 2:
            patient_entry = entry
            break
    if not patient_entry:
        patient_entry = retrieved_all[0]

    patient_id = patient_entry["patient_id"]
    patient_text = patient_entry["patient"]
    patient_qrels = qrels.get(patient_id, {})
    log.info(f"Patient: {patient_id}")
    log.info(f"Text preview: {patient_text[:200]}…")
    log.info(f"Ground truth: {len(patient_qrels)} trials annotated")

    # Build candidate trials (mix of labels)
    candidate_trials: List[TrialInfo] = []
    true_labels: Dict[str, int] = {}
    for label_str in ["2", "1", "0"]:
        label_int = int(label_str)
        for trial_dict in patient_entry.get(label_str, [])[:3]:
            ti = _trial_dict_to_info(trial_dict)
            if ti.nct_id:
                candidate_trials.append(ti)
                true_labels[ti.nct_id] = label_int

    log.info(f"Candidate trials: {len(candidate_trials)} "
             f"(2={sum(1 for v in true_labels.values() if v==2)}, "
             f"1={sum(1 for v in true_labels.values() if v==1)}, "
             f"0={sum(1 for v in true_labels.values() if v==0)})")

    # Run pipeline
    state = run_pipeline_for_patient(
        patient_id=patient_id,
        patient_text=patient_text,
        candidate_trials=candidate_trials,
        config=config,
        label="trialgpt",
    )

    # Show ranked results
    log.info("")
    log.info("── Ranked Results ──")
    print_ranked_trials(state.ranked_trials, limit=len(candidate_trials))

    # Evaluate vs ground truth
    log.info("")
    log.info("── Ground Truth Evaluation ──")
    pred_labels: List[int] = []
    gt_labels: List[int] = []
    pred_scores: Dict[str, float] = {}
    for rt in state.ranked_trials:
        gt = true_labels.get(rt.nct_id, 0)
        pred = _score_to_3class(rt.total_score, rt.aggregation_result)
        pred_labels.append(pred)
        gt_labels.append(gt)
        pred_scores[rt.nct_id] = rt.total_score
        match_mark = "✓" if pred == gt else "✗"
        log.info(f"  {rt.nct_id}: predicted={pred} true={gt} {match_mark}")

    ranked_ids = [rt.nct_id for rt in state.ranked_trials]
    ndcg5 = compute_ndcg(ranked_ids, true_labels, k=5)
    p5 = compute_precision_at_k(ranked_ids, true_labels, k=5, threshold=1)
    cls = classification_metrics(gt_labels, pred_labels)

    log.info(f"  NDCG@5:       {ndcg5:.4f}")
    log.info(f"  Precision@5:  {p5:.4f}")
    log.info(f"  Accuracy:     {cls['accuracy']:.4f}")
    log.info(f"  Macro F1:     {cls['macro_f1']:.4f}")
    log.info(f"  Kappa:        {cls['cohens_kappa']:.4f}")

    # Feedback
    log.info("")
    log.info("── Feedback (Missing Information) ──")
    print_feedback(state.feedback_requests)

    # Build result
    result = {
        "demo": "trialgpt_patient",
        "patient_id": patient_id,
        "n_candidate_trials": len(candidate_trials),
        "ranked_trials": [rt.to_dict() for rt in state.ranked_trials],
        "ground_truth": true_labels,
        "metrics": {
            "ndcg_at_5": round(ndcg5, 4),
            "precision_at_5": round(p5, 4),
            "accuracy": round(cls["accuracy"], 4),
            "macro_f1": round(cls["macro_f1"], 4),
            "cohens_kappa": round(cls["cohens_kappa"], 4),
        },
        "feedback_requests": [fb.to_dict() for fb in state.feedback_requests],
        "step_history": [s.to_dict() for s in state.step_history],
    }

    save_result(result, f"demo1_trialgpt_{patient_id}.json")
    return result


# ══════════════════════════════════════════════════════════════════════
# DEMO 2 & 3: NSCLC patient (EHR only / EHR + images)
# ══════════════════════════════════════════════════════════════════════

def _pick_nsclc_case() -> Tuple[Dict[str, Any], PatientProfile]:
    """Pick an NSCLC case with multiple images and a prebuilt profile."""
    cases = load_nsclc_cases()
    profiles = load_prebuilt_profiles()

    # Prefer a case with CT images and a rich profile
    best_case = None
    best_profile = None
    best_img_count = 0
    for c in cases:
        uid = c.get("uid", "")
        topic_id = uid.lower()
        prof = profiles.get(topic_id)
        imgs = get_resolved_images(c)
        if prof and len(imgs) > best_img_count:
            best_case = c
            best_profile = prof
            best_img_count = len(imgs)

    if not best_case:
        best_case = cases[0]
        uid = best_case.get("uid", "")
        best_profile = profiles.get(uid.lower())

    return best_case, best_profile


def _get_nsclc_candidate_trials(config: AgentConfig) -> List[TrialInfo]:
    """Build candidate NSCLC trials from the retrieved_trials corpus.

    Since trial_info.json is 1.1GB, we extract NSCLC-related trials
    from the SIGIR retrieved_trials.json and sigir_corpus.jsonl instead.
    """
    from trial_matching_agent.data_loader import load_trialgpt_corpus, _tokenize_simple
    from trial_matching_agent.config import _matches_nsclc

    nsclc_trials: Dict[str, TrialInfo] = {}

    # Source 1: retrieved_trials.json (has full criteria)
    retrieved = load_trialgpt_retrieved()
    for entry in retrieved:
        for lbl in ["0", "1", "2"]:
            for td in entry.get(lbl, []):
                text = " ".join([
                    td.get("brief_title", ""),
                    td.get("brief_summary", ""),
                    " ".join(td.get("diseases_list", td.get("diseases", []))),
                ])
                if _matches_nsclc(text):
                    ti = _trial_dict_to_info(td)
                    if ti.nct_id and ti.nct_id not in nsclc_trials:
                        nsclc_trials[ti.nct_id] = ti

    # Source 2: SIGIR corpus (has title + text, but not criteria)
    try:
        corpus = load_trialgpt_corpus()
        for entry in corpus:
            text = entry.get("title", "") + " " + entry.get("text", "")
            if _matches_nsclc(text):
                nct_id = entry.get("_id", "")
                if nct_id and nct_id not in nsclc_trials:
                    meta = entry.get("metadata", {})
                    nsclc_trials[nct_id] = TrialInfo(
                        nct_id=nct_id,
                        brief_title=entry.get("title", ""),
                        brief_summary=entry.get("text", ""),
                        diseases_list=meta.get("diseases_list", []) if isinstance(meta, dict) else [],
                    )
    except Exception:
        pass

    log.info(f"Found {len(nsclc_trials)} NSCLC trials from TrialGPT corpus")
    return list(nsclc_trials.values())


def demo_nsclc_ehr_only(
    config: AgentConfig,
    nsclc_case: Dict[str, Any],
    profile: Optional[PatientProfile],
    nsclc_trials: List[TrialInfo],
) -> Dict[str, Any]:
    """Run pipeline on NSCLC patient with pure EHR text (no images)."""
    print_banner("DEMO 2: NSCLC Patient — EHR Only (No Images)")

    uid = nsclc_case.get("uid", "unknown")
    ehr_text = build_ehr_text(nsclc_case, strip_findings=False)
    log.info(f"Patient: {uid}")
    log.info(f"Diagnosis: {nsclc_case.get('diagnosis', '')}")
    log.info(f"EHR text: {len(ehr_text)} chars")
    log.info(f"Available images: {len(get_resolved_images(nsclc_case))} (NOT used)")

    state = run_pipeline_for_patient(
        patient_id=uid,
        patient_text=ehr_text,
        candidate_trials=nsclc_trials,
        config=config,
        image_paths=[],  # No images
        prebuilt_profile=profile,
        label="nsclc_ehr_only",
    )

    log.info("")
    log.info("── Ranked Results (EHR Only) ──")
    print_ranked_trials(state.ranked_trials, limit=5)

    log.info("")
    log.info("── Feedback ──")
    print_feedback(state.feedback_requests)

    result = {
        "demo": "nsclc_ehr_only",
        "patient_id": uid,
        "diagnosis": nsclc_case.get("diagnosis", ""),
        "ehr_length": len(ehr_text),
        "images_used": 0,
        "ranked_trials": [rt.to_dict() for rt in state.ranked_trials],
        "feedback_requests": [fb.to_dict() for fb in state.feedback_requests],
        "step_history": [s.to_dict() for s in state.step_history],
    }
    save_result(result, f"demo2_nsclc_ehr_only_{uid}.json")
    return result


def demo_nsclc_ehr_plus_images(
    config: AgentConfig,
    nsclc_case: Dict[str, Any],
    profile: Optional[PatientProfile],
    nsclc_trials: List[TrialInfo],
) -> Dict[str, Any]:
    """Run pipeline on NSCLC patient with EHR text + linked CT images."""
    print_banner("DEMO 3: NSCLC Patient — EHR + CT Images")

    uid = nsclc_case.get("uid", "unknown")
    # Strip findings so the LLM must rely on image analysis
    ehr_text = build_ehr_text(nsclc_case, strip_findings=True)
    image_paths = get_resolved_images(nsclc_case)

    log.info(f"Patient: {uid}")
    log.info(f"Diagnosis: {nsclc_case.get('diagnosis', '')}")
    log.info(f"EHR text: {len(ehr_text)} chars (findings removed)")
    log.info(f"Images: {len(image_paths)} files")
    for ip in image_paths[:4]:
        log.info(f"  {ip.name}")

    state = run_pipeline_for_patient(
        patient_id=f"{uid}_with_images",
        patient_text=ehr_text,
        candidate_trials=nsclc_trials,
        config=config,
        image_paths=image_paths,
        prebuilt_profile=profile,
        label="nsclc_ehr_images",
    )

    log.info("")
    log.info("── Ranked Results (EHR + Images) ──")
    print_ranked_trials(state.ranked_trials, limit=5)

    log.info("")
    log.info("── Image Analysis Summary ──")
    if state.image_analysis:
        ia = state.image_analysis
        log.info(f"  Modality: {ia.modality}")
        log.info(f"  Body part: {ia.body_part}")
        log.info(f"  Findings: {ia.findings}")
        if ia.raw_text:
            log.info(f"  Raw: {ia.raw_text[:300]}")

    # Count imaging-relevant criteria
    n_img_criteria = 0
    for mr in state.matching_results.values():
        for cm in list(mr.inclusion.values()) + list(mr.exclusion.values()):
            if cm.imaging_relevant:
                n_img_criteria += 1
    log.info(f"  Imaging-relevant criteria flagged: {n_img_criteria}")

    log.info("")
    log.info("── Feedback ──")
    print_feedback(state.feedback_requests)

    result = {
        "demo": "nsclc_ehr_plus_images",
        "patient_id": uid,
        "diagnosis": nsclc_case.get("diagnosis", ""),
        "ehr_length": len(ehr_text),
        "images_used": len(image_paths),
        "image_files": [str(p.name) for p in image_paths],
        "image_analysis": state.image_analysis.to_dict() if state.image_analysis else None,
        "imaging_relevant_criteria": n_img_criteria,
        "ranked_trials": [rt.to_dict() for rt in state.ranked_trials],
        "feedback_requests": [fb.to_dict() for fb in state.feedback_requests],
        "step_history": [s.to_dict() for s in state.step_history],
    }
    save_result(result, f"demo3_nsclc_ehr_images_{uid}.json")
    return result


# ══════════════════════════════════════════════════════════════════════
# Comparison summary
# ══════════════════════════════════════════════════════════════════════

def print_comparison(
    result_ehr: Dict[str, Any],
    result_img: Dict[str, Any],
) -> None:
    """Print side-by-side comparison of EHR-only vs EHR+images."""
    print_banner("COMPARISON: EHR Only vs EHR + Images")

    ehr_trials = result_ehr.get("ranked_trials", [])
    img_trials = result_img.get("ranked_trials", [])

    log.info(f"  {'Metric':<35} {'EHR Only':>12} {'EHR+Images':>12}")
    log.info(f"  {'-'*35} {'-'*12} {'-'*12}")

    # Top trial comparison
    ehr_top = ehr_trials[0]["nct_id"] if ehr_trials else "—"
    img_top = img_trials[0]["nct_id"] if img_trials else "—"
    log.info(f"  {'Top trial':<35} {ehr_top:>12} {img_top:>12}")

    ehr_top_score = ehr_trials[0]["total_score"] if ehr_trials else 0
    img_top_score = img_trials[0]["total_score"] if img_trials else 0
    log.info(f"  {'Top score':<35} {ehr_top_score:>12.3f} {img_top_score:>12.3f}")

    # Average scores
    ehr_avg = sum(t["total_score"] for t in ehr_trials) / len(ehr_trials) if ehr_trials else 0
    img_avg = sum(t["total_score"] for t in img_trials) / len(img_trials) if img_trials else 0
    log.info(f"  {'Avg trial score':<35} {ehr_avg:>12.3f} {img_avg:>12.3f}")

    n_fb_ehr = len(result_ehr.get("feedback_requests", []))
    n_fb_img = len(result_img.get("feedback_requests", []))
    log.info(f"  {'Feedback items':<35} {n_fb_ehr:>12} {n_fb_img:>12}")

    n_img_crit = result_img.get("imaging_relevant_criteria", 0)
    log.info(f"  {'Imaging-relevant criteria':<35} {'—':>12} {n_img_crit:>12}")

    # Ranking agreement
    ehr_ranking = [t["nct_id"] for t in ehr_trials]
    img_ranking = [t["nct_id"] for t in img_trials]
    common = set(ehr_ranking) & set(img_ranking)
    if common:
        agreement = sum(1 for i, nid in enumerate(ehr_ranking) if i < len(img_ranking) and img_ranking[i] == nid)
        log.info(f"  {'Rank agreement (same position)':<35} {agreement:>12}/{min(len(ehr_ranking), len(img_ranking)):>3}")


# ══════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════

def save_result(result: Dict[str, Any], filename: str) -> None:
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    log.info(f"  → Saved to {path}")


def _score_to_3class(total_score: float, agg: Optional[AggregationResult]) -> int:
    """Map predicted score to 3-level relevance label."""
    if agg and total_score > 0.65:
        if agg.eligibility_score_E >= 50:
            return 2
    if total_score > 0.30:
        return 1
    return 0


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    config = AgentConfig.from_env()
    if not config.gemini_api_key:
        log.error("GEMINI_API_KEY not found. Set via env or .streamlit/secrets.toml.")
        sys.exit(1)

    # Reduce candidate count for demo (cost + time)
    config.max_candidate_trials = 5

    start_time = time.time()

    # ── Demo 1: TrialGPT patient ─────────────────────────────────────
    result1 = demo_trialgpt_patient(config)

    # ── Prepare NSCLC data (shared by demos 2 & 3) ───────────────────
    nsclc_case, profile = _pick_nsclc_case()
    nsclc_trials = _get_nsclc_candidate_trials(config)
    if not nsclc_trials:
        log.warning("No NSCLC trials found — skipping demos 2 & 3")
    else:
        # Limit to 5 most relevant trials
        nsclc_trials = nsclc_trials[:8]

        # ── Demo 2: NSCLC EHR only ──────────────────────────────────
        result2 = demo_nsclc_ehr_only(config, nsclc_case, profile, nsclc_trials)

        # ── Demo 3: NSCLC EHR + images ──────────────────────────────
        result3 = demo_nsclc_ehr_plus_images(config, nsclc_case, profile, nsclc_trials)

        # ── Comparison ───────────────────────────────────────────────
        print_comparison(result2, result3)

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print_banner("ALL DEMOS COMPLETE")
    log.info(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info(f"  Results saved to: {OUTPUT_DIR}/")

    # Save combined summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_time_sec": round(elapsed, 1),
        "demos_completed": 3 if nsclc_trials else 1,
        "demo1_patient": result1.get("patient_id"),
        "demo1_metrics": result1.get("metrics"),
    }
    if nsclc_trials:
        summary["demo2_patient"] = result2.get("patient_id")
        summary["demo2_top_trial"] = result2["ranked_trials"][0]["nct_id"] if result2.get("ranked_trials") else None
        summary["demo3_patient"] = result3.get("patient_id")
        summary["demo3_images_used"] = result3.get("images_used")
        summary["demo3_imaging_criteria"] = result3.get("imaging_relevant_criteria")
    save_result(summary, "demo_summary.json")


if __name__ == "__main__":
    main()
