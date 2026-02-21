"""TrialMatchingAgent — deterministic pipeline orchestrator.

Implements the adapted TrialGPT 3-stage pipeline:
  1. Retrieval  — keyword generation → BM25 search
  2. Matching   — criterion-level evaluation (+ image-enhanced)
  3. Ranking    — aggregation & scoring → feedback loop

The agent can be run end-to-end via ``run()``, or resumed after
feedback via ``resume()``.

Design notes
────────────
• Deterministic pipeline (not open-ended ReAct): each tool is called in
  a fixed order.  Gemini function-calling is used *within* each tool, not
  as the outer scheduling loop, because the 3-stage structure is proven
  by TrialGPT and doesn't benefit from free-form planning.
• Full reasoning trace is captured in ``AgentState`` for evaluation.
• The feedback loop adds a programmatic API: after ``run()`` returns,
  the caller inspects ``state.feedback_requests`` and supplies answers
  via ``resume(state, answers)`` for iterative refinement.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import AgentConfig, OUTPUT_DIR
from .data_loader import (
    BM25Index,
    build_ehr_text,
    get_resolved_images,
    load_nsclc_cases,
    load_prebuilt_profiles,
    load_trial_info,
    load_trialgpt_queries,
    load_trialgpt_qrels,
    load_trialgpt_retrieved,
    load_trialgpt_corpus,
)
from .schemas import (
    AgentState,
    AggregationResult,
    CriterionMatch,
    EvaluationReport,
    FeedbackRequest,
    ImageAnalysis,
    MatchingResult,
    PatientProfile,
    RankedTrial,
    StepRecord,
    TrialInfo,
)
from .scoring import (
    classification_metrics,
    compute_ndcg,
    compute_precision_at_k,
    compute_recall_at_k,
)
from .tools import (
    aggregate_and_score,
    analyze_images,
    extract_patient_profile,
    generate_search_keywords,
    identify_missing_info,
    match_criteria,
    rank_and_build_results,
    search_trials,
    validate_imaging_criteria,
)

logger = logging.getLogger(__name__)


class TrialMatchingAgent:
    """Orchestrates the full trial-matching pipeline.

    Usage::

        config = AgentConfig.from_env()
        agent  = TrialMatchingAgent(config)

        # End-to-end run
        state = agent.run(patient_text="…", patient_id="P001")

        # Inspect feedback
        for fb in state.feedback_requests:
            print(fb.question)

        # Resume with answers
        answers = {"ECOG performance status": "ECOG 1"}
        state = agent.resume(state, answers)
    """

    def __init__(
        self,
        config: AgentConfig,
        trial_index: Optional[BM25Index] = None,
        trial_info_map: Optional[Dict[str, TrialInfo]] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
    ):
        self.config = config.resolve_keys()
        self._trial_index = trial_index
        self._trial_info = trial_info_map or {}
        self._progress_cb = progress_cb or (lambda msg: logger.info(msg))

    # ──────────────────────────────────────────────────────────────────
    # Lazy BM25 index init
    # ──────────────────────────────────────────────────────────────────
    @property
    def trial_index(self) -> BM25Index:
        if self._trial_index is None:
            self._progress_cb("Building BM25 index …")
            self._trial_index = BM25Index(self.config.data_dir)
        return self._trial_index

    @property
    def trial_info(self) -> Dict[str, TrialInfo]:
        if not self._trial_info:
            self._progress_cb("Loading trial info …")
            self._trial_info = load_trial_info(self.config.data_dir)
        return self._trial_info

    # ──────────────────────────────────────────────────────────────────
    # Main entry points
    # ──────────────────────────────────────────────────────────────────
    def run(
        self,
        patient_text: str,
        patient_id: str = "patient",
        image_paths: Optional[List[Path]] = None,
        prebuilt_profile: Optional[PatientProfile] = None,
    ) -> AgentState:
        """Run the full pipeline for one patient.

        Parameters
        ----------
        patient_text : str
            Raw EHR / clinical summary text.
        patient_id : str
            Unique identifier for tracking.
        image_paths : list[Path], optional
            Paths to medical images (CT, MRI, X-ray, DICOM).
        prebuilt_profile : PatientProfile, optional
            Skip profile extraction if already available.

        Returns
        -------
        AgentState
            Full reasoning trace with ranked trials and feedback.
        """
        state = AgentState(
            patient_id=patient_id,
            patient_text=patient_text,
        )
        image_paths = image_paths or []
        step = 0

        # ── Step 1: Extract patient profile ──────────────────────────
        step += 1
        self._progress_cb(f"[Step {step}] Extracting patient profile …")
        if prebuilt_profile:
            state.patient_profile = prebuilt_profile
        else:
            state.patient_profile = extract_patient_profile(
                ehr_text=patient_text,
                patient_id=patient_id,
                config=self.config,
            )
        state.step_history.append(StepRecord(
            step=step,
            tool="extract_patient_profile",
            input_summary=f"patient_id={patient_id}, text_len={len(patient_text)}",
            output_summary=f"profile facts: {len(state.patient_profile.key_facts)}",
        ))

        # ── Step 2: Analyze images (if any) ──────────────────────────
        if image_paths:
            step += 1
            self._progress_cb(f"[Step {step}] Analyzing {len(image_paths)} images …")
            state.image_analysis = analyze_images(
                image_paths=image_paths,
                clinical_context=state.patient_profile.profile_text[:500],
                config=self.config,
            )
            state.step_history.append(StepRecord(
                step=step,
                tool="analyze_images",
                input_summary=f"n_images={len(image_paths)}",
                output_summary=f"findings: {len(state.image_analysis.findings)}",
            ))

        # ── Step 3: Generate search keywords ─────────────────────────
        step += 1
        self._progress_cb(f"[Step {step}] Generating search keywords …")
        kw_result = generate_search_keywords(
            patient_text=state.patient_profile.profile_text or patient_text[:self.config.ehr_char_limit],
            config=self.config,
        )
        state.keyword_summary = kw_result.get("summary", "")
        state.search_conditions = kw_result.get("conditions", [])
        state.step_history.append(StepRecord(
            step=step,
            tool="generate_search_keywords",
            input_summary=f"profile_len={len(state.patient_profile.profile_text or '')}",
            output_summary=f"conditions: {state.search_conditions}",
        ))

        # ── Step 4: Search trials (BM25) ─────────────────────────────
        step += 1
        self._progress_cb(f"[Step {step}] Searching trials (BM25) …")
        candidates = search_trials(
            conditions=state.search_conditions,
            bm25_index=self.trial_index,
            top_n=self.config.bm25_top_n,
        )
        # Limit to max_candidate_trials for downstream processing
        state.retrieved_trials = candidates[: self.config.max_candidate_trials]
        state.step_history.append(StepRecord(
            step=step,
            tool="search_trials",
            input_summary=f"conditions={state.search_conditions}, top_n={self.config.bm25_top_n}",
            output_summary=f"retrieved {len(candidates)}, kept {len(state.retrieved_trials)}",
        ))

        if not state.retrieved_trials:
            self._progress_cb("No trials retrieved — pipeline ends early.")
            return state

        # ── Step 5: Criterion-level matching ─────────────────────────
        step += 1
        patient_display = state.patient_profile.profile_text or patient_text
        self._progress_cb(
            f"[Step {step}] Matching criteria for {len(state.retrieved_trials)} trials …"
        )
        for trial in state.retrieved_trials:
            self._progress_cb(f"  → {trial.nct_id}: {trial.brief_title[:60]}")
            mr = match_criteria(
                patient_text=patient_display,
                trial_info=trial,
                config=self.config,
            )
            state.matching_results[trial.nct_id] = mr
        state.step_history.append(StepRecord(
            step=step,
            tool="match_criteria",
            input_summary=f"n_trials={len(state.retrieved_trials)}",
            output_summary=f"matched {len(state.matching_results)} trials",
        ))

        # ── Step 5b: Validate imaging criteria (if images exist) ─────
        if image_paths and any(
            cm.imaging_relevant
            for mr in state.matching_results.values()
            for cm in list(mr.inclusion.values()) + list(mr.exclusion.values())
        ):
            step += 1
            self._progress_cb(f"[Step {step}] Validating imaging criteria …")
            imaging_criteria = [
                cm
                for mr in state.matching_results.values()
                for cm in list(mr.inclusion.values()) + list(mr.exclusion.values())
                if cm.imaging_relevant
            ]
            updated = validate_imaging_criteria(
                imaging_criteria=imaging_criteria,
                image_paths=image_paths,
                config=self.config,
            )
            # Apply validated labels back
            _apply_imaging_updates(state.matching_results, updated)
            state.step_history.append(StepRecord(
                step=step,
                tool="validate_imaging_criteria",
                input_summary=f"n_imaging_criteria={len(imaging_criteria)}",
                output_summary=f"updated {len(updated)} criteria",
            ))

        # ── Step 6: Aggregate and score each trial ───────────────────
        step += 1
        self._progress_cb(f"[Step {step}] Aggregating scores …")
        img_summary = ""
        if state.image_analysis:
            img_summary = state.image_analysis.raw_text or ""

        for trial in state.retrieved_trials:
            mr = state.matching_results.get(trial.nct_id)
            if mr:
                agg = aggregate_and_score(
                    patient_text=patient_display,
                    trial_info=trial,
                    matching_result=mr,
                    config=self.config,
                    image_analysis_summary=img_summary[:500],
                )
                state.aggregation_results[trial.nct_id] = agg

        state.step_history.append(StepRecord(
            step=step,
            tool="aggregate_and_score",
            input_summary=f"n_trials={len(state.matching_results)}",
            output_summary=f"aggregated {len(state.aggregation_results)} trials",
        ))

        # ── Step 7: Rank trials ──────────────────────────────────────
        step += 1
        self._progress_cb(f"[Step {step}] Ranking trials …")
        state.ranked_trials = rank_and_build_results(
            matching_results=state.matching_results,
            aggregation_results=state.aggregation_results,
            trial_infos={t.nct_id: t for t in state.retrieved_trials},
        )
        state.step_history.append(StepRecord(
            step=step,
            tool="rank_and_build_results",
            input_summary=f"n_trials={len(state.matching_results)}",
            output_summary=f"top: {state.ranked_trials[0].nct_id if state.ranked_trials else 'none'}",
        ))

        # ── Step 8: Identify missing info (feedback loop) ────────────
        step += 1
        self._progress_cb(f"[Step {step}] Identifying missing information …")
        state.feedback_requests = identify_missing_info(
            patient_summary=patient_display[:1500],
            matching_results=state.matching_results,
            trial_infos={t.nct_id: t for t in state.retrieved_trials},
            config=self.config,
        )
        state.step_history.append(StepRecord(
            step=step,
            tool="identify_missing_info",
            input_summary=f"n_matching_results={len(state.matching_results)}",
            output_summary=f"feedback items: {len(state.feedback_requests)}",
        ))

        self._progress_cb(
            f"Pipeline complete. {len(state.ranked_trials)} trials ranked, "
            f"{len(state.feedback_requests)} feedback items."
        )
        return state

    def resume(
        self,
        state: AgentState,
        additional_data: Dict[str, str],
    ) -> AgentState:
        """Resume pipeline after user provides feedback answers.

        Parameters
        ----------
        state : AgentState
            Previous run state.
        additional_data : dict[str, str]
            Answers keyed by feedback field name, e.g.
            ``{"ECOG performance status": "ECOG 1"}``.

        Returns
        -------
        AgentState
            Updated state with re-evaluated matching and ranking.
        """
        state.iteration += 1
        self._progress_cb(
            f"Resuming iteration {state.iteration} with {len(additional_data)} answers …"
        )

        # Augment patient text with new info
        supplement = "\n\nAdditional patient information (provided on request):\n"
        for field_name, answer in additional_data.items():
            supplement += f"- {field_name}: {answer}\n"
        state.patient_text += supplement

        # Re-extract profile with augmented text
        state.patient_profile = extract_patient_profile(
            ehr_text=state.patient_text,
            patient_id=state.patient_id,
            config=self.config,
        )

        patient_display = state.patient_profile.profile_text or state.patient_text

        # Re-match only trials that had "not enough information" criteria
        trials_to_rematch: List[TrialInfo] = []
        for trial in state.retrieved_trials:
            mr = state.matching_results.get(trial.nct_id)
            if mr and _has_insufficient_info(mr):
                trials_to_rematch.append(trial)

        if not trials_to_rematch:
            self._progress_cb("No trials need re-matching.")
            return state

        self._progress_cb(f"Re-matching {len(trials_to_rematch)} trials …")
        for trial in trials_to_rematch:
            mr = match_criteria(
                patient_text=patient_display,
                trial_info=trial,
                config=self.config,
            )
            state.matching_results[trial.nct_id] = mr

        # Re-aggregate and re-rank
        self._progress_cb("Re-aggregating and re-ranking …")
        for trial in trials_to_rematch:
            mr = state.matching_results.get(trial.nct_id)
            if mr:
                agg = aggregate_and_score(
                    patient_text=patient_display,
                    trial_info=trial,
                    matching_result=mr,
                    config=self.config,
                )
                state.aggregation_results[trial.nct_id] = agg

        state.ranked_trials = rank_and_build_results(
            matching_results=state.matching_results,
            aggregation_results=state.aggregation_results,
            trial_infos={t.nct_id: t for t in state.retrieved_trials},
        )

        # Generate new feedback (may have fewer items now)
        state.feedback_requests = identify_missing_info(
            patient_summary=patient_display[:1500],
            matching_results=state.matching_results,
            trial_infos={t.nct_id: t for t in state.retrieved_trials},
            config=self.config,
        )

        self._progress_cb(
            f"Resume complete. {len(state.ranked_trials)} trials ranked, "
            f"{len(state.feedback_requests)} remaining feedback items."
        )
        return state

    # ──────────────────────────────────────────────────────────────────
    # TrialGPT evaluation (Track A)
    # ──────────────────────────────────────────────────────────────────
    def evaluate_trialgpt(
        self,
        patient_ids: Optional[List[str]] = None,
        max_trials_per_patient: int = 10,
    ) -> EvaluationReport:
        """Evaluate against TrialGPT ground truth (SIGIR corpus).

        Runs the pipeline for each TrialGPT patient and compares
        criterion-level labels + ranking against qrels.

        Parameters
        ----------
        patient_ids : list[str], optional
            Subset of patient IDs to evaluate. If None, evaluate all.
        max_trials_per_patient : int
            Cap candidate trials per patient for cost control.

        Returns
        -------
        EvaluationReport
            IR metrics + classification metrics.
        """
        queries = load_trialgpt_queries(self.config.data_dir)
        qrels = load_trialgpt_qrels(self.config.data_dir)
        retrieved = load_trialgpt_retrieved(self.config.data_dir)

        if patient_ids:
            queries = {k: v for k, v in queries.items() if k in patient_ids}

        all_pred_labels: List[int] = []
        all_true_labels: List[int] = []
        all_ndcg5: List[float] = []
        all_ndcg10: List[float] = []
        all_p5: List[float] = []
        total_trials = 0

        for pid, patient_text in queries.items():
            self._progress_cb(f"[Eval] Patient {pid} …")
            patient_qrels = qrels.get(pid, {})
            patient_retrieved = retrieved.get(pid, {})

            # Use pre-retrieved trial IDs (from TrialGPT dataset)
            trial_ids = set()
            for label_group in patient_retrieved.values():
                for trial_dict in label_group:
                    nct = trial_dict.get("NCTID", "")
                    if nct:
                        trial_ids.add(nct)

            # Limit for cost
            trial_ids_list = list(trial_ids)[:max_trials_per_patient]

            # Run matching for each pre-retrieved trial
            matching_results: Dict[str, MatchingResult] = {}
            aggregation_results: Dict[str, AggregationResult] = {}
            trial_info_map: Dict[str, TrialInfo] = {}

            for nct_id in trial_ids_list:
                if nct_id not in self.trial_info:
                    continue
                ti = self.trial_info[nct_id]
                trial_info_map[nct_id] = ti

                mr = match_criteria(
                    patient_text=patient_text,
                    trial_info=ti,
                    config=self.config,
                )
                matching_results[nct_id] = mr

                agg = aggregate_and_score(
                    patient_text=patient_text,
                    trial_info=ti,
                    matching_result=mr,
                    config=self.config,
                )
                aggregation_results[nct_id] = agg
                total_trials += 1

            # Rank
            ranked = rank_and_build_results(
                matching_results=matching_results,
                aggregation_results=aggregation_results,
                trial_infos=trial_info_map,
            )

            # Collect scores vs ground truth
            pred_scores: Dict[str, float] = {}
            true_scores: Dict[str, int] = {}
            for rt in ranked:
                pred_scores[rt.nct_id] = rt.total_score
                true_label = patient_qrels.get(rt.nct_id, 0)
                true_scores[rt.nct_id] = true_label

                # Convert predicted score to 3-class label for classification
                pred_label = _score_to_3class(rt.total_score, rt.aggregation_result)
                all_pred_labels.append(pred_label)
                all_true_labels.append(true_label)

            # Compute NDCG — scoring functions expect a sorted list
            if pred_scores:
                ranked_ids = [nid for nid, _ in sorted(pred_scores.items(), key=lambda x: -x[1])]
                ndcg5 = compute_ndcg(ranked_ids, true_scores, k=5)
                ndcg10 = compute_ndcg(ranked_ids, true_scores, k=10)
                p5 = compute_precision_at_k(ranked_ids, true_scores, k=5, threshold=1)
                all_ndcg5.append(ndcg5)
                all_ndcg10.append(ndcg10)
                all_p5.append(p5)

        # Aggregate metrics
        n_patients = len(queries)
        cls_metrics = classification_metrics(all_true_labels, all_pred_labels)

        report = EvaluationReport(
            n_patients=n_patients,
            n_trials_evaluated=total_trials,
            ndcg_at_5=_safe_mean(all_ndcg5),
            ndcg_at_10=_safe_mean(all_ndcg10),
            precision_at_5=_safe_mean(all_p5),
            three_class_accuracy=cls_metrics.get("accuracy", 0.0),
            macro_f1=cls_metrics.get("macro_f1", 0.0),
            cohens_kappa=cls_metrics.get("cohens_kappa", 0.0),
        )

        # Save
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.config.output_dir / "evaluation_report.json"
        out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        self._progress_cb(f"Evaluation saved to {out_path}")

        return report

    # ──────────────────────────────────────────────────────────────────
    # Save / load state
    # ──────────────────────────────────────────────────────────────────
    def save_state(self, state: AgentState, path: Optional[Path] = None) -> Path:
        """Persist agent state to JSON."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if path is None:
            path = self.config.output_dir / f"state_{state.patient_id}.json"
        path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        self._progress_cb(f"State saved to {path}")
        return path


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _has_insufficient_info(mr: MatchingResult) -> bool:
    """Check if a MatchingResult has any 'not enough information' criteria."""
    for cm in list(mr.inclusion.values()) + list(mr.exclusion.values()):
        if cm.label == "not enough information":
            return True
    return False


def _apply_imaging_updates(
    matching_results: Dict[str, MatchingResult],
    updates: Dict[str, str],
) -> None:
    """Apply imaging validation label updates back to matching results."""
    for nct_id, mr in matching_results.items():
        for criteria_dict in [mr.inclusion, mr.exclusion]:
            for crit_idx, cm in criteria_dict.items():
                if cm.criterion_idx in updates:
                    cm.label = updates[cm.criterion_idx]


def _score_to_3class(
    total_score: float,
    agg: Optional[AggregationResult],
) -> int:
    """Map a predicted total score to a 3-level relevance label.

    Thresholds calibrated to TrialGPT's scoring distribution:
    - 2 (eligible):          score > 0.65 and E >= 50
    - 1 (partially eligible): score > 0.30
    - 0 (not relevant):       otherwise
    """
    if agg and total_score > 0.65:
        e_score = agg.eligibility_score_E if agg.eligibility_score_E else 0.0
        if e_score >= 50:
            return 2
    if total_score > 0.30:
        return 1
    return 0


def _safe_mean(values: List[float]) -> float:
    """Return mean of values, or 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0
