#!/usr/bin/env python3
"""CLI runner for the trial matching agent.

Three modes:
  evaluate  — Run TrialGPT ground-truth evaluation (Track A).
  demo      — Process one NSCLC patient (Track B, image-enhanced).
  resume    — Supply feedback answers and re-run matching.

Usage::

    # Evaluate against TrialGPT SIGIR corpus
    python -m trial_matching_agent.run_demo evaluate

    # Demo with a specific NSCLC patient
    python -m trial_matching_agent.run_demo demo --patient-id 37

    # Resume with feedback
    python -m trial_matching_agent.run_demo resume \\
        --state-file output/state_37.json \\
        --answers '{"ECOG performance status": "ECOG 1"}'
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent import TrialMatchingAgent
from .config import AgentConfig, OUTPUT_DIR
from .data_loader import (
    build_ehr_text,
    get_resolved_images,
    load_nsclc_cases,
    load_prebuilt_profiles,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# MODE 1: evaluate
# ──────────────────────────────────────────────────────────────────────
def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run TrialGPT ground-truth evaluation."""
    config = AgentConfig.from_env()
    if not config.gemini_api_key:
        logger.error("GEMINI_API_KEY not found. Set via env or .streamlit/secrets.toml.")
        sys.exit(1)

    if args.max_trials:
        config.max_candidate_trials = args.max_trials

    agent = TrialMatchingAgent(config, progress_cb=_log_progress)

    patient_ids = None
    if args.patient_ids:
        patient_ids = [p.strip() for p in args.patient_ids.split(",")]

    logger.info("Starting TrialGPT evaluation (Track A) …")
    report = agent.evaluate_trialgpt(
        patient_ids=patient_ids,
        max_trials_per_patient=config.max_candidate_trials,
    )

    logger.info("═══ Evaluation Report ═══")
    for k, v in report.to_dict().items():
        logger.info(f"  {k}: {v}")


# ──────────────────────────────────────────────────────────────────────
# MODE 2: demo
# ──────────────────────────────────────────────────────────────────────
def cmd_demo(args: argparse.Namespace) -> None:
    """Run demo with one NSCLC patient."""
    config = AgentConfig.from_env()
    if not config.gemini_api_key:
        logger.error("GEMINI_API_KEY not found. Set via env or .streamlit/secrets.toml.")
        sys.exit(1)

    if args.max_trials:
        config.max_candidate_trials = args.max_trials

    # Load NSCLC cases
    cases = load_nsclc_cases()
    if not cases:
        logger.error("No NSCLC cases found. Run build_nsclc_dataset.py first.")
        sys.exit(1)

    # Select patient
    patient_id = args.patient_id
    if patient_id:
        case = next((c for c in cases if str(c.get("topic_id", "")) == str(patient_id)), None)
        if not case:
            logger.error(f"Patient {patient_id} not found. Available: {[c.get('topic_id') for c in cases[:5]]}")
            sys.exit(1)
    else:
        case = cases[0]
        patient_id = case.get("topic_id", "nsclc_0")
        logger.info(f"No patient specified; using first: {patient_id}")

    # Build EHR text
    ehr_text = build_ehr_text(case, strip_findings=False)
    logger.info(f"EHR text: {len(ehr_text)} chars")

    # Resolve images
    image_paths = get_resolved_images(case)
    logger.info(f"Images: {len(image_paths)} files")

    # Load prebuilt profile if available
    profiles = load_prebuilt_profiles()
    prebuilt = profiles.get(str(patient_id))

    agent = TrialMatchingAgent(config, progress_cb=_log_progress)

    logger.info(f"Starting demo for patient {patient_id} (Track B) …")
    state = agent.run(
        patient_text=ehr_text,
        patient_id=str(patient_id),
        image_paths=image_paths,
        prebuilt_profile=prebuilt,
    )

    # Display results
    _print_results(state)

    # Save
    out_path = agent.save_state(state)
    logger.info(f"State saved to {out_path}")


# ──────────────────────────────────────────────────────────────────────
# MODE 3: resume
# ──────────────────────────────────────────────────────────────────────
def cmd_resume(args: argparse.Namespace) -> None:
    """Resume with feedback answers."""
    config = AgentConfig.from_env()
    if not config.gemini_api_key:
        logger.error("GEMINI_API_KEY not found.")
        sys.exit(1)

    state_path = Path(args.state_file)
    if not state_path.exists():
        logger.error(f"State file not found: {state_path}")
        sys.exit(1)

    # Load state (simplified — reload from JSON)
    state_dict = json.loads(state_path.read_text("utf-8"))

    # Parse answers
    try:
        answers = json.loads(args.answers)
    except json.JSONDecodeError:
        logger.error("Invalid JSON for --answers")
        sys.exit(1)

    logger.info(f"Resuming with {len(answers)} answers …")
    logger.info("Note: Full state resumption requires re-running matching.")
    logger.info(f"Answers: {answers}")

    # For a full resume we'd deserialize AgentState; this demo shows the API
    from .schemas import AgentState
    state = AgentState(
        patient_id=state_dict.get("patient_id", "unknown"),
        patient_text=state_dict.get("patient_text", ""),
    )

    agent = TrialMatchingAgent(config, progress_cb=_log_progress)
    state = agent.resume(state, answers)

    _print_results(state)
    agent.save_state(state)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _log_progress(msg: str) -> None:
    """Progress callback for the agent."""
    logger.info(msg)


def _print_results(state: Any) -> None:
    """Pretty-print agent results."""
    print("\n" + "═" * 70)
    print(f"  TRIAL MATCHING RESULTS — Patient {state.patient_id}")
    print("═" * 70)

    if hasattr(state, "ranked_trials"):
        for i, rt in enumerate(state.ranked_trials[:10], 1):
            print(f"\n  {i}. {rt.nct_id}  (score: {rt.total_score:.3f})")
            print(f"     {rt.title[:70]}")
            if rt.aggregation_result:
                ar = rt.aggregation_result
                print(f"     R={ar.relevance_score:.0f}  E={ar.eligibility_score:.0f}  "
                      f"Label={ar.predicted_label}")

    if hasattr(state, "feedback_requests") and state.feedback_requests:
        print(f"\n  ─── Missing Information ({len(state.feedback_requests)} items) ───")
        for fb in state.feedback_requests:
            print(f"    [{fb.priority.upper() if isinstance(fb.priority, str) else fb.priority}] "
                  f"{fb.question}")
            if fb.reason:
                print(f"      Reason: {fb.reason}")

    print("\n" + "═" * 70)


# ──────────────────────────────────────────────────────────────────────
# CLI argument parser
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trial Matching Agent — CLI Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Mode of operation")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="TrialGPT ground-truth evaluation")
    p_eval.add_argument(
        "--patient-ids",
        help="Comma-separated patient IDs (default: all)",
    )
    p_eval.add_argument(
        "--max-trials",
        type=int, default=10,
        help="Max trials per patient (default: 10)",
    )

    # demo
    p_demo = sub.add_parser("demo", help="Demo with one NSCLC patient")
    p_demo.add_argument(
        "--patient-id",
        help="NSCLC patient topic_id (default: first available)",
    )
    p_demo.add_argument(
        "--max-trials",
        type=int, default=5,
        help="Max candidate trials (default: 5)",
    )

    # resume
    p_resume = sub.add_parser("resume", help="Resume with feedback answers")
    p_resume.add_argument(
        "--state-file",
        required=True,
        help="Path to saved state JSON",
    )
    p_resume.add_argument(
        "--answers",
        required=True,
        help="JSON string of field→answer pairs",
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "evaluate": cmd_evaluate,
        "demo": cmd_demo,
        "resume": cmd_resume,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
