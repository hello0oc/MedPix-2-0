#!/usr/bin/env python3
"""Filter the TrialGPT trial corpus for NSCLC-related trials.

Produces ``trial_matching_agent/data/nsclc_trial_ids.json``
listing NCT IDs that match NSCLC patterns for focused demo runs.

Usage::

    python -m trial_matching_agent.filter_nsclc_trials
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import DATA_DIR, _matches_nsclc
from .data_loader import load_trial_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def filter_nsclc_trials(data_dir: Path = DATA_DIR) -> list[str]:
    """Return NCT IDs whose title, summary, or diseases match NSCLC."""
    trials = load_trial_info(data_dir)
    nsclc_ids: list[str] = []

    for nct_id, trial in trials.items():
        text_to_check = " ".join([
            trial.brief_title or "",
            trial.brief_summary or "",
            " ".join(trial.diseases_list),
            trial.inclusion_criteria or "",
        ])
        if _matches_nsclc(text_to_check):
            nsclc_ids.append(nct_id)

    return nsclc_ids


def main() -> None:
    nsclc_ids = filter_nsclc_trials()
    out_path = DATA_DIR / "nsclc_trial_ids.json"
    out_path.write_text(json.dumps(nsclc_ids, indent=2), encoding="utf-8")
    logger.info(f"Found {len(nsclc_ids)} NSCLC-related trials → {out_path}")

    # Show a few examples
    trials = load_trial_info(DATA_DIR)
    for nct_id in nsclc_ids[:5]:
        t = trials.get(nct_id)
        if t:
            logger.info(f"  {nct_id}: {t.brief_title[:80]}")


if __name__ == "__main__":
    main()
