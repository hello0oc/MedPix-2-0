#!/usr/bin/env python3
"""
prepare_input_strategy.py
─────────────────────────
Convert PMC-Patients CSV (or V2 JSON) into the project-standard JSONL format
used by patient-ehr-image-dataset/ and medgemma_benchmark/.

Schema produced (mirrors patient-ehr-image-dataset/full_dataset.jsonl)
──────────────────────────────────────────────────────────────────────
{
  "uid":               str,   # patient_uid from PMC-Patients  (e.g. "7665777-1")
  "pmid":              str,   # PubMed ID of the source article
  "title":             str,   # Article title
  "patient":           str,   # Full patient narrative (primary clinical text)
  "age_raw":           str,   # Raw age string  (e.g. "[[60.0, 'year']]")
  "age_years":         float | null,  # Normalised age in years
  "gender":            str,   # "M" | "F" | ""
  "history":           str,   # Alias of patient (used by medgemma pipeline)
  "diagnosis":         str,   # "" — PMC-Patients has no gold diagnosis label
  "relevant_articles": dict,  # {PMID: score} relevance annotations
  "similar_patients":  dict,  # {patient_uid: score} similarity annotations
  "llm_prompt":        str,   # Ready-to-use free-text prompt for LLM
  "has_history":       bool,
  "has_diagnosis":     bool,  # always False for this dataset
  "has_images":        bool,  # always False; PMC-Patients has no images
  "is_complete":       bool
}

Input strategies
────────────────
--strategy ehr_only    (default)
    llm_prompt = "[Age: X yr / Gender: Y]\n<patient narrative>"

--strategy structured
    llm_prompt = "Age: X yr\\nGender: Y\\nPatient Summary:\\n<narrative>"

--strategy qa
    llm_prompt = turns the summary into a Q&A differential-diagnosis template
                 suitable for few-shot evaluation

Usage
─────
  # Convert default CSV (full)
  python prepare_input_strategy.py --input PMC-Patients.csv

  # Convert a sampled CSV with a specific strategy
  python prepare_input_strategy.py \\
      --input PMC-Patients-sample-5000.csv \\
      --strategy structured \\
      --output pmc_patients_structured.jsonl

  # Split into train/dev/test JSONL files (80/10/10)
  python prepare_input_strategy.py \\
      --input PMC-Patients.csv \\
      --split \\
      --seed 42

  # Convert V2 JSON instead of CSV
  python prepare_input_strategy.py --input PMC-Patients-V2.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# ── age normalisation ────────────────────────────────────────────────────────
UNIT_TO_YEARS: dict[str, float] = {
    "year":  1.0,
    "month": 1 / 12,
    "week":  1 / 52,
    "day":   1 / 365,
    "hour":  1 / 8760,
}


def _parse_age_years(age_str: str) -> Optional[float]:
    """Return age in years from raw age string, or None on failure."""
    if not age_str:
        return None
    try:
        parsed = ast.literal_eval(age_str)
        if not parsed:
            return None
        total = sum(float(v) * UNIT_TO_YEARS.get(u, 0.0) for v, u in parsed)
        return round(total, 2)
    except Exception:
        # fallback: look for a plain year number
        m = re.search(r"(\d+(?:\.\d+)?)\s*,\s*'year'", age_str)
        if m:
            return float(m.group(1))
        return None


def _age_label(age_years: Optional[float]) -> str:
    if age_years is None:
        return "unknown age"
    return f"{age_years:.0f} yr" if age_years == int(age_years) else f"{age_years:.1f} yr"


def _gender_label(gender: str) -> str:
    return {"M": "Male", "F": "Female"}.get(gender.strip().upper(), "Unknown gender")


def _safe_dict(value: str) -> dict:
    if not value or value.strip() in ("{}", ""):
        return {}
    try:
        return ast.literal_eval(value)
    except Exception:
        try:
            return json.loads(value.replace("'", '"'))
        except Exception:
            return {}


# ── LLM prompt builders ──────────────────────────────────────────────────────

def _prompt_ehr_only(patient: str, age_years: Optional[float], gender: str) -> str:
    """Compact header + narrative — mirrors medgemma_benchmark EHR-only mode."""
    header = f"[Age: {_age_label(age_years)} | Gender: {_gender_label(gender)}]"
    return f"{header}\n\n{patient.strip()}"


def _prompt_structured(patient: str, age_years: Optional[float], gender: str) -> str:
    """Labelled key-value block for structured text models."""
    lines = [
        f"Age: {_age_label(age_years)}",
        f"Gender: {_gender_label(gender)}",
        "Patient Summary:",
        patient.strip(),
    ]
    return "\n".join(lines)


def _prompt_qa(patient: str, age_years: Optional[float], gender: str) -> str:
    """Q&A differential-diagnosis template for evaluation / few-shot use."""
    return (
        f"Patient: {_age_label(age_years)}, {_gender_label(gender)}\n\n"
        f"Clinical presentation:\n{patient.strip()}\n\n"
        "Task: Based on this clinical presentation, provide:\n"
        "1. The most likely diagnosis.\n"
        "2. Two alternative differential diagnoses.\n"
        "3. A brief clinical rationale (2-4 sentences).\n"
        "Return valid JSON: "
        '{"diagnosis":"...", "differentials":["...","..."], "rationale":"..."}'
    )


PROMPT_BUILDERS = {
    "ehr_only":   _prompt_ehr_only,
    "structured": _prompt_structured,
    "qa":         _prompt_qa,
}


# ── record conversion ────────────────────────────────────────────────────────

def row_to_record(row: dict[str, str], strategy: str) -> dict[str, Any]:
    """Convert a raw PMC-Patients row to a project-standard record dict."""
    patient   = row.get("patient", "").strip()
    gender    = row.get("gender", "").strip()
    age_raw   = row.get("age", "").strip()
    age_years = _parse_age_years(age_raw)

    build_prompt = PROMPT_BUILDERS[strategy]
    llm_prompt   = build_prompt(patient, age_years, gender)

    rel_articles = _safe_dict(row.get("relevant_articles", ""))
    sim_patients = _safe_dict(row.get("similar_patients",  ""))

    has_history = bool(patient)
    is_complete = has_history and bool(gender)

    return {
        "uid":               row.get("patient_uid", ""),
        "pmid":              row.get("PMID", ""),
        "title":             row.get("title", "").strip(),
        "patient":           patient,
        "age_raw":           age_raw,
        "age_years":         age_years,
        "gender":            gender,
        # aliases for medgemma_benchmark compatibility
        "history":           patient,
        "exam":              "",
        "findings":          "",
        "diagnosis":         "",          # not available in PMC-Patients
        "differential_diagnosis": "",
        "treatment":         "",
        "relevant_articles": rel_articles,
        "similar_patients":  sim_patients,
        "llm_prompt":        llm_prompt,
        "has_history":       has_history,
        "has_diagnosis":     False,       # PMC-Patients has no gold labels
        "has_images":        False,       # text-only dataset
        "is_complete":       is_complete,
    }


# ── readers ──────────────────────────────────────────────────────────────────

def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield dict(row)


def iter_json(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        yield from data.values()
    else:
        raise ValueError(f"Unexpected JSON root type: {type(data)}")


def iter_source(path: Path) -> Iterator[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return iter_csv(path)
    elif suffix == ".json":
        return iter_json(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}")


# ── writers ──────────────────────────────────────────────────────────────────

def write_jsonl(records: list[dict], dest: Path) -> None:
    with dest.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  ✓  {len(records):>7,} records → {dest}", file=sys.stderr)


def split_records(
    records: list[dict],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n   = len(shuffled)
    n_train = int(n * ratios[0])
    n_dev   = int(n * ratios[1])
    train  = shuffled[:n_train]
    dev    = shuffled[n_train : n_train + n_dev]
    test   = shuffled[n_train + n_dev :]
    return train, dev, test


# ── quality report ───────────────────────────────────────────────────────────

def quality_report(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"total": 0}
    has_history = sum(1 for r in records if r["has_history"])
    has_gender  = sum(1 for r in records if r["gender"])
    has_age     = sum(1 for r in records if r["age_years"] is not None)
    is_complete = sum(1 for r in records if r["is_complete"])
    genders     = {"M": 0, "F": 0, "other": 0}
    for r in records:
        g = r["gender"].upper()
        genders["M" if g == "M" else ("F" if g == "F" else "other")] += 1
    avg_len = sum(len(r["patient"]) for r in records) / n
    return {
        "total":            n,
        "has_history":      has_history,
        "has_gender":       has_gender,
        "has_age":          has_age,
        "is_complete":      is_complete,
        "completeness_pct": round(is_complete / n * 100, 1),
        "gender_counts":    genders,
        "avg_patient_chars": round(avg_len, 1),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    here = Path(__file__).parent

    p = argparse.ArgumentParser(
        description="Convert PMC-Patients CSV/JSON to project-standard JSONL"
    )
    p.add_argument(
        "--input", "-i",
        default=str(here / "PMC-Patients.csv"),
        help="Path to PMC-Patients.csv or PMC-Patients-V2.json",
    )
    p.add_argument(
        "--strategy", "-s",
        choices=list(PROMPT_BUILDERS.keys()),
        default="ehr_only",
        help="LLM prompt construction strategy (default: ehr_only)",
    )
    p.add_argument(
        "--output", "-o",
        default="",
        help="Output JSONL file (default: <input_stem>_<strategy>.jsonl next to input)",
    )
    p.add_argument(
        "--split",
        action="store_true",
        help="Split output into train/dev/test JSONL files (80/10/10)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for split (default: 42)",
    )
    p.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="Process only first N rows (0 = all)",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="Print quality report to stdout after conversion",
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    src    = Path(args.input)

    if not src.exists():
        sys.exit(f"File not found: {src}\nRun download_pmc_patients.py first.")

    # ── load & convert ───────────────────────────────────────────────────────
    print(f"\nLoading  {src} …", file=sys.stderr)
    records: list[dict] = []
    for i, raw in enumerate(iter_source(src)):
        if args.limit and i >= args.limit:
            break
        records.append(row_to_record(raw, args.strategy))

    print(f"Converted {len(records):,} records using strategy '{args.strategy}'", file=sys.stderr)

    # ── quality report ────────────────────────────────────────────────────────
    report = quality_report(records)
    report_path = src.parent / "quality_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  ✓  Quality report → {report_path}", file=sys.stderr)
    if args.report:
        print(json.dumps(report, indent=2))

    # ── write output ──────────────────────────────────────────────────────────
    if args.split:
        train, dev, test = split_records(records, seed=args.seed)
        stem = src.stem
        out_dir = src.parent
        write_jsonl(train, out_dir / f"{stem}_{args.strategy}_train.jsonl")
        write_jsonl(dev,   out_dir / f"{stem}_{args.strategy}_dev.jsonl")
        write_jsonl(test,  out_dir / f"{stem}_{args.strategy}_test.jsonl")
    else:
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = src.parent / f"{src.stem}_{args.strategy}.jsonl"
        write_jsonl(records, out_path)


if __name__ == "__main__":
    main()
