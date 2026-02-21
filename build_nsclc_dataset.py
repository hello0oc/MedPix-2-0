#!/usr/bin/env python3
"""
build_nsclc_dataset.py
======================
Scans all three existing patient datasets for Non-Small Cell Lung Cancer
(NSCLC) cases and writes a unified NSCLC-focused JSONL file to:

    nsclc-dataset/nsclc_dataset.jsonl

Sources scanned
---------------
1. patient-ehr-image-dataset/full_dataset.jsonl   (MedPix 2.0)
2. Synthetic/synthetic_ehr_image_dataset.jsonl    (Synthea coherent)
3. PMCpatient/PMC-Patients-sample-1000.csv        (PMC-Patients)

Image file paths are preserved verbatim from each originating dataset —
no image files are copied.  The Streamlit app's resolve_image_path()
already handles repo-relative paths from all three source locations.

Usage
-----
    python build_nsclc_dataset.py [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

# ── Paths ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent

MEDPIX_PATH    = REPO_ROOT / "patient-ehr-image-dataset" / "full_dataset.jsonl"
SYNTHETIC_PATH = REPO_ROOT / "Synthetic" / "synthetic_ehr_image_dataset.jsonl"
PMC_PATH       = REPO_ROOT / "PMCpatient" / "PMC-Patients-sample-1000.csv"

OUTPUT_DIR  = REPO_ROOT / "nsclc-dataset"
OUTPUT_FILE = OUTPUT_DIR / "nsclc_dataset.jsonl"

# ── NSCLC keyword patterns ────────────────────────────────────────────────
# Matches terms selected by the user:
#   • non-small cell lung cancer / NSCLC
#   • adenocarcinoma of (the) lung
#   • squamous cell carcinoma of (the) lung
#   • large cell carcinoma (of lung)
NSCLC_PATTERNS: List[re.Pattern] = [
    re.compile(r"non.?small.?cell", re.IGNORECASE),
    re.compile(r"\bnsclc\b", re.IGNORECASE),
    re.compile(r"adenocarcinoma\s+of\s+(the\s+)?lung", re.IGNORECASE),
    re.compile(r"lung\s+adenocarcinoma", re.IGNORECASE),
    re.compile(r"squamous\s+cell\s+carcinoma\s+of\s+(the\s+)?lung", re.IGNORECASE),
    re.compile(r"lung\s+squamous\s+cell", re.IGNORECASE),
    re.compile(r"large\s+cell\s+carcinoma", re.IGNORECASE),
    re.compile(r"large.cell\s+lung", re.IGNORECASE),
]


def _matches_nsclc(text: str) -> bool:
    """Return True if *text* contains any NSCLC keyword."""
    return any(pat.search(text) for pat in NSCLC_PATTERNS)


def _record_matches(record: Dict[str, Any]) -> bool:
    """
    Return True if the record contains NSCLC-relevant content in any
    of the free-text or structured classification fields.
    """
    fields = [
        record.get("title", ""),
        record.get("diagnosis", ""),
        record.get("topic_title", ""),
        record.get("keywords", ""),
        record.get("history", ""),
        record.get("exam", ""),
        record.get("findings", ""),
        record.get("discussion", ""),
        record.get("disease_discussion", ""),
        record.get("category", ""),
        record.get("differential_diagnosis", ""),
        record.get("treatment", ""),
        # PMC-Patients raw narrative (before or after normalisation)
        record.get("patient", ""),
        record.get("llm_prompt", ""),
    ]
    combined = " ".join(f for f in fields if f)
    return _matches_nsclc(combined)


# ── Loaders ────────────────────────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] {path.name} line {lineno}: {exc}", file=sys.stderr)


def _normalize_pmc_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a PMC-Patients CSV row to the standard patient schema used by the
    MedPix JSONL datasets.  Images list is empty (PMC has no linked images).
    Mirrors the logic in medgemma_gui/app.py::_normalize_pmc_row().
    """
    uid     = (row.get("patient_uid") or row.get("patient_id") or "").strip()
    title   = (row.get("title") or "PMC clinical case").strip()
    history = (row.get("patient") or "").strip()
    age     = (row.get("age") or "").strip()
    gender  = (row.get("gender") or "").strip()
    pmid    = (row.get("PMID") or "").strip()

    exam_parts: List[str] = []
    if age:
        exam_parts.append(f"Age: {age}")
    if gender:
        exam_parts.append(f"Gender: {gender}")
    if pmid:
        exam_parts.append(f"PMID: {pmid}")

    findings_parts: List[str] = []
    relevant = (row.get("relevant_articles") or "").strip()
    similar  = (row.get("similar_patients") or "").strip()
    if relevant:
        findings_parts.append(f"Related articles: {relevant}")
    if similar:
        findings_parts.append(f"Similar patients: {similar}")

    return {
        "uid":                uid,
        "title":              title,
        "history":            history,
        "exam":               "\n".join(exam_parts),
        "findings":           "\n".join(findings_parts),
        "diagnosis":          "",
        "discussion":         "",
        "images":             [],
        "ct_image_ids":       [],
        "mri_image_ids":      [],
        "has_history":        bool(history),
        "has_findings":       False,
        "has_diagnosis":      False,
        "has_images":         False,
        "is_complete":        False,
        "dataset_source":     "PMC-Patients",
        "nsclc_source":       "PMC-Patients",
    }


def load_medpix_nsclc(path: Path) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for record in _iter_jsonl(path):
        if not record.get("uid"):
            continue
        if _record_matches(record):
            record.setdefault("dataset_source", "MedPix")
            record["nsclc_source"] = "MedPix"
            matched.append(record)
    return matched


def load_synthetic_nsclc(path: Path) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for record in _iter_jsonl(path):
        if not record.get("uid"):
            continue
        if _record_matches(record):
            record.setdefault("dataset_source", "Synthea coherent zip")
            record["nsclc_source"] = "Synthetic"
            matched.append(record)
    return matched


def load_pmc_nsclc(path: Path) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Test on raw row (includes the 'patient' narrative field)
            # so we match before normalisation discards fields.
            raw_text = " ".join(str(v) for v in row.values())
            if not _matches_nsclc(raw_text):
                continue
            normalized = _normalize_pmc_row(row)
            if normalized.get("uid"):
                matched.append(normalized)
    return matched


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print statistics but do not write the output file.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("NSCLC Dataset Builder")
    print("=" * 60)

    all_cases: List[Dict[str, Any]] = []

    # ── Source 1: MedPix ──
    if MEDPIX_PATH.exists():
        medpix_cases = load_medpix_nsclc(MEDPIX_PATH)
        print(f"MedPix          : {len(medpix_cases):>4} NSCLC case(s) found")
        all_cases.extend(medpix_cases)
    else:
        print(f"MedPix          : [SKIP] file not found — {MEDPIX_PATH}")

    # ── Source 2: Synthetic ──
    if SYNTHETIC_PATH.exists():
        synthetic_cases = load_synthetic_nsclc(SYNTHETIC_PATH)
        print(f"Synthetic       : {len(synthetic_cases):>4} NSCLC case(s) found")
        all_cases.extend(synthetic_cases)
    else:
        print(f"Synthetic       : [SKIP] file not found — {SYNTHETIC_PATH}")

    # ── Source 3: PMC-Patients ──
    if PMC_PATH.exists():
        pmc_cases = load_pmc_nsclc(PMC_PATH)
        print(f"PMC-Patients    : {len(pmc_cases):>4} NSCLC case(s) found")
        all_cases.extend(pmc_cases)
    else:
        print(f"PMC-Patients    : [SKIP] file not found — {PMC_PATH}")

    # ── Deduplicate by uid (different namespaces, but be safe) ──
    seen_uids: set = set()
    deduped: List[Dict[str, Any]] = []
    for case in all_cases:
        uid = case.get("uid", "")
        if uid in seen_uids:
            print(f"  [WARN] Duplicate uid skipped: {uid}")
        else:
            seen_uids.add(uid)
            deduped.append(case)

    # ── Statistics ──
    ehr_only   = sum(1 for c in deduped if not c.get("images"))
    ehr_images = len(deduped) - ehr_only
    total_imgs = sum(len(c.get("images", [])) for c in deduped)

    print("-" * 60)
    print(f"Total cases     : {len(deduped)}")
    print(f"  EHR only      : {ehr_only}")
    print(f"  EHR + images  : {ehr_images}  ({total_imgs} linked image records)")
    print("-" * 60)

    if args.dry_run:
        print("[DRY RUN] Output file not written.")
        return

    # ── Write output ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for case in deduped:
            out.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Written to      : {OUTPUT_FILE.relative_to(REPO_ROOT)}")
    print("=" * 60)
    print()
    print("No image files were copied. Image file_path values point to")
    print("their original locations within the repository.")


if __name__ == "__main__":
    main()
