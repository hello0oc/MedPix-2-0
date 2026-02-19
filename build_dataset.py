#!/usr/bin/env python3
"""
build_dataset.py
----------------
Builds a unified, LLM-ready patient dataset that links EHR (clinical text) with
their corresponding medical images, sourced entirely from the local MedPix-2-0
folder.

Inputs (all already present in the repo):
  MedPix-2-0/Case_topic.json       – 671 clinical cases (EHR + diagnosis)
  MedPix-2-0/Descriptions.json     – 2 050 per-image metadata records
  MedPix-2-0/splitted_dataset/uid_list.json – train/dev/test split assignments
  MedPix-2-0/images/*.png          – 2 050 PNG image files

Output folder:
  patient-ehr-image-dataset/
    full_dataset.jsonl              – all 671 linked patient records
    train.jsonl                     – 535 cases (train split)
    train_1.jsonl                   – 267 cases (train-1 half)
    train_2.jsonl                   – 268 cases (train-2 half)
    dev.jsonl                       – 67 cases  (dev split)
    test.jsonl                      – 69 cases  (test split)
    quality_report.json             – completeness and integrity statistics
    README.md                       – dataset card

Usage:
  python3 build_dataset.py
  python3 build_dataset.py --data-dir /path/to/MedPix-2-0
  python3 build_dataset.py --split test --out-dir my_output
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):>4} records → {path.relative_to(path.parents[1])}")


def nonempty(val) -> bool:
    """Return True when val is a non-blank string (or non-empty list)."""
    if val is None:
        return False
    if isinstance(val, list):
        return len(val) > 0
    return str(val).strip() not in ("", "None", "null", "N/A", "n/a")


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_linked_records(
    cases: list,
    descriptions: list,
    images_dir: Path,
) -> tuple[list, dict]:
    """
    Join each case record with its per-image descriptions and verify image files.

    Returns:
        linked   – list of fully linked patient dicts
        stats    – quality statistics dict
    """

    # Index descriptions by U_id
    desc_by_uid: dict[str, list] = defaultdict(list)
    for d in descriptions:
        desc_by_uid[d["U_id"]].append(d)

    linked = []
    stats = {
        "total_cases": len(cases),
        "cases_with_history": 0,
        "cases_with_findings": 0,
        "cases_with_diagnosis": 0,
        "cases_with_images": 0,
        "cases_complete": 0,          # history + findings + diagnosis + images
        "total_images_referenced": 0,
        "total_images_on_disk": 0,
        "missing_images": [],
        "modality_counts": Counter(),
        "location_category_counts": Counter(),
        "cases_missing_descriptions": [],
    }

    for case in cases:
        uid = case["U_id"]
        c   = case.get("Case", {})
        t   = case.get("Topic", {})

        # ── EHR fields ──────────────────────────────────────────────────────
        history               = c.get("History", "")
        exam                  = c.get("Exam", "")
        findings              = c.get("Findings", "")
        differential          = c.get("Differential Diagnosis", "")
        diagnosis             = c.get("Case Diagnosis", "")
        diagnosis_by          = c.get("Diagnosis By", "")
        treatment             = c.get("Treatment & Follow Up", "")
        discussion            = c.get("Discussion", "")
        title                 = c.get("Title", "")

        # ── Topic / disease background ───────────────────────────────────
        topic_title           = t.get("Title", "")
        acr_code              = t.get("ACR Code", "")
        category              = t.get("Category", "")
        disease_discussion    = t.get("Disease Discussion", "")
        keywords              = t.get("Keywords", "")

        # ── Image lists from case record ─────────────────────────────────
        ct_ids  = case.get("TAC", []) or []
        mri_ids = case.get("MRI", []) or []
        all_image_ids = ct_ids + mri_ids

        # ── Resolve each image ───────────────────────────────────────────
        images_out = []
        for img_id in all_image_ids:
            img_path = images_dir / f"{img_id}.png"
            on_disk  = img_path.exists()
            stats["total_images_referenced"] += 1
            if on_disk:
                stats["total_images_on_disk"] += 1
            else:
                stats["missing_images"].append(str(img_path.name))

        # ── Per-image descriptions for this case ─────────────────────────
        case_descs = desc_by_uid.get(uid, [])
        if not case_descs and all_image_ids:
            stats["cases_missing_descriptions"].append(uid)

        for d in case_descs:
            img_id  = d.get("image", "")
            img_path = images_dir / f"{img_id}.png"
            modality = d.get("Description", {}).get("Modality", d.get("Type", ""))
            loc_cat  = d.get("Location Category", "")

            stats["modality_counts"][modality] += 1
            if loc_cat:
                stats["location_category_counts"][loc_cat] += 1

            images_out.append({
                "image_id":         img_id,
                "file_path":        f"MedPix-2-0/images/{img_id}.png",
                "on_disk":          img_path.exists(),
                "type":             d.get("Type", ""),
                "modality":         modality,
                "plane":            d.get("Description", {}).get("Plane", ""),
                "location":         d.get("Location", ""),
                "location_category": loc_cat,
                "caption":          d.get("Description", {}).get("Caption", ""),
                "age":              d.get("Description", {}).get("Age", ""),
                "sex":              d.get("Description", {}).get("Sex", ""),
                "acr_codes":        d.get("Description", {}).get("ACR Codes", ""),
                "figure_part":      d.get("Description", {}).get("Figure Part", ""),
            })

        # ── Quality flags ────────────────────────────────────────────────
        has_history   = nonempty(history)
        has_findings  = nonempty(findings)
        has_diagnosis = nonempty(diagnosis)
        has_images    = len(images_out) > 0

        if has_history:   stats["cases_with_history"]   += 1
        if has_findings:  stats["cases_with_findings"]  += 1
        if has_diagnosis: stats["cases_with_diagnosis"] += 1
        if has_images:    stats["cases_with_images"]    += 1
        if has_history and has_findings and has_diagnosis and has_images:
            stats["cases_complete"] += 1

        # ── Build LLM-ready prompt ───────────────────────────────────────
        prompt_parts = []
        if history:
            prompt_parts.append(f"Clinical History:\n{history.strip()}")
        if exam:
            prompt_parts.append(f"Physical Exam:\n{exam.strip()}")
        if findings:
            prompt_parts.append(f"Imaging Findings:\n{findings.strip()}")
        prompt = "\n\n".join(prompt_parts) if prompt_parts else ""

        linked.append({
            # Identifiers
            "uid":              uid,
            # Core EHR
            "title":            title,
            "history":          history,
            "exam":             exam,
            "findings":         findings,
            "differential_diagnosis": differential,
            "diagnosis":        diagnosis,
            "diagnosis_by":     diagnosis_by,
            "treatment":        treatment,
            "discussion":       discussion,
            # Disease background (Topic)
            "topic_title":      topic_title,
            "acr_code":         acr_code,
            "category":         category,
            "disease_discussion": disease_discussion,
            "keywords":         keywords,
            # Images
            "ct_image_ids":     ct_ids,
            "mri_image_ids":    mri_ids,
            "images":           images_out,
            # LLM-ready prompt
            "llm_prompt":       prompt,
            # Quality flags
            "has_history":      has_history,
            "has_findings":     has_findings,
            "has_diagnosis":    has_diagnosis,
            "has_images":       has_images,
            "is_complete":      has_history and has_findings and has_diagnosis and has_images,
        })

    # Finalise stats
    stats["modality_counts"]          = dict(stats["modality_counts"])
    stats["location_category_counts"] = dict(stats["location_category_counts"])

    return linked, stats


# ---------------------------------------------------------------------------
# Quality report renderer
# ---------------------------------------------------------------------------

def print_quality_report(stats: dict, linked: list):
    n = stats["total_cases"]
    print("\n" + "=" * 60)
    print("  DATA QUALITY REPORT")
    print("=" * 60)
    print(f"  Total cases            : {n}")
    print(f"  Cases with History     : {stats['cases_with_history']:>4}  ({stats['cases_with_history']/n*100:.1f}%)")
    print(f"  Cases with Findings    : {stats['cases_with_findings']:>4}  ({stats['cases_with_findings']/n*100:.1f}%)")
    print(f"  Cases with Diagnosis   : {stats['cases_with_diagnosis']:>4}  ({stats['cases_with_diagnosis']/n*100:.1f}%)")
    print(f"  Cases with Images      : {stats['cases_with_images']:>4}  ({stats['cases_with_images']/n*100:.1f}%)")
    print(f"  Fully complete records : {stats['cases_complete']:>4}  ({stats['cases_complete']/n*100:.1f}%)")
    print(f"  Images referenced      : {stats['total_images_referenced']}")
    print(f"  Images on disk         : {stats['total_images_on_disk']}")
    print(f"  Missing image files    : {len(stats['missing_images'])}")
    if stats["cases_missing_descriptions"]:
        print(f"  Cases w/o desc entries : {len(stats['cases_missing_descriptions'])}")
    print("\n  Modality breakdown:")
    for mod, cnt in sorted(stats["modality_counts"].items(), key=lambda x: -x[1]):
        print(f"    {cnt:>4}  {mod}")
    print("\n  Location category breakdown:")
    for loc, cnt in sorted(stats["location_category_counts"].items(), key=lambda x: -x[1]):
        print(f"    {cnt:>4}  {loc}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# README generator
# ---------------------------------------------------------------------------

def generate_readme(stats: dict, out_dir: Path) -> str:
    n = stats["total_cases"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        "# Patient EHR + Medical Image Dataset (MedPix 2.0 — local build)",
        "",
        f"> Built locally on {now} from the MedPix-2-0 source files.",
        "",
        "## Overview",
        "",
        "This dataset links **clinical EHR narratives** (patient history, imaging",
        "findings, diagnoses) with their **linked radiology images** (CT / MRI).",
        "It is designed for evaluating and fine-tuning medical LLMs on visual",
        "question-answering and diagnostic reasoning tasks.",
        "",
        "## Dataset Statistics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total patient cases | {n} |",
        f"| Cases with history | {stats['cases_with_history']} ({stats['cases_with_history']/n*100:.1f}%) |",
        f"| Cases with imaging findings | {stats['cases_with_findings']} ({stats['cases_with_findings']/n*100:.1f}%) |",
        f"| Cases with diagnosis | {stats['cases_with_diagnosis']} ({stats['cases_with_diagnosis']/n*100:.1f}%) |",
        f"| Cases with linked images | {stats['cases_with_images']} ({stats['cases_with_images']/n*100:.1f}%) |",
        f"| **Fully complete records** | **{stats['cases_complete']} ({stats['cases_complete']/n*100:.1f}%)** |",
        f"| Total image files on disk | {stats['total_images_on_disk']} |",
        f"| Missing image files | {len(stats['missing_images'])} |",
        "",
        "## Files",
        "",
        "| File | Split | Cases |",
        "|---|---|---|",
        f"| `full_dataset.jsonl` | all | {n} |",
        f"| `train.jsonl` | train | 535 |",
        f"| `train_1.jsonl` | train-1 | 267 |",
        f"| `train_2.jsonl` | train-2 | 268 |",
        f"| `dev.jsonl` | dev | 67 |",
        f"| `test.jsonl` | test | 69 |",
        f"| `quality_report.json` | — | — |",
        "",
        "## Record Schema",
        "",
        "Each JSON line in the JSONL files represents one patient case:",
        "",
        "```json",
        "{",
        '  "uid":            "MPX2077",',
        '  "title":          "Choroid Plexus Carcinoma",',
        '  "history":        "15 month old girl fell off a chair...",',
        '  "exam":           "Physical exam was normal for age...",',
        '  "findings":       "CT: High density mass in the trigone...",',
        '  "differential_diagnosis": "Choroid plexus carcinoma\\n...",',
        '  "diagnosis":      "Choroid Plexus Carcinoma",',
        '  "diagnosis_by":   "Biopsy",',
        '  "treatment":      "Surgical excision with follow-up CT and MRI",',
        '  "discussion":     "...",',
        '  "topic_title":    "Choroid Plexus Neoplasm, Papilloma, Carcinoma",',
        '  "acr_code":       "1.3",',
        '  "category":       "Neoplasm, glial",',
        '  "disease_discussion": "...",',
        '  "keywords":       "choroid plexuspapillomacarcinoma",',
        '  "ct_image_ids":   ["MPX2077_synpic51017", ...],',
        '  "mri_image_ids":  ["MPX2077_synpic51021", ...],',
        '  "images": [',
        '    {',
        '      "image_id":         "MPX2077_synpic51017",',
        '      "file_path":        "MedPix-2-0/images/MPX2077_synpic51017.png",',
        '      "on_disk":          true,',
        '      "type":             "CT",',
        '      "modality":         "CT w/contrast (IV)",',
        '      "plane":            "Axial",',
        '      "location":         "Brain, Ventricular",',
        '      "location_category": "Head/Neck",',
        '      "caption":          "...",',
        '      "age":              "1",',
        '      "sex":              "female",',
        '      "acr_codes":        "1.3",',
        '      "figure_part":      "1"',
        '    }',
        '  ],',
        '  "llm_prompt":     "Clinical History:\\n...\\n\\nImaging Findings:\\n...",',
        '  "has_history":    true,',
        '  "has_findings":   true,',
        '  "has_diagnosis":  true,',
        '  "has_images":     true,',
        '  "is_complete":    true',
        "}",
        "```",
        "",
        "## How to Load",
        "",
        "```python",
        "from datasets import load_dataset",
        "",
        "# Load any split",
        "ds = load_dataset('json', data_files='patient-ehr-image-dataset/test.jsonl')['train']",
        "",
        "# Filter for fully complete records",
        "complete = ds.filter(lambda x: x['is_complete'])",
        "print(f'Complete records: {len(complete)}')",
        "",
        "# Build LLM evaluation prompt",
        "sample = complete[0]",
        "print(sample['llm_prompt'])",
        "print('Ground truth:', sample['diagnosis'])",
        "print('Images:', [img['file_path'] for img in sample['images']])",
        "```",
        "",
        "## Source",
        "",
        "Built from [MedPix 2.0](https://arxiv.org/abs/2407.02994):",
        "",
        "```bibtex",
        "@misc{siragusa2025medpix20comprehensivemultimodal,",
        "  title   = {MedPix 2.0: A Comprehensive Multimodal Biomedical Dataset},",
        "  author  = {Irene Siragusa and Salvatore Contino and Massimo La Ciura",
        "             and Rosario Alicata and Roberto Pirrone},",
        "  year    = {2025},",
        "  url     = {https://arxiv.org/abs/2407.02994}",
        "}",
        "```",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build linked EHR + Image patient dataset from MedPix 2.0")
    parser.add_argument("--data-dir",  default="MedPix-2-0",          help="Path to MedPix-2-0 folder")
    parser.add_argument("--out-dir",   default="patient-ehr-image-dataset", help="Output folder")
    parser.add_argument("--split",     default=None,
                        choices=["train", "train_1", "train_2", "dev", "test"],
                        help="Export only this split (default: all splits)")
    parser.add_argument("--complete-only", action="store_true",
                        help="Only include records with history + findings + diagnosis + images")
    args = parser.parse_args()

    repo_root = Path(__file__).parent
    data_dir  = repo_root / args.data_dir
    out_dir   = repo_root / args.out_dir
    images_dir = data_dir / "images"

    # ── Validate inputs ──────────────────────────────────────────────────
    for p in [data_dir, images_dir,
              data_dir / "Case_topic.json",
              data_dir / "Descriptions.json",
              data_dir / "splitted_dataset" / "uid_list.json"]:
        if not p.exists():
            print(f"ERROR: Expected path not found: {p}", file=sys.stderr)
            sys.exit(1)

    print("Loading source data …")
    cases        = load_json(data_dir / "Case_topic.json")
    descriptions = load_json(data_dir / "Descriptions.json")
    uid_list     = load_json(data_dir / "splitted_dataset" / "uid_list.json")

    print(f"  {len(cases)} cases, {len(descriptions)} image descriptions, {len(list(images_dir.glob('*.png')))} PNG files")

    # ── Build linked records ─────────────────────────────────────────────
    print("\nBuilding linked patient records …")
    linked, stats = build_linked_records(cases, descriptions, images_dir)

    if args.complete_only:
        print(f"  --complete-only: filtering {len(linked)} → ", end="")
        linked = [r for r in linked if r["is_complete"]]
        print(f"{len(linked)} records")

    # ── Print quality report ─────────────────────────────────────────────
    print_quality_report(stats, linked)

    # ── Write output files ───────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing output to: {out_dir}/\n")

    # Index linked records by uid for split extraction
    linked_by_uid = {r["uid"]: r for r in linked}

    def extract_split(split_name: str) -> list:
        uids = uid_list.get(split_name, [])
        return [linked_by_uid[u] for u in uids if u in linked_by_uid]

    splits_to_write = {
        "full_dataset": linked,
        "train":   extract_split("train"),
        "train_1": extract_split("train_1"),
        "train_2": extract_split("train_2"),
        "dev":     extract_split("dev"),
        "test":    extract_split("test"),
    }

    if args.split:
        # Only write the requested split (plus full dataset always)
        splits_to_write = {"full_dataset": linked, args.split: splits_to_write[args.split]}

    for name, records in splits_to_write.items():
        save_jsonl(records, out_dir / f"{name}.jsonl")

    # Save quality report
    quality_path = out_dir / "quality_report.json"
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),

            "statistics": stats,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Quality report → {quality_path.relative_to(repo_root)}")

    # Save README
    readme_content = generate_readme(stats, out_dir)
    readme_path = out_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  README         → {readme_path.relative_to(repo_root)}")

    print(f"\nDone. Dataset ready in: {out_dir}/")


if __name__ == "__main__":
    main()
