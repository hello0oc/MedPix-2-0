#!/usr/bin/env python3
"""
Show a complete patient profile by joining all CSV files in the Synthea
coherent zip on the PATIENT (or Id) column.

Usage
-----
  # Print profile for the first/richest patient found
  python Synthetic/patient_profile.py

  # Specify a patient UUID explicitly
  python Synthetic/patient_profile.py --patient 8b0484cd-3dbd-8b8d-1b72-a32f74a5a846

  # List available patient IDs
  python Synthetic/patient_profile.py --list-patients

Options
-------
  --zip PATH         Path to the zip file  [Synthetic/coherent-11-07-2022.zip]
  --patient UUID     Patient UUID to display
  --list-patients    Print a few available patient IDs and exit
  --max-rows N       Max rows per section to display  [10]
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from collections import defaultdict
from zipfile import ZipFile

SCRIPT_DIR = Path(__file__).resolve().parent
ZIP_DEFAULT = str((SCRIPT_DIR / "coherent-11-07-2022.zip").resolve())

# ── helpers ────────────────────────────────────────────────────────────────────

def _read_csv(z: ZipFile, name: str) -> list[dict]:
    with z.open(name) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        return list(csv.DictReader(text))


def _patient_col(table_name: str, row: dict) -> str | None:
    """Return the value of the patient ID column for this table.

    Critical rule:
    - For non-`patients` tables, only `PATIENT` is treated as patient foreign key.
    - For `patients` table, `Id`/`id` is the patient primary key.

    This prevents accidental indexing by unrelated generic `Id` columns from
    non-patient tables.
    """
    if "PATIENT" in row and row.get("PATIENT"):
        return str(row["PATIENT"]).strip()

    if table_name == "patients":
        for key in ("Id", "id"):
            if key in row and row.get(key):
                return str(row[key]).strip()

    return None


def _load_tables(zip_path: str) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {}
    with ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(".csv"):
                short = name.split("/")[-1].replace(".csv", "")
                try:
                    tables[short] = _read_csv(z, name)
                except Exception as e:
                    print(f"  [warn] could not read {name}: {e}", file=sys.stderr)
    return tables


def _index_by_patient(tables: dict[str, list[dict]]) -> dict[str, dict[str, list[dict]]]:
    """Returns {table_name: {patient_id: [row, ...]}}."""
    idx: dict[str, dict[str, list[dict]]] = {}
    for tname, rows in tables.items():
        by_pt: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            pid = _patient_col(tname, row)
            if pid:
                by_pt[pid].append(row)
        if by_pt:
            idx[tname] = dict(by_pt)
    return idx


def _all_patient_ids(idx: dict) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for tname, by_pt in idx.items():
        for pid in by_pt:
            counts[pid] += len(by_pt[pid])
    # sort by most records so the default patient is the richest one
    return sorted(counts, key=lambda p: counts[p], reverse=True)


# ── display ────────────────────────────────────────────────────────────────────

def _fmt_row(row: dict, skip_cols: set[str] | None = None) -> str:
    skip_cols = skip_cols or set()
    parts = [f"{k}: {v}" for k, v in row.items() if k not in skip_cols and v]
    return "  " + " | ".join(parts)


def _print_section(title: str, rows: list[dict], max_rows: int, skip_cols: set[str] | None = None) -> None:
    if not rows:
        return
    print(f"\n{'─'*70}")
    print(f"  {title.upper()}  ({len(rows)} record{'s' if len(rows)!=1 else ''})")
    print(f"{'─'*70}")
    for r in rows[:max_rows]:
        print(_fmt_row(r, skip_cols))
    if len(rows) > max_rows:
        print(f"  ... {len(rows) - max_rows} more row(s) not shown")


def print_patient_profile(
    tables: dict[str, list[dict]],
    idx: dict[str, dict[str, list[dict]]],
    patient_id: str,
    max_rows: int = 10,
) -> None:
    skip = {"PATIENT", "Id", "id"}

    print(f"\n{'═'*70}")
    print(f"  PATIENT PROFILE")
    print(f"  ID: {patient_id}")
    print(f"{'═'*70}")

    # ── demographics (patients table has Id == patient_id) ──────────────────
    if "patients" in tables:
        pt_rows = [r for r in tables["patients"] if r.get("Id") == patient_id]
        if pt_rows:
            r = pt_rows[0]
            print(f"\n  Name   : {r.get('FIRST','')} {r.get('LAST','')}")
            print(f"  Born   : {r.get('BIRTHDATE','')}   Died: {r.get('DEATHDATE','—')}")
            print(f"  Gender : {r.get('GENDER','')}   Race: {r.get('RACE','')}   Ethnicity: {r.get('ETHNICITY','')}")
            print(f"  Address: {r.get('ADDRESS','')}, {r.get('CITY','')} {r.get('STATE','')} {r.get('ZIP','')}")
            marital = r.get("MARITAL", "")
            if marital:
                print(f"  Marital: {marital}")

    # ── section order ────────────────────────────────────────────────────────
    section_order = [
        ("encounters",       "Encounters"),
        ("conditions",       "Conditions / Diagnoses"),
        ("medications",      "Medications"),
        ("procedures",       "Procedures"),
        ("observations",     "Observations / Vitals"),
        ("imaging_studies",  "Imaging Studies"),
        ("immunizations",    "Immunizations"),
        ("allergies",        "Allergies"),
        ("careplans",        "Care Plans"),
        ("devices",          "Devices"),
        ("supplies",         "Supplies"),
        ("payer_transitions","Insurance / Payer History"),
    ]

    shown = set()
    for tkey, tlabel in section_order:
        shown.add(tkey)
        rows = idx.get(tkey, {}).get(patient_id, [])
        _print_section(tlabel, rows, max_rows, skip_cols=skip)

    # ── any remaining tables we haven't listed explicitly ────────────────────
    for tkey, by_pt in idx.items():
        if tkey in shown or tkey == "patients":
            continue
        rows = by_pt.get(patient_id, [])
        if rows:
            _print_section(tkey, rows, max_rows, skip_cols=skip)

    print(f"\n{'═'*70}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Patient profile from Synthea zip")
    parser.add_argument("--zip", default=ZIP_DEFAULT, help="Path to zip file")
    parser.add_argument("--patient", default=None, help="Patient UUID")
    parser.add_argument("--list-patients", action="store_true")
    parser.add_argument("--max-rows", type=int, default=10)
    args = parser.parse_args(argv)

    zip_path = Path(args.zip)
    if not zip_path.is_absolute():
        zip_path = (Path.cwd() / zip_path).resolve()
        if not zip_path.exists():
            fallback = (SCRIPT_DIR / args.zip).resolve()
            if fallback.exists():
                zip_path = fallback

    if not zip_path.exists():
        print(f"Zip file not found: {zip_path}", file=sys.stderr)
        return 3

    print(f"Loading tables from {zip_path} …", file=sys.stderr)
    tables = _load_tables(str(zip_path))
    idx = _index_by_patient(tables)

    all_pids = _all_patient_ids(idx)
    if not all_pids:
        print("No patient IDs found.", file=sys.stderr)
        return 1

    if args.list_patients:
        print("Sample patient IDs (most records first):")
        for pid in all_pids[:20]:
            total = sum(len(idx[t].get(pid, [])) for t in idx)
            print(f"  {pid}  ({total} records)")
        return 0

    patient_id = args.patient or all_pids[0]
    if patient_id not in all_pids:
        print(f"Patient '{patient_id}' not found in any table.", file=sys.stderr)
        print("Run with --list-patients to see available IDs.", file=sys.stderr)
        return 2

    print_patient_profile(tables, idx, patient_id, max_rows=args.max_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
