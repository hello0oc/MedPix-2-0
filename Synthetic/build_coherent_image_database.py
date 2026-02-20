#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ZIP_PATH = SCRIPT_DIR / "coherent-11-07-2022.zip"
DEFAULT_IDS_OUT = SCRIPT_DIR / "synthetic_patient_ids_with_images.json"
DEFAULT_DATASET_OUT = SCRIPT_DIR / "synthetic_ehr_image_dataset.jsonl"


def _read_csv_from_zip(zf: ZipFile, member: str) -> list[dict[str, str]]:
    with zf.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        return list(csv.DictReader(text))


def _load_tables(zf: ZipFile) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for name in zf.namelist():
        if not name.startswith("csv/") or not name.endswith(".csv"):
            continue
        short = Path(name).stem
        out[short] = _read_csv_from_zip(zf, name)
    return out


def _parse_dt(value: str) -> datetime:
    if not value:
        return datetime.min
    txt = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(txt)
    except ValueError:
        return datetime.min


def _fmt(*parts: str) -> str:
    return "; ".join([p for p in parts if p])


def _group_by_patient(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pid = (row.get("PATIENT") or "").strip()
        if pid:
            out[pid].append(row)
    return dict(out)


def _latest_condition_description(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    sorted_rows = sorted(rows, key=lambda r: _parse_dt(r.get("START", "")), reverse=True)
    for row in sorted_rows:
        desc = (row.get("DESCRIPTION") or "").strip()
        if desc:
            return desc
    return ""


def _build_history(conditions: list[dict[str, str]], medications: list[dict[str, str]], max_rows: int) -> str:
    sections: list[str] = []

    if conditions:
        rows = sorted(conditions, key=lambda r: _parse_dt(r.get("START", "")), reverse=True)[:max_rows]
        lines = [_fmt((r.get("START") or "").strip(), (r.get("DESCRIPTION") or "").strip()) for r in rows]
        sections.append("Conditions:\n" + "\n".join([line for line in lines if line]))

    if medications:
        rows = sorted(medications, key=lambda r: _parse_dt(r.get("START", "")), reverse=True)[:max_rows]
        lines = [
            _fmt(
                (r.get("START") or "").strip(),
                (r.get("DESCRIPTION") or "").strip(),
                (r.get("REASONDESCRIPTION") or "").strip(),
            )
            for r in rows
        ]
        sections.append("Medications:\n" + "\n".join([line for line in lines if line]))

    return "\n\n".join(sections).strip()


def _build_exam(encounters: list[dict[str, str]], observations: list[dict[str, str]], procedures: list[dict[str, str]], max_rows: int) -> str:
    sections: list[str] = []

    if encounters:
        rows = sorted(encounters, key=lambda r: _parse_dt(r.get("START", "")), reverse=True)[:max_rows]
        lines = [
            _fmt(
                (r.get("START") or "").strip(),
                (r.get("DESCRIPTION") or "").strip(),
                (r.get("REASONDESCRIPTION") or "").strip(),
            )
            for r in rows
        ]
        sections.append("Encounters:\n" + "\n".join([line for line in lines if line]))

    if observations:
        rows = sorted(observations, key=lambda r: _parse_dt(r.get("DATE", "")), reverse=True)[:max_rows]
        lines = [
            _fmt(
                (r.get("DATE") or "").strip(),
                (r.get("DESCRIPTION") or "").strip(),
                (r.get("VALUE") or "").strip(),
                (r.get("UNITS") or "").strip(),
            )
            for r in rows
        ]
        sections.append("Observations:\n" + "\n".join([line for line in lines if line]))

    if procedures:
        rows = sorted(procedures, key=lambda r: _parse_dt(r.get("DATE", "")), reverse=True)[:max_rows]
        lines = [
            _fmt(
                (r.get("DATE") or "").strip(),
                (r.get("DESCRIPTION") or "").strip(),
                (r.get("REASONDESCRIPTION") or "").strip(),
            )
            for r in rows
        ]
        sections.append("Procedures:\n" + "\n".join([line for line in lines if line]))

    return "\n\n".join(sections).strip()


def _build_findings(imaging_studies: list[dict[str, str]], max_rows: int) -> str:
    if not imaging_studies:
        return ""
    rows = sorted(imaging_studies, key=lambda r: _parse_dt(r.get("DATE", "")), reverse=True)[:max_rows]
    lines = [
        _fmt(
            (r.get("DATE") or "").strip(),
            (r.get("MODALITY_DESCRIPTION") or "").strip(),
            (r.get("BODY_SITE_DESCRIPTION") or "").strip(),
            (r.get("SOP_DESCRIPTION") or "").strip(),
        )
        for r in rows
    ]
    return "\n".join([line for line in lines if line]).strip()


def _build_llm_prompt(history: str, exam: str, findings: str) -> str:
    parts: list[str] = []
    if history:
        parts.append(f"Clinical History:\n{history}")
    if exam:
        parts.append(f"Physical Exam:\n{exam}")
    if findings:
        parts.append(f"Imaging Findings:\n{findings}")
    return "\n\n".join(parts).strip()


def _extract_patient_ids_with_dicom(zf: ZipFile) -> dict[str, list[str]]:
    by_patient: dict[str, list[str]] = defaultdict(list)
    uuid_pattern = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
    for name in zf.namelist():
        if not name.startswith("dicom/") or not name.endswith(".dcm"):
            continue
        basename = Path(name).name
        match = uuid_pattern.search(basename)
        if not match:
            continue
        patient_id = match.group(0).lower()
        by_patient[patient_id].append(name)
    return dict(by_patient)


def _build_record(
    patient_id: str,
    patient_row: dict[str, str],
    dicom_members: list[str],
    conditions_by_patient: dict[str, list[dict[str, str]]],
    meds_by_patient: dict[str, list[dict[str, str]]],
    encounters_by_patient: dict[str, list[dict[str, str]]],
    observations_by_patient: dict[str, list[dict[str, str]]],
    procedures_by_patient: dict[str, list[dict[str, str]]],
    imaging_by_patient: dict[str, list[dict[str, str]]],
    max_rows: int,
) -> dict[str, Any]:
    first = (patient_row.get("FIRST") or "").strip()
    last = (patient_row.get("LAST") or "").strip()
    sex = (patient_row.get("GENDER") or "").strip()

    conditions = conditions_by_patient.get(patient_id, [])
    medications = meds_by_patient.get(patient_id, [])
    encounters = encounters_by_patient.get(patient_id, [])
    observations = observations_by_patient.get(patient_id, [])
    procedures = procedures_by_patient.get(patient_id, [])
    imaging_studies = imaging_by_patient.get(patient_id, [])

    history = _build_history(conditions, medications, max_rows=max_rows)
    exam = _build_exam(encounters, observations, procedures, max_rows=max_rows)
    findings = _build_findings(imaging_studies, max_rows=max_rows)
    diagnosis = _latest_condition_description(conditions)
    llm_prompt = _build_llm_prompt(history, exam, findings)

    images: list[dict[str, Any]] = []
    for member in sorted(dicom_members):
        image_name = Path(member).name
        image_stem = Path(image_name).stem
        images.append(
            {
                "image_id": image_stem,
                "file_path": f"Synthetic/coherent-11-07-2022.zip::{member}",
                "on_disk": False,
                "type": "DICOM",
                "modality": "",
                "plane": "",
                "location": "",
                "location_category": "",
                "caption": "",
                "age": "",
                "sex": sex,
                "acr_codes": "",
                "figure_part": "",
            }
        )

    return {
        "uid": patient_id,
        "title": f"Synthea case — {first} {last}".strip(),
        "history": history,
        "exam": exam,
        "findings": findings,
        "differential_diagnosis": "",
        "diagnosis": diagnosis,
        "diagnosis_by": "",
        "treatment": "",
        "discussion": "",
        "topic_title": "",
        "acr_code": "",
        "category": "",
        "disease_discussion": "",
        "keywords": "",
        "ct_image_ids": [],
        "mri_image_ids": [],
        "images": images,
        "llm_prompt": llm_prompt,
        "has_history": bool(history),
        "has_findings": bool(findings),
        "has_diagnosis": bool(diagnosis),
        "has_images": bool(images),
        "is_complete": bool(history and diagnosis and images),
        "dataset_source": "Synthea coherent zip",
    }


def build_database(zip_path: Path, ids_out: Path, dataset_out: Path, max_rows: int) -> tuple[int, int]:
    with ZipFile(zip_path) as zf:
        tables = _load_tables(zf)
        dicom_by_patient = _extract_patient_ids_with_dicom(zf)

    patients = tables.get("patients", [])
    patient_rows_by_id = {
        (row.get("Id") or "").strip().lower(): row
        for row in patients
        if (row.get("Id") or "").strip()
    }

    image_patient_ids = sorted(dicom_by_patient.keys())

    conditions_by_patient = _group_by_patient(tables.get("conditions", []))
    meds_by_patient = _group_by_patient(tables.get("medications", []))
    encounters_by_patient = _group_by_patient(tables.get("encounters", []))
    observations_by_patient = _group_by_patient(tables.get("observations", []))
    procedures_by_patient = _group_by_patient(tables.get("procedures", []))
    imaging_by_patient = _group_by_patient(tables.get("imaging_studies", []))

    ids_out.parent.mkdir(parents=True, exist_ok=True)
    dataset_out.parent.mkdir(parents=True, exist_ok=True)

    with ids_out.open("w", encoding="utf-8") as fh:
        json.dump(image_patient_ids, fh, indent=2)

    records: list[dict[str, Any]] = []
    for patient_id in image_patient_ids:
        patient_row = patient_rows_by_id.get(patient_id)
        if not patient_row:
            continue
        record = _build_record(
            patient_id=patient_id,
            patient_row=patient_row,
            dicom_members=dicom_by_patient.get(patient_id, []),
            conditions_by_patient=conditions_by_patient,
            meds_by_patient=meds_by_patient,
            encounters_by_patient=encounters_by_patient,
            observations_by_patient=observations_by_patient,
            procedures_by_patient=procedures_by_patient,
            imaging_by_patient=imaging_by_patient,
            max_rows=max_rows,
        )
        records.append(record)

    records.sort(key=lambda r: str(r.get("uid", "")))
    with dataset_out.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(image_patient_ids), len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Synthetic coherent patient database for Streamlit from ZIP."
    )
    parser.add_argument("--zip", dest="zip_path", default=str(DEFAULT_ZIP_PATH))
    parser.add_argument("--ids-out", default=str(DEFAULT_IDS_OUT))
    parser.add_argument("--dataset-out", default=str(DEFAULT_DATASET_OUT))
    parser.add_argument("--max-rows", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip_path).resolve()
    ids_out = Path(args.ids_out).resolve()
    dataset_out = Path(args.dataset_out).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    ids_count, records_count = build_database(
        zip_path=zip_path,
        ids_out=ids_out,
        dataset_out=dataset_out,
        max_rows=max(1, args.max_rows),
    )
    print(f"Image-linked patient IDs: {ids_count}")
    print(f"Records written: {records_count}")
    print(f"IDs file: {ids_out}")
    print(f"Dataset file: {dataset_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
