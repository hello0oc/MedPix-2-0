#!/usr/bin/env python3
from __future__ import annotations

import json
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medgemma_benchmark.run_medgemma_benchmark import (
    build_ehr_text,
    collect_image_paths,
    dicom_to_png_bytes,
    endpoint_chat_completion,
    image_to_data_url,
    load_jsonl,
    parse_diagnosis,
    resolve_image_path,
    sanitize_text,
    truncate_text,
)

DATASET_PATH = REPO_ROOT / "patient-ehr-image-dataset/full_dataset.jsonl"
PMC_DATASET_PATH = REPO_ROOT / "PMCpatient/PMC-Patients-sample-1000.csv"
SYNTHETIC_DATASET_PATH = REPO_ROOT / "Synthetic/synthetic_ehr_image_dataset.jsonl"
NSCLC_DATASET_PATH = REPO_ROOT / "nsclc-dataset/nsclc_dataset.jsonl"
WORKSPACE_ROOT = REPO_ROOT
PRESET_ENDPOINT_URL = "https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud"
PRESET_MODEL_ID = "google/medgemma-1-5-4b-it-hae"
TOKEN_ENV_VAR = "HF_TOKEN"
GEMINI_MODEL_ID = "gemini-2.5-pro"
GEMINI_KEY_NAME = "GEMINI_API_KEY"
# Gemini 2.5 Pro is a *thinking* model — thinking tokens consume max_output_tokens.
# We keep a separate budget so thinking doesn't starve the actual output.
GEMINI_THINKING_BUDGET = 2048
GEMINI_MIN_OUTPUT_TOKENS = 1024
HF_TOKEN_CANDIDATE_KEYS = (TOKEN_ENV_VAR, "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_API_KEY")
GEMINI_CANDIDATE_KEYS = (GEMINI_KEY_NAME, "GOOGLE_API_KEY", "GENAI_API_KEY")
LLM_MEDGEMMA = "MedGemma (HF Endpoint)"
LLM_GEMINI = "Gemini 2.5 Pro"


# ── Prompts (model-specific) ───────────────────────────────────────────────

# MedGemma: a medical-domain model — push it to use clinical vocabulary and
# produce structured, evidence-backed output.
MEDGEMMA_EXPLAIN_PROMPT = (
    "You are MedGemma, a medical vision-language model trained on clinical data. "
    "Your task: explain this clinical case to a patient or non-medical reader using simple, empathetic language. "
    "Base your explanation on ALL of the following sources of evidence, weighting them together:\n"
    "  1. The patient's EHR text (history and exam findings only — imaging/radiology reports are intentionally "
    "     withheld) provided in the user message.\n"
    "  2. Any attached medical images — you MUST independently examine and interpret these images yourself. "
    "     Do NOT rely on any pre-written radiology or imaging report; describe only what you directly observe.\n"
    "  3. If a prior AI diagnostic result is present at the end of the system prompt, you MUST reference it "
    "     explicitly: tell the patient what the diagnosis means in plain language and why the clinical findings "
    "     support it. Do not contradict the diagnostic result.\n\n"
    "Return ONLY valid JSON — no markdown, no prose outside the JSON — with exactly these keys:\n"
    "  plain_summary    : 2-3 sentence plain-language overview of what is happening, grounded in the EHR facts\n"
    "  image_findings   : what YOU directly observe in the medical images, described without jargon "
    "(empty string if no images were provided)\n"
    "  image_conclusion : what those image findings mean for the patient (empty string if no images)\n"
    "  next_steps       : what typically happens next (tests, referrals, treatment)\n\n"
    "CRITICAL: output exactly ONE JSON object and then STOP. "
    "Do NOT generate any additional clinical cases, patient histories, example reports, or any text after the closing brace."
)

MEDGEMMA_DIAGNOSIS_PROMPT = (
    "You are MedGemma, a specialist clinical diagnostic assistant. "
    "Analyse the provided patient history, examination findings, and any attached images. "
    "IMPORTANT: The EHR text provided does NOT include pre-written imaging or radiology reports — "
    "you MUST independently examine and interpret any attached images yourself. "
    "Do not assume or invent imaging findings that are not visible in the provided images.\n\n"
    "Apply systematic clinical reasoning: consider the presenting complaint, risk factors, "
    "examination signs, and your own direct interpretation of imaging characteristics to reach a single best diagnosis. "
    "Return ONLY valid JSON — no markdown, no prose outside the JSON — with exactly these keys:\n"
    "  diagnosis         : single most likely diagnosis (concise clinical term)\n"
    "  confidence        : one of 'High', 'Moderate', or 'Low'\n"
    "  rationale         : 3-5 sentence evidence-based justification citing specific findings from the case "
    "and your own image interpretation\n"
    "  key_findings      : comma-separated list of the top 3-5 findings that support this diagnosis, "
    "including image observations\n"
    "  differential      : 2-3 alternative diagnoses to consider, as a comma-separated list\n\n"
    "CRITICAL: output exactly ONE JSON object and then STOP. "
    "Do NOT generate any additional clinical cases, patient histories, example reports, or any text after the closing brace."
)

# MedGemma text-only variants — used when 'Include linked images' is unchecked.
#
# MedGemma 1.5 4B is a small fine-tuned medical VLM.  It was trained on paired
# image-text data, so any mention of "attached images" or "examine the scan" in
# the prompt body primes it to produce imaging observations even when no images
# are actually present.  Unlike Gemini 2.5 Pro (a large reasoning model), it
# cannot reliably override that priming via a meta-instruction prepended at
# inference time.  The correct solution is to give it a prompt that contains no
# image-related language whatsoever, so the conflicting activation never arises.
MEDGEMMA_EXPLAIN_PROMPT_TEXT_ONLY = (
    "You are MedGemma, a medical AI assistant trained on clinical data. "
    "Your task: explain this clinical case to a patient or non-medical reader using simple, empathetic language. "
    "Base your explanation solely on the patient's EHR text (history and examination findings) "
    "provided in the user message. No medical images are available for this request.\n"
    "If a prior AI diagnostic result is present at the end of the system prompt, you MUST reference it "
    "explicitly: tell the patient what the diagnosis means in plain language and why the clinical findings "
    "support it. Do not contradict the diagnostic result.\n\n"
    "Return ONLY valid JSON — no markdown, no prose outside the JSON — with exactly these keys:\n"
    "  plain_summary    : 2-3 sentence plain-language overview of what is happening, grounded in the EHR facts\n"
    '  image_findings   : "" (no images provided — this field must be an empty string)\n'
    '  image_conclusion : "" (no images provided — this field must be an empty string)\n'
    "  next_steps       : what typically happens next (tests, referrals, treatment)\n\n"
    "CRITICAL: output exactly ONE JSON object and then STOP. "
    "Do NOT generate any additional clinical cases, patient histories, example reports, or any text after the closing brace."
)

MEDGEMMA_DIAGNOSIS_PROMPT_TEXT_ONLY = (
    "You are MedGemma, a specialist clinical diagnostic assistant. "
    "No medical images are available for this request — base your analysis entirely on the "
    "patient history and examination findings provided in the EHR text.\n\n"
    "Apply systematic clinical reasoning: consider the presenting complaint, risk factors, "
    "and examination signs to reach a single best diagnosis. "
    "Return ONLY valid JSON — no markdown, no prose outside the JSON — with exactly these keys:\n"
    "  diagnosis         : single most likely diagnosis (concise clinical term)\n"
    "  confidence        : one of 'High', 'Moderate', or 'Low'\n"
    "  rationale         : 3-5 sentence evidence-based justification citing specific findings "
    "from the history and examination only\n"
    "  key_findings      : comma-separated list of the top 3-5 history/exam findings that support this diagnosis\n"
    "  differential      : 2-3 alternative diagnoses to consider, as a comma-separated list\n\n"
    "CRITICAL: output exactly ONE JSON object and then STOP. "
    "Do NOT generate any additional clinical cases, patient histories, example reports, or any text after the closing brace."
)

# Gemini 2.5 Pro: thinking model — be crystal-clear about the desired JSON schema
# so the model's visible output (after internal reasoning) is clean JSON.
GEMINI_EXPLAIN_PROMPT = (
    "You are a medical explainer for patients and non-medical readers. "
    "Use plain, friendly, empathetic language. Avoid jargon.\n\n"
    "Base your explanation on ALL of the following, weighting them together:\n"
    "  1. The patient's EHR text (history and exam findings only — imaging/radiology reports are intentionally "
    "     withheld) in the user message.\n"
    "  2. Any attached medical images — you MUST independently examine and interpret these images yourself. "
    "     Do NOT rely on any pre-written radiology or imaging report; describe only what you directly observe "
    "     in the images.\n"
    "  3. If a prior AI diagnostic result is appended to this system prompt, you MUST reference it "
    "     explicitly: explain to the patient what the diagnosis means in plain language and why the "
    "     available clinical facts support it. Do not contradict the diagnostic result.\n\n"
    "Return ONLY a single JSON object (no markdown, no code fences, no extra text) "
    "with exactly these four string keys:\n"
    '  "plain_summary"    – 2-3 sentence overview grounded in the EHR facts and (if present) the diagnostic result\n'
    '  "image_findings"   – what YOU directly observe in the images, described without jargon '
    '(empty string "" if no images were provided)\n'
    '  "image_conclusion" – what those findings mean for the patient (empty string "" if none)\n'
    '  "next_steps"       – what typically happens next (tests, referrals, treatment)\n'
)

GEMINI_DIAGNOSIS_PROMPT = (
    "You are a specialist clinical diagnostic assistant. "
    "Analyse all provided clinical history, examination findings, "
    "and any attached images.\n\n"
    "IMPORTANT: The EHR text provided does NOT include pre-written imaging or radiology reports — "
    "you MUST independently examine and interpret any attached images yourself. "
    "Do not assume or invent imaging findings that are not visible in the provided images.\n\n"
    "Apply systematic clinical reasoning: consider the presenting complaint, risk factors, "
    "examination signs, and your own direct interpretation of imaging characteristics.\n\n"
    "Return ONLY a single JSON object (no markdown, no code fences, no extra text) "
    "with exactly these five string keys:\n"
    '  "diagnosis"    – single most likely diagnosis (concise clinical term)\n'
    '  "confidence"   – exactly one of: "High", "Moderate", "Low"\n'
    '  "rationale"    – 3-5 sentence evidence-based justification citing specific case findings '
    'and your own image interpretation\n'
    '  "key_findings" – top 3-5 supporting findings including image observations, comma-separated\n'
    '  "differential" – 2-3 alternative diagnoses to consider, comma-separated\n'
)

# ── Clinical Trial Profile prompts ────────────────────────────────────────
#
# These prompts produce a structured patient profile intended for downstream
# clinical trial eligibility matching.  Two variants are provided:
#   TRIAL_PROFILE_PROMPT          — used when medical images are supplied
#   TRIAL_PROFILE_PROMPT_TEXT_ONLY — used when no images are available
#
# Both variants share the same output schema.  The key difference is the
# sourcing rule for the "imaging_findings" key_fact (see inline comments).
#
# NO-HALLUCINATION RULE (enforced in both variants and by post-processing):
#   Every key_fact "value" MUST be null when no supporting evidence exists in
#   the provided sources.  Filler strings such as "unknown", "N/A", or "" are
#   not acceptable — they will be coerced to null by the post-processor.
_TRIAL_PROFILE_SCHEMA = (
    'Return ONLY a single valid JSON object — no markdown, no code fences, no prose outside the object.\n'
    'The object must have exactly these four top-level keys:\n\n'
    '  "topic_id"    : string — use the value injected after "Use topic_id:" in this prompt\n'
    '  "profile_text": string — a structured free-text summary using exactly these six Markdown '
    'headings in order: "## Clinical History", "## Physical Exam", "## Imaging Findings", '
    '"## Assessment & Plan", "## Demographics", "## Missing Info". '
    'Under each heading write a concise paragraph drawn only from the provided input. '
    'Under "## Missing Info" list any fields that could not be populated from the input.\n'
    '  "key_facts"   : array of objects, each with exactly these five keys:\n'
    '      "field"        : string — field name\n'
    '      "value"        : string | object | array | null — extracted value; MUST be null if not evidenced\n'
    '      "evidence_span": string | null — exact substring from the input that supports the value, or null\n'
    '      "required"     : boolean\n'
    '      "notes"        : string | null — metadata (e.g. imaging_source)\n'
    '    Required key_fact fields (include all of these, plus any additional ones you can extract):\n'
    '      "primary_diagnosis" (required: true)\n'
    '      "demographics"      (required: true) — value should be {"age": "...", "sex": "..."}\n'
    '      "imaging_findings"  (required: true) — see sourcing rule below\n'
    '      "key_findings"      (required: true) — array of strings\n'
    '      "missing_info"      (required: false) — array of field names that lack evidence\n'
    '  "ambiguities" : array of strings — anything in the input that is contradictory or unclear\n\n'
    'STRICT NO-HALLUCINATION RULE: For every key_fact field, if you cannot find supporting evidence '
    'in the provided sources, you MUST set "value" to null. Do NOT infer, estimate, or fabricate values. '
    'Do NOT use placeholder strings such as "unknown", "N/A", "not provided", or empty string "". '
    'Add the field name to the "missing_info" key_fact value array whenever a required field is null.\n\n'
    'CRITICAL: output exactly ONE JSON object and then STOP. '
    'Do NOT generate any additional text, cases, or commentary after the closing brace.'
)

TRIAL_PROFILE_PROMPT = (
    "You are a clinical data specialist preparing a structured patient profile for clinical trial eligibility matching. "
    "You have been provided with the patient's EHR text (clinical history and examination findings) "
    "and one or more medical images.\n\n"
    "IMAGING FINDINGS SOURCING RULE (images provided):\n"
    "  - You MUST derive the 'imaging_findings' key_fact primarily from your DIRECT ANALYSIS of the "
    "supplied medical images. Examine the images yourself; do not rely on any pre-written radiology "
    "report or imaging description in the EHR text.\n"
    "  - Only supplement with EHR-sourced language where the images are genuinely ambiguous.\n"
    "  - Set the 'notes' field of the imaging_findings key_fact to exactly: "
    '"imaging_source: direct_image_analysis"\n'
    "  - If the images are present but uninterpretable, fall back to EHR text and set notes to: "
    '"imaging_source: ehr_extracted"\n\n'
    + _TRIAL_PROFILE_SCHEMA
)

TRIAL_PROFILE_PROMPT_TEXT_ONLY = (
    "You are a clinical data specialist preparing a structured patient profile for clinical trial eligibility matching. "
    "You have been provided with the patient's EHR text only (clinical history and examination findings). "
    "No medical images are available for this request.\n\n"
    "IMAGING FINDINGS SOURCING RULE (no images provided):\n"
    "  - You MUST extract the 'imaging_findings' key_fact solely from imaging-related language "
    "present in the EHR text — for example, CT/MRI/X-ray result descriptions, radiology report "
    "excerpts, or mentions of scan findings.\n"
    "  - Set the 'notes' field of the imaging_findings key_fact to exactly: "
    '"imaging_source: ehr_extracted"\n'
    "  - If no imaging information can be found anywhere in the EHR text, set 'value' to null "
    'and add "imaging_findings" to the missing_info key_fact value array.\n'
    "  - Do NOT infer, fabricate, or hallucinate any imaging findings based on the diagnosis or "
    "clinical context alone.\n\n"
    + _TRIAL_PROFILE_SCHEMA
)


def history_richness_score(case: Dict[str, Any]) -> int:
    """Score a case by the length of the patient history text only.

    Only the 'history' field is used for ranking; image findings, exam notes,
    and discussion are intentionally excluded.
    Skips N/A and placeholder values.
    """
    v = sanitize_text(case.get("history", "")).strip()
    if not v or v.lower() in {"n/a", "none", "na", "-"}:
        return 0
    return len(v)


def parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def parse_markdown_fields(text: str) -> Dict[str, str]:
    """Extract **Label:** Value pairs from markdown-style model output.

    Prefers 'Final ...' variants (common in Gemini chain-of-thought responses)
    over earlier 'Revised ...' or plain variants.
    """
    result: Dict[str, str] = {}
    # Collect all **Key:** Value pairs (value = text until next bold heading or end)
    for m in re.finditer(
        r"\*\*([^*\n]+?):\*\*\s*([^\n]+)", text
    ):
        raw_key = m.group(1).strip()
        value = m.group(2).strip().strip("*").strip()
        norm_key = raw_key.lower().replace(" ", "_")
        # Only overwrite a key if the new one is 'final_*' (more authoritative)
        if norm_key not in result or "final" in norm_key:
            result[norm_key] = value
    return result


def build_grounded_explain_prompt(base_prompt: str, diagnosis_json: Dict[str, Any]) -> str:
    """Enrich the plain-language explain system prompt with a prior LLM diagnostic result.

    When a diagnosis has already been produced for this case, we inject the key
    clinical facts into the system prompt so the non-medical explanation is
    consistent with and grounded in the diagnostic outcome.
    """
    if not diagnosis_json:
        return base_prompt

    diag = sanitize_text(diagnosis_json.get("diagnosis", ""))
    confidence = sanitize_text(diagnosis_json.get("confidence", ""))
    rationale = sanitize_text(diagnosis_json.get("rationale", ""))
    key_findings = sanitize_text(diagnosis_json.get("key_findings", ""))

    if not diag:
        return base_prompt

    parts = ["\n\n── Prior AI Diagnostic Result (use this to ground your explanation) ──"]
    parts.append(f"Diagnosis: {diag}" + (f" (Confidence: {confidence})" if confidence else ""))
    if rationale:
        parts.append(f"Rationale: {rationale}")
    if key_findings:
        parts.append(f"Key supporting findings: {key_findings}")
    parts.append(
        "\nYour plain-language explanation MUST be consistent with the above diagnosis. "
        "Translate these clinical facts into terms a non-medical person can easily understand. "
        "Explain WHY the findings led to this diagnosis without using technical jargon."
    )
    return base_prompt + "\n".join(parts)


def build_trial_profile_prompt(base_prompt: str, uid: str) -> str:
    """Inject the patient topic_id into a trial profile system prompt.

    Appends a small context block at the end of the prompt so the model
    echoes the correct topic_id in its JSON output.  Post-processing in
    validate_and_coerce_trial_profile hard-forces the value regardless, but
    having it in the prompt reduces token correction overhead.
    """
    return base_prompt + f"\n\nUse topic_id: {uid.lower()}"


# Filler strings that models commonly emit instead of null when a value is absent.
_NULL_FILLERS = frozenset({
    "", "unknown", "n/a", "na", "not provided", "not available",
    "none", "null", "unspecified", "not stated", "not documented",
})


def validate_and_coerce_trial_profile(
    result: Dict[str, Any],
    uid: str,
    used_images: bool,
) -> Dict[str, Any]:
    """Post-process a raw trial profile dict returned by the LLM.

    Guarantees:
    1. topic_id is always uid.lower() regardless of model output.
    2. Filler strings ("unknown", "N/A", "", etc.) in key_fact values are
       replaced with null so downstream consumers can rely on null meaning
       "genuinely absent".
    3. The imaging_findings key_fact always exists and carries the correct
       imaging_source annotation in its notes field.
    4. Every null required key_fact field is represented in missing_info.
    """
    # ── 1. Force topic_id ───────────────────────────────────────────────────
    result["topic_id"] = uid.lower()

    # ── 2. Ensure top-level lists exist ────────────────────────────────────
    if not isinstance(result.get("key_facts"), list):
        result["key_facts"] = []
    if not isinstance(result.get("ambiguities"), list):
        result["ambiguities"] = []
    if not isinstance(result.get("profile_text"), str):
        result["profile_text"] = ""

    key_facts: List[Dict[str, Any]] = result["key_facts"]

    # ── 3. Null-coercion pass ────────────────────────────────────────────────
    # Collect/create missing_info fact so we can append to it.
    missing_info_fact: Dict[str, Any] | None = next(
        (kf for kf in key_facts if kf.get("field") == "missing_info"), None
    )
    if missing_info_fact is None:
        missing_info_fact = {
            "field": "missing_info",
            "value": [],
            "evidence_span": None,
            "required": False,
            "notes": None,
        }
        key_facts.append(missing_info_fact)
    if not isinstance(missing_info_fact.get("value"), list):
        missing_info_fact["value"] = (
            [] if not missing_info_fact.get("value")
            else [str(missing_info_fact["value"])]
        )

    for kf in key_facts:
        if kf.get("field") == "missing_info":
            continue  # handled separately
        val = kf.get("value")
        # Coerce filler scalars to null
        if isinstance(val, str) and val.strip().lower() in _NULL_FILLERS:
            kf["value"] = None
            val = None
        # Coerce filler list items; treat an all-filler list as null
        if isinstance(val, list):
            cleaned = [
                item for item in val
                if not (isinstance(item, str) and item.strip().lower() in _NULL_FILLERS)
            ]
            kf["value"] = cleaned if cleaned else None
            val = kf["value"]
        # Register null required fields in missing_info
        if kf.get("required") and val is None:
            field_name = kf.get("field", "")
            if field_name and field_name not in missing_info_fact["value"]:
                missing_info_fact["value"].append(field_name)

    # ── 4. Guarantee imaging_findings key_fact ──────────────────────────────
    imaging_source = "direct_image_analysis" if used_images else "ehr_extracted"
    imaging_fact: Dict[str, Any] | None = next(
        (kf for kf in key_facts if kf.get("field") == "imaging_findings"), None
    )
    if imaging_fact is None:
        imaging_fact = {
            "field": "imaging_findings",
            "value": None,
            "evidence_span": None,
            "required": True,
            "notes": f"imaging_source: {imaging_source}",
        }
        key_facts.insert(0, imaging_fact)
        if "imaging_findings" not in missing_info_fact["value"]:
            missing_info_fact["value"].append("imaging_findings")
    else:
        # Ensure notes always carry the imaging_source annotation
        existing_notes = imaging_fact.get("notes") or ""
        if "imaging_source" not in existing_notes:
            imaging_fact["notes"] = (
                f"imaging_source: {imaging_source}"
                + (f"; {existing_notes}" if existing_notes else "")
            )

    return result


def _is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda" in message


def medgemma_token_budget(
    requested_tokens: int,
    ehr_text: str,
    image_count: int,
    use_images: bool,
) -> int:
    """Derive a performance-safe output-token budget for MedGemma.

    Strategy:
    - Allow large requested limits for lightweight prompts.
    - Auto-cap for heavier multimodal prompts to keep latency/stability healthy.
    """
    budget = int(requested_tokens)

    # Practical upper guardrail for 4B endpoint while still "maximizing" user limit.
    budget = min(budget, 1536)

    ehr_len = len(ehr_text or "")
    active_images = image_count if use_images else 0

    # Input-heavy requests get progressively tighter output caps.
    if active_images >= 3:
        budget = min(budget, 896)
    elif active_images == 2:
        budget = min(budget, 1024)

    if ehr_len > 1000:
        budget = min(budget, 1024)
    if ehr_len > 2000:
        budget = min(budget, 768)
    if ehr_len > 3000:
        budget = min(budget, 640)

    return max(64, budget)


@st.cache_data(show_spinner=False)
def load_cases(path_str: str) -> List[Dict[str, Any]]:
    rows = load_jsonl(Path(path_str))
    rows = [row for row in rows if row.get("uid")]
    rows.sort(key=lambda row: str(row.get("uid")))
    return rows


def _normalize_pmc_row(row: Dict[str, Any]) -> Dict[str, Any]:
    uid = sanitize_text(row.get("patient_uid") or row.get("patient_id") or "")
    title = sanitize_text(row.get("title") or "PMC clinical case")
    history = sanitize_text(row.get("patient") or "")
    age = sanitize_text(row.get("age") or "")
    gender = sanitize_text(row.get("gender") or "")
    pmid = sanitize_text(row.get("PMID") or "")
    file_path = sanitize_text(row.get("file_path") or "")
    relevant_articles = sanitize_text(row.get("relevant_articles") or "")
    similar_patients = sanitize_text(row.get("similar_patients") or "")

    exam_parts: List[str] = []
    if age:
        exam_parts.append(f"Age: {age}")
    if gender:
        exam_parts.append(f"Gender: {gender}")
    if pmid:
        exam_parts.append(f"PMID: {pmid}")
    if file_path:
        exam_parts.append(f"Source file: {file_path}")

    findings_parts: List[str] = []
    if relevant_articles:
        findings_parts.append(f"Related articles: {relevant_articles}")
    if similar_patients:
        findings_parts.append(f"Similar patients: {similar_patients}")

    return {
        "uid": uid,
        "title": title,
        "history": history,
        "exam": "\n".join(exam_parts),
        "findings": "\n".join(findings_parts),
        "diagnosis": "",
        "discussion": "",
        "images": [],
        "ct_image_ids": [],
        "mri_image_ids": [],
        "dataset_source": "PMC-Patients CSV",
    }


@st.cache_data(show_spinner=False)
def load_pmc_cases(path_str: str) -> List[Dict[str, Any]]:
    path = Path(path_str)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = _normalize_pmc_row(row)
            if case.get("uid"):
                rows.append(case)
    rows.sort(key=lambda row: str(row.get("uid")))
    return rows


def get_resolved_images(case: Dict[str, Any]) -> List[Path]:
    image_paths: List[Path] = []
    for raw_path in collect_image_paths(case):
        resolved = resolve_image_path(WORKSPACE_ROOT, raw_path)
        if resolved and resolved.exists():
            image_paths.append(resolved)
    return image_paths


def preview_payload(path: Path) -> Any:
    if path.suffix.lower() == ".dcm":
        return dicom_to_png_bytes(path)
    return str(path)


def build_user_content(
    ehr_text: str,
    image_paths: List[Path],
    use_images: bool,
    max_images: int,
) -> Any:
    if not use_images or max_images <= 0 or not image_paths:
        return ehr_text

    blocks: List[Dict[str, Any]] = [{"type": "text", "text": ehr_text}]
    for path in image_paths[:max_images]:
        try:
            image_data_url = image_to_data_url(path)
        except Exception:
            continue
        blocks.append({"type": "image_url", "image_url": {"url": image_data_url}})

    return blocks


def call_medgemma(
    endpoint_url: str,
    token: str,
    model_id: str,
    system_prompt: str,
    ehr_text: str,
    image_paths: List[Path],
    use_images: bool,
    max_images: int,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> str:
    # When images are disabled, force the effective count to 0 so the OOM
    # retry logic doesn't try to reduce an image budget that was never used.
    effective_images = max_images if use_images else 0
    effective_tokens = max_tokens
    effective_ehr = ehr_text

    for attempt in range(4):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_user_content(
                    ehr_text=effective_ehr,
                    image_paths=image_paths,
                    use_images=use_images,
                    max_images=effective_images,
                ),
            },
        ]

        try:
            return endpoint_chat_completion(
                endpoint_url=endpoint_url,
                token=token,
                model_id=model_id,
                messages=messages,
                temperature=0.1,
                max_tokens=effective_tokens,
                timeout=timeout,
                retries=retries,
                retry_backoff_sec=2.0,
            )
        except Exception as exc:
            if not _is_oom_error(exc) or attempt == 3:
                raise

            effective_tokens = max(32, int(effective_tokens * 0.7))
            effective_ehr = truncate_text(effective_ehr, max(120, int(len(effective_ehr) * 0.7)))
            if effective_images > 0:
                effective_images -= 1

    raise RuntimeError("Inference failed after adaptive retries.")


def _extract_gemini_text(response: Any) -> str:
    """Robustly extract non-thought text from a Gemini response.

    Gemini 2.5 Pro is a *thinking* model.  `response.text` concatenates only the
    non-thought parts and may return ``None`` when the entire output budget was
    consumed by internal reasoning.  This helper falls back to iterating the
    part list directly and, as a last resort, returns the raw thought text so
    the caller always gets a string.
    """
    # 1) Fast path – SDK accessor
    if response.text:
        return response.text

    # 2) Manual iteration – find any non-thought text part
    parts_text: List[str] = []
    thought_text: List[str] = []
    for candidate in (response.candidates or []):
        for part in (candidate.content.parts if candidate.content else []):
            if getattr(part, "thought", None):
                if part.text:
                    thought_text.append(part.text)
            elif part.text:
                parts_text.append(part.text)

    if parts_text:
        return "\n".join(parts_text)

    # 3) If finish_reason is MAX_TOKENS we got no real output — report clearly
    if response.candidates:
        fr = response.candidates[0].finish_reason
        if fr and "MAX_TOKENS" in str(fr):
            raise RuntimeError(
                "Gemini ran out of output tokens (all budget consumed by internal reasoning). "
                "Increase 'Max output tokens' in the sidebar or reduce input length."
            )

    # 4) Absolute fallback – return thought content (rare)
    if thought_text:
        return "\n".join(thought_text)

    raise RuntimeError("Gemini returned an empty response — check safety filters or API key.")


def call_gemini(
    api_key: str,
    model_name: str,
    system_prompt: str,
    ehr_text: str,
    image_paths: List[Path],
    use_images: bool,
    max_images: int,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> str:
    try:
        from google import genai as ggenai  # type: ignore
        from google.genai import types as gtypes  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed. Run: pip install google-genai"
        ) from exc

    client = ggenai.Client(api_key=api_key)

    # Build content parts: optional images followed by the EHR text
    contents: List[Any] = []
    if use_images and max_images > 0 and image_paths:
        try:
            from PIL import Image as PILImage  # type: ignore
            for path in image_paths[:max_images]:
                try:
                    contents.append(PILImage.open(path).copy())
                except Exception:
                    pass
        except ImportError:
            pass  # Pillow not available — text-only
    contents.append(ehr_text)

    # Gemini 2.5 Pro is a *thinking* model: internal reasoning ("thoughts")
    # consume the max_output_tokens budget.  We set a separate thinking_budget
    # so the visible JSON output is never starved of tokens.
    effective_output_tokens = max(max_tokens, GEMINI_MIN_OUTPUT_TOKENS)

    cfg = gtypes.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=effective_output_tokens + GEMINI_THINKING_BUDGET,
        temperature=0.1,
        response_mime_type="application/json",
        thinking_config=gtypes.ThinkingConfig(
            thinking_budget=GEMINI_THINKING_BUDGET,
        ),
        http_options=gtypes.HttpOptions(timeout=timeout * 1000),  # ms
    )

    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max(1, retries + 1)):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=cfg,
            )
            if not response.candidates:
                raise RuntimeError("Gemini returned no candidates — check safety filters or API key.")
            return _extract_gemini_text(response)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                import time
                time.sleep(min(30, 2 ** attempt))
    raise last_exc


def init_state() -> None:
    pass  # State is now fully per-UID and per-model; no global keys needed.


def _read_secret_value(key: str) -> str:
    try:
        value = st.secrets.get(key)
    except Exception:
        return ""
    if value is None:
        return ""
    return str(value).strip()


def resolve_credential(candidate_keys: Tuple[str, ...]) -> Tuple[str, str, str]:
    for key in candidate_keys:
        value = _read_secret_value(key)
        if value:
            return value, "secrets.toml", key

    for key in candidate_keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value, "environment", key

    return "", "", ""


def main() -> None:
    st.set_page_config(
        page_title="MedGUI — Medical AI Assistant",
        page_icon=":stethoscope:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Compact CSS tweaks ──────────────────────────────────────────────────
    st.markdown("""
    <style>
    .block-container { padding-top: 0.8rem; padding-bottom: 0.5rem; }
    section[data-testid="stSidebar"] { min-width: 270px; max-width: 320px; }
    .stButton > button { border-radius: 8px; font-weight: 600; letter-spacing: 0.02em; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; margin-bottom: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; padding: 0 18px; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)

    # Resolve credentials once (supports multiple common key aliases).
    hf_token, hf_cred_source, hf_cred_key = resolve_credential(HF_TOKEN_CANDIDATE_KEYS)
    gemini_api_key, gemini_cred_source, gemini_cred_key = resolve_credential(GEMINI_CANDIDATE_KEYS)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Settings")

        dataset_choice = st.radio(
            "Dataset Source",
            options=[
                "MedPix JSONL",
                "PMC-Patients CSV",
                "Synthea Coherent JSONL",
                "NSCLC Dataset",
            ],
            index=0,
            help="Choose which source data to load. All sources are normalized into the same downstream workflow.",
        )

        llm_choice = st.radio(
            "LLM Backend",
            options=[LLM_MEDGEMMA, LLM_GEMINI],
            index=0,
            help="Choose which AI model performs inference.",
        )

        if llm_choice == LLM_MEDGEMMA:
            endpoint_url = PRESET_ENDPOINT_URL
            model_id = PRESET_MODEL_ID
            active_credential = hf_token
            active_cred_source = hf_cred_source
            active_cred_key = hf_cred_key
            required_cred_keys = HF_TOKEN_CANDIDATE_KEYS
        else:
            endpoint_url = ""
            model_id = GEMINI_MODEL_ID
            active_credential = gemini_api_key
            active_cred_source = gemini_cred_source
            active_cred_key = gemini_cred_key
            required_cred_keys = GEMINI_CANDIDATE_KEYS

        st.caption(f"Model: `{model_id}`")
        if active_credential:
            st.caption(f"Credential: `{active_cred_key}` from {active_cred_source}.")
        else:
            st.caption(
                "Credential missing. Checked: "
                + ", ".join(f"`{k}`" for k in required_cred_keys)
                + " in `.streamlit/secrets.toml` and environment variables."
            )

        st.markdown("---")
        st.markdown("**Inference options**")
        sort_by_richness = st.checkbox("Sort patients by history richness", value=True,
                                       help="Orders the patient list richest history first.")
        use_images = st.checkbox("Include linked images", value=True)
        show_linked_images = st.checkbox("Preview images in viewer", value=True)
        max_tokens = st.slider(
            "Max output tokens",
            64, 4096, 1024, 64,
            help="For Gemini 2.5 Pro a separate thinking budget is added automatically.",
        )
        target_ehr_chars = st.slider("EHR text limit (chars)", 120, 3000, 800, 20)
        timeout = st.number_input("Timeout (sec)", 30, 300, 180, 10)
        retries = st.number_input("Retries", 0, 10, 3, 1)

    # ── Load dataset ────────────────────────────────────────────────────────
    if dataset_choice == "PMC-Patients CSV":
        active_dataset_path = PMC_DATASET_PATH
        cases = load_pmc_cases(str(PMC_DATASET_PATH))
    elif dataset_choice == "Synthea Coherent JSONL":
        active_dataset_path = SYNTHETIC_DATASET_PATH
        cases = load_cases(str(SYNTHETIC_DATASET_PATH))
    elif dataset_choice == "NSCLC Dataset":
        active_dataset_path = NSCLC_DATASET_PATH
        cases = load_cases(str(NSCLC_DATASET_PATH))
    else:
        active_dataset_path = DATASET_PATH
        cases = load_cases(str(DATASET_PATH))

    if not cases:
        st.error(f"No records found at: {active_dataset_path}")
        st.stop()

    # Pre-compute richness scores once (deterministic, cache-friendly)
    scored = sorted(
        cases,
        key=lambda c: history_richness_score(c),
        reverse=True,
    ) if sort_by_richness else cases

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("# MedGUI")
    st.caption(f"Active model: **{llm_choice}** — {model_id}")
    st.caption(f"Active dataset: **{dataset_choice}** ({active_dataset_path.name})")

    # ── Control bar: patient selector + action buttons ───────────────────────
    ctrl_uid, ctrl_explain, ctrl_diagnose, ctrl_trial = st.columns([3, 1.2, 1.2, 1.2])
    with ctrl_uid:
        # Build labels: "MPX1234  (score: 1420)" when sorted, plain UID otherwise
        if sort_by_richness:
            uid_options = [
                f"{c.get('uid')}  │  richness: {history_richness_score(c)}"
                for c in scored
            ]
        else:
            uid_options = [str(c.get("uid")) for c in scored]

        selected_label = st.selectbox(
            "Patient",
            options=uid_options,
            label_visibility="collapsed",
            placeholder="Select patient...",
        )
        # Extract plain UID regardless of label format
        selected_uid = selected_label.split("│")[0].strip()
    selected_case = next(c for c in cases if str(c.get("uid")) == selected_uid)

    # Resolve images early (needed by slider and inference)
    resolved_images = get_resolved_images(selected_case)
    _n_images = len(resolved_images)
    with st.sidebar:
        st.markdown("---")
        max_images = st.slider(
            f"Images for inference (available: {_n_images})",
            min_value=0,
            max_value=max(1, _n_images),
            value=min(1, _n_images),
            step=1,
            disabled=(_n_images == 0 or not use_images),
        )

    # Exclude 'findings' (pre-written imaging/radiology reports) so the LLM
    # must interpret the attached images independently rather than echoing
    # dataset-provided radiology conclusions.
    _case_no_findings = {k: v for k, v in selected_case.items() if k != "findings"}
    composed_ehr = truncate_text(build_ehr_text(_case_no_findings), target_ehr_chars)

    # When images are disabled, append an unambiguous note to the user-turn
    # text so the model cannot infer or hallucinate imaging findings from its
    # training knowledge.  This is the primary safeguard against the model
    # fabricating CT/MRI descriptions when no images are actually attached.
    if not use_images:
        composed_ehr = (
            composed_ehr
            + "\n\n[SYSTEM NOTE: No medical images are attached to this request. "
            "You MUST NOT describe, infer, or invent any imaging findings. "
            "Set image_findings and image_conclusion to empty strings.]"
        )
    effective_max_tokens = max_tokens
    if llm_choice == LLM_MEDGEMMA:
        effective_max_tokens = medgemma_token_budget(
            requested_tokens=max_tokens,
            ehr_text=composed_ehr,
            image_count=min(max_images, len(resolved_images)),
            use_images=use_images,
        )

    # Action buttons sit in the control bar
    with ctrl_explain:
        run_explain = st.button(
            "Explain (non-medical)",
            use_container_width=True,
            type="secondary",
            help="Generate a plain-language summary of this case.",
        )
    with ctrl_diagnose:
        run_diagnosis = st.button(
            "Diagnose",
            use_container_width=True,
            type="primary",
            help="Generate a clinical diagnostic result.",
        )
    with ctrl_trial:
        run_trial = st.button(
            "Trial Profile",
            use_container_width=True,
            type="secondary",
            help="Generate a structured clinical trial matching profile.",
        )

    if llm_choice == LLM_MEDGEMMA and effective_max_tokens < max_tokens:
        st.caption(
            f"MedGemma performance cap applied: using {effective_max_tokens} output tokens "
            f"(requested {max_tokens}) based on current context size."
        )

    if (run_explain or run_diagnosis or run_trial) and not active_credential:
        st.error(
            "Missing credential. Configure one of: "
            + ", ".join(f"`{k}`" for k in required_cred_keys)
            + " in `.streamlit/secrets.toml` or environment variables."
        )

    # ── Per-UID + per-model session cache ───────────────────────────────────
    # Include use_images in the key so that toggling the checkbox immediately
    # invalidates any result that was computed with a different image setting.
    _mk = "mg" if llm_choice == LLM_MEDGEMMA else "gf"
    _img_suffix = "img" if use_images else "noimg"
    _ek = f"explain_{_mk}_{_img_suffix}_{selected_uid}"
    _ee = f"explain_error_{_mk}_{_img_suffix}_{selected_uid}"
    _dk = f"diagnosis_{_mk}_{_img_suffix}_{selected_uid}"
    _de = f"diagnosis_error_{_mk}_{_img_suffix}_{selected_uid}"
    _tk = f"trial_{_mk}_{_img_suffix}_{selected_uid}"
    _te = f"trial_error_{_mk}_{_img_suffix}_{selected_uid}"

    def _run_inference(system_prompt: str) -> str:
        effective_system_prompt = system_prompt
        if not use_images:
            if llm_choice == LLM_MEDGEMMA:
                # MedGemma 1.5 4B is a small fine-tuned VLM: conflicting instructions
                # ("examine images" in the body vs "no images" override at the top)
                # are unreliable because fine-tuning on image-text pairs creates
                # strong activations that a meta-instruction cannot suppress.
                # We route to dedicated text-only prompt variants that contain no
                # image-related language at all, so the conflict never arises.
                if system_prompt == MEDGEMMA_DIAGNOSIS_PROMPT:
                    effective_system_prompt = MEDGEMMA_DIAGNOSIS_PROMPT_TEXT_ONLY
                elif system_prompt.startswith(MEDGEMMA_EXPLAIN_PROMPT):
                    # preserve any grounded-diagnosis suffix appended by build_grounded_explain_prompt
                    suffix = system_prompt[len(MEDGEMMA_EXPLAIN_PROMPT):]
                    effective_system_prompt = MEDGEMMA_EXPLAIN_PROMPT_TEXT_ONLY + suffix
                elif system_prompt.startswith(TRIAL_PROFILE_PROMPT_TEXT_ONLY):
                    # Already the correct text-only trial variant (includes topic_id suffix).
                    effective_system_prompt = system_prompt
                elif system_prompt.startswith(TRIAL_PROFILE_PROMPT):
                    # Switch from the images variant to the text-only variant, preserving
                    # any "Use topic_id: ..." suffix appended by build_trial_profile_prompt.
                    suffix = system_prompt[len(TRIAL_PROFILE_PROMPT):]
                    effective_system_prompt = TRIAL_PROFILE_PROMPT_TEXT_ONLY + suffix
                else:
                    effective_system_prompt = MEDGEMMA_EXPLAIN_PROMPT_TEXT_ONLY
            else:
                # Gemini 2.5 Pro is a large reasoning model — it reliably follows
                # a meta-level override prepended to the system prompt, so we use
                # that lighter-weight approach instead of a separate prompt variant.
                # For trial profile prompts that already contain explicit no-images
                # instructions, skip the redundant override prepend.
                if system_prompt.startswith(TRIAL_PROFILE_PROMPT_TEXT_ONLY) or system_prompt.startswith(TRIAL_PROFILE_PROMPT):
                    effective_system_prompt = system_prompt
                else:
                    effective_system_prompt = (
                        "IMPORTANT OVERRIDE: No medical images have been provided for this case. "
                        "Do NOT describe, infer, or fabricate any imaging findings based on your "
                        "training knowledge. Your response MUST set image_findings and "
                        "image_conclusion to empty strings, and must not reference any radiological "
                        "or imaging observations in any other field.\n\n"
                        + system_prompt
                    )
        if llm_choice == LLM_MEDGEMMA:
            return call_medgemma(
                endpoint_url=endpoint_url,
                token=active_credential,
                model_id=model_id,
                system_prompt=effective_system_prompt,
                ehr_text=composed_ehr,
                image_paths=resolved_images,
                use_images=use_images,
                max_images=max_images,
                max_tokens=effective_max_tokens,
                timeout=int(timeout),
                retries=int(retries),
            )
        else:
            return call_gemini(
                api_key=active_credential,
                model_name=model_id,
                system_prompt=effective_system_prompt,
                ehr_text=composed_ehr,
                image_paths=resolved_images,
                use_images=use_images,
                max_images=max_images,
                max_tokens=effective_max_tokens,
                timeout=int(timeout),
                retries=int(retries),
            )

    # Choose prompts based on the active model
    explain_prompt   = MEDGEMMA_EXPLAIN_PROMPT   if llm_choice == LLM_MEDGEMMA else GEMINI_EXPLAIN_PROMPT
    diagnosis_prompt = MEDGEMMA_DIAGNOSIS_PROMPT if llm_choice == LLM_MEDGEMMA else GEMINI_DIAGNOSIS_PROMPT

    if run_explain and active_credential:
        st.session_state[_ek] = ""
        st.session_state[_ee] = ""
        # Ground the explanation in any pre-existing diagnostic result for this case.
        prior_diagnosis_json = parse_json_object(st.session_state.get(_dk, ""))
        grounded_explain_prompt = build_grounded_explain_prompt(explain_prompt, prior_diagnosis_json)
        with st.spinner(f"[{llm_choice}] Generating explanation..."):
            try:
                st.session_state[_ek] = _run_inference(grounded_explain_prompt)
            except Exception as exc:
                st.session_state[_ee] = str(exc)

    if run_diagnosis and active_credential:
        st.session_state[_dk] = ""
        st.session_state[_de] = ""
        with st.spinner(f"[{llm_choice}] Generating diagnostic result..."):
            try:
                st.session_state[_dk] = _run_inference(diagnosis_prompt)
            except Exception as exc:
                st.session_state[_de] = str(exc)

    if run_trial and active_credential:
        st.session_state[_tk] = None
        st.session_state[_te] = ""
        # Select the correct prompt variant up-front so _run_inference carries
        # the right no-images version if images are disabled.
        _trial_base = (
            TRIAL_PROFILE_PROMPT
            if (use_images and max_images > 0 and resolved_images)
            else TRIAL_PROFILE_PROMPT_TEXT_ONLY
        )
        _trial_prompt = build_trial_profile_prompt(_trial_base, selected_uid)
        with st.spinner(f"[{llm_choice}] Generating clinical trial profile..."):
            try:
                _trial_raw = _run_inference(_trial_prompt)
                _trial_parsed = parse_json_object(_trial_raw)
                if _trial_parsed:
                    _trial_used_images = bool(
                        use_images and max_images > 0 and resolved_images
                    )
                    st.session_state[_tk] = validate_and_coerce_trial_profile(
                        _trial_parsed, selected_uid, _trial_used_images
                    )
                else:
                    st.session_state[_te] = (
                        "Could not parse a JSON object from the model response. "
                        "Raw output: " + _trial_raw[:500]
                    )
            except Exception as exc:
                st.session_state[_te] = str(exc)

    explain_output   = st.session_state.get(_ek, "")
    explain_error    = st.session_state.get(_ee, "")
    diagnosis_output = st.session_state.get(_dk, "")
    diagnosis_error  = st.session_state.get(_de, "")
    trial_result     = st.session_state.get(_tk, None)
    trial_error      = st.session_state.get(_te, "")

    # ── Unified 3-panel layout: EHR | AI Insights | Images ──────────────────
    # All three panels are always co-visible so the user can associate the
    # diagnostic result with the original records and images without switching tabs.
    ehr_col, ai_col, img_col = st.columns([1.05, 1.15, 0.95])

    # ────────────────────────────────────────────────────────────────────────
    # PANEL 1 — EHR Record
    # ────────────────────────────────────────────────────────────────────────
    with ehr_col:
        st.subheader("EHR Record")
        st.markdown(
            f"**Title:** {sanitize_text(selected_case.get('title', '')) or 'N/A'}"
        )
        dataset_dx = sanitize_text(selected_case.get("diagnosis", ""))
        if dataset_dx:
            st.info(f"Dataset diagnosis: {dataset_dx}")
        else:
            st.caption("Dataset diagnosis: not available")

        with st.expander("History", expanded=True):
            st.write(sanitize_text(selected_case.get("history", "")) or "Not available")
        with st.expander("Physical Exam", expanded=False):
            st.write(sanitize_text(selected_case.get("exam", "")) or "Not available")
        with st.expander("Imaging Findings (EHR record — withheld from LLM)", expanded=False):
            st.caption(
                "⚠️ These are the dataset's pre-written imaging findings. "
                "They are intentionally **not sent** to the LLM — the AI interprets the images directly. "
                "Compare the AI's image interpretation (in the AI Insights panel) against these records."
            )
            st.write(sanitize_text(selected_case.get("findings", "")) or "Not available")

    # ────────────────────────────────────────────────────────────────────────
    # PANEL 2 — AI Insights (Diagnostic Result + Grounded Explanation)
    # Rendered alongside the EHR and images so findings are immediately
    # associable with the source records.
    # ────────────────────────────────────────────────────────────────────────
    with ai_col:
        st.subheader("AI Insights")
        has_results = bool(explain_output or explain_error or diagnosis_output or diagnosis_error or trial_result or trial_error)

        if not has_results:
            st.info(
                "No AI results yet for this patient with the selected model. "
                "Use the **Diagnose** or **Explain** buttons above to run inference."
            )
        else:
            # ── Diagnostic Result ───────────────────────────────────────────
            st.markdown("##### Clinical Diagnostic Result")
            if diagnosis_error:
                st.error(f"Request failed: {diagnosis_error}")
            elif diagnosis_output:
                # 1) Try JSON
                diagnosis_json = parse_json_object(diagnosis_output)
                diagnosis_text = sanitize_text(diagnosis_json.get("diagnosis", "")) if diagnosis_json else ""
                rationale_text = sanitize_text(diagnosis_json.get("rationale", "")) if diagnosis_json else ""
                confidence_text = sanitize_text(diagnosis_json.get("confidence", "")) if diagnosis_json else ""
                key_findings_text = sanitize_text(diagnosis_json.get("key_findings", "")) if diagnosis_json else ""
                differential_text = sanitize_text(diagnosis_json.get("differential", "")) if diagnosis_json else ""

                # 2) Try markdown **Key:** Value (Gemini chain-of-thought style)
                if not diagnosis_text or not rationale_text:
                    md = parse_markdown_fields(diagnosis_output)
                    if not diagnosis_text:
                        diagnosis_text = sanitize_text(
                            md.get("final_diagnosis", "")
                            or md.get("revised_diagnosis", "")
                            or md.get("diagnosis", "")
                        )
                    if not rationale_text:
                        rationale_text = sanitize_text(
                            md.get("final_rationale", "")
                            or md.get("revised_rationale", "")
                            or md.get("rationale", "")
                        )
                    if not confidence_text:
                        confidence_text = sanitize_text(md.get("confidence", ""))
                    if not key_findings_text:
                        key_findings_text = sanitize_text(md.get("key_findings", ""))
                    if not differential_text:
                        differential_text = sanitize_text(md.get("differential", ""))

                # 3) Last-resort heuristic for diagnosis only
                if not diagnosis_text:
                    diagnosis_text = parse_diagnosis(diagnosis_output)

                _conf_colour = {"high": "#1a7f37", "moderate": "#b45309", "low": "#b91c1c"}
                _conf_label = confidence_text or ""
                _conf_style = _conf_colour.get(_conf_label.lower(), "#1f77b4")
                conf_badge = (
                    f" <span style='background:{_conf_style};color:white;padding:2px 8px;"
                    f"border-radius:10px;font-size:0.75rem;'>{_conf_label}</span>"
                    if _conf_label else ""
                )

                if diagnosis_text:
                    st.markdown(
                        f"<div style='background:#f0fdf4;border-left:4px solid #16a34a;"
                        f"padding:10px 14px;border-radius:4px;font-size:1.05rem;'>"
                        f"<strong>Diagnosis:</strong> {diagnosis_text}{conf_badge}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("Diagnosis could not be extracted — see raw output below.")

                if rationale_text:
                    st.markdown(f"**Rationale:** {rationale_text}")
                if key_findings_text:
                    st.markdown(f"**Key findings:** {key_findings_text}")
                if differential_text:
                    st.markdown(f"**Differential:** {differential_text}")

                with st.expander("Raw diagnostic output"):
                    st.code(diagnosis_output)
            else:
                st.caption("Run **Diagnose** to see a clinical diagnostic result here.")

            st.divider()

            # ── Plain-Language Explanation (grounded in EHR facts + diagnosis) ──
            st.markdown("##### Plain-Language Explanation")
            st.caption(
                "Explanation is grounded in the EHR record and the diagnostic result above — "
                "run **Diagnose** first to maximise relevance."
            )
            if explain_error:
                st.error(f"Request failed: {explain_error}")
            elif explain_output:
                explain_json = parse_json_object(explain_output)
                if explain_json:
                    summary_text = sanitize_text(explain_json.get("plain_summary", ""))
                    findings_text = sanitize_text(
                        explain_json.get("image_findings", "")
                        or explain_json.get("image_interpretation", "")
                    )
                    conclusion_text = sanitize_text(
                        explain_json.get("image_conclusion", "")
                        or explain_json.get("conclusion", "")
                    )
                    next_steps_text = sanitize_text(explain_json.get("next_steps", ""))
                else:
                    # Fallback: markdown **Key:** Value
                    md = parse_markdown_fields(explain_output)
                    summary_text = sanitize_text(
                        md.get("plain_summary", "") or md.get("summary", "")
                    )
                    findings_text = sanitize_text(
                        md.get("image_findings", "") or md.get("image_interpretation", "")
                    )
                    conclusion_text = sanitize_text(
                        md.get("image_conclusion", "") or md.get("conclusion", "")
                    )
                    next_steps_text = sanitize_text(md.get("next_steps", ""))

                if summary_text:
                    st.info(summary_text)
                elif not any([findings_text, conclusion_text, next_steps_text]):
                    st.write(explain_output)
                for label, value in [
                    ("What the images show", findings_text),
                    ("What this means", conclusion_text),
                    ("What happens next", next_steps_text),
                ]:
                    if value:
                        st.markdown(f"**{label}:** {value}")
                with st.expander("Raw explanation output"):
                    st.code(explain_output)
            else:
                st.caption("Run **Explain** to see a plain-language explanation here.")

            # ── Clinical Trial Profile ──────────────────────────────────────
            if trial_error:
                st.divider()
                st.markdown("##### 🔬 Clinical Trial Profile")
                st.error(f"Trial profile failed: {trial_error}")
            elif trial_result:
                st.divider()
                with st.expander("🔬 Clinical Trial Profile", expanded=True):
                    st.caption(f"topic_id: `{trial_result.get('topic_id', '')}`")

                    profile_text = trial_result.get("profile_text", "")
                    if profile_text:
                        st.markdown("**INGEST Summary**")
                        st.markdown(profile_text)

                    key_facts = trial_result.get("key_facts") or []
                    if key_facts:
                        st.markdown("**Key Facts**")
                        for kf in key_facts:
                            field_name = kf.get("field", "")
                            value = kf.get("value")
                            evidence_span = kf.get("evidence_span")
                            required = kf.get("required", False)
                            notes = kf.get("notes") or ""
                            is_null = value is None

                            # Source badge for imaging_findings
                            source_badge = ""
                            if field_name == "imaging_findings":
                                if "direct_image_analysis" in notes:
                                    source_badge = " 🖼️ **Image Analysis**"
                                elif "ehr_extracted" in notes:
                                    source_badge = " 📄 **EHR Extracted**"
                                else:
                                    source_badge = ""
                                if is_null:
                                    source_badge = " ⚠️ **Not Available**"

                            # Null warning for required fields
                            null_warn = " ⚠️ *missing*" if (required and is_null) else ""

                            # Format value for display
                            if is_null:
                                display_val = "_null_"
                            elif isinstance(value, list):
                                display_val = "; ".join(str(v) for v in value)
                            elif isinstance(value, dict):
                                display_val = ", ".join(
                                    f"{k}: {v}" for k, v in value.items()
                                )
                            else:
                                display_val = str(value)

                            st.markdown(
                                f"**{field_name}**{source_badge}{null_warn}: {display_val}"
                            )
                            if evidence_span:
                                st.markdown(f"> *\"{evidence_span}\"*")

                    ambiguities = trial_result.get("ambiguities") or []
                    if ambiguities:
                        st.markdown("**Ambiguities**")
                        for amb in ambiguities:
                            st.markdown(f"- {amb}")

                    st.download_button(
                        "⬇ Download JSON",
                        data=json.dumps(trial_result, indent=2),
                        file_name=f"trial_profile_{selected_uid}.json",
                        mime="application/json",
                    )

    # ────────────────────────────────────────────────────────────────────────
    # PANEL 3 — Linked Images
    # ────────────────────────────────────────────────────────────────────────
    with img_col:
        st.subheader("Linked Images")
        st.caption(
            f"{_n_images} image(s) linked to this case."
            + (" Disable 'Preview images in viewer' in the sidebar to hide." if show_linked_images else " Enable 'Preview images in viewer' in the sidebar to display.")
        )
        if show_linked_images and resolved_images:
            shown = resolved_images[:6]
            preview_items: List[Any] = []
            captions: List[str] = []
            skipped = 0
            for image_path in shown:
                try:
                    preview_items.append(preview_payload(image_path))
                    captions.append(image_path.name)
                except Exception:
                    skipped += 1

            if preview_items:
                st.image(
                    preview_items,
                    caption=captions,
                    use_container_width=True,
                )
            if skipped:
                st.caption(f"{skipped} image(s) could not be rendered for preview.")
        elif show_linked_images:
            st.info("No linked images found on disk for this case.")


if __name__ == "__main__":
    main()
