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
# Representative MRI preview slices extracted from the Synthea coherent dataset.
# Used as fallback images for Synthetic cases when the original DICOM ZIP is not
# on disk (the ZIP was removed to reduce repo size).
SYNTHETIC_MRI_SLICES_DIR = REPO_ROOT / "Synthetic" / "mri_slices"
SYNTHETIC_FALLBACK_IMAGE_NAMES = ("axial_mid.png", "coronal_mid.png", "sagittal_mid.png")
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
    "Draw on all provided text and any images. "
    "Return ONLY valid JSON — no markdown, no prose outside the JSON — with exactly these keys:\n"
    "  plain_summary    : 2-3 sentence plain-language overview of what is happening\n"
    "  image_findings   : what the medical images show, described without jargon (empty string if no images)\n"
    "  image_conclusion : what those image findings mean for the patient (empty string if no images)\n"
    "  next_steps       : what typically happens next (tests, referrals, treatment)"
)

MEDGEMMA_DIAGNOSIS_PROMPT = (
    "You are MedGemma, a specialist clinical diagnostic assistant. "
    "Analyse the provided patient history, examination findings, and any attached images. "
    "Apply systematic clinical reasoning: consider the presenting complaint, risk factors, "
    "examination signs, and imaging characteristics to reach a single best diagnosis. "
    "Return ONLY valid JSON — no markdown, no prose outside the JSON — with exactly these keys:\n"
    "  diagnosis         : single most likely diagnosis (concise clinical term)\n"
    "  confidence        : one of 'High', 'Moderate', or 'Low'\n"
    "  rationale         : 3-5 sentence evidence-based justification citing specific findings from the case\n"
    "  key_findings      : comma-separated list of the top 3-5 findings that support this diagnosis\n"
    "  differential      : 2-3 alternative diagnoses to consider, as a comma-separated list"
)

# Gemini 2.5 Pro: thinking model — be crystal-clear about the desired JSON schema
# so the model's visible output (after internal reasoning) is clean JSON.
GEMINI_EXPLAIN_PROMPT = (
    "You are a medical explainer for patients and non-medical readers. "
    "Use plain, friendly, empathetic language. Avoid jargon.\n\n"
    "If images are present, describe what you see visually, list key image findings, "
    "and provide a clear conclusion the reader can understand.\n\n"
    "Return ONLY a single JSON object (no markdown, no code fences, no extra text) "
    "with exactly these four string keys:\n"
    '  "plain_summary"    – 2-3 sentence overview of what is happening\n'
    '  "image_findings"   – what the images show (empty string "" if none)\n'
    '  "image_conclusion" – what those findings mean for the patient (empty string "" if none)\n'
    '  "next_steps"       – what typically happens next (tests, referrals, treatment)\n'
)

GEMINI_DIAGNOSIS_PROMPT = (
    "You are a specialist clinical diagnostic assistant. "
    "Analyse all provided clinical history, examination findings, "
    "and any attached images.\n\n"
    "Apply systematic clinical reasoning: consider the presenting complaint, risk factors, "
    "examination signs, and imaging characteristics.\n\n"
    "Return ONLY a single JSON object (no markdown, no code fences, no extra text) "
    "with exactly these five string keys:\n"
    '  "diagnosis"    – single most likely diagnosis (concise clinical term)\n'
    '  "confidence"   – exactly one of: "High", "Moderate", "Low"\n'
    '  "rationale"    – 3-5 sentence evidence-based justification citing specific case findings\n'
    '  "key_findings" – top 3-5 supporting findings, comma-separated\n'
    '  "differential" – 2-3 alternative diagnoses to consider, comma-separated\n'
)


def history_richness_score(case: Dict[str, Any]) -> int:
    """Score a case by the total information richness of its clinical text fields.

    Weights: history × 3 (primary), findings × 2, exam × 1, discussion × 1.
    Skips N/A and placeholder values.
    """
    def _len(field: str) -> int:
        v = sanitize_text(case.get(field, "")).strip()
        if not v or v.lower() in {"n/a", "none", "na", "-"}:
            return 0
        return len(v)

    return _len("history") * 3 + _len("findings") * 2 + _len("exam") + _len("discussion")


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


def _is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda" in message


def _is_cuda_kernel_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "cuda error" in message
        or "misaligned address" in message
        or "device-side assertions" in message
    )


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


def get_resolved_images(case: Dict[str, Any]) -> Tuple[List[Tuple[Path, Dict[str, Any]]], bool]:
    """Return (resolved_images, is_fallback), where each item is
    (resolved_path, original_image_metadata).

    For Synthea cases whose DICOM images are stored inside the (deleted) ZIP,
    falls back to the on-disk MRI preview slices so the viewer and LLM still
    receive representative images.
    """
    resolved_images: List[Tuple[Path, Dict[str, Any]]] = []
    for image in case.get("images", []):
        raw_path = image.get("file_path")
        if not raw_path:
            continue
        raw_path = str(raw_path)
        resolved = resolve_image_path(WORKSPACE_ROOT, raw_path)
        if resolved and resolved.exists():
            resolved_images.append((resolved, image))

    if resolved_images:
        return resolved_images, False

    # Fallback: Synthea cases whose images are inside the removed ZIP
    if case.get("dataset_source", "") == "Synthea coherent zip" and case.get("images"):
        fallback: List[Tuple[Path, Dict[str, Any]]] = []
        for fname in SYNTHETIC_FALLBACK_IMAGE_NAMES:
            p = SYNTHETIC_MRI_SLICES_DIR / fname
            if p.exists():
                fallback.append((p, {"type": "MRI", "plane": p.stem.replace("_mid", "")}))
        if fallback:
            return fallback, True

    return [], False


def _build_image_context_text(
    resolved_images: List[Tuple[Path, Dict[str, Any]]],
    is_fallback: bool,
) -> str:
    """Build a concise plain-text description of the images being passed to
    the LLM so it understands what each image represents.

    Injected into the EHR text *after* truncation so it is never cut off.
    """
    if not resolved_images:
        return ""

    lines: List[str] = []
    if is_fallback:
        lines.append(
            "[Note: Patient-specific DICOM images are stored in an archived ZIP "
            "that is not currently on disk. The following representative MRI views "
            "from the Synthea coherent dataset are provided for illustrative "
            "imaging context only.]"
        )

    for i, (path, img_meta) in enumerate(resolved_images):
        if is_fallback:
            # Describe by filename stem (axial_mid / coronal_mid / sagittal_mid)
            stem = path.stem.replace("_", " ")
            lines.append(f"  Attached image {i + 1}: {stem} (representative MRI preview)")
        else:
            parts: List[str] = []
            modality = sanitize_text(img_meta.get("modality") or img_meta.get("type") or "")
            plane = sanitize_text(img_meta.get("plane") or "")
            location = sanitize_text(img_meta.get("location") or "")
            caption = sanitize_text(img_meta.get("caption") or "")
            if modality:
                parts.append(modality)
            if plane:
                parts.append(f"{plane} view")
            if location:
                parts.append(f"region: {location}")
            if caption:
                parts.append(f'Caption: "{caption}"')
            desc = ", ".join(parts) if parts else "(no metadata)"
            lines.append(f"  Attached image {i + 1}: {desc}")

    header = "Imaging context (images attached):"
    return header + "\n" + "\n".join(lines)


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
    effective_images = max_images
    effective_tokens = max_tokens
    effective_ehr = ehr_text

    for attempt in range(5):
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
            if attempt == 4:
                raise

            if _is_oom_error(exc):
                effective_tokens = max(32, int(effective_tokens * 0.7))
                effective_ehr = truncate_text(effective_ehr, max(120, int(len(effective_ehr) * 0.7)))
                if effective_images > 0:
                    effective_images -= 1
                continue

            if _is_cuda_kernel_error(exc):
                # Endpoint-side CUDA kernel instability is often triggered by heavier multimodal requests.
                # Degrade request load progressively before giving up.
                if effective_images > 0:
                    effective_images -= 1
                else:
                    effective_tokens = max(64, int(effective_tokens * 0.7))
                    effective_ehr = truncate_text(effective_ehr, max(180, int(len(effective_ehr) * 0.8)))
                continue

            raise

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
            options=["MedPix JSONL", "PMC-Patients CSV", "Synthea Coherent JSONL"],
            index=0,
            help="Choose which source data to load. Both are normalized into the same downstream workflow.",
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
    ctrl_uid, ctrl_explain, ctrl_diagnose = st.columns([3, 1.2, 1.2])
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
    resolved_images, _images_are_fallback = get_resolved_images(selected_case)
    resolved_image_paths = [p for p, _meta in resolved_images]
    _n_images = len(resolved_image_paths)
    with st.sidebar:
        st.markdown("---")
        max_images = st.slider(
            f"Images for inference (available: {_n_images})",
            min_value=0,
            max_value=max(1, _n_images),
            value=min(1, _n_images),
            step=1,
            disabled=(_n_images == 0),
        )

    composed_ehr = truncate_text(build_ehr_text(selected_case), target_ehr_chars)
    # Append image metadata AFTER truncation so it is never cut off.
    # This lets the LLM know what each attached image represents (modality,
    # plane, anatomical region, caption from the dataset).
    if resolved_images and use_images:
        img_ctx = _build_image_context_text(resolved_images[:max_images], _images_are_fallback)
        if img_ctx:
            composed_ehr = composed_ehr + "\n\n" + img_ctx
    effective_max_tokens = max_tokens
    if llm_choice == LLM_MEDGEMMA:
        effective_max_tokens = medgemma_token_budget(
            requested_tokens=max_tokens,
            ehr_text=composed_ehr,
            image_count=min(max_images, len(resolved_image_paths)),
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

    if llm_choice == LLM_MEDGEMMA and effective_max_tokens < max_tokens:
        st.caption(
            f"MedGemma performance cap applied: using {effective_max_tokens} output tokens "
            f"(requested {max_tokens}) based on current context size."
        )

    if (run_explain or run_diagnosis) and not active_credential:
        st.error(
            "Missing credential. Configure one of: "
            + ", ".join(f"`{k}`" for k in required_cred_keys)
            + " in `.streamlit/secrets.toml` or environment variables."
        )

    # ── Per-UID + per-model session cache ───────────────────────────────────
    _mk = "mg" if llm_choice == LLM_MEDGEMMA else "gf"
    _ek = f"explain_{_mk}_{selected_uid}"
    _ee = f"explain_error_{_mk}_{selected_uid}"
    _dk = f"diagnosis_{_mk}_{selected_uid}"
    _de = f"diagnosis_error_{_mk}_{selected_uid}"

    def _run_inference(system_prompt: str) -> str:
        if llm_choice == LLM_MEDGEMMA:
            return call_medgemma(
                endpoint_url=endpoint_url,
                token=active_credential,
                model_id=model_id,
                system_prompt=system_prompt,
                ehr_text=composed_ehr,
                image_paths=resolved_image_paths,
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
                system_prompt=system_prompt,
                ehr_text=composed_ehr,
                image_paths=resolved_image_paths,
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
        with st.spinner(f"[{llm_choice}] Generating explanation..."):
            try:
                st.session_state[_ek] = _run_inference(explain_prompt)
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

    explain_output   = st.session_state.get(_ek, "")
    explain_error    = st.session_state.get(_ee, "")
    diagnosis_output = st.session_state.get(_dk, "")
    diagnosis_error  = st.session_state.get(_de, "")

    # ── Unified three-panel layout ─────────────────────────────────────────
    # All three panels (EHR · Images · AI Insights) sit side-by-side so the
    # user can cross-reference AI output against the source data without any
    # tab switching.
    ehr_col, img_col, ai_col = st.columns([1.2, 1.0, 1.4])

    # ── Column 1: EHR Record ───────────────────────────────────────────────
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
        with st.expander("Imaging Findings", expanded=True):
            st.write(sanitize_text(selected_case.get("findings", "")) or "Not available")

    # ── Column 2: Linked Images ────────────────────────────────────────────
    with img_col:
        st.subheader("Linked Images")
        if _images_are_fallback:
            st.caption(
                f"Patient-specific DICOM images are inside the removed ZIP archive. "
                f"Showing {_n_images} representative MRI preview slice(s) from the "
                f"Synthea dataset for visual context."
            )
        else:
            st.caption(
                f"{_n_images} image(s) linked to this case."
                + (" Enable 'Preview images in viewer' in the sidebar to display them." if not show_linked_images else "")
            )
        if show_linked_images and resolved_image_paths:
            shown = resolved_image_paths[:6]
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

    # ── Column 3: AI Insights (Diagnosis + Plain-Language Explanation) ─────
    with ai_col:
        has_results = bool(explain_output or explain_error or diagnosis_output or diagnosis_error)

        # ── 3a: Diagnostic Result ──────────────────────────────────────────
        st.subheader("Diagnostic Result")
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

            # ── Display ──
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

            with st.expander("Raw output"):
                st.code(diagnosis_output)
        elif not has_results:
            st.caption("Click **Diagnose** above to generate a clinical diagnostic result.")
        else:
            st.caption("Run 'Diagnose' to see results here.")

        st.divider()

        # ── 3b: Plain-Language Explanation ────────────────────────────────
        st.subheader("Plain-Language Explanation")
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
                # Nothing structured — show raw
                st.write(explain_output)
            for label, value in [
                ("Image findings", findings_text),
                ("Image conclusion", conclusion_text),
                ("Next steps", next_steps_text),
            ]:
                if value:
                    st.markdown(f"**{label}:** {value}")
            with st.expander("Raw output"):
                st.code(explain_output)
        elif not has_results:
            st.caption("Click **Explain (non-medical)** above to generate a plain-language summary.")
        else:
            st.caption("Run 'Explain' to see results here.")


if __name__ == "__main__":
    main()
