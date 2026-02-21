"""Tool implementations for the trial matching agent.

Each tool is a callable that performs one step of the pipeline.
Tools are invoked by the Gemini orchestrator via function-calling,
or called directly in the deterministic pipeline mode.

Model assignment rationale is documented per-tool.
"""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

from .config import AgentConfig, GEMINI_THINKING_BUDGET, GEMINI_MIN_OUTPUT_TOKENS
from .schemas import (
    AggregationResult,
    CriterionMatch,
    FeedbackRequest,
    ImageAnalysis,
    MatchingResult,
    PatientProfile,
    RankedTrial,
    TrialInfo,
)
from .prompts import (
    KEYWORD_GENERATION_SYSTEM,
    IMAGE_ANALYSIS_SYSTEM,
    IMAGE_CRITERION_VALIDATION_SYSTEM,
    FEEDBACK_GENERATION_SYSTEM,
    PROFILE_EXTRACTION_SYSTEM,
    build_keyword_generation_prompt,
    build_matching_system_prompt,
    build_matching_user_prompt,
    build_aggregation_system_prompt,
    build_aggregation_user_prompt,
    build_feedback_user_prompt,
    build_image_criterion_prompt,
    format_criterion_predictions,
    format_trial_for_prompt,
    number_patient_sentences,
)
from .scoring import get_matching_score, get_agg_score, get_trial_score


# ──────────────────────────────────────────────────────────────────────────
# Gemini API helpers
# ──────────────────────────────────────────────────────────────────────────

def _call_gemini(
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 4096,
    thinking_budget: int = GEMINI_THINKING_BUDGET,
    timeout: int = 180,
    retries: int = 3,
    image_paths: Optional[List[Path]] = None,
    response_mime_type: Optional[str] = "application/json",
) -> str:
    """Call Gemini via the google-genai SDK.

    Handles:
    - Thinking model budget (separate from output tokens)
    - Image upload (PIL objects)
    - Structured JSON output via response_mime_type
    - Retry with exponential backoff
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Build content parts
    contents: List[Any] = [user_prompt]

    # Add images if provided
    if image_paths:
        try:
            from PIL import Image
            for img_path in image_paths:
                if img_path.exists():
                    img = Image.open(img_path)
                    contents.append(img)
        except ImportError:
            pass  # PIL not available — skip images

    # Configure generation
    gen_config_kwargs: Dict[str, Any] = {
        "max_output_tokens": max_output_tokens,
        "temperature": 0.0,
    }
    if response_mime_type:
        gen_config_kwargs["response_mime_type"] = response_mime_type

    # Add thinking budget for Gemini 2.5 Pro
    if "2.5" in model_name or "2-5" in model_name:
        gen_config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget
        )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        **gen_config_kwargs,
    )

    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            # Extract text (handle thinking model response)
            return _extract_gemini_text(response)
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = min(2 ** attempt, 30)
                time.sleep(wait)
            continue

    raise RuntimeError(f"Gemini call failed after {retries} attempts: {last_exc}")


def _extract_gemini_text(response: Any) -> str:
    """Extract text from Gemini response, handling thinking model output."""
    # Fast path
    if response.text:
        return response.text

    # Manual iteration for thinking models
    parts_text: List[str] = []
    thought_text: List[str] = []
    for candidate in (response.candidates or []):
        for part in (candidate.content.parts or []):
            if hasattr(part, "thought") and part.thought:
                if part.text:
                    thought_text.append(part.text)
            elif part.text:
                parts_text.append(part.text)

    if parts_text:
        return "\n".join(parts_text)
    if thought_text:
        return "\n".join(thought_text)
    return ""


# ──────────────────────────────────────────────────────────────────────────
# MedGemma API helpers
# ──────────────────────────────────────────────────────────────────────────

def _image_to_data_url(path: Path) -> str:
    """Convert an image file to a base64 data URL for MedGemma."""
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".dcm":
        # Convert DICOM to PNG bytes
        try:
            import pydicom
            from PIL import Image
            import io
            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array
            # Normalize to 0-255
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-9) * 255).astype("uint8")
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{data}"
        except Exception:
            return ""
    else:
        mime = "image/png"  # fallback

    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def _call_medgemma(
    endpoint_url: str,
    token: str,
    model_id: str,
    system_prompt: str,
    user_text: str,
    image_paths: Optional[List[Path]] = None,
    max_tokens: int = 1024,
    timeout: int = 120,
    retries: int = 3,
) -> str:
    """Call MedGemma via HuggingFace Inference Endpoint.

    Uses the OpenAI-compatible /v1/chat/completions API.
    Implements adaptive OOM retry (reduce tokens, drop images).
    """
    effective_tokens = max_tokens
    effective_images = list(image_paths) if image_paths else []
    effective_text = user_text

    for attempt in range(4):
        # Build user content
        content: Any
        if effective_images:
            blocks: List[Dict[str, Any]] = [
                {"type": "text", "text": effective_text}
            ]
            for img_path in effective_images:
                data_url = _image_to_data_url(img_path)
                if data_url:
                    blocks.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
            content = blocks
        else:
            content = effective_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        payload = json.dumps({
            "model": model_id,
            "messages": messages,
            "max_tokens": effective_tokens,
            "temperature": 0.1,
            "stream": False,
        }).encode("utf-8")

        url = endpoint_url.rstrip("/")
        if not url.endswith("/v1/chat/completions"):
            url += "/v1/chat/completions"

        req = request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""

        except error.HTTPError as exc:
            status = exc.code
            body_text = ""
            try:
                body_text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass

            # OOM — adaptive retry
            is_oom = (
                status in (500, 503)
                and ("CUDA" in body_text or "out of memory" in body_text.lower())
            )
            if is_oom and attempt < 3:
                effective_tokens = max(32, int(effective_tokens * 0.7))
                effective_text = effective_text[:max(120, int(len(effective_text) * 0.7))]
                if effective_images:
                    effective_images = effective_images[:-1]
                continue

            # Rate limit
            if status == 429:
                retry_after = int(exc.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(retry_after, 60))
                continue

            # Retriable server errors
            if status in (500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue

            raise

        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    raise RuntimeError("MedGemma call failed after adaptive retries.")


def _parse_json_from_text(text: str) -> Any:
    """Extract JSON from LLM output, handling markdown fences."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try stripping markdown code fences
    cleaned = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```\s*$', '', cleaned, flags=re.MULTILINE)
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        pass

    # Try finding first JSON object/array
    for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue

    return None


# ══════════════════════════════════════════════════════════════════════
# TOOL 1: EXTRACT PATIENT PROFILE
# MODEL: Gemini 2.5 Pro
# ══════════════════════════════════════════════════════════════════════

def extract_patient_profile(
    ehr_text: str,
    patient_id: str,
    config: AgentConfig,
) -> PatientProfile:
    """Extract a structured patient profile from EHR text.

    MODEL RATIONALE: Gemini 2.5 Pro — needs long-context reasoning over
    multi-section EHR + structured JSON output. MedGemma's 4B params and
    limited output budget cannot handle this complex extraction reliably.
    """
    system = PROFILE_EXTRACTION_SYSTEM + f"\n\nPatient ID: {patient_id}"
    user = f"EHR Text:\n{ehr_text}\n\nJSON output:"

    raw = _call_gemini(
        api_key=config.gemini_api_key,
        model_name=config.gemini_model,
        system_prompt=system,
        user_prompt=user,
        max_output_tokens=config.gemini_max_output_tokens,
        timeout=config.timeout_sec,
        retries=config.retries,
    )

    parsed = _parse_json_from_text(raw)
    if not parsed or not isinstance(parsed, dict):
        # Return minimal profile
        return PatientProfile(
            topic_id=patient_id.lower(),
            profile_text=raw[:2000] if raw else "",
        )

    return PatientProfile.from_dict({
        **parsed,
        "topic_id": patient_id.lower(),
    })


# ══════════════════════════════════════════════════════════════════════
# TOOL 2: ANALYZE IMAGES
# MODEL: MedGemma (primary) → Gemini 2.5 Pro (fallback)
# ══════════════════════════════════════════════════════════════════════

def analyze_images(
    image_paths: List[Path],
    clinical_context: str,
    config: AgentConfig,
) -> ImageAnalysis:
    """Analyze medical images using MedGemma radiology specialist.

    MODEL RATIONALE: MedGemma (medgemma-1-5-4b-it-hae) is purpose-built
    for medical image analysis with radiology-specific fine-tuning.
    Falls back to Gemini 2.5 Pro multimodal if MedGemma is unavailable
    or encounters CUDA OOM.

    PROS of MedGemma: Domain-specific accuracy, DICOM-aware training
    CONS of MedGemma: CUDA OOM on multi-image inputs, small context
    """
    if not image_paths:
        return ImageAnalysis()

    user_text = f"Clinical context: {clinical_context}\n\nAnalyze the provided medical images."

    # Try MedGemma first
    if config.hf_token:
        try:
            raw = _call_medgemma(
                endpoint_url=config.medgemma_endpoint,
                token=config.hf_token,
                model_id=config.medgemma_model,
                system_prompt=IMAGE_ANALYSIS_SYSTEM,
                user_text=user_text,
                image_paths=image_paths,
                max_tokens=config.medgemma_max_tokens,
                timeout=config.timeout_sec,
                retries=config.retries,
            )
            return _parse_image_analysis(raw)
        except Exception:
            pass  # Fall through to Gemini

    # Fallback: Gemini 2.5 Pro multimodal
    if config.gemini_api_key:
        raw = _call_gemini(
            api_key=config.gemini_api_key,
            model_name=config.gemini_model,
            system_prompt=IMAGE_ANALYSIS_SYSTEM,
            user_prompt=user_text,
            image_paths=image_paths,
            max_output_tokens=2048,
            timeout=config.timeout_sec,
            retries=config.retries,
        )
        return _parse_image_analysis(raw)

    return ImageAnalysis(raw_text="No model available for image analysis.")


def _parse_image_analysis(raw: str) -> ImageAnalysis:
    """Parse image analysis output into structured object."""
    parsed = _parse_json_from_text(raw)
    if isinstance(parsed, dict):
        return ImageAnalysis(
            modality=parsed.get("modality", ""),
            body_part=parsed.get("body_part", ""),
            findings=parsed.get("findings", []),
            raw_text=raw,
        )
    return ImageAnalysis(raw_text=raw)


# ══════════════════════════════════════════════════════════════════════
# TOOL 3: GENERATE SEARCH KEYWORDS
# MODEL: Gemini 2.5 Pro
# ══════════════════════════════════════════════════════════════════════

def generate_search_keywords(
    patient_text: str,
    config: AgentConfig,
) -> Dict[str, Any]:
    """Generate search keywords for trial retrieval.

    MODEL RATIONALE: Gemini 2.5 Pro — needs reasoning to prioritise
    conditions and generate diverse, ranked search terms. MedGemma's
    limited vocabulary breadth makes it unsuitable for this task.

    Returns dict: {"summary": str, "conditions": list[str]}
    """
    raw = _call_gemini(
        api_key=config.gemini_api_key,
        model_name=config.gemini_model,
        system_prompt=KEYWORD_GENERATION_SYSTEM,
        user_prompt=build_keyword_generation_prompt(patient_text),
        max_output_tokens=1024,
        timeout=config.timeout_sec,
        retries=config.retries,
    )

    parsed = _parse_json_from_text(raw)
    if isinstance(parsed, dict):
        return {
            "summary": parsed.get("summary", ""),
            "conditions": parsed.get("conditions", []),
        }
    return {"summary": raw[:500], "conditions": []}


# ══════════════════════════════════════════════════════════════════════
# TOOL 4: SEARCH TRIALS (no LLM — BM25)
# ══════════════════════════════════════════════════════════════════════

def search_trials(
    conditions: List[str],
    bm25_index: Any,  # BM25Index
    top_n: int = 50,
) -> List[TrialInfo]:
    """Search the trial corpus using BM25 keyword matching.

    NO LLM NEEDED: This is a deterministic retrieval step.
    Uses BM25 with reciprocal rank fusion across conditions,
    matching TrialGPT's retrieval approach (without MedCPT).
    """
    if not conditions:
        return []

    results = bm25_index.multi_condition_search(conditions, top_n=top_n)
    trials: List[TrialInfo] = []
    for nct_id, _score in results:
        trial = bm25_index.get_trial_info(nct_id)
        if trial:
            trials.append(trial)
    return trials


# ══════════════════════════════════════════════════════════════════════
# TOOL 5: CRITERION-LEVEL MATCHING
# MODEL: Gemini 2.5 Pro
# ══════════════════════════════════════════════════════════════════════

def match_criteria(
    patient_text: str,
    trial_info: TrialInfo,
    config: AgentConfig,
) -> MatchingResult:
    """Evaluate patient eligibility criterion-by-criterion.

    MODEL RATIONALE: Gemini 2.5 Pro — core task requiring nuanced logical
    reasoning over each inclusion/exclusion criterion vs. patient evidence.
    Chain-of-thought reasoning is essential. MedGemma cannot handle this
    structured multi-criterion analysis.

    ENHANCEMENT OVER TRIALGPT: Added 4th element ``imaging_relevant``
    (bool) to flag criteria that could benefit from image review.

    Two separate calls (inclusion + exclusion) — matches TrialGPT's
    proven approach to avoid mixing label sets in one prompt.
    """
    result = MatchingResult()
    trial_dict = trial_info.to_dict()

    for inc_exc in ["inclusion", "exclusion"]:
        criteria_text = trial_dict.get(f"{inc_exc}_criteria", "")
        if not criteria_text.strip():
            continue

        system = build_matching_system_prompt(inc_exc)
        user = build_matching_user_prompt(trial_dict, inc_exc, patient_text)

        raw = _call_gemini(
            api_key=config.gemini_api_key,
            model_name=config.gemini_model,
            system_prompt=system,
            user_prompt=user,
            max_output_tokens=config.gemini_max_output_tokens,
            timeout=config.timeout_sec,
            retries=config.retries,
        )

        parsed = _parse_json_from_text(raw)
        if not isinstance(parsed, dict):
            continue

        criteria_dict: Dict[str, CriterionMatch] = {}
        for criterion_idx, info in parsed.items():
            if not isinstance(info, list):
                continue

            # Handle both 3-element (TrialGPT compat) and 4-element (ours)
            if len(info) >= 3:
                cm = CriterionMatch(
                    criterion_idx=str(criterion_idx),
                    reasoning=str(info[0]) if info[0] else "",
                    evidence_sentence_ids=info[1] if isinstance(info[1], list) else [],
                    label=str(info[2]) if info[2] else "",
                    imaging_relevant=bool(info[3]) if len(info) >= 4 else False,
                )
                criteria_dict[str(criterion_idx)] = cm

        if inc_exc == "inclusion":
            result.inclusion = criteria_dict
        else:
            result.exclusion = criteria_dict

    return result


# ══════════════════════════════════════════════════════════════════════
# TOOL 6: VALIDATE IMAGING CRITERIA
# MODEL: MedGemma (primary) → Gemini (fallback)
# ══════════════════════════════════════════════════════════════════════

def validate_imaging_criteria(
    imaging_criteria: List[CriterionMatch],
    image_paths: List[Path],
    config: AgentConfig,
) -> Dict[str, str]:
    """Validate imaging-relevant criteria against actual medical images.

    MODEL RATIONALE: MedGemma — purpose-built for radiology image
    interpretation. For criteria like "measurable disease per RECIST"
    or "brain metastases", MedGemma's domain-specific training provides
    more accurate assessment than Gemini's general multimodal capability.
    Falls back to Gemini if MedGemma unavailable.

    This is the UNIQUE IMAGE-ENHANCED step not present in TrialGPT.
    """
    if not imaging_criteria or not image_paths:
        return {}

    criteria_list = [
        {"criterion": cm.criterion_idx, "text": cm.reasoning}
        for cm in imaging_criteria
    ]
    user_text = build_image_criterion_prompt(criteria_list)

    # Try MedGemma first
    raw = ""
    if config.hf_token:
        try:
            raw = _call_medgemma(
                endpoint_url=config.medgemma_endpoint,
                token=config.hf_token,
                model_id=config.medgemma_model,
                system_prompt=IMAGE_CRITERION_VALIDATION_SYSTEM,
                user_text=user_text,
                image_paths=image_paths,
                max_tokens=config.medgemma_max_tokens,
                timeout=config.timeout_sec,
                retries=config.retries,
            )
        except Exception:
            raw = ""

    # Fallback to Gemini
    if not raw and config.gemini_api_key:
        raw = _call_gemini(
            api_key=config.gemini_api_key,
            model_name=config.gemini_model,
            system_prompt=IMAGE_CRITERION_VALIDATION_SYSTEM,
            user_prompt=user_text,
            image_paths=image_paths,
            max_output_tokens=2048,
            timeout=config.timeout_sec,
            retries=config.retries,
        )

    # Parse results
    parsed = _parse_json_from_text(raw)
    assessments: Dict[str, str] = {}

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                crit = item.get("criterion", "")
                label = item.get("revised_label")
                if crit and label:
                    assessments[str(crit)] = str(label)
    elif isinstance(parsed, dict):
        for k, v in parsed.items():
            if isinstance(v, dict):
                assessments[str(k)] = str(v.get("revised_label", ""))

    return assessments


# ══════════════════════════════════════════════════════════════════════
# TOOL 7: AGGREGATE AND SCORE
# MODEL: Gemini 2.5 Pro
# ══════════════════════════════════════════════════════════════════════

def aggregate_and_score(
    patient_text: str,
    trial_info: TrialInfo,
    matching_result: MatchingResult,
    config: AgentConfig,
    image_analysis_summary: str = "",
) -> AggregationResult:
    """Aggregate criterion-level results into trial-level scores.

    MODEL RATIONALE: Gemini 2.5 Pro — pure reasoning task that synthesises
    criterion-level results into holistic relevance (R) and eligibility (E)
    scores. Requires weighing contradictory criteria, handling partial
    evidence, and producing calibrated numeric scores. MedGemma's limited
    reasoning capacity makes it unsuitable.

    Also computes TrialGPT's algorithmic score for baseline comparison.
    """
    trial_dict = trial_info.to_dict()
    matching_dict = matching_result.to_trialgpt_format()

    # Format criterion predictions as readable string
    predictions_str = format_criterion_predictions(matching_dict, trial_dict)

    system = build_aggregation_system_prompt()
    user = build_aggregation_user_prompt(
        patient_text=patient_text,
        trial_info=trial_dict,
        criterion_predictions_str=predictions_str,
        image_analysis_summary=image_analysis_summary,
    )

    raw = _call_gemini(
        api_key=config.gemini_api_key,
        model_name=config.gemini_model,
        system_prompt=system,
        user_prompt=user,
        max_output_tokens=2048,
        timeout=config.timeout_sec,
        retries=config.retries,
    )

    parsed = _parse_json_from_text(raw)
    if isinstance(parsed, dict):
        return AggregationResult.from_dict(parsed)

    return AggregationResult()


# ══════════════════════════════════════════════════════════════════════
# TOOL 8: IDENTIFY MISSING INFO / FEEDBACK
# MODEL: Gemini 2.5 Pro
# ══════════════════════════════════════════════════════════════════════

def identify_missing_info(
    patient_summary: str,
    matching_results: Dict[str, MatchingResult],
    trial_infos: Dict[str, TrialInfo],
    config: AgentConfig,
) -> List[FeedbackRequest]:
    """Identify impactful missing patient information.

    MODEL RATIONALE: Gemini 2.5 Pro — identifying medically-relevant
    missing data requires clinical reasoning: which missing fields would
    actually flip eligibility decisions? Gemini's long context lets it
    cross-reference missing info against specific trial criteria.
    Template-based approaches can't adapt to specific trial requirements.
    """
    # Build summaries of "not enough information" criteria across trials
    matching_summaries: List[Dict[str, str]] = []
    for nct_id, mr in matching_results.items():
        nei_criteria: List[str] = []
        for cm in list(mr.inclusion.values()) + list(mr.exclusion.values()):
            if cm.label == "not enough information":
                nei_criteria.append(cm.reasoning or f"Criterion {cm.criterion_idx}")

        if nei_criteria:
            title = trial_infos[nct_id].brief_title if nct_id in trial_infos else nct_id
            matching_summaries.append({
                "trial": title,
                "not_enough_info_criteria": nei_criteria,
            })

    if not matching_summaries:
        return []

    system = FEEDBACK_GENERATION_SYSTEM
    user = build_feedback_user_prompt(patient_summary, matching_summaries)

    raw = _call_gemini(
        api_key=config.gemini_api_key,
        model_name=config.gemini_model,
        system_prompt=system,
        user_prompt=user,
        max_output_tokens=2048,
        timeout=config.timeout_sec,
        retries=config.retries,
    )

    parsed = _parse_json_from_text(raw)
    if not isinstance(parsed, list):
        return []

    requests: List[FeedbackRequest] = []
    for item in parsed:
        if isinstance(item, dict):
            requests.append(FeedbackRequest(
                field=item.get("field", ""),
                question=item.get("question", ""),
                reason=item.get("reason", ""),
                source_criteria=item.get("source_criteria", []),
                priority=item.get("priority", "medium"),
            ))
    return requests


# ══════════════════════════════════════════════════════════════════════
# Convenience: rank and build final trial list
# ══════════════════════════════════════════════════════════════════════

def rank_and_build_results(
    matching_results: Dict[str, MatchingResult],
    aggregation_results: Dict[str, AggregationResult],
    trial_infos: Dict[str, TrialInfo],
) -> List[RankedTrial]:
    """Combine matching + aggregation scores into ranked trial list."""
    trials: List[RankedTrial] = []

    for nct_id, mr in matching_results.items():
        m_dict = mr.to_trialgpt_format()
        m_score = get_matching_score(m_dict)
        a_score = 0.0
        agg = aggregation_results.get(nct_id)
        if agg:
            a_score = get_agg_score(agg.to_dict())

        total = get_trial_score(m_score, a_score)
        title = trial_infos[nct_id].brief_title if nct_id in trial_infos else ""

        trials.append(RankedTrial(
            nct_id=nct_id,
            title=title,
            matching_score=m_score,
            aggregation_score=a_score,
            total_score=total,
            matching_result=mr,
            aggregation_result=agg,
        ))

    trials.sort(key=lambda t: -t.total_score)
    return trials
