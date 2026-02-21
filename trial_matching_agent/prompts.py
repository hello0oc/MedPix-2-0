"""Prompt templates for the trial matching agent.

Adapted from TrialGPT's prompt engineering with enhancements for:
- Image-aware criterion matching (flags imaging-relevant criteria)
- Gemini 2.5 Pro structured output / thinking-model behaviour
- MedGemma radiology-specialist image analysis
- Patient feedback generation

Each prompt includes a MODEL SELECTION RATIONALE comment explaining
why that particular model was chosen for the task.
"""
from __future__ import annotations

from typing import Any, Dict, List


# ──────────────────────────────────────────────────────────────────────────
# Helper: parse criteria into numbered list (from TrialGPT)
# ──────────────────────────────────────────────────────────────────────────
def parse_criteria(criteria_text: str) -> str:
    """Convert raw criteria text into numbered lines.

    Mirrors ``trialgpt_matching/TrialGPT.py :: parse_criteria()``.
    """
    output = ""
    criteria = criteria_text.split("\n\n")
    idx = 0
    for criterion in criteria:
        criterion = criterion.strip()
        if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
            continue
        if len(criterion) < 5:
            continue
        output += f"{idx}. {criterion}\n"
        idx += 1
    return output


def format_trial_for_prompt(trial_info: Dict[str, Any], inc_exc: str) -> str:
    """Format a trial dict for inclusion in a matching prompt.

    Mirrors ``trialgpt_matching/TrialGPT.py :: print_trial()``.
    """
    diseases = trial_info.get("diseases_list", [])
    if isinstance(diseases, str):
        diseases = [diseases]
    drugs = trial_info.get("drugs_list", [])
    if isinstance(drugs, str):
        drugs = [drugs]

    trial = f"Title: {trial_info.get('brief_title', '')}\n"
    trial += f"Target diseases: {', '.join(diseases)}\n"
    trial += f"Interventions: {', '.join(drugs)}\n"
    trial += f"Summary: {trial_info.get('brief_summary', '')}\n"

    if inc_exc == "inclusion":
        trial += "Inclusion criteria:\n %s\n" % parse_criteria(
            trial_info.get("inclusion_criteria", "")
        )
    elif inc_exc == "exclusion":
        trial += "Exclusion criteria:\n %s\n" % parse_criteria(
            trial_info.get("exclusion_criteria", "")
        )
    return trial


def number_patient_sentences(patient_text: str) -> str:
    """Add sentence IDs to patient text (TrialGPT convention).

    Imports nltk sentence tokenizer; falls back to simple split if not available.
    """
    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(patient_text)
    except ImportError:
        # Simple fallback: split on period + space
        import re
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', patient_text) if s.strip()]

    # TrialGPT convention: append a compliance sentence
    sents.append(
        "The patient will provide informed consent, and will comply "
        "with the trial protocol without any practical issues."
    )
    return "\n".join(f"{idx}. {sent}" for idx, sent in enumerate(sents))


# ======================================================================
# PROMPT 1: KEYWORD GENERATION
# MODEL: Gemini 2.5 Pro
# RATIONALE: Needs reasoning to prioritise conditions and generate
#   diverse search terms from complex clinical text.  MedGemma's
#   4B-param model lacks the vocabulary breadth and ranking ability
#   needed for effective retrieval keyword generation.
# ======================================================================
KEYWORD_GENERATION_SYSTEM = (
    "You are a helpful assistant and your task is to help search relevant "
    "clinical trials for a given patient description. Please first summarize "
    "the main medical problems of the patient. Then generate up to 32 key "
    "conditions for searching relevant clinical trials for this patient. "
    "The key condition list should be ranked by priority.\n\n"
    "If imaging findings are provided, include imaging-derived conditions "
    '(e.g., "measurable disease", "brain metastases on MRI", '
    '"pleural effusion") when clinically relevant.\n\n'
    'Please output only a JSON dict formatted as '
    'Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'
)


def build_keyword_generation_prompt(patient_text: str) -> str:
    """Build the user prompt for keyword generation."""
    return (
        f"Here is the patient description:\n{patient_text}\n\n"
        "JSON output:"
    )


# ======================================================================
# PROMPT 2: CRITERION-LEVEL MATCHING
# MODEL: Gemini 2.5 Pro
# RATIONALE: Core task requiring nuanced logical reasoning over each
#   inclusion/exclusion criterion vs. patient evidence.  Gemini's
#   chain-of-thought reasoning and large context window are essential.
#   MedGemma cannot handle this structured multi-criterion analysis
#   within its limited output budget.
#
# ENHANCEMENT OVER TRIALGPT: Added 4th element ``imaging_relevant``
#   (bool) to flag criteria that could benefit from direct image review.
# ======================================================================
def build_matching_system_prompt(inc_exc: str) -> str:
    """Build the system prompt for criterion-level matching.

    Adapted from ``trialgpt_matching/TrialGPT.py :: get_matching_prompt()``.
    """
    prompt = (
        f"You are a helpful assistant for clinical trial recruitment. "
        f"Your task is to compare a given patient note and the {inc_exc} "
        f"criteria of a clinical trial to determine the patient's "
        f"eligibility at the criterion level.\n"
    )

    if inc_exc == "inclusion":
        prompt += (
            "The factors that allow someone to participate in a clinical "
            "study are called inclusion criteria. They are based on "
            "characteristics such as age, gender, the type and stage of a "
            "disease, previous treatment history, and other medical "
            "conditions.\n"
        )
    elif inc_exc == "exclusion":
        prompt += (
            "The factors that disqualify someone from participating are "
            "called exclusion criteria. They are based on characteristics "
            "such as age, gender, the type and stage of a disease, previous "
            "treatment history, and other medical conditions.\n"
        )

    prompt += (
        f"You should check the {inc_exc} criteria one-by-one, and output "
        f"the following four elements for each criterion:\n"
    )

    prompt += (
        f"\tElement 1. For each {inc_exc} criterion, briefly generate your "
        "reasoning process: First, judge whether the criterion is not "
        "applicable (not very common), where the patient does not meet the "
        "premise of the criterion. Then, check if the patient note contains "
        "direct evidence. If so, judge whether the patient meets or does not "
        "meet the criterion. If there is no direct evidence, try to infer "
        "from existing evidence, and answer one question: If the criterion "
        "is true, is it possible that a good patient note will miss such "
        "information? If impossible, then you can assume that the criterion "
        "is not true. Otherwise, there is not enough information.\n"
    )
    prompt += (
        "\tElement 2. If there is relevant information, you must generate a "
        "list of relevant sentence IDs in the patient note. If there is no "
        "relevant information, you must annotate an empty list.\n"
    )
    prompt += f"\tElement 3. Classify the patient eligibility for this specific {inc_exc} criterion: "

    if inc_exc == "inclusion":
        prompt += (
            'the label must be chosen from '
            '{"not applicable", "not enough information", "included", "not included"}. '
            '"not applicable" should only be used for criteria that are not applicable '
            'to the patient. "not enough information" should be used where the patient '
            "note does not contain sufficient information for making the classification. "
            'Try to use as few "not enough information" as possible because if the note '
            "does not mention a medically important fact, you can assume that the fact "
            'is not true for the patient. "included" denotes that the patient meets '
            'the inclusion criterion, while "not included" means the reverse.\n'
        )
    elif inc_exc == "exclusion":
        prompt += (
            'the label must be chosen from '
            '{"not applicable", "not enough information", "excluded", "not excluded"}. '
            '"not applicable" should only be used for criteria that are not applicable '
            'to the patient. "not enough information" should be used where the patient '
            "note does not contain sufficient information for making the classification. "
            'Try to use as few "not enough information" as possible because if the note '
            "does not mention a medically important fact, you can assume that the fact "
            'is not true for the patient. "excluded" denotes that the patient meets the '
            'exclusion criterion and should be excluded in the trial, while "not excluded" '
            "means the reverse.\n"
        )

    prompt += (
        "\tElement 4. A boolean indicating whether this criterion is "
        "imaging-relevant — i.e., whether evaluating it could benefit from "
        "direct review of the patient's medical images (CT, MRI, X-ray). "
        'Examples: "measurable disease per RECIST", "brain metastases", '
        '"tumor size ≥ 2cm". Set true only for criteria that genuinely '
        "require or could benefit from image interpretation.\n"
    )

    prompt += (
        "\nYou should output only a JSON dict exactly formatted as: "
        "dict{str(criterion_number): list[str(element_1_brief_reasoning), "
        "list[int(element_2_sentence_id)], str(element_3_eligibility_label), "
        "bool(element_4_imaging_relevant)]}."
    )

    return prompt


def build_matching_user_prompt(
    trial_info: Dict[str, Any],
    inc_exc: str,
    patient_text: str,
) -> str:
    """Build the user prompt for criterion-level matching."""
    numbered_patient = number_patient_sentences(patient_text)
    trial_str = format_trial_for_prompt(trial_info, inc_exc)
    return (
        f"Here is the patient note, each sentence is led by a sentence_id:\n"
        f"{numbered_patient}\n\n"
        f"Here is the clinical trial:\n{trial_str}\n\n"
        f"Plain JSON output:"
    )


# ======================================================================
# PROMPT 3: IMAGE CRITERION VALIDATION
# MODEL: MedGemma (medgemma-1-5-4b-it-hae)
# RATIONALE: Purpose-built medical VLM fine-tuned on radiology images.
#   For criteria requiring imaging assessment (e.g., "measurable disease
#   per RECIST 1.1", tumor staging), MedGemma's domain-specific training
#   provides more accurate interpretation than Gemini's general multimodal
#   capabilities.
#   FALLBACK: Gemini 2.5 Pro multimodal if MedGemma is unavailable
#   or encounters CUDA OOM.
# ======================================================================
IMAGE_CRITERION_VALIDATION_SYSTEM = (
    "You are a specialist radiologist AI. You are given a specific clinical "
    "trial eligibility criterion that requires imaging assessment, along with "
    "the patient's medical images.\n\n"
    "Your task is to evaluate whether the criterion is met based on your "
    "direct analysis of the provided images.\n\n"
    "For each criterion, output a JSON object with:\n"
    '  "criterion": the criterion text\n'
    '  "image_assessment": your detailed radiological assessment\n'
    '  "revised_label": one of "met", "not met", "inconclusive", or null '
    "(if images are insufficient)\n"
    '  "confidence": a float 0.0-1.0 indicating your confidence\n\n'
    "Be precise and cite specific imaging findings. If the images do not "
    "provide enough information to assess the criterion, set revised_label "
    "to null and explain why."
)


def build_image_criterion_prompt(
    criteria: List[Dict[str, str]],
    clinical_context: str = "",
) -> str:
    """Build user prompt for MedGemma imaging criterion validation."""
    parts = []
    if clinical_context:
        parts.append(f"Clinical context: {clinical_context}\n")
    parts.append("Criteria to evaluate against the provided images:\n")
    for i, crit in enumerate(criteria):
        parts.append(f"  {i+1}. {crit.get('criterion', crit.get('text', ''))}")
    parts.append("\nPlain JSON output (array of objects, one per criterion):")
    return "\n".join(parts)


# ======================================================================
# PROMPT 4: IMAGE ANALYSIS (general)
# MODEL: MedGemma  (with Gemini 2.5 Pro fallback)
# RATIONALE: Same as Prompt 3 — MedGemma's radiology-specific training
#   makes it the best choice for general image interpretation.
# ======================================================================
IMAGE_ANALYSIS_SYSTEM = (
    "You are a specialist radiologist AI. Examine the provided medical "
    "images and produce a structured JSON report:\n\n"
    '  "modality": imaging modality (CT, MRI, X-ray, etc.)\n'
    '  "body_part": anatomical region\n'
    '  "findings": array of distinct imaging findings\n'
    '  "summary": brief narrative summary\n\n'
    "Be specific about locations, sizes, and abnormalities observed. "
    "Do NOT infer findings — only report what is directly visible."
)


# ======================================================================
# PROMPT 5: TRIAL-LEVEL AGGREGATION
# MODEL: Gemini 2.5 Pro
# RATIONALE: Pure reasoning task — synthesise criterion-level results
#   into holistic relevance (R) and eligibility (E) scores.  Requires
#   the ability to weigh contradictory criteria, handle partial evidence,
#   and produce calibrated numeric scores.  MedGemma's limited reasoning
#   capacity makes it unsuitable.
# ======================================================================
def build_aggregation_system_prompt() -> str:
    """System prompt for TrialGPT-Ranking aggregation.

    Adapted from ``trialgpt_ranking/TrialGPT.py :: convert_pred_to_prompt()``.
    """
    prompt = (
        "You are a helpful assistant for clinical trial recruitment. "
        "You will be given a patient note, a clinical trial, and the "
        "patient eligibility predictions for each criterion.\n"
        "Your task is to output two scores, a relevance score (R) and "
        "an eligibility score (E), between the patient and the clinical "
        "trial.\n"
    )
    prompt += (
        "First explain the consideration for determining patient-trial "
        "relevance. Predict the relevance score R (0~100), which represents "
        "the overall relevance between the patient and the clinical trial. "
        "R=0 denotes the patient is totally irrelevant to the clinical "
        "trial, and R=100 denotes the patient is exactly relevant to the "
        "clinical trial.\n"
    )
    prompt += (
        "Then explain the consideration for determining patient-trial "
        "eligibility. Predict the eligibility score E (-R~R), which "
        "represents the patient's eligibility to the clinical trial. "
        "Note that -R <= E <= R (the absolute value of eligibility cannot "
        "be higher than the relevance), where E=-R denotes that the patient "
        "is ineligible (not included by any inclusion criteria, or excluded "
        "by all exclusion criteria), E=R denotes that the patient is "
        "eligible (included by all inclusion criteria, and not excluded by "
        "any exclusion criteria), E=0 denotes the patient is neutral "
        "(i.e., no relevant information for all inclusion and exclusion "
        "criteria).\n"
    )
    prompt += (
        'Please output a JSON dict formatted as '
        'Dict{"relevance_explanation": Str, "relevance_score_R": Float, '
        '"eligibility_explanation": Str, "eligibility_score_E": Float}.'
    )
    return prompt


def build_aggregation_user_prompt(
    patient_text: str,
    trial_info: Dict[str, Any],
    criterion_predictions_str: str,
    image_analysis_summary: str = "",
) -> str:
    """Build user prompt for aggregation."""
    diseases = trial_info.get("diseases_list", [])
    if isinstance(diseases, str):
        diseases = [diseases]

    trial_str = f"Title: {trial_info.get('brief_title', '')}\n"
    trial_str += f"Target conditions: {', '.join(diseases)}\n"
    trial_str += f"Summary: {trial_info.get('brief_summary', '')}"

    user_prompt = f"Here is the patient note:\n{patient_text}\n\n"
    user_prompt += f"Here is the clinical trial description:\n{trial_str}\n\n"
    user_prompt += (
        f"Here are the criterion-level eligibility predictions:\n"
        f"{criterion_predictions_str}\n\n"
    )
    if image_analysis_summary:
        user_prompt += (
            f"Additional imaging analysis findings:\n"
            f"{image_analysis_summary}\n\n"
        )
    user_prompt += "Plain JSON output:"
    return user_prompt


def format_criterion_predictions(
    matching_result: Dict[str, Any],
    trial_info: Dict[str, Any],
) -> str:
    """Convert criterion predictions to readable string for aggregation.

    Mirrors ``trialgpt_ranking/TrialGPT.py :: convert_criteria_pred_to_string()``.
    """
    output = ""
    for inc_exc in ["inclusion", "exclusion"]:
        idx2criterion: Dict[str, str] = {}
        criteria_text = trial_info.get(f"{inc_exc}_criteria", "")
        criteria = criteria_text.split("\n\n")
        idx = 0
        for criterion in criteria:
            criterion = criterion.strip()
            if "inclusion criteria" in criterion.lower() or "exclusion criteria" in criterion.lower():
                continue
            if len(criterion) < 5:
                continue
            idx2criterion[str(idx)] = criterion
            idx += 1

        predictions = matching_result.get(inc_exc, {})
        for pred_idx, (criterion_idx, preds) in enumerate(predictions.items()):
            if criterion_idx not in idx2criterion:
                continue
            criterion = idx2criterion[criterion_idx]

            if not isinstance(preds, list) or len(preds) < 3:
                continue

            output += f"{inc_exc} criterion {pred_idx}: {criterion}\n"
            output += f"\tPatient relevance: {preds[0]}\n"
            if isinstance(preds[1], list) and len(preds[1]) > 0:
                output += f"\tEvident sentences: {preds[1]}\n"
            output += f"\tPatient eligibility: {preds[2]}\n"

    return output


# ======================================================================
# PROMPT 6: FEEDBACK GENERATION
# MODEL: Gemini 2.5 Pro
# RATIONALE: Identifying medically-relevant missing data requires
#   clinical reasoning — which missing fields would actually change
#   eligibility? Gemini's reasoning + long-context lets it cross-
#   reference missing info against specific trial criteria.
# ======================================================================
FEEDBACK_GENERATION_SYSTEM = (
    "You are a clinical trial matching specialist. You have completed an "
    "initial eligibility assessment for a patient against candidate trials, "
    "but some criteria could not be evaluated due to missing patient "
    "information.\n\n"
    "Your task is to identify the MOST IMPACTFUL missing information — "
    "fields where 'not enough information' labels would FLIP the eligibility "
    "decision if resolved. Generate specific, patient-friendly questions.\n\n"
    "Output a JSON array of objects, each with:\n"
    '  "field": the clinical field name (e.g., "EGFR mutation status")\n'
    '  "question": a patient-friendly question to ask\n'
    '  "reason": why this information matters for trial eligibility\n'
    '  "source_criteria": array of trial criteria texts that need this info\n'
    '  "priority": "high", "medium", or "low"\n\n'
    "Rank by priority: high = would change eligibility for multiple trials, "
    "medium = affects one trial, low = helpful but not decisive.\n"
    "Limit to at most 8 items. Focus on information the patient could "
    "reasonably provide or request from their physician."
)


def build_feedback_user_prompt(
    patient_summary: str,
    matching_summaries: List[Dict[str, str]],
) -> str:
    """Build user prompt for feedback generation.

    Parameters
    ----------
    patient_summary : str
        Brief summary of the patient's profile.
    matching_summaries : list of dicts
        Each dict: {"trial": trial title, "not_enough_info_criteria": [criteria texts]}
    """
    parts = [f"Patient summary:\n{patient_summary}\n"]
    parts.append("Criteria with 'not enough information' across candidate trials:\n")
    for ms in matching_summaries:
        parts.append(f"\nTrial: {ms['trial']}")
        for i, crit in enumerate(ms.get("not_enough_info_criteria", []), 1):
            parts.append(f"  {i}. {crit}")
    parts.append("\nJSON output:")
    return "\n".join(parts)


# ======================================================================
# PROMPT 7: PATIENT PROFILE EXTRACTION
# MODEL: Gemini 2.5 Pro
# RATIONALE: Long-context reasoning over multi-section EHR + structured
#   JSON output matching _TRIAL_PROFILE_SCHEMA.  Produces the rich
#   structured profile used by the existing medgemma_gui/app.py system.
#   MedGemma's 4B params and limited output budget cannot handle this
#   complex structured extraction reliably.
# ======================================================================
PROFILE_EXTRACTION_SYSTEM = (
    "You are a clinical data specialist preparing a structured patient "
    "profile for clinical trial eligibility matching. You have been "
    "provided with the patient's EHR text (clinical history and "
    "examination findings).\n\n"
    "Return ONLY a single valid JSON object with exactly these four "
    "top-level keys:\n\n"
    '  "topic_id"    : string — use the provided patient ID\n'
    '  "profile_text": string — a structured free-text summary using '
    "exactly these six Markdown headings in order: "
    '"## Clinical History", "## Physical Exam", "## Imaging Findings", '
    '"## Assessment & Plan", "## Demographics", "## Missing Info"\n'
    '  "key_facts"   : array of objects with keys: '
    '"field", "value", "evidence_span", "required", "notes"\n'
    '  "ambiguities" : array of strings\n\n'
    "Required key_fact fields: primary_diagnosis, demographics, "
    "imaging_findings, key_findings, missing_info.\n\n"
    "STRICT: set value to null if no evidence exists. Do NOT fabricate."
)


# ======================================================================
# ORCHESTRATOR SYSTEM PROMPT
# MODEL: Gemini 2.5 Pro (function-calling mode)
# RATIONALE: Only model with native function-calling; needs planning
#   capability to decide tool invocation order.
# ======================================================================
ORCHESTRATOR_SYSTEM = (
    "You are an expert clinical trial matching agent. Your goal is to find "
    "the most suitable clinical trials for a patient and assess their "
    "eligibility.\n\n"
    "You have access to the following tools:\n"
    "- extract_patient_profile: Extract a structured profile from EHR text\n"
    "- analyze_images: Analyze medical images (CT/MRI) using a specialist radiology AI\n"
    "- generate_search_keywords: Generate search keywords for trial retrieval\n"
    "- search_trials: Search the clinical trial corpus for matching trials\n"
    "- match_criteria: Evaluate patient eligibility against one trial's criteria\n"
    "- validate_imaging_criteria: Use radiology AI to validate imaging-related criteria\n"
    "- aggregate_and_score: Produce trial-level relevance and eligibility scores\n"
    "- identify_missing_info: Identify impactful missing patient information\n\n"
    "WORKFLOW:\n"
    "1. First, extract the patient profile from the EHR text.\n"
    "2. If medical images are available, analyze them.\n"
    "3. Generate search keywords based on the patient profile.\n"
    "4. Search for candidate clinical trials.\n"
    "5. For each candidate trial, evaluate inclusion AND exclusion criteria.\n"
    "6. If imaging-relevant criteria are flagged AND images are available, "
    "validate them with the radiology AI.\n"
    "7. Aggregate criterion-level results into trial-level scores.\n"
    "8. Identify any missing patient information that could improve matching.\n"
    "9. Return the final ranked list of trials with eligibility assessments.\n\n"
    "IMPORTANT:\n"
    "- Always extract the profile before searching.\n"
    "- Always match criteria before aggregating.\n"
    "- Process the top candidate trials (not all of them).\n"
    "- Always check for missing info last.\n"
    "- Be thorough but efficient — minimize unnecessary tool calls."
)
