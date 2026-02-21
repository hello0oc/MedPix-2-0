from __future__ import annotations


REQUIRED_EXPLAIN_KEYS = {"plain_summary", "image_findings", "image_conclusion", "next_steps"}
REQUIRED_DIAGNOSIS_KEYS = {"diagnosis", "confidence", "rationale", "key_findings", "differential"}
ALLOWED_CONFIDENCE = {"high", "moderate", "low"}

REQUIRED_TRIAL_PROFILE_KEYS = {"topic_id", "profile_text", "key_facts", "ambiguities"}
REQUIRED_KEY_FACT_KEYS = {"field", "value", "evidence_span", "required", "notes"}


def _validate_explain_payload(payload: dict) -> bool:
    if set(payload.keys()) != REQUIRED_EXPLAIN_KEYS:
        return False
    return all(isinstance(payload[k], str) for k in REQUIRED_EXPLAIN_KEYS)


def _validate_diagnosis_payload(payload: dict) -> bool:
    if set(payload.keys()) != REQUIRED_DIAGNOSIS_KEYS:
        return False
    if not all(isinstance(payload[k], str) for k in REQUIRED_DIAGNOSIS_KEYS):
        return False
    return payload["confidence"].strip().lower() in ALLOWED_CONFIDENCE


def _validate_trial_profile(payload: dict) -> tuple[bool, str]:
    """Validate a trial profile dict.  Returns (ok, reason)."""
    if not REQUIRED_TRIAL_PROFILE_KEYS.issubset(payload.keys()):
        missing = REQUIRED_TRIAL_PROFILE_KEYS - payload.keys()
        return False, f"missing top-level keys: {missing}"
    if not isinstance(payload["key_facts"], list):
        return False, "key_facts must be a list"
    for kf in payload["key_facts"]:
        if not REQUIRED_KEY_FACT_KEYS.issubset(kf.keys()):
            missing = REQUIRED_KEY_FACT_KEYS - kf.keys()
            return False, f"key_fact missing keys: {missing}"
        # Filler strings must not appear; validate_and_coerce_trial_profile coerces
        # them to null — raw model output that bypasses post-processing should fail.
        val = kf.get("value")
        if isinstance(val, str) and val.strip().lower() in {
            "", "unknown", "n/a", "na", "not provided", "not available",
            "none", "null", "unspecified", "not stated", "not documented",
        }:
            return False, f"key_fact '{kf.get('field')}' has filler string value '{val}'"
    # Check that required null fields are in missing_info
    missing_info_values: list = []
    for kf in payload["key_facts"]:
        if kf.get("field") == "missing_info":
            missing_info_values = kf.get("value") or []
            break
    for kf in payload["key_facts"]:
        if kf.get("required") and kf.get("value") is None and kf.get("field") != "missing_info":
            if kf.get("field") not in missing_info_values:
                return False, (
                    f"required field '{kf.get('field')}' is null but not listed in missing_info"
                )
    return True, ""


def test_prompt_contract_mentions_required_explain_keys(app_module):
    for prompt in (app_module.MEDGEMMA_EXPLAIN_PROMPT, app_module.GEMINI_EXPLAIN_PROMPT):
        for key in REQUIRED_EXPLAIN_KEYS:
            assert key in prompt


def test_prompt_contract_mentions_required_diagnosis_keys(app_module):
    for prompt in (app_module.MEDGEMMA_DIAGNOSIS_PROMPT, app_module.GEMINI_DIAGNOSIS_PROMPT):
        for key in REQUIRED_DIAGNOSIS_KEYS:
            assert key in prompt


def test_valid_diagnosis_json_contract_passes(app_module):
    raw = '{"diagnosis":"Pneumonia","confidence":"High","rationale":"Findings support diagnosis.","key_findings":"fever, cough","differential":"asthma, COPD"}'
    payload = app_module.parse_json_object(raw)
    assert _validate_diagnosis_payload(payload)


def test_mutated_diagnosis_json_contract_fails(app_module):
    # Simulates schema drift (renamed and missing keys).
    raw = '{"final_diagnosis":"Pneumonia","confidence_level":"High","rationale":"x"}'
    payload = app_module.parse_json_object(raw)
    assert not _validate_diagnosis_payload(payload)


def test_valid_explain_json_contract_passes(app_module):
    raw = '{"plain_summary":"Summary","image_findings":"None","image_conclusion":"None","next_steps":"Observe"}'
    payload = app_module.parse_json_object(raw)
    assert _validate_explain_payload(payload)


def test_mutated_explain_json_contract_fails(app_module):
    raw = '{"summary":"Summary","next_step":"Observe"}'
    payload = app_module.parse_json_object(raw)
    assert not _validate_explain_payload(payload)


# ── Trial profile contract tests ────────────────────────────────────────────

_VALID_TRIAL_PROFILE = {
    "topic_id": "mpx1009",
    "profile_text": "## Clinical History\nPatient history.\n## Imaging Findings\nLarge mass.",
    "key_facts": [
        {
            "field": "primary_diagnosis",
            "value": "Stage IV NSCLC",
            "evidence_span": "biopsy confirmed NSCLC",
            "required": True,
            "notes": None,
        },
        {
            "field": "demographics",
            "value": {"age": "58", "sex": "female"},
            "evidence_span": "58-year-old female",
            "required": True,
            "notes": None,
        },
        {
            "field": "imaging_findings",
            "value": ["Large apical mass with chest wall invasion"],
            "evidence_span": None,
            "required": True,
            "notes": "imaging_source: direct_image_analysis",
        },
        {
            "field": "key_findings",
            "value": ["Transbronchial biopsy: NSCLC", "Left apical chest tenderness"],
            "evidence_span": None,
            "required": True,
            "notes": None,
        },
        {
            "field": "missing_info",
            "value": ["smoking pack-years detail"],
            "evidence_span": None,
            "required": False,
            "notes": None,
        },
    ],
    "ambiguities": [],
}


def test_valid_trial_profile_contract_passes(app_module):
    ok, reason = _validate_trial_profile(_VALID_TRIAL_PROFILE)
    assert ok, reason


def test_trial_profile_missing_key_facts_fails(app_module):
    payload = {"topic_id": "x", "profile_text": "p", "ambiguities": []}
    ok, reason = _validate_trial_profile(payload)
    assert not ok
    assert "key_facts" in reason


def test_trial_profile_imaging_findings_null_missing_from_missing_info_fails(app_module):
    import copy
    payload = copy.deepcopy(_VALID_TRIAL_PROFILE)
    # Set imaging_findings null without adding it to missing_info
    for kf in payload["key_facts"]:
        if kf["field"] == "imaging_findings":
            kf["value"] = None
    # Remove imaging_findings from missing_info if present
    for kf in payload["key_facts"]:
        if kf["field"] == "missing_info":
            kf["value"] = [v for v in kf["value"] if v != "imaging_findings"]
    ok, reason = _validate_trial_profile(payload)
    assert not ok
    assert "imaging_findings" in reason


def test_trial_profile_null_primary_diagnosis_in_missing_info_passes(app_module):
    import copy
    payload = copy.deepcopy(_VALID_TRIAL_PROFILE)
    for kf in payload["key_facts"]:
        if kf["field"] == "primary_diagnosis":
            kf["value"] = None
    for kf in payload["key_facts"]:
        if kf["field"] == "missing_info":
            kf["value"] = ["primary_diagnosis", "smoking pack-years detail"]
    ok, reason = _validate_trial_profile(payload)
    assert ok, reason


def test_trial_profile_filler_string_value_fails(app_module):
    import copy
    payload = copy.deepcopy(_VALID_TRIAL_PROFILE)
    for kf in payload["key_facts"]:
        if kf["field"] == "primary_diagnosis":
            kf["value"] = "unknown"  # filler — should fail pre-coercion
    ok, reason = _validate_trial_profile(payload)
    assert not ok
    assert "filler string" in reason


def test_trial_profile_prompt_mentions_schema_keys(app_module):
    for prompt in (app_module.TRIAL_PROFILE_PROMPT, app_module.TRIAL_PROFILE_PROMPT_TEXT_ONLY):
        for key in ("topic_id", "profile_text", "key_facts", "ambiguities"):
            assert key in prompt
        assert "imaging_source" in prompt
        assert "null" in prompt  # no-hallucination rule
