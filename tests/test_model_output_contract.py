from __future__ import annotations


REQUIRED_EXPLAIN_KEYS = {"plain_summary", "image_findings", "image_conclusion", "next_steps"}
REQUIRED_DIAGNOSIS_KEYS = {"diagnosis", "confidence", "rationale", "key_findings", "differential"}
ALLOWED_CONFIDENCE = {"high", "moderate", "low"}


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
