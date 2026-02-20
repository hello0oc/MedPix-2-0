from __future__ import annotations

from pathlib import Path


def test_history_richness_score_weights_and_placeholders(app_module):
    case = {
        "history": "abcde",
        "findings": "1234",
        "exam": "zz",
        "discussion": "n/a",
    }
    score = app_module.history_richness_score(case)
    assert score == (5 * 3) + (4 * 2) + 2


def test_parse_json_object_with_prefix_text(app_module):
    text = "Model output follows:\n\n{\"diagnosis\": \"Pneumonia\", \"confidence\": \"High\"}\nextra"
    obj = app_module.parse_json_object(text)
    assert obj.get("diagnosis") == "Pneumonia"
    assert obj.get("confidence") == "High"


def test_parse_markdown_fields_prefers_final(app_module):
    text = (
        "**Diagnosis:** Old diagnosis\n"
        "**Final Diagnosis:** New diagnosis\n"
        "**Rationale:** Why"
    )
    parsed = app_module.parse_markdown_fields(text)
    assert parsed["diagnosis"] == "Old diagnosis"
    assert parsed["final_diagnosis"] == "New diagnosis"
    assert parsed["rationale"] == "Why"


def test_medgemma_token_budget_caps_heavy_inputs(app_module):
    budget = app_module.medgemma_token_budget(
        requested_tokens=4096,
        ehr_text="x" * 2500,
        image_count=3,
        use_images=True,
    )
    assert budget == 768


def test_resolve_credential_prefers_secrets_over_env(app_module, st_stub, monkeypatch):
    st_stub.secrets["HF_TOKEN"] = "secret-token"
    monkeypatch.setenv("HF_TOKEN", "env-token")
    value, source, key = app_module.resolve_credential(("HF_TOKEN",))
    assert (value, source, key) == ("secret-token", "secrets.toml", "HF_TOKEN")


def test_resolve_credential_uses_env_when_secret_missing(app_module, st_stub, monkeypatch):
    st_stub.secrets.clear()
    monkeypatch.setenv("HF_TOKEN", "env-token")
    value, source, key = app_module.resolve_credential(("HF_TOKEN",))
    assert (value, source, key) == ("env-token", "environment", "HF_TOKEN")


def test_build_user_content_text_only_when_images_disabled(app_module):
    content = app_module.build_user_content(
        ehr_text="clinical text",
        image_paths=[Path("a.png")],
        use_images=False,
        max_images=1,
    )
    assert content == "clinical text"


def test_build_user_content_multimodal_blocks(app_module, monkeypatch, tmp_path):
    p = tmp_path / "image.png"
    p.write_bytes(b"fake")
    monkeypatch.setattr(app_module, "image_to_data_url", lambda _path: "data:image/png;base64,AAA")

    content = app_module.build_user_content(
        ehr_text="ehr",
        image_paths=[p],
        use_images=True,
        max_images=1,
    )

    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "ehr"}
    assert content[1]["type"] == "image_url"


def test_get_resolved_images_fallback_for_synthea(app_module, monkeypatch, tmp_path):
    fallback_dir = tmp_path / "mri_slices"
    fallback_dir.mkdir()
    for name in app_module.SYNTHETIC_FALLBACK_IMAGE_NAMES:
        (fallback_dir / name).write_bytes(b"x")

    monkeypatch.setattr(app_module, "SYNTHETIC_MRI_SLICES_DIR", fallback_dir)
    monkeypatch.setattr(app_module, "collect_image_paths", lambda _case: ["missing.dcm"])
    monkeypatch.setattr(app_module, "resolve_image_path", lambda _root, _raw: None)

    case = {"dataset_source": "Synthea coherent zip", "images": [{"file_path": "archive.zip::x.dcm"}]}
    paths, is_fallback = app_module.get_resolved_images(case)
    assert is_fallback is True
    assert len(paths) == len(app_module.SYNTHETIC_FALLBACK_IMAGE_NAMES)


def test_call_medgemma_retries_and_degrades_on_oom(app_module, monkeypatch):
    calls = []

    def fake_endpoint_chat_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return "ok"

    monkeypatch.setattr(app_module, "endpoint_chat_completion", fake_endpoint_chat_completion)

    output = app_module.call_medgemma(
        endpoint_url="http://example",
        token="t",
        model_id="m",
        system_prompt="sys",
        ehr_text="x" * 500,
        image_paths=[Path("a.png"), Path("b.png")],
        use_images=True,
        max_images=2,
        max_tokens=1000,
        timeout=5,
        retries=0,
    )

    assert output == "ok"
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 1000
    assert calls[1]["max_tokens"] == 700


def test_extract_gemini_text_raises_on_max_tokens(app_module):
    class Part:
        def __init__(self, text=None, thought=False):
            self.text = text
            self.thought = thought

    class Content:
        def __init__(self, parts):
            self.parts = parts

    class Candidate:
        def __init__(self, parts, finish_reason=None):
            self.content = Content(parts)
            self.finish_reason = finish_reason

    class Response:
        def __init__(self, candidates, text=None):
            self.candidates = candidates
            self.text = text

    response = Response(
        candidates=[Candidate(parts=[Part(text="thinking", thought=True)], finish_reason="MAX_TOKENS")],
        text=None,
    )

    try:
        app_module._extract_gemini_text(response)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "ran out of output tokens" in str(exc)


def test_load_pmc_cases_normalization(app_module, tmp_path):
    csv_path = tmp_path / "pmc.csv"
    csv_path.write_text(
        "patient_uid,title,patient,age,gender,PMID,file_path,relevant_articles,similar_patients\n"
        "p1,Case A,history text,35,F,12345,doc.txt,article1,sim1\n",
        encoding="utf-8",
    )
    rows = app_module.load_pmc_cases(str(csv_path))
    assert len(rows) == 1
    case = rows[0]
    assert case["uid"] == "p1"
    assert "Age: 35" in case["exam"]
    assert "Related articles: article1" in case["findings"]
