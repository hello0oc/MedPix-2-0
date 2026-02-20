from __future__ import annotations

import pytest


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeStreamlit:
    def __init__(self, radio_map=None, button_sequence=None):
        self.radio_map = radio_map or {}
        self.button_sequence = list(button_sequence or [])
        self.session_state = {}
        self.secrets = {}
        self.sidebar = _Ctx()
        self.errors = []
        self.infos = []
        self.captions = []
        self.markdowns = []
        self.subheaders = []
        self.warnings = []

    def cache_data(self, show_spinner=False):
        def _decorator(fn):
            return fn

        return _decorator

    def set_page_config(self, **kwargs):
        return None

    def markdown(self, value, **kwargs):
        self.markdowns.append(str(value))

    def caption(self, value):
        self.captions.append(str(value))

    def subheader(self, value):
        self.subheaders.append(str(value))

    def info(self, value):
        self.infos.append(str(value))

    def error(self, value):
        self.errors.append(str(value))

    def warning(self, value):
        self.warnings.append(str(value))

    def write(self, value):
        return None

    def code(self, value):
        return None

    def image(self, *args, **kwargs):
        return None

    def divider(self):
        return None

    def stop(self):
        raise RuntimeError("st.stop called")

    def spinner(self, text):
        return _Ctx()

    def columns(self, spec):
        return [_Ctx() for _ in spec]

    def radio(self, label, options, **kwargs):
        return self.radio_map.get(label, options[0])

    def checkbox(self, label, value=False, **kwargs):
        return value

    def slider(self, label, min_value, max_value, value, step=1, **kwargs):
        return value

    def number_input(self, label, min_value, max_value, value, step=1, **kwargs):
        return value

    def button(self, *args, **kwargs):
        if self.button_sequence:
            return self.button_sequence.pop(0)
        return False

    def selectbox(self, label, options, **kwargs):
        return options[0]

    def expander(self, *args, **kwargs):
        return _Ctx()


def _sample_case(uid="u1"):
    return {
        "uid": uid,
        "title": "Sample Case",
        "history": "History text",
        "exam": "Exam text",
        "findings": "Findings text",
        "diagnosis": "",
        "discussion": "",
        "images": [],
    }


def test_main_smoke_renders_cached_results(monkeypatch, app_module):
    fake_st = FakeStreamlit(button_sequence=[False, False])
    fake_st.secrets["HF_TOKEN"] = "token"
    fake_st.session_state["explain_mg_u1"] = (
        '{"plain_summary":"Summary text","image_findings":"","image_conclusion":"","next_steps":"Follow up"}'
    )
    fake_st.session_state["diagnosis_mg_u1"] = (
        '{"diagnosis":"Pneumonia","confidence":"High","rationale":"Because.","key_findings":"A,B","differential":"X,Y"}'
    )

    monkeypatch.setattr(app_module, "st", fake_st)
    monkeypatch.setattr(app_module, "load_cases", lambda _path: [_sample_case()])
    monkeypatch.setattr(app_module, "get_resolved_images", lambda _case: ([], False))

    app_module.main()

    assert "Diagnostic Result" in fake_st.subheaders
    assert "Plain-Language Explanation" in fake_st.subheaders
    assert any("Summary text" in value for value in fake_st.infos)
    assert not any("Request failed:" in value for value in fake_st.errors)


def test_main_smoke_missing_credential_error(monkeypatch, app_module):
    # Explain=False, Diagnose=True to trigger missing credential branch.
    fake_st = FakeStreamlit(button_sequence=[False, True])

    monkeypatch.setattr(app_module, "st", fake_st)
    monkeypatch.setattr(app_module, "resolve_credential", lambda _keys: ("", "", ""))
    monkeypatch.setattr(app_module, "load_cases", lambda _path: [_sample_case()])
    monkeypatch.setattr(app_module, "get_resolved_images", lambda _case: ([], False))

    app_module.main()

    assert any("Missing credential." in value for value in fake_st.errors)


@pytest.mark.parametrize(
    ("dataset_choice", "expected_filename", "expect_pmc_loader"),
    [
        ("MedPix JSONL", "full_dataset.jsonl", False),
        ("PMC-Patients CSV", "PMC-Patients-sample-1000.csv", True),
        ("Synthea Coherent JSONL", "synthetic_ehr_image_dataset.jsonl", False),
    ],
)
def test_main_dataset_switch_uses_correct_loader(
    monkeypatch,
    app_module,
    dataset_choice,
    expected_filename,
    expect_pmc_loader,
):
    fake_st = FakeStreamlit(
        radio_map={
            "Dataset Source": dataset_choice,
            "LLM Backend": "MedGemma (HF Endpoint)",
        },
        button_sequence=[False, False],
    )
    fake_st.secrets["HF_TOKEN"] = "token"

    calls = {"load_cases": [], "load_pmc_cases": []}

    def _load_cases(path):
        calls["load_cases"].append(str(path))
        return [_sample_case("u1")]

    def _load_pmc_cases(path):
        calls["load_pmc_cases"].append(str(path))
        return [_sample_case("p1")]

    monkeypatch.setattr(app_module, "st", fake_st)
    monkeypatch.setattr(app_module, "load_cases", _load_cases)
    monkeypatch.setattr(app_module, "load_pmc_cases", _load_pmc_cases)
    monkeypatch.setattr(app_module, "get_resolved_images", lambda _case: ([], False))

    app_module.main()

    if expect_pmc_loader:
        assert len(calls["load_pmc_cases"]) == 1
        assert len(calls["load_cases"]) == 0
    else:
        assert len(calls["load_cases"]) == 1
        assert len(calls["load_pmc_cases"]) == 0

    assert any(expected_filename in value for value in fake_st.captions)


@pytest.mark.parametrize(
    ("llm_choice", "expected_cache_key", "expected_call"),
    [
        ("MedGemma (HF Endpoint)", "diagnosis_mg_u1", "medgemma"),
        ("Gemini 2.5 Pro", "diagnosis_gf_u1", "gemini"),
    ],
)
def test_main_model_switch_routes_to_correct_backend(
    monkeypatch,
    app_module,
    llm_choice,
    expected_cache_key,
    expected_call,
):
    fake_st = FakeStreamlit(
        radio_map={
            "Dataset Source": "MedPix JSONL",
            "LLM Backend": llm_choice,
        },
        # Explain=False, Diagnose=True
        button_sequence=[False, True],
    )

    mg_calls = []
    gemini_calls = []

    def _resolve_credential(candidate_keys):
        if "HF_TOKEN" in candidate_keys:
            return ("hf-token", "secrets.toml", "HF_TOKEN")
        return ("gemini-token", "secrets.toml", "GEMINI_API_KEY")

    def _call_medgemma(**kwargs):
        mg_calls.append(kwargs)
        return (
            '{"diagnosis":"Pneumonia","confidence":"High",'
            '"rationale":"Because","key_findings":"a,b","differential":"x,y"}'
        )

    def _call_gemini(**kwargs):
        gemini_calls.append(kwargs)
        return (
            '{"diagnosis":"Pneumonia","confidence":"High",'
            '"rationale":"Because","key_findings":"a,b","differential":"x,y"}'
        )

    monkeypatch.setattr(app_module, "st", fake_st)
    monkeypatch.setattr(app_module, "resolve_credential", _resolve_credential)
    monkeypatch.setattr(app_module, "load_cases", lambda _path: [_sample_case("u1")])
    monkeypatch.setattr(app_module, "get_resolved_images", lambda _case: ([], False))
    monkeypatch.setattr(app_module, "call_medgemma", _call_medgemma)
    monkeypatch.setattr(app_module, "call_gemini", _call_gemini)

    app_module.main()

    if expected_call == "medgemma":
        assert len(mg_calls) == 1
        assert len(gemini_calls) == 0
    else:
        assert len(gemini_calls) == 1
        assert len(mg_calls) == 0

    assert expected_cache_key in fake_st.session_state


def test_main_parser_resilience_diagnosis_malformed_multiblock(monkeypatch, app_module):
    fake_st = FakeStreamlit(button_sequence=[False, False])
    fake_st.secrets["HF_TOKEN"] = "token"
    fake_st.session_state["diagnosis_mg_u1"] = (
        "```json\n{invalid json block}\n```\n"
        "**Revised Diagnosis:** Viral syndrome\n"
        "**Final Diagnosis:** Acute appendicitis\n"
        "**Final Rationale:** Localized right-lower-quadrant pain supports appendicitis.\n"
        "**Confidence:** Moderate\n"
        "**Key Findings:** RLQ tenderness, rebound\n"
        "**Differential:** gastroenteritis, mesenteric adenitis"
    )

    monkeypatch.setattr(app_module, "st", fake_st)
    monkeypatch.setattr(app_module, "load_cases", lambda _path: [_sample_case()])
    monkeypatch.setattr(app_module, "get_resolved_images", lambda _case: ([], False))

    app_module.main()

    assert any("Acute appendicitis" in value for value in fake_st.markdowns)
    assert not any("could not be extracted" in value.lower() for value in fake_st.warnings)


def test_main_parser_resilience_explain_malformed_multiblock(monkeypatch, app_module):
    fake_st = FakeStreamlit(button_sequence=[False, False])
    fake_st.secrets["HF_TOKEN"] = "token"
    fake_st.session_state["explain_mg_u1"] = (
        "Some preface\n"
        "{ malformed json payload\n"
        "**Summary:** Old summary\n"
        "**Plain Summary:** Final patient-friendly summary.\n"
        "**Image Findings:** No acute image findings.\n"
        "**Image Conclusion:** Imaging is reassuring.\n"
        "**Next Steps:** Outpatient follow-up."
    )

    monkeypatch.setattr(app_module, "st", fake_st)
    monkeypatch.setattr(app_module, "load_cases", lambda _path: [_sample_case()])
    monkeypatch.setattr(app_module, "get_resolved_images", lambda _case: ([], False))

    app_module.main()

    assert any("Final patient-friendly summary." in value for value in fake_st.infos)
    assert any("Image findings" in value for value in fake_st.markdowns)
