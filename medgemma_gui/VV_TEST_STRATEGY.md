# MedGUI V&V Test Strategy (Phase 1-3)

## Scope

This strategy covers **MedGUI app workflows** implemented in [medgemma_gui/app.py](app.py):

- dataset loading and normalization at UI boundary,
- patient ranking and case text preparation,
- image resolution and synthetic fallback behavior,
- response parsing and structured rendering helpers,
- credential resolution and inference resilience logic.

## Explicit Exclusion

The MedGemma endpoint connectivity benchmark is **excluded from required V&V pass/fail** due to current connection issues:

- [medgemma_benchmark/run_medgemma_benchmark.py](../medgemma_benchmark/run_medgemma_benchmark.py)
- [medgemma_benchmark/output/summary.json](../medgemma_benchmark/output/summary.json)

Connectivity tests may run separately as informational checks when endpoint stability is restored.

## Verification Dimensions

1. **Functional correctness**
   - canonical parsing/normalization helpers return expected schema-compatible values,
   - multimodal request payload composition remains stable.
2. **Robustness / error handling**
   - adaptive retries degrade token/image load for OOM-like failures,
   - parser fallbacks tolerate mixed JSON/markdown model outputs.
3. **Data interface compatibility**
   - PMC CSV row normalization preserves required canonical keys,
   - synthetic image fallback is deterministic when DICOM archive is unavailable.
4. **Security / config hygiene**
   - credential source precedence is deterministic (`secrets.toml` before environment).

## Automated Test Suite (Implemented)

Test file: [tests/test_medgui_vv_offline.py](../tests/test_medgui_vv_offline.py)

- `test_history_richness_score_weights_and_placeholders`
- `test_parse_json_object_with_prefix_text`
- `test_parse_markdown_fields_prefers_final`
- `test_medgemma_token_budget_caps_heavy_inputs`
- `test_resolve_credential_prefers_secrets_over_env`
- `test_resolve_credential_uses_env_when_secret_missing`
- `test_build_user_content_text_only_when_images_disabled`
- `test_build_user_content_multimodal_blocks`
- `test_get_resolved_images_fallback_for_synthea`
- `test_call_medgemma_retries_and_degrades_on_oom`
- `test_extract_gemini_text_raises_on_max_tokens`
- `test_load_pmc_cases_normalization`

Headless mocked UI smoke tests:

- [tests/test_medgui_ui_smoke_mocked.py](../tests/test_medgui_ui_smoke_mocked.py)
   - cached explain/diagnosis render smoke test,
   - missing credential error-path smoke test,
   - dataset switching coverage for MedPix / PMC / Synthea modes,
   - model switching coverage for MedGemma and Gemini inference routing,
   - parser resilience coverage for malformed multi-block diagnosis/explain outputs.

Schema/mutation contract checks:

- [tests/test_model_output_contract.py](../tests/test_model_output_contract.py)
   - prompt schema key contracts pinned,
   - valid/invalid diagnosis payload contract checks,
   - valid/invalid explain payload contract checks.

## CI Gate (Implemented)

Workflow: [.github/workflows/medgui-vv-offline.yml](../.github/workflows/medgui-vv-offline.yml)

- trigger on MedGUI/test changes,
- run offline-only test suite on Python 3.11,
- fail PR on any offline V&V regression.

## Nightly Connectivity Monitoring (Implemented, Non-Blocking)

- Workflow: [.github/workflows/medgui-connectivity-nightly.yml](../.github/workflows/medgui-connectivity-nightly.yml)
- Script: [scripts/nightly_connectivity_check.py](../scripts/nightly_connectivity_check.py)
- Trend summarizer: [scripts/summarize_connectivity_trends.py](../scripts/summarize_connectivity_trends.py)
- Dependency file: [requirements-connectivity.txt](../requirements-connectivity.txt)

Behavior:

- runs on schedule and manual dispatch,
- checks MedGemma and Gemini transport paths when credentials are available,
- writes JSON report artifact,
- archives timestamped reports and publishes trend summaries (JSON + Markdown),
- never blocks merges or releases.

## Execution

Local run:

```bash
pip install -r requirements-test.txt
pytest -q tests/test_medgui_vv_offline.py tests/test_medgui_ui_smoke_mocked.py tests/test_model_output_contract.py
```

## Next Phases

- Add optional strict mode gate for release branches when connectivity stability is restored.
- Add richer parser fuzz cases for edge Unicode/escaped JSON fragments.
