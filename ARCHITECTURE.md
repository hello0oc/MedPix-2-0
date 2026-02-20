# Solution Architecture

## 1) Purpose and Scope
This document explains the repository from three angles:
1. available datasets and their structures,
2. canonical data model used by the app,
3. Streamlit + backend runtime workflow.

The main integration point is the Streamlit application at `medgemma_gui/app.py`.

## 2) Dataset Inventory

### 2.1 MedPix JSONL (primary EHR-image dataset)
- Path: `patient-ehr-image-dataset/full_dataset.jsonl`
- Row characteristics:
  - clinical text fields (`history`, `exam`, `findings`)
  - diagnosis label (`diagnosis`)
  - image references (`images`, `ct_image_ids`, `mri_image_ids`)
- Used directly by the UI loader `load_cases(...)`.

### 2.2 PMC-Patients CSV
- Path: `PMCpatient/PMC-Patients-sample-1000.csv`
- Native fields include:
  - `patient_uid`, `title`, `patient`, `age`, `gender`, `PMID`, `file_path`,
  - `relevant_articles`, `similar_patients`
- Ingested by `load_pmc_cases(...)`, then normalized with `_normalize_pmc_row(...)`.

### 2.3 Synthea Coherent JSONL
- Path: `Synthetic/synthetic_ehr_image_dataset.jsonl`
- Loaded through the same `load_cases(...)` JSONL path when selected in the UI.

## 3) Canonical Case Model (shared downstream contract)
All selected datasets are transformed into a common case dictionary so ranking, prompts, and rendering remain unchanged.

Required/expected keys in downstream workflow:
- identity: `uid`, `title`
- clinical text: `history`, `exam`, `findings`
- optional context: `discussion`, `diagnosis`
- imaging references: `images`, `ct_image_ids`, `mri_image_ids`
- optional metadata: `dataset_source`

### 3.1 Mapping summary
- **MedPix JSONL**: already close to canonical schema, minimal adaptation.
- **PMC CSV**:
  - `patient_uid -> uid`
  - `patient -> history`
  - `age/gender/PMID/file_path -> exam` (composed text)
  - `relevant_articles/similar_patients -> findings` (composed text)
  - diagnosis/images default to empty values for schema consistency.
- **Synthea JSONL**: read as JSONL and consumed under the same canonical keys expected by UI logic.

## 4) Streamlit App Architecture (`medgemma_gui/app.py`)

### 4.1 Functional layers
- **Input layer**
  - sidebar controls:
    - dataset selector (`MedPix JSONL`, `PMC-Patients CSV`, `Synthea Coherent JSONL`)
    - model selector (`MedGemma`, `Gemini`)
    - inference options (image usage, token budget, timeout, retries)
- **Data layer**
  - `load_cases(...)` for JSONL sources
  - `load_pmc_cases(...)` + `_normalize_pmc_row(...)` for PMC CSV
  - `history_richness_score(...)` for optional ranking (richest first)
- **Inference layer**
  - `call_medgemma(...)` for HF endpoint inference
  - `call_gemini(...)` for Google GenAI inference
  - prompt routing by backend:
    - `MEDGEMMA_EXPLAIN_PROMPT`, `MEDGEMMA_DIAGNOSIS_PROMPT`
    - `GEMINI_EXPLAIN_PROMPT`, `GEMINI_DIAGNOSIS_PROMPT`
- **Parsing layer**
  - `parse_json_object(...)` first
  - `parse_markdown_fields(...)` fallback for markdown-style responses
  - `parse_diagnosis(...)` heuristic fallback from benchmark module
- **Presentation layer**
  - tab 1: `Patient Data`
  - tab 2: `AI Insights`
  - diagnosis panel + rationale + supporting findings + differential + raw output expander

### 4.2 Session state and caching strategy
- Outputs are cached per `(model, patient_uid)` in `st.session_state`:
  - explain keys: `explain_<model_key>_<uid>`
  - diagnosis keys: `diagnosis_<model_key>_<uid>`
- Benefit: switching patients/models does not discard previously computed outputs.

### 4.3 Image handling strategy
- `get_resolved_images(...)` resolves on-disk paths from case metadata.
- Preview path:
  - regular image paths rendered directly
  - DICOM preview support via `preview_payload(...)` and `dicom_to_png_bytes(...)`
- Inference path:
  - MedGemma uses data URLs via `build_user_content(...)`
  - Gemini uses PIL image objects when available.

## 5) Backend Utility Dependencies
The Streamlit app imports shared backend utilities from `medgemma_benchmark/run_medgemma_benchmark.py`:
- data helpers: `load_jsonl`, `build_ehr_text`, `collect_image_paths`, `resolve_image_path`
- image helpers: `image_to_data_url`, `dicom_to_png_bytes`
- inference transport: `endpoint_chat_completion`
- post-processing helpers: `sanitize_text`, `truncate_text`, `parse_diagnosis`

This keeps transport/format logic centralized and reduces duplicate implementations.

## 6) End-to-End Runtime Sequence
1. User selects dataset source and model backend in sidebar.
2. App loads source rows and normalizes to canonical case schema.
3. Cases are ranked by richness (optional) and presented in patient selector.
4. App resolves images and builds EHR text context.
5. User triggers explanation or diagnosis.
6. Dispatcher calls selected backend function (`call_medgemma` or `call_gemini`).
7. Response parser extracts structured fields with layered fallback.
8. UI renders structured clinical output and stores it in session cache.

## 7) Extension Guidance
- Add a new dataset by implementing a loader that outputs canonical case keys.
- Keep prompt output contracts stable (JSON keys) to preserve parser/UI compatibility.
- Prefer adding backend-specific logic behind dispatcher functions; avoid UI-layer branching explosion.
- Keep ranking function independent of source schema by scoring canonical text fields only.
