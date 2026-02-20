# MedGUI — Streamlit App

`medgemma_gui/app.py` is the main entry point for MedGUI. It provides a unified browser for three medical datasets and runs AI inference using MedGemma or Gemini 2.5 Pro.

## Features

- Browse **three dataset sources** (selectable in sidebar):
  - `patient-ehr-image-dataset/full_dataset.jsonl` — 671 MedPix EHR + image cases
  - `PMCpatient/PMC-Patients-sample-1000.csv` — 1 000 PubMed clinical case reports
  - `Synthetic/synthetic_ehr_image_dataset.jsonl` — Synthea synthetic EHR + DICOM
- **Richness ranking** — sort patients by clinical text completeness
- **Image preview** — inline viewer for PNG and DICOM images
- **Two AI tasks per patient:**
  - *Explain* — plain-language summary for non-medical readers
  - *Diagnose* — structured clinical result: diagnosis, confidence, rationale, key findings, differential
- **Two LLM backends** (selectable in sidebar):
  - MedGemma — HuggingFace Inference Endpoint
  - Gemini 2.5 Pro — Google AI Studio (thinking model; thinking budget handled automatically)
- **Per-patient session cache** — results persist across patient/model switches

## Requirements

```bash
pip install streamlit pillow google-genai
```

Optional — for DICOM preview of Synthetic cases:

```bash
pip install pydicom numpy
```

## Authentication

Credentials are loaded from `.streamlit/secrets.toml` (preferred) or environment variables.

```toml
# .streamlit/secrets.toml
HF_TOKEN = "hf_..."
GEMINI_API_KEY = "AIza..."
```

Accepted key aliases:

| Service | Keys tried (in order) |
|---|---|
| HuggingFace | `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, `HUGGINGFACE_API_KEY` |
| Google Gemini | `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GENAI_API_KEY` |

The sidebar shows exactly which key and source (secrets.toml / environment) is active.

## Run

From the repository root:

```bash
streamlit run medgemma_gui/app.py
```

Open `http://localhost:8501`. Use the sidebar to select dataset, model, and inference options, then click **Explain** or **Diagnose**.

## Gemini 2.5 Pro — Thinking Model Note

Gemini 2.5 Pro uses internal reasoning tokens ("thinking") that consume the `max_output_tokens` budget. The app automatically adds a separate `thinking_budget` so the visible JSON output is never starved of tokens. You can safely raise the *Max output tokens* slider; the app handles the budget split internally.
