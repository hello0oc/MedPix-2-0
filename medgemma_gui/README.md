# MedGemma GUI

This Streamlit app provides a user-friendly interface to:
- browse and analyze multiple datasets through one unified workflow:
  - `patient-ehr-image-dataset/full_dataset.jsonl` (MedPix JSONL)
  - `PMCpatient/PMC-Patients-sample-1000.csv` (PMC CSV)
  - `Synthetic/synthetic_ehr_image_dataset.jsonl` (Synthea coherent JSONL)
- rank patient records by clinical text richness
- preview linked images (including DICOM preview support where available)
- run two AI tasks:
  - plain-language explanation for non-medical users
  - diagnostic insight with rationale, confidence, key findings, and differential
- switch model backend:
  - MedGemma (HF Endpoint)
  - Gemini 2.5 Pro

## Requirements

Install dependencies:

```bash
pip install streamlit pillow
```

For synthetic coherent ZIP DICOM preview/inference support:

```bash
pip install pydicom numpy
```

## Authentication

Credentials are loaded from Streamlit secrets (preferred) or environment variables.

Accepted key names:
- Hugging Face: `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, `HUGGINGFACE_API_KEY`
- Gemini: `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GENAI_API_KEY`

Create `.streamlit/secrets.toml` from `.streamlit/secrets.example.toml` and set:

```toml
HF_TOKEN = "your_huggingface_token"
GEMINI_API_KEY = "your_gemini_api_key"
```

Environment variable alternative:

```bash
export HF_TOKEN='your_huggingface_token'
export GEMINI_API_KEY='your_gemini_api_key'
```

## Run

From repository root:

```bash
streamlit run medgemma_gui/app.py
```

In the sidebar:
- choose dataset source,
- choose LLM backend,
- optionally enable richness-based ranking,
- run `Explain` or `Diagnose` on the selected patient.
