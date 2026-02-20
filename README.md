# MedGUI

**MedGUI** is a multimodal medical AI assistant built on Streamlit. It lets clinicians and researchers browse patient EHR records from three independent dataset sources, preview linked radiology images, and run AI-powered inference — plain-language explanations and clinical diagnostic reasoning — using either a MedGemma HuggingFace endpoint or the Gemini 2.5 Pro API.

---

## Quick Start

```bash
# 1. Install dependencies
pip install streamlit pillow google-genai

# 2. Set credentials
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
# Fill in HF_TOKEN and GEMINI_API_KEY in .streamlit/secrets.toml

# 3. Run
streamlit run medgemma_gui/app.py
```

Open `http://localhost:8501` in your browser.

---

## Features

- **Three dataset sources** — selectable in the sidebar:
  - MedPix JSONL (671 clinical cases with linked CT/MRI images)
  - PMC-Patients CSV (1 000-row sample of PubMed Central case reports)
  - Synthea Coherent JSONL (synthetically generated EHR + DICOM imaging)
- **Richness ranking** — patients ordered by clinical text completeness
- **Image preview** — inline radiology image viewer (PNG and DICOM supported)
- **Two AI tasks** per patient:
  - *Explain* — plain-language summary for non-medical readers
  - *Diagnose* — structured clinical diagnosis with confidence, rationale, key findings, and differential
- **Two LLM backends**:
  - MedGemma (HuggingFace Inference Endpoint)
  - Gemini 2.5 Pro (Google AI Studio)
- **Per-patient session cache** — results persist when switching patients or models

---

## Credentials

Credentials are loaded from `.streamlit/secrets.toml` first, then environment variables. Accepted key names:

| Service | Accepted keys |
|---|---|
| HuggingFace | `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, `HUGGINGFACE_API_KEY` |
| Google Gemini | `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GENAI_API_KEY` |

`.streamlit/secrets.toml` is in `.gitignore` and is never committed.

---

## Repository Structure

```
MedGUI/
├── medgemma_gui/              # Streamlit app (main entry point)
│   └── app.py
├── medgemma_benchmark/        # Backend utilities + standalone benchmark runner
│   └── run_medgemma_benchmark.py
├── patient-ehr-image-dataset/ # Primary dataset: 671 MedPix EHR+image cases
├── MedPix-2-0/                # Source data: raw MedPix JSON + 2 050 images
├── PMCpatient/                # PMC-Patients 1 000-row sample + prep scripts
├── Synthetic/                 # Synthea coherent EHR+DICOM dataset + build scripts
├── build_dataset.py           # Builds patient-ehr-image-dataset/ from MedPix source
├── extract_llm_subset.py      # Extracts a curated LLM evaluation subset
├── scripts/                   # Recovery utilities
│   ├── recover_deleted_datasets.sh
│   └── recover_deleted_code.sh
├── ARCHITECTURE.md            # Cross-module architecture & data-flow reference
└── .streamlit/
    └── secrets.toml           # Local credentials (gitignored)
```

---

## Architecture

For a detailed cross-module data-flow and runtime sequence, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Datasets

| Dataset | Path | Records | Notes |
|---|---|---|---|
| MedPix JSONL | `patient-ehr-image-dataset/full_dataset.jsonl` | 671 | Primary; EHR + linked images |
| PMC-Patients | `PMCpatient/PMC-Patients-sample-1000.csv` | 1 000 | Text-only; PubMed case reports |
| Synthea JSONL | `Synthetic/synthetic_ehr_image_dataset.jsonl` | — | Synthetic EHR + DICOM |

To rebuild `patient-ehr-image-dataset/` from raw MedPix source:

```bash
python build_dataset.py
```

---

## Benchmark

A standalone MedGemma benchmark (EHR-only vs EHR+images) ships in `medgemma_benchmark/`:

```bash
export HF_TOKEN='your_token'
python medgemma_benchmark/run_medgemma_benchmark.py \
  --endpoint-url 'https://YOUR-ENDPOINT.aws.endpoints.huggingface.cloud' \
  --model-id 'google/medgemma-1-5-4b-it-hae'
```

---

## License

Dataset source: MedPix 2.0 (Siragusa et al., 2025).  
PMC-Patients source: Zhao et al., Scientific Data 2023 — CC BY-NC-SA 4.0.
