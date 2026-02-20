# PMCpatient

This folder contains scripts and data for the **PMC-Patients** dataset, sourced from:

> Zhao et al., *A large-scale dataset of patient summaries for retrieval-based clinical decision support systems*, Scientific Data 2023.  
> HuggingFace: [zhengyun21/PMC-Patients](https://huggingface.co/datasets/zhengyun21/PMC-Patients)

---

## Dataset Overview

| Property | Value |
|---|---|
| Source | PubMed Central case reports |
| Original version (CSV) | ~167 k patient summaries |
| V2 version (JSON) | ~250 k patient summaries |
| License | CC BY-NC-SA 4.0 |
| Language | English |
| Images | ❌ text-only |
| Gold diagnosis labels | ❌ not provided |

### Field Schema (raw PMC-Patients)

| Field | Type | Description |
|---|---|---|
| `patient_id` | int | Sequential row identifier |
| `patient_uid` | str | Unique ID: `{PMID}-{index}` |
| `PMID` | str | PubMed article identifier |
| `file_path` | str | PMC XML source path |
| `title` | str | Source article title |
| `patient` | str | Patient narrative (primary text) |
| `age` | str | Age as list of `[value, unit]` pairs |
| `gender` | str | `M` or `F` |
| `relevant_articles` | dict | `{PMID: score}` relevance annotations |
| `similar_patients` | dict | `{uid: score}` similarity annotations |

---

## File Structure

```
PMCpatient/
├── download_pmc_patients.py           ← Step 1: download raw data (optional)
├── prepare_input_strategy.py          ← Step 2: convert to project JSONL (optional)
├── README.md                          ← this file
├── PMC-Patients-sample-1000.csv       ← 1 000-row dev sample (tracked in repo)
└── quality_report.json                ← auto-generated stats
```

> **Note:** The full `PMC-Patients.csv` (~520 MB) and all derived JSONL files are excluded from the repository (see `.gitignore`). Only the 1 000-row sample is committed because it is the only file loaded by the MedGUI Streamlit app. Run `bash scripts/recover_deleted_datasets.sh` to regenerate derived files locally.

---

## Step 1 — Download

```bash
cd PMCpatient/

# Download CSV only (default, ~500 MB)
python download_pmc_patients.py

# Download CSV and keep only 1 000 rows for development
python download_pmc_patients.py --sample 1000

# Download CSV + V2 JSON (~1.5 GB extra)
python download_pmc_patients.py --all

# If the dataset becomes gated, pass a HuggingFace token
HF_TOKEN=hf_xxxx python download_pmc_patients.py
```

---

## Step 2 — Prepare Input Strategy

The converter normalises the raw CSV/JSON into the project-standard JSONL schema
(compatible with `patient-ehr-image-dataset/` and `medgemma_benchmark/`).

### Three strategies

| Strategy | `llm_prompt` shape | Best for |
|---|---|---|
| `ehr_only` | `[Age: X yr \| Gender: Y]\n\n<narrative>` | MedGemma EHR benchmarking |
| `structured` | labelled key-value block | Fine-tuning structured models |
| `qa` | instruction + JSON-output template | Differential diagnosis evaluation |

### Examples

```bash
# Default: EHR-only strategy, full CSV → JSONL
python prepare_input_strategy.py

# Structured strategy on the sample file
python prepare_input_strategy.py \
    --input PMC-Patients-sample-1000.csv \
    --strategy structured

# QA strategy + train/dev/test split (80/10/10)
python prepare_input_strategy.py \
    --input PMC-Patients.csv \
    --strategy qa \
    --split --seed 42

# Show quality report after conversion
python prepare_input_strategy.py --report
```

### Output JSONL schema

```json
{
  "uid":               "7665777-1",
  "pmid":              "33492400",
  "title":             "Early Physical Therapist Interventions …",
  "patient":           "This 60-year-old male was hospitalized …",
  "age_raw":           "[[60.0, 'year']]",
  "age_years":         60.0,
  "gender":            "M",
  "history":           "<same as patient — medgemma_benchmark alias>",
  "exam":              "",
  "findings":          "",
  "diagnosis":         "",
  "differential_diagnosis": "",
  "treatment":         "",
  "relevant_articles": {"32320506": 1, "33492400": 2},
  "similar_patients":  {"7665777-2": 2, "7665777-3": 2},
  "llm_prompt":        "[Age: 60 yr | Gender: Male]\n\nThis 60-year-old male …",
  "has_history":       true,
  "has_diagnosis":     false,
  "has_images":        false,
  "is_complete":       true
}
```

---

## Integration with medgemma_benchmark

Because the output JSONL uses the same keys as `patient-ehr-image-dataset/full_dataset.jsonl`,
you can pass any strategy file directly to the benchmark runner:

```bash
python medgemma_benchmark/run_medgemma_benchmark.py \
    --dataset PMCpatient/PMC-Patients_ehr_only.jsonl \
    --mode ehr_only \
    --n-eval 50
```

---

## Citation

```bibtex
@article{zhao2023large,
  title={A large-scale dataset of patient summaries for retrieval-based clinical decision support systems},
  author={Zhao, Zhengyun and Jin, Qiao and Chen, Fangyuan and Peng, Tuorui and Yu, Sheng},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={909},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
