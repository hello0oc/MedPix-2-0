# Patient EHR + Medical Image Dataset (MedPix 2.0 — local build)

> Built locally on 2026-02-19 from the MedPix-2-0 source files.

## Overview

This dataset links **clinical EHR narratives** (patient history, imaging
findings, diagnoses) with their **linked radiology images** (CT / MRI).
It is designed for evaluating and fine-tuning medical LLMs on visual
question-answering and diagnostic reasoning tasks.

## Dataset Statistics

| Metric | Value |
|---|---|
| Total patient cases | 671 |
| Cases with history | 663 (98.8%) |
| Cases with imaging findings | 654 (97.5%) |
| Cases with diagnosis | 671 (100.0%) |
| Cases with linked images | 671 (100.0%) |
| **Fully complete records** | **650 (96.9%)** |
| Total image files on disk | 2050 |
| Missing image files | 0 |

## Files

| File | Split | Cases |
|---|---|---|
| `full_dataset.jsonl` | all | 671 |
| `train.jsonl` | train | 535 |
| `train_1.jsonl` | train-1 | 267 |
| `train_2.jsonl` | train-2 | 268 |
| `dev.jsonl` | dev | 67 |
| `test.jsonl` | test | 69 |
| `quality_report.json` | — | — |

## Record Schema

Each JSON line in the JSONL files represents one patient case:

```json
{
  "uid":            "MPX2077",
  "title":          "Choroid Plexus Carcinoma",
  "history":        "15 month old girl fell off a chair...",
  "exam":           "Physical exam was normal for age...",
  "findings":       "CT: High density mass in the trigone...",
  "differential_diagnosis": "Choroid plexus carcinoma\n...",
  "diagnosis":      "Choroid Plexus Carcinoma",
  "diagnosis_by":   "Biopsy",
  "treatment":      "Surgical excision with follow-up CT and MRI",
  "discussion":     "...",
  "topic_title":    "Choroid Plexus Neoplasm, Papilloma, Carcinoma",
  "acr_code":       "1.3",
  "category":       "Neoplasm, glial",
  "disease_discussion": "...",
  "keywords":       "choroid plexuspapillomacarcinoma",
  "ct_image_ids":   ["MPX2077_synpic51017", ...],
  "mri_image_ids":  ["MPX2077_synpic51021", ...],
  "images": [
    {
      "image_id":         "MPX2077_synpic51017",
      "file_path":        "MedPix-2-0/images/MPX2077_synpic51017.png",
      "on_disk":          true,
      "type":             "CT",
      "modality":         "CT w/contrast (IV)",
      "plane":            "Axial",
      "location":         "Brain, Ventricular",
      "location_category": "Head/Neck",
      "caption":          "...",
      "age":              "1",
      "sex":              "female",
      "acr_codes":        "1.3",
      "figure_part":      "1"
    }
  ],
  "llm_prompt":     "Clinical History:\n...\n\nImaging Findings:\n...",
  "has_history":    true,
  "has_findings":   true,
  "has_diagnosis":  true,
  "has_images":     true,
  "is_complete":    true
}
```

## How to Load

```python
from datasets import load_dataset

# Load any split
ds = load_dataset('json', data_files='patient-ehr-image-dataset/test.jsonl')['train']

# Filter for fully complete records
complete = ds.filter(lambda x: x['is_complete'])
print(f'Complete records: {len(complete)}')

# Build LLM evaluation prompt
sample = complete[0]
print(sample['llm_prompt'])
print('Ground truth:', sample['diagnosis'])
print('Images:', [img['file_path'] for img in sample['images']])
```

## Source

Built from [MedPix 2.0](https://arxiv.org/abs/2407.02994):

```bibtex
@misc{siragusa2025medpix20comprehensivemultimodal,
  title   = {MedPix 2.0: A Comprehensive Multimodal Biomedical Dataset},
  author  = {Irene Siragusa and Salvatore Contino and Massimo La Ciura
             and Rosario Alicata and Roberto Pirrone},
  year    = {2025},
  url     = {https://arxiv.org/abs/2407.02994}
}
```