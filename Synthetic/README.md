# Synthetic — Coherent EHR + DICOM Dataset

This folder contains scripts and data for the **Synthea coherent simulation** dataset: a set of synthetically generated patient records with linked DICOM imaging.

## Files

```
Synthetic/
├── build_coherent_image_database.py   ← builds synthetic_ehr_image_dataset.jsonl
├── patient_profile.py                  ← patient profile generation utilities
├── download_and_preview.py             ← downloads and previews DICOM ZIP files
├── synthetic_ehr_image_dataset.jsonl   ← primary output (loaded by the GUI)
├── synthetic_patient_ids_with_images.json  ← patient-to-image ID mapping
├── synthetic_quality_report.json       ← build quality statistics
└── mri_slices/                         ← sample MRI slice preview images
    ├── axial_mid.png
    ├── coronal_mid.png
    ├── sagittal_mid.png
    ├── axial_mosaic_6x6.png
    └── viewer.html
```

> **Note:** The raw Synthea DICOM ZIP (`coherent-11-07-2022.zip`, ~600 MB) and the extracted `.cache/` directory are excluded from the repository. They are auto-populated on first Synthea case access in the GUI, or can be pre-warmed with:
>
> ```bash
> bash scripts/recover_deleted_datasets.sh --prewarm-cache
> ```

## Usage

The dataset is loaded directly by the MedGUI Streamlit app when **Synthea Coherent JSONL** is selected in the sidebar.

To rebuild from scratch:

```bash
python Synthetic/build_coherent_image_database.py
```
