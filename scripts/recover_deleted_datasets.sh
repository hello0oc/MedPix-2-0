#!/usr/bin/env bash
# =============================================================================
# recover_deleted_datasets.sh
#
# Regenerates the non-essential datasets that were removed to save ~965 MB.
# Run from the repository root:
#
#   bash scripts/recover_deleted_datasets.sh
#
# Prerequisites:
#   pip install -r requirements.txt   (or your active venv)
#   HF_TOKEN / GEMINI_API_KEY set in .streamlit/secrets.toml or env
# =============================================================================
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PYTHON="${PYTHON:-$(which python3)}"

echo "========================================"
echo " MedPix-2-0 dataset recovery script"
echo " Repo root: $REPO"
echo "========================================"

# ── 1. Full PMC-Patients CSV (520 MB) ────────────────────────────────────────
# The raw 167K-row PubMed patients dump.  Download via the bundled helper.
if [ ! -f "$REPO/PMCpatient/PMC-Patients.csv" ]; then
  echo "[1/3] Downloading PMC-Patients.csv (~520 MB) ..."
  "$PYTHON" "$REPO/PMCpatient/download_pmc_patients.py"
  echo "      Done: PMCpatient/PMC-Patients.csv"
else
  echo "[1/3] PMC-Patients.csv already present — skipping."
fi

# ── 2. PMC sample derived JONLs (ehr_only, structured, qa splits) ─────────────
# These are outputs of prepare_input_strategy.py applied to the 1000-row sample.
SAMPLE_CSV="$REPO/PMCpatient/PMC-Patients-sample-1000.csv"
PREP="$REPO/PMCpatient/prepare_input_strategy.py"

if [ ! -f "$REPO/PMCpatient/PMC-Patients-sample-1000_ehr_only.jsonl" ]; then
  echo "[2/3] Regenerating PMC sample JONLs ..."

  echo "      Strategy: ehr_only ..."
  "$PYTHON" "$PREP" \
    --input "$SAMPLE_CSV" \
    --strategy ehr_only \
    --output "$REPO/PMCpatient/PMC-Patients-sample-1000_ehr_only.jsonl"

  echo "      Strategy: structured ..."
  "$PYTHON" "$PREP" \
    --input "$SAMPLE_CSV" \
    --strategy structured \
    --output "$REPO/PMCpatient/PMC-Patients-sample-1000_structured.jsonl"

  echo "      QA splits (train/dev/test) ..."
  "$PYTHON" "$PREP" \
    --input "$SAMPLE_CSV" \
    --strategy qa \
    --output "$REPO/PMCpatient/PMC-Patients-sample-1000_qa_train.jsonl" \
    --split train

  "$PYTHON" "$PREP" \
    --input "$SAMPLE_CSV" \
    --strategy qa \
    --output "$REPO/PMCpatient/PMC-Patients-sample-1000_qa_dev.jsonl" \
    --split dev

  "$PYTHON" "$PREP" \
    --input "$SAMPLE_CSV" \
    --strategy qa \
    --output "$REPO/PMCpatient/PMC-Patients-sample-1000_qa_test.jsonl" \
    --split test

  echo "      Done: PMC sample JONLs regenerated."
else
  echo "[2/3] PMC sample JONLs already present — skipping."
fi

# ── 3. Synthetic DICOM .cache (417 MB) ────────────────────────────────────────
# The .cache/zip_media/ directory is populated on first use by
# Synthetic/download_and_preview.py (or by the GUI when Synthea cases are viewed).
# It is NOT needed to run the GUI — the cache re-fills automatically on first access.
# To pre-warm it manually:
if [ "$1" == "--prewarm-cache" ]; then
  echo "[3/3] Pre-warming Synthetic DICOM cache (requires internet) ..."
  "$PYTHON" "$REPO/Synthetic/download_and_preview.py"
  echo "      Done: .cache/zip_media/ re-populated."
else
  echo "[3/3] Synthetic .cache not pre-warmed (pass --prewarm-cache to do so)."
  echo "      It will auto-populate on first GUI access to Synthea Coherent JSONL."
fi

echo ""
echo "Recovery complete."
echo "Note: to recover deleted CODE modules (code-DRMinerva, code-KG, MongoDB-UI)"
echo "  run:  bash scripts/recover_deleted_code.sh"
