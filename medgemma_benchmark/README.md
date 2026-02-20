# MedGemma Benchmark (EHR vs EHR+Images)

This benchmark runs diagnostic inference on `patient-ehr-image-dataset/full_dataset.jsonl` in two modes:

1. `ehr_only`
2. `ehr_plus_images`

Both modes use:
- fixed 3-shot exemplars
- 3 target patients
- output format: diagnosis + short rationale

## Model Target

- Model ID: `google/medgemma-1-5-4b-it-hae`
- Endpoint: your Hugging Face Inference Endpoint URL

## Security

Do **not** hardcode your token in files.
Set it as an environment variable only.

## Run

From repository root:

```bash
export HF_TOKEN='YOUR_HF_TOKEN'
python3 medgemma_benchmark/run_medgemma_benchmark.py \
  --endpoint-url 'https://YOUR-ENDPOINT.aws.endpoints.huggingface.cloud' \
  --model-id 'google/medgemma-1-5-4b-it-hae' \
  --input 'patient-ehr-image-dataset/full_dataset.jsonl' \
  --output-dir 'medgemma_benchmark/output' \
  --num-shots 3 \
  --num-targets 3 \
  --max-images 2
```

## Dry Run (no endpoint call)

```bash
python3 medgemma_benchmark/run_medgemma_benchmark.py \
  --endpoint-url 'https://YOUR-ENDPOINT.aws.endpoints.huggingface.cloud' \
  --dry-run
```

## Outputs

- `medgemma_benchmark/output/summary.json`
- `medgemma_benchmark/output/predictions_ehr_only.jsonl`
- `medgemma_benchmark/output/predictions_ehr_plus_images.jsonl`

`summary.json` contains:
- selected exemplar and target UIDs
- per-mode aggregate metrics (`exact_match`, `avg_token_f1`, `avg_latency_sec`)

## Notes

- The script uses records with diagnosis, EHR text, and linked on-disk images.
- Multimodal requests send local images as data URLs (base64).
- If your endpoint uses a non-chat schema, adjust `endpoint_chat_completion` in `run_medgemma_benchmark.py`.
