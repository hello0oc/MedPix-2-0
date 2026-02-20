#!/usr/bin/env python3
"""
extract_llm_subset.py
Create a compact LLM-ready JSONL file with prompt, image paths, and ground-truth.
"""
import argparse
import json
from pathlib import Path


def load_jsonl(p: Path):
    with open(p, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_prompt(rec):
    if rec.get('llm_prompt'):
        return rec['llm_prompt']
    parts = []
    if rec.get('history'):
        parts.append(f"History:\n{rec['history'].strip()}")
    if rec.get('findings'):
        parts.append(f"Findings:\n{rec['findings'].strip()}")
    return "\n\n".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='patient-ehr-image-dataset/full_dataset.jsonl')
    p.add_argument('--output', default='patient-ehr-image-dataset/llm_subset.jsonl')
    p.add_argument('--max', type=int, default=200, help='max records to include')
    p.add_argument('--complete-only', action='store_true', help='only include fully complete records')
    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out, 'w', encoding='utf-8') as fo:
        for rec in load_jsonl(inp):
            if args.complete_only and not rec.get('is_complete'):
                continue
            # ensure there is a diagnosis and at least one image
            if not rec.get('diagnosis'):
                continue
            imgs = [im for im in rec.get('images', []) if im.get('on_disk')]
            if not imgs:
                continue
            prompt = build_prompt(rec)
            if not prompt:
                continue
            first_img = imgs[0]
            sample = {
                'case_id': rec.get('uid'),
                'prompt': prompt,
                'ground_truth': rec.get('diagnosis'),
                'image_paths': [im['file_path'] for im in imgs],
                'modalities': sorted({im.get('modality', '').strip() for im in imgs if im.get('modality')}),
                'age': first_img.get('age', ''),
                'sex': first_img.get('sex', ''),
            }
            fo.write(json.dumps(sample, ensure_ascii=False) + '\n')
            written += 1
            if written >= args.max:
                break

    print(f'Wrote {written} records → {out}')

if __name__ == '__main__':
    main()
