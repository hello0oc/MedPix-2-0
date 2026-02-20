#!/usr/bin/env python3
"""
Download PMC-Patients dataset from HuggingFace Hub.

Files available:
  PMC-Patients.csv       ─ original 167k patient summaries
  PMC-Patients-V2.json   ─ updated 250k patient summaries (same schema)

Usage
-----
  # Download CSV (default, ~optimised for iterative work)
  python download_pmc_patients.py

  # Download CSV + V2 JSON
  python download_pmc_patients.py --all

  # Download CSV and keep only first N rows as a sample
  python download_pmc_patients.py --sample 5000

  # Provide a HuggingFace token (needed only if the dataset becomes gated)
  python download_pmc_patients.py --token hf_xxxx
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_BASE = "https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main"
CSV_FILENAME  = "PMC-Patients.csv"
JSON_FILENAME = "PMC-Patients-V2.json"
OUT_DIR       = Path(__file__).parent          # save inside PMCpatient/

CHUNK = 1 << 20  # 1 MB streaming chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_request(url: str, token: str | None) -> urllib.request.Request:
    headers: dict[str, str] = {"User-Agent": "MedPix-2-0/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def stream_download(url: str, dest: Path, token: str | None = None) -> Path:
    """Stream-download *url* to *dest*, printing progress to stderr."""
    req = _build_request(url, token)
    print(f"  ↓  {url}", file=sys.stderr)
    try:
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done  = 0
            t0    = time.time()
            with dest.open("wb") as fh:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        mb  = done / 1e6
                        elapsed = time.time() - t0
                        rate = done / elapsed / 1e6 if elapsed else 0
                        print(
                            f"\r     {pct:5.1f}%  {mb:.1f} MB  {rate:.1f} MB/s  ",
                            end="",
                            file=sys.stderr,
                        )
    except urllib.error.HTTPError as exc:
        print(f"\nHTTP {exc.code} for {url}", file=sys.stderr)
        raise
    print(f"\n  ✓  Saved → {dest}", file=sys.stderr)
    return dest


def sample_csv(src: Path, dest: Path, n: int) -> Path:
    """Write only the first *n* data rows of *src* CSV to *dest*."""
    with src.open("r", encoding="utf-8") as fin, dest.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        writer.writerow(next(reader))  # header
        for i, row in enumerate(reader):
            if i >= n:
                break
            writer.writerow(row)
    print(f"  ✓  Sampled {n} rows → {dest}", file=sys.stderr)
    return dest


def sample_json(src: Path, dest: Path, n: int) -> Path:
    """Write only the first *n* entries of *src* JSON array to *dest*."""
    with src.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    subset = data[:n] if isinstance(data, list) else list(data.values())[:n]
    with dest.open("w", encoding="utf-8") as fout:
        json.dump(subset, fout, ensure_ascii=False, indent=2)
    print(f"  ✓  Sampled {n} entries → {dest}", file=sys.stderr)
    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download PMC-Patients dataset")
    p.add_argument("--all",    action="store_true", help="Also download V2 JSON (large ~1.5 GB)")
    p.add_argument("--sample", type=int, default=0, metavar="N",
                   help="Keep only first N records (0 = keep everything)")
    p.add_argument("--token",  default=os.getenv("HF_TOKEN", ""),
                   help="HuggingFace token (or set HF_TOKEN env var)")
    p.add_argument("--out",    default=str(OUT_DIR),
                   help="Output directory (default: same folder as this script)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out  = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    token = args.token or None

    # ── CSV ──────────────────────────────────────────────────────────────────
    print("\n[1/2] Downloading PMC-Patients CSV …", file=sys.stderr)
    csv_dest = out / CSV_FILENAME
    stream_download(f"{HF_BASE}/{CSV_FILENAME}", csv_dest, token)

    if args.sample > 0:
        sampled_csv = out / f"PMC-Patients-sample-{args.sample}.csv"
        sample_csv(csv_dest, sampled_csv, args.sample)

    # ── JSON (optional) ──────────────────────────────────────────────────────
    if args.all:
        print("\n[2/2] Downloading PMC-Patients-V2 JSON …", file=sys.stderr)
        json_dest = out / JSON_FILENAME
        stream_download(f"{HF_BASE}/{JSON_FILENAME}", json_dest, token)

        if args.sample > 0:
            sampled_json = out / f"PMC-Patients-V2-sample-{args.sample}.json"
            sample_json(json_dest, sampled_json, args.sample)
    else:
        print("\n[2/2] Skipping V2 JSON (use --all to include it).", file=sys.stderr)

    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
