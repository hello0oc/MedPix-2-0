#!/usr/bin/env python3
"""Download a Synthea dataset zip and preview files inside it without unzipping.

Usage examples:
  python Synthetic/download_and_preview.py
  python Synthetic/download_and_preview.py --path Synthetic/coherent-11-07-2022.zip --preview
  python Synthetic/download_and_preview.py --no-download --path Synthetic/coherent-11-07-2022.zip --max-files 3 --lines 8
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from zipfile import ZipFile
from typing import List


DOWNLOAD_URL = "https://synthea-open-data.s3.amazonaws.com/coherent/coherent-11-07-2022.zip"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_file(url: str, dest_path: str) -> str:
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return dest_path

    print(f"Downloading {url} -> {dest_path}")
    ensure_dir(os.path.dirname(dest_path) or ".")

    def _reporthook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 / total_size)
        end = "\r" if downloaded < total_size else "\n"
        sys.stdout.write(f"{pct:5.1f}% downloaded...{end}")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, _reporthook)
    return dest_path


def _is_text_file(name: str) -> bool:
    name = name.lower()
    return any(name.endswith(ext) for ext in (".csv", ".json", ".ndjson", ".txt", ".xml"))


def preview_zip(zip_path: str, max_files: int = 5, lines: int = 5) -> None:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    print(f"Opening zip: {zip_path}")
    with ZipFile(zip_path, "r") as z:
        namelist = [n for n in z.namelist() if not n.endswith("/")]
        if not namelist:
            print("No files found in zip.")
            return

        # Prefer human-readable files
        text_files = [n for n in namelist if _is_text_file(n)]
        others = [n for n in namelist if n not in text_files]
        candidates: List[str] = (text_files + others)[:max_files]

        for name in candidates:
            print("\n---\nFile: {}\n---".format(name))
            try:
                with z.open(name) as fh:
                    # read small chunk and decode as text
                    try:
                        for i in range(lines):
                            raw = fh.readline()
                            if not raw:
                                break
                            # decode safely
                            try:
                                line = raw.decode("utf-8")
                            except Exception:
                                line = raw.decode("latin-1", errors="ignore")
                            print(line.rstrip("\n"))
                    except Exception as e:
                        print(f"Could not read lines from {name}: {e}")
            except Exception as e:
                print(f"Could not open {name} in zip: {e}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and preview a Synthea zip dataset")
    parser.add_argument("--url", default=DOWNLOAD_URL, help="URL to download")
    parser.add_argument("--dir", default="Synthetic", help="Directory to store the zip")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading if the file already exists or path provided")
    parser.add_argument("--path", default=None, help="Explicit path to the zip file (overrides dir+url basename)")
    parser.add_argument("--preview", action="store_true", help="Show a preview after download")
    parser.add_argument("--max-files", type=int, default=5, help="Max files to preview")
    parser.add_argument("--lines", type=int, default=5, help="Lines per file to preview")

    args = parser.parse_args(argv)

    ensure_dir(args.dir)

    if args.path:
        zip_path = args.path
    else:
        basename = os.path.basename(args.url)
        zip_path = os.path.join(args.dir, basename)

    if not args.no_download:
        try:
            download_file(args.url, zip_path)
        except Exception as e:
            print(f"Download failed: {e}")
            return 2
    else:
        if not os.path.exists(zip_path):
            print(f"Skipping download but file does not exist: {zip_path}")
            return 3

    if args.preview:
        try:
            preview_zip(zip_path, max_files=args.max_files, lines=args.lines)
        except Exception as e:
            print(f"Preview failed: {e}")
            return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
