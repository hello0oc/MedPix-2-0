#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medgemma_benchmark.run_medgemma_benchmark import endpoint_chat_completion


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_medgemma() -> Dict[str, Any]:
    endpoint_url = os.environ.get("HF_ENDPOINT_URL", "").strip()
    token = os.environ.get("HF_TOKEN", "").strip()
    model_id = os.environ.get("HF_MODEL_ID", "google/medgemma-1-5-4b-it-hae").strip()

    if not endpoint_url or not token:
        return {
            "status": "skipped",
            "reason": "HF_ENDPOINT_URL or HF_TOKEN not set",
        }

    start = time.time()
    try:
        output = endpoint_chat_completion(
            endpoint_url=endpoint_url,
            token=token,
            model_id=model_id,
            messages=[
                {"role": "system", "content": "Return compact JSON only."},
                {"role": "user", "content": "Return exactly: {\"ok\":\"true\"}"},
            ],
            temperature=0.0,
            max_tokens=48,
            timeout=60,
            retries=1,
            retry_backoff_sec=2.0,
        )
        return {
            "status": "pass",
            "latency_sec": round(time.time() - start, 3),
            "sample": str(output)[:240],
        }
    except Exception as exc:
        return {
            "status": "fail",
            "latency_sec": round(time.time() - start, 3),
            "error": str(exc),
        }


def check_gemini() -> Dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL_ID", "gemini-2.5-pro").strip()

    if not api_key:
        return {
            "status": "skipped",
            "reason": "GEMINI_API_KEY not set",
        }

    start = time.time()
    try:
        from google import genai as ggenai  # type: ignore
        from google.genai import types as gtypes  # type: ignore
    except Exception as exc:
        return {
            "status": "fail",
            "error": f"google-genai import failed: {exc}",
        }

    try:
        client = ggenai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=["Return exactly: {\"ok\":\"true\"}"],
            config=gtypes.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=96,
                response_mime_type="application/json",
                http_options=gtypes.HttpOptions(timeout=60000),
            ),
        )
        text = response.text or ""
        return {
            "status": "pass",
            "latency_sec": round(time.time() - start, 3),
            "sample": text[:240],
        }
    except Exception as exc:
        return {
            "status": "fail",
            "latency_sec": round(time.time() - start, 3),
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Nightly non-blocking connectivity checks")
    parser.add_argument(
        "--output",
        default="medgemma_gui/output/connectivity_report.json",
        help="Path to JSON report",
    )
    args = parser.parse_args()

    report = {
        "generated_at_utc": _ts(),
        "checks": {
            "medgemma": check_medgemma(),
            "gemini": check_gemini(),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # Non-blocking by design for nightly informational monitoring.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
