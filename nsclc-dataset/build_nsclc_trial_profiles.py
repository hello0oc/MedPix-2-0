#!/usr/bin/env python3
"""Batch-generate clinical trial profiles for all NSCLC dataset cases.

Uses Gemini 2.5 Pro with the same prompt, image handling, and
post-processing logic as the Streamlit app (medgemma_gui/app.py).
All images for each case are passed to the model.

Output:  nsclc-dataset/nsclc_trial_profiles.json
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent   # nsclc-dataset/
REPO_ROOT  = SCRIPT_DIR.parent                 # MedPix-2-0/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "medgemma_gui"))

# ── Import shared helpers from the benchmark module ───────────────────────────
from medgemma_benchmark.run_medgemma_benchmark import (
    build_ehr_text,
    collect_image_paths,
    load_jsonl,
    resolve_image_path,
    truncate_text,
)


# ── Load app.py as a module without starting the Streamlit server ─────────────
def _load_app_module() -> types.ModuleType:
    """Import medgemma_gui/app.py with a minimal Streamlit stub."""
    if "streamlit" not in sys.modules:
        stub = types.ModuleType("streamlit")

        class _Ctx:
            def __enter__(self): return self
            def __exit__(self, *_): return False

        class _Secrets(dict):
            def get(self, k, d=None): return super().get(k, d)

        stub.secrets       = _Secrets()
        stub.session_state = {}
        stub.sidebar       = _Ctx()
        stub.cache_data    = lambda show_spinner=False: (lambda fn: fn)
        stub.spinner       = lambda *a, **k: _Ctx()
        stub.columns       = lambda spec: [_Ctx() for _ in spec]
        for _fn in (
            "radio", "checkbox", "slider", "number_input", "button",
            "selectbox", "expander", "set_page_config", "markdown",
            "caption", "subheader", "info", "error", "warning", "write",
            "code", "image", "divider", "stop", "download_button",
        ):
            setattr(stub, _fn, lambda *a, **k: None)
        sys.modules["streamlit"] = stub

    spec = importlib.util.spec_from_file_location(
        "medgui_app", REPO_ROOT / "medgemma_gui" / "app.py"
    )
    mod = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(mod)                  # type: ignore[union-attr]
    return mod


app = _load_app_module()

# ── Populate stub st.secrets from .streamlit/secrets.toml (if present) ───────
def _load_secrets_into_stub() -> None:
    """Read .streamlit/secrets.toml and inject values into the stub's secrets dict.

    Outside the Streamlit server st.secrets is an empty dict, so
    resolve_credential() can't find keys that live only in the TOML file.
    """
    secrets_path = REPO_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            # Fallback: very simple key = "value" line parser
            import re
            data: Dict[str, str] = {}
            for line in secrets_path.read_text(encoding="utf-8").splitlines():
                m = re.match(r'^\s*(\w+)\s*=\s*["\'](.+?)["\']\s*$', line)
                if m:
                    data[m.group(1)] = m.group(2)
            sys.modules["streamlit"].secrets.update(data)
            return
    with secrets_path.open("rb") as fh:
        sys.modules["streamlit"].secrets.update(tomllib.load(fh))

_load_secrets_into_stub()

# ── Configuration ──────────────────────────────────────────────────────────────
NSCLC_PATH  = SCRIPT_DIR / "nsclc_dataset.jsonl"
OUTPUT_PATH = SCRIPT_DIR / "nsclc_trial_profiles.json"
WORKSPACE   = REPO_ROOT

GEMINI_MODEL      = app.GEMINI_MODEL_ID   # "gemini-2.5-pro"
MAX_OUTPUT_TOKENS = 2048
TIMEOUT_SEC       = 180
RETRIES           = 3
EHR_CHAR_LIMIT    = 3000   # generous limit for batch; no interactive slider


# ── Helpers ────────────────────────────────────────────────────────────────────

def resolve_gemini_key() -> str:
    """Read the Gemini API key using the same resolver as the app."""
    key, _, _ = app.resolve_credential(app.GEMINI_CANDIDATE_KEYS)
    if not key:
        raise RuntimeError(
            "No Gemini API key found.  Set GEMINI_API_KEY in "
            ".streamlit/secrets.toml or as an environment variable."
        )
    return key


def get_all_images(case: Dict[str, Any]) -> List[Path]:
    """Resolve every on-disk image for a case — no cap, all images."""
    paths: List[Path] = []
    for raw in collect_image_paths(case):
        resolved = resolve_image_path(WORKSPACE, raw)
        if resolved and resolved.exists():
            paths.append(resolved)
    return paths


def build_ehr_no_findings(case: Dict[str, Any]) -> str:
    """Build EHR text excluding pre-written imaging findings (same as app)."""
    stripped = {k: v for k, v in case.items() if k != "findings"}
    return truncate_text(build_ehr_text(stripped), EHR_CHAR_LIMIT)


def run_trial_profile(case: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Generate and post-process a trial profile for one case."""
    uid         = str(case.get("uid", "unknown"))
    ehr_text    = build_ehr_no_findings(case)
    image_paths = get_all_images(case)
    use_images  = bool(image_paths)

    # Mirror the app's prompt-selection logic
    base_prompt = (
        app.TRIAL_PROFILE_PROMPT
        if use_images
        else app.TRIAL_PROFILE_PROMPT_TEXT_ONLY
    )
    system_prompt = app.build_trial_profile_prompt(base_prompt, uid)

    raw_output = app.call_gemini(
        api_key=api_key,
        model_name=GEMINI_MODEL,
        system_prompt=system_prompt,
        ehr_text=ehr_text,
        image_paths=image_paths,
        use_images=use_images,
        max_images=len(image_paths),   # pass ALL images to the model
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=TIMEOUT_SEC,
        retries=RETRIES,
    )

    parsed = app.parse_json_object(raw_output)
    if not parsed:
        raise ValueError(
            "Could not parse a JSON object from model output. "
            f"Raw (first 400 chars): {raw_output[:400]}"
        )

    return app.validate_and_coerce_trial_profile(parsed, uid, use_images)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading NSCLC dataset:  {NSCLC_PATH}")
    cases = [c for c in load_jsonl(NSCLC_PATH) if c.get("uid")]
    print(f"Cases found:            {len(cases)}")

    api_key = resolve_gemini_key()
    print(f"Gemini API key:         resolved")
    print(f"Model:                  {GEMINI_MODEL}")
    print(f"Output:                 {OUTPUT_PATH}\n")

    results: List[Dict[str, Any]] = []
    errors:  List[Dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        uid        = str(case.get("uid", f"case_{i}"))
        n_images   = len(get_all_images(case))
        use_images = n_images > 0
        label      = f"🖼️  {n_images} image(s)" if use_images else "📄 text-only"
        print(f"[{i:02d}/{len(cases)}] {uid:12s}  {label}  ...", end="", flush=True)

        try:
            profile = run_trial_profile(case, api_key)
            results.append(profile)
            n_facts = len(profile.get("key_facts") or [])
            missing = next(
                (kf.get("value") for kf in (profile.get("key_facts") or [])
                 if kf.get("field") == "missing_info"),
                [],
            )
            miss_str = f", ⚠️ missing: {missing}" if missing else ""
            print(f"  ✓  ({n_facts} key_facts{miss_str})")
        except Exception as exc:
            errors.append({"uid": uid, "error": str(exc)})
            results.append({
                "topic_id":    uid.lower(),
                "profile_text": "",
                "key_facts":   [],
                "ambiguities": [],
                "_error":      str(exc),
            })
            print(f"  ✗  ERROR: {exc}")

        # Brief pause to respect Gemini rate limits
        if i < len(cases):
            time.sleep(1)

    # ── Write output ───────────────────────────────────────────────────────
    envelope = {
        "generated_at":  datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model":         GEMINI_MODEL,
        "dataset":       NSCLC_PATH.name,
        "total_cases":   len(cases),
        "success_count": len(cases) - len(errors),
        "error_count":   len(errors),
        "profiles":      results,
    }
    OUTPUT_PATH.write_text(json.dumps(envelope, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Done.  {len(cases) - len(errors)}/{len(cases)} profiles generated successfully.")
    if errors:
        print(f"\nFailed cases ({len(errors)}):")
        for e in errors:
            print(f"  • {e['uid']}: {e['error'][:120]}")
    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
