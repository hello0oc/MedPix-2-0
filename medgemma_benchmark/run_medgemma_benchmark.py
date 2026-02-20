#!/usr/bin/env python3
"""
Run a small MedGemma diagnosis benchmark on MedPix full_dataset.jsonl in two modes:
1) EHR-only
2) EHR + linked images

Default behavior is aligned to the request:
- 3 fixed few-shot exemplars
- 3 evaluation patients
- diagnosis + short rationale output

Auth token is read from env var only (default: HF_TOKEN).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib import error, request
from zipfile import ZipFile


SYSTEM_INSTRUCTION = (
    "You are a careful medical AI assistant. Given patient EHR data and optionally medical images, "
    "provide: (1) most likely diagnosis, (2) brief rationale in 2-4 lines. "
    "Return ONLY valid JSON with this schema and no markdown/code fences: "
    '{"diagnosis":"<single best diagnosis>","rationale":"<2-4 lines concise clinical reasoning>"}. '
    "Do not include any other keys."
)


@dataclass
class EvalRecord:
    uid: str
    ehr_text: str
    diagnosis: str
    image_paths: List[str]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sanitize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def build_ehr_text(record: Dict[str, Any]) -> str:
    llm_prompt = sanitize_text(record.get("llm_prompt"))
    if llm_prompt:
        return llm_prompt

    pieces: List[str] = []
    history = sanitize_text(record.get("history"))
    exam = sanitize_text(record.get("exam"))
    findings = sanitize_text(record.get("findings"))

    if history:
        pieces.append(f"Clinical History: {history}")
    if exam:
        pieces.append(f"Physical Exam: {exam}")
    if findings:
        pieces.append(f"Imaging Findings: {findings}")

    return "\n".join(pieces).strip()


def collect_image_paths(record: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for image in record.get("images", []):
        raw_path = image.get("file_path")
        if not raw_path:
            continue
        raw_path = str(raw_path)
        if image.get("on_disk") or "::" in raw_path:
            paths.append(raw_path)
    return paths


def _resolve_workspace_path(workspace_root: Path, raw_path: str) -> Optional[Path]:
    candidate = workspace_root / raw_path
    if candidate.exists():
        return candidate

    if raw_path.startswith("MedPix-2-0/"):
        fallback = workspace_root / raw_path.split("MedPix-2-0/", 1)[1]
        if fallback.exists():
            return fallback

    fallback = workspace_root / "MedPix-2-0" / raw_path
    if fallback.exists():
        return fallback
    return None


def _extract_zip_member(workspace_root: Path, raw_path: str) -> Optional[Path]:
    if "::" not in raw_path:
        return None

    zip_rel, member = raw_path.split("::", 1)
    member = member.strip()
    if not member:
        return None

    member_path = Path(member)
    if member_path.is_absolute() or ".." in member_path.parts:
        return None

    zip_path = _resolve_workspace_path(workspace_root, zip_rel.strip())
    if not zip_path or not zip_path.exists():
        return None

    cache_key = hashlib.sha1(str(zip_path.resolve()).encode("utf-8")).hexdigest()[:16]
    target = workspace_root / ".cache" / "zip_media" / cache_key / member
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "r") as zf:
        try:
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        except KeyError:
            return None
    return target


def resolve_image_path(workspace_root: Path, raw_path: str) -> Optional[Path]:
    extracted = _extract_zip_member(workspace_root, raw_path)
    if extracted and extracted.exists():
        return extracted
    return _resolve_workspace_path(workspace_root, raw_path)


def normalize_for_match(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = normalize_for_match(prediction).split()
    tgt_tokens = normalize_for_match(target).split()
    if not pred_tokens or not tgt_tokens:
        return 0.0

    pred_count: Dict[str, int] = {}
    for token in pred_tokens:
        pred_count[token] = pred_count.get(token, 0) + 1

    overlap = 0
    for token in tgt_tokens:
        if pred_count.get(token, 0) > 0:
            overlap += 1
            pred_count[token] -= 1

    precision = overlap / len(pred_tokens)
    recall = overlap / len(tgt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_diagnosis(model_output: str) -> str:
    text = model_output.strip()
    if not text:
        return ""

    decoder = json.JSONDecoder()
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "diagnosis" in payload:
            diagnosis = sanitize_text(str(payload.get("diagnosis", "")))
            if diagnosis:
                return diagnosis

    key_match = re.search(r'"diagnosis"\s*:\s*"([^\"]*)"', text, flags=re.IGNORECASE)
    if key_match:
        return sanitize_text(key_match.group(1))

    if text.lstrip().startswith("{") and "diagnosis" in text.lower():
        return ""

    for line in text.splitlines():
        if re.match(r"^\s*diagnosis\s*:", line, flags=re.IGNORECASE):
            diagnosis = re.sub(r"^\s*diagnosis\s*:\s*", "", line, flags=re.IGNORECASE).strip()
            if diagnosis:
                return diagnosis

    bold_line = re.search(r"\*\*\s*([^*\n]+?)\s*\*\*", text)
    if bold_line:
        diagnosis = sanitize_text(bold_line.group(1))
        if diagnosis and len(diagnosis.split()) <= 8:
            return diagnosis

    first_line = text.splitlines()[0]
    first_line = re.sub(r"^[-*\d\.)\s]+", "", first_line).strip()
    sentence = re.split(r"[.;]", first_line)[0].strip()
    return sentence


def build_records(rows: Sequence[Dict[str, Any]], require_images: bool) -> List[EvalRecord]:
    out: List[EvalRecord] = []
    for row in rows:
        uid = sanitize_text(str(row.get("uid", "")))
        if not uid:
            continue

        diagnosis = sanitize_text(row.get("diagnosis"))
        if not diagnosis:
            continue

        ehr_text = build_ehr_text(row)
        if not ehr_text:
            continue

        image_paths = collect_image_paths(row)
        if require_images and not image_paths:
            continue

        out.append(
            EvalRecord(
                uid=uid,
                ehr_text=ehr_text,
                diagnosis=diagnosis,
                image_paths=image_paths,
            )
        )

    out.sort(key=lambda record: record.uid)
    return out


def choose_exemplars_and_targets(records: Sequence[EvalRecord], num_shots: int, num_targets: int) -> Tuple[List[EvalRecord], List[EvalRecord]]:
    if len(records) < (num_shots + num_targets):
        raise ValueError(
            f"Need at least {num_shots + num_targets} records, found {len(records)}."
        )

    exemplars = list(records[:num_shots])
    targets = list(records[num_shots : num_shots + num_targets])
    return exemplars, targets


def make_fewshot_block(exemplars: Sequence[EvalRecord]) -> str:
    blocks: List[str] = []
    for idx, item in enumerate(exemplars, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Example {idx}:",
                    f"Patient UID: {item.uid}",
                    f"EHR: {item.ehr_text}",
                    (
                        "Answer JSON: "
                        + json.dumps(
                            {
                                "diagnosis": item.diagnosis,
                                "rationale": "Ground-truth diagnosis from case metadata.",
                            },
                            ensure_ascii=False,
                        )
                    ),
                ]
            )
        )
    return "\n\n".join(blocks)


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".dcm":
        encoded = base64.b64encode(dicom_to_png_bytes(path)).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def dicom_to_png_bytes(path: Path) -> bytes:
    try:
        import numpy as np  # type: ignore
        import pydicom  # type: ignore
        from PIL import Image  # type: ignore
        from pydicom.pixel_data_handlers.util import apply_voi_lut  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "DICOM support requires packages: pydicom, numpy, pillow"
        ) from exc

    ds = pydicom.dcmread(str(path), force=True)
    pixels = ds.pixel_array

    # Apply VOI LUT when available to avoid noisy renders for MR/CT data.
    try:
        pixels = apply_voi_lut(pixels, ds)
    except Exception:
        pass

    # Apply rescale slope/intercept if present.
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    pixels = pixels.astype(np.float32) * slope + intercept

    if getattr(pixels, "ndim", 0) == 4:
        pixels = pixels[0]

    arr = np.asarray(pixels, dtype=np.float32)

    # If volume data (slices, rows, cols), pick the slice with most structure.
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        flat = arr.reshape(arr.shape[0], -1)
        slice_scores = flat.std(axis=1)
        best_idx = int(slice_scores.argmax())
        arr = arr[best_idx]

    # Windowing if metadata is present; otherwise, use full min/max.
    window_center = getattr(ds, "WindowCenter", None)
    window_width = getattr(ds, "WindowWidth", None)
    if isinstance(window_center, (list, tuple)):
        window_center = window_center[0]
    if isinstance(window_width, (list, tuple)):
        window_width = window_width[0]

    if window_center is not None and window_width is not None:
        wc = float(window_center)
        ww = float(window_width)
        if ww > 0:
            lower = wc - ww / 2
            upper = wc + ww / 2
            arr = arr.clip(lower, upper)

    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max <= arr_min:
        arr_u8 = np.zeros(arr.shape, dtype=np.uint8)
    else:
        arr_u8 = ((arr - arr_min) / (arr_max - arr_min) * 255.0).clip(0, 255).astype(np.uint8)

    # Invert MONOCHROME1 (where higher values are darker).
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr_u8 = 255 - arr_u8

    if arr_u8.ndim == 2:
        image = Image.fromarray(arr_u8, mode="L")
    elif arr_u8.ndim == 3 and arr_u8.shape[-1] == 3:
        image = Image.fromarray(arr_u8, mode="RGB")
    elif arr_u8.ndim == 3 and arr_u8.shape[-1] == 4:
        image = Image.fromarray(arr_u8, mode="RGBA")
    else:
        image = Image.fromarray(arr_u8[..., 0], mode="L")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _strip_data_url_prefix(value: str) -> str:
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def _messages_to_generate_payload(
    messages: Sequence[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    system_text = ""
    user_text = ""
    images_b64: List[str] = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "system" and isinstance(content, str):
            system_text = content.strip()
            continue

        if role == "user":
            if isinstance(content, str):
                user_text = content.strip()
            elif isinstance(content, list):
                text_parts: List[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        text_parts.append(str(block.get("text", "")).strip())
                    elif block_type == "image_url":
                        image_url = block.get("image_url", {})
                        if isinstance(image_url, dict) and image_url.get("url"):
                            images_b64.append(_strip_data_url_prefix(str(image_url["url"])))
                user_text = "\n\n".join(part for part in text_parts if part)

    prompt_parts: List[str] = []
    if system_text:
        prompt_parts.append(system_text)

    if images_b64:
        image_tokens = "\n".join("<image>" for _ in images_b64)
        prompt_parts.append(image_tokens)

    if user_text:
        prompt_parts.append(user_text)

    prompt = "\n\n".join(prompt_parts).strip()
    payload: Dict[str, Any] = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": False,
            "return_full_text": False,
        },
    }
    if images_b64:
        payload["images"] = images_b64
    return payload


def endpoint_chat_completion(
    endpoint_url: str,
    token: str,
    model_id: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_backoff_sec: float,
) -> str:
    endpoint_base = endpoint_url.rstrip("/")
    chat_url = endpoint_base
    if not chat_url.endswith("/v1/chat/completions"):
        chat_url = f"{chat_url}/v1/chat/completions"

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    def post_json(url: str, payload_obj: Dict[str, Any]) -> Any:
        body = json.dumps(payload_obj).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )

        raw = ""
        for attempt in range(retries + 1):
            try:
                with request.urlopen(req, timeout=timeout) as response:
                    raw = response.read().decode("utf-8")
                return json.loads(raw)
            except error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code in {429, 500, 502, 503, 504}
                if retryable and attempt < retries:
                    time.sleep(retry_backoff_sec * (2 ** attempt))
                    continue
                raise RuntimeError(
                    f"Endpoint HTTP {exc.code} at {url}. Response: {raw[:1200]}"
                ) from exc
            except Exception as exc:
                if attempt < retries:
                    time.sleep(retry_backoff_sec * (2 ** attempt))
                    continue
                raise RuntimeError(f"Endpoint request failed at {url}: {exc}") from exc

    try:
        data = post_json(chat_url, payload)
        if isinstance(data, dict) and data.get("choices"):
            message = data["choices"][0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                text_segments: List[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_segments.append(str(item.get("text", "")))
                return "\n".join(seg for seg in text_segments if seg).strip()
            return str(content).strip()
        if isinstance(data, dict) and data.get("generated_text"):
            return str(data["generated_text"]).strip()
    except RuntimeError as exc:
        message = str(exc)
        if "HTTP 404" not in message and "HTTP 405" not in message:
            raise

    fallback_payload = _messages_to_generate_payload(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    data = post_json(endpoint_base, fallback_payload)

    if isinstance(data, list) and data:
        generated = str(data[0].get("generated_text", "")).strip()
        prompt_text = str(data[0].get("input_text", "")).strip()
        if prompt_text and generated.startswith(prompt_text):
            generated = generated[len(prompt_text) :].strip()
        return generated

    if isinstance(data, dict) and data.get("generated_text"):
        return str(data["generated_text"]).strip()

    raise RuntimeError(f"Unexpected endpoint response schema: {str(data)[:1000]}")


def run_mode(
    mode_name: str,
    endpoint_url: str,
    token: str,
    model_id: str,
    exemplars: Sequence[EvalRecord],
    targets: Sequence[EvalRecord],
    workspace_root: Path,
    max_images: int,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_backoff_sec: float,
    exemplar_ehr_chars: int,
    target_ehr_chars: int,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    compact_exemplars: List[EvalRecord] = [
        EvalRecord(
            uid=item.uid,
            ehr_text=truncate_text(item.ehr_text, exemplar_ehr_chars),
            diagnosis=item.diagnosis,
            image_paths=item.image_paths,
        )
        for item in exemplars
    ]
    fewshot = make_fewshot_block(compact_exemplars)
    outputs: List[Dict[str, Any]] = []

    for item in targets:
        effective_target_chars = target_ehr_chars
        effective_max_tokens = max_tokens
        effective_max_images = max_images
        used_images: List[str] = []

        start = time.time()
        model_text = ""
        error_message = ""
        success = False

        for oom_attempt in range(4):
            user_text = (
                f"{fewshot}\n\n"
                f"Now diagnose the next patient.\n"
                f"Patient UID: {item.uid}\n"
                f"EHR: {truncate_text(item.ehr_text, effective_target_chars)}\n"
                "Return only one JSON object with keys diagnosis and rationale. "
                "Diagnosis must never be empty."
            )

            user_content: Any = user_text
            used_images = []

            if mode_name == "ehr_plus_images":
                content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
                for raw_path in item.image_paths:
                    if len(used_images) >= effective_max_images:
                        break
                    resolved = resolve_image_path(workspace_root, raw_path)
                    if not resolved:
                        continue
                    try:
                        data_url = image_to_data_url(resolved)
                    except Exception:
                        continue
                    content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
                    used_images.append(str(resolved))
                user_content = content_blocks

            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_content},
            ]

            if dry_run:
                model_text = "Diagnosis: DRY_RUN\nRationale: Dry run only."
                success = True
                break

            try:
                model_text = endpoint_chat_completion(
                    endpoint_url=endpoint_url,
                    token=token,
                    model_id=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                    timeout=timeout,
                    retries=retries,
                    retry_backoff_sec=retry_backoff_sec,
                )
                success = True
                break
            except RuntimeError as exc:
                message = str(exc).lower()
                oom_like = "out of memory" in message or "cuda" in message
                if not oom_like or oom_attempt == 3:
                    error_message = str(exc)
                    break

                effective_target_chars = max(90, int(effective_target_chars * 0.7))
                effective_max_tokens = max(32, int(effective_max_tokens * 0.7))
                if mode_name == "ehr_plus_images" and effective_max_images > 1:
                    effective_max_images -= 1
                time.sleep(min(6, 2 + oom_attempt))

        if not success and not model_text:
            model_text = (
                '{"diagnosis":"","rationale":"Inference failed after retries.'
                ' Endpoint returned an error."}'
            )

        latency = time.time() - start

        pred_diag = parse_diagnosis(model_text)
        exact = int(normalize_for_match(pred_diag) == normalize_for_match(item.diagnosis))
        f1 = token_f1(pred_diag, item.diagnosis)

        outputs.append(
            {
                "mode": mode_name,
                "uid": item.uid,
                "ground_truth_diagnosis": item.diagnosis,
                "predicted_diagnosis": pred_diag,
                "exact_match": exact,
                "token_f1": round(f1, 4),
                "latency_sec": round(latency, 3),
                "effective_target_ehr_chars": effective_target_chars,
                "effective_max_tokens": effective_max_tokens,
                "images_used": used_images,
                "raw_output": model_text,
                "inference_error": error_message,
            }
        )

    return outputs


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"n": 0, "exact_match": 0.0, "avg_token_f1": 0.0, "avg_latency_sec": 0.0}

    n = len(records)
    exact = sum(r["exact_match"] for r in records) / n
    f1 = sum(r["token_f1"] for r in records) / n
    latency = sum(r["latency_sec"] for r in records) / n
    return {
        "n": n,
        "exact_match": round(exact, 4),
        "avg_token_f1": round(f1, 4),
        "avg_latency_sec": round(latency, 3),
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="MedGemma benchmark on MedPix full_dataset.")
    parser.add_argument("--input", default="patient-ehr-image-dataset/full_dataset.jsonl")
    parser.add_argument("--output-dir", default="medgemma_benchmark/output")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--endpoint-url", required=True, help="HF Inference Endpoint base URL")
    parser.add_argument("--model-id", default="google/medgemma-1-5-4b-it-hae")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Env var containing HF token")
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument("--num-targets", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--exemplar-ehr-chars", type=int, default=500)
    parser.add_argument("--target-ehr-chars", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ.get(args.token_env, "")
    if not token and not args.dry_run:
        raise SystemExit(
            f"Missing token in env var '{args.token_env}'. Export it before running."
        )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    workspace_root = Path(args.workspace_root).resolve()

    rows = load_jsonl(input_path)
    records = build_records(rows, require_images=True)
    exemplars, targets = choose_exemplars_and_targets(
        records,
        num_shots=args.num_shots,
        num_targets=args.num_targets,
    )

    ehr_only = run_mode(
        mode_name="ehr_only",
        endpoint_url=args.endpoint_url,
        token=token,
        model_id=args.model_id,
        exemplars=exemplars,
        targets=targets,
        workspace_root=workspace_root,
        max_images=0,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff_sec=args.retry_backoff_sec,
        exemplar_ehr_chars=args.exemplar_ehr_chars,
        target_ehr_chars=args.target_ehr_chars,
        dry_run=args.dry_run,
    )

    ehr_plus_images = run_mode(
        mode_name="ehr_plus_images",
        endpoint_url=args.endpoint_url,
        token=token,
        model_id=args.model_id,
        exemplars=exemplars,
        targets=targets,
        workspace_root=workspace_root,
        max_images=args.max_images,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff_sec=args.retry_backoff_sec,
        exemplar_ehr_chars=args.exemplar_ehr_chars,
        target_ehr_chars=args.target_ehr_chars,
        dry_run=args.dry_run,
    )

    summary = {
        "config": {
            "input": str(input_path),
            "endpoint_url": args.endpoint_url,
            "model_id": args.model_id,
            "num_shots": args.num_shots,
            "num_targets": args.num_targets,
            "max_images": args.max_images,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "retries": args.retries,
            "retry_backoff_sec": args.retry_backoff_sec,
            "exemplar_ehr_chars": args.exemplar_ehr_chars,
            "target_ehr_chars": args.target_ehr_chars,
            "dry_run": args.dry_run,
        },
        "selected_uids": {
            "exemplars": [r.uid for r in exemplars],
            "targets": [r.uid for r in targets],
        },
        "metrics": {
            "ehr_only": summarize(ehr_only),
            "ehr_plus_images": summarize(ehr_plus_images),
        },
    }

    save_json(output_dir / "summary.json", summary)
    save_jsonl(output_dir / "predictions_ehr_only.jsonl", ehr_only)
    save_jsonl(output_dir / "predictions_ehr_plus_images.jsonl", ehr_plus_images)

    print("Benchmark complete.")
    print(f"Output directory: {output_dir}")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
