#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


SERVICES = ("medgemma", "gemini")


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _load_reports(history_dir: Path) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for path in sorted(history_dir.glob("connectivity_report_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_source"] = str(path)
            payload["_generated_dt"] = _parse_ts(str(payload.get("generated_at_utc", "")))
            reports.append(payload)
        except Exception:
            continue
    reports.sort(key=lambda item: item.get("_generated_dt", datetime.min))
    return reports


def _service_stats(reports: List[Dict[str, Any]], service: str) -> Dict[str, Any]:
    status_counts = {"pass": 0, "fail": 0, "skipped": 0, "unknown": 0}
    latencies: List[float] = []
    recent_statuses: List[str] = []

    for report in reports:
        check = (report.get("checks") or {}).get(service) or {}
        status = str(check.get("status", "unknown")).lower()
        status_counts[status if status in status_counts else "unknown"] += 1
        if isinstance(check.get("latency_sec"), (int, float)):
            latencies.append(float(check["latency_sec"]))
        recent_statuses.append(status)

    n = len(reports)
    fail_rate = (status_counts["fail"] / n) if n else 0.0
    pass_rate = (status_counts["pass"] / n) if n else 0.0
    avg_latency = (sum(latencies) / len(latencies)) if latencies else None
    p95_latency = None
    if latencies:
        values = sorted(latencies)
        idx = int(round(0.95 * (len(values) - 1)))
        p95_latency = values[idx]

    return {
        "total_runs": n,
        "counts": status_counts,
        "pass_rate": round(pass_rate, 4),
        "fail_rate": round(fail_rate, 4),
        "avg_latency_sec": round(avg_latency, 3) if avg_latency is not None else None,
        "p95_latency_sec": round(p95_latency, 3) if p95_latency is not None else None,
        "recent_5_statuses": recent_statuses[-5:],
    }


def build_summary(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_service = {service: _service_stats(reports, service) for service in SERVICES}
    latest_report = reports[-1] if reports else {}
    return {
        "history_size": len(reports),
        "latest_generated_at_utc": latest_report.get("generated_at_utc"),
        "services": by_service,
    }


def build_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = ["# Connectivity Trend Summary", ""]
    lines.append(f"- History size: {summary.get('history_size', 0)}")
    lines.append(f"- Latest report: {summary.get('latest_generated_at_utc') or 'n/a'}")
    lines.append("")
    lines.append("| Service | Pass | Fail | Skipped | Pass rate | Fail rate | Avg latency (s) | P95 latency (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    services = summary.get("services", {})
    for service in SERVICES:
        item = services.get(service, {})
        counts = item.get("counts", {})
        lines.append(
            "| {service} | {pass_n} | {fail_n} | {skip_n} | {pass_rate:.2%} | {fail_rate:.2%} | {avg} | {p95} |".format(
                service=service,
                pass_n=counts.get("pass", 0),
                fail_n=counts.get("fail", 0),
                skip_n=counts.get("skipped", 0),
                pass_rate=float(item.get("pass_rate", 0.0)),
                fail_rate=float(item.get("fail_rate", 0.0)),
                avg=item.get("avg_latency_sec", "n/a"),
                p95=item.get("p95_latency_sec", "n/a"),
            )
        )

    lines.append("")
    lines.append("## Recent statuses (last 5)")
    lines.append("")
    for service in SERVICES:
        item = services.get(service, {})
        statuses = item.get("recent_5_statuses") or []
        lines.append(f"- {service}: {', '.join(statuses) if statuses else 'n/a'}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize connectivity report trends")
    parser.add_argument(
        "--history-dir",
        default="medgemma_gui/output/connectivity_history",
        help="Directory containing connectivity_report_*.json files",
    )
    parser.add_argument(
        "--output-json",
        default="medgemma_gui/output/connectivity_trend_summary.json",
        help="Summary JSON output path",
    )
    parser.add_argument(
        "--output-md",
        default="medgemma_gui/output/connectivity_trend_summary.md",
        help="Summary Markdown output path",
    )
    args = parser.parse_args()

    history_dir = Path(args.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    reports = _load_reports(history_dir)
    summary = build_summary(reports)
    markdown = build_markdown(summary)

    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
