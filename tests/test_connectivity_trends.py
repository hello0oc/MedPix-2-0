from __future__ import annotations

from scripts.summarize_connectivity_trends import build_markdown, build_summary


def test_build_summary_counts_and_rates():
    reports = [
        {
            "generated_at_utc": "2026-02-20T00:00:00+00:00",
            "checks": {
                "medgemma": {"status": "pass", "latency_sec": 1.2},
                "gemini": {"status": "fail", "latency_sec": 2.0},
            },
        },
        {
            "generated_at_utc": "2026-02-21T00:00:00+00:00",
            "checks": {
                "medgemma": {"status": "fail", "latency_sec": 1.6},
                "gemini": {"status": "skipped"},
            },
        },
        {
            "generated_at_utc": "2026-02-22T00:00:00+00:00",
            "checks": {
                "medgemma": {"status": "pass", "latency_sec": 1.4},
                "gemini": {"status": "pass", "latency_sec": 2.2},
            },
        },
    ]

    summary = build_summary(reports)
    medgemma = summary["services"]["medgemma"]
    gemini = summary["services"]["gemini"]

    assert summary["history_size"] == 3
    assert summary["latest_generated_at_utc"] == "2026-02-22T00:00:00+00:00"
    assert medgemma["counts"]["pass"] == 2
    assert medgemma["counts"]["fail"] == 1
    assert medgemma["pass_rate"] == 0.6667
    assert medgemma["fail_rate"] == 0.3333
    assert medgemma["avg_latency_sec"] == 1.4
    assert medgemma["p95_latency_sec"] == 1.6
    assert gemini["counts"]["skipped"] == 1


def test_build_markdown_renders_table_and_recent_statuses():
    summary = {
        "history_size": 2,
        "latest_generated_at_utc": "2026-02-20T00:00:00+00:00",
        "services": {
            "medgemma": {
                "counts": {"pass": 1, "fail": 1, "skipped": 0, "unknown": 0},
                "pass_rate": 0.5,
                "fail_rate": 0.5,
                "avg_latency_sec": 1.5,
                "p95_latency_sec": 2.0,
                "recent_5_statuses": ["pass", "fail"],
            },
            "gemini": {
                "counts": {"pass": 0, "fail": 0, "skipped": 2, "unknown": 0},
                "pass_rate": 0.0,
                "fail_rate": 0.0,
                "avg_latency_sec": None,
                "p95_latency_sec": None,
                "recent_5_statuses": ["skipped", "skipped"],
            },
        },
    }

    md = build_markdown(summary)
    assert "Connectivity Trend Summary" in md
    assert "| medgemma | 1 | 1 | 0 | 50.00% | 50.00%" in md
    assert "- medgemma: pass, fail" in md
    assert "- gemini: skipped, skipped" in md
