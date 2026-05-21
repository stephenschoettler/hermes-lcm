"""Report output helpers for deterministic LCM benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from .types import ReplayMetrics


def summarize_metrics(metrics: Iterable[ReplayMetrics]) -> dict[str, object]:
    rows = list(metrics)
    if not rows:
        return {"benchmark_version": "1", "runs": 0, "policies": [], "fixtures": []}
    return {
        "benchmark_version": "1",
        "runs": len(rows),
        "policies": sorted({row.policy_name for row in rows}),
        "fixtures": sorted({row.fixture_name for row in rows}),
        "total_failures": sum(len(row.failures) for row in rows),
        "compression_count": sum(row.compression_count for row in rows),
        "compaction_attempts": sum(row.compaction_attempts for row in rows),
        "avg_prompt_tokens_before": mean(row.prompt_tokens_before for row in rows),
        "avg_prompt_tokens_after": mean(row.prompt_tokens_after for row in rows),
        "total_active_canaries_found": sum(row.active_canaries_found for row in rows),
        "total_retrieval_canaries_found": sum(row.retrieval_canaries_found for row in rows),
        "total_canaries": sum(row.total_canaries for row in rows),
        "metrics": [row.to_dict() for row in rows],
    }


def write_metrics_jsonl(path: str | Path, metrics: Iterable[ReplayMetrics]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in metrics:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def write_summary(path: str | Path, metrics: Iterable[ReplayMetrics]) -> dict[str, object]:
    rows = list(metrics)
    summary = summarize_metrics(rows)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
