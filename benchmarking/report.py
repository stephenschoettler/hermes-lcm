"""Report output helpers for deterministic LCM benchmark runs."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from .types import LCMPolicy, ReplayMetrics

BENCHMARK_VERSION = "2"
FRESH_TAIL_PRESSURE_THRESHOLD = 0.30


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_mean(values: Iterable[int | float]) -> float:
    selected = list(values)
    return mean(selected) if selected else 0.0


def _safe_ratio(numerator: int | float, denominator: int | float, *, default: float = 0.0) -> float:
    if not denominator:
        return default
    return numerator / denominator


def _fixture_suite(rows: list[ReplayMetrics]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = grouped.setdefault(
            row.fixture_name,
            {"name": row.fixture_name, "runs": 0, "tags": set()},
        )
        entry["runs"] += 1
        entry["tags"].update(row.fixture_tags)
    return [
        {"name": name, "runs": entry["runs"], "tags": sorted(entry["tags"])}
        for name, entry in sorted(grouped.items())
    ]


def _policy_versions(rows: list[ReplayMetrics]) -> dict[str, str | list[str]]:
    versions_by_policy: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        versions_by_policy[row.policy_name].add(row.policy_version)
    return {
        policy_name: versions[0] if len(versions) == 1 else versions
        for policy_name, raw_versions in sorted(versions_by_policy.items())
        for versions in [sorted(raw_versions)]
    }


def _summary_failure_modes(rows: list[ReplayMetrics]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        mode = row.summary_failure_mode.value if hasattr(row.summary_failure_mode, "value") else str(row.summary_failure_mode)
        counts[mode] += 1
    return dict(sorted(counts.items()))


def _summary_level_runs(rows: list[ReplayMetrics]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.summary_level)] += 1
    return dict(sorted(counts.items()))


def _metric_summary(rows: list[ReplayMetrics]) -> dict[str, object]:
    total_canaries = sum(row.total_canaries for row in rows)
    return {
        "total_failures": sum(len(row.failures) for row in rows),
        "compression_count": sum(row.compression_count for row in rows),
        "compaction_attempts": sum(row.compaction_attempts for row in rows),
        "repeated_compaction_risk_count": sum(1 for row in rows if row.repeated_compaction_risk),
        "summary_failure_modes": _summary_failure_modes(rows),
        "summary_level_runs": _summary_level_runs(rows),
        "fresh_tail_pressure_events": sum(
            1 for row in rows if row.fresh_tail_pressure_ratio >= FRESH_TAIL_PRESSURE_THRESHOLD
        ),
        "avg_prompt_tokens_before": _safe_mean(row.prompt_tokens_before for row in rows),
        "avg_prompt_tokens_after": _safe_mean(row.prompt_tokens_after for row in rows),
        "avg_post_compaction_headroom_tokens": _safe_mean(
            row.post_compaction_headroom_tokens for row in rows
        ),
        "min_post_compaction_headroom_tokens": min(
            (row.post_compaction_headroom_tokens for row in rows), default=0
        ),
        "total_active_canaries_found": sum(row.active_canaries_found for row in rows),
        "total_retrieval_canaries_found": sum(row.retrieval_canaries_found for row in rows),
        "total_canaries": total_canaries,
        "active_canary_recall": _safe_ratio(
            sum(row.active_canaries_found for row in rows), total_canaries, default=1.0
        ),
        "retrieval_canary_recall": _safe_ratio(
            sum(row.retrieval_canaries_found for row in rows), total_canaries, default=1.0
        ),
    }


def compare_policies(metrics: Iterable[ReplayMetrics]) -> list[dict[str, object]]:
    """Return ranked aggregate policy comparison rows."""
    grouped: dict[tuple[str, str], list[ReplayMetrics]] = defaultdict(list)
    for row in metrics:
        grouped[(row.policy_name, row.policy_version)].append(row)

    comparison: list[dict[str, object]] = []
    for (policy_name, policy_version), rows in grouped.items():
        summary = _metric_summary(rows)
        runs = len(rows)
        failure_count = int(summary["total_failures"])
        risk_count = int(summary["repeated_compaction_risk_count"])
        pressure_events = int(summary["fresh_tail_pressure_events"])
        active_recall = float(summary["active_canary_recall"])
        retrieval_recall = float(summary["retrieval_canary_recall"])
        stability = 1.0 - _safe_ratio(risk_count, runs)
        recall_score = (retrieval_recall * 0.70) + (active_recall * 0.30)
        score = round(
            (recall_score * 100.0)
            + (stability * 20.0)
            - (failure_count * 100.0)
            - (pressure_events * 5.0),
            3,
        )
        comparison.append({
            "policy_name": policy_name,
            "policy_version": policy_version,
            "policy_key": f"{policy_name}@{policy_version}",
            "runs": runs,
            "score": score,
            **summary,
        })

    return sorted(
        comparison,
        key=lambda row: (-float(row["score"]), str(row["policy_key"])),
    )


def summarize_metrics(metrics: Iterable[ReplayMetrics]) -> dict[str, object]:
    rows = list(metrics)
    if not rows:
        return {
            "benchmark_version": BENCHMARK_VERSION,
            "generated_at_utc": _utc_timestamp(),
            "runs": 0,
            "policies": [],
            "policy_versions": {},
            "fixtures": [],
            "fixture_suite": [],
            "metric_summary": _metric_summary([]),
            "policy_comparison": [],
            "metrics": [],
        }
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at_utc": _utc_timestamp(),
        "runs": len(rows),
        "policies": sorted({row.policy_name for row in rows}),
        "policy_versions": _policy_versions(rows),
        "fixtures": sorted({row.fixture_name for row in rows}),
        "fixture_suite": _fixture_suite(rows),
        "total_failures": sum(len(row.failures) for row in rows),
        "compression_count": sum(row.compression_count for row in rows),
        "compaction_attempts": sum(row.compaction_attempts for row in rows),
        "avg_prompt_tokens_before": mean(row.prompt_tokens_before for row in rows),
        "avg_prompt_tokens_after": mean(row.prompt_tokens_after for row in rows),
        "total_active_canaries_found": sum(row.active_canaries_found for row in rows),
        "total_retrieval_canaries_found": sum(row.retrieval_canaries_found for row in rows),
        "total_canaries": sum(row.total_canaries for row in rows),
        "metric_summary": _metric_summary(rows),
        "policy_comparison": compare_policies(rows),
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


def _policy_settings(policies: Iterable[LCMPolicy]) -> dict[str, dict[str, object]]:
    settings: dict[str, dict[str, object]] = {}
    for policy in policies:
        data = policy.to_dict()
        data.pop("notes", None)
        settings[f"{policy.name}@{policy.policy_version}"] = data
    return settings


def build_community_export(
    summary: Mapping[str, Any],
    *,
    policies: Iterable[LCMPolicy],
    provider: str = "",
    model: str = "",
) -> dict[str, object]:
    """Build a scrubbed benchmark result export for community sharing.

    The export intentionally omits per-run metrics because those contain local
    paths such as ``database_path`` and ``hermes_home``. It includes only the
    aggregate comparison/provenance fields needed to reproduce and discuss a
    preset result.
    """

    return {
        "schema_version": "1",
        "benchmark_version": summary.get("benchmark_version", BENCHMARK_VERSION),
        "generated_at_utc": summary.get("generated_at_utc", _utc_timestamp()),
        "provider": provider,
        "model": model,
        "transcript_contents_included": False,
        "fixtures": list(summary.get("fixtures", [])),
        "fixture_suite": list(summary.get("fixture_suite", [])),
        "policies": list(summary.get("policies", [])),
        "policy_versions": dict(summary.get("policy_versions", {})),
        "policy_settings": _policy_settings(policies),
        "metric_summary": dict(summary.get("metric_summary", {})),
        "policy_comparison": list(summary.get("policy_comparison", [])),
    }


def write_community_export(
    path: str | Path,
    summary: Mapping[str, Any],
    *,
    policies: Iterable[LCMPolicy],
    provider: str = "",
    model: str = "",
) -> dict[str, object]:
    export = build_community_export(summary, policies=policies, provider=provider, model=model)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(export, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return export
