"""Steady-state per-turn hot-path benchmark.

The existing replay/stress benchmarks time ``compress()``. They do not measure
the path that runs on *every* turn regardless of compaction: the post-LLM
``ingest()`` hook plus ``should_compress_preflight()``. That path re-scans the
active history each turn (quarantine / redaction / placeholder passes, token
counts, ignore/store-id maps), so its cost can grow with conversation length
even when nothing is compacted.

This module replays N turns *without* triggering compaction (a very high
threshold) and records per-turn wall time for ``ingest`` + ``preflight`` at a
range of history sizes, under a few configurations (baseline, ignore-message
patterns on, sensitive patterns on). It is a regression guard: per-turn cost
should stay roughly flat as history grows, not scale with it.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .replay import _ensure_hermes_lcm_package


DEFAULT_HISTORY_SIZES = (200, 1000, 4000)
DEFAULT_ITERATIONS = 25
# Effectively disable compaction so we isolate the ingest/preflight cost.
_NO_COMPACT_CONTEXT_LENGTH = 100_000_000


@dataclass
class SteadyStateCase:
    """One measured configuration."""

    name: str
    ignore_message_patterns: tuple[str, ...] = ()
    sensitive_patterns: tuple[str, ...] = ()


@dataclass
class SteadyStateSample:
    case: str
    history_size: int
    iterations: int
    ingest_p50_ms: float
    ingest_p95_ms: float
    preflight_p50_ms: float
    preflight_p95_ms: float
    row_writes_per_turn: float


@dataclass
class SteadyStateReport:
    history_sizes: tuple[int, ...]
    iterations: int
    samples: list[SteadyStateSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_sizes": list(self.history_sizes),
            "iterations": self.iterations,
            "samples": [sample.__dict__ for sample in self.samples],
        }


DEFAULT_CASES: tuple[SteadyStateCase, ...] = (
    SteadyStateCase(name="baseline"),
    SteadyStateCase(name="ignore_patterns", ignore_message_patterns=("^HEARTBEAT",)),
    SteadyStateCase(
        name="sensitive_patterns",
        sensitive_patterns=("api_key", "bearer_token", "password_assignment", "private_key"),
    ),
)


def _build_engine(case: SteadyStateCase, run_dir: Path):
    _ensure_hermes_lcm_package()
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine

    db_path = run_dir / f"steady_{case.name}.lcm.db"
    hermes_home = run_dir / f"hermes-home-{case.name}"
    hermes_home.mkdir(parents=True, exist_ok=True)

    config = LCMConfig(
        database_path=str(db_path),
        # A large fresh tail + very high context length keep both the preflight
        # threshold and force-overflow recovery from ever firing.
        fresh_tail_count=8,
        leaf_chunk_tokens=100_000,
        context_threshold=0.99,
    )
    if case.ignore_message_patterns:
        setattr(config, "ignore_message_patterns", list(case.ignore_message_patterns))
    if case.sensitive_patterns:
        setattr(config, "sensitive_patterns_enabled", True)
        setattr(config, "sensitive_patterns", list(case.sensitive_patterns))

    engine = LCMEngine(config=config, hermes_home=str(hermes_home))
    engine.on_session_start(
        f"steady-{case.name}",
        platform="benchmark",
        conversation_id=f"steady-{case.name}",
        model="benchmark-model",
        provider="benchmark",
        context_length=_NO_COMPACT_CONTEXT_LENGTH,
    )
    engine.update_model(
        model="benchmark-model",
        context_length=_NO_COMPACT_CONTEXT_LENGTH,
        provider="benchmark",
    )
    return engine


def _synthetic_turn(index: int) -> list[dict[str, Any]]:
    # A user/assistant pair with enough varied content to exercise the passes
    # without tripping externalization or externally-observable heuristics.
    return [
        {
            "role": "user",
            "content": (
                f"Turn {index}: please review module_{index % 37}.py and summarize "
                f"the change to function handler_{index % 53} regarding value {index * 7}."
            ),
        },
        {
            "role": "assistant",
            "content": (
                f"Reviewed module_{index % 37}.py. handler_{index % 53} now clamps "
                f"value {index * 7} and returns early on empty input; no other changes."
            ),
        },
    ]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _total_changes(engine: Any) -> int:
    total = 0
    for attr in ("_store", "_dag", "_lifecycle"):
        holder = getattr(engine, attr, None)
        conn = getattr(holder, "_conn", None)
        if conn is not None:
            try:
                total += int(conn.total_changes)
            except Exception:
                pass
    return total


def _measure_case(
    case: SteadyStateCase,
    run_dir: Path,
    history_sizes: tuple[int, ...],
    iterations: int,
) -> list[SteadyStateSample]:
    samples: list[SteadyStateSample] = []

    for target in sorted(history_sizes):
        target_run_dir = run_dir / f"{case.name}-{target}"
        engine = _build_engine(case, target_run_dir)
        history: list[dict[str, Any]] = []
        next_index = 0
        try:
            # Build an independent history for each target so a nearby smaller
            # target cannot pre-populate store/identity maps for the next one.
            while len(history) < target:
                history.extend(_synthetic_turn(next_index))
                next_index += 1
                engine.ingest(history)

            ingest_ms: list[float] = []
            preflight_ms: list[float] = []
            writes: list[int] = []
            for _ in range(iterations):
                history.extend(_synthetic_turn(next_index))
                next_index += 1

                before_changes = _total_changes(engine)
                t0 = time.perf_counter()
                engine.ingest(history)
                t1 = time.perf_counter()
                engine.should_compress_preflight(history)
                t2 = time.perf_counter()

                ingest_ms.append((t1 - t0) * 1000.0)
                preflight_ms.append((t2 - t1) * 1000.0)
                writes.append(max(0, _total_changes(engine) - before_changes))

            samples.append(
                SteadyStateSample(
                    case=case.name,
                    history_size=len(history),
                    iterations=iterations,
                    ingest_p50_ms=round(statistics.median(ingest_ms), 4),
                    ingest_p95_ms=round(_percentile(ingest_ms, 95), 4),
                    preflight_p50_ms=round(statistics.median(preflight_ms), 4),
                    preflight_p95_ms=round(_percentile(preflight_ms, 95), 4),
                    row_writes_per_turn=round(statistics.mean(writes), 2) if writes else 0.0,
                )
            )
        finally:
            engine.shutdown()
    return samples


def _ignore_message_filtering_active(case: SteadyStateCase) -> bool:
    """Return whether an ignore-message benchmark case can exercise filtering."""

    if not case.ignore_message_patterns:
        return True
    _ensure_hermes_lcm_package()
    from hermes_lcm.message_patterns import compile_message_patterns

    return bool(compile_message_patterns(case.ignore_message_patterns))


def run_steady_state(
    run_dir: Path,
    *,
    history_sizes: tuple[int, ...] = DEFAULT_HISTORY_SIZES,
    iterations: int = DEFAULT_ITERATIONS,
    cases: tuple[SteadyStateCase, ...] = DEFAULT_CASES,
    progress: Optional[Callable[[str], None]] = None,
) -> SteadyStateReport:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"steady-state output directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    report = SteadyStateReport(history_sizes=tuple(sorted(history_sizes)), iterations=iterations)
    for case in cases:
        if not _ignore_message_filtering_active(case):
            if progress is not None:
                progress(
                    f"skipping inactive case: {case.name} "
                    "(ignore message regex filtering unavailable)"
                )
            continue
        if progress is not None:
            progress(f"measuring case: {case.name}")
        report.samples.extend(_measure_case(case, run_dir, history_sizes, iterations))
    return report


def format_report(report: SteadyStateReport) -> str:
    lines = [
        f"Steady-state per-turn hot path (iterations={report.iterations})",
        f"history sizes: {', '.join(str(s) for s in report.history_sizes)}",
        "",
        f"{'case':<20}{'history':>9}{'ingest p50':>13}{'ingest p95':>13}"
        f"{'pre p50':>10}{'pre p95':>10}{'writes/turn':>13}",
    ]
    for sample in report.samples:
        lines.append(
            f"{sample.case:<20}{sample.history_size:>9}"
            f"{sample.ingest_p50_ms:>12.3f}m{sample.ingest_p95_ms:>12.3f}m"
            f"{sample.preflight_p50_ms:>9.3f}m{sample.preflight_p95_ms:>9.3f}m"
            f"{sample.row_writes_per_turn:>13.2f}"
        )
    return "\n".join(lines)
