"""Tests for the steady-state per-turn hot-path benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from benchmarking.steady_state import (
    SteadyStateCase,
    format_report,
    run_steady_state,
)


class _FakeRegexPattern:
    pattern = "^HEARTBEAT"

    def search(self, text, timeout=None):
        return None


class _FakeRegexEngine:
    error = ValueError

    @staticmethod
    def compile(pattern):
        return _FakeRegexPattern()


def _enable_message_regex(monkeypatch):
    import hermes_lcm.message_patterns as message_patterns

    monkeypatch.setattr(message_patterns, "_regex_engine", _FakeRegexEngine)
    monkeypatch.setattr(message_patterns, "_MISSING_REGEX_WARNING_EMITTED", False)


def _load_steady_state_cli():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "lcm_steady_state_bench.py"
    spec = importlib.util.spec_from_file_location("lcm_steady_state_bench_cli", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_steady_state_report_covers_all_cases_and_sizes(tmp_path, monkeypatch):
    _enable_message_regex(monkeypatch)

    report = run_steady_state(tmp_path, history_sizes=(20, 60), iterations=3)

    # One sample per (case, history size). Default cases: baseline,
    # ignore_patterns, sensitive_patterns.
    assert len(report.samples) == 3 * 2
    assert {s.case for s in report.samples} == {
        "baseline",
        "ignore_patterns",
        "sensitive_patterns",
    }
    # History grows in whole turns (2 msgs), so sizes are >= the requested targets.
    baseline_sizes = sorted(s.history_size for s in report.samples if s.case == "baseline")
    assert baseline_sizes[0] >= 20
    assert baseline_sizes[1] >= 60
    for sample in report.samples:
        assert sample.iterations == 3
        assert sample.ingest_p50_ms >= 0.0
        assert sample.ingest_p95_ms >= sample.ingest_p50_ms
        assert sample.preflight_p95_ms >= sample.preflight_p50_ms


def test_steady_state_does_not_compact(tmp_path):
    # The whole point is to isolate ingest/preflight, so compaction must never
    # fire during the run (very high threshold + context length).
    report = run_steady_state(
        tmp_path,
        history_sizes=(40,),
        iterations=2,
        cases=(SteadyStateCase(name="baseline"),),
    )
    assert len(report.samples) == 1
    assert "baseline" in format_report(report)


def test_steady_state_report_serializes(tmp_path):
    report = run_steady_state(
        tmp_path,
        history_sizes=(20,),
        iterations=2,
        cases=(SteadyStateCase(name="baseline"),),
    )
    data = report.to_dict()
    assert data["iterations"] == 2
    assert data["history_sizes"] == [20]
    assert data["samples"] and data["samples"][0]["case"] == "baseline"


def test_steady_state_rejects_populated_output_directory(tmp_path):
    (tmp_path / "old.json").write_text("{}")

    try:
        run_steady_state(tmp_path, history_sizes=(20,), iterations=1, cases=(SteadyStateCase(name="baseline"),))
    except FileExistsError as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("expected FileExistsError")


def test_steady_state_isolates_each_target_size(tmp_path, monkeypatch):
    import benchmarking.steady_state as steady

    built = []
    original = steady._build_engine

    def tracking_build(case, run_dir):
        built.append(run_dir.name)
        return original(case, run_dir)

    monkeypatch.setattr(steady, "_build_engine", tracking_build)
    report = run_steady_state(
        tmp_path,
        history_sizes=(20, 22),
        iterations=1,
        cases=(SteadyStateCase(name="baseline"),),
    )

    assert len(report.samples) == 2
    assert built == ["baseline-20", "baseline-22"]


def test_steady_state_skips_ignore_case_when_regex_filtering_is_inactive(tmp_path, monkeypatch):
    import hermes_lcm.message_patterns as message_patterns

    monkeypatch.setattr(message_patterns, "_regex_engine", None)
    monkeypatch.setattr(message_patterns, "_MISSING_REGEX_WARNING_EMITTED", False)
    progress: list[str] = []

    report = run_steady_state(
        tmp_path,
        history_sizes=(20,),
        iterations=1,
        progress=progress.append,
    )

    assert {sample.case for sample in report.samples} == {"baseline", "sensitive_patterns"}
    assert all(sample.case != "ignore_patterns" for sample in report.samples)
    assert any(
        "skipping inactive case: ignore_patterns" in message
        and "unavailable" in message
        for message in progress
    )


def test_steady_state_cli_suppresses_inactive_ignore_case(tmp_path, monkeypatch, capsys):
    import hermes_lcm.message_patterns as message_patterns

    monkeypatch.setattr(message_patterns, "_regex_engine", None)
    monkeypatch.setattr(message_patterns, "_MISSING_REGEX_WARNING_EMITTED", False)
    cli = _load_steady_state_cli()

    result = cli.main([
        "--history-size",
        "20",
        "--iterations",
        "1",
        "--output",
        str(tmp_path),
        "--json",
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert result == 0
    assert {sample["case"] for sample in payload["samples"]} == {"baseline", "sensitive_patterns"}
    assert "ignore_patterns" not in captured.out
    assert "[steady-state] skipping inactive case: ignore_patterns" in captured.err
