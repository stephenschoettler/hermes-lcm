"""Tests for the steady-state per-turn hot-path benchmark."""

from benchmarking.steady_state import (
    SteadyStateCase,
    format_report,
    run_steady_state,
)


def test_steady_state_report_covers_all_cases_and_sizes(tmp_path):
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
