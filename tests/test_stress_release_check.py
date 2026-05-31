"""Tests for the deterministic LCM stress release-check CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest


def _load_stress_cli():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "lcm_stress_check.py"
    spec = importlib.util.spec_from_file_location("lcm_stress_check_cli", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_results_hash(results: dict) -> str:
    payload = dict(results)
    payload.pop("artifact_hashes", None)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return hashlib.sha256(data).hexdigest()


def _assert_path_under(path_value: str, root: Path) -> None:
    assert Path(path_value).resolve().is_relative_to(root.resolve())


def test_stress_cli_refuses_non_empty_output_directory(tmp_path):
    cli = _load_stress_cli()
    output_dir = tmp_path / "existing-output"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("old run", encoding="utf-8")

    with pytest.raises(SystemExit, match="Refusing to reuse non-empty output directory"):
        cli.main([
            "--output",
            str(output_dir),
            "--tier",
            "smoke",
        ])


@pytest.mark.filterwarnings("ignore:.*__package__ != __spec__.*:DeprecationWarning")
def test_stress_cli_smoke_writes_results_summary_and_uses_output_sandbox(tmp_path):
    cli = _load_stress_cli()
    output_dir = tmp_path / "stress-smoke"

    result = cli.main([
        "--output",
        str(output_dir),
        "--tier",
        "smoke",
        "--json",
    ])

    results_path = output_dir / "results" / "stress-results.json"
    summary_path = output_dir / "stress-summary.md"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    serialized = json.dumps(results, sort_keys=True)

    assert result == 0
    assert results_path.exists()
    assert summary_path.exists()
    assert results["tier"] == "smoke"
    assert results["failure_count"] == 0
    assert set(results["cases"]) >= {
        "multi_cycle_canary_recall",
        "redaction_and_externalization_boundaries",
        "cross_session_scope_and_pagination",
        "query_fuzz_no_crash",
        "concurrent_read_write_smoke",
    }
    assert all(case["ok"] is True for case in results["cases"].values())
    assert str(output_dir) in serialized
    _assert_path_under(results["run_dir"], output_dir)
    _assert_path_under(results["results_path"], output_dir)
    _assert_path_under(results["summary_path"], output_dir)
    for case_payload in results["cases"].values():
        for externalized_file in case_payload.get("externalized_files", []):
            _assert_path_under(externalized_file, output_dir)
    assert "failure_count: 0" in summary_path.read_text(encoding="utf-8")
    assert results["artifact_hashes"] == {
        "results/stress-results.json_without_artifact_hashes": _canonical_results_hash(results),
        "stress-summary.md": _sha256(summary_path),
    }


def test_stress_cli_rejects_existing_file_output_path(tmp_path):
    cli = _load_stress_cli()
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("not a dir", encoding="utf-8")

    with pytest.raises(SystemExit, match="Output path exists and is not a directory"):
        cli.main(["--output", str(output_file), "--tier", "smoke"])


def test_stress_run_blanks_provider_keys_and_restores_environment(tmp_path, monkeypatch):
    from benchmarking import stress

    monkeypatch.setenv("HOME", "/tmp/original-home")
    monkeypatch.setenv("HERMES_HOME", "/tmp/original-hermes")
    monkeypatch.setenv("OPENAI_API_KEY", "live-openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "live-openrouter")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "live-anthropic")
    observed = {}

    def env_probe(run):
        observed["HOME"] = os.environ.get("HOME")
        observed["HERMES_HOME"] = os.environ.get("HERMES_HOME")
        observed["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
        observed["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY")
        observed["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY")
        run.record("env_probe", "ok", True)

    monkeypatch.setitem(stress._SCENARIO_FUNCTIONS, "env_probe", env_probe)

    result = stress.run_stress_check(output_dir=tmp_path / "env-run", tier="smoke", scenarios=["env_probe"])

    assert result["failure_count"] == 0
    assert observed["HOME"].startswith(str(tmp_path / "env-run"))
    assert observed["HERMES_HOME"].startswith(str(tmp_path / "env-run"))
    assert observed["OPENAI_API_KEY"] == ""
    assert observed["OPENROUTER_API_KEY"] == ""
    assert observed["ANTHROPIC_API_KEY"] == ""
    assert os.environ["HOME"] == "/tmp/original-home"
    assert os.environ["HERMES_HOME"] == "/tmp/original-hermes"
    assert os.environ["OPENAI_API_KEY"] == "live-openai"
    assert os.environ["OPENROUTER_API_KEY"] == "live-openrouter"
    assert os.environ["ANTHROPIC_API_KEY"] == "live-anthropic"


@pytest.mark.filterwarnings("ignore:.*__package__ != __spec__.*:DeprecationWarning")
def test_stress_runner_reloads_partial_hermes_lcm_submodules(tmp_path, monkeypatch):
    from benchmarking import stress

    partial_engine = types.ModuleType("hermes_lcm.engine")
    partial_engine.__file__ = str(Path(__file__).resolve().parents[1] / "engine.py")
    monkeypatch.setitem(sys.modules, "hermes_lcm.engine", partial_engine)

    def module_probe(run):
        import hermes_lcm.engine as engine_mod

        run.record("module_probe", "has_summarizer", hasattr(engine_mod, "summarize_with_escalation"))

    monkeypatch.setitem(stress._SCENARIO_FUNCTIONS, "module_probe", module_probe)

    result = stress.run_stress_check(output_dir=tmp_path / "partial-module-run", tier="smoke", scenarios=["module_probe"])

    assert result["failure_count"] == 0
    assert result["cases"]["module_probe"]["has_summarizer"] is True


def test_stress_cli_exits_nonzero_when_any_case_fails(tmp_path, monkeypatch):
    cli = _load_stress_cli()

    def fake_run_stress_check(*args, **kwargs):
        output_dir = Path(kwargs["output_dir"])
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True)
        payload = {
            "failure_count": 1,
            "failures": [{"case": "fake", "bug_id": "fake.failure"}],
            "cases": {"fake": {"ok": False}},
            "results_path": str(results_dir / "stress-results.json"),
            "summary_path": str(output_dir / "stress-summary.md"),
        }
        Path(payload["results_path"]).write_text(json.dumps(payload), encoding="utf-8")
        Path(payload["summary_path"]).write_text("failure_count: 1\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(cli, "run_stress_check", fake_run_stress_check)

    assert cli.main(["--output", str(tmp_path / "out"), "--tier", "smoke"]) == 1
