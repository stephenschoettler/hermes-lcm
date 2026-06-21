"""Tests for the deterministic benchmark CLI."""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_benchmark_cli():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "lcm_benchmark.py"
    spec = importlib.util.spec_from_file_location("lcm_benchmark_cli", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _block_agent_imports(monkeypatch):
    for name in list(sys.modules):
        if name == "agent" or name.startswith("agent.") or name == "hermes_lcm" or name.startswith("hermes_lcm."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__
    real_import_module = importlib.import_module

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and (name == "agent" or name.startswith("agent.")) and name not in sys.modules:
            missing = "agent.context_engine" if name.startswith("agent.") else "agent"
            raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)
        return real_import(name, globals, locals, fromlist, level)

    def blocked_import_module(name, package=None):
        if name == "agent.context_engine" and name not in sys.modules:
            raise ModuleNotFoundError("No module named 'agent.context_engine'", name=name)
        return real_import_module(name, package)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    monkeypatch.setattr(importlib, "import_module", blocked_import_module)


def test_cli_synthetic_fixture_runs_without_hermes_agent_importable(tmp_path, monkeypatch):
    cli = _load_benchmark_cli()
    _block_agent_imports(monkeypatch)

    result = cli.main([
        "--synthetic-fixture",
        "standalone_probe:2:1:3",
        "--policy",
        "benchmarks/policies/pressure_smoke.yaml",
        "--output",
        str(tmp_path),
        "--allow-external-output",
        "--json",
    ])

    assert result == 0
    assert (tmp_path / "summary.json").exists()
    assert not hasattr(sys.modules["agent.context_engine"], "__file__")


def test_cli_accepts_synthetic_fixture_specs(tmp_path):
    cli = _load_benchmark_cli()

    result = cli.main([
        "--synthetic-fixture",
        "cli_probe:2:1:3",
        "--policy",
        "benchmarks/policies/pressure_smoke.yaml",
        "--output",
        str(tmp_path),
        "--allow-external-output",
        "--json",
    ])

    summary = json.loads((tmp_path / "summary.json").read_text())
    metrics = json.loads((tmp_path / "metrics.jsonl").read_text())

    assert result == 0
    assert summary["fixtures"] == ["cli_probe"]
    assert summary["policies"] == ["pressure_smoke"]
    assert metrics["fixture_name"] == "cli_probe"
    assert metrics["policy_name"] == "pressure_smoke"


def test_cli_missing_fixture_does_not_create_output_directory(tmp_path):
    cli = _load_benchmark_cli()
    output_dir = tmp_path / "missing-input-output"

    with pytest.raises(SystemExit, match="At least one --fixture or --synthetic-fixture is required"):
        cli.main([
            "--output",
            str(output_dir),
            "--allow-external-output",
        ])

    assert not output_dir.exists()


def test_cli_empty_argv_does_not_fall_back_to_process_argv(tmp_path, monkeypatch):
    cli = _load_benchmark_cli()
    output_dir = tmp_path / "process-argv-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pytest",
            "--synthetic-fixture",
            "argv_leak:1:1:1",
            "--policy",
            "benchmarks/policies/pressure_smoke.yaml",
            "--output",
            str(output_dir),
            "--allow-external-output",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main([])

    assert excinfo.value.code == 2
    assert not output_dir.exists()


def test_cli_invalid_synthetic_spec_does_not_create_output_directory(tmp_path):
    cli = _load_benchmark_cli()
    output_dir = tmp_path / "invalid-synthetic-output"

    with pytest.raises(ValueError, match="message_pairs must be positive"):
        cli.main([
            "--synthetic-fixture",
            "bad:0:1:5",
            "--output",
            str(output_dir),
            "--allow-external-output",
        ])

    assert not output_dir.exists()


def test_cli_missing_fixture_path_does_not_create_output_directory(tmp_path):
    cli = _load_benchmark_cli()
    output_dir = tmp_path / "missing-fixture-output"

    with pytest.raises(FileNotFoundError):
        cli.main([
            "--fixture",
            "benchmarks/fixtures/does-not-exist.json",
            "--output",
            str(output_dir),
            "--allow-external-output",
        ])

    assert not output_dir.exists()


def test_cli_missing_policy_path_does_not_create_output_directory(tmp_path):
    cli = _load_benchmark_cli()
    output_dir = tmp_path / "missing-policy-output"

    with pytest.raises(FileNotFoundError):
        cli.main([
            "--synthetic-fixture",
            "policy_probe:1:1:1",
            "--policy",
            "benchmarks/policies/does-not-exist.yaml",
            "--output",
            str(output_dir),
            "--allow-external-output",
        ])

    assert not output_dir.exists()


def test_cli_writes_scrubbed_community_export(tmp_path):
    cli = _load_benchmark_cli()
    output_dir = tmp_path / "benchmark-output"
    export_path = output_dir / "community-export.json"

    result = cli.main([
        "--synthetic-fixture",
        "export_probe:2:1:3",
        "--policy",
        "benchmarks/policies/codex_gpt_long_context.yaml",
        "--output",
        str(output_dir),
        "--allow-external-output",
        "--export",
        str(export_path),
        "--provider",
        "openai-codex",
        "--model",
        "gpt-5.5",
    ])

    export = json.loads(export_path.read_text())
    serialized = json.dumps(export, sort_keys=True)

    assert result == 0
    assert export["schema_version"] == "1"
    assert export["provider"] == "openai-codex"
    assert export["model"] == "gpt-5.5"
    assert export["transcript_contents_included"] is False
    assert export["policy_settings"]["codex_gpt_long_context@1"]["context_length"] == 272000
    assert "notes" not in export["policy_settings"]["codex_gpt_long_context@1"]
    assert "database_path" not in serialized
    assert "hermes_home" not in serialized
    assert "messages" not in serialized


def test_cli_export_outside_output_directory_is_rejected(tmp_path):
    cli = _load_benchmark_cli()
    output_dir = Path("benchmarks/runs") / f"export-policy-test-{tmp_path.name}"
    export_path = tmp_path / "community-export.json"

    with pytest.raises(SystemExit, match="Refusing --export outside output directory"):
        cli.main([
            "--synthetic-fixture",
            "export_probe:2:1:3",
            "--policy",
            "benchmarks/policies/codex_gpt_long_context.yaml",
            "--output",
            str(output_dir),
            "--export",
            str(export_path),
        ])

    assert not export_path.exists()
    assert not output_dir.exists()


def test_cli_export_refuses_existing_repo_file():
    cli = _load_benchmark_cli()
    readme_path = Path("README.md")
    original_readme = readme_path.read_text(encoding="utf-8")

    with pytest.raises(SystemExit, match="Refusing to overwrite existing export file"):
        cli.main([
            "--synthetic-fixture",
            "export_probe:2:1:3",
            "--policy",
            "benchmarks/policies/codex_gpt_long_context.yaml",
            "--output",
            ".",
            "--export",
            "README.md",
        ])

    assert readme_path.read_text(encoding="utf-8") == original_readme


def test_cli_export_refuses_git_directory():
    cli = _load_benchmark_cli()

    with pytest.raises(SystemExit, match="Refusing --export inside .git"):
        cli.main([
            "--synthetic-fixture",
            "export_probe:2:1:3",
            "--policy",
            "benchmarks/policies/codex_gpt_long_context.yaml",
            "--output",
            ".",
            "--export",
            ".git/config",
        ])
