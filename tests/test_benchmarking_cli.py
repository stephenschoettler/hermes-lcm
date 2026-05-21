"""Tests for the deterministic benchmark CLI."""

from __future__ import annotations

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
