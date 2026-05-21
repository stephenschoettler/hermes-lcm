#!/usr/bin/env python3
"""Run deterministic hermes-lcm benchmark replays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.fixtures import load_fixtures
from benchmarking.policies import load_policies
from benchmarking.replay import run_replays
from benchmarking.report import write_metrics_jsonl, write_summary


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="append", default=[], help="Fixture JSON path. Repeatable.")
    parser.add_argument("--policy", action="append", default=[], help="Policy JSON/YAML path. Repeatable.")
    parser.add_argument("--output", required=True, help="Benchmark output directory.")
    parser.add_argument("--json", action="store_true", help="Print summary JSON to stdout.")
    parser.add_argument(
        "--allow-external-output",
        action="store_true",
        help="Allow --output outside this repository.",
    )
    return parser.parse_args(argv)


def _validate_output_path(path: Path, *, allow_external: bool) -> Path:
    resolved = path.resolve()
    repo_root = REPO_ROOT.resolve()
    if not allow_external and not resolved.is_relative_to(repo_root):
        raise SystemExit(
            f"Refusing output outside repo: {resolved}. "
            "Pass --allow-external-output to override."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    output_dir = _validate_output_path(Path(args.output), allow_external=args.allow_external_output)
    if not args.fixture:
        raise SystemExit("At least one --fixture path is required")
    fixtures = load_fixtures(args.fixture)
    policies = load_policies(args.policy)
    metrics = run_replays(fixtures, policies, output_dir=output_dir)
    write_metrics_jsonl(output_dir / "metrics.jsonl", metrics)
    summary = write_summary(output_dir / "summary.json", metrics)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
