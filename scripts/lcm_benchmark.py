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
from benchmarking.report import write_community_export, write_metrics_jsonl, write_summary


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="append", default=[], help="Fixture JSON path. Repeatable.")
    parser.add_argument(
        "--synthetic-fixture",
        action="append",
        default=[],
        help="Deterministic synthetic fixture spec: name:pairs:canaries:filler_words. Repeatable.",
    )
    parser.add_argument("--policy", action="append", default=[], help="Policy JSON/YAML path. Repeatable.")
    parser.add_argument("--output", required=True, help="Benchmark output directory.")
    parser.add_argument("--json", action="store_true", help="Print summary JSON to stdout.")
    parser.add_argument(
        "--export",
        help="Write a scrubbed community benchmark export JSON file under --output.",
    )
    parser.add_argument("--provider", default="", help="Provider label for --export metadata.")
    parser.add_argument("--model", default="", help="Model label for --export metadata.")
    parser.add_argument(
        "--allow-external-output",
        action="store_true",
        help="Allow --output outside this repository. --export must still be under --output.",
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
    return resolved


def _validate_export_path(path: Path, *, output_dir: Path) -> Path:
    output_root = output_dir.resolve()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        cwd_relative = path.resolve()
        resolved = (
            cwd_relative
            if cwd_relative.is_relative_to(output_root)
            else (output_root / path).resolve()
        )

    if not resolved.is_relative_to(output_root):
        raise SystemExit(
            f"Refusing --export outside output directory: {resolved}. "
            "Place the export under --output."
        )

    repo_root = REPO_ROOT.resolve()
    if resolved.is_relative_to(repo_root):
        repo_relative_parts = resolved.relative_to(repo_root).parts
        if repo_relative_parts and repo_relative_parts[0] == ".git":
            raise SystemExit(f"Refusing --export inside .git: {resolved}")

    if resolved.exists():
        raise SystemExit(f"Refusing to overwrite existing export file: {resolved}")

    return resolved


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not args.fixture and not args.synthetic_fixture:
        raise SystemExit("At least one --fixture or --synthetic-fixture is required")
    output_dir = _validate_output_path(Path(args.output), allow_external=args.allow_external_output)
    export_path = None
    if args.export:
        export_path = _validate_export_path(Path(args.export), output_dir=output_dir)
    fixtures = load_fixtures(args.fixture, synthetic_specs=args.synthetic_fixture)
    policies = load_policies(args.policy)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_replays(fixtures, policies, output_dir=output_dir)
    write_metrics_jsonl(output_dir / "metrics.jsonl", metrics)
    summary = write_summary(output_dir / "summary.json", metrics)
    if export_path:
        write_community_export(
            export_path,
            summary,
            policies=policies,
            provider=args.provider,
            model=args.model,
        )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
