#!/usr/bin/env python3
"""Run deterministic hermes-lcm stress release checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.stress import DEFAULT_SCENARIOS, TIERS, run_stress_check


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Fresh output directory for stress artifacts.")
    parser.add_argument("--tier", choices=sorted(TIERS), default="release", help="Stress tier to run.")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        choices=sorted(DEFAULT_SCENARIOS),
        help="Scenario to run. Repeatable. Defaults to all scenarios.",
    )
    parser.add_argument("--json", action="store_true", help="Print a compact JSON summary to stdout.")
    return parser.parse_args(argv)


def _validate_output_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and not resolved.is_dir():
        raise SystemExit(f"Output path exists and is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty output directory: {resolved}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    output_dir = _validate_output_path(Path(args.output))
    results = run_stress_check(
        output_dir=output_dir,
        tier=args.tier,
        scenarios=args.scenario or None,
    )
    if args.json:
        compact = {
            "failure_count": results.get("failure_count", 0),
            "failures": results.get("failures", []),
            "results_path": results.get("results_path"),
            "run_dir": results.get("run_dir"),
            "summary_path": results.get("summary_path"),
            "tier": results.get("tier"),
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
    return 0 if int(results.get("failure_count", 0) or 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
