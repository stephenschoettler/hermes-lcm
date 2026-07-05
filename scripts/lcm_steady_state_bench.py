#!/usr/bin/env python3
"""Measure the per-turn ingest/preflight hot path at a range of history sizes.

Unlike lcm_benchmark.py (which times compaction), this replays turns *without*
compacting and reports how per-turn ingest/preflight latency scales with
conversation length. Use it as a regression guard: the per-turn cost should stay
roughly flat as history grows, not scale with it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.steady_state import (
    DEFAULT_HISTORY_SIZES,
    DEFAULT_ITERATIONS,
    format_report,
    run_steady_state,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-size",
        type=int,
        action="append",
        default=[],
        help="History size (message count) to sample. Repeatable. "
        f"Default: {', '.join(str(s) for s in DEFAULT_HISTORY_SIZES)}.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Measured turns per history size.",
    )
    parser.add_argument("--output", help="Directory for benchmark DBs (default: a temp dir).")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    history_sizes = tuple(args.history_size) if args.history_size else DEFAULT_HISTORY_SIZES
    run_dir = Path(args.output) if args.output else Path(tempfile.mkdtemp(prefix="lcm-steady-"))

    report = run_steady_state(
        run_dir,
        history_sizes=history_sizes,
        iterations=args.iterations,
        progress=lambda msg: print(f"[steady-state] {msg}", file=sys.stderr),
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
