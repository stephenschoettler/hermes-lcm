#!/usr/bin/env python3
"""Measure LCM SQLite storage sharing across retained engine clones.

Example:
    python scripts/measure_clone_storage.py --samples 25 --clones 10 --json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(os.environ.get("LCM_BENCH_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
PACKAGE_NAME = "hermes_lcm"
if PACKAGE_NAME not in sys.modules:
    package = ModuleType(PACKAGE_NAME)
    package.__path__ = [str(REPO_ROOT)]
    package.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = package

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _open_fd_count() -> int:
    """Return currently open descriptors on platforms exposing /dev/fd."""
    try:
        return len(os.listdir("/dev/fd"))
    except OSError:
        return -1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=25, help="Timed samples per measurement.")
    parser.add_argument("--clones", type=int, default=10, help="Retained clones for FD and batch measurements.")
    parser.add_argument("--database", help="SQLite database path (default: temporary database).")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a readable report.")
    return parser.parse_args(argv)


def _shutdown_all(engines: list[LCMEngine]) -> None:
    for engine in reversed(engines):
        engine.shutdown()


def run(samples: int, clones: int, database: Path) -> dict[str, float | int]:
    if samples < 1 or clones < 1:
        raise ValueError("--samples and --clones must both be positive")

    startup_ms: list[float] = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        engine = LCMEngine(config=LCMConfig(database_path=str(database)))
        startup_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        engine.shutdown()

    prototype = LCMEngine(config=LCMConfig(database_path=str(database)))
    retained: list[LCMEngine] = []
    try:
        fd_before = _open_fd_count()
        clone_ms: list[float] = []
        for _ in range(samples):
            started = time.perf_counter_ns()
            clone = prototype.clone_for_agent()
            clone_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            clone.shutdown()

        started = time.perf_counter_ns()
        retained = [prototype.clone_for_agent() for _ in range(clones)]
        ten_clone_ms = (time.perf_counter_ns() - started) / 1_000_000
        fd_after = _open_fd_count()
    finally:
        _shutdown_all(retained)
        prototype.shutdown()

    return {
        "fd_before_retained_clones": fd_before,
        "fd_after_retained_clones": fd_after,
        "retained_clone_fd_delta": fd_after - fd_before if fd_before >= 0 and fd_after >= 0 else -1,
        "median_clone_setup_ms": statistics.median(clone_ms),
        "ten_clone_setup_ms": ten_clone_ms,
        "median_initial_startup_ms": statistics.median(startup_ms),
        "samples": samples,
        "retained_clones": clones,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.database:
        report = run(args.samples, args.clones, Path(args.database))
    else:
        with tempfile.TemporaryDirectory(prefix="lcm-clone-storage-") as directory:
            report = run(args.samples, args.clones, Path(directory) / "lcm.db")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for name, value in report.items():
            print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
