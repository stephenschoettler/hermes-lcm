#!/usr/bin/env python3
"""Time first-query/warm #171 KNN and compare simple vs streaming paths.

The first query builds process-local residency but runs against page-cache-hot
data seeded by this process; it is not an I/O-cold measurement.
"""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.replay import _ensure_hermes_lcm_package


SIZES = (10_000, 50_000, 185_000)
CROSSOVER_SIZES = (
    500,
    1_000,
    2_000,
    2_499,
    2_500,
    5_000,
    10_000,
    20_000,
)
MODEL = "fast-scan-synthetic"
PROVIDER = "bench"
DIM = 384


def _seed(
    db_path: Path,
    count: int,
    rng: np.random.Generator,
    *,
    dtype: str,
) -> None:
    for suffix in ("", "-wal", "-shm"):
        db_path.with_name(f"{db_path.name}{suffix}").unlink(missing_ok=True)
    _ensure_hermes_lcm_package()
    from hermes_lcm.vector_store import VectorStore

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            source TEXT DEFAULT '',
            role TEXT NOT NULL,
            content TEXT,
            timestamp REAL NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO messages VALUES (?, 'synthetic', 'bench', 'user', '', ?)",
        ((index, float(index)) for index in range(count)),
    )
    conn.commit()
    conn.close()

    store = VectorStore(db_path)
    try:
        store.ensure_chunk_schema()
        identity_hash = store.register_profile(
            MODEL, PROVIDER, DIM, dtype=dtype, task="chunk"
        )
        conn = store.connection
        conn.execute("BEGIN IMMEDIATE")
        try:
            for start in range(0, count, 2_000):
                end = min(count, start + 2_000)
                vectors = rng.standard_normal((end - start, DIM)).astype(
                    np.float32
                )
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                vector_rows = []
                meta_rows = []
                for offset, index in enumerate(range(start, end)):
                    chunk_id = f"{index}:0"
                    if dtype == "int8":
                        vector = vectors[offset]
                        max_abs = float(np.max(np.abs(vector)))
                        scale = max_abs / 127.0 if max_abs > 0.0 else 0.0
                        quantized = np.rint(vector / scale).clip(
                            -127, 127
                        ).astype(np.int8) if scale > 0.0 else np.zeros(
                            DIM, dtype=np.int8
                        )
                        blob = quantized.tobytes() + struct.pack("<f", scale)
                    else:
                        blob = vectors[offset].astype("<f4", copy=False).tobytes()
                    vector_rows.append((chunk_id, identity_hash, blob))
                    meta_rows.append(
                        (
                            chunk_id,
                            identity_hash,
                            index,
                            0,
                            0,
                            1,
                            1,
                            "synthetic",
                        )
                    )
                conn.executemany(
                    "INSERT INTO lcm_chunk_vectors VALUES (?, ?, ?)",
                    vector_rows,
                )
                conn.executemany(
                    """
                    INSERT INTO lcm_chunk_meta(
                        chunk_id, identity_hash, store_id, chunk_index,
                        char_start, char_end, token_estimate, embedded_at, archived
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    meta_rows,
                )
            conn.execute(
                "UPDATE lcm_embedding_profile SET data_version = data_version + 1 "
                "WHERE identity_hash = ?",
                (identity_hash,),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    finally:
        store.close()


def _time_one(
    db_path: Path,
    count: int,
    *,
    resident_max_mb: int,
    warm_runs: int,
    batch_rows: int,
    path: str,
) -> dict[str, object]:
    if warm_runs < 1:
        raise ValueError("warm_runs must be >= 1")
    _ensure_hermes_lcm_package()
    import hermes_lcm.vector_store as vector_store_module
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.vector_store import VectorStore

    original_crossover = vector_store_module._FAST_SCAN_STREAMING_MIN_ROWS
    if path == "simple":
        vector_store_module._FAST_SCAN_STREAMING_MIN_ROWS = sys.maxsize
    elif path == "streaming":
        vector_store_module._FAST_SCAN_STREAMING_MIN_ROWS = 0
    store = VectorStore(
        db_path,
        config=LCMConfig(
            embedding_bounded_scan_rows=batch_rows,
            knn_resident_max_mb=resident_max_mb,
        ),
    )
    query = np.zeros(DIM, dtype=np.float32)
    query[0] = 1.0
    try:
        started = time.perf_counter()
        cold = store.knn_chunks(
            query.tolist(),
            k=50,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )
        first_query_ms = (time.perf_counter() - started) * 1_000.0
        if cold.coverage != "full":
            raise RuntimeError(f"first-query coverage mismatch: {cold.coverage}")
        expected_k = min(50, count)
        if len(cold) != expected_k:
            raise RuntimeError(
                f"first-query result count mismatch: {len(cold)} != {expected_k}"
            )
        warm_ms = []
        for _ in range(warm_runs):
            started = time.perf_counter()
            warm = store.knn_chunks(
                query.tolist(),
                k=50,
                model=MODEL,
                provider=PROVIDER,
                full_scan=True,
            )
            warm_ms.append((time.perf_counter() - started) * 1_000.0)
        if warm.coverage != "full":
            raise RuntimeError(f"warm coverage mismatch: {warm.coverage}")
        if [row[0] for row in cold] != [row[0] for row in warm]:
            raise RuntimeError("first-query and warm top-k ids differ")
        git_sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "n": count,
            "dim": DIM,
            "path": path,
            "batch_rows": batch_rows,
            "embedding_bounded_scan_rows": batch_rows,
            "resident_max_mb": resident_max_mb,
            "resident_used": bool(store._resident_matrix_cache),
            "first_query_ms": round(first_query_ms, 3),
            "warm_ms": [round(value, 3) for value in warm_ms],
            "warm_p50_ms": round(float(np.median(warm_ms)), 3),
            "coverage": warm.coverage,
            "scoring": warm.scoring,
            "git_sha": git_sha,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "warm_runs": warm_runs,
            "_ranked": list(warm),
        }
    finally:
        store.close()
        vector_store_module._FAST_SCAN_STREAMING_MIN_ROWS = original_crossover


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--resident-max-mb", type=int, default=128)
    parser.add_argument("--batch-rows", type=int, default=2_000)
    parser.add_argument("--dtype", choices=("float32", "int8"), default="float32")
    parser.add_argument("--compare-loaders", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    sizes = args.sizes or list(CROSSOVER_SIZES if args.compare_loaders else SIZES)
    invalid = [size for size in sizes if size <= 0]
    if invalid:
        parser.error(f"--sizes must be positive: {invalid}")
    if args.warm_runs < 1:
        parser.error("--warm-runs must be >= 1")

    def run(root: Path) -> None:
        for count in sizes:
            db_path = root / f"fast-scan-{count}.db"
            _seed(
                db_path,
                count,
                np.random.default_rng(171 + count),
                dtype=args.dtype,
            )
            paths = ("simple", "streaming") if args.compare_loaders else ("selected",)
            ranked = None
            for path in paths:
                result = _time_one(
                    db_path,
                    count,
                    resident_max_mb=(
                        0 if args.compare_loaders else max(0, args.resident_max_mb)
                    ),
                    warm_runs=args.warm_runs,
                    batch_rows=max(1, args.batch_rows),
                    path=path,
                )
                current_ranked = result.pop("_ranked")
                if ranked is not None and current_ranked != ranked:
                    raise RuntimeError(
                        f"ranking mismatch at n={count}: simple != streaming"
                    )
                ranked = current_ranked
                print(json.dumps(result, sort_keys=True))

    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        run(args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="lcm-fast-scan-") as temp_dir:
            run(Path(temp_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
