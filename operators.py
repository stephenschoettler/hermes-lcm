"""Persistent engine-managed JSONL map operators.

``LLMMap`` and ``AgenticMap`` move iteration, bounded concurrency, item claims,
schema validation, and retry state out of model-written control flow.  The
execution callable is injected by the host so this module remains provider and
network independent.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import secrets
import sqlite3
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .db_bootstrap import configure_connection
from .file_registry import FileRegistry
from .operator_schemas import (
    JSONSchemaError,
    JSONSchemaValidationError,
    compile_json_schema,
)


DEFAULT_CONCURRENCY = 16
DEFAULT_MAX_RETRIES = 2
MAX_CONCURRENCY = 256


@dataclass(frozen=True)
class ClaimedItem:
    batch_id: str
    item_index: int
    item: Any
    attempt: int
    validation_error: str | None


@dataclass(frozen=True)
class BatchResult:
    batch_id: str
    kind: str
    status: str
    total: int
    completed: int
    failed: int
    pending: int
    running: int
    concurrency: int
    output_path: str
    output_file_id: str | None


class OperatorStore:
    """Transactional persistent state for map batches and their items."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        configure_connection(conn)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS lcm_operator_batches (
                    batch_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL CHECK (kind IN ('llm_map', 'agentic_map')),
                    input_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    output_file_id TEXT,
                    prompt TEXT NOT NULL,
                    output_schema TEXT NOT NULL,
                    concurrency INTEGER NOT NULL CHECK (concurrency BETWEEN 1 AND 256),
                    max_retries INTEGER NOT NULL CHECK (max_retries >= 0),
                    read_only INTEGER,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'running', 'completed', 'completed_with_errors')
                    ),
                    total INTEGER NOT NULL CHECK (total >= 0),
                    created_at REAL NOT NULL DEFAULT (unixepoch('subsec')),
                    updated_at REAL NOT NULL DEFAULT (unixepoch('subsec')),
                    completed_at REAL
                );

                CREATE TABLE IF NOT EXISTS lcm_operator_items (
                    batch_id TEXT NOT NULL,
                    item_index INTEGER NOT NULL,
                    input_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'running', 'completed', 'failed')
                    ),
                    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                    output_json TEXT,
                    error TEXT,
                    claimed_by TEXT,
                    claim_started_at REAL,
                    completed_at REAL,
                    PRIMARY KEY (batch_id, item_index),
                    FOREIGN KEY (batch_id) REFERENCES lcm_operator_batches(batch_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_lcm_operator_claims
                ON lcm_operator_items(batch_id, status, item_index);
                """
            )

    def create_batch(
        self,
        *,
        kind: str,
        input_path: str | Path,
        output_path: str | Path,
        prompt: str,
        output_schema: dict[str, Any],
        concurrency: int = DEFAULT_CONCURRENCY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        read_only: bool | None = None,
    ) -> str:
        if kind not in {"llm_map", "agentic_map"}:
            raise ValueError(f"unsupported operator kind: {kind}")
        concurrency = _validate_concurrency(concurrency)
        max_retries = _validate_max_retries(max_retries)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if kind == "agentic_map" and not isinstance(read_only, bool):
            raise ValueError("agentic_map requires an explicit read_only boolean")
        _compile_validator(output_schema)

        source = Path(input_path).expanduser().resolve(strict=True)
        destination = Path(output_path).expanduser().resolve()
        if source == destination:
            raise ValueError("output_path must differ from input_path")
        items = _read_jsonl(source)
        batch_id = "map_" + secrets.token_urlsafe(18)
        schema_json = json.dumps(output_schema, ensure_ascii=False, sort_keys=True)

        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO lcm_operator_batches (
                    batch_id, kind, input_path, output_path, prompt,
                    output_schema, concurrency, max_retries, read_only,
                    status, total
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    batch_id,
                    kind,
                    str(source),
                    str(destination),
                    prompt,
                    schema_json,
                    concurrency,
                    max_retries,
                    int(read_only) if read_only is not None else None,
                    len(items),
                ),
            )
            conn.executemany(
                """
                INSERT INTO lcm_operator_items (
                    batch_id, item_index, input_json, status
                ) VALUES (?, ?, ?, 'pending')
                """,
                [
                    (
                        batch_id,
                        index,
                        json.dumps(item, ensure_ascii=False, sort_keys=True),
                    )
                    for index, item in enumerate(items)
                ],
            )
        return batch_id

    def batch_config(self, batch_id: str) -> dict[str, Any]:
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                "SELECT * FROM lcm_operator_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown operator batch: {batch_id}")
        result = dict(row)
        result["output_schema"] = json.loads(result["output_schema"])
        if result["read_only"] is not None:
            result["read_only"] = bool(result["read_only"])
        return result

    def mark_running(self, batch_id: str) -> None:
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                UPDATE lcm_operator_batches
                SET status = 'running', updated_at = unixepoch('subsec'), completed_at = NULL
                WHERE batch_id = ?
                  AND status IN ('pending', 'running', 'completed_with_errors')
                """,
                (batch_id,),
            )
            if cursor.rowcount != 1:
                row = conn.execute(
                    "SELECT status FROM lcm_operator_batches WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown operator batch: {batch_id}")
                if row["status"] == "completed":
                    return
                raise RuntimeError(f"batch {batch_id} cannot run from {row['status']}")

    def claim_next(self, batch_id: str, worker_id: str) -> ClaimedItem | None:
        """Atomically claim one pending item and increment its attempt count."""

        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT item_index, input_json, attempts, error
                FROM lcm_operator_items
                WHERE batch_id = ? AND status = 'pending'
                ORDER BY item_index
                LIMIT 1
                """,
                (batch_id,),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            attempt = int(row["attempts"]) + 1
            updated = conn.execute(
                """
                UPDATE lcm_operator_items
                SET status = 'running', attempts = ?, claimed_by = ?,
                    claim_started_at = unixepoch('subsec')
                WHERE batch_id = ? AND item_index = ? AND status = 'pending'
                """,
                (attempt, worker_id, batch_id, int(row["item_index"])),
            )
            if updated.rowcount != 1:  # pragma: no cover - BEGIN IMMEDIATE serializes this
                conn.rollback()
                return None
            conn.commit()
        return ClaimedItem(
            batch_id=batch_id,
            item_index=int(row["item_index"]),
            item=json.loads(row["input_json"]),
            attempt=attempt,
            validation_error=str(row["error"]) if row["error"] else None,
        )

    def complete_item(self, item: ClaimedItem, output: Any) -> None:
        output_json = json.dumps(output, ensure_ascii=False, sort_keys=True)
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                UPDATE lcm_operator_items
                SET status = 'completed', output_json = ?, error = NULL,
                    claimed_by = NULL, claim_started_at = NULL,
                    completed_at = unixepoch('subsec')
                WHERE batch_id = ? AND item_index = ? AND status = 'running'
                """,
                (output_json, item.batch_id, item.item_index),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"item {item.batch_id}/{item.item_index} lost its running claim"
                )

    def fail_item(self, item: ClaimedItem, error: str, max_retries: int) -> None:
        terminal = item.attempt > max_retries
        status = "failed" if terminal else "pending"
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                UPDATE lcm_operator_items
                SET status = ?, error = ?, claimed_by = NULL,
                    claim_started_at = NULL,
                    completed_at = CASE WHEN ? THEN unixepoch('subsec') ELSE NULL END
                WHERE batch_id = ? AND item_index = ? AND status = 'running'
                """,
                (
                    status,
                    error[:8_000],
                    int(terminal),
                    item.batch_id,
                    item.item_index,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"item {item.batch_id}/{item.item_index} lost its running claim"
                )

    def requeue_running(self, batch_id: str) -> int:
        """Recover items whose process exited after claiming them."""

        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                UPDATE lcm_operator_items
                SET status = 'pending', claimed_by = NULL, claim_started_at = NULL,
                    error = COALESCE(error, 'recovered abandoned running claim')
                WHERE batch_id = ? AND status = 'running'
                """,
                (batch_id,),
            )
            return int(cursor.rowcount)

    def finish_batch(self, batch_id: str) -> BatchResult:
        with closing(self._connect()) as conn, conn:
            counts = _item_counts(conn, batch_id)
            if counts["pending"] or counts["running"]:
                raise RuntimeError(f"batch {batch_id} still has unfinished items")
            status = "completed_with_errors" if counts["failed"] else "completed"
            conn.execute(
                """
                UPDATE lcm_operator_batches
                SET status = ?, updated_at = unixepoch('subsec'),
                    completed_at = unixepoch('subsec')
                WHERE batch_id = ?
                """,
                (status, batch_id),
            )
        return self.status(batch_id)

    def set_output_file_id(self, batch_id: str, file_id: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                UPDATE lcm_operator_batches
                SET output_file_id = ?, updated_at = unixepoch('subsec')
                WHERE batch_id = ?
                """,
                (file_id, batch_id),
            )

    def write_output(self, batch_id: str) -> Path:
        config = self.batch_config(batch_id)
        destination = Path(config["output_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".{batch_id}.tmp")
        with closing(self._connect()) as conn, conn, temporary.open("w", encoding="utf-8") as handle:
            rows = conn.execute(
                """
                SELECT item_index, status, attempts, output_json, error
                FROM lcm_operator_items
                WHERE batch_id = ?
                ORDER BY item_index
                """,
                (batch_id,),
            )
            for row in rows:
                output_row = {
                    "index": int(row["item_index"]),
                    "status": str(row["status"]),
                    "attempts": int(row["attempts"]),
                }
                if row["output_json"] is not None:
                    output_row["output"] = json.loads(row["output_json"])
                if row["error"] is not None:
                    output_row["error"] = str(row["error"])
                handle.write(json.dumps(output_row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(destination)
        return destination

    def status(self, batch_id: str) -> BatchResult:
        with closing(self._connect()) as conn, conn:
            batch = conn.execute(
                "SELECT * FROM lcm_operator_batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if batch is None:
                raise KeyError(f"unknown operator batch: {batch_id}")
            counts = _item_counts(conn, batch_id)
        return BatchResult(
            batch_id=batch_id,
            kind=str(batch["kind"]),
            status=str(batch["status"]),
            total=int(batch["total"]),
            completed=counts["completed"],
            failed=counts["failed"],
            pending=counts["pending"],
            running=counts["running"],
            concurrency=int(batch["concurrency"]),
            output_path=str(batch["output_path"]),
            output_file_id=str(batch["output_file_id"]) if batch["output_file_id"] else None,
        )


class _MapOperator:
    kind: str

    def __init__(
        self,
        db_path: str | Path,
        *,
        executor: Callable[..., Any],
        file_registry: FileRegistry | None = None,
    ):
        if not callable(executor):
            raise TypeError("executor must be callable")
        # A bound engine method otherwise creates
        # engine -> operator -> bound method -> engine.  That cycle delays the
        # engine's SQLite-backed children until cyclic GC, violating the host's
        # refcount-time deallocation contract.  WeakMethod keeps normal call
        # semantics without retaining the owner.
        self._executor_ref: weakref.WeakMethod | Callable[..., Any]
        self.executor = executor
        self.store = OperatorStore(db_path)
        self.file_registry = file_registry or FileRegistry(db_path)

    @property
    def executor(self) -> Callable[..., Any]:
        """Current injected execution callable (replaceable by hosts/tests)."""

        return self._executor()

    @executor.setter
    def executor(self, executor: Callable[..., Any]) -> None:
        if not callable(executor):
            raise TypeError("executor must be callable")
        if inspect.ismethod(executor) and executor.__self__ is not None:
            self._executor_ref = weakref.WeakMethod(executor)
        else:
            self._executor_ref = executor

    def _executor(self) -> Callable[..., Any]:
        if isinstance(self._executor_ref, weakref.WeakMethod):
            executor = self._executor_ref()
            if executor is None:
                raise RuntimeError("map operator owner has been released")
            return executor
        return self._executor_ref

    def _create_and_run(
        self,
        *,
        input_path: str | Path,
        output_path: str | Path | None,
        prompt: str,
        output_schema: dict[str, Any],
        concurrency: int,
        max_retries: int,
        read_only: bool | None,
    ) -> BatchResult:
        source = Path(input_path).expanduser().resolve(strict=True)
        if output_path is None:
            output_path = source.with_name(source.name + ".lcm-output.jsonl")
        batch_id = self.store.create_batch(
            kind=self.kind,
            input_path=source,
            output_path=output_path,
            prompt=prompt,
            output_schema=output_schema,
            concurrency=concurrency,
            max_retries=max_retries,
            read_only=read_only,
        )
        return self._execute(batch_id, recover=False)

    def resume(self, batch_id: str) -> BatchResult:
        config = self.store.batch_config(batch_id)
        if config["kind"] != self.kind:
            raise ValueError(
                f"batch {batch_id} belongs to {config['kind']}, not {self.kind}"
            )
        if config["status"] == "completed":
            return self._materialize_output(batch_id)
        return self._execute(batch_id, recover=True)

    def status(self, batch_id: str) -> BatchResult:
        return self.store.status(batch_id)

    def _execute(self, batch_id: str, *, recover: bool) -> BatchResult:
        config = self.store.batch_config(batch_id)
        validator = _compile_validator(config["output_schema"])
        if recover:
            self.store.requeue_running(batch_id)
        self.store.mark_running(batch_id)

        def worker(worker_number: int) -> None:
            worker_id = f"{batch_id}:{worker_number}:{secrets.token_hex(4)}"
            while True:
                item = self.store.claim_next(batch_id, worker_id)
                if item is None:
                    return
                try:
                    output = self._invoke(
                        item=item.item,
                        prompt=config["prompt"],
                        attempt=item.attempt,
                        validation_error=item.validation_error,
                        read_only=config["read_only"],
                    )
                    validator.validate(output)
                    # JSON serialization is part of the output contract, not a
                    # best-effort logging concern.
                    json.dumps(output, ensure_ascii=False, sort_keys=True)
                except JSONSchemaValidationError as exc:
                    self.store.fail_item(
                        item,
                        _format_validation_error(exc),
                        int(config["max_retries"]),
                    )
                except Exception as exc:
                    self.store.fail_item(
                        item,
                        f"{type(exc).__name__}: {exc}",
                        int(config["max_retries"]),
                    )
                else:
                    self.store.complete_item(item, output)

        worker_count = int(config["concurrency"])
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=f"lcm-{self.kind}",
        ) as pool:
            futures = [pool.submit(worker, index) for index in range(worker_count)]
            for future in futures:
                future.result()

        self.store.finish_batch(batch_id)
        return self._materialize_output(batch_id)

    def _materialize_output(self, batch_id: str) -> BatchResult:
        output_path = self.store.write_output(batch_id)
        output_record = self.file_registry.register(output_path)
        self.store.set_output_file_id(batch_id, output_record.file_id)
        return self.store.status(batch_id)

    def _invoke(
        self,
        *,
        item: Any,
        prompt: str,
        attempt: int,
        validation_error: str | None,
        read_only: bool | None,
    ) -> Any:
        kwargs = {
            "item": item,
            "prompt": prompt,
            "attempt": attempt,
            "validation_error": validation_error,
        }
        if self.kind == "agentic_map":
            kwargs["read_only"] = read_only
        result = self._executor()(**kwargs)
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result


class LLMMap(_MapOperator):
    """Stateless, side-effect-free per-item map execution."""

    kind = "llm_map"

    def run(
        self,
        *,
        input_path: str | Path,
        prompt: str,
        output_schema: dict[str, Any],
        output_path: str | Path | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> BatchResult:
        return self._create_and_run(
            input_path=input_path,
            output_path=output_path,
            prompt=prompt,
            output_schema=output_schema,
            concurrency=concurrency,
            max_retries=max_retries,
            read_only=None,
        )


class AgenticMap(_MapOperator):
    """Tool-capable per-item sub-agent map with an explicit capability mode."""

    kind = "agentic_map"

    def run(
        self,
        *,
        input_path: str | Path,
        prompt: str,
        output_schema: dict[str, Any],
        read_only: bool,
        output_path: str | Path | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> BatchResult:
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be an explicit boolean")
        return self._create_and_run(
            input_path=input_path,
            output_path=output_path,
            prompt=prompt,
            output_schema=output_schema,
            concurrency=concurrency,
            max_retries=max_retries,
            read_only=read_only,
        )


def _validate_concurrency(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("concurrency must be an integer")
    if not 1 <= value <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be between 1 and {MAX_CONCURRENCY}")
    return value


def _validate_max_retries(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_retries must be an integer")
    if value < 0:
        raise ValueError("max_retries cannot be negative")
    return value


def _read_jsonl(path: Path) -> list[Any]:
    items: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc.msg}") from exc
    return items


def _compile_validator(schema: dict[str, Any]):
    try:
        return compile_json_schema(schema)
    except JSONSchemaError as exc:
        raise ValueError(f"invalid output_schema: {exc}") from exc


def _format_validation_error(error: JSONSchemaValidationError) -> str:
    location = ".".join(str(part) for part in error.path)
    prefix = f"output.{location}: " if location else "output: "
    return prefix + error.message


def _item_counts(conn: sqlite3.Connection, batch_id: str) -> dict[str, int]:
    counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS count
        FROM lcm_operator_items
        WHERE batch_id = ?
        GROUP BY status
        """,
        (batch_id,),
    ).fetchall()
    for row in rows:
        counts[str(row["status"])] = int(row["count"])
    return counts
