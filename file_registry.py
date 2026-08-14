"""Filesystem-backed large-file awareness for LCM.

The registry stores a stable opaque identifier, the canonical path, metadata,
and a compact structural exploration summary.  File bytes remain exclusively
on the filesystem; they are never copied into the LCM database.
"""

from __future__ import annotations

import ast
import csv
import json
import mimetypes
import re
import secrets
import sqlite3
from dataclasses import dataclass
from contextlib import closing
from pathlib import Path
from typing import Any

from .db_bootstrap import configure_connection


_CODE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cs", ".go", ".h", ".hpp", ".java", ".js",
    ".jsx", ".kt", ".php", ".rb", ".rs", ".swift", ".ts", ".tsx",
}
_SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
_MAX_SUMMARY_CHARS = 8_000
_MAX_TEXT_SCAN_BYTES = 4 * 1024 * 1024
_MAX_JSON_PARSE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class FileRecord:
    file_id: str
    path: str
    mime_type: str
    size_bytes: int
    token_count: int
    exploration_summary: str
    mtime_ns: int


class FileRegistry:
    """Persistent stable identifiers and structural summaries for large files."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        configure_connection(conn)
        return conn

    def _initialize(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lcm_files (
                    file_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                    token_count INTEGER NOT NULL CHECK (token_count >= 0),
                    exploration_summary TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    created_at REAL NOT NULL DEFAULT (unixepoch('subsec')),
                    updated_at REAL NOT NULL DEFAULT (unixepoch('subsec'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lcm_files_path ON lcm_files(path)"
            )

    def register(self, path: str | Path) -> FileRecord:
        canonical = Path(path).expanduser().resolve(strict=True)
        if not canonical.is_file():
            raise ValueError(f"not a regular file: {canonical}")
        stat = canonical.stat()
        mime_type = _detect_mime(canonical)
        summary = _explore(canonical, mime_type)
        token_count = max(1, (stat.st_size + 3) // 4) if stat.st_size else 0

        with closing(self._connect()) as conn, conn:
            existing = conn.execute(
                "SELECT file_id FROM lcm_files WHERE path = ?", (str(canonical),)
            ).fetchone()
            file_id = str(existing["file_id"]) if existing else _new_file_id()
            conn.execute(
                """
                INSERT INTO lcm_files (
                    file_id, path, mime_type, size_bytes, token_count,
                    exploration_summary, mtime_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    mime_type = excluded.mime_type,
                    size_bytes = excluded.size_bytes,
                    token_count = excluded.token_count,
                    exploration_summary = excluded.exploration_summary,
                    mtime_ns = excluded.mtime_ns,
                    updated_at = unixepoch('subsec')
                """,
                (
                    file_id,
                    str(canonical),
                    mime_type,
                    stat.st_size,
                    token_count,
                    summary[:_MAX_SUMMARY_CHARS],
                    stat.st_mtime_ns,
                ),
            )
        record = self.get(file_id)
        if record is None:  # pragma: no cover - defensive against disk failure
            raise RuntimeError(f"file registry write did not persist {file_id}")
        return record

    def get(self, file_id: str) -> FileRecord | None:
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                """
                SELECT file_id, path, mime_type, size_bytes, token_count,
                       exploration_summary, mtime_ns
                FROM lcm_files WHERE file_id = ?
                """,
                (file_id,),
            ).fetchone()
        return _row_to_record(row) if row else None

    def get_by_path(self, path: str | Path) -> FileRecord | None:
        canonical = str(Path(path).expanduser().resolve())
        with closing(self._connect()) as conn, conn:
            row = conn.execute(
                """
                SELECT file_id, path, mime_type, size_bytes, token_count,
                       exploration_summary, mtime_ns
                FROM lcm_files WHERE path = ?
                """,
                (canonical,),
            ).fetchone()
        return _row_to_record(row) if row else None


def union_file_ids(*lineage: Any) -> tuple[str, ...]:
    """Return a stable union of file IDs referenced anywhere in lineage data.

    Only explicit ``file_id`` and ``file_ids`` fields are considered; generic
    node/message ``id`` and ``source_ids`` fields are intentionally ignored.
    """

    seen: set[str] = set()
    ordered: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str):
            candidates = (
                [value]
                if value.startswith("file_") and not any(char.isspace() for char in value)
                else re.findall(r"\bfile_[A-Za-z0-9_-]{8,}\b", value)
            )
            for candidate in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    ordered.append(candidate)
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "file_id":
                    add(nested)
                elif key == "file_ids":
                    if isinstance(nested, (list, tuple, set, frozenset)):
                        for item in nested:
                            add(item)
                    else:
                        add(nested)
                elif isinstance(nested, (dict, list, tuple, str)):
                    add(nested)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                add(item)

    for value in lineage:
        add(value)
    return tuple(ordered)


def _new_file_id() -> str:
    return "file_" + secrets.token_urlsafe(18)


def _row_to_record(row: sqlite3.Row) -> FileRecord:
    return FileRecord(
        file_id=str(row["file_id"]),
        path=str(row["path"]),
        mime_type=str(row["mime_type"]),
        size_bytes=int(row["size_bytes"]),
        token_count=int(row["token_count"]),
        exploration_summary=str(row["exploration_summary"]),
        mtime_ns=int(row["mtime_ns"]),
    )


def _detect_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    overrides = {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".ndjson": "application/x-ndjson",
        ".csv": "text/csv",
        ".sql": "application/sql",
        ".sqlite": "application/vnd.sqlite3",
        ".sqlite3": "application/vnd.sqlite3",
        ".db": "application/vnd.sqlite3",
        ".py": "text/x-python",
    }
    return overrides.get(suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _explore(path: Path, mime_type: str) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            return _summarize_json(path)
        if suffix in {".jsonl", ".ndjson"}:
            return _summarize_jsonl(path)
        if suffix == ".csv":
            return _summarize_csv(path)
        if suffix in _SQLITE_SUFFIXES:
            return _summarize_sqlite(path)
        if suffix == ".sql":
            return _summarize_sql(path)
        if suffix == ".py":
            return _summarize_python(path)
        if suffix in _CODE_SUFFIXES:
            return _summarize_code(path)
        if mime_type.startswith("text/"):
            return _summarize_text(path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, csv.Error, sqlite3.Error, SyntaxError) as exc:
        return f"{path.name}: {path.stat().st_size} bytes; structural exploration failed ({type(exc).__name__})"
    return f"{path.name}: binary file, {path.stat().st_size} bytes"


def _shape(value: Any, *, depth: int = 0) -> str:
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, dict):
        entries = [f"{key}: {_shape(nested, depth=depth + 1)}" for key, nested in list(value.items())[:50]]
        suffix = ", ..." if len(value) > 50 else ""
        return "{" + ", ".join(entries) + suffix + "}"
    if isinstance(value, list):
        samples = {_shape(item, depth=depth + 1) for item in value[:50]}
        return f"array[{len(value)}]<{' | '.join(sorted(samples)) or 'empty'}>"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _summarize_json(path: Path) -> str:
    if path.stat().st_size > _MAX_JSON_PARSE_BYTES:
        sample = _read_bounded_text(path)
        stripped = sample.lstrip()
        if stripped.startswith("{"):
            keys = re.findall(r'"((?:[^"\\]|\\.)*)"\s*:', sample)
            unique_keys = list(dict.fromkeys(keys))[:100]
            return (
                f"JSON {path.name}: large object ({path.stat().st_size} bytes); "
                f"sampled keys [{', '.join(unique_keys)}]"
            )
        if stripped.startswith("["):
            return f"JSON {path.name}: large array ({path.stat().st_size} bytes); prefix sampled"
        return f"JSON {path.name}: large JSON value ({path.stat().st_size} bytes); prefix sampled"
    value = json.loads(path.read_text(encoding="utf-8"))
    return f"JSON {path.name}: {_shape(value)}"


def _summarize_jsonl(path: Path) -> str:
    shapes: set[str] = set()
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            count += 1
            if len(shapes) < 10:
                try:
                    shapes.add(_shape(json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL line {line_number}") from exc
    return f"JSONL {path.name}: {count} records; shapes: {' | '.join(sorted(shapes)) or 'empty'}"


def _summarize_csv(path: Path) -> str:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    columns = ", ".join(header[:100])
    if len(header) > 100:
        columns += ", ..."
    return f"CSV {path.name}: {rows} data rows, {len(header)} columns [{columns}]"


def _summarize_sqlite(path: Path) -> str:
    uri = path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        descriptions: list[str] = []
        for (table_name,) in tables[:100]:
            escaped = str(table_name).replace('"', '""')
            columns = conn.execute(f'PRAGMA table_info("{escaped}")').fetchall()
            column_text = ", ".join(
                f"{column[1]} {column[2] or 'ANY'}" + (" NOT NULL" if column[3] else "")
                for column in columns
            )
            descriptions.append(f"{table_name}({column_text})")
    return f"SQLite {path.name}: " + ("; ".join(descriptions) if descriptions else "no user tables")


def _summarize_sql(path: Path) -> str:
    text = _read_bounded_text(path)
    tables = re.findall(r"(?is)\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.\"`\[\]-]+)", text)
    views = re.findall(r"(?is)\bCREATE\s+VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.\"`\[\]-]+)", text)
    return f"SQL {path.name}: tables [{', '.join(tables[:100])}]; views [{', '.join(views[:100])}]"


def _format_python_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [argument.arg for argument in node.args.posonlyargs + node.args.args]
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    args.extend(argument.arg for argument in node.args.kwonlyargs)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    # ast.unparse preserves useful annotations while the fallback above remains
    # available on the oldest supported Python if an unusual node fails.
    rendered: list[str] = []
    all_nodes = node.args.posonlyargs + node.args.args
    annotation_by_name = {
        argument.arg: ast.unparse(argument.annotation)
        for argument in all_nodes + node.args.kwonlyargs
        if argument.annotation is not None
    }
    for arg in args:
        prefix = ""
        name = arg
        if arg.startswith("**"):
            prefix, name = "**", arg[2:]
        elif arg.startswith("*"):
            prefix, name = "*", arg[1:]
        annotation = annotation_by_name.get(name)
        rendered.append(prefix + name + (f": {annotation}" if annotation else ""))
    return ", ".join(rendered)


def _summarize_python(path: Path) -> str:
    tree = ast.parse(_read_bounded_text(path))
    structures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            structures.append(f"class {node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            structures.append(f"{prefix}{node.name}({_format_python_args(node)})")
    return f"Python {path.name}: " + ("; ".join(structures[:200]) if structures else "no classes or functions")


def _summarize_code(path: Path) -> str:
    text = _read_bounded_text(path)
    declarations = re.findall(
        r"(?m)^\s*(?:export\s+)?(?:public\s+|private\s+|protected\s+)?(?:async\s+)?"
        r"(?:class|interface|struct|enum|function|fn|func)\s+[A-Za-z_$][\w$]*(?:\s*\([^\n{;]*\))?",
        text,
    )
    return f"Code {path.name}: " + ("; ".join(item.strip() for item in declarations[:200]) if declarations else "no recognized declarations")


def _summarize_text(path: Path) -> str:
    text = _read_bounded_text(path)
    lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    words = len(re.findall(r"\S+", text))
    truncated = path.stat().st_size > _MAX_TEXT_SCAN_BYTES
    suffix = f" (first {_MAX_TEXT_SCAN_BYTES} bytes scanned)" if truncated else ""
    return f"Text {path.name}: {lines} lines, {words} words{suffix}"


def _read_bounded_text(path: Path) -> str:
    with path.open("rb") as handle:
        data = handle.read(_MAX_TEXT_SCAN_BYTES)
    return data.decode("utf-8")
