#!/usr/bin/env python3
"""Import raw messages from a lossless-claw/OpenClaw LCM SQLite DB.

This is an operator script, not an agent tool. It only writes when --apply is
passed; dry-run is the default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PLUGIN_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "hermes_lcm"


def _ensure_local_package_importable() -> None:
    """Make local plugin modules importable when this file is run directly."""
    if PACKAGE_NAME in sys.modules:
        return
    pkg = types.ModuleType(PACKAGE_NAME)
    pkg.__path__ = [str(PLUGIN_DIR)]
    pkg.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = pkg


_ensure_local_package_importable()

from hermes_lcm.config import LCMConfig  # noqa: E402
from hermes_lcm.ingest_protection import protect_message_for_ingest  # noqa: E402
from hermes_lcm.message_content import normalize_content_value  # noqa: E402
from hermes_lcm.store import MessageStore, _normalize_source_value  # noqa: E402
from hermes_lcm.tokens import count_message_tokens  # noqa: E402


VALID_SESSION_IDENTITIES = frozenset({"session_id", "session_key"})


@dataclass(frozen=True)
class ImportCandidate:
    source_message_id: int
    source_conversation_id: int
    source_session: str
    target_session_id: str
    source: str
    role: str
    content: str
    tool_call_id: str | None
    tool_calls: list[dict[str, Any]] | None
    tool_name: str | None
    timestamp: float
    token_estimate: int


@dataclass(frozen=True)
class ImportResult:
    source_db: str
    target_db: str
    import_id: str
    scanned: int = 0
    eligible: int = 0
    would_import: int = 0
    imported: int = 0
    skipped_existing: int = 0
    skipped_empty: int = 0
    conversations: int = 0
    backup_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_db": self.source_db,
            "target_db": self.target_db,
            "import_id": self.import_id,
            "scanned": self.scanned,
            "eligible": self.eligible,
            "would_import": self.would_import,
            "imported": self.imported,
            "skipped_existing": self.skipped_existing,
            "skipped_empty": self.skipped_empty,
            "conversations": self.conversations,
            "backup_path": self.backup_path,
        }


def _readonly_sqlite_uri(db_path: Path) -> str:
    return db_path.resolve().as_uri() + "?mode=ro"


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise FileNotFoundError(f"source DB not found: {db_path}")
    conn = sqlite3.connect(_readonly_sqlite_uri(db_path), uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _quote_identifier(identifier: str) -> str:
    if not identifier.replace("_", "").isalnum():
        raise ValueError(f"unsafe SQLite identifier: {identifier!r}")
    return '"' + identifier.replace('"', '""') + '"'


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    sql = "PRAGMA table_info(" + _quote_identifier(table) + ")"
    return {row[1] for row in conn.execute(sql)}


def _require_columns(conn: sqlite3.Connection, table: str, columns: Iterable[str]) -> None:
    actual = _table_columns(conn, table)
    missing = [column for column in columns if column not in actual]
    if missing:
        raise ValueError(f"source DB table {table!r} missing required columns: {', '.join(missing)}")


def _default_import_id(source_db: Path) -> str:
    return hashlib.sha256(str(source_db.resolve()).encode("utf-8")).hexdigest()[:16]


def _parse_timestamp(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return fallback
    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                dt = None
        if dt is None:
            return fallback
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).timestamp()


def _safe_segment(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _target_source(namespace: str, agent: str, source_session: str) -> str:
    return f"{_safe_segment(namespace, 'openclaw-lcm')}:agent:{_safe_segment(agent, 'unknown')}:{source_session}"


def _resolve_source_session(
    row: sqlite3.Row,
    *,
    conversation_id: int,
    session_identity: str,
) -> str:
    if session_identity not in VALID_SESSION_IDENTITIES:
        raise ValueError(
            "session_identity must be one of "
            + ", ".join(sorted(VALID_SESSION_IDENTITIES))
        )
    fallback = f"conversation:{conversation_id}"
    if session_identity == "session_key":
        return _safe_segment(
            row["conversation_session_key"] or row["conversation_session_id"],
            fallback,
        )
    return _safe_segment(
        row["conversation_session_id"] or row["conversation_session_key"],
        fallback,
    )


def _load_parts(conn: sqlite3.Connection) -> dict[int, list[sqlite3.Row]]:
    if not _table_exists(conn, "message_parts"):
        return {}
    columns = _table_columns(conn, "message_parts")
    if "message_id" not in columns or "ordinal" not in columns:
        return {}

    wanted = [
        "message_id",
        "part_type",
        "ordinal",
        "text_content",
        "is_ignored",
        "is_synthetic",
        "tool_call_id",
        "tool_name",
        "tool_input",
        "tool_output",
        "tool_error",
        "metadata",
    ]
    select_cols = [column if column in columns else f"NULL AS {column}" for column in wanted]
    rows = conn.execute(
        f"SELECT {', '.join(select_cols)} FROM message_parts ORDER BY message_id, ordinal"
    ).fetchall()
    by_message: dict[int, list[sqlite3.Row]] = {}
    for row in rows:
        by_message.setdefault(int(row["message_id"]), []).append(row)
    return by_message


def _metadata_value(part: sqlite3.Row, *keys: str) -> Any:
    raw = part["metadata"]
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    raw_obj = data.get("raw")
    if isinstance(raw_obj, dict):
        for key in keys:
            if key in raw_obj and raw_obj[key] is not None:
                return raw_obj[key]
    return None


def _part_value(part: sqlite3.Row, column: str, *metadata_keys: str) -> Any:
    value = part[column]
    if value not in (None, ""):
        return value
    return _metadata_value(part, *metadata_keys)


def _stringify_tool_payload(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _message_from_parts(role: str, content: str, parts: list[sqlite3.Row]) -> tuple[str, str | None, list[dict[str, Any]] | None, str | None]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_result_parts: list[str] = []

    for part in parts:
        if part["is_ignored"] or part["is_synthetic"]:
            continue
        part_type = str(part["part_type"] or "")
        text_content = part["text_content"]
        if part_type == "text" and text_content:
            text_parts.append(str(text_content))
            continue
        if part_type != "tool":
            continue

        candidate_tool_call_id = _part_value(
            part,
            "tool_call_id",
            "toolCallId",
            "tool_call_id",
            "toolUseId",
            "tool_use_id",
            "call_id",
            "id",
        )
        candidate_tool_name = _part_value(part, "tool_name", "name", "toolName", "tool_name")

        if role == "assistant":
            if candidate_tool_call_id or candidate_tool_name:
                tool_calls.append(
                    {
                        "id": str(candidate_tool_call_id or f"lossless_tool_{len(tool_calls)}"),
                        "type": "function",
                        "function": {
                            "name": str(candidate_tool_name or "unknown"),
                            "arguments": _stringify_tool_payload(
                                _part_value(part, "tool_input", "input", "arguments", "toolInput", "tool_input")
                            ),
                        },
                    }
                )
        elif role == "tool":
            tool_call_id = str(candidate_tool_call_id) if candidate_tool_call_id else tool_call_id
            tool_name = str(candidate_tool_name) if candidate_tool_name else tool_name
            output = _part_value(part, "tool_output", "output", "toolOutput", "tool_output")
            error = _part_value(part, "tool_error", "error", "toolError", "tool_error")
            if output not in (None, ""):
                tool_result_parts.append(str(output))
            elif error not in (None, ""):
                tool_result_parts.append(str(error))
            elif text_content:
                tool_result_parts.append(str(text_content))

    if not content and text_parts:
        content = "\n".join(text_parts)
    if role == "tool" and not content and tool_result_parts:
        content = "\n".join(tool_result_parts)
    return content, tool_call_id, tool_calls or None, tool_name


def _collect_candidates(
    conn: sqlite3.Connection,
    *,
    namespace: str,
    agent: str,
    session_identity: str = "session_id",
) -> tuple[list[ImportCandidate], int, int, int]:
    _require_columns(conn, "conversations", ["conversation_id", "session_id"])
    _require_columns(conn, "messages", ["message_id", "conversation_id", "seq", "role", "content"])

    conversation_cols = _table_columns(conn, "conversations")
    message_cols = _table_columns(conn, "messages")
    session_key_expr = "c.session_key" if "session_key" in conversation_cols else "NULL"
    conversation_created_expr = "c.created_at" if "created_at" in conversation_cols else "NULL"
    message_created_expr = "m.created_at" if "created_at" in message_cols else "NULL"
    token_count_expr = "m.token_count" if "token_count" in message_cols else "0"

    parts_by_message = _load_parts(conn)
    rows = conn.execute(
        f"""
        SELECT
            m.message_id,
            m.conversation_id,
            m.seq,
            m.role,
            m.content,
            {token_count_expr} AS token_count,
            {message_created_expr} AS message_created_at,
            c.session_id AS conversation_session_id,
            {session_key_expr} AS conversation_session_key,
            {conversation_created_expr} AS conversation_created_at
        FROM messages m
        JOIN conversations c ON c.conversation_id = m.conversation_id
        ORDER BY m.conversation_id, m.seq
        """
    ).fetchall()

    now = time.time()
    candidates: list[ImportCandidate] = []
    skipped_empty = 0
    conversation_ids: set[int] = set()
    for row in rows:
        role = str(row["role"] or "unknown")
        content = str(row["content"] or "")
        parts = parts_by_message.get(int(row["message_id"]), [])
        content, tool_call_id, tool_calls, tool_name = _message_from_parts(role, content, parts)
        if not content and not tool_calls:
            skipped_empty += 1
            continue

        conversation_id = int(row["conversation_id"])
        conversation_ids.add(conversation_id)
        source_session = _resolve_source_session(
            row,
            conversation_id=conversation_id,
            session_identity=session_identity,
        )
        source = _target_source(namespace, agent, source_session)
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        token_estimate = count_message_tokens(msg)
        timestamp = _parse_timestamp(
            row["message_created_at"],
            _parse_timestamp(row["conversation_created_at"], now),
        )
        candidates.append(
            ImportCandidate(
                source_message_id=int(row["message_id"]),
                source_conversation_id=conversation_id,
                source_session=source_session,
                target_session_id=source,
                source=source,
                role=role,
                content=content,
                tool_call_id=tool_call_id,
                tool_calls=tool_calls,
                tool_name=tool_name,
                timestamp=timestamp,
                token_estimate=token_estimate,
            )
        )
    return candidates, len(rows), skipped_empty, len(conversation_ids)


def _target_has_import_table(conn: sqlite3.Connection) -> bool:
    return _table_exists(conn, "lcm_imported_messages")


def _ensure_import_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_imported_messages (
            import_id TEXT NOT NULL,
            source_message_id INTEGER NOT NULL,
            source_conversation_id INTEGER NOT NULL,
            source_session TEXT NOT NULL,
            target_store_id INTEGER NOT NULL,
            imported_at REAL NOT NULL,
            PRIMARY KEY (import_id, source_message_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_lcm_imported_messages_target
            ON lcm_imported_messages(target_store_id)
        """
    )


def _existing_source_message_ids(target_db: Path, import_id: str) -> set[int]:
    if not target_db.exists():
        return set()
    conn = sqlite3.connect(target_db)
    try:
        if not _target_has_import_table(conn):
            return set()
        rows = conn.execute(
            "SELECT source_message_id FROM lcm_imported_messages WHERE import_id = ?",
            (import_id,),
        ).fetchall()
        return {int(row[0]) for row in rows}
    finally:
        conn.close()


def _backup_target(target_db: Path) -> str | None:
    if not target_db.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    backup_path = target_db.with_name(f"{target_db.name}.backup-{stamp}")
    suffix = 1
    while backup_path.exists():
        backup_path = target_db.with_name(f"{target_db.name}.backup-{stamp}-{suffix}")
        suffix += 1

    source_conn = sqlite3.connect(_readonly_sqlite_uri(target_db), uri=True)
    backup_conn = sqlite3.connect(backup_path)
    try:
        source_conn.backup(backup_conn)
    finally:
        backup_conn.close()
        source_conn.close()
    return str(backup_path)


def import_lossless_claw(
    *,
    source_db: str | Path,
    target_db: str | Path,
    namespace: str = "openclaw-lcm",
    agent: str = "unknown",
    import_id: str | None = None,
    session_identity: str = "session_id",
    apply: bool = False,
) -> ImportResult:
    source_path = Path(source_db)
    target_path = Path(target_db)
    resolved_import_id = import_id or _default_import_id(source_path)
    if session_identity not in VALID_SESSION_IDENTITIES:
        raise ValueError(
            "session_identity must be one of "
            + ", ".join(sorted(VALID_SESSION_IDENTITIES))
        )

    with _connect_readonly(source_path) as source_conn:
        candidates, scanned, skipped_empty, conversations = _collect_candidates(
            source_conn,
            namespace=namespace,
            agent=agent,
            session_identity=session_identity,
        )

    existing_ids = _existing_source_message_ids(target_path, resolved_import_id)
    to_import = [candidate for candidate in candidates if candidate.source_message_id not in existing_ids]
    skipped_existing = len(candidates) - len(to_import)

    if not apply:
        return ImportResult(
            source_db=str(source_path),
            target_db=str(target_path),
            import_id=resolved_import_id,
            scanned=scanned,
            eligible=len(candidates),
            would_import=len(to_import),
            imported=0,
            skipped_existing=skipped_existing,
            skipped_empty=skipped_empty,
            conversations=conversations,
            backup_path=None,
        )

    if not to_import:
        return ImportResult(
            source_db=str(source_path),
            target_db=str(target_path),
            import_id=resolved_import_id,
            scanned=scanned,
            eligible=len(candidates),
            would_import=0,
            imported=0,
            skipped_existing=skipped_existing,
            skipped_empty=skipped_empty,
            conversations=conversations,
            backup_path=None,
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = _backup_target(target_path)
    protection_config = LCMConfig.from_env()
    protection_config.database_path = str(target_path)
    store = MessageStore(
        target_path,
        ingest_protection_config=protection_config,
        hermes_home=str(target_path.parent),
    )
    conn = store._conn
    _ensure_import_table(conn)

    imported = 0
    try:
        for candidate in to_import:
            msg: dict[str, Any] = {
                "role": candidate.role,
                "content": candidate.content,
            }
            if candidate.tool_call_id:
                msg["tool_call_id"] = candidate.tool_call_id
            if candidate.tool_calls:
                msg["tool_calls"] = candidate.tool_calls
            if candidate.tool_name:
                msg["tool_name"] = candidate.tool_name
            protected_msg = protect_message_for_ingest(
                msg,
                config=protection_config,
                hermes_home=str(target_path.parent),
                session_id=candidate.target_session_id,
            )
            tool_calls_json = json.dumps(protected_msg.get("tool_calls")) if protected_msg.get("tool_calls") else None
            cur = conn.execute(
                """INSERT INTO messages
                   (session_id, source, role, content, tool_call_id, tool_calls,
                    tool_name, timestamp, token_estimate, pinned)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (
                    candidate.target_session_id,
                    _normalize_source_value(candidate.source),
                    protected_msg.get("role", candidate.role),
                    normalize_content_value(protected_msg.get("content")),
                    protected_msg.get("tool_call_id"),
                    tool_calls_json,
                    protected_msg.get("tool_name"),
                    candidate.timestamp,
                    count_message_tokens(protected_msg),
                ),
            )
            conn.execute(
                """INSERT INTO lcm_imported_messages
                   (import_id, source_message_id, source_conversation_id, source_session,
                    target_store_id, imported_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    resolved_import_id,
                    candidate.source_message_id,
                    candidate.source_conversation_id,
                    candidate.source_session,
                    int(cur.lastrowid),
                    time.time(),
                ),
            )
            imported += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        store.close()

    return ImportResult(
        source_db=str(source_path),
        target_db=str(target_path),
        import_id=resolved_import_id,
        scanned=scanned,
        eligible=len(candidates),
        would_import=0,
        imported=imported,
        skipped_existing=skipped_existing,
        skipped_empty=skipped_empty,
        conversations=conversations,
        backup_path=backup_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import raw messages from a lossless-claw/OpenClaw LCM SQLite DB into hermes-lcm.",
    )
    parser.add_argument("--source-db", required=True, help="Path to the source lossless-claw/OpenClaw LCM SQLite DB")
    parser.add_argument("--target-db", required=True, help="Path to the target hermes-lcm SQLite DB")
    parser.add_argument("--namespace", default="openclaw-lcm", help="Provenance namespace for imported rows")
    parser.add_argument("--agent", default="unknown", help="Source OpenClaw agent/profile label for provenance")
    parser.add_argument("--import-id", help="Stable idempotency key. Defaults to a hash of the source DB path")
    parser.add_argument(
        "--session-identity",
        choices=sorted(VALID_SESSION_IDENTITIES),
        default="session_id",
        help=(
            "Source conversation field used for imported session_id/source provenance. "
            "Default session_id preserves concrete source conversation boundaries; "
            "session_key intentionally groups conversations sharing the same key."
        ),
    )
    parser.add_argument("--apply", action="store_true", help="Write rows to the target DB. Omit for dry-run")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = import_lossless_claw(
        source_db=args.source_db,
        target_db=args.target_db,
        namespace=args.namespace,
        agent=args.agent,
        import_id=args.import_id,
        session_identity=args.session_identity,
        apply=args.apply,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        mode = "apply" if args.apply else "dry-run"
        print(f"lossless-claw import {mode}")
        print(f"  source_db: {result.source_db}")
        print(f"  target_db: {result.target_db}")
        print(f"  import_id: {result.import_id}")
        print(f"  conversations: {result.conversations}")
        print(f"  scanned: {result.scanned}")
        print(f"  eligible: {result.eligible}")
        print(f"  would_import: {result.would_import}")
        print(f"  imported: {result.imported}")
        print(f"  skipped_existing: {result.skipped_existing}")
        print(f"  skipped_empty: {result.skipped_empty}")
        if result.backup_path:
            print(f"  backup_path: {result.backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
