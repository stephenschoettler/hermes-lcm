"""Slash-style /lcm command helpers for Hermes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any

from .session_patterns import build_session_match_keys, matches_session_pattern


def _fmt_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _fmt_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024
        unit += 1
    precision = 0 if value >= 100 else 1 if value >= 10 else 2
    return f"{value:.{precision}f} {units[unit]}"


def _help_text(error: str | None = None) -> str:
    lines = []
    if error:
        lines.append(error)
        lines.append("")
    lines.extend([
        "LCM command help",
        "- /lcm or /lcm status: show current LCM runtime/session status",
        "- /lcm doctor: run read-only LCM health checks",
        "- /lcm doctor clean: best-effort scan of obvious junk/noise session candidates without deleting anything",
        "- /lcm backup: create a timestamped SQLite backup before any future cleanup workflow",
        "- /lcm help: show this help",
    ])
    return "\n".join(lines)


def _status_text(engine) -> str:
    status = engine.get_status()
    db_path = Path(engine._store.db_path)
    db_exists = db_path.exists()
    db_size = db_path.stat().st_size if db_exists else 0
    session_bound = bool(engine._session_id)

    def _safe_scalar(conn, query: str) -> int | str:
        try:
            return int(conn.execute(query).fetchone()[0])
        except Exception as exc:  # pragma: no cover - defensive
            return f"error: {exc}"

    lines = [
        "LCM status",
        f"engine: {status.get('engine', engine.name)}",
        f"session_id: {engine._session_id or '(unbound)'}",
        f"session_platform: {status.get('session_platform') or ('(unbound)' if not session_bound else '(unknown)')}",
        f"database_path: {db_path}",
        f"database_exists: {_fmt_bool(db_exists)}",
        f"database_size: {_fmt_size(db_size) if db_exists else 'missing'}",
        f"compression_count: {engine.compression_count}",
        f"threshold_tokens: {engine.threshold_tokens if session_bound else '(uninitialized)'}",
        f"session_ignored: {_fmt_bool(status.get('session_ignored'))}",
        f"session_stateless: {_fmt_bool(status.get('session_stateless'))}",
    ]

    if session_bound:
        lines.extend([
            f"store_messages: {status.get('store_messages', 0)}",
            f"dag_nodes: {status.get('dag_nodes', 0)}",
        ])
    else:
        lines.extend([
            f"messages_total: {_safe_scalar(engine._store._conn, 'SELECT COUNT(*) FROM messages')}",
            f"message_sessions_total: {_safe_scalar(engine._store._conn, 'SELECT COUNT(DISTINCT session_id) FROM messages')}",
            f"summary_nodes_total: {_safe_scalar(engine._dag._conn, 'SELECT COUNT(*) FROM summary_nodes')}",
            f"summary_node_sessions_total: {_safe_scalar(engine._dag._conn, 'SELECT COUNT(DISTINCT session_id) FROM summary_nodes')}",
            "note: no active Hermes session has initialized LCM in this process yet — after a fresh restart, send one normal message first if you want live per-session runtime details",
        ])

    if "ignore_session_patterns_source" in status:
        lines.append(
            f"ignore_session_patterns_source: {status.get('ignore_session_patterns_source')}"
        )
    if "stateless_session_patterns_source" in status:
        lines.append(
            f"stateless_session_patterns_source: {status.get('stateless_session_patterns_source')}"
        )
    return "\n".join(lines)


def _doctor_text(engine) -> str:
    db_path = Path(engine._store.db_path)
    store_conn = engine._store._conn
    dag_conn = engine._dag._conn

    issues: list[str] = []

    def _safe_count(conn, query: str, issue_key: str) -> int | str:
        try:
            return int(conn.execute(query).fetchone()[0])
        except Exception as exc:  # pragma: no cover - defensive
            issues.append(issue_key)
            return f"error: {exc}"

    try:
        integrity_row = store_conn.execute("PRAGMA integrity_check").fetchone()
        integrity = str(integrity_row[0]) if integrity_row else "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        integrity = f"error: {exc}"
        issues.append("sqlite_integrity")

    try:
        store_fts_count = int(store_conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0])
        store_fts = "ok"
    except Exception as exc:  # pragma: no cover - defensive
        store_fts_count = f"error: {exc}"
        store_fts = f"error: {exc}"
        issues.append("messages_fts")

    try:
        node_fts_count = int(dag_conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0])
        node_fts = "ok"
    except Exception as exc:  # pragma: no cover - defensive
        node_fts_count = f"error: {exc}"
        node_fts = f"error: {exc}"
        issues.append("nodes_fts")

    total_messages = _safe_count(store_conn, "SELECT COUNT(*) FROM messages", "messages_total")
    total_message_sessions = _safe_count(
        store_conn,
        "SELECT COUNT(DISTINCT session_id) FROM messages",
        "message_sessions_total",
    )
    total_nodes = _safe_count(dag_conn, "SELECT COUNT(*) FROM summary_nodes", "summary_nodes_total")
    total_node_sessions = _safe_count(
        dag_conn,
        "SELECT COUNT(DISTINCT session_id) FROM summary_nodes",
        "summary_node_sessions_total",
    )

    db_exists = db_path.exists()
    db_size = db_path.stat().st_size if db_exists else 0

    doctor_status = "ok" if integrity == "ok" and not issues else "issues-found"
    lines = [
        "LCM doctor",
        f"status: {doctor_status}",
        f"database_path: {db_path}",
        f"database_exists: {_fmt_bool(db_exists)}",
        f"database_size: {_fmt_size(db_size) if db_exists else 'missing'}",
        f"sqlite_integrity: {integrity}",
        f"messages_total: {total_messages}",
        f"message_sessions_total: {total_message_sessions}",
        f"summary_nodes_total: {total_nodes}",
        f"summary_node_sessions_total: {total_node_sessions}",
        f"messages_fts: {store_fts}",
        f"messages_fts_rows: {store_fts_count}",
        f"nodes_fts: {node_fts}",
        f"nodes_fts_rows: {node_fts_count}",
    ]
    if issues:
        lines.append(f"issues: {', '.join(issues)}")
    else:
        lines.append("issues: none")
    return "\n".join(lines)


def _doctor_clean_text(engine) -> str:
    conn = engine._store._conn
    try:
        rows = conn.execute(
            """
            WITH session_ids AS (
                SELECT session_id FROM messages
                UNION
                SELECT session_id FROM summary_nodes
            ),
            message_stats AS (
                SELECT session_id,
                       COUNT(*) AS message_count,
                       COALESCE(SUM(token_estimate), 0) AS token_total
                FROM messages
                GROUP BY session_id
            ),
            node_stats AS (
                SELECT session_id, COUNT(*) AS node_count
                FROM summary_nodes
                GROUP BY session_id
            )
            SELECT s.session_id,
                   COALESCE(m.message_count, 0) AS message_count,
                   COALESCE(m.token_total, 0) AS token_total,
                   COALESCE(n.node_count, 0) AS node_count
            FROM session_ids s
            LEFT JOIN message_stats m ON m.session_id = s.session_id
            LEFT JOIN node_stats n ON n.session_id = s.session_id
            ORDER BY s.session_id
            """
        ).fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        return "\n".join([
            "LCM doctor clean",
            "status: error",
            f"error: {exc}",
            "note: read-only scan only — no rows were deleted",
        ])

    candidates = []
    ignored_count = 0
    stateless_count = 0

    for session_id, message_count, token_total, node_count in rows:
        keys = build_session_match_keys(session_id)
        matched_classes = []
        if matches_session_pattern(keys, engine._compiled_ignore_session_patterns):
            matched_classes.append("ignored-pattern")
            ignored_count += 1
        elif matches_session_pattern(keys, engine._compiled_stateless_session_patterns):
            matched_classes.append("stateless-pattern")
            stateless_count += 1
        if not matched_classes:
            continue
        candidates.append(
            {
                "session_id": session_id,
                "classes": matched_classes,
                "message_count": int(message_count),
                "node_count": int(node_count),
                "token_total": int(token_total),
            }
        )

    lines = [
        "LCM doctor clean",
        f"status: {'candidates-found' if candidates else 'ok'}",
        f"candidate_sessions: {len(candidates)}",
        f"ignored_pattern_matches: {ignored_count}",
        f"stateless_pattern_matches: {stateless_count}",
    ]

    if not candidates:
        lines.append("result: no obvious junk/noise session candidates detected")
        return "\n".join(lines)

    lines.append("candidates:")
    for item in candidates[:20]:
        classes = ", ".join(item["classes"])
        lines.append(
            "- "
            f"{item['session_id']} | class={classes} | messages={item['message_count']} | "
            f"nodes={item['node_count']} | tokens={item['token_total']}"
        )
    if len(candidates) > 20:
        lines.append(f"... {len(candidates) - 20} more candidate session(s) omitted")
    lines.append("note: best-effort stored-session scan only — platform-only matches may not be reconstructable from the SQLite state")
    lines.append("note: read-only scan only — no rows were deleted")
    return "\n".join(lines)


def _backup_text(engine) -> str:
    db_path = Path(engine._store.db_path)
    if not db_path.exists():
        return "\n".join([
            "LCM backup",
            "status: error",
            f"database_path: {db_path}",
            "error: database file does not exist",
        ])

    backup_root = Path(engine._hermes_home).expanduser() if getattr(engine, "_hermes_home", "") else db_path.parent
    backup_dir = backup_root / "backups" / "lcm"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{db_path.stem}-{timestamp}.sqlite3"

    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        engine._store._conn.commit()
        engine._dag._conn.commit()

        dest = sqlite3.connect(str(backup_path))
        try:
            engine._store._conn.backup(dest)
        finally:
            dest.close()
    except (OSError, sqlite3.Error) as exc:
        return "\n".join([
            "LCM backup",
            "status: error",
            f"database_path: {db_path}",
            f"error: {exc}",
        ])

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return "\n".join([
        "LCM backup",
        "status: ok",
        f"database_path: {db_path}",
        f"backup_path: {backup_path}",
        f"backup_size: {_fmt_size(backup_size)}",
        "note: backup created before any future cleanup/apply workflow",
    ])


def handle_lcm_command(raw_args: str | None, engine) -> str:
    tokens = [part.strip() for part in (raw_args or "").strip().split() if part.strip()]
    if not tokens:
        return _status_text(engine)

    head = tokens[0].lower()
    rest = tokens[1:]

    if head == "status":
        if rest:
            return _help_text("`/lcm status` does not accept extra arguments.")
        return _status_text(engine)

    if head == "doctor":
        if not rest:
            return _doctor_text(engine)
        if len(rest) == 1 and rest[0].lower() == "clean":
            return _doctor_clean_text(engine)
        if len(rest) == 2 and rest[0].lower() == "clean" and rest[1].lower() == "apply":
            return _help_text("`/lcm doctor clean apply` is not implemented yet. This slice is read-only diagnostics only.")
        return _help_text("`/lcm doctor` currently supports only `clean` as an extra subcommand.")

    if head == "backup":
        if rest:
            return _help_text("`/lcm backup` does not accept extra arguments.")
        return _backup_text(engine)

    if head == "help":
        return _help_text()

    return _help_text(f"Unknown subcommand: {tokens[0]}")
