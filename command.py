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
        "- /lcm doctor clean apply: backup-first cleanup for safe pattern-matched candidates only",
        "- /lcm doctor retention: read-only retention analysis for stored session footprint and age",
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
    source_stats = status.get("source_lineage") or {}
    source_stats = {
        "messages_total": int(source_stats.get("messages_total", 0) or 0),
        "attributed_messages": int(source_stats.get("attributed_messages", 0) or 0),
        "normalized_unknown_messages": int(source_stats.get("normalized_unknown_messages", 0) or 0),
        "legacy_blank_source_messages": int(source_stats.get("legacy_blank_source_messages", 0) or 0),
        "effective_unknown_messages": int(source_stats.get("effective_unknown_messages", 0) or 0),
        **({"error": source_stats.get("error")} if source_stats.get("error") else {}),
    }


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
        f"source_messages_total: {source_stats['messages_total']}",
        f"source_attributed_messages: {source_stats['attributed_messages']}",
        f"source_unknown_messages: {source_stats['normalized_unknown_messages']}",
        f"source_legacy_blank_messages: {source_stats['legacy_blank_source_messages']}",
        f"source_effective_unknown_messages: {source_stats['effective_unknown_messages']}",
    ]

    if session_bound:
        lines.extend([
            f"store_messages: {status.get('store_messages', 0)}",
            f"dag_nodes: {status.get('dag_nodes', 0)}",
        ])
    else:
        lines.append(
            "note: no active Hermes session has initialized LCM in this process yet — after a fresh restart, send one normal message first if you want live per-session runtime details"
        )

    if "ignore_session_patterns_source" in status:
        lines.append(
            f"ignore_session_patterns_source: {status.get('ignore_session_patterns_source')}"
        )
    if "stateless_session_patterns_source" in status:
        lines.append(
            f"stateless_session_patterns_source: {status.get('stateless_session_patterns_source')}"
        )
    if source_stats.get("error"):
        lines.append(f"source_lineage_error: {source_stats['error']}")
    return "\n".join(lines)


def _scan_clean_candidates(engine) -> dict[str, Any]:
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
        return {
            "error": str(exc),
            "candidates": [],
            "ignored_count": 0,
            "stateless_count": 0,
            "protected_count": 0,
        }

    candidates = []
    ignored_count = 0
    stateless_count = 0
    protected_count = 0

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
        if session_id == getattr(engine, "_session_id", ""):
            protected_count += 1
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

    return {
        "error": None,
        "candidates": candidates,
        "ignored_count": ignored_count,
        "stateless_count": stateless_count,
        "protected_count": protected_count,
    }


def _scan_retention_candidates(engine) -> dict[str, Any]:
    conn = engine._store._conn
    now = datetime.now().timestamp()
    session_id = getattr(engine, "_session_id", "")
    if not session_id:
        return {
            "error": None,
            "sessions": [],
            "sessions_analyzed": 0,
            "stale_sessions_30d": 0,
            "stale_sessions_90d": 0,
            "retained_tokens_30d": 0,
            "retained_tokens_90d": 0,
            "protected_count": 0,
        }
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
                       COALESCE(SUM(token_estimate), 0) AS token_total,
                       MIN(timestamp) AS first_message_at,
                       MAX(timestamp) AS last_message_at
                FROM messages
                GROUP BY session_id
            ),
            node_stats AS (
                SELECT session_id,
                       COUNT(*) AS node_count,
                       COALESCE(SUM(token_count), 0) AS node_token_total,
                       MIN(COALESCE(earliest_at, created_at)) AS first_node_at,
                       MAX(COALESCE(latest_at, created_at)) AS last_node_at
                FROM summary_nodes
                GROUP BY session_id
            )
            SELECT s.session_id,
                   COALESCE(m.message_count, 0) AS message_count,
                   COALESCE(m.token_total, 0) AS token_total,
                   COALESCE(n.node_count, 0) AS node_count,
                   COALESCE(n.node_token_total, 0) AS node_token_total,
                   m.first_message_at,
                   m.last_message_at,
                   n.first_node_at,
                   n.last_node_at
            FROM session_ids s
            LEFT JOIN message_stats m ON m.session_id = s.session_id
            LEFT JOIN node_stats n ON n.session_id = s.session_id
            WHERE s.session_id = ?
            ORDER BY s.session_id
            """,
            (session_id,),
        ).fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "error": str(exc),
            "sessions": [],
            "sessions_analyzed": 0,
            "stale_sessions_30d": 0,
            "stale_sessions_90d": 0,
            "retained_tokens_30d": 0,
            "retained_tokens_90d": 0,
            "protected_count": 0,
        }

    sessions = []
    protected_count = 0
    stale_sessions_30d = 0
    stale_sessions_90d = 0
    retained_tokens_30d = 0
    retained_tokens_90d = 0

    for row in rows:
        (
            session_id,
            message_count,
            token_total,
            node_count,
            node_token_total,
            first_message_at,
            last_message_at,
            first_node_at,
            last_node_at,
        ) = row
        timestamps = [
            ts for ts in (first_message_at, last_message_at, first_node_at, last_node_at)
            if ts is not None
        ]
        if not timestamps:
            continue
        first_activity_at = min(float(ts) for ts in (first_message_at, first_node_at) if ts is not None)
        last_activity_at = max(float(ts) for ts in (last_message_at, last_node_at) if ts is not None)
        age_days = max(0.0, (now - last_activity_at) / 86400.0)
        protected = session_id == getattr(engine, "_session_id", "")
        total_footprint_tokens = int(token_total) + int(node_token_total)
        if protected:
            protected_count += 1
        if age_days >= 30.0:
            stale_sessions_30d += 1
            retained_tokens_30d += total_footprint_tokens
        if age_days >= 90.0:
            stale_sessions_90d += 1
            retained_tokens_90d += total_footprint_tokens
        sessions.append(
            {
                "session_id": session_id,
                "protected": protected,
                "message_count": int(message_count),
                "node_count": int(node_count),
                "token_total": total_footprint_tokens,
                "raw_token_total": int(token_total),
                "summary_token_total": int(node_token_total),
                "first_activity_at": float(first_activity_at),
                "last_activity_at": float(last_activity_at),
                "age_days": age_days,
            }
        )

    sessions.sort(
        key=lambda item: (
            1 if item["protected"] else 0,
            0 if item["age_days"] >= 30.0 else 1,
            -item["token_total"],
            -item["node_count"],
            -item["message_count"],
            item["last_activity_at"],
            item["session_id"],
        )
    )

    return {
        "error": None,
        "sessions": sessions,
        "sessions_analyzed": len(sessions),
        "stale_sessions_30d": stale_sessions_30d,
        "stale_sessions_90d": stale_sessions_90d,
        "retained_tokens_30d": retained_tokens_30d,
        "retained_tokens_90d": retained_tokens_90d,
        "protected_count": protected_count,
    }


def _backup_database(engine) -> dict[str, Any]:
    db_path = Path(engine._store.db_path)
    if not db_path.exists():
        return {
            "ok": False,
            "db_path": db_path,
            "error": "database file does not exist",
        }

    backup_root = Path(engine._hermes_home).expanduser() if getattr(engine, "_hermes_home", "") else db_path.parent
    backup_dir = backup_root / "backups" / "lcm"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{db_path.stem}-{timestamp}.sqlite3"

    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        engine._store._conn.commit()
        engine._dag._conn.commit()
        lifecycle_conn = getattr(getattr(engine, "_lifecycle", None), "_conn", None)
        if lifecycle_conn is not None:
            lifecycle_conn.commit()

        dest = sqlite3.connect(str(backup_path))
        try:
            engine._store._conn.backup(dest)
        finally:
            dest.close()
    except (OSError, sqlite3.Error) as exc:
        return {
            "ok": False,
            "db_path": db_path,
            "error": str(exc),
        }

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return {
        "ok": True,
        "db_path": db_path,
        "backup_path": backup_path,
        "backup_size": backup_size,
    }


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
    clean_scan = _scan_clean_candidates(engine)

    debt_rows = []
    lifecycle_conn = getattr(getattr(engine, "_lifecycle", None), "_conn", None)
    if lifecycle_conn is not None:
        try:
            debt_rows = lifecycle_conn.execute(
                """
                SELECT conversation_id, debt_kind, debt_size_estimate
                FROM lcm_lifecycle_state
                WHERE debt_kind IS NOT NULL AND debt_size_estimate > 0
                ORDER BY updated_at DESC
                """
            ).fetchall()
        except Exception as exc:  # pragma: no cover - defensive
            issues.append("lifecycle_state")
            debt_rows = [(f"error: {exc}", "error", 0)]

    observations: list[str] = []
    recommended_actions: list[str] = []

    if debt_rows:
        first = debt_rows[0]
        observations.append(
            f"maintenance_debt: {len(debt_rows)} conversation(s) currently carry deferred maintenance debt; first={first[0]} kind={first[1]} size={first[2]}"
        )
        recommended_actions.append(
            "let normal compaction turns reduce maintenance debt before attempting broader cleanup"
        )

    if clean_scan["error"]:
        observations.append(f"cleanup_candidates: scan error: {clean_scan['error']}")
    elif clean_scan["candidates"]:
        observations.append(
            f"cleanup_candidates: {len(clean_scan['candidates'])} pattern-matched junk/noise session candidate(s) detected"
        )
        recommended_actions.append("inspect candidate sessions with `/lcm doctor clean`")
        recommended_actions.append("create a safety snapshot first with `/lcm backup`")
    else:
        observations.append("cleanup_candidates: none")

    try:
        source_stats = engine._store.get_source_stats()
    except Exception as exc:  # pragma: no cover - defensive
        issues.append("source_lineage")
        source_stats = {
            "messages_total": 0,
            "attributed_messages": 0,
            "normalized_unknown_messages": 0,
            "legacy_blank_source_messages": 0,
            "effective_unknown_messages": 0,
            "error": str(exc),
        }
    observations.append(
        "source_lineage: "
        f"attributed={source_stats['attributed_messages']} "
        f"unknown={source_stats['normalized_unknown_messages']} "
        f"legacy_blank={source_stats['legacy_blank_source_messages']} "
        f"effective_unknown={source_stats['effective_unknown_messages']}"
    )
    if source_stats.get("error"):
        observations.append(f"source_lineage_error: {source_stats['error']}")
    if source_stats["legacy_blank_source_messages"]:
        recommended_actions.append("review legacy blank-source rows before any destructive cleanup or migration step")
        recommended_actions.append("treat `source=unknown` as the back-compat filter until legacy blank-source rows are normalized")

    if clean_scan.get("protected_count"):
        observations.append(
            f"protected_sessions: skipped {clean_scan['protected_count']} currently bound session(s) from cleanup candidates"
        )

    doctor_status = "issues-found" if integrity != "ok" or issues else (
        "action-recommended" if recommended_actions else "ok"
    )
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
    lines.append("observations:")
    for item in observations:
        lines.append(f"- {item}")
    lines.append("recommended_actions:")
    if recommended_actions:
        for item in recommended_actions:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def _doctor_clean_text(engine) -> str:
    scan = _scan_clean_candidates(engine)
    if scan["error"]:
        return "\n".join([
            "LCM doctor clean",
            "status: error",
            f"error: {scan['error']}",
            "note: read-only scan only — no rows were deleted",
        ])

    candidates = scan["candidates"]
    lines = [
        "LCM doctor clean",
        f"status: {'candidates-found' if candidates else 'ok'}",
        f"candidate_sessions: {len(candidates)}",
        f"ignored_pattern_matches: {scan['ignored_count']}",
        f"stateless_pattern_matches: {scan['stateless_count']}",
    ]
    if scan["protected_count"]:
        lines.append(f"protected_sessions_skipped: {scan['protected_count']}")

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
    lines.append("note: use `/lcm doctor clean apply` only after a backup-first review of these safe candidates")
    return "\n".join(lines)


def _doctor_retention_text(engine) -> str:
    scan = _scan_retention_candidates(engine)
    if scan["error"]:
        return "\n".join([
            "LCM doctor retention",
            "status: error",
            f"error: {scan['error']}",
            "note: read-only analysis only — no rows were deleted",
        ])

    sessions = scan["sessions"]
    lines = [
        "LCM doctor retention",
        f"status: {'analysis-ready' if sessions else 'ok'}",
        f"sessions_analyzed: {scan['sessions_analyzed']}",
        f"stale_sessions_30d: {scan['stale_sessions_30d']}",
        f"stale_sessions_90d: {scan['stale_sessions_90d']}",
        f"retained_tokens_30d: {scan['retained_tokens_30d']}",
        f"retained_tokens_90d: {scan['retained_tokens_90d']}",
    ]
    if scan["protected_count"]:
        lines.append(f"protected_sessions: {scan['protected_count']}")

    if not sessions:
        lines.append("result: no stored sessions found for retention analysis")
        lines.append("note: read-only analysis only — no rows were deleted")
        return "\n".join(lines)

    lines.append("retention_candidates:")
    for item in sessions[:20]:
        lines.append(
            "- "
            f"{item['session_id']} | protected={'yes' if item['protected'] else 'no'} | "
            f"messages={item['message_count']} | nodes={item['node_count']} | "
            f"tokens={item['token_total']} | age_days={item['age_days']:.1f}"
        )
    if len(sessions) > 20:
        lines.append(f"... {len(sessions) - 20} more session(s) omitted")
    lines.append("note: retention analysis is scoped to the active session only")
    lines.append("note: stale sessions are listed before fresh ones; within each bucket, candidates are sorted by footprint (tokens/nodes/messages), with protected current-session entries listed after non-protected ones")
    lines.append("note: read-only analysis only — no rows were deleted")
    lines.append("note: if you prune later, create a safety snapshot first with `/lcm backup`")
    return "\n".join(lines)


def _doctor_clean_apply_text(engine) -> str:
    if not getattr(getattr(engine, "_config", None), "doctor_clean_apply_enabled", False):
        return "\n".join([
            "LCM doctor clean apply",
            "status: denied",
            "error: destructive cleanup is disabled by default",
            "note: set LCM_DOCTOR_CLEAN_APPLY_ENABLED=true only in trusted operator environments",
            "note: no rows were deleted",
        ])

    scan = _scan_clean_candidates(engine)
    if scan["error"]:
        return "\n".join([
            "LCM doctor clean apply",
            "status: error",
            f"error: {scan['error']}",
            "note: cleanup apply aborted before any rows were deleted",
        ])

    candidates = scan["candidates"]
    if not candidates:
        return "\n".join([
            "LCM doctor clean apply",
            "status: ok",
            "candidate_sessions: 0",
            "result: no safe cleanup candidates detected",
            "note: nothing was deleted",
        ])

    backup = _backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM doctor clean apply",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: backup failed: {backup['error']}",
            "note: cleanup apply aborted before any rows were deleted",
        ])

    session_ids = {item["session_id"] for item in candidates}
    messages_deleted = sum(engine._store.delete_session_messages(session_id) for session_id in session_ids)
    nodes_deleted = sum(engine._dag.delete_session_nodes(session_id) for session_id in session_ids)
    lifecycle_deleted, lifecycle_skipped = engine._lifecycle.delete_safe_rows_for_sessions(
        session_ids,
        protected_session_ids={getattr(engine, "_session_id", "")},
    )

    return "\n".join([
        "LCM doctor clean apply",
        "status: ok",
        f"database_path: {backup['db_path']}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size: {_fmt_size(int(backup['backup_size']))}",
        f"candidate_sessions: {len(candidates)}",
        f"messages_deleted: {messages_deleted}",
        f"nodes_deleted: {nodes_deleted}",
        f"lifecycle_rows_deleted: {lifecycle_deleted}",
        f"lifecycle_rows_skipped: {lifecycle_skipped}",
        "note: backup created before cleanup apply",
    ])


def _backup_text(engine) -> str:
    backup = _backup_database(engine)
    if not backup["ok"]:
        return "\n".join([
            "LCM backup",
            "status: error",
            f"database_path: {backup['db_path']}",
            f"error: {backup['error']}",
        ])

    return "\n".join([
        "LCM backup",
        "status: ok",
        f"database_path: {backup['db_path']}",
        f"backup_path: {backup['backup_path']}",
        f"backup_size: {_fmt_size(int(backup['backup_size']))}",
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
        if len(rest) == 1 and rest[0].lower() == "retention":
            return _doctor_retention_text(engine)
        if len(rest) == 2 and rest[0].lower() == "clean" and rest[1].lower() == "apply":
            return _doctor_clean_apply_text(engine)
        return _help_text("`/lcm doctor` currently supports `clean`, `clean apply`, and `retention` as extra subcommands.")

    if head == "backup":
        if rest:
            return _help_text("`/lcm backup` does not accept extra arguments.")
        return _backup_text(engine)

    if head == "help":
        return _help_text()

    return _help_text(f"Unknown subcommand: {tokens[0]}")
