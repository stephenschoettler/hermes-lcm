"""Immutable message store — the source of truth.

Every message is persisted verbatim and never modified. The store is
append-only with optional pruning of very old messages (configurable).

Each message gets a monotonic store_id used as a stable reference.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db_bootstrap import (
    ExternalContentFtsSpec,
    configure_connection,
    ensure_external_content_fts,
    run_versioned_migrations,
)
from .search_query import (
    build_snippet,
    count_term_matches,
    escape_like,
    extract_search_terms,
    requires_like_fallback,
)

logger = logging.getLogger(__name__)

AGE_DECAY_RATE = 0.001


def _normalize_search_sort(sort: str | None) -> str:
    normalized = (sort or "recency").strip().lower()
    return normalized if normalized in {"recency", "relevance", "hybrid"} else "recency"


def _build_search_order_by(
    sort: str | None,
    timestamp_expr: str,
    role_penalty_expr: str | None = None,
) -> str:
    normalized = _normalize_search_sort(sort)
    order_parts: list[str] = []
    if role_penalty_expr:
        order_parts.append(f"{role_penalty_expr} ASC")
    if normalized == "relevance":
        order_parts.extend(["rank ASC", f"{timestamp_expr} DESC"])
        return ", ".join(order_parts)
    if normalized == "hybrid":
        order_parts.extend([
            f"(rank / (1 + (((strftime('%s','now') - {timestamp_expr}) / 3600.0) * {AGE_DECAY_RATE}))) ASC",
            f"{timestamp_expr} DESC",
        ])
        return ", ".join(order_parts)
    order_parts.extend([f"{timestamp_expr} DESC", "rank ASC"])
    return ", ".join(order_parts)


def _fallback_result_sort_key(result: Dict[str, Any], sort: str | None) -> tuple[float, float, float]:
    normalized = _normalize_search_sort(sort)
    score = float(result.get("_fallback_score") or 0.0)
    timestamp = float(result.get("timestamp") or 0.0)
    role_bias = 1.0 if result.get("role") == "tool" else 0.0

    if normalized == "relevance":
        return (role_bias, -score, -timestamp)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - timestamp) / 3600.0)
        blended = score / (1 + (age_hours * AGE_DECAY_RATE))
        return (role_bias, -blended, -timestamp)
    return (role_bias, -timestamp, -score)


class MessageStore:
    """SQLite-backed immutable message store."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
        configure_connection(self._conn)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                store_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_estimate INTEGER DEFAULT 0,
                pinned INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_msg_session
                ON messages(session_id, store_id);
            CREATE INDEX IF NOT EXISTS idx_msg_session_ts
                ON messages(session_id, timestamp);

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        ensure_external_content_fts(
            self._conn,
            ExternalContentFtsSpec(
                table_name="messages_fts",
                content_table="messages",
                content_rowid="store_id",
                indexed_column="content",
                trigger_sqls=(
                    """
                    CREATE TRIGGER IF NOT EXISTS msg_fts_insert
                        AFTER INSERT ON messages BEGIN
                        INSERT INTO messages_fts(rowid, content)
                            VALUES (new.store_id, new.content);
                    END;
                    """,
                    """
                    CREATE TRIGGER IF NOT EXISTS msg_fts_delete
                        AFTER DELETE ON messages BEGIN
                        INSERT INTO messages_fts(messages_fts, rowid, content)
                            VALUES('delete', old.store_id, old.content);
                    END;
                    """,
                ),
            ),
        )
        run_versioned_migrations(self._conn)
        self._conn.commit()

    # -- Write operations ---------------------------------------------------

    def append(self, session_id: str, msg: Dict[str, Any],
               token_estimate: int = 0) -> int:
        """Persist a message and return its store_id."""
        tool_calls = msg.get("tool_calls")
        tc_json = json.dumps(tool_calls) if tool_calls else None

        cur = self._conn.execute(
            """INSERT INTO messages
               (session_id, role, content, tool_call_id, tool_calls,
                tool_name, timestamp, token_estimate, pinned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                msg.get("role", "unknown"),
                msg.get("content"),
                msg.get("tool_call_id"),
                tc_json,
                msg.get("tool_name"),
                time.time(),
                token_estimate,
                0,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def append_batch(self, session_id: str,
                     messages: List[Dict[str, Any]],
                     token_estimates: List[int] | None = None) -> List[int]:
        """Persist multiple messages in one transaction. Returns store_ids."""
        if token_estimates is None:
            token_estimates = [0] * len(messages)

        ids = []
        ts = time.time()
        with self._conn:
            for msg, est in zip(messages, token_estimates):
                tc = msg.get("tool_calls")
                tc_json = json.dumps(tc) if tc else None
                cur = self._conn.execute(
                    """INSERT INTO messages
                       (session_id, role, content, tool_call_id, tool_calls,
                        tool_name, timestamp, token_estimate, pinned)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        msg.get("role", "unknown"),
                        msg.get("content"),
                        msg.get("tool_call_id"),
                        tc_json,
                        msg.get("tool_name"),
                        ts,
                        est,
                        0,
                    ),
                )
                ids.append(cur.lastrowid)
        return ids

    def pin(self, store_id: int) -> None:
        """Mark a message as pinned (protected from pruning)."""
        self._conn.execute(
            "UPDATE messages SET pinned = 1 WHERE store_id = ?", (store_id,)
        )
        self._conn.commit()

    def unpin(self, store_id: int) -> None:
        self._conn.execute(
            "UPDATE messages SET pinned = 0 WHERE store_id = ?", (store_id,)
        )
        self._conn.commit()

    # -- Read operations ----------------------------------------------------

    def get(self, store_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single message by store_id."""
        row = self._conn.execute(
            "SELECT * FROM messages WHERE store_id = ?", (store_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_range(self, session_id: str, start_id: int = 0,
                  end_id: int | None = None,
                  limit: int = 1000) -> List[Dict[str, Any]]:
        """Get messages in a store_id range for a session."""
        if end_id is not None:
            rows = self._conn.execute(
                """SELECT * FROM messages
                   WHERE session_id = ? AND store_id >= ? AND store_id <= ?
                   ORDER BY store_id LIMIT ?""",
                (session_id, start_id, end_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM messages
                   WHERE session_id = ? AND store_id >= ?
                   ORDER BY store_id LIMIT ?""",
                (session_id, start_id, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_session_messages(self, session_id: str,
                             limit: int = 10000) -> List[Dict[str, Any]]:
        """Get all messages for a session, ordered by store_id."""
        rows = self._conn.execute(
            """SELECT * FROM messages
               WHERE session_id = ?
               ORDER BY store_id LIMIT ?""",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_session_count(self, session_id: str) -> int:
        """Count messages in a session."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_session_token_total(self, session_id: str) -> int:
        """Sum of token estimates for a session."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(token_estimate), 0) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_time_bounds(self, store_ids: List[int]) -> tuple[float | None, float | None]:
        if not store_ids:
            return None, None
        placeholders = ",".join("?" * len(store_ids))
        row = self._conn.execute(
            f"SELECT MIN(timestamp), MAX(timestamp) FROM messages WHERE store_id IN ({placeholders})",
            store_ids,
        ).fetchone()
        if not row:
            return None, None
        return row[0], row[1]

    # -- Search -------------------------------------------------------------

    def search(self, query: str, session_id: str | None = None,
               limit: int = 20, sort: str | None = None) -> List[Dict[str, Any]]:
        """FTS5 search across messages. Returns matches with snippets."""
        if requires_like_fallback(query):
            return self._search_like(query, session_id=session_id, limit=limit, sort=sort)

        order_by = _build_search_order_by(
            sort,
            "m.timestamp",
            "CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END",
        )
        try:
            if session_id:
                rows = self._conn.execute(
                    f"""SELECT m.*, rank as search_rank,
                              snippet(messages_fts, 0, '>>>', '<<<', '...', 40) as snippet
                       FROM messages_fts fts
                       JOIN messages m ON m.store_id = fts.rowid
                       WHERE messages_fts MATCH ? AND m.session_id = ?
                       ORDER BY {order_by} LIMIT ?""",
                    (query, session_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"""SELECT m.*, rank as search_rank,
                              snippet(messages_fts, 0, '>>>', '<<<', '...', 40) as snippet
                       FROM messages_fts fts
                       JOIN messages m ON m.store_id = fts.rowid
                       WHERE messages_fts MATCH ?
                       ORDER BY {order_by} LIMIT ?""",
                    (query, limit),
                ).fetchall()
        except sqlite3.Error:
            return self._search_like(query, session_id=session_id, limit=limit, sort=sort)
        results = []
        for r in rows:
            d = self._row_to_dict(r)
            d["search_rank"] = r[10] if len(r) > 10 else None
            d["snippet"] = r[11] if len(r) > 11 else ""
            results.append(d)
        return results

    def _search_like(self, query: str, session_id: str | None = None,
                     limit: int = 20, sort: str | None = None) -> List[Dict[str, Any]]:
        terms = extract_search_terms(query)
        if not terms:
            return []

        where: list[str] = ["content IS NOT NULL"]
        args: list[Any] = []
        if session_id:
            where.append("session_id = ?")
            args.append(session_id)
        like_clauses = []
        for term in terms:
            like_clauses.append("content LIKE ? ESCAPE '\\'")
            args.append(f"%{escape_like(term)}%")
        where.append("(" + " OR ".join(like_clauses) + ")")

        rows = self._conn.execute(
            f"SELECT * FROM messages WHERE {' AND '.join(where)}"
            , args,
        ).fetchall()

        results: list[Dict[str, Any]] = []
        for row in rows:
            result = self._row_to_dict(row)
            content = result.get("content") or ""
            score = sum(count_term_matches(content, term) for term in terms)
            if score <= 0:
                continue
            result["search_rank"] = -float(score)
            result["snippet"] = build_snippet(content, terms)
            result["_fallback_score"] = float(score)
            results.append(result)

        results.sort(key=lambda result: _fallback_result_sort_key(result, sort))
        for result in results:
            result.pop("_fallback_score", None)
        return results[:limit]

    # -- Helpers ------------------------------------------------------------

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a sqlite3 row to a dict."""
        if row is None:
            return {}
        cols = [
            "store_id", "session_id", "role", "content", "tool_call_id",
            "tool_calls", "tool_name", "timestamp", "token_estimate", "pinned",
        ]
        d = dict(zip(cols, row[:len(cols)]))
        # Deserialize tool_calls JSON
        if d.get("tool_calls"):
            try:
                d["tool_calls"] = json.loads(d["tool_calls"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d

    def to_openai_msg(self, stored: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a stored message back to OpenAI format."""
        msg: Dict[str, Any] = {"role": stored["role"]}
        if stored.get("content") is not None:
            msg["content"] = stored["content"]
        if stored.get("tool_calls"):
            msg["tool_calls"] = stored["tool_calls"]
        if stored.get("tool_call_id"):
            msg["tool_call_id"] = stored["tool_call_id"]
        if stored.get("tool_name"):
            msg["name"] = stored["tool_name"]
        return msg

    # -- Lifecycle ----------------------------------------------------------

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
