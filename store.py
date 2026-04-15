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
    set_schema_version,
)

logger = logging.getLogger(__name__)


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
                trigger_name="msg_fts_insert",
                trigger_sql="""
                    CREATE TRIGGER IF NOT EXISTS msg_fts_insert
                        AFTER INSERT ON messages BEGIN
                        INSERT INTO messages_fts(rowid, content)
                            VALUES (new.store_id, new.content);
                    END;
                """,
            ),
        )
        set_schema_version(self._conn)
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

    # -- Search -------------------------------------------------------------

    def search(self, query: str, session_id: str | None = None,
               limit: int = 20) -> List[Dict[str, Any]]:
        """FTS5 search across messages. Returns matches with snippets."""
        if session_id:
            rows = self._conn.execute(
                """SELECT m.*, snippet(messages_fts, 0, '>>>', '<<<', '...', 40) as snippet
                   FROM messages_fts fts
                   JOIN messages m ON m.store_id = fts.rowid
                   WHERE messages_fts MATCH ? AND m.session_id = ?
                   ORDER BY rank LIMIT ?""",
                (query, session_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT m.*, snippet(messages_fts, 0, '>>>', '<<<', '...', 40) as snippet
                   FROM messages_fts fts
                   JOIN messages m ON m.store_id = fts.rowid
                   WHERE messages_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
        results = []
        for r in rows:
            d = self._row_to_dict(r)
            # snippet is the extra column
            d["snippet"] = r[-1] if len(r) > 10 else ""
            results.append(d)
        return results

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
