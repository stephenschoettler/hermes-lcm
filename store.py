"""Immutable-first message store — the source of truth.

Every message is persisted durably in SQLite. The normal model is append-only,
with one narrow opt-in exception: already-externalized summarized tool-result
rows may be rewritten to compact GC tombstones while preserving the original
row identity (`store_id`) for DAG/source lookup.
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
    compute_directness_rank_bonus_upper_bound,
    compute_directness_score,
    compute_search_fetch_limit,
    contains_risky_fts_ascii,
    count_term_matches,
    escape_like,
    extract_quoted_phrases,
    extract_search_terms,
    normalize_search_sort,
    requires_like_fallback,
    AGE_DECAY_RATE,
    should_apply_directness_rank_adjustment,
)
from .tokens import count_message_tokens

logger = logging.getLogger(__name__)


_MESSAGE_ROLE_BIAS_SQL = "CASE m.role WHEN 'user' THEN 0 WHEN 'assistant' THEN 1 WHEN 'tool' THEN 2 ELSE 1 END"
_MESSAGE_SELECT_COLUMNS = (
    "store_id, session_id, source, role, content, tool_call_id, "
    "tool_calls, tool_name, timestamp, token_estimate, pinned"
)


def _message_role_bias(role: str | None) -> float:
    if role == "user":
        return 0.0
    if role == "assistant":
        return 1.0
    if role == "tool":
        return 2.0
    return 1.0


def _message_directness_score(role: str | None, content: str | None, terms: List[str], phrases: List[str] | None = None) -> float:
    score = compute_directness_score(content or "", terms, phrases)
    if role == "tool":
        stripped = (content or "").lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            score -= 4.0
    return score


def _build_search_order_by(
    sort: str | None,
    timestamp_expr: str,
    role_penalty_expr: str | None = None,
) -> str:
    normalized = normalize_search_sort(sort)
    order_parts: list[str] = []
    if normalized == "relevance":
        if role_penalty_expr:
            order_parts.extend(["rank ASC", f"{role_penalty_expr} ASC", f"{timestamp_expr} DESC"])
        else:
            order_parts.extend(["rank ASC", f"{timestamp_expr} DESC"])
        return ", ".join(order_parts)
    if normalized == "hybrid":
        blended = f"(rank / (1 + (MAX(0.0, ((strftime('%s','now') - {timestamp_expr}) / 3600.0)) * {AGE_DECAY_RATE})))"
        if role_penalty_expr:
            order_parts.extend([f"{blended} ASC", f"{role_penalty_expr} ASC", f"{timestamp_expr} DESC"])
        else:
            order_parts.extend([f"{blended} ASC", f"{timestamp_expr} DESC"])
        return ", ".join(order_parts)
    order_parts.append(f"{timestamp_expr} DESC")
    if role_penalty_expr:
        order_parts.append(f"{role_penalty_expr} ASC")
    order_parts.append("rank ASC")
    return ", ".join(order_parts)


def _fallback_result_sort_key(result: Dict[str, Any], sort: str | None) -> tuple[float, float, float, float]:
    normalized = normalize_search_sort(sort)
    score = float(result.get("_fallback_score") or 0.0)
    directness = float(result.get("_directness_score") or 0.0)
    timestamp = float(result.get("timestamp") or 0.0)
    role_bias = _message_role_bias(result.get("role"))

    if normalized == "relevance":
        return (-score, -directness, role_bias, -timestamp)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - timestamp) / 3600.0)
        blended = score / (1 + (age_hours * AGE_DECAY_RATE))
        return (-blended, -directness, role_bias, -timestamp)
    return (-timestamp, role_bias, -score, -directness)


def _fts_result_sort_key(result: Dict[str, Any], sort: str | None) -> tuple[float, float, float, float]:
    normalized = normalize_search_sort(sort)
    rank = result.get("search_rank")
    rank_value = float(rank) if rank is not None else float("inf")
    directness = float(result.get("_directness_score") or 0.0)
    timestamp = float(result.get("timestamp") or 0.0)
    role_bias = _message_role_bias(result.get("role"))

    if normalized == "relevance":
        return (rank_value, -directness, role_bias, -timestamp)
    if normalized == "hybrid":
        age_hours = max(0.0, (time.time() - timestamp) / 3600.0)
        blended = rank_value / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("inf")
        return (blended, -directness, role_bias, -timestamp)
    return (-timestamp, role_bias, rank_value, 0.0)


def _fts_primary_value(result: Dict[str, Any], sort: str | None) -> float:
    normalized = normalize_search_sort(sort)
    rank = result.get("search_rank")
    rank_value = float(rank) if rank is not None else float("inf")
    if normalized == "hybrid":
        timestamp = float(result.get("timestamp") or 0.0)
        age_hours = max(0.0, (time.time() - timestamp) / 3600.0)
        return rank_value / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("inf")
    return rank_value


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
                source TEXT DEFAULT '',
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
                    """
                    CREATE TRIGGER IF NOT EXISTS msg_fts_update
                        AFTER UPDATE OF content ON messages BEGIN
                        INSERT INTO messages_fts(messages_fts, rowid, content)
                            VALUES('delete', old.store_id, old.content);
                        INSERT INTO messages_fts(rowid, content)
                            VALUES (new.store_id, new.content);
                    END;
                    """,
                ),
            ),
        )
        run_versioned_migrations(self._conn)
        self._ensure_source_column()
        self._conn.commit()

    def _ensure_source_column(self) -> None:
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "source" not in columns:
            self._conn.execute("ALTER TABLE messages ADD COLUMN source TEXT DEFAULT ''")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_msg_source_session ON messages(source, session_id, store_id)"
        )

    # -- Write operations ---------------------------------------------------

    def append(self, session_id: str, msg: Dict[str, Any],
               token_estimate: int = 0, source: str = "") -> int:
        """Persist a message and return its store_id."""
        tool_calls = msg.get("tool_calls")
        tc_json = json.dumps(tool_calls) if tool_calls else None

        cur = self._conn.execute(
            """INSERT INTO messages
               (session_id, source, role, content, tool_call_id, tool_calls,
                tool_name, timestamp, token_estimate, pinned)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                source or "",
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
                     token_estimates: List[int] | None = None,
                     source: str = "") -> List[int]:
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
                       (session_id, source, role, content, tool_call_id, tool_calls,
                        tool_name, timestamp, token_estimate, pinned)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        source or "",
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

    def delete_session_messages(self, session_id: str) -> int:
        """Delete all messages for a session. Returns count deleted."""
        cur = self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?",
            (session_id,),
        )
        self._conn.commit()
        deleted = cur.rowcount if cur.rowcount is not None else 0
        return deleted

    def gc_externalized_tool_result(self, store_id: int, placeholder: str) -> bool:
        """Rewrite one unpinned tool-result row to a compact GC placeholder."""
        row = self._conn.execute(
            "SELECT role, pinned, content, tool_call_id FROM messages WHERE store_id = ?",
            (store_id,),
        ).fetchone()
        if row is None:
            return False
        role, pinned, current_content, tool_call_id = row
        if role != "tool" or bool(pinned) or current_content == placeholder:
            return False
        placeholder_tokens = count_message_tokens(
            {
                "role": "tool",
                "content": placeholder,
                "tool_call_id": tool_call_id,
            }
        )
        self._conn.execute(
            "UPDATE messages SET content = ?, token_estimate = ? WHERE store_id = ?",
            (placeholder, placeholder_tokens, store_id),
        )
        self._conn.commit()
        return True

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
            f"SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages WHERE store_id = ?", (store_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_batch(self, store_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Retrieve multiple messages by store_id in a single query.

        Returns a dict mapping store_id → message dict.
        """
        if not store_ids:
            return {}
        placeholders = ",".join("?" for _ in store_ids)
        rows = self._conn.execute(
            f"SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages WHERE store_id IN ({placeholders})",
            store_ids,
        ).fetchall()
        return {row[0]: self._row_to_dict(row) for row in rows}

    def get_range(self, session_id: str, start_id: int = 0,
                  end_id: int | None = None,
                  limit: int = 1000) -> List[Dict[str, Any]]:
        """Get messages in a store_id range for a session."""
        if end_id is not None:
            rows = self._conn.execute(
                f"""SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages
                   WHERE session_id = ? AND store_id >= ? AND store_id <= ?
                   ORDER BY store_id LIMIT ?""",
                (session_id, start_id, end_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"""SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages
                   WHERE session_id = ? AND store_id >= ?
                   ORDER BY store_id LIMIT ?""",
                (session_id, start_id, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_session_messages(self, session_id: str,
                             limit: int = 10000) -> List[Dict[str, Any]]:
        """Get all messages for a session, ordered by store_id."""
        rows = self._conn.execute(
            f"""SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages
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
               limit: int = 20, sort: str | None = None,
               source: str | None = None) -> List[Dict[str, Any]]:
        """FTS5 search across messages. Returns matches with snippets."""
        terms = extract_search_terms(query)
        phrases = extract_quoted_phrases(query)
        if requires_like_fallback(query):
            return self._search_like(query, session_id=session_id, limit=limit, sort=sort, source=source)

        order_by = _build_search_order_by(
            sort,
            "m.timestamp",
            _MESSAGE_ROLE_BIAS_SQL,
        )
        fetch_limit = compute_search_fetch_limit(limit, terms, phrases)
        apply_directness_adjustment = should_apply_directness_rank_adjustment(terms, phrases)
        max_rank_bonus = compute_directness_rank_bonus_upper_bound(terms, phrases) * 3e-7
        offset = 0
        results: list[Dict[str, Any]] = []
        while True:
            try:
                if session_id:
                    if source:
                        rows = self._conn.execute(
                            f"""SELECT m.store_id, m.session_id, m.source, m.role, m.content, m.tool_call_id,
                                      m.tool_calls, m.tool_name, m.timestamp, m.token_estimate, m.pinned,
                                      rank as search_rank,
                                      snippet(messages_fts, 0, '>>>', '<<<', '...', 40) as snippet
                               FROM messages_fts fts
                               JOIN messages m ON m.store_id = fts.rowid
                               WHERE messages_fts MATCH ? AND m.session_id = ? AND m.source = ?
                               ORDER BY {order_by} LIMIT ? OFFSET ?""",
                            (query, session_id, source, fetch_limit, offset),
                        ).fetchall()
                    else:
                        rows = self._conn.execute(
                            f"""SELECT m.store_id, m.session_id, m.source, m.role, m.content, m.tool_call_id,
                                      m.tool_calls, m.tool_name, m.timestamp, m.token_estimate, m.pinned,
                                      rank as search_rank,
                                      snippet(messages_fts, 0, '>>>', '<<<', '...', 40) as snippet
                               FROM messages_fts fts
                               JOIN messages m ON m.store_id = fts.rowid
                               WHERE messages_fts MATCH ? AND m.session_id = ?
                               ORDER BY {order_by} LIMIT ? OFFSET ?""",
                            (query, session_id, fetch_limit, offset),
                        ).fetchall()
                else:
                    if source:
                        rows = self._conn.execute(
                            f"""SELECT m.store_id, m.session_id, m.source, m.role, m.content, m.tool_call_id,
                                      m.tool_calls, m.tool_name, m.timestamp, m.token_estimate, m.pinned,
                                      rank as search_rank,
                                      snippet(messages_fts, 0, '>>>', '<<<', '...', 40) as snippet
                               FROM messages_fts fts
                               JOIN messages m ON m.store_id = fts.rowid
                               WHERE messages_fts MATCH ? AND m.source = ?
                               ORDER BY {order_by} LIMIT ? OFFSET ?""",
                            (query, source, fetch_limit, offset),
                        ).fetchall()
                    else:
                        rows = self._conn.execute(
                            f"""SELECT m.store_id, m.session_id, m.source, m.role, m.content, m.tool_call_id,
                                      m.tool_calls, m.tool_name, m.timestamp, m.token_estimate, m.pinned,
                                      rank as search_rank,
                                      snippet(messages_fts, 0, '>>>', '<<<', '...', 40) as snippet
                               FROM messages_fts fts
                               JOIN messages m ON m.store_id = fts.rowid
                               WHERE messages_fts MATCH ?
                               ORDER BY {order_by} LIMIT ? OFFSET ?""",
                            (query, fetch_limit, offset),
                        ).fetchall()
            except sqlite3.Error as exc:
                logger.warning("FTS message search failed, falling back to LIKE: %s", exc)
                return self._search_like(query, session_id=session_id, limit=limit, sort=sort, source=source)

            raw_primary_values: list[float] = []
            for r in rows:
                d = self._row_to_dict(r)
                base_columns = 11
                d["search_rank"] = r[base_columns] if len(r) > base_columns else None
                d["snippet"] = r[base_columns + 1] if len(r) > (base_columns + 1) else ""
                d["_directness_score"] = _message_directness_score(d.get("role"), d.get("content"), terms, phrases)
                if apply_directness_adjustment and d["search_rank"] is not None:
                    rank_adjustment = max(float(d["_directness_score"]), 0.0)
                    d["search_rank"] = float(d["search_rank"]) - (rank_adjustment * 3e-7)
                raw_primary_values.append(_fts_primary_value(d, sort))
                results.append(d)
            results.sort(key=lambda result: _fts_result_sort_key(result, sort))

            if not apply_directness_adjustment or len(rows) < fetch_limit or len(results) <= limit:
                return results[:limit]

            worst_visible_primary = _fts_primary_value(results[min(limit, len(results)) - 1], sort)
            last_fetched_primary = raw_primary_values[-1]
            best_unseen_primary = last_fetched_primary - max_rank_bonus
            if best_unseen_primary > worst_visible_primary:
                return results[:limit]

            offset += len(rows)
            fetch_limit *= 2

    def _search_like(self, query: str, session_id: str | None = None,
                     limit: int = 20, sort: str | None = None,
                     source: str | None = None) -> List[Dict[str, Any]]:
        terms = extract_search_terms(query)
        phrases = extract_quoted_phrases(query)
        if not terms:
            return []

        where: list[str] = ["content IS NOT NULL"]
        args: list[Any] = []
        if session_id:
            where.append("session_id = ?")
            args.append(session_id)
        if source:
            where.append("source = ?")
            args.append(source)
        like_clauses = []
        for term in terms:
            like_clauses.append("content LIKE ? ESCAPE '\\'")
            args.append(f"%{escape_like(term)}%")
        where.append("(" + " OR ".join(like_clauses) + ")")

        rows = self._conn.execute(
            f"SELECT {_MESSAGE_SELECT_COLUMNS} FROM messages WHERE {' AND '.join(where)}",
            args,
        ).fetchall()
        results: List[Dict[str, Any]] = []
        collapse_risky_repeats = contains_risky_fts_ascii(query)
        for row in rows:
            result = self._row_to_dict(row)
            content = result.get("content") or ""
            score = sum(
                min(count_term_matches(content, term), 1) if collapse_risky_repeats else count_term_matches(content, term)
                for term in terms
            )
            if score <= 0:
                continue
            result["search_rank"] = -float(score)
            result["snippet"] = build_snippet(content, terms)
            result["_fallback_score"] = float(score)
            result["_directness_score"] = _message_directness_score(result.get("role"), content, terms, phrases)
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
            "store_id", "session_id", "source", "role", "content", "tool_call_id",
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
