"""Crash-safe background preparation and atomic summary publication.

Prepared summaries deliberately live outside ``summary_nodes``.  They become
visible only when promotion validates the live session, policy, route,
frontier, source identities, and overlap constraints in one SQLite
transaction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from .db_bootstrap import configure_connection
from .file_registry import union_file_ids
from .tokens import count_messages_tokens, count_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedCompactionBatch:
    batch_id: str
    state: str
    source_ids: tuple[int, ...] = ()
    frontier_start_store_id: int = 0
    frontier_end_store_id: int = 0
    expected_leaf_count: int = 0
    failure_reason: str = ""


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    reason: str = ""
    batch_id: str = ""
    node_ids: tuple[int, ...] = ()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class AsyncCompactionMixin:
    """Engine mixin implementing the paper's soft/background publication path."""

    def _initialize_async_compaction(self) -> None:
        self._async_compaction_lock = threading.RLock()
        self._async_compaction_worker: threading.Thread | None = None
        self._async_compaction_closed = False
        conn = self._store.connection
        if conn is None:
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS compaction_batches (
                batch_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                state TEXT NOT NULL,
                frontier_start_store_id INTEGER NOT NULL,
                frontier_end_store_id INTEGER NOT NULL DEFAULT 0,
                fresh_tail_start_store_id INTEGER NOT NULL DEFAULT 0,
                policy_fingerprint TEXT NOT NULL,
                route_fingerprint TEXT NOT NULL,
                source_identity_hash TEXT NOT NULL DEFAULT '',
                source_ids TEXT NOT NULL DEFAULT '[]',
                expected_leaf_count INTEGER NOT NULL DEFAULT 0,
                prepared_leaf_count INTEGER NOT NULL DEFAULT 0,
                failure_reason TEXT NOT NULL DEFAULT '',
                retry_after REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_compaction_batches_scope_state
                ON compaction_batches(conversation_id, session_id, state, created_at);
            CREATE TABLE IF NOT EXISTS pending_summary_nodes (
                pending_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL REFERENCES compaction_batches(batch_id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                depth INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                source_token_count INTEGER NOT NULL,
                source_ids TEXT NOT NULL,
                source_identity_hash TEXT NOT NULL,
                earliest_at REAL,
                latest_at REAL,
                expand_hint TEXT NOT NULL DEFAULT '',
                file_ids TEXT NOT NULL DEFAULT '[]',
                UNIQUE(batch_id, ordinal)
            );
            """
        )
        pending_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(pending_summary_nodes)").fetchall()
        }
        if "file_ids" not in pending_columns:
            conn.execute(
                "ALTER TABLE pending_summary_nodes ADD COLUMN file_ids TEXT NOT NULL DEFAULT '[]'"
            )
        conn.commit()
        # A process that died while preparing cannot prove it finished writing
        # every pending leaf.  Fail it closed; ready rows remain promotable.
        now = time.time()
        conn.execute(
            """UPDATE compaction_batches
               SET state='failed', failure_reason='interrupted_preparation',
                   retry_after=?, updated_at=?
               WHERE state='preparing'""",
            (now + float(self._config.async_background_compaction_retry_backoff_seconds), now),
        )
        conn.commit()

    def _async_policy_fingerprint(self) -> str:
        c = self._config
        return _stable_hash({
            "fresh_tail_count": int(c.fresh_tail_count),
            "fresh_tail_max_tokens": int(c.fresh_tail_max_tokens),
            "leaf_chunk_tokens": int(c.leaf_chunk_tokens),
            "context_threshold": float(c.context_threshold),
            "hard_context_threshold": float(c.hard_context_threshold),
            "l2_budget_ratio": float(c.l2_budget_ratio),
            "l3_truncate_tokens": int(c.l3_truncate_tokens),
            "custom_instructions": str(c.custom_instructions),
        })

    def _async_route_fingerprint(self) -> str:
        return _stable_hash({
            "summary_model": str(self._config.summary_model),
            "fallback_models": list(self._config.summary_fallback_models),
            "provider": str(self.provider),
            "base_url": str(self.base_url),
        })

    @staticmethod
    def _row_identity(row: sqlite3.Row) -> tuple[Any, ...]:
        keys = (
            "store_id", "session_id", "role", "content", "tool_call_id",
            "tool_calls", "tool_name", "timestamp", "token_estimate",
            "conversation_id", "observed_at", "observed_at_source",
        )
        if isinstance(row, sqlite3.Row):
            return tuple(row[key] for key in keys)
        return tuple(row)

    def _source_identity_hash(self, conn: sqlite3.Connection, source_ids: Iterable[int]) -> str:
        ids = tuple(int(value) for value in source_ids)
        if not ids:
            return _stable_hash([])
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"""SELECT store_id, session_id, role, content, tool_call_id, tool_calls,
                       tool_name, timestamp, token_estimate, conversation_id,
                       observed_at, observed_at_source
                FROM messages WHERE store_id IN ({placeholders}) ORDER BY store_id""",
            ids,
        ).fetchall()
        return _stable_hash([self._row_identity(row) for row in rows])

    def _batch_from_row(self, row: sqlite3.Row) -> PreparedCompactionBatch:
        if not isinstance(row, sqlite3.Row):
            columns = (
                "batch_id", "conversation_id", "session_id", "state",
                "frontier_start_store_id", "frontier_end_store_id",
                "fresh_tail_start_store_id", "policy_fingerprint",
                "route_fingerprint", "source_identity_hash", "source_ids",
                "expected_leaf_count", "prepared_leaf_count", "failure_reason",
                "retry_after", "created_at", "updated_at",
            )
            row = dict(zip(columns, row))
        return PreparedCompactionBatch(
            batch_id=str(row["batch_id"]),
            state=str(row["state"]),
            source_ids=tuple(int(value) for value in json.loads(row["source_ids"] or "[]")),
            frontier_start_store_id=int(row["frontier_start_store_id"] or 0),
            frontier_end_store_id=int(row["frontier_end_store_id"] or 0),
            expected_leaf_count=int(row["expected_leaf_count"] or 0),
            failure_reason=str(row["failure_reason"] or ""),
        )

    def prepare_background_compaction_once(
        self,
        messages: list[dict[str, Any]],
        *,
        leave_state: str | None = None,
    ) -> PreparedCompactionBatch | None:
        if not bool(self._config.async_background_compaction_enabled):
            return None
        if not self._session_id or not self._conversation_id or not messages:
            return PreparedCompactionBatch("", "failed", failure_reason="unbound_or_empty")
        # The foreground compactor has additional dependency filtering for
        # operator-configured ignore patterns.  Do not let the asynchronous
        # path summarize data that policy intentionally excludes; hard
        # pressure still falls back to that foreground path.
        if self._compiled_ignore_message_patterns:
            return PreparedCompactionBatch(
                "",
                "failed",
                failure_reason="message_filter_requires_foreground_compaction",
            )
        conn = self._store.connection
        assert conn is not None
        with self._async_compaction_lock:
            active = conn.execute(
                """SELECT COUNT(*) FROM compaction_batches
                   WHERE conversation_id=? AND session_id=? AND state IN ('preparing','ready')""",
                (self._conversation_id, self._session_id),
            ).fetchone()[0]
            if int(active) >= max(1, int(self._config.async_background_compaction_max_batches)):
                row = conn.execute(
                    """SELECT * FROM compaction_batches
                       WHERE conversation_id=? AND session_id=? AND state='ready'
                       ORDER BY created_at LIMIT 1""",
                    (self._conversation_id, self._session_id),
                ).fetchone()
                return self._batch_from_row(row) if row is not None else None

            lifecycle = self._lifecycle.get_by_conversation(self._conversation_id)
            frontier_start = int(lifecycle.current_frontier_store_id if lifecycle else 0)
            tail_start = self._fresh_tail_start(messages)
            leading = self._leading_anchor_count(messages)
            candidates = list(messages[leading:tail_start])
            source_ids = tuple(
                sid for sid in self._get_store_ids_for_messages(candidates)
                if int(sid) > frontier_start
            )
            if not source_ids:
                return PreparedCompactionBatch("", "failed", failure_reason="no_eligible_sources")
            rows = conn.execute(
                f"SELECT store_id, timestamp FROM messages WHERE store_id IN ({','.join('?' for _ in source_ids)}) ORDER BY store_id",
                source_ids,
            ).fetchall()
            if len(rows) != len(source_ids):
                return PreparedCompactionBatch("", "failed", failure_reason="missing_sources")

            batch_id = uuid.uuid4().hex
            now = time.time()
            identity_hash = self._source_identity_hash(conn, source_ids)
            policy_hash = self._async_policy_fingerprint()
            route_hash = self._async_route_fingerprint()
            frontier_end = max(source_ids)
            fresh_tail_ids = self._get_store_ids_for_messages(messages[tail_start:])
            fresh_tail_start_id = min(fresh_tail_ids) if fresh_tail_ids else frontier_end + 1
            conn.execute(
                """INSERT INTO compaction_batches
                   (batch_id,conversation_id,session_id,state,frontier_start_store_id,
                    frontier_end_store_id,fresh_tail_start_store_id,policy_fingerprint,
                    route_fingerprint,source_identity_hash,source_ids,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (batch_id, self._conversation_id, self._session_id, "preparing",
                 frontier_start, frontier_end, fresh_tail_start_id, policy_hash,
                 route_hash, identity_hash, json.dumps(source_ids), now, now),
            )
            conn.commit()
            if leave_state == "preparing":
                row = conn.execute("SELECT * FROM compaction_batches WHERE batch_id=?", (batch_id,)).fetchone()
                return self._batch_from_row(row)

        try:
            chosen, source_tokens, summary, _level, _attempts = self._summarize_leaf_chunk_with_rescue(candidates)
            chosen_ids = tuple(self._get_store_ids_for_messages(chosen))
            if not chosen_ids:
                raise RuntimeError("background compaction resolved no durable source lineage")
            identity_hash = self._source_identity_hash(conn, chosen_ids)
            timestamps = [float(row[1]) for row in rows if row[1] is not None]
            with self._async_compaction_lock:
                conn.execute(
                    """INSERT INTO pending_summary_nodes
                       (batch_id,ordinal,depth,summary,token_count,source_token_count,
                        source_ids,source_identity_hash,earliest_at,latest_at,expand_hint,file_ids)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (batch_id, 0, 0, summary, count_tokens(summary), source_tokens,
                     json.dumps(chosen_ids), identity_hash,
                     min(timestamps) if timestamps else None,
                     max(timestamps) if timestamps else None,
                     "Expand for the original messages in this summary.",
                     json.dumps(union_file_ids(chosen))),
                )
                conn.execute(
                    """UPDATE compaction_batches SET state='ready', source_ids=?,
                       source_identity_hash=?, frontier_end_store_id=?,
                       expected_leaf_count=1, prepared_leaf_count=1, updated_at=?
                       WHERE batch_id=?""",
                    (json.dumps(chosen_ids), identity_hash, max(chosen_ids), time.time(), batch_id),
                )
                conn.commit()
                row = conn.execute("SELECT * FROM compaction_batches WHERE batch_id=?", (batch_id,)).fetchone()
                return self._batch_from_row(row)
        except Exception as exc:
            with self._async_compaction_lock:
                now = time.time()
                conn.execute(
                    """UPDATE compaction_batches SET state='failed', failure_reason=?,
                       retry_after=?, updated_at=? WHERE batch_id=?""",
                    (str(exc), now + float(self._config.async_background_compaction_retry_backoff_seconds), now, batch_id),
                )
                conn.commit()
                row = conn.execute("SELECT * FROM compaction_batches WHERE batch_id=?", (batch_id,)).fetchone()
                return self._batch_from_row(row)

    def reject_prepared_compaction(self, batch_id: str, *, reason: str) -> None:
        conn = self._store.connection
        assert conn is not None
        with self._async_compaction_lock:
            conn.execute(
                "UPDATE compaction_batches SET state='rejected', failure_reason=?, updated_at=? WHERE batch_id=? AND state IN ('preparing','ready')",
                (reason, time.time(), batch_id),
            )
            conn.commit()

    def _reject_promotion(self, conn: sqlite3.Connection, batch_id: str, reason: str) -> PromotionResult:
        conn.execute(
            "UPDATE compaction_batches SET state=?, failure_reason=?, updated_at=? WHERE batch_id=?",
            ("superseded" if reason in {"frontier_mismatch", "canonical_source_overlap"} else "rejected", reason, time.time(), batch_id),
        )
        conn.commit()
        return PromotionResult(False, reason, batch_id)

    def promote_prepared_compaction(
        self,
        batch_id: str,
        messages: list[dict[str, Any]],
    ) -> PromotionResult:
        # Use a dedicated connection: every canonical mutation below therefore
        # shares one transaction even though normal stores own separate handles.
        conn = sqlite3.connect(str(self._store.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        configure_connection(conn)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM compaction_batches WHERE batch_id=?", (batch_id,)).fetchone()
            if row is None or row["state"] != "ready":
                conn.rollback()
                return PromotionResult(False, "batch_not_ready", batch_id)
            if not bool(self._config.async_background_compaction_enabled):
                conn.rollback()
                return self._reject_promotion(conn, batch_id, "feature_disabled")
            if row["session_id"] != self._session_id or row["conversation_id"] != self._conversation_id:
                conn.rollback()
                return self._reject_promotion(conn, batch_id, "session_mismatch")
            if row["policy_fingerprint"] != self._async_policy_fingerprint():
                conn.rollback()
                return self._reject_promotion(conn, batch_id, "policy_fingerprint_mismatch")
            if row["route_fingerprint"] != self._async_route_fingerprint():
                conn.rollback()
                return self._reject_promotion(conn, batch_id, "summary_route_fingerprint_mismatch")
            lifecycle = conn.execute(
                "SELECT * FROM lcm_lifecycle_state WHERE conversation_id=?",
                (self._conversation_id,),
            ).fetchone()
            live_frontier = int(lifecycle["current_frontier_store_id"] if lifecycle else 0)
            if live_frontier != int(row["frontier_start_store_id"]):
                conn.rollback()
                return self._reject_promotion(conn, batch_id, "frontier_mismatch")
            source_ids = tuple(int(value) for value in json.loads(row["source_ids"] or "[]"))
            if not source_ids or self._source_identity_hash(conn, source_ids) != row["source_identity_hash"]:
                conn.rollback()
                return self._reject_promotion(conn, batch_id, "source_identity_mismatch")
            # Fresh-tail safety is recomputed from the live messages, not trusted
            # from persisted preparation metadata.
            live_tail_start = self._fresh_tail_start(messages)
            live_tail_ids = set(self._get_store_ids_for_messages(messages[live_tail_start:]))
            if live_tail_ids.intersection(source_ids):
                conn.rollback()
                return self._reject_promotion(conn, batch_id, "fresh_tail_overlap")
            canonical_rows = conn.execute(
                "SELECT source_ids FROM summary_nodes WHERE session_id=? AND source_type='messages'",
                (self._session_id,),
            ).fetchall()
            canonical_sources = {
                int(value)
                for canonical in canonical_rows
                for value in json.loads(canonical[0] or "[]")
            }
            if canonical_sources.intersection(source_ids):
                conn.rollback()
                return self._reject_promotion(conn, batch_id, "canonical_source_overlap")
            pending = conn.execute(
                "SELECT * FROM pending_summary_nodes WHERE batch_id=? ORDER BY ordinal",
                (batch_id,),
            ).fetchall()
            if len(pending) != int(row["expected_leaf_count"] or 0) or not pending:
                conn.rollback()
                return self._reject_promotion(conn, batch_id, "incomplete_preparation")
            node_ids: list[int] = []
            for node in pending:
                cursor = conn.execute(
                    """INSERT INTO summary_nodes
                       (session_id,depth,summary,token_count,source_token_count,
                        source_ids,source_type,created_at,earliest_at,latest_at,expand_hint,file_ids)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (self._session_id, int(node["depth"]), node["summary"],
                     int(node["token_count"]), int(node["source_token_count"]),
                     node["source_ids"], "messages", time.time(), node["earliest_at"],
                     node["latest_at"], node["expand_hint"], node["file_ids"]),
                )
                node_id = int(cursor.lastrowid)
                node_source_ids = [int(value) for value in json.loads(node["source_ids"] or "[]")]
                conn.executemany(
                    """INSERT INTO lcm_summary_message_sources
                       (summary_node_id,message_store_id,source_ordinal)
                       VALUES (?,?,?)""",
                    ((node_id, source_id, ordinal) for ordinal, source_id in enumerate(node_source_ids)),
                )
                node_ids.append(node_id)
            if getattr(self, "_async_compaction_publish_failure_hook", None) == "after_canonical_insert":
                raise RuntimeError("injected async promotion failure")
            cursor = conn.execute(
                """UPDATE lcm_lifecycle_state
                   SET current_frontier_store_id=?, updated_at=?
                   WHERE conversation_id=? AND current_session_id=?
                     AND current_frontier_store_id=?""",
                (int(row["frontier_end_store_id"]), time.time(), self._conversation_id,
                 self._session_id, int(row["frontier_start_store_id"])),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                return self._reject_promotion(conn, batch_id, "frontier_mismatch")
            conn.execute(
                "UPDATE compaction_batches SET state='promoted', updated_at=? WHERE batch_id=?",
                (time.time(), batch_id),
            )
            conn.execute(
                """UPDATE compaction_batches SET state='superseded',
                   failure_reason='newer_batch_promoted', updated_at=?
                   WHERE conversation_id=? AND session_id=? AND state='ready' AND batch_id<>?""",
                (time.time(), self._conversation_id, self._session_id, batch_id),
            )
            conn.commit()
            self._last_compacted_store_id = max(self._last_compacted_store_id, int(row["frontier_end_store_id"]))
            return PromotionResult(True, "", batch_id, tuple(node_ids))
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_async_compaction_status(self) -> dict[str, Any]:
        enabled = bool(self._config.async_background_compaction_enabled)
        status = {
            "enabled": enabled,
            "worker_enabled": bool(self._config.async_background_compaction_worker_enabled),
            "pending_batches": 0,
            "preparing_batches": 0,
            "prepared_batches": 0,
            "promoted_batches": 0,
            "rejected_batches": 0,
            "superseded_batches": 0,
            "failed_batches": 0,
        }
        conn = self._store.connection
        if conn is None:
            return status
        rows = conn.execute(
            "SELECT state, COUNT(*) FROM compaction_batches WHERE conversation_id=? GROUP BY state",
            (self.current_conversation_id or self._conversation_id or "",),
        ).fetchall()
        counts = {str(row[0]): int(row[1]) for row in rows}
        status.update({
            "pending_batches": counts.get("preparing", 0),
            "preparing_batches": counts.get("preparing", 0),
            "prepared_batches": counts.get("ready", 0),
            "promoted_batches": counts.get("promoted", 0),
            "rejected_batches": counts.get("rejected", 0),
            "superseded_batches": counts.get("superseded", 0),
            "failed_batches": counts.get("failed", 0),
        })
        return status

    def _promote_oldest_ready_compaction(
        self,
        messages: list[dict[str, Any]],
    ) -> PromotionResult | None:
        if not bool(self._config.async_background_compaction_enabled):
            return None
        conn = self._store.connection
        if conn is None:
            return None
        row = conn.execute(
            """SELECT batch_id FROM compaction_batches
               WHERE conversation_id=? AND session_id=? AND state='ready'
               ORDER BY created_at LIMIT 1""",
            (self._conversation_id, self._session_id),
        ).fetchone()
        if row is None:
            return None
        return self.promote_prepared_compaction(str(row[0]), messages)

    def _schedule_background_compaction(self, messages: list[dict[str, Any]]) -> None:
        if not (
            self._config.async_background_compaction_enabled
            and self._config.async_background_compaction_worker_enabled
            and count_messages_tokens(messages) >= self.threshold_tokens > 0
        ):
            return
        with self._async_compaction_lock:
            if self._async_compaction_closed:
                return
            if self._async_compaction_worker is not None and self._async_compaction_worker.is_alive():
                return
            snapshot = [dict(message) for message in messages]

            def run() -> None:
                try:
                    self.prepare_background_compaction_once(snapshot)
                except Exception:
                    logger.exception("LCM background compaction preparation failed")

            self._async_compaction_worker = threading.Thread(
                target=run,
                name="lcm-background-compaction",
                daemon=True,
            )
            self._async_compaction_worker.start()

    def _close_async_compaction(self) -> None:
        with self._async_compaction_lock:
            self._async_compaction_closed = True
            worker = self._async_compaction_worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
