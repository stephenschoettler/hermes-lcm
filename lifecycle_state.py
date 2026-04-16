"""Durable lifecycle/checkpoint state for hermes-lcm.

This is the smallest viable substrate for cross-turn/session lifecycle state:
- which logical conversation a session belongs to
- which session is currently bound
- which session was last finalized
- the active session frontier/checkpoint marker
- the last finalized frontier marker
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .db_bootstrap import configure_connection, run_versioned_migrations


@dataclass
class LifecycleState:
    conversation_id: str
    current_session_id: str | None
    last_finalized_session_id: str | None
    current_frontier_store_id: int
    last_finalized_frontier_store_id: int
    current_bound_at: float | None
    last_finalized_at: float | None
    last_rollover_at: float | None
    last_reset_at: float | None
    updated_at: float


class LifecycleStateStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        configure_connection(self._conn)
        self._conn.row_factory = sqlite3.Row
        run_versioned_migrations(self._conn)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def row_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS count FROM lcm_lifecycle_state").fetchone()
        return int(row["count"] if row else 0)

    def _row_to_state(self, row: sqlite3.Row | None) -> LifecycleState | None:
        if row is None:
            return None
        return LifecycleState(
            conversation_id=row["conversation_id"],
            current_session_id=row["current_session_id"],
            last_finalized_session_id=row["last_finalized_session_id"],
            current_frontier_store_id=int(row["current_frontier_store_id"] or 0),
            last_finalized_frontier_store_id=int(row["last_finalized_frontier_store_id"] or 0),
            current_bound_at=row["current_bound_at"],
            last_finalized_at=row["last_finalized_at"],
            last_rollover_at=row["last_rollover_at"],
            last_reset_at=row["last_reset_at"],
            updated_at=float(row["updated_at"] or 0.0),
        )

    def get_by_conversation(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        row = self._conn.execute(
            "SELECT * FROM lcm_lifecycle_state WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return self._row_to_state(row)

    def get_by_session(self, session_id: str | None) -> LifecycleState | None:
        if not session_id:
            return None
        row = self._conn.execute(
            """
            SELECT *
            FROM lcm_lifecycle_state
            WHERE current_session_id = ? OR last_finalized_session_id = ?
            ORDER BY CASE WHEN current_session_id = ? THEN 0 ELSE 1 END, updated_at DESC
            LIMIT 1
            """,
            (session_id, session_id, session_id),
        ).fetchone()
        return self._row_to_state(row)

    def bind_session(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> LifecycleState:
        existing = self.get_by_conversation(conversation_id) if conversation_id else self.get_by_session(session_id)
        conversation_id = conversation_id or (existing.conversation_id if existing else session_id)
        now = time.time()
        current_frontier = 0
        current_bound_at = now
        last_finalized_session_id = None
        last_finalized_frontier = 0
        last_finalized_at = None
        last_rollover_at = None
        last_reset_at = None

        if existing is not None:
            if existing.current_session_id == session_id:
                return existing
            current_frontier = (
                existing.current_frontier_store_id if existing.current_session_id == session_id else 0
            )
            current_bound_at = (
                existing.current_bound_at if existing.current_session_id == session_id else now
            )
            last_finalized_session_id = existing.last_finalized_session_id
            last_finalized_frontier = existing.last_finalized_frontier_store_id
            last_finalized_at = existing.last_finalized_at
            last_rollover_at = (
                now
                if (
                    (existing.current_session_id and existing.current_session_id != session_id)
                    or (
                        existing.current_session_id is None
                        and existing.last_finalized_session_id
                        and existing.last_finalized_session_id != session_id
                    )
                )
                else existing.last_rollover_at
            )
            last_reset_at = existing.last_reset_at

        self._conn.execute(
            """
            INSERT INTO lcm_lifecycle_state(
                conversation_id,
                current_session_id,
                last_finalized_session_id,
                current_frontier_store_id,
                last_finalized_frontier_store_id,
                current_bound_at,
                last_finalized_at,
                last_rollover_at,
                last_reset_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                current_session_id = excluded.current_session_id,
                last_finalized_session_id = excluded.last_finalized_session_id,
                current_frontier_store_id = excluded.current_frontier_store_id,
                last_finalized_frontier_store_id = excluded.last_finalized_frontier_store_id,
                current_bound_at = excluded.current_bound_at,
                last_finalized_at = excluded.last_finalized_at,
                last_rollover_at = excluded.last_rollover_at,
                last_reset_at = excluded.last_reset_at,
                updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                session_id,
                last_finalized_session_id,
                current_frontier,
                last_finalized_frontier,
                current_bound_at,
                last_finalized_at,
                last_rollover_at,
                last_reset_at,
                now,
            ),
        )
        self._conn.commit()
        state = self.get_by_conversation(conversation_id)
        assert state is not None
        return state

    def finalize_session(
        self,
        conversation_id: str | None,
        session_id: str,
        frontier_store_id: int = 0,
    ) -> LifecycleState | None:
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        current_session_id = state.current_session_id
        current_frontier = state.current_frontier_store_id
        if current_session_id == session_id:
            current_session_id = None
            current_frontier = 0
        finalized_frontier = max(
            int(frontier_store_id or 0),
            state.last_finalized_frontier_store_id,
        )
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET current_session_id = ?,
                last_finalized_session_id = ?,
                current_frontier_store_id = ?,
                last_finalized_frontier_store_id = ?,
                last_finalized_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (
                current_session_id,
                session_id,
                current_frontier,
                finalized_frontier,
                now,
                now,
                state.conversation_id,
            ),
        )
        self._conn.commit()
        return self.get_by_conversation(state.conversation_id)

    def record_rollover(
        self,
        conversation_id: str,
        *,
        old_session_id: str,
        new_session_id: str,
        finalized_frontier_store_id: int = 0,
    ) -> LifecycleState:
        state = self.get_by_conversation(conversation_id)
        if (
            state is not None
            and state.current_session_id == new_session_id
            and state.last_finalized_session_id == old_session_id
        ):
            return state

        now = time.time()
        last_finalized_frontier = max(
            int(finalized_frontier_store_id or 0),
            state.last_finalized_frontier_store_id if state else 0,
        )
        self._conn.execute(
            """
            INSERT INTO lcm_lifecycle_state(
                conversation_id,
                current_session_id,
                last_finalized_session_id,
                current_frontier_store_id,
                last_finalized_frontier_store_id,
                current_bound_at,
                last_finalized_at,
                last_rollover_at,
                last_reset_at,
                updated_at
            ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                current_session_id = excluded.current_session_id,
                last_finalized_session_id = excluded.last_finalized_session_id,
                current_frontier_store_id = 0,
                last_finalized_frontier_store_id = excluded.last_finalized_frontier_store_id,
                current_bound_at = excluded.current_bound_at,
                last_finalized_at = excluded.last_finalized_at,
                last_rollover_at = excluded.last_rollover_at,
                last_reset_at = excluded.last_reset_at,
                updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                new_session_id,
                old_session_id,
                last_finalized_frontier,
                now,
                now,
                now,
                now,
                now,
            ),
        )
        self._conn.commit()
        updated = self.get_by_conversation(conversation_id)
        assert updated is not None
        return updated

    def record_reset(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET last_reset_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (now, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)

    def advance_frontier(
        self,
        conversation_id: str | None,
        session_id: str,
        frontier_store_id: int,
    ) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None or state.current_session_id != session_id:
            return state
        frontier = max(int(frontier_store_id or 0), state.current_frontier_store_id)
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET current_frontier_store_id = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (frontier, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)
