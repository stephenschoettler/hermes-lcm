"""Lifecycle state regression tests."""

from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore


def _lifecycle_row_count(lifecycle: LifecycleStateStore, session_id: str) -> int:
    row = lifecycle.connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM lcm_lifecycle_state
        WHERE last_finalized_session_id = ?
        """,
        (session_id,),
    ).fetchone()
    return int(row["count"])


def test_finalize_session_skips_empty_session(tmp_path):
    db_path = tmp_path / "empty-finalize.db"
    messages = MessageStore(db_path)
    lifecycle = LifecycleStateStore(db_path)
    try:
        lifecycle.bind_session("empty-session", conversation_id="conversation-a")

        lifecycle.finalize_session("conversation-a", "empty-session", frontier_store_id=0)

        assert _lifecycle_row_count(lifecycle, "empty-session") == 0
    finally:
        lifecycle.close()
        messages.close()


def test_finalize_session_keeps_non_empty_session_behavior(tmp_path):
    db_path = tmp_path / "non-empty-finalize.db"
    messages = MessageStore(db_path)
    lifecycle = LifecycleStateStore(db_path)
    try:
        lifecycle.bind_session("session-a", conversation_id="conversation-a")
        store_id = messages.append(
            "session-a",
            {"role": "user", "content": "durable context"},
            conversation_id="conversation-a",
        )

        lifecycle.finalize_session("conversation-a", "session-a", frontier_store_id=store_id)

        assert _lifecycle_row_count(lifecycle, "session-a") == 1
    finally:
        lifecycle.close()
        messages.close()
