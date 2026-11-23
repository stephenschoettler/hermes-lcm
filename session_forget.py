"""Explicit /new forgets the conversation's summary graph.

Compression carry is a different boundary: it may move summary nodes into the
next segment. A user /new is not that boundary. The Hermes transcript rotates
on its own; this drops the nodes /new was supposed to leave behind and clears
``last_finalized_session_id`` so the next compression cannot reassign them.

Raw messages stay. Explicit recall can still read them. Other conversations
are not touched, and the retain-depth used by ordinary session reset is not
changed: /new deletes every depth on the sessions this conversation owns.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _database_path() -> Path | None:
    configured = (os.environ.get("LCM_DATABASE_PATH") or "").strip()
    if configured:
        path = Path(configured)
        return path if path.is_file() else None
    home = (os.environ.get("HERMES_HOME") or "").strip()
    candidate = Path(home) / "lcm.db" if home else Path.home() / ".hermes" / "lcm.db"
    return candidate if candidate.is_file() else None


def forget_explicit_new(
    conversation_id: str | None = None,
    *,
    extra_session_ids: list[str] | None = None,
) -> dict:
    """Delete summary nodes for one conversation and clear its carry pointer.

    No-ops when the conversation is not in the lifecycle table. Session ids
    that belong to some other conversation are not deleted.
    """
    conversation_id = str(conversation_id or "").strip()
    extras = [str(item).strip() for item in (extra_session_ids or []) if str(item or "").strip()]
    if not conversation_id and not extras:
        return {"found": False, "deleted_nodes": 0}

    db_path = _database_path()
    if db_path is None:
        return {"found": False, "deleted_nodes": 0, "reason": "no_database"}

    from .dag import SummaryDAG
    from .lifecycle_state import LifecycleStateStore

    lifecycle = LifecycleStateStore(str(db_path))
    dag = SummaryDAG(str(db_path))
    try:
        state = lifecycle.get_by_conversation(conversation_id) if conversation_id else None
        if state is None:
            for session_id in extras:
                state = lifecycle.get_by_session(session_id)
                if state is not None:
                    break
        if state is None:
            return {"found": False, "deleted_nodes": 0}

        owned = {
            str(session_id)
            for session_id in (state.current_session_id, state.last_finalized_session_id, *extras)
            if session_id
        }
        # extras may name the Hermes id that was just ended. Only delete an extra
        # when it is this conversation's current or finalized session, so a hook
        # payload cannot wipe a different topic.
        owned = {
            session_id
            for session_id in owned
            if session_id in {state.current_session_id, state.last_finalized_session_id}
        }
        deleted = 0
        for session_id in owned:
            deleted += int(dag.delete_session_nodes(session_id) or 0)
        lifecycle.drop_carry_for_explicit_new(state.conversation_id)
        logger.info(
            "LCM explicit /new forgot conversation=%s sessions=%s deleted_nodes=%d",
            state.conversation_id,
            ",".join(sorted(owned)),
            deleted,
        )
        return {
            "found": True,
            "conversation_id": state.conversation_id,
            "sessions": sorted(owned),
            "deleted_nodes": deleted,
        }
    finally:
        dag.close()
        lifecycle.close()
