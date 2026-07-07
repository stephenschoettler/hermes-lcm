"""Process-wide registry of active LCM runtime clones by session/lane.

Isolated from ``engine.py`` (WS5 seam): LCM clones register their own
session/conversation binding so post-turn ingest can follow the active clone
instead of the process-wide plugin singleton. The lock and the two weak
registries live here alongside the pure resolver/matcher helpers that read
them. ``engine.py`` imports the shared lock, the two registries, the removal
helper, and the public ``resolve_active_lcm_engine`` entry point; the binding
methods on ``LCMEngine`` mutate the same shared objects by reference.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any

_ACTIVE_ENGINE_REGISTRY_LOCK = threading.RLock()
_ACTIVE_ENGINES_BY_SESSION_ID = weakref.WeakValueDictionary()
_ACTIVE_ENGINES_BY_CONVERSATION_ID = weakref.WeakValueDictionary()


def _is_usable_lcm_engine(engine: Any) -> bool:
    return bool(
        engine is not None
        and getattr(engine, "name", None) == "lcm"
        and hasattr(engine, "ingest")
    )


def _engine_matches_session_binding(engine: Any, session_id: str) -> bool:
    return bool(
        _is_usable_lcm_engine(engine)
        and session_id
        and str(getattr(engine, "_session_id", "") or "") == session_id
    )


def _engine_matches_conversation_binding(engine: Any, conversation_id: str) -> bool:
    return bool(
        _is_usable_lcm_engine(engine)
        and conversation_id
        and str(getattr(engine, "_conversation_id", "") or "") == conversation_id
    )


def _remove_registry_entries_for_engine(
    engine: Any,
    *,
    keep_session_id: str = "",
    keep_conversation_id: str = "",
) -> None:
    for registered_session_id, registered_engine in list(_ACTIVE_ENGINES_BY_SESSION_ID.items()):
        if registered_engine is engine and registered_session_id != keep_session_id:
            _ACTIVE_ENGINES_BY_SESSION_ID.pop(registered_session_id, None)
    for registered_conversation_id, registered_engine in list(
        _ACTIVE_ENGINES_BY_CONVERSATION_ID.items()
    ):
        if registered_engine is engine and registered_conversation_id != keep_conversation_id:
            _ACTIVE_ENGINES_BY_CONVERSATION_ID.pop(registered_conversation_id, None)


def resolve_active_lcm_engine(session_id: str = "", conversation_id: str = "") -> Any:
    """Return the LCM runtime clone most recently bound to a session/lane.

    Newer Hermes Agent hosts pass the active per-agent context engine directly
    to ``post_llm_call`` hooks. Older hosts may only pass session/lane ids. LCM
    clones register their own session binding when ``on_session_start`` runs so
    post-turn ingest can still follow the active clone instead of rebinding the
    process-wide plugin singleton.
    """
    session_id = str(session_id or "")
    conversation_id = str(conversation_id or "")
    with _ACTIVE_ENGINE_REGISTRY_LOCK:
        if session_id:
            engine = _ACTIVE_ENGINES_BY_SESSION_ID.get(session_id)
            if _engine_matches_session_binding(engine, session_id):
                return engine
            if engine is not None:
                _ACTIVE_ENGINES_BY_SESSION_ID.pop(session_id, None)
        if conversation_id:
            engine = _ACTIVE_ENGINES_BY_CONVERSATION_ID.get(conversation_id)
            conversation_matches = _engine_matches_conversation_binding(
                engine,
                conversation_id,
            )
            session_matches = not session_id or _engine_matches_session_binding(
                engine,
                session_id,
            )
            if conversation_matches and session_matches:
                return engine
            if engine is not None and not conversation_matches:
                _ACTIVE_ENGINES_BY_CONVERSATION_ID.pop(conversation_id, None)
    return None
