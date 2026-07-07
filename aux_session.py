"""Auxiliary / subagent session tracking for the LCM engine (WS5 Seam 5).

The ``AuxiliarySessionMixin`` holds the auxiliary-session lifecycle tracking:
the per-thread auxiliary stack, active/lineage session-id sets, register /
deactivate / handoff, and the host-coupled caller-frame + state.db ancestor
detection. These methods were lifted verbatim out of ``LCMEngine`` and continue
to run bound to the engine instance (``self`` is the ``LCMEngine``), so the
auxiliary state (initialised in ``__init__``, cleared in
``_reset_profile_runtime_state``, guarded by ``_auxiliary_session_lock``) stays
owned by the engine and resolves via ``self`` — as do the callbacks
(``_end_host_fallback_compressor_for_session``, ``_state_db_path``). ``LCMEngine``
mixes this in, so no call site and no test changes.
"""

from __future__ import annotations

import inspect
import logging
import sqlite3
import threading
import weakref
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


# --- Explicit subagent lineage (WS5.7) --------------------------------------
#
# Hosts that expose the plugin hook bus fire ``subagent_start`` / ``subagent_stop``
# with the explicit ``child_session_id`` / ``parent_session_id`` linkage when a
# delegate/subagent is spawned (see ``__init__.py`` registration). We record that
# linkage here, keyed by child session id, so auxiliary-session detection can use
# the host's own explicit signal instead of walking the call stack and reading
# private agent attributes. The ``inspect.currentframe()`` frame walk below stays
# as a legacy fallback for hosts that do not fire these hooks.
_SUBAGENT_LINEAGE_LOCK = threading.RLock()
_SUBAGENT_LINEAGE_BY_SESSION_ID: "dict[str, dict[str, str]]" = {}
_SUBAGENT_LINEAGE_MAX = 4096


def record_subagent_start(payload: Dict[str, Any]) -> None:
    """Record an explicit child→parent subagent linkage from a ``subagent_start`` hook."""
    child_session_id = str((payload or {}).get("child_session_id") or "")
    if not child_session_id:
        return
    record = {
        "parent_session_id": str(payload.get("parent_session_id") or ""),
        "child_subagent_id": str(payload.get("child_subagent_id") or ""),
        "parent_subagent_id": str(payload.get("parent_subagent_id") or ""),
        "role": str(payload.get("child_role") or ""),
    }
    with _SUBAGENT_LINEAGE_LOCK:
        _SUBAGENT_LINEAGE_BY_SESSION_ID[child_session_id] = record
        # Bound memory: the child's on_session_start consumes the record long
        # before subagent_stop, so evicting the oldest entries is safe.
        overflow = len(_SUBAGENT_LINEAGE_BY_SESSION_ID) - _SUBAGENT_LINEAGE_MAX
        if overflow > 0:
            for stale in list(_SUBAGENT_LINEAGE_BY_SESSION_ID)[:overflow]:
                _SUBAGENT_LINEAGE_BY_SESSION_ID.pop(stale, None)


def record_subagent_stop(payload: Dict[str, Any]) -> None:
    """Drop an explicit subagent linkage when its ``subagent_stop`` hook fires."""
    child_session_id = str((payload or {}).get("child_session_id") or "")
    if not child_session_id:
        return
    with _SUBAGENT_LINEAGE_LOCK:
        _SUBAGENT_LINEAGE_BY_SESSION_ID.pop(child_session_id, None)


def explicit_subagent_lineage(session_id: str) -> Dict[str, str]:
    """Return the recorded explicit subagent linkage for a session id, or ``{}``."""
    key = str(session_id or "")
    if not key:
        return {}
    with _SUBAGENT_LINEAGE_LOCK:
        record = _SUBAGENT_LINEAGE_BY_SESSION_ID.get(key)
        return dict(record) if record else {}


class AuxiliarySessionMixin:
    def _thread_context_auxiliary_stack(self) -> list[str]:
        stack = getattr(self._thread_context, "auxiliary_session_stack", None)
        if stack is None:
            current = str(getattr(self._thread_context, "current_auxiliary_session_id", "") or "")
            stack = [current] if current else []
            self._thread_context.auxiliary_session_stack = stack
        return stack

    def _sync_thread_context_current_auxiliary(self) -> list[str]:
        stack = self._thread_context_auxiliary_stack()
        active_ids = self._active_auxiliary_session_ids()
        stack[:] = [session_id for session_id in stack if session_id in active_ids]
        self._thread_context.current_auxiliary_session_id = stack[-1] if stack else ""
        return stack

    def _thread_context_session_id(self) -> str:
        stack = self._sync_thread_context_current_auxiliary()
        stack_session_id = self._in_process_auxiliary_session_id_from_stack()
        if stack_session_id:
            return stack_session_id
        if stack:
            return stack[-1]
        return ""

    def _current_auxiliary_prompt_tokens(self, session_id: str) -> int:
        if not session_id:
            return 0
        caller_generation = self._in_process_auxiliary_caller_generation(session_id)
        with self._auxiliary_session_lock:
            if session_id not in self._auxiliary_session_ids:
                return 0
            active_generation = self._auxiliary_session_generations.get(session_id)
            stack = self._thread_context_auxiliary_stack()
            stack_marks_current_session = bool(
                active_generation is not None
                and caller_generation == 0
                and stack
                and stack[-1] == session_id
            )
            generation_matches = (
                caller_generation == 0
                if active_generation is None
                else caller_generation == active_generation or stack_marks_current_session
            )
            if not generation_matches:
                return 0
            return self._auxiliary_last_prompt_tokens.get(session_id, 0)

    def _thread_context_has_auxiliary_session(self, session_id: str) -> bool:
        with self._auxiliary_session_lock:
            return session_id in self._auxiliary_session_ids

    def _active_auxiliary_session_ids(self) -> set[str]:
        with self._auxiliary_session_lock:
            return set(self._auxiliary_session_ids)

    def _known_auxiliary_lineage_session_ids(self) -> set[str]:
        with self._auxiliary_session_lock:
            return set(self._auxiliary_lineage_session_ids)

    def _known_auxiliary_parent_lineage_session_ids(self) -> set[str]:
        with self._auxiliary_session_lock:
            return set(self._auxiliary_lineage_session_ids) - set(
                self._auxiliary_foreground_reused_session_ids
            )

    def _has_auxiliary_lineage_session(self, session_id: str) -> bool:
        with self._auxiliary_session_lock:
            return session_id in self._auxiliary_lineage_session_ids

    def _auxiliary_lineage_suppressed_as_foreground(self, session_id: str) -> bool:
        with self._auxiliary_session_lock:
            return session_id in self._auxiliary_foreground_reused_session_ids

    def _thread_context_stateless(self) -> bool:
        return bool(self._thread_context_session_id())

    def _register_auxiliary_session(
        self,
        session_id: str,
        *,
        preserve_foreground_reuse_marker: bool = False,
    ) -> bool:
        generation = self._in_process_auxiliary_caller_generation(session_id)
        with self._auxiliary_session_lock:
            previous_generation = self._auxiliary_session_generations.get(session_id)
            had_active_session = session_id in self._auxiliary_session_ids
            if generation and self._auxiliary_generation_is_retired(session_id, generation):
                if previous_generation is not None and previous_generation != generation:
                    return False
                if had_active_session and previous_generation is None:
                    return False
                generation = self._in_process_auxiliary_caller_generation(
                    session_id,
                    refresh_retired=True,
                )
            had_cached_usage = session_id in self._auxiliary_last_prompt_tokens
            self._auxiliary_session_ids.add(session_id)
            self._auxiliary_lineage_session_ids.add(session_id)
            if not preserve_foreground_reuse_marker:
                self._auxiliary_foreground_reused_session_ids.discard(session_id)
            if generation:
                if previous_generation != generation:
                    if previous_generation is not None:
                        self._retire_auxiliary_generation(session_id, previous_generation)
                    if self._host_fallback_session_id == session_id:
                        self._end_host_fallback_compressor_for_session(
                            session_id,
                            [],
                            current_session_bypasses=True,
                        )
                    if (
                        had_active_session
                        or had_cached_usage
                        or previous_generation is not None
                    ):
                        self._auxiliary_direct_end_guard_session_ids.add(session_id)
                    self._auxiliary_last_prompt_tokens.pop(session_id, None)
                self._auxiliary_session_generations[session_id] = generation
                self._auxiliary_handoff_parent_session_ids.pop(session_id, None)
            else:
                if previous_generation is not None and had_active_session:
                    return True
                if previous_generation is not None:
                    self._retire_auxiliary_generation(session_id, previous_generation)
                self._auxiliary_last_prompt_tokens.pop(session_id, None)
                self._auxiliary_session_generations.pop(session_id, None)
                self._auxiliary_direct_end_guard_session_ids.discard(session_id)
                self._auxiliary_handoff_parent_session_ids.pop(session_id, None)
            return True

    def _retire_auxiliary_generation(self, session_id: str, generation: int | None) -> None:
        if session_id and generation:
            self._auxiliary_retired_session_generations.setdefault(session_id, set()).add(generation)

    def _drop_auxiliary_generation_token(
        self,
        object_id: int,
        token: int,
        token_ref: weakref.ReferenceType[Any],
    ) -> None:
        with self._auxiliary_session_lock:
            existing = self._auxiliary_generation_tokens.get(object_id)
            if existing is not None and existing == (token_ref, token):
                self._auxiliary_generation_tokens.pop(object_id, None)

    def _auxiliary_generation_is_retired(self, session_id: str, generation: int) -> bool:
        return bool(generation) and generation in self._auxiliary_retired_session_generations.get(
            session_id,
            set(),
        )

    def _deactivate_auxiliary_session(self, session_id: str, *, generation: int = 0) -> bool:
        if not session_id:
            return False
        with self._auxiliary_session_lock:
            if self._auxiliary_generation_is_retired(session_id, generation):
                return False
            active_generation = self._auxiliary_session_generations.get(session_id)
            if active_generation is None:
                expected_parent = self._auxiliary_handoff_parent_session_ids.get(session_id)
                caller_parent = self._in_process_parent_session_id(
                    {},
                    session_id=session_id,
                    include_explicit=False,
                )
                if session_id in self._auxiliary_direct_end_guard_session_ids:
                    has_usage = session_id in self._auxiliary_last_prompt_tokens
                    if not generation and not has_usage:
                        return False
                    if expected_parent and caller_parent and caller_parent != expected_parent:
                        return False
            elif generation != active_generation:
                expected_parent = self._auxiliary_handoff_parent_session_ids.get(session_id)
                has_cached_usage = session_id in self._auxiliary_last_prompt_tokens
                if generation != 0 or (
                    session_id in self._auxiliary_direct_end_guard_session_ids
                    and not has_cached_usage
                ):
                    return False
            self._auxiliary_session_ids.discard(session_id)
            self._auxiliary_last_prompt_tokens.pop(session_id, None)
            self._retire_auxiliary_generation(session_id, active_generation or generation)
            self._auxiliary_session_generations.pop(session_id, None)
            self._auxiliary_direct_end_guard_session_ids.discard(session_id)
            self._auxiliary_handoff_parent_session_ids.pop(session_id, None)
            return True

    def _mark_thread_context_stateless(
        self,
        session_id: str,
        *,
        preserve_foreground_reuse_marker: bool = False,
    ) -> None:
        if not self._register_auxiliary_session(
            session_id,
            preserve_foreground_reuse_marker=preserve_foreground_reuse_marker,
        ):
            return
        stack = self._thread_context_auxiliary_stack()
        stack[:] = [existing for existing in stack if existing != session_id]
        stack.append(session_id)
        self._thread_context.current_auxiliary_session_id = session_id

    def _clear_thread_context_stateless(self, session_id: str = "") -> None:
        stack = self._thread_context_auxiliary_stack()
        if session_id:
            stack[:] = [existing for existing in stack if existing != session_id]
        else:
            stack.clear()
        self._sync_thread_context_current_auxiliary()

    def _handoff_auxiliary_session(
        self,
        old_session_id: str,
        new_session_id: str,
        *,
        preserve_old_session: bool = False,
        preserve_old_foreground_marker: bool = False,
    ) -> None:
        generation = self._in_process_auxiliary_caller_generation(new_session_id)
        if generation and self._auxiliary_generation_is_retired(new_session_id, generation):
            with self._auxiliary_session_lock:
                if new_session_id:
                    self._auxiliary_lineage_session_ids.add(new_session_id)
            return
        stack = self._thread_context_auxiliary_stack()
        handoff_from_active_thread_marker = old_session_id in stack
        with self._auxiliary_session_lock:
            old_session_was_active = old_session_id in self._auxiliary_session_ids
            if old_session_id:
                if not preserve_old_session:
                    self._auxiliary_last_prompt_tokens.pop(old_session_id, None)
                    self._retire_auxiliary_generation(
                        old_session_id,
                        self._auxiliary_session_generations.get(old_session_id),
                    )
                    self._auxiliary_session_generations.pop(old_session_id, None)
                    self._auxiliary_direct_end_guard_session_ids.discard(old_session_id)
                    self._auxiliary_handoff_parent_session_ids.pop(old_session_id, None)
                    self._auxiliary_session_ids.discard(old_session_id)
                self._auxiliary_lineage_session_ids.add(old_session_id)
            if new_session_id:
                previous_new_generation = self._auxiliary_session_generations.get(new_session_id)
                had_new_runtime_state = (
                    new_session_id in self._auxiliary_session_ids
                    or new_session_id in self._auxiliary_last_prompt_tokens
                    or previous_new_generation is not None
                    or new_session_id in self._auxiliary_direct_end_guard_session_ids
                )
                had_new_session_state = (
                    had_new_runtime_state
                    or new_session_id in self._auxiliary_lineage_session_ids
                )
                had_direct_end_guard = new_session_id in self._auxiliary_direct_end_guard_session_ids
                expected_new_parent = self._auxiliary_handoff_parent_session_ids.get(new_session_id)
                stale_generationless_parent_handoff = bool(
                    previous_new_generation is not None
                    and not generation
                    and old_session_id
                    and not old_session_was_active
                    and self._auxiliary_retired_session_generations.get(old_session_id)
                    and old_session_id != expected_new_parent
                )
                if stale_generationless_parent_handoff:
                    self._auxiliary_lineage_session_ids.add(new_session_id)
                    return
                new_generation_replaces_old = (
                    previous_new_generation is not None
                    and (
                        (bool(generation) and previous_new_generation != generation)
                        or (
                            not generation
                            and had_direct_end_guard
                            and (
                                new_session_id not in self._auxiliary_last_prompt_tokens
                                or not expected_new_parent
                                or old_session_id == expected_new_parent
                            )
                        )
                    )
                )
                if had_new_session_state and new_generation_replaces_old:
                    self._retire_auxiliary_generation(new_session_id, previous_new_generation)
                boundary_from_live_new_generation = bool(generation) and previous_new_generation == generation
                if (
                    previous_new_generation is not None
                    and not generation
                    and had_direct_end_guard
                    and new_session_id in self._auxiliary_last_prompt_tokens
                    and expected_new_parent
                    and old_session_id
                    and not old_session_was_active
                    and old_session_id != expected_new_parent
                ):
                    self._auxiliary_lineage_session_ids.add(new_session_id)
                    return
                preserve_new_foreground_marker = (
                    new_session_id in self._auxiliary_foreground_reused_session_ids
                    and not boundary_from_live_new_generation
                )
                if not preserve_new_foreground_marker:
                    self._auxiliary_last_prompt_tokens.pop(new_session_id, None)
                    self._auxiliary_direct_end_guard_session_ids.discard(new_session_id)
                    if had_new_session_state and (
                        new_generation_replaces_old
                        or had_direct_end_guard
                        or (bool(generation) and previous_new_generation is None and had_new_runtime_state)
                        or (
                            previous_new_generation is None
                            and not generation
                            and new_session_id in self._auxiliary_lineage_session_ids
                            and (
                                handoff_from_active_thread_marker
                                or (
                                    old_session_id
                                    and had_new_runtime_state
                                    and (
                                        old_session_id in self._auxiliary_lineage_session_ids
                                        or old_session_id in self._auxiliary_session_ids
                                    )
                                )
                            )
                        )
                    ):
                        self._auxiliary_direct_end_guard_session_ids.add(new_session_id)
                        if old_session_id:
                            self._auxiliary_handoff_parent_session_ids[new_session_id] = old_session_id
                    self._auxiliary_session_ids.add(new_session_id)
                    self._auxiliary_foreground_reused_session_ids.discard(new_session_id)
                    if generation:
                        if previous_new_generation != generation and self._host_fallback_session_id == new_session_id:
                            self._end_host_fallback_compressor_for_session(
                                new_session_id,
                                [],
                                current_session_bypasses=True,
                            )
                        self._auxiliary_session_generations[new_session_id] = generation
                        self._auxiliary_handoff_parent_session_ids.pop(new_session_id, None)
                    elif previous_new_generation is None or new_generation_replaces_old:
                        self._auxiliary_session_generations.pop(new_session_id, None)
                elif generation:
                    if previous_new_generation != generation:
                        if previous_new_generation is not None:
                            self._retire_auxiliary_generation(new_session_id, previous_new_generation)
                        self._auxiliary_last_prompt_tokens.pop(new_session_id, None)
                        self._auxiliary_direct_end_guard_session_ids.discard(new_session_id)
                        if self._host_fallback_session_id == new_session_id:
                            self._end_host_fallback_compressor_for_session(
                                new_session_id,
                                [],
                                current_session_bypasses=True,
                            )
                    self._auxiliary_session_ids.add(new_session_id)
                    self._auxiliary_session_generations[new_session_id] = generation
                    self._auxiliary_handoff_parent_session_ids.pop(new_session_id, None)
                else:
                    if previous_new_generation is not None:
                        self._retire_auxiliary_generation(new_session_id, previous_new_generation)
                    self._auxiliary_last_prompt_tokens.pop(new_session_id, None)
                    self._auxiliary_direct_end_guard_session_ids.discard(new_session_id)
                    if self._host_fallback_session_id == new_session_id:
                        self._end_host_fallback_compressor_for_session(
                            new_session_id,
                            [],
                            current_session_bypasses=True,
                        )
                    self._auxiliary_session_ids.add(new_session_id)
                    self._auxiliary_session_generations.pop(new_session_id, None)
                    if old_session_id:
                        self._auxiliary_handoff_parent_session_ids[new_session_id] = old_session_id
                self._auxiliary_lineage_session_ids.add(new_session_id)
        stack = self._thread_context_auxiliary_stack()
        had_thread_marker = old_session_id in stack or new_session_id in stack
        stack[:] = [
            existing
            for existing in stack
            if existing not in {old_session_id, new_session_id}
        ]
        if had_thread_marker and new_session_id:
            stack.append(new_session_id)
        self._sync_thread_context_current_auxiliary()

    def _unmark_thread_context_auxiliary_session(
        self,
        session_id: str,
        *,
        suppress_as_foreground_reuse: bool = True,
    ) -> None:
        with self._auxiliary_session_lock:
            self._auxiliary_session_ids.discard(session_id)
            if suppress_as_foreground_reuse and session_id in self._auxiliary_lineage_session_ids:
                self._auxiliary_foreground_reused_session_ids.add(session_id)
            self._auxiliary_last_prompt_tokens.pop(session_id, None)
            self._retire_auxiliary_generation(
                session_id,
                self._auxiliary_session_generations.pop(session_id, None),
            )
            self._auxiliary_direct_end_guard_session_ids.discard(session_id)
            self._auxiliary_handoff_parent_session_ids.pop(session_id, None)
        self._clear_thread_context_stateless(session_id)

    def _caller_is_auxiliary_agent_frame(self, caller_self: Any) -> bool:
        if caller_self is None:
            return False
        if getattr(caller_self, "_subagent_id", None):
            return True
        if getattr(caller_self, "_parent_subagent_id", None):
            return True
        try:
            if int(getattr(caller_self, "_delegate_depth", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
        memory_origin = str(getattr(caller_self, "_memory_write_origin", "") or "")
        memory_context = str(getattr(caller_self, "_memory_write_context", "") or "")
        if memory_origin == "background_review" or memory_context == "background_review":
            return True
        log_prefix = str(getattr(caller_self, "log_prefix", "") or "").strip()
        if log_prefix.startswith("[subagent-"):
            return True
        enabled_toolsets = getattr(caller_self, "enabled_toolsets", None)
        if enabled_toolsets is not None:
            try:
                toolsets = {str(toolset) for toolset in enabled_toolsets}
            except TypeError:
                toolsets = set()
            if toolsets and toolsets <= {"memory", "skills"}:
                return True
        if getattr(caller_self, "ephemeral_system_prompt", None) and log_prefix.startswith("[subagent-"):
            return True
        return False

    def _auxiliary_generation_token_for(self, caller_self: object, *, force_new: bool = False) -> int:
        """Return a process-local, non-reusable token for an auxiliary frame.

        ``id(obj)`` can be reused after GC, which makes retired-generation
        checks incorrectly reject later live frames. Key by id for lookup speed,
        but verify object identity with ``is`` before reusing a token. Weak refs
        avoid retaining ordinary frames; objects that cannot be weak-referenced
        are held strongly so their id cannot be recycled while the token remains
        live in this engine runtime.
        """
        with self._auxiliary_session_lock:
            object_id = id(caller_self)
            existing = self._auxiliary_generation_tokens.get(object_id)
            if existing is not None and not force_new:
                existing_ref, token = existing
                if isinstance(existing_ref, weakref.ReferenceType):
                    existing_obj = existing_ref()
                else:
                    existing_obj = existing_ref
                if existing_obj is caller_self:
                    return token

            self._auxiliary_next_generation_token += 1
            token = self._auxiliary_next_generation_token
            try:
                token_ref: Any = weakref.ref(
                    caller_self,
                    lambda ref, object_id=object_id, token=token: self._drop_auxiliary_generation_token(
                        object_id,
                        token,
                        ref,
                    ),
                )
            except TypeError:
                token_ref = caller_self
            self._auxiliary_generation_tokens[object_id] = (token_ref, token)
            return token

    def _in_process_auxiliary_caller_generation(
        self,
        session_id: str,
        *,
        refresh_retired: bool = False,
    ) -> int:
        frame = inspect.currentframe()
        try:
            frame = frame.f_back if frame is not None else None
            for _ in range(32):
                if frame is None:
                    return 0
                caller_self = frame.f_locals.get("self")
                if not self._caller_is_auxiliary_agent_frame(caller_self):
                    frame = frame.f_back
                    continue
                caller_session = str(getattr(caller_self, "session_id", "") or "")
                if not session_id or caller_session == session_id:
                    token = self._auxiliary_generation_token_for(caller_self)
                    if refresh_retired and self._auxiliary_generation_is_retired(
                        session_id,
                        token,
                    ):
                        token = self._auxiliary_generation_token_for(caller_self, force_new=True)
                    return token
                frame = frame.f_back
        finally:
            del frame
        return 0

    def _in_process_parent_session_id(
        self,
        kwargs: Dict[str, Any],
        session_id: str = "",
        include_explicit: bool = True,
    ) -> str:
        explicit = str(kwargs.get("parent_session_id") or "")
        if include_explicit and explicit:
            return explicit
        target_session_id = str(session_id or kwargs.get("session_id") or "")
        if include_explicit and target_session_id:
            explicit_parent = str(
                explicit_subagent_lineage(target_session_id).get("parent_session_id") or ""
            )
            if explicit_parent:
                return explicit_parent
        frame = inspect.currentframe()
        try:
            frame = frame.f_back if frame is not None else None
            for _ in range(32):
                if frame is None:
                    return ""
                caller_self = frame.f_locals.get("self")
                if not self._caller_is_auxiliary_agent_frame(caller_self):
                    frame = frame.f_back
                    continue
                parent = str(getattr(caller_self, "_parent_session_id", "") or "")
                caller_session = str(getattr(caller_self, "session_id", "") or "")
                if parent and caller_session and (
                    not target_session_id or caller_session == target_session_id
                ):
                    return parent
                frame = frame.f_back
        finally:
            del frame
        return ""

    def _in_process_auxiliary_session_id_from_stack(self) -> str:
        active_ids = self._active_auxiliary_session_ids()
        lineage_ids = self._known_auxiliary_lineage_session_ids()
        if not active_ids and not lineage_ids and not self._session_id:
            return ""
        frame = inspect.currentframe()
        try:
            frame = frame.f_back if frame is not None else None
            for _ in range(32):
                if frame is None:
                    return ""
                caller_self = frame.f_locals.get("self")
                if not self._caller_is_auxiliary_agent_frame(caller_self):
                    frame = frame.f_back
                    continue
                session_id = str(getattr(caller_self, "session_id", "") or "")
                parent_id = str(getattr(caller_self, "_parent_session_id", "") or "")
                if session_id and parent_id and (
                    session_id in active_ids
                    or session_id in lineage_ids
                    or parent_id == self._session_id
                    or parent_id in lineage_ids
                ):
                    return session_id
                frame = frame.f_back
        finally:
            del frame
        return ""

    def _is_live_auxiliary_child_session(
        self,
        session_id: str,
        parent_session_id: str,
        kwargs: Dict[str, Any],
    ) -> bool:
        """Return True when a same-process child agent should not rebind LCM.

        Detect Hermes auxiliary/background child sessions without treating real
        foreground branches as stateless. In-process auxiliary agent frames are
        trusted even when this engine is fresh and has no bound foreground yet.
        Explicit parent metadata by itself is not enough, because legitimate
        foreground branches can also carry parent ids before their state.db row
        is visible to the plugin.
        """
        if not session_id or session_id == parent_session_id:
            return False
        known_auxiliary_ids = self._known_auxiliary_lineage_session_ids()
        known_auxiliary_parent_ids = self._known_auxiliary_parent_lineage_session_ids()
        explicit_parent_id = str(kwargs.get("parent_session_id") or "")
        in_process_parent_id = self._in_process_parent_session_id(
            kwargs,
            session_id,
            include_explicit=False,
        )
        if in_process_parent_id:
            if not parent_session_id or in_process_parent_id == parent_session_id:
                return True
            if in_process_parent_id in known_auxiliary_ids:
                return True
        if explicit_parent_id:
            if self._thread_context_has_auxiliary_session(explicit_parent_id):
                return True
            if explicit_parent_id in known_auxiliary_parent_ids and explicit_parent_id != self._session_id:
                return True
            if (
                explicit_parent_id in known_auxiliary_ids
                and self._lcm_session_last_bypassed.get(explicit_parent_id)
            ):
                return True
            return False
        if not parent_session_id:
            return False

        path = self._state_db_path(kwargs)
        if not path.exists():
            return False
        try:
            uri = path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                row = conn.execute(
                    """
                    SELECT
                        child.parent_session_id,
                        child.started_at,
                        child.ended_at,
                        parent.id,
                        parent.ended_at
                    FROM sessions AS child
                    LEFT JOIN sessions AS parent
                        ON parent.id = child.parent_session_id
                    WHERE child.id = ?
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
            finally:
                conn.close()
        except Exception as exc:  # pragma: no cover - defensive against host DB drift
            logger.debug("LCM auxiliary child session probe failed: %s", exc)
            return False
        if not row:
            return False
        child_parent_id, child_started_at, child_ended_at, actual_parent_id, parent_ended_at = row
        if child_ended_at is not None or actual_parent_id is None:
            return False

        active_auxiliary_ids = self._active_auxiliary_session_ids()
        known_auxiliary_ids = self._known_auxiliary_lineage_session_ids()
        known_auxiliary_parent_ids = self._known_auxiliary_parent_lineage_session_ids()
        bypassed_auxiliary_ids = {
            str(auxiliary_id or "")
            for auxiliary_id in known_auxiliary_ids
            if self._lcm_session_last_bypassed.get(str(auxiliary_id or ""))
        }
        if child_parent_id in active_auxiliary_ids:
            return True
        if child_parent_id in known_auxiliary_parent_ids and child_parent_id != self._session_id:
            return True
        if child_parent_id in bypassed_auxiliary_ids:
            return True
        if child_parent_id != parent_session_id:
            return self._session_has_auxiliary_ancestor(
                str(child_parent_id or ""),
                known_auxiliary_parent_ids | active_auxiliary_ids | bypassed_auxiliary_ids,
                path,
            )
        return False

    def _session_has_auxiliary_ancestor(
        self,
        session_id: str,
        auxiliary_lineage_ids: set[str],
        state_db_path: Path,
    ) -> bool:
        if not session_id or not auxiliary_lineage_ids or not state_db_path.exists():
            return False
        visited: set[str] = set()
        current = session_id
        try:
            uri = state_db_path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                for _ in range(32):
                    if not current or current in visited:
                        return False
                    if current in auxiliary_lineage_ids:
                        return True
                    visited.add(current)
                    row = conn.execute(
                        "SELECT parent_session_id FROM sessions WHERE id = ? LIMIT 1",
                        (current,),
                    ).fetchone()
                    if not row:
                        return False
                    current = str(row[0] or "")
            finally:
                conn.close()
        except Exception as exc:  # pragma: no cover - defensive against host DB drift
            logger.debug("LCM auxiliary ancestor probe failed: %s", exc)
            return False
        return False
