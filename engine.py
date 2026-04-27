"""LCM Engine — Lossless Context Management.

Implements the ContextEngine ABC. Replaces the built-in ContextCompressor
with a DAG-based summarization system that preserves every message.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

from .config import LCMConfig
from .dag import SummaryDAG, SummaryNode
from .escalation import summarize_with_escalation
from .externalize import (
    build_transcript_gc_placeholder,
    maybe_externalize_tool_output,
    find_externalized_payload_for_message,
    reassign_externalized_payloads,
)
from .extraction import (
    extract_before_compaction,
    sanitize_pre_compaction_content,
    sanitize_pre_compaction_tool_arguments,
)
from .schemas import LCM_DESCRIBE, LCM_DOCTOR, LCM_EXPAND, LCM_EXPAND_QUERY, LCM_GREP, LCM_STATUS
from .session_patterns import (
    build_session_match_keys,
    compile_session_patterns,
    matches_session_pattern,
)
from .lifecycle_state import LifecycleStateStore
from .store import MessageStore
from .tokens import count_message_tokens, count_messages_tokens, count_tokens
from . import tools as lcm_tools

logger = logging.getLogger(__name__)


class LCMEngine(ContextEngine):
    """Lossless Context Management engine.

    Architecture:
      1. Every message is persisted verbatim in an immutable MessageStore
      2. When context pressure builds, older messages outside the fresh tail
         are summarized into leaf nodes (D0) in a SummaryDAG
      3. When enough nodes accumulate at a depth, they're condensed into
         higher-depth nodes (D1, D2, ...)
      4. The agent gets tools (lcm_grep, lcm_describe, lcm_expand) to
         search and drill into compacted history
      5. Active context = system prompt + DAG summaries + fresh tail
    """

    def __init__(self, config: LCMConfig | None = None,
                 hermes_home: str = ""):
        self._config = config or LCMConfig.from_env()
        self._hermes_home = hermes_home

        # Resolve DB path
        if self._config.database_path:
            db_path = Path(self._config.database_path)
        elif hermes_home:
            db_path = Path(hermes_home) / "lcm.db"
        else:
            db_path = Path.home() / ".hermes" / "lcm.db"

        self._store = MessageStore(db_path)
        self._dag = SummaryDAG(db_path)
        self._lifecycle = LifecycleStateStore(db_path)

        self._session_id: str = ""
        self._session_platform: str = ""
        self._conversation_id: str = ""
        self._session_match_keys: list[str] = []
        self._session_ignored = False
        self._session_stateless = False
        self._compiled_ignore_session_patterns = compile_session_patterns(
            self._config.ignore_session_patterns
        )
        self._compiled_stateless_session_patterns = compile_session_patterns(
            self._config.stateless_session_patterns
        )

        # Track which store_ids have been ingested into the DAG
        self._last_compacted_store_id: int = 0

        # Cursor: index in the current messages list up to which all
        # messages have been persisted.  After compress() shortens the
        # list, the cursor resets to len(compressed) so that only
        # genuinely new messages (appended after compaction) get ingested.
        self._ingest_cursor: int = 0

        # State required by ContextEngine ABC and run_agent.py compatibility
        self.model = ""
        self.base_url = ""
        self.api_key = ""
        self.provider = ""
        self.context_length = 0
        self.threshold_tokens = 0
        self.threshold_percent = self._config.context_threshold
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        # run_agent.py reads these for preflight checks
        self.protect_first_n = 3
        self.protect_last_n = self._config.fresh_tail_count
        # run_agent.py reads these for context probing
        self._context_probed = False
        self._context_probe_persistable = False
        self.quiet_mode = False
        self.summary_model = self._config.summary_model
        self._last_overflow_recovery_failed = False
        self._last_condensation_suppressed_reason = ""
        self._logged_filter_config = False

    @property
    def name(self) -> str:
        return "lcm"

    # -- ContextEngine required methods ------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        if self._session_ignored or self._session_stateless:
            return False
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if self._should_force_overflow_recovery(observed_tokens=tokens):
            return True
        if self.threshold_tokens <= 0:
            return False
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages):
        """Pre-flight check — also ingests messages into the store."""
        if self._session_ignored or self._session_stateless:
            return False
        if self._session_id and messages:
            try:
                self._ingest_messages(messages)
            except Exception as e:
                logger.warning("Ingest during preflight: %s", e)
        from .tokens import count_messages_tokens
        rough = count_messages_tokens(messages)
        if self._should_force_overflow_recovery(observed_tokens=rough):
            return True
        if self.threshold_tokens > 0 and rough >= self.threshold_tokens:
            return True
        self._refresh_raw_backlog_debt(messages)
        return self._should_run_deferred_maintenance(messages)

    def _working_leaf_chunk_tokens(self, raw_tokens_outside_tail: int) -> int:
        base = max(1, self._config.leaf_chunk_tokens)
        if not self._config.dynamic_leaf_chunk_enabled:
            return base
        ceiling = max(base, self._config.dynamic_leaf_chunk_max)
        working = base
        while working < ceiling and raw_tokens_outside_tail > working * 2:
            working = min(ceiling, working * 2)
        return working

    def _select_oldest_leaf_chunk(
        self,
        candidate_raw: List[Dict[str, Any]],
        working_leaf_chunk_tokens: int,
    ) -> List[Dict[str, Any]]:
        selected: list[Dict[str, Any]] = []
        used = 0
        for msg in candidate_raw:
            msg_tokens = count_message_tokens(msg)
            if used + msg_tokens > working_leaf_chunk_tokens and selected:
                break
            selected.append(msg)
            used += msg_tokens
        return selected

    def _is_retry_worthy_leaf_summary_error(self, exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        message = str(exc).lower()
        retry_markers = (
            "context length",
            "maximum context",
            "max context",
            "too many tokens",
            "token limit",
            "prompt is too long",
            "input too long",
            "request too large",
            "timed out",
            "timeout",
        )
        return any(marker in message for marker in retry_markers)

    def _next_leaf_rescue_chunk(
        self,
        current_chunk: List[Dict[str, Any]],
        current_source_tokens: int,
    ) -> List[Dict[str, Any]]:
        if len(current_chunk) <= 1:
            return []

        floor_tokens = max(1, self._config.leaf_chunk_tokens)
        shrink_targets = [
            max(floor_tokens, int(current_source_tokens * 0.75)),
            max(floor_tokens, int(current_source_tokens * 0.50)),
        ]

        for target in shrink_targets:
            if target >= current_source_tokens:
                continue
            smaller = self._select_oldest_leaf_chunk(current_chunk, target)
            if smaller and len(smaller) < len(current_chunk):
                return smaller

        return current_chunk[:-1]

    def _summarize_leaf_chunk_with_rescue(
        self,
        initial_chunk: List[Dict[str, Any]],
        focus_topic: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], int, str, int, int]:
        attempt_chunk = list(initial_chunk)
        max_attempts = 3
        attempt_number = 0

        while attempt_chunk and attempt_number < max_attempts:
            attempt_number += 1
            source_tokens = count_messages_tokens(attempt_chunk)
            serialized = self._serialize_messages(attempt_chunk)
            token_budget = max(2000, int(source_tokens * 0.20))
            token_budget = min(token_budget, 12000)

            try:
                summary_text, level = summarize_with_escalation(
                    text=serialized,
                    source_tokens=source_tokens,
                    token_budget=token_budget,
                    depth=0,
                    model=self._config.summary_model,
                    timeout=self._config.summary_timeout_ms / 1000,
                    l2_budget_ratio=self._config.l2_budget_ratio,
                    l3_truncate_tokens=self._config.l3_truncate_tokens,
                    focus_topic=focus_topic or "",
                    custom_instructions=self._config.custom_instructions,
                )
                return attempt_chunk, source_tokens, summary_text, level, attempt_number
            except Exception as exc:
                if attempt_number >= max_attempts or not self._is_retry_worthy_leaf_summary_error(exc):
                    raise
                smaller_chunk = self._next_leaf_rescue_chunk(attempt_chunk, source_tokens)
                if not smaller_chunk or len(smaller_chunk) >= len(attempt_chunk):
                    raise
                logger.warning(
                    "LCM leaf summarization retrying with smaller oldest chunk after retry-worthy failure: %s (attempt %d/%d, %d→%d messages)",
                    exc,
                    attempt_number,
                    max_attempts,
                    len(attempt_chunk),
                    len(smaller_chunk),
                )
                attempt_chunk = smaller_chunk

        raise RuntimeError("adaptive leaf rescue exhausted without a valid chunk")

    def compress(self, messages: List[Dict[str, Any]],
                 current_tokens: int = None,
                 focus_topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """Main compaction entry point.

        1. Ingest any new messages into the store
        2. Identify messages outside the fresh tail
        3. Summarize them into DAG leaf nodes
        4. Check if condensation is needed
        5. Assemble new active context: summaries + fresh tail
        """
        if not messages:
            return messages

        if self._session_ignored or self._session_stateless:
            logger.debug(
                "LCM compress bypassed for %s session %s",
                "ignored" if self._session_ignored else "stateless",
                self._session_id or "(unknown)",
            )
            return messages

        observed_prompt_tokens = current_tokens if current_tokens is not None else None
        force_overflow = self._should_force_overflow_recovery(
            observed_tokens=observed_prompt_tokens,
            messages=messages,
        )
        recovery_assembly_cap = (
            self._overflow_recovery_assembly_cap(
                observed_tokens=observed_prompt_tokens,
                messages=messages,
            )
            if force_overflow
            else None
        )

        # Step 1: Ingest new messages into the immutable store
        self._ingest_messages(messages)

        working_messages = list(messages)
        leaf_compacted_this_turn = False
        leaf_passes = 0
        deferred_maintenance_active = (
            not force_overflow and self._should_run_deferred_maintenance(working_messages)
        )
        if deferred_maintenance_active:
            self._lifecycle.record_maintenance_attempt(self._conversation_id)
        base_max_leaf_passes = 4 if self._config.dynamic_leaf_chunk_enabled else 1
        max_leaf_passes = base_max_leaf_passes
        if deferred_maintenance_active:
            max_leaf_passes = max(1, self._config.deferred_maintenance_max_passes)
        estimated_active_tokens = (
            observed_prompt_tokens
            if observed_prompt_tokens is not None and observed_prompt_tokens > 0
            else count_messages_tokens(messages)
        )

        while leaf_passes < max_leaf_passes:
            n = len(working_messages)
            fresh_tail_start = max(0, n - self._config.fresh_tail_count)

            # Protect system prompt (always index 0)
            if fresh_tail_start <= 1:
                break

            # Skip system prompt (index 0), candidate raw backlog is indices 1..fresh_tail_start
            candidate_raw = working_messages[1:fresh_tail_start]
            if not candidate_raw:
                break

            raw_tokens_outside_tail = count_messages_tokens(candidate_raw)
            if self._config.dynamic_leaf_chunk_enabled:
                working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens_outside_tail)
                if raw_tokens_outside_tail < working_leaf_chunk_tokens and not force_overflow:
                    break
                if force_overflow:
                    to_compact = candidate_raw
                else:
                    to_compact = self._select_oldest_leaf_chunk(candidate_raw, working_leaf_chunk_tokens)
            else:
                if raw_tokens_outside_tail < self._config.leaf_chunk_tokens and not force_overflow:
                    break
                to_compact = candidate_raw

            if not to_compact:
                break

            # Pre-compaction extraction: best-effort, never blocks compaction
            if self._config.extraction_enabled:
                self._run_pre_compaction_extraction(to_compact)

            compacted_chunk, source_tokens, summary_text, _level, _rescue_attempts = self._summarize_leaf_chunk_with_rescue(
                to_compact,
                focus_topic=focus_topic,
            )
            remaining_messages = working_messages[1 + len(compacted_chunk):]

            source_store_ids = self._get_store_ids_for_messages(compacted_chunk)
            earliest_at, latest_at = self._store.get_time_bounds(source_store_ids)
            summary_tokens = count_tokens(summary_text)

            node = SummaryNode(
                session_id=self._session_id,
                depth=0,
                summary=summary_text,
                token_count=summary_tokens,
                source_token_count=source_tokens,
                source_ids=source_store_ids,
                source_type="messages",
                created_at=time.time(),
                earliest_at=earliest_at,
                latest_at=latest_at,
                expand_hint=self._extract_expand_hint(summary_text),
            )
            self._dag.add_node(node)
            self._maybe_gc_compacted_tool_results(compacted_chunk, source_store_ids)
            self._last_compacted_store_id = max(source_store_ids) if source_store_ids else 0
            self._persist_frontier_marker()

            working_messages = [working_messages[0]] + remaining_messages
            leaf_compacted_this_turn = True
            leaf_passes += 1
            estimated_active_tokens = max(0, estimated_active_tokens - source_tokens + summary_tokens)

            if not self._config.dynamic_leaf_chunk_enabled:
                break

            if not force_overflow:
                if (not deferred_maintenance_active) and self.threshold_tokens > 0 and estimated_active_tokens < self.threshold_tokens:
                    break
                remaining_raw = working_messages[1:max(0, len(working_messages) - self._config.fresh_tail_count)]
                if not remaining_raw:
                    break
                remaining_raw_tokens = count_messages_tokens(remaining_raw)
                remaining_threshold = self._working_leaf_chunk_tokens(remaining_raw_tokens)
                if remaining_raw_tokens < remaining_threshold:
                    break

        if not leaf_compacted_this_turn:
            self._refresh_raw_backlog_debt(working_messages)
            if force_overflow and len(messages) >= 1:
                compressed = self._assemble_overflow_recovery_context(
                    messages[0],
                    messages[1:],
                    assembly_cap_override=recovery_assembly_cap,
                )
                return self._finalize_forced_overflow_result(
                    messages,
                    compressed,
                    assembly_cap_override=recovery_assembly_cap,
                )
            return messages

        # Step 6: Check if condensation is needed
        self._maybe_condense(
            focus_topic=focus_topic,
            leaf_compacted_this_turn=True,
            force_overflow=force_overflow,
        )

        # Step 7: Assemble new active context
        self._refresh_raw_backlog_debt(working_messages)
        compressed = self._assemble_context(
            working_messages[0],
            working_messages[1:],
            assembly_cap_override=recovery_assembly_cap,
        )
        self.compression_count += 1
        if recovery_assembly_cap is None:
            self._last_overflow_recovery_failed = False
        else:
            self._last_overflow_recovery_failed = count_messages_tokens(compressed) > recovery_assembly_cap
            if self._last_overflow_recovery_failed:
                logger.warning(
                    "LCM overflow recovery could not get under cap=%d after compaction; returning best-effort context (%d tokens)",
                    recovery_assembly_cap,
                    count_messages_tokens(compressed),
                )
        # Reset cursor to the length of the compressed context so that
        # only messages appended *after* this point get ingested next time.
        self._ingest_cursor = len(compressed)

        logger.info(
            "LCM compaction #%d: %d messages → %d (%d leaf pass%s, %d→%d tokens, %d DAG nodes%s)",
            self.compression_count,
            len(messages),
            len(compressed),
            leaf_passes,
            "es" if leaf_passes != 1 else "",
            count_messages_tokens(messages),
            count_messages_tokens(compressed),
            len(self._dag.get_session_nodes(self._session_id)),
            ", forced overflow recovery" if force_overflow else "",
        )

        return compressed

    # -- ContextEngine optional methods ------------------------------------

    def _bind_lifecycle_state(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> None:
        state = self._lifecycle.bind_session(session_id, conversation_id=conversation_id)
        self._conversation_id = state.conversation_id
        self._last_compacted_store_id = state.current_frontier_store_id

    def _persist_frontier_marker(self) -> None:
        if not self._session_id or not self._conversation_id:
            return
        self._lifecycle.advance_frontier(
            self._conversation_id,
            self._session_id,
            self._last_compacted_store_id,
        )

    def _raw_backlog_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n = len(messages)
        fresh_tail_start = max(0, n - self._config.fresh_tail_count)
        if fresh_tail_start <= 1:
            return []
        return messages[1:fresh_tail_start]

    def _raw_backlog_tokens(self, messages: List[Dict[str, Any]]) -> int:
        backlog = self._raw_backlog_messages(messages)
        if not backlog:
            return 0
        return count_messages_tokens(backlog)

    def _raw_backlog_threshold(self, raw_tokens: int) -> int:
        if self._config.dynamic_leaf_chunk_enabled:
            return self._working_leaf_chunk_tokens(raw_tokens)
        return max(1, self._config.leaf_chunk_tokens)

    def _has_raw_backlog_debt(self) -> bool:
        if not self._config.deferred_maintenance_enabled or not self._conversation_id:
            return False
        state = self._lifecycle.get_by_conversation(self._conversation_id)
        return bool(state and state.debt_kind == "raw_backlog" and state.debt_size_estimate > 0)

    def _should_run_deferred_maintenance(self, messages: List[Dict[str, Any]]) -> bool:
        if not self._has_raw_backlog_debt():
            return False
        raw_tokens = self._raw_backlog_tokens(messages)
        if raw_tokens <= 0:
            return False
        return raw_tokens >= self._raw_backlog_threshold(raw_tokens)

    def _refresh_raw_backlog_debt(self, messages: List[Dict[str, Any]]) -> None:
        if not self._config.deferred_maintenance_enabled or not self._conversation_id:
            return
        raw_tokens = self._raw_backlog_tokens(messages)
        threshold = self._raw_backlog_threshold(raw_tokens) if raw_tokens > 0 else 0
        if raw_tokens > 0 and raw_tokens >= threshold:
            self._lifecycle.record_debt(
                self._conversation_id,
                kind="raw_backlog",
                size_estimate=raw_tokens,
            )
            return
        if self._has_raw_backlog_debt():
            self._lifecycle.clear_debt(self._conversation_id)

    def _reset_session_scoped_runtime_state(self) -> None:
        self.compression_count = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self._last_compacted_store_id = 0
        self._ingest_cursor = 0
        self._context_probed = False
        self._context_probe_persistable = False
        self._last_overflow_recovery_failed = False
        self._last_condensation_suppressed_reason = ""

    def _apply_session_start_metadata(self, session_id: str, kwargs: Dict[str, Any]) -> None:
        self._session_id = session_id
        self._session_platform = str(kwargs.get("platform") or "")
        self._refresh_session_filters()
        if "hermes_home" in kwargs:
            self._hermes_home = kwargs["hermes_home"]
        # Pick up context_length from kwargs if provided
        if "context_length" in kwargs:
            self.context_length = kwargs["context_length"]
            self.threshold_tokens = int(
                self.context_length * self._config.context_threshold
            )

    def _continue_compression_boundary(
        self,
        session_id: str,
        old_session_id: str,
        kwargs: Dict[str, Any],
    ) -> None:
        previous_session_id = self._session_id
        prior_state = self._lifecycle.get_by_session(old_session_id)
        conversation_id = (
            kwargs.get("conversation_id")
            or self._conversation_id
            or (prior_state.conversation_id if prior_state else None)
            or old_session_id
            or session_id
        )
        frontier = max(
            int(self._last_compacted_store_id or 0),
            int(prior_state.current_frontier_store_id if prior_state else 0),
        )
        can_reassign = bool(
            old_session_id
            and session_id
            and old_session_id != session_id
            and (not previous_session_id or previous_session_id == old_session_id)
        )

        if can_reassign:
            self._lifecycle.finalize_session(
                conversation_id,
                old_session_id,
                frontier_store_id=frontier,
            )
            moved_messages = self._store.reassign_session_messages(old_session_id, session_id)
            moved_nodes = self._dag.reassign_session_nodes(old_session_id, session_id)
            moved_payloads = reassign_externalized_payloads(
                old_session_id,
                session_id,
                config=self._config,
                hermes_home=self._hermes_home,
            )
            logger.debug(
                "LCM compression boundary continued %s -> %s: moved %d messages, %d DAG nodes, %d externalized payloads",
                old_session_id,
                session_id,
                moved_messages,
                moved_nodes,
                moved_payloads,
            )
        elif old_session_id:
            logger.warning(
                "LCM compression boundary skipped carry-over: old_session_id=%s does not match bound session=%s",
                old_session_id,
                previous_session_id,
            )
            self._reset_session_scoped_runtime_state()
            self._apply_session_start_metadata(session_id, kwargs)
            self._bind_lifecycle_state(
                session_id,
                conversation_id=kwargs.get("conversation_id"),
            )
            self._log_session_filter_diagnostics()
            return

        self._apply_session_start_metadata(session_id, kwargs)
        self._bind_lifecycle_state(session_id, conversation_id=conversation_id)
        if frontier > 0:
            state = self._lifecycle.advance_frontier(
                self._conversation_id,
                session_id,
                frontier,
            )
            if state is not None:
                self._last_compacted_store_id = state.current_frontier_store_id
        self._log_session_filter_diagnostics()

    def on_session_start(self, session_id: str, **kwargs) -> None:
        boundary_reason = str(kwargs.get("boundary_reason") or "")
        old_session_id = str(kwargs.get("old_session_id") or "")
        if boundary_reason == "compression" and old_session_id and old_session_id != session_id:
            self._continue_compression_boundary(session_id, old_session_id, kwargs)
            return

        previous_session_id = self._session_id
        if previous_session_id and previous_session_id != session_id:
            self._reset_session_scoped_runtime_state()
        else:
            self._ingest_cursor = 0
            self._last_compacted_store_id = 0
            self._last_overflow_recovery_failed = False
            self._last_condensation_suppressed_reason = ""
        self._apply_session_start_metadata(session_id, kwargs)
        self._bind_lifecycle_state(
            session_id,
            conversation_id=kwargs.get("conversation_id"),
        )
        self._log_session_filter_diagnostics()

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        # Ensure all messages are persisted
        self._ingest_messages(messages)
        self._lifecycle.finalize_session(
            self._conversation_id,
            session_id,
            frontier_store_id=self._last_compacted_store_id,
        )

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._lifecycle.record_reset(self._conversation_id)
        self._last_compacted_store_id = 0
        self._ingest_cursor = 0
        self._context_probed = False
        self._context_probe_persistable = False
        self._last_overflow_recovery_failed = False
        self._last_condensation_suppressed_reason = ""

        # Retain DAG nodes across sessions based on config.
        #   -1  → keep all nodes
        #    0  → delete everything
        #    N  → keep nodes at depth >= N (e.g. 2 keeps d2+)
        retain = self._config.new_session_retain_depth
        if self._session_id and retain != -1:
            if retain == 0:
                self._dag.delete_session_nodes(self._session_id)
            else:
                self._dag.delete_below_depth(self._session_id, retain)

    def carry_over_new_session_context(self, old_session_id: str, new_session_id: str) -> int:
        """Move retained summaries from the old session into the new one.

        This reassigns session ownership for retained summary nodes, but it does
        not rewrite the nodes' descendant raw-message lineage. Retrieval under
        ``session_scope='current'`` may therefore include a carried-over node in
        the new session, while ``source`` filtering still evaluates against the
        node's original descendant message sources.
        """
        if not old_session_id or not new_session_id or old_session_id == new_session_id:
            return 0
        if self._session_ignored and new_session_id == self._session_id:
            logger.debug(
                "LCM carry-over skipped for ignored session %s",
                new_session_id,
            )
            return 0
        return self._dag.reassign_session_nodes(old_session_id, new_session_id)

    def rollover_session(
        self,
        old_session_id: str,
        new_session_id: str,
        previous_messages: List[Dict[str, Any]] | None = None,
        carry_over_context: bool = True,
        **kwargs,
    ) -> int:
        """Complete a Hermes-style `/new` rollover for this engine.

        This is a small helper for host/runtime integrations that need the
        correct lifecycle ordering in one call:
        1. flush old-session messages into the store
        2. prune/reset retained DAG state on the old session
        3. bind the engine to the new session
        4. optionally move retained summaries into the new session
        """
        previous_messages = previous_messages or []
        conversation_id = self._conversation_id or old_session_id or new_session_id
        bound_session_id = self._session_id
        can_carry_over = bool(
            old_session_id and bound_session_id and old_session_id == bound_session_id
        )

        if old_session_id and can_carry_over:
            self.on_session_end(old_session_id, previous_messages)
            self.on_session_reset()
        elif old_session_id and not carry_over_context:
            logger.warning(
                "LCM rollover skipped old-session finalization: old_session_id=%s does not match bound session=%s",
                old_session_id,
                bound_session_id,
            )
        elif old_session_id and not can_carry_over:
            logger.warning(
                "LCM carry-over skipped: old_session_id=%s does not match bound session=%s",
                old_session_id,
                bound_session_id,
            )

        self.on_session_start(new_session_id, conversation_id=conversation_id, **kwargs)

        if not carry_over_context:
            return 0
        if old_session_id and not can_carry_over:
            return 0
        return self.carry_over_new_session_context(old_session_id, new_session_id)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [LCM_GREP, LCM_DESCRIBE, LCM_EXPAND, LCM_EXPAND_QUERY, LCM_STATUS, LCM_DOCTOR]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        # Ingest live messages if passed (enables current-turn search)
        messages = kwargs.get("messages")

        if messages and self._session_id and not (self._session_ignored or self._session_stateless):
            try:
                self._ingest_messages(messages)
            except Exception as e:
                logger.warning("Ingest during tool call failed: %s", e)

        handlers = {
            "lcm_grep": lcm_tools.lcm_grep,
            "lcm_describe": lcm_tools.lcm_describe,
            "lcm_expand": lcm_tools.lcm_expand,
            "lcm_expand_query": lcm_tools.lcm_expand_query,
            "lcm_status": lcm_tools.lcm_status,
            "lcm_doctor": lcm_tools.lcm_doctor,
        }
        handler = handlers.get(name)
        if handler:
            return handler(args, engine=self)
        return json.dumps({"error": f"Unknown LCM tool: {name}"})

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "compression_count": self.compression_count,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_completion_tokens": self.last_completion_tokens,
            "last_total_tokens": self.last_total_tokens,
            "context_length": self.context_length,
            "threshold_tokens": self.threshold_tokens,
        })
        lifecycle_state = self._lifecycle.get_by_conversation(self._conversation_id)
        status["engine"] = "lcm"
        try:
            status["source_lineage"] = self._store.get_source_stats(self._session_id or None)
        except Exception as exc:  # pragma: no cover - defensive
            status["source_lineage"] = {"error": str(exc)}
        if self._session_id:
            status["store_messages"] = self._store.get_session_count(self._session_id)
            status["dag_nodes"] = len(self._dag.get_session_nodes(self._session_id))
            status["session_platform"] = self._session_platform
            status["session_ignored"] = self._session_ignored
            status["session_stateless"] = self._session_stateless
            status["ignore_session_patterns"] = list(self._config.ignore_session_patterns)
            status["stateless_session_patterns"] = list(self._config.stateless_session_patterns)
            status["ignore_session_patterns_source"] = self._config.ignore_session_patterns_source
            status["stateless_session_patterns_source"] = self._config.stateless_session_patterns_source
            status["overflow_recovery_failed"] = self._last_overflow_recovery_failed
            status["condensation_suppressed_reason"] = self._last_condensation_suppressed_reason
            status["conversation_id"] = self._conversation_id
            if lifecycle_state is not None:
                status["lifecycle"] = {
                    "conversation_id": lifecycle_state.conversation_id,
                    "current_session_id": lifecycle_state.current_session_id,
                    "last_finalized_session_id": lifecycle_state.last_finalized_session_id,
                    "current_frontier_store_id": lifecycle_state.current_frontier_store_id,
                    "last_finalized_frontier_store_id": lifecycle_state.last_finalized_frontier_store_id,
                    "debt_kind": lifecycle_state.debt_kind,
                    "debt_size_estimate": lifecycle_state.debt_size_estimate,
                    "current_bound_at": lifecycle_state.current_bound_at,
                    "last_finalized_at": lifecycle_state.last_finalized_at,
                    "debt_updated_at": lifecycle_state.debt_updated_at,
                    "last_maintenance_attempt_at": lifecycle_state.last_maintenance_attempt_at,
                    "last_rollover_at": lifecycle_state.last_rollover_at,
                    "last_reset_at": lifecycle_state.last_reset_at,
                    "updated_at": lifecycle_state.updated_at,
                }
        return status

    def update_model(self, model: str, context_length: int,
                     base_url: str = "", api_key: str = "",
                     provider: str = "",
                     api_mode: str = "") -> None:
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self._config.context_threshold)

    def _refresh_session_filters(self) -> None:
        self._session_match_keys = build_session_match_keys(
            self._session_id,
            platform=self._session_platform,
        )
        self._session_ignored = matches_session_pattern(
            self._session_match_keys,
            self._compiled_ignore_session_patterns,
        )
        self._session_stateless = (
            not self._session_ignored
            and matches_session_pattern(
                self._session_match_keys,
                self._compiled_stateless_session_patterns,
            )
        )

    def _log_session_filter_diagnostics(self) -> None:
        if not self._logged_filter_config:
            if self._config.ignore_session_patterns:
                logger.info(
                    "LCM ignore_session_patterns from %s: %s",
                    self._config.ignore_session_patterns_source,
                    ", ".join(self._config.ignore_session_patterns),
                )
            if self._config.stateless_session_patterns:
                logger.info(
                    "LCM stateless_session_patterns from %s: %s",
                    self._config.stateless_session_patterns_source,
                    ", ".join(self._config.stateless_session_patterns),
                )
            self._logged_filter_config = True
        if self._session_ignored:
            logger.info(
                "LCM session %s matched ignore_session_patterns via %s — skipping writes and compaction",
                self._session_id,
                ", ".join(self._session_match_keys),
            )
        elif self._session_stateless:
            logger.info(
                "LCM session %s matched stateless_session_patterns via %s — read-only mode (no LCM writes)",
                self._session_id,
                ", ".join(self._session_match_keys),
            )

    # -- Internal: message ingestion ---------------------------------------

    def _ingest_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Persist new messages to the store.

        Uses a cursor to track which portion of the current messages list
        has already been persisted.  After compress() shortens the list,
        the cursor is reset to len(compressed), so only messages appended
        after compaction are ingested — regardless of how the store count
        compares to the current list length.
        """
        if not self._session_id:
            logger.debug("Ingest skipped: no session_id")
            return

        if self._session_ignored or self._session_stateless:
            logger.debug(
                "Ingest skipped for %s session %s",
                "ignored" if self._session_ignored else "stateless",
                self._session_id,
            )
            return

        n = len(messages)
        cursor = self._ingest_cursor
        logger.debug(
            "Ingest: session=%s cursor=%d incoming=%d",
            self._session_id, cursor, n,
        )

        new_messages = messages[cursor:] if cursor < n else []

        if not new_messages:
            return

        estimates = [count_message_tokens(m) for m in new_messages]
        self._store.append_batch(
            self._session_id,
            new_messages,
            estimates,
            source=self._session_platform,
        )
        self._ingest_cursor = n
        logger.debug("Ingested %d messages into LCM store", len(new_messages))

    def _get_store_ids_for_messages(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Map current raw messages back to store_ids in stable store order.

        Matching starts strictly after ``_last_compacted_store_id`` so repeated
        content from older already-compacted history cannot hijack the mapping.
        Synthetic summary messages simply fail to match and are skipped.
        """
        candidates = [
            stored for stored in self._store.get_session_messages(self._session_id)
            if stored["store_id"] > self._last_compacted_store_id
        ]

        ids: list[int] = []
        store_idx = 0
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content") or ""
            probe_idx = store_idx
            while probe_idx < len(candidates):
                stored = candidates[probe_idx]
                if stored.get("role", "") == role and (stored.get("content") or "") == content:
                    ids.append(stored["store_id"])
                    store_idx = probe_idx + 1
                    break
                probe_idx += 1

        return ids

    # -- Internal: summarization -------------------------------------------

    def _run_pre_compaction_extraction(self, messages: List[Dict[str, Any]]) -> None:
        """Best-effort extraction of decisions before compaction."""
        try:
            serialized = self._serialize_messages(messages)
            output_path = self._config.extraction_output_path
            if not output_path:
                base = self._hermes_home or os.path.expanduser("~/.hermes")
                output_path = os.path.join(base, "lcm-extractions")
            extraction_model = self._config.extraction_model or self._config.summary_model
            extract_before_compaction(
                serialized_messages=serialized,
                output_path=output_path,
                session_id=self._session_id or "",
                model=extraction_model,
                timeout=self._config.summary_timeout_ms / 1000,
            )
        except Exception as e:
            logger.warning("Pre-compaction extraction failed (non-blocking): %s", e)

    def _maybe_gc_compacted_tool_results(
        self,
        compacted_chunk: List[Dict[str, Any]],
        source_store_ids: List[int],
    ) -> None:
        if not getattr(self._config, "large_output_transcript_gc_enabled", False):
            return
        if not compacted_chunk or not source_store_ids:
            return

        stored_by_id = self._store.get_batch(source_store_ids)
        for store_id in source_store_ids:
            stored = stored_by_id.get(store_id)
            if not stored or stored.get("session_id") != self._session_id:
                continue
            if stored.get("role") != "tool":
                continue
            content = stored.get("content", "") or ""
            tool_call_id = stored.get("tool_call_id", "") or ""
            if not content:
                continue

            lookup_candidates = []
            sanitized_content = sanitize_pre_compaction_content(content)
            if sanitized_content and sanitized_content != content:
                lookup_candidates.append(sanitized_content)
            lookup_candidates.append(content)

            externalized = None
            for candidate in lookup_candidates:
                externalized = find_externalized_payload_for_message(
                    candidate,
                    tool_call_id=tool_call_id,
                    session_id=self._session_id,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                if externalized is not None:
                    break
            if externalized is None:
                continue

            placeholder = build_transcript_gc_placeholder(externalized)
            self._store.gc_externalized_tool_result(store_id, placeholder)

    def _serialize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Serialize messages into labeled text for the summarizer."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            content = sanitize_pre_compaction_content(content)

            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                externalized = maybe_externalize_tool_output(
                    content,
                    tool_call_id=tool_id,
                    session_id=self._session_id,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                if externalized:
                    content = externalized["placeholder"]
                elif len(content) > 3000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            if role == "assistant":
                if len(content) > 3000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = fn.get("arguments", "")
                            args = sanitize_pre_compaction_tool_arguments(args)
                            if len(args) > 500:
                                args = args[:400] + "..."
                            tc_parts.append(f"  {name}({args})")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            if len(content) > 3000:
                content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
            parts.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(parts)

    # -- Internal: condensation --------------------------------------------

    def _should_allow_follow_on_condensation(
        self,
        *,
        uncondensed_count: int,
        leaf_compacted_this_turn: bool,
        force_overflow: bool,
    ) -> tuple[bool, str]:
        if not leaf_compacted_this_turn:
            return True, ""
        if not self._config.cache_friendly_condensation_enabled:
            return True, ""
        if force_overflow:
            return True, ""

        fanin = max(1, self._config.condensation_fanin)
        debt_threshold = fanin * max(1, self._config.cache_friendly_min_debt_groups)
        if uncondensed_count >= debt_threshold:
            return True, ""
        if uncondensed_count == fanin:
            return False, "cache_friendly_single_group"
        return False, "cache_friendly_low_debt"

    def _maybe_condense(
        self,
        focus_topic: Optional[str] = None,
        *,
        leaf_compacted_this_turn: bool = False,
        force_overflow: bool = False,
    ) -> None:
        """Check if any depth level has enough nodes for condensation."""
        self._last_condensation_suppressed_reason = ""

        max_depth = self._config.incremental_max_depth
        if max_depth == 0:
            return  # condensation disabled

        # When max_depth is -1 (unlimited), derive the upper bound from
        # the deepest existing node + 1, so condensation can always
        # create the next depth level.
        if max_depth < 0:
            all_nodes = self._dag.get_session_nodes(self._session_id)
            upper = (max(n.depth for n in all_nodes) + 1) if all_nodes else 1
        else:
            upper = max_depth

        condensed_any = False
        suppression_reason = ""

        for depth in range(upper):
            uncondensed = self._dag.get_uncondensed_at_depth(
                self._session_id, depth
            )
            if len(uncondensed) < self._config.condensation_fanin:
                continue

            allow_condense, reason = self._should_allow_follow_on_condensation(
                uncondensed_count=len(uncondensed),
                leaf_compacted_this_turn=leaf_compacted_this_turn,
                force_overflow=force_overflow,
            )
            if not allow_condense:
                suppression_reason = reason or suppression_reason
                continue

            # Take the first fanin nodes and condense
            to_condense = uncondensed[:self._config.condensation_fanin]
            combined_text = "\n\n---\n\n".join(n.summary for n in to_condense)
            source_tokens = sum(n.token_count for n in to_condense)
            token_budget = max(1000, int(source_tokens * 0.40))

            summary_text, level = summarize_with_escalation(
                text=combined_text,
                source_tokens=source_tokens,
                token_budget=token_budget,
                depth=depth + 1,
                model=self._config.summary_model,
                timeout=self._config.summary_timeout_ms / 1000,
                l2_budget_ratio=self._config.l2_budget_ratio,
                l3_truncate_tokens=self._config.l3_truncate_tokens,
                focus_topic=focus_topic or "",
                custom_instructions=self._config.custom_instructions,
            )

            earliest_at, latest_at = self._dag.get_source_time_window([n.node_id for n in to_condense])
            node = SummaryNode(
                session_id=self._session_id,
                depth=depth + 1,
                summary=summary_text,
                token_count=count_tokens(summary_text),
                source_token_count=source_tokens,
                source_ids=[n.node_id for n in to_condense],
                source_type="nodes",
                created_at=time.time(),
                earliest_at=earliest_at,
                latest_at=latest_at,
                expand_hint=self._extract_expand_hint(summary_text),
            )
            self._dag.add_node(node)
            condensed_any = True

            logger.info(
                "LCM condensation: d%d × %d → d%d (L%d, %d→%d tokens)",
                depth, len(to_condense), depth + 1, level,
                source_tokens, count_tokens(summary_text),
            )

            if leaf_compacted_this_turn and self._config.cache_friendly_condensation_enabled:
                break

        if not condensed_any and leaf_compacted_this_turn and self._config.cache_friendly_condensation_enabled:
            self._last_condensation_suppressed_reason = suppression_reason

    # -- Internal: context assembly ----------------------------------------

    def _assemble_context(
        self,
        system_msg: Dict[str, Any],
        tail_messages: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
        include_lcm_note: bool = True,
    ) -> List[Dict[str, Any]]:
        """Build the active context from DAG summaries + fresh tail.

        Structure:
          [system prompt (with LCM note)]
          [highest-depth summary nodes first, then lower]
          [fresh tail messages]
        """
        result = []

        # System prompt with LCM annotation
        sys_msg = system_msg.copy()
        if self.compression_count == 0 and include_lcm_note:
            sys_content = sys_msg.get("content", "")
            sys_msg["content"] = (
                sys_content
                + "\n\n[Note: This conversation uses Lossless Context Management (LCM). "
                "Earlier turns have been compacted into hierarchical summaries below. "
                "Use lcm_grep to search history, lcm_describe to inspect the DAG, "
                "and lcm_expand to recover original details from any summary.]"
            )
        result.append(sys_msg)

        assembly_cap = (
            assembly_cap_override
            if assembly_cap_override is not None
            else self._effective_assembly_token_cap()
        )

        tail_selected = tail_messages
        summary_budget = None
        if assembly_cap is not None:
            used = count_message_tokens(sys_msg)
            kept_tail_reversed: list[Dict[str, Any]] = []
            tail_token_total = 0
            for msg in reversed(tail_messages):
                msg_tokens = count_message_tokens(msg)
                if used + tail_token_total + msg_tokens > assembly_cap:
                    break
                kept_tail_reversed.append(msg)
                tail_token_total += msg_tokens
            tail_selected = list(reversed(kept_tail_reversed))
            summary_budget = max(0, assembly_cap - used - tail_token_total)

        # Collect DAG summaries — highest depth first for context hierarchy
        all_nodes = self._dag.get_session_nodes(self._session_id)
        if all_nodes:
            # Group by depth, take the most recent uncondensed at each level
            # For active context, we want the highest-level summaries
            # that haven't been condensed into even higher levels
            depths = sorted(set(n.depth for n in all_nodes), reverse=True)
            summary_parts = []
            for d in depths:
                uncondensed = self._dag.get_uncondensed_at_depth(self._session_id, d)
                for node in uncondensed:
                    depth_label = {
                        0: "Recent",
                        1: "Session Arc",
                        2: "Durable",
                    }.get(d, f"Depth-{d}")
                    summary_parts.append(
                        f"[{depth_label} Summary (d{d}, node {node.node_id})]\n"
                        f"{node.summary}\n"
                        f"[Expand for details: {node.expand_hint}]"
                    )

            if summary_parts:
                # Choose role to avoid consecutive same-role
                last_role = result[-1].get("role", "system")
                summary_role = "assistant" if last_role != "assistant" else "user"
                selected_parts = summary_parts
                if summary_budget is not None:
                    selected_parts = []
                    for part in summary_parts:
                        candidate = "\n\n---\n\n".join(selected_parts + [part])
                        candidate_msg = {"role": summary_role, "content": candidate}
                        if count_message_tokens(candidate_msg) > summary_budget:
                            break
                        selected_parts.append(part)
                if selected_parts:
                    combined = "\n\n---\n\n".join(selected_parts)
                    result.append({"role": summary_role, "content": combined})

        # Fresh tail
        result.extend(tail_selected)

        return result

    def _finalize_forced_overflow_result(
        self,
        original_messages: List[Dict[str, Any]],
        compressed: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if compressed != original_messages:
            self._ingest_cursor = len(compressed)
            logger.info(
                "LCM assembly guardrail recovery: %d messages → %d (no new summary node)",
                len(original_messages),
                len(compressed),
            )

        effective_cap = (
            assembly_cap_override
            if assembly_cap_override is not None
            else self._effective_assembly_token_cap()
        )
        if effective_cap is None:
            self._last_overflow_recovery_failed = False
        else:
            self._last_overflow_recovery_failed = count_messages_tokens(compressed) > effective_cap
            if self._last_overflow_recovery_failed:
                logger.warning(
                    "LCM overflow recovery could not get under cap=%d; returning best-effort context (%d tokens)",
                    effective_cap,
                    count_messages_tokens(compressed),
                )
        return compressed

    def _should_force_overflow_recovery(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is None:
            return False

        tokens = self._overflow_recovery_signal_tokens(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        if tokens is None:
            return False
        return tokens >= assembly_cap

    def _overflow_recovery_signal_tokens(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        candidates: list[int] = []
        if observed_tokens is not None and observed_tokens > 0:
            candidates.append(observed_tokens)
        if messages is not None:
            candidates.append(count_messages_tokens(messages))
        if not candidates:
            return None
        return max(candidates)

    def _overflow_recovery_assembly_cap(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is None:
            return None
        if messages is None or observed_tokens is None or observed_tokens <= 0:
            return assembly_cap

        message_tokens = count_messages_tokens(messages)
        overhead_tokens = max(0, observed_tokens - message_tokens)
        return max(1, assembly_cap - overhead_tokens)

    def _effective_assembly_token_cap(self) -> Optional[int]:
        """Return the active assembly cap, if any.

        Two knobs can constrain the assembled active context:
        - max_assembly_tokens: explicit hard cap
        - reserve_tokens_floor: keep headroom inside context_length
        """
        caps: list[int] = []

        if self._config.max_assembly_tokens > 0:
            caps.append(self._config.max_assembly_tokens)

        if self.context_length > 0 and self._config.reserve_tokens_floor > 0:
            reserve_cap = self.context_length - self._config.reserve_tokens_floor
            if reserve_cap > 0:
                caps.append(reserve_cap)
            else:
                logger.warning(
                    "LCM reserve_tokens_floor=%d disables reserve-based assembly cap because context_length=%d",
                    self._config.reserve_tokens_floor,
                    self.context_length,
                )

        if not caps:
            return None

        return max(1, min(caps))

    # -- Internal: helpers -------------------------------------------------

    def _assemble_overflow_recovery_context(
        self,
        system_msg: Dict[str, Any],
        tail_messages: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if tail_messages:
            first = tail_messages[0]
            content = first.get("content") or ""
            role = first.get("role") or ""
            if role == "assistant" and self._looks_like_active_summary_blob(content):
                candidate = self._assemble_context(
                    system_msg,
                    tail_messages[1:],
                    assembly_cap_override=assembly_cap_override,
                    include_lcm_note=False,
                )
                if any(
                    (msg.get("content") or "") == content
                    for msg in candidate[1:]
                ):
                    return candidate

        candidate = self._assemble_context(
            system_msg,
            tail_messages,
            assembly_cap_override=assembly_cap_override,
            include_lcm_note=False,
        )
        if len(candidate) == 1 and tail_messages:
            return [system_msg, tail_messages[-1]]
        return candidate

    @staticmethod
    def _looks_like_active_summary_blob(content: str) -> bool:
        if not isinstance(content, str) or not content:
            return False
        block = (
            r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]\n"
            r".*?\n"
            r"\[Expand for details: .*?\]"
        )
        pattern = rf"^{block}(?:\n\n---\n\n{block})*$"
        return re.fullmatch(pattern, content, flags=re.DOTALL) is not None

    @staticmethod
    def _extract_expand_hint(summary: str) -> str:
        """Extract the 'Expand for details about:' line from a summary."""
        marker = "Expand for details about:"
        idx = summary.rfind(marker)
        if idx >= 0:
            hint = summary[idx + len(marker):].strip()
            # Take first line only
            return hint.split("\n")[0].strip()
        return ""

    # -- Lifecycle ---------------------------------------------------------

    def shutdown(self):
        self._store.close()
        self._dag.close()
