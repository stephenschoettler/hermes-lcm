"""LCM Engine — Lossless Context Management.

Implements the ContextEngine ABC. Replaces the built-in ContextCompressor
with a DAG-based summarization system that preserves every message.
"""

import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

from .codex_routing import (
    _codex_oauth_context_cap,
    _is_codex_gpt55_route,
)
from .config import LCMConfig
from .dag import SummaryDAG, SummaryNode
from .diagnostics import _enforce_state_db_containment
from .engine_registry import (
    _ACTIVE_ENGINE_REGISTRY_LOCK,
    _ACTIVE_ENGINES_BY_CONVERSATION_ID,
    _ACTIVE_ENGINES_BY_SESSION_ID,
    _remove_registry_entries_for_engine,
    resolve_active_lcm_engine,  # noqa: F401  (re-exported: hosts import it from .engine)
)
from .escalation import (
    SummaryCircuitBreaker,
    SummarySpendGuard,
    summarize_with_escalation,
)
from .externalize import (
    build_transcript_gc_placeholder,
    extract_externalized_ref,
    find_externalized_payload_for_message,
    find_externalized_tool_result_content_for_call,
    is_externalized_placeholder,
    load_externalized_payload,
    maybe_externalize_tool_output,
)
from .extraction import (
    extract_before_compaction,
    sanitize_pre_compaction_content,
    sanitize_pre_compaction_tool_arguments,
    strip_injected_context_blocks,
)
from .ingest_protection import (
    _expected_persisted_output_chars,
    _has_lossy_sensitive_redaction,
    _is_hermes_persisted_output_marker,
    _persisted_output_inline_preview_sha256,
    _persisted_output_preview_prefix_digest,
    _persisted_output_saved_path,
    assistant_output_quarantine_reason,
    extract_all_externalized_payload_refs,
    extract_ingest_externalized_refs,
    protect_inline_payloads_in_text,
    protect_messages_for_ingest,
    quarantine_suspicious_assistant_messages,
    recover_hermes_persisted_output_with_file_stat,
    redact_sensitive_text,
    redact_sensitive_value,
    restore_ingest_payload_placeholders,
    sensitive_pattern_status,
)
from .runtime_identity import (
    _PLUGIN_ROOT,
    _git_runtime_identity,
    _plugin_metadata,
)
from .schemas import (
    LCM_DESCRIBE,
    LCM_DOCTOR,
    LCM_EXPAND,
    LCM_EXPAND_QUERY,
    LCM_GREP,
    LCM_INSPECT,
    LCM_LOAD_SESSION,
    LCM_STATUS,
)
from .sanitize import (
    _clean_active_assistant_message,
    _should_drop_active_assistant_message,
)
from .session_patterns import (
    build_session_match_keys,
    compile_session_patterns,
    matches_session_pattern,
)
from .message_analysis import (
    _is_synthetic_assistant_noise,
    _matched_tool_call_ids,
    _tool_call_id,
)
from .message_patterns import compile_message_patterns, matches_message_pattern
from .aux_session import AuxiliarySessionMixin
from .placeholder_ledger import PlaceholderLedgerMixin
from .reconcile import ReconcileMixin, _PRESERVED_OBJECTIVE_CONTEXT_PREFIX
from .compaction import CompactionMixin
from .reset_state import ResetStateMixin
from .bypass import BypassMixin
from .lifecycle_state import LifecycleStateStore
from .message_content import (
    normalize_content_value,
    stored_text_content_for_pattern_matching,
    text_content_for_pattern_matching,
)
from .sqlite_util import (
    _is_sqlite_locked_error,
    _temporary_sqlite_busy_timeout,
)
from .store import MessageStore
from .tokens import count_message_tokens, count_messages_tokens, count_tokens
from . import tools as lcm_tools

logger = logging.getLogger(__name__)

_SESSION_END_BUSY_TIMEOUT_MS = 50
_CODEX_GPT55_COMPACTION_THRESHOLD = 0.85

# Auto-focus topic derivation: infer a compact focus hint from the most recent
# real user turns so that summarization can prioritise current user intent.
# Mirrors Hermes upstream fix/compression-auto-focus-topic (#44687 branch).
_AUTO_FOCUS_MAX_TURNS = 3
_AUTO_FOCUS_TURN_MAX_CHARS = 260
_AUTO_FOCUS_MAX_CHARS = 700

_PRESERVED_TODO_CONTEXT_PREFIX = "[Your active task list was preserved across context compression]"
_LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT = 8


class LCMEngine(CompactionMixin, ResetStateMixin, ReconcileMixin, AuxiliarySessionMixin, PlaceholderLedgerMixin, BypassMixin, ContextEngine):
    """Lossless Context Management engine.

    Automatic LCM compaction is routine background maintenance. Hosts that
    support user-visible compaction status opt-outs should keep successful
    automatic LCM passes silent unless the user explicitly asks for diagnostics.

    Architecture:
      1. Every message is persisted verbatim in an immutable MessageStore
      2. When context pressure builds, older messages outside the fresh tail
         are summarized into leaf nodes (D0) in a SummaryDAG
      3. When enough nodes accumulate at a depth, they're condensed into
         higher-depth nodes (D1, D2, ...)
      4. The agent gets tools (lcm_grep, lcm_load_session, lcm_describe,
         lcm_expand) to search and drill into compacted history
      5. Active context = system prompt + DAG summaries + fresh tail
    """

    def __init__(self, config: LCMConfig | None = None,
                 hermes_home: str = ""):
        self._config = config or LCMConfig.from_env()
        self._hermes_home = hermes_home

        db_path = self._resolve_db_path(hermes_home)
        self._bind_storage(db_path, hermes_home)

        self._session_id: str = ""
        self._session_platform: str = ""
        # Tracks the most recent non-ignored, non-stateless binding so that
        # user-facing tools (lcm_status, lcm_grep default scope, lcm_describe,
        # lcm_expand_query, lcm_doctor) keep showing the foreground session
        # even while a side-channel session (cron, debug) temporarily owns the
        # engine's _session_id binding. Updated alongside _session_id only
        # when _refresh_session_filters classifies the new session as a real
        # foreground (neither ignored nor stateless). Read via the
        # `current_session_id` / `current_session_platform` properties and
        # `current_session_ignored` / `current_session_stateless` /
        # `side_channel_active` companion predicates.
        self._foreground_session_id: str = ""
        self._foreground_session_platform: str = ""
        self._foreground_conversation_id: str = ""
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
        self._compiled_ignore_message_patterns = compile_message_patterns(
            self._config.ignore_message_patterns
        )
        self._ignored_message_count: int = 0
        # Raw messages permanently dropped because they matched
        # ignore_message_patterns. These are NOT persisted anywhere, so an
        # over-broad operator pattern silently discards substantive turns from
        # the "lossless" store. Count + log them so the loss is at least
        # visible; full lossless retention (store with ignored=1) is a larger
        # follow-up that touches cursor reconciliation and FTS.
        self._ignore_pattern_dropped_count: int = 0

        # Track which store_ids have been ingested into the DAG
        self._last_compacted_store_id: int = 0

        # Cursor: index in the current messages list up to which all
        # messages have been persisted.  After compress() shortens the
        # list, the cursor resets to len(compressed) so that only
        # genuinely new messages (appended after compaction) get ingested.
        # The cursor is process-local; existing sessions rebound after a
        # gateway restart reconcile it against the durable store on the
        # next ingest.
        self._ingest_cursor: int = 0
        self._ingest_cursor_needs_reconcile = False
        self._last_ingest_reconciliation: Dict[str, Any] = {
            "action": "none",
            "reason": "not run",
        }

        # State required by ContextEngine ABC and run_agent.py compatibility
        self.model = ""
        self.base_url = ""
        self.api_key = ""
        self.provider = ""
        self.api_mode = ""
        self.raw_context_length = 0
        self.context_length = 0
        self.effective_context_length_cap: int | None = None
        self.effective_context_length_reason = ""
        self._context_length_source = ""
        self._update_model_pending_session_start = False
        self.threshold_tokens = 0
        self.context_threshold = self._config.context_threshold
        self.threshold_percent = self.context_threshold
        self._context_threshold_source = (
            self._config.config_sources.get("context_threshold", "manual_or_default")
            if getattr(self._config, "config_sources", None)
            else "manual_or_default"
        )
        self._context_threshold_autoraised: dict[str, float] | None = None
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_reasoning_tokens = 0
        self.cache_metrics_available = False
        self.compression_count = 0
        # Wall-clock of the last leaf compaction (ms); surfaced via telemetry only.
        self._last_compaction_duration_ms = 0.0
        # run_agent.py reads these for preflight checks
        self.protect_first_n = 3
        self.protect_last_n = self._config.fresh_tail_count
        # run_agent.py reads these for context probing
        self._context_probed = False
        self._context_probe_persistable = False
        # Host compatibility: LCM treats successful automatic compaction as
        # silent maintenance. Manual /lcm diagnostics and warning/error paths
        # remain explicit.
        self.emit_automatic_compaction_status = False
        self.quiet_mode = True
        self.summary_model = self._config.summary_model
        self._summary_circuit_breaker = SummaryCircuitBreaker(
            failure_threshold=self._config.summary_circuit_breaker_failure_threshold,
            cooldown_seconds=self._config.summary_circuit_breaker_cooldown_seconds,
        )
        # Summary spend guard: process-local sliding window so a loop that
        # keeps succeeding cannot burn auxiliary-model budget without bound. When
        # tripped, escalation falls back to deterministic L3 truncation. Set
        # summary_spend_max_calls=0 to disable.
        self._summary_spend_guard = SummarySpendGuard(
            max_calls=int(self._config.summary_spend_max_calls),
            window_seconds=float(self._config.summary_spend_window_seconds),
            backoff_seconds=float(self._config.summary_spend_backoff_seconds),
        )
        self._last_overflow_recovery_failed = False
        self._last_condensation_suppressed_reason = ""
        self._last_compression_status = "idle"
        self._last_compression_noop_reason = ""
        # Ingest-failure tracking. The core promise is that nothing is ever
        # lost, but a swallowed persistence error (disk full, DB locked,
        # corruption) silently breaks it: the turn continues while messages
        # exist only in the volatile host list. Surface it instead of hiding
        # it in a debug log so get_status()/doctor can escalate. Store-scoped,
        # not session-scoped, so it is not cleared on session reset.
        self._ingest_failure_count = 0
        self._consecutive_ingest_failures = 0
        self._last_ingest_error = ""
        self._last_ingest_error_time: float = 0
        # Cooldown timestamp to prevent compression cascade after boundary skip.
        # Set when skip-carry-over path is taken in _continue_compression_boundary.
        self._last_boundary_skip_time: float = 0
        # Temporary source window used only while compress() assembles context.
        # _assemble_context also serves tests and recovery paths directly, so
        # keep anchoring opt-in rather than changing its public behavior.
        self._pending_context_anchor_messages: Optional[List[Dict[str, Any]]] = None
        self._current_compress_store_ids_by_message_id: dict[int, int] = {}
        self._current_compress_placeholder_identity_counts: dict[tuple[str, str, str, str], int] = {}
        self._last_active_replay_source_identities: list[tuple[Any, ...]] = []
        self._last_active_replay_messages: list[Dict[str, Any]] = []
        self._generated_ignored_active_replay_placeholder_message_ids: set[int] = set()
        self._logged_filter_config = False
        self._pending_reset_session_id: str = ""
        self._pending_reset_conversation_id: str = ""
        self._pending_reset_frontier_store_id: int = 0
        self._compression_boundary_ingest_pending = False
        self._compression_boundary_active_placeholder_digest_budget: dict[str, int] = {}
        self._compression_boundary_active_placeholder_digest_ordinals: dict[str, set[int]] = {}
        self._compression_boundary_stored_placeholder_digest_counts: dict[str, int] = {}
        self._thread_context = threading.local()
        self._auxiliary_session_ids: set[str] = set()
        self._auxiliary_lineage_session_ids: set[str] = set()
        self._auxiliary_last_prompt_tokens: dict[str, int] = {}
        self._auxiliary_session_generations: dict[str, int] = {}
        self._auxiliary_generation_tokens: dict[int, tuple[Any, int]] = {}
        self._auxiliary_next_generation_token = 0
        self._auxiliary_direct_end_guard_session_ids: set[str] = set()
        self._auxiliary_handoff_parent_session_ids: dict[str, str] = {}
        self._auxiliary_retired_session_generations: dict[str, set[int]] = {}
        self._auxiliary_foreground_reused_session_ids: set[str] = set()
        self._lcm_bypass_lineage_session_ids: set[str] = set()
        self._lcm_bypass_lineage_platforms: dict[str, set[str]] = {}
        self._lcm_non_bypass_platforms: dict[str, set[str]] = {}
        self._lcm_session_last_platform: dict[str, str] = {}
        self._lcm_session_last_normal_platform: dict[str, str] = {}
        self._lcm_session_last_bypassed: dict[str, bool] = {}
        self._lcm_session_last_conversation_id: dict[str, str] = {}
        self._lcm_session_last_normal_conversation_id: dict[str, str] = {}
        self._lcm_bypass_message_prefix_fingerprints: dict[
            str, list[tuple[list[str], bool]]
        ] = {}
        self._lcm_normal_message_prefix_fingerprints: dict[tuple[str, str], list[str]] = {}
        self._lcm_current_start_allows_bypass_lineage = False
        self._auxiliary_session_lock = threading.RLock()
        self._host_fallback_compressor: Any = None
        self._host_fallback_session_id = ""
        self._host_fallback_import_warning_logged = False

    def clone_for_agent(self) -> "LCMEngine":
        """Return a fresh runtime engine for one AIAgent instance.

        Hermes registers plugin context engines process-wide, while gateway
        runtimes may keep multiple cached AIAgent instances alive at once
        (different platforms, chats, cron jobs, etc.).  LCM stores mutable
        session binding and ingest cursor state on the engine object itself, so
        sharing one registered instance across agents can let one conversation
        rebind another conversation's raw-message ingest and lifecycle state.

        The clone shares the same durable SQLite database path/configuration,
        but gets independent session/cursor/lifecycle runtime state. Runtime
        model and context-window metadata is copied so the clone is immediately
        budget-aware even before a compatible Hermes host calls update_model().
        """
        clone = type(self)(
            config=copy.deepcopy(self._config),
            hermes_home=self._hermes_home,
        )
        clone.model = self.model
        clone.base_url = self.base_url
        clone.api_key = self.api_key
        clone.provider = self.provider
        clone.api_mode = self.api_mode
        if self._context_length_source:
            clone._set_context_length(
                self.raw_context_length,
                source=self._context_length_source,
                model=self.model,
                provider=self.provider,
            )
        elif self.raw_context_length or self.context_length:
            clone._set_context_length(
                self.raw_context_length or self.context_length,
                source="clone_for_agent",
                model=self.model,
                provider=self.provider,
            )
        # ``update_model()`` authority is a per-runtime lifecycle edge, not
        # durable metadata.  Compatible hosts call update_model() on the clone
        # before binding it; hosts that bind only through on_session_start()
        # must still be able to replace the copied prototype route.
        clone._update_model_pending_session_start = False
        clone._lcm_current_start_allows_bypass_lineage = False
        return clone

    def __deepcopy__(self, memo: dict[int, object]) -> "LCMEngine":
        """Copy the plugin runtime without pickling SQLite-backed helpers.

        Hermes core may deepcopy plugin context engines while creating isolated
        AIAgent instances. A default object deepcopy walks into MessageStore,
        SummaryDAG, and LifecycleStateStore sqlite3.Connection handles, which
        cannot be pickled. LCM already exposes clone_for_agent() as the safe
        boundary: share durable configuration/database path, but allocate fresh
        per-agent runtime/storage helper objects.
        """
        clone = self.clone_for_agent()
        memo[id(self)] = clone
        return clone

    def _resolve_db_path(self, hermes_home: str = "") -> Path:
        """Resolve the SQLite path for the active Hermes profile/home."""
        if self._config.database_path:
            return Path(self._config.database_path)
        if hermes_home:
            return Path(hermes_home) / "lcm.db"
        return Path.home() / ".hermes" / "lcm.db"

    def _bind_storage(self, db_path: str | Path, hermes_home: str = "") -> None:
        """Bind store/DAG/lifecycle helpers to one SQLite database."""
        self._store = MessageStore(
            db_path,
            ingest_protection_config=self._config,
            hermes_home=hermes_home,
        )
        self._dag = SummaryDAG(db_path)
        self._lifecycle = LifecycleStateStore(db_path)

    def _close_storage(self) -> None:
        """Best-effort close of currently bound SQLite helpers."""
        for attr in ("_store", "_dag", "_lifecycle"):
            helper = getattr(self, attr, None)
            close = getattr(helper, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("LCM failed closing %s during profile rebind", attr, exc_info=True)

    def _reset_profile_runtime_state(self) -> None:
        """Clear process-local session state that cannot cross profile homes."""
        self._unregister_active_engine_binding()
        self._session_id = ""
        self._session_platform = ""
        self._foreground_session_id = ""
        self._foreground_session_platform = ""
        self._foreground_conversation_id = ""
        self._conversation_id = ""
        self._session_match_keys = []
        self._session_ignored = False
        self._session_stateless = False
        self._clear_pending_reset_boundary()
        self._compression_boundary_ingest_pending = False
        self._compression_boundary_active_placeholder_digest_budget = {}
        self._compression_boundary_active_placeholder_digest_ordinals = {}
        self._compression_boundary_stored_placeholder_digest_counts = {}
        with self._auxiliary_session_lock:
            self._auxiliary_session_ids.clear()
            self._auxiliary_lineage_session_ids.clear()
            self._auxiliary_last_prompt_tokens.clear()
            self._auxiliary_session_generations.clear()
            self._auxiliary_generation_tokens.clear()
            self._auxiliary_next_generation_token = 0
            self._auxiliary_direct_end_guard_session_ids.clear()
            self._auxiliary_handoff_parent_session_ids.clear()
            self._auxiliary_retired_session_generations.clear()
            self._auxiliary_foreground_reused_session_ids.clear()
            self._lcm_bypass_lineage_session_ids.clear()
            self._lcm_bypass_lineage_platforms.clear()
            self._lcm_non_bypass_platforms.clear()
            self._lcm_session_last_platform.clear()
            self._lcm_session_last_normal_platform.clear()
            self._lcm_session_last_bypassed.clear()
            self._lcm_session_last_conversation_id.clear()
            self._lcm_session_last_normal_conversation_id.clear()
            self._lcm_bypass_message_prefix_fingerprints.clear()
            self._lcm_normal_message_prefix_fingerprints.clear()
        self._lcm_current_start_allows_bypass_lineage = False
        self._host_fallback_compressor = None
        self._host_fallback_session_id = ""
        self._host_fallback_import_warning_logged = False
        self._clear_thread_context_stateless()
        self._reset_session_scoped_runtime_state()

    def _rebind_storage_for_home(self, hermes_home: str = "") -> bool:
        """Switch SQLite-backed state when a reused engine serves another profile.

        Hermes core passes the active ``hermes_home`` on session start.  Older
        Hermes versions may still reuse the same plugin/context-engine object
        after ``HERMES_HOME`` changes, so the plugin must not assume the store
        captured during ``register()`` is still correct.
        """
        if not hermes_home:
            return False
        if self._config.database_path:
            current_home = str(self._hermes_home or "")
            current_store_home = str(getattr(getattr(self, "_store", None), "_hermes_home", "") or "")
            if current_home == str(hermes_home) and current_store_home == str(hermes_home):
                return False
            self._hermes_home = hermes_home
            store = getattr(self, "_store", None)
            if store is not None:
                store._hermes_home = hermes_home
            self._reset_profile_runtime_state()
            logger.info("LCM rebound Hermes home for configured database path %s", hermes_home)
            return True

        db_path = self._resolve_db_path(hermes_home)
        current_db = Path(getattr(getattr(self, "_store", None), "db_path", ""))
        if current_db == db_path and str(self._hermes_home or "") == str(hermes_home):
            return False

        self._close_storage()
        self._hermes_home = hermes_home
        self._bind_storage(db_path, hermes_home)
        self._reset_profile_runtime_state()
        logger.info("LCM rebound storage for Hermes home %s", hermes_home)
        return True

    def _runtime_context_threshold(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> tuple[float, str, dict[str, float] | None]:
        configured = float(self._config.context_threshold)
        source = (
            self._config.config_sources.get("context_threshold", "manual_or_default")
            if getattr(self._config, "config_sources", None)
            else "manual_or_default"
        )
        explicit_lcm_override = source in {
            "env:LCM_CONTEXT_THRESHOLD",
            "config_yaml:lcm.context_threshold",
        }
        route_model = self.model if model is None else model
        route_provider = self.provider if provider is None else provider
        if (
            _is_codex_gpt55_route(route_model, route_provider)
            and self._config.codex_gpt55_autoraise_enabled
            and not explicit_lcm_override
            and configured < _CODEX_GPT55_COMPACTION_THRESHOLD
        ):
            return (
                _CODEX_GPT55_COMPACTION_THRESHOLD,
                "codex_gpt55_autoraise",
                {"from": configured, "to": _CODEX_GPT55_COMPACTION_THRESHOLD},
            )
        return configured, source, None

    def _effective_context_length(
        self,
        raw_context_length: int,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> tuple[int, int | None, str]:
        route_model = self.model if model is None else model
        route_provider = self.provider if provider is None else provider
        cap = _codex_oauth_context_cap(route_model, route_provider)
        if cap is not None and raw_context_length > cap:
            return (
                cap,
                cap,
                "codex_oauth_context_cap",
            )
        return raw_context_length, None, ""

    def _effective_threshold_tokens(self, context_threshold_tokens: int) -> int:
        """Return the host-visible preflight trigger token count.

        Hermes core uses ``threshold_tokens`` as a cheap gate before it pays for
        the full request estimate that includes system prompt and tool schemas.
        LCM can enforce a stricter active-context assembly cap than the normal
        context-threshold value, so expose the stricter cap here; otherwise a
        tool/schema-heavy request can skip host preflight entirely.
        """
        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is not None and assembly_cap > 0:
            if context_threshold_tokens > 0:
                return min(context_threshold_tokens, assembly_cap)
            return assembly_cap
        return context_threshold_tokens

    def _set_context_length(
        self,
        context_length: Any,
        *,
        source: str,
        model: str | None = None,
        provider: str | None = None,
    ) -> bool:
        try:
            parsed_context_length = int(context_length)
        except (TypeError, ValueError):
            logger.debug("LCM ignored invalid %s context_length: %r", source, context_length)
            return False
        if parsed_context_length <= 0:
            logger.debug(
                "LCM cleared non-positive %s context_length: %r",
                source,
                context_length,
            )
            self.raw_context_length = 0
            self.context_length = 0
            self.effective_context_length_cap = None
            self.effective_context_length_reason = ""
            self._context_length_source = source
            self.threshold_tokens = 0
            self.context_threshold, self._context_threshold_source, self._context_threshold_autoraised = (
                self._runtime_context_threshold(model=model, provider=provider)
            )
            self.threshold_percent = self.context_threshold
            return True
        self.raw_context_length = parsed_context_length
        effective_context_length, cap, reason = self._effective_context_length(
            parsed_context_length,
            model=model,
            provider=provider,
        )
        self.context_length = effective_context_length
        self.effective_context_length_cap = cap
        self.effective_context_length_reason = reason
        self._context_length_source = source
        self.context_threshold, self._context_threshold_source, self._context_threshold_autoraised = (
            self._runtime_context_threshold(model=model, provider=provider)
        )
        self.threshold_percent = self.context_threshold
        context_threshold_tokens = int(
            effective_context_length * self.context_threshold
        )
        self.threshold_tokens = self._effective_threshold_tokens(
            context_threshold_tokens
        )
        return True

    def _session_metadata_matches_active_runtime(
        self,
        kwargs: Dict[str, Any],
        *,
        ignore_empty_optional: bool = False,
    ) -> bool:
        if "model" in kwargs and str(kwargs.get("model") or "") != self.model:
            return False
        for key in ("provider", "base_url", "api_key", "api_mode"):
            if key not in kwargs:
                continue
            incoming = str(kwargs.get(key) or "")
            if ignore_empty_optional and not incoming:
                continue
            if incoming != str(getattr(self, key, "") or ""):
                return False
        return True

    @property
    def name(self) -> str:
        return "lcm"

    @property
    def last_compression_status(self) -> str:
        """Public status for the most recent compression/preflight attempt.

        Host runtimes use this to distinguish a real compaction boundary from
        an LCM no-op (for example, when request pressure is high but all
        compactable raw backlog is protected by the fresh tail).
        """
        return self._last_compression_status

    @property
    def last_compression_noop_reason(self) -> str:
        """Human-readable reason for the latest no-op compression decision."""
        return self._last_compression_noop_reason

    @property
    def last_compression_was_noop(self) -> bool:
        """Whether the most recent compression/preflight decision was a no-op."""
        return self._last_compression_status == "noop"

    def _mark_preflight_compression_requested(self) -> bool:
        """Record that preflight found work and clear any stale no-op reason."""
        self._last_compression_status = "pending"
        self._last_compression_noop_reason = ""
        return True

    @property
    def current_session_id(self) -> str:
        """User-facing "current session" id surfaced by LCM tools.

        Returns the most recent foreground binding (the last session id that
        ``_refresh_session_filters`` classified as neither ignored nor
        stateless). Falls back to ``_session_id`` when no foreground has
        ever been bound, so unattended cron-only or stateless-only processes
        remain observable via ``lcm_status``.

        Lifecycle paths (compress, ingest, on_session_end, etc.) must keep
        reading ``_session_id`` directly because those paths must follow the
        binding the engine is actually servicing. Only tool-surface code
        paths that report a "current session" view to operators should read
        this property.
        """
        return self._foreground_session_id or self._session_id

    @property
    def current_session_platform(self) -> str:
        """Platform string paired with ``current_session_id``."""
        if self._foreground_session_id:
            return self._foreground_session_platform
        return self._session_platform

    @property
    def current_conversation_id(self) -> str:
        """Conversation id paired with ``current_session_id``."""
        if self._foreground_session_id:
            return self._foreground_conversation_id
        return self._conversation_id

    @property
    def side_channel_active(self) -> bool:
        """True when an ignored or stateless session has temporarily rebound
        ``_session_id`` while a real foreground binding still exists.

        Operators reading lcm_status during this window see the foreground
        session id and counts (because tools read ``current_session_id``)
        but the engine itself is servicing the side channel. This predicate
        lets diagnostic surfaces (lcm_status, /lcm command) make the
        divergence explicit without recomputing the underlying invariant.
        """
        return bool(self._foreground_session_id) and self._foreground_session_id != self._session_id

    @property
    def current_session_ignored(self) -> bool:
        """``_session_ignored`` reported for ``current_session_id``.

        When a side channel is in flight the foreground is by definition
        non-ignored; otherwise this is the bound session's ignore flag.
        """
        if self.side_channel_active:
            return False
        return self._session_ignored

    @property
    def current_session_stateless(self) -> bool:
        """``_session_stateless`` reported for ``current_session_id``.

        When a side channel is in flight the foreground is by definition
        non-stateless; otherwise this is the bound session's stateless flag.
        """
        if self.side_channel_active:
            return False
        return self._session_stateless

    # -- ContextEngine required methods ------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        if self._thread_context_stateless():
            auxiliary_session_id = self._thread_context_session_id()
            if auxiliary_session_id:
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                caller_generation = self._in_process_auxiliary_caller_generation(
                    auxiliary_session_id
                )
                with self._auxiliary_session_lock:
                    if auxiliary_session_id not in self._auxiliary_session_ids:
                        return
                    if self._auxiliary_generation_is_retired(
                        auxiliary_session_id,
                        caller_generation,
                    ):
                        return
                    active_generation = self._auxiliary_session_generations.get(
                        auxiliary_session_id
                    )
                    if active_generation is None and caller_generation:
                        expected_parent = self._auxiliary_handoff_parent_session_ids.get(
                            auxiliary_session_id
                        )
                        if expected_parent and self._in_process_parent_session_id(
                            {},
                            session_id=auxiliary_session_id,
                            include_explicit=False,
                        ) != expected_parent:
                            return
                        if auxiliary_session_id in self._auxiliary_last_prompt_tokens:
                            self._auxiliary_direct_end_guard_session_ids.add(
                                auxiliary_session_id
                            )
                        self._auxiliary_last_prompt_tokens.pop(auxiliary_session_id, None)
                        if self._host_fallback_session_id == auxiliary_session_id:
                            self._end_host_fallback_compressor_for_session(
                                auxiliary_session_id,
                                [],
                                current_session_bypasses=True,
                            )
                        self._auxiliary_session_generations[
                            auxiliary_session_id
                        ] = caller_generation
                        active_generation = caller_generation
                    stack = self._thread_context_auxiliary_stack()
                    stack_marks_current_session = bool(
                        active_generation is not None
                        and caller_generation == 0
                        and stack
                        and stack[-1] == auxiliary_session_id
                    )
                    generation_matches = (
                        caller_generation == 0
                        if active_generation is None
                        else caller_generation == active_generation or stack_marks_current_session
                    )
                    if generation_matches:
                        self._auxiliary_last_prompt_tokens[auxiliary_session_id] = prompt_tokens
            return
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.last_total_tokens = int(usage.get("total_tokens", 0) or 0)

        cache_keys = {"cache_read_tokens", "cache_write_tokens"}
        self.cache_metrics_available = any(key in usage for key in cache_keys)
        self.last_input_tokens = int(usage.get("input_tokens", self.last_prompt_tokens) or 0)
        self.last_output_tokens = int(
            usage.get("output_tokens", self.last_completion_tokens) or 0
        )
        self.last_cache_read_tokens = int(usage.get("cache_read_tokens", 0) or 0)
        self.last_cache_write_tokens = int(usage.get("cache_write_tokens", 0) or 0)
        self.last_reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)
        self._record_turn_compaction_telemetry()

    @property
    def cache_read_ratio(self) -> float:
        if self.last_prompt_tokens <= 0:
            return 0.0
        return self.last_cache_read_tokens / self.last_prompt_tokens

    def _record_turn_compaction_telemetry(self) -> None:
        """Persist a per-conversation compaction-telemetry snapshot for this turn.

        Best-effort and diagnostic only: any failure is logged at debug and never
        affects the turn. Turns with no token or cache signal are skipped so idle
        turns do not churn the record. The since-compaction accumulators reset off
        the monotonic ``compression_count`` (which also drops to 0 on a session
        reset) rather than instrumenting the compaction hot path.
        """
        conversation_id = self._conversation_id
        if not conversation_id:
            return
        prompt_tokens = self.last_prompt_tokens
        cache_read = self.last_cache_read_tokens
        cache_write = self.last_cache_write_tokens
        if (
            prompt_tokens <= 0
            and cache_read <= 0
            and cache_write <= 0
            and not self.cache_metrics_available
        ):
            return
        try:
            existing = self._store.read_compaction_telemetry(conversation_id) or {}

            if cache_read > 0 or cache_write > 0:
                cache_state = "hot"
            elif self.cache_metrics_available:
                cache_state = "cold"
            else:
                cache_state = "unknown"
            cold_streak = int(existing.get("consecutive_cold_observations", 0) or 0)
            if cache_state == "hot":
                cold_streak = 0
            elif cache_state == "cold":
                cold_streak += 1

            prev_count = int(existing.get("compression_count_at_record", 0) or 0)
            compacted = self.compression_count > prev_count
            rebaselined = self.compression_count != prev_count  # compaction or session reset
            if rebaselined:
                turns_since = 0
                peak_tokens_since = prompt_tokens
            else:
                turns_since = int(existing.get("turns_since_leaf_compaction", 0) or 0) + 1
                peak_tokens_since = max(
                    int(existing.get("peak_prompt_tokens_since_leaf_compaction", 0) or 0),
                    prompt_tokens,
                )
            total_compactions = int(existing.get("total_compactions", 0) or 0)
            if compacted:
                total_compactions += self.compression_count - prev_count
                last_leaf_compaction_at = time.time()
                last_compaction_duration_ms = round(self._last_compaction_duration_ms, 3)
            else:
                last_leaf_compaction_at = existing.get("last_leaf_compaction_at")
                last_compaction_duration_ms = existing.get("last_compaction_duration_ms")

            record = dict(existing)
            record.update({
                "conversation_id": conversation_id,
                "last_observed_prompt_tokens": prompt_tokens,
                "last_observed_cache_read": cache_read,
                "last_observed_cache_write": cache_write,
                "cache_state": cache_state,
                "consecutive_cold_observations": cold_streak,
                "turns_since_leaf_compaction": turns_since,
                "peak_prompt_tokens_since_leaf_compaction": peak_tokens_since,
                # Reserved carry-forward field; no live 'medium'/'high' computation yet.
                "activity_band": existing.get("activity_band", "low"),
                "provider": self.provider or existing.get("provider"),
                "model": self.model or existing.get("model"),
                "last_api_call_at": time.time(),
                "last_leaf_compaction_at": last_leaf_compaction_at,
                "last_compaction_duration_ms": last_compaction_duration_ms,
                "total_compactions": total_compactions,
                "compression_count_at_record": self.compression_count,
            })
            if cache_state == "hot":
                record["last_cache_hit_at"] = time.time()
            self._store.write_compaction_telemetry(conversation_id, record)
        except Exception:
            logger.debug("LCM compaction telemetry update failed", exc_info=True)

    def _compression_boundary_cooldown_active(self) -> bool:
        """Return true while a boundary skip is in its short no-compress window."""
        if self._last_boundary_skip_time <= 0:
            return False
        elapsed = time.time() - self._last_boundary_skip_time
        if elapsed < 60:
            logger.debug(
                "LCM compression cooldown active: %.1f seconds since boundary skip",
                elapsed,
            )
            return True
        self._last_boundary_skip_time = 0
        return False

    def _record_ingest_success(self) -> None:
        self._consecutive_ingest_failures = 0

    def _record_ingest_failure(self, where: str, error: Exception) -> None:
        """Track a swallowed ingest error so it is operator-visible.

        Escalates to error level once failures are consecutive: a single
        transient lock is a warning, but a sustained inability to persist
        means the lossless guarantee is broken and must not stay hidden.
        """
        self._ingest_failure_count += 1
        self._consecutive_ingest_failures += 1
        self._last_ingest_error = f"{type(error).__name__}: {error}"
        self._last_ingest_error_time = time.time()
        message = "LCM ingest failed (%s): %s [consecutive=%d, total=%d]"
        args = (
            where,
            error,
            self._consecutive_ingest_failures,
            self._ingest_failure_count,
        )
        if self._consecutive_ingest_failures >= 3:
            logger.error(message, *args)
        else:
            logger.warning(message, *args)

    def ingest(self, messages: List[Dict[str, Any]]) -> None:
        """Persist messages to the durable store every turn.

        Called by the post_llm_call plugin hook so messages land in LCM
        regardless of whether compression triggers — short WebUI
        conversations never hit the compression threshold and never
        expire like Telegram sessions do, so without this they'd never
        be ingested.

        Uses the same _ingest_messages cursor as compress(), so if
        compression runs later the same turn, already-ingested messages
        are skipped (no duplicates).
        """
        if self._bypasses_lcm_context_management():
            self._remember_lcm_bypass_message_prefix(self._bypass_lcm_session_id(), messages)
            return
        if self._session_id and messages:
            try:
                self._remember_lcm_normal_message_prefix(
                    self._session_id,
                    messages,
                    conversation_id=self._conversation_id,
                )
                self._ingest_messages(messages)
                self._record_ingest_success()
                logger.debug(
                    "Per-turn ingest OK: session=%s msgs=%d cursor=%d",
                    self._session_id, len(messages), self._ingest_cursor,
                )
            except Exception as e:
                self._record_ingest_failure("per-turn ingest()", e)

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
                    fallback_models=self._config.summary_fallback_models,
                    circuit_breaker=self._summary_circuit_breaker,
                    spend_guard=self._summary_spend_guard,
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

    # -- ContextEngine optional methods ------------------------------------

    def _bind_lifecycle_state(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> None:
        state = self._lifecycle.bind_session(session_id, conversation_id=conversation_id)
        self._conversation_id = state.conversation_id
        self._lcm_session_last_conversation_id[session_id] = state.conversation_id
        self._last_compacted_store_id = state.current_frontier_store_id
        self._register_active_engine_binding()
        if not self._session_ignored and not self._session_stateless:
            self._lcm_session_last_normal_conversation_id[session_id] = state.conversation_id
            self._foreground_session_id = session_id
            self._foreground_session_platform = self._session_platform
            self._foreground_conversation_id = state.conversation_id

        # Garbage-collect empty lifecycle rows when the table exceeds threshold.
        # Gateway restarts, ephemeral cron ticks, and crash-loops all create
        # lifecycle rows that never ingest data — prune them here so they
        # don't accumulate forever.
        if (
            self._config.empty_lifecycle_gc_enabled
            and self._lifecycle.row_count() > self._config.empty_lifecycle_gc_threshold
        ):
            protected = {str(self._session_id)} if self._session_id else None
            max_age = self._config.empty_lifecycle_gc_max_age_hours
            try:
                deleted = self._lifecycle.prune_empty_sessions(
                    protected_session_ids=protected,
                    max_age_hours=max_age,
                )
            except Exception:
                deleted = 0
            if deleted:
                logger.info(
                    "LCM pruned %d lifecycle rows with zero stored data "
                    "(table exceeded threshold of %d rows)",
                    deleted,
                    self._config.empty_lifecycle_gc_threshold,
                )

    def _register_active_engine_binding(self) -> None:
        session_id = str(self._session_id or "")
        conversation_id = str(self._conversation_id or "")
        if not session_id:
            return
        with _ACTIVE_ENGINE_REGISTRY_LOCK:
            _remove_registry_entries_for_engine(
                self,
                keep_session_id=session_id,
                keep_conversation_id=conversation_id,
            )
            _ACTIVE_ENGINES_BY_SESSION_ID[session_id] = self
            if conversation_id:
                _ACTIVE_ENGINES_BY_CONVERSATION_ID[conversation_id] = self

    def _unregister_active_engine_binding(self) -> None:
        with _ACTIVE_ENGINE_REGISTRY_LOCK:
            _remove_registry_entries_for_engine(self)

    def _persist_frontier_marker(self) -> None:
        if not self._session_id or not self._conversation_id:
            return
        self._lifecycle.advance_frontier(
            self._conversation_id,
            self._session_id,
            self._last_compacted_store_id,
        )

    def _has_lcm_bypass_lineage_session(self, session_id: str, *, platform: Optional[str] = None) -> bool:
        with self._auxiliary_session_lock:
            if session_id not in self._lcm_bypass_lineage_session_ids:
                return False
            if platform is None:
                return True
            platforms = self._lcm_bypass_lineage_platforms.get(session_id) or set()
            return not platforms or platform in platforms

    def _mark_lcm_bypass_lineage_session(self, session_id: str, *, platform: Optional[str] = None) -> None:
        if not session_id:
            return
        platform = self._session_platform if platform is None else str(platform or "")
        with self._auxiliary_session_lock:
            self._lcm_bypass_lineage_session_ids.add(session_id)
            self._lcm_bypass_lineage_platforms.setdefault(session_id, set()).add(platform)
            self._lcm_session_last_platform[session_id] = platform
            self._lcm_session_last_bypassed[session_id] = True

    def _unmark_lcm_bypass_lineage_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._auxiliary_session_lock:
            self._lcm_bypass_lineage_session_ids.discard(session_id)
            self._lcm_bypass_lineage_platforms.pop(session_id, None)

    def _handoff_lcm_bypass_lineage(
        self,
        old_session_id: str,
        new_session_id: str,
        *,
        new_platform: str = "",
    ) -> None:
        with self._auxiliary_session_lock:
            if old_session_id:
                self._lcm_bypass_lineage_session_ids.add(old_session_id)
            if new_session_id:
                new_platform = str(new_platform or "")
                self._lcm_bypass_lineage_session_ids.add(new_session_id)
                self._lcm_bypass_lineage_platforms.setdefault(new_session_id, set()).add(new_platform)
                self._lcm_session_last_platform[new_session_id] = new_platform
                self._lcm_session_last_bypassed[new_session_id] = True

    def _compression_boundary_from_lcm_bypassed_session(self, old_session_id: str) -> bool:
        if not old_session_id:
            return False
        if old_session_id in self._lcm_session_last_bypassed:
            return bool(self._lcm_session_last_bypassed.get(old_session_id))
        if old_session_id == self._session_id:
            return bool(
                self._bypasses_lcm_context_management()
                or self._session_id_matches_lcm_bypass_filters(
                    old_session_id,
                    platform=self._session_platform,
                )
            )
        return bool(
            self._has_lcm_bypass_lineage_session(old_session_id)
            or self._session_id_matches_lcm_bypass_filters(old_session_id)
        )

    def _get_allowed_hermes_base(self) -> Path | None:
        """Get the allowed base directory for hermes_home, or None if not restricted."""
        env_base = os.environ.get("LCM_HERMES_BASE_DIR")
        if env_base:
            return Path(env_base).expanduser().resolve()
        return None  # No restriction when env var not set

    def _state_db_path(self, kwargs: Dict[str, Any] | None = None) -> Path:
        kwargs = kwargs or {}
        hermes_home = str(kwargs.get("hermes_home") or self._hermes_home or "")
        if hermes_home:
            return _enforce_state_db_containment(
                Path(hermes_home) / "state.db",
                description=f"hermes_home {hermes_home}",
            )
        db_path = Path(self._store.db_path)
        return _enforce_state_db_containment(
            db_path.parent / "state.db",
            description=f"state database fallback from LCM database {db_path}",
        )

    def _clear_pending_reset_boundary(self) -> None:
        self._pending_reset_session_id = ""
        self._pending_reset_conversation_id = ""
        self._pending_reset_frontier_store_id = 0

    def _finalize_pending_reset_boundary(self, session_id: str) -> None:
        if not self._pending_reset_session_id:
            return
        if self._pending_reset_session_id != session_id:
            self._clear_pending_reset_boundary()
            return
        if not self._pending_reset_conversation_id:
            self._clear_pending_reset_boundary()
            return
        state = self._lifecycle.get_by_conversation(self._pending_reset_conversation_id)
        frontier_store_id = self._pending_reset_frontier_store_id
        if state is not None and state.current_session_id == session_id:
            frontier_store_id = max(
                frontier_store_id,
                int(state.current_frontier_store_id or 0),
            )
        self._lifecycle.finalize_session(
            self._pending_reset_conversation_id,
            self._pending_reset_session_id,
            frontier_store_id=frontier_store_id,
        )
        self._clear_pending_reset_boundary()

    def _raw_backlog_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n = len(messages)
        fresh_tail_start = max(0, n - self._config.fresh_tail_count)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return []
        return messages[leading_anchor_count:fresh_tail_start]

    @staticmethod
    def _leading_anchor_count(messages: List[Dict[str, Any]]) -> int:
        """Return the number of non-compactable leading messages.

        Only the system prompt is a safe permanent anchor. Hermes gateway
        sessions can begin with a user message when core passes conversation
        history without a system prompt; preserving that first user turn as raw
        active context lets stale requests look current after later compaction.
        """
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            return 1
        return 0

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

    def _budget_pressure_ratio(
        self,
        *,
        observed_tokens: int | None = None,
        messages: List[Dict[str, Any]] | None = None,
    ) -> float | None:
        if self.context_length <= 0:
            return None
        token_count: int | None = None
        if observed_tokens is not None and observed_tokens > 0:
            token_count = observed_tokens
        elif messages is not None:
            token_count = count_messages_tokens(messages)
        elif self.last_prompt_tokens > 0:
            token_count = self.last_prompt_tokens
        if token_count is None or token_count <= 0:
            return None
        return token_count / self.context_length

    def _critical_budget_pressure_reached(
        self,
        *,
        observed_tokens: int | None = None,
        messages: List[Dict[str, Any]] | None = None,
    ) -> bool:
        threshold = self._config.critical_budget_pressure_ratio
        if threshold <= 0:
            return False
        pressure = self._budget_pressure_ratio(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        return pressure is not None and pressure >= threshold

    def _should_run_deferred_maintenance(
        self,
        messages: List[Dict[str, Any]],
        *,
        observed_tokens: int | None = None,
    ) -> bool:
        if not self._has_raw_backlog_debt():
            return False
        raw_tokens = self._raw_backlog_tokens(messages)
        if raw_tokens <= 0:
            return False
        if raw_tokens >= self._raw_backlog_threshold(raw_tokens):
            return True
        return self._critical_budget_pressure_reached(
            observed_tokens=observed_tokens,
            messages=messages,
        )

    def _refresh_raw_backlog_debt(
        self,
        messages: List[Dict[str, Any]],
        *,
        observed_tokens: int | None = None,
    ) -> None:
        if not self._config.deferred_maintenance_enabled or not self._conversation_id:
            return
        raw_tokens = self._raw_backlog_tokens(messages)
        threshold = self._raw_backlog_threshold(raw_tokens) if raw_tokens > 0 else 0
        keep_under_critical_pressure = (
            raw_tokens > 0
            and self._has_raw_backlog_debt()
            and self._critical_budget_pressure_reached(
                observed_tokens=observed_tokens,
                messages=messages,
            )
        )
        if raw_tokens > 0 and (raw_tokens >= threshold or keep_under_critical_pressure):
            self._lifecycle.record_debt(
                self._conversation_id,
                kind="raw_backlog",
                size_estimate=raw_tokens,
            )
            return
        if self._has_raw_backlog_debt():
            self._lifecycle.clear_debt(self._conversation_id)

    def _apply_session_start_metadata(self, session_id: str, kwargs: Dict[str, Any]) -> None:
        self._session_id = session_id
        self._session_platform = str(kwargs.get("platform") or "")
        self._refresh_session_filters()
        # Hold the foreground view stable when the new binding is a side
        # channel (cron tick inside the gateway process, debug probe, etc.).
        # Tools that report "current session" to operators must keep pointing
        # at the real foreground rather than the ignored/stateless session
        # that just stole _session_id. Lifecycle paths still read _session_id
        # directly so cron's compress short-circuits correctly via the
        # _session_ignored / _session_stateless gates.
        if not self._session_ignored and not self._session_stateless:
            self._foreground_session_id = session_id
            self._foreground_session_platform = self._session_platform
        if "hermes_home" in kwargs:
            self._hermes_home = kwargs["hermes_home"]

        update_model_is_authoritative = (
            self._context_length_source == "update_model"
            and self._update_model_pending_session_start
        )

        # Pick up context_length from kwargs if provided, but do not let stale
        # session metadata undo the authoritative runtime update_model() call.
        # Hermes Agent calls update_model() with the resolver output before it
        # binds a fresh agent/session.  Older or buggy host paths can still pass
        # a context_length copied from the previously bound runtime; treating
        # that as authoritative makes /model switches keep compressing against
        # the old model window.
        if "context_length" in kwargs:
            incoming_context_length = kwargs["context_length"]
            try:
                parsed_context_length = int(incoming_context_length)
            except (TypeError, ValueError):
                logger.debug(
                    "LCM ignored invalid session-start context_length: %r",
                    incoming_context_length,
                )
                self._update_model_pending_session_start = False
                return
            if parsed_context_length <= 0:
                if update_model_is_authoritative:
                    if self._session_metadata_matches_active_runtime(
                        kwargs,
                        ignore_empty_optional=True,
                    ):
                        logger.debug(
                            "LCM ignored missing session-start context_length=%r for model=%s; active update_model context_length=%s",
                            incoming_context_length,
                            self.model or str(kwargs.get("model") or ""),
                            self.context_length,
                        )
                    else:
                        logger.warning(
                            "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                            str(kwargs.get("model") or ""),
                            self.model,
                        )
                    self._update_model_pending_session_start = False
                    return
                self._set_context_length(parsed_context_length, source="session_start")
                update_model_is_authoritative = False
            else:
                if (
                    update_model_is_authoritative
                    and parsed_context_length not in {self.context_length, self.raw_context_length}
                ):
                    logger.warning(
                        "LCM ignored stale session-start context_length=%s for model=%s; active update_model raw_context_length=%s effective_context_length=%s",
                        parsed_context_length,
                        self.model or str(kwargs.get("model") or ""),
                        self.raw_context_length,
                        self.context_length,
                    )
                    self._update_model_pending_session_start = False
                    return
                if update_model_is_authoritative:
                    if not self._session_metadata_matches_active_runtime(kwargs):
                        logger.warning(
                            "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                            str(kwargs.get("model") or ""),
                            self.model,
                        )
                        self._update_model_pending_session_start = False
                        return
                else:
                    self._set_context_length(
                        parsed_context_length,
                        source="session_start",
                        model=str(kwargs.get("model") or self.model),
                        provider=str(kwargs.get("provider") or self.provider),
                    )
                    update_model_is_authoritative = False
        if (
            update_model_is_authoritative
            and not self._session_metadata_matches_active_runtime(kwargs)
        ):
            logger.warning(
                "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                str(kwargs.get("model") or ""),
                self.model,
            )
            self._update_model_pending_session_start = False
            return
        if "model" in kwargs:
            self.model = str(kwargs.get("model") or "")
        route_affects_context = "model" in kwargs or "provider" in kwargs
        for key in ("base_url", "api_key", "provider", "api_mode"):
            if key in kwargs:
                setattr(self, key, str(kwargs.get(key) or ""))
        if (
            "context_length" not in kwargs
            and route_affects_context
            and (self.raw_context_length or self.context_length)
        ):
            self._set_context_length(
                self.raw_context_length or self.context_length,
                source=self._context_length_source or "session_start",
                model=self.model,
                provider=self.provider,
            )
        self._update_model_pending_session_start = False

    def _continue_compression_boundary(
        self,
        session_id: str,
        old_session_id: str,
        kwargs: Dict[str, Any],
    ) -> None:
        previous_session_id = self._session_id
        requested_conversation_id = kwargs.get("conversation_id")
        session_state = self._lifecycle.get_by_session(old_session_id)
        conversation_state = self._lifecycle.get_by_conversation(old_session_id)

        def _state_conversation_matches(state: Any) -> bool:
            return bool(
                state
                and (
                    not requested_conversation_id
                    or state.conversation_id == requested_conversation_id
                )
            )

        def _has_summary_nodes(candidate_session_id: str | None) -> bool:
            return bool(candidate_session_id and self._dag.get_session_nodes(candidate_session_id))

        def _host_source_from_conversation_state(state: Any) -> tuple[str, Any]:
            if not _state_conversation_matches(state):
                return "", None
            if state.current_session_id == old_session_id and _has_summary_nodes(old_session_id):
                return old_session_id, state
            if (
                state.conversation_id == old_session_id
                and state.current_session_id
                and _has_summary_nodes(state.current_session_id)
            ):
                return state.current_session_id, state
            if (
                state.current_session_id is None
                and state.last_finalized_session_id
                and _has_summary_nodes(state.last_finalized_session_id)
            ):
                return state.last_finalized_session_id, state
            return "", None

        def _host_source_from_session_state(state: Any) -> tuple[str, Any]:
            if not _state_conversation_matches(state):
                return "", None
            if state.current_session_id == old_session_id and _has_summary_nodes(old_session_id):
                return old_session_id, state
            if (
                state.current_session_id is None
                and state.last_finalized_session_id == old_session_id
                and _has_summary_nodes(old_session_id)
            ):
                return old_session_id, state
            return "", None

        host_source_session_id, host_source_state = _host_source_from_conversation_state(
            conversation_state
        )
        if not host_source_session_id:
            host_source_session_id, host_source_state = _host_source_from_session_state(
                session_state
            )

        source_session_id = host_source_session_id or old_session_id
        source_state = host_source_state or session_state

        if previous_session_id and previous_session_id != old_session_id:
            # Hermes passes the session that actually crossed the compression
            # boundary as old_session_id. A different bound session can be a
            # short-lived subagent/cron/WebUI side channel that ran after the
            # foreground compaction. Prefer the host-authoritative source when
            # durable lifecycle + DAG evidence proves it belongs to LCM, then
            # fall back to the older bound-session recovery path. When the host
            # old_session_id is the durable conversation id, use that row's
            # current/finalized LCM source instead of unrelated auxiliary rows
            # where the id appears only as last_finalized_session_id.
            if host_source_session_id:
                logger.warning(
                    "LCM compression boundary using host old_session_id %s as carry-over source=%s despite bound session drift=%s",
                    old_session_id,
                    host_source_session_id,
                    previous_session_id,
                )
            else:
                bound_state = self._lifecycle.get_by_session(previous_session_id)
                bound_conversation_matches = bool(
                    bound_state
                    and (not self._conversation_id or bound_state.conversation_id == self._conversation_id)
                    and (
                        not requested_conversation_id
                        or bound_state.conversation_id == requested_conversation_id
                    )
                )
                bound_is_active_source = bool(
                    bound_state and bound_state.current_session_id == previous_session_id
                )
                bound_is_finalized_source = bool(
                    bound_state
                    and bound_state.current_session_id is None
                    and bound_state.last_finalized_session_id == previous_session_id
                )
                bound_has_summary_nodes = bool(self._dag.get_session_nodes(previous_session_id))
                if (
                    bound_conversation_matches
                    and (bound_is_active_source or bound_is_finalized_source)
                    and bound_has_summary_nodes
                ):
                    source_session_id = previous_session_id
                    source_state = bound_state
                    logger.warning(
                        "LCM compression boundary using bound session %s as carry-over source; host old_session_id=%s does not match",
                        previous_session_id,
                        old_session_id,
                    )
                else:
                    # Fallback: sibling chain with zero-DAG parent.
                    # When stale old_session_id has no DAG nodes AND the
                    # bound session belongs to a different conversation_id
                    # but shares the same last_finalized_session_id
                    # (parent) — prefer the bound session despite the
                    # conversation_id mismatch. This handles the lifecycle
                    # fork case where two sessions on the same channel
                    # received different conversation_ids.
                    bound_shares_parent_with_host = bool(
                        bound_state
                        and bound_state.last_finalized_session_id == old_session_id
                    )
                    host_has_no_dag = not bool(
                        self._dag.get_session_nodes(old_session_id)
                    )
                    if (
                        bound_shares_parent_with_host
                        and host_has_no_dag
                        and (bound_is_active_source or bound_is_finalized_source)
                        and bound_has_summary_nodes
                    ):
                        source_session_id = previous_session_id
                        source_state = bound_state
                        logger.warning(
                            "LCM compression boundary using bound session %s on sibling chain as carry-over source; host old_session_id=%s has zero DAG, parent=%s matches",
                            previous_session_id,
                            old_session_id,
                            bound_state.last_finalized_session_id,
                        )
                    else:
                        source_session_id = ""
                        source_state = None

        conversation_id = (
            (source_state.conversation_id if source_state else None)
            or kwargs.get("conversation_id")
            or self._conversation_id
            or source_session_id
            or old_session_id
            or session_id
        )
        process_local_frontier = (
            int(self._last_compacted_store_id or 0)
            if source_session_id and previous_session_id == source_session_id
            else 0
        )
        pending_reset_frontier = int(
            self._pending_reset_frontier_store_id
            if self._pending_reset_session_id
            and self._pending_reset_session_id == source_session_id
            else 0
        )
        frontier = max(
            process_local_frontier,
            int(source_state.current_frontier_store_id if source_state else 0),
            int(source_state.last_finalized_frontier_store_id if source_state else 0),
            pending_reset_frontier,
        )
        can_reassign = bool(
            source_session_id
            and session_id
            and source_session_id != session_id
        )
        boundary_placeholder_budget = {}
        boundary_placeholder_ordinals: dict[str, set[int]] = {}
        if can_reassign:
            if previous_session_id == source_session_id:
                boundary_placeholder_budget = self._active_replay_generated_placeholder_digest_budget()
                boundary_placeholder_ordinals = self._generated_placeholder_digest_ordinals_for_active_replay(
                    self._last_active_replay_messages
                )
            if not boundary_placeholder_budget:
                boundary_placeholder_budget = self._load_generated_ignored_placeholder_hash_counts(
                    self._session_scoped_hash_metadata_keys(
                        "ignored_active_replay_placeholder_hash_counts",
                        source_session_id,
                    )
                )
            if not boundary_placeholder_ordinals:
                boundary_placeholder_ordinals = self._load_generated_ignored_placeholder_hash_ordinals(
                    self._session_scoped_hash_metadata_keys(
                        "ignored_active_replay_placeholder_hash_ordinals",
                        source_session_id,
                    )
                )
            for digest, ordinals in boundary_placeholder_ordinals.items():
                boundary_placeholder_budget[digest] = max(
                    boundary_placeholder_budget.get(digest, 0),
                    len(ordinals),
                )
            self._compression_boundary_stored_placeholder_digest_counts = (
                self._stored_active_replay_placeholder_digest_counts(
                    source_session_id,
                    after_store_id=frontier,
                )
            )

        if can_reassign:
            self._lifecycle.finalize_session(
                conversation_id,
                source_session_id,
                frontier_store_id=frontier,
            )
            self._copy_generated_ignore_hashes_to_session(
                source_session_id,
                session_id,
                copy_dependent_content=True,
                source_frontier_store_id=frontier,
            )
            self._write_generated_ignored_placeholder_hash_counts(
                boundary_placeholder_budget,
                self._session_scoped_hash_metadata_keys(
                    "ignored_active_replay_placeholder_hash_counts",
                    session_id,
                ),
            )
            self._write_generated_ignored_placeholder_hash_ordinals(
                boundary_placeholder_ordinals,
                self._session_scoped_hash_metadata_keys(
                    "ignored_active_replay_placeholder_hash_ordinals",
                    session_id,
                ),
            )
            # Compression rollover carries derived context forward, but raw
            # messages remain owned by the session that produced them. Moving
            # raw rows here makes session-scoped transcript recovery report the
            # old/child session as missing even though its payload was only
            # reassigned to the next compression segment.
            moved_nodes = self._dag.reassign_session_nodes(source_session_id, session_id)
            logger.debug(
                "LCM compression boundary continued %s -> %s: carried %d DAG nodes; preserved raw message ownership",
                source_session_id,
                session_id,
                moved_nodes,
            )
        elif old_session_id:
            logger.warning(
                "LCM compression boundary skipped carry-over: old_session_id=%s does not match bound session=%s",
                old_session_id,
                previous_session_id,
            )
            self._finalize_pending_reset_boundary(previous_session_id)
            self._reset_session_scoped_runtime_state()
            self._last_boundary_skip_time = time.time()
            self._apply_session_start_metadata(session_id, kwargs)
            self._bind_lifecycle_state(
                session_id,
                conversation_id=kwargs.get("conversation_id"),
            )
            self._schedule_ingest_cursor_reconciliation()
            self._clear_pending_reset_boundary()
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
        self._clear_pending_reset_boundary()
        self._compression_boundary_ingest_pending = can_reassign
        self._compression_boundary_active_placeholder_digest_budget = boundary_placeholder_budget
        self._compression_boundary_active_placeholder_digest_ordinals = boundary_placeholder_ordinals
        self._log_session_filter_diagnostics()

    def on_session_start(self, session_id: str, **kwargs) -> None:
        if "hermes_home" in kwargs:
            self._rebind_storage_for_home(str(kwargs.get("hermes_home") or ""))

        boundary_reason = str(kwargs.get("boundary_reason") or "")
        old_session_id = str(kwargs.get("old_session_id") or "")
        previous_session_id = self._session_id
        self._lcm_current_start_allows_bypass_lineage = False
        requested_platform = str(kwargs.get("platform") or self._session_platform or "")
        pre_reset_preserve_ambiguous_no_frame_old_session = False
        if boundary_reason == "compression" and old_session_id and old_session_id != session_id:
            old_session_auxiliary_generation = self._in_process_auxiliary_caller_generation(
                old_session_id
            )
            new_session_auxiliary_parent = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
            )
            new_session_auxiliary_generation = self._in_process_auxiliary_caller_generation(session_id)
            with self._auxiliary_session_lock:
                active_old_auxiliary_generation = self._auxiliary_session_generations.get(
                    old_session_id
                )
                old_session_auxiliary_generation_is_stale = bool(
                    (
                        old_session_auxiliary_generation
                        and self._auxiliary_generation_is_retired(
                            old_session_id,
                            old_session_auxiliary_generation,
                        )
                    )
                    or (
                        new_session_auxiliary_generation
                        and self._auxiliary_generation_is_retired(
                            session_id,
                            new_session_auxiliary_generation,
                        )
                    )
                    or (
                        active_old_auxiliary_generation is not None
                        and (
                            (
                                old_session_auxiliary_generation
                                and active_old_auxiliary_generation != old_session_auxiliary_generation
                            )
                        )
                    )
                )
                old_session_has_retired_generation = bool(
                    self._auxiliary_retired_session_generations.get(old_session_id)
                )
            if old_session_auxiliary_generation_is_stale:
                logger.info(
                    "LCM ignored stale auxiliary compression boundary from %s to %s",
                    old_session_id,
                    session_id,
                )
                return
            pre_reset_preserve_ambiguous_no_frame_old_session = bool(
                active_old_auxiliary_generation is not None
                and not old_session_auxiliary_generation
                and old_session_id != self._session_id
                and old_session_has_retired_generation
                and old_session_id not in self._auxiliary_direct_end_guard_session_ids
                and new_session_auxiliary_parent != old_session_id
            )
        if self._host_fallback_compressor is not None and (
            self._host_fallback_session_id != session_id or requested_platform != self._session_platform
        ) and not (
            pre_reset_preserve_ambiguous_no_frame_old_session
            and self._host_fallback_session_id == old_session_id
        ):
            compressor = self._host_fallback_compressor
            fallback_session_id = self._host_fallback_session_id or previous_session_id
            on_session_end = getattr(compressor, "on_session_end", None)
            if callable(on_session_end) and fallback_session_id:
                try:
                    on_session_end(fallback_session_id, [])
                except Exception:
                    logger.debug("LCM host fallback compressor session-start reset failed", exc_info=True)
            on_session_reset = getattr(compressor, "on_session_reset", None)
            if callable(on_session_reset):
                try:
                    on_session_reset()
                except Exception:
                    logger.debug("LCM host fallback compressor reset failed", exc_info=True)
            self._host_fallback_compressor = None
            self._host_fallback_session_id = ""
        if boundary_reason == "compression" and old_session_id and old_session_id != session_id:
            old_session_is_suppressed_foreground = self._auxiliary_lineage_suppressed_as_foreground(
                old_session_id
            )
            old_session_auxiliary_generation = self._in_process_auxiliary_caller_generation(
                old_session_id
            )
            new_session_auxiliary_parent = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
            )
            new_session_auxiliary_generation = self._in_process_auxiliary_caller_generation(session_id)
            with self._auxiliary_session_lock:
                active_old_auxiliary_generation = self._auxiliary_session_generations.get(
                    old_session_id
                )
                old_session_auxiliary_generation_is_stale = bool(
                    (
                        old_session_auxiliary_generation
                        and self._auxiliary_generation_is_retired(
                            old_session_id,
                            old_session_auxiliary_generation,
                        )
                    )
                    or (
                        new_session_auxiliary_generation
                        and self._auxiliary_generation_is_retired(
                            session_id,
                            new_session_auxiliary_generation,
                        )
                    )
                    or (
                        active_old_auxiliary_generation is not None
                        and (
                            (
                                old_session_auxiliary_generation
                                and active_old_auxiliary_generation != old_session_auxiliary_generation
                            )
                        )
                    )
                )
                old_session_has_retired_generation = bool(
                    self._auxiliary_retired_session_generations.get(old_session_id)
                )
            new_session_auxiliary_parent = self._in_process_parent_session_id(
                {},
                session_id=session_id,
                include_explicit=False,
            )
            new_session_is_auxiliary_continuation = new_session_auxiliary_parent == old_session_id
            preserve_ambiguous_no_frame_old_session = bool(
                active_old_auxiliary_generation is not None
                and not old_session_auxiliary_generation
                and old_session_id != self._session_id
                and old_session_has_retired_generation
                and old_session_id not in self._auxiliary_direct_end_guard_session_ids
                and not new_session_is_auxiliary_continuation
            )
            if old_session_auxiliary_generation_is_stale:
                logger.info(
                    "LCM ignored stale auxiliary compression boundary from %s to %s",
                    old_session_id,
                    session_id,
                )
                return
            if (
                self._has_auxiliary_lineage_session(old_session_id)
                and not old_session_auxiliary_generation_is_stale
                and (
                    old_session_id != self._session_id
                    or old_session_auxiliary_generation
                    or new_session_is_auxiliary_continuation
                )
                and (
                    not old_session_is_suppressed_foreground
                    or old_session_auxiliary_generation
                    or new_session_is_auxiliary_continuation
                )
            ):
                self._handoff_auxiliary_session(
                    old_session_id,
                    session_id,
                    preserve_old_session=(
                        old_session_id == self._session_id
                        or preserve_ambiguous_no_frame_old_session
                    ),
                    preserve_old_foreground_marker=old_session_is_suppressed_foreground,
                )
                logger.info(
                    "LCM auxiliary session %s compressed to %s — keeping boundary stateless",
                    old_session_id,
                    session_id,
                )
                return
            if self._compression_boundary_from_lcm_bypassed_session(old_session_id):
                self._handoff_lcm_bypass_lineage(
                    old_session_id,
                    session_id,
                    new_platform=str(kwargs.get("platform") or ""),
                )
                self._clear_thread_context_stateless()
                if previous_session_id and previous_session_id != session_id:
                    self._finalize_pending_reset_boundary(previous_session_id)
                    self._reset_session_scoped_runtime_state()
                else:
                    self._clear_pending_reset_boundary()
                    self._ingest_cursor = 0
                    self._last_compacted_store_id = 0
                    self._last_overflow_recovery_failed = False
                    self._last_condensation_suppressed_reason = ""
                self._lcm_current_start_allows_bypass_lineage = True
                self._apply_session_start_metadata(session_id, kwargs)
                self._bind_lifecycle_state(
                    session_id,
                    conversation_id=kwargs.get("conversation_id"),
                )
                self._schedule_ingest_cursor_reconciliation()
                self._log_session_filter_diagnostics()
                logger.info(
                    "LCM compression boundary %s -> %s stayed stateless because the source session bypasses LCM storage",
                    old_session_id,
                    session_id,
                )
                return
            self._clear_thread_context_stateless()
            self._continue_compression_boundary(session_id, old_session_id, kwargs)
            return

        if self._is_live_auxiliary_child_session(session_id, previous_session_id, kwargs):
            explicit_parent_id = str(kwargs.get("parent_session_id") or "")
            preserve_foreground_reuse_marker = bool(
                (
                    explicit_parent_id
                    and self._lcm_session_last_bypassed.get(explicit_parent_id)
                )
                or self._lcm_session_last_normal_conversation_id.get(session_id)
            )
            if preserve_foreground_reuse_marker:
                if self._lcm_session_last_normal_conversation_id.get(session_id):
                    with self._auxiliary_session_lock:
                        self._auxiliary_foreground_reused_session_ids.add(session_id)
                self._mark_thread_context_stateless(
                    session_id,
                    preserve_foreground_reuse_marker=True,
                )
            else:
                self._register_auxiliary_session(session_id)
            logger.info(
                "LCM session %s is a live child of bound session %s — treating it as auxiliary/stateless",
                session_id,
                previous_session_id,
            )
            return
        start_platform = str(kwargs.get("platform") or "")
        side_channel_rebind = self._session_id_matches_lcm_bypass_filters(
            session_id,
            platform=start_platform,
        ) or self._has_lcm_bypass_lineage_session(session_id, platform=start_platform)
        self._unmark_thread_context_auxiliary_session(
            session_id,
            suppress_as_foreground_reuse=not side_channel_rebind,
        )
        self._clear_thread_context_stateless()
        if previous_session_id and previous_session_id != session_id:
            self._finalize_pending_reset_boundary(previous_session_id)
            self._reset_session_scoped_runtime_state()
        else:
            self._clear_pending_reset_boundary()
            self._ingest_cursor = 0
            self._last_compacted_store_id = 0
            self._last_overflow_recovery_failed = False
            self._last_condensation_suppressed_reason = ""
        self._apply_session_start_metadata(session_id, kwargs)
        self._bind_lifecycle_state(
            session_id,
            conversation_id=kwargs.get("conversation_id"),
        )
        self._schedule_ingest_cursor_reconciliation()
        self._log_session_filter_diagnostics()

    def _session_end_matches_current_store_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        prefix_count = self._session_end_store_prefix_count(session_id, messages)
        return prefix_count is not None and prefix_count > 0

    def _session_end_prefix_compare_value(self, value: Any, *, session_id: str) -> Any:
        if isinstance(value, dict):
            return {
                key: self._session_end_prefix_compare_value(child, session_id=session_id)
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [
                self._session_end_prefix_compare_value(child, session_id=session_id)
                for child in value
            ]
        if not isinstance(value, str):
            return value

        text = restore_ingest_payload_placeholders(
            value,
            config=self._config,
            hermes_home=self._hermes_home,
            session_id=session_id,
        )
        stripped = text.strip()
        ingest_refs = extract_ingest_externalized_refs(stripped)
        if (
            len(ingest_refs) == 1
            and stripped.startswith("[Externalized LCM ingest payload:")
            and stripped.endswith("]")
        ):
            payload = load_externalized_payload(
                ingest_refs[0],
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if payload is not None:
                payload_session_id = str(payload.get("session_id") or "")
                if not session_id or not payload_session_id or payload_session_id == session_id:
                    content = payload.get("content")
                    if isinstance(content, str):
                        return content

        if is_externalized_placeholder(stripped):
            ref = extract_externalized_ref(stripped)
            payload = load_externalized_payload(
                ref or "",
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if payload is not None:
                payload_session_id = str(payload.get("session_id") or "")
                if not session_id or not payload_session_id or payload_session_id == session_id:
                    content = payload.get("content")
                    if isinstance(content, str):
                        return content
        return text

    def _session_end_prefix_compare_content(
        self,
        message: Dict[str, Any],
        *,
        session_id: str,
    ) -> str:
        content = self._session_end_prefix_compare_value(
            (message or {}).get("content"),
            session_id=session_id,
        )
        content = redact_sensitive_value(
            content,
            self._config,
            parse_json_strings=False,
        )
        return normalize_content_value(content)

    def _session_end_prefix_compare_tool_calls(
        self,
        message: Dict[str, Any],
        *,
        session_id: str,
    ) -> str:
        tool_calls = self._session_end_prefix_compare_value(
            (message or {}).get("tool_calls"),
            session_id=session_id,
        )
        tool_calls = redact_sensitive_value(
            tool_calls,
            self._config,
            parse_json_strings=True,
        )
        if tool_calls is None or tool_calls == [] or tool_calls == {}:
            tool_calls = None
        return json.dumps(
            tool_calls,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _session_end_prefix_compare_identity(
        self,
        message: Dict[str, Any],
        *,
        session_id: str,
    ) -> tuple[str, str, str, str, str]:
        return (
            str((message or {}).get("role") or ""),
            self._session_end_prefix_compare_content(message, session_id=session_id),
            str((message or {}).get("tool_call_id") or ""),
            str((message or {}).get("tool_name") or ""),
            self._session_end_prefix_compare_tool_calls(message, session_id=session_id),
        )

    def _session_end_store_prefix_count(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> Optional[int]:
        try:
            stored_messages = self._store.get_range(
                session_id,
                limit=max(1, len(messages)),
                conversation_id=conversation_id,
            )
        except Exception:
            logger.debug("LCM session-end prefix check failed", exc_info=True)
            return None
        if not stored_messages:
            return 0
        if len(messages) < len(stored_messages):
            return None
        for idx, stored_msg in enumerate(stored_messages):
            msg = messages[idx]
            try:
                message_identity = self._session_end_prefix_compare_identity(
                    msg,
                    session_id=session_id,
                )
                stored_identity = self._session_end_prefix_compare_identity(
                    stored_msg,
                    session_id=session_id,
                )
            except Exception:
                logger.debug("LCM session-end prefix compare normalization failed", exc_info=True)
                return None
            if message_identity != stored_identity:
                return None
        return len(stored_messages)

    @staticmethod
    def _lcm_bypass_message_fingerprint(message: Dict[str, Any]) -> str:
        tool_calls = message.get("tool_calls")
        if tool_calls is None or tool_calls == [] or tool_calls == {}:
            tool_calls = None
        payload = {
            "role": message.get("role"),
            "content": normalize_content_value(message.get("content")),
            "tool_call_id": message.get("tool_call_id"),
            "tool_calls": tool_calls,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()

    def _remember_lcm_bypass_message_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        if not session_id or not messages:
            return
        fingerprints = [
            self._lcm_bypass_message_fingerprint(msg)
            for msg in messages[:_LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT]
        ]
        if fingerprints:
            remembered = self._lcm_bypass_message_prefix_fingerprints.setdefault(session_id, [])
            truncated = len(messages) > _LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT
            retained: list[tuple[list[str], bool]] = []
            for existing_fingerprints, existing_truncated in remembered:
                if existing_fingerprints == fingerprints:
                    truncated = truncated or bool(existing_truncated)
                    continue
                retained.append((existing_fingerprints, existing_truncated))
            retained.append((fingerprints, truncated))
            remembered[:] = retained

    def _remember_lcm_normal_message_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> None:
        if not session_id or not messages:
            return
        fingerprints = [
            self._lcm_bypass_message_fingerprint(msg)
            for msg in messages[:_LCM_MESSAGE_PREFIX_FINGERPRINT_LIMIT]
        ]
        if fingerprints:
            self._lcm_normal_message_prefix_fingerprints[
                self._lcm_normal_prefix_key(session_id, conversation_id=conversation_id)
            ] = fingerprints

    def _lcm_normal_prefix_key(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> tuple[str, str]:
        return (
            session_id,
            str(
                conversation_id
                or self._lcm_session_last_normal_conversation_id.get(session_id)
                or ""
            ),
        )

    def _messages_match_fingerprint_prefix(
        self,
        fingerprints: list[str],
        messages: List[Dict[str, Any]],
    ) -> bool:
        return self._matching_fingerprint_prefix_count(fingerprints, messages) > 0

    def _matching_fingerprint_prefix_count(
        self,
        fingerprints: list[str],
        messages: List[Dict[str, Any]],
    ) -> int:
        if not fingerprints or not messages:
            return 0
        compare_count = min(len(fingerprints), len(messages))
        if compare_count <= 0:
            return 0
        candidate = [self._lcm_bypass_message_fingerprint(msg) for msg in messages[:compare_count]]
        if candidate == fingerprints[:compare_count]:
            return compare_count
        return 0

    def _messages_match_lcm_bypass_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        return self._matching_lcm_bypass_prefix_count(session_id, messages) > 0

    def _matching_lcm_bypass_prefix_count(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> int:
        count, _truncated = self._matching_lcm_bypass_prefix_evidence(session_id, messages)
        return count

    def _matching_lcm_bypass_prefix_evidence(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> tuple[int, bool]:
        best_count = 0
        best_truncated = False
        for fingerprints, truncated in self._lcm_bypass_message_prefix_fingerprints.get(session_id, []):
            count = self._matching_fingerprint_prefix_count(fingerprints, messages)
            count_truncated = bool(truncated and count > 0 and count == len(fingerprints))
            if count > best_count:
                best_count = count
                best_truncated = count_truncated
            elif count == best_count:
                best_truncated = best_truncated or count_truncated
        return best_count, best_truncated

    def _messages_match_lcm_normal_prefix(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> bool:
        return self._matching_lcm_normal_prefix_count(
            session_id,
            messages,
            conversation_id=conversation_id,
        ) > 0

    def _matching_lcm_normal_prefix_count(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        conversation_id: str | None = None,
    ) -> int:
        return self._matching_fingerprint_prefix_count(
            self._lcm_normal_message_prefix_fingerprints.get(
                self._lcm_normal_prefix_key(session_id, conversation_id=conversation_id)
            ) or [],
            messages,
        )

    def _append_off_current_session_end_suffix(
        self,
        session_id: str,
        suffix: List[Dict[str, Any]],
        *,
        source: str,
        conversation_id: str,
    ) -> list[int]:
        if not session_id or not suffix:
            return []
        kept: list[Dict[str, Any]] = []
        for msg in suffix:
            if self._matches_ignore_message_patterns(msg):
                self._ignored_message_count += 1
                excerpt = (text_content_for_pattern_matching(msg.get("content")) or "")[:80].replace("\n", " ")
                logger.debug(
                    "LCM ignore_message_patterns dropped late session-end %s message: %r",
                    msg.get("role", "unknown"),
                    excerpt,
                )
                continue
            kept.append(msg)
        if not kept:
            return []
        protected_messages = protect_messages_for_ingest(
            kept,
            session_id=session_id,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        return self._store._append_protected_batch(
            session_id,
            protected_messages,
            [count_message_tokens(msg) for msg in protected_messages],
            source=source,
            conversation_id=conversation_id,
        )

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        ended_generation = self._in_process_auxiliary_caller_generation(session_id)
        active_auxiliary_end = session_id in self._active_auxiliary_session_ids()
        if (
            self._has_auxiliary_lineage_session(session_id)
            and session_id != self._session_id
            and (
                active_auxiliary_end
                or not self._auxiliary_lineage_suppressed_as_foreground(session_id)
                or ended_generation
                or (
                    session_id in self._auxiliary_last_prompt_tokens
                    and not self._auxiliary_lineage_suppressed_as_foreground(session_id)
                )
            )
        ):
            current_thread_session_id = self._thread_context_session_id()
            deactivated = self._deactivate_auxiliary_session(
                session_id,
                generation=ended_generation,
            )
            if deactivated:
                if current_thread_session_id == session_id or active_auxiliary_end:
                    self._remember_lcm_bypass_message_prefix(session_id, messages)
                self._end_host_fallback_compressor_for_session(
                    session_id,
                    messages,
                    current_session_bypasses=False,
                )
                if current_thread_session_id == session_id:
                    self._clear_thread_context_stateless(session_id)
            return
        current_session_bypasses = session_id == self._session_id and self._bypasses_lcm_context_management()
        ended_session_directly_bypasses = self._ended_session_directly_bypasses_lcm(session_id)
        direct_bypass_normal_conversation_id = self._lcm_session_last_normal_conversation_id.get(session_id)
        direct_bypass_normal_prefix_count = None
        if (
            session_id != self._session_id
            and self._auxiliary_lineage_suppressed_as_foreground(session_id)
            and direct_bypass_normal_conversation_id
            and not ended_generation
        ):
            direct_bypass_normal_prefix_count = self._session_end_store_prefix_count(
                session_id,
                messages,
                conversation_id=direct_bypass_normal_conversation_id,
            )
        direct_bypass_is_suppressed_reused_normal = (
            session_id != self._session_id
            and self._auxiliary_lineage_suppressed_as_foreground(session_id)
            and bool(direct_bypass_normal_conversation_id)
            and not ended_generation
            and session_id != self._thread_context_session_id()
            and direct_bypass_normal_prefix_count is not None
            and direct_bypass_normal_prefix_count > 0
        )
        if ended_session_directly_bypasses and not direct_bypass_is_suppressed_reused_normal:
            self._remember_lcm_bypass_message_prefix(session_id, messages)
            self._end_host_fallback_compressor_for_session(
                session_id,
                messages,
                current_session_bypasses=current_session_bypasses,
            )
            if session_id == self._thread_context_session_id():
                self._deactivate_auxiliary_session(session_id, generation=ended_generation)
                self._clear_thread_context_stateless(session_id)
            return
        same_id_has_bypass_lineage = (
            session_id == self._session_id
            and not current_session_bypasses
            and self._has_lcm_bypass_lineage_session(session_id)
        )
        same_id_normal_prefix_count = None
        same_id_recorded_normal_prefix_count = 0
        same_id_bypass_prefix_count = 0
        same_id_bypass_prefix_truncated = False
        if same_id_has_bypass_lineage:
            same_id_conversation_id = (
                self._conversation_id
                or self._lcm_session_last_normal_conversation_id.get(session_id)
                or None
            )
            (
                same_id_bypass_prefix_count,
                same_id_bypass_prefix_truncated,
            ) = self._matching_lcm_bypass_prefix_evidence(session_id, messages)
            same_id_normal_prefix_count = self._session_end_store_prefix_count(
                session_id,
                messages,
                conversation_id=same_id_conversation_id,
            )
            same_id_recorded_normal_prefix_count = self._matching_lcm_normal_prefix_count(
                session_id,
                messages,
                conversation_id=same_id_conversation_id,
            )
        same_id_store_prefix_positive = (
            same_id_normal_prefix_count is not None
            and same_id_normal_prefix_count > 0
        )
        same_id_strongest_normal_prefix_count = max(
            same_id_recorded_normal_prefix_count,
            same_id_normal_prefix_count if same_id_store_prefix_positive else 0,
        )
        same_id_truncated_bypass_prefix_ambiguous = (
            same_id_bypass_prefix_truncated
            and same_id_bypass_prefix_count > 0
            and same_id_strongest_normal_prefix_count >= same_id_bypass_prefix_count
            and len(messages) > same_id_bypass_prefix_count
        )
        same_id_matches_stronger_normal_prefix = (
            same_id_strongest_normal_prefix_count > 0
            and not same_id_truncated_bypass_prefix_ambiguous
            and (
                same_id_bypass_prefix_count <= 0
                or same_id_strongest_normal_prefix_count >= same_id_bypass_prefix_count
            )
        )
        off_current_auxiliary_reused_normal = (
            session_id != self._session_id
            and self._auxiliary_lineage_suppressed_as_foreground(session_id)
            and bool(direct_bypass_normal_conversation_id)
            and not ended_generation
        )
        off_current_lineage = (
            session_id != self._session_id
            and (
                self._has_lcm_bypass_lineage_session(session_id)
                or off_current_auxiliary_reused_normal
            )
        )
        off_current_normal_conversation_id = (
            self._lcm_session_last_normal_conversation_id.get(session_id)
            if off_current_lineage
            else ""
        )
        off_current_store_prefix_count = None
        off_current_recorded_prefix_count = 0
        off_current_bypass_prefix_count = 0
        off_current_bypass_prefix_truncated = False
        if off_current_lineage:
            (
                off_current_bypass_prefix_count,
                off_current_bypass_prefix_truncated,
            ) = self._matching_lcm_bypass_prefix_evidence(session_id, messages)
        if off_current_lineage and off_current_normal_conversation_id:
            off_current_store_prefix_count = self._session_end_store_prefix_count(
                session_id,
                messages,
                conversation_id=off_current_normal_conversation_id,
            )
            off_current_recorded_prefix_count = self._matching_lcm_normal_prefix_count(
                session_id,
                messages,
                conversation_id=off_current_normal_conversation_id,
            )
        off_current_prefix_count = None
        off_current_store_prefix_positive = (
            off_current_store_prefix_count is not None
            and off_current_store_prefix_count > 0
        )
        off_current_store_prefix_for_append = int(off_current_store_prefix_count or 0)
        off_current_recorded_prefix_for_append = 0
        if off_current_recorded_prefix_count > 0 and off_current_normal_conversation_id:
            try:
                stored_normal_rows = self._store.get_range(
                    session_id,
                    limit=off_current_recorded_prefix_count + 1,
                    conversation_id=off_current_normal_conversation_id,
                )
            except Exception:
                logger.debug("LCM off-current recorded-prefix row-count probe failed", exc_info=True)
                stored_normal_rows = []
            if len(stored_normal_rows) == off_current_recorded_prefix_count:
                off_current_recorded_prefix_for_append = off_current_recorded_prefix_count
        off_current_strongest_normal_prefix_count = max(
            off_current_store_prefix_for_append if off_current_store_prefix_positive else 0,
            off_current_recorded_prefix_for_append,
        )
        off_current_truncated_bypass_prefix_ambiguous = (
            off_current_bypass_prefix_truncated
            and off_current_bypass_prefix_count > 0
            and off_current_strongest_normal_prefix_count >= off_current_bypass_prefix_count
            and len(messages) > off_current_bypass_prefix_count
        )
        if (
            off_current_store_prefix_positive
            and not off_current_truncated_bypass_prefix_ambiguous
            and (
                off_current_bypass_prefix_count <= 0
                or off_current_store_prefix_for_append > off_current_bypass_prefix_count
            )
        ):
            off_current_prefix_count = off_current_store_prefix_for_append
        elif (
            off_current_recorded_prefix_for_append > 0
            and not off_current_truncated_bypass_prefix_ambiguous
            and (
                off_current_bypass_prefix_count <= 0
                or off_current_recorded_prefix_for_append > off_current_bypass_prefix_count
            )
        ):
            off_current_prefix_count = off_current_recorded_prefix_for_append
        if (
            off_current_lineage
            and not off_current_auxiliary_reused_normal
            and off_current_normal_conversation_id
            and off_current_store_prefix_count == 0
            and off_current_bypass_prefix_count <= 0
            and self._lcm_session_last_bypassed.get(session_id) is False
        ):
            off_current_prefix_count = 0
        same_id_should_bypass = (
            same_id_has_bypass_lineage
            and same_id_bypass_prefix_count > 0
            and not same_id_matches_stronger_normal_prefix
        )
        off_current_matches_bypass_prefix = (
            session_id != self._session_id
            and self._has_lcm_bypass_lineage_session(session_id)
            and off_current_bypass_prefix_count > 0
            and off_current_prefix_count is None
        )
        ended_lineage_bypasses = (
            session_id != self._session_id
            and self._has_lcm_bypass_lineage_session(session_id)
            and bool(self._lcm_session_last_bypassed.get(session_id))
            and not self._session_end_matches_current_store_prefix(session_id, messages)
        )
        off_current_should_bypass = off_current_lineage and off_current_prefix_count is None
        if off_current_prefix_count is not None:
            prefix_count = off_current_prefix_count
            suffix = messages[prefix_count:]
            if suffix:
                self._append_off_current_session_end_suffix(
                    session_id,
                    suffix,
                    source=(
                        self._lcm_session_last_normal_platform.get(session_id)
                        or self._lcm_session_last_platform.get(session_id, self._session_platform)
                    ),
                    conversation_id=off_current_normal_conversation_id,
                )
            try:
                state = self._lifecycle.get_by_conversation(off_current_normal_conversation_id)
                frontier_store_id = state.current_frontier_store_id if state is not None else 0
                self._lifecycle.finalize_session(
                    off_current_normal_conversation_id,
                    session_id,
                    frontier_store_id=frontier_store_id,
                )
            except Exception:
                logger.debug("LCM off-current session-end lifecycle finalization failed", exc_info=True)
            return
        if (
            current_session_bypasses
            or same_id_should_bypass
            or off_current_should_bypass
            or off_current_matches_bypass_prefix
            or ended_lineage_bypasses
        ):
            self._end_host_fallback_compressor_for_session(
                session_id,
                messages,
                current_session_bypasses=current_session_bypasses,
            )
            return
        try:
            with _temporary_sqlite_busy_timeout(
                [
                    getattr(self._store, "_conn", None),
                    getattr(self._lifecycle, "_conn", None),
                ],
                _SESSION_END_BUSY_TIMEOUT_MS,
            ):
                try:
                    # Best-effort final flush. Keep this path bounded because
                    # host gateways call session-end hooks from lifecycle paths
                    # that must not wait through SQLite's normal busy timeout.
                    self._ingest_messages(messages)
                except KeyboardInterrupt:
                    logger.warning(
                        "LCM session-end raw-message ingest interrupted; "
                        "final messages may be absent from the plugin-local store"
                    )
                    return
                except Exception as exc:
                    if _is_sqlite_locked_error(exc):
                        logger.warning(
                            "LCM session-end raw-message ingest skipped due to SQLite lock after short wait; "
                            "final messages may be absent from the plugin-local store: %s",
                            exc,
                        )
                        return
                    raise

                try:
                    self._lifecycle.finalize_session(
                        self._conversation_id,
                        session_id,
                        frontier_store_id=self._last_compacted_store_id,
                    )
                except KeyboardInterrupt:
                    logger.warning(
                        "LCM session-end lifecycle finalization interrupted; "
                        "raw messages may be ingested but lifecycle state may be finalized later"
                    )
                    return
                except Exception as exc:
                    if _is_sqlite_locked_error(exc):
                        logger.warning(
                            "LCM session-end lifecycle finalization skipped due to SQLite lock after short wait; "
                            "raw messages were ingested but lifecycle state may be finalized later: %s",
                            exc,
                        )
                        return
                    raise
        except KeyboardInterrupt:
            logger.warning("LCM session-end ingest/finalize interrupted before bounded flush completed")
            return
        except Exception as exc:
            if _is_sqlite_locked_error(exc):
                logger.warning(
                    "LCM session-end ingest/finalize skipped due to SQLite lock before bounded flush: %s",
                    exc,
                )
                return
            raise

    def on_session_reset(self) -> None:
        if self._host_fallback_compressor is not None:
            compressor = self._host_fallback_compressor
            on_session_reset = getattr(compressor, "on_session_reset", None)
            if callable(on_session_reset):
                try:
                    on_session_reset()
                except Exception:
                    logger.debug("LCM host fallback compressor reset failed", exc_info=True)
            self._host_fallback_compressor = None
            self._host_fallback_session_id = ""
        self._pending_reset_session_id = self._session_id
        self._pending_reset_conversation_id = self._conversation_id
        self._pending_reset_frontier_store_id = self._last_compacted_store_id
        super().on_session_reset()
        self._lifecycle.record_reset(self._conversation_id)
        self._reset_session_scoped_runtime_state()

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
        boundary_reason = str(kwargs.get("boundary_reason") or "")
        conversation_id = self._conversation_id or old_session_id or new_session_id
        bound_session_id = self._session_id
        can_carry_over = bool(
            old_session_id and bound_session_id and old_session_id == bound_session_id
        )

        if carry_over_context and boundary_reason == "compression" and old_session_id and old_session_id != new_session_id:
            before_node_ids = {node.node_id for node in self._dag.get_session_nodes(new_session_id)}
            if can_carry_over:
                self.on_session_end(old_session_id, previous_messages)
            else:
                logger.warning(
                    "LCM compression rollover old_session_id=%s does not match bound session=%s; using boundary handler fallback",
                    old_session_id,
                    bound_session_id,
                )
            self.on_session_start(
                new_session_id,
                old_session_id=old_session_id,
                **kwargs,
            )
            after_node_ids = {node.node_id for node in self._dag.get_session_nodes(new_session_id)}
            return len(after_node_ids - before_node_ids)

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
        return [
            LCM_GREP,
            LCM_LOAD_SESSION,
            LCM_DESCRIBE,
            LCM_EXPAND,
            LCM_EXPAND_QUERY,
            LCM_STATUS,
            LCM_INSPECT,
            LCM_DOCTOR,
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        # Ingest live messages if passed (enables current-turn search)
        messages = kwargs.get("messages")

        if name != "lcm_inspect" and messages and self._session_id and not (
            self._session_ignored or self._session_stateless or self._thread_context_stateless()
        ):
            try:
                self._ingest_messages(messages)
                self._record_ingest_success()
            except Exception as e:
                self._record_ingest_failure("tool-call ingest", e)

        handlers = {
            "lcm_grep": lcm_tools.lcm_grep,
            "lcm_load_session": lcm_tools.lcm_load_session,
            "lcm_describe": lcm_tools.lcm_describe,
            "lcm_expand": lcm_tools.lcm_expand,
            "lcm_expand_query": lcm_tools.lcm_expand_query,
            "lcm_status": lcm_tools.lcm_status,
            "lcm_inspect": lcm_tools.lcm_inspect,
            "lcm_doctor": lcm_tools.lcm_doctor,
        }
        handler = handlers.get(name)
        if handler:
            return handler(args, engine=self)
        return json.dumps({"error": f"Unknown LCM tool: {name}"})

    def _database_path_source(self) -> str:
        if self._config.database_path:
            return "config.database_path"
        if self._hermes_home:
            return "hermes_home"
        return "default_home"

    def get_runtime_identity(self) -> Dict[str, Any]:
        """Return operator-facing identity for the loaded LCM runtime.

        The public identity follows the same foreground-session view as
        ``lcm_status`` and other tools. When a side-channel session is bound,
        the bound session details are still exposed separately for diagnostics.
        """
        metadata = _plugin_metadata()
        git_identity = _git_runtime_identity(_PLUGIN_ROOT)
        session_id = self.current_session_id
        conversation_id = self.current_conversation_id
        lifecycle_state = None
        lifecycle_error = ""
        if conversation_id:
            try:
                lifecycle_state = self._lifecycle.get_by_conversation(conversation_id)
            except Exception as exc:  # pragma: no cover - defensive
                lifecycle_error = str(exc)

        identity: Dict[str, Any] = {
            "engine": self.name,
            "plugin_name": metadata.get("name", "hermes-lcm"),
            "plugin_version": metadata.get("version", "unknown"),
            "plugin_path": str(_PLUGIN_ROOT),
            "module_path": str(Path(__file__).resolve()),
            "hermes_home": str(self._hermes_home or ""),
            "database_path": str(self._store.db_path),
            "database_path_source": self._database_path_source(),
            "session_id": session_id,
            "session_platform": self.current_session_platform,
            "session_bound": bool(session_id),
            "conversation_id": conversation_id,
            "lifecycle_current_session_id": "",
            "lifecycle_last_finalized_session_id": "",
        }
        if self.side_channel_active:
            identity.update({
                "bound_session_id": self._session_id,
                "bound_session_platform": self._session_platform,
                "bound_conversation_id": self._conversation_id,
            })
        identity.update(git_identity)
        if lifecycle_state is not None:
            identity.update({
                "lifecycle_current_session_id": lifecycle_state.current_session_id or "",
                "lifecycle_last_finalized_session_id": lifecycle_state.last_finalized_session_id or "",
            })
        if lifecycle_error:
            identity["lifecycle_error"] = lifecycle_error
        return identity

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "compression_count": self.compression_count,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_completion_tokens": self.last_completion_tokens,
            "last_total_tokens": self.last_total_tokens,
            "last_input_tokens": self.last_input_tokens,
            "last_output_tokens": self.last_output_tokens,
            "last_cache_read_tokens": self.last_cache_read_tokens,
            "last_cache_write_tokens": self.last_cache_write_tokens,
            "last_reasoning_tokens": self.last_reasoning_tokens,
            "cache_metrics_available": self.cache_metrics_available,
            "cache_read_ratio": round(self.cache_read_ratio, 4),
            "raw_context_length": self.raw_context_length,
            "context_length": self.context_length,
            "effective_context_length_cap": self.effective_context_length_cap,
            "effective_context_length_reason": self.effective_context_length_reason,
            "threshold_tokens": self.threshold_tokens,
            "last_compression_status": self._last_compression_status,
            "last_compression_noop_reason": self._last_compression_noop_reason,
            "ingest_failure_count": self._ingest_failure_count,
            "consecutive_ingest_failures": self._consecutive_ingest_failures,
            "last_ingest_error": self._last_ingest_error,
            "last_ingest_error_time": self._last_ingest_error_time,
            "model": self.model,
            "provider": self.provider,
            "context_length_source": self._context_length_source,
            "configured_context_threshold": self._config.context_threshold,
            "context_threshold": self.context_threshold,
            "context_threshold_source": self._context_threshold_source,
            "context_threshold_autoraised": self._context_threshold_autoraised,
            "config_sources": dict(getattr(self._config, "config_sources", {}) or {}),
            "config_source_warnings": list(getattr(self._config, "config_source_warnings", []) or []),
            "ignored_config_yaml_lcm_keys": list(getattr(self._config, "ignored_config_yaml_lcm_keys", []) or []),
        })
        session_id = self.current_session_id
        conversation_id = self.current_conversation_id
        lifecycle_state = self._lifecycle.get_by_conversation(conversation_id) if conversation_id else None
        status["engine"] = "lcm"
        status["runtime_identity"] = self.get_runtime_identity()
        status["ingest_protection"] = sensitive_pattern_status(self._config)
        try:
            status["source_lineage"] = self._store.get_source_stats(session_id or None)
        except Exception as exc:  # pragma: no cover - defensive
            status["source_lineage"] = {"error": str(exc)}
        try:
            status["lifecycle_fragmentation"] = self._lifecycle.get_fragmentation_stats(
                state_db_path=self._state_db_path()
            )
        except Exception as exc:  # pragma: no cover - defensive
            status["lifecycle_fragmentation"] = {"error": str(exc), "read_only": True}
        try:
            rotate_backup_path = self.rotate_backup_path()
            status["rotate_backup_path"] = str(rotate_backup_path)
            # Single stat() to avoid a TOCTOU window where the rolling slot
            # could be atomically replaced between separate mtime and size reads.
            try:
                rotate_stat = rotate_backup_path.stat()
            except FileNotFoundError:
                rotate_stat = None
            if rotate_stat is not None:
                status["last_rotate_at"] = rotate_stat.st_mtime
                status["rotate_backup_size"] = rotate_stat.st_size
            else:
                status["last_rotate_at"] = None
                status["rotate_backup_size"] = 0
        except Exception as exc:  # pragma: no cover - defensive
            status["rotate_backup_path"] = None
            status["last_rotate_at"] = None
            status["rotate_backup_size"] = 0
            status["rotate_backup_error"] = str(exc)
        if session_id:
            status["store_messages"] = self._store.get_session_count(session_id)
            status["dag_nodes"] = self._dag.get_session_node_count(session_id)
            status["session_platform"] = self.current_session_platform
            status["session_ignored"] = self.current_session_ignored
            status["session_stateless"] = self.current_session_stateless
            status["ignore_session_patterns"] = list(self._config.ignore_session_patterns)
            status["stateless_session_patterns"] = list(self._config.stateless_session_patterns)
            status["ignore_message_patterns"] = list(self._config.ignore_message_patterns)
            status["ignore_session_patterns_source"] = self._config.ignore_session_patterns_source
            status["stateless_session_patterns_source"] = self._config.stateless_session_patterns_source
            status["ignore_message_patterns_source"] = self._config.ignore_message_patterns_source
            status["ignored_message_count"] = self._ignored_message_count
            status["ignore_pattern_dropped_count"] = self._ignore_pattern_dropped_count
            status["ingest_reconciliation"] = dict(self._last_ingest_reconciliation)
            status["overflow_recovery_failed"] = self._last_overflow_recovery_failed
            status["condensation_suppressed_reason"] = self._last_condensation_suppressed_reason
            status["conversation_id"] = conversation_id
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
            try:
                telemetry = self._store.read_compaction_telemetry(conversation_id)
            except Exception:
                telemetry = None
            if telemetry:
                status["compaction_telemetry"] = {
                    "cache_state": telemetry.get("cache_state", "unknown"),
                    "consecutive_cold_observations": telemetry.get(
                        "consecutive_cold_observations", 0
                    ),
                    "turns_since_leaf_compaction": telemetry.get(
                        "turns_since_leaf_compaction", 0
                    ),
                    "peak_prompt_tokens_since_leaf_compaction": telemetry.get(
                        "peak_prompt_tokens_since_leaf_compaction", 0
                    ),
                    "last_observed_prompt_tokens": telemetry.get(
                        "last_observed_prompt_tokens", 0
                    ),
                    "last_observed_cache_read": telemetry.get("last_observed_cache_read", 0),
                    "last_observed_cache_write": telemetry.get("last_observed_cache_write", 0),
                    "activity_band": telemetry.get("activity_band", "low"),
                    "total_compactions": telemetry.get("total_compactions", 0),
                    "last_leaf_compaction_at": telemetry.get("last_leaf_compaction_at"),
                    "last_compaction_duration_ms": telemetry.get("last_compaction_duration_ms"),
                    "provider": telemetry.get("provider"),
                    "model": telemetry.get("model"),
                    "last_api_call_at": telemetry.get("last_api_call_at"),
                }
        return status

    def update_model(self, model: str, context_length: int,
                     base_url: str = "", api_key: str = "",
                     provider: str = "",
                     api_mode: str = "") -> None:
        parent_session_id = self._in_process_parent_session_id({})
        if parent_session_id:
            logger.debug(
                "LCM model update ignored for auxiliary child of %s",
                parent_session_id,
            )
            return
        self.model = str(model or "")
        self.base_url = str(base_url or "")
        self.api_key = str(api_key or "")
        self.provider = str(provider or "")
        self.api_mode = str(api_mode or "")
        self._set_context_length(context_length, source="update_model")
        self._update_model_pending_session_start = True

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
            and (
                (
                    self._lcm_current_start_allows_bypass_lineage
                    and self._has_lcm_bypass_lineage_session(self._session_id, platform=self._session_platform)
                )
                or matches_session_pattern(
                    self._session_match_keys,
                    self._compiled_stateless_session_patterns,
                )
            )
        )
        if self._session_id:
            self._lcm_session_last_platform[self._session_id] = self._session_platform
            self._lcm_session_last_bypassed[self._session_id] = bool(self._session_ignored or self._session_stateless)
            if not self._session_ignored and not self._session_stateless:
                self._lcm_non_bypass_platforms.setdefault(self._session_id, set()).add(self._session_platform)
                self._lcm_session_last_normal_platform[self._session_id] = self._session_platform
        if self._session_ignored or self._session_stateless:
            self._mark_lcm_bypass_lineage_session(self._session_id, platform=self._session_platform)

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
            if self._config.ignore_message_patterns:
                logger.info(
                    "LCM ignore_message_patterns from %s: %s",
                    self._config.ignore_message_patterns_source,
                    ", ".join(self._config.ignore_message_patterns),
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

    def _schedule_ingest_cursor_reconciliation(self) -> None:
        """Mark existing-session rebinds for cursor repair on next ingest."""
        self._ingest_cursor_needs_reconcile = False
        if not self._session_id or self._session_ignored or self._session_stateless:
            return
        try:
            self._ingest_cursor_needs_reconcile = self._store.get_session_count(self._session_id) > 0
        except Exception as exc:  # pragma: no cover - defensive only
            logger.debug("LCM ingest cursor reconciliation probe failed: %s", exc)
            self._ingest_cursor_needs_reconcile = False

    def _stored_row_externalized_text_parts_for_pattern_matching(self, msg: Dict[str, Any]) -> list[str]:
        ref_sources: list[str] = []
        content = msg.get("content")
        if isinstance(content, str):
            ref_sources.append(content)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            try:
                ref_sources.append(json.dumps(tool_calls, ensure_ascii=False))
            except (TypeError, ValueError):
                ref_sources.append(str(tool_calls))
        refs: list[str] = []
        for source in ref_sources:
            for ref in extract_all_externalized_payload_refs(source):
                if ref not in refs:
                    refs.append(ref)
        parts: list[str] = []
        session_id = str(msg.get("session_id") or self._session_id or "")
        for ref in refs:
            payload = load_externalized_payload(
                ref,
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if not payload:
                continue
            payload_session_id = str(payload.get("session_id") or "")
            if session_id and payload_session_id and payload_session_id != session_id:
                continue
            payload_content = payload.get("content")
            if isinstance(payload_content, str):
                parts.append(payload_content)
        return parts

    def _stored_row_externalized_text_for_pattern_matching(self, msg: Dict[str, Any]) -> str:
        return "\n".join(self._stored_row_externalized_text_parts_for_pattern_matching(msg))

    def _is_cached_active_replay_message_at_index(self, idx: int, msg: Dict[str, Any]) -> bool:
        if idx < 0 or idx >= len(self._last_active_replay_messages):
            return False
        return self._message_replay_identity(msg) == self._message_replay_identity(
            self._last_active_replay_messages[idx]
        )

    def _matches_ignore_message_patterns(self, msg: Dict[str, Any], *, stored_row: bool = False) -> bool:
        if not self._compiled_ignore_message_patterns:
            return False
        content = msg.get("content")
        text = (
            stored_text_content_for_pattern_matching(content)
            if stored_row
            else text_content_for_pattern_matching(content)
        ) or ""
        if matches_message_pattern(text, self._compiled_ignore_message_patterns):
            return True
        if stored_row:
            externalized_parts = self._stored_row_externalized_text_parts_for_pattern_matching(msg)
            for externalized_text in externalized_parts:
                if externalized_text and matches_message_pattern(externalized_text, self._compiled_ignore_message_patterns):
                    return True
            externalized_text = "\n".join(externalized_parts)
            if externalized_text and externalized_text != text:
                return matches_message_pattern(externalized_text, self._compiled_ignore_message_patterns)
        return False

    def _content_has_externalized_placeholder_ref(self, content: str) -> bool:
        return bool(extract_externalized_ref(content) or extract_ingest_externalized_refs(content))

    def _has_prior_raw_externalized_placeholder_row(self, store_id: int, msg: Dict[str, Any]) -> bool:
        if not self._session_id:
            return False
        raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
        after_store_id = 0
        while True:
            rows = self._store.get_session_messages_after(
                self._session_id,
                after_store_id=after_store_id,
                limit=1000,
            )
            if not rows:
                return False
            for row in rows:
                row_store_id = int(row.get("store_id") or 0)
                if row_store_id >= store_id:
                    return False
                if self._raw_externalized_placeholder_replay_identity(row) == raw_identity:
                    return True
                after_store_id = max(after_store_id, row_store_id)

    def _mapped_stored_row_matches_ignore_message_patterns(self, msg: Dict[str, Any]) -> bool:
        store_id = msg.get("store_id")
        content = normalize_content_value(msg.get("content")) or ""
        has_externalized_placeholder = self._content_has_externalized_placeholder_ref(content)
        mapped_from_active_placeholder = False
        if store_id is None:
            store_id = self._current_compress_store_ids_by_message_id.get(id(msg))
            mapped_from_active_placeholder = has_externalized_placeholder and store_id is not None
        if store_id is None:
            return False
        if mapped_from_active_placeholder and self._has_prior_raw_externalized_placeholder_row(int(store_id), msg):
            raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
            if self._current_compress_placeholder_identity_counts.get(raw_identity, 0) <= 1:
                return False
        try:
            stored = self._store.get(int(store_id))
        except Exception:
            logger.debug("LCM stored ignore-pattern lookup failed", exc_info=True)
            return False
        return bool(stored and self._matches_ignore_message_patterns(stored, stored_row=True))

    def _copy_active_replay_messages_preserving_generated_ids(
        self,
        active_replay_messages: List[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        copied_replay_messages: list[Dict[str, Any]] = []
        generated_message_ids = getattr(
            self,
            "_generated_ignored_active_replay_placeholder_message_ids",
            set(),
        )
        for message in active_replay_messages:
            copied_message = dict(message)
            if id(message) in generated_message_ids:
                self._generated_ignored_active_replay_placeholder_message_ids.add(id(copied_message))
            copied_replay_messages.append(copied_message)
        return copied_replay_messages

    def _remember_active_replay_messages(
        self,
        original_messages: List[Dict[str, Any]],
        active_replay_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        self._last_active_replay_source_identities = [
            self._message_replay_identity(message) for message in original_messages
        ]
        self._last_active_replay_messages = self._copy_active_replay_messages_preserving_generated_ids(
            active_replay_messages
        )
        self._write_generated_ignored_placeholder_hash_counts(
            self._generated_placeholder_digest_budget_for_active_replay(active_replay_messages)
        )
        self._write_generated_ignored_placeholder_hash_ordinals(
            self._generated_placeholder_digest_ordinals_for_active_replay(active_replay_messages)
        )
        return active_replay_messages

    def _cached_active_replay_messages(
        self,
        original_messages: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        identities = [self._message_replay_identity(message) for message in original_messages]
        if identities == getattr(self, "_last_active_replay_source_identities", None):
            cached = getattr(self, "_last_active_replay_messages", None)
            if cached is not None:
                return self._copy_active_replay_messages_preserving_generated_ids(cached)
        return None

    def _is_replayed_context_scaffold_message(self, msg: Dict[str, Any]) -> bool:
        """Return true for active-context scaffolding that should not be re-ingested."""
        role = str(msg.get("role") or "")
        content = normalize_content_value(msg.get("content")) or ""
        if role == "system":
            return (
                "[Note: This conversation uses Lossless Context Management (LCM)." in content
                and "Earlier turns have been compacted into hierarchical summaries below." in content
            )
        if content.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX):
            return True
        if "[Expand for details:" not in content:
            return False
        return bool(
            re.search(
                r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]",
                content,
            )
        )

    def _restore_ingest_payload_placeholders_in_value(self, value: Any, *, session_id: str) -> Any:
        if isinstance(value, dict):
            return {
                self._restore_ingest_payload_placeholders_in_value(key, session_id=session_id)
                if isinstance(key, str)
                else key: self._restore_ingest_payload_placeholders_in_value(val, session_id=session_id)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [self._restore_ingest_payload_placeholders_in_value(item, session_id=session_id) for item in value]
        if isinstance(value, str):
            return restore_ingest_payload_placeholders(
                value,
                config=self._config,
                hermes_home=self._hermes_home,
                session_id=session_id,
            )
        return value

    def _restore_ingest_payload_placeholders_in_content_identity(self, content: str, *, session_id: str) -> str:
        if not content:
            return content
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return restore_ingest_payload_placeholders(
                content,
                config=self._config,
                hermes_home=self._hermes_home,
                session_id=session_id,
            )
        restore_as_structured = False
        if isinstance(decoded, (dict, list)) and normalize_content_value(decoded) == content:
            for ref in extract_ingest_externalized_refs(content):
                payload = load_externalized_payload(
                    ref,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                payload_session_id = (payload or {}).get("session_id") or ""
                if session_id and payload_session_id and payload_session_id != session_id:
                    continue
                field_path = str((payload or {}).get("field_path") or "")
                if field_path and field_path != "content":
                    restore_as_structured = True
                    break
        if restore_as_structured:
            restored = self._restore_ingest_payload_placeholders_in_value(decoded, session_id=session_id)
            return normalize_content_value(restored) or ""
        return restore_ingest_payload_placeholders(
            content,
            config=self._config,
            hermes_home=self._hermes_home,
            session_id=session_id,
        )

    def _recovered_content_matches_durable_identity(self, recovered_content: str, durable_content: str) -> bool:
        recovered_identity_content = normalize_content_value(
            redact_sensitive_value(
                recovered_content,
                self._config,
                parse_json_strings=False,
            )
        )
        if recovered_identity_content == durable_content:
            return True
        redaction_names = sorted(set(re.findall(r"\[LCM sensitive redaction: name=([^;\]]+)", durable_content)))
        if not redaction_names or bool(getattr(self._config, "sensitive_patterns_enabled", False)):
            return False
        compat_config = copy.copy(self._config)
        compat_config.sensitive_patterns_enabled = True
        compat_config.sensitive_patterns = redaction_names
        compat_identity_content = normalize_content_value(
            redact_sensitive_value(
                recovered_content,
                compat_config,
                parse_json_strings=False,
            )
        )
        return compat_identity_content == durable_content

    @staticmethod
    def _persisted_output_marker_replay_proof(content: str) -> tuple[str | None, bool]:
        inline_preview_sha256 = _persisted_output_inline_preview_sha256(content)
        preview_sha256 = inline_preview_sha256 or _persisted_output_preview_prefix_digest(content)
        if not preview_sha256:
            return None, False
        allow_redacted_preview_match = inline_preview_sha256 is None and not _has_lossy_sensitive_redaction(content)
        return preview_sha256, allow_redacted_preview_match

    def _has_any_durable_persisted_output_payload_for_marker(self, msg: Dict[str, Any]) -> bool:
        role = str(msg.get("role") or "unknown")
        content = normalize_content_value(msg.get("content")) or ""
        if role != "tool" or not _is_hermes_persisted_output_marker(content):
            return False
        expected_chars = _expected_persisted_output_chars(content)
        persisted_output_source_path = _persisted_output_saved_path(content)
        persisted_output_preview_sha256, allow_redacted_preview_match = self._persisted_output_marker_replay_proof(content)
        if expected_chars is None or not persisted_output_source_path or not persisted_output_preview_sha256:
            return False
        if recover_hermes_persisted_output_with_file_stat(content) is None:
            return False
        durable_content = find_externalized_tool_result_content_for_call(
            tool_call_id=str(msg.get("tool_call_id") or ""),
            session_id=str(msg.get("session_id") or self._session_id or ""),
            expected_chars=expected_chars,
            persisted_output_source_path=persisted_output_source_path,
            persisted_output_preview_sha256=persisted_output_preview_sha256,
            allow_redacted_preview_match=allow_redacted_preview_match,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        return durable_content is not None

    @classmethod
    def _is_active_context_droppable_identity(cls, identity: tuple[str, str, str, str]) -> bool:
        """Return true for durable rows sanitized out of active replay only."""
        role, content, _tool_call_id, tool_calls = identity
        if role != "assistant" or tool_calls:
            return False
        return _should_drop_active_assistant_message({
            "role": role,
            "content": cls._identity_content_for_active_cleanup(content),
        })

    def _ignored_message_is_quarantinable_assistant(self, msg: Dict[str, Any]) -> bool:
        if self._is_volatile_ignored_quarantine_placeholder(
            msg,
            text_content_for_pattern_matching(msg.get("content")) or "",
        ):
            return True
        identity = self._message_replay_identity(msg)
        if self._is_quarantined_assistant_replay_identity(identity):
            return True
        if not self._matches_ignore_message_patterns(msg):
            return False
        if identity[0] != "assistant":
            return False
        content = normalize_content_value(msg.get("content")) or ""
        return assistant_output_quarantine_reason(content) is not None

    def _redact_active_replay_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        redacted_replay_messages: list[Dict[str, Any]] = []
        generated_message_ids = getattr(
            self,
            "_generated_ignored_active_replay_placeholder_message_ids",
            set(),
        )
        for message in messages:
            redacted_message = dict(message)
            if "content" in redacted_message:
                redacted_content = redact_sensitive_value(
                    redacted_message.get("content"),
                    self._config,
                    parse_json_strings=False,
                )
                redacted_message["content"] = redacted_content

            if "tool_calls" in redacted_message:
                redacted_message["tool_calls"] = redact_sensitive_value(
                    redacted_message.get("tool_calls"),
                    self._config,
                    parse_json_strings=True,
                )
            if id(message) in generated_message_ids:
                self._generated_ignored_active_replay_placeholder_message_ids.add(id(redacted_message))
            redacted_replay_messages.append(redacted_message)
        return redacted_replay_messages

    def _ingest_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Persist new messages to the store.

        Uses a cursor to track which portion of the current messages list
        has already been persisted.  After compress() shortens the list,
        the cursor is reset to len(compressed), so only messages appended
        after compaction are ingested — regardless of how the store count
        compares to the current list length.

        Returns a replay-safe copy of ``messages`` with obviously broken
        assistant loops replaced by quarantine placeholders. Existing callers may
        ignore the return value when they only need durable persistence.
        """
        if not self._session_id:
            logger.debug("Ingest skipped: no session_id")
            return self._redact_active_replay_messages(messages)

        if self._session_ignored or self._session_stateless:
            logger.debug(
                "Ingest skipped for %s session %s",
                "ignored" if self._session_ignored else "stateless",
                self._session_id,
            )
            return self._redact_active_replay_messages(messages)

        n = len(messages)
        cursor = min(max(self._ingest_cursor, 0), n)
        scan_start = 0 if self._ingest_cursor_needs_reconcile else cursor
        ignored_original_messages = [False] * n
        if self._compiled_ignore_message_patterns:
            previous_store_id_map = self._current_compress_store_ids_by_message_id
            self._current_compress_store_ids_by_message_id = self._get_store_id_map_for_messages(messages)
            try:
                for idx in range(scan_start, n):
                    mapped_ignore = self._mapped_stored_row_matches_ignore_message_patterns(messages[idx])
                    ignored_original_messages[idx] = (
                        self._matches_ignore_message_patterns(messages[idx])
                        or mapped_ignore
                    )
            finally:
                self._current_compress_store_ids_by_message_id = previous_store_id_map
        externalize_messages = [False] * n
        prefer_existing_externalized = [False] * n
        for idx in range(scan_start, n):
            externalize_messages[idx] = not ignored_original_messages[idx]
        for idx in range(0, scan_start):
            prefer_existing_externalized[idx] = not ignored_original_messages[idx]
        replay_messages = quarantine_suspicious_assistant_messages(
            messages,
            session_id=self._session_id,
            config=self._config,
            hermes_home=self._hermes_home,
            externalize=externalize_messages,
            prefer_existing_externalized=prefer_existing_externalized,
        )
        replay_messages = self._redact_active_replay_messages(replay_messages)
        replay_messages = self._apply_ignored_active_replay_placeholders(
            messages,
            replay_messages,
            scan_start=scan_start,
            ignored_messages=ignored_original_messages,
        )
        if self._ingest_cursor_needs_reconcile:
            reconcile_messages = [
                original_msg
                if (
                    (
                        str(original_msg.get("role") or "") == "tool"
                        and _is_hermes_persisted_output_marker(
                            normalize_content_value(original_msg.get("content")) or ""
                        )
                        and self._has_any_durable_persisted_output_payload_for_marker(original_msg)
                    )
                    or (
                        self._compiled_ignore_message_patterns
                        and ignored_original_messages[idx]
                    )
                )
                else replay_msg
                for idx, (original_msg, replay_msg) in enumerate(zip(messages, replay_messages))
            ]
            self._ingest_cursor = self._reconcile_ingest_cursor_from_store(reconcile_messages)
            self._ingest_cursor_needs_reconcile = False
        cursor = min(max(self._ingest_cursor, 0), n)
        if cursor > 0:
            cached_source_identities = getattr(self, "_last_active_replay_source_identities", None)
            cached_active_replay_messages = getattr(self, "_last_active_replay_messages", None)
            if (
                cached_source_identities is not None
                and cached_active_replay_messages is not None
                and len(cached_source_identities) >= cursor
                and len(cached_active_replay_messages) >= cursor
            ):
                current_prefix_identities = [
                    self._message_replay_identity(message) for message in messages[:cursor]
                ]
                if current_prefix_identities == cached_source_identities[:cursor]:
                    replay_messages = (
                        self._copy_active_replay_messages_preserving_generated_ids(
                            cached_active_replay_messages[:cursor]
                        )
                        + replay_messages[cursor:]
                    )
        logger.debug(
            "Ingest: session=%s cursor=%d incoming=%d",
            self._session_id, cursor, n,
        )

        new_messages = replay_messages[cursor:] if cursor < n else []
        original_new_messages = messages[cursor:] if cursor < n else []

        if not new_messages:
            cached_replay = self._cached_active_replay_messages(messages)
            self._compression_boundary_ingest_pending = False
            self._compression_boundary_active_placeholder_digest_budget = {}
            self._compression_boundary_active_placeholder_digest_ordinals = {}
            self._compression_boundary_stored_placeholder_digest_counts = {}
            if cached_replay is not None:
                return cached_replay
            return self._remember_active_replay_messages(messages, replay_messages)

        active_replay_messages = replay_messages
        compression_boundary_ingest_pending = self._compression_boundary_ingest_pending
        empty_session_placeholder_budget: dict[str, int] = {}
        empty_session_placeholder_ordinals: dict[str, set[int]] = {}
        if not compression_boundary_ingest_pending and self._session_id:
            try:
                if self._store.get_session_count(self._session_id) == 0:
                    empty_session_placeholder_budget = self._load_generated_ignored_placeholder_hash_counts()
                    empty_session_placeholder_ordinals = self._load_generated_ignored_placeholder_hash_ordinals()
            except Exception:
                empty_session_placeholder_budget = {}
                empty_session_placeholder_ordinals = {}
        messages_to_store_with_index: list[tuple[int, Dict[str, Any]]] = [
            (cursor + offset, replay_msg)
            for offset, replay_msg in enumerate(new_messages)
        ]
        if messages_to_store_with_index:
            kept: list[tuple[int, Dict[str, Any]]] = []
            boundary_placeholder_seen: dict[str, int] = {}
            boundary_seen_synthetic_summary_before = False
            empty_session_placeholder_seen: dict[str, int] = {}
            if empty_session_placeholder_ordinals and cursor > 0:
                for replay_msg in replay_messages[:cursor]:
                    replay_text = text_content_for_pattern_matching(replay_msg.get("content")) or ""
                    digest = self._active_replay_placeholder_digest(replay_text)
                    if digest:
                        empty_session_placeholder_seen[digest] = empty_session_placeholder_seen.get(digest, 0) + 1
            boundary_all_placeholder_replay_batch = (
                compression_boundary_ingest_pending
                and len(new_messages) > 1
                and all(
                    self._is_ignored_active_replay_placeholder(
                        msg,
                        text_content_for_pattern_matching(msg.get("content")) or "",
                    )
                    for msg in new_messages
                )
            )
            if compression_boundary_ingest_pending:
                boundary_budget = self._compression_boundary_active_placeholder_digest_budget
                stored_counts = self._compression_boundary_stored_placeholder_digest_counts
                if boundary_budget and stored_counts:
                    incoming_counts: dict[str, int] = {}
                    relevant_digests = set(boundary_budget) | set(stored_counts)
                    for msg in new_messages:
                        text = text_content_for_pattern_matching(msg.get("content")) or ""
                        digest = self._active_replay_placeholder_digest(text)
                        if digest in relevant_digests:
                            incoming_counts[digest] = incoming_counts.get(digest, 0) + 1
                    adjusted_budget: dict[str, int] = {}
                    for digest, count in boundary_budget.items():
                        parsed_count = max(0, int(count or 0))
                        incoming_count = max(0, int(incoming_counts.get(digest, 0) or 0))
                        stored_count = max(0, int(stored_counts.get(digest, 0) or 0))
                        remaining = min(parsed_count, max(0, incoming_count - stored_count))
                        if remaining > 0:
                            adjusted_budget[digest] = remaining
                    self._compression_boundary_active_placeholder_digest_budget = adjusted_budget
            empty_session_all_placeholder_replay_batch = (
                bool(empty_session_placeholder_ordinals)
                and len(new_messages) > 1
                and all(
                    self._is_ignored_active_replay_placeholder(
                        msg,
                        text_content_for_pattern_matching(msg.get("content")) or "",
                    )
                    for msg in new_messages
                )
            )
            for offset, (original_msg, replay_msg) in enumerate(zip(original_new_messages, new_messages)):
                absolute_idx = cursor + offset
                replay_text = text_content_for_pattern_matching(replay_msg.get("content")) or ""
                original_text = text_content_for_pattern_matching(original_msg.get("content")) or ""
                volatile_placeholder = self._is_volatile_ignored_quarantine_placeholder(
                    replay_msg,
                    replay_text,
                )
                volatile_digest = self._active_replay_placeholder_digest(replay_text)
                generated_volatile_placeholder = volatile_placeholder and (
                    original_text != replay_text
                    or (
                        volatile_digest is not None
                        and volatile_digest in self._load_generated_ignored_placeholder_hashes()
                    )
                )
                active_replay_placeholder = self._is_ignored_active_replay_placeholder(replay_msg, replay_text)
                active_replay_placeholder_digest = self._active_replay_placeholder_digest(replay_text)
                if not active_replay_placeholder:
                    replay_text_stripped = replay_text.strip()
                    if (
                        self._is_context_summary_content(replay_text)
                        or replay_text_stripped.startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX)
                        or replay_text_stripped.startswith(_PRESERVED_TODO_CONTEXT_PREFIX)
                    ):
                        boundary_seen_synthetic_summary_before = True
                compression_carried_active_placeholder = False
                metadata_replayed_active_placeholder = False
                if (
                    empty_session_placeholder_budget
                    and empty_session_placeholder_ordinals
                    and active_replay_placeholder
                    and active_replay_placeholder_digest is not None
                ):
                    empty_session_placeholder_seen[active_replay_placeholder_digest] = (
                        empty_session_placeholder_seen.get(active_replay_placeholder_digest, 0) + 1
                    )
                    ordinal = empty_session_placeholder_seen[active_replay_placeholder_digest]
                    remaining = empty_session_placeholder_budget.get(active_replay_placeholder_digest, 0)
                    if (
                        remaining > 0
                        and ordinal in empty_session_placeholder_ordinals.get(
                            active_replay_placeholder_digest,
                            set(),
                        )
                        and (ordinal > 1 or empty_session_all_placeholder_replay_batch)
                    ):
                        metadata_replayed_active_placeholder = True
                        if remaining == 1:
                            empty_session_placeholder_budget.pop(active_replay_placeholder_digest, None)
                        else:
                            empty_session_placeholder_budget[active_replay_placeholder_digest] = remaining - 1
                if (
                    compression_boundary_ingest_pending
                    and active_replay_placeholder
                    and active_replay_placeholder_digest is not None
                ):
                    boundary_placeholder_seen[active_replay_placeholder_digest] = (
                        boundary_placeholder_seen.get(active_replay_placeholder_digest, 0) + 1
                    )
                    current_placeholder_ordinal = boundary_placeholder_seen[active_replay_placeholder_digest]
                    boundary_budget = self._compression_boundary_active_placeholder_digest_budget
                    boundary_ordinals = self._compression_boundary_active_placeholder_digest_ordinals
                    generated_message_ids = getattr(
                        self,
                        "_generated_ignored_active_replay_placeholder_message_ids",
                        set(),
                    )
                    has_generated_provenance = (
                        id(replay_msg) in generated_message_ids
                        or id(original_msg) in generated_message_ids
                    )
                    ordinal_matches_generated = (
                        current_placeholder_ordinal in boundary_ordinals.get(
                            active_replay_placeholder_digest,
                            set(),
                        )
                        and (
                            current_placeholder_ordinal > 1
                            or boundary_seen_synthetic_summary_before
                            or boundary_all_placeholder_replay_batch
                        )
                    )
                    if boundary_budget and (
                        has_generated_provenance
                        or (not has_generated_provenance and ordinal_matches_generated)
                    ):
                        remaining = boundary_budget.get(active_replay_placeholder_digest, 0)
                        if remaining > 0:
                            compression_carried_active_placeholder = True
                            if remaining == 1:
                                boundary_budget.pop(active_replay_placeholder_digest, None)
                            else:
                                boundary_budget[active_replay_placeholder_digest] = remaining - 1
                replayed_active_placeholder = active_replay_placeholder and (
                    self._is_cached_active_replay_message_at_index(absolute_idx, replay_msg)
                    or compression_carried_active_placeholder
                    or metadata_replayed_active_placeholder
                )
                if (
                    ignored_original_messages[absolute_idx]
                    or generated_volatile_placeholder
                    or replayed_active_placeholder
                ):
                    self._ignored_message_count += 1
                    if generated_volatile_placeholder and volatile_digest is not None:
                        self._remember_generated_ignored_placeholder_hash(volatile_digest)
                    replay_preserves_ignore_decision = (
                        self._is_volatile_ignored_quarantine_placeholder(replay_msg, replay_text)
                        or self._is_ignored_active_replay_placeholder(replay_msg, replay_text)
                    )
                    if ignored_original_messages[absolute_idx] and not replay_preserves_ignore_decision:
                        if active_replay_messages is replay_messages:
                            active_replay_messages = self._copy_active_replay_messages_preserving_generated_ids(
                                replay_messages
                            )
                        active_message = dict(active_replay_messages[absolute_idx])
                        active_message["content"] = self._ignored_active_replay_placeholder(original_text)
                        active_replay_messages[absolute_idx] = active_message
                    excerpt = original_text[:80].replace("\n", " ")
                    if ignored_original_messages[absolute_idx]:
                        # A raw message matched ignore_message_patterns and is
                        # discarded here - never persisted anywhere. Count and
                        # log it (INFO) so an over-broad pattern silently eating
                        # substantive turns is at least visible to the operator.
                        self._ignore_pattern_dropped_count += 1
                        logger.info(
                            "LCM ignore_message_patterns dropped %s message "
                            "(not persisted; total dropped=%d): %r",
                            original_msg.get("role", "unknown"),
                            self._ignore_pattern_dropped_count,
                            excerpt,
                        )
                    else:
                        logger.debug(
                            "LCM ignore_message_patterns dropped %s message: %r",
                            original_msg.get("role", "unknown"),
                            excerpt,
                        )
                    continue
                store_msg = replay_msg
                if (
                    str(original_msg.get("role") or "") == "tool"
                    and _is_hermes_persisted_output_marker(
                        normalize_content_value(original_msg.get("content")) or ""
                    )
                ):
                    store_msg = original_msg
                kept.append((absolute_idx, store_msg))
            messages_to_store_with_index = kept

        if not messages_to_store_with_index:
            self._ingest_cursor = n
            self._compression_boundary_ingest_pending = False
            self._compression_boundary_active_placeholder_digest_budget = {}
            self._compression_boundary_active_placeholder_digest_ordinals = {}
            self._compression_boundary_stored_placeholder_digest_counts = {}
            return self._remember_active_replay_messages(messages, active_replay_messages)

        protected_messages = protect_messages_for_ingest(
            [msg for _idx, msg in messages_to_store_with_index],
            session_id=self._session_id,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        for (absolute_idx, _replay_msg), protected_msg in zip(
            messages_to_store_with_index,
            protected_messages,
        ):
            if self._protected_message_uses_raw_payload_active_stub(protected_msg):
                if active_replay_messages is replay_messages:
                    active_replay_messages = self._copy_active_replay_messages_preserving_generated_ids(
                        replay_messages
                    )
                active_message = dict(active_replay_messages[absolute_idx])
                active_message["content"] = protected_msg["content"]
                active_replay_messages[absolute_idx] = active_message

        estimates = [count_message_tokens(m) for m in protected_messages]
        self._store._append_protected_batch(
            self._session_id,
            protected_messages,
            estimates,
            source=self._session_platform,
            conversation_id=self._conversation_id,
        )
        self._ingest_cursor = n
        self._compression_boundary_ingest_pending = False
        self._compression_boundary_active_placeholder_digest_budget = {}
        self._compression_boundary_active_placeholder_digest_ordinals = {}
        self._compression_boundary_stored_placeholder_digest_counts = {}
        logger.debug("Ingested %d messages into LCM store", len(messages_to_store_with_index))
        # Most ``protected_messages`` changes are storage-only: inline media,
        # tool results, and data/base64 substrings must stay provider-usable in
        # active replay. Whole-message ``raw_payload`` externalization is the
        # exception: it intentionally returns a compact active stub so the host
        # does not replay huge opaque text while SQLite stores only the stub.
        return self._remember_active_replay_messages(messages, active_replay_messages)

    @staticmethod
    def _protected_message_uses_raw_payload_active_stub(message: Dict[str, Any]) -> bool:
        content = message.get("content")
        return isinstance(content, str) and content.startswith(
            "[Externalized payload: kind=raw_payload;"
        )

    def _get_store_ids_for_messages(self, messages: List[Dict[str, Any]]) -> List[int]:
        ids_by_message_id = self._get_store_id_map_for_messages(messages)
        return [ids_by_message_id[id(msg)] for msg in messages if id(msg) in ids_by_message_id]

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

            # Only take the fast ref-branch when the ENTIRE row is the
            # externalized placeholder. A ref merely embedded in surrounding
            # text (e.g. a recall-tool result that quotes a placeholder) must
            # fall through to the content-equality lookup below, which tombstones
            # only when the full row content matches the stored payload -
            # otherwise the surrounding, never-externalized text is lost.
            ref = extract_externalized_ref(content) if is_externalized_placeholder(content) else None
            if ref:
                externalized = load_externalized_payload(
                    ref,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                if externalized is not None and externalized.get("kind", "tool_result") == "tool_result":
                    placeholder = build_transcript_gc_placeholder(externalized)
                    self._store.gc_externalized_tool_result(store_id, placeholder)
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
        matched_tool_ids = _matched_tool_call_ids(messages)
        for msg in messages:
            role = msg.get("role", "unknown")
            content = redact_sensitive_value(
                msg.get("content") or "",
                self._config,
                parse_json_strings=False,
            )
            if role == "tool":
                tool_id = str(msg.get("tool_call_id") or "").strip()
                externalized = maybe_externalize_tool_output(
                    content,
                    tool_call_id=tool_id,
                    session_id=self._session_id,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                if externalized:
                    content = externalized["placeholder"]
                else:
                    content = sanitize_pre_compaction_content(content)
                    if len(content) > 3000:
                        content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            content = sanitize_pre_compaction_content(content)

            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                matched_tool_calls = [
                    tc for tc in tool_calls
                    if not _tool_call_id(tc) or _tool_call_id(tc) in matched_tool_ids
                ]
                if _is_synthetic_assistant_noise(content):
                    if not matched_tool_calls:
                        continue
                    content = ""
                if len(content) > 3000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                if matched_tool_calls:
                    tc_parts = []
                    for tc in matched_tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = fn.get("arguments", "")
                            args = redact_sensitive_value(
                                args,
                                self._config,
                                parse_json_strings=True,
                            )
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

    # -- Internal: tool-pair sanitization ------------------------------------

    def _sanitize_active_context_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        insert_missing_tool_stubs: bool = True,
    ) -> List[Dict[str, Any]]:
        """Drop unsafe assistant-only noise, then repair tool sequencing.

        This is intentionally active-context-only: callers pass the selected
        provider replay context, and this helper never mutates stored rows,
        source mappings, or DAG nodes.
        """
        cleaned: list[Dict[str, Any]] = []
        dropped_assistant_messages = 0
        stripped_assistant_messages = 0
        for msg in messages:
            msg = self._sanitize_active_preserved_objective_message(msg)
            if msg.get("role") == "assistant":
                cleaned_msg = _clean_active_assistant_message(msg)
                if cleaned_msg is None:
                    dropped_assistant_messages += 1
                    continue
                if cleaned_msg is not msg:
                    stripped_assistant_messages += 1
                cleaned.append(cleaned_msg)
                continue
            cleaned.append(msg)

        if dropped_assistant_messages:
            logger.info(
                "LCM active-context cleanup: dropped %d assistant message(s) with no visible content",
                dropped_assistant_messages,
            )
        if stripped_assistant_messages:
            logger.info(
                "LCM active-context cleanup: stripped internal content from %d assistant message(s)",
                stripped_assistant_messages,
            )

        return self._sanitize_tool_pairs(
            cleaned,
            insert_missing_tool_stubs=insert_missing_tool_stubs,
        )

    def _sanitize_tool_pairs(
        self,
        messages: List[Dict[str, Any]],
        *,
        insert_missing_tool_stubs: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return provider-safe active-context tool-call/result sequencing.

        Raw store and DAG history remain lossless. This guardrail only sanitizes
        the active context emitted back to providers, where assistant tool calls
        must be followed immediately by their contiguous tool results. Late,
        duplicate, out-of-order, and orphan tool results are dropped; missing
        direct results get synthetic stubs.
        """
        sanitized: List[Dict[str, Any]] = []
        dropped_tool_results = 0
        inserted_stub_results = 0

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.get("role") == "tool":
                dropped_tool_results += 1
                i += 1
                continue

            sanitized.append(msg)

            if msg.get("role") == "assistant":
                expected_ids = [
                    call_id
                    for call_id in (_tool_call_id(tool_call) for tool_call in (msg.get("tool_calls") or []))
                    if call_id
                ]

                for expected_id in expected_ids:
                    matched_direct_result = False
                    while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                        next_msg = messages[i + 1]
                        next_id = str(next_msg.get("tool_call_id") or "").strip()
                        if next_id == expected_id:
                            sanitized.append(next_msg)
                            i += 1
                            matched_direct_result = True
                            break
                        dropped_tool_results += 1
                        i += 1

                    if not matched_direct_result and insert_missing_tool_stubs:
                        sanitized.append({
                            "role": "tool",
                            "content": "[Result from earlier conversation — see context summary above]",
                            "tool_call_id": expected_id,
                        })
                        inserted_stub_results += 1

                while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                    dropped_tool_results += 1
                    i += 1

            i += 1

        if dropped_tool_results:
            logger.info(
                "LCM tool-pair guardrail: dropped %d late/orphan/duplicate tool result(s)",
                dropped_tool_results,
            )
        if inserted_stub_results:
            logger.info(
                "LCM tool-pair guardrail: inserted %d missing tool-result stub(s)",
                inserted_stub_results,
            )

        return sanitized

    # -- Internal: condensation --------------------------------------------

    def _should_allow_follow_on_condensation(
        self,
        *,
        uncondensed_count: int,
        leaf_compacted_this_turn: bool,
        force_overflow: bool,
        critical_budget_pressure: bool = False,
    ) -> tuple[bool, str]:
        if not leaf_compacted_this_turn:
            return True, ""
        if not self._config.cache_friendly_condensation_enabled:
            return True, ""
        if force_overflow:
            return True, ""
        if critical_budget_pressure:
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
        critical_budget_pressure: bool = False,
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
                critical_budget_pressure=critical_budget_pressure,
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
                fallback_models=self._config.summary_fallback_models,
                circuit_breaker=self._summary_circuit_breaker,
                spend_guard=self._summary_spend_guard,
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

    @staticmethod
    def _append_lcm_note_to_content(content: Any) -> Any:
        note = (
            "\n\n[Note: This conversation uses Lossless Context Management (LCM). "
            "Earlier turns have been compacted into hierarchical summaries below. "
            "Use lcm_grep to search history, lcm_describe to inspect the DAG, "
            "and lcm_expand to recover original details from any summary.]"
        )
        if isinstance(content, str):
            return content + note
        note_part = {"type": "text", "text": note.lstrip()}
        if content is None:
            return note.lstrip()
        if isinstance(content, list):
            return list(content) + [note_part]
        normalized = normalize_content_value(content) or ""
        return normalized + note

    @staticmethod
    def _is_preserved_todo_context_message(message: Dict[str, Any]) -> bool:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        return content.lstrip().startswith(_PRESERVED_TODO_CONTEXT_PREFIX)

    @staticmethod
    def _preserved_objective_context_content(message: Dict[str, Any]) -> str:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        return content if content.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX) else ""

    def _sanitized_preserved_objective_context_content(self, message: Dict[str, Any]) -> str:
        preserved_objective = self._preserved_objective_context_content(message)
        if not preserved_objective:
            return ""
        return self._sanitize_preserved_objective_content(
            preserved_objective,
            role=str(message.get("role") or "user"),
        )

    def _sanitize_active_preserved_objective_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        sanitized_content = self._sanitized_preserved_objective_context_content(message)
        if not sanitized_content or sanitized_content == message.get("content"):
            return message
        sanitized = dict(message)
        sanitized["content"] = sanitized_content
        return sanitized

    def _sanitize_preserved_objective_content(self, content: str, role: str = "user") -> str:
        content = strip_injected_context_blocks(content)
        content = protect_inline_payloads_in_text(
            content,
            role=role,
            session_id=self._session_id,
            field_path="preserved_objective.content",
            config=self._config,
            hermes_home=self._hermes_home,
        )
        return content

    def _build_preserved_objective_summary_part(self, message: Dict[str, Any]) -> str:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        content = self._sanitize_preserved_objective_content(
            content,
            role=str(message.get("role") or "user"),
        )
        return f"{_PRESERVED_OBJECTIVE_CONTEXT_PREFIX}\n{content}"

    def _latest_user_context_anchor(
        self,
        messages: List[Dict[str, Any]],
        selected_tail: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Return a scaffolded newest real user objective omitted from the tail.

        Tool-heavy turns can push the operative user request outside the fresh
        tail while retaining only assistant/tool traces from that turn.  The
        returned text is active-context scaffolding, not raw conversation: it is
        emitted inside the summary block so restart reconciliation ignores it
        instead of ingesting a duplicate non-contiguous user message.

        Previous preserved-objective scaffolds are derived context, not real
        user turns, so they are not eligible as the next anchor source. Once a
        reverse scan reaches one, older user turns are stale relative to that
        synthetic continuity marker and must not be promoted as current intent.
        """
        selected_tail_messages = [msg for msg in selected_tail if isinstance(msg, dict)]
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            content_text = text_content_for_pattern_matching(message.get("content")) or ""
            if (
                self._matches_ignore_message_patterns(message)
                or self._mapped_stored_row_matches_ignore_message_patterns(message)
                or self._is_volatile_ignored_quarantine_placeholder(
                    message,
                    content_text,
                )
                or self._is_ignored_active_replay_placeholder(message, content_text)
            ):
                continue
            if self._preserved_objective_context_content(message):
                return None
            if message.get("role") != "user":
                continue
            if self._is_preserved_todo_context_message(message):
                continue
            if any(message == selected for selected in selected_tail_messages):
                return None
            return self._build_preserved_objective_summary_part(message)
        return None

    def _assemble_context(
        self,
        system_msg: Optional[Dict[str, Any]],
        tail_messages: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
        include_lcm_note: bool = True,
    ) -> List[Dict[str, Any]]:
        """Build the active context from DAG summaries + fresh tail.

        Structure:
          [leading anchor, normally system prompt]
          [highest-depth summary nodes first, then lower]
          [fresh tail messages]
        """
        result = []

        # Leading anchor with optional LCM annotation. Only a true system prompt
        # is a safe permanent anchor; gateway sessions can start directly with
        # user messages, and those user turns must remain compactable.
        leading_msg = system_msg.copy() if system_msg is not None else None
        if leading_msg is not None:
            if (
                leading_msg.get("role") == "system"
                and self.compression_count == 0
                and include_lcm_note
            ):
                leading_msg["content"] = self._append_lcm_note_to_content(
                    leading_msg.get("content", "")
                )
            result.append(leading_msg)

        assembly_cap = (
            assembly_cap_override
            if assembly_cap_override is not None
            else self._effective_assembly_token_cap()
        )

        tail_selected = tail_messages
        anchor_source = getattr(self, "_pending_context_anchor_messages", None)
        if anchor_source is None:
            anchor_source = tail_messages
        anchor_part: Optional[str] = None
        summary_budget = None
        if assembly_cap is not None:
            used = count_message_tokens(leading_msg) if leading_msg is not None else 0
            kept_tail_reversed: list[Dict[str, Any]] = []
            tail_token_total = 0
            tail_for_selection = self._sanitize_active_context_messages(
                tail_messages,
                insert_missing_tool_stubs=False,
            )
            skipped_tail_gap = False
            for msg in reversed(tail_for_selection):
                msg_tokens = count_message_tokens(msg)
                if used + tail_token_total + msg_tokens > assembly_cap:
                    if self._is_budget_droppable_tail_message(msg):
                        skipped_tail_gap = True
                        continue
                    break
                if skipped_tail_gap:
                    break
                kept_tail_reversed.append(msg)
                tail_token_total += msg_tokens
            tail_selected = list(reversed(kept_tail_reversed))
            summary_budget = max(0, assembly_cap - used - tail_token_total)
        if anchor_source is not None:
            anchor_part = self._latest_user_context_anchor(anchor_source, tail_selected)

        # Collect DAG summaries — highest depth first for context hierarchy
        summary_parts: list[str] = []
        last_role = result[-1].get("role", "system") if result else "system"
        if not result or result[-1].get("role") == "system":
            # The summary becomes the first provider-visible message: either no
            # leading anchor exists (gateway-style assembly) or the system
            # prompt is the only anchor, which Anthropic extracts into a
            # separate field. Either way messages[0] must be role "user"; an
            # assistant summary here is rejected with HTTP 400 after the second
            # compaction.
            summary_role = "user"
        else:
            summary_role = "assistant" if last_role != "assistant" else "user"
        if anchor_part is not None:
            anchor_msg = {"role": summary_role, "content": anchor_part}
            if summary_budget is None or count_message_tokens(anchor_msg) <= summary_budget:
                summary_parts.append(anchor_part)

        all_nodes = self._dag.get_session_nodes(self._session_id)
        if all_nodes:
            # Group by depth, take the most recent uncondensed at each level
            # For active context, we want the highest-level summaries
            # that haven't been condensed into even higher levels
            depths = sorted(set(n.depth for n in all_nodes), reverse=True)
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
            selected_parts = summary_parts
            if summary_budget is not None:
                selected_parts = []
                for part in summary_parts:
                    candidate = "\n\n---\n\n".join(selected_parts + [part])
                    candidate_msg = {"role": summary_role, "content": candidate}
                    if count_message_tokens(candidate_msg) > summary_budget:
                        if part == anchor_part:
                            continue
                        continue
                    selected_parts.append(part)
            if selected_parts:
                combined = "\n\n---\n\n".join(selected_parts)
                result.append({"role": summary_role, "content": combined})

        # Fresh tail
        result.extend(tail_selected)

        # ── Active-context cleanup / tool-pair guardrail ──
        # Drop assistant turns that carry only blank/internal structured content,
        # then ensure provider-valid tool-call/result sequencing.
        result = self._sanitize_active_context_messages(result)
        if leading_msg is None:
            while result and result[0].get("role") in {"assistant", "tool"}:
                result = result[1:]
        if (
            assembly_cap is not None
            and anchor_part is not None
            and count_messages_tokens(result) > assembly_cap
        ):
            trimmed_result: list[Dict[str, Any]] = []
            for msg in result:
                content = normalize_content_value(msg.get("content")) or ""
                if _PRESERVED_OBJECTIVE_CONTEXT_PREFIX not in content:
                    trimmed_result.append(msg)
                    continue
                parts = [
                    part for part in content.split("\n\n---\n\n")
                    if not part.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX)
                ]
                if parts:
                    trimmed = msg.copy()
                    trimmed["content"] = "\n\n---\n\n".join(parts)
                    trimmed_result.append(trimmed)
            result = self._sanitize_active_context_messages(trimmed_result)

        return result

    def _is_budget_droppable_tail_message(self, message: Dict[str, Any]) -> bool:
        """Return whether an over-budget tail message may be evicted.

        User turns are prompt-bearing context and stop tail selection when they
        cannot fit. Assistant/tool turns are derived context; if one bulky turn
        blocks older prompt material, skip it and keep scanning for budgetable
        user intent or compact status that still fits.
        """
        role = message.get("role")
        if role not in {"assistant", "tool"}:
            return False
        content = normalize_content_value(message.get("content")) or ""
        if _PRESERVED_TODO_CONTEXT_PREFIX in content:
            return False
        if _PRESERVED_OBJECTIVE_CONTEXT_PREFIX in content:
            return False
        return True

    def _finalize_forced_overflow_result(
        self,
        original_messages: List[Dict[str, Any]],
        compressed: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if compressed != original_messages:
            self._last_compression_status = "overflow_recovery"
            self._last_compression_noop_reason = ""
            self._ingest_cursor = len(compressed)
            self._ingest_cursor_needs_reconcile = False
            logger.info(
                "LCM assembly guardrail recovery: %d messages → %d (no new summary node)",
                len(original_messages),
                len(compressed),
            )
        else:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = (
                "forced overflow recovery found no droppable active-context messages"
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
        system_msg: Optional[Dict[str, Any]],
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
                    for msg in (candidate[1:] if system_msg is not None else candidate)
                ):
                    return candidate

        candidate = self._assemble_context(
            system_msg,
            tail_messages,
            assembly_cap_override=assembly_cap_override,
            include_lcm_note=False,
        )
        minimum_candidate_len = 1 if system_msg is not None else 0
        if len(candidate) == minimum_candidate_len and tail_messages:
            fallback = ([system_msg] if system_msg is not None else []) + [tail_messages[-1]]
            return self._sanitize_active_context_messages(fallback)
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

    def _derive_auto_focus_topic(
        self,
        messages: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Infer a compact focus hint from the most recent real user turns.

        Walks the message list backwards, collecting up to
        ``_AUTO_FOCUS_MAX_TURNS`` user messages (skipping context summaries
        and empty turns).  Returns a brief text block suitable for injection
        into the summarizer prompt as ``focus_topic``.

        IMPORTANT: The ``messages`` parameter must be ``working_messages``
        (output of ``_ingest_messages``), not raw messages.  ``working_messages``
        has already been redacted by ``_redact_active_replay_messages``.

        As an additional safety layer, text extracted by
        ``text_content_for_pattern_matching`` is run through
        ``redact_sensitive_text`` with the active config.  This covers
        sensitive values that ``_redact_active_replay_messages`` misses
        (e.g., dict/JSON token content deserialized into text,
        bearer-style auth text that survived structured-content flattening).

        Mirrors Hermes upstream ``ContextCompressor._derive_auto_focus_topic``
        from ``fix/compression-auto-focus-topic``.
        """
        candidates: list[str] = []
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            # Skip context compaction summaries — they are synthetic, not
            # real user intent.
            if self._is_context_summary_content(content):
                continue
            text = (text_content_for_pattern_matching(content) or "").strip()
            if self._matches_ignore_message_patterns(msg) or self._is_volatile_ignored_quarantine_placeholder(
                msg,
                text,
            ) or self._is_ignored_active_replay_placeholder(msg, text):
                continue
            # Additional redaction safety net: run extracted text through the
            # configured redaction path.  _redact_active_replay_messages uses
            # parse_json_strings=False for content, so structured content
            # (dict/JSON tokens, bearer-style auth text) may not be fully
            # covered.  This extra pass ensures the same redaction rules apply
            # to whatever text is extracted for the focus topic.
            text = redact_sensitive_text(text, self._config)
            if not text:
                continue
            text = " ".join(text.split())
            if len(text) > _AUTO_FOCUS_TURN_MAX_CHARS:
                text = text[: _AUTO_FOCUS_TURN_MAX_CHARS - 1].rstrip() + "…"
            candidates.append(text)
            if len(candidates) >= _AUTO_FOCUS_MAX_TURNS:
                break

        if not candidates:
            return None

        candidates.reverse()
        focus = "Recent user focus:\n" + "\n".join(f"- {item}" for item in candidates)
        if len(focus) > _AUTO_FOCUS_MAX_CHARS:
            focus = focus[: _AUTO_FOCUS_MAX_CHARS - 1].rstrip() + "…"
        return focus

    @staticmethod
    def _is_context_summary_content(content: Any) -> bool:
        """Check whether message content is a synthetic context summary.

        Only checks string content — LCM/ Hermes compression summaries are
        always stored as plain strings, never as structured multimodal parts.
        """
        if not isinstance(content, str):
            return False
        return (
            "CONTEXT COMPACTION" in content
            or "CONTEXT SUMMARY" in content
            or "Earlier turns have been compacted" in content
            or "Earlier turns were compacted" in content
        )

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

    # -- Rotate ------------------------------------------------------------

    def backup_dir(self) -> Path:
        """Return the directory where LCM backup snapshots are written.

        Centralized so the timestamped ``/lcm backup`` slot and the rolling
        ``/lcm rotate apply`` slot share the same directory derivation.
        """
        db_path = Path(self._store.db_path)
        backup_root = (
            Path(self._hermes_home).expanduser()
            if getattr(self, "_hermes_home", "")
            else db_path.parent
        )
        return backup_root / "backups" / "lcm"

    def rotate_backup_path(self) -> Path:
        """Return the rolling rotate-latest SQLite backup path for this engine.

        Centralized so command.py (which writes the backup) and get_status()
        (which reads its mtime to surface last_rotate_at) cannot drift.
        """
        db_path = Path(self._store.db_path)
        return self.backup_dir() / f"{db_path.stem}-rotate-latest.sqlite3"

    def rotate_active_session(
        self,
        *,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Compact the active session in-place without changing identity.

        Read-only by default (``apply=False``). Returns a preview describing
        what would change. When ``apply=True``, advances the lifecycle frontier
        marker past the pre-tail raw messages so they are no longer replayed
        into active context on subsequent bootstrap. Raw messages remain in
        the SQLite store and are recoverable through ``lcm_load_session`` and
        ``lcm_expand`` — the lossless raw recovery contract is preserved.

        Refuses on sessions that are unbound, ignored, or stateless.

        Two frontier markers are intentionally kept separate:

        - The **persisted lifecycle frontier**
          (``lifecycle_state.current_frontier_store_id``) is the
          bootstrap signal — on next session start, raw rows at or
          below it are not replayed into the active context. Rotate
          advances this marker.
        - The **in-process source-mapping marker**
          (``self._last_compacted_store_id``) tracks raw rows that the
          *current process* has already moved into summary DAG nodes.
          ``_get_store_ids_for_messages`` uses it to filter candidates
          when mapping in-memory active messages back to ``store_id``.
          Rotate deliberately does NOT advance this marker: pre-tail
          raw messages remain in the in-memory active context until
          the host rebuilds it, so a normal ``compress()`` later in
          the same process can still summarize them with correct
          ``source_ids`` lineage. On next process start,
          ``_bind_lifecycle_state`` reads the persisted frontier into
          the in-process marker — at that point the active context is
          being built from scratch, so the contract holds.

        Refusal/no-op reason codes (returned as ``reason``):

        - ``no_active_session``: engine has no bound session or conversation.
        - ``session_ignored``: foreground session matched
          ``LCM_IGNORE_SESSION_PATTERNS``.
        - ``session_stateless``: foreground session matched
          ``LCM_STATELESS_SESSION_PATTERNS``.
        - ``no_pre_tail_content``: total stored messages do not exceed
          ``fresh_tail_count``; nothing to rotate.
        - ``empty_tail``: tail query returned no rows despite a non-zero
          count (concurrent deletion race); rotate cannot compute a boundary.
        - ``frontier_already_ahead``: lifecycle frontier is already at or
          past the proposed new frontier; rotate is a no-op.
        - ``stale_lifecycle_state``: apply requested but lifecycle's
          ``current_session_id`` did not match this engine's session, so
          ``advance_frontier`` did not persist the change.
        """
        session_id = self._session_id
        conversation_id = self._conversation_id

        if not session_id or not conversation_id:
            return {"ok": False, "reason": "no_active_session"}
        if self._session_ignored:
            return {"ok": False, "reason": "session_ignored", "session_id": session_id}
        if self._session_stateless:
            return {"ok": False, "reason": "session_stateless", "session_id": session_id}

        fresh_tail_count = max(1, int(self._config.fresh_tail_count))
        total_count = int(self._store.get_session_count(session_id))

        state = self._lifecycle.get_by_conversation(conversation_id)
        current_frontier = int(state.current_frontier_store_id) if state else 0

        base = {
            "ok": True,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "total_message_count": total_count,
            "fresh_tail_count": fresh_tail_count,
            "current_frontier_store_id": current_frontier,
            "mode": "apply" if apply else "preview",
        }

        if total_count <= fresh_tail_count:
            return {
                **base,
                "noop": True,
                "reason": "no_pre_tail_content",
                "pre_tail_message_count": 0,
                "new_frontier_store_id": current_frontier,
            }

        tail = self._store.get_session_tail(session_id, fresh_tail_count)
        if not tail:
            # Concurrent deletion can empty the tail after the count check.
            # Surface the same shape callers expect for any other no-op so
            # downstream formatters can render it without KeyError.
            return {
                **base,
                "noop": True,
                "reason": "empty_tail",
                "pre_tail_message_count": 0,
                "new_frontier_store_id": current_frontier,
            }

        smallest_tail_store_id = int(tail[0].get("store_id") or 0)
        new_frontier = max(0, smallest_tail_store_id - 1)
        pre_tail_count = max(0, total_count - len(tail))

        is_noop = new_frontier <= current_frontier
        result = {
            **base,
            "pre_tail_message_count": pre_tail_count,
            "new_frontier_store_id": new_frontier,
            "noop": is_noop,
        }
        if is_noop:
            # Set the reason for both preview and apply so downstream
            # formatters can render a stable explanation. Preview previously
            # omitted the reason, which left _rotate_apply_text's preflight
            # check unable to distinguish frontier-already-ahead from other
            # no-ops.
            result["reason"] = "frontier_already_ahead"

        if not apply:
            return result

        if is_noop:
            return result

        new_state = self._lifecycle.advance_frontier(
            conversation_id,
            session_id,
            new_frontier,
        )
        # advance_frontier silently returns the unchanged state when its
        # session_id check fails (lifecycle_state.py:557-559). Detect that
        # by checking whether the persisted frontier actually advanced; only
        # promote the in-process marker on a confirmed persist.
        persisted_frontier = (
            int(new_state.current_frontier_store_id) if new_state else current_frontier
        )
        if persisted_frontier < new_frontier:
            return {
                **{k: v for k, v in result.items() if k != "ok"},
                "ok": False,
                "noop": False,
                "reason": "stale_lifecycle_state",
                "applied_frontier_store_id": persisted_frontier,
            }
        # Deliberately do NOT touch self._last_compacted_store_id here.
        # The in-process source-mapping marker must stay aligned with the
        # in-memory active context the host is still using. Pre-tail raw
        # messages remain in that active context until the host rebuilds
        # it; advancing the marker would make
        # _get_store_ids_for_messages filter out those rows on the next
        # in-process compress(), producing summary nodes whose text
        # covers pre-rotate messages but whose source_ids reference only
        # post-rotate rows. The persisted lifecycle frontier we just
        # advanced is the bootstrap signal for the next process start,
        # where _bind_lifecycle_state will read it into the marker
        # against a freshly-built active context.
        result["applied_frontier_store_id"] = persisted_frontier
        return result

    # -- Lifecycle ---------------------------------------------------------

    def shutdown(self):
        self._unregister_active_engine_binding()
        self._store.close()
        self._dag.close()
        self._lifecycle.close()
