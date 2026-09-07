"""Session-scoped runtime-state resets.

Extracted verbatim from :mod:`hermes_lcm.engine` as ``ResetStateMixin``
(WS5 seam). The methods clear the session-scoped counters, compaction
progress, and per-turn placeholder-boundary bookkeeping when a session is
reset or rolled over. State stays on the engine (accessed via ``self``);
mixing this in leaves every call site and ``self._*`` reference unchanged.
"""

import uuid


class ResetStateMixin:
    def _reset_session_counters(self) -> None:
        """Reset session-scoped counters and token tracking.

        Safe to call on boundary skip because it does not affect compaction progress.
        """
        self.compression_count = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_reasoning_tokens = 0
        self.cache_metrics_available = False
        self._compaction_telemetry_counter_epoch = uuid.uuid4().hex
        self._context_probed = False
        self._context_probe_persistable = False
        self._last_overflow_recovery_failed = False
        self._last_condensation_suppressed_reason = ""
        self._last_compression_status = "idle"
        self._last_compression_noop_reason = ""
        self._last_boundary_skip_time = 0
        self._compaction_telemetry_counter_rebaseline_pending = True
        self._compaction_telemetry_turn_reset_pending = False

    def _reset_compaction_progress(self) -> None:
        """Reset process-local compaction markers for a fresh/unproven session."""
        self._last_compacted_store_id = 0
        self._ingest_cursor = 0
        self._ingest_cursor_needs_reconcile = False
        self._last_ingest_reconciliation = {"action": "none", "reason": "not run"}

    def _reset_session_scoped_runtime_state(self) -> None:
        """Reset all session-scoped runtime state.

        Calls both _reset_session_counters and _reset_compaction_progress.
        Proven carry-over paths must restore/advance state from the verified
        source lifecycle after rebinding.
        """
        self._reset_session_counters()
        self._reset_compaction_progress()
        self._generated_ignored_active_replay_placeholder_hashes = set()
        self._generated_ignored_active_replay_placeholder_message_ids = set()
        self._compression_boundary_ingest_pending = False
        self._compression_boundary_active_placeholder_digest_budget = {}
        self._compression_boundary_active_placeholder_digest_ordinals = {}
        self._compression_boundary_stored_placeholder_digest_counts = {}
