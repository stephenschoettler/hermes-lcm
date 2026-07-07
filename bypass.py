"""LCM-bypass compaction and host-fallback-compressor handling.

Extracted verbatim from :mod:`hermes_lcm.engine` as ``BypassMixin`` (WS5 seam).
The methods manage sessions that opt out of LCM context management: detecting
the bypass, mirroring the host's native fallback compressor, and applying the
deterministic tail-compaction fallback. State stays on the engine (accessed via
``self``); mixing this in leaves every call site and ``self._*`` reference
unchanged.
"""

import importlib
import logging
from typing import Any, Dict, List, Optional

from .message_analysis import _assistant_tool_call_ids
from .message_content import normalize_content_value
from .session_patterns import build_session_match_keys, matches_session_pattern
from .tokens import count_messages_tokens

logger = logging.getLogger(__name__)


class BypassMixin:
    def _bypasses_lcm_context_management(self) -> bool:
        """Return True when this binding must not write/manage LCM state.

        Ignored, stateless, and in-process auxiliary sessions are excluded from
        LCM storage. They still need context-size protection because Hermes has
        exactly one active context engine; returning a pure no-op here would
        disable every compaction layer for the session.
        """
        return bool(
            self._session_ignored
            or self._session_stateless
            or self._thread_context_stateless()
        )

    def _bypass_lcm_reason(self) -> str:
        if self._thread_context_stateless():
            return "auxiliary thread context"
        if self._session_ignored:
            return "ignored session"
        if self._session_stateless:
            return "stateless session"
        return "active session"

    def _bypass_lcm_session_id(self) -> str:
        return self._thread_context_session_id() or self._session_id or "(unknown)"

    def _session_id_matches_lcm_bypass_filters(
        self,
        session_id: str,
        *,
        platform: str = "",
    ) -> bool:
        if not session_id:
            return False
        match_keys = build_session_match_keys(session_id, platform=platform)
        if matches_session_pattern(match_keys, self._compiled_ignore_session_patterns):
            return True
        return matches_session_pattern(match_keys, self._compiled_stateless_session_patterns)

    def _ended_session_directly_bypasses_lcm(self, session_id: str) -> bool:
        """Classify a session-end callback by the ended id, not the active binding."""
        if not session_id:
            return False
        if session_id == self._thread_context_session_id():
            return True
        return self._session_id_matches_lcm_bypass_filters(session_id)

    def _end_host_fallback_compressor_for_session(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        current_session_bypasses: bool,
    ) -> None:
        if self._host_fallback_compressor is not None and (
            current_session_bypasses
            or not self._host_fallback_session_id
            or self._host_fallback_session_id == session_id
        ):
            compressor = self._host_fallback_compressor
            fallback_session_id = self._host_fallback_session_id or session_id
            on_session_end = getattr(compressor, "on_session_end", None)
            if callable(on_session_end) and fallback_session_id:
                try:
                    on_session_end(fallback_session_id, messages)
                except Exception:
                    logger.debug("LCM host fallback compressor session-end reset failed", exc_info=True)
            on_session_reset = getattr(compressor, "on_session_reset", None)
            if callable(on_session_reset):
                try:
                    on_session_reset()
                except Exception:
                    logger.debug("LCM host fallback compressor reset failed", exc_info=True)
            self._host_fallback_compressor = None
            self._host_fallback_session_id = ""

    def _get_host_fallback_compressor(self) -> Any:
        """Return Hermes' native compressor for LCM-bypassed sessions if available."""
        session_id = self._bypass_lcm_session_id()
        if self._host_fallback_compressor is not None:
            if session_id == self._host_fallback_session_id:
                return self._host_fallback_compressor
            previous = self._host_fallback_compressor
            on_session_end = getattr(previous, "on_session_end", None)
            if callable(on_session_end) and self._host_fallback_session_id:
                try:
                    on_session_end(self._host_fallback_session_id, [])
                except Exception:
                    logger.debug("LCM host fallback compressor session-end reset failed", exc_info=True)
            on_session_reset = getattr(previous, "on_session_reset", None)
            if callable(on_session_reset):
                try:
                    on_session_reset()
                except Exception:
                    logger.debug("LCM host fallback compressor reset failed", exc_info=True)
            self._host_fallback_compressor = None
            self._host_fallback_session_id = ""
        try:
            ContextCompressor = getattr(
                importlib.import_module("agent.context_compressor"),
                "ContextCompressor",
            )
        except Exception as exc:  # pragma: no cover - only hit on non-Hermes hosts
            if not self._host_fallback_import_warning_logged:
                logger.warning(
                    "LCM could not load Hermes native ContextCompressor for bypassed session fallback: %s",
                    exc,
                )
                self._host_fallback_import_warning_logged = True
            return None

        kwargs = {
            "model": self.model or "unknown",
            "threshold_percent": self.context_threshold or self.threshold_percent or 0.50,
            "protect_first_n": self.protect_first_n,
            "protect_last_n": self.protect_last_n,
            "quiet_mode": True,
            "summary_model_override": self._config.summary_model or None,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "config_context_length": self.context_length or self.raw_context_length or None,
            "provider": self.provider,
            "api_mode": self.api_mode,
        }
        try:
            compressor = ContextCompressor(**kwargs)
        except TypeError:
            # Older Hermes hosts may not expose all constructor kwargs. Keep the
            # fallback deliberately conservative rather than failing open to an
            # unbounded ignored/stateless transcript.
            try:
                compressor = ContextCompressor(
                    self.model or "unknown",
                    threshold_percent=self.context_threshold or self.threshold_percent or 0.50,
                    protect_first_n=self.protect_first_n,
                    protect_last_n=self.protect_last_n,
                    quiet_mode=True,
                )
            except Exception as exc:
                logger.warning(
                    "LCM could not initialize Hermes native ContextCompressor for bypassed session fallback; using deterministic trim: %s",
                    exc,
                )
                return None
        except Exception as exc:
            logger.warning(
                "LCM could not initialize Hermes native ContextCompressor for bypassed session fallback; using deterministic trim: %s",
                exc,
            )
            return None
        self._host_fallback_compressor = compressor
        self._host_fallback_session_id = session_id
        self._sync_host_fallback_compressor(compressor)
        return compressor

    def _sync_host_fallback_compressor(self, compressor: Any) -> None:
        """Keep the delegated native compressor aligned with LCM runtime metadata."""
        update_model = getattr(compressor, "update_model", None)
        context_length = self.context_length or self.raw_context_length
        if callable(update_model) and context_length > 0:
            try:
                update_model(
                    model=self.model or "unknown",
                    context_length=context_length,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    provider=self.provider,
                    api_mode=self.api_mode,
                )
            except TypeError:
                try:
                    update_model(self.model or "unknown", context_length, self.base_url, self.api_key)
                except TypeError:
                    pass
                except Exception:
                    logger.debug("LCM host fallback compressor model sync failed", exc_info=True)
            except Exception:
                logger.debug("LCM host fallback compressor model sync failed", exc_info=True)
        for attr, value in (
            ("threshold_percent", self.context_threshold or self.threshold_percent),
            ("protect_first_n", self.protect_first_n),
            ("protect_last_n", self.protect_last_n),
        ):
            try:
                setattr(compressor, attr, value)
            except Exception:
                pass
        on_session_start = getattr(compressor, "on_session_start", None)
        session_id = self._bypass_lcm_session_id()
        if callable(on_session_start) and session_id:
            try:
                on_session_start(
                    session_id,
                    platform=self._session_platform,
                    model=self.model,
                    provider=self.provider,
                    context_length=context_length,
                )
            except Exception:
                logger.debug("LCM host fallback compressor session bind failed", exc_info=True)

    def _mirror_host_fallback_state(self, compressor: Any) -> None:
        for attr in (
            "_last_compress_aborted",
            "_last_summary_error",
            "_last_summary_auth_failure",
            "_last_summary_network_failure",
            "_last_summary_dropped_count",
            "_last_summary_fallback_used",
            "_last_aux_model_failure_error",
            "_last_aux_model_failure_model",
        ):
            if hasattr(compressor, attr):
                setattr(self, attr, getattr(compressor, attr))

    def _bypass_compaction_target_tokens(
        self,
        *,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        caps: list[int] = []
        assembly_cap = self._overflow_recovery_assembly_cap(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        if assembly_cap is not None:
            caps.append(assembly_cap)
        if self.threshold_tokens > 0:
            caps.append(self.threshold_tokens)
        elif self.context_length > 0:
            caps.append(self.context_length)
        return min(caps) if caps else None

    @staticmethod
    def _truncate_bypass_content_value(content: Any, char_budget: int, *, suffix: str = "") -> Any:
        if char_budget < 0:
            char_budget = 0
        if isinstance(content, str):
            if len(content) <= char_budget:
                return content
            return content[:char_budget] + (suffix if char_budget > 0 else "")
        if isinstance(content, list):
            truncated_parts: list[Any] = []
            changed = False
            for part in content:
                if isinstance(part, str):
                    next_part = part if len(part) <= char_budget else part[:char_budget] + (suffix if char_budget > 0 else "")
                    changed = changed or next_part != part
                    truncated_parts.append(next_part)
                    continue
                if isinstance(part, dict):
                    next_part = dict(part)
                    for key in ("text", "content"):
                        value = next_part.get(key)
                        if isinstance(value, str) and len(value) > char_budget:
                            next_part[key] = value[:char_budget] + (suffix if char_budget > 0 else "")
                            changed = True
                        elif isinstance(value, dict):
                            nested = dict(value)
                            for nested_key in ("value", "content"):
                                nested_value = nested.get(nested_key)
                                if isinstance(nested_value, str) and len(nested_value) > char_budget:
                                    nested[nested_key] = nested_value[:char_budget] + (suffix if char_budget > 0 else "")
                                    changed = True
                            next_part[key] = nested
                    truncated_parts.append(next_part)
                    continue
                truncated_parts.append(part)
            if changed:
                return truncated_parts
        normalized = normalize_content_value(content)
        if isinstance(normalized, str) and len(normalized) > char_budget:
            return normalized[:char_budget] + (suffix if char_budget > 0 else "")
        return content

    def _trim_bypass_compacted_to_cap(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: Optional[int],
    ) -> List[Dict[str, Any]]:
        compacted = self._sanitize_active_context_messages(messages)
        if target_tokens is None or target_tokens <= 0:
            return compacted

        while len(compacted) > 2 and count_messages_tokens(compacted) > target_tokens:
            remove_indices: list[int] = []
            for idx, msg in enumerate(compacted):
                if idx == 0:
                    continue
                if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                    continue
                call_ids = _assistant_tool_call_ids([msg])
                remove_indices = [idx]
                remove_indices.extend(
                    follow_idx
                    for follow_idx in range(idx + 1, len(compacted))
                    if compacted[follow_idx].get("role") == "tool"
                    and str(compacted[follow_idx].get("tool_call_id") or "") in call_ids
                )
                break
            if not remove_indices:
                remove_index = 1
                if (
                    compacted[1].get("role") == "tool"
                    and compacted[0].get("role") == "assistant"
                    and compacted[0].get("tool_calls")
                ):
                    remove_index = 0
                remove_indices = [remove_index]
            before_shape = [
                (msg.get("role"), msg.get("tool_call_id"), bool(msg.get("tool_calls")))
                for msg in compacted
            ]
            before_tokens = count_messages_tokens(compacted)
            for remove_index in sorted(set(remove_indices), reverse=True):
                if 0 <= remove_index < len(compacted):
                    del compacted[remove_index]
            compacted = self._sanitize_active_context_messages(compacted)
            after_shape = [
                (msg.get("role"), msg.get("tool_call_id"), bool(msg.get("tool_calls")))
                for msg in compacted
            ]
            if after_shape == before_shape and count_messages_tokens(compacted) >= before_tokens:
                break

        if count_messages_tokens(compacted) <= target_tokens:
            return compacted

        char_budget = max(0, min(500, target_tokens * 4 // max(1, len(compacted))))
        truncated = compacted
        previous_budget = -1
        for _ in range(12):
            next_messages: list[Dict[str, Any]] = []
            for msg in compacted:
                next_msg = dict(msg)
                content = next_msg.get("content")
                next_msg["content"] = self._truncate_bypass_content_value(content, char_budget, suffix="…")
                next_messages.append(next_msg)
            truncated = self._sanitize_active_context_messages(next_messages)
            token_count = count_messages_tokens(truncated)
            if token_count <= target_tokens:
                return truncated
            if char_budget == 0 or char_budget == previous_budget:
                break
            previous_budget = char_budget
            ratio = target_tokens / max(1, token_count)
            char_budget = max(0, min(char_budget - 1, int(char_budget * max(0.25, ratio * 0.8))))

        compacted = truncated
        while len(compacted) > 1 and count_messages_tokens(compacted) > target_tokens:
            compacted = self._sanitize_active_context_messages(compacted[1:])

        char_budget = max(0, min(80, target_tokens * 4))
        previous_budget = -1
        while count_messages_tokens(compacted) > target_tokens and char_budget != previous_budget:
            previous_budget = char_budget
            shrunk: list[Dict[str, Any]] = []
            for msg in compacted:
                next_msg = dict(msg)
                content = next_msg.get("content")
                next_msg["content"] = self._truncate_bypass_content_value(content, char_budget)
                shrunk.append(next_msg)
            compacted = self._sanitize_active_context_messages(shrunk)
            char_budget = max(0, char_budget // 2)

        return compacted

    def _fallback_tail_compaction(
        self,
        messages: List[Dict[str, Any]],
        *,
        target_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Last-resort size guard when Hermes' native compressor is unavailable."""
        if len(messages) <= 2:
            return self._trim_bypass_compacted_to_cap(messages, target_tokens)
        head_count = max(1, min(self.protect_first_n, len(messages)))
        tail_count = max(1, min(self.protect_last_n, len(messages) - head_count))
        marker = {
            "role": "user",
            "content": (
                "[Context omitted: this session is ignored/stateless for LCM, "
                "and Hermes native compression was unavailable. Older messages "
                "were dropped to keep the request within the model context window.]"
            ),
        }
        compacted = list(messages[:head_count]) + [marker] + list(messages[-tail_count:])
        return self._trim_bypass_compacted_to_cap(compacted, target_tokens)

    def _compress_lcm_bypassed_session(
        self,
        messages: List[Dict[str, Any]],
        *,
        current_tokens: int | None = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Delegate ignored/stateless context bounding without writing to LCM."""
        reason = self._bypass_lcm_reason()
        session_id = self._bypass_lcm_session_id()
        self._remember_lcm_bypass_message_prefix(session_id, messages)
        observed_tokens = current_tokens if current_tokens and current_tokens > 0 else count_messages_tokens(messages)
        force_overflow = self._should_force_overflow_recovery(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        if not force and not force_overflow and self.threshold_tokens > 0 and observed_tokens < self.threshold_tokens:
            logger.debug(
                "LCM compaction bypass no-op for %s %s below threshold (%s < %s)",
                reason,
                session_id,
                observed_tokens,
                self.threshold_tokens,
            )
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = f"LCM bypassed below threshold: {reason}"
            return self._redact_active_replay_messages(messages)

        logger.debug("LCM delegating compaction for bypassed %s %s", reason, session_id)
        self._last_compression_status = "host_fallback"
        self._last_compression_noop_reason = f"LCM bypassed: {reason}"
        safe_messages = self._redact_active_replay_messages(messages)
        target_tokens = self._bypass_compaction_target_tokens(
            observed_tokens=observed_tokens,
            messages=safe_messages,
        )

        compressor = self._get_host_fallback_compressor()
        if compressor is None:
            compacted = self._fallback_tail_compaction(safe_messages, target_tokens=target_tokens)
            if compacted != messages:
                self.compression_count += 1
            return compacted

        self._sync_host_fallback_compressor(compressor)
        before_count = int(getattr(compressor, "compression_count", 0) or 0)
        try:
            try:
                compacted = compressor.compress(
                    safe_messages,
                    current_tokens=current_tokens,
                    focus_topic=focus_topic,
                    force=force,
                )
            except TypeError:
                compacted = compressor.compress(
                    safe_messages,
                    current_tokens=current_tokens,
                    focus_topic=focus_topic,
                )
        except Exception as exc:
            self._mirror_host_fallback_state(compressor)
            logger.warning(
                "LCM Hermes native ContextCompressor failed for bypassed %s %s; using deterministic trim: %s",
                reason,
                session_id,
                exc,
            )
            self._host_fallback_compressor = None
            self._host_fallback_session_id = ""
            compacted = self._fallback_tail_compaction(safe_messages, target_tokens=target_tokens)
            if compacted != safe_messages:
                self.compression_count += 1
                self._last_compress_aborted = False
            return compacted
        self._mirror_host_fallback_state(compressor)
        after_count = int(getattr(compressor, "compression_count", before_count) or before_count)
        self.compression_count += max(1, after_count - before_count)
        compacted = self._sanitize_active_context_messages(compacted)
        if target_tokens is not None and count_messages_tokens(compacted) > target_tokens:
            compacted = self._fallback_tail_compaction(safe_messages, target_tokens=target_tokens)
            if compacted != safe_messages:
                self._last_compress_aborted = False
        return compacted
