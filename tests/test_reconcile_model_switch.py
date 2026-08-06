"""Tests for model-switch notification handling in reconciliation.

The host injects "[Note: model was just switched from X to Y...]" into
the active message list when the model changes.  LCM persists it, but
the host removes it after processing.  The next turn's incoming list no
longer contains it, so the stored tail can never match → cursor=0 →
full re-ingest → duplication.

Fix: _message_replay_identity treats model-switch notifications as
identity-empty, and _is_replayed_context_scaffold_message recognises
them as scaffolding.

All tests use synthetic messages.  No real session data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from hermes_lcm.config import LCMConfig


def _make_engine(tmp_path: Path, *, session_id: str = "model-switch"):
    # engine.py imports agent.context_engine at module level; provide a
    # stub here (not at module level, so collection order of other test
    # files is unaffected), then clear any partial import left by conftest.
    if "agent.context_engine" not in sys.modules:
        _agent_mod = ModuleType("agent")
        _agent_mod.__path__ = []
        _ce_mod = ModuleType("agent.context_engine")

        class _StubContextEngine:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self.last_prompt_tokens = 0

            def get_status(self):
                return {}

        _ce_mod.ContextEngine = _StubContextEngine
        sys.modules["agent"] = _agent_mod
        sys.modules["agent.context_engine"] = _ce_mod

    _existing = sys.modules.get("hermes_lcm.engine")
    if _existing is not None and not hasattr(_existing, "LCMEngine"):
        sys.modules.pop("hermes_lcm.engine", None)
    from hermes_lcm.engine import LCMEngine

    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine._session_id = session_id
    return engine


class TestModelSwitchIdentity:
    """Model-switch notifications must be identity-transparent."""

    def test_identity_strips_prefix_preserves_content(self, tmp_path):
        """Model-switch prefix is stripped but user content survives."""
        engine = _make_engine(tmp_path)
        try:
            msg_with_prefix = {
                "role": "user",
                "content": "[Note: model was just switched from qwen3.8 to kimi-k3 via router. Adjust.]\n\nARGENT — FIX STAGING PREFLIGHT",
            }
            msg_without_prefix = {
                "role": "user",
                "content": "ARGENT — FIX STAGING PREFLIGHT",
            }
            id_with = engine._message_replay_identity(msg_with_prefix)
            id_without = engine._message_replay_identity(msg_without_prefix)
            assert id_with == id_without, (
                "Prefix-stripped identity must match plain content identity"
            )
            # Content must NOT be empty — user's message survives
            assert id_with[1] == "ARGENT — FIX STAGING PREFLIGHT"
        finally:
            engine.shutdown()

    def test_identity_stable_across_different_switches(self, tmp_path):
        """Different model-switch prefixes with same user content → same identity."""
        engine = _make_engine(tmp_path)
        try:
            msg_a = {
                "role": "user",
                "content": "[Note: model was just switched from qwen3.8 to kimi-k3 via router.]\n\nUser task here",
            }
            msg_b = {
                "role": "user",
                "content": "[Note: model was just switched from kimi-k3 to deepseek-v4 via opencode.]\n\nUser task here",
            }
            assert (
                engine._message_replay_identity(msg_a)
                == engine._message_replay_identity(msg_b)
            )
        finally:
            engine.shutdown()

    def test_normal_message_not_affected(self, tmp_path):
        """Regular messages starting with '[Note:' but not model-switch are unaffected."""
        engine = _make_engine(tmp_path)
        try:
            msg_note = {"role": "user", "content": "[Note: this is a regular note]"}
            msg_model = {
                "role": "user",
                "content": "[Note: model was just switched from X to Y]",
            }
            assert (
                engine._message_replay_identity(msg_note)
                != engine._message_replay_identity(msg_model)
            )
        finally:
            engine.shutdown()


class TestModelSwitchScaffoldDetection:
    """Model-switch messages are NOT scaffolding — they carry user content."""

    def test_model_switch_with_content_not_scaffold(self, tmp_path):
        """A model-switch prefixed message with user content is NOT scaffold."""
        engine = _make_engine(tmp_path)
        try:
            msg = {
                "role": "user",
                "content": "[Note: model was just switched from qwen3.8 to kimi-k3.]\n\nUser task here",
            }
            assert engine._is_replayed_context_scaffold_message(msg) is False
        finally:
            engine.shutdown()

    def test_regular_note_not_scaffold(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            msg = {"role": "user", "content": "[Note: remember to update docs]"}
            assert engine._is_replayed_context_scaffold_message(msg) is False
        finally:
            engine.shutdown()


class TestModelSwitchReconciliation:
    """End-to-end: model switch prefix must not cause re-ingest."""

    def test_no_duplication_after_model_switch(self, tmp_path):
        """Simulates: ingest with model-switch prefix on user message →
        next turn host strips prefix → reconcile matches without duplication."""
        engine = _make_engine(tmp_path)
        try:
            # Phase 1: normal ingest (pre-switch)
            original = [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
                {"role": "assistant", "content": "Second answer"},
            ]
            engine._ingest_messages(original)
            assert engine._store.get_session_count("model-switch") == 5

            # Phase 2: model switch happens — host PREPENDS notification
            # to the user's new message
            with_switch = original + [
                {
                    "role": "user",
                    "content": "[Note: model was just switched from qwen3.8 to kimi-k3 via router. Adjust.]\n\nThird question after switch",
                },
            ]
            switch_engine = _make_engine(tmp_path)
            try:
                switch_engine._ingest_cursor_needs_reconcile = True
                switch_engine._ingest_messages(with_switch)
                # 6 messages (5 + user msg with prefix)
                assert switch_engine._store.get_session_count("model-switch") == 6
            finally:
                switch_engine.shutdown()

            # Phase 3: next turn — host STRIPPED the prefix,
            # same user content without prefix
            after_switch = original + [
                {"role": "user", "content": "Third question after switch"},
                {"role": "assistant", "content": "Third answer"},
            ]
            replay_engine = _make_engine(tmp_path)
            try:
                replay_engine._ingest_cursor_needs_reconcile = True
                replay_engine._ingest_messages(after_switch)

                count = replay_engine._store.get_session_count("model-switch")
                # Must be 7 (5 + switch-msg + new answer), no duplication
                assert count == 7, (
                    f"Expected 7 (5 + switch + answer), got {count}. "
                    "Model switch prefix caused re-ingest duplication."
                )

                rows = replay_engine._store.get_session_messages("model-switch")
                contents = [r["content"] for r in rows]
                assert contents.count("First question") == 1
                assert contents.count("Second answer") == 1
                assert contents.count("Third answer") == 1
            finally:
                replay_engine.shutdown()
        finally:
            engine.shutdown()

    def test_model_switch_at_tail_does_not_block_reconcile(self, tmp_path):
        """When the model-switch prefix is on the LAST stored message,
        reconcile must still match after host strips the prefix."""
        engine = _make_engine(tmp_path)
        try:
            # Ingest with switch prefix on last user message
            msgs = [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "Question A"},
                {"role": "assistant", "content": "Answer A"},
                {
                    "role": "user",
                    "content": "[Note: model was just switched from X to Y.]\n\nQuestion B",
                },
            ]
            engine._ingest_messages(msgs)
            assert engine._store.get_session_count("model-switch") == 4

            # Next turn: prefix stripped, new message added
            replay = _make_engine(tmp_path)
            try:
                replay._ingest_cursor_needs_reconcile = True
                incoming = [
                    {"role": "system", "content": "System prompt."},
                    {"role": "user", "content": "Question A"},
                    {"role": "assistant", "content": "Answer A"},
                    {"role": "user", "content": "Question B"},
                    {"role": "assistant", "content": "Answer B"},
                ]
                replay._ingest_messages(incoming)

                count = replay._store.get_session_count("model-switch")
                assert count == 5, (
                    f"Expected 5 (4 stored + 1 new), got {count}."
                )
            finally:
                replay.shutdown()
        finally:
            engine.shutdown()
