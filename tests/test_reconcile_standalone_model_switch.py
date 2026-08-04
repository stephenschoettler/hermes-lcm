"""Standalone model-switch notes must not break replay reconciliation.

The host injects "[Note: model was just switched from X to Y ...]" as a
STANDALONE user message (not merely a prefix on a real user message)
when the model changes mid-session.  LCM persists that row.  On the
next turn the host removes the note from its active list entirely — the
note was ephemeral.  The stored tail therefore contains a row the
incoming replay lacks; suffix matching shifts, cursor falls to 0 and the
whole session is re-ingested (duplication).

Fix: standalone model-switch notes are recognised as replay scaffolding
and filtered from the stored-tail row set used for reconciliation, so
the stored tail and the incoming replay align.

All tests use synthetic messages.  No real session data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

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

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

STANDALONE_NOTE = (
    "[Note: model was just switched from model-a to model-b "
    "via the router. Adjust your self-identification accordingly.]"
)


def _make_engine(tmp_path: Path, **overrides) -> LCMEngine:
    defaults = dict(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_enabled=False,
        fresh_tail_count=64,
    )
    defaults.update(overrides)
    config = LCMConfig(**defaults)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "standalone-note-test",
        platform="cli",
        context_length=1000000,
    )
    return engine


class TestStandaloneModelSwitchNote:
    """Standalone model-switch notes are identity-transparent scaffolding."""

    def test_standalone_note_is_scaffold(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            msg = {"role": "user", "content": STANDALONE_NOTE}
            assert engine._is_replayed_context_scaffold_message(msg) is True
        finally:
            engine.shutdown()

    def test_note_with_user_content_not_scaffold(self, tmp_path):
        """Control: prefix + real user content stays a real message."""
        engine = _make_engine(tmp_path)
        try:
            msg = {
                "role": "user",
                "content": STANDALONE_NOTE + "\n\nPlease continue the audit.",
            }
            assert engine._is_replayed_context_scaffold_message(msg) is False
        finally:
            engine.shutdown()

    def test_standalone_note_stripped_no_reingest(self, tmp_path):
        """Stored with note, replayed without note -> no re-ingest."""
        engine = _make_engine(tmp_path)
        try:
            turn1 = [
                {"role": "user", "content": "Run the audit"},
                {"role": "assistant", "content": "Starting."},
                {"role": "tool", "tool_call_id": "c1", "content": '{"output": "ok"}'},
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content": STANDALONE_NOTE},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("standalone-note-test") == 5

            # Host strips the note from the replay and adds one new turn.
            turn2 = [
                {"role": "user", "content": "Run the audit"},
                {"role": "assistant", "content": "Starting."},
                {"role": "tool", "tool_call_id": "c1", "content": '{"output": "ok"}'},
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content": "Continue"},
            ]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            count = engine._store.get_session_count("standalone-note-test")
            assert count == 6, (
                f"Expected 6 (5 stored + 1 new), got {count}. "
                "Standalone model-switch note broke suffix matching -> "
                "cursor=0 -> full re-ingest."
            )
        finally:
            engine.shutdown()

    def test_standalone_note_present_on_both_sides(self, tmp_path):
        """Note still in the replay (same turn) must also stay clean."""
        engine = _make_engine(tmp_path)
        try:
            turn1 = [
                {"role": "user", "content": "Run the audit"},
                {"role": "assistant", "content": "Starting."},
                {"role": "user", "content": STANDALONE_NOTE},
                {"role": "assistant", "content": "Done."},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("standalone-note-test") == 4

            turn2 = turn1 + [{"role": "user", "content": "Verify"}]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            count = engine._store.get_session_count("standalone-note-test")
            assert count == 5, f"Expected 5, got {count}"
        finally:
            engine.shutdown()
