"""Out-of-band user messages appended to tool output must not break
replay reconciliation.

When the user sends a message while the agent is mid-turn, the host
appends it to the tool output currently being delivered, wrapped in an
"[OUT-OF-BAND USER MESSAGE ...] ... [/OUT-OF-BAND USER MESSAGE]" block.
LCM persists the tool row with the block attached.  On resume/restart
the host replays the same tool result WITHOUT the block (the user
message was already delivered separately), so the stored row and the
incoming row differ by exactly that block -> identity mismatch ->
cursor=0 -> full session re-ingest (duplication).

Fix: _message_replay_identity strips the out-of-band block so identity
matching survives the delivery split.

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

OOB_BLOCK = (
    "\n\n[OUT-OF-BAND USER MESSAGE — a direct message from the user, "
    "delivered mid-turn; not tool output]\ncheck the 2026 logs instead\n"
    "[/OUT-OF-BAND USER MESSAGE]"
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
        "oob-block-test",
        platform="cli",
        context_length=1000000,
    )
    return engine


class TestOutOfBandBlockIdentity:
    """The OOB block is identity-transparent."""

    def test_identity_strips_oob_block_preserves_content(self, tmp_path):
        """Stored row (with block) and incoming row (without) match."""
        engine = _make_engine(tmp_path)
        try:
            base = '{"output": "build finished, 0 errors", "exit_code": 0}'
            id_with = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base + OOB_BLOCK}
            )
            id_without = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base}
            )
            assert id_with == id_without, (
                "OOB-block-stripped identity must match plain content identity"
            )
            # The tool's own content must survive
            assert "build finished" in id_with[1]
        finally:
            engine.shutdown()

    def test_identity_multiple_oob_blocks_stripped(self, tmp_path):
        """Multiple appended blocks (repeated interruptions) all strip."""
        engine = _make_engine(tmp_path)
        try:
            base = '{"output": "step one done", "exit_code": 0}'
            twice = base + OOB_BLOCK + OOB_BLOCK.replace(
                "check the 2026 logs instead", "and also run the tests"
            )
            id_twice = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": twice}
            )
            id_plain = engine._message_replay_identity(
                {"role": "tool", "tool_call_id": "c1", "content": base}
            )
            assert id_twice == id_plain
        finally:
            engine.shutdown()

    def test_oob_text_without_markers_not_stripped(self, tmp_path):
        """Content merely mentioning the phrase keeps its identity."""
        engine = _make_engine(tmp_path)
        try:
            content = "The docs mention OUT-OF-BAND USER MESSAGE markers."
            id_a = engine._message_replay_identity({"role": "user", "content": content})
            assert content in id_a[1], "Text without real markers must not be stripped"
        finally:
            engine.shutdown()


class TestOutOfBandBlockReconciliation:
    """Restart reconciliation across the OOB delivery split."""

    def test_oob_block_no_reingest_on_restart(self, tmp_path):
        """Stored with block, replayed without block -> no re-ingest."""
        engine = _make_engine(tmp_path)
        try:
            tool_content = '{"output": "compile succeeded", "exit_code": 0}'
            turn1 = [
                {"role": "user", "content": "Build the module"},
                {"role": "assistant", "content": "Building."},
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": tool_content + OOB_BLOCK,  # persisted mid-turn
                },
                {"role": "assistant", "content": "Done."},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("oob-block-test") == 4

            # Restart: host replays the tool result WITHOUT the OOB block
            # (the user message was delivered separately), plus one new turn.
            turn2 = [
                {"role": "user", "content": "Build the module"},
                {"role": "assistant", "content": "Building."},
                {"role": "tool", "tool_call_id": "c1", "content": tool_content},
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content": "Now run the tests"},
            ]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            count = engine._store.get_session_count("oob-block-test")
            assert count == 5, (
                f"Expected 5 (4 stored + 1 new), got {count}. "
                "OOB block identity mismatch -> cursor=0 -> full re-ingest."
            )
        finally:
            engine.shutdown()

    def test_reverse_direction_no_reingest(self, tmp_path):
        """Stored without block, replayed with block -> no re-ingest."""
        engine = _make_engine(tmp_path)
        try:
            tool_content = '{"output": "deploy ok", "exit_code": 0}'
            turn1 = [
                {"role": "user", "content": "Deploy"},
                {"role": "assistant", "content": "Deploying."},
                {"role": "tool", "tool_call_id": "c1", "content": tool_content},
                {"role": "assistant", "content": "Done."},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("oob-block-test") == 4

            turn2 = turn1.copy()
            turn2[2] = {
                "role": "tool",
                "tool_call_id": "c1",
                "content": tool_content + OOB_BLOCK,
            }
            turn2.append({"role": "user", "content": "Verify"})
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            count = engine._store.get_session_count("oob-block-test")
            assert count == 5, f"Expected 5, got {count}"
        finally:
            engine.shutdown()
