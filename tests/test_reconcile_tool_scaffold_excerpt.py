"""Tool messages containing summary-like excerpts must not be treated as
scaffolding during replay reconciliation.

A tool result can legitimately quote a summary excerpt (delegation receipts,
session links, embedded "[Expand for details:" blocks).  If
_is_replayed_context_scaffold_message marks such a tool row as scaffolding,
the replay identity list drops it — the incoming list becomes shorter than
the stored tail, suffix matching shifts, cursor falls to 0 and the whole
session is re-ingested (duplication).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from hermes_lcm.config import LCMConfig


def _make_engine(tmp_path: Path, **overrides):
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

    defaults = dict(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=200,
        large_output_externalization_path=str(tmp_path / "externalized"),
        fresh_tail_count=64,
    )
    defaults.update(overrides)
    config = LCMConfig(**defaults)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "tool-scaffold-test",
        platform="cli",
        context_length=1000000,
    )
    return engine


class TestToolSummaryExcerptNotScaffold:
    """Tool rows quoting summary excerpts keep their replay identity."""

    def test_tool_row_with_summary_excerpt_is_not_scaffold(self, tmp_path):
        engine = _make_engine(tmp_path)
        try:
            summary_quote = (
                "Delegation receipt with embedded excerpt:\n"
                "[Expand for details: recent summary content]\n"
                "[Recent Summary (d0, node 42)]\n"
                "Actual tool payload continues here."
            )
            msg = {"role": "tool", "tool_call_id": "call_x", "content": summary_quote}
            assert engine._is_replayed_context_scaffold_message(msg) is False, (
                "Tool row with summary excerpt must NOT be scaffolding"
            )
        finally:
            engine.shutdown()

    def test_tool_summary_excerpt_no_reingest_on_restart(self, tmp_path):
        """Restart replay with a summary-quoting tool row must not re-ingest."""
        engine = _make_engine(tmp_path)
        try:
            summary_quote = (
                "Receipt:\n"
                "[Expand for details: abc]\n"
                "[Recent Summary (d0, node 7)]\n"
                + ("x" * 3000)
            )
            turn1 = [
                {"role": "user", "content": "Deploy"},
                {"role": "assistant", "content": "Running."},
                {"role": "tool", "tool_call_id": "c1", "content": summary_quote},
                {"role": "assistant", "content": "Done."},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("tool-scaffold-test") == 4

            # Simulated restart: same messages + one new
            turn2 = turn1 + [{"role": "user", "content": "Verify"}]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            count = engine._store.get_session_count("tool-scaffold-test")
            assert count == 5, (
                f"Expected 5 (4 stored + 1 new), got {count}. "
                "Summary-quoting tool row was treated as scaffolding → "
                "replay identity dropped it → full re-ingest."
            )
        finally:
            engine.shutdown()

    def test_assistant_summary_excerpt_still_scaffold(self, tmp_path):
        """Control: assistant summary rows remain scaffolding."""
        engine = _make_engine(tmp_path)
        try:
            summary = (
                "[Recent Summary (d0, node 42)]\n"
                "[Expand for details: ...]\n"
                "Context was compacted."
            )
            msg = {"role": "assistant", "content": summary}
            assert engine._is_replayed_context_scaffold_message(msg) is True
        finally:
            engine.shutdown()
