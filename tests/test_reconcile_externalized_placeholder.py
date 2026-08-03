"""Reconcile must not re-ingest when stored tail has externalized placeholders.

When _ingest_messages externalizes a large message, the DB row holds a
compact placeholder while the active context keeps the original content.
On the next turn, reconcile compares the stored tail (placeholders) against
the incoming list (originals).  The exact tuple equality in
_matches_store_tail_suffix fails, cursor falls to 0, and the entire
message list is re-persisted — creating duplicates.

These tests reproduce that cycle and verify the fix: externalized
placeholder rows in the stored tail must be treated as role-scoped
wildcards during suffix matching so the cursor advances correctly.
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


def _make_engine(tmp_path: Path, **overrides) -> LCMEngine:
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
        "ext-reconcile-test",
        platform="cli",
        context_length=200000,
    )
    return engine


class TestExternalizedPlaceholderReconciliation:
    """Stored externalized placeholders must not break suffix matching."""

    def test_tool_externalization_does_not_cause_reingest(self, tmp_path):
        """Large tool output externalized in DB, original in active context.

        Turn 1: ingest [system, user, assistant, tool(large)]
                → DB stores tool as placeholder
        Turn 2: same messages + new assistant reply
                → reconcile must advance cursor, not re-ingest
        """
        engine = _make_engine(tmp_path)
        try:
            large_tool = "TOOL_RESULT:" + ("x" * 5000)
            turn1 = [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "Run the build"},
                {"role": "assistant", "content": "Running build now."},
                {"role": "tool", "tool_call_id": "call_1", "content": large_tool},
            ]
            engine._ingest_messages(turn1)
            count_after_t1 = engine._store.get_session_count("ext-reconcile-test")
            assert count_after_t1 == 4, f"Turn 1 should store 4, got {count_after_t1}"

            # Verify the tool row was externalized in the store
            stored = engine._store.get_session_messages("ext-reconcile-test")
            tool_row = [r for r in stored if r["role"] == "tool"][0]
            assert "[Externalized" in tool_row["content"], (
                f"Tool row should be externalized, got: {tool_row['content'][:80]}"
            )

            # Turn 2: same messages + new reply (active context has originals)
            turn2 = turn1 + [
                {"role": "assistant", "content": "Build succeeded."},
            ]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            count_after_t2 = engine._store.get_session_count("ext-reconcile-test")
            assert count_after_t2 == 5, (
                f"Expected 5 (4 stored + 1 new), got {count_after_t2}. "
                "Externalized placeholder broke reconcile → re-ingest duplication."
            )
        finally:
            engine.shutdown()

    def test_assistant_externalization_does_not_cause_reingest(self, tmp_path):
        """Large assistant response externalized in DB, original in active.

        Same pattern but with role=assistant.
        """
        engine = _make_engine(tmp_path)
        try:
            large_assistant = "ANALYSIS:" + ("y" * 5000)
            turn1 = [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "Analyze the binary"},
                {"role": "assistant", "content": large_assistant},
            ]
            engine._ingest_messages(turn1)
            count_after_t1 = engine._store.get_session_count("ext-reconcile-test")
            assert count_after_t1 == 3

            stored = engine._store.get_session_messages("ext-reconcile-test")
            asst_rows = [r for r in stored if r["role"] == "assistant"]
            assert "[Externalized" in asst_rows[0]["content"], (
                f"Assistant row should be externalized, got: {asst_rows[0]['content'][:80]}"
            )

            turn2 = turn1 + [
                {"role": "user", "content": "What about the imports?"},
            ]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            count_after_t2 = engine._store.get_session_count("ext-reconcile-test")
            assert count_after_t2 == 4, (
                f"Expected 4 (3 stored + 1 new), got {count_after_t2}. "
                "Externalized assistant placeholder broke reconcile."
            )
        finally:
            engine.shutdown()

    def test_multiple_externalized_messages_do_not_cause_reingest(self, tmp_path):
        """Multiple large messages externalized — all must be wildcard-matched."""
        engine = _make_engine(tmp_path)
        try:
            turn1 = [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "Do the analysis"},
                {"role": "assistant", "content": "STEP1:" + ("a" * 3000)},
                {"role": "tool", "tool_call_id": "c1", "content": "OUT1:" + ("b" * 3000)},
                {"role": "assistant", "content": "STEP2:" + ("c" * 3000)},
                {"role": "tool", "tool_call_id": "c2", "content": "OUT2:" + ("d" * 3000)},
            ]
            engine._ingest_messages(turn1)
            count_t1 = engine._store.get_session_count("ext-reconcile-test")
            assert count_t1 == 6

            turn2 = turn1 + [
                {"role": "assistant", "content": "Analysis complete."},
            ]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            count_t2 = engine._store.get_session_count("ext-reconcile-test")
            assert count_t2 == 7, (
                f"Expected 7 (6 stored + 1 new), got {count_t2}. "
                "Multiple externalized placeholders broke reconcile."
            )
        finally:
            engine.shutdown()

    def test_no_externalization_still_reconciles(self, tmp_path):
        """Control: small messages (no externalization) reconcile normally."""
        engine = _make_engine(tmp_path)
        try:
            turn1 = [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("ext-reconcile-test") == 3

            turn2 = turn1 + [{"role": "user", "content": "How are you?"}]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            assert engine._store.get_session_count("ext-reconcile-test") == 4
        finally:
            engine.shutdown()

    def test_both_sides_externalized_uses_exact_match(self, tmp_path):
        """When incoming ALSO has placeholders, exact match applies."""
        engine = _make_engine(tmp_path)
        try:
            large = "DATA:" + ("z" * 5000)
            turn1 = [
                {"role": "system", "content": "System prompt."},
                {"role": "tool", "tool_call_id": "c1", "content": large},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("ext-reconcile-test") == 2

            # Simulate: active context also has the placeholder (e.g. after
            # compress() rebuilds from store).  Exact match should work.
            stored = engine._store.get_session_messages("ext-reconcile-test")
            placeholder_content = stored[1]["content"]
            assert "[Externalized" in placeholder_content

            turn2 = [
                {"role": "system", "content": "System prompt."},
                {"role": "tool", "tool_call_id": "c1", "content": placeholder_content},
                {"role": "assistant", "content": "Noted."},
            ]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            count = engine._store.get_session_count("ext-reconcile-test")
            assert count == 3, (
                f"Expected 3 (2 stored + 1 new), got {count}. "
                "Placeholder-to-placeholder exact match should work."
            )
        finally:
            engine.shutdown()

    def test_role_mismatch_does_not_wildcard_match(self, tmp_path):
        """Divergent conversation: tool row skipped, but system matches.

        Stored: [system, tool(c1)].  Incoming: [system, user, assistant].
        The stored tool row has no counterpart (compaction removed it) so it
        is skipped; the system prompt still matches at cursor=1, and only the
        user + assistant rows are new.  Total = 2 stored + 2 new = 4.
        """
        engine = _make_engine(tmp_path)
        try:
            turn1 = [
                {"role": "system", "content": "System prompt."},
                {"role": "tool", "tool_call_id": "c1", "content": "BIG:" + ("q" * 5000)},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("ext-reconcile-test") == 2

            turn2 = [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "Completely different message"},
                {"role": "assistant", "content": "Response"},
            ]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            count = engine._store.get_session_count("ext-reconcile-test")
            # 4 = 2 stored (system+tool) + 2 new (user+assistant)
            assert count == 4, (
                f"Expected 4 (system matches, tool skipped, 2 new), got {count}."
            )
        finally:
            engine.shutdown()

    def test_system_prompt_change_does_not_cause_reingest(self, tmp_path):
        """Model switch rebuilds system prompt — must not break reconcile.

        The system prompt is volatile host scaffolding that changes on
        every model/provider switch.  When its content is included in
        the replay identity, any switch breaks suffix matching at
        position 0 → cursor=0 → full re-ingest → duplication.
        """
        engine = _make_engine(tmp_path)
        try:
            turn1 = [
                {"role": "system", "content": "You are model-A. Provider: X."},
                {"role": "user", "content": "Analyze the binary"},
                {"role": "assistant", "content": "Starting analysis."},
                {"role": "user", "content": "Check imports"},
                {"role": "assistant", "content": "Imports look normal."},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("ext-reconcile-test") == 5

            # Model switch: system prompt rebuilt with different identity
            turn2 = [
                {"role": "system", "content": "You are model-B. Provider: Y."},
                {"role": "user", "content": "Analyze the binary"},
                {"role": "assistant", "content": "Starting analysis."},
                {"role": "user", "content": "Check imports"},
                {"role": "assistant", "content": "Imports look normal."},
                {"role": "user", "content": "What about exports?"},
            ]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            count = engine._store.get_session_count("ext-reconcile-test")
            assert count == 6, (
                f"Expected 6 (5 stored + 1 new), got {count}. "
                "System prompt change broke reconcile → full re-ingest."
            )
        finally:
            engine.shutdown()

    def test_system_prompt_change_with_externalized_tail(self, tmp_path):
        """Combined: system prompt change + externalized tool in tail.

        This is the exact production scenario: model switch rebuilds
        system prompt while the stored tail has externalized placeholders.
        """
        engine = _make_engine(tmp_path)
        try:
            turn1 = [
                {"role": "system", "content": "You are model-A."},
                {"role": "user", "content": "Run build"},
                {"role": "assistant", "content": "Building."},
                {"role": "tool", "tool_call_id": "c1", "content": "BUILD:" + ("o" * 5000)},
                {"role": "assistant", "content": "Build done."},
            ]
            engine._ingest_messages(turn1)
            assert engine._store.get_session_count("ext-reconcile-test") == 5

            # Model switch + new message
            turn2 = [
                {"role": "system", "content": "You are model-B."},
                {"role": "user", "content": "Run build"},
                {"role": "assistant", "content": "Building."},
                {"role": "tool", "tool_call_id": "c1", "content": "BUILD:" + ("o" * 5000)},
                {"role": "assistant", "content": "Build done."},
                {"role": "user", "content": "Run tests"},
            ]
            engine._ingest_cursor_needs_reconcile = True
            engine._ingest_messages(turn2)

            count = engine._store.get_session_count("ext-reconcile-test")
            assert count == 6, (
                f"Expected 6 (5 stored + 1 new), got {count}. "
                "System prompt change + externalized tail broke reconcile."
            )
        finally:
            engine.shutdown()
