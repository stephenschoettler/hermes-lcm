"""Reproduce: compression followed by reconcile must not re-ingest.

Production evidence (session 20260802_180245_53f74a):
- 19:03: compression 285→85 messages (200 compacted)
- 19:05: 200 rows persisted immediately after, 198 of them duplicates
- 22:38: compression 482→67 (415 compacted)
- 22:40: 415 rows persisted, duplicates
- Each compression re-writes exactly the compacted message count

Hypothesis: after compress() shortens the list and resets
_ingest_cursor = len(compressed), the host rebinds via on_session_start
which sets _ingest_cursor_needs_reconcile = True.  The reconcile then
compares the stored tail (pre-compaction rows, some externalized as
placeholders) against the incoming list (fresh tail + summaries) and
fails to find a suffix match → cursor=0 → full re-ingest.
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
        fresh_tail_count=32,
        leaf_chunk_tokens=100000,
    )
    defaults.update(overrides)
    config = LCMConfig(**defaults)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "compress-reconcile-test",
        platform="cli",
        context_length=1000000,
    )
    return engine


def _turn(messages, content):
    return messages + [{"role": "user", "content": content}]


class TestCompressionThenReconcile:
    """Compression must not cause full re-ingest on the next turn."""

    def test_compression_no_reingest_on_next_turn(self, tmp_path):
        """After compress() shortens the list, next ingest must only add new."""
        engine = _make_engine(tmp_path)
        try:
            # Build a session of ~60 messages with some large tool outputs
            messages = [{"role": "system", "content": "System prompt."}]
            for i in range(60):
                if i % 5 == 0:
                    messages.append(
                        {"role": "tool", "tool_call_id": f"c{i}",
                         "content": f"BIG{i}:" + ("x" * 500)}
                    )
                else:
                    role = "user" if i % 2 == 0 else "assistant"
                    messages.append({"role": role, "content": f"msg{i}"})

            engine._ingest_messages(messages)
            count_before = engine._store.get_session_count("compress-reconcile-test")

            # Compress — this is what the host calls
            compressed = engine.compress(messages)
            assert len(compressed) < len(messages), (
                f"Compression should shrink: {len(messages)} -> {len(compressed)}"
            )

            # Host rebinds after compression (conversation_compression.py does this)
            engine.on_session_start(
                "compress-reconcile-test",
                platform="cli",
                context_length=1000000,
                boundary_reason="compression",
            )

            # Next turn: compressed context + one new user message
            next_turn = compressed + [{"role": "user", "content": "new question"}]
            engine._ingest_messages(next_turn)

            count_after = engine._store.get_session_count("compress-reconcile-test")
            expected_new = 1  # only the new user message
            expected_total = count_before + expected_new
            assert count_after == expected_total, (
                f"After compression + rebind, expected {expected_total} "
                f"({count_before} stored + 1 new), got {count_after}. "
                "Compression caused full re-ingest (duplication)."
            )
        finally:
            engine.shutdown()

    def test_compression_with_externalized_tail_no_reingest(self, tmp_path):
        """Compression + externalized tool rows + rebind must not re-ingest."""
        engine = _make_engine(tmp_path)
        try:
            messages = [{"role": "system", "content": "System prompt."}]
            for i in range(40):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": f"m{i}"})
            # Large tool output that will be externalized
            messages.append(
                {"role": "tool", "tool_call_id": "big",
                 "content": "HUGE:" + ("z" * 5000)}
            )

            engine._ingest_messages(messages)
            count_before = engine._store.get_session_count("compress-reconcile-test")

            compressed = engine.compress(messages)
            assert len(compressed) < len(messages)

            engine.on_session_start(
                "compress-reconcile-test",
                platform="cli",
                context_length=1000000,
                boundary_reason="compression",
            )

            next_turn = compressed + [{"role": "user", "content": "follow up"}]
            engine._ingest_messages(next_turn)

            count_after = engine._store.get_session_count("compress-reconcile-test")
            assert count_after == count_before + 1, (
                f"Expected {count_before + 1}, got {count_after}. "
                "Externalized rows + compression caused re-ingest."
            )
        finally:
            engine.shutdown()

    def test_compression_no_rebind_no_reingest(self, tmp_path):
        """Control: compression WITHOUT rebind must not re-ingest."""
        engine = _make_engine(tmp_path)
        try:
            messages = [{"role": "system", "content": "System prompt."}]
            for i in range(40):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": f"m{i}"})

            engine._ingest_messages(messages)
            count_before = engine._store.get_session_count("compress-reconcile-test")

            compressed = engine.compress(messages)
            # NO on_session_start here

            next_turn = compressed + [{"role": "user", "content": "follow up"}]
            engine._ingest_messages(next_turn)

            count_after = engine._store.get_session_count("compress-reconcile-test")
            assert count_after == count_before + 1, (
                f"Expected {count_before + 1}, got {count_after}. "
                "Compression alone caused re-ingest."
            )
        finally:
            engine.shutdown()
