"""Context-aware clamping for leaf chunk and fresh tail sizing.

When context_length is known, leaf_chunk_tokens and fresh_tail_max_tokens
must be clamped proportionally so compression remains possible on small
context models.  Without clamping, a 250K leaf chunk on a 272K model means
the fresh tail alone consumes the entire window and compression never fires.

Production evidence: coder profile session 20260801_113735_47946d
(model gpt-5.6-sol, ctx 272,000) hit 416K tokens with repeated
"LCM compression no-op: raw backlog outside fresh tail is below leaf
chunk threshold" because leaf_chunk_tokens=250000 > 272000 * 0.4.
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


def _make_engine(
    tmp_path: Path,
    *,
    context_length: int = 0,
    leaf_chunk: int = 250000,
    dynamic_max: int = 500000,
    fresh_tail_count: int = 64,
    fresh_tail_max_tokens: int = 0,
) -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
        leaf_chunk_tokens=leaf_chunk,
        dynamic_leaf_chunk_enabled=True,
        dynamic_leaf_chunk_max=dynamic_max,
        fresh_tail_count=fresh_tail_count,
        fresh_tail_max_tokens=fresh_tail_max_tokens,
    )
    engine = LCMEngine(config=config)
    if context_length > 0:
        engine._set_context_length(context_length, source="test")
    return engine


class TestContextAwareLeafCap:
    """_context_aware_leaf_cap returns 40% of context_length."""

    def test_no_context_length_returns_none(self, tmp_path):
        engine = _make_engine(tmp_path, context_length=0)
        try:
            assert engine._context_aware_leaf_cap() is None
        finally:
            engine.shutdown()

    def test_small_context_returns_none(self, tmp_path):
        """Context < 50K (test fixtures) → no clamping."""
        engine = _make_engine(tmp_path, context_length=10000)
        try:
            assert engine._context_aware_leaf_cap() is None
        finally:
            engine.shutdown()

    def test_272k_model_caps_at_40_percent(self, tmp_path):
        engine = _make_engine(tmp_path, context_length=272000)
        try:
            cap = engine._context_aware_leaf_cap()
            assert cap == int(272000 * 0.4)  # 108800
        finally:
            engine.shutdown()

    def test_1m_model_caps_at_40_percent(self, tmp_path):
        engine = _make_engine(tmp_path, context_length=1048576)
        try:
            cap = engine._context_aware_leaf_cap()
            assert cap == int(1048576 * 0.4)  # 419430
        finally:
            engine.shutdown()


class TestWorkingLeafChunkClamping:
    """_working_leaf_chunk_tokens respects context-aware cap."""

    def test_250k_leaf_clamped_on_272k_model(self, tmp_path):
        """250K leaf chunk on 272K model -> clamped to 108.8K."""
        engine = _make_engine(tmp_path, context_length=272000, leaf_chunk=250000)
        try:
            working = engine._working_leaf_chunk_tokens(50000)
            assert working == int(272000 * 0.4)  # 108800, not 250000
        finally:
            engine.shutdown()

    def test_250k_leaf_not_clamped_on_1m_model(self, tmp_path):
        """250K leaf chunk on 1M model -> no clamping needed."""
        engine = _make_engine(tmp_path, context_length=1048576, leaf_chunk=250000)
        try:
            working = engine._working_leaf_chunk_tokens(50000)
            assert working == 250000  # below 419K cap, no change
        finally:
            engine.shutdown()

    def test_dynamic_ceiling_clamped_on_small_model(self, tmp_path):
        """Dynamic ceiling (500K) on 272K model -> clamped to 108.8K."""
        engine = _make_engine(
            tmp_path, context_length=272000, leaf_chunk=250000, dynamic_max=500000
        )
        try:
            working = engine._working_leaf_chunk_tokens(500000)
            assert working == int(272000 * 0.4)  # ceiling clamped
        finally:
            engine.shutdown()

    def test_no_context_length_no_clamping(self, tmp_path):
        """Without context_length, original values preserved."""
        engine = _make_engine(tmp_path, context_length=0, leaf_chunk=250000)
        try:
            working = engine._working_leaf_chunk_tokens(50000)
            assert working == 250000
        finally:
            engine.shutdown()


class TestEffectiveFreshTailMaxTokens:
    """_effective_fresh_tail_max_tokens derives context-proportional default."""

    def test_explicit_setting_preserved(self, tmp_path):
        """User-set LCM_FRESH_TAIL_MAX_TOKENS is never overridden."""
        engine = _make_engine(
            tmp_path, context_length=272000, fresh_tail_max_tokens=100000
        )
        try:
            assert engine._effective_fresh_tail_max_tokens() == 100000
        finally:
            engine.shutdown()

    def test_implicit_default_50_percent_of_context(self, tmp_path):
        """No explicit setting -> 50% of context_length."""
        engine = _make_engine(tmp_path, context_length=272000)
        try:
            assert engine._effective_fresh_tail_max_tokens() == int(272000 * 0.5)
        finally:
            engine.shutdown()

    def test_no_context_returns_zero(self, tmp_path):
        """No context_length and no explicit setting -> 0 (disabled)."""
        engine = _make_engine(tmp_path, context_length=0)
        try:
            assert engine._effective_fresh_tail_max_tokens() == 0
        finally:
            engine.shutdown()

    def test_small_context_returns_zero(self, tmp_path):
        """Context < 50K (test fixtures) -> 0 (disabled)."""
        engine = _make_engine(tmp_path, context_length=10000)
        try:
            assert engine._effective_fresh_tail_max_tokens() == 0
        finally:
            engine.shutdown()

    def test_1m_model_gets_500k_tail_cap(self, tmp_path):
        """1M model -> 500K fresh tail cap (generous, won't over-compress)."""
        engine = _make_engine(tmp_path, context_length=1048576)
        try:
            assert engine._effective_fresh_tail_max_tokens() == int(1048576 * 0.5)
        finally:
            engine.shutdown()


class TestCompressionEligibilityWithClamping:
    """End-to-end: compression must be eligible on small context models."""

    def test_272k_model_compression_eligible(self, tmp_path):
        """Simulates the coder profile scenario: 272K model, large tool outputs.

        Before fix: leaf_chunk=250K > backlog -> no-op forever.
        After fix: leaf_chunk clamped to 108.8K -> compression fires.
        """
        engine = _make_engine(tmp_path, context_length=272000, leaf_chunk=250000)
        try:
            # 200 messages: 64 in fresh tail, 136 in backlog
            # 136 * ~2K tokens = ~272K backlog > 108.8K clamped leaf chunk
            messages = [{"role": "system", "content": "System prompt."}]
            for i in range(200):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": f"Message {i}. " + "x" * 6000})

            eligible, reason = engine._leaf_compaction_candidate_status(messages)
            assert eligible, (
                f"Compression must be eligible on 272K model with 272K+ backlog. "
                f"Got: {reason}"
            )
        finally:
            engine.shutdown()

    def test_1m_model_still_works(self, tmp_path):
        """1M model with same settings -> no behavioral change."""
        engine = _make_engine(tmp_path, context_length=1048576, leaf_chunk=250000)
        try:
            messages = [{"role": "system", "content": "System prompt."}]
            for i in range(80):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": f"Message {i}. " + "x" * 6000})

            eligible, reason = engine._leaf_compaction_candidate_status(messages)
            # 160K backlog < 250K leaf chunk -> not eligible (correct for 1M model)
            assert not eligible
            assert "below leaf chunk threshold" in reason
        finally:
            engine.shutdown()
