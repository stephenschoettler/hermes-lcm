"""Only a foreground response may consume pending compaction usage."""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "memory.db")))
    instance._session_id = "example-session"
    try:
        yield instance
    finally:
        instance.shutdown()


@pytest.mark.parametrize("usage", [{}, {"prompt_tokens": 123, "completion_tokens": 7, "total_tokens": 130}])
def test_foreground_response_consumes_pending_usage_even_without_counts(engine, usage):
    engine.awaiting_real_usage_after_compression = True
    engine._verify_compaction_cleared_threshold = True
    engine.update_from_response(usage)
    assert engine.awaiting_real_usage_after_compression is False
    assert engine._verify_compaction_cleared_threshold is False
    assert engine.last_prompt_tokens == usage.get("prompt_tokens", 0)


def test_auxiliary_response_does_not_consume_foreground_gate(engine, monkeypatch):
    engine.awaiting_real_usage_after_compression = True
    engine._verify_compaction_cleared_threshold = True
    engine.last_prompt_tokens = 123
    monkeypatch.setattr(engine, "_thread_context_stateless", lambda: True)
    monkeypatch.setattr(engine, "_thread_context_session_id", lambda: "")
    engine.update_from_response({"prompt_tokens": 77})
    assert engine.awaiting_real_usage_after_compression is True
    assert engine._verify_compaction_cleared_threshold is True
    assert engine.last_prompt_tokens == 123


def test_session_reset_clears_pending_gate(engine):
    engine.awaiting_real_usage_after_compression = True
    engine._verify_compaction_cleared_threshold = True
    engine.on_session_reset()
    assert engine.awaiting_real_usage_after_compression is False
    assert engine._verify_compaction_cleared_threshold is False
