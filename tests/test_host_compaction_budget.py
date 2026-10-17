"""Exercise the Hermes compaction budget with a real LCM engine.

The host commits an engine result, then uses the next provider prompt count
to decide whether it can reset its per-turn attempt counter. These tests
use that contract without a model request or a Hermes installation.
"""

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


class _CompactionHost:
    """Apply the commit and usage gates from Hermes conversation_compression/loop."""

    def __init__(self, engine):
        self.engine = engine
        self.attempts = 0

    def compact(self, messages):
        if self.attempts >= 3:
            return messages
        self.attempts += 1
        compressed = self.engine.compress(messages)
        # The real host sets this only after the transcript commit succeeds.
        if getattr(self.engine, "_last_compression_made_progress", False):
            self.engine._verify_compaction_cleared_threshold = True
        return compressed

    def receive_usage(self, prompt_tokens):
        pending = getattr(self.engine, "_verify_compaction_cleared_threshold", False)
        self.engine.update_from_response({"prompt_tokens": prompt_tokens})
        if pending and 0 < prompt_tokens < self.engine.threshold_tokens:
            self.attempts = 0


@pytest.fixture
def host(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "host-budget.db"),
        fresh_tail_count=4,
        leaf_chunk_tokens=400,
    )
    engine = LCMEngine(config=config)
    engine._session_id = "long-tool-turn"
    engine.context_length = 200_000
    engine.threshold_tokens = 4_000
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        lambda **kwargs: ("Completed prior checks. Continue the current task.", 1),
    )
    try:
        yield _CompactionHost(engine)
    finally:
        engine.shutdown()


def _append_history(messages, batch):
    for index in range(20):
        messages.extend([
            {"role": "user", "content": f"Check {batch}/{index}. " + "detail " * 100},
            {"role": "assistant", "content": f"Result {batch}/{index}. " + "result " * 100},
        ])


def test_long_turn_can_complete_more_than_three_compactions(host):
    messages = [{"role": "system", "content": "Complete the checks."}]
    for batch in range(5):
        _append_history(messages, batch)
        before = count_messages_tokens(messages)
        assert before >= host.engine.threshold_tokens

        messages = host.compact(messages)
        after = count_messages_tokens(messages)
        assert after < host.engine.threshold_tokens
        host.receive_usage(after)

    assert host.engine.compression_count == 5
    assert host.attempts == 0


@pytest.mark.parametrize("first_prompt", [0, 4_000, 6_000])
def test_unverified_result_does_not_reset_budget_on_later_usage(host, first_prompt):
    messages = [{"role": "system", "content": "Complete the checks."}]
    _append_history(messages, 0)
    host.compact(messages)

    host.receive_usage(first_prompt)
    assert host.attempts == 1
    host.receive_usage(100)
    assert host.attempts == 1


def test_unchanged_result_does_not_reuse_previous_success(host):
    messages = [{"role": "system", "content": "Complete the checks."}]
    _append_history(messages, 0)
    host.compact(messages)
    host.receive_usage(100)

    assert host.compact([]) == []
    host.receive_usage(100)
    assert host.attempts == 1


def test_exception_clears_previous_progress(host, monkeypatch):
    messages = [{"role": "system", "content": "Complete the checks."}]
    _append_history(messages, 0)
    host.compact(messages)
    host.receive_usage(100)

    def fail(*args, **kwargs):
        raise RuntimeError("summary failed")

    monkeypatch.setattr(host.engine, "_compress_impl", fail)
    with pytest.raises(RuntimeError, match="summary failed"):
        host.compact(messages)

    assert not host.engine._last_compression_made_progress
    host.receive_usage(100)
    assert host.attempts == 1


def test_auxiliary_usage_does_not_consume_main_verification(host, monkeypatch):
    messages = [{"role": "system", "content": "Complete the checks."}]
    _append_history(messages, 0)
    host.compact(messages)

    with monkeypatch.context() as patch:
        patch.setattr(host.engine, "_thread_context_stateless", lambda: True)
        patch.setattr(host.engine, "_thread_context_session_id", lambda: "")
        host.engine.update_from_response({"prompt_tokens": 100})

    host.receive_usage(100)
    assert host.attempts == 0


def test_session_reset_discards_pending_verification(host):
    messages = [{"role": "system", "content": "Complete the checks."}]
    _append_history(messages, 0)
    host.compact(messages)

    host.engine.on_session_reset()
    host.receive_usage(100)
    assert host.attempts == 1
    assert not host.engine._last_compression_made_progress
