"""Regression: auxiliary caller detection must not starve per-agent clones.

Match the host's order: clone an unconfigured registered prototype, update the
clone from an auxiliary agent frame, then bind with the clone's context length.
Do not mock auxiliary detection or initialize the prototype with the child model.
"""
from pathlib import Path
from typing import Iterator
from unittest.mock import Mock

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


class ChildAgentFrame:
    """The attributes available in AIAgent.__init__ before delegate IDs are set."""

    def __init__(self, prototype: LCMEngine, *, parent_id: str = "parent") -> None:
        self.session_id = "child"
        self._parent_session_id = parent_id
        self.log_prefix = "[subagent-0]"
        self.platform = "subagent"
        self.engine = prototype.clone_for_agent()
        self.update_model(self.engine, 1_000_000)
        self.engine.on_session_start(
            self.session_id,
            platform=self.platform,
            model="claude-sonnet-5-5",
            context_length=self.engine.context_length,
        )

    def update_model(self, engine: LCMEngine, context_length: int) -> None:
        engine.update_model(
            model="claude-sonnet-5-5",
            context_length=context_length,
            provider="anthropic",
            base_url="https://child.example.invalid",
            api_mode="messages",
        )

    def gates(self, tokens: int) -> tuple[bool, bool]:
        self.engine.update_from_response({"prompt_tokens": tokens})
        return (
            self.engine.should_compress(tokens),
            self.engine.should_compress(),
        )

    def preflight(self, messages: list[dict[str, str]]) -> bool:
        return self.engine.should_compress_preflight(messages)

    def compress_if_due(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if self.engine.should_compress(468_000):
            return self.engine.compress(messages, current_tokens=468_000)
        return messages


@pytest.fixture
def prototype(tmp_path: Path) -> Iterator[LCMEngine]:
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            context_threshold=0.22,
        ),
        hermes_home=str(tmp_path),
    )
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.mark.parametrize("primed", [False, True], ids=["empty-prototype", "different-parent-model"])
def test_child_clone_receives_own_runtime_budget(prototype: LCMEngine, primed: bool) -> None:
    if primed:
        prototype.update_model("parent-model", 200_000, provider="parent-provider")
        prototype.on_session_start("parent", platform="cli", context_length=200_000)
    before = (prototype.model, prototype.context_length, prototype.threshold_tokens, prototype._session_id)
    child = ChildAgentFrame(prototype)
    try:
        assert child.engine.model == "claude-sonnet-5-5"
        assert child.engine.provider == "anthropic"
        assert child.engine.base_url == "https://child.example.invalid"
        assert child.engine.context_length == 1_000_000
        assert child.engine.threshold_tokens == 220_000
        assert child.engine._session_id == ""  # Auxiliary classification remains intact.
        assert child.engine._thread_context_has_auxiliary_session("child")
        assert (prototype.model, prototype.context_length, prototype.threshold_tokens, prototype._session_id) == before
    finally:
        child.engine.shutdown()


@pytest.mark.parametrize("tokens, expected", [(219_999, False), (220_000, True), (468_000, True)])
def test_child_compaction_gates_use_resolved_window(
    prototype: LCMEngine, monkeypatch: pytest.MonkeyPatch, tokens: int, expected: bool,
) -> None:
    child = ChildAgentFrame(prototype)
    messages = [{"role": "user", "content": "child work"}]
    try:
        # Isolate tokenization cost, not the threshold, caller-frame, or usage paths.
        monkeypatch.setattr("hermes_lcm.compaction.count_messages_tokens", lambda _messages: tokens)
        assert child.gates(tokens) == (expected, expected)
        assert child.preflight(messages) is expected
        assert child.engine._store.get_session_count("child") == 0
        assert child.engine._store.get_session_count("parent") == 0
    finally:
        child.engine.shutdown()


def test_child_pressure_reaches_host_fallback(prototype: LCMEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    child = ChildAgentFrame(prototype)
    messages = [{"role": "user", "content": "old work"}, {"role": "assistant", "content": "results"}]
    compacted = [{"role": "user", "content": "summarized child work"}]
    fallback = Mock(compression_count=0)
    fallback.compress.return_value = compacted
    monkeypatch.setattr(child.engine, "_get_host_fallback_compressor", lambda: fallback)
    try:
        assert child.compress_if_due(messages) == compacted
        fallback.compress.assert_called_once()
        assert fallback.update_model.call_args.kwargs["context_length"] == 1_000_000
        assert child.engine.compression_count == 1
        assert child.engine._store.get_session_count("child") == 0
    finally:
        child.engine.shutdown()


def test_shared_bound_parent_remains_protected(prototype: LCMEngine) -> None:
    prototype.update_model("parent-model", 200_000, provider="parent-provider")
    prototype.on_session_start("parent", platform="cli", context_length=200_000)
    child = ChildAgentFrame(prototype)
    try:
        child.update_model(prototype, 1_000_000)
        assert prototype.model == "parent-model"
        assert prototype.provider == "parent-provider"
        assert prototype.context_length == 200_000
        assert prototype.threshold_tokens == 44_000
        # A later model switch on the child's own unbound auxiliary clone works.
        child.update_model(child.engine, 500_000)
        assert child.engine.context_length == 500_000
        assert child.engine.threshold_tokens == 110_000
    finally:
        child.engine.shutdown()
