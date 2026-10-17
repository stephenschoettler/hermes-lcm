"""Generated summary context must not become a current user objective."""

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


def summary(role="user"):
    return {"role": role, "content": "[Recent Summary (d0, node 1)]\nHistorical work\n[Expand for details: 1]"}


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_summary_boundary_blocks_stale_user_objective(engine, role):
    messages = [
        {"role": "user", "content": "an older request"}, summary(role),
        {"role": "assistant", "content": "continuation"},
    ]
    assert engine._latest_user_context_anchor(messages, messages[-1:]) is None


def test_new_real_user_after_summary_remains_eligible(engine):
    messages = [summary(), {"role": "user", "content": "a new request"}, {"role": "assistant", "content": "continuation"}]
    anchor = engine._latest_user_context_anchor(messages, messages[-1:])
    assert "a new request" in anchor
    assert "Historical work" not in anchor


def test_user_request_already_in_tail_needs_no_anchor(engine):
    messages = [summary(), {"role": "user", "content": "a new request"}]
    assert engine._latest_user_context_anchor(messages, messages[-1:]) is None


def test_ordinary_user_text_is_not_mistaken_for_generated_summary(engine):
    messages = [{"role": "user", "content": "Please write a summary"}, {"role": "assistant", "content": "continuation"}]
    assert "Please write a summary" in engine._latest_user_context_anchor(messages, messages[-1:])


def test_lcm_system_note_is_also_a_continuity_boundary(engine):
    note = {"role": "system", "content": "[Note: This conversation uses Lossless Context Management (LCM). Earlier turns have been compacted into hierarchical summaries below.]"}
    messages = [{"role": "user", "content": "an older request"}, note, {"role": "assistant", "content": "continuation"}]
    assert engine._latest_user_context_anchor(messages, messages[-1:]) is None
