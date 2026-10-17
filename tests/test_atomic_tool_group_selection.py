"""Budget selection keeps complete exchanges and upstream replay protection."""
import copy
import pytest
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(LCMConfig(database_path=str(tmp_path / "lcm.db")))
    instance._session_id = "test"
    instance.compression_count = 1
    try:
        yield instance
    finally:
        instance.shutdown()


def group(label, call_id="shared"):
    return [{"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "function": {"name": "read", "arguments": '{"path":"' + label + '"}'}},
        {"id": call_id + "-b", "function": {"name": "read", "arguments": "{}"}},
    ]}, {"role": "tool", "tool_call_id": call_id, "content": label + " result A " * 15},
        {"role": "tool", "tool_call_id": call_id + "-b", "content": label + " result B " * 15}]


def test_budget_cannot_keep_half_exchange_or_replace_real_result_with_stub(engine):
    user = {"role": "user", "content": "Read both files and compare them."}
    exchange = group("new")
    tail = [user] + exchange
    original = copy.deepcopy(tail)
    cap = count_messages_tokens([user] + exchange[1:])
    result = engine._assemble_context(None, tail, assembly_cap_override=cap)
    assert len(result) == 1
    assert result[0]["content"] == engine._build_preserved_objective_summary_part(user)
    assert tail == original


def test_complete_exchange_reserves_true_user_ahead_of_synthetic_todo(engine):
    user = {"role": "user", "content": "Compare my files."}
    todo = {"role": "user", "content": "[Your active task list was preserved across context compression]\nsynthetic task"}
    tail = [user] + group("new") + [todo]
    cap = count_messages_tokens(tail)
    result = engine._assemble_context(None, tail, assembly_cap_override=cap)
    assert user in result
    assert [m['content'] for m in result if m['role']=='tool'] == [m['content'] for m in tail if m['role']=='tool']


def test_reused_ids_never_borrow_result_from_other_turn(engine):
    old = [{"role": "user", "content": "old request"}] + group("old")
    new = [{"role": "user", "content": "new request"}] + group("new")
    cap = count_messages_tokens(new)
    result = engine._assemble_context(None, old + new, assembly_cap_override=cap)
    assert result == new
    assert all("old result" not in str(message) for message in result)


def test_selection_does_not_change_raw_rows_or_dag(engine):
    tail = [{"role": "user", "content": "request"}] + group("new")
    engine.ingest(tail)
    before = engine._store.get_session_messages("test")
    nodes = engine._dag.get_session_nodes("test")
    engine._assemble_context(None, tail, assembly_cap_override=20)
    assert engine._store.get_session_messages("test") == before
    assert engine._dag.get_session_nodes("test") == nodes


def test_irreducible_user_is_retained_with_visible_overflow_failure(engine):
    user = {"role": "user", "content": "critical request " * 100}
    result = engine._assemble_context(None, [user], assembly_cap_override=10)
    assert user in result
    engine._finalize_forced_overflow_result([user], result, assembly_cap_override=10)
    assert engine._last_overflow_recovery_failed


def test_dropped_protected_exchange_is_not_reported_as_success(engine):
    tail = [{"role": "user", "content": "request"}] + group("new")
    cap = count_messages_tokens([{"role": "user", "content": engine._build_preserved_objective_summary_part(tail[0])}])
    result = engine._assemble_context(None, tail, assembly_cap_override=cap)
    assert len(result) == 1
    assert result[0]["content"] == engine._build_preserved_objective_summary_part(tail[0])
    engine._finalize_forced_overflow_result(tail, result, assembly_cap_override=cap)
    assert engine._last_overflow_recovery_failed


def test_duplicate_ids_in_one_call_list_preserve_concrete_result_order(engine):
    tail = [{"role": "user", "content": "request"}] + group("same")
    tail[1]["tool_calls"][1]["id"] = "shared"
    tail[3]["tool_call_id"] = "shared"
    assert engine._assemble_context(None, tail, assembly_cap_override=count_messages_tokens(tail)) == tail


def test_userless_and_media_tool_groups_remain_atomic(engine):
    tail = group("media")
    tail[1]["content"] = [{"type": "text", "text": "image extracted"}]
    system = {"role": "system", "content": "system"}
    result = engine._assemble_context(system, tail, assembly_cap_override=count_messages_tokens([system] + tail), include_lcm_note=False)
    assert result == [system] + tail


def test_live_user_survives_an_over_budget_group_before_generated_todo(engine):
    user = {"role": "user", "content": "Compare my files."}
    todo = {"role": "user", "content": "[Your active task list was preserved across context compression]\nsynthetic task"}
    objective = engine._build_preserved_objective_summary_part(user)
    cap = count_messages_tokens([{"role": "user", "content": objective}, todo])
    tail = [user] + group("new") + [todo]
    result = engine._assemble_context(None, tail, assembly_cap_override=cap)
    assert any(objective in message.get("content", "") for message in result)
    assert not any(message["role"] == "tool" for message in result)
    engine._finalize_forced_overflow_result(tail, result, assembly_cap_override=cap)
    assert engine.get_status()["overflow_recovery_failed"]
