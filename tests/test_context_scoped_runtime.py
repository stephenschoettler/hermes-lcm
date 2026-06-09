import sqlite3
import types

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _make_engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm_context_scope.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        dynamic_leaf_chunk_enabled=False,
        extraction_enabled=False,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))

    def fake_summary(self, chunk, focus_topic=None):
        joined = " | ".join(str(m.get("content", "")) for m in chunk)
        return chunk, 123, f"SUMMARY_FROM_{joined}", "fake", 1

    engine._summarize_leaf_chunk_with_rescue = types.MethodType(fake_summary, engine)
    return engine


def _rows(tmp_path, query):
    con = sqlite3.connect(tmp_path / "lcm_context_scope.db")
    con.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in con.execute(query)]
    finally:
        con.close()


def test_shared_engine_compress_uses_gateway_conversation_context(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path)

    engine.on_session_start(
        "A-session",
        platform="discord",
        conversation_id="conv-A-thread",
        context_length=200000,
    )
    engine.on_session_start(
        "B-session",
        platform="discord",
        conversation_id="conv-B-thread",
        context_length=200000,
    )

    # Simulate a cached AIAgent turn: no fresh on_session_start() runs for A,
    # but the gateway's task-local session key identifies the active thread.
    monkeypatch.setattr(
        engine,
        "_gateway_context_value",
        lambda name: "conv-A-thread" if name == "HERMES_SESSION_KEY" else "",
    )

    engine.compress(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "A_ONLY_user_1"},
            {"role": "assistant", "content": "A_ONLY_assistant_1"},
            {"role": "user", "content": "A_ONLY_tail"},
        ],
        current_tokens=999999,
    )

    messages = _rows(
        tmp_path,
        "select session_id, role, content from messages order by store_id",
    )
    nodes = _rows(
        tmp_path,
        "select session_id, summary from summary_nodes order by node_id",
    )
    lifecycle = _rows(
        tmp_path,
        "select conversation_id, current_session_id, current_frontier_store_id "
        "from lcm_lifecycle_state order by conversation_id",
    )

    assert {row["session_id"] for row in messages} == {"A-session"}
    assert nodes == [
        {
            "session_id": "A-session",
            "summary": "SUMMARY_FROM_A_ONLY_user_1 | A_ONLY_assistant_1",
        }
    ]
    assert lifecycle == [
        {
            "conversation_id": "conv-A-thread",
            "current_session_id": "A-session",
            "current_frontier_store_id": 3,
        },
        {
            "conversation_id": "conv-B-thread",
            "current_session_id": "B-session",
            "current_frontier_store_id": 0,
        },
    ]


def test_compression_boundary_updates_the_scoped_conversation_state(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path)

    engine.on_session_start(
        "A-old",
        platform="discord",
        conversation_id="conv-A-thread",
        context_length=200000,
    )
    monkeypatch.setattr(
        engine,
        "_gateway_context_value",
        lambda name: "conv-A-thread" if name == "HERMES_SESSION_KEY" else "",
    )
    engine.compress(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "A_OLD_user"},
            {"role": "assistant", "content": "A_OLD_assistant"},
            {"role": "user", "content": "A_OLD_tail"},
        ],
        current_tokens=999999,
    )

    engine.on_session_start(
        "B-session",
        platform="discord",
        conversation_id="conv-B-thread",
        context_length=200000,
    )
    engine.on_session_start(
        "A-child",
        platform="discord",
        conversation_id="conv-A-thread",
        boundary_reason="compression",
        old_session_id="A-old",
        context_length=200000,
    )

    monkeypatch.setattr(
        engine,
        "_gateway_context_value",
        lambda name: "conv-A-thread" if name == "HERMES_SESSION_KEY" else "",
    )
    engine.compress(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "A_CHILD_user"},
            {"role": "assistant", "content": "A_CHILD_assistant"},
            {"role": "user", "content": "A_CHILD_tail"},
        ],
        current_tokens=999999,
    )

    nodes = _rows(
        tmp_path,
        "select session_id, summary from summary_nodes order by node_id",
    )
    lifecycle = _rows(
        tmp_path,
        "select conversation_id, current_session_id, last_finalized_session_id "
        "from lcm_lifecycle_state order by conversation_id",
    )

    assert [row["session_id"] for row in nodes] == ["A-child", "A-child"]
    assert "A_CHILD_user" in nodes[1]["summary"]
    assert lifecycle == [
        {
            "conversation_id": "conv-A-thread",
            "current_session_id": "A-child",
            "last_finalized_session_id": "A-old",
        },
        {
            "conversation_id": "conv-B-thread",
            "current_session_id": "B-session",
            "last_finalized_session_id": None,
        },
    ]
