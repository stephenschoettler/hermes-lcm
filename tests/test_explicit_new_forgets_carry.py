"""Explicit /new drops the predecessor summary graph and cannot be carried back."""

import os

from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.session_forget import forget_explicit_new


def test_explicit_new_deletes_predecessor_nodes_and_clears_carry(tmp_path, monkeypatch):
    db_path = tmp_path / "lcm.db"
    monkeypatch.setenv("LCM_DATABASE_PATH", str(db_path))
    dag = SummaryDAG(db_path)
    lifecycle = LifecycleStateStore(db_path)
    try:
        lifecycle.bind_session("old-session", conversation_id="agent:main:telegram:group:-1:11")
        lifecycle.finalize_session("agent:main:telegram:group:-1:11", "old-session")
        lifecycle.bind_session("new-session", conversation_id="agent:main:telegram:group:-1:11")
        dag.add_node(SummaryNode(session_id="old-session", depth=0, summary="old photo handoff"))
        dag.add_node(SummaryNode(session_id="old-session", depth=2, summary="kept by retain depth, wiped by /new"))
        dag.add_node(SummaryNode(session_id="projects-session", depth=0, summary="other topic"))
        lifecycle.bind_session("projects-session", conversation_id="agent:main:telegram:group:-1:106")
    finally:
        dag.close()
        lifecycle.close()

    result = forget_explicit_new(
        "agent:main:telegram:group:-1:11",
        extra_session_ids=["old-session", "projects-session"],
    )

    assert result["found"] is True
    assert result["deleted_nodes"] == 2
    dag = SummaryDAG(db_path)
    lifecycle = LifecycleStateStore(db_path)
    try:
        assert dag.get_session_nodes("old-session") == []
        assert len(dag.get_session_nodes("projects-session")) == 1
        state = lifecycle.get_by_conversation("agent:main:telegram:group:-1:11")
        assert state.last_finalized_session_id is None
        assert state.last_reset_at is not None
        assert dag.reassign_session_nodes("old-session", "new-session") == 0
        projects = lifecycle.get_by_conversation("agent:main:telegram:group:-1:106")
        assert projects.current_session_id == "projects-session"
    finally:
        dag.close()
        lifecycle.close()
        os.environ.pop("LCM_DATABASE_PATH", None)
