import json
import time

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


def test_compression_boundary_carries_summaries_without_moving_raw_messages(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start(
            "parent-session",
            platform="discord",
            conversation_id="discord-thread",
            context_length=200_000,
        )
        store_ids = engine._store.append_batch(
            "parent-session",
            [
                {"role": "user", "content": "raw parent payload"},
                {"role": "assistant", "content": "raw assistant payload"},
            ],
            source="discord",
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="parent-session",
                depth=0,
                summary="summary carried across compression boundary",
                token_count=8,
                source_token_count=12,
                source_ids=store_ids,
                source_type="messages",
                created_at=time.time(),
                earliest_at=time.time(),
                latest_at=time.time(),
                expand_hint="Expand for raw parent payload",
            )
        )
        externalized_dir = tmp_path / "externalized"
        externalized_dir.mkdir()
        payload_path = externalized_dir / "payload.json"
        payload_path.write_text(
            json.dumps(
                {
                    "kind": "ingest_payload",
                    "role": "tool",
                    "session_id": "parent-session",
                    "content": "large raw payload",
                    "created_at": time.time(),
                }
            ),
            encoding="utf-8",
        )

        engine.on_session_start(
            "child-session",
            platform="discord",
            conversation_id="discord-thread",
            context_length=200_000,
            boundary_reason="compression",
            old_session_id="parent-session",
        )

        assert engine._store.get_session_count("parent-session") == 2
        assert engine._store.get_session_count("child-session") == 0
        assert engine._dag.get_session_nodes("parent-session") == []
        child_nodes = engine._dag.get_session_nodes("child-session")
        assert len(child_nodes) == 1
        assert child_nodes[0].source_ids == store_ids
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert payload["session_id"] == "parent-session"
    finally:
        engine.shutdown()
