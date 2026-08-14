from __future__ import annotations

import json

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path):
    engine = LCMEngine(LCMConfig(database_path=str(tmp_path / "lcm.db")))
    engine.on_session_start(
        "session", conversation_id="conversation", context_length=1_000
    )
    return engine


def test_regex_grep_and_summary_scope(tmp_path):
    engine = _engine(tmp_path)
    try:
        messages = [
            {"role": "user", "content": "ticket ABC-123 is open"},
            {"role": "assistant", "content": "unrelated"},
        ]
        engine.ingest(messages)
        ids = engine._get_store_ids_for_messages(messages)
        node = SummaryNode(
            session_id="session",
            depth=0,
            summary="ticket summary",
            source_ids=[ids[0]],
            source_type="messages",
        )
        engine._dag.add_node(node)

        payload = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"mode": "regex", "query": r"ABC-\d+", "summary_id": node.node_id},
        ))
        assert [hit["store_id"] for hit in payload["results"]] == [ids[0]]
    finally:
        engine.shutdown()


def test_expand_is_not_exposed_to_root_and_is_defensively_denied(tmp_path):
    engine = _engine(tmp_path)
    try:
        root_names = {schema["name"] for schema in engine.get_tool_schemas(is_subagent=False)}
        child_names = {schema["name"] for schema in engine.get_tool_schemas(is_subagent=True)}
        assert "lcm_expand" not in root_names
        assert "lcm_expand_query" not in root_names
        assert {"lcm_expand", "lcm_expand_query"} <= child_names
        denied = json.loads(engine.handle_tool_call(
            "lcm_expand", {"store_id": 1}, is_subagent=False
        ))
        assert "restricted to sub-agents" in denied["error"]
    finally:
        engine.shutdown()


def test_llm_map_is_wired_to_engine_tool_surface(tmp_path):
    engine = _engine(tmp_path)
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"value": 2}\n{"value": 4}\n', encoding="utf-8")
    engine._llm_map_operator.executor = lambda **kwargs: {
        "doubled": kwargs["item"]["value"] * 2
    }
    try:
        names = {schema["name"] for schema in engine.get_tool_schemas()}
        assert {"llm_map", "agentic_map"} <= names
        payload = json.loads(engine.handle_tool_call("llm_map", {
            "input_path": str(input_path),
            "prompt": "double value",
            "output_schema": {
                "type": "object",
                "properties": {"doubled": {"type": "integer"}},
                "required": ["doubled"],
                "additionalProperties": False,
            },
        }))
        assert payload["status"] == "completed"
        assert payload["completed"] == 2
        assert payload["output_file_id"].startswith("file_")
    finally:
        engine.shutdown()


def test_describe_accepts_opaque_file_ids(tmp_path):
    engine = _engine(tmp_path)
    source = tmp_path / "large.json"
    source.write_text('{"records":[{"id":1}]}', encoding="utf-8")
    try:
        record = engine._file_registry.register(source)
        payload = json.loads(engine.handle_tool_call(
            "lcm_describe", {"file_id": record.file_id}
        ))
        assert payload["file_id"] == record.file_id
        assert payload["path"] == str(source.resolve())
        assert "records" in payload["exploration_summary"]
    finally:
        engine.shutdown()
