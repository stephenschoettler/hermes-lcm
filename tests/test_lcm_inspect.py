"""Read-only lcm_inspect tool contract tests."""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.externalize import get_large_output_storage_dir
from hermes_lcm import tools as lcm_tools


def _import_lcm_engine():
    try:
        from hermes_lcm.engine import LCMEngine
        return LCMEngine
    except ModuleNotFoundError as exc:
        if exc.name not in {"agent", "agent.context_engine"}:
            raise
        agent_module = sys.modules.get("agent")
        if agent_module is None:
            agent_module = ModuleType("agent")
            agent_module.__path__ = []
            sys.modules["agent"] = agent_module

        context_engine_module = ModuleType("agent.context_engine")

        class ContextEngine:
            def on_session_reset(self):
                return None

        setattr(context_engine_module, "ContextEngine", ContextEngine)
        sys.modules["agent.context_engine"] = context_engine_module
        setattr(agent_module, "context_engine", context_engine_module)
        sys.modules.pop("hermes_lcm.engine", None)
        from hermes_lcm.engine import LCMEngine
        return LCMEngine


LCMEngine = _import_lcm_engine()


def _make_engine(tmp_path: Path, **config_overrides):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=config_overrides.pop("fresh_tail_count", 2),
        **config_overrides,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "sess-current",
        platform="discord",
        conversation_id="discord:channel:thread",
    )
    return engine


def _seed_messages(engine):
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow up"},
        {"role": "assistant", "content": "final answer"},
    ]
    engine.ingest(messages)
    return engine._store.load_session_page("sess-current", limit=10)


def _write_externalized_payload(engine, *, ref="payload-test.json", session_id="sess-current"):
    storage_dir = get_large_output_storage_dir(
        engine._config,
        hermes_home=engine._hermes_home,
        create=True,
    )
    payload = {
        "kind": "tool_result",
        "tool_call_id": "call-1",
        "role": "tool",
        "session_id": session_id,
        "field_path": "content",
        "content_chars": 17,
        "content_bytes": 17,
        "created_at": 123.0,
        "content": "hidden tool output",
    }
    (storage_dir / ref).write_text(json.dumps(payload), encoding="utf-8")
    return ref


def test_lcm_inspect_reports_bounded_metadata_without_content(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        rows = _seed_messages(engine)
        compacted_store_ids = [rows[0]["store_id"], rows[1]["store_id"], rows[2]["store_id"]]
        engine._dag.add_node(
            SummaryNode(
                session_id="sess-current",
                depth=0,
                summary="compacted setup",
                token_count=7,
                source_token_count=40,
                source_ids=compacted_store_ids,
                source_type="messages",
                created_at=100.0,
                earliest_at=1.0,
                latest_at=3.0,
                expand_hint="setup",
            )
        )
        ref = _write_externalized_payload(engine)
        engine._store._conn.execute(
            "UPDATE messages SET content = content || ? WHERE store_id = ?",
            (f" [Externalized tool output: tool_call_id=call-1; chars=17; bytes=17; ref={ref}]", rows[-1]["store_id"]),
        )
        engine._store._conn.commit()
        engine._last_compression_status = "noop"
        engine._last_compression_noop_reason = "no eligible raw backlog outside fresh tail"
        engine._last_compacted_store_id = max(compacted_store_ids)

        result = json.loads(engine.handle_tool_call("lcm_inspect", {"limit": 2}))

        assert result["read_only"] is True
        assert result["session_id"] == "sess-current"
        assert result["conversation_id"] == "discord:channel:thread"
        assert result["messages"]["total"] == 5
        assert result["messages"]["fresh_tail"]["returned"] == 2
        assert [item["store_id"] for item in result["messages"]["fresh_tail"]["items"]] == [rows[-2]["store_id"], rows[-1]["store_id"]]
        assert all("content" not in item for item in result["messages"]["fresh_tail"]["items"])
        assert result["compaction"]["frontier"]["runtime_last_compacted_store_id"] == max(compacted_store_ids)
        assert result["compaction"]["frontier"]["highest_compacted_source_store_id"] == max(compacted_store_ids)
        assert result["compaction"]["last"]["status"] == "noop"
        assert "no eligible raw backlog" in result["compaction"]["last"]["noop_reason"]
        assert result["dag"]["total_nodes"] == 1
        assert result["dag"]["latest_nodes"][0]["node_id"] >= 1
        assert "expand_hint" not in result["dag"]["latest_nodes"][0]
        assert result["dag"]["latest_nodes"][0]["expand_hint_available"] is True
        assert result["dag"]["latest_nodes"][0]["expand_hint_chars"] == len("setup")
        assert result["externalized_refs"]["total_known"] == 1
        assert result["externalized_refs"]["items"][0]["externalized_ref"] == ref
        assert result["externalized_refs"]["items"][0]["readable"] is True
        assert result["externalized_refs"]["items"][0]["file_size_bytes"] > 0
        assert result["externalized_refs"]["items"][0]["payload_session_id"] == "sess-current"
        assert result["externalized_refs"]["items"][0]["payload_validation"] == "metadata_prefix"
        assert result["externalized_refs"]["items"][0]["store_id"] == rows[-1]["store_id"]
        assert "content_preview" not in result["externalized_refs"]["items"][0]
        assert "content" not in result["externalized_refs"]["items"][0]
        assert "content_chars" not in result["externalized_refs"]["items"][0]
        assert result["ingest_protection"]["sensitive_patterns_enabled"] is False
        assert "api_key" in result["ingest_protection"]["sensitive_patterns"]
    finally:
        engine.shutdown()


def test_lcm_inspect_includes_sensitive_pattern_status(tmp_path):
    engine = _make_engine(
        tmp_path,
        sensitive_patterns_enabled=True,
        sensitive_patterns=["api_key", "typoed_pattern"],
    )
    try:
        result = json.loads(engine.handle_tool_call("lcm_inspect", {}))

        protection = result["ingest_protection"]
        assert protection["sensitive_patterns_enabled"] is True
        assert protection["enabled"] is True
        assert protection["sensitive_patterns"] == ["api_key", "typoed_pattern"]
        assert protection["patterns"] == ["api_key", "typoed_pattern"]
        assert protection["active_patterns"] == ["api_key"]
        assert protection["unknown_patterns"] == ["typoed_pattern"]
        assert protection["source"] == "default"
        assert protection["lossless_recovery"] is False
        assert "placeholder_format" in protection
    finally:
        engine.shutdown()


def test_lcm_inspect_skips_tool_dispatch_ingest_with_messages(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        before = {
            "store_changes": engine._store._conn.total_changes,
            "message_count": engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        }

        result = json.loads(
            engine.handle_tool_call(
                "lcm_inspect",
                {},
                messages=[{"role": "user", "content": "current turn must not be ingested"}],
            )
        )

        after = {
            "store_changes": engine._store._conn.total_changes,
            "message_count": engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        }
        assert result["read_only"] is True
        assert result["messages"]["total"] == 0
        assert after == before
    finally:
        engine.shutdown()


def test_lcm_inspect_honors_zero_fresh_tail_count(tmp_path):
    engine = _make_engine(tmp_path, fresh_tail_count=0)
    try:
        _seed_messages(engine)

        result = json.loads(engine.handle_tool_call("lcm_inspect", {"limit": 2}))

        assert result["messages"]["fresh_tail_count"] == 0
        assert result["messages"]["pre_tail_message_count"] == 5
        assert result["messages"]["fresh_tail"]["returned"] == 0
        assert result["messages"]["fresh_tail"]["items"] == []
    finally:
        engine.shutdown()


def test_lcm_inspect_externalized_ref_scan_reports_truncation(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path)
    try:
        ref = _write_externalized_payload(engine, ref="payload-after-scan-window.json")
        engine._store.append(
            "sess-current",
            {"role": "user", "content": "first row without refs"},
            source="discord",
            conversation_id="discord:channel:thread",
        )
        engine._store.append(
            "sess-current",
            {"role": "assistant", "content": f"[Externalized tool output: tool_call_id=call-1; chars=17; bytes=17; ref={ref}]"},
            source="discord",
            conversation_id="discord:channel:thread",
        )
        monkeypatch.setattr(lcm_tools, "_LCM_INSPECT_REF_SCAN_MESSAGE_LIMIT", 1)

        result = json.loads(engine.handle_tool_call("lcm_inspect", {}))

        assert result["externalized_refs"]["scanned_messages"] == 1
        assert result["externalized_refs"]["scan_truncated"] is True
        assert result["externalized_refs"]["total_known_exact"] is False
        assert result["externalized_refs"]["has_more"] is True
        assert result["externalized_refs"]["total_known"] == 0
    finally:
        engine.shutdown()


def test_lcm_inspect_payload_metadata_prefix_stops_before_content(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text(
        json.dumps(
            {
                "kind": "tool_result",
                "session_id": "sess-current",
                "content": "SECRET_PAYLOAD_BODY" * 1024,
                "content_chars": 19 * 1024,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    prefix_text, stopped_at_content, prefix_truncated = lcm_tools._read_externalized_payload_metadata_prefix(path)

    assert stopped_at_content is True
    assert prefix_truncated is False
    assert '"session_id"' in prefix_text
    assert '"content"' in prefix_text
    assert "SECRET_PAYLOAD_BODY" not in prefix_text


def test_lcm_inspect_payload_metadata_prefix_accepts_valid_non_ascii_before_content(tmp_path):
    path = tmp_path / "payload.json"
    path.write_text(
        json.dumps(
            {
                "kind": "tool_result",
                "session_id": "sess-current",
                "label": "café",
                "content": "SECRET_PAYLOAD_BODY" * 1024,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    prefix_text, stopped_at_content, prefix_truncated = lcm_tools._read_externalized_payload_metadata_prefix(path)

    assert stopped_at_content is True
    assert prefix_truncated is False
    assert '"label"' in prefix_text
    assert "café" in prefix_text
    assert "SECRET_PAYLOAD_BODY" not in prefix_text


def test_lcm_inspect_payload_metadata_parser_uses_top_level_session_owner():
    fields, content_key_seen = lcm_tools._inspect_top_level_json_string_fields_before_content(
        json.dumps(
            {
                "metadata": {"session_id": "nested-session"},
                "session_id": "top-level-session",
                "content": "payload body",
            }
        )
    )

    assert content_key_seen is True
    assert fields["session_id"] == "top-level-session"

    fields, content_key_seen = lcm_tools._inspect_top_level_json_string_fields_before_content(
        '{"session_id":"first","session_id":"second","content":"payload body"}'
    )

    assert content_key_seen is True
    assert fields["session_id"] == "second"

    fields, content_key_seen = lcm_tools._inspect_top_level_json_string_fields_before_content(
        '{"session_id":"first","session_id":123,"content":"payload body"}'
    )

    assert content_key_seen is True
    assert "session_id" not in fields


def test_lcm_inspect_rejects_cross_session_externalized_refs(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        ref = _write_externalized_payload(
            engine,
            ref="payload-other-session.json",
            session_id="other-session",
        )
        store_id = engine._store.append(
            "sess-current",
            {"role": "assistant", "content": f"pasted placeholder [Externalized tool output: tool_call_id=call-1; chars=17; bytes=17; ref={ref}]"},
            source="discord",
            conversation_id="discord:channel:thread",
        )

        result = json.loads(engine.handle_tool_call("lcm_inspect", {}))

        item = result["externalized_refs"]["items"][0]
        assert item["externalized_ref"] == ref
        assert item["store_id"] == store_id
        assert item["readable"] is False
        assert item["error"] == "session_mismatch"
        assert "file_size_bytes" not in item
        assert "modified_at" not in item
        assert "payload_session_id" not in item
    finally:
        engine.shutdown()


def test_lcm_inspect_rejects_nested_session_spoofed_externalized_refs(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        storage_dir = get_large_output_storage_dir(
            engine._config,
            hermes_home=engine._hermes_home,
            create=True,
        )
        ref = "payload-nested-spoof.json"
        (storage_dir / ref).write_text(
            json.dumps(
                {
                    "metadata": {"session_id": "sess-current"},
                    "session_id": "other-session",
                    "content": "payload body",
                    "content_chars": 12,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        engine._store.append(
            "sess-current",
            {"role": "assistant", "content": f"pasted placeholder [Externalized tool output: tool_call_id=call-1; chars=12; bytes=12; ref={ref}]"},
            source="discord",
            conversation_id="discord:channel:thread",
        )

        result = json.loads(engine.handle_tool_call("lcm_inspect", {}))

        item = result["externalized_refs"]["items"][0]
        assert item["readable"] is False
        assert item["error"] == "session_mismatch"
        assert "file_size_bytes" not in item
        assert "modified_at" not in item
    finally:
        engine.shutdown()


def test_lcm_inspect_rejects_payload_refs_without_session_metadata(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        storage_dir = get_large_output_storage_dir(
            engine._config,
            hermes_home=engine._hermes_home,
            create=True,
        )
        ref = "payload-without-session.json"
        (storage_dir / ref).write_text(
            json.dumps(
                {
                    "kind": "tool_result",
                    "content": "legacy payload body",
                    "content_chars": 19,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        engine._store.append(
            "sess-current",
            {"role": "assistant", "content": f"pasted placeholder [Externalized tool output: tool_call_id=call-1; chars=19; bytes=19; ref={ref}]"},
            source="discord",
            conversation_id="discord:channel:thread",
        )

        result = json.loads(engine.handle_tool_call("lcm_inspect", {}))

        item = result["externalized_refs"]["items"][0]
        assert item["readable"] is False
        assert item["error"] == "session_metadata_unavailable"
        assert "file_size_bytes" not in item
        assert "modified_at" not in item
    finally:
        engine.shutdown()


def test_lcm_inspect_rejects_payload_refs_when_session_metadata_is_beyond_prefix(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        storage_dir = get_large_output_storage_dir(
            engine._config,
            hermes_home=engine._hermes_home,
            create=True,
        )
        ref = "payload-session-beyond-prefix.json"
        (storage_dir / ref).write_text(
            json.dumps(
                {
                    "kind": "tool_result",
                    "padding": "x" * 20_000,
                    "session_id": "sess-current",
                    "content": "payload body",
                    "content_chars": 12,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        engine._store.append(
            "sess-current",
            {"role": "assistant", "content": f"pasted placeholder [Externalized tool output: tool_call_id=call-1; chars=12; bytes=12; ref={ref}]"},
            source="discord",
            conversation_id="discord:channel:thread",
        )

        result = json.loads(engine.handle_tool_call("lcm_inspect", {}))

        item = result["externalized_refs"]["items"][0]
        assert item["readable"] is False
        assert item["error"] == "session_metadata_unavailable"
        assert "file_size_bytes" not in item
        assert "modified_at" not in item
    finally:
        engine.shutdown()


def test_lcm_inspect_uses_bounded_prefix_for_payload_readability(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        storage_dir = get_large_output_storage_dir(
            engine._config,
            hermes_home=engine._hermes_home,
            create=True,
        )
        ref = "payload-corrupt-after-content.json"
        (storage_dir / ref).write_text(
            '{"kind":"tool_result","session_id":"sess-current","content":"unterminated',
            encoding="utf-8",
        )
        engine._store.append(
            "sess-current",
            {"role": "assistant", "content": f"pasted placeholder [Externalized tool output: tool_call_id=call-1; chars=12; bytes=12; ref={ref}]"},
            source="discord",
            conversation_id="discord:channel:thread",
        )

        result = json.loads(engine.handle_tool_call("lcm_inspect", {}))

        item = result["externalized_refs"]["items"][0]
        assert item["readable"] is True
        assert item["payload_session_id"] == "sess-current"
        assert item["payload_validation"] == "metadata_prefix"
        assert item["file_size_bytes"] > 0
        assert "content" not in item
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("payload_text", "ref"),
    [
        ('{"kind":"tool_result","session_id":"sess-current",\f"content":"x"}', "payload-formfeed.json"),
        ('{"kind":"tool_result","session_id":"sess-current",\u00a0"content":"x"}', "payload-nbsp.json"),
        ('{"kind":"tool_result","session_id":"sess-current" "content":"x"}', "payload-missing-comma.json"),
        ('{"kind":"tool_result","session_id":"sess-current"}', "payload-missing-content.json"),
        ('{"kind":"tool_result","session_id":"sess-current","content":1}', "payload-non-string-content.json"),
    ],
)
def test_lcm_inspect_rejects_invalid_payload_metadata_prefix(tmp_path, payload_text, ref):
    engine = _make_engine(tmp_path)
    try:
        storage_dir = get_large_output_storage_dir(
            engine._config,
            hermes_home=engine._hermes_home,
            create=True,
        )
        (storage_dir / ref).write_text(payload_text, encoding="utf-8")
        engine._store.append(
            "sess-current",
            {"role": "assistant", "content": f"pasted placeholder [Externalized tool output: tool_call_id=call-1; chars=12; bytes=12; ref={ref}]"},
            source="discord",
            conversation_id="discord:channel:thread",
        )

        result = json.loads(engine.handle_tool_call("lcm_inspect", {}))

        item = result["externalized_refs"]["items"][0]
        assert item["readable"] is False
        assert item["error"] == "invalid_payload"
        assert "file_size_bytes" not in item
        assert "modified_at" not in item
    finally:
        engine.shutdown()


def test_lcm_inspect_rejects_invalid_utf8_payload_metadata_prefix(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        storage_dir = get_large_output_storage_dir(
            engine._config,
            hermes_home=engine._hermes_home,
            create=True,
        )
        ref = "payload-invalid-utf8-prefix.json"
        (storage_dir / ref).write_bytes(b'{"kind":"tool_result","session_id":"sess-current",\xff"content":"x"}')
        engine._store.append(
            "sess-current",
            {"role": "assistant", "content": f"pasted placeholder [Externalized tool output: tool_call_id=call-1; chars=12; bytes=12; ref={ref}]"},
            source="discord",
            conversation_id="discord:channel:thread",
        )

        result = json.loads(engine.handle_tool_call("lcm_inspect", {}))

        item = result["externalized_refs"]["items"][0]
        assert item["readable"] is False
        assert item["error"] == "invalid_payload"
        assert "file_size_bytes" not in item
        assert "modified_at" not in item
    finally:
        engine.shutdown()


def test_lcm_inspect_caps_payload_validation_to_returned_refs(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path)
    try:
        for index in range(5):
            ref = _write_externalized_payload(engine, ref=f"payload-{index}.json")
            engine._store.append(
                "sess-current",
                {"role": "assistant", "content": f"placeholder [Externalized tool output: tool_call_id=call-{index}; chars=17; bytes=17; ref={ref}]"},
                source="discord",
                conversation_id="discord:channel:thread",
            )

        calls = []

        def fake_metadata(_engine, ref, session_id):
            calls.append((ref, session_id))
            return {"readable": True, "payload_session_id": session_id}

        monkeypatch.setattr(lcm_tools, "_inspect_externalized_payload_metadata", fake_metadata)

        result = json.loads(engine.handle_tool_call("lcm_inspect", {"limit": 2}))

        assert result["externalized_refs"]["total_known"] == 5
        assert result["externalized_refs"]["returned"] == 2
        assert result["externalized_refs"]["has_more"] is True
        assert len(calls) == 2
    finally:
        engine.shutdown()


def test_lcm_inspect_finds_externalized_refs_inside_decoded_tool_calls(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        ref = _write_externalized_payload(engine, ref="payload-tool-call.json")
        store_id = engine._store.append(
            "sess-current",
            {
                "role": "assistant",
                "content": "tool call with externalized arguments",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "demo",
                            "arguments": f"[Externalized payload: kind=tool_call; role=assistant; chars=17; bytes=17; ref={ref}]",
                        },
                    }
                ],
            },
            source="discord",
            conversation_id="discord:channel:thread",
        )

        row = engine._store.load_session_page("sess-current", limit=1)[0]
        assert isinstance(row["tool_calls"], list)

        result = json.loads(engine.handle_tool_call("lcm_inspect", {}))

        assert result["externalized_refs"]["total_known"] == 1
        item = result["externalized_refs"]["items"][0]
        assert item["externalized_ref"] == ref
        assert item["store_id"] == store_id
        assert item["readable"] is True
        assert result["messages"]["fresh_tail"]["items"][0]["externalized_refs"] == [ref]
    finally:
        engine.shutdown()


def test_lcm_inspect_surfaces_matched_session_patterns(tmp_path):
    engine = _make_engine(
        tmp_path,
        ignore_session_patterns=["discord:*"],
        stateless_session_patterns=["sess-*"],
    )
    try:
        result = json.loads(engine.handle_tool_call("lcm_inspect", {}))

        assert result["filters"]["session_keys"] == [
            "sess-current",
            "discord",
            "discord:sess-current",
        ]
        assert result["filters"]["ignored"] is True
        # An ignored session is not actively stateless, but the inspector still
        # reports which stateless patterns would match the session keys.
        assert result["filters"]["stateless"] is False
        assert result["filters"]["matched_ignore_session_patterns"] == ["discord:*"]
        assert result["filters"]["matched_stateless_session_patterns"] == ["sess-*"]
    finally:
        engine.shutdown()


def test_lcm_inspect_is_read_only_for_database_connections(tmp_path):
    engine = _make_engine(tmp_path)
    try:
        _seed_messages(engine)
        before = {
            "store_changes": engine._store._conn.total_changes,
            "dag_changes": engine._dag._conn.total_changes,
            "lifecycle_changes": engine._lifecycle._conn.total_changes,
            "max_store_id": engine._store._conn.execute("SELECT MAX(store_id) FROM messages").fetchone()[0],
            "node_count": engine._dag._conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0],
            "lifecycle_rows": engine._lifecycle._conn.execute("SELECT COUNT(*) FROM lcm_lifecycle_state").fetchone()[0],
        }

        result = json.loads(engine.handle_tool_call("lcm_inspect", {"limit": 500}))

        after = {
            "store_changes": engine._store._conn.total_changes,
            "dag_changes": engine._dag._conn.total_changes,
            "lifecycle_changes": engine._lifecycle._conn.total_changes,
            "max_store_id": engine._store._conn.execute("SELECT MAX(store_id) FROM messages").fetchone()[0],
            "node_count": engine._dag._conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0],
            "lifecycle_rows": engine._lifecycle._conn.execute("SELECT COUNT(*) FROM lcm_lifecycle_state").fetchone()[0],
        }
        assert result["limit"] == 200
        assert result["limit_clamped_from"] == 500
        assert after == before
    finally:
        engine.shutdown()
