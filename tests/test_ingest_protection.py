"""Regression tests for storage-boundary payload protection."""

from __future__ import annotations

import base64
import importlib.util
import json
import re
import sqlite3
import stat
import sys
from copy import deepcopy
from pathlib import Path

from hermes_lcm import tools as lcm_tools
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
import hermes_lcm.engine as lcm_engine_module
import hermes_lcm.store as lcm_store_module
from hermes_lcm.extraction import sanitize_pre_compaction_tool_arguments
import hermes_lcm.externalize as externalize_module
from hermes_lcm.externalize import (
    build_transcript_gc_placeholder,
    externalize_ingest_payload,
    extract_externalized_ref,
    extract_externalized_refs,
    reassign_externalized_payloads,
)
from hermes_lcm.ingest_protection import (
    extract_all_externalized_payload_refs,
    extract_ingest_externalized_refs,
    redact_sensitive_text,
    scan_externalized_payload_integrity,
)
from hermes_lcm.tokens import count_messages_tokens


DATA_PAYLOAD = base64.b64encode(("LCM payload boundary repro ".encode("ascii")) * 900).decode("ascii")
DATA_URI = "data:image/png;base64," + DATA_PAYLOAD
GENERIC_BASE64 = DATA_PAYLOAD * 2
GENERIC_BASE64URL = base64.urlsafe_b64encode(bytes(range(256)) * 40).decode("ascii")


def _engine(tmp_path: Path) -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "payload-session",
        platform="telegram",
        conversation_id="payload-conversation",
        context_length=200_000,
    )
    return engine


def _sensitive_engine(tmp_path: Path, **overrides) -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    setattr(config, "sensitive_patterns_enabled", True)
    setattr(config, "sensitive_patterns", ["api_key", "bearer_token", "password_assignment", "private_key"])
    for key, value in overrides.items():
        setattr(config, key, value)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "payload-session",
        platform="telegram",
        conversation_id="payload-conversation",
        context_length=200_000,
    )
    return engine


def _single_message_row(engine: LCMEngine, *, role: str | None = None):
    where = "WHERE session_id = ?"
    args = [engine.current_session_id]
    if role:
        where += " AND role = ?"
        args.append(role)
    return engine._store._conn.execute(
        f"SELECT store_id, content, tool_calls FROM messages {where} ORDER BY store_id DESC LIMIT 1",
        args,
    ).fetchone()


def _extract_ref(text: str) -> str:
    match = re.search(r";\s*ref=([^;\]\s]+)", text)
    assert match, text
    return match.group(1)


def _extract_refs(text: str) -> list[str]:
    refs = re.findall(r";\s*ref=([^;\]\s]+)", text)
    assert refs, text
    return refs


def _expand_ref(engine: LCMEngine, ref: str) -> dict:
    return json.loads(lcm_tools.lcm_expand({"externalized_ref": ref, "max_tokens": 100_000}, engine=engine))


def _externalized_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "externalized").glob("*.json"))


def test_engine_ingest_does_not_reprotect_messages_in_store(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    def fail_if_store_protects_again(*_args, **_kwargs):
        raise AssertionError("engine ingest already protected this batch")

    monkeypatch.setattr(lcm_store_module, "protect_messages_for_ingest", fail_if_store_protects_again)

    engine._ingest_messages([{"role": "user", "content": "hello"}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert content == "hello"


def test_store_append_batch_still_protects_direct_callers(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    calls = []

    def mark_protected(messages, **_kwargs):
        calls.append(len(messages))
        return [dict(message, content="protected by store") for message in messages]

    monkeypatch.setattr(lcm_store_module, "protect_messages_for_ingest", mark_protected)

    ids = engine._store.append_batch("direct-session", [{"role": "user", "content": "raw"}], [1])

    assert calls == [1]
    stored = engine._store.get(ids[0])
    assert stored["content"] == "protected by store"


def test_sensitive_patterns_disabled_by_default_preserves_lossless_raw_text(tmp_path):
    engine = _engine(tmp_path)
    secret = "sk-defaultpreserved1234567890abcdef"

    engine._ingest_messages([{"role": "user", "content": f"api_key={secret}"}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert secret in content


def test_sensitive_redaction_continues_after_existing_placeholder(tmp_path):
    engine = _sensitive_engine(tmp_path)
    first_secret = "sk-firstsecret1234567890abcdef"
    second_secret = "sk-secondsecret1234567890abcdef"
    partially_redacted = redact_sensitive_text(f"api_key={first_secret}", engine._config)
    assert first_secret not in partially_redacted

    redacted = redact_sensitive_text(
        f"prefix {partially_redacted} and api_key={second_secret}",
        engine._config,
    )

    assert first_secret not in redacted
    assert second_secret not in redacted
    assert redacted.count("[LCM sensitive redaction:") == 2


def test_sensitive_patterns_redact_user_assistant_tool_and_tool_calls_before_sqlite_write(tmp_path):
    engine = _sensitive_engine(tmp_path)
    user_secret = "sk-usersecret1234567890abcdef"
    assistant_secret = "Bearer assistantTOKEN1234567890abcdef"
    tool_secret = "password=tool-secret-value"
    tool_call_secret = "sk-toolcall1234567890abcdef"

    engine._ingest_messages([
        {"role": "user", "content": f"please use api_key={user_secret} for the request"},
        {
            "role": "assistant",
            "content": f"I would call it with {assistant_secret}",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "deploy",
                        "arguments": json.dumps({"api_key": tool_call_secret}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": tool_secret},
    ])

    rows = engine._store._conn.execute(
        "SELECT content, COALESCE(tool_calls, '') FROM messages ORDER BY store_id"
    ).fetchall()
    stored_text = "\n".join("\n".join(row) for row in rows)
    for raw in (user_secret, assistant_secret, "tool-secret-value", tool_call_secret):
        assert raw not in stored_text
        assert engine._store.search(raw, session_id=engine.current_session_id) == []
    assert stored_text.count("[LCM sensitive redaction:") >= 4
    assert "please use api_key=" in stored_text


def test_sensitive_patterns_redact_before_large_payload_externalization(tmp_path):
    engine = _sensitive_engine(
        tmp_path,
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=40,
    )
    secret = "sk-externalized1234567890abcdef"
    content = f"api_key={secret}\n" + ("large payload line\n" * 20)

    engine._ingest_messages([{"role": "user", "content": content}])

    _store_id, stored_content, _tool_calls = _single_message_row(engine, role="user")
    assert secret not in stored_content
    ref = _extract_ref(stored_content)
    expanded = _expand_ref(engine, ref)
    assert secret not in expanded["content"]
    assert "[LCM sensitive redaction:" in expanded["content"]


def test_sensitive_patterns_redact_before_summary_serialization(tmp_path, monkeypatch):
    engine = _sensitive_engine(tmp_path, fresh_tail_count=1, leaf_chunk_tokens=1)
    secret = "sk-summary1234567890abcdef"
    captured = {}

    def fake_summarize(**kwargs):
        captured["text"] = kwargs["text"]
        assert secret not in kwargs["text"]
        return "summary without raw credential", 1

    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", fake_summarize)

    engine.compress(
        [
            {"role": "user", "content": f"api_key={secret} should be hidden"},
            {"role": "user", "content": "fresh tail"},
        ],
        current_tokens=10_000,
    )

    assert captured["text"]
    assert "[LCM sensitive redaction:" in captured["text"]
    node = engine._dag.get_session_nodes(engine.current_session_id)[0]
    assert secret not in node.summary


def test_sensitive_patterns_visible_in_status_and_doctor(tmp_path):
    engine = _sensitive_engine(tmp_path)

    status = json.loads(lcm_tools.lcm_status({}, engine=engine))
    assert status["ingest_protection"]["sensitive_patterns_enabled"] is True
    assert "api_key" in status["ingest_protection"]["sensitive_patterns"]

    doctor = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
    protection = next(check for check in doctor["checks"] if check["check"] == "sensitive_pattern_handling")
    assert protection["status"] == "pass"
    assert protection["detail"]["enabled"] is True
    assert "api_key" in protection["detail"]["patterns"]


def test_sensitive_patterns_redact_bypassed_active_replay_without_storage(tmp_path):
    secret = "sk-bypasssecret1234567890abcdef"
    tool_secret = "sk-bypasstoolsecret1234567890abcdef"
    messages = [
        {"role": "user", "content": f"api_key={secret}"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": json.dumps({"api_key": tool_secret}),
                    },
                }
            ],
        },
    ]

    config = LCMConfig(
        database_path=str(tmp_path / "no-session.db"),
        large_output_externalization_path=str(tmp_path / "no-session-externalized"),
        sensitive_patterns_enabled=True,
    )
    no_session_engine = LCMEngine(config=config, hermes_home=str(tmp_path / "no-session-home"))
    no_session_active = no_session_engine._ingest_messages(deepcopy(messages))

    config = LCMConfig(
        database_path=str(tmp_path / "ignored.db"),
        large_output_externalization_path=str(tmp_path / "ignored-externalized"),
        sensitive_patterns_enabled=True,
        ignore_session_patterns=["cron:*"],
    )
    ignored_engine = LCMEngine(config=config, hermes_home=str(tmp_path / "ignored-home"))
    ignored_engine.on_session_start(
        "nightly",
        platform="cron",
        conversation_id="ignored-conversation",
        context_length=200_000,
    )
    ignored_active = ignored_engine.compress(deepcopy(messages), current_tokens=0)

    config = LCMConfig(
        database_path=str(tmp_path / "stateless.db"),
        large_output_externalization_path=str(tmp_path / "stateless-externalized"),
        sensitive_patterns_enabled=True,
        stateless_session_patterns=["debug:*"],
    )
    stateless_engine = LCMEngine(config=config, hermes_home=str(tmp_path / "stateless-home"))
    stateless_engine.on_session_start(
        "scratch",
        platform="debug",
        conversation_id="stateless-conversation",
        context_length=200_000,
    )
    stateless_active = stateless_engine.compress(deepcopy(messages), current_tokens=0)

    for active in (no_session_active, ignored_active, stateless_active):
        active_text = json.dumps(active, sort_keys=True)
        assert secret not in active_text
        assert tool_secret not in active_text
        assert active_text.count("[LCM sensitive redaction:") == 2
        assert "content" not in active[1]

    assert ignored_engine._store.get_session_messages(ignored_engine.current_session_id) == []
    assert stateless_engine._store.get_session_messages(stateless_engine.current_session_id) == []


def test_sensitive_patterns_cover_client_secret_duplicate_json_and_quoted_passwords(tmp_path):
    engine = _sensitive_engine(tmp_path)
    client_secret = "oauthsupersecret1234567890"
    first_json_secret = "oldoauthclientsecret1234567890"
    second_json_secret = "newoauthclientsecret1234567890"
    escaped_first_json_secret = "alphaescapedclientsecret1234567890"
    escaped_second_json_secret = "betaescapedclientsecret1234567890"
    password_phrase = "correct horse battery staple"
    duplicate_key_json = (
        f'{{"client_secret":"{first_json_secret}",'
        f'"client_secret":"{second_json_secret}"}}'
    )
    escaped_duplicate_key_json = (
        f'{{\\"client_secret\\":\\"{escaped_first_json_secret}\\",'
        f'\\"client_secret\\":\\"{escaped_second_json_secret}\\"}}'
    )

    engine._ingest_messages([
        {
            "role": "user",
            "content": f"client_secret={client_secret} password=\"{password_phrase}\"",
        },
        {
            "role": "assistant",
            "content": "prepared oauth call",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "oauth", "arguments": duplicate_key_json},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "oauth", "arguments": escaped_duplicate_key_json},
                },
            ],
        },
    ])

    rows = engine._store._conn.execute(
        "SELECT content, COALESCE(tool_calls, '') FROM messages ORDER BY store_id"
    ).fetchall()
    stored_text = "\n".join("\n".join(row) for row in rows)
    for raw in (
        client_secret,
        first_json_secret,
        second_json_secret,
        escaped_first_json_secret,
        escaped_second_json_secret,
        password_phrase,
        "horse battery staple",
    ):
        assert raw not in stored_text
        assert engine._store.search(raw, session_id=engine.current_session_id) == []
    assert "client_secret=" in stored_text
    assert 'password="[LCM sensitive redaction:' in stored_text
    assert "sha256=" not in stored_text.split('password="', 1)[1].split('"', 1)[0]


def test_extract_externalized_ref_recovers_tool_and_non_tool_placeholders():
    placeholders = [
        "[Externalized tool output: tool_call_id=call_1; chars=1200; bytes=1200; ref=tool.json]",
        "[GC'd externalized tool output: tool_call_id=call_1; chars=1200; ref=tool-gc.json]",
        "[Externalized payload: kind=raw_payload; role=assistant; chars=1200; bytes=1200; ref=raw.json]",
        "[GC'd externalized payload: kind=raw_payload; role=assistant; chars=1200; ref=raw-gc.json]",
    ]

    assert [extract_externalized_ref(value) for value in placeholders] == [
        "tool.json",
        "tool-gc.json",
        "raw.json",
        "raw-gc.json",
    ]


def test_extract_all_externalized_payload_refs_recovers_real_placeholders_only():
    text = "\n".join(
        [
            "[Externalized LCM ingest payload: kind=ingest_payload; field=content; chars=1; bytes=1; ref=ingest.json]",
            "[Externalized tool output: tool_call_id=call_1; chars=1200; bytes=1200; ref=tool.json]",
            "[GC'd externalized tool output: tool_call_id=call_1; chars=1200; ref=tool-gc.json]",
            "[Externalized payload: kind=raw_payload; role=assistant; chars=1200; bytes=1200; ref=raw.json]",
            "[GC'd externalized payload: kind=raw_payload; role=assistant; chars=1200; ref=raw-gc.json]",
            "docs mention ref=docs.json without the real placeholder prefix",
            "[Externalized payload example: ref=example.json]",
            "[Externalized payload: kind=raw_payload; role=assistant; chars=1; ref=nested/not-basename.json]",
            "[Externalized payload: kind=raw_payload; role=assistant; chars=1; ref=nested\\not-basename.json]",
            "[Externalized LCM ingest payload: kind=ingest_payload; field=content; chars=1; bytes=1; ref=../escape.json]",
            "[Externalized LCM ingest payload: kind=ingest_payload; field=content; chars=1; bytes=1; ref=..\\escape.json]",
            "[Externalized tool output: tool_call_id=call_1; chars=1200; bytes=1200; ref=tool.json]",
        ]
    )

    assert extract_externalized_refs(text) == ["tool.json", "tool-gc.json", "raw.json", "raw-gc.json"]
    assert extract_all_externalized_payload_refs(text) == [
        "ingest.json",
        "tool.json",
        "tool-gc.json",
        "raw.json",
        "raw-gc.json",
    ]


def test_transcript_gc_placeholder_sanitizes_non_tool_metadata_before_ref():
    placeholder = build_transcript_gc_placeholder(
        {
            "kind": "raw_payload; ref=kind-bogus]",
            "role": "user; ref=role-bogus]",
            "content_chars": 1200,
            "ref": "real-ref.json",
        }
    )

    assert "kind-bogus" in placeholder
    assert "role-bogus" in placeholder
    assert "; ref=kind-bogus]" not in placeholder
    assert "; ref=role-bogus]" not in placeholder
    assert extract_externalized_ref(placeholder) == "real-ref.json"


def test_ingest_payload_placeholder_sanitizes_custom_kind_metadata_before_ref(tmp_path):
    engine = _engine(tmp_path)
    result = externalize_ingest_payload(
        "payload content",
        role="user",
        session_id=engine.current_session_id,
        field_path="content",
        config=engine._config,
        hermes_home=str(tmp_path),
        kind="ingest_payload; ref=kind-bogus]",
    )

    assert result is not None
    placeholder = result["placeholder"]
    assert "kind-bogus" in placeholder
    assert "; ref=kind-bogus]" not in placeholder
    refs = extract_ingest_externalized_refs(placeholder)
    assert refs == [result["path"].name]
    assert result["path"].name.startswith("20")
    assert "ref=kind-bogus" not in result["path"].name


def test_externalized_payload_write_fsyncs_file_and_parent_directory(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    file_fsync_calls = []
    fsynced_dirs = []

    monkeypatch.setattr(externalize_module.os, "fsync", lambda fd: file_fsync_calls.append(fd))
    monkeypatch.setattr(externalize_module, "_fsync_directory", lambda path: fsynced_dirs.append(Path(path)))

    result = externalize_ingest_payload(
        "durable ingest payload" * 20,
        role="user",
        session_id=engine.current_session_id,
        field_path="content",
        config=engine._config,
        hermes_home=str(tmp_path),
    )

    assert result is not None
    assert result["path"].exists()
    assert file_fsync_calls
    assert result["path"].parent in fsynced_dirs


def test_first_externalized_payload_fsyncs_new_storage_directory_parent(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    fsynced_dirs = []

    monkeypatch.setattr(externalize_module, "_fsync_directory", lambda path: fsynced_dirs.append(Path(path)))

    result = externalize_ingest_payload(
        "first durable ingest payload" * 20,
        role="user",
        session_id=engine.current_session_id,
        field_path="content",
        config=engine._config,
        hermes_home=str(tmp_path),
    )

    assert result is not None
    assert tmp_path in fsynced_dirs
    assert tmp_path / "externalized" in fsynced_dirs


def test_externalized_payload_fsync_failure_keeps_ingest_payload_inline(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    def fail_fsync(_fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(externalize_module.os, "fsync", fail_fsync)

    result = externalize_ingest_payload(
        "payload should stay inline when durability fails" * 20,
        role="user",
        session_id=engine.current_session_id,
        field_path="content",
        config=engine._config,
        hermes_home=str(tmp_path),
    )

    assert result is None
    assert _externalized_files(tmp_path) == []


def test_ingest_keeps_original_content_when_externalized_payload_durability_fails(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    def fail_fsync(_fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(externalize_module.os, "fsync", fail_fsync)

    engine._ingest_messages([{"role": "user", "content": "see image " + DATA_URI}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert DATA_URI in content
    assert extract_ingest_externalized_refs(content) == []
    assert _externalized_files(tmp_path) == []


def test_externalized_payload_reassignment_fsyncs_replacement(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    result = externalize_ingest_payload(
        "payload moved across compression boundary" * 20,
        role="user",
        session_id="old-session",
        field_path="content",
        config=engine._config,
        hermes_home=str(tmp_path),
    )
    assert result is not None

    fsync_calls = []
    monkeypatch.setattr(externalize_module.os, "fsync", lambda fd: fsync_calls.append(fd))

    moved = reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=engine._config,
        hermes_home=str(tmp_path),
    )

    assert moved == 1
    assert len(fsync_calls) >= 3
    payload = json.loads(result["path"].read_text(encoding="utf-8"))
    assert payload["session_id"] == "new-session"


def test_ingest_externalizes_plain_data_uri_user_content_before_sqlite_write(tmp_path):
    engine = _engine(tmp_path)

    engine._ingest_messages([{"role": "user", "content": "see image " + DATA_URI}])

    store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert "data:image" not in content
    assert DATA_PAYLOAD[:80] not in content
    ref = _extract_ref(content)
    assert _externalized_files(tmp_path)
    assert engine._store.search("base64", session_id=engine.current_session_id) == []
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == DATA_URI
    assert expanded["kind"] == "ingest_payload"

    raw_message = json.loads(lcm_tools.lcm_expand({"store_id": store_id, "max_tokens": 100_000}, engine=engine))
    assert raw_message["externalized_ref"] == ref
    assert raw_message["externalized"]["kind"] == "ingest_payload"
    assert raw_message["externalized"]["field_path"] == "content"


def test_ingest_preserves_trailing_text_after_data_uri(tmp_path):
    engine = _engine(tmp_path)

    engine._ingest_messages([{"role": "user", "content": DATA_URI + " please analyze this"}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert "data:image" not in content
    assert content.endswith(" please analyze this")
    assert engine._store.search("analyze", session_id=engine.current_session_id)
    ref = _extract_ref(content)
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == DATA_URI


def test_ingest_externalizes_data_uri_without_media_type_before_sqlite_write(tmp_path):
    engine = _engine(tmp_path)
    bare_data_uri = "data:;base64," + DATA_PAYLOAD

    engine._ingest_messages([{"role": "user", "content": bare_data_uri}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert "data:;base64" not in content
    ref = _extract_ref(content)
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == bare_data_uri


def test_ingest_externalizes_medium_data_uri_with_hyphenated_parameter_value(tmp_path):
    engine = _engine(tmp_path)
    medium_payload = base64.b64encode(b"medium-data-uri-payload" * 12).decode("ascii")
    assert 256 <= len(medium_payload) < 4096
    data_uri = "data:image/png;charset=utf-8;base64," + medium_payload

    engine._ingest_messages([{"role": "user", "content": "image " + data_uri}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert "charset=utf-8" not in content
    assert medium_payload[:120] not in content
    assert engine._store.search("charset", session_id=engine.current_session_id) == []
    assert engine._store.search(medium_payload[:64], session_id=engine.current_session_id) == []
    ref = _extract_ref(content)
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == data_uri


def test_ingest_externalizes_structured_image_url_data_uri_before_sqlite_write(tmp_path):
    engine = _engine(tmp_path)
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this image"},
            {"type": "image_url", "image_url": {"url": DATA_URI}},
        ],
    }

    engine._ingest_messages([message])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert "look at this image" in content
    assert "data:image" not in content
    ref = _extract_ref(content)
    assert engine._store.search("look", session_id=engine.current_session_id)
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == DATA_URI


def test_ingest_externalizes_generic_long_base64_string(tmp_path):
    engine = _engine(tmp_path)

    engine._ingest_messages([{"role": "user", "content": GENERIC_BASE64}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert GENERIC_BASE64[:120] not in content
    ref = _extract_ref(content)
    assert len(content) < 300
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == GENERIC_BASE64


def test_ingest_externalizes_generic_long_base64url_string(tmp_path):
    engine = _engine(tmp_path)
    assert "-" in GENERIC_BASE64URL and "_" in GENERIC_BASE64URL
    assert "+" not in GENERIC_BASE64URL and "/" not in GENERIC_BASE64URL

    engine._ingest_messages([{"role": "user", "content": f"prefix {GENERIC_BASE64URL} suffix"}])

    store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert "prefix " in content
    assert " suffix" in content
    assert GENERIC_BASE64URL[:120] not in content
    ref = _extract_ref(content)
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == GENERIC_BASE64URL
    raw_message = json.loads(lcm_tools.lcm_expand({"store_id": store_id, "max_tokens": 100_000}, engine=engine))
    assert raw_message["externalized_ref"] == ref
    assert GENERIC_BASE64URL[:120] not in json.dumps(raw_message)


def test_ingest_externalizes_base64url_tool_call_arguments(tmp_path):
    engine = _engine(tmp_path)
    message = {
        "role": "assistant",
        "content": "calling upload",
        "tool_calls": [
            {
                "id": "call_upload_urlsafe",
                "type": "function",
                "function": {
                    "name": "upload_blob",
                    "arguments": json.dumps({"blob": GENERIC_BASE64URL}),
                },
            }
        ],
    }

    engine._ingest_messages([message])

    store_id, _content, tool_calls = _single_message_row(engine, role="assistant")
    assert GENERIC_BASE64URL[:120] not in tool_calls
    ref = _extract_ref(tool_calls)
    parsed_tool_calls = json.loads(tool_calls)
    parsed_args = json.loads(parsed_tool_calls[0]["function"]["arguments"])
    assert parsed_args["blob"].startswith("[Externalized LCM ingest payload:")
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == GENERIC_BASE64URL
    raw_message = json.loads(lcm_tools.lcm_expand({"store_id": store_id, "max_tokens": 100_000}, engine=engine))
    assert raw_message["externalized_refs"] == [ref]
    assert GENERIC_BASE64URL[:120] not in json.dumps(raw_message)


def test_ingest_externalizes_embedded_generic_long_base64_run(tmp_path):
    engine = _engine(tmp_path)

    engine._ingest_messages([{"role": "user", "content": f"prefix {GENERIC_BASE64} suffix"}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert GENERIC_BASE64[:120] not in content
    assert content.startswith("prefix ")
    assert content.endswith(" suffix")
    ref = _extract_ref(content)
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == GENERIC_BASE64


def test_ingest_externalizes_data_uri_and_generic_base64_in_same_text(tmp_path):
    engine = _engine(tmp_path)

    engine._ingest_messages([{"role": "user", "content": f"image {DATA_URI} blob {GENERIC_BASE64} done"}])

    store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert "data:image" not in content
    assert DATA_PAYLOAD[:120] not in content
    assert GENERIC_BASE64[:120] not in content
    assert content.startswith("image ")
    assert content.endswith(" done")
    refs = _extract_refs(content)
    assert len(refs) == 2
    expanded_payloads = [_expand_ref(engine, ref)["content"] for ref in refs]
    assert expanded_payloads == [DATA_URI, GENERIC_BASE64]
    raw_message = json.loads(lcm_tools.lcm_expand({"store_id": store_id, "max_tokens": 100_000}, engine=engine))
    assert raw_message["externalized_refs"] == refs


def test_ingest_leaves_repeated_non_base64_text_inline(tmp_path):
    engine = _engine(tmp_path)
    repeated_text = "A" * 8000

    engine._ingest_messages([{"role": "user", "content": repeated_text}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert content == repeated_text
    assert _externalized_files(tmp_path) == []


def test_ingest_does_not_mutate_input_message(tmp_path):
    engine = _engine(tmp_path)
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "uploading"}],
        "tool_calls": [{"function": {"name": "upload", "arguments": json.dumps({"image": DATA_URI})}}],
    }
    original = deepcopy(message)

    engine._ingest_messages([message])

    assert message == original


def test_ingest_preserves_provider_active_context_while_protecting_storage(tmp_path):
    engine = _engine(tmp_path)
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "inspect this image"},
            {"type": "image_url", "image_url": {"url": DATA_URI}},
        ],
    }

    active = engine._ingest_messages([deepcopy(message)])

    assert active == [message]
    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert "data:image" not in content
    refs = extract_ingest_externalized_refs(content)
    assert len(refs) == 1
    assert _expand_ref(engine, refs[0])["content"] == DATA_URI


def test_tool_result_ingest_preserves_active_context_while_protecting_storage(tmp_path):
    engine = _engine(tmp_path)
    message = {"role": "tool", "tool_call_id": "call_media", "content": "tool saw " + DATA_URI}

    active = engine._ingest_messages([deepcopy(message)])

    assert active == [message]
    _store_id, content, _tool_calls = _single_message_row(engine, role="tool")
    assert "data:image" not in content
    ref = _extract_ref(content)
    assert _expand_ref(engine, ref)["content"] == DATA_URI


def test_preflight_storage_protection_does_not_force_noop_compaction(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
        context_threshold=0.0001,
        fresh_tail_count=64,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "preflight-storage-protection-session",
        platform="cli",
        conversation_id="preflight-storage-protection-conversation",
        context_length=200_000,
    )
    messages = [
        {"role": "system", "content": "system anchor"},
        {"role": "user", "content": "see image " + DATA_URI},
    ]

    assert engine.should_compress_preflight(deepcopy(messages)) is False
    assert engine.should_compress_preflight(deepcopy(messages)) is False
    assert engine._last_compression_status == "noop"
    assert engine._last_compression_noop_reason == "no eligible raw backlog outside fresh tail"
    assert engine._store.count_session_load_messages(engine.current_session_id) == 2
    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert "data:image" not in content


def test_ingest_protection_is_idempotent_for_existing_placeholder(tmp_path):
    engine = _engine(tmp_path)
    engine._store.append(engine.current_session_id, {"role": "user", "content": DATA_URI})
    _store_id, placeholder, _tool_calls = _single_message_row(engine, role="user")
    files_before = _externalized_files(tmp_path)

    engine._store.append(engine.current_session_id, {"role": "user", "content": placeholder})

    _store_id, second_content, _tool_calls = _single_message_row(engine, role="user")
    assert second_content == placeholder
    assert _externalized_files(tmp_path) == files_before


def test_engine_ingest_does_not_double_externalize_existing_externalized_payload_placeholder(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=50,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "double-protection-session",
        platform="telegram",
        conversation_id="double-protection-conversation",
        context_length=200_000,
    )
    original = "see image " + DATA_URI

    engine._ingest_messages([{"role": "user", "content": original}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert content.startswith("[Externalized payload: kind=media_payload;")
    assert len(_externalized_files(tmp_path)) == 1
    expanded = _expand_ref(engine, _extract_ref(content))
    assert expanded["content"] == original


def test_ingest_keeps_scanning_after_existing_placeholder_prefix(tmp_path):
    engine = _engine(tmp_path)
    engine._store.append(engine.current_session_id, {"role": "user", "content": DATA_URI})
    _store_id, placeholder, _tool_calls = _single_message_row(engine, role="user")
    first_ref = _extract_ref(placeholder)

    engine._store.append(engine.current_session_id, {"role": "user", "content": f"{placeholder} plus {DATA_URI}"})

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert "data:image" not in content
    assert content.startswith(placeholder)
    assert " plus " in content
    refs = _extract_refs(content)
    assert len(refs) == 2
    assert refs[0] == first_ref
    assert refs[1] != first_ref
    assert _expand_ref(engine, refs[0])["content"] == DATA_URI
    assert _expand_ref(engine, refs[1])["content"] == DATA_URI


def test_replayed_original_payload_reconciles_with_placeholder_row(tmp_path):
    engine = _engine(tmp_path)
    original_messages = [
        {"role": "system", "content": "system anchor"},
        {"role": "user", "content": "see image " + DATA_URI},
    ]
    engine._ingest_messages(original_messages)
    assert engine._store.count_session_load_messages(engine.current_session_id) == 2
    engine._store.close()

    engine = _engine(tmp_path)
    engine._ingest_messages(original_messages)

    assert engine._store.count_session_load_messages(engine.current_session_id) == 2


def test_replayed_original_tool_call_payload_reconciles_with_placeholder_row(tmp_path):
    engine = _engine(tmp_path)
    original_messages = [
        {"role": "system", "content": "system anchor"},
        {
            "role": "assistant",
            "content": "calling upload",
            "tool_calls": [
                {
                    "id": "call_upload",
                    "type": "function",
                    "function": {
                        "name": "upload_image",
                        "arguments": json.dumps({"image": DATA_URI}, indent=2),
                    },
                }
            ],
        },
    ]
    engine._ingest_messages(original_messages)
    assert engine._store.count_session_load_messages(engine.current_session_id) == 2
    engine._store.close()

    engine = _engine(tmp_path)
    engine._ingest_messages(original_messages)

    assert engine._store.count_session_load_messages(engine.current_session_id) == 2


def test_restart_replay_matches_escaped_structured_content_payload(tmp_path):
    medium_payload = base64.b64encode(b"medium-data-uri-payload" * 12).decode("ascii")
    assert 256 <= len(medium_payload) < 4096
    escaped_data_uri = f"data:image\\/png;base64,{medium_payload}"
    original_messages = [
        {"role": "system", "content": "system anchor"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "inspect this image"},
                {"type": "image_url", "image_url": {"url": escaped_data_uri}},
            ],
        },
    ]

    engine = _engine(tmp_path)
    engine._ingest_messages(original_messages)
    assert engine._store.count_session_load_messages(engine.current_session_id) == 2
    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert medium_payload[:120] not in content
    refs = extract_ingest_externalized_refs(content)
    assert len(refs) == 1
    assert _expand_ref(engine, refs[0])["content"] == escaped_data_uri
    engine._store.close()

    replay_engine = _engine(tmp_path)
    replay_engine._ingest_messages(original_messages)

    assert replay_engine._store.count_session_load_messages(replay_engine.current_session_id) == 2


def test_restart_replay_matches_plain_json_string_content_payload(tmp_path):
    medium_payload = base64.b64encode(b"plain-json-data-uri-payload" * 12).decode("ascii")
    assert 256 <= len(medium_payload) < 4096
    escaped_data_uri = f"data:image\\/png;base64,{medium_payload}"
    original_messages = [
        {"role": "system", "content": "system anchor"},
        {"role": "user", "content": f'{{"url": "{escaped_data_uri}"}}'},
    ]

    engine = _engine(tmp_path)
    engine._ingest_messages(original_messages)
    assert engine._store.count_session_load_messages(engine.current_session_id) == 2
    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert medium_payload[:120] not in content
    refs = extract_ingest_externalized_refs(content)
    assert len(refs) == 1
    assert _expand_ref(engine, refs[0])["content"] == escaped_data_uri
    engine._store.close()

    replay_engine = _engine(tmp_path)
    replay_engine._ingest_messages(original_messages)

    assert replay_engine._store.count_session_load_messages(replay_engine.current_session_id) == 2


def test_restart_replay_does_not_skip_changed_structured_content_payload(tmp_path):
    payload_a = base64.b64encode(b"structured-content-a" * 16).decode("ascii")
    payload_b = base64.b64encode(b"structured-content-b" * 16).decode("ascii")
    assert 256 <= len(payload_a) < 4096
    assert 256 <= len(payload_b) < 4096

    def messages(payload: str) -> list[dict]:
        escaped_data_uri = f"data:image\\/png;base64,{payload}"
        return [
            {"role": "system", "content": "system anchor"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect this image"},
                    {"type": "image_url", "image_url": {"url": escaped_data_uri}},
                ],
            },
        ]

    engine = _engine(tmp_path)
    engine._ingest_messages(messages(payload_a))
    assert engine._store.count_session_load_messages(engine.current_session_id) == 2
    engine._store.close()

    replay_engine = _engine(tmp_path)
    replay_engine._ingest_messages(messages(payload_b))

    assert replay_engine._store.count_session_load_messages(replay_engine.current_session_id) == 4
    _store_id, content, _tool_calls = _single_message_row(replay_engine, role="user")
    assert payload_b[:120] not in content
    refs = extract_ingest_externalized_refs(content)
    assert len(refs) == 1
    assert _expand_ref(replay_engine, refs[0])["content"] == f"data:image\\/png;base64,{payload_b}"


def test_tool_call_replay_identity_preserves_duplicate_key_argument_payload_text(tmp_path):
    engine = _engine(tmp_path)
    payload_a = "data:image/png;base64," + base64.b64encode(b"payload-a" * 48).decode("ascii")
    payload_b = "data:image/png;base64," + base64.b64encode(b"payload-b" * 48).decode("ascii")
    arguments_a = json.dumps({"image": payload_a}, separators=(",", ":"))[:-1] + ',"image":"plain"}'
    arguments_b = json.dumps({"image": payload_b}, separators=(",", ":"))[:-1] + ',"image":"plain"}'

    identity_a = engine._message_replay_identity(
        {
            "role": "assistant",
            "content": "calling upload",
            "tool_calls": [{"id": "call_upload", "function": {"name": "upload_image", "arguments": arguments_a}}],
        }
    )
    identity_b = engine._message_replay_identity(
        {
            "role": "assistant",
            "content": "calling upload",
            "tool_calls": [{"id": "call_upload", "function": {"name": "upload_image", "arguments": arguments_b}}],
        }
    )

    assert identity_a != identity_b
    assert payload_a in identity_a[3]
    assert payload_b in identity_b[3]


def test_restart_replay_does_not_skip_changed_duplicate_key_tool_argument_payload(tmp_path):
    payload_a = "data:image/png;base64," + base64.b64encode(b"payload-a" * 48).decode("ascii")
    payload_b = "data:image/png;base64," + base64.b64encode(b"payload-b" * 48).decode("ascii")
    arguments_a = json.dumps({"image": payload_a}, separators=(",", ":"))[:-1] + ',"image":"plain"}'
    arguments_b = json.dumps({"image": payload_b}, separators=(",", ":"))[:-1] + ',"image":"plain"}'

    def messages(arguments: str) -> list[dict]:
        return [
            {"role": "system", "content": "system anchor"},
            {
                "role": "assistant",
                "content": "calling upload",
                "tool_calls": [{"id": "call_upload", "function": {"name": "upload_image", "arguments": arguments}}],
            },
        ]

    engine = _engine(tmp_path)
    engine._ingest_messages(messages(arguments_a))
    assert engine._store.count_session_load_messages(engine.current_session_id) == 2
    engine._store.close()

    engine = _engine(tmp_path)
    engine._ingest_messages(messages(arguments_b))

    assert engine._store.count_session_load_messages(engine.current_session_id) == 4
    _store_id, _content, tool_calls = _single_message_row(engine, role="assistant")
    assert payload_b not in tool_calls
    refs = extract_ingest_externalized_refs(tool_calls)
    assert refs
    assert _expand_ref(engine, refs[0])["content"] == payload_b


def test_ingest_preserves_inline_payload_when_externalization_fails(tmp_path, monkeypatch):
    from hermes_lcm import ingest_protection

    engine = _engine(tmp_path)
    monkeypatch.setattr(ingest_protection, "externalize_ingest_payload", lambda *args, **kwargs: None)

    engine._store.append(engine.current_session_id, {"role": "user", "content": DATA_URI})

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert content == DATA_URI
    assert _externalized_files(tmp_path) == []


def test_externalized_payload_files_are_private(tmp_path):
    engine = _engine(tmp_path)

    engine._store.append(engine.current_session_id, {"role": "user", "content": DATA_URI})

    storage_dir = tmp_path / "externalized"
    payload_file = _externalized_files(tmp_path)[0]
    assert stat.S_IMODE(storage_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(payload_file.stat().st_mode) == 0o600


def test_ingest_leaves_normal_media_reference_inline(tmp_path):
    engine = _engine(tmp_path)
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "normal media ref"},
            {"type": "image_url", "image_url": {"url": "file:///tmp/example.png"}},
        ],
    }

    engine._ingest_messages([message])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert "file:///tmp/example.png" in content
    assert "ref=" not in content
    assert _externalized_files(tmp_path) == []


def test_ingest_externalizes_tool_result_data_uri_before_sqlite_write(tmp_path):
    engine = _engine(tmp_path)

    engine._ingest_messages([{"role": "tool", "tool_call_id": "call_media", "content": "tool saw " + DATA_URI}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="tool")
    assert "data:image" not in content
    ref = _extract_ref(content)
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == DATA_URI
    assert expanded["role"] == "tool"


def test_ingest_externalizes_tool_calls_function_arguments(tmp_path):
    engine = _engine(tmp_path)
    message = {
        "role": "assistant",
        "content": "calling upload",
        "tool_calls": [
            {
                "id": "call_upload",
                "type": "function",
                "function": {
                    "name": "upload_image",
                    "arguments": json.dumps({"image": DATA_URI}),
                },
            }
        ],
    }

    engine._ingest_messages([message])

    store_id, _content, tool_calls = _single_message_row(engine, role="assistant")
    assert "data:image" not in tool_calls
    ref = _extract_ref(tool_calls)
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == DATA_URI
    parsed_tool_calls = json.loads(tool_calls)
    parsed_args = json.loads(parsed_tool_calls[0]["function"]["arguments"])
    assert parsed_args["image"].startswith("[Externalized LCM ingest payload:")
    raw_message = json.loads(lcm_tools.lcm_expand({"store_id": store_id, "max_tokens": 100_000}, engine=engine))
    assert raw_message["externalized_refs"] == [ref]
    assert raw_message["externalized_payloads"][0]["field_path"] == "tool_calls[0].function.arguments"
    assert "tool_calls" not in raw_message


def test_ingest_preserves_json_argument_scaffold_when_externalizing_payload(tmp_path):
    engine = _engine(tmp_path)
    original_arguments = json.dumps({"image": DATA_URI, "caption": "keep formatting"}, indent=2)
    assert "\n  \"image\": " in original_arguments
    message = {
        "role": "assistant",
        "content": "calling formatted upload",
        "tool_calls": [
            {
                "id": "call_formatted_upload",
                "type": "function",
                "function": {
                    "name": "upload_image",
                    "arguments": original_arguments,
                },
            }
        ],
    }

    engine._ingest_messages([message])

    _store_id, _content, tool_calls = _single_message_row(engine, role="assistant")
    parsed_tool_calls = json.loads(tool_calls)
    protected_arguments = parsed_tool_calls[0]["function"]["arguments"]
    assert "data:image" not in protected_arguments
    assert DATA_PAYLOAD[:80] not in protected_arguments
    assert protected_arguments.startswith("{\n  \"image\": \"")
    assert protected_arguments.endswith('\",\n  \"caption\": \"keep formatting\"\n}')
    assert ",\n  \"caption\": " in protected_arguments
    assert json.loads(protected_arguments)["caption"] == "keep formatting"
    refs = extract_ingest_externalized_refs(protected_arguments)
    assert len(refs) == 1
    assert _expand_ref(engine, refs[0])["content"] == DATA_URI


def test_ingest_externalizes_tool_call_argument_payload_keys(tmp_path):
    engine = _engine(tmp_path)
    message = {
        "role": "assistant",
        "content": "calling keyed upload",
        "tool_calls": [
            {
                "id": "call_keyed_upload",
                "type": "function",
                "function": {
                    "name": "upload_image",
                    "arguments": json.dumps({DATA_URI: "plain-value"}),
                },
            }
        ],
    }

    engine._ingest_messages([message])

    store_id, _content, tool_calls = _single_message_row(engine, role="assistant")
    assert "data:image" not in tool_calls
    assert DATA_PAYLOAD[:80] not in tool_calls
    ref = _extract_ref(tool_calls)
    parsed_tool_calls = json.loads(tool_calls)
    parsed_args = json.loads(parsed_tool_calls[0]["function"]["arguments"])
    protected_key = next(iter(parsed_args))
    assert protected_key.startswith("[Externalized LCM ingest payload:")
    assert parsed_args[protected_key] == "plain-value"
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == DATA_URI
    raw_message = json.loads(lcm_tools.lcm_expand({"store_id": store_id, "max_tokens": 100_000}, engine=engine))
    assert raw_message["externalized_refs"] == [ref]
    assert DATA_PAYLOAD[:80] not in json.dumps(raw_message)


def test_ingest_externalizes_structured_content_payload_keys(tmp_path):
    engine = _engine(tmp_path)
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "keep this searchable"},
            {DATA_URI: "plain-value"},
        ],
    }

    engine._ingest_messages([message])

    store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert "keep this searchable" in content
    assert "data:image" not in content
    assert DATA_PAYLOAD[:80] not in content
    ref = _extract_ref(content)
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == DATA_URI
    raw_message = json.loads(lcm_tools.lcm_expand({"store_id": store_id, "max_tokens": 100_000}, engine=engine))
    assert raw_message["externalized_refs"] == [ref]
    assert DATA_PAYLOAD[:80] not in json.dumps(raw_message)


def test_pre_compaction_tool_arguments_sanitize_payload_keys():
    sanitized = sanitize_pre_compaction_tool_arguments({DATA_URI: "plain-value"})

    assert "data:image" not in sanitized
    assert DATA_PAYLOAD[:80] not in sanitized
    parsed = json.loads(sanitized)
    assert parsed == {"[Media attachment]": "plain-value"}


def test_payload_bearing_key_uses_neutral_child_field_path(tmp_path):
    engine = _engine(tmp_path)
    mixed_key = f"raw-label {DATA_URI} raw-suffix"
    message = {
        "role": "assistant",
        "content": "calling keyed upload",
        "tool_calls": [
            {
                "id": "call_keyed_upload",
                "type": "function",
                "function": {
                    "name": "upload_image",
                    "arguments": json.dumps({mixed_key: DATA_URI}),
                },
            }
        ],
    }

    engine._ingest_messages([message])

    _store_id, _content, tool_calls = _single_message_row(engine, role="assistant")
    assert "data:image" not in tool_calls
    assert DATA_PAYLOAD[:80] not in tool_calls
    payloads = [json.loads(path.read_text()) for path in _externalized_files(tmp_path)]
    field_paths = [payload["field_path"] for payload in payloads]
    assert len(field_paths) == 2
    assert all("data:image" not in field_path for field_path in field_paths)
    assert all(DATA_PAYLOAD[:80] not in field_path for field_path in field_paths)
    assert all("raw-label" not in field_path for field_path in field_paths)
    assert all("raw-suffix" not in field_path for field_path in field_paths)
    assert "tool_calls[0].function.arguments" in field_paths


def test_ingest_preserves_json_argument_string_when_no_payload_changes(tmp_path):
    engine = _engine(tmp_path)
    original_arguments = '{"b": 1, "a": 2, "b": 3}'
    message = {
        "role": "assistant",
        "content": "calling ordinary function",
        "tool_calls": [
            {
                "id": "call_plain",
                "type": "function",
                "function": {
                    "name": "ordinary_function",
                    "arguments": original_arguments,
                },
            }
        ],
    }

    engine._ingest_messages([message])

    _store_id, _content, tool_calls = _single_message_row(engine, role="assistant")
    parsed_tool_calls = json.loads(tool_calls)
    assert parsed_tool_calls[0]["function"]["arguments"] == original_arguments
    assert "ref=" not in tool_calls
    assert _externalized_files(tmp_path) == []


def test_ingest_externalizes_duplicate_key_json_argument_payload_without_collapsing_string(tmp_path):
    engine = _engine(tmp_path)
    original_arguments = json.dumps({"image": DATA_URI}, separators=(",", ":"))[:-1] + ',"image":"plain"}'
    message = {
        "role": "assistant",
        "content": "calling duplicate-key function",
        "tool_calls": [
            {
                "id": "call_duplicate",
                "type": "function",
                "function": {
                    "name": "duplicate_key_function",
                    "arguments": original_arguments,
                },
            }
        ],
    }

    engine._ingest_messages([message])

    _store_id, _content, tool_calls = _single_message_row(engine, role="assistant")
    parsed_tool_calls = json.loads(tool_calls)
    protected_arguments = parsed_tool_calls[0]["function"]["arguments"]
    assert "data:image" not in protected_arguments
    assert ',"image":"plain"}' in protected_arguments
    refs = extract_ingest_externalized_refs(protected_arguments)
    assert len(refs) == 1
    expanded = _expand_ref(engine, refs[0])
    assert expanded["content"] == DATA_URI
    assert expanded["field_path"] == "tool_calls[0].function.arguments"


def test_ingest_externalizes_duplicate_key_json_argument_escaped_data_uri(tmp_path):
    engine = _engine(tmp_path)
    medium_payload = base64.b64encode(b"medium-data-uri-payload" * 12).decode("ascii")
    assert 256 <= len(medium_payload) < 4096
    escaped_data_uri = f"data:image\\/png;base64,{medium_payload}"
    original_arguments = f'{{"image":"{escaped_data_uri}","image":"plain"}}'
    message = {
        "role": "assistant",
        "content": "calling escaped duplicate-key function",
        "tool_calls": [
            {
                "id": "call_duplicate_escaped",
                "type": "function",
                "function": {
                    "name": "duplicate_key_function",
                    "arguments": original_arguments,
                },
            }
        ],
    }

    engine._ingest_messages([message])

    store_id, _content, tool_calls = _single_message_row(engine, role="assistant")
    parsed_tool_calls = json.loads(tool_calls)
    protected_arguments = parsed_tool_calls[0]["function"]["arguments"]
    assert "data:image" not in protected_arguments
    assert medium_payload[:120] not in protected_arguments
    assert ',"image":"plain"}' in protected_arguments
    refs = extract_ingest_externalized_refs(protected_arguments)
    assert len(refs) == 1
    expanded = _expand_ref(engine, refs[0])
    assert expanded["content"] == escaped_data_uri
    assert expanded["field_path"] == "tool_calls[0].function.arguments"
    raw_message = json.loads(lcm_tools.lcm_expand({"store_id": store_id, "max_tokens": 100_000}, engine=engine))
    assert raw_message["externalized_refs"] == refs
    assert medium_payload[:120] not in json.dumps(raw_message)


def test_restart_replay_matches_escaped_duplicate_key_tool_argument_payload(tmp_path):
    medium_payload = base64.b64encode(b"medium-data-uri-payload" * 12).decode("ascii")
    assert 256 <= len(medium_payload) < 4096
    escaped_data_uri = f"data:image\\/png;base64,{medium_payload}"
    original_arguments = f'{{"image":"{escaped_data_uri}","image":"plain"}}'
    original_messages = [
        {"role": "system", "content": "system anchor"},
        {
            "role": "assistant",
            "content": "calling escaped duplicate-key function",
            "tool_calls": [
                {
                    "id": "call_duplicate_escaped",
                    "type": "function",
                    "function": {
                        "name": "duplicate_key_function",
                        "arguments": original_arguments,
                    },
                }
            ],
        },
    ]

    engine = _engine(tmp_path)
    engine._ingest_messages(original_messages)
    assert engine._store.count_session_load_messages(engine.current_session_id) == 2
    engine._store.close()

    engine = _engine(tmp_path)
    engine._ingest_messages(original_messages)

    assert engine._store.count_session_load_messages(engine.current_session_id) == 2
    _store_id, _content, tool_calls = _single_message_row(engine, role="assistant")
    assert medium_payload[:120] not in tool_calls
    refs = extract_ingest_externalized_refs(tool_calls)
    assert len(refs) == 1
    assert _expand_ref(engine, refs[0])["content"] == escaped_data_uri


def test_ingest_externalizes_unicode_escaped_slash_duplicate_key_tool_argument(tmp_path):
    medium_payload = base64.b64encode(b"unicode-slash-data-uri-payload" * 10).decode("ascii")
    assert 256 <= len(medium_payload) < 4096

    for label, slash_escape in (("lower", "\\u002f"), ("upper", "\\u002F")):
        variant_path = tmp_path / label
        variant_path.mkdir()
        engine = _engine(variant_path)
        escaped_data_uri = "data:image" + slash_escape + "png;base64," + medium_payload
        original_arguments = f'{{"image":"{escaped_data_uri}","image":"plain"}}'

        engine._ingest_messages([
            {
                "role": "assistant",
                "content": f"calling {label} unicode escaped duplicate-key function",
                "tool_calls": [
                    {
                        "id": f"call_duplicate_unicode_escaped_{label}",
                        "type": "function",
                        "function": {
                            "name": "duplicate_key_function",
                            "arguments": original_arguments,
                        },
                    }
                ],
            }
        ])

        _store_id, _content, tool_calls = _single_message_row(engine, role="assistant")
        assert "data:image" not in tool_calls
        assert slash_escape + "png" not in tool_calls
        assert medium_payload[:120] not in tool_calls
        refs = extract_ingest_externalized_refs(tool_calls)
        assert len(refs) == 1
        assert _expand_ref(engine, refs[0])["content"] == escaped_data_uri


def test_ingest_ref_parser_ignores_ref_text_in_tool_argument_key(tmp_path):
    engine = _engine(tmp_path)
    tricky_key = "; ref=bogus]"
    message = {
        "role": "assistant",
        "content": "calling upload",
        "tool_calls": [
            {
                "id": "call_upload",
                "type": "function",
                "function": {
                    "name": "upload_image",
                    "arguments": json.dumps({tricky_key: DATA_URI}),
                },
            }
        ],
    }
    messages = [{"role": "system", "content": "stable replay anchor"}, message]

    engine._ingest_messages(messages)

    store_id, _content, tool_calls = _single_message_row(engine, role="assistant")
    assert "data:image" not in tool_calls
    assert tricky_key in tool_calls
    assert "field=tool_calls-0-.function.arguments;" in tool_calls
    refs = extract_ingest_externalized_refs(tool_calls)
    assert len(refs) == 1
    assert refs[0] != "bogus"
    expanded = _expand_ref(engine, refs[0])
    assert expanded["content"] == DATA_URI
    raw_message = json.loads(lcm_tools.lcm_expand({"store_id": store_id, "max_tokens": 100_000}, engine=engine))
    assert raw_message["externalized_refs"] == refs
    assert raw_message["externalized_payloads"][0]["field_path"] == "tool_calls[0].function.arguments"
    assert engine._store.count_session_load_messages(engine.current_session_id) == 2
    engine._store.close()

    replay_engine = _engine(tmp_path)
    replay_engine._ingest_messages(messages)

    assert replay_engine._store.count_session_load_messages(replay_engine.current_session_id) == 2


def test_ingest_externalizes_nested_json_string_tool_arguments(tmp_path):
    engine = _engine(tmp_path)
    nested_arguments = json.dumps({"outer": json.dumps({"inner": DATA_URI})})

    engine._ingest_messages([
        {
            "role": "assistant",
            "content": "calling nested upload",
            "tool_calls": [
                {
                    "id": "call_nested",
                    "type": "function",
                    "function": {"name": "upload_nested", "arguments": nested_arguments},
                }
            ],
        }
    ])

    _store_id, _content, tool_calls = _single_message_row(engine, role="assistant")
    assert "data:image" not in tool_calls
    ref = _extract_ref(tool_calls)
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == DATA_URI


def _load_importer_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "import_lossless_claw.py"
    spec = importlib.util.spec_from_file_location("import_lossless_claw_payload_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _create_lossless_source(db_path: Path):
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE conversations (
                conversation_id INTEGER PRIMARY KEY,
                session_id TEXT,
                session_key TEXT,
                created_at TEXT
            );
            CREATE TABLE messages (
                message_id INTEGER PRIMARY KEY,
                conversation_id INTEGER NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT
            );
            """
        )
        conn.execute("INSERT INTO conversations VALUES (1, 'legacy-session', 'legacy-key', '2026-01-01 00:00')")
        conn.execute(
            "INSERT INTO messages VALUES (1, 1, 1, 'user', ?, 123, '2026-01-01 00:01')",
            ("legacy " + DATA_URI,),
        )
        conn.commit()
    finally:
        conn.close()


def test_import_lossless_claw_externalizes_legacy_data_uri_content(tmp_path):
    importer = _load_importer_module()
    source_db = tmp_path / "source.db"
    target_db = tmp_path / "target.db"
    _create_lossless_source(source_db)

    first = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        namespace="openclaw-lcm",
        agent="repro",
        import_id="payload-import",
        apply=True,
    )
    second = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        namespace="openclaw-lcm",
        agent="repro",
        import_id="payload-import",
        apply=True,
    )

    assert first.imported == 1
    assert second.imported == 0
    conn = sqlite3.connect(target_db)
    try:
        content = conn.execute("SELECT content FROM messages").fetchone()[0]
    finally:
        conn.close()
    assert "data:image" not in content
    ref = _extract_ref(content)
    engine = LCMEngine(
        config=LCMConfig(database_path=str(target_db)),
        hermes_home=str(tmp_path),
    )
    engine.on_session_start("openclaw-lcm:agent:repro:legacy-session", platform="import", context_length=200_000)
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == DATA_URI


def test_import_lossless_claw_respects_externalization_path_env(tmp_path, monkeypatch):
    importer = _load_importer_module()
    source_db = tmp_path / "source.db"
    target_db = tmp_path / "target.db"
    custom_externalized = tmp_path / "custom-externalized"
    _create_lossless_source(source_db)
    monkeypatch.setenv("LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH", str(custom_externalized))

    result = importer.import_lossless_claw(
        source_db=source_db,
        target_db=target_db,
        namespace="openclaw-lcm",
        agent="repro",
        import_id="payload-import-custom-path",
        apply=True,
    )

    assert result.imported == 1
    conn = sqlite3.connect(target_db)
    try:
        content = conn.execute("SELECT content FROM messages").fetchone()[0]
    finally:
        conn.close()
    ref = _extract_ref(content)
    assert (custom_externalized / ref).exists()
    engine = LCMEngine(
        config=LCMConfig(database_path=str(target_db), large_output_externalization_path=str(custom_externalized)),
        hermes_home=str(tmp_path),
    )
    engine.on_session_start("openclaw-lcm:agent:repro:legacy-session", platform="import", context_length=200_000)
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == DATA_URI


def test_store_id_expand_never_returns_raw_historical_tool_calls(tmp_path):
    engine = _engine(tmp_path)
    tool_calls = json.dumps([{"function": {"arguments": DATA_URI}}])
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "historical row with raw tool args",
            None,
            tool_calls,
            None,
            1.0,
            5,
            0,
        ),
    )
    engine._store._conn.commit()
    store_id = engine._store._conn.execute("SELECT max(store_id) FROM messages").fetchone()[0]

    raw_message_text = lcm_tools.lcm_expand({"store_id": store_id, "max_tokens": 100_000}, engine=engine)
    raw_message = json.loads(raw_message_text)

    assert "tool_calls" not in raw_message
    assert DATA_URI not in raw_message_text
    assert DATA_PAYLOAD[:120] not in raw_message_text


def test_lcm_doctor_reports_largest_and_suspicious_payload_rows(tmp_path):
    engine = _engine(tmp_path)
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "small content",
            None,
            json.dumps([{"function": {"arguments": DATA_URI}}]),
            None,
            1.0,
            5,
            0,
        ),
    )
    engine._store._conn.commit()

    result = handle_lcm_command("doctor", engine)

    assert "largest_content_rows:" in result
    assert "largest_tool_calls_rows:" in result
    assert "suspicious_data_uri_content_rows:" in result
    assert "suspicious_data_uri_tool_calls_rows:" in result
    assert "suspicious_base64_like_rows:" in result


def test_lcm_doctor_ignores_literal_data_uri_like_scaffold(tmp_path):
    engine = _engine(tmp_path)
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "diagnostic pattern: data:%;base64,%",
            None,
            json.dumps([{"function": {"arguments": "LIKE pattern data:%;base64,%"}}]),
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    json_result = json.loads(lcm_tools.lcm_doctor({}, engine=engine))

    payload_check = next(check for check in json_result["checks"] if check["check"] == "payload_storage")
    assert payload_check["status"] == "pass"
    assert payload_check["detail"]["suspicious_data_uri_content_rows"] == []
    assert payload_check["detail"]["suspicious_data_uri_tool_calls_rows"] == []


def test_lcm_doctor_ignores_code_data_uri_prefix_without_payload(tmp_path):
    engine = _engine(tmp_path)
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "tool",
            'DATA_URI = "data:image/png;base64," + DATA_PAYLOAD',
            None,
            None,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    json_result = json.loads(lcm_tools.lcm_doctor({}, engine=engine))

    payload_check = next(check for check in json_result["checks"] if check["check"] == "payload_storage")
    assert payload_check["status"] == "pass"
    assert payload_check["detail"]["suspicious_data_uri_content_rows"] == []


def test_lcm_doctor_reports_embedded_generic_base64_without_raw_preview(tmp_path):
    engine = _engine(tmp_path)
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "small content",
            None,
            json.dumps([{"function": {"arguments": "prefix " + GENERIC_BASE64 + " suffix"}}]),
            None,
            1.0,
            5,
            0,
        ),
    )
    engine._store._conn.commit()

    json_result_text = lcm_tools.lcm_doctor({}, engine=engine)
    json_result = json.loads(json_result_text)

    assert GENERIC_BASE64[:120] not in json_result_text
    payload_check = next(check for check in json_result["checks"] if check["check"] == "payload_storage")
    rows = payload_check["detail"]["suspicious_base64_like_rows"]
    assert rows
    assert rows[0]["field"] == "tool_calls"
    assert rows[0]["suspicious_category"] == "base64_like"


def test_lcm_doctor_reports_externalized_payload_stats(tmp_path):
    engine = _engine(tmp_path)
    engine._ingest_messages([{"role": "user", "content": DATA_URI}])

    text_result = handle_lcm_command("doctor", engine)
    json_result = json.loads(lcm_tools.lcm_doctor({}, engine=engine))

    assert "externalized_payload_count: 1" in text_result
    assert "externalized_payload_bytes:" in text_result
    externalized_check = next(check for check in json_result["checks"] if check["check"] == "payload_storage")
    assert externalized_check["detail"]["externalized_payload_count"] == 1
    assert externalized_check["detail"]["externalized_payload_bytes"] > 0


def test_externalized_payload_integrity_scan_reports_missing_and_unreferenced_refs_without_payload_previews(tmp_path):
    engine = _engine(tmp_path)
    storage_dir = tmp_path / "externalized"
    storage_dir.mkdir()
    for ref, content in {
        "ingest-present.json": "present ingest payload",
        "legacy-present.json": "present legacy payload",
        "orphan.json": "UNREFERENCED_RAW_PAYLOAD",
    }.items():
        (storage_dir / ref).write_text(json.dumps({"content": content, "content_chars": len(content)}))

    content = "\n".join(
        [
            "[Externalized LCM ingest payload: kind=ingest_payload; field=content; chars=1; bytes=1; ref=ingest-present.json]",
            "[Externalized payload: kind=raw_payload; role=assistant; chars=1; bytes=1; ref=missing-raw.json]",
            "[Externalized payload: kind=raw_payload; role=assistant; chars=1; ref=nested/not-counted.json]",
            "docs mention ref=doc-only.json but not in a real placeholder",
        ]
    )
    tool_calls = json.dumps(
        [
            {
                "function": {
                    "arguments": "[GC'd externalized tool output: tool_call_id=call_1; chars=1; ref=legacy-present.json]"
                }
            }
        ]
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            content,
            None,
            tool_calls,
            None,
            1.0,
            5,
            0,
        ),
    )
    engine._store._conn.commit()
    before_rows = engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    before_files = sorted(path.name for path in storage_dir.glob("*.json"))

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == before_rows
    assert sorted(path.name for path in storage_dir.glob("*.json")) == before_files
    assert detail["externalized_payload_refs_total"] == 3
    assert detail["externalized_payload_refs_existing"] == 2
    assert detail["externalized_payload_refs_missing"] == 1
    assert detail["externalized_payload_files_unreferenced"] == 1
    assert detail["missing_externalized_payload_refs"] == [
        {
            "store_id": 1,
            "session_id": engine.current_session_id,
            "source": "telegram",
            "role": "assistant",
            "field": "content",
            "externalized_ref": "missing-raw.json",
        }
    ]
    assert detail["unreferenced_externalized_payload_files"] == [{"externalized_ref": "orphan.json"}]
    encoded = json.dumps(detail)
    assert "UNREFERENCED_RAW_PAYLOAD" not in encoded
    assert "doc-only.json" not in encoded
    assert "not-counted.json" not in encoded


def test_externalized_payload_integrity_scan_detects_embedded_content_placeholder_with_trailing_text(tmp_path):
    engine = _engine(tmp_path)
    (tmp_path / "externalized").mkdir()
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "user",
            "see image [Externalized LCM ingest payload: kind=media_payload; field=content; chars=1; bytes=1; ref=missing-embedded.json] please inspect",
            None,
            None,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 1
    assert detail["externalized_payload_refs_missing"] == 1
    assert detail["missing_externalized_payload_refs"] == [
        {
            "store_id": 1,
            "session_id": engine.current_session_id,
            "source": "telegram",
            "role": "user",
            "field": "content",
            "externalized_ref": "missing-embedded.json",
        }
    ]


def test_externalized_payload_integrity_scan_ignores_escaped_placeholder_examples_in_logs(tmp_path):
    engine = _engine(tmp_path)
    (tmp_path / "externalized").mkdir()
    escaped_output = (
        'pytest output: \\\\"[Externalized LCM ingest payload: kind=ingest_payload; '
        'field=content; chars=1; bytes=1; '
        'ref=example-log-ref.json]\\\\"'
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "tool",
            escaped_output,
            None,
            None,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 0
    assert detail["externalized_payload_refs_missing"] == 0
    assert detail["missing_externalized_payload_refs"] == []


def test_externalized_payload_integrity_scan_detects_nested_tool_call_argument_json_placeholder(tmp_path):
    engine = _engine(tmp_path)
    (tmp_path / "externalized").mkdir()
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=tool_calls; "
        "chars=1; bytes=1; ref=missing-tool-call-media.json]"
    )
    tool_calls = json.dumps(
        [
            {
                "function": {
                    "name": "analyze_image",
                    "arguments": json.dumps({"image": placeholder}),
                }
            }
        ]
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "calling tool",
            None,
            tool_calls,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 1
    assert detail["externalized_payload_refs_missing"] == 1
    assert detail["missing_externalized_payload_refs"] == [
        {
            "store_id": 1,
            "session_id": engine.current_session_id,
            "source": "telegram",
            "role": "assistant",
            "field": "tool_calls",
            "externalized_ref": "missing-tool-call-media.json",
        }
    ]


def test_externalized_payload_integrity_scan_detects_embedded_tool_call_metadata_placeholder(tmp_path):
    engine = _engine(tmp_path)
    storage_dir = tmp_path / "externalized"
    storage_dir.mkdir()
    (storage_dir / "present-tool-call-metadata-media.json").write_text(json.dumps({"content": "payload"}))
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=tool_calls; "
        "chars=1; bytes=1; ref=present-tool-call-metadata-media.json]"
    )
    tool_calls = json.dumps(
        [
            {
                "function": {
                    "name": "analyze_image",
                    "arguments": "{}",
                },
                "metadata": f"prefix {placeholder} suffix",
            }
        ]
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "calling tool",
            None,
            tool_calls,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 1
    assert detail["externalized_payload_refs_existing"] == 1
    assert detail["externalized_payload_refs_missing"] == 0
    assert detail["externalized_payload_files_unreferenced"] == 0


def test_externalized_payload_integrity_scan_counts_duplicate_provider_custom_field_placeholder(tmp_path):
    engine = _engine(tmp_path)
    storage_dir = tmp_path / "externalized"
    storage_dir.mkdir()
    (storage_dir / "present-provider-custom.json").write_text(json.dumps({"content": "payload"}))
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=tool_calls; "
        "chars=1; bytes=1; ref=present-provider-custom.json]"
    )
    tool_calls = (
        '[{"metadata":"'
        + placeholder
        + '","metadata":"fallback","function":{"arguments":"{}"}}]'
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "calling tool",
            None,
            tool_calls,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 1
    assert detail["externalized_payload_refs_existing"] == 1
    assert detail["externalized_payload_refs_missing"] == 0
    assert detail["externalized_payload_files_unreferenced"] == 0


def test_externalized_payload_integrity_scan_reports_missing_ref_in_malformed_tool_calls(tmp_path):
    engine = _engine(tmp_path)
    (tmp_path / "externalized").mkdir()
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=tool_calls; "
        "chars=1; bytes=1; ref=missing-malformed-tool-calls.json]"
    )
    tool_calls = '[{"function":{"arguments":"{}"},"metadata":"' + placeholder + '"'
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "calling tool",
            None,
            tool_calls,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 1
    assert detail["externalized_payload_refs_missing"] == 1
    assert detail["missing_externalized_payload_refs"] == [
        {
            "store_id": 1,
            "session_id": engine.current_session_id,
            "source": "telegram",
            "role": "assistant",
            "field": "tool_calls",
            "externalized_ref": "missing-malformed-tool-calls.json",
        }
    ]


def test_externalized_payload_integrity_scan_detects_embedded_tool_call_argument_placeholder_with_duplicate_keys(tmp_path):
    engine = _engine(tmp_path)
    storage_dir = tmp_path / "externalized"
    storage_dir.mkdir()
    (storage_dir / "present-tool-call-media.json").write_text(json.dumps({"content": "payload"}))
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=tool_calls; "
        "chars=1; bytes=1; ref=present-tool-call-media.json]"
    )
    duplicate_key_arguments = (
        '{"note":"said \\\"hi\\\"","image":"'
        + placeholder
        + '","image":"plain text fallback"}'
    )
    tool_calls = json.dumps(
        [
            {
                "function": {
                    "name": "analyze_image",
                    "arguments": duplicate_key_arguments,
                }
            }
        ]
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "calling tool",
            None,
            tool_calls,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 1
    assert detail["externalized_payload_refs_existing"] == 1
    assert detail["externalized_payload_refs_missing"] == 0
    assert detail["externalized_payload_files_unreferenced"] == 0


def test_externalized_payload_integrity_scan_detects_free_form_tool_call_argument_placeholder(tmp_path):
    engine = _engine(tmp_path)
    storage_dir = tmp_path / "externalized"
    storage_dir.mkdir()
    (storage_dir / "present-free-form-tool-call-media.json").write_text(json.dumps({"content": "payload"}))
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=tool_calls; "
        "chars=1; bytes=1; ref=present-free-form-tool-call-media.json]"
    )
    tool_calls = json.dumps(
        [
            {
                "function": {
                    "name": "analyze_image",
                    "arguments": f"prefix {placeholder} suffix",
                }
            }
        ]
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "calling tool",
            None,
            tool_calls,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 1
    assert detail["externalized_payload_refs_existing"] == 1
    assert detail["externalized_payload_refs_missing"] == 0
    assert detail["externalized_payload_files_unreferenced"] == 0


def test_externalized_payload_integrity_scan_detects_json_tool_call_argument_placeholder_after_caption_quotes(tmp_path):
    engine = _engine(tmp_path)
    storage_dir = tmp_path / "externalized"
    storage_dir.mkdir()
    (storage_dir / "present-caption-tool-call-media.json").write_text(json.dumps({"content": "payload"}))
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=tool_calls; "
        "chars=1; bytes=1; ref=present-caption-tool-call-media.json]"
    )
    arguments = json.dumps({"image": f'caption says "front" {placeholder}'})
    tool_calls = json.dumps(
        [
            {
                "function": {
                    "name": "analyze_image",
                    "arguments": arguments,
                }
            }
        ]
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "calling tool",
            None,
            tool_calls,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 1
    assert detail["externalized_payload_refs_existing"] == 1
    assert detail["externalized_payload_refs_missing"] == 0
    assert detail["externalized_payload_files_unreferenced"] == 0


def test_externalized_payload_integrity_scan_detects_json_tool_call_argument_placeholder_after_unmatched_caption_quote(tmp_path):
    engine = _engine(tmp_path)
    storage_dir = tmp_path / "externalized"
    storage_dir.mkdir()
    (storage_dir / "present-unmatched-caption-tool-call-media.json").write_text(json.dumps({"content": "payload"}))
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=tool_calls; "
        "chars=1; bytes=1; ref=present-unmatched-caption-tool-call-media.json]"
    )
    arguments = json.dumps({"image": f'caption says "front {placeholder}'})
    tool_calls = json.dumps(
        [
            {
                "function": {
                    "name": "analyze_image",
                    "arguments": arguments,
                }
            }
        ]
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "calling tool",
            None,
            tool_calls,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 1
    assert detail["externalized_payload_refs_existing"] == 1
    assert detail["externalized_payload_refs_missing"] == 0
    assert detail["externalized_payload_files_unreferenced"] == 0


def test_externalized_payload_integrity_scan_detects_parsed_object_tool_call_argument_placeholder(tmp_path):
    engine = _engine(tmp_path)
    storage_dir = tmp_path / "externalized"
    storage_dir.mkdir()
    (storage_dir / "present-object-tool-call-media.json").write_text(json.dumps({"content": "payload"}))
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=tool_calls; "
        "chars=1; bytes=1; ref=present-object-tool-call-media.json]"
    )
    tool_calls = json.dumps(
        [
            {
                "function": {
                    "name": "analyze_image",
                    "arguments": {"image": f"user's screenshot {placeholder} suffix"},
                }
            }
        ]
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "calling tool",
            None,
            tool_calls,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 1
    assert detail["externalized_payload_refs_existing"] == 1
    assert detail["externalized_payload_refs_missing"] == 0
    assert detail["externalized_payload_files_unreferenced"] == 0


def test_externalized_payload_integrity_scan_ignores_escaped_placeholder_examples_inside_tool_call_json(tmp_path):
    engine = _engine(tmp_path)
    (tmp_path / "externalized").mkdir()
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=tool_calls; "
        "chars=1; bytes=1; ref=example-missing.json]"
    )
    arguments = json.dumps({"log": f'pytest output: "prefix before placeholder {placeholder}"'})
    tool_calls = json.dumps(
        [
            {
                "function": {
                    "name": "inspect_log",
                    "arguments": arguments,
                }
            }
        ]
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "calling tool",
            None,
            tool_calls,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 0
    assert detail["externalized_payload_refs_missing"] == 0
    assert detail["missing_externalized_payload_refs"] == []


def test_externalized_payload_integrity_scan_ignores_single_quoted_placeholder_examples_inside_tool_call_json(tmp_path):
    engine = _engine(tmp_path)
    (tmp_path / "externalized").mkdir()
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=tool_calls; "
        "chars=1; bytes=1; ref=example-single-quote.json]"
    )
    arguments = json.dumps({"log": f"pytest output: 'prefix before placeholder {placeholder}'"})
    tool_calls = json.dumps(
        [
            {
                "function": {
                    "name": "inspect_log",
                    "arguments": arguments,
                }
            }
        ]
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "calling tool",
            None,
            tool_calls,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 0
    assert detail["externalized_payload_refs_missing"] == 0
    assert detail["missing_externalized_payload_refs"] == []


def test_externalized_payload_integrity_scan_counts_real_refs_inside_tool_call_log_quotes(tmp_path):
    engine = _engine(tmp_path)
    storage_dir = tmp_path / "externalized"
    storage_dir.mkdir()
    (storage_dir / "20260625-real-tool-call-media.json").write_text(json.dumps({"content": "payload"}))
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=tool_calls; "
        "chars=1; bytes=1; ref=20260625-real-tool-call-media.json]"
    )
    arguments = json.dumps({"log": f'pytest output: "prefix before placeholder {placeholder}"'})
    tool_calls = json.dumps(
        [
            {
                "function": {
                    "name": "inspect_log",
                    "arguments": arguments,
                }
            }
        ]
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "calling tool",
            None,
            tool_calls,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 1
    assert detail["externalized_payload_refs_existing"] == 1
    assert detail["externalized_payload_refs_missing"] == 0
    assert detail["externalized_payload_files_unreferenced"] == 0


def test_externalized_payload_integrity_scan_detects_embedded_tool_content_placeholder(tmp_path):
    engine = _engine(tmp_path)
    (tmp_path / "externalized").mkdir()
    content = (
        'log returned \\\"preview\\\" plus '
        "[Externalized LCM ingest payload: kind=media_payload; field=content; "
        "chars=1; bytes=1; ref=missing-tool-content-media.json]"
    )
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "tool",
            content,
            None,
            None,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 1
    assert detail["externalized_payload_refs_missing"] == 1
    assert detail["missing_externalized_payload_refs"] == [
        {
            "store_id": 1,
            "session_id": engine.current_session_id,
            "source": "telegram",
            "role": "tool",
            "field": "content",
            "externalized_ref": "missing-tool-content-media.json",
        }
    ]


def test_externalized_payload_integrity_scan_detects_escaped_json_tool_content_placeholder(tmp_path):
    engine = _engine(tmp_path)
    storage_dir = tmp_path / "externalized"
    storage_dir.mkdir()
    (storage_dir / "present-tool-content-media.json").write_text(json.dumps({"content": "payload"}))
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=content; "
        "chars=1; bytes=1; ref=present-tool-content-media.json]"
    )
    content = '{\\"output\\":\\"' + placeholder + '\\",\\"output\\":\\"fallback\\"}'
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "tool",
            content,
            None,
            None,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 1
    assert detail["externalized_payload_refs_existing"] == 1
    assert detail["externalized_payload_refs_missing"] == 0
    assert detail["externalized_payload_files_unreferenced"] == 0


def test_externalized_payload_integrity_scan_ignores_escaped_placeholder_examples_inside_tool_content_json(tmp_path):
    engine = _engine(tmp_path)
    (tmp_path / "externalized").mkdir()
    placeholder = (
        "[Externalized LCM ingest payload: kind=media_payload; field=content; "
        "chars=1; bytes=1; ref=example-tool-content.json]"
    )
    content = '{\\"log\\":\\"pytest output: \\\\\\\"prefix before placeholder ' + placeholder + '\\\\\\\"\\"}'
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "tool",
            content,
            None,
            None,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    detail = scan_externalized_payload_integrity(engine._store._conn, engine._config, hermes_home=engine._hermes_home)

    assert detail["externalized_payload_refs_total"] == 0
    assert detail["externalized_payload_refs_missing"] == 0
    assert detail["missing_externalized_payload_refs"] == []


def test_lcm_doctor_warns_on_missing_externalized_payload_refs_when_inline_payloads_are_clean(tmp_path):
    engine = _engine(tmp_path)
    (tmp_path / "externalized").mkdir()
    engine._store._conn.execute(
        """INSERT INTO messages
           (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            engine.current_session_id,
            "telegram",
            "assistant",
            "[Externalized payload: kind=raw_payload; role=assistant; chars=1; bytes=1; ref=missing-doctor.json]",
            None,
            None,
            None,
            1.0,
            1,
            0,
        ),
    )
    engine._store._conn.commit()

    json_result = json.loads(lcm_tools.lcm_doctor({}, engine=engine))

    payload_check = next(check for check in json_result["checks"] if check["check"] == "payload_storage")
    assert payload_check["status"] == "warn"
    assert payload_check["detail"]["suspicious_data_uri_content_rows"] == []
    assert payload_check["detail"]["suspicious_base64_like_rows"] == []
    assert payload_check["detail"]["externalized_payload_refs_total"] == 1
    assert payload_check["detail"]["externalized_payload_refs_missing"] == 1
    assert payload_check["detail"]["missing_externalized_payload_refs"][0]["externalized_ref"] == "missing-doctor.json"


BROKEN_ASSISTANT_MARKER = "BROKEN_ASSISTANT_LOOP_MARKER_196"


def _broken_assistant_output() -> str:
    paragraph = (
        f"{BROKEN_ASSISTANT_MARKER}: I am drafting the same response again. "
        "I will repeat the plan, restate the same tool loop, and continue without adding new information. "
        "This paragraph is intentionally repetitive model-loop output.\n"
    )
    return paragraph * 520


def _legitimate_long_report() -> str:
    return "\n".join(
        f"Section {idx:04d}: finding={idx * 17}; evidence=case-{idx:04d}; "
        f"next_step=review-module-{idx % 97}; note=unique-context-{idx * idx}."
        for idx in range(1800)
    )


def test_repetitive_assistant_output_is_quarantined_before_sqlite_and_fts(tmp_path):
    engine = _engine(tmp_path)
    broken = _broken_assistant_output()
    assert len(broken) > 70_000

    engine._ingest_messages([{"role": "assistant", "content": broken}])

    store_id, content, _tool_calls = _single_message_row(engine, role="assistant")
    assert "assistant output quarantined" in content
    assert "high_repetition" in content
    assert BROKEN_ASSISTANT_MARKER not in content
    assert engine._store.search(BROKEN_ASSISTANT_MARKER, session_id=engine.current_session_id) == []

    ref = _extract_ref(content)
    expanded = _expand_ref(engine, ref)
    assert expanded["kind"] == "quarantined_assistant_output"
    assert expanded["content"] == broken

    raw_message = json.loads(lcm_tools.lcm_expand({"store_id": store_id, "max_tokens": 100_000}, engine=engine))
    assert raw_message["externalized_ref"] == ref
    assert raw_message["externalized"]["kind"] == "quarantined_assistant_output"

    doctor = handle_lcm_command("doctor", engine)
    assert "quarantined_assistant_rows:" in doctor
    assert "suspicious_repetitive_assistant_rows: []" in doctor


def test_quarantined_assistant_output_does_not_enter_summaries_or_active_context(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=100,
        context_threshold=0.10,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "quarantine-summary-session",
        platform="telegram",
        conversation_id="quarantine-summary-conversation",
        context_length=10_000,
    )
    broken = _broken_assistant_output()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "please help"},
        {"role": "assistant", "content": broken},
        {"role": "user", "content": "fresh tail"},
    ]

    def summarize_without_marker(**kwargs):
        text = kwargs["text"]
        assert BROKEN_ASSISTANT_MARKER not in text
        return text, 1

    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", summarize_without_marker)

    active_context = engine.compress(messages)

    nodes = engine._dag.get_session_nodes(engine.current_session_id)
    assert nodes
    assert all(BROKEN_ASSISTANT_MARKER not in node.summary for node in nodes)
    assert all(BROKEN_ASSISTANT_MARKER not in str(message.get("content", "")) for message in active_context)
    assert any("assistant output quarantined" in str(message.get("content", "")) for message in active_context)


def test_quarantined_assistant_output_does_not_enter_summaries_or_active_context_after_preflight(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=100,
        context_threshold=0.10,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "quarantine-preflight-session",
        platform="telegram",
        conversation_id="quarantine-preflight-conversation",
        context_length=10_000,
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "please help"},
        {"role": "assistant", "content": _broken_assistant_output()},
        {"role": "user", "content": "fresh tail"},
    ]

    assert engine.should_compress_preflight(messages)

    def summarize_without_marker(**kwargs):
        text = kwargs["text"]
        assert BROKEN_ASSISTANT_MARKER not in text
        return text, 1

    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", summarize_without_marker)

    active_context = engine.compress(messages)

    assert all(BROKEN_ASSISTANT_MARKER not in str(message.get("content", "")) for message in active_context)
    assert any("assistant output quarantined" in str(message.get("content", "")) for message in active_context)
    nodes = engine._dag.get_session_nodes(engine.current_session_id)
    assert nodes
    assert all(BROKEN_ASSISTANT_MARKER not in node.summary for node in nodes)


def test_preflight_quarantined_assistant_rebind_keeps_durable_placeholder(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    messages = [
        {"role": "user", "content": "please help"},
        {"role": "assistant", "content": _broken_assistant_output()},
    ]

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first.on_session_start(
        "quarantine-preflight-rebind-session",
        platform="telegram",
        conversation_id="quarantine-preflight-rebind-conversation",
        context_length=1_000_000,
    )
    assert first.should_compress_preflight(messages)

    preflight_rows = first._store.get_session_messages(first.current_session_id)
    assert len(preflight_rows) == 2
    assert "Externalized LCM ingest payload" in str(preflight_rows[1].get("content", ""))
    assert "LCM active replay placeholder" not in str(preflight_rows[1].get("content", ""))

    active_context = first.compress(messages)
    assert len(active_context) == 2
    assert all(BROKEN_ASSISTANT_MARKER not in str(message.get("content", "")) for message in active_context)
    assert "Externalized LCM ingest payload" in str(active_context[1].get("content", ""))
    assert "LCM active replay placeholder" not in str(active_context[1].get("content", ""))
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second.on_session_start(
        "quarantine-preflight-rebind-session",
        platform="telegram",
        conversation_id="quarantine-preflight-rebind-conversation",
        context_length=1_000_000,
    )
    second.compress(active_context)

    rebound_rows = second._store.get_session_messages(second.current_session_id)
    assert len(rebound_rows) == 2
    assert [row["role"] for row in rebound_rows] == ["user", "assistant"]
    assert "Externalized LCM ingest payload" in str(rebound_rows[1].get("content", ""))
    assert all("LCM active replay placeholder" not in str(row.get("content", "")) for row in rebound_rows)


def test_dynamic_quarantined_assistant_pressure_continues_after_first_leaf_pass(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=100,
        dynamic_leaf_chunk_enabled=True,
        dynamic_leaf_chunk_max=100,
        context_threshold=0.10,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "dynamic-quarantine-pressure-session",
        platform="telegram",
        conversation_id="dynamic-quarantine-pressure-conversation",
        context_length=10_000,
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": " ".join(["large-user"] * 130)},
        {"role": "assistant", "content": _broken_assistant_output()},
        {"role": "user", "content": "fresh tail"},
    ]
    summarized_texts: list[str] = []

    def summarize_without_marker(**kwargs):
        text = kwargs["text"]
        assert BROKEN_ASSISTANT_MARKER not in text
        summarized_texts.append(text)
        return "summary", 1

    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", summarize_without_marker)

    active_context = engine.compress(messages, current_tokens=count_messages_tokens(messages))

    nodes = engine._dag.get_session_nodes(engine.current_session_id)
    assert len(nodes) >= 2
    assert len(summarized_texts) >= 2
    assert any("assistant output quarantined" in text for text in summarized_texts)
    assert all(BROKEN_ASSISTANT_MARKER not in str(message.get("content", "")) for message in active_context)


def test_quarantined_assistant_tool_call_content_does_not_enter_summarization(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=100,
        context_threshold=0.10,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "quarantine-tool-call-summary-session",
        platform="telegram",
        conversation_id="quarantine-tool-call-summary-conversation",
        context_length=10_000,
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "please call the tool"},
        {
            "role": "assistant",
            "content": _broken_assistant_output(),
            "tool_calls": [{"id": "call_loop", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_loop", "content": "tool result"},
        {"role": "user", "content": "fresh tail"},
    ]

    def summarize_without_marker(**kwargs):
        text = kwargs["text"]
        assert BROKEN_ASSISTANT_MARKER not in text
        assert "assistant output quarantined" in text
        return text, 1

    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", summarize_without_marker)

    active_context = engine.compress(messages)

    assert all(BROKEN_ASSISTANT_MARKER not in str(message.get("content", "")) for message in active_context)
    _store_id, content, _tool_calls = _single_message_row(engine, role="assistant")
    assert "assistant output quarantined" in content
    assert BROKEN_ASSISTANT_MARKER not in content


def test_quarantined_assistant_tool_call_content_is_removed_from_noop_active_replay(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "quarantine-tool-call-noop-session",
        platform="telegram",
        conversation_id="quarantine-tool-call-noop-conversation",
        context_length=10_000,
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "please call the tool"},
        {
            "role": "assistant",
            "content": _broken_assistant_output(),
            "tool_calls": [{"id": "call_loop", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_loop", "content": "tool result"},
    ]

    active_context = engine.compress(messages)

    assistant = next(message for message in active_context if message.get("role") == "assistant")
    assert assistant["tool_calls"] == messages[2]["tool_calls"]
    assert "assistant output quarantined" in assistant["content"]
    assert BROKEN_ASSISTANT_MARKER not in assistant["content"]


def test_quarantined_assistant_rebind_reconciliation_does_not_duplicate_rows(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    messages = [
        {"role": "user", "content": "please help"},
        {"role": "assistant", "content": _broken_assistant_output()},
    ]

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first.on_session_start(
        "quarantine-rebind-session",
        platform="telegram",
        conversation_id="quarantine-rebind-conversation",
        context_length=10_000,
    )
    first.compress(messages)
    first_count = len(first._store.get_session_messages(first.current_session_id))
    assert first_count == 2
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second.on_session_start(
        "quarantine-rebind-session",
        platform="telegram",
        conversation_id="quarantine-rebind-conversation",
        context_length=10_000,
    )
    second.compress(messages)

    rows = second._store.get_session_messages(second.current_session_id)
    assert len(rows) == first_count
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert "assistant output quarantined" in rows[-1]["content"]
    assert BROKEN_ASSISTANT_MARKER not in rows[-1]["content"]


class _PrefixPattern:
    pattern = "^Cronjob Response:"

    def search(self, text, timeout=None):
        return object() if str(text).startswith("Cronjob Response:") else None


class _CountingIgnorePattern:
    pattern = "^DROP:"

    def __init__(self):
        self.seen: list[str] = []

    def search(self, text, timeout=None):
        self.seen.append(str(text))
        return object() if str(text).startswith("DROP:") else None


def test_ignore_message_patterns_scan_only_new_tail_after_cursor(tmp_path):
    engine = _engine(tmp_path)
    pattern = _CountingIgnorePattern()
    engine._compiled_ignore_message_patterns = [pattern]
    messages = [
        {"role": "user", "content": f"old message {idx}"}
        for idx in range(50)
    ]

    engine._ingest_messages(messages)
    pattern.seen.clear()
    engine._ingest_messages(messages + [{"role": "user", "content": "new message"}])

    assert pattern.seen == ["new message"]
    rows = engine._store.get_session_messages(engine.current_session_id)
    assert [row["content"] for row in rows][-2:] == ["old message 49", "new message"]


def test_ignore_message_pattern_drop_is_counted_and_surfaced_in_status(tmp_path):
    engine = _engine(tmp_path)
    engine._compiled_ignore_message_patterns = [_CountingIgnorePattern()]

    engine._ingest_messages([
        {"role": "user", "content": "keep this substantive turn"},
        {"role": "user", "content": "DROP: noisy heartbeat"},
    ])

    # The matched message is not persisted (unchanged behavior)...
    rows = engine._store.get_session_messages(engine.current_session_id)
    assert [row["content"] for row in rows] == ["keep this substantive turn"]
    # ...but the drop is no longer silent: it is counted and visible in status.
    assert engine._ignore_pattern_dropped_count == 1
    status = engine.get_status()
    assert status["ignore_pattern_dropped_count"] == 1

    doctor = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
    drop_check = next(c for c in doctor["checks"] if c["check"] == "ignore_pattern_drops")
    assert drop_check["status"] == "warn"


def test_live_placeholder_text_does_not_match_ignore_pattern_via_payload(tmp_path):
    engine = _engine(tmp_path)
    pattern = _CountingIgnorePattern()
    engine._compiled_ignore_message_patterns = [pattern]
    result = externalize_ingest_payload(
        "DROP: hidden payload should not filter copied placeholder text",
        role="user",
        session_id=engine.current_session_id,
        field_path="content",
        config=engine._config,
        hermes_home=str(tmp_path / "home"),
    )
    assert result is not None
    placeholder = result["placeholder"]

    engine.compress([{"role": "user", "content": placeholder}])

    rows = engine._store.get_session_messages(engine.current_session_id)
    assert [row["content"] for row in rows] == [placeholder]


def test_ignore_message_patterns_remain_storage_only_for_compress_replay(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine._compiled_ignore_message_patterns = [_PrefixPattern()]
    engine.on_session_start(
        "ignore-storage-only-session",
        platform="telegram",
        conversation_id="ignore-storage-only-conversation",
        context_length=10_000,
    )
    ignored = "Cronjob Response: noisy heartbeat"
    kept = "real user request"
    messages = [
        {"role": "user", "content": ignored},
        {"role": "user", "content": kept},
    ]

    active_context = engine.compress(messages)

    active_contents = [message.get("content") for message in active_context]
    assert active_contents[0].startswith("[LCM active replay placeholder: message ignored;")
    assert ignored not in active_contents[0]
    assert active_contents[1] == kept
    stored_contents = [row["content"] for row in engine._store.get_session_messages(engine.current_session_id)]
    assert stored_contents == [kept]
    assert engine._store.search("noisy heartbeat", session_id=engine.current_session_id) == []


def test_quarantined_assistant_preflight_requests_cleanup_when_no_compaction_needed(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "quarantine-preflight-noop-session",
        platform="telegram",
        conversation_id="quarantine-preflight-noop-conversation",
        context_length=10_000,
    )
    messages = [
        {"role": "user", "content": "please help"},
        {"role": "assistant", "content": _broken_assistant_output()},
    ]

    assert engine.should_compress_preflight(messages)

    active_context = engine.compress(messages)
    assert any("assistant output quarantined" in str(message.get("content", "")) for message in active_context)
    assert all(BROKEN_ASSISTANT_MARKER not in str(message.get("content", "")) for message in active_context)


class _ContainsBrokenAssistantPattern:
    pattern = BROKEN_ASSISTANT_MARKER

    def search(self, text, timeout=None):
        return object() if BROKEN_ASSISTANT_MARKER in str(text) else None


class _NeverMatchesPattern:
    pattern = "NEVER_MATCHES_BROKEN_ASSISTANT"

    def search(self, text, timeout=None):
        return None


def test_ignore_message_patterns_match_original_suspicious_assistant_before_storage(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine._compiled_ignore_message_patterns = [_ContainsBrokenAssistantPattern()]
    engine.on_session_start(
        "ignore-quarantine-storage-session",
        platform="telegram",
        conversation_id="ignore-quarantine-storage-conversation",
        context_length=10_000,
    )
    messages = [
        {"role": "assistant", "content": _broken_assistant_output()},
        {"role": "user", "content": "fresh request"},
    ]

    active_context = engine.compress(messages)

    assert "assistant output quarantined" in str(active_context[0].get("content", ""))
    assert BROKEN_ASSISTANT_MARKER not in str(active_context[0].get("content", ""))
    stored_rows = engine._store.get_session_messages(engine.current_session_id)
    assert [row["role"] for row in stored_rows] == ["user"]
    assert engine._store.search(BROKEN_ASSISTANT_MARKER, session_id=engine.current_session_id) == []
    assert _externalized_files(tmp_path) == []


def test_ignored_quarantined_assistant_rebind_reconciliation_does_not_duplicate_rows(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": _broken_assistant_output()},
        {"role": "user", "content": "fresh request"},
    ]

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first._compiled_ignore_message_patterns = [_ContainsBrokenAssistantPattern()]
    first.on_session_start(
        "ignore-quarantine-rebind-session",
        platform="telegram",
        conversation_id="ignore-quarantine-rebind-conversation",
        context_length=10_000,
    )
    first_active = first.compress(messages)
    assert "assistant output quarantined" in str(first_active[1].get("content", ""))
    first_rows = first._store.get_session_messages(first.current_session_id)
    assert [row["role"] for row in first_rows] == ["system", "user"]
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second._compiled_ignore_message_patterns = [_ContainsBrokenAssistantPattern()]
    second.on_session_start(
        "ignore-quarantine-rebind-session",
        platform="telegram",
        conversation_id="ignore-quarantine-rebind-conversation",
        context_length=10_000,
    )
    second_active = second.compress(first_active)

    second_rows = second._store.get_session_messages(second.current_session_id)
    assert [row["role"] for row in second_rows] == ["system", "user"]
    assert "assistant output quarantined" in str(second_active[1].get("content", ""))
    assert BROKEN_ASSISTANT_MARKER not in str(second_active[1].get("content", ""))


def test_existing_quarantined_assistant_row_rebinds_after_ignore_pattern_added(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": _broken_assistant_output()},
        {"role": "user", "content": "fresh request"},
    ]

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first.on_session_start(
        "ignore-added-after-quarantine-session",
        platform="telegram",
        conversation_id="ignore-added-after-quarantine-conversation",
        context_length=10_000,
    )
    first_active = first.compress(messages)
    assert "assistant output quarantined" in str(first_active[1].get("content", ""))
    first_rows = first._store.get_session_messages(first.current_session_id)
    assert [row["role"] for row in first_rows] == ["system", "assistant", "user"]
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second._compiled_ignore_message_patterns = [_ContainsBrokenAssistantPattern()]
    second.on_session_start(
        "ignore-added-after-quarantine-session",
        platform="telegram",
        conversation_id="ignore-added-after-quarantine-conversation",
        context_length=10_000,
    )
    second_active = second.compress(first_active)

    second_rows = second._store.get_session_messages(second.current_session_id)
    assert [row["role"] for row in second_rows] == ["system", "assistant", "user"]
    assert "LCM active replay placeholder: message ignored" in str(second_active[1].get("content", ""))
    assert BROKEN_ASSISTANT_MARKER not in str(second_active[1].get("content", ""))


def test_nonmatching_ignore_pattern_preserves_existing_quarantine_rebind_prefix(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": _broken_assistant_output()},
        {"role": "user", "content": "fresh request"},
    ]

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first.on_session_start(
        "nonmatching-ignore-quarantine-session",
        platform="telegram",
        conversation_id="nonmatching-ignore-quarantine-conversation",
        context_length=10_000,
    )
    first_active = first.compress(messages)
    assert "assistant output quarantined" in str(first_active[1].get("content", ""))
    assert [row["role"] for row in first._store.get_session_messages(first.current_session_id)] == [
        "system",
        "assistant",
        "user",
    ]
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second._compiled_ignore_message_patterns = [_NeverMatchesPattern()]
    second.on_session_start(
        "nonmatching-ignore-quarantine-session",
        platform="telegram",
        conversation_id="nonmatching-ignore-quarantine-conversation",
        context_length=10_000,
    )
    second.compress(first_active)

    second_rows = second._store.get_session_messages(second.current_session_id)
    assert [row["role"] for row in second_rows] == ["system", "assistant", "user"]
    assert len(second_rows) == 3


def test_singleton_quarantined_assistant_row_rebinds_after_ignore_pattern_added(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    messages = [{"role": "assistant", "content": _broken_assistant_output()}]

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first.on_session_start(
        "singleton-ignore-added-session",
        platform="telegram",
        conversation_id="singleton-ignore-added-conversation",
        context_length=10_000,
    )
    first_active = first.compress(messages)
    assert "assistant output quarantined" in str(first_active[0].get("content", ""))
    assert len(first._store.get_session_messages(first.current_session_id)) == 1
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second._compiled_ignore_message_patterns = [_ContainsBrokenAssistantPattern()]
    second.on_session_start(
        "singleton-ignore-added-session",
        platform="telegram",
        conversation_id="singleton-ignore-added-conversation",
        context_length=10_000,
    )
    second.compress(first_active)

    second_rows = second._store.get_session_messages(second.current_session_id)
    assert len(second_rows) == 1
    assert [row["role"] for row in second_rows] == ["assistant"]


def test_singleton_quarantined_assistant_rebind_reconciliation_does_not_duplicate_row(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    messages = [{"role": "assistant", "content": _broken_assistant_output()}]

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first.on_session_start(
        "singleton-quarantine-rebind-session",
        platform="telegram",
        conversation_id="singleton-quarantine-rebind-conversation",
        context_length=10_000,
    )
    first_active = first.compress(messages)
    assert "assistant output quarantined" in str(first_active[0].get("content", ""))
    assert len(first._store.get_session_messages(first.current_session_id)) == 1
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second.on_session_start(
        "singleton-quarantine-rebind-session",
        platform="telegram",
        conversation_id="singleton-quarantine-rebind-conversation",
        context_length=10_000,
    )
    second_active = second.compress(messages)

    second_rows = second._store.get_session_messages(second.current_session_id)
    assert len(second_rows) == 1
    assert "assistant output quarantined" in str(second_active[0].get("content", ""))
    assert BROKEN_ASSISTANT_MARKER not in str(second_active[0].get("content", ""))


def test_fresh_singleton_quarantined_assistant_delta_after_rebind_is_stored(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first.on_session_start(
        "fresh-singleton-quarantine-delta-session",
        platform="telegram",
        conversation_id="fresh-singleton-quarantine-delta-conversation",
        context_length=10_000,
    )
    first.compress([{"role": "assistant", "content": _broken_assistant_output()}])
    assert len(first._store.get_session_messages(first.current_session_id)) == 1
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second.on_session_start(
        "fresh-singleton-quarantine-delta-session",
        platform="telegram",
        conversation_id="fresh-singleton-quarantine-delta-conversation",
        context_length=10_000,
    )
    second.compress([{"role": "assistant", "content": _broken_assistant_output() + " distinct fresh delta"}])

    second_rows = second._store.get_session_messages(second.current_session_id)
    assert len(second_rows) == 2
    assert second._last_ingest_reconciliation["action"] == "persisted batch"


def test_no_system_ignored_quarantined_assistant_rebind_does_not_duplicate_tail(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    messages = [
        {"role": "assistant", "content": _broken_assistant_output()},
        {"role": "user", "content": "fresh request"},
    ]

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first._compiled_ignore_message_patterns = [_ContainsBrokenAssistantPattern()]
    first.on_session_start(
        "no-system-ignore-quarantine-rebind-session",
        platform="telegram",
        conversation_id="no-system-ignore-quarantine-rebind-conversation",
        context_length=10_000,
    )
    first_active = first.compress(messages)
    assert [row["role"] for row in first._store.get_session_messages(first.current_session_id)] == ["user"]
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second._compiled_ignore_message_patterns = [_ContainsBrokenAssistantPattern()]
    second.on_session_start(
        "no-system-ignore-quarantine-rebind-session",
        platform="telegram",
        conversation_id="no-system-ignore-quarantine-rebind-conversation",
        context_length=10_000,
    )
    second_active = second.compress(first_active)

    second_rows = second._store.get_session_messages(second.current_session_id)
    assert [row["role"] for row in second_rows] == ["user"]
    assert [row["content"] for row in second_rows] == ["fresh request"]
    assert "assistant output quarantined" in str(second_active[0].get("content", ""))
    assert "sha256=" in str(second_active[0].get("content", ""))
    assert BROKEN_ASSISTANT_MARKER not in str(second_active[0].get("content", ""))


def test_no_system_trailing_ignored_quarantined_assistant_rebind_does_not_duplicate_tail(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    messages = [
        {"role": "user", "content": "fresh request"},
        {"role": "assistant", "content": _broken_assistant_output()},
    ]

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first._compiled_ignore_message_patterns = [_ContainsBrokenAssistantPattern()]
    first.on_session_start(
        "no-system-trailing-ignore-quarantine-rebind-session",
        platform="telegram",
        conversation_id="no-system-trailing-ignore-quarantine-rebind-conversation",
        context_length=10_000,
    )
    first_active = first.compress(messages)
    assert [row["content"] for row in first._store.get_session_messages(first.current_session_id)] == ["fresh request"]
    assert "sha256=" in str(first_active[-1].get("content", ""))
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second._compiled_ignore_message_patterns = [_ContainsBrokenAssistantPattern()]
    second.on_session_start(
        "no-system-trailing-ignore-quarantine-rebind-session",
        platform="telegram",
        conversation_id="no-system-trailing-ignore-quarantine-rebind-conversation",
        context_length=10_000,
    )
    second.compress(first_active)

    second_rows = second._store.get_session_messages(second.current_session_id)
    assert [row["content"] for row in second_rows] == ["fresh request"]


def test_only_ignored_quarantined_assistant_rebind_does_not_store_placeholder(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first._compiled_ignore_message_patterns = [_ContainsBrokenAssistantPattern()]
    first.on_session_start(
        "only-ignored-quarantine-rebind-session",
        platform="telegram",
        conversation_id="only-ignored-quarantine-rebind-conversation",
        context_length=10_000,
    )
    first_active = first.compress([{"role": "assistant", "content": _broken_assistant_output()}])
    assert first._store.get_session_messages(first.current_session_id) == []
    assert "sha256=" in str(first_active[0].get("content", ""))
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second._compiled_ignore_message_patterns = [_ContainsBrokenAssistantPattern()]
    second.on_session_start(
        "only-ignored-quarantine-rebind-session",
        platform="telegram",
        conversation_id="only-ignored-quarantine-rebind-conversation",
        context_length=10_000,
    )
    second.compress(first_active)

    assert second._store.get_session_messages(second.current_session_id) == []


def test_ignore_message_patterns_do_not_drop_plain_text_that_mentions_quarantine_markers(tmp_path):
    class _DropOnlyPattern:
        pattern = "drop me only"

        def search(self, text, timeout=None):
            return object() if "drop me only" in str(text) else None

    engine = _engine(tmp_path)
    engine._compiled_ignore_message_patterns = [_DropOnlyPattern()]
    text = (
        "This is a normal note mentioning scope=ignored_message_pattern, "
        "assistant output quarantined, and quarantined_assistant_output."
    )

    engine.compress([{"role": "user", "content": text}])

    rows = engine._store.get_session_messages(engine.current_session_id)
    assert [row["content"] for row in rows] == [text]


def test_rebind_with_ignore_patterns_preserves_assistant_text_that_mentions_quarantine_markers(tmp_path):
    class _DropOnlyPattern:
        pattern = "drop me only"

        def search(self, text, timeout=None):
            return object() if "drop me only" in str(text) else None

    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    text = (
        "This assistant message literally mentions assistant output quarantined "
        "and quarantined_assistant_output, but it is not an LCM placeholder."
    )

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first.on_session_start(
        "literal-quarantine-marker-session",
        platform="telegram",
        conversation_id="literal-quarantine-marker-conversation",
        context_length=10_000,
    )
    first.compress([{"role": "user", "content": "seed"}])
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second._compiled_ignore_message_patterns = [_DropOnlyPattern()]
    second.on_session_start(
        "literal-quarantine-marker-session",
        platform="telegram",
        conversation_id="literal-quarantine-marker-conversation",
        context_length=10_000,
    )
    second.compress([{"role": "assistant", "content": text}])

    rows = second._store.get_session_messages(second.current_session_id)
    assert [row["content"] for row in rows] == ["seed", text]
    assert second._last_ingest_reconciliation["action"] == "persisted batch"


def test_rebind_with_ignore_patterns_preserves_trailing_literal_quarantine_placeholder_text(tmp_path):
    class _DropOnlyPattern:
        pattern = "drop me only"

        def search(self, text, timeout=None):
            return object() if "drop me only" in str(text) else None

    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    literal = (
        "[LCM active replay placeholder: assistant output quarantined; "
        "kind=quarantined_assistant_output; reason=high_repetition; "
        "scope=ignored_message_pattern; field=content; chars=65536; bytes=65536]"
    )

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first.on_session_start(
        "trailing-literal-quarantine-placeholder-session",
        platform="telegram",
        conversation_id="trailing-literal-quarantine-placeholder-conversation",
        context_length=10_000,
    )
    first.compress([{"role": "user", "content": "seed"}])
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second._compiled_ignore_message_patterns = [_DropOnlyPattern()]
    second.on_session_start(
        "trailing-literal-quarantine-placeholder-session",
        platform="telegram",
        conversation_id="trailing-literal-quarantine-placeholder-conversation",
        context_length=10_000,
    )
    second.compress([{"role": "assistant", "content": literal}])

    rows = second._store.get_session_messages(second.current_session_id)
    assert [row["content"] for row in rows] == ["seed", literal]


def test_rebind_with_ignore_patterns_preserves_literal_quarantine_placeholder_before_repeated_tail(tmp_path):
    class _DropOnlyPattern:
        pattern = "drop me only"

        def search(self, text, timeout=None):
            return object() if "drop me only" in str(text) else None

    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    literal = (
        "[LCM active replay placeholder: assistant output quarantined; "
        "kind=quarantined_assistant_output; reason=high_repetition; "
        "scope=ignored_message_pattern; field=content; chars=65536; bytes=65536]"
    )

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first.on_session_start(
        "literal-quarantine-placeholder-before-tail-session",
        platform="telegram",
        conversation_id="literal-quarantine-placeholder-before-tail-conversation",
        context_length=10_000,
    )
    first.compress([{"role": "user", "content": "fresh request"}])
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second._compiled_ignore_message_patterns = [_DropOnlyPattern()]
    second.on_session_start(
        "literal-quarantine-placeholder-before-tail-session",
        platform="telegram",
        conversation_id="literal-quarantine-placeholder-before-tail-conversation",
        context_length=10_000,
    )
    second.compress([
        {"role": "assistant", "content": literal},
        {"role": "user", "content": "fresh request"},
    ])

    rows = second._store.get_session_messages(second.current_session_id)
    assert [row["content"] for row in rows] == ["fresh request", literal, "fresh request"]
    assert second._last_ingest_reconciliation["action"] == "persisted batch"


def test_ignore_message_patterns_do_not_drop_literal_quarantine_placeholder_text(tmp_path):
    class _DropOnlyPattern:
        pattern = "drop me only"

        def search(self, text, timeout=None):
            return object() if "drop me only" in str(text) else None

    engine = _engine(tmp_path)
    engine._compiled_ignore_message_patterns = [_DropOnlyPattern()]
    text = (
        "[LCM active replay placeholder: assistant output quarantined; "
        "kind=quarantined_assistant_output; reason=high_repetition; "
        "scope=ignored_message_pattern; field=content; chars=65536; bytes=65536]"
    )

    engine.compress([{"role": "assistant", "content": text}])

    rows = engine._store.get_session_messages(engine.current_session_id)
    assert [row["content"] for row in rows] == [text]


def test_rebind_does_not_skip_literal_quarantine_placeholder_without_ignore_patterns(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    literal = (
        "[LCM active replay placeholder: assistant output quarantined; "
        "kind=quarantined_assistant_output; reason=high_repetition; "
        "scope=ignored_message_pattern; field=content; chars=65536; bytes=65536]"
    )

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first.on_session_start(
        "literal-placeholder-rebind-session",
        platform="telegram",
        conversation_id="literal-placeholder-rebind-conversation",
        context_length=10_000,
    )
    first.compress([{"role": "user", "content": "seed"}])
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second.on_session_start(
        "literal-placeholder-rebind-session",
        platform="telegram",
        conversation_id="literal-placeholder-rebind-conversation",
        context_length=10_000,
    )
    second.compress([{"role": "assistant", "content": literal}])

    rows = second._store.get_session_messages(second.current_session_id)
    assert [row["content"] for row in rows] == ["seed", literal]


def test_no_system_raw_ignored_quarantined_assistant_rebind_preserves_repeated_tail_delta(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    messages = [
        {"role": "assistant", "content": _broken_assistant_output()},
        {"role": "user", "content": "fresh request"},
    ]

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first._compiled_ignore_message_patterns = [_ContainsBrokenAssistantPattern()]
    first.on_session_start(
        "no-system-raw-ignore-quarantine-rebind-session",
        platform="telegram",
        conversation_id="no-system-raw-ignore-quarantine-rebind-conversation",
        context_length=10_000,
    )
    first.compress(messages)
    assert [row["content"] for row in first._store.get_session_messages(first.current_session_id)] == ["fresh request"]
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second._compiled_ignore_message_patterns = [_ContainsBrokenAssistantPattern()]
    second.on_session_start(
        "no-system-raw-ignore-quarantine-rebind-session",
        platform="telegram",
        conversation_id="no-system-raw-ignore-quarantine-rebind-conversation",
        context_length=10_000,
    )
    second_active = second.compress(messages)

    second_rows = second._store.get_session_messages(second.current_session_id)
    assert [row["content"] for row in second_rows] == ["fresh request", "fresh request"]
    assert "assistant output quarantined" in str(second_active[0].get("content", ""))
    assert BROKEN_ASSISTANT_MARKER not in str(second_active[0].get("content", ""))


def test_no_system_filtered_non_quarantine_rebind_preserves_repeated_tail_delta(tmp_path):
    class _CronPattern:
        pattern = "^Cronjob Response:"

        def search(self, text, timeout=None):
            return object() if str(text).startswith("Cronjob Response:") else None

    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=10,
        leaf_chunk_tokens=10_000,
        context_threshold=0.95,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    first = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    first._compiled_ignore_message_patterns = [_CronPattern()]
    first.on_session_start(
        "no-system-filtered-delta-session",
        platform="telegram",
        conversation_id="no-system-filtered-delta-conversation",
        context_length=10_000,
    )
    first.compress([{"role": "user", "content": "fresh request"}])
    first.shutdown()

    second = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    second._compiled_ignore_message_patterns = [_CronPattern()]
    second.on_session_start(
        "no-system-filtered-delta-session",
        platform="telegram",
        conversation_id="no-system-filtered-delta-conversation",
        context_length=10_000,
    )
    second.compress([
        {"role": "user", "content": "Cronjob Response: noisy heartbeat"},
        {"role": "user", "content": "fresh request"},
    ])

    second_rows = second._store.get_session_messages(second.current_session_id)
    assert [row["content"] for row in second_rows] == ["fresh request", "fresh request"]


def test_legitimate_long_assistant_report_is_not_quarantined(tmp_path):
    engine = _engine(tmp_path)
    report = _legitimate_long_report()
    assert len(report) > 70_000

    engine._ingest_messages([{"role": "assistant", "content": report}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="assistant")
    assert "assistant output quarantined" not in content
    assert "Section 0001" in content
    assert "Section 1799" in content


def test_readme_documents_storage_boundary_payload_guard():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")

    assert "Storage-boundary payload guard" in readme
    assert "messages.content" in readme
    assert "messages.tool_calls" in readme
    assert "data:*;base64" in readme
    assert "Doctor output is metadata-only" in readme
    assert "state.db" in readme
    assert "upstream/outside LCM scope" in readme
    assert "historical rows already present in `lcm.db`" in readme
    assert "backup-first cleanup or migration" in readme


def test_sensitive_private_key_redaction_is_redos_safe_on_pathological_input(tmp_path):
    import time as _time

    engine = _sensitive_engine(tmp_path)

    small = "-----BEGIN RSA PRIVATE KEY-----\nabcdef\n-----END RSA PRIVATE KEY-----"
    assert "BEGIN RSA PRIVATE KEY" not in redact_sensitive_text(small, engine._config)

    pathological = ("-----BEGIN PRIVATE KEY-----\n" + "A" * 64 + "\n") * 20000
    start = _time.perf_counter()
    result = redact_sensitive_text(pathological, engine._config)
    assert _time.perf_counter() - start < 3.0
    assert isinstance(result, str)


def test_sensitive_private_key_fallback_bounds_input_without_regex(tmp_path, monkeypatch):
    import hermes_lcm.ingest_protection as ip

    monkeypatch.setattr(ip, "_regex_engine", None)
    ip._SENSITIVE_REGEX_CATALOG.clear()
    engine = _sensitive_engine(tmp_path)

    small = "-----BEGIN RSA PRIVATE KEY-----\nabcdef\n-----END RSA PRIVATE KEY-----"
    assert "BEGIN RSA PRIVATE KEY" not in ip.redact_sensitive_text(small, engine._config)

    big = "-----BEGIN PRIVATE KEY-----\n" + "A" * (ip._SENSITIVE_STDLIB_MAX_CHARS + 10)
    assert ip.redact_sensitive_text(big, engine._config) == big


def test_ingest_externalizes_line_wrapped_base64_block(tmp_path):
    engine = _engine(tmp_path)
    wrapped = "\n".join(GENERIC_BASE64[i:i + 64] for i in range(0, len(GENERIC_BASE64), 64))

    engine._ingest_messages([{"role": "user", "content": f"attachment:\n{wrapped}\nend"}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert GENERIC_BASE64[:120] not in content
    assert wrapped[:200] not in content
    assert "[Externalized" in content


def test_ingest_externalizes_crlf_wrapped_base64_block(tmp_path):
    engine = _engine(tmp_path)
    wrapped = "\r\n".join(GENERIC_BASE64[i:i + 76] for i in range(0, len(GENERIC_BASE64), 76))

    engine._ingest_messages([{"role": "user", "content": f"attachment:\r\n{wrapped}\r\nend"}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert GENERIC_BASE64[:120] not in content
    assert "[Externalized" in content


def test_ingest_externalizes_wrapped_base64_with_short_terminal_line(tmp_path):
    engine = _engine(tmp_path)
    payload = base64.b64encode(bytes((i * 37) % 256 for i in range(3096))).decode("ascii")
    assert len(payload) == 4096 + 32
    wrapped = "\n".join(payload[i:i + 64] for i in range(0, len(payload), 64))
    terminal_line = wrapped.rsplit("\n", 1)[1]
    assert len(terminal_line) == 32

    engine._ingest_messages([{"role": "user", "content": f"attachment:\n{wrapped}\nend"}])

    _store_id, content, _tool_calls = _single_message_row(engine, role="user")
    assert payload[:120] not in content
    assert terminal_line not in content
    assert content.startswith("attachment:\n[Externalized")
    assert content.endswith("end")
    ref = _extract_ref(content)
    expanded = _expand_ref(engine, ref)
    assert expanded["content"] == wrapped + "\n"


def test_private_key_redaction_fallback_is_case_insensitive(tmp_path, monkeypatch):
    import hermes_lcm.ingest_protection as ip

    engine = _sensitive_engine(tmp_path)
    monkeypatch.setattr(ip, "_regex_engine", None)
    monkeypatch.setattr(ip, "_SENSITIVE_REGEX_CATALOG", {})
    begin = "-----begin " + "private key" + "-----"
    end = "-----EnD " + "PrIvAtE kEy" + "-----"
    key = begin + "\n" + ("A" * 64) + "\n" + end

    redacted = ip.redact_sensitive_text("prefix " + key + " suffix", engine._config)

    assert "begin private key" not in redacted.lower()
    assert "end private key" not in redacted.lower()
    assert "[LCM sensitive redaction: name=private_key" in redacted
    assert redacted.startswith("prefix ")
    assert redacted.endswith(" suffix")


def test_private_key_redaction_fallback_preserves_large_complete_key(tmp_path, monkeypatch):
    import hermes_lcm.ingest_protection as ip

    engine = _sensitive_engine(tmp_path)
    monkeypatch.setattr(ip, "_regex_engine", None)
    monkeypatch.setattr(ip, "_SENSITIVE_REGEX_CATALOG", {})
    key = (
        "-----BEGIN PRIVATE KEY-----\n"
        + "A" * (ip._SENSITIVE_STDLIB_MAX_CHARS + 10)
        + "\n-----END PRIVATE KEY-----"
    )

    redacted = ip.redact_sensitive_text("prefix " + key + " suffix", engine._config)

    assert "BEGIN PRIVATE KEY" not in redacted
    assert "END PRIVATE KEY" not in redacted
    assert "[LCM sensitive redaction: name=private_key" in redacted
    assert redacted.startswith("prefix ")
    assert redacted.endswith(" suffix")


def test_wrapped_base64_scan_ignores_long_single_line_without_regex_backtracking():
    from hermes_lcm.ingest_protection import contains_long_base64_run

    not_payload = "A" * 80_000

    assert contains_long_base64_run(not_payload) is False

def test_sensitive_private_key_regex_timeout_preserves_prior_redactions(tmp_path, monkeypatch):
    import hermes_lcm.ingest_protection as ip

    class TimeoutPattern:
        def sub(self, repl, text, timeout=None):
            raise TimeoutError("synthetic timeout")

    engine = _sensitive_engine(tmp_path)
    monkeypatch.setattr(ip, "_regex_pattern_for", lambda name: TimeoutPattern())
    text = "api_key=sk-test-secret-value-123456 and -----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"

    redacted = ip.redact_sensitive_text(text, engine._config)

    assert "sk-test-secret-value" not in redacted
    assert "BEGIN PRIVATE KEY" not in redacted
    assert "[LCM sensitive redaction: name=api_key" in redacted
    assert "[LCM sensitive redaction: name=private_key" in redacted


def test_wrapped_base64_scan_ignores_hex_hash_inventory():
    from hermes_lcm.ingest_protection import contains_long_base64_run

    hex_lines = "\n".join(f"{i:064x}" for i in range(96))

    assert contains_long_base64_run(hex_lines) is False

def test_wrapped_base64_scan_preserves_short_terminal_line():
    from hermes_lcm.ingest_protection import contains_long_base64_run

    full_line = "QUJD" * 16
    terminal = "REVG" * 4
    payload = "\n".join([full_line] * 70 + [terminal])

    assert contains_long_base64_run(payload) is True
