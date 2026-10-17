"""Replay uses the host's stable identity, preserving genuinely repeated turns."""
import copy
import sqlite3
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_message_tokens


SID = "message-uid-session"


@pytest.fixture
def profile(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    return home


def engine(home, session=SID):
    instance = LCMEngine(config=LCMConfig(database_path=str(home / "lcm.db")), hermes_home=str(home))
    instance.on_session_start(session, hermes_home=str(home))
    return instance


def close(instance):
    instance._close_storage()


def contents(instance, session=SID):
    return [row["content"] for row in instance._store.get_session_messages(session)]


def message(text, uid, role="user", **extra):
    return {"role": role, "content": text, "message_uid": uid, **extra}


def test_uid_cold_replay_filters_known_rows_around_new_occurrence(profile):
    before = engine(profile)
    old = [message("first", "uid-a"), message("last", "uid-c", "assistant")]
    before._ingest_messages(old)
    close(before)
    after = engine(profile)
    try:
        after._ingest_messages([old[0], message("new in the middle", "uid-b"), old[1]])
        assert contents(after) == ["first", "last", "new in the middle"]
    finally:
        close(after)


def test_warm_same_content_new_uid_is_a_new_occurrence(profile):
    instance = engine(profile)
    try:
        instance._ingest_messages([message("repeat", "uid-old")])
        output = instance._ingest_messages([message("repeat", "uid-new")])
        assert contents(instance) == ["repeat", "repeat"]
        assert output[0]["message_uid"] == "uid-new"
    finally:
        close(instance)


def test_new_uid_cannot_be_adopted_from_matching_content_suffix(profile):
    before = engine(profile)
    old = [message("repeat", "uid-1"), message("ack", "uid-2", "assistant")]
    before._ingest_messages(old)
    close(before)
    after = engine(profile)
    try:
        fresh = [message("repeat", "uid-3"), message("ack", "uid-4", "assistant")]
        after._ingest_messages(fresh)
        assert contents(after) == ["repeat", "ack", "repeat", "ack"]
    finally:
        close(after)


def test_mixed_absorbed_user_witness_preserves_new_content_once(profile):
    before = engine(profile)
    before._ingest_messages([message("old request", "uid-a")])
    close(before)
    composite = message("old request\n\nnew request", "uid-a", _absorbed_message_uids=["uid-b"])
    once = engine(profile)
    once._ingest_messages([composite])
    assert contents(once) == ["old request", composite["content"]]
    close(once)
    twice = engine(profile)
    try:
        twice._ingest_messages([copy.deepcopy(composite)])
        assert contents(twice) == ["old request", composite["content"]]
    finally:
        close(twice)


@pytest.mark.parametrize("change", ["content", "reasoning", "witness", "tool_name", "tool_uid"])
def test_changed_registered_projection_is_preserved(profile, change):
    before = engine(profile)
    old = message("original", "uid-a", "assistant", reasoning_content="reason", tool_name="probe",
                  _tool_call_uids={"reused": "call-a"})
    before._ingest_messages([old])
    close(before)
    changed = copy.deepcopy(old)
    if change == "content":
        changed["content"] = "edited"
    elif change == "reasoning":
        changed["reasoning_content"] = "different reasoning"
    elif change == "witness":
        changed["_absorbed_message_uids"] = ["uid-b"]
    elif change == "tool_name":
        changed["tool_name"] = "different_probe"
    else:
        changed["_tool_call_uids"]["reused"] = "call-b"
    after = engine(profile)
    try:
        after._ingest_messages([changed])
        assert after._store.get_session_count(SID) == 2
    finally:
        close(after)


@pytest.mark.parametrize("mutation", ["edit", "delete"])
def test_changed_source_invalidates_uid_proof(profile, mutation):
    before = engine(profile)
    original = message("source", "uid-a")
    before._ingest_messages([original])
    if mutation == "edit":
        before._store._conn.execute("UPDATE messages SET content='modified source' WHERE session_id=?", (SID,))
    else:
        before._store._conn.execute("DELETE FROM messages WHERE session_id=?", (SID,))
    before._store._conn.commit()
    close(before)
    after = engine(profile)
    try:
        after._ingest_messages([original])
        assert contents(after)[-1] == "source"
        assert len(contents(after)) == (2 if mutation == "edit" else 1)
    finally:
        close(after)


def test_metadata_failure_rolls_back_append_and_cursor(profile):
    instance = engine(profile)
    try:
        conn = instance._store._conn
        conn.execute("""CREATE TEMP TRIGGER refuse_uid_bridge BEFORE INSERT ON metadata
                      WHEN NEW.key LIKE 'core_message_origins:%'
                      BEGIN SELECT RAISE(ABORT, 'uid bridge failed'); END""")
        incoming = [message("new", "uid-a")]
        with pytest.raises(sqlite3.IntegrityError, match="uid bridge failed"):
            instance._ingest_messages(incoming)
        with sqlite3.connect(profile / "lcm.db") as reader:
            assert reader.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
            assert reader.execute("SELECT count(*) FROM metadata WHERE key LIKE 'core_message_origins:%'").fetchone()[0] == 0
        assert instance._ingest_cursor == 0
        assert not conn.in_transaction
        conn.execute("DROP TRIGGER refuse_uid_bridge")
        instance._ingest_messages(incoming)
        assert contents(instance) == ["new"]
    finally:
        close(instance)


def test_recovery_projection_preserves_host_identity(profile):
    original = message("text", "uid-a", _absorbed_message_uids=["uid-b"],
                       _tool_call_uids={"call": ["occurrence-a", "occurrence-b"]},
                       _tool_call_uid="occurrence-b", reasoning_content="unpriced reasoning")
    instance = engine(profile)
    try:
        projected = instance._assemble_context(None, [original], assembly_cap_override=1000, include_lcm_note=False)[-1]
        assert projected["message_uid"] == "uid-a"
        for key in ("_absorbed_message_uids", "_tool_call_uids", "_tool_call_uid"):
            assert projected[key] == original[key]
        assert count_message_tokens(projected) == count_message_tokens({"role": "user", "content": "text"})
        assert original["reasoning_content"] == "unpriced reasoning"
    finally:
        close(instance)


def test_actual_core_cold_restore_after_physical_reissue(profile):
    core = pytest.importorskip("hermes_state")
    db = core.SessionDB(db_path=profile / "state.db")
    db.create_session(SID, "cli", model="test/model")
    live = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"raw {i}"} for i in range(6)]
    db.append_messages_batch(SID, live)
    original = db.get_messages_as_conversation(SID, repair_alternation=False)
    original_ids = {row[0] for row in db._conn.execute("SELECT id FROM messages WHERE session_id=?", (SID,))}
    before = engine(profile)
    before._ingest_messages(original)
    before._store.append_batch(SID, [{"role": "tool", "content": "stored-only recovery", "tool_call_id": "orphan"}])
    close(before)
    selected = [original[0], original[-1]]
    db.replace_messages(SID, selected, active_only=True)
    active_ids = {row[0] for row in db._conn.execute("SELECT id FROM messages WHERE session_id=? AND active=1", (SID,))}
    assert active_ids.isdisjoint(original_ids)
    db.close()
    reopened = core.SessionDB(db_path=profile / "state.db")
    restored = reopened.get_messages_as_conversation(SID, repair_alternation=False)
    assert all("_row_id" not in msg for msg in restored)
    assert [msg["message_uid"] for msg in restored] == [msg["message_uid"] for msg in selected]
    after = engine(profile)
    try:
        after._ingest_messages(restored)
        assert contents(after) == [f"raw {i}" for i in range(6)] + ["stored-only recovery"]
    finally:
        close(after)
        reopened.close()


def test_actual_core_assigns_new_uid_to_identical_warm_turn(profile):
    core = pytest.importorskip("hermes_state")
    db = core.SessionDB(db_path=profile / "state.db")
    db.create_session(SID, "cli", model="test/model")
    first = [{"role": "user", "content": "repeat", "timestamp": 1780000000.0}]
    second = copy.deepcopy(first)
    db.append_messages_batch(SID, first)
    instance = engine(profile)
    instance._ingest_messages(db.get_messages_as_conversation(SID, repair_alternation=False))
    db.append_messages_batch(SID, second)
    assert first[0]["message_uid"] != second[0]["message_uid"]
    restored = db.get_messages_as_conversation(SID, repair_alternation=False)[-1:]
    try:
        instance._ingest_messages(restored)
        assert contents(instance) == ["repeat", "repeat"]
    finally:
        close(instance)
        db.close()


def test_rejected_compaction_old_history_and_new_uid_are_preserved(profile):
    instance = engine(profile)
    original = [message("first", "uid-a"), message("middle", "uid-b", "assistant"), message("last", "uid-c")]
    try:
        instance._ingest_messages(original)
        candidate = [original[0], original[-1], message("candidate delta", "uid-d", "assistant")]
        instance._ingest_messages(candidate)
        # The host rejected the candidate and supplies its old history again.
        instance._ingest_messages(copy.deepcopy(original) + [message("last", "uid-new")])
        assert contents(instance) == ["first", "middle", "last", "candidate delta", "last"]
    finally:
        close(instance)


@pytest.mark.parametrize("restart", [False, True], ids=["warm", "cold"])
def test_actual_compress_rejected_by_host_preserves_original_and_new_core_uids(profile, monkeypatch, restart):
    core = pytest.importorskip("hermes_state")
    from hermes_lcm import engine as engine_module

    monkeypatch.setattr(
        engine_module, "summarize_with_escalation",
        lambda **_kwargs: ("Earlier turns summarized.\nExpand for details about: earlier turns", 1),
    )
    config = LCMConfig(
        database_path=str(profile / "lcm.db"), fresh_tail_count=2,
        leaf_chunk_tokens=1, incremental_max_depth=0,
    )
    db = core.SessionDB(db_path=profile / "state.db")
    db.create_session(SID, "cli", model="test/model")
    db.append_messages_batch(SID, [
        {"role": "user" if index % 2 == 0 else "assistant",
         "content": f"Original turn {index}: " + "history detail " * 80}
        for index in range(6)
    ])
    original = db.get_messages_as_conversation(SID, repair_alternation=False)
    original_snapshot = copy.deepcopy(original)
    instance = LCMEngine(config=config, hermes_home=str(profile))
    instance.on_session_start(SID, hermes_home=str(profile), context_length=200_000)
    try:
        instance._ingest_messages(original)
        assert instance._ingest_cursor == 6
        candidate = instance.compress(original)
        assert original == original_snapshot
        assert instance._last_compression_status == "compacted"
        assert len(candidate) == 3 < len(original)
        assert sum(count_message_tokens(row) for row in candidate) < sum(count_message_tokens(row) for row in original)
        assert instance._ingest_cursor == len(candidate)
        assert contents(instance) == [row["content"] for row in original]

        # Simulate a host declining the proposal (or failing its archive): Core
        # retains the original history rather than committing the short list.
        assert db.get_messages_as_conversation(SID, repair_alternation=False) == original_snapshot
        db.append_messages_batch(SID, [
            {"role": row["role"], "content": row["content"]} for row in original[-2:]
        ])
        replay = db.get_messages_as_conversation(SID, repair_alternation=False)
        assert len(replay) == 8
        assert {row["message_uid"] for row in replay[-2:]}.isdisjoint(
            row["message_uid"] for row in original
        )
        if restart:
            close(instance)
            instance = LCMEngine(config=config, hermes_home=str(profile))
            instance.on_session_start(SID, hermes_home=str(profile), context_length=200_000)
        replay_snapshot = copy.deepcopy(replay)
        instance._ingest_messages(replay)
        assert replay == replay_snapshot
        # With a lowered cursor alone, the old suffix would be appended too.
        assert instance._store.get_session_count(SID) == 8
        assert contents(instance) == [row["content"] for row in replay]
        instance._ingest_messages(copy.deepcopy(replay))
        assert instance._store.get_session_count(SID) == 8
    finally:
        close(instance)
        db.close()


@pytest.mark.parametrize("identity", [
    {"message_uid": ""}, {"message_uid": 7},
    {"message_uid": "uid-a", "_absorbed_message_uids": "uid-b"},
    {"message_uid": "uid-a", "_absorbed_message_uids": ["uid-b", "uid-b"]},
    {"message_uid": "uid-a", "_tool_call_uids": {"call": []}},
])
def test_invalid_identity_never_certifies_matching_content(profile, identity):
    before = engine(profile)
    incoming = {"role": "user", "content": "repeat", **identity}
    before._ingest_messages([incoming])
    close(before)
    after = engine(profile)
    try:
        after._ingest_messages([copy.deepcopy(incoming)])
        assert contents(after) == ["repeat", "repeat"]
    finally:
        close(after)


@pytest.mark.parametrize("aliased", [False, True])
def test_duplicate_uid_proof_is_consumed_once(profile, aliased):
    before = engine(profile)
    original = message("repeat", "uid-a")
    before._ingest_messages([original])
    close(before)
    after = engine(profile)
    try:
        repeated = original if aliased else copy.deepcopy(original)
        after._ingest_messages([original, repeated])
        assert contents(after) == ["repeat", "repeat"]
    finally:
        close(after)


def test_uid_scope_never_crosses_session(profile):
    before = engine(profile)
    original = message("same", "uid-a")
    before._ingest_messages([original])
    close(before)
    after = engine(profile, session="other-session")
    try:
        after._ingest_messages([original])
        assert contents(after, session="other-session") == ["same"]
    finally:
        close(after)


def test_actual_core_tool_occurrence_and_wire_fields(profile):
    core = pytest.importorskip("hermes_state")
    metadata = pytest.importorskip("agent.message_metadata")
    turn_context = pytest.importorskip("agent.turn_context")
    db = core.SessionDB(db_path=profile / "state.db")
    db.create_session(SID, "cli", model="test/model")
    call = {"id": "reused", "type": "function", "function": {"name": "probe", "arguments": "{}"}}
    live = [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "content": "same result", "tool_call_id": "reused"},
        {"role": "user", "content": "second request"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "content": "same result", "tool_call_id": "reused"},
    ]
    db.append_messages_batch(SID, live)
    restored = db.get_messages_as_conversation(SID, repair_alternation=False)
    assert restored[2]["_tool_call_uid"] != restored[5]["_tool_call_uid"]
    assert restored[1]["_tool_call_uids"]["reused"] == restored[2]["_tool_call_uid"]
    instance = engine(profile)
    try:
        instance._ingest_messages(restored)
        assembled = instance._assemble_context(None, restored, include_lcm_note=False)
        for row in assembled:
            original = next(msg for msg in restored if msg["message_uid"] == row["message_uid"])
            for key in ("message_uid", "_tool_call_uids", "_tool_call_uid"):
                assert row.get(key) == original.get(key)
            outgoing = metadata.without_persistence_fields(row)
            assert not (set(outgoing) & metadata.PERSISTENCE_ONLY_MESSAGE_FIELDS)
            assert "message_uid" in row
        agent = SimpleNamespace(
            _current_turn_timestamp=1780000000.0,
            _copy_reasoning_content_for_api=lambda _original, _wire: None,
            _should_sanitize_tool_calls=lambda: False,
            ephemeral_system_prompt="",
        )
        wire, _system = turn_context.build_api_messages(
            agent, assembled, current_turn_user_idx=None, ext_prefetch_cache=None,
            plugin_user_context=None, moa_config=None, active_system_prompt="",
        )
        assert all(not (set(row) & metadata.PERSISTENCE_ONLY_MESSAGE_FIELDS) for row in wire)
        assert all("message_uid" in row for row in assembled)
    finally:
        close(instance)
        db.close()


@pytest.mark.parametrize("changed_scope", ["profile", "core_db"])
def test_uid_proof_is_scoped_to_profile_and_core_database(profile, tmp_path, monkeypatch, changed_scope):
    before = engine(profile)
    original = message("same", "uid-a")
    before._ingest_messages([original])
    close(before)
    after = engine(profile)
    if changed_scope == "profile":
        other_home = tmp_path / "other-profile"
        other_home.mkdir(mode=0o700)
        after._hermes_home = str(other_home)
    else:
        monkeypatch.setattr(after, "_state_db_path", lambda: tmp_path / "other-state.db")
    try:
        after._ingest_messages([original])
        assert contents(after) == ["same", "same"]
    finally:
        close(after)


def test_source_content_presence_change_invalidates_proof(profile):
    before = engine(profile)
    original = message(None, "uid-a", "assistant", tool_calls=[{"id": "call", "function": {"name": "probe"}}])
    before._ingest_messages([original])
    before._store._conn.execute("UPDATE messages SET content='' WHERE session_id=?", (SID,))
    before._store._conn.commit()
    close(before)
    after = engine(profile)
    try:
        after._ingest_messages([original])
        assert after._store.get_session_count(SID) == 2
    finally:
        close(after)


def test_actual_core_merge_witness_keeps_new_absorbed_content(profile):
    core = pytest.importorskip("hermes_state")
    runtime = pytest.importorskip("agent.agent_runtime_helpers")
    db = core.SessionDB(db_path=profile / "state.db")
    db.create_session(SID, "cli", model="test/model")
    old = [{"role": "user", "content": "old request"}]
    db.append_messages_batch(SID, old)
    before = engine(profile)
    before._ingest_messages(db.get_messages_as_conversation(SID, repair_alternation=False))
    close(before)
    new = [{"role": "user", "content": "new request"}]
    db.append_messages_batch(SID, new)
    merged, _repairs = runtime._merge_consecutive_users(db.get_messages_as_conversation(SID, repair_alternation=False))
    assert merged[0]["message_uid"] == old[0]["message_uid"]
    assert merged[0]["_absorbed_message_uids"] == [new[0]["message_uid"]]
    db.replace_messages(SID, merged, active_only=True)
    db.close()
    reopened = core.SessionDB(db_path=profile / "state.db")
    composite = reopened.get_messages_as_conversation(SID, repair_alternation=False)
    once = engine(profile)
    once._ingest_messages(composite)
    assert contents(once) == ["old request", "old request\n\nnew request"]
    close(once)
    twice = engine(profile)
    try:
        twice._ingest_messages(composite)
        assert contents(twice) == ["old request", "old request\n\nnew request"]
    finally:
        close(twice)
        reopened.close()


def test_source_changed_during_raw_preparation_aborts_new_batch(profile, monkeypatch):
    from hermes_lcm import engine as engine_module

    instance = engine(profile)
    old = message("original", "uid-a")
    instance._ingest_messages([old])
    original_protect = engine_module.protect_messages_for_ingest

    def change_source(messages, **kwargs):
        with sqlite3.connect(profile / "lcm.db") as writer:
            writer.execute("UPDATE messages SET content='changed source' WHERE session_id=?", (SID,))
        return original_protect(messages, **kwargs)

    monkeypatch.setattr(engine_module, "protect_messages_for_ingest", change_source)
    try:
        with pytest.raises(ValueError, match="origins changed during ingest"):
            instance._ingest_messages([old, message("new", "uid-b")])
        assert contents(instance) == ["changed source"]
        assert not instance._store._conn.in_transaction
        monkeypatch.setattr(engine_module, "protect_messages_for_ingest", original_protect)
        instance._ingest_messages([old, message("new", "uid-b")])
        assert contents(instance) == ["changed source", "original", "new"]
    finally:
        close(instance)


def test_proof_only_callback_holds_sqlite_writer_reservation(profile):
    instance = engine(profile)
    try:
        writer = sqlite3.connect(profile / "lcm.db", timeout=0)
        visited = []

        def inspect_writer(conn, ids):
            assert conn.in_transaction and ids == []
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                writer.execute("INSERT INTO metadata(key,value) VALUES('other-writer','{}')")
            visited.append(True)

        instance._store._append_protected_batch(SID, [], before_commit=inspect_writer)
        assert visited == [True]
        assert not instance._store._conn.in_transaction
        writer.close()
    finally:
        close(instance)


def test_proof_only_ingest_revalidates_before_confirming_skip(profile, monkeypatch):
    instance = engine(profile)
    old = message("original", "uid-a")
    instance._ingest_messages([old])
    original_append = instance._store._append_protected_batch

    def change_source_then_append(*args, **kwargs):
        with sqlite3.connect(profile / "lcm.db") as writer:
            writer.execute("UPDATE messages SET content='changed source' WHERE session_id=?", (SID,))
        return original_append(*args, **kwargs)

    monkeypatch.setattr(instance._store, "_append_protected_batch", change_source_then_append)
    try:
        with pytest.raises(ValueError, match="origins changed during ingest"):
            instance._ingest_messages([old])
        assert contents(instance) == ["changed source"]
        assert not instance._store._conn.in_transaction
        monkeypatch.setattr(instance._store, "_append_protected_batch", original_append)
        instance._ingest_messages([old])
        assert contents(instance) == ["changed source", "original"]
    finally:
        close(instance)


def test_mixed_legacy_system_keeps_existing_warm_cursor_proof(profile):
    instance = engine(profile)
    mixed = [{"role": "system", "content": "legacy system prompt"}, message("request", "uid-a")]
    try:
        instance._ingest_messages(mixed)
        instance._ingest_messages(copy.deepcopy(mixed))
        assert contents(instance).count("request") == 1
        assert contents(instance).count("legacy system prompt") == 1
    finally:
        close(instance)


@pytest.mark.parametrize("changed", ["cold", "new_uid", "inserted_legacy"])
def test_mixed_legacy_without_complete_warm_prefix_is_preserved(profile, changed):
    instance = engine(profile)
    mixed = [{"role": "system", "content": "legacy system prompt"}, message("request", "uid-a")]
    instance._ingest_messages(mixed)
    incoming = copy.deepcopy(mixed)
    if changed == "cold":
        close(instance)
        instance = engine(profile)
    elif changed == "new_uid":
        incoming[-1]["message_uid"] = "uid-new"
    else:
        incoming.insert(1, {"role": "user", "content": "new legacy occurrence"})
    try:
        instance._ingest_messages(incoming)
        assert contents(instance).count("legacy system prompt") == 2
        assert contents(instance).count("request") == (2 if changed == "new_uid" else 1)
        if changed == "inserted_legacy":
            assert contents(instance).count("new legacy occurrence") == 1
    finally:
        close(instance)


def test_invalid_uid_never_borrows_warm_legacy_prefix_proof(profile):
    instance = engine(profile)
    invalid = {"role": "user", "content": "repeat", "message_uid": ""}
    try:
        instance._ingest_messages([invalid])
        instance._ingest_messages([copy.deepcopy(invalid)])
        assert contents(instance) == ["repeat", "repeat"]
    finally:
        close(instance)
