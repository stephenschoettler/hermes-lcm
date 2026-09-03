"""Contract tests for the V1 durable-message retrieval-reference adapter."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from hermes_lcm.db_bootstrap import (
    RETRIEVAL_REFERENCES_TABLE,
    get_database_uuid,
    rotate_database_uuid,
)
from hermes_lcm.message_references import (
    MESSAGE_REFERENCE_KIND,
    MessageReferenceAdapter,
    MessageReferenceResult,
    issue_message_reference,
    message_revision,
    message_revision_preimage,
    resolve_message_reference,
)
from hermes_lcm.retrieval_references import (
    ReferenceError,
    ReferenceErrorCode,
    revoke_reference,
)
from hermes_lcm.store import MessageStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Every adapter test gets a fresh home and explicit database path."""
    home = tmp_path / "hermes-home"
    db_path = home / "lcm.db"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("LCM_DATABASE_PATH", str(db_path))
    message_store = MessageStore(db_path, hermes_home=str(home))
    try:
        yield message_store
    finally:
        message_store.close()


def _allow_issue(context, kind, target, scope, revision):
    return (
        context == "issuer"
        and kind == MESSAGE_REFERENCE_KIND
        and target.keys() == {"store_id"}
        and isinstance(scope, dict)
        and revision["algorithm"] == "sha256"
    )


def _allow_target(context, store_id, scope):
    return (
        context == "issuer"
        and isinstance(store_id, int)
        and store_id > 0
        and scope == {"session_id": "session-a"}
    )


def _allow(context, envelope):
    return context == "reader" and envelope.kind == MESSAGE_REFERENCE_KIND


def _allow_scope(context, envelope, scope):
    return context == "reader" and scope == {"session_id": "session-a"}


def _seed(store: MessageStore, *, session_id: str = "session-a", source: str = "") -> int:
    store_id = store.append(
        session_id,
        {
            "role": "user",
            "content": "hello",
            "tool_call_id": None,
            "tool_name": None,
        },
        token_estimate=3,
        source=source,
        conversation_id="conversation-a",
    )
    store.connection.execute(
        "UPDATE messages SET timestamp = ?, tool_calls = ? WHERE store_id = ?",
        (100.0, '{"b": 2, "a": 1}', store_id),
    )
    store.connection.commit()
    return store_id


def _issue(store: MessageStore, store_id: int, *, expires_at=None):
    return issue_message_reference(
        store,
        store_id,
        {"session_id": "session-a"},
        authorization_context="issuer",
        authorize_target=_allow_target,
        authorize_issue=_allow_issue,
        issued_at=100.0,
        expires_at=expires_at,
    )


def _resolve(store: MessageStore, envelope, **kwargs):
    return resolve_message_reference(
        store,
        envelope,
        authorization_context="reader",
        authorize=_allow,
        authorize_scope=_allow_scope,
        **kwargs,
    )


def test_connection_bound_facade_works_with_a_raw_sqlite_connection(store):
    store_id = _seed(store)
    adapter = MessageReferenceAdapter(store.connection)
    envelope = adapter.issue(
        store_id,
        {"session_id": "session-a"},
        authorization_context="issuer",
        authorize_target=_allow_target,
        authorize_issue=_allow_issue,
        issued_at=100.0,
    )

    result = adapter.resolve(
        envelope,
        authorization_context="reader",
        authorize=_allow,
        authorize_scope=_allow_scope,
    )

    assert result.ok
    assert result.message is not None
    assert result.message["store_id"] == store_id


def test_round_trip_returns_normalized_message_and_exact_revision(store):
    store_id = _seed(store, source=" ")
    envelope = _issue(store, store_id)

    result = _resolve(store, envelope, expected_scope={"session_id": "session-a"})

    assert result.ok
    assert result.record is not None
    assert result.message is not None
    assert result.record.kind == "message"
    assert result.record.target == {"store_id": store_id}
    assert result.record.revision == message_revision(result.message)
    assert result.message["source"] == "unknown"
    assert result.message["conversation_id"] == "conversation-a"
    assert result.message["tool_calls"] == {"a": 1, "b": 2}
    assert set(result.message) == {
        "store_id",
        "session_id",
        "source",
        "conversation_id",
        "role",
        "content",
        "tool_call_id",
        "tool_calls",
        "tool_name",
        "timestamp",
        "token_estimate",
        "pinned",
    }


def test_registry_persists_only_digest_not_wire_token(store):
    store_id = _seed(store)
    envelope = _issue(store, store_id)
    digest = hashlib.sha256(envelope.token.encode("ascii")).hexdigest()
    row = store.connection.execute(
        f"SELECT token_digest, target_json, scope_json, revision_json FROM {RETRIEVAL_REFERENCES_TABLE} WHERE token_digest = ?",
        (digest,),
    ).fetchone()

    assert row is not None
    assert row[0] == digest
    assert envelope.token not in json.dumps(tuple(row))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("content", "rewritten"),
        ("session_id", "session-b"),
        ("source", "other-source"),
        ("conversation_id", "conversation-b"),
        ("role", "assistant"),
        ("tool_call_id", "call-b"),
        ("tool_name", "other-tool"),
        ("timestamp", 101.0),
        ("tool_calls", '[{"id":"call-b"}]'),
    ],
)
def test_semantic_message_mutations_stale_the_reference(store, column, value):
    store_id = _seed(store)
    envelope = _issue(store, store_id)
    store.connection.execute(
        f"UPDATE messages SET {column} = ? WHERE store_id = ?",
        (value, store_id),
    )
    store.connection.commit()

    result = _resolve(store, envelope)

    assert result.to_dict() == {
        "error_code": ReferenceErrorCode.REFERENCE_STALE.value,
        "error": "message reference is stale",
    }


def test_externalized_tool_gc_rewrite_is_stale_but_operational_changes_are_not(store):
    store_id = _seed(store)
    store.connection.execute(
        "UPDATE messages SET role = 'tool', tool_call_id = 'call-a' WHERE store_id = ?",
        (store_id,),
    )
    store.connection.commit()
    envelope = _issue(store, store_id)
    store.pin(store_id)
    store.connection.execute(
        "UPDATE messages SET token_estimate = ?, pinned = 1 WHERE store_id = ?",
        (999, store_id),
    )
    store.connection.commit()
    assert _resolve(store, envelope).ok

    store.unpin(store_id)
    assert store.gc_externalized_tool_result(store_id, "[externalized output gc]")
    stale = _resolve(store, envelope)

    assert stale.error_code == ReferenceErrorCode.REFERENCE_STALE.value


def test_malformed_message_timestamp_fails_closed_as_stale(store):
    store_id = _seed(store)
    envelope = _issue(store, store_id)
    store.connection.execute(
        "UPDATE messages SET timestamp = ? WHERE store_id = ?",
        ("not-a-finite-timestamp", store_id),
    )
    store.connection.commit()

    result = _resolve(store, envelope)

    assert result.error_code == ReferenceErrorCode.REFERENCE_STALE.value
    assert "timestamp" in (result.error or "")


def test_deleted_message_is_not_found_after_authorization(store):
    store_id = _seed(store)
    envelope = _issue(store, store_id)
    store.connection.execute("DELETE FROM messages WHERE store_id = ?", (store_id,))
    store.connection.commit()

    result = _resolve(store, envelope)

    assert result.error_code == ReferenceErrorCode.REFERENCE_NOT_FOUND.value


def test_denial_runs_no_registry_or_message_select_and_discloses_nothing(store):
    store_id = _seed(store)
    envelope = _issue(store, store_id)
    statements: list[str] = []
    store.connection.set_trace_callback(statements.append)

    denied = resolve_message_reference(
        store,
        envelope,
        authorization_context="reader",
        authorize=lambda _context, _envelope: False,
        authorize_scope=_allow_scope,
    )

    assert denied.to_dict() == {
        "error_code": ReferenceErrorCode.REFERENCE_FORBIDDEN.value,
        "error": "reference use is not authorized by the host",
    }
    assert "target" not in denied.to_dict()
    assert "revision" not in denied.to_dict()
    assert not any("FROM messages" in statement.upper() for statement in statements)
    assert not any(
        f"FROM {RETRIEVAL_REFERENCES_TABLE}" in statement.upper()
        for statement in statements
    )


def test_stored_scope_denial_runs_no_message_select(store):
    store_id = _seed(store)
    envelope = _issue(store, store_id)
    statements: list[str] = []
    store.connection.set_trace_callback(statements.append)

    denied = resolve_message_reference(
        store,
        envelope,
        authorization_context="reader",
        authorize=_allow,
        authorize_scope=lambda _context, _envelope, _scope: False,
    )

    assert denied.error_code == ReferenceErrorCode.REFERENCE_FORBIDDEN.value
    assert not any("FROM messages" in statement.upper() for statement in statements)


def test_wrong_kind_malformed_binding_scope_and_lifecycle_fail_closed(store):
    store_id = _seed(store)
    envelope = _issue(store, store_id)

    wrong_kind = resolve_message_reference(
        store,
        replace(envelope, kind="summary_node"),
        authorization_context="reader",
        authorize=lambda context, _envelope: context == "reader",
        authorize_scope=_allow_scope,
    )
    assert wrong_kind.error_code == ReferenceErrorCode.REFERENCE_KIND_MISMATCH.value

    from hermes_lcm.retrieval_references import reference_token_digest

    digest = reference_token_digest(envelope.token)
    store.connection.execute(
        f"UPDATE {RETRIEVAL_REFERENCES_TABLE} SET target_json = ? WHERE token_digest = ?",
        ('{"store_id":0}', digest),
    )
    store.connection.commit()
    assert _resolve(store, envelope).error_code == ReferenceErrorCode.REFERENCE_STALE.value

    store.connection.execute(
        f"UPDATE {RETRIEVAL_REFERENCES_TABLE} SET target_json = ?, revision_json = ? WHERE token_digest = ?",
        (json.dumps({"store_id": store_id}), json.dumps({"algorithm": "sha256", "semantic_digest": "BAD"}), digest),
    )
    store.connection.commit()
    assert _resolve(store, envelope).error_code == ReferenceErrorCode.REFERENCE_STALE.value

    expired = _issue(store, store_id, expires_at=101.0)
    assert _resolve(store, expired, now=101.0).error_code == ReferenceErrorCode.REFERENCE_STALE.value
    assert _resolve(store, envelope, expected_scope={"session_id": "session-b"}).error_code == ReferenceErrorCode.REFERENCE_SCOPE_MISMATCH.value

    clone = _issue(store, store_id)
    old_uuid = get_database_uuid(store.connection)
    rotate_database_uuid(
        store.connection,
        authorization_context="admin",
        authorize_rotation=lambda context: context == "admin",
        revoked_at=200.0,
    )
    assert get_database_uuid(store.connection) != old_uuid
    assert _resolve(store, clone).error_code == ReferenceErrorCode.REFERENCE_DATABASE_MISMATCH.value


def test_revocation_uses_foundation_taxonomy(store):
    store_id = _seed(store)
    envelope = _issue(store, store_id)
    revoked = revoke_reference(
        store.connection,
        envelope,
        authorization_context="reader",
        authorize=_allow,
        authorize_scope=_allow_scope,
        revoked_at=110.0,
    )

    assert revoked.ok
    assert _resolve(store, envelope).error_code == ReferenceErrorCode.REFERENCE_NOT_FOUND.value


def test_issue_rejects_non_positive_or_boolean_store_ids_without_a_row_read(store):
    statements: list[str] = []
    store.connection.set_trace_callback(statements.append)

    with pytest.raises(ReferenceError) as zero:
        issue_message_reference(
            store,
            0,
            {"session_id": "session-a"},
            authorization_context="issuer",
            authorize_issue=_allow_issue,
        )
    with pytest.raises(ReferenceError) as boolean:
        issue_message_reference(
            store,
            True,
            {"session_id": "session-a"},
            authorization_context="issuer",
            authorize_issue=_allow_issue,
        )

    assert zero.value.code == ReferenceErrorCode.INVALID_REQUEST
    assert boolean.value.code == ReferenceErrorCode.INVALID_REQUEST
    assert not any("FROM messages" in statement.upper() for statement in statements)


def test_issue_missing_message_is_reference_not_found(store):
    with pytest.raises(ReferenceError) as error:
        _issue(store, 999999)

    assert error.value.code == ReferenceErrorCode.REFERENCE_NOT_FOUND
    assert store.connection.in_transaction is False


@pytest.mark.parametrize("target_exists", [True, False])
def test_issue_target_preauthorization_hides_message_existence(store, target_exists):
    store_id = _seed(store) if target_exists else 999999
    statements: list[str] = []
    calls: list[tuple[object, int, object]] = []
    store.connection.set_trace_callback(statements.append)

    def deny_target(context, candidate_store_id, scope):
        calls.append((context, candidate_store_id, scope))
        return False

    with pytest.raises(ReferenceError) as denied:
        issue_message_reference(
            store,
            store_id,
            {"session_id": "session-a"},
            authorization_context="issuer",
            authorize_target=deny_target,
            authorize_issue=_allow_issue,
        )

    assert denied.value.code == ReferenceErrorCode.REFERENCE_FORBIDDEN
    assert calls == [("issuer", store_id, {"session_id": "session-a"})]
    assert not any("BEGIN" in statement.upper() for statement in statements)
    assert not any("FROM MESSAGES" in statement.upper() for statement in statements)
    assert store.connection.in_transaction is False


def test_issue_missing_context_hides_message_existence(store):
    existing_id = _seed(store)
    missing_id = 999999
    calls: list[tuple[object, int, object]] = []

    def allow_requested_target(context, candidate_store_id, scope):
        calls.append((context, candidate_store_id, scope))
        return True

    outcomes = []
    for candidate_store_id in (existing_id, missing_id):
        with pytest.raises(ReferenceError) as denied:
            issue_message_reference(
                store,
                candidate_store_id,
                {"session_id": "session-a"},
                authorize_target=allow_requested_target,
                authorize_issue=_allow_issue,
            )
        outcomes.append(denied.value.as_dict())

    assert outcomes == [
        {
            "error_code": ReferenceErrorCode.REFERENCE_FORBIDDEN.value,
            "error": "message reference issuance is not authorized by the host",
        }
    ] * 2
    assert calls == []


def test_issue_requires_target_preauthorization(store):
    store_id = _seed(store)
    statements: list[str] = []
    store.connection.set_trace_callback(statements.append)

    with pytest.raises(ReferenceError) as denied:
        issue_message_reference(
            store,
            store_id,
            {"session_id": "session-a"},
            authorization_context="issuer",
            authorize_issue=_allow_issue,
        )

    assert denied.value.code == ReferenceErrorCode.REFERENCE_FORBIDDEN
    assert not statements


def test_caller_transaction_rollback_removes_issued_row(store):
    store_id = _seed(store)
    store.connection.execute("BEGIN")
    _issue(store, store_id)
    assert store.connection.in_transaction
    store.connection.rollback()

    assert store.connection.execute(
        f"SELECT COUNT(*) FROM {RETRIEVAL_REFERENCES_TABLE}"
    ).fetchone() == (0,)


def test_callback_failures_roll_back_owned_issue_and_keep_caller_transaction(store):
    store_id = _seed(store)
    with pytest.raises(ReferenceError) as denied:
        issue_message_reference(
            store,
            store_id,
            {"session_id": "session-a"},
            authorization_context="issuer",
            authorize_target=_allow_target,
            authorize_issue=lambda *_args: (_ for _ in ()).throw(RuntimeError("policy")),
        )
    assert denied.value.code == ReferenceErrorCode.REFERENCE_FORBIDDEN
    assert store.connection.in_transaction is False

    store.connection.execute("BEGIN")
    with pytest.raises(ReferenceError):
        issue_message_reference(
            store,
            store_id,
            {"session_id": "session-a"},
            authorization_context="issuer",
            authorize_target=_allow_target,
            authorize_issue=lambda *_args: False,
        )
    assert store.connection.in_transaction is True
    store.connection.rollback()


def test_legacy_blank_and_semantic_tool_call_normalization_is_deterministic(store):
    store_id = _seed(store)
    store.connection.execute(
        "UPDATE messages SET source = ?, conversation_id = ?, tool_calls = ? WHERE store_id = ?",
        (" \t\n", " \n", " false ", store_id),
    )
    store.connection.commit()
    row = store.connection.execute(
        f"SELECT {_MESSAGE_COLUMNS_FOR_TEST} FROM messages WHERE store_id = ?",
        (store_id,),
    ).fetchone()
    # The preimage uses the same normalized values for legacy SQL rows and
    # already-normalized adapter input, including a valid falsy JSON value.
    from hermes_lcm.message_references import _row_mapping

    message = _row_mapping(row)
    assert message_revision_preimage(message)["source"] == "unknown"
    assert message_revision_preimage(message)["conversation_id"] == ""
    assert message_revision_preimage(message)["tool_calls"] is False
    assert message_revision(message) == message_revision(
        {
            **message,
            "source": None,
            "conversation_id": None,
            "tool_calls": False,
        }
    )


def test_invalid_tool_call_json_is_hashed_as_the_stored_string(store):
    store_id = _seed(store)
    store.connection.execute(
        "UPDATE messages SET tool_calls = ? WHERE store_id = ?",
        ("not-json", store_id),
    )
    store.connection.commit()
    row = store.connection.execute(
        f"SELECT {_MESSAGE_COLUMNS_FOR_TEST} FROM messages WHERE store_id = ?",
        (store_id,),
    ).fetchone()
    from hermes_lcm.message_references import _row_mapping

    message = _row_mapping(row)
    assert message_revision_preimage(message)["tool_calls"] == "not-json"


def test_wal_race_keeps_one_resolve_snapshot_then_next_call_is_stale(store):
    store_id = _seed(store)
    envelope = _issue(store, store_id)
    db_path = Path(store.db_path)
    other = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
    other.execute("PRAGMA journal_mode=WAL")
    other.execute("PRAGMA busy_timeout=5000")
    callback_entered = threading.Event()
    release_callback = threading.Event()
    results: list[MessageReferenceResult] = []

    def pause_scope(context, parsed, scope):
        assert context == "reader"
        callback_entered.set()
        assert release_callback.wait(5.0)
        return scope == {"session_id": "session-a"}

    def resolve_in_thread():
        results.append(
            resolve_message_reference(
                store.connection,
                envelope,
                authorization_context="reader",
                authorize=_allow,
                authorize_scope=pause_scope,
            )
        )

    worker = threading.Thread(target=resolve_in_thread)
    worker.start()
    assert callback_entered.wait(5.0)
    other.execute("UPDATE messages SET content = ? WHERE store_id = ?", ("rewritten", store_id))
    other.commit()
    release_callback.set()
    worker.join(5.0)
    try:
        assert len(results) == 1
        assert results[0].ok
        assert results[0].message is not None
        assert results[0].message["content"] == "hello"
        assert _resolve(store, envelope).error_code == ReferenceErrorCode.REFERENCE_STALE.value
    finally:
        other.close()


def test_resolve_snapshot_holds_message_store_write_lock(store):
    store_id = _seed(store)
    envelope = _issue(store, store_id)
    callback_entered = threading.Event()
    release_callback = threading.Event()
    writer_finished = threading.Event()
    results: list[MessageReferenceResult] = []

    def pause_scope(context, parsed, scope):
        assert context == "reader"
        callback_entered.set()
        assert release_callback.wait(5.0)
        return scope == {"session_id": "session-a"}

    def resolve_in_thread():
        results.append(
            resolve_message_reference(
                store,
                envelope,
                authorization_context="reader",
                authorize=_allow,
                authorize_scope=pause_scope,
            )
        )

    def write_in_thread():
        store.append(
            "session-a",
            {"role": "user", "content": "later"},
            source="test",
        )
        writer_finished.set()

    resolver = threading.Thread(target=resolve_in_thread)
    resolver.start()
    assert callback_entered.wait(5.0)
    writer = threading.Thread(target=write_in_thread)
    writer.start()
    assert not writer_finished.wait(0.1)
    release_callback.set()
    resolver.join(5.0)
    writer.join(5.0)

    assert len(results) == 1
    assert results[0].ok
    assert writer_finished.is_set()


_MESSAGE_COLUMNS_FOR_TEST = (
    "store_id, session_id, source, conversation_id, role, content, "
    "tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned"
)
