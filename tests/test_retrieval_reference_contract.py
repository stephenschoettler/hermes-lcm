"""Contract tests for the frozen #476 V1 retrieval-reference foundation."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.db_bootstrap import (
    DATABASE_UUID_MIGRATION,
    RETRIEVAL_REFERENCES_MIGRATION,
    RETRIEVAL_REFERENCES_TABLE,
    SCHEMA_VERSION,
    DatabaseIdentityError,
    RetrievalReferenceSchemaError,
    SchemaVersionTooNewError,
    VERSION_MISMATCH_GENUINELY_NEWER,
    classify_version_mismatch,
    ensure_retrieval_reference_migrations,
    get_database_uuid,
    get_schema_version,
    rotate_database_uuid,
    run_versioned_migrations,
    verify_retrieval_references_schema,
)
from hermes_lcm.retrieval_references import (
    ReferenceError,
    ReferenceErrorCode,
    canonical_json,
    decode_reference_envelope,
    issue_reference,
    reference_token_digest,
    resolve_reference,
    revoke_reference,
    rotate_explicit_clone,
    serialize_reference_envelope,
)


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    run_versioned_migrations(conn)
    conn.commit()
    return conn


def _migration_markers(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT step_name FROM lcm_migration_state WHERE step_name IN (?, ?)",
            (DATABASE_UUID_MIGRATION, RETRIEVAL_REFERENCES_MIGRATION),
        ).fetchall()
    }


def _issue(conn, *, expires_at=None, kind="message", scope=None):
    return issue_reference(
        conn,
        kind,
        {"store_id": 7, "source": "test"},
        {"session_id": "session-a"} if scope is None else scope,
        {"content_digest": "rev-1"},
        authorization_context={"trusted": True},
        authorize_issue=_allow_issue,
        consistency="live",
        issued_at=100.0,
        expires_at=expires_at,
    )


def _allow_issue(context, kind, target, scope, revision):
    return (
        context == {"trusted": True}
        and isinstance(kind, str)
        and isinstance(target, dict)
        and isinstance(scope, dict)
        and isinstance(revision, dict)
    )


def _allow_rotation(context):
    return context == {"admin": True}


def _allow(context, envelope):
    assert context == {"trusted": True}
    assert envelope.kind == "message"
    return True


def _allow_scope(context, envelope, scope):
    assert context == {"trusted": True}
    assert envelope.kind == "message"
    return scope == {"session_id": "session-a"}


def test_fresh_and_populated_pre_v1_initialization_preserves_data(tmp_path):
    fresh = _open(tmp_path / "fresh.db")
    try:
        assert get_schema_version(fresh) == SCHEMA_VERSION
        assert get_database_uuid(fresh)
        assert verify_retrieval_references_schema(fresh) == []
    finally:
        fresh.close()

    populated_path = tmp_path / "populated.db"
    old = sqlite3.connect(populated_path)
    old.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    old.execute("INSERT INTO metadata(key, value) VALUES('schema_version', '5')")
    old.execute("CREATE TABLE preserved(id INTEGER PRIMARY KEY, value TEXT)")
    old.execute("INSERT INTO preserved(value) VALUES('before-v1')")
    old.commit()
    old.close()

    populated = _open(populated_path)
    try:
        assert populated.execute("SELECT value FROM preserved").fetchone() == ("before-v1",)
        markers = {
            row[0]
            for row in populated.execute(
                "SELECT step_name FROM lcm_migration_state WHERE step_name IN (?, ?)",
                (DATABASE_UUID_MIGRATION, RETRIEVAL_REFERENCES_MIGRATION),
            )
        }
        assert markers == {DATABASE_UUID_MIGRATION, RETRIEVAL_REFERENCES_MIGRATION}
    finally:
        populated.close()


def test_database_uuid_is_canonical_and_persists_across_restart(tmp_path):
    path = tmp_path / "restart.db"
    first = _open(path)
    first_uuid = get_database_uuid(first)
    first.close()

    second = _open(path)
    try:
        assert get_database_uuid(second) == first_uuid
    finally:
        second.close()


def test_completed_named_migrations_use_read_only_healthy_fast_path(tmp_path):
    conn = _open(tmp_path / "healthy.db")
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        run_versioned_migrations(conn)
    finally:
        conn.close()

    normalized = [" ".join(statement.upper().split()) for statement in statements]
    assert not any("BEGIN IMMEDIATE" in statement for statement in normalized)
    assert not any(
        "CREATE TABLE IF NOT EXISTS METADATA" in statement
        or "CREATE TABLE IF NOT EXISTS LCM_MIGRATION_STATE" in statement
        for statement in normalized
    )
    assert not any(
        RETRIEVAL_REFERENCES_TABLE.upper() in statement
        and ("CREATE TABLE" in statement or "CREATE INDEX" in statement)
        for statement in normalized
    )


def test_named_migration_validation_uses_one_snapshot_during_startup_race(
    tmp_path, monkeypatch
):
    path = tmp_path / "migration-race.db"
    seed = sqlite3.connect(path)
    seed.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE lcm_migration_state(
            step_name TEXT PRIMARY KEY,
            completed_at REAL NOT NULL
        );
        """
    )
    seed.commit()
    seed.close()

    loser = sqlite3.connect(path, timeout=30.0)
    winner = sqlite3.connect(path, timeout=30.0)
    db_bootstrap.configure_connection(loser)
    db_bootstrap.configure_connection(winner)
    triggered = False

    original_marker_exists = db_bootstrap._migration_marker_exists

    def interleave_marker(conn, step_name):
        nonlocal triggered
        result = original_marker_exists(conn, step_name)
        if not triggered and conn is loser and step_name == DATABASE_UUID_MIGRATION:
            triggered = True
            ensure_retrieval_reference_migrations(winner)
        return result

    monkeypatch.setattr(db_bootstrap, "_migration_marker_exists", interleave_marker)
    try:
        database_uuid = ensure_retrieval_reference_migrations(loser)
        assert database_uuid == get_database_uuid(winner)
    finally:
        loser.close()
        winner.close()

    assert triggered


def test_missing_or_corrupt_post_migration_uuid_fails_closed(tmp_path):
    path = tmp_path / "identity.db"
    conn = _open(path)
    conn.execute("DELETE FROM metadata WHERE key='database_uuid'")
    conn.commit()
    conn.close()

    missing = sqlite3.connect(path)
    with pytest.raises(DatabaseIdentityError):
        run_versioned_migrations(missing)
    missing.close()

    corrupt = sqlite3.connect(path)
    corrupt.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('database_uuid', 'not-a-uuid')")
    corrupt.commit()
    corrupt.close()

    reopened = sqlite3.connect(path)
    with pytest.raises(DatabaseIdentityError):
        run_versioned_migrations(reopened)
    reopened.close()


def test_existing_registry_state_without_identity_is_not_adopted(tmp_path):
    path = tmp_path / "orphaned-registry.db"
    conn = _open(path)
    conn.execute("DELETE FROM metadata WHERE key='database_uuid'")
    conn.execute(
        "DELETE FROM lcm_migration_state WHERE step_name=?",
        (DATABASE_UUID_MIGRATION,),
    )
    conn.commit()
    with pytest.raises(DatabaseIdentityError, match="unmarked"):
        run_versioned_migrations(conn)
    assert conn.execute(
        "SELECT value FROM metadata WHERE key='database_uuid'"
    ).fetchone() is None
    conn.close()


@pytest.mark.parametrize(
    ("remove_identity_marker", "remove_registry_marker", "drop_registry", "error"),
    [
        (True, True, False, DatabaseIdentityError),
        (True, True, True, DatabaseIdentityError),
        (False, True, False, RetrievalReferenceSchemaError),
    ],
)
def test_unmarked_authority_or_registry_state_is_never_adopted(
    tmp_path,
    remove_identity_marker,
    remove_registry_marker,
    drop_registry,
    error,
):
    conn = _open(tmp_path / "unmarked-state.db")
    if remove_identity_marker:
        conn.execute(
            "DELETE FROM lcm_migration_state WHERE step_name=?",
            (DATABASE_UUID_MIGRATION,),
        )
    if remove_registry_marker:
        conn.execute(
            "DELETE FROM lcm_migration_state WHERE step_name=?",
            (RETRIEVAL_REFERENCES_MIGRATION,),
        )
    if drop_registry:
        conn.execute(f"DROP TABLE {RETRIEVAL_REFERENCES_TABLE}")
    conn.commit()

    with pytest.raises(error, match="unmarked"):
        ensure_retrieval_reference_migrations(conn)
    assert _migration_markers(conn) == {
        marker
        for marker, removed in (
            (DATABASE_UUID_MIGRATION, remove_identity_marker),
            (RETRIEVAL_REFERENCES_MIGRATION, remove_registry_marker),
        )
        if not removed
    }
    conn.close()


def test_valid_identity_only_state_completes_registry_without_rotating_uuid(tmp_path):
    conn = _open(tmp_path / "identity-only.db")
    database_uuid = get_database_uuid(conn)
    conn.execute(f"DROP TABLE {RETRIEVAL_REFERENCES_TABLE}")
    conn.execute(
        "DELETE FROM lcm_migration_state WHERE step_name=?",
        (RETRIEVAL_REFERENCES_MIGRATION,),
    )
    conn.commit()

    assert ensure_retrieval_reference_migrations(conn) == database_uuid
    assert get_database_uuid(conn) == database_uuid
    assert _migration_markers(conn) == {
        DATABASE_UUID_MIGRATION,
        RETRIEVAL_REFERENCES_MIGRATION,
    }
    assert verify_retrieval_references_schema(conn) == []
    conn.close()


def test_public_operations_require_complete_registry_provenance(tmp_path):
    conn = _open(tmp_path / "unmarked-public-operations.db")
    database_uuid = get_database_uuid(conn)
    envelope = _issue(conn)
    conn.execute(
        "DELETE FROM lcm_migration_state WHERE step_name=?",
        (RETRIEVAL_REFERENCES_MIGRATION,),
    )
    conn.commit()

    with pytest.raises(RetrievalReferenceSchemaError, match="unmarked"):
        _issue(conn)
    resolved = resolve_reference(
        conn,
        envelope,
        authorization_context={"trusted": True},
        authorize=_allow,
        authorize_scope=_allow_scope,
    )
    assert resolved.to_dict() == {
        "error_code": ReferenceErrorCode.REFERENCE_INVALID.value,
        "error": "reference authority identity is unavailable",
    }
    revoked = revoke_reference(
        conn,
        envelope,
        authorization_context={"trusted": True},
        authorize=_allow,
        authorize_scope=_allow_scope,
    )
    assert revoked.to_dict() == resolved.to_dict()
    with pytest.raises(RetrievalReferenceSchemaError, match="unmarked"):
        rotate_database_uuid(
            conn,
            authorization_context={"admin": True},
            authorize_rotation=_allow_rotation,
            revoked_at=200.0,
        )

    assert get_database_uuid(conn) == database_uuid
    assert conn.execute(
        f"SELECT COUNT(*), MAX(revoked_at) FROM {RETRIEVAL_REFERENCES_TABLE}"
    ).fetchone() == (1, None)
    assert not conn.in_transaction
    conn.close()


def test_direct_named_helper_refuses_too_new_schema_before_ddl(tmp_path):
    conn = _open(tmp_path / "too-new-helper.db")
    conn.execute("UPDATE metadata SET value=? WHERE key='schema_version'", ("99",))
    conn.commit()
    before = conn.execute(
        "SELECT name, sql FROM sqlite_master ORDER BY name"
    ).fetchall()
    with pytest.raises(SchemaVersionTooNewError):
        ensure_retrieval_reference_migrations(conn)
    assert conn.execute(
        "SELECT name, sql FROM sqlite_master ORDER BY name"
    ).fetchall() == before
    conn.close()


def test_too_new_exact_unmarked_registry_is_genuinely_newer(tmp_path):
    conn = _open(tmp_path / "unmarked-too-new.db")
    conn.execute(
        "DELETE FROM lcm_migration_state WHERE step_name=?",
        (RETRIEVAL_REFERENCES_MIGRATION,),
    )
    conn.execute("UPDATE metadata SET value=? WHERE key='schema_version'", ("99",))
    conn.commit()
    assert classify_version_mismatch(conn) == VERSION_MISMATCH_GENUINELY_NEWER
    conn.close()


@pytest.mark.parametrize("damage", ["missing-marker", "missing-value"])
def test_too_new_classifier_requires_database_uuid_provenance(tmp_path, damage):
    conn = _open(tmp_path / f"too-new-identity-{damage}.db")
    conn.execute(f"DROP TABLE {RETRIEVAL_REFERENCES_TABLE}")
    conn.execute(
        "DELETE FROM lcm_migration_state WHERE step_name=?",
        (RETRIEVAL_REFERENCES_MIGRATION,),
    )
    if damage == "missing-marker":
        conn.execute(
            "DELETE FROM lcm_migration_state WHERE step_name=?",
            (DATABASE_UUID_MIGRATION,),
        )
    else:
        conn.execute("DELETE FROM metadata WHERE key='database_uuid'")
    conn.execute(
        "UPDATE metadata SET value=? WHERE key='schema_version'",
        (str(SCHEMA_VERSION + 94),),
    )
    conn.commit()
    assert classify_version_mismatch(conn) == VERSION_MISMATCH_GENUINELY_NEWER
    conn.close()


def test_completed_registry_marker_does_not_recreate_missing_table(tmp_path):
    path = tmp_path / "missing-registry.db"
    conn = _open(path)
    conn.execute(f"DROP TABLE {RETRIEVAL_REFERENCES_TABLE}")
    conn.commit()
    with pytest.raises(RetrievalReferenceSchemaError, match="completed.*damaged"):
        run_versioned_migrations(conn)
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (RETRIEVAL_REFERENCES_TABLE,),
    ).fetchone() is None
    conn.close()


def test_token_is_random_rh1_256_bit_and_registry_is_digest_only(tmp_path):
    conn = _open(tmp_path / "tokens.db")
    try:
        first = _issue(conn)
        second = _issue(conn)
        assert first.token.startswith("rh1_")
        assert len(first.token) == len("rh1_") + 64
        assert first.token != second.token
        digest = hashlib.sha256(first.token.encode("ascii")).hexdigest()
        row = conn.execute(
            f"SELECT token_digest, target_json, scope_json, revision_json FROM {RETRIEVAL_REFERENCES_TABLE} WHERE token_digest=?",
            (digest,),
        ).fetchone()
        assert row is not None
        assert row[0] == digest
        assert first.token not in json.dumps(row)
        token_column = {
            str(item[1]): (str(item[2]).upper(), int(item[3]), int(item[5]))
            for item in conn.execute(
                f"PRAGMA table_info({RETRIEVAL_REFERENCES_TABLE})"
            ).fetchall()
        }["token_digest"]
        assert token_column == ("TEXT", 1, 1)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"""
                INSERT INTO {RETRIEVAL_REFERENCES_TABLE}(
                    token_digest, kind, target_json, scope_json, revision_json,
                    consistency, issued_at
                ) VALUES (NULL, 'message', '{{}}', '{{}}', '{{}}', 'live', 100.0)
                """
            )
        conn.rollback()
    finally:
        conn.close()


def test_issue_and_clone_rotation_require_explicit_operation_authorization(tmp_path):
    conn = _open(tmp_path / "operation-authorization.db")
    original_uuid = get_database_uuid(conn)
    with pytest.raises(ReferenceError) as issue_error:
        issue_reference(
            conn,
            "message",
            {"store_id": 7},
            {"session_id": "session-a"},
            {"content_digest": "rev-1"},
        )
    assert issue_error.value.code == ReferenceErrorCode.REFERENCE_FORBIDDEN.value
    with pytest.raises(ReferenceError) as missing_context_error:
        issue_reference(
            conn,
            "message",
            {"store_id": 7},
            {"session_id": "session-a"},
            {"content_digest": "rev-1"},
            authorize_issue=lambda context, kind, target, scope, revision: True,
        )
    assert missing_context_error.value.code == ReferenceErrorCode.REFERENCE_FORBIDDEN.value

    observed_issue = []

    def deny_issue(context, kind, target, scope, revision):
        observed_issue.append((context, kind, target, scope, revision))
        return False

    with pytest.raises(ReferenceError) as denied_issue_error:
        issue_reference(
            conn,
            "message",
            {"z": 2, "a": 1},
            {"tenant": "tenant-a"},
            {"content_digest": "rev-1"},
            authorization_context={"trusted": True},
            authorize_issue=deny_issue,
        )
    assert denied_issue_error.value.code == ReferenceErrorCode.REFERENCE_FORBIDDEN.value
    assert list(observed_issue[0][2]) == ["a", "z"]
    assert observed_issue == [
        (
            {"trusted": True},
            "message",
            {"a": 1, "z": 2},
            {"tenant": "tenant-a"},
            {"content_digest": "rev-1"},
        )
    ]
    assert conn.execute(
        f"SELECT COUNT(*) FROM {RETRIEVAL_REFERENCES_TABLE}"
    ).fetchone() == (0,)

    with pytest.raises(PermissionError, match="not authorized"):
        rotate_database_uuid(conn)
    with pytest.raises(PermissionError, match="not authorized"):
        rotate_database_uuid(conn, authorize_rotation=lambda context: True)
    with pytest.raises(PermissionError, match="not authorized"):
        rotate_database_uuid(
            conn,
            authorization_context={"admin": False},
            authorize_rotation=_allow_rotation,
        )
    assert get_database_uuid(conn) == original_uuid
    conn.close()


def test_issue_revoke_and_clone_validate_identity_inside_write_transaction(tmp_path):
    conn = _open(tmp_path / "transaction-order.db")

    issue_statements = []
    conn.set_trace_callback(issue_statements.append)
    envelope = _issue(conn)
    issue_sql = [statement.upper() for statement in issue_statements]
    issue_begin = next(i for i, sql in enumerate(issue_sql) if "BEGIN IMMEDIATE" in sql)
    issue_identity = next(
        i
        for i, sql in enumerate(issue_sql)
        if "SELECT VALUE FROM METADATA" in sql and "DATABASE_UUID" in sql
    )
    issue_insert = next(
        i
        for i, sql in enumerate(issue_sql)
        if f"INSERT INTO {RETRIEVAL_REFERENCES_TABLE}".upper() in sql
    )
    assert issue_begin < issue_identity < issue_insert

    revoke_statements = []
    conn.set_trace_callback(revoke_statements.append)
    assert revoke_reference(
        conn,
        envelope,
        authorization_context={"trusted": True},
        authorize=_allow,
        authorize_scope=_allow_scope,
        revoked_at=120.0,
    ).ok
    revoke_sql = [statement.upper() for statement in revoke_statements]
    revoke_begin = next(i for i, sql in enumerate(revoke_sql) if "BEGIN IMMEDIATE" in sql)
    revoke_identity = next(
        i
        for i, sql in enumerate(revoke_sql)
        if "SELECT VALUE FROM METADATA" in sql and "DATABASE_UUID" in sql
    )
    revoke_update = next(
        i
        for i, sql in enumerate(revoke_sql)
        if f"UPDATE {RETRIEVAL_REFERENCES_TABLE}".upper() in sql
    )
    assert revoke_begin < revoke_identity < revoke_update

    clone_statements = []
    conn.set_trace_callback(clone_statements.append)
    rotate_explicit_clone(
        conn,
        authorization_context={"admin": True},
        authorize_rotation=_allow_rotation,
        revoked_at=130.0,
    )
    clone_sql = [statement.upper() for statement in clone_statements]
    clone_begin = next(i for i, sql in enumerate(clone_sql) if "BEGIN IMMEDIATE" in sql)
    clone_identity = next(
        i
        for i, sql in enumerate(clone_sql)
        if "SELECT VALUE FROM METADATA" in sql and "DATABASE_UUID" in sql
    )
    clone_update = next(
        i
        for i, sql in enumerate(clone_sql)
        if "UPDATE METADATA SET VALUE" in sql
    )
    assert clone_begin < clone_identity < clone_update
    conn.close()


def test_canonical_json_and_wire_envelope_round_trip(tmp_path):
    conn = _open(tmp_path / "codec.db")
    try:
        envelope = _issue(conn)
        assert canonical_json({"b": 2, "a": [1, True]}) == '{"a":[1,true],"b":2}'
        assert serialize_reference_envelope(envelope) == (
            '{"database_uuid":"%s","kind":"message","token":"%s","v":1}'
            % (envelope.database_uuid, envelope.token)
        )
        assert decode_reference_envelope(envelope.to_json()) == envelope
    finally:
        conn.close()


def test_malformed_and_unsupported_envelopes_are_typed():
    malformed = resolve_reference(None, {"v": 1}, authorization_context={"trusted": True}, authorize=_allow)
    assert malformed.to_dict() == {
        "error_code": ReferenceErrorCode.REFERENCE_INVALID.value,
        "error": "reference envelope has an invalid shape",
    }
    unsupported = resolve_reference(
        None,
        {"v": 2, "kind": "message", "database_uuid": "00000000-0000-4000-8000-000000000000", "token": "rh1_" + "0" * 64},
        authorization_context={"trusted": True},
        authorize=_allow,
    )
    assert unsupported.error_code == ReferenceErrorCode.REFERENCE_UNSUPPORTED_VERSION.value
    assert unsupported.error

    duplicate = resolve_reference(
        None,
        '{"v":1,"kind":"message","database_uuid":"00000000-0000-4000-8000-000000000000",'
        '"token":"rh1_0000000000000000000000000000000000000000000000000000000000000000",'
        '"token":"rh1_ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}',
        authorization_context={"trusted": True},
        authorize=_allow,
    )
    assert duplicate.error_code == ReferenceErrorCode.REFERENCE_INVALID.value
    assert duplicate.error == "reference envelope contains a duplicate JSON key"


def test_database_replacement_mismatch_precedes_registry_disclosure(tmp_path):
    first = _open(tmp_path / "first.db")
    envelope = _issue(first)
    first.close()

    replacement = _open(tmp_path / "replacement.db")
    statements = []
    replacement.set_trace_callback(statements.append)
    try:
        result = resolve_reference(
            replacement,
            envelope,
            authorization_context={"trusted": True},
            authorize=lambda context, parsed: True,
        )
        assert result.error_code == ReferenceErrorCode.REFERENCE_DATABASE_MISMATCH.value
        assert not any(
            f"FROM {RETRIEVAL_REFERENCES_TABLE}" in statement.upper()
            for statement in statements
        )
    finally:
        replacement.close()


def test_kind_and_scope_mismatch_do_not_return_binding_details(tmp_path):
    conn = _open(tmp_path / "scope.db")
    envelope = _issue(conn, scope={"session_id": "session-a"})
    try:
        wrong_kind = resolve_reference(
            conn,
            envelope,
            authorization_context={"trusted": True},
            authorize=_allow,
            authorize_scope=_allow_scope,
            expected_kind="summary_node",
        )
        wrong_scope = resolve_reference(
            conn,
            envelope,
            authorization_context={"trusted": True},
            authorize=_allow,
            authorize_scope=_allow_scope,
            expected_scope={"session_id": "session-b"},
        )
        assert wrong_kind.error_code == ReferenceErrorCode.REFERENCE_KIND_MISMATCH.value
        assert wrong_scope.error_code == ReferenceErrorCode.REFERENCE_SCOPE_MISMATCH.value
        assert "target" not in wrong_kind.to_dict()
        assert "scope" not in wrong_scope.to_dict()
    finally:
        conn.close()


def test_expiry_and_revocation_have_stable_errors(tmp_path):
    conn = _open(tmp_path / "lifecycle.db")
    try:
        expired = _issue(conn, expires_at=150.0)
        expired_result = resolve_reference(
            conn,
            expired,
            authorization_context={"trusted": True},
            authorize=_allow,
            authorize_scope=_allow_scope,
            now=150.0,
        )
        assert expired_result.error_code == ReferenceErrorCode.REFERENCE_STALE.value

        live = _issue(conn)
        assert revoke_reference(
            conn,
            live,
            authorization_context={"trusted": True},
            authorize=_allow,
            authorize_scope=_allow_scope,
            revoked_at=120.0,
        ).ok
        revoked = resolve_reference(
            conn,
            live,
            authorization_context={"trusted": True},
            authorize=_allow,
            authorize_scope=_allow_scope,
            now=120.0,
        )
        assert revoked.error_code == ReferenceErrorCode.REFERENCE_NOT_FOUND.value
    finally:
        conn.close()


def test_resolution_rechecks_lifecycle_after_mid_resolve_revocation(tmp_path):
    path = tmp_path / "mid-resolve-revocation.db"
    conn = _open(path)
    revoker = sqlite3.connect(path)
    envelope = _issue(conn)
    revocation = []
    statements = []
    conn.set_trace_callback(statements.append)

    def revoke_during_scope(context, parsed, scope):
        assert any(
            "SCOPE_JSON" in statement.upper() and "SELECT" in statement.upper()
            for statement in statements
        )
        assert not any("TARGET_JSON" in statement.upper() for statement in statements)
        result = revoke_reference(
            revoker,
            envelope,
            authorization_context={"trusted": True},
            authorize=_allow,
            authorize_scope=_allow_scope,
            revoked_at=120.0,
        )
        revocation.append(result.ok)
        return True

    try:
        result = resolve_reference(
            conn,
            envelope,
            authorization_context={"trusted": True},
            authorize=_allow,
            authorize_scope=revoke_during_scope,
            now=100.0,
        )
        assert revocation == [True]
        assert any("TARGET_JSON" in statement.upper() for statement in statements)
        assert result.ok is False
        assert result.error_code == ReferenceErrorCode.REFERENCE_NOT_FOUND.value
        assert result.record is None
    finally:
        revoker.close()
        conn.close()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("issued_at", "not-a-number"),
        ("issued_at", "NaN"),
        ("expires_at", "not-a-number"),
        ("expires_at", "NaN"),
        ("expires_at", 99.0),
    ],
)
def test_malformed_registry_timestamps_fail_closed_as_stale(
    tmp_path, column, value
):
    conn = _open(tmp_path / f"malformed-{column}-{value}.db")
    envelope = _issue(conn)
    digest = reference_token_digest(envelope.token)
    conn.execute(
        f"UPDATE {RETRIEVAL_REFERENCES_TABLE} SET {column}=? WHERE token_digest=?",
        (value, digest),
    )
    conn.commit()

    result = resolve_reference(
        conn,
        envelope,
        authorization_context={"trusted": True},
        authorize=_allow,
        authorize_scope=_allow_scope,
        now=100.0,
    )
    assert result.to_dict() == {
        "error_code": ReferenceErrorCode.REFERENCE_STALE.value,
        "error": "reference binding is no longer valid",
    }
    conn.close()


def test_authorization_callback_runs_before_registry_and_denial_is_indistinguishable(tmp_path):
    conn = _open(tmp_path / "auth.db")
    existing = _issue(conn)
    missing = existing.to_dict() | {"token": "rh1_" + "f" * 64}
    events = []
    statements = []
    conn.set_trace_callback(statements.append)

    def deny(context, envelope):
        events.append((context, envelope.kind))
        assert not any(f"FROM {RETRIEVAL_REFERENCES_TABLE}" in sql.upper() for sql in statements)
        return False

    try:
        denied_existing = resolve_reference(conn, existing, authorization_context="trusted", authorize=deny)
        denied_missing = resolve_reference(conn, missing, authorization_context="trusted", authorize=deny)
        assert denied_existing.to_dict() == denied_missing.to_dict()
        assert denied_existing.error_code == ReferenceErrorCode.REFERENCE_FORBIDDEN.value
        assert events == [("trusted", "message"), ("trusted", "message")]
    finally:
        conn.close()


def test_stored_scope_requires_host_authorization_before_binding_disclosure(tmp_path):
    conn = _open(tmp_path / "scope-authorization.db")
    envelope = _issue(conn, scope={"tenant": "tenant-b"})

    missing_scope_guard = resolve_reference(
        conn,
        envelope,
        authorization_context={"tenant": "tenant-a"},
        authorize=lambda context, parsed: bool(context),
    )
    assert missing_scope_guard.error_code == ReferenceErrorCode.REFERENCE_FORBIDDEN.value
    assert "target" not in missing_scope_guard.to_dict()
    assert "scope" not in missing_scope_guard.to_dict()
    assert "revision" not in missing_scope_guard.to_dict()

    observed_scopes = []

    def authorize_tenant(context, parsed, scope):
        observed_scopes.append(scope)
        return context.get("tenant") == scope.get("tenant")

    denied = resolve_reference(
        conn,
        envelope,
        authorization_context={"tenant": "tenant-a"},
        authorize=lambda context, parsed: True,
        authorize_scope=authorize_tenant,
    )
    assert denied.error_code == ReferenceErrorCode.REFERENCE_FORBIDDEN.value
    assert "target" not in denied.to_dict()
    assert "scope" not in denied.to_dict()
    assert "revision" not in denied.to_dict()

    allowed = resolve_reference(
        conn,
        envelope,
        authorization_context={"tenant": "tenant-b"},
        authorize=lambda context, parsed: True,
        authorize_scope=authorize_tenant,
    )
    assert allowed.ok
    assert allowed.record is not None
    assert allowed.record.target == {"source": "test", "store_id": 7}
    assert observed_scopes == [{"tenant": "tenant-b"}, {"tenant": "tenant-b"}]
    conn.close()


def test_scope_denial_normalizes_missing_kind_and_state_oracles(tmp_path):
    conn = _open(tmp_path / "scope-oracles.db")
    active = _issue(conn)
    expired = _issue(conn, expires_at=110.0)
    revoked = _issue(conn)
    assert revoke_reference(
        conn,
        revoked,
        authorization_context={"trusted": True},
        authorize=_allow,
        authorize_scope=_allow_scope,
        revoked_at=120.0,
    ).ok

    missing = active.to_dict()
    missing["token"] = "rh1_" + "0" * 64
    wrong_kind = active.to_dict()
    wrong_kind["kind"] = "summary_node"
    statements = []
    conn.set_trace_callback(statements.append)
    failures = [
        resolve_reference(
            conn,
            envelope,
            authorization_context={"trusted": True},
            authorize=lambda context, parsed: True,
            authorize_scope=lambda context, parsed, scope: False,
            now=150.0,
        ).to_dict()
        for envelope in (missing, wrong_kind, expired, revoked)
    ]
    assert failures == [
        {
            "error_code": ReferenceErrorCode.REFERENCE_FORBIDDEN.value,
            "error": "reference scope is not authorized by the host",
        }
    ] * 4
    assert not any(
        "TARGET_JSON" in statement.upper() or "REVISION_JSON" in statement.upper()
        for statement in statements
    )
    conn.close()


def test_revocation_denial_precedes_registry_lookup(tmp_path):
    conn = _open(tmp_path / "revoke-authorization.db")
    envelope = _issue(conn)
    statements = []
    conn.set_trace_callback(statements.append)

    denied = revoke_reference(
        conn,
        envelope,
        authorization_context={"trusted": False},
        authorize=lambda context, parsed: False,
        authorize_scope=lambda context, parsed, scope: True,
        revoked_at=120.0,
    )
    assert denied.error_code == ReferenceErrorCode.REFERENCE_FORBIDDEN.value
    assert not any(
        f"FROM {RETRIEVAL_REFERENCES_TABLE}" in statement.upper()
        for statement in statements
    )
    assert conn.execute(
        f"SELECT revoked_at FROM {RETRIEVAL_REFERENCES_TABLE}"
    ).fetchone() == (None,)
    conn.close()


def test_issue_failure_rolls_back_registry_insert(tmp_path):
    conn = _open(tmp_path / "rollback.db")
    try:
        conn.execute(
            f"""
            CREATE TRIGGER fail_retrieval_issue
            BEFORE INSERT ON {RETRIEVAL_REFERENCES_TABLE}
            BEGIN SELECT RAISE(ABORT, 'injected issue failure'); END
            """
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="injected issue failure"):
            _issue(conn)
        assert conn.execute(f"SELECT COUNT(*) FROM {RETRIEVAL_REFERENCES_TABLE}").fetchone() == (0,)
    finally:
        conn.close()


def test_explicit_clone_rotation_changes_uuid_and_revokes_copied_rows_atomically(tmp_path):
    path = tmp_path / "clone.db"
    conn = _open(path)
    envelope = _issue(conn)
    old_uuid = get_database_uuid(conn)
    new_uuid = rotate_explicit_clone(
        conn,
        authorization_context={"admin": True},
        authorize_rotation=_allow_rotation,
        revoked_at=200.0,
    )
    try:
        assert new_uuid != old_uuid
        assert get_database_uuid(conn) == new_uuid
        assert conn.execute(
            f"SELECT revoked_at FROM {RETRIEVAL_REFERENCES_TABLE}"
        ).fetchone() == (200.0,)
        result = resolve_reference(
            conn,
            envelope,
            authorization_context={"trusted": True},
            authorize=_allow,
        )
        assert result.error_code == ReferenceErrorCode.REFERENCE_DATABASE_MISMATCH.value
    finally:
        conn.close()

    rollback = _open(tmp_path / "clone-rollback.db")
    _issue(rollback)
    rollback_uuid = get_database_uuid(rollback)
    rollback.execute(
        """
        CREATE TRIGGER fail_clone_rotation
        BEFORE UPDATE ON metadata WHEN OLD.key='database_uuid'
        BEGIN SELECT RAISE(ABORT, 'injected clone failure'); END
        """
    )
    rollback.commit()
    with pytest.raises(sqlite3.IntegrityError, match="injected clone failure"):
        rotate_database_uuid(
            rollback,
            authorization_context={"admin": True},
            authorize_rotation=_allow_rotation,
            revoked_at=300.0,
        )
    assert get_database_uuid(rollback) == rollback_uuid
    assert rollback.execute(
        f"SELECT revoked_at FROM {RETRIEVAL_REFERENCES_TABLE}"
    ).fetchone() == (None,)
    rollback.close()


def test_registry_writes_respect_caller_transaction_rollback(tmp_path):
    conn = _open(tmp_path / "caller-transaction.db")
    original_uuid = get_database_uuid(conn)

    conn.execute("BEGIN")
    _issue(conn)
    assert conn.in_transaction
    conn.rollback()
    assert conn.execute(
        f"SELECT COUNT(*) FROM {RETRIEVAL_REFERENCES_TABLE}"
    ).fetchone() == (0,)

    durable = _issue(conn)
    conn.execute("BEGIN")
    revoked = revoke_reference(
        conn,
        durable,
        authorization_context={"trusted": True},
        authorize=_allow,
        authorize_scope=_allow_scope,
        revoked_at=310.0,
    )
    assert revoked.ok
    assert conn.in_transaction
    conn.rollback()
    assert conn.execute(
        f"SELECT revoked_at FROM {RETRIEVAL_REFERENCES_TABLE}"
    ).fetchone() == (None,)

    conn.execute("BEGIN")
    transient_uuid = rotate_database_uuid(
        conn,
        authorization_context={"admin": True},
        authorize_rotation=_allow_rotation,
        revoked_at=320.0,
    )
    assert transient_uuid != original_uuid
    assert conn.in_transaction
    conn.rollback()
    assert get_database_uuid(conn) == original_uuid
    assert conn.execute(
        f"SELECT revoked_at FROM {RETRIEVAL_REFERENCES_TABLE}"
    ).fetchone() == (None,)
    conn.close()


def test_byte_restore_preserves_uuid_and_registry(tmp_path):
    original_path = tmp_path / "original.db"
    original = _open(original_path)
    envelope = _issue(original)
    original_uuid = get_database_uuid(original)
    original.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    original.close()
    restored_path = tmp_path / "restored.db"
    shutil.copy2(original_path, restored_path)

    restored = _open(restored_path)
    try:
        assert get_database_uuid(restored) == original_uuid
        result = resolve_reference(
            restored,
            envelope,
            authorization_context={"trusted": True},
            authorize=_allow,
            authorize_scope=_allow_scope,
        )
        assert result.ok
        assert result.record is not None
        assert result.record.target == {"source": "test", "store_id": 7}
    finally:
        restored.close()


def test_registry_constraints_indexes_and_numeric_schema_version(tmp_path):
    conn = _open(tmp_path / "schema.db")
    try:
        assert get_schema_version(conn) == 5
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (RETRIEVAL_REFERENCES_TABLE,),
        ).fetchone()[0].lower()
        assert "check" in sql
        indexes = {
            row[1]
            for row in conn.execute(
                f"PRAGMA index_list({RETRIEVAL_REFERENCES_TABLE})"
            ).fetchall()
        }
        assert f"idx_{RETRIEVAL_REFERENCES_TABLE}_kind_state" in indexes
        assert f"idx_{RETRIEVAL_REFERENCES_TABLE}_scope" in indexes
        assert verify_retrieval_references_schema(conn) == []
    finally:
        conn.close()


def test_registry_index_names_cannot_be_owned_by_another_table(tmp_path):
    conn = _open(tmp_path / "foreign-index-owner.db")
    kind_index = f"idx_{RETRIEVAL_REFERENCES_TABLE}_kind_state"
    scope_index = f"idx_{RETRIEVAL_REFERENCES_TABLE}_scope"
    conn.execute(f"DROP INDEX {kind_index}")
    conn.execute(f"DROP INDEX {scope_index}")
    conn.execute(
        "CREATE TABLE other(kind TEXT, revoked_at REAL, expires_at REAL, scope_json TEXT)"
    )
    conn.execute(
        f"CREATE INDEX {kind_index} ON other(kind, revoked_at, expires_at)"
    )
    conn.execute(f"CREATE INDEX {scope_index} ON other(kind, scope_json)")
    conn.commit()

    assert verify_retrieval_references_schema(conn) == [
        f"malformed index:{kind_index}",
        f"malformed index:{scope_index}",
    ]
    with pytest.raises(RetrievalReferenceSchemaError, match="completed.*damaged"):
        run_versioned_migrations(conn)
    assert conn.execute(
        "SELECT name, tbl_name FROM sqlite_master WHERE name IN (?, ?) ORDER BY name",
        (kind_index, scope_index),
    ).fetchall() == [(kind_index, "other"), (scope_index, "other")]
    conn.close()


def test_named_migration_failure_rolls_back_identity_and_new_registry(tmp_path, monkeypatch):
    conn = sqlite3.connect(tmp_path / "migration-rollback.db")
    original_verify = db_bootstrap.verify_retrieval_references_schema
    monkeypatch.setattr(
        db_bootstrap,
        "verify_retrieval_references_schema",
        lambda connection: ["injected schema failure"],
    )
    try:
        with pytest.raises(RetrievalReferenceSchemaError, match="injected schema failure"):
            run_versioned_migrations(conn)
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retrieval_references_v1'"
        ).fetchone() is None
        assert conn.execute(
            "SELECT value FROM metadata WHERE key='database_uuid'"
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name=?",
            (DATABASE_UUID_MIGRATION,),
        ).fetchone() is None
    finally:
        monkeypatch.setattr(db_bootstrap, "verify_retrieval_references_schema", original_verify)
        conn.close()


def test_registry_index_recreated_as_unique_is_rejected(tmp_path):
    """Name, table and columns do not pin an index's semantics.

    A unique idx_..._scope keeps all three and would verify as healthy, while
    rejecting the second reference issued for a scope -- a correctness change
    this verification exists to catch.
    """
    path = tmp_path / "index-uniqueness.db"
    conn = _open(path)
    ensure_retrieval_reference_migrations(conn)
    assert db_bootstrap.verify_retrieval_references_schema(conn) == []

    table = db_bootstrap.RETRIEVAL_REFERENCES_TABLE
    index_name = f"idx_{table}_scope"
    conn.execute(f"DROP INDEX {index_name}")
    conn.execute(f"CREATE UNIQUE INDEX {index_name} ON {table}(kind, scope_json)")
    conn.commit()
    try:
        assert db_bootstrap.verify_retrieval_references_schema(conn) == [
            f"malformed index:{index_name}"
        ]
    finally:
        conn.close()
