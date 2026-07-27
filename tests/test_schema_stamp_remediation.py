"""Tests for interim-build schema-stamp detection and guided remediation (fix #7).

A database touched by an interim development build can carry a numeric
``schema_version`` ahead of this build's ladder while its actual schema has a
compatible core plus named feature markers. These tests cover classification,
refusal-message guidance, and the explicit backup-first repair.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.command import (
    _doctor_repair_schema_stamp_apply_text,
    _doctor_repair_schema_stamp_text,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.db_bootstrap import (
    SchemaVersionTooNewError,
    classify_version_mismatch,
    remediate_interim_schema_stamp,
)
from hermes_lcm.engine import LCMEngine
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.store import MessageStore
from hermes_lcm.vector_store import VectorStore


def _build_v5_db(path: Path, *, with_features: bool = False) -> None:
    """Materialize the current core tables and both FTS indexes."""
    store = MessageStore(path)
    store.close()
    dag = SummaryDAG(path)
    dag.close()
    if with_features:
        rollups = RollupStore(path)
        rollups.close()
        conn = sqlite3.connect(path)
        try:
            db_bootstrap.ensure_embedding_tables(conn)
            conn.commit()
        finally:
            conn.close()


def _add_early_feature_tables(path: Path) -> None:
    """Create EARLY-variant feature tables (missing later-added columns/tables).

    Mirrors the real interim operator DB: family-prefixed tables that predate
    later schema additions (lcm_rollups without generation/lease_nonce/failed_at
    and no lcm_rollup_invalidations; lcm_embedding_profile keyed on model_name
    without identity_hash/data_version). These fail the final-shape verifiers.
    """
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE lcm_rollups (
                rollup_id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_kind TEXT, period_start TEXT, scope TEXT,
                summary TEXT, token_count INTEGER, status TEXT,
                built_at TEXT, source_fingerprint TEXT, error TEXT
            );
            CREATE TABLE lcm_rollup_sources (
                rollup_id INTEGER, node_id INTEGER,
                PRIMARY KEY(rollup_id, node_id)
            );
            CREATE TABLE lcm_rollup_state (
                period_kind TEXT PRIMARY KEY,
                last_build_cursor TEXT, last_built_at TEXT
            );
            CREATE TABLE lcm_embedding_profile (
                model_name TEXT PRIMARY KEY, provider TEXT, dim INTEGER,
                registered_at TEXT, active INTEGER DEFAULT 1, archived_at TEXT
            );
            CREATE TABLE lcm_embedding_meta (
                embedded_id TEXT, embedded_kind TEXT, model_name TEXT,
                embedded_at TEXT, PRIMARY KEY(embedded_id, embedded_kind)
            );
            CREATE TABLE lcm_embedding_vectors (
                embedded_id TEXT PRIMARY KEY, vec BLOB
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _add_early_chunk_tables(path: Path) -> None:
    """Create an EARLY-variant chunk schema (missing later-added columns/indexes).

    Mirrors a DB whose chunk corpus predates char-span columns and the required
    partial index — it fails ``verify_chunk_schema`` with missing-object /
    malformed-table findings (never an unexpected-column one).
    """
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE lcm_chunk_meta (
                chunk_id TEXT, identity_hash TEXT, store_id INTEGER,
                chunk_index INTEGER, embedded_at TEXT, archived INTEGER DEFAULT 0,
                PRIMARY KEY(chunk_id, identity_hash)
            );
            CREATE TABLE lcm_chunk_vectors (
                chunk_id TEXT, identity_hash TEXT, vec BLOB NOT NULL,
                PRIMARY KEY(chunk_id, identity_hash)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _table_names(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
    finally:
        conn.close()


def _stamp(path: Path, version: int) -> None:
    conn = sqlite3.connect(path)
    try:
        db_bootstrap.set_schema_version(conn, version)
        conn.commit()
    finally:
        conn.close()


def _stored_version(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        return db_bootstrap.read_existing_schema_version(conn)
    finally:
        conn.close()


# --- classification --------------------------------------------------------


def test_classify_interim_stamp_on_v5_shape(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        assert (
            classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
        )
    finally:
        conn.close()


def test_classify_interim_stamp_on_previous_core_shape(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE session_project_metadata")
        db_bootstrap.set_schema_version(conn, db_bootstrap.SCHEMA_VERSION + 1)
        conn.commit()

        assert (
            classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
        )
    finally:
        conn.close()


@pytest.mark.parametrize(
    "malformation",
    ["missing_constraints", "missing_index", "wrong_index"],
)
def test_classify_genuinely_newer_on_malformed_present_project_table(
    tmp_path, malformation
):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        if malformation == "missing_constraints":
            conn.execute("DROP TABLE session_project_metadata")
            conn.execute(
                "CREATE TABLE session_project_metadata("
                "session_id TEXT, project_id TEXT, project_root TEXT, cwd TEXT)"
            )
        else:
            conn.execute("DROP INDEX idx_session_project_metadata_project")
            if malformation == "wrong_index":
                conn.execute(
                    "CREATE INDEX idx_session_project_metadata_project "
                    "ON session_project_metadata(cwd, session_id)"
                )
        db_bootstrap.set_schema_version(conn, db_bootstrap.SCHEMA_VERSION + 1)
        conn.commit()

        assert (
            classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
        )
    finally:
        conn.close()


def test_classify_interim_stamp_with_feature_marker_tables(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path, with_features=True)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        # temporal-rollup + embedding tables are known feature markers, so the
        # DB is still classified as an interim stamp, not a genuinely newer DB.
        assert (
            classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
        )
    finally:
        conn.close()


def test_classify_genuinely_newer_on_unknown_table(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE lcm_future_widgets (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        assert (
            classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
        )
    finally:
        conn.close()


def test_classify_genuinely_newer_on_extra_feature_family_column(tmp_path):
    """An EXTRA column on a feature-family table is a newer-build signature.

    Reproduces F2-schema-stamp-drops-newer-data: a future release adds a column
    to ``lcm_rollups``. The old classifier ignored feature-table internal shape
    and called this an interim stamp, so remediation DROPPED the table and its
    siblings. It must classify ``genuinely_newer`` instead — an unexpected
    (extra) column is never an early-variant signature.
    """
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path, with_features=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("ALTER TABLE lcm_rollups ADD COLUMN future_col INTEGER DEFAULT 0")
        conn.commit()
    finally:
        conn.close()
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        assert (
            classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
        )
    finally:
        conn.close()


def test_remediate_apply_refuses_and_preserves_extra_column_family(tmp_path):
    """Remediation must NOT drop a family table that carries an extra column.

    The data-destruction guard: with an extra ``lcm_rollups`` column present,
    ``remediate_interim_schema_stamp(apply=True)`` refuses and leaves every
    feature table (and the stamp) untouched.
    """
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path, with_features=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("ALTER TABLE lcm_rollups ADD COLUMN future_col INTEGER DEFAULT 0")
        conn.commit()
    finally:
        conn.close()
    stamped = db_bootstrap.SCHEMA_VERSION + 1
    _stamp(db_path, stamped)
    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert result["status"] == "refused"
    assert result["classification"] == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
    assert result["applied"] is False
    assert result["dropped_tables"] == []
    # Nothing dropped, stamp untouched — real data survives.
    assert _stored_version(db_path) == stamped
    assert "lcm_rollups" in _table_names(db_path)


def test_classify_interim_stamp_on_missing_feature_family_column(tmp_path):
    """A feature table only MISSING a later-added column stays an interim stamp.

    The counterpart to the extra-column case: an early variant omits pieces (no
    ``generation``/``lease_nonce``/``failed_at`` on ``lcm_rollups``) and must
    still be classified interim so remediation can drop-and-rebuild it.
    """
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    _add_early_feature_tables(db_path)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        assert (
            classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
        )
    finally:
        conn.close()


def test_classify_genuinely_newer_on_unknown_core_column(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN future_flag INTEGER DEFAULT 0")
        conn.commit()
    finally:
        conn.close()
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        assert (
            classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
        )
    finally:
        conn.close()


# --- refusal-message guidance ---------------------------------------------


def test_refuse_message_points_at_remediation_for_interim_stamp(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    with pytest.raises(SchemaVersionTooNewError) as excinfo:
        MessageStore(db_path)
    message = str(excinfo.value)
    assert "schema-stamp" in message
    assert "do NOT upgrade" in message


def test_refuse_message_stays_generic_for_genuinely_newer(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE lcm_future_widgets (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    with pytest.raises(SchemaVersionTooNewError) as excinfo:
        MessageStore(db_path)
    message = str(excinfo.value)
    assert "restore a pre-upgrade backup" in message
    assert "schema-stamp" not in message


# --- remediation helper ----------------------------------------------------


def test_remediate_dry_run_reports_without_mutating(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    stamped = db_bootstrap.SCHEMA_VERSION + 1
    _stamp(db_path, stamped)
    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=False)
    finally:
        conn.close()
    assert result["status"] == "dry-run"
    assert result["classification"] == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
    assert result["applied"] is False
    assert _stored_version(db_path) == stamped


def test_remediate_apply_resets_stamp_and_db_reopens(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert result["status"] == "ok"
    assert result["applied"] is True
    assert _stored_version(db_path) == db_bootstrap.SCHEMA_VERSION
    # After the reset the store opens again without refusing.
    store = MessageStore(db_path)
    store.close()


def test_remediate_refuses_genuinely_newer(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE lcm_future_widgets (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    stamped = db_bootstrap.SCHEMA_VERSION + 1
    _stamp(db_path, stamped)
    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert result["status"] == "refused"
    assert result["classification"] == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
    assert result["applied"] is False
    assert _stored_version(db_path) == stamped


def test_early_variant_feature_tables_remediate_end_to_end(tmp_path):
    """Early-variant feature tables classify as interim and recover after apply.

    This is the real-operator-DB shape: clean v5 core plus family-prefixed
    tables that are EARLY variants failing the final-shape verifiers.
    """
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    _add_early_feature_tables(db_path)
    stamped = db_bootstrap.SCHEMA_VERSION + 1
    _stamp(db_path, stamped)

    # Classification ignores feature-table internal shape → interim_stamp.
    conn = sqlite3.connect(db_path)
    try:
        assert (
            classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
        )
        dry = remediate_interim_schema_stamp(conn, apply=False)
    finally:
        conn.close()
    assert dry["status"] == "dry-run"
    would_drop = {t for fam in dry["drop_plan"] for t in fam["tables"]}
    assert {"lcm_rollups", "lcm_embedding_profile"} <= would_drop
    # Dry-run mutates nothing.
    assert _stored_version(db_path) == stamped
    assert "lcm_rollups" in _table_names(db_path)

    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert result["status"] == "ok"
    dropped = set(result["dropped_tables"])
    assert {"lcm_rollups", "lcm_rollup_sources", "lcm_rollup_state"} <= dropped
    assert {
        "lcm_embedding_profile",
        "lcm_embedding_meta",
        "lcm_embedding_vectors",
    } <= dropped
    assert _stored_version(db_path) == db_bootstrap.SCHEMA_VERSION
    # Early feature tables are gone; core tables remain untouched.
    remaining = _table_names(db_path)
    assert not any(t.startswith(("lcm_rollup", "lcm_embedding")) for t in remaining)
    assert {"messages", "summary_nodes"} <= remaining

    # refuse now passes, and each feature store reconstructs the final shape.
    conn = sqlite3.connect(db_path)
    try:
        db_bootstrap.refuse_schema_version_too_new(conn)  # must not raise
    finally:
        conn.close()
    rollups = RollupStore(db_path)
    try:
        assert db_bootstrap.verify_temporal_rollup_schema(rollups.connection) == []
    finally:
        rollups.close()
    vectors = VectorStore(db_path)
    try:
        assert db_bootstrap.verify_embedding_schema(vectors._conn) == []
    finally:
        vectors.close()


def test_early_variant_chunk_family_remediates(tmp_path):
    """A broken chunk schema is dropped by remediation, not silently kept.

    Reproduces F2-schema-stamp-chunk-family-missing / F4-chunk-family-verifier-
    missing: with no ``lcm_chunk`` entry in the interim feature families the
    remediator reported ``status: ok, dropped_tables: []`` while leaving a broken
    ``lcm_chunk_meta``/``lcm_chunk_vectors`` in place. The family must now be
    verified and dropped so its marker-gated init rebuilds the final shape.
    """
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path, with_features=True)
    _add_early_chunk_tables(db_path)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)

    conn = sqlite3.connect(db_path)
    try:
        assert (
            classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
        )
        # The broken chunk schema fails its verifier (missing pieces, not extra).
        assert db_bootstrap.verify_chunk_schema(conn) != []
        assert not db_bootstrap._family_reports_newer_shape(
            db_bootstrap.verify_chunk_schema(conn)
        )
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert result["status"] == "ok"
    dropped = set(result["dropped_tables"])
    assert {"lcm_chunk_meta", "lcm_chunk_vectors"} <= dropped
    assert not any(t.startswith("lcm_chunk") for t in _table_names(db_path))

    # The chunk feature's own init recreates the final, verifier-clean shape.
    conn = sqlite3.connect(db_path)
    try:
        db_bootstrap.ensure_embedding_tables(conn)
        db_bootstrap.ensure_chunk_tables(conn)
        conn.commit()
        assert db_bootstrap.verify_chunk_schema(conn) == []
    finally:
        conn.close()


def test_remediate_noop_when_version_supported(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert result["status"] == "noop"
    assert result["applied"] is False


# --- /lcm doctor repair schema-stamp command path --------------------------


def _healthy_engine(tmp_path: Path) -> LCMEngine:
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    return LCMEngine(config=config, hermes_home=str(tmp_path / "home"))


def test_doctor_repair_schema_stamp_dry_run_and_apply(tmp_path):
    engine = _healthy_engine(tmp_path)
    db_path = Path(engine._store.db_path)
    # Add early-variant feature tables + stamp ahead of the ladder to simulate
    # an interim build, then drive the operator-facing command path.
    _add_early_feature_tables(db_path)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)

    dry = _doctor_repair_schema_stamp_text(engine)
    assert "status: repair-needed" in dry
    assert "classification: interim_stamp" in dry
    assert "would_drop: lcm_rollups" in dry
    assert "/lcm rollups rebuild" in dry
    assert "would_drop: lcm_embedding_profile" in dry
    assert "no schema changes were made" in dry
    # Dry-run must not mutate the stamp or drop anything.
    assert _stored_version(db_path) == db_bootstrap.SCHEMA_VERSION + 1
    assert "lcm_rollups" in _table_names(db_path)

    applied = _doctor_repair_schema_stamp_apply_text(engine)
    assert "status: ok" in applied
    assert "backup_path:" in applied
    assert f"schema_version_reset_to: {db_bootstrap.SCHEMA_VERSION}" in applied
    assert "dropped: lcm_rollups" in applied
    assert _stored_version(db_path) == db_bootstrap.SCHEMA_VERSION
    assert not any(
        t.startswith(("lcm_rollup", "lcm_embedding")) for t in _table_names(db_path)
    )


def test_doctor_repair_schema_stamp_apply_refuses_genuinely_newer(tmp_path):
    engine = _healthy_engine(tmp_path)
    db_path = Path(engine._store.db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE lcm_future_widgets (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)

    applied = _doctor_repair_schema_stamp_apply_text(engine)
    assert "status: refused" in applied
    assert "classification: genuinely_newer" in applied
    # No backup and no mutation on the refused path.
    assert "backup_path:" not in applied
    assert _stored_version(db_path) == db_bootstrap.SCHEMA_VERSION + 1
