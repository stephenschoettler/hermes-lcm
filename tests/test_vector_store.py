from __future__ import annotations

import json
import math
import sqlite3
import struct
import threading
from array import array

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore
import hermes_lcm.vector_store as vector_store_module


EMBEDDING_TABLES = {
    "lcm_embedding_profile",
    "lcm_embedding_meta",
    "lcm_embedding_vectors",
}
MIGRATION_STEP = "embeddings_v1"


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


@pytest.fixture
def stores(tmp_path):
    db_path = tmp_path / "vectors.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path)
    try:
        yield dag, store
    finally:
        store.close()
        dag.close()


def _add_summary(
    dag: SummaryDAG,
    *,
    session_id: str = "conversation-a",
    source_token_count: int = 100,
    created_at: float = 1.0,
) -> int:
    return dag.add_node(
        SummaryNode(
            session_id=session_id,
            summary=f"summary for {session_id} at {created_at}",
            source_token_count=source_token_count,
            created_at=created_at,
            earliest_at=created_at,
            latest_at=created_at,
        )
    )


def _record_embedding(
    store: VectorStore,
    embedded_id: str | int,
    kind: str,
    model: str,
    vec,
    *,
    identity: EmbeddingIdentity | None = None,
) -> None:
    """Test helper that captures identity before invoking the write API."""
    if identity is None:
        identity = store.capture_identity(model)
    store.record_embedding(
        embedded_id, kind, model, vec, identity=identity
    )


def test_core_migrations_omit_embedding_tables(tmp_path):
    """A disabled install stays at schema_version 5 with no embedding tables.

    Embedding tables are opt-in and never created by the core migration path,
    so a base build can open the DB and the numeric counter stays free for the
    temporal train (no v6 collision).
    """
    conn = sqlite3.connect(tmp_path / "core_only.db")
    try:
        db_bootstrap.run_versioned_migrations(conn)
        conn.commit()

        assert db_bootstrap.SCHEMA_VERSION == 5
        assert db_bootstrap.get_schema_version(conn) == 5
        assert not (EMBEDDING_TABLES & _table_names(conn))
        marker = conn.execute(
            "SELECT step_name FROM lcm_migration_state WHERE step_name = ?",
            (MIGRATION_STEP,),
        ).fetchall()
        assert marker == []
    finally:
        conn.close()


def test_vector_store_creates_embedding_tables_lazily_and_idempotently(tmp_path):
    """VectorStore materializes the opt-in tables on first use, still at v5."""
    db_path = tmp_path / "idempotent.db"
    first = VectorStore(db_path)
    first.close()
    # Re-opening must not duplicate the marker or fail on existing tables.
    store = VectorStore(db_path)
    try:
        assert EMBEDDING_TABLES <= _table_names(store.connection)
        assert db_bootstrap.get_schema_version(store.connection) == 5
        steps = store.connection.execute(
            "SELECT step_name FROM lcm_migration_state WHERE step_name = ?",
            (MIGRATION_STEP,),
        ).fetchall()
        assert [tuple(row) for row in steps] == [(MIGRATION_STEP,)]
        index_sql = store.connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'index'
              AND name = 'idx_lcm_embedding_meta_identity_embedded_at'
            """
        ).fetchone()[0]
        assert "WHERE archived = 0" in index_sql
    finally:
        store.close()


def test_vector_store_upgrades_previous_schema_version(tmp_path):
    db_path = tmp_path / "previous.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(db_bootstrap.SCHEMA_VERSION - 1),),
    )
    conn.commit()
    conn.close()

    store = VectorStore(db_path)
    try:
        assert EMBEDDING_TABLES <= _table_names(store.connection)
        assert db_bootstrap.get_schema_version(store.connection) == db_bootstrap.SCHEMA_VERSION
        completed = store.connection.execute(
            "SELECT completed_at FROM lcm_migration_state WHERE step_name = ?",
            (MIGRATION_STEP,),
        ).fetchone()
        assert completed is not None
    finally:
        store.close()


def test_vector_store_refuses_newer_schema_before_configuring_connection(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "future.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(db_bootstrap.SCHEMA_VERSION + 1),),
    )
    conn.commit()
    conn.close()

    configure_called = False

    def fail_if_called(conn):
        nonlocal configure_called
        configure_called = True
        raise AssertionError("configure_connection should not run for future schemas")

    monkeypatch.setattr(vector_store_module, "configure_connection", fail_if_called)

    with pytest.raises(db_bootstrap.SchemaVersionTooNewError):
        VectorStore(db_path)
    assert configure_called is False
    check = sqlite3.connect(db_path)
    try:
        assert _table_names(check) == {"metadata"}
        assert check.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    finally:
        check.close()


def test_profile_identity_distinguishes_provider_without_clobber(tmp_path):
    """Same model_name under two providers is two profiles, not a silent overwrite."""
    store = VectorStore(tmp_path / "profiles.db")
    try:
        identity_a = store.register_profile("model-a", "provider-a", 3)
        identity_b = store.register_profile("model-a", "provider-b", 3)
        assert identity_a != identity_b

        rows = store.connection.execute(
            """
            SELECT provider, dim, active
            FROM lcm_embedding_profile
            WHERE model_name = 'model-a'
            ORDER BY provider
            """
        ).fetchall()
        # Both provider rows survive; provider-a's metadata was not clobbered.
        assert [tuple(r) for r in rows] == [
            ("provider-a", 3, 0),
            ("provider-b", 3, 1),
        ]
        # A different dim is a different identity (new row), not a locked error.
        identity_c = store.register_profile("model-a", "provider-a", 4)
        assert identity_c not in {identity_a, identity_b}
        assert store.connection.execute(
            "SELECT COUNT(*) FROM lcm_embedding_profile WHERE model_name = 'model-a'"
        ).fetchone()[0] == 3
    finally:
        store.close()


def test_switch_provider_a_b_a_reactivates_without_rebackfill(stores):
    """Switching config A→B→A reactivates A's profile with its vectors intact."""
    dag, store = stores
    node_a = _add_summary(dag, created_at=1.0)
    node_b = _add_summary(dag, created_at=2.0)
    store.register_profile("shared-model", "provider-a", 3)
    _record_embedding(store, node_a, "summary", "shared-model", [1.0, 0.0, 0.0])

    # Switch to provider B (same model name); A is retained but deactivated.
    store.register_profile("shared-model", "provider-b", 3)
    _record_embedding(store, node_b, "summary", "shared-model", [0.0, 1.0, 0.0])
    current = store._current_profile()
    assert current["provider"] == "provider-b"

    # Switch back to A: no re-backfill, A's vector must resolve again.
    store.register_profile("shared-model", "provider-a", 3)
    current = store._current_profile()
    assert current["provider"] == "provider-a"
    result = store.knn([1.0, 0.0, 0.0])
    assert [row[0] for row in result] == [str(node_a)]
    # Only exactly one profile is active at a time.
    active = store.connection.execute(
        "SELECT COUNT(*) FROM lcm_embedding_profile WHERE active = 1 AND archived_at IS NULL"
    ).fetchone()[0]
    assert active == 1


def test_record_and_knn_match_hand_computed_cosines(stores):
    dag, store = stores
    axis_x = _add_summary(dag, source_token_count=11, created_at=1.0)
    diagonal = _add_summary(dag, source_token_count=22, created_at=2.0)
    axis_y = _add_summary(dag, source_token_count=33, created_at=3.0)
    store.register_profile("three-d", "local", 3)
    _record_embedding(store, axis_x, "summary", "three-d", [1.0, 0.0, 0.0])
    _record_embedding(store, diagonal, "summary", "three-d", [1.0, 1.0, 0.0])
    _record_embedding(store, axis_y, "summary", "three-d", [0.0, 1.0, 0.0])

    result = store.knn([1.0, 0.0, 0.0], k=3, model="three-d")

    assert [row[0] for row in result] == [str(axis_x), str(diagonal), str(axis_y)]
    assert [row[1] for row in result] == pytest.approx(
        [1.0, 1.0 / math.sqrt(2.0), 0.0],
        abs=1e-6,
    )
    assert [row[2] for row in result] == ["summary", "summary", "summary"]
    assert result.coverage in {"full", "bounded"}
    token_counts = store.connection.execute(
        """
        SELECT embedded_id, source_token_count
        FROM lcm_embedding_meta
        ORDER BY CAST(embedded_id AS INTEGER)
        """
    ).fetchall()
    assert [tuple(row) for row in token_counts] == [
        (str(axis_x), 11),
        (str(diagonal), 22),
        (str(axis_y), 33),
    ]


def test_record_normalizes_vector_before_packing(stores):
    dag, store = stores
    node_id = _add_summary(dag)
    store.register_profile("normalized", "local", 3)
    _record_embedding(store, node_id, "summary", "normalized", [3.0, 4.0, 0.0])

    blob = store.connection.execute(
        "SELECT vec FROM lcm_embedding_vectors WHERE embedded_id = ?",
        (str(node_id),),
    ).fetchone()[0]
    unpacked = array("f")
    unpacked.frombytes(blob)
    assert list(unpacked) == pytest.approx([0.6, 0.8, 0.0], abs=1e-6)
    result = store.knn([3.0, 4.0, 0.0], model="normalized")
    assert result[0][1] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "identity_field",
    [
        {"dtype": "float64"},
        {"byteorder": "big"},
        {"task": "query"},
    ],
)
def test_profile_rejects_unsupported_vector_representation(stores, identity_field):
    _dag, store = stores
    with pytest.raises(ValueError, match=r"supported representation is float32\|int8/little/summary"):
        store.register_profile("unsupported", "local", 2, **identity_field)


def test_vector_wire_format_is_explicit_little_endian_float32(stores):
    dag, store = stores
    node_id = _add_summary(dag)
    store.register_profile("wire", "local", 2)
    _record_embedding(store, node_id, "summary", "wire", [3.0, 4.0])

    blob = store.connection.execute(
        "SELECT vec FROM lcm_embedding_vectors WHERE embedded_id = ?",
        (str(node_id),),
    ).fetchone()[0]
    assert blob == struct.pack("<2f", 0.6, 0.8)
    assert store.knn([3.0, 4.0], model="wire")[0][0] == str(node_id)


def test_numpy_absent_reports_full_when_scan_covers_corpus(stores, monkeypatch):
    dag, store = stores
    first = _add_summary(dag, created_at=1.0)
    second = _add_summary(dag, created_at=2.0)
    third = _add_summary(dag, created_at=3.0)
    store.register_profile("fallback", "local", 3)
    _record_embedding(store, first, "summary", "fallback", [1.0, 0.0, 0.0])
    _record_embedding(store, second, "summary", "fallback", [1.0, 1.0, 0.0])
    _record_embedding(store, third, "summary", "fallback", [0.0, 1.0, 0.0])

    def unavailable():
        raise ImportError("numpy not installed")

    monkeypatch.setattr(vector_store_module, "_load_numpy", unavailable)
    result = store.knn([1.0, 0.0, 0.0], k=3, model="fallback")

    assert result.coverage == "full"
    assert [row[0] for row in result] == [str(first), str(second), str(third)]
    assert [row[1] for row in result] == pytest.approx(
        [1.0, 1.0 / math.sqrt(2.0), 0.0],
        abs=1e-6,
    )


def test_bounded_scan_uses_source_recency_after_newest_first_backfill(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "bounded.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=1)
    try:
        oldest = _add_summary(dag, created_at=1.0)
        middle = _add_summary(dag, created_at=2.0)
        newest = _add_summary(dag, created_at=3.0)
        store.register_profile("bounded", "local", 3)
        # Backfill discovers newest summaries first. That makes the oldest source
        # the most recently embedded row, but the bounded retrieval window must
        # still follow source chronology rather than vector write chronology.
        _record_embedding(store, newest, "summary", "bounded", [0.0, 0.0, 1.0])
        _record_embedding(store, middle, "summary", "bounded", [0.0, 1.0, 0.0])
        _record_embedding(store, oldest, "summary", "bounded", [1.0, 0.0, 0.0])
        store.connection.executemany(
            "UPDATE lcm_embedding_meta SET embedded_at = ? WHERE embedded_id = ?",
            [
                ("2026-07-15T01:00:00+00:00", str(newest)),
                ("2026-07-15T02:00:00+00:00", str(middle)),
                ("2026-07-15T03:00:00+00:00", str(oldest)),
            ],
        )
        store.connection.commit()

        def unavailable():
            raise ImportError("numpy not installed")

        monkeypatch.setattr(vector_store_module, "_load_numpy", unavailable)
        result = store.knn([1.0, 0.0, 0.0], k=3, model="bounded")

        assert result.coverage == "bounded"
        assert [row[0] for row in result] == [str(newest)]
    finally:
        store.close()
        dag.close()


def test_full_scan_reaches_best_vector_outside_recency_bound(tmp_path):
    db_path = tmp_path / "full-scan.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=1)
    try:
        oldest = _add_summary(dag, created_at=1.0)
        newest = _add_summary(dag, created_at=2.0)
        store.register_profile("full-scan", "local", 2)
        _record_embedding(store, oldest, "summary", "full-scan", [1.0, 0.0])
        _record_embedding(store, newest, "summary", "full-scan", [0.0, 1.0])

        bounded = store.knn([1.0, 0.0], k=1, model="full-scan")
        exhaustive = store.knn(
            [1.0, 0.0],
            k=1,
            model="full-scan",
            full_scan=True,
        )

        assert bounded.coverage == "bounded"
        assert [row[0] for row in bounded] == [str(newest)]
        assert exhaustive.coverage == "full"
        assert exhaustive.reason is None
        assert exhaustive.scanned == exhaustive.total == 2
        assert [row[0] for row in exhaustive] == [str(oldest)]
    finally:
        store.close()
        dag.close()


def test_full_scan_reports_incomplete_coverage_when_vector_decode_fails(tmp_path):
    db_path = tmp_path / "full-scan-malformed.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=1)
    try:
        valid = _add_summary(dag, created_at=1.0)
        malformed = _add_summary(dag, created_at=2.0)
        store.register_profile("full-scan", "local", 2)
        _record_embedding(store, valid, "summary", "full-scan", [1.0, 0.0])
        _record_embedding(store, malformed, "summary", "full-scan", [0.0, 1.0])
        store.connection.execute(
            "UPDATE lcm_embedding_vectors SET vec = ? WHERE embedded_id = ?",
            (sqlite3.Binary(b"\x00"), str(malformed)),
        )

        result = store.knn(
            [1.0, 0.0],
            k=2,
            model="full-scan",
            full_scan=True,
        )

        assert result.coverage == "bounded"
        assert result.reason == "malformed_vectors_skipped"
        assert result.scanned == 1
        assert result.total == 2
        assert [row[0] for row in result] == [str(valid)]
    finally:
        store.close()
        dag.close()


def test_full_scan_timeout_releases_read_snapshot(tmp_path, monkeypatch):
    db_path = tmp_path / "full-scan-timeout.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=1)
    try:
        node_id = _add_summary(dag, created_at=1.0)
        store.register_profile("full-scan", "local", 2)
        _record_embedding(store, node_id, "summary", "full-scan", [1.0, 0.0])
        ticks = iter([0.0, 1.0])
        monkeypatch.setattr(
            vector_store_module.time, "monotonic", lambda: next(ticks, 1.0)
        )

        with pytest.raises(TimeoutError, match="deadline exhausted"):
            store.knn(
                [1.0, 0.0],
                k=1,
                model="full-scan",
                full_scan=True,
                deadline=0.5,
            )
        assert store.connection.in_transaction is False

        monkeypatch.setattr(vector_store_module.time, "monotonic", lambda: 0.0)
        recovered = store.knn(
            [1.0, 0.0],
            k=1,
            model="full-scan",
            full_scan=True,
            deadline=1.0,
        )
        assert recovered.coverage == "full"
        assert [row[0] for row in recovered] == [str(node_id)]
    finally:
        store.close()
        dag.close()


def test_full_scan_timeout_clears_progress_handler_before_rollback_and_refreshes(
    tmp_path, monkeypatch
):
    """An expired progress handler must not interrupt read-snapshot cleanup."""
    db_path = tmp_path / "full-scan-progress-timeout.db"
    dag = SummaryDAG(db_path)
    real_connect = sqlite3.connect

    class InterruptingRollbackConnection(sqlite3.Connection):
        progress_handler_active = False

        def set_progress_handler(self, progress_handler, n):
            self.progress_handler_active = progress_handler is not None
            return super().set_progress_handler(progress_handler, n)

        def rollback(self):
            if self.progress_handler_active:
                raise sqlite3.OperationalError("interrupted")
            return super().rollback()

    def tracked_connect(*args, **kwargs):
        kwargs["factory"] = InterruptingRollbackConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(vector_store_module.sqlite3, "connect", tracked_connect)
    store = VectorStore(db_path, bounded_scan_rows=1)
    try:
        first = _add_summary(dag, created_at=1.0)
        store.register_profile("full-scan", "local", 2)
        _record_embedding(store, first, "summary", "full-scan", [1.0, 0.0])
        store.connection.set_progress_handler(lambda: 0, 1000)
        ticks = iter([0.0, 1.0])
        monkeypatch.setattr(
            vector_store_module.time, "monotonic", lambda: next(ticks, 1.0)
        )

        with pytest.raises(TimeoutError, match="deadline exhausted"):
            store.knn(
                [0.0, 1.0],
                k=1,
                model="full-scan",
                full_scan=True,
                deadline=0.5,
            )
        assert store.connection.in_transaction is False
        assert store.connection.progress_handler_active is False

        second = _add_summary(dag, created_at=2.0)
        writer = VectorStore(db_path)
        try:
            _record_embedding(writer, second, "summary", "full-scan", [0.0, 1.0])
        finally:
            writer.close()

        monkeypatch.setattr(vector_store_module.time, "monotonic", lambda: 0.0)
        refreshed = store.knn(
            [0.0, 1.0],
            k=1,
            model="full-scan",
            full_scan=True,
            deadline=1.0,
        )
        assert refreshed.coverage == "full"
        assert [row[0] for row in refreshed] == [str(second)]
        assert store.connection.in_transaction is False
    finally:
        store.close()
        dag.close()


def test_full_scan_exception_releases_owned_read_transaction(tmp_path, monkeypatch):
    db_path = tmp_path / "full-scan-exception.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=1)
    try:
        node_id = _add_summary(dag, created_at=1.0)
        store.register_profile("full-scan", "local", 2)
        _record_embedding(store, node_id, "summary", "full-scan", [1.0, 0.0])

        def fail_decode(*_args, **_kwargs):
            raise RuntimeError("simulated decode failure")

        monkeypatch.setattr(store, "_decode_stored_vec", fail_decode)
        with pytest.raises(RuntimeError, match="simulated decode failure"):
            store.knn(
                [1.0, 0.0],
                k=1,
                model="full-scan",
                full_scan=True,
            )
        assert store.connection.in_transaction is False
    finally:
        store.close()
        dag.close()


def test_full_scan_releases_preexisting_stale_read_transaction(tmp_path):
    db_path = tmp_path / "full-scan-stale-transaction.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=1)
    try:
        node_id = _add_summary(dag, created_at=1.0)
        store.register_profile("full-scan", "local", 2)
        _record_embedding(store, node_id, "summary", "full-scan", [1.0, 0.0])
        store.connection.execute("BEGIN")
        store.connection.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()

        result = store.knn(
            [1.0, 0.0],
            k=1,
            model="full-scan",
            full_scan=True,
        )

        assert result.coverage == "full"
        assert store.connection.in_transaction is False
    finally:
        store.close()
        dag.close()


def test_bounded_scan_keeps_null_latest_at_legacy_rows_by_created_at(
    tmp_path, monkeypatch
):
    """The DAG migration adds latest_at without backfilling legacy rows; the
    bounded window must fall back to created_at per-row so upgraded databases
    do not silently lose legacy summaries from candidate enumeration."""
    db_path = tmp_path / "bounded_null.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=1)
    try:
        older = _add_summary(dag, created_at=1.0)
        legacy_newest = _add_summary(dag, created_at=5.0)
        store.register_profile("bounded", "local", 3)
        _record_embedding(store, older, "summary", "bounded", [1.0, 0.0, 0.0])
        _record_embedding(store, legacy_newest, "summary", "bounded", [0.0, 1.0, 0.0])
        # Simulate a legacy row: migration added the column, no backfill.
        store.connection.execute(
            "UPDATE summary_nodes SET latest_at = NULL WHERE node_id = ?",
            (legacy_newest,),
        )
        store.connection.commit()

        result = store.knn([0.0, 1.0, 0.0], k=2, model="bounded")

        assert result.coverage == "bounded"
        # The legacy row is the chronologically newest by created_at, so the
        # bound-1 window must contain it — not drop it for having NULL latest_at.
        assert [row[0] for row in result] == [str(legacy_newest)]
    finally:
        store.close()
        dag.close()

def test_suppressed_summaries_are_filtered_and_purge_removes_embeddings(stores):
    dag, store = stores
    suppressed = _add_summary(dag, created_at=1.0)
    kept = _add_summary(dag, created_at=2.0)
    store.register_profile("suppression", "local", 3)
    _record_embedding(store, suppressed, "summary", "suppression", [1.0, 0.0, 0.0])
    _record_embedding(store, kept, "summary", "suppression", [0.0, 1.0, 0.0])
    store.connection.execute("ALTER TABLE summary_nodes ADD COLUMN suppressed_at TEXT")
    store.connection.execute(
        "UPDATE summary_nodes SET suppressed_at = '2026-07-15' WHERE node_id = ?",
        (suppressed,),
    )
    store.connection.commit()

    result = store.knn([1.0, 0.0, 0.0], k=2, model="suppression")
    assert [row[0] for row in result] == [str(kept)]

    assert store.purge_embeddings_for_nodes([kept, kept]) == 1
    assert store.connection.execute(
        "SELECT COUNT(*) FROM lcm_embedding_meta WHERE embedded_id = ?",
        (str(kept),),
    ).fetchone()[0] == 0
    assert store.connection.execute(
        "SELECT COUNT(*) FROM lcm_embedding_vectors WHERE embedded_id = ?",
        (str(kept),),
    ).fetchone()[0] == 0


def test_filter_overfetch_uses_summary_time_and_session_columns(stores):
    dag, store = stores
    old_a = _add_summary(dag, session_id="conversation-a", created_at=10.0)
    new_a = _add_summary(dag, session_id="conversation-a", created_at=20.0)
    new_b = _add_summary(dag, session_id="conversation-b", created_at=30.0)
    store.register_profile("filters", "local", 3)
    _record_embedding(store, old_a, "summary", "filters", [1.0, 0.0, 0.0])
    _record_embedding(store, new_a, "summary", "filters", [0.9, 0.1, 0.0])
    _record_embedding(store, new_b, "summary", "filters", [0.8, 0.2, 0.0])

    result = store.knn(
        [1.0, 0.0, 0.0],
        k=1,
        model="filters",
        since=15.0,
        conversation_ids=["conversation-a"],
    )
    assert [row[0] for row in result] == [str(new_a)]


def test_filters_are_applied_before_score_top_k_truncation(stores):
    pytest.importorskip("numpy")
    dag, store = stores
    store.register_profile("filter-before-top-k", "local", 2)

    for index in range(501):
        unfiltered = _add_summary(
            dag,
            session_id="conversation-other",
            created_at=float(index + 1),
        )
        _record_embedding(
            store,
            unfiltered,
            "summary",
            "filter-before-top-k",
            [1.0, 0.0],
        )

    filtered_ids = [
        _add_summary(
            dag,
            session_id="conversation-target",
            created_at=1_000.0 + index,
        )
        for index in range(2)
    ]
    for node_id in filtered_ids:
        _record_embedding(
            store,
            node_id,
            "summary",
            "filter-before-top-k",
            [0.0, 1.0],
        )

    result = store.knn(
        [1.0, 0.0],
        k=2,
        model="filter-before-top-k",
        conversation_ids=["conversation-target"],
    )

    assert result.coverage == "full"
    assert {row[0] for row in result} == {str(node_id) for node_id in filtered_ids}


def test_matrix_cache_is_invalidated_on_write(stores):
    pytest.importorskip("numpy")
    dag, store = stores
    first = _add_summary(dag, created_at=1.0)
    second = _add_summary(dag, created_at=2.0)
    store.register_profile("cache", "local", 3)
    _record_embedding(store, first, "summary", "cache", [1.0, 0.0, 0.0])

    initial = store.knn([0.0, 1.0, 0.0], model="cache")
    assert initial[0][0] == str(first)
    assert store._matrix_cache

    _record_embedding(store, second, "summary", "cache", [0.0, 1.0, 0.0])
    assert store._matrix_cache == {}
    updated = store.knn([0.0, 1.0, 0.0], model="cache")
    assert updated[0][0] == str(second)


def test_time_to_filter_excludes_before_top_k(stores):
    pytest.importorskip("numpy")
    dag, store = stores
    store.register_profile("time-to", "local", 2)
    # 501 high-scoring but too-new vectors must not consume the top-k slots.
    for index in range(501):
        too_new = _add_summary(dag, created_at=10_000.0 + index)
        _record_embedding(store, too_new, "summary", "time-to", [1.0, 0.0])
    eligible = _add_summary(dag, created_at=5.0)
    _record_embedding(store, eligible, "summary", "time-to", [0.0, 1.0])

    result = store.knn([1.0, 0.0], k=2, model="time-to", until=100.0)
    assert result.coverage == "full"
    assert [row[0] for row in result] == [str(eligible)]


def test_source_filter_enforced_before_top_k(stores):
    pytest.importorskip("numpy")
    dag, store = stores
    conn = store.connection
    conn.execute("CREATE TABLE IF NOT EXISTS messages (store_id INTEGER PRIMARY KEY, source TEXT)")
    conn.execute("INSERT INTO messages(store_id, source) VALUES (1, 'keep'), (2, 'drop')")
    conn.commit()
    store.register_profile("source-filter", "local", 2)

    # Many high-scoring vectors from the wrong source must be excluded before cap.
    for index in range(300):
        wrong = dag.add_node(
            SummaryNode(
                session_id="conversation-a",
                summary=f"wrong {index}",
                source_ids=[2],
                created_at=float(index + 1),
            )
        )
        _record_embedding(store, wrong, "summary", "source-filter", [1.0, 0.0])
    right = dag.add_node(
        SummaryNode(
            session_id="conversation-a",
            summary="right",
            source_ids=[1],
            created_at=1_000.0,
        )
    )
    _record_embedding(store, right, "summary", "source-filter", [0.0, 1.0])

    result = store.knn([1.0, 0.0], k=3, model="source-filter", source="keep")
    assert result.coverage == "bounded"
    assert [row[0] for row in result] == [str(right)]


def test_data_version_bump_invalidates_cross_process_cache(stores, tmp_path):
    pytest.importorskip("numpy")
    dag, store = stores
    first = _add_summary(dag, created_at=1.0)
    second = _add_summary(dag, created_at=2.0)
    store.register_profile("shared", "local", 3)
    _record_embedding(store, first, "summary", "shared", [1.0, 0.0, 0.0])

    # Process A opens its own connection and warms its matrix cache.
    process_a = VectorStore(store.db_path)
    try:
        warmed = process_a.knn([0.0, 1.0, 0.0], model="shared")
        assert [row[0] for row in warmed] == [str(first)]
        assert process_a._matrix_cache

        # Process B writes a new vector, bumping the durable data_version in the
        # same transaction. max_rowid/row_count alone would not reveal an
        # in-place rewrite, but the counter forces process A to reload.
        _record_embedding(store, second, "summary", "shared", [0.0, 1.0, 0.0])

        refreshed = process_a.knn([0.0, 1.0, 0.0], model="shared")
        assert refreshed[0][0] == str(second)
    finally:
        process_a.close()


def test_large_id_metadata_resolve_scales_past_variable_limit(stores):
    pytest.importorskip("numpy")
    dag, store = stores
    store.register_profile("bulk", "local", 2)
    conn = store.connection
    # Insert 40k summary+vector rows directly for speed; a giant WHERE id IN
    # (...) resolve would raise "too many SQL variables" at ~33k on this runtime.
    now = 1.0
    vec = array("f", store._normalized([1.0, 0.0], expected_dim=2)).tobytes()
    identity = store._current_profile()["identity_hash"]
    conn.execute("BEGIN")
    for node_id in range(1, 40_001):
        conn.execute(
            "INSERT INTO summary_nodes(node_id, session_id, depth, summary, "
            "source_token_count, source_ids, source_type, created_at, "
            "earliest_at, latest_at) VALUES (?, 'conversation-a', 0, 's', 1, "
            "'[]', 'messages', ?, ?, ?)",
            (node_id, now, now, now),
        )
    conn.executemany(
        "INSERT INTO lcm_embedding_vectors(embedded_id, identity_hash, vec) VALUES (?, ?, ?)",
        [(str(node_id), identity, vec) for node_id in range(1, 40_001)],
    )
    conn.executemany(
        "INSERT INTO lcm_embedding_meta(embedded_id, embedded_kind, identity_hash, "
        "embedded_at, source_token_count, archived) VALUES (?, 'summary', ?, '2026', 1, 0)",
        [(str(node_id), identity) for node_id in range(1, 40_001)],
    )
    conn.commit()
    store._matrix_cache.clear()

    result = store.knn(
        [1.0, 0.0],
        k=5,
        model="bulk",
        conversation_ids=["conversation-a"],
    )
    assert result.coverage == "bounded"
    assert len(result) == 5


def test_no_profile_or_vectors_returns_none_coverage(tmp_path):
    store = VectorStore(tmp_path / "none.db")
    try:
        no_profile = store.knn([1.0, 0.0, 0.0])
        assert no_profile == []
        assert no_profile.coverage == "none"

        store.register_profile("empty", "local", 3)
        no_vectors = store.knn([1.0, 0.0, 0.0])
        assert no_vectors == []
        assert no_vectors.coverage == "none"
    finally:
        store.close()


def test_embedding_config_defaults_are_inert_and_read_environment(monkeypatch):
    defaults = LCMConfig()
    assert defaults.embeddings_enabled is False
    assert defaults.embedding_bounded_scan_rows == 2_000

    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "true")
    monkeypatch.setenv("LCM_EMBEDDING_BOUNDED_SCAN_ROWS", "123")
    configured = LCMConfig.from_env()
    assert configured.embeddings_enabled is True
    assert configured.embedding_bounded_scan_rows == 123

    store = VectorStore(":memory:")
    try:
        assert store.bounded_scan_rows == 123
    finally:
        store.close()


def _numpy_unavailable():
    raise ImportError("numpy not installed")


def test_legacy_db_missing_latest_at_and_source_fails_filters_closed(tmp_path):
    """A VectorStore-only worker DB predating the DAG/source migrations.

    Its summary_nodes lacks latest_at (added by the DAG migration) and its
    messages lacks source (added by MessageStore._ensure_source_column). Time-
    and source-scoped KNN must degrade gracefully instead of raising
    "no such column: latest_at" / "no such column: source".
    """
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE summary_nodes (
            node_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            depth INTEGER DEFAULT 0,
            summary TEXT,
            source_token_count INTEGER,
            source_ids TEXT NOT NULL DEFAULT '[]',
            source_type TEXT NOT NULL DEFAULT 'messages',
            created_at REAL
        );
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT
        );
        """
    )
    cur = conn.execute(
        "INSERT INTO messages(session_id, role, content) VALUES ('s', 'user', 'hi')"
    )
    msg_id = cur.lastrowid
    conn.execute(
        "INSERT INTO summary_nodes(session_id, summary, source_token_count, source_ids, source_type, created_at) "
        "VALUES ('s', 'legacy summary', 10, ?, 'messages', 5.0)",
        (json.dumps([msg_id]),),
    )
    node_id = conn.execute("SELECT node_id FROM summary_nodes").fetchone()[0]
    conn.commit()
    conn.close()

    store = VectorStore(db_path)
    try:
        store.register_profile("m", "p", 3)
        _record_embedding(store, node_id, "summary", "m", [1.0, 0.0, 0.0])

        # Publication time is not coverage time.  If latest_at is absent a
        # requested time filter is unverifiable and must fail closed.
        in_range = store.knn([1.0, 0.0, 0.0], model="m", since=1.0, until=10.0)
        assert list(in_range) == []
        assert in_range.reason == "unverifiable_provenance"

        # source filter FAILS CLOSED (not crashed, not fail-open) when
        # messages.source is absent: provenance is unverifiable, so a
        # source-filtered query returns nothing rather than surfacing the
        # legacy summary whose source could not be confirmed.
        with_source = store.knn([1.0, 0.0, 0.0], model="m", source="whatever")
        assert list(with_source) == []
    finally:
        store.close()


def test_record_embedding_rejects_non_integer_id_without_crashing(stores):
    dag, store = stores
    store.register_profile("m", "p", 3)
    with pytest.raises(ValueError, match="summary node does not exist"):
        _record_embedding(store, "not-a-node", "summary", "m", [1.0, 0.0, 0.0])


def test_candidate_filter_uses_node_id_pk_index(stores):
    """The candidate JOIN binds the integer node_id so the PK index is used."""
    dag, store = stores
    node = _add_summary(dag, created_at=1.0)
    store.register_profile("m", "p", 3)
    _record_embedding(store, node, "summary", "m", [1.0, 0.0, 0.0])
    plan = "\n".join(
        str(row[-1])
        for row in store.connection.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT t.id FROM (SELECT ? AS id) t "
            "JOIN summary_nodes sn ON sn.node_id = CAST(t.id AS INTEGER)",
            (str(node),),
        ).fetchall()
    )
    # An integer-keyed lookup uses the INTEGER PRIMARY KEY, never a full scan.
    assert "SCAN summary_nodes" not in plan


def test_bounded_path_filters_before_applying_recency_bound(tmp_path, monkeypatch):
    """No-numpy path: a filtered match beyond the recency window is not lost.

    With bounded_scan_rows=1 the old behavior loaded only the single most-recent
    vector and then filtered it out, dropping the eligible older match. Filtering
    before the bound keeps it.
    """
    db_path = tmp_path / "before_bound.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=1)
    try:
        keep = _add_summary(dag, session_id="conversation-a", created_at=1.0)
        drop = _add_summary(dag, session_id="conversation-b", created_at=2.0)
        store.register_profile("m", "local", 3)
        _record_embedding(store, keep, "summary", "m", [1.0, 0.0, 0.0])
        _record_embedding(store, drop, "summary", "m", [1.0, 0.0, 0.0])
        # The wrong-conversation row is the most recent by embedded_at, so a
        # bound-first scan would pick only it and then filter to empty.
        store.connection.executemany(
            "UPDATE lcm_embedding_meta SET embedded_at = ? WHERE embedded_id = ?",
            [
                ("2026-07-15T05:00:00+00:00", str(drop)),
                ("2026-07-15T01:00:00+00:00", str(keep)),
            ],
        )
        store.connection.commit()
        monkeypatch.setattr(vector_store_module, "_load_numpy", _numpy_unavailable)

        result = store.knn(
            [1.0, 0.0, 0.0], k=5, model="m", conversation_ids=["conversation-a"]
        )
        assert result.coverage == "full"
        assert [row[0] for row in result] == [str(keep)]
    finally:
        store.close()
        dag.close()


def test_knn_resolves_by_provider_identity_not_model_name(stores):
    """Reads follow the configured provider identity, not the bare model name."""
    dag, store = stores
    node = _add_summary(dag, created_at=1.0)
    store.register_profile("shared", "provider-a", 3)
    _record_embedding(store, node, "summary", "shared", [1.0, 0.0, 0.0])
    # Switch config to provider-b for the same model (now active, no vectors).
    store.register_profile("shared", "provider-b", 3)

    # provider-b identity has no backfilled vectors -> coverage none -> degrade.
    res_b = store.knn([1.0, 0.0, 0.0], model="shared", provider="provider-b")
    assert res_b.coverage == "none"
    assert list(res_b) == []

    # provider-a identity still resolves its own vectors.
    res_a = store.knn([1.0, 0.0, 0.0], model="shared", provider="provider-a")
    assert [row[0] for row in res_a] == [str(node)]


def test_orphaned_embeddings_are_not_ranked_and_purge_reclaims(stores):
    dag, store = stores
    live = _add_summary(dag, created_at=1.0)
    orphan = _add_summary(dag, created_at=2.0)
    store.register_profile("m", "local", 3)
    _record_embedding(store, live, "summary", "m", [1.0, 0.0, 0.0])
    _record_embedding(store, orphan, "summary", "m", [1.0, 0.0, 0.0])

    # Delete the orphan's summary node (a deletion path that has not purged yet).
    store.connection.execute("DELETE FROM summary_nodes WHERE node_id = ?", (orphan,))
    store.connection.commit()

    result = store.knn([1.0, 0.0, 0.0], k=5, model="m")
    assert [row[0] for row in result] == [str(live)]

    assert store.purge_embeddings_for_nodes([orphan]) == 1
    assert store.connection.execute(
        "SELECT COUNT(*) FROM lcm_embedding_vectors WHERE embedded_id = ?",
        (str(orphan),),
    ).fetchone()[0] == 0


def test_large_session_delete_purges_exact_ids_in_bounded_batches(tmp_path):
    db_path = tmp_path / "batched-delete.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path)
    try:
        node_ids = list(range(1, 5_001))
        dag.connection.executemany(
            "INSERT INTO summary_nodes("
            "node_id, session_id, depth, summary, source_token_count, "
            "source_ids, source_type, created_at) "
            "VALUES (?, 'large', 0, 'summary', 1, '[]', 'messages', ?)",
            ((node_id, float(node_id)) for node_id in node_ids),
        )
        dag.connection.commit()
        identity = store.register_profile("bulk-delete", "local", 2)
        vec = struct.pack("<2f", 1.0, 0.0)
        store.connection.executemany(
            "INSERT INTO lcm_embedding_vectors(embedded_id, identity_hash, vec) "
            "VALUES (?, ?, ?)",
            ((str(node_id), identity, vec) for node_id in node_ids),
        )
        store.connection.executemany(
            "INSERT INTO lcm_embedding_meta("
            "embedded_id, embedded_kind, identity_hash, embedded_at, "
            "source_token_count, archived) "
            "VALUES (?, 'summary', ?, '2026-01-01', 1, 0)",
            ((str(node_id), identity) for node_id in node_ids),
        )
        store.connection.commit()
        batches: list[list[int]] = []

        def purge_batch(batch: list[int]) -> None:
            batches.append(list(batch))
            store.purge_embeddings_for_nodes(batch)

        deleted = dag.delete_session_nodes(
            "large", on_deleted_batch=purge_batch
        )

        assert deleted == len(node_ids)
        assert batches
        assert max(map(len, batches)) == 256
        assert [node_id for batch in batches for node_id in batch] == node_ids
        assert store.connection.execute(
            "SELECT COUNT(*) FROM lcm_embedding_vectors"
        ).fetchone()[0] == 0
        assert store.connection.execute(
            "SELECT COUNT(*) FROM lcm_embedding_meta"
        ).fetchone()[0] == 0
    finally:
        store.close()
        dag.close()


def test_delete_node_batch_stages_scope_past_sqlite_bind_cap(tmp_path):
    dag = SummaryDAG(tmp_path / "large-scope.db")
    try:
        session_ids = [f"session-{index}" for index in range(250_001)]
        assert SummaryDAG.delete_node_batch(dag.connection, session_ids) == []
    finally:
        dag.close()


def test_temp_id_tables_are_unique_per_call_and_dropped(stores):
    """Overlapping scratch tables get distinct names and never clobber each other.

    The old single fixed name meant a second call's ``DELETE FROM _lcm_id_scratch``
    wiped the first call's candidate set. Unique-per-call names keep both sets
    intact, and each table is dropped when its context exits.
    """
    dag, store = stores
    with store._temp_id_table(["1", "2"]) as first:
        with store._temp_id_table(["3", "4"]) as second:
            assert first != second
            assert {r[0] for r in store.connection.execute(f"SELECT id FROM {first}")} == {"1", "2"}
            assert {r[0] for r in store.connection.execute(f"SELECT id FROM {second}")} == {"3", "4"}
    leftover = store.connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE '_lcm_id_scratch%'"
    ).fetchall()
    assert leftover == []


def test_concurrent_knn_on_separate_stores_are_correct(tmp_path):
    """Production runs one VectorStore (own connection) per query; that is safe."""
    db_path = tmp_path / "concurrent.db"
    dag = SummaryDAG(db_path)
    seed = VectorStore(db_path)
    node = _add_summary(dag, created_at=1.0)
    seed.register_profile("m", "local", 3)
    _record_embedding(seed, node, "summary", "m", [1.0, 0.0, 0.0])
    seed.close()

    results: list[list[str]] = []
    errors: list[BaseException] = []

    def run():
        try:
            store = VectorStore(db_path)
            try:
                for _ in range(20):
                    res = store.knn(
                        [1.0, 0.0, 0.0], k=5, model="m", conversation_ids=["conversation-a"]
                    )
                    results.append([row[0] for row in res])
            finally:
                store.close()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the assertion
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    dag.close()

    assert errors == []
    assert results and all(rows == [str(node)] for rows in results)


def test_record_embedding_publishes_under_captured_identity_not_active(stores):
    """A1: an A-vector must publish under A's captured identity, never B's.

    Maintainer repro: produce a vector under provider A, switch the same
    model/dim to provider B (now active) before publication, then record. The
    provider-A vector must be stored under A's identity — the durable identity
    captured at provider-resolution time is carried through the write, and
    record_embedding does not silently rebind onto whatever is active now.
    """
    dag, store = stores
    node = _add_summary(dag, created_at=1.0)
    identity_a_hash = store.register_profile("shared-model", "provider-a", 3)
    identity_a = store.capture_identity("shared-model", provider="provider-a")
    # Flip the active identity to provider B for the SAME model/dim mid-flight.
    identity_b_hash = store.register_profile("shared-model", "provider-b", 3)
    assert identity_a_hash != identity_b_hash
    active = store._current_profile()["identity_hash"]
    assert active == identity_b_hash  # B is active; a bare-name resolve would pick B.

    _record_embedding(
        store,
        node, "summary", "shared-model", [1.0, 0.0, 0.0], identity=identity_a
    )

    vectors = store.connection.execute(
        "SELECT identity_hash FROM lcm_embedding_vectors WHERE embedded_id = ?",
        (str(node),),
    ).fetchall()
    meta = store.connection.execute(
        "SELECT identity_hash FROM lcm_embedding_meta WHERE embedded_id = ?",
        (str(node),),
    ).fetchall()
    assert [row[0] for row in vectors] == [identity_a_hash]
    assert [row[0] for row in meta] == [identity_a_hash]
    # Nothing is ever written under B's active identity.
    assert identity_b_hash not in {row[0] for row in vectors}


def test_record_embedding_rejects_unregistered_captured_identity(stores):
    """A1: a captured identity that is not a registered profile is rejected."""
    dag, store = stores
    node = _add_summary(dag, created_at=1.0)
    store.register_profile("m", "p", 3)
    with pytest.raises(ValueError, match="profile is not registered"):
        _record_embedding(
            store,
            node,
            "summary",
            "m",
            [1.0, 0.0, 0.0],
            identity=EmbeddingIdentity.canonical(
                "other-provider", "m", "", 3, "float32", "little", "summary"
            ),
        )


def test_source_filter_fails_closed_on_legacy_db_without_source_column(tmp_path):
    """A2: a source filter on a DB whose messages lacks ``source`` returns nothing.

    Maintainer repro: query for a source that definitely did not exist and the
    legacy summary was returned (fail-open). Provenance being unverifiable must
    fail CLOSED — no false-positive hit.
    """
    db_path = tmp_path / "legacy_source.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE summary_nodes (
            node_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            depth INTEGER DEFAULT 0,
            summary TEXT,
            source_token_count INTEGER,
            source_ids TEXT NOT NULL DEFAULT '[]',
            source_type TEXT NOT NULL DEFAULT 'messages',
            created_at REAL
        );
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT
        );
        """
    )
    cur = conn.execute(
        "INSERT INTO messages(session_id, role, content) VALUES ('s', 'user', 'hi')"
    )
    msg_id = cur.lastrowid
    conn.execute(
        "INSERT INTO summary_nodes(session_id, summary, source_token_count, "
        "source_ids, source_type, created_at) VALUES ('s', 'legacy', 10, ?, "
        "'messages', 5.0)",
        (json.dumps([msg_id]),),
    )
    node_id = conn.execute("SELECT node_id FROM summary_nodes").fetchone()[0]
    conn.commit()
    conn.close()

    store = VectorStore(db_path)
    try:
        store.register_profile("m", "p", 3)
        _record_embedding(store, node_id, "summary", "m", [1.0, 0.0, 0.0])
        # Without a source filter the summary is returned (baseline).
        assert [row[0] for row in store.knn([1.0, 0.0, 0.0], model="m")] == [
            str(node_id)
        ]
        # With a source filter on an unverifiable schema: no false-positive hit,
        # on BOTH the numpy and the dependency-free bounded paths.
        assert list(store.knn([1.0, 0.0, 0.0], model="m", source="nope")) == []
    finally:
        store.close()


def test_source_filter_fails_closed_on_bounded_path(tmp_path, monkeypatch):
    """A2: fail-closed also holds on the no-numpy bounded path."""
    db_path = tmp_path / "legacy_source_bounded.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=50)
    try:
        # Drop messages.source: emulate a DB predating the source repair.
        store.connection.execute(
            "CREATE TABLE IF NOT EXISTS messages (store_id INTEGER PRIMARY KEY)"
        )
        store.connection.commit()
        node = dag.add_node(
            SummaryNode(
                session_id="conversation-a",
                summary="legacy",
                source_ids=[1],
                created_at=1.0,
            )
        )
        store.register_profile("m", "p", 3)
        _record_embedding(store, node, "summary", "m", [1.0, 0.0, 0.0])
        monkeypatch.setattr(vector_store_module, "_load_numpy", _numpy_unavailable)
        result = store.knn([1.0, 0.0, 0.0], model="m", source="nope")
        assert result.coverage == "none"
        assert result.reason == "unverifiable_provenance"
        assert list(result) == []
    finally:
        store.close()
        dag.close()


def test_source_lineage_overflow_fails_closed_with_bounded_sql_work(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "bounded-lineage.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=10)
    try:
        store.connection.execute(
            "CREATE TABLE IF NOT EXISTS messages("
            "store_id INTEGER PRIMARY KEY, source TEXT)"
        )
        store.connection.execute(
            "INSERT INTO messages(store_id, source) VALUES (1, 'keep')"
        )
        child = dag.add_node(
            SummaryNode(
                session_id="conversation-a",
                summary="leaf",
                source_ids=[1],
                source_type="messages",
                created_at=1.0,
            )
        )
        for index in range(40):
            child = dag.add_node(
                SummaryNode(
                    session_id="conversation-a",
                    summary=f"parent-{index}",
                    source_ids=[child],
                    source_type="nodes",
                    created_at=float(index + 2),
                )
            )
        store.register_profile("lineage", "local", 2)
        _record_embedding(store, child, "summary", "lineage", [1.0, 0.0])
        monkeypatch.setattr(vector_store_module, "_SOURCE_LINEAGE_WORK_LIMIT", 16)
        statements: list[str] = []
        store.connection.set_trace_callback(statements.append)
        try:
            result = store.knn(
                [1.0, 0.0], model="lineage", source="keep"
            )
        finally:
            store.connection.set_trace_callback(None)

        assert list(result) == []
        assert result.coverage == "none"
        assert result.reason == "unverifiable_provenance"
        recursive = [sql for sql in statements if "WITH RECURSIVE walk" in sql]
        assert recursive
        assert "LIMIT 17" in recursive[0]
    finally:
        store.close()
        dag.close()


def test_bounded_candidate_enumeration_is_capped_at_sql_layer(tmp_path, monkeypatch):
    """A3: enumeration is bounded at the SQL layer, not the whole corpus.

    Maintainer repro: a 100-row corpus with bound 10 still loaded 100 ids. The
    candidate enumeration must ORDER BY recency + LIMIT at SQL, so only ~bound
    rows are enumerated.
    """
    db_path = tmp_path / "bounded_enum.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=10)
    try:
        identity = store.register_profile("m", "p", 3)
        for index in range(100):
            node = _add_summary(dag, created_at=float(index + 1))
            _record_embedding(store, node, "summary", "m", [1.0, 0.0, 0.0])

        # Direct: the enumerator returns exactly the bound from a 100-row corpus.
        bounded = store._bounded_candidate_ids(
            identity,
            since=None,
            until=None,
            conversation_ids=None,
            source=None,
            limit=10,
        )
        assert len(bounded) == 10

        # Spy: the enumeration SQL carries a LIMIT (the bound lives in SQL, not
        # a host-side slice over the whole corpus).
        statements: list[str] = []
        store.connection.set_trace_callback(statements.append)
        try:
            monkeypatch.setattr(
                vector_store_module, "_load_numpy", _numpy_unavailable
            )
            result = store.knn([1.0, 0.0, 0.0], k=5, model="m")
        finally:
            store.connection.set_trace_callback(None)
        assert result.coverage == "bounded"
        assert len(result) == 5
        enum_stmts = [
            sql
            for sql in statements
            if "FROM lcm_embedding_meta m" in sql and "LIMIT" in sql
        ]
        assert enum_stmts, "bounded enumeration query must carry a SQL LIMIT"
    finally:
        store.close()
        dag.close()


def test_embedding_schema_repaired_when_marker_set_but_table_dropped(tmp_path):
    """A4: a set ``embeddings_v1`` marker over a missing table is repaired on init."""
    db_path = tmp_path / "repair.db"
    store = VectorStore(db_path)
    store.close()

    conn = sqlite3.connect(db_path)
    marker = conn.execute(
        "SELECT 1 FROM lcm_migration_state WHERE step_name = ?", ("embeddings_v1",)
    ).fetchone()
    assert marker is not None  # marker present ...
    conn.execute("DROP TABLE lcm_embedding_vectors")  # ... but a table is gone.
    conn.commit()
    conn.close()

    # Re-opening must VERIFY + repair rather than trust the marker.
    repaired = VectorStore(db_path)
    try:
        assert "lcm_embedding_vectors" in _table_names(repaired.connection)
        assert db_bootstrap.embedding_schema_missing(repaired.connection) == set()
    finally:
        repaired.close()


def test_record_embedding_requires_captured_identity(stores):
    dag, store = stores
    node = _add_summary(dag)
    store.register_profile("shared", "provider-a", 3)
    with pytest.raises(TypeError, match="identity"):
        store.record_embedding(node, "summary", "shared", [1.0, 0.0, 0.0])
    identity_hash = store._current_profile()["identity_hash"]
    with pytest.raises(TypeError, match="EmbeddingIdentity"):
        store.record_embedding(
            node,
            "summary",
            "shared",
            [1.0, 0.0, 0.0],
            identity=identity_hash,
        )


def test_provider_identity_fields_are_canonical_at_write_and_lookup(stores):
    dag, store = stores
    node = _add_summary(dag)
    identity_hash = store.register_profile(" shared ", " Voyage ", 3)
    identity = store.capture_identity("shared", provider="voyage")
    _record_embedding(
        store, node, "summary", "shared", [1.0, 0.0, 0.0], identity=identity
    )
    profile = store._profile_by_identity(identity_hash)
    assert profile["provider"] == "voyage"
    assert profile["model_name"] == "shared"
    canonical = store.knn(
        [1.0, 0.0, 0.0], model="shared", provider="voyage"
    )
    display_form = store.knn(
        [1.0, 0.0, 0.0], model=" shared ", provider=" Voyage "
    )
    assert [row[0] for row in canonical] == [str(node)]
    assert list(display_form) == list(canonical)


def test_numpy_candidate_load_is_sql_bounded(tmp_path, monkeypatch):
    numpy = pytest.importorskip("numpy")
    db_path = tmp_path / "bounded_numpy.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=10)
    try:
        identity = store.register_profile("m", "p", 2)
        vec = array("f", [1.0, 0.0]).tobytes()
        for index in range(30):
            node = _add_summary(dag, created_at=float(index + 1))
            store.connection.execute(
                "INSERT INTO lcm_embedding_vectors(embedded_id, identity_hash, vec) "
                "VALUES (?, ?, ?)",
                (str(node), identity, vec),
            )
            store.connection.execute(
                "INSERT INTO lcm_embedding_meta(embedded_id, embedded_kind, "
                "identity_hash, embedded_at, source_token_count, archived) "
                "VALUES (?, 'summary', ?, ?, 1, 0)",
                (str(node), identity, f"2026-01-01T00:00:{index:02d}+00:00"),
            )
        store.connection.commit()
        loaded: list[int] = []
        original = store._numpy_rows

        def counted(np, identity_hash, dim, ids, dtype="float32"):
            loaded.append(len(ids))
            return original(np, identity_hash, dim, ids, dtype)

        monkeypatch.setattr(store, "_numpy_rows", counted)
        result = store.knn([1.0, 0.0], k=1, model="m")
        assert result.coverage == "bounded"
        assert loaded == [10]
        assert len(result) == 1
        assert numpy is not None
    finally:
        store.close()
        dag.close()


def test_conversation_filter_above_sqlite_bind_cap_is_staged(stores, monkeypatch):
    dag, store = stores
    node = _add_summary(dag, session_id="target")
    store.register_profile("m", "p", 3)
    _record_embedding(store, node, "summary", "m", [1.0, 0.0, 0.0])
    monkeypatch.setattr(vector_store_module, "_load_numpy", _numpy_unavailable)
    conversation_ids = [f"other-{index}" for index in range(250_000)] + ["target"]
    result = store.knn(
        [1.0, 0.0, 0.0], model="m", conversation_ids=conversation_ids
    )
    assert [row[0] for row in result] == [str(node)]


def test_malformed_same_name_embedding_table_is_rejected(tmp_path):
    db_path = tmp_path / "malformed.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE lcm_embedding_profile("
        "identity_hash TEXT, provider TEXT, model_name TEXT)"
    )
    conn.commit()
    conn.close()

    with pytest.raises(sqlite3.OperationalError, match="malformed table"):
        VectorStore(db_path)
    check = sqlite3.connect(db_path)
    try:
        marker = check.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='lcm_migration_state'"
        ).fetchone()
        if marker:
            assert check.execute(
                "SELECT 1 FROM lcm_migration_state WHERE step_name='embeddings_v1'"
            ).fetchone() is None
    finally:
        check.close()


def test_malformed_same_name_embedding_index_is_rejected(tmp_path):
    db_path = tmp_path / "bad_index.db"
    store = VectorStore(db_path)
    store.close()
    conn = sqlite3.connect(db_path)
    conn.execute("DROP INDEX idx_lcm_embedding_meta_identity_embedded_at")
    conn.execute(
        "CREATE UNIQUE INDEX idx_lcm_embedding_meta_identity_embedded_at "
        "ON lcm_embedding_meta(identity_hash, embedded_at ASC)"
    )
    conn.commit()
    conn.close()
    with pytest.raises(sqlite3.OperationalError, match="malformed index"):
        VectorStore(db_path)


def test_malformed_embedding_index_collation_is_rejected(tmp_path):
    db_path = tmp_path / "bad_index_collation.db"
    store = VectorStore(db_path)
    store.close()
    conn = sqlite3.connect(db_path)
    conn.execute("DROP INDEX idx_lcm_embedding_profile_model")
    conn.execute(
        "CREATE INDEX idx_lcm_embedding_profile_model "
        "ON lcm_embedding_profile(model_name COLLATE NOCASE, provider)"
    )
    conn.commit()
    conn.close()

    with pytest.raises(sqlite3.OperationalError, match="malformed index"):
        VectorStore(db_path)


@pytest.mark.parametrize(
    "dim_clause,data_version_default",
    [
        ("CHECK(dim BETWEEN 0 AND 4096)", "0"),
        ("CHECK(dim BETWEEN 1 AND 4096)", "1"),
    ],
)
def test_malformed_embedding_constraint_or_default_is_rejected(
    tmp_path, dim_clause, data_version_default
):
    db_path = tmp_path / f"bad_shape_{data_version_default}.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE lcm_embedding_profile ("
        "identity_hash TEXT PRIMARY KEY, provider TEXT NOT NULL, "
        "model_name TEXT NOT NULL, revision TEXT NOT NULL DEFAULT '', "
        f"dim INTEGER {dim_clause}, "
        "dtype TEXT NOT NULL DEFAULT 'float32', "
        "byteorder TEXT NOT NULL DEFAULT 'little', "
        "task TEXT NOT NULL DEFAULT 'summary', registered_at TEXT, "
        "active INTEGER DEFAULT 1, archived_at TEXT NULL, "
        f"data_version INTEGER NOT NULL DEFAULT {data_version_default})"
    )
    conn.commit()
    conn.close()
    with pytest.raises(
        sqlite3.OperationalError, match="malformed (table|constraints)"
    ):
        VectorStore(db_path)


def test_marker_write_skipped_when_already_stamped(tmp_path, monkeypatch):
    """sprint-opt-1: constructing a VectorStore over an already-stamped DB does
    not re-write the embeddings_v1 / chunk_vectors_v1 markers."""
    db_path = tmp_path / "markers.db"
    first = VectorStore(db_path)  # first construction stamps both markers
    first.ensure_chunk_schema()
    first.close()

    calls: list[str] = []
    real_mark = vector_store_module.mark_migration_step_complete

    def counting_mark(conn, step_name):
        calls.append(step_name)
        return real_mark(conn, step_name)

    monkeypatch.setattr(vector_store_module, "mark_migration_step_complete", counting_mark)
    vs = VectorStore(db_path)
    try:
        vs.ensure_chunk_schema()
        # Neither the embedding nor the chunk marker is re-written: both were
        # already stamped by the first construction.
        assert "embeddings_v1" not in calls
        assert "chunk_vectors_v1" not in calls
    finally:
        vs.close()


def test_summary_knn_survives_chunk_profile_registration(tmp_path):
    """Registering the chunk-corpus profile for the same (model, provider) must
    not redirect the summary knn to the chunk identity: _resolve_profile is
    task-scoped. Regression for the harness summary-arm zeroing (coverage
    'none' after H2 began registering both profiles per store)."""
    db_path = tmp_path / "taskscope.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=100)
    try:
        node = _add_summary(dag, created_at=1.0)
        store.register_profile("m1", "local", 3)
        _record_embedding(store, node, "summary", "m1", [1.0, 0.0, 0.0])

        before = store.knn([1.0, 0.0, 0.0], k=5, model="m1", provider="local")
        assert before.coverage == "full"
        assert len(list(before)) == 1

        store.register_profile("m1", "local", 3, task="chunk")

        after = store.knn([1.0, 0.0, 0.0], k=5, model="m1", provider="local")
        assert after.coverage == "full"
        assert [row[0] for row in after] == [str(node)]
    finally:
        store.close()
        dag.close()
