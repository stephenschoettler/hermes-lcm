from __future__ import annotations

import json
import math
import sqlite3
import struct
import threading
import time
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



def _seed_scan_corpus(dag, store, size, *, gold_vector, filler_vector):
    """Seed ``size`` summaries whose OLDEST carries ``gold_vector``."""
    store.register_profile("scan", "local", 3)
    gold = _add_summary(dag, created_at=1.0)
    _record_embedding(store, gold, "summary", "scan", gold_vector)
    for index in range(1, size):
        node = _add_summary(dag, created_at=1.0 + index)
        _record_embedding(store, node, "summary", "scan", filler_vector)
    return gold


def test_full_scan_batches_the_whole_corpus_and_reaches_the_oldest(tmp_path):
    """F31 §2: with full_scan the bound is a BATCH SIZE, not a recency window.

    The gold vector is the OLDEST of a corpus more than 2x the batch size —
    the exact shape that made 86% of a 185k-vector store invisible and drove
    all-gold recall to 0.000 as content aged out.
    """
    db_path = tmp_path / "full-scan.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=3)  # 3-row batches, 7 vectors
    try:
        gold = _seed_scan_corpus(
            dag, store, 7, gold_vector=[1.0, 0.0, 0.0], filler_vector=[0.0, 1.0, 0.0]
        )

        windowed = store.knn([1.0, 0.0, 0.0], k=1, model="scan")
        assert windowed.coverage == "bounded"
        assert [row[0] for row in windowed] != [str(gold)]  # aged out of the window

        result = store.knn([1.0, 0.0, 0.0], k=1, model="scan", full_scan=True)

        assert result.coverage == "full"
        assert result.reason is None
        assert [row[0] for row in result] == [str(gold)]
    finally:
        store.close()
        dag.close()


def test_full_scan_without_numpy_also_reaches_the_oldest(tmp_path, monkeypatch):
    """The pure-Python scoring path batches identically (no numpy install)."""
    db_path = tmp_path / "full-scan-nonumpy.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=2)
    try:
        gold = _seed_scan_corpus(
            dag, store, 5, gold_vector=[1.0, 0.0, 0.0], filler_vector=[0.0, 1.0, 0.0]
        )

        def unavailable():
            raise ImportError("numpy not installed")

        monkeypatch.setattr(vector_store_module, "_load_numpy", unavailable)
        result = store.knn([1.0, 0.0, 0.0], k=1, model="scan", full_scan=True)

        assert result.coverage == "full"
        assert [row[0] for row in result] == [str(gold)]
    finally:
        store.close()
        dag.close()


def test_multi_batch_scan_does_not_populate_or_thrash_the_matrix_cache(tmp_path):
    """Review finding 2: a sweep needing more batches than the 4-entry LRU
    evicts each entry before the next sweep reaches it — zero hits, while
    retaining 4 float32 matrices and breaking the one-batch memory bound.
    Multi-batch sweeps therefore stream past the cache entirely."""
    db_path = tmp_path / "cache-thrash.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=2,
    )  # 2-row batches, 12 vectors
    try:
        _seed_scan_corpus(
            dag, store, 12, gold_vector=[1.0, 0.0, 0.0], filler_vector=[0.0, 1.0, 0.0]
        )
        assert len(store._matrix_cache) == 0

        for _ in range(3):  # repeat: the old code re-loaded all 6 batches each time
            store.knn([1.0, 0.0, 0.0], k=1, model="scan", full_scan=True)

        assert len(store._matrix_cache) == 0
    finally:
        store.close()
        dag.close()


def test_multi_batch_scan_releases_matrices_warmed_before_it(tmp_path, monkeypatch):
    """Delta-review residual on finding 2: cache=False keeps NEW batches out of
    the LRU but leaves matrices warmed by EARLIER calls resident for the pooled
    store's whole lifetime — the reviewer's probe measured cache_before=4 /
    cache_after=4 across a two-batch scan, i.e. ~192MB coexisting with the
    streamed batch. The invariant: at first-batch allocation, nothing is left.
    """
    numpy = pytest.importorskip("numpy")
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "cache-release.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=2,
    )  # 2-row batches, 8 vectors
    try:
        _seed_scan_corpus(
            dag, store, 8, gold_vector=[1.0, 0.0, 0.0], filler_vector=[0.0, 1.0, 0.0]
        )
        identity = str(store._current_profile()["identity_hash"])
        ids = store._bounded_candidate_ids(
            identity,
            since=None,
            until=None,
            conversation_ids=None,
            source=None,
            limit=vector_store_module._SCAN_ALL_ROWS,
        )
        # Warm the summary LRU to capacity the way earlier bounded calls would,
        # and plant sentinels in the sibling caches that are additive to it.
        for index in range(store._MATRIX_CACHE_MAX_ENTRIES):
            store._numpy_rows(numpy, identity, 3, ids[index:index + 2])
        store._chunk_matrix_cache[("sentinel", 0, ())] = ([], [], [], None)
        store._binary_matrix_cache[("sentinel", 0)] = ([], None)
        store._chunk_binary_matrix_cache[("sentinel", 0)] = ([], None)
        cache_before = len(store._matrix_cache)
        assert cache_before == store._MATRIX_CACHE_MAX_ENTRIES

        observed: list[tuple[int, int]] = []
        original = VectorStore._vectorized_batch

        def probe(np, rows, dim, dtype):
            observed.append((len(store._matrix_cache), len(store._chunk_matrix_cache)))
            return original(np, rows, dim, dtype)

        monkeypatch.setattr(VectorStore, "_vectorized_batch", staticmethod(probe))
        result = store.knn([1.0, 0.0, 0.0], k=1, model="scan", full_scan=True)

        assert result.coverage == "full"
        assert len(observed) == 4  # 8 vectors, 2 per batch
        # Released BEFORE the first batch is allocated, and never repopulated.
        assert observed[0] == (0, 0)
        assert all(sizes == (0, 0) for sizes in observed)
        assert len(store._matrix_cache) == 0
        assert len(store._chunk_matrix_cache) == 0
        # Persistent accelerator caches survive the transient-matrix release.
        assert len(store._binary_matrix_cache) == 1
        assert len(store._chunk_binary_matrix_cache) == 1
    finally:
        store.close()
        dag.close()


def test_single_batch_scan_still_caches_its_matrix(tmp_path):
    """The warm pooled-store path is unchanged when one batch covers the corpus."""
    db_path = tmp_path / "cache-single.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=50,
    )  # one batch covers all 5
    try:
        _seed_scan_corpus(
            dag, store, 5, gold_vector=[1.0, 0.0, 0.0], filler_vector=[0.0, 1.0, 0.0]
        )

        store.knn([1.0, 0.0, 0.0], k=1, model="scan", full_scan=True)

        assert len(store._matrix_cache) == 1
    finally:
        store.close()
        dag.close()


def test_vectorized_multibatch_ranking_is_bit_identical_to_legacy_loader(
    tmp_path, monkeypatch
):
    """Frozen #171 parity: the R1 cursor produces the old exact top-k bytes."""
    numpy = pytest.importorskip("numpy")
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "r1-parity.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=2,
    )
    try:
        _seed_scan_corpus(
            dag,
            store,
            9,
            gold_vector=[1.0, 0.0, 0.0],
            filler_vector=[0.0, 1.0, 0.0],
        )
        identity = str(store._current_profile()["identity_hash"])
        ids = store._bounded_candidate_ids(
            identity,
            since=None,
            until=None,
            conversation_ids=None,
            source=None,
            limit=vector_store_module._SCAN_ALL_ROWS,
        )
        old_rowids: list[int] = []
        old_ids: list[str] = []
        old_kinds: list[str] = []
        old_scores: list[float] = []
        query = numpy.asarray([1.0, 0.0, 0.0], dtype=numpy.float32)
        for start in range(0, len(ids), 2):
            with store._temp_id_table(ids[start:start + 2]) as table:
                rows = store.connection.execute(
                    f"""
                    SELECT v.rowid, v.embedded_id, m.embedded_kind, v.vec
                    FROM {table} t
                    JOIN lcm_embedding_vectors v
                      ON v.embedded_id = t.id AND v.identity_hash = ?
                    JOIN lcm_embedding_meta m
                      ON m.embedded_id = v.embedded_id
                     AND m.identity_hash = v.identity_hash
                    WHERE m.archived = 0
                    """,
                    (identity,),
                ).fetchall()
            vectors = [
                list(store._decode_stored_vec(bytes(row["vec"]), 3, "float32"))
                for row in rows
            ]
            scores = numpy.asarray(vectors, dtype=numpy.float32) @ query
            old_rowids.extend(int(row["rowid"]) for row in rows)
            old_ids.extend(str(row["embedded_id"]) for row in rows)
            old_kinds.extend(str(row["embedded_kind"]) for row in rows)
            old_scores.extend(float(score) for score in scores)
        expected = store._ranked(
            old_rowids, old_ids, old_kinds, old_scores, limit=5
        )

        actual = store.knn(
            query.tolist(), k=5, model="scan", full_scan=True
        )

        assert actual.coverage == "full"
        assert list(actual) == expected
    finally:
        store.close()
        dag.close()


def test_summary_int8_full_scan_uses_exact_residency(tmp_path, monkeypatch):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "summary-resident.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=1),
        bounded_scan_rows=1,
    )
    identity = EmbeddingIdentity.canonical(
        "local", "resident", "", 3, "int8", "little", "summary"
    )
    try:
        store.register_profile(
            "resident", "local", 3, dtype="int8", task="summary"
        )
        for index, vec in enumerate(
            (
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            )
        ):
            node_id = _add_summary(dag, created_at=float(index + 1))
            _record_embedding(
                store,
                node_id,
                "summary",
                "resident",
                vec,
                identity=identity,
            )
            if index == 0:
                expected_top = node_id

        result = store.knn(
            [1.0, 0.0, 0.0],
            k=3,
            model="resident",
            provider="local",
            full_scan=True,
        )

        assert result.coverage == "full"
        assert result.scoring == "int8_quantized"
        assert len(result) == 3
        assert result[0][0] == str(expected_top)
        assert len(store._resident_matrix_cache) == 1
        assert next(iter(store._resident_matrix_cache))[0] is False
    finally:
        store.close()
        dag.close()


def test_resident_deadline_does_not_start_a_count_query(tmp_path, monkeypatch):
    db_path = tmp_path / "resident-deadline.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=1),
        bounded_scan_rows=1,
    )
    try:
        _seed_scan_corpus(
            dag,
            store,
            3,
            gold_vector=[1.0, 0.0, 0.0],
            filler_vector=[0.0, 1.0, 0.0],
        )

        def expire(*args, **kwargs):
            raise vector_store_module._PrescreenDeadlineExpired(1)

        def count_after_deadline(*args, **kwargs):
            raise AssertionError("deadline expiry must not start COUNT(*)")

        monkeypatch.setattr(
            store, "_resident_int8_matrix", lambda *args, **kwargs: object()
        )
        monkeypatch.setattr(store, "_rank_resident_int8", expire)
        monkeypatch.setattr(
            store, "_count_embedded_vectors", count_after_deadline
        )
        result = store.knn(
            [1.0, 0.0, 0.0],
            k=1,
            model="scan",
            full_scan=True,
            deadline=vector_store_module._monotonic() + 60.0,
        )

        assert result.coverage == "bounded"
        assert result.scanned == 1
        assert result.total is None
    finally:
        store.close()
        dag.close()


@pytest.fixture
def empty_resident_registry():
    registry = vector_store_module._RESIDENT_MATRIX_REGISTRY
    with vector_store_module._RESIDENT_MATRIX_REGISTRY_LOCK:
        registry.clear()
        vector_store_module._RESIDENT_MATRIX_BUILD_LOCKS.clear()
    try:
        yield registry
    finally:
        with vector_store_module._RESIDENT_MATRIX_REGISTRY_LOCK:
            registry.clear()
            vector_store_module._RESIDENT_MATRIX_BUILD_LOCKS.clear()


def test_residency_survives_close_and_invalidates_on_data_version(
    tmp_path, monkeypatch, empty_resident_registry
):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "pooled-resident.db"
    dag = SummaryDAG(db_path)
    config = LCMConfig(knn_resident_max_mb=1)
    first = VectorStore(db_path, config=config, bounded_scan_rows=1)
    second = None
    builds: list[str] = []
    original = VectorStore._vector_rows_cursor

    def counted(store, identity_hash, *, chunk, candidate_ids=None):
        builds.append(store._resident_registry_db_path)
        return original(
            store,
            identity_hash,
            chunk=chunk,
            candidate_ids=candidate_ids,
        )

    monkeypatch.setattr(VectorStore, "_vector_rows_cursor", counted)
    try:
        _seed_scan_corpus(
            dag,
            first,
            3,
            gold_vector=[1.0, 0.0, 0.0],
            filler_vector=[0.0, 1.0, 0.0],
        )
        cold = first.knn(
            [1.0, 0.0, 0.0], k=2, model="scan", full_scan=True
        )
        assert cold.coverage == "full"
        assert len(builds) == 1
        pooled_key = next(iter(empty_resident_registry))
        assert pooled_key == (
            str(db_path.resolve()),
            *first._resident_registry_incarnation,
            3,
            first.capture_identity("scan").identity_hash,
            1,
        )

        first.close()
        assert list(empty_resident_registry) == [pooled_key]

        second = VectorStore(db_path, config=config, bounded_scan_rows=1)
        warm = second.knn(
            [1.0, 0.0, 0.0], k=2, model="scan", full_scan=True
        )
        assert list(warm) == list(cold)
        assert len(builds) == 1  # second instance performed zero rebuilds

        node = _add_summary(dag, created_at=4.0)
        _record_embedding(
            second,
            node,
            "summary",
            "scan",
            [0.5, 0.5, 0.0],
        )
        assert empty_resident_registry == {}
        refreshed = second.knn(
            [1.0, 0.0, 0.0], k=4, model="scan", full_scan=True
        )
        assert refreshed.coverage == "full"
        assert len(builds) == 2
        assert next(iter(empty_resident_registry))[3] == 4
    finally:
        first.close()
        if second is not None:
            second.close()
        dag.close()


def test_replaced_database_path_cannot_reuse_prior_incarnation_residency(
    tmp_path, monkeypatch, empty_resident_registry
):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "replaced-resident.db"
    config = LCMConfig(knn_resident_max_mb=1)

    old_dag = SummaryDAG(db_path)
    old_store = VectorStore(db_path, config=config, bounded_scan_rows=1)
    try:
        old_node = _add_summary(old_dag, created_at=1.0)
        old_store.register_profile("scan", "local", 3)
        _record_embedding(
            old_store,
            old_node,
            "summary",
            "scan",
            [1.0, 0.0, 0.0],
        )
        old_result = old_store.knn(
            [1.0, 0.0, 0.0], k=1, model="scan", full_scan=True
        )
        assert [row[0] for row in old_result] == [str(old_node)]
        old_incarnation = old_store._resident_registry_incarnation
    finally:
        old_store.close()
        old_dag.close()

    replacement_path = tmp_path / "replacement.db"
    replacement_dag = SummaryDAG(replacement_path)
    replacement_store = VectorStore(
        replacement_path, config=config, bounded_scan_rows=1
    )
    try:
        _add_summary(replacement_dag, created_at=1.0)
        replacement_node = _add_summary(replacement_dag, created_at=2.0)
        replacement_store.register_profile("scan", "local", 3)
        _record_embedding(
            replacement_store,
            replacement_node,
            "summary",
            "scan",
            [1.0, 0.0, 0.0],
        )
    finally:
        replacement_store.close()
        replacement_dag.close()

    replacement_path.replace(db_path)
    reopened = VectorStore(db_path, config=config, bounded_scan_rows=1)
    try:
        assert reopened._resident_registry_incarnation != old_incarnation
        replacement_result = reopened.knn(
            [1.0, 0.0, 0.0], k=1, model="scan", full_scan=True
        )
        assert [row[0] for row in replacement_result] == [
            str(replacement_node)
        ]
    finally:
        reopened.close()


def test_live_knn_residency_builds_once_and_ranks_every_query_above_crossover(
    tmp_path, monkeypatch, empty_resident_registry
):
    db_path = tmp_path / "live-resident-crossover.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=1),
        bounded_scan_rows=500,
    )
    build_count = 0
    resident_ranked_count = 0
    original_cursor = VectorStore._vector_rows_cursor
    original_rank = VectorStore._rank_resident_int8

    def counted_cursor(store, identity_hash, *, chunk, candidate_ids=None):
        nonlocal build_count
        if candidate_ids is None:
            build_count += 1
        return original_cursor(
            store,
            identity_hash,
            chunk=chunk,
            candidate_ids=candidate_ids,
        )

    def counted_rank(store, *args, **kwargs):
        nonlocal resident_ranked_count
        resident_ranked_count += 1
        return original_rank(store, *args, **kwargs)

    monkeypatch.setattr(VectorStore, "_vector_rows_cursor", counted_cursor)
    monkeypatch.setattr(VectorStore, "_rank_resident_int8", counted_rank)
    try:
        _seed_scan_corpus(
            dag,
            store,
            vector_store_module._FAST_SCAN_STREAMING_MIN_ROWS,
            gold_vector=[1.0, 0.0, 0.0],
            filler_vector=[0.0, 1.0, 0.0],
        )
        query_count = 3
        for _ in range(query_count):
            result = store.knn(
                [1.0, 0.0, 0.0],
                k=1,
                model="scan",
                full_scan=True,
                deadline=vector_store_module._monotonic() + 60.0,
            )
            assert result.coverage == "full"
            assert result.scoring == "int8_quantized"

        assert resident_ranked_count == query_count
        assert build_count == 1
    finally:
        store.close()
        dag.close()


def test_resident_hit_is_sql_free_and_constant_time_above_crossover(
    tmp_path, monkeypatch, empty_resident_registry
):
    numpy = pytest.importorskip("numpy")
    db_path = tmp_path / "resident-hit-cost.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=1),
        bounded_scan_rows=500,
    )
    try:
        _seed_scan_corpus(
            dag,
            store,
            vector_store_module._FAST_SCAN_STREAMING_MIN_ROWS,
            gold_vector=[1.0, 0.0, 0.0],
            filler_vector=[0.0, 1.0, 0.0],
        )
        profile = store._current_profile()
        assert profile is not None
        identity = str(profile["identity_hash"])
        kwargs = {
            "data_version": int(profile["data_version"]),
            "chunk": False,
            "deadline": None,
        }
        built = store._resident_int8_matrix(
            numpy, identity, 3, "float32", **kwargs
        )
        assert built is not None

        statements: list[str] = []
        store.connection.set_trace_callback(statements.append)
        try:
            hit = store._resident_int8_matrix(
                numpy, identity, 3, "float32", **kwargs
            )
        finally:
            store.connection.set_trace_callback(None)
        assert hit is built
        assert statements == []

        service_statements: list[str] = []
        store.connection.set_trace_callback(service_statements.append)
        try:
            service_hit = store.knn(
                [1.0, 0.0, 0.0],
                k=1,
                model="scan",
                full_scan=True,
            )
        finally:
            store.connection.set_trace_callback(None)
        assert service_hit.coverage == "full"
        assert not [
            statement
            for statement in service_statements
            if "count(" in statement.lower()
            or "from lcm_embedding_vectors" in statement.lower()
            or "from lcm_embedding_meta" in statement.lower()
        ]

        hit_ms: list[float] = []
        hit_scan_statement_counts: list[int] = []
        for _ in range(9):
            hit_statements: list[str] = []
            store.connection.set_trace_callback(hit_statements.append)
            started = time.perf_counter()
            try:
                service_hit = store.knn(
                    [1.0, 0.0, 0.0],
                    k=1,
                    model="scan",
                    full_scan=True,
                )
            finally:
                store.connection.set_trace_callback(None)
            assert service_hit.coverage == "full"
            hit_ms.append((time.perf_counter() - started) * 1000)
            hit_scan_statement_counts.append(
                sum(
                    "count(" in statement.lower()
                    or "from lcm_embedding_vectors" in statement.lower()
                    or "from lcm_embedding_meta" in statement.lower()
                    for statement in hit_statements
                )
            )
        assert hit_scan_statement_counts == [0] * len(hit_ms)
        assert sorted(hit_ms)[len(hit_ms) // 2] < 250.0
    finally:
        store.close()
        dag.close()


def test_resident_summary_filters_suppressed_rows_before_and_after_warmup(
    tmp_path, monkeypatch, empty_resident_registry
):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "resident-suppressed.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=1),
        bounded_scan_rows=1,
    )
    try:
        store.connection.execute(
            "ALTER TABLE summary_nodes ADD COLUMN suppressed_at TEXT"
        )
        first = _add_summary(dag, created_at=1.0)
        second = _add_summary(dag, created_at=2.0)
        third = _add_summary(dag, created_at=3.0)
        store.register_profile("scan", "local", 3)
        _record_embedding(
            store, first, "summary", "scan", [1.0, 0.0, 0.0]
        )
        _record_embedding(
            store, second, "summary", "scan", [0.8, 0.2, 0.0]
        )
        _record_embedding(
            store, third, "summary", "scan", [0.0, 1.0, 0.0]
        )
        store.connection.execute(
            "UPDATE summary_nodes SET suppressed_at = 'before' "
            "WHERE node_id = ?",
            (first,),
        )

        cold = store.knn(
            [1.0, 0.0, 0.0], k=3, model="scan", full_scan=True
        )
        assert cold.coverage == "full"
        assert str(first) not in {row[0] for row in cold}
        assert len(cold) == 2

        identity = store.capture_identity("scan").identity_hash
        before_version = store._data_version(identity)
        store.connection.execute(
            "UPDATE summary_nodes SET suppressed_at = 'after' "
            "WHERE node_id = ?",
            (second,),
        )
        assert store._data_version(identity) == before_version + 1
        warm = store.knn(
            [1.0, 0.0, 0.0], k=3, model="scan", full_scan=True
        )
        assert warm.coverage == "full"
        assert [row[0] for row in warm] == [str(third)]
        assert len(next(iter(empty_resident_registry.values()))[1]) == 1
    finally:
        store.close()
        dag.close()


def test_profile_swap_bumps_task_versions(tmp_path):
    store = VectorStore(tmp_path / "profile-swap.db")
    try:
        first = store.register_profile("shared", "provider-a", 3)
        assert store._data_version(first) == 0

        second = store.register_profile("shared", "provider-b", 3)
        assert store._data_version(first) == 1
        assert store._data_version(second) == 1

        reactivated = store.register_profile("shared", "provider-a", 3)
        assert reactivated == first
        assert store._data_version(first) == 2
        assert store._data_version(second) == 2
    finally:
        store.close()


def test_resident_count_growth_falls_back_without_matrix_overrun(
    tmp_path, monkeypatch, empty_resident_registry
):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "resident-count-race.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=1),
        bounded_scan_rows=1,
    )
    try:
        _seed_scan_corpus(
            dag,
            store,
            3,
            gold_vector=[1.0, 0.0, 0.0],
            filler_vector=[0.0, 1.0, 0.0],
        )
        actual_count = VectorStore._live_vector_count

        def stale_count(self, identity_hash, *, chunk):
            return actual_count(self, identity_hash, chunk=chunk) - 1

        monkeypatch.setattr(VectorStore, "_live_vector_count", stale_count)
        result = store.knn(
            [1.0, 0.0, 0.0], k=3, model="scan", full_scan=True
        )

        assert result.coverage == "full"
        assert result.scoring == "float32_exact"
        assert len(result) == 3
        assert empty_resident_registry == {}
    finally:
        store.close()
        dag.close()


def test_in_memory_resident_registry_namespaces_are_unique():
    first = VectorStore(":memory:")
    second = VectorStore(":memory:")
    try:
        assert first._resident_registry_db_path != second._resident_registry_db_path
        assert first._resident_registry_db_path.startswith(":memory:#")
        assert second._resident_registry_db_path.startswith(":memory:#")
    finally:
        first.close()
        second.close()


def test_resident_registry_evicts_lru_across_databases(
    tmp_path, monkeypatch, empty_resident_registry
):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    stores_and_dags = []
    builds: list[str] = []
    original = VectorStore._vector_rows_cursor

    def counted(store, identity_hash, *, chunk, candidate_ids=None):
        builds.append(store._resident_registry_db_path)
        return original(
            store,
            identity_hash,
            chunk=chunk,
            candidate_ids=candidate_ids,
        )

    monkeypatch.setattr(VectorStore, "_vector_rows_cursor", counted)
    try:
        for name in ("first", "second"):
            db_path = tmp_path / f"{name}.db"
            dag = SummaryDAG(db_path)
            store = VectorStore(
                db_path,
                config=LCMConfig(knn_resident_max_mb=1),
                bounded_scan_rows=1,
            )
            _seed_scan_corpus(
                dag,
                store,
                2,
                gold_vector=[1.0, 0.0, 0.0],
                filler_vector=[0.0, 1.0, 0.0],
            )
            store.knn_resident_max_bytes = 800
            stores_and_dags.append((store, dag))

        first, _ = stores_and_dags[0]
        second, _ = stores_and_dags[1]
        first.knn([1.0, 0.0, 0.0], k=1, model="scan", full_scan=True)
        second.knn([1.0, 0.0, 0.0], k=1, model="scan", full_scan=True)

        assert len(empty_resident_registry) == 1
        assert next(iter(empty_resident_registry))[0] == str(
            second.db_path.resolve()
        )
        assert len(first._resident_matrix_cache) == 0
        assert builds == [
            str(first.db_path.resolve()),
            str(second.db_path.resolve()),
        ]

        first.knn([1.0, 0.0, 0.0], k=1, model="scan", full_scan=True)
        assert next(iter(empty_resident_registry))[0] == str(
            first.db_path.resolve()
        )
        assert len(second._resident_matrix_cache) == 0
        assert builds[-1] == str(first.db_path.resolve())
    finally:
        for store, dag in stores_and_dags:
            store.close()
            dag.close()


def test_resident_budget_partitions_do_not_evict_other_store_budgets(
    tmp_path, monkeypatch, empty_resident_registry
):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    stores_and_dags = []
    try:
        for name, budget_mb in (("small", 1), ("large", 2)):
            db_path = tmp_path / f"{name}.db"
            dag = SummaryDAG(db_path)
            store = VectorStore(
                db_path,
                config=LCMConfig(knn_resident_max_mb=budget_mb),
                bounded_scan_rows=1,
            )
            _seed_scan_corpus(
                dag,
                store,
                2,
                gold_vector=[1.0, 0.0, 0.0],
                filler_vector=[0.0, 1.0, 0.0],
            )
            store.knn_resident_max_bytes = 800
            stores_and_dags.append((store, dag))

        for store, _ in stores_and_dags:
            result = store.knn(
                [1.0, 0.0, 0.0],
                k=1,
                model="scan",
                full_scan=True,
            )
            assert result.coverage == "full"

        assert len(empty_resident_registry) == 2
        assert {key[5] for key in empty_resident_registry} == {1, 2}
    finally:
        for store, dag in stores_and_dags:
            store.close()
            dag.close()


def test_resident_budget_counts_metadata_not_only_numpy_arrays(
    tmp_path, monkeypatch, empty_resident_registry
):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "resident-metadata-budget.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=1),
        bounded_scan_rows=1,
    )
    try:
        _seed_scan_corpus(
            dag,
            store,
            2,
            gold_vector=[1.0, 0.0, 0.0],
            filler_vector=[0.0, 1.0, 0.0],
        )
        store.knn_resident_max_bytes = 20
        result = store.knn(
            [1.0, 0.0, 0.0], k=2, model="scan", full_scan=True
        )

        assert result.coverage == "full"
        assert result.scoring == "float32_exact"
        assert empty_resident_registry == {}
    finally:
        store.close()
        dag.close()


def test_same_key_concurrent_cold_scans_build_one_resident_matrix(
    tmp_path, monkeypatch, empty_resident_registry
):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "resident-concurrent.db"
    dag = SummaryDAG(db_path)
    first = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=1),
        bounded_scan_rows=1,
    )
    second = None
    builds = 0
    builds_lock = threading.Lock()
    original = VectorStore._vector_rows_cursor

    def counted(store, identity_hash, *, chunk, candidate_ids=None):
        nonlocal builds
        with builds_lock:
            builds += 1
        return original(
            store,
            identity_hash,
            chunk=chunk,
            candidate_ids=candidate_ids,
        )

    monkeypatch.setattr(VectorStore, "_vector_rows_cursor", counted)
    try:
        _seed_scan_corpus(
            dag,
            first,
            3,
            gold_vector=[1.0, 0.0, 0.0],
            filler_vector=[0.0, 1.0, 0.0],
        )
        second = VectorStore(
            db_path,
            config=LCMConfig(knn_resident_max_mb=1),
            bounded_scan_rows=1,
        )
        barrier = threading.Barrier(2)
        results = []
        errors = []

        def recall(store):
            try:
                barrier.wait()
                results.append(
                    store.knn(
                        [1.0, 0.0, 0.0],
                        k=2,
                        model="scan",
                        full_scan=True,
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=recall, args=(first,)),
            threading.Thread(target=recall, args=(second,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert len(results) == 2
        assert list(results[0]) == list(results[1])
        assert builds == 1
        assert len(empty_resident_registry) == 1
        assert len(vector_store_module._RESIDENT_MATRIX_BUILD_LOCKS) == 0
    finally:
        first.close()
        if second is not None:
            second.close()
        dag.close()


def test_full_scan_max_rows_caps_the_scan_and_discloses_it(tmp_path):
    """scan_max_rows is the pathological-corpus escape hatch, and it discloses."""
    db_path = tmp_path / "full-scan-capped.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=2,
    )
    try:
        gold = _seed_scan_corpus(
            dag, store, 6, gold_vector=[1.0, 0.0, 0.0], filler_vector=[0.0, 1.0, 0.0]
        )

        result = store.knn(
            [1.0, 0.0, 0.0], k=1, model="scan", full_scan=True, scan_max_rows=2
        )

        assert result.coverage == "bounded"
        assert result.scanned == 2
        assert result.total == 6
        assert str(gold) not in {row[0] for row in result}
    finally:
        store.close()
        dag.close()


def test_full_scan_budget_stops_early_and_reports_bounded(tmp_path, monkeypatch):
    """An exhausted latency budget is the only thing that truncates a default
    scan, and it degrades to the same disclosed 'bounded' coverage."""
    db_path = tmp_path / "full-scan-budget.db"
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=2,
    )
    try:
        gold = _seed_scan_corpus(
            dag, store, 6, gold_vector=[1.0, 0.0, 0.0], filler_vector=[0.0, 1.0, 0.0]
        )
        # Spend the budget while loading the first batch, so the scan stops with
        # 4 of 6 vectors unscored. Candidate enumeration itself stays at t=0;
        # its separate regression below proves that it shares this budget.
        with monkeypatch.context() as clock_patch:
            now = [0.0]
            original_load = VectorStore._vectorized_batch
            count_calls = 0
            original_count = store._count_embedded_vectors

            def counted_total(*args, **kwargs):
                nonlocal count_calls
                count_calls += 1
                return original_count(*args, **kwargs)

            def timed_load(np, rows, dim, dtype):
                loaded = original_load(np, rows, dim, dtype)
                now[0] = 1.0
                return loaded

            clock_patch.setattr(
                vector_store_module, "_monotonic", lambda: now[0]
            )
            clock_patch.setattr(
                VectorStore, "_vectorized_batch", staticmethod(timed_load)
            )
            clock_patch.setattr(
                store, "_count_embedded_vectors", counted_total
            )
            result = store.knn(
                [1.0, 0.0, 0.0],
                k=1,
                model="scan",
                full_scan=True,
                scan_budget_s=0.5,
            )

        assert result.coverage == "bounded"
        assert result.scanned == 2
        assert result.total == 6
        assert count_calls == 1
        assert str(gold) not in {row[0] for row in result}
    finally:
        store.close()
        dag.close()


def test_full_scan_deadline_on_final_batch_reports_complete_total(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "full-scan-final-batch-deadline.db"
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=2,
    )
    try:
        _seed_scan_corpus(
            dag,
            store,
            4,
            gold_vector=[1.0, 0.0, 0.0],
            filler_vector=[0.0, 1.0, 0.0],
        )
        now = [0.0]
        loads = 0
        original_load = VectorStore._vectorized_batch

        def timed_load(np, rows, dim, dtype):
            nonlocal loads
            loaded = original_load(np, rows, dim, dtype)
            loads += 1
            if loads == 2:
                now[0] = 2.0
            return loaded

        monkeypatch.setattr(vector_store_module, "_monotonic", lambda: now[0])
        monkeypatch.setattr(
            VectorStore, "_vectorized_batch", staticmethod(timed_load)
        )

        def count_should_not_run(*args, **kwargs):
            raise AssertionError(
                "completed final-batch scan must not run COUNT(*)"
            )

        monkeypatch.setattr(
            store,
            "_count_embedded_vectors",
            count_should_not_run,
        )

        result = store.knn(
            [1.0, 0.0, 0.0],
            k=1,
            model="scan",
            full_scan=True,
            scan_budget_s=0.0,
            deadline=1.0,
        )

        assert loads == 2
        assert result.coverage == "full"
        assert result.scanned == 4
        assert result.total == 4
    finally:
        store.close()
        dag.close()


def test_full_scan_deadline_on_final_batch_without_numpy_reports_complete_total(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "full-scan-final-batch-deadline-nonumpy.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=2,
    )
    try:
        _seed_scan_corpus(
            dag,
            store,
            4,
            gold_vector=[1.0, 0.0, 0.0],
            filler_vector=[0.0, 1.0, 0.0],
        )
        now = [0.0]
        loads = 0
        original_load = store._load_vectors_for_ids

        def unavailable():
            raise ImportError("numpy not installed")

        def timed_load(*args, **kwargs):
            nonlocal loads
            loaded = original_load(*args, **kwargs)
            loads += 1
            if loads == 2:
                now[0] = 2.0
            return loaded

        monkeypatch.setattr(vector_store_module, "_load_numpy", unavailable)
        monkeypatch.setattr(vector_store_module, "_monotonic", lambda: now[0])
        monkeypatch.setattr(store, "_load_vectors_for_ids", timed_load)

        def count_should_not_run(*args, **kwargs):
            raise AssertionError(
                "completed final-batch scan must not run COUNT(*)"
            )

        monkeypatch.setattr(
            store,
            "_count_embedded_vectors",
            count_should_not_run,
        )

        result = store.knn(
            [1.0, 0.0, 0.0],
            k=1,
            model="scan",
            full_scan=True,
            scan_budget_s=0.0,
            deadline=1.0,
        )

        assert loads == 2
        assert result.coverage == "full"
        assert result.scanned == 4
        assert result.total == 4
    finally:
        store.close()
        dag.close()


def test_streaming_truncation_keeps_recency_order_below_host_limit(
    tmp_path, monkeypatch
):
    if not hasattr(sqlite3.Connection, "setlimit"):
        pytest.skip("sqlite runtime does not expose setlimit")
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "streaming-recency-host-limit.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=2,
    )
    try:
        gold = _seed_scan_corpus(
            dag,
            store,
            12,
            gold_vector=[1.0, 0.0, 0.0],
            filler_vector=[0.0, 1.0, 0.0],
        )
        identity = store.capture_identity("scan").identity_hash
        expected_newest = store._bounded_candidate_ids(
            identity,
            since=None,
            until=None,
            conversation_ids=None,
            source=None,
            limit=vector_store_module._SCAN_ALL_ROWS,
        )[0]
        store.connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 8)
        now = [0.0]
        original_load = VectorStore._vectorized_batch

        def timed_load(np, rows, dim, dtype):
            loaded = original_load(np, rows, dim, dtype)
            now[0] = 1.0
            return loaded

        monkeypatch.setattr(vector_store_module, "_monotonic", lambda: now[0])
        monkeypatch.setattr(
            VectorStore, "_vectorized_batch", staticmethod(timed_load)
        )
        statements: list[str] = []
        store.connection.set_trace_callback(statements.append)
        try:
            result = store.knn(
                [0.0, 1.0, 0.0],
                k=1,
                model="scan",
                full_scan=True,
                scan_budget_s=0.5,
            )
        finally:
            store.connection.set_trace_callback(None)

        assert result.coverage == "bounded"
        assert result.scanned == 2
        assert result[0][0] == expected_newest
        assert str(gold) not in {row[0] for row in result}
        assert not any("_lcm_ordered_id_scratch" in sql for sql in statements)
    finally:
        store.close()
        dag.close()


def test_full_scan_budget_includes_candidate_enumeration(tmp_path, monkeypatch):
    db_path = tmp_path / "full-scan-enumeration-budget.db"
    dag = SummaryDAG(db_path)
    store = VectorStore(db_path, bounded_scan_rows=2)
    try:
        _seed_scan_corpus(
            dag,
            store,
            6,
            gold_vector=[1.0, 0.0, 0.0],
            filler_vector=[0.0, 1.0, 0.0],
        )
        now = [0.0]
        original = VectorStore._bounded_candidate_ids

        def slow_enumeration(self, *args, **kwargs):
            candidate_ids = original(self, *args, **kwargs)
            now[0] = 1.0
            return candidate_ids

        monkeypatch.setattr(vector_store_module, "_monotonic", lambda: now[0])
        monkeypatch.setattr(
            VectorStore, "_bounded_candidate_ids", slow_enumeration
        )

        result = store.knn(
            [1.0, 0.0, 0.0],
            k=1,
            model="scan",
            full_scan=True,
            scan_budget_s=0.5,
        )

        assert result.coverage == "bounded"
        assert result.scanned == 0
        assert result.total == 6
        assert result == []
    finally:
        store.close()
        dag.close()


def test_full_scan_absolute_deadline_stops_between_batches(tmp_path, monkeypatch):
    """The operation deadline remains a hard stop when the relative scan budget
    is disabled (zero), so recall cannot start another full batch after expiry."""
    db_path = tmp_path / "full-scan-deadline.db"
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    dag = SummaryDAG(db_path)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=2,
    )
    try:
        gold = _seed_scan_corpus(
            dag, store, 6, gold_vector=[1.0, 0.0, 0.0], filler_vector=[0.0, 1.0, 0.0]
        )
        with monkeypatch.context() as clock_patch:
            now = [0.0]
            original_load = VectorStore._vectorized_batch

            def count_after_deadline(*args, **kwargs):
                raise AssertionError("deadline expiry must not start COUNT(*)")

            def timed_load(np, rows, dim, dtype):
                loaded = original_load(np, rows, dim, dtype)
                now[0] = 2.0
                return loaded

            clock_patch.setattr(
                vector_store_module, "_monotonic", lambda: now[0]
            )
            clock_patch.setattr(
                VectorStore, "_vectorized_batch", staticmethod(timed_load)
            )
            clock_patch.setattr(
                store, "_count_embedded_vectors", count_after_deadline
            )
            result = store.knn(
                [1.0, 0.0, 0.0],
                k=1,
                model="scan",
                full_scan=True,
                scan_budget_s=0.0,
                deadline=1.0,
            )

        assert result.coverage == "bounded"
        assert result.scanned == 2
        assert result.total is None
        assert str(gold) not in {row[0] for row in result}
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
    assert defaults.knn_resident_max_mb == 128

    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "true")
    monkeypatch.setenv("LCM_EMBEDDING_BOUNDED_SCAN_ROWS", "123")
    monkeypatch.setenv("LCM_KNN_RESIDENT_MAX_MB", "64")
    configured = LCMConfig.from_env()
    assert configured.embeddings_enabled is True
    assert configured.embedding_bounded_scan_rows == 123
    assert configured.knn_resident_max_mb == 64

    store = VectorStore(":memory:")
    try:
        assert store.bounded_scan_rows == 123
        assert store.knn_resident_max_bytes == 64 * 1024 * 1024
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

        def counted(np, identity_hash, dim, ids, dtype="float32", cache=True):
            loaded.append(len(ids))
            return original(np, identity_hash, dim, ids, dtype, cache=cache)

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
