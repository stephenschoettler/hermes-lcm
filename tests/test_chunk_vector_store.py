from __future__ import annotations

import math
import sqlite3

import pytest

import hermes_lcm.vector_store as vector_store_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore

MODEL = "voyage-context-4"
PROVIDER = "voyage"
DIM = 4


def _seed_messages(db_path, rows):
    """Create the messages columns the chunk KNN filters need, then insert rows."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            store_id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            source TEXT DEFAULT '',
            role TEXT NOT NULL,
            content TEXT,
            timestamp REAL NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO messages(store_id, session_id, source, role, content, timestamp) "
        "VALUES(?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def _chunk_identity():
    return EmbeddingIdentity.canonical(
        PROVIDER, MODEL, "", DIM, "float32", "little", "chunk"
    )


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "lcm.db"
    _seed_messages(
        db_path,
        [
            (10, "sess-a", "history", "user", "first message", 100.0),
            (11, "sess-a", "history", "assistant", "second message", 200.0),
            (12, "sess-b", "other", "tool", "tool output", 300.0),
        ],
    )
    vs = VectorStore(db_path)
    vs.register_profile(MODEL, PROVIDER, DIM, task="chunk")
    yield vs
    vs.close()


def _write(store, chunk_id, store_id, chunk_index, vec, *, identity=None):
    store.record_chunk_embedding(
        chunk_id,
        MODEL,
        vec,
        store_id=store_id,
        chunk_index=chunk_index,
        char_start=0,
        char_end=10,
        token_estimate=5,
        identity=identity or _chunk_identity(),
    )


class TestChunkWriteAndKnn:
    def test_write_and_retrieve(self, store):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "11:0", 11, 0, [0.0, 1.0, 0.0, 0.0])
        result = store.knn_chunks([1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER)
        assert result.coverage == "full"
        assert result[0][0] == "10:0"
        assert {row[0] for row in result} == {"10:0", "11:0"}
        assert all(row[2] == "chunk" for row in result)

    def test_unbackfilled_identity_returns_none(self, store):
        result = store.knn_chunks([1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER)
        assert result.coverage == "none"

    def test_session_filter(self, store):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "12:0", 12, 0, [1.0, 0.0, 0.0, 0.0])
        result = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER,
            conversation_ids=["sess-b"],
        )
        assert {row[0] for row in result} == {"12:0"}

    def test_source_filter(self, store):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "12:0", 12, 0, [1.0, 0.0, 0.0, 0.0])
        result = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER, source="other",
        )
        assert {row[0] for row in result} == {"12:0"}

    def test_recency_window(self, store):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "12:0", 12, 0, [1.0, 0.0, 0.0, 0.0])
        result = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER, since=250.0,
        )
        assert {row[0] for row in result} == {"12:0"}

    def test_bounded_coverage(self, tmp_path):
        db_path = tmp_path / "lcm.db"
        _seed_messages(
            db_path,
            [(i, "s", "history", "user", "m", float(i)) for i in range(5)],
        )
        vs = VectorStore(db_path, bounded_scan_rows=2)
        vs.register_profile(MODEL, PROVIDER, DIM, task="chunk")
        try:
            for i in range(5):
                _write(vs, f"{i}:0", i, 0, [1.0, 0.0, 0.0, 0.0])
            result = vs.knn_chunks([1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER)
            assert result.coverage == "bounded"
            assert len(result) <= 2
        finally:
            vs.close()

    def test_numpy_absent_reports_full_when_scan_covers_corpus(
        self, store, monkeypatch
    ):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "11:0", 11, 0, [0.0, 1.0, 0.0, 0.0])

        def unavailable():
            raise ImportError("numpy not installed")

        monkeypatch.setattr(vector_store_module, "_load_numpy", unavailable)
        result = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER
        )

        assert result.coverage == "full"
        assert {row[0] for row in result} == {"10:0", "11:0"}


class TestArchiveOnPurge:
    def test_archive_drops_from_knn(self, store):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "11:0", 11, 0, [0.0, 1.0, 0.0, 0.0])
        identity = str(store._current_chunk_profile()["identity_hash"])
        before_version = store._data_version(identity)
        archived = store.archive_chunks_for_messages([10])
        assert archived == 1
        assert store._data_version(identity) > before_version
        result = store.knn_chunks([1.0, 0.0, 0.0, 0.0], k=5, model=MODEL, provider=PROVIDER)
        assert {row[0] for row in result} == {"11:0"}

    def test_archive_noop_without_schema(self, tmp_path):
        db_path = tmp_path / "lcm.db"
        _seed_messages(db_path, [(1, "s", "", "user", "m", 1.0)])
        vs = VectorStore(db_path)
        try:
            assert vs.archive_chunks_for_messages([1]) == 0
        finally:
            vs.close()

    def test_archive_batch_on_connection(self, store):
        _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
        conn = store.connection
        archived = VectorStore.archive_chunks_for_messages_on_connection(conn, [10])
        assert archived == 1


class TestCoexistence:
    def test_summary_and_chunk_profiles_coexist(self, store):
        # Register a summary profile alongside the existing chunk profile.
        store.register_profile("summary-model", PROVIDER, DIM, task="summary")
        chunk = store._current_chunk_profile()
        summary = store._current_profile()
        assert chunk is not None and summary is not None
        assert chunk["task"] == "chunk"
        assert summary["task"] == "summary"
        assert chunk["identity_hash"] != summary["identity_hash"]
        # Both remain active.
        assert int(chunk["active"]) == 1
        assert int(summary["active"]) == 1


def test_chunk_vectorized_multibatch_ranking_matches_legacy_loader(
    tmp_path, monkeypatch
):
    numpy = pytest.importorskip("numpy")
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "chunk-r1-parity.db"
    _seed_messages(
        db_path,
        [
            (i, "s", "history", "user", "m", float(i))
            for i in range(6)
        ],
    )
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=2,
    )
    store.register_profile(MODEL, PROVIDER, DIM, task="chunk")
    try:
        for i, vec in enumerate(
            (
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            )
        ):
            _write(store, f"{i}:0", i, 0, vec)
        identity = str(store._current_chunk_profile()["identity_hash"])
        ids = store._bounded_chunk_candidate_ids(
            identity,
            since=None,
            until=None,
            conversation_ids=None,
            source=None,
            limit=vector_store_module._SCAN_ALL_ROWS,
        )
        old_rowids: list[int] = []
        old_ids: list[str] = []
        old_scores: list[float] = []
        query = numpy.asarray([1.0, 0.0, 0.0, 0.0], dtype=numpy.float32)
        for start in range(0, len(ids), 2):
            with store._temp_id_table(ids[start:start + 2]) as table:
                rows = store.connection.execute(
                    f"""
                    SELECT v.rowid, v.chunk_id, v.vec
                    FROM {table} t
                    JOIN lcm_chunk_vectors v
                      ON v.chunk_id = t.id AND v.identity_hash = ?
                    JOIN lcm_chunk_meta m
                      ON m.chunk_id = v.chunk_id
                     AND m.identity_hash = v.identity_hash
                    WHERE m.archived = 0
                    """,
                    (identity,),
                ).fetchall()
            vectors = [
                list(store._decode_stored_vec(bytes(row["vec"]), DIM, "float32"))
                for row in rows
            ]
            scores = numpy.asarray(vectors, dtype=numpy.float32) @ query
            old_rowids.extend(int(row["rowid"]) for row in rows)
            old_ids.extend(str(row["chunk_id"]) for row in rows)
            old_scores.extend(float(score) for score in scores)
        expected = store._ranked(
            old_rowids,
            old_ids,
            ["chunk"] * len(old_ids),
            old_scores,
            limit=4,
        )

        actual = store.knn_chunks(
            query.tolist(),
            k=4,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )

        assert actual.coverage == "full"
        assert list(actual) == expected
    finally:
        store.close()


def test_float32_residency_overlaps_exact_top_k_with_quantized_scores(
    tmp_path, monkeypatch
):
    numpy = pytest.importorskip("numpy")
    dim = 32
    count = 64
    k = 12
    db_path = tmp_path / "chunk-float-resident.db"
    _seed_messages(
        db_path,
        [
            (i, "s", "history", "user", "m", float(i))
            for i in range(count)
        ],
    )
    rng = numpy.random.default_rng(171)
    vectors = rng.standard_normal((count, dim)).astype(numpy.float32)
    vectors /= numpy.linalg.norm(vectors, axis=1, keepdims=True)
    query = rng.standard_normal(dim).astype(numpy.float32)
    query /= numpy.linalg.norm(query)

    exact_store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=7,
    )
    exact_store.register_profile(MODEL, PROVIDER, dim, task="chunk")
    identity = EmbeddingIdentity.canonical(
        PROVIDER, MODEL, "", dim, "float32", "little", "chunk"
    )
    try:
        for index, vector in enumerate(vectors):
            _write(
                exact_store,
                f"{index}:0",
                index,
                0,
                vector.tolist(),
                identity=identity,
            )
        monkeypatch.setattr(
            vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0
        )
        exact = exact_store.knn_chunks(
            query.tolist(),
            k=k,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )
    finally:
        exact_store.close()

    resident_store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=1),
        bounded_scan_rows=7,
    )
    try:
        resident = resident_store.knn_chunks(
            query.tolist(),
            k=k,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )

        exact_ids = {row[0] for row in exact}
        resident_ids = {row[0] for row in resident}
        assert exact.coverage == resident.coverage == "full"
        assert exact.scoring == "float32_exact"
        assert resident.scoring == "int8_quantized"
        assert len(exact_ids & resident_ids) / k >= 0.9
        exact_scores = dict((row[0], row[1]) for row in exact)
        assert any(
            not math.isclose(
                exact_scores[row[0]], row[1], rel_tol=0.0, abs_tol=0.0
            )
            for row in resident
            if row[0] in exact_scores
        )
        assert len(resident_store._resident_matrix_cache) == 1
    finally:
        resident_store.close()


def test_size_aware_scan_routes_below_and_at_threshold(tmp_path, monkeypatch):
    db_path = tmp_path / "chunk-size-route.db"
    _seed_messages(
        db_path,
        [
            (i, "s", "history", "user", "m", float(i))
            for i in range(4)
        ],
    )
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 4)
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=2,
    )
    store.register_profile(MODEL, PROVIDER, DIM, task="chunk")
    simple_calls: list[int] = []
    streaming_calls: list[int] = []
    original_simple = store._load_chunk_vectors_for_ids
    original_streaming = store._scan_vectorized_ranked

    def counted_simple(identity_hash, dim, ids, dtype="float32"):
        simple_calls.append(len(ids))
        return original_simple(identity_hash, dim, ids, dtype)

    def counted_streaming(**kwargs):
        streaming_calls.append(len(kwargs["candidate_ids"]))
        return original_streaming(**kwargs)

    monkeypatch.setattr(store, "_load_chunk_vectors_for_ids", counted_simple)
    monkeypatch.setattr(store, "_scan_vectorized_ranked", counted_streaming)
    try:
        for index in range(3):
            _write(
                store,
                f"{index}:0",
                index,
                0,
                [1.0, float(index), 0.0, 0.0],
            )

        below = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0],
            k=3,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )
        assert below.coverage == "full"
        assert simple_calls
        assert streaming_calls == []

        simple_calls.clear()
        _write(
            store,
            "3:0",
            3,
            0,
            [1.0, 3.0, 0.0, 0.0],
        )
        at_threshold = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0],
            k=4,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )
        assert at_threshold.coverage == "full"
        assert simple_calls == []
        assert streaming_calls == [4]
    finally:
        store.close()


def test_int8_residency_reuses_matrix_and_write_forces_full_reload(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "chunk-resident.db"
    _seed_messages(
        db_path,
        [
            (i, "s", "history", "user", "m", float(i))
            for i in range(4)
        ],
    )
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=1),
        bounded_scan_rows=2,
    )
    identity = EmbeddingIdentity.canonical(
        PROVIDER, MODEL, "", DIM, "int8", "little", "chunk"
    )
    store.register_profile(
        MODEL, PROVIDER, DIM, dtype="int8", task="chunk"
    )
    try:
        for i, vec in enumerate(
            (
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            )
        ):
            store.record_chunk_embedding(
                f"{i}:0",
                MODEL,
                vec,
                store_id=i,
                chunk_index=0,
                char_start=0,
                char_end=1,
                token_estimate=1,
                identity=identity,
            )
        calls: list[bool] = []
        original = store._vector_rows_cursor

        def counted(identity_hash, *, chunk, candidate_ids=None):
            calls.append(chunk)
            return original(
                identity_hash,
                chunk=chunk,
                candidate_ids=candidate_ids,
            )

        monkeypatch.setattr(store, "_vector_rows_cursor", counted)
        cold = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0],
            k=2,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )
        warm = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0],
            k=2,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )

        assert cold.coverage == warm.coverage == "full"
        assert cold.scoring == warm.scoring == "int8_quantized"
        assert list(cold) == list(warm)
        rows = store.connection.execute(
            """
            SELECT v.rowid, v.chunk_id, v.vec
            FROM lcm_chunk_vectors v
            JOIN lcm_chunk_meta m
              ON m.chunk_id = v.chunk_id
             AND m.identity_hash = v.identity_hash
            JOIN messages msg ON msg.store_id = m.store_id
            WHERE v.identity_hash = ? AND m.archived = 0
            """,
            (identity.identity_hash,),
        ).fetchall()
        query = [1.0, 0.0, 0.0, 0.0]
        decoded = [
            vector_store_module._decode_int8_vector(bytes(row["vec"]), DIM)
            for row in rows
        ]
        legacy_scores = [
            sum(value * query_value for value, query_value in zip(vector, query))
            for vector in decoded
        ]
        expected = store._ranked(
            [int(row["rowid"]) for row in rows],
            [str(row["chunk_id"]) for row in rows],
            ["chunk"] * len(rows),
            legacy_scores,
            2,
        )
        assert [row[0] for row in cold] == [row[0] for row in expected]
        assert [row[1] for row in cold] == pytest.approx(
            [row[1] for row in expected], abs=1e-7
        )
        assert calls == [True]
        assert len(store._resident_matrix_cache) == 1

        store.record_chunk_embedding(
            "3:0",
            MODEL,
            [0.9, 0.1, 0.0, 0.0],
            store_id=3,
            chunk_index=0,
            char_start=0,
            char_end=1,
            token_estimate=1,
            identity=identity,
        )
        assert len(store._resident_matrix_cache) == 0
        reloaded = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0],
            k=4,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )
        assert reloaded.coverage == "full"
        assert calls == [True, True]
        assert {row[0] for row in reloaded} == {
            "0:0", "1:0", "2:0", "3:0"
        }
    finally:
        store.close()


def test_warm_chunk_residency_drops_orphans_when_message_is_deleted(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "chunk-resident-orphan.db"
    _seed_messages(
        db_path,
        [
            (0, "s", "history", "user", "m", 0.0),
            (1, "s", "history", "user", "m", 1.0),
        ],
    )
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=1),
        bounded_scan_rows=1,
    )
    store.register_profile(MODEL, PROVIDER, DIM, task="chunk")
    try:
        _write(store, "0:0", 0, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "1:0", 1, 0, [0.0, 1.0, 0.0, 0.0])
        cold = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0],
            k=2,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )
        assert {row[0] for row in cold} == {"0:0", "1:0"}

        identity = str(store._current_chunk_profile()["identity_hash"])
        before_version = store._data_version(identity)
        store.connection.execute("DELETE FROM messages WHERE store_id = 0")
        assert store._data_version(identity) == before_version + 1
        warm = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0],
            k=2,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )
        assert warm.coverage == "full"
        assert [row[0] for row in warm] == ["1:0"]
    finally:
        store.close()


def test_message_delete_bumps_only_chunk_profiles_for_affected_store_id(tmp_path):
    db_path = tmp_path / "chunk-delete-scope.db"
    _seed_messages(
        db_path,
        [
            (0, "s", "history", "user", "m", 0.0),
            (1, "s", "history", "user", "m", 1.0),
            (2, "s", "history", "user", "m", 2.0),
        ],
    )
    store = VectorStore(db_path)
    first_identity = EmbeddingIdentity.canonical(
        "provider-a", MODEL, "", DIM, "float32", "little", "chunk"
    )
    second_identity = EmbeddingIdentity.canonical(
        "provider-b", MODEL, "", DIM, "float32", "little", "chunk"
    )
    try:
        store.register_profile(MODEL, "provider-a", DIM, task="chunk")
        _write(
            store,
            "0:0",
            0,
            0,
            [1.0, 0.0, 0.0, 0.0],
            identity=first_identity,
        )
        store.register_profile(MODEL, "provider-b", DIM, task="chunk")
        _write(
            store,
            "1:0",
            1,
            0,
            [0.0, 1.0, 0.0, 0.0],
            identity=second_identity,
        )

        before_first = store._data_version(first_identity.identity_hash)
        before_second = store._data_version(second_identity.identity_hash)
        store.connection.execute("DELETE FROM messages WHERE store_id = 2")
        assert store._data_version(first_identity.identity_hash) == before_first
        assert store._data_version(second_identity.identity_hash) == before_second

        store.connection.execute("DELETE FROM messages WHERE store_id = 0")
        assert (
            store._data_version(first_identity.identity_hash)
            == before_first + 1
        )
        assert store._data_version(second_identity.identity_hash) == before_second
    finally:
        store.close()


def test_warm_chunk_residency_invalidates_when_vector_is_deleted(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "chunk-resident-delete.db"
    _seed_messages(
        db_path,
        [
            (0, "s", "history", "user", "m", 0.0),
            (1, "s", "history", "user", "m", 1.0),
        ],
    )
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=1),
        bounded_scan_rows=1,
    )
    store.register_profile(MODEL, PROVIDER, DIM, task="chunk")
    try:
        _write(store, "0:0", 0, 0, [1.0, 0.0, 0.0, 0.0])
        _write(store, "1:0", 1, 0, [0.0, 1.0, 0.0, 0.0])
        cold = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0],
            k=2,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )
        assert {row[0] for row in cold} == {"0:0", "1:0"}

        identity = str(store._current_chunk_profile()["identity_hash"])
        before_version = store._data_version(identity)
        store.connection.execute(
            "DELETE FROM lcm_chunk_vectors "
            "WHERE chunk_id = '0:0' AND identity_hash = ?",
            (identity,),
        )
        assert store._data_version(identity) == before_version + 1

        warm = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0],
            k=2,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )
        assert warm.coverage == "full"
        assert [row[0] for row in warm] == ["1:0"]
    finally:
        store.close()


def test_chunk_deadline_on_final_batch_reports_total_without_count(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "chunk-final-batch-deadline.db"
    _seed_messages(
        db_path,
        [
            (index, "s", "history", "user", "m", float(index))
            for index in range(4)
        ],
    )
    store = VectorStore(
        db_path,
        config=LCMConfig(knn_resident_max_mb=0),
        bounded_scan_rows=2,
    )
    store.register_profile(MODEL, PROVIDER, DIM, task="chunk")
    try:
        for index in range(4):
            _write(
                store,
                f"{index}:0",
                index,
                0,
                [1.0, float(index), 0.0, 0.0],
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

        def count_should_not_run(*args, **kwargs):
            raise AssertionError(
                "completed final-batch chunk scan must not run COUNT(*)"
            )

        monkeypatch.setattr(vector_store_module, "_monotonic", lambda: now[0])
        monkeypatch.setattr(
            VectorStore,
            "_vectorized_batch",
            staticmethod(timed_load),
        )
        monkeypatch.setattr(
            store,
            "_count_embedded_vectors",
            count_should_not_run,
        )

        result = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0],
            k=1,
            model=MODEL,
            provider=PROVIDER,
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


def test_int8_over_resident_budget_falls_back_to_exact_r1(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(vector_store_module, "_FAST_SCAN_STREAMING_MIN_ROWS", 0)
    db_path = tmp_path / "chunk-over-budget.db"
    _seed_messages(
        db_path,
        [
            (i, "s", "history", "user", "m", float(i))
            for i in range(3)
        ],
    )
    store = VectorStore(
        db_path,
        bounded_scan_rows=1,
    )
    identity = EmbeddingIdentity.canonical(
        PROVIDER, MODEL, "", DIM, "int8", "little", "chunk"
    )
    store.register_profile(
        MODEL, PROVIDER, DIM, dtype="int8", task="chunk"
    )
    try:
        for i in range(3):
            store.record_chunk_embedding(
                f"{i}:0",
                MODEL,
                [1.0, float(i), 0.0, 0.0],
                store_id=i,
                chunk_index=0,
                char_start=0,
                char_end=1,
                token_estimate=1,
                identity=identity,
            )
        store.knn_resident_max_bytes = 2 * (DIM + 4)

        result = store.knn_chunks(
            [1.0, 0.0, 0.0, 0.0],
            k=3,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
        )

        assert result.coverage == "full"
        assert len(result) == 3
        assert len(store._resident_matrix_cache) == 0
    finally:
        store.close()


def test_exact_scan_does_not_overstate_coverage_for_unscorable_live_vector(
    store,
):
    _write(store, "10:0", 10, 0, [1.0, 0.0, 0.0, 0.0])
    _write(store, "11:0", 11, 0, [0.0, 1.0, 0.0, 0.0])
    store.connection.execute(
        "UPDATE lcm_chunk_vectors SET vec = X'00' WHERE chunk_id = '11:0'"
    )

    result = store.knn_chunks(
        [1.0, 0.0, 0.0, 0.0],
        k=2,
        model=MODEL,
        provider=PROVIDER,
        full_scan=True,
    )

    assert result.coverage == "bounded"
    assert result.scanned == 1
    assert result.total == 2
