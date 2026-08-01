from __future__ import annotations

import sqlite3

import pytest

import hermes_lcm.vector_store as vector_store_module
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


def _write(store, chunk_id, store_id, chunk_index, vec):
    store.record_chunk_embedding(
        chunk_id,
        MODEL,
        vec,
        store_id=store_id,
        chunk_index=chunk_index,
        char_start=0,
        char_end=10,
        token_estimate=5,
        identity=_chunk_identity(),
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

    def test_full_scan_reaches_best_chunk_outside_recency_bound(self, tmp_path):
        db_path = tmp_path / "lcm.db"
        _seed_messages(
            db_path,
            [
                (1, "s", "history", "user", "old exact match", 1.0),
                (2, "s", "history", "user", "new distractor", 2.0),
            ],
        )
        vs = VectorStore(db_path, bounded_scan_rows=1)
        vs.register_profile(MODEL, PROVIDER, DIM, task="chunk")
        try:
            _write(vs, "1:0", 1, 0, [1.0, 0.0, 0.0, 0.0])
            _write(vs, "2:0", 2, 0, [0.0, 1.0, 0.0, 0.0])

            bounded = vs.knn_chunks(
                [1.0, 0.0, 0.0, 0.0], k=1, model=MODEL, provider=PROVIDER
            )
            exhaustive = vs.knn_chunks(
                [1.0, 0.0, 0.0, 0.0],
                k=1,
                model=MODEL,
                provider=PROVIDER,
                full_scan=True,
            )

            assert bounded.coverage == "bounded"
            assert [row[0] for row in bounded] == ["2:0"]
            assert exhaustive.coverage == "full"
            assert exhaustive.scanned == exhaustive.total == 2
            assert [row[0] for row in exhaustive] == ["1:0"]
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
        archived = store.archive_chunks_for_messages([10])
        assert archived == 1
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
