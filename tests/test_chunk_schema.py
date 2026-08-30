from __future__ import annotations

import sqlite3

from hermes_lcm.db_bootstrap import (
    chunk_schema_missing,
    ensure_chunk_tables,
    ensure_embedding_tables,
    verify_chunk_schema,
)


def _fresh_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


class TestChunkSchema:
    def test_ensure_is_idempotent_and_verifies_clean(self):
        conn = _fresh_conn()
        ensure_embedding_tables(conn)
        ensure_chunk_tables(conn)
        ensure_chunk_tables(conn)  # second run must be a no-op
        assert chunk_schema_missing(conn) == set()
        assert verify_chunk_schema(conn) == []

    def test_missing_reports_absent_objects(self):
        conn = _fresh_conn()
        missing = chunk_schema_missing(conn)
        assert "lcm_chunk_meta" in missing
        assert "lcm_chunk_vectors" in missing
        assert "idx_lcm_chunk_meta_identity_embedded_at" in missing

    def test_verify_rejects_wrong_shaped_table(self):
        conn = _fresh_conn()
        conn.execute(
            "CREATE TABLE lcm_chunk_meta (chunk_id TEXT PRIMARY KEY, wrong INTEGER)"
        )
        conn.execute("CREATE TABLE lcm_chunk_vectors (chunk_id TEXT, identity_hash TEXT, vec BLOB NOT NULL, PRIMARY KEY(chunk_id, identity_hash))")
        conn.execute("CREATE TABLE lcm_chunk_binary (chunk_id TEXT, identity_hash TEXT, bits BLOB NOT NULL, PRIMARY KEY(chunk_id, identity_hash))")
        conn.execute(
            "CREATE INDEX idx_lcm_chunk_meta_identity_embedded_at "
            "ON lcm_chunk_meta(chunk_id)"
        )
        conn.execute("CREATE INDEX idx_lcm_chunk_meta_store ON lcm_chunk_meta(chunk_id)")
        errors = verify_chunk_schema(conn)
        assert any("malformed" in error for error in errors)

    def test_pragma_shapes_match_declared(self):
        conn = _fresh_conn()
        ensure_embedding_tables(conn)
        ensure_chunk_tables(conn)
        cols = [
            row["name"]
            for row in conn.execute("PRAGMA table_info(lcm_chunk_meta)").fetchall()
        ]
        assert cols == [
            "chunk_id",
            "identity_hash",
            "store_id",
            "chunk_index",
            "char_start",
            "char_end",
            "token_estimate",
            "embedded_at",
            "archived",
            "access_scope",
        ]

    def test_partial_index_predicate_is_present(self):
        conn = _fresh_conn()
        ensure_embedding_tables(conn)
        ensure_chunk_tables(conn)
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' "
            "AND name='idx_lcm_chunk_meta_identity_embedded_at'"
        ).fetchone()
        assert "where archived = 0" in " ".join(row["sql"].lower().split())
