"""int8/Matryoshka storage + two-stage (binary Hamming -> int8 rescore) KNN.

SPEC C1. These exercise the additive, default-off int8 storage dtype and the
full-corpus two-stage KNN. The legacy float32 path is covered by the existing
vector-store suites (which must stay green); here we assert the new int8 identity
is distinct, the sign-bit prescreen is written and consulted, coverage='full_approx'
reported, and stage-1 recall@M meets the spec bar on a synthetic 5k set.
"""
from __future__ import annotations

import sqlite3
import struct

import numpy as np
import hermes_lcm.vector_store as vector_store_module

from hermes_lcm.vector_store import (
    EmbeddingIdentity,
    VectorStore,
    _decode_int8_vector,
    _encode_int8_vector,
    _pack_sign_bits,
    _POPCOUNT_TABLE,
)

MODEL = "voyage-context-4"
PROVIDER = "voyage"


def _expire_after_first_prescreen_batch(deadline, scanned_rows):
    """Position-keyed deadline stub: expire once the first prescreen batch
    has completed (scanned_rows >= 1), regardless of clock call patterns."""
    return deadline is not None and scanned_rows >= 1


def _seed_messages(db_path, count, *, session="s", source="hist", ts0=0.0):
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
        [(i, session, source, "user", "m", ts0 + i) for i in range(count)],
    )
    conn.commit()
    conn.close()


def _int8_identity(dim):
    return EmbeddingIdentity.canonical(
        PROVIDER, MODEL, "", dim, "int8", "little", "chunk"
    )


def _unit(rng, dim):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# -- Quantization round-trip -------------------------------------------------


def test_int8_roundtrip_is_high_fidelity():
    rng = np.random.default_rng(0)
    worst = 1.0
    for _ in range(100):
        v = _unit(rng, 256)
        decoded = np.asarray(_decode_int8_vector(_encode_int8_vector(v.tolist()), 256))
        cos = float(v @ decoded / np.linalg.norm(decoded))
        worst = min(worst, cos)
    # symmetric per-vector int8 keeps cosine within a tiny quantization error.
    assert worst >= 0.999


def test_float_matrix_quantization_matches_shipped_encoder_semantics():
    matrix = np.asarray(
        [
            [0.6, 0.8, 0.0, 0.0],
            [-0.5, 0.5, 0.25, -0.25],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    quantized, scales = VectorStore._quantize_float_matrix(np, matrix)

    for index, vector in enumerate(matrix):
        encoded = quantized[index].tobytes() + struct.pack(
            "<f", float(scales[index])
        )
        assert encoded == _encode_int8_vector(vector.tolist())


def test_int8_blob_layout_and_bad_length_rejected():
    blob = _encode_int8_vector([0.6, 0.8, 0.0, 0.0])
    assert len(blob) == 4 + 4  # dim int8 bytes + float32 scale
    assert _decode_int8_vector(blob, 4) is not None
    assert _decode_int8_vector(blob, 8) is None  # wrong dim => rejected, not misread


def test_sign_bits_match_numpy_packbits():
    rng = np.random.default_rng(1)
    v = rng.standard_normal(37).astype(np.float32)
    expected = np.packbits((v >= 0.0).astype(np.uint8)).tobytes()
    assert _pack_sign_bits(v.tolist()) == expected


# -- Identity isolation: int8 never mixes with float32 -----------------------


def test_int8_identity_distinct_and_writes_binary(tmp_path):
    db_path = tmp_path / "lcm.db"
    _seed_messages(db_path, 3)
    vs = VectorStore(db_path)
    try:
        f32 = EmbeddingIdentity.canonical(PROVIDER, MODEL, "", 4, "float32", "little", "chunk")
        i8 = _int8_identity(4)
        assert f32.identity_hash != i8.identity_hash

        vs.register_profile(MODEL, PROVIDER, 4, dtype="float32", task="chunk")
        vs.record_chunk_embedding(
            "0:0", MODEL, [1.0, 0.0, 0.0, 0.0], store_id=0, chunk_index=0,
            char_start=0, char_end=1, token_estimate=1, identity=f32,
        )
        # float32 writes NO binary row (byte-identical legacy behavior).
        assert vs.connection.execute("SELECT COUNT(*) FROM lcm_chunk_binary").fetchone()[0] == 0

        vs.register_profile(MODEL, PROVIDER, 4, dtype="int8", task="chunk")
        vs.record_chunk_embedding(
            "1:0", MODEL, [0.0, 1.0, 0.0, 0.0], store_id=1, chunk_index=0,
            char_start=0, char_end=1, token_estimate=1, identity=i8,
        )
        assert vs.connection.execute("SELECT COUNT(*) FROM lcm_chunk_binary").fetchone()[0] == 1
        # int8 vec blob is dim + 4 bytes, distinct layout from float32 (dim*4).
        blob = vs.connection.execute(
            "SELECT vec FROM lcm_chunk_vectors WHERE identity_hash = ?",
            (i8.identity_hash,),
        ).fetchone()[0]
        assert len(blob) == 4 + 4
    finally:
        vs.close()


def test_two_stage_reports_full_coverage(tmp_path):
    db_path = tmp_path / "lcm.db"
    _seed_messages(db_path, 3)
    vs = VectorStore(db_path, bounded_scan_rows=1)  # tiny bound: proves it is NOT bounded
    try:
        i8 = _int8_identity(4)
        vs.register_profile(MODEL, PROVIDER, 4, dtype="int8", task="chunk")
        for idx, vec in enumerate([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]):
            vs.record_chunk_embedding(
                f"{idx}:0", MODEL, vec, store_id=idx, chunk_index=0,
                char_start=0, char_end=1, token_estimate=1, identity=i8,
            )
        result = vs.knn_chunks([1.0, 0.0, 0.0, 0.0], k=2, model=MODEL, provider=PROVIDER)
        # Full corpus reached despite bounded_scan_rows=1 -> the two-stage path fired.
        assert result.coverage == "full_approx"
        assert result[0][0] == "0:0"
    finally:
        vs.close()


def test_in_memory_two_stage_query_with_deadline_uses_current_connection():
    vs = VectorStore(":memory:", bounded_scan_rows=1)
    try:
        vs.connection.execute(
            """
            CREATE TABLE messages (
                store_id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                source TEXT DEFAULT '',
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL
            )
            """
        )
        vs.connection.executemany(
            "INSERT INTO messages("
            "store_id, session_id, source, role, content, timestamp"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            [
                (0, "s", "hist", "user", "m", 0.0),
                (1, "s", "hist", "user", "m", 1.0),
            ],
        )
        i8 = _int8_identity(4)
        vs.register_profile(MODEL, PROVIDER, 4, dtype="int8", task="chunk")
        for idx, vec in enumerate(
            ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
        ):
            vs.record_chunk_embedding(
                f"{idx}:0",
                MODEL,
                vec,
                store_id=idx,
                chunk_index=0,
                char_start=0,
                char_end=1,
                token_estimate=1,
                identity=i8,
            )

        result = vs.knn_chunks(
            [1.0, 0.0, 0.0, 0.0],
            k=1,
            model=MODEL,
            provider=PROVIDER,
            deadline=vector_store_module._monotonic() + 5.0,
        )

        assert result.coverage == "full_approx"
        assert result[0][0] == "0:0"
    finally:
        vs.close()


def test_chunk_deadline_bounds_a_synced_binary_prescreen(tmp_path, monkeypatch):
    db_path = tmp_path / "lcm.db"
    _seed_messages(db_path, 2)
    vs = VectorStore(db_path, bounded_scan_rows=1)
    try:
        i8 = _int8_identity(4)
        vs.register_profile(MODEL, PROVIDER, 4, dtype="int8", task="chunk")
        for idx, vec in enumerate(
            ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
        ):
            vs.record_chunk_embedding(
                f"{idx}:0",
                MODEL,
                vec,
                store_id=idx,
                chunk_index=0,
                char_start=0,
                char_end=1,
                token_estimate=1,
                identity=i8,
            )

        unbounded = vs.knn_chunks(
            [1.0, 0.0, 0.0, 0.0], k=1, model=MODEL, provider=PROVIDER
        )
        monkeypatch.setattr(
            vector_store_module,
            "_prescreen_deadline_expired",
            _expire_after_first_prescreen_batch,
        )
        monkeypatch.setattr(
            vs,
            "_count_embedded_vectors",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("deadline expiry must not start COUNT(*)")
            ),
        )
        expired = vs.knn_chunks(
            [1.0, 0.0, 0.0, 0.0],
            k=1,
            model=MODEL,
            provider=PROVIDER,
            full_scan=True,
            # Genuinely-future deadline: only the patched position-keyed seam
            # trips; the real clock never expires the load path.
            deadline=vector_store_module._monotonic() + 60.0,
        )

        assert unbounded.coverage == "full_approx"
        assert expired.coverage == "bounded"
        assert expired.scanned == 1
        assert expired.total is None
    finally:
        vs.close()


# -- Stage-1 Hamming recall@M on a synthetic 5k set (the spec bar) -----------


def test_stage1_hamming_recall_at_4k_on_synthetic_5k():
    """Stage-1 (sign-bit Hamming) recall@(4k) >= 0.98 of the exact-cosine top-k.

    Uses the production sign-bit packing + popcount table over a synthetic 5k set
    with genuine near-neighbors planted around each query (the structure real
    embeddings have), matching the SPEC C1 acceptance bar.
    """
    rng = np.random.default_rng(7)
    n, dim, k, mult = 5000, 512, 10, 4
    popcount = np.asarray(_POPCOUNT_TABLE, dtype=np.uint16)

    def planted(base, eps):
        u = rng.standard_normal(dim).astype(np.float32)
        v = base + eps * u / np.linalg.norm(u)
        return v / np.linalg.norm(v)

    total = 0.0
    trials = 40
    for _ in range(trials):
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        base = _unit(rng, dim)
        idx = rng.choice(n, k, replace=False)
        for j in idx:
            vecs[j] = planted(base, 0.4)
        bits = np.packbits((vecs >= 0.0).astype(np.uint8), axis=1)
        query = planted(base, 0.2)

        exact = set(np.argsort(-(vecs @ query))[:k].tolist())
        query_bits = np.packbits((query >= 0.0).astype(np.uint8))
        hamming = popcount[np.bitwise_xor(bits, query_bits)].sum(axis=1)
        m = mult * k
        survivors = set(np.argpartition(hamming, m - 1)[:m].tolist())
        total += len(exact & survivors) / k
    assert total / trials >= 0.98


def test_two_stage_recall_vs_exact_float_through_store(tmp_path):
    """End-to-end store two-stage top-k recovers the exact float top-k on planted data."""
    db_path = tmp_path / "lcm.db"
    n, dim, k = 2000, 256, 10
    _seed_messages(db_path, n)
    rng = np.random.default_rng(11)
    base = _unit(rng, dim)

    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    neighbor_idx = rng.choice(n, k, replace=False)
    for j in neighbor_idx:
        u = rng.standard_normal(dim).astype(np.float32)
        v = base + 0.4 * u / np.linalg.norm(u)
        vecs[j] = v / np.linalg.norm(v)

    vs = VectorStore(db_path)
    try:
        i8 = _int8_identity(dim)
        vs.register_profile(MODEL, PROVIDER, dim, dtype="int8", task="chunk")
        for i in range(n):
            vs.record_chunk_embedding(
                f"{i}:0", MODEL, vecs[i].tolist(), store_id=i, chunk_index=0,
                char_start=0, char_end=1, token_estimate=1, identity=i8,
            )
        query = base
        result = vs.knn_chunks(query.tolist(), k=k, model=MODEL, provider=PROVIDER)
        assert result.coverage == "full_approx"
        got = {row[0] for row in result}
        exact = {f"{i}:0" for i in np.argsort(-(vecs @ query))[:k].tolist()}
        assert len(got & exact) / k >= 0.98
    finally:
        vs.close()


# -- Matryoshka store_dim truncation -----------------------------------------


def test_store_dim_truncates_and_renormalizes(tmp_path):
    db_path = tmp_path / "lcm.db"
    _seed_messages(db_path, 2)
    vs = VectorStore(db_path)
    try:
        store_dim = 2  # register the profile at the truncated (identity) dim
        i8 = _int8_identity(store_dim)
        vs.register_profile(MODEL, PROVIDER, store_dim, dtype="int8", task="chunk")
        # Feed a full 4-d vector; only the leading 2 dims are stored + renormalized.
        vs.record_chunk_embedding(
            "0:0", MODEL, [3.0, 4.0, 99.0, 99.0], store_id=0, chunk_index=0,
            char_start=0, char_end=1, token_estimate=1, identity=i8,
        )
        blob = vs.connection.execute(
            "SELECT vec FROM lcm_chunk_vectors WHERE chunk_id = '0:0'"
        ).fetchone()[0]
        assert len(blob) == store_dim + 4
        decoded = np.asarray(_decode_int8_vector(blob, store_dim))
        expected = np.array([0.6, 0.8], dtype=np.float32)  # [3,4] renormalized
        assert np.allclose(decoded / np.linalg.norm(decoded), expected, atol=2e-2)
    finally:
        vs.close()


def test_prescreen_multiplier_respected(tmp_path):
    from hermes_lcm.config import LCMConfig

    cfg = LCMConfig(knn_prescreen_multiplier=9)
    db_path = tmp_path / "lcm.db"
    _seed_messages(db_path, 1)
    vs = VectorStore(db_path, config=cfg)
    try:
        assert vs.knn_prescreen_multiplier == 9
    finally:
        vs.close()


# -- float32 + binary prescreen: full-corpus two-stage with EXACT float rescore --


def test_float32_prescreen_opt_in_writes_binary_and_stays_exact(tmp_path):
    """A float32 identity with LCM_EMBEDDING_BINARY_PRESCREEN keeps float32 vec
    bytes but gains a prescreen, giving the two-stage path with exact rescore."""
    from hermes_lcm.config import LCMConfig

    db_path = tmp_path / "lcm.db"
    n, dim, k = 1500, 128, 10
    _seed_messages(db_path, n)
    rng = np.random.default_rng(5)
    base = _unit(rng, dim)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    for j in rng.choice(n, k, replace=False):
        u = rng.standard_normal(dim).astype(np.float32)
        vecs[j] = (base + 0.4 * u / np.linalg.norm(u))
        vecs[j] /= np.linalg.norm(vecs[j])

    cfg = LCMConfig(embedding_binary_prescreen=True)
    vs = VectorStore(db_path, config=cfg)
    try:
        # Distinct identity (revision) so prescreen rows never mix with a legacy
        # float32 identity; dtype stays float32 -> vec bytes byte-identical.
        ident = EmbeddingIdentity.canonical(
            PROVIDER, MODEL, "prescreen", dim, "float32", "little", "chunk"
        )
        vs.register_profile(MODEL, PROVIDER, dim, revision="prescreen", dtype="float32", task="chunk")
        for i in range(n):
            vs.record_chunk_embedding(
                f"{i}:0", MODEL, vecs[i].tolist(), store_id=i, chunk_index=0,
                char_start=0, char_end=1, token_estimate=1, identity=ident,
            )
        # float32 layout preserved (dim*4 bytes), and a binary row was written.
        blob = vs.connection.execute(
            "SELECT vec FROM lcm_chunk_vectors WHERE chunk_id='0:0'"
        ).fetchone()[0]
        assert len(blob) == dim * 4
        assert vs.connection.execute("SELECT COUNT(*) FROM lcm_chunk_binary").fetchone()[0] == n

        result = vs.knn_chunks(base.tolist(), k=k, model=MODEL, provider=PROVIDER)
        assert result.coverage == "full_approx"
        got = {row[0] for row in result}
        # Exact float rescore of survivors: score-threshold recall (ties-safe).
        scores = vecs @ base
        thr = np.sort(scores)[-k]
        idx = {int(cid.split(":")[0]) for cid in got}
        assert np.mean([scores[i] >= thr - 1e-5 for i in idx]) >= 0.95
    finally:
        vs.close()
