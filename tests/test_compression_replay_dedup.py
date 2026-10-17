"""End-to-end: a post-compression replay that re-sends an already-persisted
turn (mixed NEW/REPLAY ordering, the upstream issue #513 shape) must not
create duplicate rows once the identity guard is in place.

Scenario (issue #513):
    ingest batch A  ->  compress()  ->  on_session_start(boundary_reason="compression")
    ->  ingest batch B = [NEW, NEW, REPLAY(T1), NEW, REPLAY(A1), REPLAY(U1)]
Expected: exactly the NEW rows are stored; the REPLAY rows map to their
existing store_ids and appear in the store exactly once each.

Harness note: the engine has no host-facing compress() beyond
LCMEngine.compress (CompactionMixin), and the compression boundary in the
same session is driven by on_session_start(boundary_reason="compression").
We mirror tests/test_lcm_engine.py's ``engine`` fixture (direct _session_id,
low leaf threshold) and force compression so no LLM provider is required.
"""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def env(tmp_path):
    config = LCMConfig()
    config.fresh_tail_count = 4  # small for testing
    config.leaf_chunk_tokens = 100  # low threshold for testing
    config.database_path = str(tmp_path / "lcm_test.db")
    e = LCMEngine(config=config)
    e._session_id = "dedup-session"
    e.context_length = 200000
    e.threshold_tokens = int(200000 * config.context_threshold)
    try:
        yield e, e._store, "dedup-session"
    finally:
        e.shutdown()


def test_partial_turn_replay_after_compress_no_duplicate(env):
    engine, store, session_id = env
    u1 = {"role": "user", "content": "original user question", "timestamp": 100.0}
    a1 = {"role": "assistant", "content": "original assistant answer", "timestamp": 101.0}
    t1 = {"role": "tool", "content": "tool output before compression",
          "timestamp": 102.0, "tool_call_id": "tc-1"}

    def id_of(msg):
        rows = [r for r in store.get_session_tail(session_id, limit=50)
                if r["content"] == msg["content"] and r["role"] == msg["role"]]
        return rows[-1]["store_id"] if rows else None

    # Batch A: the turn that will be re-sent after compression.
    engine.ingest([u1, a1, t1])
    id_u1, id_a1, id_t1 = id_of(u1), id_of(a1), id_of(t1)
    assert all(x is not None for x in (id_u1, id_a1, id_t1))
    assert len({id_u1, id_a1, id_t1}) == 3

    # Compression boundary (real compress path per the #513 repro).
    engine.compress([u1, a1, t1], force=True)

    # Post-compression session start in the SAME session: cursor reset to 0
    # + reconciliation rescan — this is the re-send trigger.
    engine.on_session_start(session_id, boundary_reason="compression")

    # Batch B: mixed NEW/REPLAY — replays interleaved with genuinely new rows,
    # which defeats any scalar prefix cursor.
    n0a = {"role": "tool", "content": "new tool output during compression",
           "timestamp": 110.0, "tool_call_id": "tc-2"}
    n0b = {"role": "assistant", "content": "new assistant text during compression",
           "timestamp": 111.0}
    n1 = {"role": "assistant", "content": "new assistant continuation", "timestamp": 112.0}
    batch_b = [n0a, n0b, t1, n1, a1, u1]   # NEW, NEW, REPLAY, NEW, REPLAY, REPLAY
    engine.ingest(batch_b)

    tail = store.get_session_tail(session_id, limit=100)
    counts = {}
    for row in tail:
        key = (row["role"], row["content"])
        counts[key] = counts.get(key, 0) + 1
    for msg in (u1, a1, t1):
        assert counts[(msg["role"], msg["content"])] == 1, (
            f"duplicate rows for {msg['content']!r}: {counts[(msg['role'], msg['content'])]}"
        )
    for msg in (n0a, n0b, n1):
        assert counts[(msg["role"], msg["content"])] == 1, (
            f"expected new row missing for {msg['content']!r}: {counts[(msg['role'], msg['content'])]}"
        )
    # Replay rows are the ORIGINAL store_ids (identity preserved, not re-stamped).
    assert id_of(u1) == id_u1 and id_of(a1) == id_a1 and id_of(t1) == id_t1


def test_storage_guard_stops_re_stamp_when_cursor_is_defeated(env):
    """Second #513 trigger shape: the scalar ingest cursor misses entirely.

    If the engine's cursor lands at 0 (e.g. a reconciliation miss or a
    mid-turn re-send before reconciliation has run), the whole post-
    compression window is written to the store as if new — including the
    already-persisted rows. The storage-boundary identity guard
    (_insert_message_conflict_safe / idx_msg_identity) must catch every
    replayed row and return the ORIGINAL store_ids, so the store stays
    deduped and all id lists stay position-aligned for engine callers.
    """
    engine, store, session_id = env
    u1 = {"role": "user", "content": "original user question", "timestamp": 100.0}
    a1 = {"role": "assistant", "content": "original assistant answer", "timestamp": 101.0}
    t1 = {"role": "tool", "content": "tool output before compression",
          "timestamp": 102.0, "tool_call_id": "tc-1"}

    engine.ingest([u1, a1, t1])
    tail = store.get_session_tail(session_id, limit=50)
    id_by_content = {r["content"]: r["store_id"] for r in tail}
    id_u1, id_a1, id_t1 = (id_by_content[u1["content"]],
                           id_by_content[a1["content"]],
                           id_by_content[t1["content"]])
    assert len({id_u1, id_a1, id_t1}) == 3

    n0a = {"role": "tool", "content": "new tool output during compression",
           "timestamp": 110.0, "tool_call_id": "tc-2"}
    n0b = {"role": "assistant", "content": "new assistant text during compression",
           "timestamp": 111.0}
    n1 = {"role": "assistant", "content": "new assistant continuation", "timestamp": 112.0}
    batch_b = [n0a, n0b, t1, n1, a1, u1]

    # Defeat the cursor: force the engine to treat the ENTIRE batch as new.
    engine._ingest_cursor = 0
    engine._ingest_cursor_needs_reconcile = False
    engine.ingest(batch_b)

    rows = store.get_session_tail(session_id, limit=100)
    by_content = {}
    for r in rows:
        by_content.setdefault(r["content"], []).append(r["store_id"])
    # Replayed rows: exactly one row each, the ORIGINAL id (no re-stamp).
    assert by_content[t1["content"]] == [id_t1], f"t1 re-stamped: {by_content[t1['content']]}"
    assert by_content[a1["content"]] == [id_a1], f"a1 re-stamped: {by_content[a1['content']]}"
    assert by_content[u1["content"]] == [id_u1], f"u1 re-stamped: {by_content[u1['content']]}"
    # New rows: exactly one row each, new ids.
    assert len(by_content[n0a["content"]]) == 1
    assert len(by_content[n0b["content"]]) == 1
    assert len(by_content[n1["content"]]) == 1
    assert by_content[n0a["content"]] != by_content[n1["content"]]
    assert store.get_session_count(session_id) == 6
