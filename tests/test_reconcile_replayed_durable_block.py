"""Existing-session rebinds must not re-append the whole history once the
durable store holds rows the incoming history lacks.

The stored-tail suffix rule only advances the ingest cursor when the incoming
prefix covers the entire stored session. After a rewound turn (the discarded
pair stays in the store), a stored/replayed text mismatch, or one earlier
ambiguous-delta re-append, that proof is unavailable forever, and every rebind
re-appended the full history (each burst larger than the last). The
replayed-durable-block rule accepts the longest incoming prefix that replays a
contiguous stored run of at least four rows with a user and an assistant turn.
"""

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _turns(start: int, count: int) -> list[dict]:
    rows = []
    for i in range(start, start + count):
        rows.append({"role": "user", "content": f"question {i}"})
        rows.append({"role": "assistant", "content": f"answer {i}"})
    return rows


def _rebind(config: LCMConfig, session: str) -> LCMEngine:
    engine = LCMEngine(config=config)
    engine.on_session_start(
        session,
        platform="telegram",
        conversation_id=f"{session}-conversation",
        context_length=200000,
    )
    return engine


def _close(engine: LCMEngine) -> None:
    engine._store.close()
    engine._dag.close()
    engine._lifecycle.close()


def test_rebind_after_rewind_replays_durable_block_instead_of_full_history(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "rewind.db"))
    session = "rewind-session"
    engine = _rebind(config, session)
    history = _turns(0, 6)
    engine._ingest_messages(history)
    assert engine._store.get_session_count(session) == 12
    _close(engine)

    # The user rewinds the last turn and retries: the store keeps the discarded
    # pair, the replayed history carries the retry instead.
    retried = history[:-2] + [
        {"role": "user", "content": "question 5 (retry)"},
        {"role": "assistant", "content": "answer 5 (retry)"},
    ]
    engine = _rebind(config, session)
    engine._ingest_messages(retried)

    rows = engine._store.get_session_messages(session)
    assert len(rows) == 14
    assert [row["content"] for row in rows[-2:]] == [
        "question 5 (retry)",
        "answer 5 (retry)",
    ]
    assert engine._last_ingest_reconciliation["reason"] == "replayed contiguous durable block"
    assert engine._ingest_cursor == len(retried)
    _close(engine)


def test_rebind_with_outgrown_store_does_not_reappend_history(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "outgrown.db"))
    session = "outgrown-session"
    engine = _rebind(config, session)
    history = _turns(0, 6)
    engine._ingest_messages(history)
    _close(engine)

    # A rewind leaves the discarded pair behind (14 stored rows, 12 replayed).
    engine = _rebind(config, session)
    engine._ingest_messages(history + _turns(6, 1))
    assert engine._store.get_session_count(session) == 14
    _close(engine)

    # The retry plus two further turns arrive across per-turn rebinds.
    retried = history + [
        {"role": "user", "content": "question 6 (retry)"},
        {"role": "assistant", "content": "answer 6 (retry)"},
    ]
    engine = _rebind(config, session)
    engine._ingest_messages(retried + _turns(7, 1))
    count_after_first = engine._store.get_session_count(session)
    assert count_after_first == 18  # 12 replayed rows skipped, 4 tail rows persisted
    _close(engine)

    engine = _rebind(config, session)
    engine._ingest_messages(retried + _turns(7, 2))
    assert engine._store.get_session_count(session) == 20  # only the newest turn
    assert engine._last_ingest_reconciliation["reason"] == "replayed contiguous durable block"
    _close(engine)

    # Rebinding with an unchanged history appends nothing.
    engine = _rebind(config, session)
    engine._ingest_messages(retried + _turns(7, 2))
    assert engine._store.get_session_count(session) == 20
    assert engine._ingest_cursor == len(retried) + 4
    _close(engine)


def test_short_delta_matching_stored_run_is_still_persisted(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "short-delta.db"))
    session = "short-delta-session"
    engine = _rebind(config, session)
    engine._ingest_messages(_turns(0, 6))
    _close(engine)

    # A three-row delta that happens to repeat stored rows stays below the
    # replay-evidence threshold: duplicate-over-loss still applies.
    delta = [
        {"role": "user", "content": "question 0"},
        {"role": "assistant", "content": "answer 0"},
        {"role": "user", "content": "question 1"},
    ]
    engine = _rebind(config, session)
    engine._ingest_messages(delta)
    assert engine._store.get_session_count(session) == 15
    assert engine._last_ingest_reconciliation["reason"] == "persisted ambiguous delta"
    _close(engine)
