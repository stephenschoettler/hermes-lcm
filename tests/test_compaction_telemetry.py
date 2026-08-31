"""Tests for B2 compaction telemetry (per-conversation snapshot in metadata)."""

import json
import sqlite3
import threading
import time

import pytest

from hermes_lcm import tools as lcm_tools
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_test.db")
    e = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    e._session_id = "test-session"
    e._conversation_id = "conv-1"
    e.update_model("gpt-test", 200000, provider="openai-codex", api_mode="responses")
    return e


def _hot_usage(prompt=1000, read=400, write=50):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": 100,
        "total_tokens": prompt + 100,
        "input_tokens": prompt - read - write,
        "cache_read_tokens": read,
        "cache_write_tokens": write,
    }


def _cold_usage(prompt=1000):
    # Cache keys present (so cache_metrics_available) but both zero -> cold.
    return {"prompt_tokens": prompt, "cache_read_tokens": 0, "cache_write_tokens": 0}


def _telemetry(engine):
    return engine.get_status().get("compaction_telemetry")


def _slash_status_fields(engine):
    return {
        key.strip(): value.strip()
        for line in handle_lcm_command("status", engine).splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }


def _assert_total_compaction_surfaces(engine, expected):
    status = engine.get_status()
    tool_status = json.loads(lcm_tools.lcm_status({}, engine=engine))
    slash_status = _slash_status_fields(engine)

    assert status["total_compactions"] == expected
    assert status["total_compactions_scope"] == "current_conversation"
    assert tool_status["total_compactions"] == expected
    assert tool_status["total_compactions_scope"] == "current_conversation"
    assert slash_status["total_compactions"] == str(expected)
    assert slash_status["total_compactions_scope"] == "current_conversation"


def test_records_per_turn_snapshot(engine):
    engine.update_from_response(_hot_usage(prompt=1050, read=400, write=50))
    t = _telemetry(engine)
    assert t is not None
    assert t["cache_state"] == "hot"
    assert t["consecutive_cold_observations"] == 0
    assert t["turns_since_leaf_compaction"] == 1
    assert t["last_observed_prompt_tokens"] == 1050
    assert t["last_observed_cache_read"] == 400
    assert t["last_observed_cache_write"] == 50
    assert t["peak_prompt_tokens_since_leaf_compaction"] == 1050
    assert t["provider"] == "openai-codex"
    assert t["model"] == "gpt-test"
    assert t["activity_band"] == "low"


def test_turns_since_increments_and_peak_is_max(engine):
    engine.update_from_response(_hot_usage(prompt=1000))
    engine.update_from_response(_hot_usage(prompt=600))
    t = _telemetry(engine)
    assert t["turns_since_leaf_compaction"] == 2
    assert t["peak_prompt_tokens_since_leaf_compaction"] == 1000  # max across turns


def test_cold_streak_counts_and_hot_resets(engine):
    engine.update_from_response(_cold_usage())
    engine.update_from_response(_cold_usage())
    assert _telemetry(engine)["cache_state"] == "cold"
    assert _telemetry(engine)["consecutive_cold_observations"] == 2
    engine.update_from_response(_hot_usage())
    assert _telemetry(engine)["cache_state"] == "hot"
    assert _telemetry(engine)["consecutive_cold_observations"] == 0


def test_idle_turn_is_skipped(engine):
    engine.update_from_response(_hot_usage())
    before = _telemetry(engine)["turns_since_leaf_compaction"]
    # No prompt tokens and no cache keys at all -> no signal -> no write.
    engine.update_from_response({"completion_tokens": 5})
    assert _telemetry(engine)["turns_since_leaf_compaction"] == before


def test_resets_on_compaction(engine):
    engine.update_from_response(_hot_usage(prompt=1000))
    engine.update_from_response(_hot_usage(prompt=1000))
    assert _telemetry(engine)["turns_since_leaf_compaction"] == 2

    # Simulate a leaf compaction happening between turns.
    engine.compression_count += 1
    engine._last_compaction_duration_ms = 12.5
    engine.update_from_response(_hot_usage(prompt=400))

    t = _telemetry(engine)
    assert t["turns_since_leaf_compaction"] == 0
    assert t["total_compactions"] == 1
    assert t["last_leaf_compaction_at"] is not None
    assert t["last_compaction_duration_ms"] == 12.5
    assert t["peak_prompt_tokens_since_leaf_compaction"] == 400  # reset to current


def test_total_compactions_status_includes_pending_compaction(engine):
    engine.compression_count = 3
    engine.update_from_response(_hot_usage())
    _assert_total_compaction_surfaces(engine, 3)

    # Status can be requested after compaction increments the live counter but
    # before the response hook persists the next telemetry snapshot.
    engine.compression_count += 1

    _assert_total_compaction_surfaces(engine, 4)


def test_total_compactions_status_survives_session_rollover_for_conversation(engine):
    engine.compression_count = 3
    engine.update_from_response(_hot_usage())
    _assert_total_compaction_surfaces(engine, 3)

    engine.on_session_start(
        "test-session-next",
        platform="discord",
        conversation_id="conv-1",
    )

    assert engine.compression_count == 0
    _assert_total_compaction_surfaces(engine, 3)


@pytest.mark.parametrize("previous_session_count", [1, 3])
def test_total_compactions_includes_first_compaction_after_session_rollover(
    engine,
    previous_session_count,
):
    engine.compression_count = previous_session_count
    engine.update_from_response(_hot_usage())

    engine.on_session_start(
        "test-session-next",
        platform="discord",
        conversation_id="conv-1",
    )
    assert engine.compression_count == 0

    # The first response telemetry can arrive after the new session has already
    # compacted. Count it even when the new session's counter is equal to or
    # below the persisted baseline from the previous session.
    engine.compression_count = 1
    engine.update_from_response(_hot_usage())

    _assert_total_compaction_surfaces(engine, previous_session_count + 1)


def test_total_compactions_includes_first_compaction_after_engine_restart(tmp_path):
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_test.db")

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    first.on_session_start(
        "test-session",
        platform="discord",
        conversation_id="conv-1",
    )
    first.compression_count = 3
    first.update_from_response(_hot_usage())
    first.shutdown()

    restarted = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    try:
        restarted.on_session_start(
            "test-session-next",
            platform="discord",
            conversation_id="conv-1",
        )
        restarted.compression_count = 1
        restarted.update_from_response(_hot_usage())

        _assert_total_compaction_surfaces(restarted, 4)
    finally:
        restarted.shutdown()


def test_total_compactions_rebaseline_when_session_id_binds_new_conversation(engine):
    engine.compression_count = 3
    engine.update_from_response(_hot_usage())
    _assert_total_compaction_surfaces(engine, 3)

    engine.on_session_start(
        "test-session",
        platform="discord",
        conversation_id="conv-2",
    )

    assert engine.compression_count == 0
    _assert_total_compaction_surfaces(engine, 0)
    assert engine._store.read_compaction_telemetry("conv-1")["total_compactions"] == 3

    engine.compression_count = 1
    engine.update_from_response(_hot_usage())
    _assert_total_compaction_surfaces(engine, 1)


@pytest.mark.parametrize("record_mode", ["successful_compaction", "response_hook"])
def test_compactions_increment_total_atomically_across_engines(tmp_path, record_mode):
    config = LCMConfig(database_path=str(tmp_path / "lcm_test.db"))
    engines = [
        LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home_a")),
        LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home_b")),
    ]
    barrier = threading.Barrier(2, timeout=5)
    errors = []

    try:
        for index, current in enumerate(engines):
            current.on_session_start(
                f"test-session-{index}",
                platform="cli",
                conversation_id="conv-1",
            )
            current.compression_count = 1
            original_increment = current._store.increment_compaction_telemetry

            def paused_increment(
                conversation_id,
                increment,
                updates,
                *,
                _increment=original_increment,
            ):
                barrier.wait()
                return _increment(conversation_id, increment, updates)

            current._store.increment_compaction_telemetry = paused_increment

        def record(current):
            try:
                if record_mode == "successful_compaction":
                    current._record_successful_compaction_telemetry()
                else:
                    current.update_from_response(_hot_usage())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=record, args=(current,)) for current in engines]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not [thread for thread in threads if thread.is_alive()]
        assert errors == []
        telemetry = engines[0]._store.read_compaction_telemetry("conv-1")
        assert telemetry["total_compactions"] == 2
    finally:
        for current in engines:
            current.shutdown()


def test_zero_delta_snapshot_cannot_overwrite_concurrent_increment(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_test.db"))
    engines = [
        LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home_a")),
        LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home_b")),
    ]
    barrier = threading.Barrier(2, timeout=5)
    errors = []

    try:
        for index, current in enumerate(engines):
            current.on_session_start(
                f"test-session-{index}",
                platform="cli",
                conversation_id="conv-1",
            )
            original_increment = current._store.increment_compaction_telemetry

            def paused_increment(
                conversation_id,
                increment,
                updates,
                *,
                _increment=original_increment,
            ):
                barrier.wait()
                return _increment(conversation_id, increment, updates)

            current._store.increment_compaction_telemetry = paused_increment

        engines[1].compression_count = 1

        def record_snapshot():
            try:
                engines[0].update_from_response(_hot_usage())
            except Exception as exc:
                errors.append(exc)

        def record_increment():
            try:
                engines[1]._record_successful_compaction_telemetry()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=record_snapshot),
            threading.Thread(target=record_increment),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not [thread for thread in threads if thread.is_alive()]
        assert errors == []
        telemetry = engines[0]._store.read_compaction_telemetry("conv-1")
        assert telemetry["total_compactions"] == 1
    finally:
        for current in engines:
            current.shutdown()


def test_zero_delta_snapshot_preserves_concurrent_compaction_metadata(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_test.db"))
    snapshot_engine = LCMEngine(
        config=config,
        hermes_home=str(tmp_path / "hermes_home_snapshot"),
    )
    compaction_engine = LCMEngine(
        config=config,
        hermes_home=str(tmp_path / "hermes_home_compaction"),
    )
    snapshot_ready = threading.Event()
    compaction_done = threading.Event()
    errors = []

    try:
        snapshot_engine.on_session_start(
            "snapshot-session",
            platform="cli",
            conversation_id="conv-1",
        )
        compaction_engine.on_session_start(
            "compaction-session",
            platform="cli",
            conversation_id="conv-1",
        )
        original_increment = snapshot_engine._store.increment_compaction_telemetry

        def paused_snapshot(conversation_id, increment, updates):
            snapshot_ready.set()
            assert compaction_done.wait(timeout=5)
            return original_increment(conversation_id, increment, updates)

        snapshot_engine._store.increment_compaction_telemetry = paused_snapshot

        def record_snapshot():
            try:
                snapshot_engine.update_from_response(_hot_usage(prompt=1000))
            except Exception as exc:
                errors.append(exc)

        snapshot_thread = threading.Thread(target=record_snapshot)
        snapshot_thread.start()
        assert snapshot_ready.wait(timeout=5)

        compaction_engine.compression_count = 1
        compaction_engine._last_compaction_duration_ms = 12.5
        compaction_engine._record_successful_compaction_telemetry()
        compaction_done.set()
        snapshot_thread.join(timeout=10)

        assert not snapshot_thread.is_alive()
        assert errors == []
        telemetry = snapshot_engine._store.read_compaction_telemetry("conv-1")
        assert telemetry["total_compactions"] == 1
        assert telemetry["turns_since_leaf_compaction"] == 0
        assert telemetry["peak_prompt_tokens_since_leaf_compaction"] == 0
        assert telemetry["last_leaf_compaction_at"] is not None
        assert telemetry["last_compaction_duration_ms"] == 12.5
    finally:
        compaction_done.set()
        snapshot_engine.shutdown()
        compaction_engine.shutdown()


def test_compaction_telemetry_lock_contention_is_nonblocking(tmp_path):
    db_path = tmp_path / "lcm_test.db"
    lock_owner = MessageStore(db_path)
    telemetry_store = MessageStore(db_path)
    lock_ready = threading.Event()
    release_lock = threading.Event()
    write_done = threading.Event()
    write_errors = []

    def hold_write_lock():
        lock_owner._conn.execute("BEGIN IMMEDIATE")
        lock_ready.set()
        release_lock.wait(timeout=5)
        lock_owner._conn.rollback()

    def write_telemetry():
        try:
            telemetry_store.increment_compaction_telemetry(
                "conv-1",
                1,
                {"compression_count_at_record": 1},
            )
        except Exception as exc:
            write_errors.append(exc)
        finally:
            write_done.set()

    holder = threading.Thread(target=hold_write_lock)
    writer = threading.Thread(target=write_telemetry)
    try:
        holder.start()
        assert lock_ready.wait(timeout=5)
        assert telemetry_store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000

        started = time.monotonic()
        writer.start()
        assert write_done.wait(timeout=0.25)
        elapsed = time.monotonic() - started

        assert elapsed < 0.25
        assert len(write_errors) == 1
        assert isinstance(write_errors[0], sqlite3.OperationalError)
        assert "locked" in str(write_errors[0]).lower()
        assert telemetry_store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        assert telemetry_store.read_compaction_telemetry("conv-1") is None
    finally:
        release_lock.set()
        writer.join(timeout=5)
        holder.join(timeout=5)
        lock_owner.close()
        telemetry_store.close()


def test_successful_compaction_is_durable_before_response_hook(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm_test.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=1,
    )
    conversation_id = "conv-1"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old question one"},
        {"role": "assistant", "content": "old answer one"},
        {"role": "user", "content": "old question two"},
        {"role": "assistant", "content": "old answer two"},
        {"role": "user", "content": "fresh question"},
        {"role": "assistant", "content": "fresh answer"},
    ]

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    first.on_session_start(
        "test-session",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )

    def summarize(initial_chunk, **_kwargs):
        return (
            list(initial_chunk),
            100,
            "Durable summary.\nExpand for details about: old turns",
            1,
            1,
        )

    monkeypatch.setattr(first, "_summarize_leaf_chunk_with_rescue", summarize)
    first.compress(messages)
    assert first.compression_count == 1
    persisted = first._store.read_compaction_telemetry(conversation_id)
    assert persisted is not None
    assert persisted["total_compactions"] == 1
    assert persisted["compression_count_at_record"] == 1

    # Simulate process interruption before update_from_response() can persist
    # the next per-turn telemetry snapshot.
    first.shutdown()

    restarted = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    try:
        restarted.on_session_start(
            "test-session-next",
            platform="cli",
            conversation_id=conversation_id,
            context_length=200_000,
        )

        _assert_total_compaction_surfaces(restarted, 1)
    finally:
        restarted.shutdown()


def test_response_hook_does_not_recount_durably_recorded_compaction(engine):
    engine.compression_count = 1
    engine._last_compaction_duration_ms = 12.5
    engine._record_successful_compaction_telemetry()

    engine.update_from_response(_hot_usage(prompt=400))

    telemetry = _telemetry(engine)
    assert telemetry["total_compactions"] == 1
    assert telemetry["turns_since_leaf_compaction"] == 0
    assert telemetry["peak_prompt_tokens_since_leaf_compaction"] == 400
    assert telemetry["last_compaction_duration_ms"] == 12.5


def test_response_hook_does_not_recount_ambiguous_committed_increment(engine, monkeypatch):
    real_increment = engine._store.increment_compaction_telemetry

    def commit_then_raise(conversation_id, increment, updates):
        real_increment(conversation_id, increment, updates)
        raise RuntimeError("simulated ambiguous commit outcome")

    engine.compression_count = 1
    engine._last_compaction_duration_ms = 12.5
    monkeypatch.setattr(engine._store, "increment_compaction_telemetry", commit_then_raise)
    engine._record_successful_compaction_telemetry()

    committed = engine._store.read_compaction_telemetry("conv-1")
    assert committed["total_compactions"] == 1
    assert engine._compaction_telemetry_counter_rebaseline_pending is True

    monkeypatch.setattr(engine._store, "increment_compaction_telemetry", real_increment)
    engine.update_from_response(_hot_usage(prompt=400))

    telemetry = engine._store.read_compaction_telemetry("conv-1")
    assert telemetry["total_compactions"] == 1
    assert telemetry["turns_since_leaf_compaction"] == 0
    assert telemetry["peak_prompt_tokens_since_leaf_compaction"] == 400

    engine.compression_count = 2
    engine._record_successful_compaction_telemetry()
    assert engine._store.read_compaction_telemetry("conv-1")["total_compactions"] == 2


def test_total_compactions_status_defaults_to_zero_without_telemetry(engine):
    _assert_total_compaction_surfaces(engine, 0)


def test_total_compactions_status_is_scoped_to_current_conversation(engine):
    engine._store.write_compaction_telemetry(
        "conv-1",
        {"conversation_id": "conv-1", "total_compactions": 2},
    )
    engine._store.write_compaction_telemetry(
        "other-conversation",
        {"conversation_id": "other-conversation", "total_compactions": 99},
    )

    _assert_total_compaction_surfaces(engine, 2)


@pytest.mark.parametrize("malformed", [-1, "7", 1.5, True, None, [], {}])
def test_total_compactions_status_fails_closed_for_malformed_telemetry(engine, malformed):
    engine._store.write_compaction_telemetry(
        "conv-1",
        {"conversation_id": "conv-1", "total_compactions": malformed},
    )

    _assert_total_compaction_surfaces(engine, 0)
    assert _telemetry(engine)["total_compactions"] == 0


def test_no_conversation_records_nothing(engine):
    engine._conversation_id = ""
    engine.update_from_response(_hot_usage())
    assert _telemetry(engine) is None


def test_unknown_cache_state_when_no_cache_signal(engine):
    # Prompt tokens but no cache keys -> still recorded, state unknown.
    engine.update_from_response({"prompt_tokens": 800})
    t = _telemetry(engine)
    assert t is not None
    assert t["cache_state"] == "unknown"


def test_store_roundtrip_and_skip_unchanged(tmp_path):
    store = MessageStore(tmp_path / "t.db")
    assert store.read_compaction_telemetry("c1") is None
    record = {"conversation_id": "c1", "cache_state": "hot", "turns_since_leaf_compaction": 3}
    store.write_compaction_telemetry("c1", record)
    assert store.read_compaction_telemetry("c1") == record

    # Unchanged payload must not rewrite the row.
    key = store._compaction_telemetry_key("c1")
    store.write_compaction_telemetry("c1", dict(record))
    row = store._conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    assert row is not None

    # Empty conversation id is a no-op.
    store.write_compaction_telemetry("", {"x": 1})
    assert store.read_compaction_telemetry("") is None
    store.close()
