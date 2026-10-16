"""Lazy storage binding tests for cloned LCM engines."""

import copy
import sqlite3

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def test_deepcopy_does_not_open_sqlite_until_first_storage_use(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "lazy-copy.db"),
        temporal_rollups_enabled=True,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    try:
        engine._store.append("session-a", {"role": "user", "content": "hello"})

        real_connect = sqlite3.connect
        connect_calls = 0
        rollup_initializations = 0

        def counting_connect(*args, **kwargs):
            nonlocal connect_calls
            connect_calls += 1
            return real_connect(*args, **kwargs)

        def counting_initialize_rollups(dag):
            nonlocal rollup_initializations
            rollup_initializations += 1

        monkeypatch.setattr(sqlite3, "connect", counting_connect)
        monkeypatch.setattr(
            "hermes_lcm.engine.initialize_rollup_invalidation_outbox",
            counting_initialize_rollups,
        )

        clone = copy.deepcopy(engine)

        assert connect_calls == 0
        assert rollup_initializations == 0
        assert clone._store.get_session_count("session-a") == 1
        assert connect_calls > 0
        assert rollup_initializations == 1

        connect_calls_after_bind = connect_calls
        assert clone._store.get_session_count("session-a") == 1
        assert connect_calls == connect_calls_after_bind
        assert rollup_initializations == 1
    finally:
        engine.shutdown()
        if "clone" in locals():
            clone.shutdown()


def test_lazy_clone_shutdown_is_noop_without_sqlite_connect(tmp_path, monkeypatch):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "lazy-shutdown.db")),
        hermes_home=str(tmp_path / "hermes"),
    )
    try:
        clone = copy.deepcopy(engine)

        def fail_connect(*args, **kwargs):
            raise AssertionError("lazy clone shutdown opened sqlite")

        monkeypatch.setattr(sqlite3, "connect", fail_connect)

        clone.shutdown()
    finally:
        engine.shutdown()


def test_deepcopy_loop_does_not_open_sqlite(tmp_path, monkeypatch):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "lazy-loop.db")),
        hermes_home=str(tmp_path / "hermes"),
    )
    try:
        real_connect = sqlite3.connect
        connect_calls = 0

        def counting_connect(*args, **kwargs):
            nonlocal connect_calls
            connect_calls += 1
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", counting_connect)

        clones = [copy.deepcopy(engine) for _ in range(20)]

        assert connect_calls == 0
    finally:
        engine.shutdown()
        for clone in locals().get("clones", []):
            clone.shutdown()


def test_storage_quick_check_refuses_corrupt_database_before_write_connect(
    tmp_path, monkeypatch, caplog
):
    db_path = tmp_path / "corrupt.db"
    db_path.write_bytes(b"SQLite format 3\x00" + b"not a valid sqlite database" * 64)
    lcm_engine._STORAGE_QUICKCHECK_CACHE.clear()

    real_connect = sqlite3.connect
    write_connects = 0

    def counting_connect(*args, **kwargs):
        nonlocal write_connects
        if not kwargs.get("uri"):
            write_connects += 1
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", counting_connect)
    caplog.set_level("CRITICAL", logger="hermes_lcm.engine")

    with pytest.raises(RuntimeError, match="failed quick_check"):
        LCMEngine(
            config=LCMConfig(database_path=str(db_path)),
            hermes_home=str(tmp_path / "hermes"),
        )

    assert write_connects == 0
    assert "plugin stays offline, repair required" in caplog.text


def test_storage_quick_check_allows_healthy_database(tmp_path):
    db_path = tmp_path / "healthy.db"
    engine = LCMEngine(
        config=LCMConfig(database_path=str(db_path)),
        hermes_home=str(tmp_path / "hermes"),
    )
    try:
        engine._store.append("session-a", {"role": "user", "content": "hello"})
        assert engine._store.get_session_count("session-a") == 1
    finally:
        engine.shutdown()
