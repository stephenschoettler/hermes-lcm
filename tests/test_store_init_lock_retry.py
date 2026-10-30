"""MessageStore open retries transient 'database is locked' instead of failing plugin load."""
import sqlite3

import pytest

from hermes_lcm.store import MessageStore


def test_open_retries_transient_lock(tmp_path, monkeypatch):
    real = MessageStore._init_db_once
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return real(self)

    monkeypatch.setattr(MessageStore, "_init_db_once", flaky)
    monkeypatch.setattr("hermes_lcm.store.time.sleep", lambda s: None)
    store = MessageStore(tmp_path / "lcm.db")
    assert calls["n"] == 3
    assert store._conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 0
    store._conn.close()


def test_open_gives_up_after_budget(tmp_path, monkeypatch):
    def always(self):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(MessageStore, "_init_db_once", always)
    monkeypatch.setattr(MessageStore, "_INIT_LOCK_RETRY_BUDGET_S", 0.0)
    with pytest.raises(sqlite3.OperationalError):
        MessageStore(tmp_path / "lcm.db")


def test_non_lock_errors_not_retried(tmp_path, monkeypatch):
    calls = {"n": 0}

    def boom(self):
        calls["n"] += 1
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(MessageStore, "_init_db_once", boom)
    with pytest.raises(sqlite3.OperationalError):
        MessageStore(tmp_path / "lcm.db")
    assert calls["n"] == 1
