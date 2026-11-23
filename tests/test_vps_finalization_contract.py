"""A02 synthetic real-SQLite contention experiment; no production data."""

import json
import sqlite3
import threading
import time
import pytest
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
import hermes_lcm.engine as engine_module


@pytest.mark.parametrize("repeat", range(3))
@pytest.mark.parametrize(
    "hold_ms",
    [0, 150, 350, None],
    ids=["uncontended", "release150ms", "release350ms", "held-through-return"],
)
@pytest.mark.parametrize("timeout_ms", [50, 500])
def test_session_end_contention_tradeoff(
    tmp_path, monkeypatch, timeout_ms, hold_ms, repeat
):
    monkeypatch.setattr(engine_module, "_SESSION_END_BUSY_TIMEOUT_MS", timeout_ms)
    e = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "synthetic.db")),
        hermes_home=str(tmp_path / "home"),
    )
    ready = threading.Event()
    begin = threading.Event()
    finish = threading.Event()
    failures = []
    worker = None
    try:
        e.on_session_start("synthetic-session", platform="cli")
        e._store._conn.execute("PRAGMA busy_timeout=750")
        e._lifecycle._conn.execute("PRAGMA busy_timeout=750")

        def lock_holder():
            try:
                c = sqlite3.connect(
                    str(e._store.db_path), timeout=2, isolation_level=None
                )
                c.execute("BEGIN IMMEDIATE")
                ready.set()
                assert begin.wait(3)
                if hold_ms is None:
                    assert finish.wait(4)
                else:
                    time.sleep(hold_ms / 1000)
                c.execute("ROLLBACK")
                c.close()
            except BaseException as ex:
                failures.append(repr(ex))
                ready.set()

        if hold_ms != 0:
            worker = threading.Thread(target=lock_holder)
            worker.start()
            assert ready.wait(3) and not failures
        original = e._ingest_messages

        def ingest(messages):
            begin.set()
            return original(messages)

        monkeypatch.setattr(e, "_ingest_messages", ingest)
        started = time.monotonic()
        e.on_session_end(
            "synthetic-session",
            [{"role": "user", "content": "A02 SYNTHETIC FINAL MESSAGE"}],
        )
        elapsed = time.monotonic() - started
        finish.set()
        if worker:
            worker.join(4)
            assert not worker.is_alive()
        assert not failures
        rows = e._store.get_session_count("synthetic-session")
        expected = (
            1 if hold_ms == 0 or (hold_ms is not None and hold_ms < timeout_ms) else 0
        )
        assert rows == expected
        assert e._store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 750
        assert e._lifecycle._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 750
        assert elapsed < 1.2
        if hold_ms is None:
            assert elapsed >= timeout_ms / 1000 * 0.85
        elif hold_ms and expected:
            assert elapsed >= hold_ms / 1000 * 0.85
        result = {
            "timeout_ms": timeout_ms,
            "hold_ms": hold_ms,
            "repeat": repeat,
            "elapsed_ms": round(elapsed * 1000, 3),
            "stored_final_messages": rows,
            "prior_busy_timeouts_restored": True,
        }
        with (tmp_path / "contention-result.json").open("w") as f:
            f.write(json.dumps(result) + "\n")
    finally:
        begin.set()
        finish.set()
        if worker:
            worker.join(4)
        e.shutdown()


def test_installed_default_retains_bounded_finalization_wait():
    assert engine_module._SESSION_END_BUSY_TIMEOUT_MS == 500
