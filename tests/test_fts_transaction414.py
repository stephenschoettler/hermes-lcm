"""Real SQLite VM controls for automatic FTS transaction ownership (review413).

Adapted from the independent review repro, which remains unchanged. The clock
expires at the real FTS command; neither the command nor SQLite's error is faked.
"""
import sqlite3
from types import SimpleNamespace

import pytest

from . import test_rework411 as rework
from hermes_lcm import db_bootstrap as boot
from hermes_lcm.sqlite_util import _InitializingConnection
from hermes_lcm.store import MessageStore, build_message_fts_spec


@pytest.fixture
def ready(tmp_path, monkeypatch):
    return rework.ready.__wrapped__(tmp_path, monkeypatch)


def _arm_scan(monkeypatch, *, interrupt):
    clock, entered, errors = [0.0], [], []
    original = _InitializingConnection.execute

    def execute(conn, sql, parameters=(), /):
        if 'integrity-check' in sql:
            entered.append(conn)
            if interrupt:
                clock[0] = 1.0
        try:
            return original(conn, sql, parameters)
        except sqlite3.OperationalError as exc:
            errors.append(getattr(exc, 'sqlite_errorcode', None))
            raise

    monkeypatch.setattr(boot, 'time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 123.0))
    monkeypatch.setattr(_InitializingConnection, 'execute', execute)
    monkeypatch.setenv('LCM_FTS_INTEGRITY_BACKGROUND', 'false')
    monkeypatch.setattr(boot, '_should_run_integrity_check', lambda *a, **kw: True)
    return entered, errors


def _prepare(store, conn):
    store.append_batch('scan-data', [{'role': 'user', 'content': 'alpha beta gamma delta ' * 30}] * 2000, [1] * 2000)
    conn.execute("DELETE FROM metadata WHERE key='fts_integrity_checked_at:messages_fts'")
    conn.execute("INSERT INTO metadata VALUES('fts_integrity_failed:messages_fts','prior-failure')")
    conn.commit()


def _markers(conn):
    return [tuple(row) for row in conn.execute("SELECT key,value FROM metadata WHERE key LIKE 'fts_integrity_%:messages_fts' ORDER BY key")]


def _call(conn, public_entry):
    spec = build_message_fts_spec()
    return (boot.ensure_external_content_fts(conn, spec) if public_entry
            else boot._bounded_synchronous_integrity(conn, spec))


def _handler(conn):
    calls = []

    def callback():
        calls.append(True)
        return 0

    conn.set_progress_handler(callback, 7)
    conn.execute('PRAGMA busy_timeout=137')
    return callback, calls


def _assert_settings(conn, callback, calls, managed):
    assert conn.execute('PRAGMA busy_timeout').fetchone()[0] == 137
    if managed:
        assert conn._progress_handler == (callback, 7)
    calls.clear()
    conn.execute('WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100) SELECT sum(x) FROM n').fetchone()
    assert calls  # Also verify the opaque stdlib caller handler is still installed.


@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('public_entry', [False, True])
def test_automatic_fts_defers_outer_transaction_before_real_vm(ready, monkeypatch, managed, public_entry):
    store = MessageStore(ready)
    conn = store._conn if managed else sqlite3.connect(ready, timeout=0)
    try:
        _prepare(store, conn)
        before = _markers(conn)
        callback, calls = _handler(conn)
        conn.execute("INSERT INTO metadata VALUES('outer-414','must-survive')")
        entered, errors = _arm_scan(monkeypatch, interrupt=True)
        result = _call(conn, public_entry)
        # Check the actual caller state first: an unchecked return cannot hide loss.
        assert conn.in_transaction
        assert conn.execute("SELECT value FROM metadata WHERE key='outer-414'").fetchone()[0] == 'must-survive'
        if not public_entry:
            assert result['status'] == 'unchecked'
        assert entered == [] and errors == []
        assert _markers(conn) == before
        _assert_settings(conn, callback, calls, managed)
        with sqlite3.connect(ready, timeout=0) as peer:
            assert peer.execute("SELECT value FROM metadata WHERE key='outer-414'").fetchone() is None
        conn.rollback()
        assert conn.execute("SELECT value FROM metadata WHERE key='outer-414'").fetchone() is None
    finally:
        conn.rollback()
        if not managed:
            conn.close()
        store.close()


@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('public_entry', [False, True])
@pytest.mark.parametrize('interrupt', [False, True])
def test_transaction_free_automatic_fts_runs_real_bounded_vm(ready, monkeypatch, managed, public_entry, interrupt):
    store = MessageStore(ready)
    conn = store._conn if managed else sqlite3.connect(ready, timeout=0)
    try:
        _prepare(store, conn)
        before = _markers(conn)
        callback, calls = _handler(conn)
        assert not conn.in_transaction
        entered, errors = _arm_scan(monkeypatch, interrupt=interrupt)
        result = _call(conn, public_entry)
        assert len(entered) == 1  # No accidental permanent deferral or synchronous retry.
        if interrupt:
            assert sqlite3.SQLITE_INTERRUPT in errors  # Raised by the real SQLite VM.
        else:
            assert errors == []
        if not public_entry:
            assert result['status'] == ('unchecked' if interrupt else 'pass')
        # Only a successful public maintenance scan publishes the checked marker
        # and clears the prior failure. Direct diagnostics do neither.
        expected = [('fts_integrity_checked_at:messages_fts', '123.0')]
        assert _markers(conn) == (expected if public_entry and not interrupt else before)
        assert not conn.in_transaction
        _assert_settings(conn, callback, calls, managed)
        with sqlite3.connect(ready, timeout=0) as peer:
            peer.execute('BEGIN IMMEDIATE')
            peer.rollback()
    finally:
        conn.rollback()
        if not managed:
            conn.close()
        store.close()
