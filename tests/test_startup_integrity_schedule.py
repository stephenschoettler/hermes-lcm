"""Private-DB regressions for post-commit, worker-owned FTS scan claims."""

from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.db_bootstrap import ExternalContentFtsSpec, ensure_external_content_fts


@pytest.fixture(autouse=True)
def background_defaults(monkeypatch):
    monkeypatch.setenv(db_bootstrap.INTEGRITY_CHECK_INTERVAL_ENV, "24")
    monkeypatch.delenv(db_bootstrap.BACKGROUND_INTEGRITY_ENV, raising=False)
    yield
    db_bootstrap.join_background_integrity_scans(timeout=5)


def _spec(table="messages_fts"):
    return ExternalContentFtsSpec(
        table_name=table,
        content_table="messages",
        content_rowid="store_id",
        indexed_column="content",
        trigger_sqls=(),
    )


def _database(tmp_path, *, tables=("messages_fts",)):
    path = tmp_path / "private.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        "CREATE TABLE messages(store_id INTEGER PRIMARY KEY, content TEXT);"
        "INSERT INTO messages VALUES(1, 'searchable message');"
    )
    for table in tables:
        spec = _spec(table)
        ensure_external_content_fts(conn, spec)
        conn.execute(
            "UPDATE metadata SET value=? WHERE key=?",
            (str(time.time() - 100 * 3600), db_bootstrap._integrity_marker_key(spec)),
        )
    conn.commit()
    return path, conn


def _row(conn, key):
    return conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()


class _ObservedConnection:
    def __init__(self, conn, operations):
        self.conn = conn
        self.operations = operations

    def execute(self, sql, *args):
        self.operations.append((threading.get_ident(), sql))
        # Avoid making the red run wait for the known 30-second claim bug.
        if sql == "BEGIN IMMEDIATE" and threading.current_thread() is threading.main_thread():
            raise sqlite3.OperationalError("caller-thread claim forbidden")
        return self.conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.conn, name)


def _observe_connections(monkeypatch):
    operations = []
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        return _ObservedConnection(real_connect(*args, **kwargs), operations)

    monkeypatch.setattr(db_bootstrap.sqlite3, "connect", connect)
    return operations


def test_bootstrap_scope_dispatches_only_after_writer_commit(tmp_path, monkeypatch):
    _, conn = _database(tmp_path)
    launches = []
    real_start = threading.Thread.start

    def start(thread):
        launches.append(conn.in_transaction)
        real_start(thread)

    monkeypatch.setattr(db_bootstrap.threading.Thread, "start", start)
    try:
        with db_bootstrap.defer_integrity_scans(conn):
            conn.execute("UPDATE messages SET content=content WHERE store_id=1")
            ensure_external_content_fts(conn, _spec())
            assert launches == []
            conn.commit()
            assert launches == []
        assert launches == [False]
        db_bootstrap.join_background_integrity_scans(timeout=5)
        assert float(_row(conn, db_bootstrap._integrity_marker_key(_spec()))[0]) > time.time() - 10
    finally:
        conn.close()


@pytest.mark.parametrize("exit_kind", ["uncommitted", "rollback", "exception"])
def test_aborted_bootstrap_does_not_dispatch(tmp_path, monkeypatch, exit_kind):
    _, conn = _database(tmp_path)
    launches = []
    monkeypatch.setattr(db_bootstrap.threading.Thread, "start", lambda thread: launches.append(thread))
    try:
        try:
            with db_bootstrap.defer_integrity_scans(conn):
                conn.execute("UPDATE messages SET content=content WHERE store_id=1")
                ensure_external_content_fts(conn, _spec())
                # Later bootstrap migrations can open a fresh writer even if
                # the FTS helper committed its own initialization work.
                conn.execute("UPDATE messages SET content=content WHERE store_id=1")
                if exit_kind == "exception":
                    raise RuntimeError("bootstrap failed")
                if exit_kind == "rollback":
                    conn.rollback()
                    # A rollback alone cannot be distinguished from commit by SQLite;
                    # callers signal failed bootstrap by unwinding the scope.
                    raise RuntimeError("bootstrap rolled back")
        except RuntimeError:
            pass
        assert launches == []
    finally:
        conn.rollback()
        conn.close()


def test_nested_bootstrap_scopes_deduplicate_at_outer_commit(tmp_path, monkeypatch):
    _, conn = _database(tmp_path)
    scans = []
    monkeypatch.setattr(
        db_bootstrap,
        "check_external_content_fts_integrity",
        lambda connection, spec: scans.append(spec.table_name) or {"status": "pass", "detail": "ok"},
    )
    try:
        with db_bootstrap.defer_integrity_scans(conn):
            conn.execute("UPDATE messages SET content=content WHERE store_id=1")
            with db_bootstrap.defer_integrity_scans(conn):
                ensure_external_content_fts(conn, _spec())
                ensure_external_content_fts(conn, _spec())
            assert scans == []
            conn.commit()
        db_bootstrap.join_background_integrity_scans(timeout=5)
        assert scans == ["messages_fts"]
    finally:
        conn.close()


def test_scan_claim_is_worker_owned_and_has_bounded_busy_timeout(tmp_path, monkeypatch):
    _, conn = _database(tmp_path)
    operations = _observe_connections(monkeypatch)
    try:
        ensure_external_content_fts(conn, _spec())
        db_bootstrap.join_background_integrity_scans(timeout=5)
        claims = [tid for tid, sql in operations if sql == "BEGIN IMMEDIATE"]
        assert claims
        assert threading.get_ident() not in claims
        timeouts = [int(sql.split("=")[1]) for _, sql in operations if sql.startswith("PRAGMA busy_timeout=")]
        assert timeouts and max(timeouts) <= 1000
    finally:
        conn.close()


def test_unscoped_active_writer_defers_without_scan_or_synchronous_fallback(tmp_path, monkeypatch):
    _, conn = _database(tmp_path)
    scans = []
    monkeypatch.setattr(
        db_bootstrap,
        "check_external_content_fts_integrity",
        lambda connection, spec: scans.append(threading.get_ident()) or {"status": "pass", "detail": "ok"},
    )
    operations = _observe_connections(monkeypatch)
    try:
        conn.execute("UPDATE messages SET content=content WHERE store_id=1")
        ensure_external_content_fts(conn, _spec())
        db_bootstrap.join_background_integrity_scans(timeout=5)
        assert scans == []
        assert not any(sql == "BEGIN IMMEDIATE" for _, sql in operations)
    finally:
        conn.rollback()
        conn.close()


def test_claim_failure_defers_without_expensive_check(tmp_path, monkeypatch):
    _, conn = _database(tmp_path)
    real_connect = sqlite3.connect
    scans = []

    class ClaimFailure(_ObservedConnection):
        def execute(self, sql, *args):
            if sql == "BEGIN IMMEDIATE":
                raise sqlite3.OperationalError("database is locked")
            return self.conn.execute(sql, *args)

    monkeypatch.setattr(
        db_bootstrap.sqlite3, "connect", lambda *args, **kwargs: ClaimFailure(real_connect(*args, **kwargs), [])
    )
    monkeypatch.setattr(
        db_bootstrap,
        "check_external_content_fts_integrity",
        lambda connection, spec: scans.append(threading.get_ident()) or {"status": "pass", "detail": "ok"},
    )
    try:
        old = _row(conn, db_bootstrap._integrity_marker_key(_spec()))
        ensure_external_content_fts(conn, _spec())
        db_bootstrap.join_background_integrity_scans(timeout=5)
        assert scans == []
        assert _row(conn, db_bootstrap._integrity_marker_key(_spec())) == old
        assert _row(conn, db_bootstrap._integrity_scan_started_key(_spec())) is None
        assert db_bootstrap.load_integrity_failed(conn, _spec()) is None
    finally:
        conn.close()


def test_worker_rereads_fresh_integrity_marker_before_claim(tmp_path, monkeypatch):
    _, conn = _database(tmp_path)
    gate = threading.Event()
    scans = []
    real_worker = db_bootstrap._run_background_integrity_scan

    def delayed_worker(*args):
        assert gate.wait(5)
        real_worker(*args)

    monkeypatch.setattr(db_bootstrap, "_run_background_integrity_scan", delayed_worker)
    monkeypatch.setattr(
        db_bootstrap,
        "check_external_content_fts_integrity",
        lambda connection, spec: scans.append(spec.table_name) or {"status": "pass", "detail": "ok"},
    )
    try:
        ensure_external_content_fts(conn, _spec())
        db_bootstrap._record_integrity_checked(conn, _spec())
        conn.commit()
        gate.set()
        db_bootstrap.join_background_integrity_scans(timeout=5)
        assert scans == []
    finally:
        gate.set()
        db_bootstrap.join_background_integrity_scans(timeout=5)
        conn.close()


def test_worker_rereads_other_process_claim_and_does_not_clear_it(tmp_path, monkeypatch):
    _, conn = _database(tmp_path)
    gate = threading.Event()
    scans = []
    real_worker = db_bootstrap._run_background_integrity_scan

    def delayed_worker(*args):
        assert gate.wait(5)
        real_worker(*args)

    monkeypatch.setattr(db_bootstrap, "_run_background_integrity_scan", delayed_worker)
    monkeypatch.setattr(
        db_bootstrap,
        "check_external_content_fts_integrity",
        lambda connection, spec: scans.append(spec.table_name) or {"status": "pass", "detail": "ok"},
    )
    try:
        ensure_external_content_fts(conn, _spec())
        other_claim = time.time() + 1
        db_bootstrap._record_scan_started(conn, _spec(), now=other_claim)
        conn.commit()
        gate.set()
        db_bootstrap.join_background_integrity_scans(timeout=5)
        assert scans == []
        assert float(_row(conn, db_bootstrap._integrity_scan_started_key(_spec()))[0]) == other_claim
    finally:
        gate.set()
        db_bootstrap.join_background_integrity_scans(timeout=5)
        conn.close()


def test_one_scan_per_database_table(tmp_path, monkeypatch):
    _, conn = _database(tmp_path, tables=("messages_fts", "other_fts"))
    started = threading.Event()
    release = threading.Event()
    scans = []

    def scan(connection, spec):
        scans.append(spec.table_name)
        started.set()
        assert release.wait(5)
        return {"status": "pass", "detail": "ok"}

    monkeypatch.setattr(db_bootstrap, "check_external_content_fts_integrity", scan)
    try:
        ensure_external_content_fts(conn, _spec())
        assert started.wait(5)
        ensure_external_content_fts(conn, _spec())
        ensure_external_content_fts(conn, _spec("other_fts"))
        release.set()
        db_bootstrap.join_background_integrity_scans(timeout=5)
        assert sorted(scans) == ["messages_fts", "other_fts"]
    finally:
        release.set()
        db_bootstrap.join_background_integrity_scans(timeout=5)
        conn.close()


def test_dispatch_does_not_claim_until_worker_runs(tmp_path, monkeypatch):
    _, conn = _database(tmp_path)
    gate = threading.Event()
    entered = threading.Event()
    real_worker = db_bootstrap._run_background_integrity_scan

    def delayed_worker(*args):
        entered.set()
        assert gate.wait(5)
        real_worker(*args)

    monkeypatch.setattr(db_bootstrap, "_run_background_integrity_scan", delayed_worker)
    try:
        ensure_external_content_fts(conn, _spec())
        assert entered.wait(5)
        assert db_bootstrap._load_scan_started_at(conn, _spec()) is None
        gate.set()
        db_bootstrap.join_background_integrity_scans(timeout=5)
    finally:
        gate.set()
        db_bootstrap.join_background_integrity_scans(timeout=5)
        conn.close()


def test_real_writer_contention_is_bounded_and_does_not_scan(tmp_path, monkeypatch):
    path, conn = _database(tmp_path)
    scans = []
    blocker = sqlite3.connect(path)
    blocker.execute("BEGIN IMMEDIATE")
    monkeypatch.setattr(
        db_bootstrap,
        "check_external_content_fts_integrity",
        lambda connection, spec: scans.append(spec.table_name) or {"status": "pass", "detail": "ok"},
    )
    try:
        before = time.monotonic()
        assert db_bootstrap._dispatch_background_integrity_scan(conn, _spec())
        db_bootstrap.join_background_integrity_scans(timeout=2)
        elapsed = time.monotonic() - before
        assert elapsed < 1.5
        assert scans == []
        assert db_bootstrap._load_scan_started_at(conn, _spec()) is None
    finally:
        blocker.rollback()
        blocker.close()
        conn.close()


def test_replacement_lease_is_not_cleared_by_old_worker(tmp_path, monkeypatch):
    path, conn = _database(tmp_path)
    replacement = time.time() + 1

    def scan(connection, spec):
        other = sqlite3.connect(path)
        try:
            db_bootstrap._record_scan_started(other, spec, now=replacement)
            other.commit()
        finally:
            other.close()
        return {"status": "pass", "detail": "ok"}

    monkeypatch.setattr(db_bootstrap, "check_external_content_fts_integrity", scan)
    try:
        old = _row(conn, db_bootstrap._integrity_marker_key(_spec()))
        ensure_external_content_fts(conn, _spec())
        db_bootstrap.join_background_integrity_scans(timeout=5)
        assert db_bootstrap._load_scan_started_at(conn, _spec()) == replacement
        assert _row(conn, db_bootstrap._integrity_marker_key(_spec())) == old
    finally:
        conn.close()


def test_separate_processes_scan_database_table_once(tmp_path):
    path, conn = _database(tmp_path)
    conn.close()
    module_path = Path(db_bootstrap.__file__).resolve()
    # Import only db_bootstrap in the children: no host bootstrap or live DBs.
    script = r"""
import importlib.util, os, pathlib, sqlite3, sys, time
module_path, db_path, directory, number = sys.argv[1:]
root = pathlib.Path(directory)
spec = importlib.util.spec_from_file_location('private_bootstrap', module_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
fts = m.ExternalContentFtsSpec(table_name='messages_fts', content_table='messages',
    content_rowid='store_id', indexed_column='content', trigger_sqls=())
def wait(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, str(path)
        time.sleep(.01)
def scan(connection, spec):
    fd = os.open(root / 'scans', os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    try:
        os.write(fd, (number + '\n').encode())
    finally:
        os.close(fd)
    wait(root / 'release')
    return {'status': 'pass', 'detail': 'ok'}
m.check_external_content_fts_integrity = scan
conn = sqlite3.connect(db_path)
(root / ('ready' + number)).touch()
wait(root / 'go')
assert m._dispatch_background_integrity_scan(conn, fts)
m.join_background_integrity_scans(timeout=10)
assert not m._integrity_scan_threads
conn.close()
(root / ('done' + number)).touch()
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(module_path), str(path), str(tmp_path), str(number)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for number in range(2)
    ]

    def wait_until(predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            assert time.monotonic() < deadline, "private worker did not reach barrier"
            time.sleep(.01)

    try:
        wait_until(lambda: all((tmp_path / ("ready" + str(n))).exists() for n in range(2)))
        (tmp_path / "go").touch()
        wait_until(lambda: (tmp_path / "scans").exists())
        # The losing process must finish without scanning while the winning
        # process holds the committed lease but no SQLite writer transaction.
        wait_until(lambda: any((tmp_path / ("done" + str(n))).exists() for n in range(2)))
        assert len((tmp_path / "scans").read_text().splitlines()) == 1
        (tmp_path / "release").touch()
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, (stdout, stderr)
        assert len((tmp_path / "scans").read_text().splitlines()) == 1
    finally:
        (tmp_path / "release").touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)


def test_memory_database_still_checks_synchronously(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE messages(store_id INTEGER PRIMARY KEY, content TEXT);"
        "INSERT INTO messages VALUES(1, 'searchable message');"
    )
    scans = []
    try:
        ensure_external_content_fts(conn, _spec())
        conn.execute("UPDATE metadata SET value='0' WHERE key=?", (db_bootstrap._integrity_marker_key(_spec()),))
        conn.commit()
        real_check = db_bootstrap.check_external_content_fts_integrity

        def scan(connection, spec):
            scans.append(threading.get_ident())
            return real_check(connection, spec)

        monkeypatch.setattr(db_bootstrap, "check_external_content_fts_integrity", scan)
        with db_bootstrap.defer_integrity_scans(conn):
            ensure_external_content_fts(conn, _spec())
            conn.commit()
        assert scans == [threading.get_ident()]
    finally:
        conn.close()
