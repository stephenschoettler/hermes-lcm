"""Tests for FTS startup integrity-check throttling (issue #235).

The FTS5 ``integrity-check`` is O(index size) and was run unconditionally on
every startup where the index already exists and is structurally sound,
dominating launch time on large databases. These tests pin the throttled
behavior: the deep check runs at most once per configurable interval, while the
cheap structural checks always run.

Note on behavior model: a brand-new database takes the ``structural -> rebuild``
path and does NOT run integrity-check; the expensive check only fires on
subsequent startups of an existing, structurally-sound index. The tests build
the index first, then exercise the existing-index path.
"""

import sqlite3
import threading
import time
import types

import pytest

from hermes_lcm import command, db_bootstrap
from hermes_lcm.db_bootstrap import (
    ExternalContentFtsSpec,
    ensure_external_content_fts,
)

INTERVAL_ENV = "LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS"
MARKER_KEY = "fts_integrity_checked_at:messages_fts"


def _make_conn(tmp_path, name="t.db"):
    conn = sqlite3.connect(str(tmp_path / name))
    conn.executescript(
        """
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT
        );
        INSERT INTO messages(content) VALUES ('hello world');
        INSERT INTO messages(content) VALUES ('second searchable message');
        """
    )
    return conn


def _spec():
    return ExternalContentFtsSpec(
        table_name="messages_fts",
        content_table="messages",
        content_rowid="store_id",
        indexed_column="content",
        trigger_sqls=(),
    )


def _make_future_schema_db(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
            (str(db_bootstrap.SCHEMA_VERSION + 1),),
        )
        conn.commit()
    finally:
        conn.close()


def _journal_mode(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()


def _table_names(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()


@pytest.fixture
def integrity_calls(monkeypatch):
    """Spy that counts real integrity-check invocations by table name."""
    calls = []
    real = db_bootstrap.check_external_content_fts_integrity

    def spy(conn, spec):
        calls.append(spec.table_name)
        return real(conn, spec)

    monkeypatch.setattr(db_bootstrap, "check_external_content_fts_integrity", spy)
    return calls


def _marker(conn):
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?", (MARKER_KEY,)
    ).fetchone()
    return row[0] if row else None


def test_existing_index_without_marker_runs_check_and_records_marker(tmp_path, monkeypatch, integrity_calls):
    # Kill-switch off pins the synchronous throttle decision this test asserts;
    # the async dispatch is covered separately below.
    monkeypatch.setenv("LCM_FTS_INTEGRITY_BACKGROUND", "false")
    conn = _make_conn(tmp_path)
    ensure_external_content_fts(conn, _spec())  # builds index (rebuild path)
    # Simulate an existing DB upgraded to the throttling version: no marker yet.
    conn.execute("DELETE FROM metadata WHERE key = ?", (MARKER_KEY,))
    integrity_calls.clear()

    ensure_external_content_fts(conn, _spec())

    assert integrity_calls == ["messages_fts"]
    assert _marker(conn) is not None
    conn.close()


def test_fresh_marker_skips_integrity_check(tmp_path, monkeypatch, integrity_calls):
    monkeypatch.setenv(INTERVAL_ENV, "24")
    conn = _make_conn(tmp_path)
    ensure_external_content_fts(conn, _spec())  # build records a fresh marker
    integrity_calls.clear()

    ensure_external_content_fts(conn, _spec())

    assert integrity_calls == []  # fresh marker -> deep check skipped
    conn.close()


def test_expired_marker_reruns_integrity_check(tmp_path, monkeypatch, integrity_calls):
    monkeypatch.setenv(INTERVAL_ENV, "24")
    monkeypatch.setenv("LCM_FTS_INTEGRITY_BACKGROUND", "false")
    conn = _make_conn(tmp_path)
    ensure_external_content_fts(conn, _spec())
    # Age the marker well past the 24h interval.
    conn.execute(
        "UPDATE metadata SET value = ? WHERE key = ?",
        (str(time.time() - 100 * 3600), MARKER_KEY),
    )
    integrity_calls.clear()

    ensure_external_content_fts(conn, _spec())

    assert integrity_calls == ["messages_fts"]
    conn.close()


def test_interval_zero_checks_every_init(tmp_path, monkeypatch, integrity_calls):
    monkeypatch.setenv(INTERVAL_ENV, "0")
    monkeypatch.setenv("LCM_FTS_INTEGRITY_BACKGROUND", "false")
    conn = _make_conn(tmp_path)
    ensure_external_content_fts(conn, _spec())  # build
    integrity_calls.clear()

    ensure_external_content_fts(conn, _spec())
    ensure_external_content_fts(conn, _spec())

    assert integrity_calls == ["messages_fts", "messages_fts"]
    conn.close()


def test_negative_interval_never_checks_on_startup(tmp_path, monkeypatch, integrity_calls):
    monkeypatch.setenv(INTERVAL_ENV, "-1")
    conn = _make_conn(tmp_path)
    ensure_external_content_fts(conn, _spec())  # build
    integrity_calls.clear()

    ensure_external_content_fts(conn, _spec())

    assert integrity_calls == []
    conn.close()


def test_structural_mismatch_rebuilds_despite_fresh_marker(tmp_path, monkeypatch, integrity_calls):
    monkeypatch.setenv(INTERVAL_ENV, "24")
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)  # build + fresh marker, index has 2 docs

    # Insert a row without a trigger (spec has none): the FTS index now lags
    # content. Marker is fresh, so the deep integrity-check is throttled, but
    # the structural check must still detect the desync and rebuild.
    conn.execute("INSERT INTO messages(content) VALUES ('untracked row')")
    integrity_calls.clear()

    ensure_external_content_fts(conn, spec)

    assert integrity_calls == []  # repaired via structural path, not deep check
    assert db_bootstrap._fts_needs_rebuild_structural(conn, spec) is False
    conn.close()


def test_external_content_desync_detected_via_docsize(tmp_path):
    """Content-vs-index row-count comparison must detect real desync.

    For an external-content FTS5 table, ``COUNT(*) FROM <fts>`` reads through to
    the content table and cannot reveal a lagging index; ``<fts>_docsize`` holds
    the true indexed-document count. This guards the switch to docsize.
    """
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)
    assert db_bootstrap._fts_needs_rebuild_structural(conn, spec) is False

    # Insert without a trigger: indexed doc count (2) now lags content (3).
    conn.execute("INSERT INTO messages(content) VALUES ('untracked row')")
    assert db_bootstrap._fts_needs_rebuild_structural(conn, spec) is True
    conn.close()


def test_explicit_repair_fixes_same_count_corruption_despite_fresh_marker(tmp_path, monkeypatch):
    """`/lcm doctor repair apply` must deep-check/repair regardless of throttle.

    Regression for review on PR #236: the startup throttle must not leak into
    the explicit repair path. Same-row-count stale drift passes structural
    checks but fails the FTS5 integrity-check; with a fresh marker the throttle
    would otherwise skip the repair entirely.
    """
    monkeypatch.setenv(INTERVAL_ENV, "24")
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)  # build + fresh marker (startup path)

    # Content changes but the index does not (spec has no update trigger): the
    # row count is unchanged, so structural checks pass, but the indexed tokens
    # are stale and the integrity-check fails.
    conn.execute(
        "UPDATE messages SET content = 'completely different searchable text' WHERE store_id = 1"
    )
    assert db_bootstrap._fts_needs_rebuild_structural(conn, spec) is False
    assert db_bootstrap.check_external_content_fts_integrity(conn, spec)["status"] == "fail"

    # Explicit repair (doctor path) is unthrottled and must rebuild + fix it.
    repaired = db_bootstrap.repair_external_content_fts(conn, spec)
    assert repaired["rebuilt"] is True
    assert db_bootstrap.check_external_content_fts_integrity(conn, spec)["status"] == "pass"
    conn.close()


def test_explicit_repair_clears_stuck_integrity_failed_flag(tmp_path, monkeypatch):
    """A successful repair clears a prior background-scan corruption flag (F1).

    Regression for F1: `repair_external_content_fts` never cleared
    `fts_integrity_failed:<table>`, so after `/lcm doctor repair apply` succeeded
    `/lcm doctor` kept reporting issues-found forever and the next self-healing
    scan was pushed out a full interval.
    """
    monkeypatch.setenv(INTERVAL_ENV, "24")
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)  # build + fresh marker

    # A background scan flagged the index as corrupt.
    db_bootstrap._record_integrity_failed(conn, spec, detail="synthetic corruption")
    conn.commit()
    assert db_bootstrap.load_integrity_failed(conn, spec) is not None

    # Drive a same-count stale-drift corruption so the explicit repair rebuilds.
    conn.execute(
        "UPDATE messages SET content = 'completely different searchable text' WHERE store_id = 1"
    )
    repaired = db_bootstrap.repair_external_content_fts(conn, spec)
    assert repaired["rebuilt"] is True
    # The flag is cleared in the same transaction as the rebuild.
    assert db_bootstrap.load_integrity_failed(conn, spec) is None
    conn.close()


def test_repair_without_rebuild_still_clears_integrity_failed_flag(tmp_path, monkeypatch):
    """Even a no-op repair (nothing to rebuild) clears a stale corruption flag."""
    monkeypatch.setenv(INTERVAL_ENV, "24")
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)

    db_bootstrap._record_integrity_failed(conn, spec, detail="stale flag")
    conn.commit()
    assert db_bootstrap.load_integrity_failed(conn, spec) is not None

    # Index is healthy: repair makes no rebuild but must still clear the flag.
    repaired = db_bootstrap.repair_external_content_fts(conn, spec, throttle=True)
    assert repaired["rebuilt"] is False
    assert db_bootstrap.load_integrity_failed(conn, spec) is None
    conn.close()


class _RaisingConn:
    """Delegates to a real connection but raises on the integrity-check INSERT."""

    def __init__(self, real, exc):
        self._real = real
        self._exc = exc

    def execute(self, sql, *args):
        if "integrity-check" in sql:
            raise self._exc
        return self._real.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_is_fts_corruption_error_classification():
    """Only real corruption signatures classify as corruption (F3)."""
    assert db_bootstrap._is_fts_corruption_error("database disk image is malformed")
    assert db_bootstrap._is_fts_corruption_error("malformed database schema")
    assert db_bootstrap._is_fts_corruption_error("file is not a database")
    assert db_bootstrap._is_fts_corruption_error(
        'fts5: checksum mismatch for table "messages_fts"'
    )
    assert not db_bootstrap._is_fts_corruption_error("database is locked")
    assert not db_bootstrap._is_fts_corruption_error("database table is locked")
    assert not db_bootstrap._is_fts_corruption_error("query timeout expired")


def test_integrity_check_lock_error_is_unchecked_not_corruption(tmp_path, monkeypatch):
    """A transient lock/busy error classifies as 'unchecked', never 'fail' (F3).

    Reclassifying a lock-timeout as corruption would wedge a false
    fts_integrity_failed flag (the same stuck false-positive as F1, via a race).
    """
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)
    monkeypatch.setattr(db_bootstrap, "_fts_needs_rebuild_structural", lambda *a, **k: False)

    fake = _RaisingConn(conn, sqlite3.OperationalError("database is locked"))
    result = db_bootstrap.check_external_content_fts_integrity(fake, spec)
    assert result["status"] == "unchecked"
    conn.close()


def test_integrity_check_malformed_error_is_fail(tmp_path, monkeypatch):
    """A genuine corruption signature still classifies as 'fail' (F3)."""
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)
    monkeypatch.setattr(db_bootstrap, "_fts_needs_rebuild_structural", lambda *a, **k: False)

    fake = _RaisingConn(conn, sqlite3.DatabaseError("database disk image is malformed"))
    result = db_bootstrap.check_external_content_fts_integrity(fake, spec)
    assert result["status"] == "fail"
    conn.close()


def test_doctor_repair_apply_joins_background_scans_first(tmp_path, monkeypatch):
    """Explicit repair joins in-flight background scans before repairing (F3)."""
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine

    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
        hermes_home=str(tmp_path / "home"),
    )
    order: list[str] = []
    monkeypatch.setattr(
        command, "join_background_integrity_scans",
        lambda *a, **k: order.append("join"),
    )
    real_repair = command.repair_external_content_fts

    def spy_repair(conn, spec, **kwargs):
        order.append("repair")
        return real_repair(conn, spec, **kwargs)

    monkeypatch.setattr(command, "repair_external_content_fts", spy_repair)
    command._doctor_repair_apply_text(engine)
    assert order[0] == "join"
    assert "repair" in order


def test_startup_throttle_still_skips_explicitly(tmp_path, monkeypatch, integrity_calls):
    """The throttle remains available on the startup path via throttle=True."""
    monkeypatch.setenv(INTERVAL_ENV, "24")
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)  # build + fresh marker
    integrity_calls.clear()

    db_bootstrap.repair_external_content_fts(conn, spec, throttle=True)

    assert integrity_calls == []  # fresh marker -> throttled path skips deep check
    conn.close()


def _db_file(tmp_path, name="t.db"):
    return str(tmp_path / name)


def _age_marker(conn):
    conn.execute(
        "UPDATE metadata SET value = ? WHERE key = ?",
        (str(time.time() - 100 * 3600), MARKER_KEY),
    )
    conn.commit()


def test_due_marker_runs_deep_check_in_background_and_stamps_marker(tmp_path, monkeypatch):
    """SPEC E (a): a due marker dispatches the deep scan to a background thread.

    The bind path returns without running the O(index) check itself; the scan
    runs on a daemon thread and stamps the throttle marker on clean completion.
    """
    monkeypatch.setenv(INTERVAL_ENV, "24")
    monkeypatch.delenv("LCM_FTS_INTEGRITY_BACKGROUND", raising=False)  # default: on
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)  # build + fresh marker (no deep check)
    _age_marker(conn)
    aged = float(_marker(conn))

    ran_on = {}
    real = db_bootstrap.check_external_content_fts_integrity

    def spy(conn_, spec_):
        ran_on["thread"] = threading.current_thread()
        return real(conn_, spec_)

    monkeypatch.setattr(db_bootstrap, "check_external_content_fts_integrity", spy)

    ensure_external_content_fts(conn, spec)  # should dispatch, not block
    db_bootstrap.join_background_integrity_scans(timeout=30)

    # The deep check ran on a background (non-main) thread, not the bind thread.
    assert ran_on.get("thread") is not None
    assert ran_on["thread"] is not threading.main_thread()

    # The background scan stamped a fresh marker (via its own connection).
    verify = sqlite3.connect(_db_file(tmp_path))
    try:
        new = float(
            verify.execute(
                "SELECT value FROM metadata WHERE key = ?", (MARKER_KEY,)
            ).fetchone()[0]
        )
    finally:
        verify.close()
    assert new > aged
    conn.close()


def test_background_scan_flags_corruption_without_rebuilding(tmp_path, monkeypatch):
    """SPEC E (b): corruption found in the background writes a flag, no rebuild."""
    monkeypatch.setenv(INTERVAL_ENV, "24")
    monkeypatch.delenv("LCM_FTS_INTEGRITY_BACKGROUND", raising=False)
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)  # build + fresh marker

    # Same-row-count stale drift: structural checks pass, deep check fails (the
    # spec has no update trigger, so the indexed tokens go stale in place).
    conn.execute(
        "UPDATE messages SET content = 'completely different searchable text' WHERE store_id = 1"
    )
    conn.commit()
    assert db_bootstrap._fts_needs_rebuild_structural(conn, spec) is False
    _age_marker(conn)

    ensure_external_content_fts(conn, spec)  # dispatch background scan
    db_bootstrap.join_background_integrity_scans(timeout=30)

    verify = sqlite3.connect(_db_file(tmp_path))
    try:
        flag = db_bootstrap.load_integrity_failed(verify, spec)
        # The background thread flags rather than rebuilds: corruption persists.
        assert (
            db_bootstrap.check_external_content_fts_integrity(verify, spec)["status"]
            == "fail"
        )
    finally:
        verify.close()
    assert flag is not None
    assert flag["at"] > 0
    conn.close()


def test_dispatch_stamps_scan_started_before_thread_runs(tmp_path, monkeypatch):
    """The dispatcher durably stamps scan_started_at before the thread runs (F6).

    A second process racing dispatch in the window before the thread commits its
    own stamp must see the claim and not launch a duplicate deep scan.
    """
    monkeypatch.setenv(INTERVAL_ENV, "24")
    monkeypatch.delenv("LCM_FTS_INTEGRITY_BACKGROUND", raising=False)
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)
    _age_marker(conn)

    # Block the scan thread body so it can NOT be the one that stamps.
    release = threading.Event()
    monkeypatch.setattr(
        db_bootstrap, "_run_background_integrity_scan",
        lambda *a, **k: release.wait(30),
    )

    dispatched = db_bootstrap._dispatch_background_integrity_scan(conn, spec)
    assert dispatched is True
    try:
        # A separate connection sees the stamp already committed by the dispatcher.
        verify = sqlite3.connect(_db_file(tmp_path))
        try:
            started = db_bootstrap._load_scan_started_at(verify, spec)
        finally:
            verify.close()
        # The dispatcher wrote it — the thread body is blocked and cannot have.
        assert started is not None
    finally:
        release.set()
        db_bootstrap.join_background_integrity_scans(timeout=30)
    conn.close()


def test_plugin_unload_waits_for_background_integrity_scan(tmp_path, monkeypatch):
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine

    db_path = tmp_path / "lcm.db"
    engine = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    scan_started = threading.Event()
    release_scan = threading.Event()

    def blocking_scan(path, _spec, _started_at):
        scan_conn = sqlite3.connect(path, check_same_thread=False)
        try:
            scan_started.set()
            if not release_scan.wait(timeout=3):
                raise AssertionError("test did not release integrity scan")
        finally:
            scan_conn.close()

    monkeypatch.setattr(
        db_bootstrap,
        "_run_background_integrity_scan",
        blocking_scan,
    )
    assert db_bootstrap._dispatch_background_integrity_scan(
        engine._store._conn,
        _spec(),
    )
    assert scan_started.wait(timeout=1)

    unload_done = threading.Event()
    unload_errors = []

    def unload_plugin():
        try:
            engine.shutdown_all_instances()
        except BaseException as exc:  # captured for assertion in the main thread
            unload_errors.append(exc)
        finally:
            unload_done.set()

    unload_thread = threading.Thread(target=unload_plugin)
    unload_thread.start()
    unload_waited = not unload_done.wait(timeout=0.1)
    release_scan.set()
    unload_thread.join(timeout=3)
    db_bootstrap.join_background_integrity_scans(timeout=2)

    assert unload_waited
    assert not unload_thread.is_alive()
    assert unload_errors == []
    db_path.unlink()
    assert not db_path.exists()


def test_kill_switch_false_runs_synchronously_without_a_thread(tmp_path, monkeypatch, integrity_calls):
    """SPEC E (c): LCM_FTS_INTEGRITY_BACKGROUND=false = exact old synchronous path."""
    monkeypatch.setenv(INTERVAL_ENV, "24")
    monkeypatch.setenv("LCM_FTS_INTEGRITY_BACKGROUND", "false")
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)  # build + fresh marker
    _age_marker(conn)
    integrity_calls.clear()

    ensure_external_content_fts(conn, spec)  # runs the deep check synchronously

    assert integrity_calls == ["messages_fts"]
    assert (_db_file(tmp_path), "messages_fts") not in db_bootstrap._integrity_scan_threads
    conn.close()


def test_only_one_background_scan_per_table_at_a_time(tmp_path, monkeypatch):
    """SPEC E (d): a second dispatch while a scan is in flight does not spawn another."""
    monkeypatch.setenv(INTERVAL_ENV, "24")
    monkeypatch.delenv("LCM_FTS_INTEGRITY_BACKGROUND", raising=False)
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)  # build + fresh marker
    _age_marker(conn)

    started = threading.Event()
    release = threading.Event()
    real = db_bootstrap.check_external_content_fts_integrity

    def slow(conn_, spec_):
        started.set()
        release.wait(5)
        return real(conn_, spec_)

    monkeypatch.setattr(db_bootstrap, "check_external_content_fts_integrity", slow)
    key = (_db_file(tmp_path), "messages_fts")

    assert db_bootstrap._dispatch_background_integrity_scan(conn, spec) is True
    assert started.wait(5)
    first = db_bootstrap._integrity_scan_threads[key]

    # Second dispatch while the first scan is still running: no new thread.
    assert db_bootstrap._dispatch_background_integrity_scan(conn, spec) is True
    assert db_bootstrap._integrity_scan_threads[key] is first

    release.set()
    db_bootstrap.join_background_integrity_scans(timeout=5)
    conn.close()


def test_stale_scan_stamp_does_not_wedge_future_dispatch(tmp_path, monkeypatch):
    """A crashed scan (stale started-stamp, no live thread) must not block re-dispatch."""
    monkeypatch.setenv(INTERVAL_ENV, "24")
    monkeypatch.delenv("LCM_FTS_INTEGRITY_BACKGROUND", raising=False)
    conn = _make_conn(tmp_path)
    spec = _spec()
    ensure_external_content_fts(conn, spec)
    # Simulate a crashed scan: an old started-stamp with no in-process thread.
    stale = time.time() - db_bootstrap.INTEGRITY_SCAN_STALE_SECONDS - 60
    db_bootstrap._record_scan_started(conn, spec, now=stale)
    conn.commit()

    assert db_bootstrap._dispatch_background_integrity_scan(conn, spec) is True
    assert (_db_file(tmp_path), "messages_fts") in db_bootstrap._integrity_scan_threads
    db_bootstrap.join_background_integrity_scans(timeout=30)
    conn.close()


def test_non_finite_interval_falls_back_to_default(monkeypatch):
    """nan/inf must not parse as a valid interval (would suppress checks forever)."""
    for value in ("nan", "inf", "-inf", "Infinity"):
        monkeypatch.setenv(INTERVAL_ENV, value)
        assert (
            db_bootstrap._integrity_check_interval_hours()
            == db_bootstrap.DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
        )


def test_check_disk_space_uses_portable_fallback_when_statvfs_is_unavailable(monkeypatch, tmp_path):
    """Windows lacks os.statvfs, so startup FTS repair must not crash there."""
    monkeypatch.delattr(db_bootstrap.os, "statvfs", raising=False)
    monkeypatch.setattr(
        db_bootstrap,
        "shutil",
        types.SimpleNamespace(
            disk_usage=lambda path: types.SimpleNamespace(
                free=db_bootstrap._MIN_DISK_SPACE_BYTES
            )
        ),
        raising=False,
    )

    assert db_bootstrap._check_disk_space(str(tmp_path / "lcm.db")) is True


def test_run_versioned_migrations_refuses_newer_schema_before_migration_state_ddl(tmp_path):
    conn = sqlite3.connect(tmp_path / "future-no-ddl.db")
    try:
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
            (str(db_bootstrap.SCHEMA_VERSION + 1),),
        )
        conn.commit()

        with pytest.raises(db_bootstrap.SchemaVersionTooNewError):
            db_bootstrap.run_versioned_migrations(conn)

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert tables == {"metadata"}
    finally:
        conn.close()


def test_run_versioned_migrations_refuses_newer_schema(tmp_path):
    from hermes_lcm.db_bootstrap import (
        SchemaVersionTooNewError,
        ensure_metadata_table,
        run_versioned_migrations,
    )

    conn = sqlite3.connect(tmp_path / "future.db")
    try:
        ensure_metadata_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', '99')"
        )
        conn.commit()
        with pytest.raises(SchemaVersionTooNewError):
            run_versioned_migrations(conn)
    finally:
        conn.close()


def test_run_versioned_migrations_accepts_current_schema(tmp_path):
    from hermes_lcm.db_bootstrap import run_versioned_migrations, get_schema_version, SCHEMA_VERSION

    conn = sqlite3.connect(tmp_path / "fresh.db")
    try:
        run_versioned_migrations(conn)
        assert get_schema_version(conn) == SCHEMA_VERSION
    finally:
        conn.close()


def test_message_store_refuses_newer_schema_before_startup_ddl(tmp_path):
    from hermes_lcm.store import MessageStore

    db_path = tmp_path / "newer-message.db"
    _make_future_schema_db(db_path)
    assert _journal_mode(db_path) == "delete"

    with pytest.raises(db_bootstrap.SchemaVersionTooNewError):
        MessageStore(db_path)

    assert _journal_mode(db_path) == "delete"
    assert _table_names(db_path) == {"metadata"}


def test_summary_dag_refuses_newer_schema_before_startup_ddl(tmp_path):
    from hermes_lcm.dag import SummaryDAG

    db_path = tmp_path / "newer-dag.db"
    _make_future_schema_db(db_path)
    assert _journal_mode(db_path) == "delete"

    with pytest.raises(db_bootstrap.SchemaVersionTooNewError):
        SummaryDAG(db_path)

    assert _journal_mode(db_path) == "delete"
    assert _table_names(db_path) == {"metadata"}


def test_lifecycle_state_store_refuses_newer_schema_before_writable_pragmas_or_ddl(tmp_path):
    from hermes_lcm.lifecycle_state import LifecycleStateStore

    db_path = tmp_path / "newer-lifecycle.db"
    _make_future_schema_db(db_path)
    assert _journal_mode(db_path) == "delete"

    with pytest.raises(db_bootstrap.SchemaVersionTooNewError):
        LifecycleStateStore(db_path)

    assert _journal_mode(db_path) == "delete"
    assert _table_names(db_path) == {"metadata"}

def test_message_store_refuses_newer_schema_before_configuring_connection(tmp_path, monkeypatch):
    from hermes_lcm.store import MessageStore
    import hermes_lcm.store as store_module

    db_path = tmp_path / "newer-before-pragmas.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(db_bootstrap.SCHEMA_VERSION + 1),),
    )
    conn.commit()
    conn.close()

    called = False

    def fail_if_called(conn):
        nonlocal called
        called = True
        raise AssertionError("configure_connection should not run for future schemas")

    monkeypatch.setattr(store_module, "configure_connection", fail_if_called)

    with pytest.raises(db_bootstrap.SchemaVersionTooNewError):
        MessageStore(db_path)
    assert called is False
