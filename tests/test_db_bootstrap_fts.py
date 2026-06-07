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
import time
import types

import pytest

from hermes_lcm import db_bootstrap
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


def test_existing_index_without_marker_runs_check_and_records_marker(tmp_path, integrity_calls):
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
