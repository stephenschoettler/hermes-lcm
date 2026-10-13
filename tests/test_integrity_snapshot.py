"""Real SQLite deep checks must never hold the active database writer lock."""
import os
from pathlib import Path
import sqlite3
import threading
import time

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.db_bootstrap import ExternalContentFtsSpec


def test_snapshot_low_disk_defers_without_flag_or_copy(tmp_path, monkeypatch):
    path = tmp_path / 'disk.db'
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE messages(store_id INTEGER PRIMARY KEY, content TEXT)')
    conn.commit()
    conn.close()
    from collections import namedtuple
    Usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(db_bootstrap.shutil, 'disk_usage', lambda directory: Usage(100, 100, 0))
    monkeypatch.setattr(db_bootstrap, 'check_external_content_fts_integrity',
                        lambda *args: (_ for _ in ()).throw(AssertionError('scan after disk failure')))
    spec = ExternalContentFtsSpec(table_name='messages_fts',content_table='messages',content_rowid='store_id',indexed_column='content',trigger_sqls=())
    result = db_bootstrap._check_committed_integrity_snapshot(str(path), spec)
    assert result['status'] == 'unchecked'
    assert 'space' in result['detail']


def test_failed_backup_cleans_private_files_before_deferring(tmp_path, monkeypatch):
    path = tmp_path / 'source.db'
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE messages(store_id INTEGER PRIMARY KEY, content TEXT)')
    conn.commit()
    conn.close()
    real_connect = sqlite3.connect
    real_tempdir = db_bootstrap.tempfile.TemporaryDirectory
    directories = []

    class BrokenBackup(sqlite3.Connection):
        def backup(self, *args, **kwargs):
            raise sqlite3.OperationalError('controlled snapshot copy failure')

    def connect(*args, **kwargs):
        kwargs.setdefault('factory', BrokenBackup)
        return real_connect(*args, **kwargs)

    def tempdir(*args, **kwargs):
        kwargs['dir'] = tmp_path
        result = real_tempdir(*args, **kwargs)
        directories.append(Path(result.name))
        return result

    monkeypatch.setattr(db_bootstrap.sqlite3, 'connect', connect)
    monkeypatch.setattr(db_bootstrap.tempfile, 'TemporaryDirectory', tempdir)
    spec = ExternalContentFtsSpec(table_name='messages_fts',content_table='messages',content_rowid='store_id',indexed_column='content',trigger_sqls=())
    result = db_bootstrap._check_committed_integrity_snapshot(str(path), spec)
    assert result['status'] == 'unchecked'
    assert 'copy failure' in result['detail']
    assert directories and all(not d.exists() for d in directories)


@pytest.mark.parametrize('snapshot_status', ['pass', 'fail'])
def test_live_repair_invalidates_older_snapshot_result(tmp_path, monkeypatch, snapshot_status):
    path = tmp_path / 'repair.db'
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript('''
        CREATE TABLE messages(store_id INTEGER PRIMARY KEY, content TEXT);
        CREATE VIRTUAL TABLE messages_fts USING fts5(content,
            content='messages', content_rowid='store_id');
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO messages VALUES(1, 'original text');
        INSERT INTO messages_fts(messages_fts) VALUES('rebuild');
    ''')
    if snapshot_status == 'fail':
        # Same row count, actual checksum drift.
        conn.execute("UPDATE messages SET content='repaired text' WHERE store_id=1")
    conn.commit()
    spec = ExternalContentFtsSpec(table_name='messages_fts',content_table='messages',content_rowid='store_id',indexed_column='content',trigger_sqls=())
    entered, release = threading.Event(), threading.Event()
    original = db_bootstrap._check_committed_integrity_snapshot

    def pause_after_check(db_path, fts):
        result = original(db_path, fts)
        assert result['status'] == snapshot_status
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(db_bootstrap, '_check_committed_integrity_snapshot', pause_after_check)
    worker = threading.Thread(target=db_bootstrap._run_background_integrity_scan,args=(str(path),spec,time.time()))
    worker.start()
    try:
        assert entered.wait(5)
        repair_time = time.time() + 1
        result = db_bootstrap.repair_external_content_fts(conn, spec, now=repair_time, throttle=False)
        if snapshot_status == 'fail':
            assert result['rebuilt'] is True
        assert db_bootstrap.check_external_content_fts_integrity(conn, spec)['status'] == 'pass'
        assert db_bootstrap.load_integrity_failed(conn, spec) is None
        release.set()
        worker.join(5)
        assert not worker.is_alive()
        assert db_bootstrap.load_integrity_failed(conn, spec) is None
        checked = conn.execute('SELECT value FROM metadata WHERE key=?', (db_bootstrap._integrity_marker_key(spec),)).fetchone()
        assert float(checked[0]) == repair_time
    finally:
        release.set()
        worker.join(5)
        conn.close()


def test_real_background_integrity_statement_does_not_block_live_writer(tmp_path, monkeypatch):
    path = tmp_path / 'active.db'
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript('''
        CREATE TABLE messages(store_id INTEGER PRIMARY KEY, content TEXT);
        CREATE VIRTUAL TABLE messages_fts USING fts5(content,
            content='messages', content_rowid='store_id');
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
    ''')
    conn.executemany('INSERT INTO messages VALUES(?,?)', [(i, 'real indexed document '+str(i)) for i in range(1,201)])
    conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()
    spec = ExternalContentFtsSpec(table_name='messages_fts',content_table='messages',content_rowid='store_id',indexed_column='content',trigger_sqls=())
    entered, release = threading.Event(), threading.Event()
    real_connect = sqlite3.connect
    checked_paths = []

    class Observed(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if "VALUES('integrity-check', 1)" not in sql:
                return super().execute(sql, *args, **kwargs)
            self.set_progress_handler(lambda: 0, 1)
            calls = 0

            def pause_inside_real_statement():
                nonlocal calls
                calls += 1
                # Let the real VM enter its write transaction first.
                if calls == 100:
                    checked_paths.append(self.execute('PRAGMA database_list').fetchone()[2])
                    entered.set()
                    if not release.wait(5):
                        return 1
                return 0

            self.set_progress_handler(pause_inside_real_statement, 1)
            try:
                return super().execute(sql, *args, **kwargs)
            finally:
                self.set_progress_handler(None, 0)

    def connect(*args, **kwargs):
        kwargs.setdefault('factory', Observed)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(db_bootstrap.sqlite3, 'connect', connect)
    worker = threading.Thread(target=db_bootstrap._run_background_integrity_scan,args=(str(path),spec,time.time()))
    worker.start()
    try:
        assert entered.wait(5), 'real FTS statement did not reach barrier'
        writer = real_connect(path, timeout=.25)
        try:
            before = time.monotonic()
            writer.execute('BEGIN IMMEDIATE')
            writer.execute("INSERT INTO messages VALUES(201, 'concurrent ingest')")
            writer.commit()
            assert time.monotonic()-before < 1
        finally:
            writer.close()
        assert checked_paths and Path(checked_paths[0]) != path
        assert os.stat(checked_paths[0]).st_mode & 0o777 == 0o600
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    assert not Path(checked_paths[0]).exists(), 'private snapshot not cleaned'
