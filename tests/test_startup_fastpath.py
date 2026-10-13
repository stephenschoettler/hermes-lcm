"""Startup regressions: work stays outside registration and runs only once."""
import importlib.util
import logging
from pathlib import Path
import sqlite3
import sys
import types

from hermes_lcm.store import MessageStore

STEP = 'message_ingested_at_backfill_v1'


def _columns(conn):
    conn.executescript('''
        CREATE TABLE messages(store_id INTEGER PRIMARY KEY, timestamp REAL,
            ingested_at REAL, observed_at REAL, observed_at_source TEXT);
        CREATE TABLE lcm_migration_state(step_name TEXT PRIMARY KEY,
            completed_at REAL NOT NULL);
        INSERT INTO messages VALUES(1, 123.5, NULL, NULL, NULL);
        INSERT INTO messages VALUES(2, 456.5, 999.0, 33.0, 'host');
    ''')


def _helper(conn):
    store = object.__new__(MessageStore)
    store._conn = conn
    return store


def test_timestamp_backfill_is_transactional_and_once():
    conn = sqlite3.connect(':memory:')
    _columns(conn)
    sql = []
    conn.set_trace_callback(sql.append)
    store = _helper(conn)
    store._ensure_time_contract_columns()
    assert conn.execute('SELECT ingested_at FROM messages ORDER BY store_id').fetchall() == [(123.5,), (999.0,)]
    assert conn.execute('SELECT observed_at FROM messages ORDER BY store_id').fetchall() == [(None,), (33.0,)]
    assert conn.execute('SELECT 1 FROM lcm_migration_state WHERE step_name=?', (STEP,)).fetchone()
    conn.commit()
    sql.clear()
    store._ensure_time_contract_columns()
    assert not any(s.lstrip().upper().startswith('UPDATE MESSAGES') for s in sql)
    conn.close()


def test_timestamp_marker_rolls_back_with_failed_bootstrap():
    conn = sqlite3.connect(':memory:')
    _columns(conn)
    store = _helper(conn)
    store._ensure_time_contract_columns()
    conn.rollback()
    assert conn.execute('SELECT ingested_at FROM messages WHERE store_id=1').fetchone() == (None,)
    assert conn.execute('SELECT 1 FROM lcm_migration_state WHERE step_name=?', (STEP,)).fetchone() is None
    store._ensure_time_contract_columns()
    assert conn.execute('SELECT ingested_at FROM messages WHERE store_id=1').fetchone() == (123.5,)
    conn.close()


def test_new_empty_store_marks_backfill_without_full_update(tmp_path):
    db = tmp_path / 'fresh.db'
    store = MessageStore(db)
    assert store._conn.execute('SELECT 1 FROM lcm_migration_state WHERE step_name=?', (STEP,)).fetchone()
    store.close()
    sql = []
    conn = sqlite3.connect(db)
    conn.set_trace_callback(sql.append)
    store = _helper(conn)
    store._ensure_time_contract_columns()
    assert not any(s.lstrip().upper().startswith('UPDATE MESSAGES') for s in sql)
    conn.close()


def _plugin(monkeypatch):
    name = 'lcm_startup_registration_test'
    directory = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(name, directory / '__init__.py', submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class Context:
    def __init__(self):
        self.engine = None
        self.hooks = {}

    def register_context_engine(self, engine):
        self.engine = engine

    def register_hook(self, name, callback):
        self.hooks[name] = callback


def test_deferred_status_reports_pending_without_opening_database(tmp_path):
    from hermes_lcm.engine import LCMEngine
    engine = LCMEngine(hermes_home=str(tmp_path), defer_storage=True)
    status = engine.get_status()
    assert status['storage_ready'] is False
    assert status['source_lineage'] == {'status': 'pending_initialization'}
    assert not (tmp_path / 'lcm.db').exists()
    engine.shutdown()


def test_deferred_prototype_shutdown_does_not_initialize_storage(tmp_path):
    from hermes_lcm.engine import LCMEngine
    engine = LCMEngine(hermes_home=str(tmp_path), defer_storage=True)
    engine.shutdown()
    assert not (tmp_path / 'lcm.db').exists()


def test_register_does_not_open_database_and_first_session_initializes(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    module = _plugin(monkeypatch)
    original = sqlite3.connect
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('register must not open SQLite')

    monkeypatch.setattr(sqlite3, 'connect', forbidden)
    ctx = Context()
    module.register(ctx)
    assert ctx.engine is not None
    assert calls == []
    assert not (tmp_path / 'lcm.db').exists()
    monkeypatch.setattr(sqlite3, 'connect', original)
    ctx.engine.on_session_start('startup-proof', hermes_home=str(tmp_path), platform='test')
    ctx.engine.ingest([{'role': 'user', 'content': 'recorded after lazy startup'}])
    assert ctx.engine._store.get_session_count('startup-proof') == 1
    ctx.engine._close_storage()


def test_post_turn_hook_uses_guarded_context_registration(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    private_hooks = {}
    fake_plugins = types.ModuleType('hermes_cli.plugins')
    fake_plugins.get_plugin_manager = lambda: types.SimpleNamespace(_hooks=private_hooks)
    monkeypatch.setitem(sys.modules, 'hermes_cli.plugins', fake_plugins)
    module = _plugin(monkeypatch)
    ctx = Context()
    module.register(ctx)
    assert 'post_llm_call' in ctx.hooks
    assert private_hooks == {}
    ctx.engine._close_storage()


def test_abandoned_registration_never_logs_active_or_adds_hooks(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    module = _plugin(monkeypatch)
    ctx = Context()
    ctx._load_abandoned = True
    with caplog.at_level(logging.INFO):
        module.register(ctx)
    assert ctx.engine is None
    assert not ctx.hooks
    assert not any('lossless context management active' in r.message for r in caplog.records)
