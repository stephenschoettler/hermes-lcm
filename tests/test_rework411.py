"""Durable-ingest and public registration ownership regressions from review409."""
import ast
import functools
import logging
import os
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from . import test_contention_contract as contracts
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore
from hermes_lcm.sqlite_util import _InitializingConnection

entrypoint = contracts.entrypoint


@pytest.fixture
def ready(tmp_path, monkeypatch):
    import hermes_cli.config
    assert Path(hermes_cli.config.__file__).resolve().is_relative_to(Path(os.environ['LCM_TEST_CORE_SOURCE']))
    return contracts.ready.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize('count', [1, 3])
@pytest.mark.parametrize('finish', ['commit', 'rollback', 'commit_failure'])
def test_nested_engine_ingest_rejects_without_raw_ack(ready, monkeypatch, count, finish):
    engine = LCMEngine(config=LCMConfig(database_path=str(ready)))
    engine.on_session_start('nested-fixture', platform='cli')
    conn = engine._store._conn
    history = [{'role':'user', 'content':f'synthetic {i}'} for i in range(count)]
    original_commit = _InitializingConnection.commit
    successes = []
    monkeypatch.setattr(engine, '_record_ingest_success', lambda: successes.append(True))
    try:
        conn.execute("INSERT INTO metadata VALUES('outer-rework','kept')")
        with pytest.raises(RuntimeError, match='caller-owned transaction'):
            engine._ingest_messages(history)
        assert conn.in_transaction and engine._ingest_cursor == 0
        assert conn.execute("SELECT value FROM metadata WHERE key='outer-rework'").fetchone()[0] == 'kept'
        assert engine._store.get_session_count('nested-fixture') == 0
        engine.ingest(history)
        assert engine._ingest_failure_count == 1 and engine._consecutive_ingest_failures == 1
        assert successes == []
        assert engine._ingest_cursor == 0
        if finish == 'commit_failure':
            def fail(candidate):
                if candidate is conn:
                    raise sqlite3.OperationalError('fixture outer commit failure')
                return original_commit(candidate)
            monkeypatch.setattr(_InitializingConnection, 'commit', fail)
            with pytest.raises(sqlite3.OperationalError, match='outer commit failure'):
                conn.commit()
            assert conn.in_transaction and engine._ingest_cursor == 0
            conn.rollback()
            monkeypatch.setattr(_InitializingConnection, 'commit', original_commit)
        elif finish == 'commit':
            conn.commit()
        else:
            conn.rollback()
        engine._ingest_messages(history)
        engine._ingest_messages(history)
        assert engine._ingest_cursor == count
        assert engine._store.get_session_count('nested-fixture') == count
    finally:
        conn.rollback()
        engine.shutdown()


@pytest.mark.parametrize('count', [1, 3])
@pytest.mark.parametrize('finish', ['rollback', 'commit', 'commit_failure'])
def test_store_nested_write_remains_provisional_until_caller_commit(ready, monkeypatch, count, finish):
    store = MessageStore(ready)
    conn = store._conn
    original_commit = _InitializingConnection.commit
    messages = [{'role':'user','content':f'fixture {i}'} for i in range(count)]
    def append():
        return ([store.append('provisional', messages[0])] if count == 1
                else store.append_batch('provisional', messages))
    try:
        conn.execute('BEGIN')
        assert len(append()) == count and conn.in_transaction
        with sqlite3.connect(ready) as peer:
            assert peer.execute("SELECT count(*) FROM messages WHERE session_id='provisional'").fetchone()[0] == 0
        if finish == 'commit_failure':
            def fail(candidate):
                if candidate is conn:
                    raise sqlite3.OperationalError('fixture caller commit failure')
                return original_commit(candidate)
            monkeypatch.setattr(_InitializingConnection, 'commit', fail)
            with pytest.raises(sqlite3.OperationalError, match='caller commit failure'):
                conn.commit()
            assert conn.in_transaction and store.get_session_count('provisional') == count
            conn.rollback()
            monkeypatch.setattr(_InitializingConnection, 'commit', original_commit)
        elif finish == 'rollback':
            conn.rollback()
        else:
            conn.commit()
        if finish != 'commit':
            assert store.get_session_count('provisional') == 0
            append()
        assert store.get_session_count('provisional') == count
        assert not conn.in_transaction
    finally:
        conn.rollback()
        store.close()


def test_registration_finishing_after_cleanup_disposes_its_handle(ready, monkeypatch):
    # Keep the review's real registrar/decorator boundary; imports never execute
    # the installed host or read its .env. The source copy is fixture-only.
    source = (Path(os.environ['LCM_TEST_CORE_SOURCE']) / 'hermes_cli/plugins.py').read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PluginContext')
    registrar = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'register_context_engine')
    guard = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_ignore_after_abandoned_load')
    entered, release = threading.Event(), threading.Event()
    def serialized(fn):
        @functools.wraps(fn)
        def wrapped(*a, **kw):
            entered.set()
            assert release.wait(5)
            return fn(*a, **kw)
        return wrapped
    namespace = dict(_serialized_replacement=serialized, wraps=functools.wraps, logger=logging.getLogger('rework411'))
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), registrar, guard], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'disposable-host-registrar', 'exec'), namespace)
    class Context:
        _load_abandoned = False
        _load_deadline = None
        manifest = SimpleNamespace(name='fixture')
        def __init__(self):
            self._manager = SimpleNamespace(_context_engine=None)
            self.handles = []
        def _wrong_type(self, *args):
            return False
        def _track_replacement(self, *args, **kwargs):
            handle = SimpleNamespace(dispose=lambda: setattr(self._manager, '_context_engine', None))
            self.handles.append(handle)
            return handle
        register_context_engine = namespace['_ignore_after_abandoned_load'](namespace['register_context_engine'])
    ctx = Context()
    monkeypatch.setattr(LCMConfig, 'from_env', lambda: LCMConfig(database_path=str(ready)))
    failures = []
    def load():
        try:
            entrypoint().register(ctx)
        except BaseException as exc:
            failures.append(exc)
    worker = threading.Thread(target=load)
    worker.start()
    try:
        assert entered.wait(5)
        ctx._load_abandoned = True
        for handle in list(ctx.handles):
            handle.dispose()
        assert ctx._manager._context_engine is None
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert failures and isinstance(failures[0], TimeoutError)
    assert ctx._manager._context_engine is None


@pytest.mark.parametrize('cancel_stage', ['engine', 'skill', 'hook', 'tool'])
def test_partial_public_registration_handles_are_disposed(ready, monkeypatch, cancel_stage):
    module = entrypoint()
    monkeypatch.setattr(module, '_host_forwards_registered_tool_messages', lambda ctx: True)
    monkeypatch.setattr(LCMConfig, 'from_env', lambda: LCMConfig(database_path=str(ready)))
    class Context:
        _load_abandoned = False
        _load_deadline = None
        def __init__(self):
            self.handles = []
            self.live = []
        def publish(self, stage):
            self.live.append(stage)
            handle = SimpleNamespace(dispose=lambda: self.live.remove(stage) if stage in self.live else None)
            self.handles.append(handle)
            if stage == cancel_stage:
                self._load_abandoned = True
            return handle
        def register_context_engine(self, engine):
            self.engine = engine
            return self.publish('engine')
        def register_skill(self, *args, **kwargs):
            return self.publish('skill')
        def register_hook(self, *args, **kwargs):
            return self.publish('hook')
        def register_tool(self, *args, **kwargs):
            return self.publish('tool')
    ctx = Context()
    with pytest.raises(TimeoutError):
        module.register(ctx)
    assert ctx.handles and ctx.live == []
    assert ctx.engine._store._conn is None


def test_runtime_executescript_retains_sqlite_contract(ready):
    store = MessageStore(ready)
    try:
        conn = store._conn
        conn.execute("INSERT INTO metadata VALUES('runtime-script-rework','committed')")
        conn.executescript('SELECT 1;')
        assert not conn.in_transaction
        with sqlite3.connect(ready) as peer:
            assert peer.execute("SELECT value FROM metadata WHERE key='runtime-script-rework'").fetchone()[0] == 'committed'
    finally:
        store.close()


@pytest.mark.parametrize('action', [
    "open('/nonfixture/.env').read()",
    "__import__('sqlite3').connect('/nonfixture/live.db')",
    "__import__('socket').socket().connect(('203.0.113.1',443))",
    "__import__('os').execv('/bin/cat',['cat','/nonfixture/.env'])",
])
def test_subprocess_guard_blocks_network_database_and_sensitive_reads(action):
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c', action], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'forbidden' in result.stderr


def test_disposable_host_source_is_used():
    import hermes_cli.config
    assert Path(hermes_cli.config.__file__).resolve().is_relative_to(Path(os.environ['LCM_TEST_FIXTURE_ROOT']))


@pytest.mark.parametrize('off_current', [False, True])
def test_nested_session_end_never_acknowledges_raw_or_finalizes(ready, monkeypatch, caplog, off_current):
    engine = LCMEngine(config=LCMConfig(database_path=str(ready), ignore_session_patterns=['cron']))
    first = {'role':'user','content':'normal prefix'}
    history = [first, {'role':'assistant','content':'retryable final'}]
    if off_current:
        engine.on_session_start('ended', platform='cron')
        engine.ingest([{'role':'user','content':'ignored'}])
    engine.on_session_start('ended', platform='cli')
    engine.ingest([first])
    if off_current:
        engine.on_session_start('foreground', platform='cli')
    conn = engine._store._conn
    finalized = []
    original = engine._lifecycle.finalize_session
    def finalize(*args, **kwargs):
        finalized.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(engine._lifecycle, 'finalize_session', finalize)
    try:
        conn.execute("INSERT INTO metadata VALUES('outer-session-end','kept')")
        engine.on_session_end('ended', history)
        assert conn.in_transaction and finalized == []
        assert engine._store.get_session_count('ended') == 1
        assert 'raw-message ingest rejected' in caplog.text
        assert 'raw messages were ingested' not in caplog.text
        conn.rollback()
        engine.on_session_end('ended', history)
        engine.on_session_end('ended', history)
        assert finalized and engine._store.get_session_count('ended') == 2
    finally:
        conn.rollback()
        engine.shutdown()


@pytest.mark.parametrize('failure', [ValueError, KeyboardInterrupt])
def test_partial_registration_exception_disposes_returned_handles_preserving_cause(ready, monkeypatch, failure):
    module = entrypoint()
    monkeypatch.setattr(module, '_host_forwards_registered_tool_messages', lambda ctx: True)
    monkeypatch.setattr(LCMConfig, 'from_env', lambda: LCMConfig(database_path=str(ready)))
    monkeypatch.setenv('LCM_ENABLE_SLASH_COMMAND', '1')
    class Context:
        _load_abandoned = False
        _load_deadline = None
        def __init__(self):
            self.live = ['preexisting']
            self.disposed = []
        def publish(self, stage):
            self.live.append(stage)
            def dispose():
                self.live.remove(stage)
                self.disposed.append(stage)
                if stage == 'skill':
                    raise RuntimeError('secondary disposal failure')
            return SimpleNamespace(dispose=dispose)
        def register_context_engine(self, engine):
            self.engine = engine
            return self.publish('engine')
        def register_skill(self, *args, **kwargs):
            return self.publish('skill')
        def register_hook(self, *args, **kwargs):
            return self.publish('hook')
        def register_tool(self, *args, **kwargs):
            # Exercise registrars returning multiple public handles as well.
            return [self.publish('tool')]
        def register_command(self, *args, **kwargs):
            raise failure('fixture registration failure')
    ctx = Context()
    with pytest.raises(failure, match='fixture registration failure'):
        module.register(ctx)
    assert ctx.live == ['preexisting']
    assert set(ctx.disposed) == {'engine', 'skill', 'hook', 'tool'}
    assert ctx.engine._store._conn is None
