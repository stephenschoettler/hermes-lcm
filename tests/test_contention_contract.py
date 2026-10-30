"""Disposable contention regressions; no live databases or provider calls."""
import importlib.util
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_lcm import db_bootstrap as boot, tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.store import MessageStore, build_message_fts_spec
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.assertion_store import AssertionStore
from hermes_lcm.query_view_store import QueryViewStore, _ensure_query_view_schema
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.vector_store import VectorStore
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, '_encoder_ready', True)
    monkeypatch.setattr(tokens, '_encoder', None)
    monkeypatch.setattr(boot, '_should_run_integrity_check', lambda *a, **k: False)
    monkeypatch.setattr(boot, 'SQLITE_BUSY_TIMEOUT_MS', 0)
    monkeypatch.setattr(MessageStore, '_INIT_LOCK_RETRY_BUDGET_S', 0)
    db = tmp_path / 'fixture.db'
    for cls in (MessageStore, SummaryDAG, LifecycleStateStore, AssertionStore,
                QueryViewStore, RollupStore, VectorStore):
        obj = cls(db)
        obj.close()
    return db


@pytest.mark.parametrize('cls', [MessageStore, SummaryDAG, LifecycleStateStore,
                                 AssertionStore, QueryViewStore, RollupStore, VectorStore])
def test_ready_open_needs_no_writer(ready, cls):
    with sqlite3.connect(ready, timeout=0) as holder:
        holder.execute('BEGIN IMMEDIATE')
        obj = cls(ready)
        try:
            assert not obj._conn.in_transaction
        finally:
            obj.close()
        holder.rollback()


def test_ready_query_schema_does_not_scan_corpus(ready):
    store = MessageStore(ready)
    store.append_batch('synthetic', [{'role': 'user', 'content': 'fixture'}] * 500)
    q = QueryViewStore(ready)
    trace = []
    q._conn.set_trace_callback(trace.append)
    try:
        _ensure_query_view_schema(q._conn)
        assert not any('FROM MESSAGES' in sql.upper() and 'COUNT(' in sql.upper() for sql in trace)
        assert not q._conn.in_transaction
    finally:
        q.close()
        store.close()


@pytest.mark.parametrize('operation', ['append', 'metadata'])
def test_failed_owned_write_releases_transaction(ready, operation):
    store = MessageStore(ready)
    if operation == 'append':
        def fail():
            return store.append('s', {'role': None, 'content': 'bad'})
    else:
        store._conn.execute("CREATE TRIGGER reject_metadata BEFORE INSERT ON metadata WHEN NEW.key='bad' BEGIN SELECT RAISE(ABORT,'fixture'); END")
        def fail():
            return store.write_metadata_json(['good', 'bad'], '{}')
    try:
        with pytest.raises(sqlite3.IntegrityError):
            fail()
        assert not store._conn.in_transaction
        with sqlite3.connect(ready, timeout=0) as peer:
            peer.execute('BEGIN IMMEDIATE')
            assert peer.execute("SELECT value FROM metadata WHERE key='good'").fetchone() is None
            peer.rollback()
    finally:
        store.close()


def test_dag_constructor_interrupt_closes_connection(ready, monkeypatch):
    opened = []
    original = sqlite3.connect
    def connect(*a, **kw):
        conn = original(*a, **kw)
        opened.append(conn)
        return conn
    monkeypatch.setattr(sqlite3, 'connect', connect)
    def interrupt(self):
        self._conn.execute("INSERT INTO metadata VALUES('fixture-interrupt','1')")
        raise KeyboardInterrupt('fixture')
    monkeypatch.setattr(SummaryDAG, '_ensure_source_window_columns', interrupt)
    with pytest.raises(KeyboardInterrupt, match='fixture') as retained:
        SummaryDAG(ready)
    assert retained.value.__traceback__ is not None
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        opened[-1].execute('SELECT 1')

from hermes_lcm.sqlite_util import InitializationBudget, _InitializingConnection
import subprocess
import sys
import threading
from types import SimpleNamespace


def entrypoint():
    path = Path(__file__).resolve().parents[1] / '__init__.py'
    spec = importlib.util.spec_from_file_location('hermes_lcm', path, submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixtureContext:
    _load_abandoned = False
    _load_deadline = None
    def __init__(self):
        self.engines = []
        self.hooks = {}
    def register_context_engine(self, engine):
        self.engines.append(engine)
    def register_hook(self, name, callback):
        self.hooks.setdefault(name, []).append(callback)


def test_full_registration_and_clone_under_writer(ready, monkeypatch):
    config = LCMConfig(database_path=str(ready), assertions_enabled=True,
                       query_views_enabled=True, temporal_rollups_enabled=True,
                       adaptive_retrieval_enabled=True)
    # Engine rollup trigger installation must already be ready as well.
    prototype = LCMEngine(config=config)
    prototype.shutdown()
    monkeypatch.setattr(LCMConfig, 'from_env', lambda: config)
    ctx = FixtureContext()
    with sqlite3.connect(ready, timeout=0) as writer:
        writer.execute('BEGIN IMMEDIATE')
        entrypoint().register(ctx)
        clone = ctx.engines[0].clone_for_agent()
        assert clone._assertions is not None and clone._query_views is not None
        assert 'post_llm_call' in ctx.hooks
        clone.shutdown()
        ctx.engines[0].shutdown()
        writer.rollback()


def test_initialization_cancellation_has_no_late_hooks(ready, monkeypatch):
    ctx = FixtureContext()
    opened = []
    original = MessageStore._init_db_once
    def cancel(self):
        original(self)
        opened.append(self._conn)
        ctx._load_abandoned = True
    monkeypatch.setattr(MessageStore, '_init_db_once', cancel)
    monkeypatch.setattr(LCMConfig, 'from_env', lambda: LCMConfig(database_path=str(ready)))
    with pytest.raises(TimeoutError):
        entrypoint().register(ctx)
    assert ctx.engines == [] and ctx.hooks == {}
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute('SELECT 1')


def test_shared_remaining_budget_across_helpers(ready, monkeypatch):
    import hermes_lcm.sqlite_util as util
    observed = []
    original = util.connect_initializing
    def delayed(owner, *args, **kw):
        owner._init_budget.deadline -= 2
        conn = original(owner, *args, **kw)
        observed.append((owner._init_budget, conn))
        return conn
    for name in ('store', 'dag', 'lifecycle_state', 'assertion_store', 'query_view_store'):
        monkeypatch.setattr(sys.modules['hermes_lcm.' + name], 'connect_initializing', delayed)
    start = time.monotonic()
    budget = InitializationBudget(5)
    with pytest.raises(TimeoutError):
        LCMEngine(config=LCMConfig(database_path=str(ready), assertions_enabled=True,
                                   query_views_enabled=True), init_budget=budget)
    assert time.monotonic() - start < .4
    assert len(observed) >= 2 and all(b is budget for b, _ in observed)
    for _, conn in observed:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute('SELECT 1')


@pytest.mark.parametrize('failure', ['commit', 'interrupt'])
@pytest.mark.parametrize('operation', ['append', 'metadata', 'dag'])
def test_owned_write_commit_failure_and_interrupt(ready, monkeypatch, failure, operation):
    obj = SummaryDAG(ready) if operation == 'dag' else MessageStore(ready)
    original = _InitializingConnection.commit
    def fail_commit(conn):
        if conn is obj._conn:
            if failure == 'interrupt':
                raise KeyboardInterrupt('fixture commit')
            raise sqlite3.OperationalError('fixture commit')
        return original(conn)
    if operation == 'dag':
        from hermes_lcm.dag import SummaryNode
        node = SummaryNode(session_id='fixture', depth=0, summary='synthetic')
        def action():
            return obj.add_node(node)
    elif operation == 'append':
        def action():
            return obj.append('fixture', {'role': 'user', 'content': 'synthetic'})
    else:
        def action():
            return obj.write_metadata_json(['first', 'second'], '{}')
    try:
        monkeypatch.setattr(_InitializingConnection, 'commit', fail_commit)
        with pytest.raises(KeyboardInterrupt if failure == 'interrupt' else sqlite3.OperationalError):
            action()
        assert not obj._conn.in_transaction
        with sqlite3.connect(ready, timeout=0) as peer:
            peer.execute('BEGIN IMMEDIATE')
            assert peer.execute("SELECT count(*) FROM metadata WHERE key IN ('first','second')").fetchone()[0] == 0
            assert peer.execute("SELECT count(*) FROM messages WHERE session_id='fixture'").fetchone()[0] == 0
            assert peer.execute("SELECT count(*) FROM summary_nodes WHERE session_id='fixture'").fetchone()[0] == 0
            peer.rollback()
        monkeypatch.setattr(_InitializingConnection, 'commit', original)
        action()
        table = 'summary_nodes' if operation == 'dag' else 'metadata' if operation == 'metadata' else 'messages'
        assert obj._conn.execute(f'SELECT count(*) FROM {table}' + (" WHERE key IN ('first','second')" if operation == 'metadata' else " WHERE session_id='fixture'")).fetchone()[0] == (2 if operation == 'metadata' else 1)
    finally:
        obj.close()


def test_failed_inner_metadata_preserves_outer_transaction(ready):
    obj = MessageStore(ready)
    try:
        obj._conn.execute("CREATE TRIGGER reject_key BEFORE INSERT ON metadata WHEN NEW.key='bad' BEGIN SELECT RAISE(ABORT,'fixture'); END")
        obj._conn.execute("INSERT INTO metadata VALUES('outer','kept')")
        with pytest.raises(sqlite3.IntegrityError):
            obj.write_metadata_json(['inner', 'bad'], '{}')
        assert obj._conn.in_transaction
        assert obj._conn.execute("SELECT value FROM metadata WHERE key='outer'").fetchone()[0] == 'kept'
        assert obj._conn.execute("SELECT 1 FROM metadata WHERE key='inner'").fetchone() is None
        obj.append('s', {'role':'user','content':'outer message'})
        assert obj._conn.in_transaction
        obj._conn.rollback()
        assert obj.get_session_count('s') == 0
    finally:
        obj.close()


class ProcessWriter:
    def __init__(self, path):
        code = "import sqlite3,sys; c=sqlite3.connect(sys.argv[1],timeout=0); c.execute('BEGIN IMMEDIATE'); print('ready',flush=True); sys.stdin.readline(); c.rollback(); c.close()"
        self.process = subprocess.Popen([sys.executable, '-c', code, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert self.process.stdout.readline().strip() == 'ready'
    def release(self):
        self.process.communicate('\n', timeout=5)
        assert self.process.returncode == 0


def test_process_writer_ready_and_repair_retry(ready, monkeypatch):
    monkeypatch.setattr(MessageStore, '_INIT_LOCK_RETRY_BUDGET_S', 5)
    # Prove the absence of writer work, rather than treating 200ms of thread
    # scheduling as a database-lock contract. Use the product's normal budget;
    # forbidden writer statements fail before SQLite or its retry loop can run.
    original_connect = sqlite3.connect
    statements = []
    class ReadyReadConnection(_InitializingConnection):
        def execute(self, sql, parameters=(), /):
            statements.append(sql)
            assert not sql.lstrip().upper().startswith(
                ('BEGIN', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER')
            ), f'ready constructor attempted writer work: {sql}'
            return super().execute(sql, parameters)
    def connect(database, *args, **kwargs):
        if str(database) == str(ready):
            kwargs['factory'] = ReadyReadConnection
        return original_connect(database, *args, **kwargs)
    holder = ProcessWriter(ready)
    try:
        with monkeypatch.context() as probe:
            probe.setattr(sqlite3, 'connect', connect)
            for cls in (MessageStore, SummaryDAG, LifecycleStateStore, AssertionStore, QueryViewStore, RollupStore, VectorStore):
                obj = cls(ready, init_budget=InitializationBudget())
                assert not obj._conn.in_transaction
                obj.close()
        assert statements and holder.process.poll() is None
        with original_connect(ready, timeout=0) as peer:
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                peer.execute('BEGIN IMMEDIATE')
    finally:
        holder.release()
    with sqlite3.connect(ready) as conn:
        conn.execute('DROP TRIGGER msg_fts_insert')
    holder = ProcessWriter(ready)
    timer = threading.Timer(.05, holder.release)
    timer.start()
    try:
        obj = MessageStore(ready, init_budget=InitializationBudget(1))
        assert obj.append('retry', {'role':'user','content':'exactly once'}) > 0
        assert obj.get_session_count('retry') == 1
        obj.close()
    finally:
        timer.join(5)


def test_fts_contended_claim_defers_without_full_scan(ready, monkeypatch):
    monkeypatch.setattr(boot, '_should_run_integrity_check', lambda *a, **k: True)
    monkeypatch.setattr(boot, 'check_external_content_fts_integrity', lambda *a: pytest.fail('foreground scan'))
    holder = ProcessWriter(ready)
    try:
        start = time.monotonic()
        obj = MessageStore(ready, init_budget=InitializationBudget(.2))
        assert time.monotonic() - start < .2
        obj.close()
    finally:
        holder.release()


def test_fts_vm_budget_releases_writer_leaves_pending(ready, monkeypatch):
    # Actual SQLite VM work, not a sleeping mock of an integrity scan.
    monkeypatch.setattr(boot, 'AUTOMATIC_FTS_BUDGET_S', .015)
    acquired = threading.Event()
    def expensive(conn, spec):
        acquired.set()
        conn.execute('WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100000000) SELECT sum(x) FROM n').fetchone()
        return {'status':'pass','detail':'unexpected completion'}
    monkeypatch.setattr(boot, 'check_external_content_fts_integrity', expensive)
    spec = build_message_fts_spec()
    with sqlite3.connect(ready) as conn:
        conn.execute('DELETE FROM metadata WHERE key LIKE "fts_integrity_checked_at:%"')
    with sqlite3.connect(ready) as conn:
        assert boot._dispatch_background_integrity_scan(conn, spec)
    assert acquired.wait(2)
    boot.join_background_integrity_scans(2)
    with sqlite3.connect(ready, timeout=0) as conn:
        conn.execute('BEGIN IMMEDIATE')
        assert conn.execute("SELECT value FROM metadata WHERE key='fts_integrity_checked_at:messages_fts'").fetchone() is None
        assert boot.load_integrity_failed(conn, spec) is None
        conn.rollback()

@pytest.mark.parametrize('repair, cls, expected', [
    ("DELETE FROM metadata WHERE key='schema_version'", MessageStore, "SELECT value FROM metadata WHERE key='schema_version'"),
    ("DELETE FROM lcm_migration_state WHERE step_name='v5_message_conversation_id'", LifecycleStateStore, "SELECT step_name FROM lcm_migration_state WHERE step_name='v5_message_conversation_id'"),
    ('DROP INDEX idx_msg_session', MessageStore, "SELECT name FROM sqlite_master WHERE name='idx_msg_session'"),
    ('DROP TRIGGER msg_fts_update', MessageStore, "SELECT name FROM sqlite_master WHERE name='msg_fts_update'"),
    ('DROP TABLE lcm_lifecycle_state', LifecycleStateStore, "SELECT name FROM sqlite_master WHERE name='lcm_lifecycle_state'"),
    ('DROP TRIGGER lcm_assertion_source_insert_guard', AssertionStore, "SELECT name FROM sqlite_master WHERE name='lcm_assertion_source_insert_guard'"),
    ('DROP TRIGGER lcm_query_corpus_message_insert', QueryViewStore, "SELECT name FROM sqlite_master WHERE name='lcm_query_corpus_message_insert'"),
    ('DELETE FROM lcm_query_corpus_state', QueryViewStore, "SELECT singleton FROM lcm_query_corpus_state"),
    ("DELETE FROM lcm_migration_state WHERE step_name='temporal_rollups_v1'", RollupStore, "SELECT step_name FROM lcm_migration_state WHERE step_name='temporal_rollups_v1'"),
    ('DROP INDEX idx_lcm_query_views_lifecycle', QueryViewStore, "SELECT name FROM sqlite_master WHERE name='idx_lcm_query_views_lifecycle'"),
])
def test_genuine_schema_repairs(ready, repair, cls, expected):
    with sqlite3.connect(ready) as conn:
        conn.execute(repair)
    obj = cls(ready)
    try:
        assert obj._conn.execute(expected).fetchone() is not None
    finally:
        obj.close()


def test_assertion_legacy_guard_repaired_atomically(ready):
    with sqlite3.connect(ready) as conn:
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='lcm_assertion_source_insert_guard'").fetchone()[0]
        conn.execute('DROP TRIGGER lcm_assertion_source_insert_guard')
        conn.execute(sql.replace('COALESCE(m.observed_at, m.timestamp)', 'm.timestamp'))
    obj = AssertionStore(ready)
    try:
        assert not boot.verify_assertion_schema(obj._conn)
    finally:
        obj.close()


def test_query_trigger_repair_reconciles_untracked_rows(ready):
    with sqlite3.connect(ready) as conn:
        conn.execute('DROP TRIGGER lcm_query_corpus_message_insert')
    obj = MessageStore(ready)
    obj.append('s', {'role':'user','content':'untracked fixture'})
    q = QueryViewStore(ready)
    try:
        assert q._conn.execute('SELECT row_count FROM lcm_query_corpus_state').fetchone()[0] == 1
        obj.append('s', {'role':'user','content':'tracked fixture'})
        assert q._conn.execute('SELECT row_count FROM lcm_query_corpus_state').fetchone()[0] == 2
    finally:
        q.close()
        obj.close()


def test_repair_deadline_does_not_bless_missing_schema(ready, monkeypatch):
    with sqlite3.connect(ready) as conn:
        conn.execute('DROP TRIGGER msg_fts_insert')
    monkeypatch.setattr(MessageStore, '_INIT_LOCK_RETRY_BUDGET_S', 5)
    holder = ProcessWriter(ready)
    start = time.monotonic()
    try:
        with pytest.raises((TimeoutError, sqlite3.OperationalError)):
            MessageStore(ready, init_budget=InitializationBudget(.06))
        assert time.monotonic() - start < .4
        with sqlite3.connect(ready) as conn:
            assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='msg_fts_insert'").fetchone() is None
    finally:
        holder.release()
    obj = MessageStore(ready)
    obj.close()


def test_trajectory_ready_open_needs_no_writer(ready, tmp_path):
    from hermes_lcm.trajectory_store import TrajectoryStore, CorpusIdentity
    identity = CorpusIdentity(dataset_name='synthetic', dataset_revision='r1', harness_commit='fixture', tier='small', domain='fixture', ingest_config_digest='fixture')
    assets = tmp_path / 'assets'
    assets.mkdir()
    obj = TrajectoryStore(ready, identity, asset_root=assets)
    obj.close()
    holder = ProcessWriter(ready)
    try:
        obj = TrajectoryStore(ready, identity, asset_root=assets, init_budget=InitializationBudget(.2))
        obj.close()
    finally:
        holder.release()


@pytest.mark.parametrize('off_current', [False, True])
@pytest.mark.parametrize('stage', ['raw', 'lifecycle'])
@pytest.mark.parametrize('failure', ['lock', 'interrupt', 'other'])
def test_session_end_failures_are_distinct(ready, monkeypatch, caplog, off_current, stage, failure):
    engine = LCMEngine(config=LCMConfig(database_path=str(ready), ignore_session_patterns=['cron']))
    first = {'role':'user','content':'normal opener'}
    history = [first, {'role':'assistant','content':'final suffix'}]
    if off_current:
        engine.on_session_start('ended', platform='cron')
        engine.ingest([{'role':'user','content':'ignored'}])
    engine.on_session_start('ended', platform='cli')
    engine.ingest([first])
    if off_current:
        engine.on_session_start('foreground', platform='cli')
    error = sqlite3.OperationalError('database is locked') if failure == 'lock' else KeyboardInterrupt('fixture') if failure == 'interrupt' else ValueError('fixture')
    def fail(*a, **kw):
        raise error
    owner, name = ((engine, '_append_off_current_session_end_suffix' if off_current else '_ingest_messages') if stage == 'raw' else (engine._lifecycle, 'finalize_session'))
    original = getattr(owner, name)
    monkeypatch.setattr(owner, name, fail)
    try:
        if failure == 'other':
            with pytest.raises(ValueError, match='fixture'):
                engine.on_session_end('ended', history)
        else:
            engine.on_session_end('ended', history)
            assert ('raw-message ingest' if stage == 'raw' else 'lifecycle finalization') in caplog.text
            assert ('raw messages were ingested' in caplog.text) == (stage == 'lifecycle')
        assert engine._store.get_session_count('ended') == (1 if stage == 'raw' else 2)
        monkeypatch.setattr(owner, name, original)
        engine.on_session_end('ended', history)
        engine.on_session_end('ended', history)
        assert engine._store.get_session_count('ended') == 2
    finally:
        engine.shutdown()


@pytest.mark.parametrize('off_current', [False, True])
def test_session_end_process_writer_release_retry_once(ready, caplog, off_current):
    engine = LCMEngine(config=LCMConfig(database_path=str(ready), ignore_session_patterns=['cron']))
    first = {'role':'user','content':'normal prefix'}
    history = [first, {'role':'assistant','content':'exactly once suffix'}]
    if off_current:
        engine.on_session_start('ended', platform='cron')
        engine.ingest([{'role':'user','content':'ignored'}])
    engine.on_session_start('ended', platform='cli')
    engine.ingest([first])
    if off_current:
        engine.on_session_start('foreground', platform='cli')
    holder = ProcessWriter(ready)
    try:
        start = time.monotonic()
        engine.on_session_end('ended', history)
        assert time.monotonic() - start < .4
        assert 'raw-message ingest skipped' in caplog.text
        assert engine._store.get_session_count('ended') == 1
    finally:
        holder.release()
    try:
        engine.on_session_end('ended', history)
        engine.on_session_end('ended', history)
        assert engine._store.get_session_count('ended') == 2
    finally:
        engine.shutdown()

@pytest.mark.parametrize('deadline_contract', [False, True])
def test_host_loader_abandonment_stops_plugin_work(ready, monkeypatch, deadline_contract):
    import ast
    import contextvars
    source = (Path(__file__).resolve().parent / 'fixtures' / 'host_load_deadline.py').read_text()
    if deadline_contract:
        source = source.replace('    timeout = _resolve_plugin_load_timeout()\n', '    timeout = _resolve_plugin_load_timeout()\n    ctx._load_deadline = time.monotonic() + timeout if timeout > 0 else None\n', 1)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run_with_load_deadline')
    workers = []
    def make_worker(*a, **kw):
        worker = threading.Thread(*a, **kw)
        workers.append(worker)
        return worker
    scope = dict(threading=SimpleNamespace(Thread=make_worker), time=time,
                 contextvars=contextvars,
                 _IN_PLUGIN_LOAD=contextvars.ContextVar('fixture_load', default=None),
                 _resolve_plugin_load_timeout=lambda: .045,
                 _reserve_abandoned_loader_slot=lambda: None,
                 _LOADER_THREAD_PREFIX='fixture-loader:', _ABANDONED_LOADERS=[],
                 _ABANDONED_LOADERS_LOCK=threading.Lock(), PluginLoadTimeout=TimeoutError)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'loader_fixture', 'exec'), scope)
    ctx = FixtureContext()
    ctx._abandon_load = lambda: setattr(ctx, '_load_abandoned', True)
    with sqlite3.connect(ready) as conn:
        conn.execute('DROP TRIGGER msg_fts_insert')
    monkeypatch.setattr(MessageStore, '_INIT_LOCK_RETRY_BUDGET_S', 5)
    monkeypatch.setattr(LCMConfig, 'from_env', lambda: LCMConfig(database_path=str(ready)))
    mod = entrypoint()
    holder = ProcessWriter(ready)
    try:
        with pytest.raises((TimeoutError, sqlite3.OperationalError)):
            scope['run_with_load_deadline']('fixture', ctx, lambda: mod.register(ctx))
        for worker in workers:
            worker.join(1)
            assert not worker.is_alive()
        assert ctx.engines == [] and ctx.hooks == {}
    finally:
        holder.release()
    with sqlite3.connect(ready, timeout=0) as conn:
        conn.execute('BEGIN IMMEDIATE')
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='msg_fts_insert'").fetchone() is None
        conn.rollback()


def test_expired_host_deadline_registers_nothing(ready, monkeypatch):
    ctx = FixtureContext()
    ctx._load_deadline = time.monotonic() - 1
    monkeypatch.setattr(LCMConfig, 'from_env', lambda: LCMConfig(database_path=str(ready)))
    with pytest.raises(TimeoutError):
        entrypoint().register(ctx)
    assert ctx.engines == [] and ctx.hooks == {}


def test_automatic_scan_restores_progress_and_timeout(ready, monkeypatch):
    obj = MessageStore(ready)
    conn = obj._conn
    def callback():
        return 0
    conn.set_progress_handler(callback, 7)
    conn.execute('PRAGMA busy_timeout=123')
    monkeypatch.setattr(boot, 'AUTOMATIC_FTS_BUDGET_S', .001)
    def work(conn, spec):
        conn.execute('WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100000000) SELECT sum(x) FROM n').fetchone()
    monkeypatch.setattr(boot, 'check_external_content_fts_integrity', work)
    try:
        result = boot._bounded_synchronous_integrity(conn, build_message_fts_spec())
        assert result['status'] == 'unchecked'
        assert conn._progress_handler == (callback, 7)
        assert conn.execute('PRAGMA busy_timeout').fetchone()[0] == 123
        assert not conn.in_transaction
    finally:
        obj.close()


def test_two_processes_bootstrap_all_optional_stores(tmp_path):
    root = Path(__file__).resolve().parents[1]
    db = tmp_path / 'bootstrap.db'
    code = '''
import sys, types
from pathlib import Path
pkg=types.ModuleType('hermes_lcm'); pkg.__path__=[sys.argv[1]]; sys.modules['hermes_lcm']=pkg
from hermes_lcm import tokens
from hermes_lcm.engine import LCMEngine
from hermes_lcm.config import LCMConfig
from hermes_lcm.vector_store import VectorStore
from hermes_lcm.trajectory_store import TrajectoryStore,CorpusIdentity
tokens._encoder_ready=True; tokens._encoder=None
print('ready',flush=True); sys.stdin.readline()
engine=LCMEngine(config=LCMConfig(database_path=sys.argv[2],assertions_enabled=True,query_views_enabled=True,temporal_rollups_enabled=True))
vec=VectorStore(sys.argv[2]); vec.close()
identity=CorpusIdentity(dataset_name='fixture',dataset_revision='1',harness_commit='fixture',tier='small',domain='fixture',ingest_config_digest='fixture')
trajectory=TrajectoryStore(sys.argv[2],identity,asset_root=Path(sys.argv[2]).parent); trajectory.close()
engine._store.append(sys.argv[3],{'role':'user','content':'fixture bootstrap'})
engine.shutdown(); print('done',flush=True)
'''
    workers = [subprocess.Popen([sys.executable, '-c', code, str(root), str(db), str(i)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(2)]
    try:
        for worker in workers:
            assert worker.stdout.readline().strip() == 'ready'
        for worker in workers:
            worker.stdin.write('\n')
            worker.stdin.flush()
        for worker in workers:
            out, err = worker.communicate(timeout=20)
            assert worker.returncode == 0, err
            assert out.strip() == 'done'
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT count(*) FROM messages').fetchone()[0] == 2
        assert conn.execute('SELECT row_count FROM lcm_query_corpus_state').fetchone()[0] == 2
        assert not boot.verify_assertion_schema(conn)
        assert not boot.verify_temporal_rollup_schema(conn)


@pytest.mark.parametrize('cls', [MessageStore, SummaryDAG, LifecycleStateStore, AssertionStore, QueryViewStore, RollupStore, VectorStore])
@pytest.mark.parametrize('failure', [KeyboardInterrupt, sqlite3.OperationalError])
def test_constructor_commit_failure_closes_before_rethrow(ready, monkeypatch, cls, failure):
    observed = []
    original = sqlite3.connect
    def connect(*a, **kw):
        conn = original(*a, **kw)
        if isinstance(conn, _InitializingConnection):
            observed.append(conn)
        return conn
    monkeypatch.setattr(sqlite3, 'connect', connect)
    def fail(conn):
        raise failure('fixture causal commit')
    monkeypatch.setattr(_InitializingConnection, 'commit', fail)
    # VectorStore's verified ready schema skips commit until the constructor's
    # final commit; every helper is therefore tested at its actual boundary.
    expected = RuntimeError if cls is QueryViewStore and failure is sqlite3.OperationalError else failure
    with pytest.raises(expected, match='fixture causal commit') as retained:
        cls(ready)
    assert retained.value.__traceback__ is not None
    if expected is RuntimeError:
        assert isinstance(retained.value.__cause__, failure)
    assert observed
    for conn in observed:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute('SELECT 1')
    with original(ready, timeout=0) as peer:
        peer.execute('BEGIN IMMEDIATE')
        peer.rollback()


def test_failed_batch_keeps_cursor_and_source_then_retries_once(ready, monkeypatch):
    engine = LCMEngine(config=LCMConfig(database_path=str(ready)))
    engine.on_session_start('batch-fixture', platform='discord', conversation_id='fixture-conversation')
    history = [{'role':'user','content':'prefix'}, {'role':'assistant','content':'suffix'}]
    conn = engine._store._conn
    original = _InitializingConnection.commit
    def fail_commit(candidate):
        if candidate is conn:
            raise sqlite3.OperationalError('fixture batch commit')
        return original(candidate)
    try:
        monkeypatch.setattr(_InitializingConnection, 'commit', fail_commit)
        with pytest.raises(sqlite3.OperationalError, match='fixture batch commit'):
            engine._ingest_messages(history)
        assert engine._ingest_cursor == 0 and engine._store.get_session_count('batch-fixture') == 0
        assert not conn.in_transaction
        monkeypatch.setattr(_InitializingConnection, 'commit', original)
        engine._ingest_messages(history)
        engine._ingest_messages(history)
        rows = engine._store.get_range('batch-fixture')
        assert len(rows) == 2 and engine._ingest_cursor == 2
        assert all(row['source'] == 'discord' and row['conversation_id'] == 'fixture-conversation' for row in rows)
    finally:
        engine.shutdown()


def test_constructor_cleanup_does_not_mask_causal_exception(ready, monkeypatch):
    original_close = _InitializingConnection.close
    def close_then_raise(conn):
        original_close(conn)
        raise RuntimeError('secondary cleanup failure')
    def fail(self):
        self._conn.execute("INSERT INTO metadata VALUES('causal-fixture','1')")
        raise ValueError('causal initialization failure')
    monkeypatch.setattr(_InitializingConnection, 'close', close_then_raise)
    monkeypatch.setattr(SummaryDAG, '_ensure_source_window_columns', fail)
    with pytest.raises(ValueError, match='causal initialization failure'):
        SummaryDAG(ready)
    with sqlite3.connect(ready, timeout=0) as conn:
        conn.execute('BEGIN IMMEDIATE')
        assert conn.execute("SELECT 1 FROM metadata WHERE key='causal-fixture'").fetchone() is None
        conn.rollback()


def test_integrity_interrupt_rolls_back_only_its_savepoint(ready, monkeypatch):
    obj = MessageStore(ready)
    conn = obj._conn
    original = _InitializingConnection.execute
    def interrupt(candidate, sql, parameters=(), /):
        if 'integrity-check' in sql:
            raise KeyboardInterrupt('fixture FTS interruption')
        return original(candidate, sql, parameters)
    try:
        conn.execute("INSERT INTO metadata VALUES('outer-fts','kept')")
        monkeypatch.setattr(_InitializingConnection, 'execute', interrupt)
        with pytest.raises(KeyboardInterrupt, match='fixture FTS interruption'):
            boot.check_external_content_fts_integrity(conn, build_message_fts_spec())
        assert conn.in_transaction
        assert conn.execute("SELECT value FROM metadata WHERE key='outer-fts'").fetchone()[0] == 'kept'
        conn.rollback()
        with pytest.raises(KeyboardInterrupt):
            boot.check_external_content_fts_integrity(conn, build_message_fts_spec())
        assert not conn.in_transaction
    finally:
        obj.close()


def test_real_fts_scan_yields_writer_to_foreground_ingest(ready, monkeypatch):
    obj = MessageStore(ready)
    obj.append_batch('fts-fixture', [{'role':'user','content':f'synthetic item {i} ' + 'searchable fixture words ' * 40} for i in range(6000)])
    conn = obj._conn
    conn.execute("DELETE FROM metadata WHERE key LIKE 'fts_integrity_checked_at:%'")
    conn.commit()
    conn.execute('PRAGMA busy_timeout=0')
    started = threading.Event()
    results = []
    original = boot.check_external_content_fts_integrity
    def actual_scan(candidate, spec):
        started.set()
        result = original(candidate, spec)
        results.append(result)
        return result
    monkeypatch.setattr(boot, 'check_external_content_fts_integrity', actual_scan)
    # Reach the actual FTS5 integrity command rather than exhausting this tiny
    # test budget during its preliminary schema/count diagnostics.
    monkeypatch.setattr(boot, '_fts_needs_rebuild_structural', lambda *a, **kw: False)
    monkeypatch.setattr(boot, 'AUTOMATIC_FTS_BUDGET_S', .001)
    try:
        assert boot._dispatch_background_integrity_scan(conn, build_message_fts_spec())
        assert started.wait(2)
        deadline = time.monotonic() + .4
        while True:
            try:
                obj.append('foreground-fixture', {'role':'user','content':'foreground retained'})
                break
            except sqlite3.OperationalError as exc:
                assert boot._is_sqlite_lock_error(exc)
                assert time.monotonic() < deadline
                time.sleep(0)
        boot.join_background_integrity_scans(2)
        assert obj.get_session_count('foreground-fixture') == 1
        assert results and results[0]['status'] == 'unchecked'
        assert conn.execute("SELECT 1 FROM metadata WHERE key='fts_integrity_checked_at:messages_fts'").fetchone() is None
        assert boot.load_integrity_failed(conn, build_message_fts_spec()) is None
    finally:
        obj.close()


def test_contended_maintenance_registry_does_not_block_open(ready, monkeypatch):
    monkeypatch.setattr(boot, '_should_run_integrity_check', lambda *a, **kw: True)
    held = threading.Lock()
    held.acquire()
    probes = []
    class RegistryProbe:
        def acquire(self, blocking=True):
            assert blocking is False, 'automatic dispatch attempted a blocking registry wait'
            probes.append(True)
            return held.acquire(blocking=False)
        def release(self):
            pytest.fail('a contended registry was not acquired')
        def __enter__(self):
            pytest.fail('automatic dispatch used a blocking registry context')
    monkeypatch.setattr(boot, '_integrity_scan_lock', RegistryProbe())
    try:
        obj = MessageStore(ready, init_budget=InitializationBudget())
        obj.close()
        assert probes
    finally:
        held.release()


@pytest.mark.parametrize("damage", ["DELETE FROM lcm_query_corpus_state", "DROP TABLE lcm_query_corpus_state"])
def test_missing_query_singleton_resumes_past_retained_event_cursor(ready, damage):
    obj = MessageStore(ready)
    obj.append_batch('fixture', [{'role':'user','content':'one'}, {'role':'assistant','content':'two'}])
    with sqlite3.connect(ready) as conn:
        old_generation = conn.execute('SELECT generation FROM lcm_query_corpus_state').fetchone()[0]
        conn.execute(damage)
    q = QueryViewStore(ready)
    try:
        restored = q._conn.execute('SELECT generation,row_count FROM lcm_query_corpus_state').fetchone()
        assert restored[0] > old_generation and restored[1] == 2
        obj.append('fixture', {'role':'assistant','content':'after repair'})
        assert q._conn.execute('SELECT row_count FROM lcm_query_corpus_state').fetchone()[0] == 3
        assert q._conn.execute('SELECT max(generation) FROM lcm_query_corpus_events').fetchone()[0] == restored[0] + 1
    finally:
        q.close()
        obj.close()


def test_constructor_checks_remaining_budget_before_commit(ready, monkeypatch):
    def expire(self):
        self._conn.execute("INSERT INTO metadata VALUES('before-expired-commit','fixture')")
        self._init_budget.deadline = 0
    monkeypatch.setattr(SummaryDAG, '_ensure_source_window_columns', expire)
    with pytest.raises(TimeoutError):
        SummaryDAG(ready)
    with sqlite3.connect(ready, timeout=0) as conn:
        conn.execute('BEGIN IMMEDIATE')
        assert conn.execute("SELECT 1 FROM metadata WHERE key='before-expired-commit'").fetchone() is None
        conn.rollback()


@pytest.mark.parametrize('caller_transaction', [False, True])
def test_synchronous_fts_result_owns_only_its_metadata_transaction(ready, monkeypatch, caller_transaction):
    monkeypatch.setenv('LCM_FTS_INTEGRITY_BACKGROUND', 'false')
    monkeypatch.setattr(boot, '_should_run_integrity_check', lambda *a, **kw: True)
    conn = sqlite3.connect(ready, timeout=0)  # Real legacy/default transaction mode.
    try:
        conn.execute("DELETE FROM metadata WHERE key='fts_integrity_checked_at:messages_fts'")
        conn.commit()
        if caller_transaction:
            conn.execute("INSERT INTO metadata VALUES('outer-default-fts','kept')")
        boot.ensure_external_content_fts(conn, build_message_fts_spec())
        assert conn.in_transaction == caller_transaction
        if caller_transaction:
            assert conn.execute("SELECT value FROM metadata WHERE key='outer-default-fts'").fetchone()[0] == 'kept'
            conn.rollback()
        with sqlite3.connect(ready, timeout=0) as peer:
            peer.execute('BEGIN IMMEDIATE')
            peer.rollback()
    finally:
        conn.close()


@pytest.mark.parametrize('caller_transaction', [False, True])
def test_fts_result_marker_interrupt_preserves_ownership(ready, monkeypatch, caller_transaction):
    conn = sqlite3.connect(ready, timeout=0)
    original = boot._record_integrity_checked
    def interrupt(candidate, spec, *, now=None):
        original(candidate, spec, now=now)
        raise KeyboardInterrupt('fixture result marker')
    try:
        conn.execute("DELETE FROM metadata WHERE key='fts_integrity_checked_at:messages_fts'")
        conn.commit()
        if caller_transaction:
            conn.execute("INSERT INTO metadata VALUES('outer-result-marker','kept')")
        monkeypatch.setattr(boot, '_record_integrity_checked', interrupt)
        with pytest.raises(KeyboardInterrupt, match='fixture result marker'):
            boot.repair_external_content_fts(conn, build_message_fts_spec(), throttle=False)
        assert conn.in_transaction == caller_transaction
        assert conn.execute("SELECT 1 FROM metadata WHERE key='fts_integrity_checked_at:messages_fts'").fetchone() is None
        if caller_transaction:
            assert conn.execute("SELECT value FROM metadata WHERE key='outer-result-marker'").fetchone()[0] == 'kept'
            conn.rollback()
        with sqlite3.connect(ready, timeout=0) as peer:
            peer.execute('BEGIN IMMEDIATE')
            peer.rollback()
    finally:
        conn.close()
