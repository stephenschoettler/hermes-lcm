"""Fail-closed profile transitions and deferred lifecycle regressions."""
import json
import threading

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.engine import LCMEngine


def test_failed_rebind_retry_never_carries_previous_profile_identity(tmp_path, monkeypatch):
    a, b = tmp_path / 'a', tmp_path / 'b'
    engine = LCMEngine(hermes_home=str(a))
    engine.on_session_start('A-session', hermes_home=str(a), platform='test')
    original = engine_module.SummaryDAG

    def fail_b(path):
        if str(path) == str(b / 'lcm.db'):
            raise RuntimeError('B initialization failed')
        return original(path)

    monkeypatch.setattr(engine_module, 'SummaryDAG', fail_b)
    with pytest.raises(RuntimeError, match='B initialization failed'):
        engine.on_session_start('B-session', hermes_home=str(b), platform='test')
    assert engine.current_session_id == ''
    assert engine._conversation_id == ''
    assert engine._ingest_cursor == 0
    monkeypatch.setattr(engine_module, 'SummaryDAG', original)
    status = json.loads(engine.handle_tool_call('lcm_status', {}))
    assert status.get('error') == 'No active session'
    engine.ingest([{'role': 'user', 'content': 'must not write A into B'}])
    assert engine._store.get_session_count('A-session') == 0
    engine.on_session_start('B-session', hermes_home=str(b), platform='test')
    engine.ingest([{'role': 'user', 'content': 'belongs to B'}])
    assert engine._store.get_session_count('B-session') == 1
    engine.shutdown()


def test_slash_status_handles_deferred_prototype_without_storage(tmp_path):
    engine = LCMEngine(hermes_home=str(tmp_path), defer_storage=True)
    text = handle_lcm_command('status', engine)
    assert 'storage_ready: no' in text
    assert 'pending_initialization' in text
    assert not (tmp_path / 'lcm.db').exists()
    engine.shutdown()


def test_shutdown_serializes_with_initialization_and_cannot_resurrect(tmp_path, monkeypatch):
    entered, release, shutdown_done = threading.Event(), threading.Event(), threading.Event()
    helpers = []

    class Helper:
        def __init__(self, path, **kwargs):
            self.db_path = path
            self.closed = False
            self._hermes_home = str(tmp_path)
            helpers.append(self)

        def close(self):
            self.closed = True

    def blocked_store(path, **kwargs):
        entered.set()
        assert release.wait(5)
        return Helper(path, **kwargs)

    monkeypatch.setattr(engine_module, 'MessageStore', blocked_store)
    monkeypatch.setattr(engine_module, 'SummaryDAG', Helper)
    monkeypatch.setattr(engine_module, 'LifecycleStateStore', Helper)
    engine = LCMEngine(hermes_home=str(tmp_path), defer_storage=True)
    errors = []

    def initialize():
        try:
            engine._ensure_storage()
        except Exception as exc:
            errors.append(exc)

    def stop():
        engine.shutdown()
        shutdown_done.set()

    init = threading.Thread(target=initialize)
    stop_thread = threading.Thread(target=stop)
    init.start()
    assert entered.wait(3)
    stop_thread.start()
    assert not shutdown_done.wait(0.1)
    release.set()
    init.join(5)
    stop_thread.join(5)
    assert not init.is_alive() and not stop_thread.is_alive()
    assert not errors
    assert all(h.closed for h in helpers)
    assert engine._storage_ready is False
    with pytest.raises(RuntimeError, match='shut down'):
        engine._ensure_storage()
