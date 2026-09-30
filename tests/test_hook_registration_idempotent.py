"""Regression: register() must not accumulate post_llm_call ingest hooks.

The host calls register(ctx) once per agent build (load_context_engine() ->
instance_from_module() -> register()).  Every build creates a fresh LCMEngine
with its own SQLite descriptors, and before this regression was covered, every
build also appended a fresh post_llm_call closure to the process-wide plugin
manager without removing the previous one.  Each stale closure pinned its
build's engine for the life of the process, so long-running gateways
accumulated thousands of lcm.db file descriptors; once open descriptors exceed
FD_SETSIZE, select()-based terminal drains start failing silently (empty tool
output, exit code 0).

With the fix in place the descriptor count must plateau: retired engines are
collectable, they release their WAL handles on collection, and SQLite's unix
VFS reuses deferred-close descriptors for subsequent connections, so repeated
builds stabilise instead of growing ~6 descriptors per build.
"""
from pathlib import Path
import gc
import importlib.util
import os
import sqlite3
import sys
import types
import weakref

import pytest


def _load_plugin_entrypoint_module(module_name: str):
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(repo_root / "__init__.py"),
        submodule_search_locations=[str(repo_root)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_agent_context_engine_importable(monkeypatch):
    """Install the smallest host stub needed for isolated plugin tests."""
    agent_module = sys.modules.get("agent")
    if agent_module is None:
        agent_module = types.ModuleType("agent")
        agent_module.__path__ = []
        monkeypatch.setitem(sys.modules, "agent", agent_module)
    context_engine_module = types.ModuleType("agent.context_engine")

    class ContextEngine:
        def on_session_reset(self):
            return None

    context_engine_module.ContextEngine = ContextEngine
    monkeypatch.setitem(sys.modules, "agent.context_engine", context_engine_module)
    monkeypatch.setattr(
        agent_module, "context_engine", context_engine_module, raising=False
    )


class _CtxNoTool:
    def __init__(self):
        self.engine = None

    def register_context_engine(self, engine):
        self.engine = engine


def _make_host(monkeypatch, tmp_path, module_name):
    _ensure_agent_context_engine_importable(monkeypatch)
    module = _load_plugin_entrypoint_module(module_name)
    manager = types.SimpleNamespace(_hooks={})
    fake_plugins = types.SimpleNamespace(get_plugin_manager=lambda: manager)
    fake_hermes_cli = types.SimpleNamespace(plugins=fake_plugins)
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", fake_plugins)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    return module, manager


def _lcm_fd_count(home_dir):
    """Count open descriptors whose target lives under this test's HERMES_HOME.

    Scoped to the test's own home so descriptors from engines owned by other
    tests (or earlier phases) can never skew the measurement.
    """
    if sys.platform != "linux":
        return None
    home_marker = str(home_dir)
    try:
        names = os.listdir("/proc/self/fd")
    except OSError:
        return None
    count = 0
    for name in names:
        try:
            target = os.readlink(f"/proc/self/fd/{name}")
        except OSError:
            continue
        if home_marker in target and "lcm.db" in target:
            count += 1
    return count


def _sqlite_connection_count():
    return sum(1 for obj in gc.get_objects() if isinstance(obj, sqlite3.Connection))


def test_register_replaces_superseded_post_llm_hook(monkeypatch, tmp_path):
    module, manager = _make_host(monkeypatch, tmp_path, "hermes_lcm_hook_idempotency")

    first = _CtxNoTool()
    module.register(first)
    assert first.engine is not None
    assert len(manager._hooks["post_llm_call"]) == 1

    second = _CtxNoTool()
    module.register(second)
    assert second.engine is not None

    # The newer build must replace the stale ingest hook, not stack onto it.
    assert len(manager._hooks["post_llm_call"]) == 1

    # ...and the superseded engine must become collectable (the stale closure
    # used to pin it, and with it every SQLite descriptor the engine owns).
    stale_ref = weakref.ref(first.engine)
    del first
    gc.collect()
    assert stale_ref() is None

    second.engine.shutdown()


def test_register_does_not_leak_engine_file_descriptors(monkeypatch, tmp_path):
    module, manager = _make_host(monkeypatch, tmp_path, "hermes_lcm_fd_leak")
    home_dir = tmp_path / "hermes_home"
    if _lcm_fd_count(home_dir) is None:
        pytest.skip("no per-process fd table on this platform")

    # Warm up: the first engine bootstraps the SQLite schema in the temporary
    # home and settles one-time descriptors before the baseline is taken.
    warm = _CtxNoTool()
    module.register(warm)
    warm.engine.shutdown()
    del warm
    gc.collect()
    baseline_fds = _lcm_fd_count(home_dir)
    baseline_conns = _sqlite_connection_count()

    # Repeated builds, retiring the previous engine each cycle.  Fixed code
    # plateaus (~10 descriptors regardless of cycle count); unfixed code grows
    # ~6 descriptors per build because every stale hook pins its engine.
    totals = []
    live = None
    for _ in range(6):
        ctx = _CtxNoTool()
        module.register(ctx)
        if live is not None:
            del live
            gc.collect()
        live = ctx.engine
        totals.append(_lcm_fd_count(home_dir))

    assert totals[0] >= baseline_fds + 6, (
        f"sanity: a live engine should hold descriptors, got {totals}"
    )
    assert totals[-1] <= baseline_fds + 20, (
        f"file descriptors grew with every build: {totals} "
        f"(baseline={baseline_fds}); a stale post_llm_call hook is pinning engines"
    )
    assert totals[-1] == totals[-2], (
        f"descriptor count did not plateau across builds: {totals}"
    )

    live_conns = _sqlite_connection_count()
    assert live_conns <= baseline_conns + 4, (
        "superseded engines' SQLite connections are still alive "
        f"({live_conns - baseline_conns} extra); the stale hook still pins them"
    )

    # Full teardown: after the newest engine is shut down and collected, the
    # whole chain must release every descriptor.
    live.shutdown()
    del live
    gc.collect()
    assert _lcm_fd_count(home_dir) <= baseline_fds + 4
    assert _sqlite_connection_count() <= baseline_conns + 2
