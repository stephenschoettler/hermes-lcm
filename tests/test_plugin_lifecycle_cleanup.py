"""Plugin-owned cleanup for finalized per-agent LCM clones."""

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_plugin_module(name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        str(REPO_ROOT / "__init__.py"),
        submodule_search_locations=[str(REPO_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _registered_plugin(tmp_path, monkeypatch, module_name: str):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    module = _load_plugin_module(module_name)
    hooks = {}

    class _Ctx:
        def register_context_engine(self, engine):
            self.engine = engine

        def register_hook(self, name, callback):
            hooks[name] = callback

    ctx = _Ctx()
    module.register(ctx)
    return ctx.engine, hooks


def test_finalize_closes_only_the_clone_bound_to_that_session(tmp_path, monkeypatch):
    prototype, hooks = _registered_plugin(
        tmp_path,
        monkeypatch,
        "hermes_lcm_finalize_exact_clone",
    )
    clone_a = prototype.clone_for_agent()
    clone_b = prototype.clone_for_agent()
    clone_a.on_session_start("session-a", platform="desktop")
    clone_b.on_session_start("session-b", platform="desktop")

    try:
        hooks["on_session_finalize"](
            session_id="session-a",
            platform="desktop",
            reason="tui_close",
        )

        assert clone_a._store._conn is None
        assert clone_a._dag._conn is None
        assert clone_a._lifecycle._conn is None
        assert clone_b._store._conn is not None
        assert clone_b._dag._conn is not None
        assert clone_b._lifecycle._conn is not None
        assert prototype._store._conn is not None
        assert prototype._dag._conn is not None
        assert prototype._lifecycle._conn is not None
    finally:
        clone_a.shutdown()
        clone_b.shutdown()
        prototype.shutdown()


def test_cli_finalize_keeps_the_reused_clone_open(tmp_path, monkeypatch):
    prototype, hooks = _registered_plugin(
        tmp_path,
        monkeypatch,
        "hermes_lcm_finalize_cli_reuse",
    )
    clone = prototype.clone_for_agent()
    clone.on_session_start("cli-session", platform="cli")

    try:
        hooks["on_session_finalize"](
            session_id="cli-session",
            platform="cli",
            reason="session_boundary",
        )

        assert clone._store._conn is not None
        assert clone._dag._conn is not None
        assert clone._lifecycle._conn is not None
    finally:
        clone.shutdown()
        prototype.shutdown()
