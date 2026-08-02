"""Plugin-owned cleanup for finalized per-agent LCM clones."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_plugin_module(name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        str(REPO_ROOT / "__init__.py"),
        submodule_search_locations=[str(REPO_ROOT)],
    )
    assert spec is not None
    assert spec.loader is not None
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
    clone_a_shutdown = clone_a.shutdown
    clone_b_shutdown = clone_b.shutdown
    prototype_shutdown = prototype.shutdown
    clone_a.shutdown = Mock(wraps=clone_a_shutdown)
    clone_b.shutdown = Mock(wraps=clone_b_shutdown)
    prototype.shutdown = Mock(wraps=prototype_shutdown)

    try:
        hooks["on_session_finalize"](
            session_id="session-a",
            platform="desktop",
            reason="tui_close",
        )

        clone_a.shutdown.assert_called_once_with()
        clone_b.shutdown.assert_not_called()
        prototype.shutdown.assert_not_called()
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
    clone_shutdown = clone.shutdown
    clone.shutdown = Mock(wraps=clone_shutdown)

    try:
        hooks["on_session_finalize"](
            session_id="cli-session",
            platform="cli",
            reason="session_boundary",
        )

        clone.shutdown.assert_not_called()
    finally:
        clone.shutdown()
        prototype.shutdown()
