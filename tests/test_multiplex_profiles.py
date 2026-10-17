"""Routed homes are task-local; the gateway environment stays unchanged."""

from contextvars import ContextVar
import copy
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def routed_plugin(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    homes = {}
    for name, threshold, timeout in (
        ("default", 0.71, 11), ("alpha", 0.43, 23), ("beta", 0.62, 37),
    ):
        home = tmp_path / name
        home.mkdir()
        (home / "config.yaml").write_text(
            f"compression:\n  enabled: true\n  threshold: {threshold}\n"
            f"auxiliary:\n  compression:\n    timeout: {timeout}\n"
        )
        homes[name] = home
    monkeypatch.setenv("HERMES_HOME", str(homes["default"]))
    active_home = ContextVar("test_hermes_home", default=homes["default"])
    constants = ModuleType("hermes_constants")
    constants.get_hermes_home = active_home.get
    monkeypatch.setitem(sys.modules, "hermes_constants", constants)
    # No real host config, plugin manager, credentials, or session DB is loaded.
    manager = SimpleNamespace(_hooks={})
    cli = ModuleType("hermes_cli")
    cli.__path__ = []
    cli_config = ModuleType("hermes_cli.config")
    cli_config.get_hermes_home = active_home.get
    plugins = ModuleType("hermes_cli.plugins")
    plugins.get_plugin_manager = lambda: manager
    for name, module in (("hermes_cli", cli), ("hermes_cli.config", cli_config),
                         ("hermes_cli.plugins", plugins)):
        monkeypatch.setitem(sys.modules, name, module)

    name = "multiplex_test_plugin"
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    engines = []
    hooks = {}
    ctx = SimpleNamespace(
        register_context_engine=engines.append,
        register_hook=lambda name, handler: hooks.update({name: handler}),
    )
    try:
        yield SimpleNamespace(
            module=module, ctx=ctx, homes=homes, active_home=active_home,
            engines=engines, hooks=hooks, manager=manager,
        )
    finally:
        for engine in reversed(engines):
            engine.shutdown()
        for module_name in list(sys.modules):
            if module_name.startswith(name + "."):
                monkeypatch.delitem(sys.modules, module_name)


def test_registered_prototype_clones_use_routed_config_and_storage(routed_plugin):
    host = routed_plugin
    host.module.register(host.ctx)
    prototype = host.engines[0]
    token = host.active_home.set(host.homes["alpha"])
    try:
        clone = prototype.clone_for_agent()
        host.engines.append(clone)
        assert clone._config.context_threshold == 0.43
        assert clone._config.summary_timeout_ms == 23000
        assert clone._config.config_sources["context_threshold"] == "config_yaml:compression.threshold"
        assert clone._store.db_path == host.homes["alpha"] / "lcm.db"
        clone.on_session_start("alpha-session", context_length=100000)
        assert clone.context_threshold == 0.43
        assert clone.threshold_tokens == 43000
        assert prototype._config.context_threshold == 0.71
        assert prototype._store.db_path == host.homes["default"] / "lcm.db"
        assert os.environ["HERMES_HOME"] == str(host.homes["default"])
    finally:
        host.active_home.reset(token)


@pytest.mark.parametrize("route_mode", ["ambient", "explicit", "empty"])
def test_session_start_rebinds_config_and_clears_profile_state(routed_plugin, route_mode):
    host = routed_plugin
    host.module.register(host.ctx)
    engine = host.engines[0].clone_for_agent()
    host.engines.append(engine)
    engine.on_session_start("old-session", conversation_id="old-lane", context_length=100000)
    engine.ingest([{"role": "user", "content": "default-only record"}])
    engine._lifecycle.record_debt("old-lane", kind="raw_backlog", size_estimate=123)
    old_store = engine._store
    # An explicit lifecycle home wins even if the ambient context is different;
    # an empty host value falls back to the context-local route.
    token = host.active_home.set(host.homes["alpha"])
    try:
        if route_mode == "explicit":
            kwargs = {"hermes_home": str(host.homes["beta"])}
            home = host.homes["beta"]
        elif route_mode == "empty":
            kwargs = {"hermes_home": ""}
            home = host.homes["alpha"]
        else:
            kwargs = {}
            home = host.homes["alpha"]
        engine.on_session_start("new-session", conversation_id="new-lane", **kwargs)
        expected_threshold = 0.62 if route_mode == "explicit" else 0.43
        expected_timeout = 37000 if route_mode == "explicit" else 23000
        assert engine._config.context_threshold == expected_threshold
        assert engine._config.summary_timeout_ms == expected_timeout
        assert engine.threshold_tokens == int(expected_threshold * 100000)
        assert engine._store.db_path == home / "lcm.db"
        assert engine._dag.db_path == home / "lcm.db"
        assert engine._lifecycle.db_path == home / "lcm.db"
        assert old_store._conn is None
        assert engine._store.get_session_messages("old-session") == []
        assert engine._lifecycle.get_by_conversation("old-lane") is None
        assert engine.current_session_id == "new-session"
        assert engine.current_conversation_id == "new-lane"
        engine.ingest([{"role": "user", "content": "routed-only record"}])
        assert len(engine._store.get_session_messages("new-session")) == 1
        assert host.engines[0]._store.get_session_messages("new-session") == []
        assert os.environ["HERMES_HOME"] == str(host.homes["default"])
    finally:
        host.active_home.reset(token)


def test_interleaved_profiles_dispatch_hooks_without_sibling_state(routed_plugin):
    host = routed_plugin
    host.module.register(host.ctx)
    prototype = host.engines[0]
    resolve = sys.modules[prototype.__module__].resolve_active_lcm_engine
    clones = {}
    # Session/lane IDs can repeat in independently persisted profile state.
    for name in ("alpha", "beta"):
        token = host.active_home.set(host.homes[name])
        try:
            clone = prototype.clone_for_agent()
            host.engines.append(clone)
            clones[name] = clone
            clone.on_session_start("same-session", conversation_id="same-lane", context_length=100000)
            clone._lifecycle.record_debt("same-lane", kind="raw_backlog", size_estimate=123 if name == "alpha" else 456)
        finally:
            host.active_home.reset(token)

    for name in ("alpha", "beta", "alpha", "beta"):
        token = host.active_home.set(host.homes[name])
        try:
            clone = clones[name]
            assert resolve(session_id="same-session") is clone
            assert resolve(conversation_id="same-lane") is clone
            assert host.hooks["pre_llm_call"](session_id="same-session") == {"context": host.module.get_recall_policy()}
            # Exercise legacy hook lookup without a direct context_compressor.
            host.manager._hooks["post_llm_call"][0](
                session_id="same-session", conversation_id="same-lane",
                conversation_history=[{"role": "user", "content": name + " private record"}],
            )
            assert [row["content"] for row in clone._store.get_session_messages("same-session")] == [name + " private record"]
            state = clone._lifecycle.get_by_conversation("same-lane")
            assert state.debt_size_estimate == (123 if name == "alpha" else 456)
            assert clone._config.summary_timeout_ms == (23000 if name == "alpha" else 37000)
            assert clone.threshold_tokens == (43000 if name == "alpha" else 62000)
            assert os.environ["HERMES_HOME"] == str(host.homes["default"])
        finally:
            host.active_home.reset(token)
    assert prototype._store.get_session_messages("same-session") == []
    assert prototype._lifecycle.get_by_conversation("same-lane") is None
    assert resolve(session_id="same-session", conversation_id="same-lane", allow_foreground=True) is None


def test_lcm_environment_overrides_win_for_routed_yaml_and_do_not_mutate_home(
    routed_plugin, monkeypatch
):
    host = routed_plugin
    from hermes_lcm.config import LCMConfig

    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.91")
    monkeypatch.setenv("LCM_SUMMARY_TIMEOUT_MS", "91000")
    before = os.environ["HERMES_HOME"]
    config = LCMConfig.from_env(hermes_home=host.homes["alpha"])

    assert config.context_threshold == 0.91
    assert config.summary_timeout_ms == 91000
    assert config.config_sources["context_threshold"] == "env:LCM_CONTEXT_THRESHOLD"
    assert config.config_sources["summary_timeout_ms"] == "env:LCM_SUMMARY_TIMEOUT_MS"
    assert os.environ["HERMES_HOME"] == before


def test_legacy_home_resolution_falls_back_to_process_environment(monkeypatch, tmp_path):
    from hermes_lcm.config import resolve_hermes_home

    legacy_home = tmp_path / "legacy-home"
    monkeypatch.setenv("HERMES_HOME", str(legacy_home))
    # Force both optional context-aware host modules to be unavailable. A
    # legacy standalone host then retains the documented environment fallback.
    for module_name in ("hermes_constants", "hermes_cli", "hermes_cli.config"):
        monkeypatch.setitem(sys.modules, module_name, None)

    assert resolve_hermes_home() == legacy_home


def test_legacy_post_hook_rebinds_same_session_id_across_profiles(routed_plugin):
    host = routed_plugin
    host.module.register(host.ctx)
    engine = host.engines[0]
    hook = host.manager._hooks["post_llm_call"][0]
    expected_by_profile = {"alpha": [], "beta": []}
    for name in ("alpha", "beta", "alpha"):
        token = host.active_home.set(host.homes[name])
        try:
            hook(session_id="same-session", conversation_history=[{"role": "user", "content": name}])
            expected_by_profile[name].append(name)
            assert engine._store.db_path == host.homes[name] / "lcm.db"
            assert [row["content"] for row in engine._store.get_session_messages("same-session")] == expected_by_profile[name]
        finally:
            host.active_home.reset(token)


@pytest.mark.parametrize("use_yaml", [False, True])
def test_routed_registration_and_direct_config_loading(routed_plugin, monkeypatch, use_yaml):
    host = routed_plugin
    config_module = importlib.import_module(host.module.__name__ + ".config")
    if not use_yaml:
        monkeypatch.setattr(config_module, "yaml", None)
    # A stale legacy alias must not shadow the canonical context-local API.
    monkeypatch.setattr(sys.modules["hermes_cli.config"], "get_hermes_home", lambda: host.homes["default"])
    token = host.active_home.set(host.homes["beta"])
    try:
        config = config_module.LCMConfig.from_env()
        assert config.context_threshold == 0.62
        assert config.summary_timeout_ms == 37000
        host.module.register(host.ctx)
        engine = host.engines[0]
        assert engine._config.context_threshold == 0.62
        assert engine._store.db_path == host.homes["beta"] / "lcm.db"
        assert os.environ["HERMES_HOME"] == str(host.homes["default"])
    finally:
        host.active_home.reset(token)


def test_operator_overrides_survive_routed_clone_and_rebind(routed_plugin, monkeypatch, tmp_path):
    host = routed_plugin
    db_path = tmp_path / "operator.db"
    monkeypatch.setenv("LCM_DATABASE_PATH", str(db_path))
    monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.52")
    monkeypatch.setenv("LCM_SUMMARY_TIMEOUT_MS", "19000")
    monkeypatch.setenv("LCM_ASSERTIONS_ENABLED", "true")
    monkeypatch.setenv("LCM_QUERY_VIEWS_ENABLED", "true")
    host.module.register(host.ctx)
    token = host.active_home.set(host.homes["alpha"])
    try:
        clone = copy.deepcopy(host.engines[0])
        host.engines.append(clone)
        clone.on_session_start("alpha-session", context_length=100000)
        for name in ("alpha", "beta"):
            clone.on_session_start(name + "-session", hermes_home=str(host.homes[name]))
            assert clone._config.context_threshold == 0.52
            assert clone.threshold_tokens == 52000
            assert clone._config.summary_timeout_ms == 19000
            assert clone._config.config_sources["context_threshold"] == "env:LCM_CONTEXT_THRESHOLD"
            assert clone._config.config_sources["summary_timeout_ms"] == "env:LCM_SUMMARY_TIMEOUT_MS"
            for helper in (clone._store, clone._dag, clone._lifecycle, clone._assertions, clone._query_views):
                assert helper.db_path == db_path
            assert clone._store._hermes_home == str(host.homes[name])
            assert clone._store._ingest_protection_config is clone._config
        assert host.engines[0]._hermes_home == str(host.homes["default"])
        assert clone._config is not host.engines[0]._config
    finally:
        host.active_home.reset(token)


def test_explicit_config_is_caller_owned_across_routed_clones(routed_plugin):
    host = routed_plugin
    config_module = importlib.import_module(host.module.__name__ + ".config")
    engine_module = importlib.import_module(host.module.__name__ + ".engine")
    config = config_module.LCMConfig(context_threshold=0.39, summary_timeout_ms=17000)
    prototype = engine_module.LCMEngine(config=config, hermes_home=str(host.homes["default"]))
    host.engines.append(prototype)
    token = host.active_home.set(host.homes["alpha"])
    try:
        clone = prototype.clone_for_agent()
        host.engines.append(clone)
        clone.on_session_start("manual-session", hermes_home=str(host.homes["beta"]), context_length=100000)
        assert clone._config.context_threshold == 0.39
        assert clone._config.summary_timeout_ms == 17000
        assert clone.threshold_tokens == 39000
        assert clone._store.db_path == host.homes["beta"] / "lcm.db"
        clone._config.ignore_session_patterns.append("manual-only")
        assert "manual-only" not in prototype._config.ignore_session_patterns
    finally:
        host.active_home.reset(token)


@pytest.mark.parametrize("fallback", ["legacy_api", "environment", "default_home"])
def test_standalone_and_legacy_home_fallback(routed_plugin, monkeypatch, tmp_path, fallback):
    host = routed_plugin
    monkeypatch.setitem(sys.modules, "hermes_constants", None)
    if fallback != "legacy_api":
        monkeypatch.setitem(sys.modules, "hermes_cli.config", None)
    if fallback == "default_home":
        monkeypatch.delenv("HERMES_HOME")
        monkeypatch.setenv("HOME", str(tmp_path / "standalone"))
        home = tmp_path / "standalone" / ".hermes"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("compression:\n  threshold: 0.47\nauxiliary:\n  compression:\n    timeout: 13\n")
        threshold, timeout = 0.47, 13000
    elif fallback == "legacy_api":
        host.active_home.set(host.homes["alpha"])
        home, threshold, timeout = host.homes["alpha"], 0.43, 23000
    else:
        home, threshold, timeout = host.homes["default"], 0.71, 11000
    host.module.register(host.ctx)
    prototype = host.engines[0]
    clone = prototype.clone_for_agent()
    host.engines.append(clone)
    clone.on_session_start("fallback-session", context_length=100000)
    assert clone._store.db_path == home / "lcm.db"
    assert clone._config.context_threshold == threshold
    assert clone._config.summary_timeout_ms == timeout
    if fallback == "default_home":
        assert "HERMES_HOME" not in os.environ


@pytest.mark.parametrize("contents", [None, "not: [valid yaml"])
def test_missing_or_invalid_profile_yaml_keeps_defaults(routed_plugin, contents):
    host = routed_plugin
    path = host.homes["default"] / "config.yaml"
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents)
    host.module.register(host.ctx)
    engine = host.engines[0]
    config_module = importlib.import_module(host.module.__name__ + ".config")
    defaults = config_module.LCMConfig()
    assert engine._config.context_threshold == defaults.context_threshold
    assert engine._config.summary_timeout_ms == defaults.summary_timeout_ms
    assert engine._store.db_path == host.homes["default"] / "lcm.db"


def test_broken_context_resolver_does_not_fall_back_to_sibling(routed_plugin, monkeypatch):
    host = routed_plugin

    def broken_resolver():
        raise RuntimeError("synthetic route failure")

    monkeypatch.setattr(sys.modules["hermes_constants"], "get_hermes_home", broken_resolver)
    with pytest.raises(RuntimeError, match="synthetic route failure"):
        host.module.register(host.ctx)
    assert not (host.homes["default"] / "lcm.db").exists()
