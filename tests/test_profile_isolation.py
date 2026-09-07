"""Regression tests for routed Hermes profile isolation."""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from contextvars import ContextVar
from types import ModuleType
from pathlib import Path

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _write_profile_config(home: Path, *, threshold: float, timeout: float) -> None:
    home.mkdir()
    (home / "config.yaml").write_text(
        "\n".join(
            [
                "compression:",
                "  enabled: true",
                f"  threshold: {threshold}",
                "auxiliary:",
                "  compression:",
                f"    timeout: {timeout}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _load_plugin_module(name: str):
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        name,
        str(repo_root / "__init__.py"),
        submodule_search_locations=[str(repo_root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_from_env_accepts_routed_home_without_mutating_process_environment(tmp_path, monkeypatch):
    default_home = tmp_path / "default"
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    _write_profile_config(default_home, threshold=0.11, timeout=11)
    _write_profile_config(profile_a, threshold=0.23, timeout=23)
    _write_profile_config(profile_b, threshold=0.79, timeout=79)

    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.delenv("LCM_CONTEXT_THRESHOLD", raising=False)
    monkeypatch.delenv("LCM_SUMMARY_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    before = os.environ["HERMES_HOME"]

    config_a = LCMConfig.from_env(hermes_home=str(profile_a))
    config_b = LCMConfig.from_env(hermes_home=str(profile_b))

    assert config_a.context_threshold == 0.23
    assert config_b.context_threshold == 0.79
    assert config_a.summary_timeout_ms == 23_000
    assert config_b.summary_timeout_ms == 79_000
    assert os.environ["HERMES_HOME"] == before

    monkeypatch.setenv("LCM_SUMMARY_TIMEOUT_MS", "1234")
    monkeypatch.setenv("LCM_DATABASE_PATH", str(tmp_path / "shared.db"))
    override_a = LCMConfig.from_env(hermes_home=str(profile_a))
    override_b = LCMConfig.from_env(hermes_home=str(profile_b))
    assert override_a.summary_timeout_ms == 1234
    assert override_b.summary_timeout_ms == 1234
    assert override_a.database_path == override_b.database_path == str(tmp_path / "shared.db")


def test_context_local_home_is_used_when_host_omits_lifecycle_home(tmp_path, monkeypatch):
    default_home = tmp_path / "default"
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    _write_profile_config(default_home, threshold=0.13, timeout=13)
    _write_profile_config(profile_a, threshold=0.29, timeout=29)
    _write_profile_config(profile_b, threshold=0.83, timeout=83)

    active_home = ContextVar("active_home", default=str(default_home))
    core = ModuleType("hermes_constants")
    setattr(core, "get_hermes_home", lambda: active_home.get())
    monkeypatch.setitem(sys.modules, "hermes_constants", core)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.delenv("LCM_CONTEXT_THRESHOLD", raising=False)
    monkeypatch.setenv("LCM_SUMMARY_SPEND_MAX_CALLS", "4")
    monkeypatch.setenv("LCM_SUMMARY_SPEND_WINDOW_SECONDS", "10")
    monkeypatch.setenv("LCM_SUMMARY_SPEND_BACKOFF_SECONDS", "20")
    monkeypatch.setenv("LCM_SUMMARY_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("LCM_SUMMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "30")

    engine = LCMEngine(
        config=LCMConfig.from_env(hermes_home=str(profile_a)),
        hermes_home=str(profile_a),
    )
    spend_guard = engine._summary_spend_guard
    circuit_breaker = engine._summary_circuit_breaker
    assert spend_guard.try_record_call(now=1.0) is True
    circuit_breaker.record_failure("test-model", now=1.0)
    try:
        monkeypatch.setenv("LCM_SUMMARY_SPEND_MAX_CALLS", "9")
        monkeypatch.setenv("LCM_SUMMARY_SPEND_WINDOW_SECONDS", "90")
        monkeypatch.setenv("LCM_SUMMARY_SPEND_BACKOFF_SECONDS", "180")
        monkeypatch.setenv("LCM_SUMMARY_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")
        monkeypatch.setenv("LCM_SUMMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "300")
        token = active_home.set(str(profile_b))
        try:
            engine.on_session_start("session-b")
        finally:
            active_home.reset(token)
        assert engine._config.context_threshold == 0.83
        assert engine._config.summary_timeout_ms == 83_000
        assert engine._store.db_path == profile_b / "lcm.db"
        assert spend_guard.max_calls == 9
        assert spend_guard.window_seconds == 90.0
        assert spend_guard.backoff_seconds == 180.0
        assert circuit_breaker.failure_threshold == 5
        assert circuit_breaker.cooldown_seconds == 300
        assert os.environ["HERMES_HOME"] == str(default_home)

        token = active_home.set(str(profile_a))
        try:
            engine.on_session_start("session-a")
        finally:
            active_home.reset(token)
        assert engine._config.context_threshold == 0.29
        assert engine._store.db_path == profile_a / "lcm.db"
        assert engine._summary_spend_guard is spend_guard
        assert spend_guard._calls == [1.0]
        assert engine._summary_circuit_breaker is circuit_breaker
        assert circuit_breaker._failures["test-model"] == 1
    finally:
        engine.shutdown()


def test_plugin_registration_uses_context_local_home(tmp_path, monkeypatch):
    default_home = tmp_path / "default"
    profile_b = tmp_path / "profile-b"
    _write_profile_config(default_home, threshold=0.17, timeout=17)
    _write_profile_config(profile_b, threshold=0.77, timeout=77)

    active_home = ContextVar("active_home", default=str(default_home))
    core = ModuleType("hermes_constants")
    setattr(core, "get_hermes_home", lambda: active_home.get())
    monkeypatch.setitem(sys.modules, "hermes_constants", core)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.delenv("LCM_CONTEXT_THRESHOLD", raising=False)

    class Context:
        engine = None

        def register_context_engine(self, engine):
            self.engine = engine

    token = active_home.set(str(profile_b))
    ctx = Context()
    try:
        _load_plugin_module("hermes_lcm_profile_registration").register(ctx)
    finally:
        active_home.reset(token)
    engine = ctx.engine
    assert engine is not None
    try:
        assert engine._config.context_threshold == 0.77
        assert engine._config.summary_timeout_ms == 77_000
        assert engine._store.db_path == profile_b / "lcm.db"
    finally:
        engine.shutdown()


def test_constructor_reconciles_config_home_with_storage_home(tmp_path, monkeypatch):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    _write_profile_config(profile_a, threshold=0.27, timeout=27)
    _write_profile_config(profile_b, threshold=0.73, timeout=73)
    monkeypatch.delenv("LCM_CONTEXT_THRESHOLD", raising=False)

    config_a = LCMConfig.from_env(hermes_home=str(profile_a))
    engine = LCMEngine(config=config_a, hermes_home=str(profile_b))
    try:
        assert engine._config.context_threshold == 0.73
        assert engine._config.summary_timeout_ms == 73_000
        assert engine._config.config_hermes_home == str(profile_b)
        assert engine._store.db_path == profile_b / "lcm.db"
    finally:
        engine.shutdown()


def test_distinct_context_local_profiles_can_rebind_concurrently(tmp_path, monkeypatch):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    _write_profile_config(profile_a, threshold=0.37, timeout=37)
    _write_profile_config(profile_b, threshold=0.71, timeout=71)

    active_home = ContextVar("active_home", default="")
    core = ModuleType("hermes_constants")
    setattr(core, "get_hermes_home", lambda: active_home.get())
    monkeypatch.setitem(sys.modules, "hermes_constants", core)
    monkeypatch.delenv("LCM_CONTEXT_THRESHOLD", raising=False)
    prototype = LCMEngine(
        config=LCMConfig.from_env(hermes_home=str(profile_a)),
        hermes_home=str(profile_a),
    )
    clone_a = prototype.clone_for_agent()
    clone_b = prototype.clone_for_agent()
    barrier = threading.Barrier(2)
    seen: dict[str, tuple[float, Path]] = {}

    def bind(name: str, engine: LCMEngine, home: Path) -> None:
        token = active_home.set(str(home))
        try:
            barrier.wait(timeout=5)
            engine.on_session_start(name)
            seen[name] = (engine._config.context_threshold, engine._store.db_path)
        finally:
            active_home.reset(token)

    threads = [
        threading.Thread(target=bind, args=("session-a", clone_a, profile_a)),
        threading.Thread(target=bind, args=("session-b", clone_b, profile_b)),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert all(not thread.is_alive() for thread in threads)
        assert seen == {
            "session-a": (0.37, profile_a / "lcm.db"),
            "session-b": (0.71, profile_b / "lcm.db"),
        }
    finally:
        clone_a.shutdown()
        clone_b.shutdown()
        prototype.shutdown()


def test_cloned_engine_rebinds_profile_config_storage_and_override_precedence(
    tmp_path, monkeypatch
):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    _write_profile_config(profile_a, threshold=0.31, timeout=31)
    _write_profile_config(profile_b, threshold=0.67, timeout=67)

    monkeypatch.delenv("LCM_CONTEXT_THRESHOLD", raising=False)
    prototype = LCMEngine(
        config=LCMConfig.from_env(hermes_home=str(profile_a)),
        hermes_home=str(profile_a),
    )
    clone_a = prototype.clone_for_agent()
    clone_b = prototype.clone_for_agent()
    try:
        clone_a.on_session_start("session-a", hermes_home=str(profile_a))
        clone_b.on_session_start("session-b", hermes_home=str(profile_b))

        assert clone_a._config.context_threshold == 0.31
        assert clone_b._config.context_threshold == 0.67
        assert clone_a._config.summary_timeout_ms == 31_000
        assert clone_b._config.summary_timeout_ms == 67_000
        assert clone_a._store.db_path == profile_a / "lcm.db"
        assert clone_b._store.db_path == profile_b / "lcm.db"
        assert clone_a._store._hermes_home == str(profile_a)
        assert clone_b._store._hermes_home == str(profile_b)
        assert clone_a._config is not clone_b._config

        monkeypatch.setenv("LCM_CONTEXT_THRESHOLD", "0.91")
        clone_b.on_session_start("session-a-override", hermes_home=str(profile_a))
        clone_b.on_session_start("session-b-override", hermes_home=str(profile_b))
        assert clone_b._config.context_threshold == 0.91
        assert clone_b._config.config_sources["context_threshold"] == "env:LCM_CONTEXT_THRESHOLD"
        assert clone_a._config.context_threshold == 0.31
    finally:
        clone_a.shutdown()
        clone_b.shutdown()
        prototype.shutdown()
