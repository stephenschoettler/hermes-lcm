"""Tests for host capability detection before registering lcm_* tools."""

import importlib.util
import sys
import threading
from pathlib import Path

import pytest


EXPECTED_LCM_TOOLS = {
    "lcm_grep",
    "lcm_recall",
    "lcm_query_state",
    "lcm_compute",
    "lcm_compile_evidence",
    "lcm_evidence_pack",
    "lcm_retrieve",
    "lcm_recent",
    "lcm_load_session",
    "lcm_describe",
    "lcm_expand",
    "lcm_expand_query",
    "lcm_status",
    "lcm_inspect",
    "lcm_doctor",
}


def _load_plugin_module(name: str):
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        name, str(repo_root / "__init__.py"), submodule_search_locations=[str(repo_root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestHostCapabilityDetection:
    """Verify explicit host capability detection for registered lcm_* tools."""

    def test_returns_false_when_ctx_lacks_capability(self):
        module = _load_plugin_module("hermes_lcm_cap_no_attr")

        class _Ctx:
            pass

        assert module._host_forwards_registered_tool_messages(_Ctx()) is False

    def test_returns_false_when_capability_is_false(self):
        module = _load_plugin_module("hermes_lcm_cap_false")

        class _Ctx:
            context_engine_tool_handlers_receive_messages = False

        assert module._host_forwards_registered_tool_messages(_Ctx()) is False

    def test_returns_true_when_capability_is_true(self):
        module = _load_plugin_module("hermes_lcm_cap_true")

        class _Ctx:
            context_engine_tool_handlers_receive_messages = True

        assert module._host_forwards_registered_tool_messages(_Ctx()) is True

    def test_supports_callable_capability(self):
        module = _load_plugin_module("hermes_lcm_cap_callable")

        class _Ctx:
            def context_engine_tool_handlers_receive_messages(self):
                return True

        assert module._host_forwards_registered_tool_messages(_Ctx()) is True

    def test_callable_capability_failure_fails_closed(self):
        module = _load_plugin_module("hermes_lcm_cap_callable_raises")

        class _Ctx:
            def context_engine_tool_handlers_receive_messages(self):
                raise RuntimeError("host capability unavailable")

        assert module._host_forwards_registered_tool_messages(_Ctx()) is False


class TestRegistrationGating:
    """Verify register() skips ctx.register_tool unless messages forwarding is explicit."""

    def test_skips_register_tool_without_explicit_message_forwarding(self):
        module = _load_plugin_module("hermes_lcm_gating_skip")
        registered_tools = []

        class _CtxNoForwarding:
            def __init__(self):
                self.engine = None

            def register_context_engine(self, engine):
                self.engine = engine

            def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
                registered_tools.append(name)

        ctx = _CtxNoForwarding()
        module.register(ctx)

        assert ctx.engine is not None
        assert ctx.engine.name == "lcm"
        assert registered_tools == []
        assert EXPECTED_LCM_TOOLS.issubset(
            {schema["name"] for schema in ctx.engine.get_tool_schemas()}
        )

    def test_registers_tools_when_host_explicitly_supports_message_forwarding(self):
        module = _load_plugin_module("hermes_lcm_gating_register")
        registered_tools = []

        class _CtxWithForwarding:
            context_engine_tool_handlers_receive_messages = True

            def __init__(self):
                self.engine = None

            def register_context_engine(self, engine):
                self.engine = engine

            def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
                registered_tools.append(name)

        ctx = _CtxWithForwarding()
        module.register(ctx)

        assert ctx.engine is not None
        assert set(registered_tools) == EXPECTED_LCM_TOOLS

    def test_existing_context_engine_path_still_loads_without_register_tool(self):
        module = _load_plugin_module("hermes_lcm_gating_no_register_tool")

        class _Ctx:
            def __init__(self):
                self.engine = None

            def register_context_engine(self, engine):
                self.engine = engine

        ctx = _Ctx()
        module.register(ctx)
        assert ctx.engine is not None
        assert ctx.engine.name == "lcm"

    def test_registers_engine_shutdown_with_host_unload(self, tmp_path, monkeypatch):
        module = _load_plugin_module("hermes_lcm_unload_cleanup")
        callbacks = []
        db_path = tmp_path / "lcm.db"
        monkeypatch.setenv("LCM_DATABASE_PATH", str(db_path))

        class _Ctx:
            def register_context_engine(self, engine):
                self.engine = engine

            def on_unload(self, callback):
                callbacks.append(callback)

        ctx = _Ctx()
        module.register(ctx)

        assert len(callbacks) == 1
        assert callbacks[0].__self__ is ctx.engine
        clone = ctx.engine.clone_for_agent()
        assert db_path.is_file()

        callbacks[0]()

        assert ctx.engine._store._conn is None
        assert clone._store._conn is None
        db_path.unlink()
        assert not db_path.exists()

        late_clone = None
        try:
            with pytest.raises(RuntimeError, match="unloading"):
                late_clone = ctx.engine.clone_for_agent()
        finally:
            if late_clone is not None:
                late_clone.shutdown()
        if db_path.exists():
            db_path.unlink()
        assert not db_path.exists()

    def test_unload_waits_for_clone_construction_already_in_flight(
        self, tmp_path, monkeypatch
    ):
        module = _load_plugin_module("hermes_lcm_unload_inflight_clone")
        callbacks = []
        db_path = tmp_path / "lcm.db"
        monkeypatch.setenv("LCM_DATABASE_PATH", str(db_path))

        class _Ctx:
            def register_context_engine(self, engine):
                self.engine = engine

            def on_unload(self, callback):
                callbacks.append(callback)

        ctx = _Ctx()
        module.register(ctx)
        engine_type = type(ctx.engine)
        original_init = engine_type.__init__
        clone_constructed = threading.Event()
        release_clone = threading.Event()

        def blocking_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            clone_constructed.set()
            if not release_clone.wait(timeout=3):
                raise AssertionError("test did not release clone construction")

        monkeypatch.setattr(engine_type, "__init__", blocking_init)
        clone_errors = []

        def clone_engine():
            try:
                ctx.engine.clone_for_agent()
            except BaseException as exc:  # captured for assertion in the main thread
                clone_errors.append(exc)

        clone_thread = threading.Thread(target=clone_engine)
        clone_thread.start()
        assert clone_constructed.wait(timeout=1)

        unload_done = threading.Event()
        unload_errors = []

        def unload_plugin():
            try:
                callbacks[0]()
            except BaseException as exc:  # captured for assertion in the main thread
                unload_errors.append(exc)
            finally:
                unload_done.set()

        unload_thread = threading.Thread(target=unload_plugin)
        unload_thread.start()
        unload_waited = not unload_done.wait(timeout=0.1)
        release_clone.set()
        clone_thread.join(timeout=3)
        unload_thread.join(timeout=3)

        assert unload_waited
        assert not clone_thread.is_alive()
        assert not unload_thread.is_alive()
        assert unload_errors == []
        assert len(clone_errors) == 1
        assert isinstance(clone_errors[0], RuntimeError)
        assert "unloading" in str(clone_errors[0])
        db_path.unlink()
        assert not db_path.exists()

    def test_failed_clone_initialization_closes_partial_engine(self, tmp_path, monkeypatch):
        module = _load_plugin_module("hermes_lcm_unload_failed_clone")
        callbacks = []
        db_path = tmp_path / "lcm.db"
        monkeypatch.setenv("LCM_DATABASE_PATH", str(db_path))

        class _Ctx:
            def register_context_engine(self, engine):
                self.engine = engine

            def on_unload(self, callback):
                callbacks.append(callback)

        ctx = _Ctx()
        module.register(ctx)
        engine_type = type(ctx.engine)
        engine_module = sys.modules[engine_type.__module__]

        def fail_after_storage_bind(*_args, **_kwargs):
            raise RuntimeError("synthetic clone initialization failure")

        monkeypatch.setattr(
            engine_module,
            "compile_session_patterns",
            fail_after_storage_bind,
        )
        with pytest.raises(RuntimeError, match="synthetic clone initialization failure") as excinfo:
            ctx.engine.clone_for_agent()

        partial_engine = None
        traceback = excinfo.value.__traceback__
        while traceback is not None:
            candidate = traceback.tb_frame.f_locals.get("self")
            if isinstance(candidate, engine_type) and candidate is not ctx.engine:
                partial_engine = candidate
                break
            traceback = traceback.tb_next
        assert partial_engine is not None

        callbacks[0]()
        partial_closed = partial_engine._store._conn is None
        if not partial_closed:
            for name in ("_adaptive_retrieval", "_store", "_dag", "_lifecycle", "_assertions", "_query_views"):
                resource = getattr(partial_engine, name, None)
                if resource is not None:
                    resource.close()
        if db_path.exists():
            db_path.unlink()

        assert partial_closed
        assert not db_path.exists()

    def test_unload_serializes_profile_storage_rebind(
        self, tmp_path, monkeypatch, request
    ):
        from hermes_lcm import retrieval_core
        from hermes_lcm import tools as lcm_tools
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        request.addfinalizer(retrieval_core._reset_vector_store_pool)
        request.addfinalizer(lcm_tools._open_deadline_worker_registry)
        home_a = tmp_path / "profile-a"
        home_b = tmp_path / "profile-b"
        home_a.mkdir()
        home_b.mkdir()
        engine = LCMEngine(
            config=LCMConfig(database_path=""),
            hermes_home=str(home_a),
        )
        real_bind = engine._bind_storage
        bind_started = threading.Event()
        release_bind = threading.Event()
        unload_done = threading.Event()
        rebind_errors = []
        unload_errors = []

        def blocking_bind(db_path, hermes_home):
            if str(hermes_home) == str(home_b):
                bind_started.set()
                if not release_bind.wait(timeout=3):
                    raise AssertionError("test did not release profile storage bind")
            return real_bind(db_path, hermes_home)

        def rebind_storage():
            try:
                engine._rebind_storage_for_home(str(home_b))
            except BaseException as exc:  # captured for assertion in the main thread
                rebind_errors.append(exc)

        def unload_plugin():
            try:
                engine.shutdown_all_instances()
            except BaseException as exc:  # captured for assertion in the main thread
                unload_errors.append(exc)
            finally:
                unload_done.set()

        monkeypatch.setattr(engine, "_bind_storage", blocking_bind)
        rebind_thread = threading.Thread(target=rebind_storage)
        unload_thread = threading.Thread(target=unload_plugin)
        db_paths = (home_a / "lcm.db", home_b / "lcm.db")
        try:
            rebind_thread.start()
            assert bind_started.wait(timeout=1)
            unload_thread.start()
            unload_waited = not unload_done.wait(timeout=0.1)
            release_bind.set()
            rebind_thread.join(timeout=3)
            unload_thread.join(timeout=3)

            assert unload_waited
            assert not rebind_thread.is_alive()
            assert not unload_thread.is_alive()
            assert rebind_errors == []
            assert unload_errors == []
            assert engine._store._conn is None
            with pytest.raises(RuntimeError, match="plugin is unloading"):
                engine._rebind_storage_for_home(str(home_a))
            for path in db_paths:
                if path.exists():
                    path.unlink()
                assert not path.exists()
        finally:
            release_bind.set()
            rebind_thread.join(timeout=3)
            unload_thread.join(timeout=3)
            engine._close_storage()
            for path in db_paths:
                if path.exists():
                    path.unlink()


class TestHermesAgentRegression:
    """Regression: Hermes Agent-shaped hosts must not shadow native LCM routing."""

    def test_hermes_agent_shaped_host_uses_context_engine_path(self):
        module = _load_plugin_module("hermes_lcm_hermes_agent_regression")
        registered_via_tool = []
        registered_via_engine = []

        class _HermesAgentCtx:
            def __init__(self):
                self.engine = None

            def register_context_engine(self, engine):
                self.engine = engine
                registered_via_engine.extend(
                    s["name"] for s in engine.get_tool_schemas()
                )

            def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
                registered_via_tool.append(name)

        ctx = _HermesAgentCtx()
        module.register(ctx)

        assert ctx.engine is not None
        assert registered_via_tool == []
        assert set(registered_via_engine) == EXPECTED_LCM_TOOLS

    def test_messages_forwarded_through_context_engine_path(self):
        module = _load_plugin_module("hermes_lcm_messages_forward_regression")

        class _HermesAgentCtx:
            def __init__(self):
                self.engine = None

            def register_context_engine(self, engine):
                self.engine = engine

            def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
                raise AssertionError("Hermes Agent-shaped host must not register lcm_* tools")

        ctx = _HermesAgentCtx()
        module.register(ctx)
        assert ctx.engine is not None

        test_messages = [{"role": "user", "content": "test context"}]
        result = ctx.engine.handle_tool_call(
            "lcm_status", {}, messages=test_messages
        )

        assert isinstance(result, str)
        assert len(result) > 0
