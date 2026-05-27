"""Tests for host capability detection before registering lcm_* tools."""

import importlib.util
import sys
from pathlib import Path


EXPECTED_LCM_TOOLS = {
    "lcm_grep",
    "lcm_load_session",
    "lcm_describe",
    "lcm_expand",
    "lcm_expand_query",
    "lcm_status",
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
