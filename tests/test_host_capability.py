"""Tests for host capability detection before registering lcm_* tools."""

import importlib.util
import sys
from pathlib import Path


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
    """Verify _host_supports_message_forwarding() correctly inspects ctx."""

    def test_returns_false_when_ctx_lacks_register_tool(self):
        """Host without register_tool cannot support message-forwarding."""
        module = _load_plugin_module("hermes_lcm_cap_no_tool")

        class _Ctx:
            pass

        assert module._host_supports_message_forwarding(_Ctx()) is False

    def test_returns_false_when_register_tool_does_not_accept_messages(self):
        """Host whose register_tool doesn't forward kwargs cannot support messages."""
        module = _load_plugin_module("hermes_lcm_cap_no_messages")

        class _Ctx:
            def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
                pass  # No **kwargs, no messages param

        assert module._host_supports_message_forwarding(_Ctx()) is False

    def test_returns_true_when_register_tool_accepts_kwargs(self):
        """Host whose register_tool accepts **kwargs may forward messages."""
        module = _load_plugin_module("hermes_lcm_cap_kwargs")

        class _Ctx:
            def register_tool(self, name, toolset, schema, handler, **kwargs):
                pass

        assert module._host_supports_message_forwarding(_Ctx()) is True

    def test_returns_true_when_register_tool_has_messages_param(self):
        """Host that explicitly declares messages param supports forwarding."""
        module = _load_plugin_module("hermes_lcm_cap_messages_param")

        class _Ctx:
            def register_tool(self, name, toolset, schema, handler, messages=None, **kwargs):
                pass

        assert module._host_supports_message_forwarding(_Ctx()) is True


class TestRegistrationGating:
    """Verify register() skips ctx.register_tool when host lacks message-forwarding."""

    def test_skips_register_tool_when_host_lacks_message_forwarding(self):
        """Plugin should NOT call ctx.register_tool when host cannot forward messages."""
        module = _load_plugin_module("hermes_lcm_gating_skip")

        registered_tools = []

        class _CtxNoForwarding:
            """Host with register_tool but no **kwargs — cannot forward messages."""
            def __init__(self):
                self.engine = None
            def register_context_engine(self, engine):
                self.engine = engine
            def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
                registered_tools.append(name)

        ctx = _CtxNoForwarding()
        module.register(ctx)

        # Engine should still be registered (context engine path)
        assert ctx.engine is not None
        assert ctx.engine.name == "lcm"

        # But NO lcm_* tools should be registered via ctx.register_tool
        assert len(registered_tools) == 0, (
            f"Expected no tools registered, but got: {registered_tools}"
        )

    def test_registers_tools_when_host_supports_message_forwarding(self):
        """Plugin SHOULD call ctx.register_tool when host can forward messages."""
        module = _load_plugin_module("hermes_lcm_gating_register")

        registered_tools = []

        class _CtxWithForwarding:
            """Host with register_tool accepting **kwargs — can forward messages."""
            def __init__(self):
                self.engine = None
            def register_context_engine(self, engine):
                self.engine = engine
            def register_tool(self, name, toolset, schema, handler, **kwargs):
                registered_tools.append(name)

        ctx = _CtxWithForwarding()
        module.register(ctx)

        # Engine should be registered
        assert ctx.engine is not None

        # All lcm_* tools should be registered
        expected_tools = {
            "lcm_grep", "lcm_load_session", "lcm_describe",
            "lcm_expand", "lcm_expand_query", "lcm_status", "lcm_doctor"
        }
        assert set(registered_tools) == expected_tools

    def test_existing_tests_still_pass_with_forwarding_capable_host(self):
        """Regression: existing behavior preserved when host supports forwarding."""
        module = _load_plugin_module("hermes_lcm_gating_existing")

        class _Ctx:
            def __init__(self):
                self.engine = None
            def register_context_engine(self, engine):
                self.engine = engine
            def register_tool(self, name, toolset, schema, handler, **kwargs):
                pass

        ctx = _Ctx()
        module.register(ctx)
        assert ctx.engine is not None
        assert ctx.engine.name == "lcm"
