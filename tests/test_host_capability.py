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
