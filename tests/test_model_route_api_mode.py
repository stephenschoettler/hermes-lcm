"""Explicit model routes carry the selected provider's wire protocol."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from hermes_lcm.escalation import _call_llm_for_summary
from hermes_lcm.model_routing import apply_lcm_model_route, parse_lcm_model_override


@pytest.fixture
def provider_modules(monkeypatch):
    package = ModuleType("hermes_cli")
    package.__path__ = []
    runtime = ModuleType("hermes_cli.runtime_provider")
    entries = {}
    runtime._get_named_custom_provider = entries.get
    auth = ModuleType("hermes_cli.auth")
    auth.PROVIDER_REGISTRY = {}
    monkeypatch.setitem(sys.modules, "hermes_cli", package)
    monkeypatch.setitem(sys.modules, "hermes_cli.runtime_provider", runtime)
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", auth)
    return entries


@pytest.mark.parametrize("prefix", ["example-provider", "custom:example-provider"])
def test_named_route_carries_api_mode(provider_modules, prefix):
    provider_modules["example-provider"] = {"base_url": "https://example.invalid", "api_mode": " anthropic_messages "}
    route = parse_lcm_model_override(f"{prefix}/model-a")
    assert (route.provider, route.model, route.api_mode) == ("example-provider", "model-a", "anthropic_messages")


def test_explicit_route_overrides_inherited_api_mode(provider_modules):
    provider_modules["example-provider"] = {"api_mode": "anthropic_messages"}
    kwargs = {"api_mode": "chat_completions"}
    apply_lcm_model_route(kwargs, "example-provider/model-a")
    assert kwargs == {"provider": "example-provider", "model": "model-a", "api_mode": "anthropic_messages"}


@pytest.mark.parametrize("entry", [{}, {"api_mode": None}, {"api_mode": 123}, {"api_mode": "  "}])
def test_absent_api_mode_preserves_existing_call_behavior(provider_modules, entry):
    provider_modules["example-provider"] = {"model": "model-a", **entry}
    kwargs = {"api_mode": "chat_completions"}
    apply_lcm_model_route(kwargs, "example-provider/model-a")
    assert kwargs["api_mode"] == "chat_completions"


def test_summary_request_receives_selected_mode(monkeypatch, provider_modules):
    provider_modules["example-provider"] = {"api_mode": "anthropic_messages"}
    auxiliary = ModuleType("agent.auxiliary_client")
    calls = []
    def call_llm(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))])
    auxiliary.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", auxiliary)
    assert _call_llm_for_summary("summarize", 200, model="custom:example-provider/model-a") == "summary"
    assert calls[0]["api_mode"] == "anthropic_messages"
    assert calls[0]["provider"] == "example-provider"
    assert calls[0]["model"] == "model-a"


def test_minimax_cn_builtin_uses_messages_api(provider_modules):
    sys.modules["hermes_cli.auth"].PROVIDER_REGISTRY["minimax-cn"] = {}
    kwargs = {}
    apply_lcm_model_route(kwargs, "minimax-cn/MiniMax-M3")
    assert kwargs == {"provider": "minimax-cn", "model": "MiniMax-M3", "api_mode": "anthropic_messages"}
