"""Tests for Hermes-backed Codex OAuth context-window resolution."""

import sys
from types import ModuleType

import hermes_lcm.codex_routing as codex_routing
import hermes_lcm.engine as lcm_engine

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _install_model_metadata_resolver(monkeypatch, resolver):
    import agent

    metadata_module = ModuleType("agent.model_metadata")
    setattr(metadata_module, "get_model_context_length", resolver)
    monkeypatch.setitem(sys.modules, "agent.model_metadata", metadata_module)
    monkeypatch.setattr(agent, "model_metadata", metadata_module, raising=False)


def test_codex_oauth_context_cap_uses_hermes_provider_resolver(monkeypatch):
    calls = {}

    def resolve_context_length(model, *, base_url="", api_key="", provider="", **kwargs):
        calls.update(
            {
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "provider": provider,
                "kwargs": kwargs,
            }
        )
        return 900_000

    _install_model_metadata_resolver(monkeypatch, resolve_context_length)

    assert codex_routing._codex_oauth_context_cap(
        "gpt-5.6-luna",
        "openai-codex",
        api_key="oauth-token",
    ) == 900_000
    assert calls == {
        "model": "gpt-5.6-luna",
        "base_url": "",
        "api_key": "oauth-token",
        "provider": "openai-codex",
        "kwargs": {},
    }


def test_codex_oauth_context_cap_uses_lower_live_value_over_static_fallback(monkeypatch):
    def resolve_context_length(*args, **kwargs):
        return 272_000

    _install_model_metadata_resolver(monkeypatch, resolve_context_length)

    assert codex_routing._codex_oauth_context_cap(
        "gpt-5.6-luna",
        "openai-codex",
    ) == 272_000


def test_codex_oauth_context_cap_keeps_existing_fallback_on_resolver_failure(
    monkeypatch,
):
    def fail_resolver(*args, **kwargs):
        raise RuntimeError("provider metadata unavailable")

    _install_model_metadata_resolver(monkeypatch, fail_resolver)

    assert codex_routing._codex_oauth_context_cap(
        "gpt-5.6-luna",
        "openai-codex",
    ) == 372_000


def test_engine_uses_resolved_codex_cap_and_forwards_route_credentials(
    tmp_path,
    monkeypatch,
):
    calls = {}

    def resolve_cap(model, provider, *, api_key=""):
        calls.update(
            {
                "model": model,
                "provider": provider,
                "api_key": api_key,
            }
        )
        return 900_000

    monkeypatch.setattr(lcm_engine, "_codex_oauth_context_cap", resolve_cap)
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "codex-routing.db")),
    )
    try:
        engine.update_model(
            model="gpt-5.6-luna",
            context_length=1_000_000,
            api_key="oauth-token",
            provider="openai-codex",
        )

        assert engine.context_length == 900_000
        assert engine.effective_context_length_cap == 900_000
        assert engine.effective_context_length_reason == "codex_oauth_context_cap"
        assert calls == {
            "model": "gpt-5.6-luna",
            "provider": "openai-codex",
            "api_key": "oauth-token",
        }
    finally:
        engine.shutdown()
