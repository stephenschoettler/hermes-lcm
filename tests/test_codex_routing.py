"""Regression tests for Codex OAuth route-cap matching."""

import copy
import logging
import sys
from types import ModuleType

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

from hermes_lcm.codex_routing import _codex_oauth_context_cap


@pytest.fixture(autouse=True)
def model_metadata(monkeypatch):
    # Existing alias tests exercise LCM's fallback, not a developer's live host.
    module = ModuleType("agent.model_metadata")
    monkeypatch.setitem(sys.modules, "agent.model_metadata", module)
    return module


EXACT_CODEX_900K_MODELS = (
    "gpt-5.6-terra-900k",
    "gpt-5.6-sol-900k",
    "gpt-5.6-luna-900k",
)


@pytest.mark.parametrize("model", EXACT_CODEX_900K_MODELS)
def test_exact_codex_900k_routes_receive_proven_cap(model):
    assert _codex_oauth_context_cap(model, "openai-codex") == 900_000


def test_codex_900k_route_matches_normalized_bare_slug():
    assert (
        _codex_oauth_context_cap(
            "  openai/GPT-5.6-SOL-900K  ",
            "  OPENAI-CODEX  ",
        )
        == 900_000
    )


@pytest.mark.parametrize("provider", [None, "openai", "openai-codex-proxy"])
def test_codex_900k_routes_require_exact_provider(provider):
    assert _codex_oauth_context_cap("gpt-5.6-sol-900k", provider) is None


@pytest.mark.parametrize(
    ("model", "expected_cap"),
    [
        ("gpt-5.6", 372_000),
        ("gpt-5.6-preview", 372_000),
        ("gpt-5.5", 272_000),
        ("gpt-5.4", 272_000),
        ("gpt-5.3-codex-spark", 128_000),
    ],
)
def test_existing_codex_route_caps_are_preserved(model, expected_cap):
    assert _codex_oauth_context_cap(model, "openai-codex") == expected_cap


@pytest.mark.parametrize(
    ("model", "expected_cap"),
    [
        ("gpt-5.5-900k", 272_000),
        ("gpt-5.6-terra-900k-pro", 372_000),
        ("fake-gpt-5.6-sol-900k", 372_000),
        ("gpt-5.6-luna-900k.fake", 372_000),
        ("gpt-5.6-900k", 372_000),
        ("gpt-5.7-terra-900k", 272_000),
    ],
)
def test_900k_suffix_and_malformed_aliases_do_not_gain_900k_cap(
    model,
    expected_cap,
):
    assert _codex_oauth_context_cap(model, "openai-codex") == expected_cap


@pytest.mark.parametrize("resolved", [128_000, 272_000, 900_000, 1_000_000])
@pytest.mark.parametrize("model", ["gpt-5.6-luna", "openai/gpt-5.6-sol-900k"])
def test_host_resolver_result_precedes_local_tables(model_metadata, model, resolved):
    calls = []

    def resolve(model, **kwargs):
        calls.append((model, kwargs))
        return resolved

    model_metadata.get_model_context_length = resolve
    assert _codex_oauth_context_cap(model, "openai-codex", api_key="test-token") == resolved
    assert calls == [(model, {"api_key": "test-token", "provider": "openai-codex"})]


def test_normalized_provider_selects_codex_host_resolver(model_metadata):
    calls = []

    def resolve(model, *, provider, api_key):
        calls.append((model, provider))
        return 800_000

    model_metadata.get_model_context_length = resolve
    assert _codex_oauth_context_cap(
        "  openai/GPT-5.6-SOL-900K  ", "  OPENAI-CODEX  "
    ) == 800_000
    assert calls == [("  openai/GPT-5.6-SOL-900K  ", "openai-codex")]


@pytest.mark.parametrize("value", [None, 0, -1, True, False, "900000", 900000.0])
@pytest.mark.parametrize("model, expected", [("gpt-5.6-sol", 372_000), ("gpt-5.6-sol-900k", 900_000)])
def test_invalid_host_results_keep_local_fallback(model_metadata, value, model, expected):
    model_metadata.get_model_context_length = lambda *args, **kwargs: value
    assert _codex_oauth_context_cap(model, "openai-codex") == expected


@pytest.mark.parametrize("failure", ["missing_module", "old_signature", "exception"])
def test_unavailable_resolver_keeps_exact_alias_fallback(
    monkeypatch, model_metadata, caplog, failure
):
    token = "synthetic-secret-must-not-be-logged"
    if failure == "missing_module":
        monkeypatch.setitem(sys.modules, "agent.model_metadata", None)
    elif failure == "old_signature":
        model_metadata.get_model_context_length = lambda model: 1_000_000
    else:
        def fail(*args, **kwargs):
            raise RuntimeError(token)
        model_metadata.get_model_context_length = fail
    with caplog.at_level(logging.DEBUG, logger="hermes_lcm.codex_routing"):
        assert _codex_oauth_context_cap("gpt-5.6-sol", "openai-codex", api_key=token) == 372_000
        for model in EXACT_CODEX_900K_MODELS:
            assert _codex_oauth_context_cap(model, "openai-codex", api_key=token) == 900_000
        assert _codex_oauth_context_cap("unknown-model", "openai-codex", api_key=token) is None
    assert token not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize("model, provider", [(None, "openai-codex"), ("", "openai-codex"), ("vendor/", "openai-codex"), ("gpt-5.6-sol-900k", "openai")])
def test_inapplicable_routes_do_not_call_resolver(model_metadata, model, provider):
    calls = []
    model_metadata.get_model_context_length = lambda *args, **kwargs: calls.append(args)
    assert _codex_oauth_context_cap(model, provider) is None
    assert calls == []


@pytest.fixture
def routing_engine(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "routing.db")))
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.mark.parametrize("raw, resolved", [(1_000_000, 900_000), (1_000_000, 272_000), (128_000, 900_000), (900_000, 900_000)])
def test_engine_preserves_lower_host_context(model_metadata, routing_engine, raw, resolved):
    calls = []

    def resolve(model, **kwargs):
        calls.append((model, kwargs))
        return resolved

    model_metadata.get_model_context_length = resolve
    routing_engine.update_model(
        model="gpt-5.6-luna", context_length=raw, provider="openai-codex",
        api_key="test-token", base_url="https://custom.invalid/v1",
    )
    assert routing_engine.raw_context_length == raw
    assert routing_engine.context_length == min(raw, resolved)
    assert routing_engine.effective_context_length_cap == (resolved if raw > resolved else None)
    assert routing_engine.effective_context_length_reason == ("codex_oauth_context_cap" if raw > resolved else "")
    assert calls == [("gpt-5.6-luna", {"api_key": "test-token", "provider": "openai-codex"})]
    clone = copy.deepcopy(routing_engine)
    try:
        assert clone.context_length == min(raw, resolved)
        assert calls[-1] == calls[0]
        assert len(calls) == 2
    finally:
        clone.shutdown()


@pytest.mark.parametrize("credential", ["session-token", "", None])
def test_session_start_uses_incoming_credentials_before_context_resolution(
    model_metadata, routing_engine, credential
):
    calls = []

    def resolve(model, **kwargs):
        calls.append((model, kwargs))
        return 272_000 if kwargs["api_key"] == "session-token" else 900_000

    model_metadata.get_model_context_length = resolve
    routing_engine.api_key = "stale-token"
    routing_engine.on_session_start(
        "session-route", model="gpt-5.6-sol-900k", provider="openai-codex",
        api_key=credential, context_length=1_000_000,
    )
    assert calls == [("gpt-5.6-sol-900k", {"provider": "openai-codex", "api_key": credential or ""})]
    assert routing_engine.context_length == (272_000 if credential else 900_000)
    assert routing_engine.api_key == (credential or "")


def test_session_start_credential_rotation_recomputes_existing_context(model_metadata, routing_engine):
    calls = []

    def resolve(model, **kwargs):
        calls.append(kwargs["api_key"])
        return 272_000 if kwargs["api_key"] == "new-token" else 900_000

    model_metadata.get_model_context_length = resolve
    routing_engine.on_session_start(
        "rotation", model="gpt-5.6-sol-900k", provider="openai-codex",
        api_key="old-token", context_length=1_000_000,
    )
    routing_engine.on_session_start("rotation", api_key="new-token")
    assert routing_engine.context_length == 272_000
    assert calls == ["old-token", "new-token"]


def test_session_start_cleared_provider_does_not_use_old_codex_route(model_metadata, routing_engine):
    calls = []
    model_metadata.get_model_context_length = lambda *args, **kwargs: calls.append(args)
    routing_engine.provider = "openai-codex"
    routing_engine.on_session_start(
        "cleared-provider", model="gpt-5.6-sol", provider="", context_length=1_000_000,
    )
    assert routing_engine.context_length == 1_000_000
    assert calls == []


def test_session_start_without_credential_keeps_active_credential(model_metadata, routing_engine):
    calls = []

    def resolve(model, **kwargs):
        calls.append(kwargs["api_key"])
        return 272_000

    model_metadata.get_model_context_length = resolve
    routing_engine.api_key = "active-token"
    routing_engine.on_session_start(
        "keep-credential", model="gpt-5.6-sol", provider="openai-codex", context_length=1_000_000,
    )
    assert calls == ["active-token"]
    assert routing_engine.context_length == 272_000


def test_stale_session_credentials_do_not_replace_update_model_route(model_metadata, routing_engine):
    calls = []

    def resolve(model, **kwargs):
        calls.append(kwargs["api_key"])
        return 272_000

    model_metadata.get_model_context_length = resolve
    routing_engine.update_model(
        model="gpt-5.6-sol", provider="openai-codex", api_key="active-token", context_length=1_000_000,
    )
    routing_engine.on_session_start(
        "stale-credential", model="gpt-5.6-sol", provider="openai-codex",
        api_key="stale-token", context_length=1_000_000,
    )
    assert calls == ["active-token"]
    assert routing_engine.api_key == "active-token"
    assert routing_engine.context_length == 272_000
