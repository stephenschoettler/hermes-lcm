"""Regression tests for Codex OAuth route-cap matching."""

import sys
from types import ModuleType

import pytest

from hermes_lcm.codex_routing import _codex_oauth_context_cap


def _install_host_variant_predicate(monkeypatch, predicate):
    host_metadata = ModuleType("agent.model_metadata")
    host_metadata.is_codex_context_variant = predicate
    monkeypatch.setitem(sys.modules, "agent.model_metadata", host_metadata)


EXACT_CODEX_900K_MODELS = (
    "gpt-5.6-terra-900k",
    "gpt-5.6-sol-900k",
    "gpt-5.6-luna-900k",
)


@pytest.mark.parametrize("model", EXACT_CODEX_900K_MODELS)
def test_exact_codex_900k_routes_receive_proven_cap(model, monkeypatch):
    _install_host_variant_predicate(monkeypatch, lambda candidate: candidate == model)

    assert _codex_oauth_context_cap(model, "openai-codex") == 900_000


def test_codex_900k_route_matches_normalized_bare_slug(monkeypatch):
    model = "  openai/GPT-5.6-SOL-900K  "
    _install_host_variant_predicate(monkeypatch, lambda candidate: candidate == model)

    assert _codex_oauth_context_cap(model, "  OPENAI-CODEX  ") == 900_000


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
    monkeypatch,
):
    _install_host_variant_predicate(monkeypatch, lambda _candidate: False)

    assert _codex_oauth_context_cap(model, "openai-codex") == expected_cap


def test_base_gpt56_route_keeps_conservative_cap_without_host_lookup(monkeypatch):
    calls = []
    _install_host_variant_predicate(monkeypatch, calls.append)

    assert _codex_oauth_context_cap("gpt-5.6-sol", "openai-codex") == 372_000
    assert calls == []


def test_unrecognised_900k_suffix_does_not_bypass_cap(monkeypatch):
    _install_host_variant_predicate(monkeypatch, lambda _model: False)

    assert _codex_oauth_context_cap("gpt-5.5-900k", "openai-codex") == 272_000


def test_missing_host_variant_predicate_fails_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.model_metadata", None)

    assert (
        _codex_oauth_context_cap("gpt-5.6-sol-900k", "openai-codex")
        == 372_000
    )


def test_broken_host_variant_predicate_fails_closed(monkeypatch):
    def fail(_model):
        raise RuntimeError("host predicate failed")

    _install_host_variant_predicate(monkeypatch, fail)

    assert (
        _codex_oauth_context_cap("gpt-5.6-sol-900k", "openai-codex")
        == 372_000
    )


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.6-sol-900k",
        "openai/GPT-5.6-TERRA-900K",
        "gpt-5.6-luna-2026-08-16-900k",
        "gpt-5.4-900k",
        "gpt-daybreak-blue-latest-900k",
    ],
)
def test_host_recognised_900k_variants_use_named_cap(model, monkeypatch):
    recognised = {
        "gpt-5.6-sol-900k",
        "openai/GPT-5.6-TERRA-900K",
        "gpt-5.6-luna-2026-08-16-900k",
        "gpt-5.4-900k",
        "gpt-daybreak-blue-latest-900k",
    }
    _install_host_variant_predicate(monkeypatch, lambda candidate: candidate in recognised)

    assert _codex_oauth_context_cap(model, "openai-codex") == 900_000
