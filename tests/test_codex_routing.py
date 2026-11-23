"""Regression tests for Codex OAuth route-cap matching."""

import sys
import types

import pytest

from hermes_lcm.codex_routing import _codex_oauth_context_cap


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
        ("gpt-5.6", 272_000),
        ("gpt-5.6-preview", 272_000),
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
        ("gpt-5.6-terra-900k-pro", 272_000),
        ("fake-gpt-5.6-sol-900k", 272_000),
        ("gpt-5.6-luna-900k.fake", 272_000),
        ("gpt-5.6-900k", 272_000),
        ("gpt-5.7-terra-900k", 272_000),
    ],
)
def test_900k_suffix_and_malformed_aliases_do_not_gain_900k_cap(
    model,
    expected_cap,
):
    assert _codex_oauth_context_cap(model, "openai-codex") == expected_cap


def _install_host_helper(monkeypatch, variant_predicate):
    """Simulate a host that exposes ``agent.model_metadata.is_codex_context_variant``."""
    mod = types.ModuleType("agent.model_metadata")
    mod.is_codex_context_variant = variant_predicate
    monkeypatch.setitem(sys.modules, "agent.model_metadata", mod)


def test_host_declared_variant_defers_to_host_context_length(monkeypatch):
    _install_host_helper(monkeypatch, lambda m: m.endswith("-900k"))
    # A host that has resolved the variant itself must not be re-capped.
    assert _codex_oauth_context_cap("gpt-5.6-sol-900k", "openai-codex") is None
    assert _codex_oauth_context_cap("gpt-6-astra-900k", "openai-codex") is None


def test_host_helper_that_rejects_slug_keeps_table_caps(monkeypatch):
    _install_host_helper(monkeypatch, lambda m: False)
    assert _codex_oauth_context_cap("gpt-5.6-sol-900k", "openai-codex") == 900_000
    assert _codex_oauth_context_cap("gpt-5.6-sol", "openai-codex") == 272_000


def test_missing_host_helper_falls_back_to_exact_table(monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.model_metadata", None)  # import raises
    assert _codex_oauth_context_cap("gpt-5.6-sol-900k", "openai-codex") == 900_000


def test_broken_host_helper_is_fail_open(monkeypatch):
    def _boom(_m):
        raise RuntimeError("host helper exploded")

    _install_host_helper(monkeypatch, _boom)
    assert _codex_oauth_context_cap("gpt-5.6-sol-900k", "openai-codex") == 900_000
