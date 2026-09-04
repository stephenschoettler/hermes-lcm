"""Regression tests for Codex OAuth route-cap matching."""

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
