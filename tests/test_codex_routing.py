"""Focused tests for Codex OAuth context-cap routing."""

import sys
from types import ModuleType

import pytest

from hermes_lcm.codex_routing import _codex_oauth_context_cap


def _install_host_variant_predicate(monkeypatch, predicate):
    host_metadata = ModuleType("agent.model_metadata")
    host_metadata.is_codex_context_variant = predicate
    monkeypatch.setitem(sys.modules, "agent.model_metadata", host_metadata)


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
