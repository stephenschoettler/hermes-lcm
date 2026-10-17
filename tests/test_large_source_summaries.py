"""Reject unusually thin summaries without bypassing escalation."""

import pytest

from hermes_lcm import escalation
from hermes_lcm.config import LCMConfig
from hermes_lcm.tokens import count_tokens


def summarize(monkeypatch, responses, **kwargs):
    calls = []
    def fake_summary(prompt, max_tokens, model="", timeout=None):
        calls.append(model)
        return responses[model]
    monkeypatch.setattr(escalation, "_call_llm_for_summary", fake_summary)
    result = escalation.summarize_with_escalation(
        "source detail " * 1000, source_tokens=100_000, token_budget=4000,
        model="primary", fallback_models=["backup"], **kwargs,
    )
    return result, calls


def test_thin_large_summary_uses_next_model(monkeypatch):
    detail = "summary detail " * 300
    (summary, level), calls = summarize(monkeypatch, {"primary": "tiny", "backup": detail})
    assert summary == detail
    assert level == 1
    assert calls == ["primary", "backup"]


def test_all_thin_summaries_reach_deterministic_fallback(monkeypatch):
    (summary, level), calls = summarize(monkeypatch, {"primary": "tiny", "backup": "tiny"})
    assert level == 3
    assert summary != "tiny"
    assert count_tokens(summary) <= 512
    assert calls == ["primary", "backup", "primary", "backup"]


@pytest.mark.parametrize("thresholds", [
    {"large_source_summary_min_source_tokens": 0},
    {"large_source_summary_min_result_tokens": 0},
    {"large_source_summary_min_source_tokens": 100_001},
])
def test_guard_can_be_disabled_or_below_source_threshold(monkeypatch, thresholds):
    (summary, level), calls = summarize(monkeypatch, {"primary": "tiny", "backup": "unused"}, **thresholds)
    assert (summary, level) == ("tiny", 1)
    assert calls == ["primary"]


def test_guard_is_configurable_from_environment(monkeypatch):
    monkeypatch.setenv("LCM_LARGE_SOURCE_SUMMARY_MIN_SOURCE_TOKENS", "200000")
    monkeypatch.setenv("LCM_LARGE_SOURCE_SUMMARY_MIN_RESULT_TOKENS", "256")
    config = LCMConfig.from_env()
    assert config.large_source_summary_min_source_tokens == 200_000
    assert config.large_source_summary_min_result_tokens == 256
