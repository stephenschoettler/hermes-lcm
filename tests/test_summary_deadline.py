"""Every model/level in one escalation spends the same finite timeout."""

from hermes_lcm import escalation


def test_fallbacks_and_l2_only_receive_remaining_budget(monkeypatch):
    clock = [100.0]
    calls = []
    monkeypatch.setattr(escalation.time, "monotonic", lambda: clock[0])

    def summary(prompt, max_tokens, model="", timeout=None):
        calls.append((model, timeout, max_tokens))
        clock[0] += 3.0
        return None

    monkeypatch.setattr(escalation, "_call_llm_for_summary", summary)
    result, level = escalation.summarize_with_escalation(
        "source facts " * 1000, source_tokens=3000, token_budget=100,
        model="primary", fallback_models=["backup"], timeout=10,
    )
    assert [(model, timeout) for model, timeout, _ in calls] == [
        ("primary", 10), ("backup", 7), ("primary", 4), ("backup", 1),
    ]
    assert result and level == 3


def test_spent_l1_budget_never_starts_another_provider_call(monkeypatch):
    clock = [100.0]
    calls = []
    monkeypatch.setattr(escalation.time, "monotonic", lambda: clock[0])

    def summary(prompt, max_tokens, model="", timeout=None):
        calls.append(model)
        clock[0] += 10.0
        return "x" * 10000  # An overlong L1 is rejected even on a successful HTTP call.

    monkeypatch.setattr(escalation, "_call_llm_for_summary", summary)
    result, level = escalation.summarize_with_escalation(
        "source facts " * 1000, source_tokens=3000, token_budget=100,
        model="primary", fallback_models=["backup"], timeout=10,
    )
    assert calls == ["primary"]
    assert result and level == 3
