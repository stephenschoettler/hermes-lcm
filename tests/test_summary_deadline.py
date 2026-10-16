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
        return "oversized summary " * 10000  # Larger than the source on either tokenizer.

    monkeypatch.setattr(escalation, "_call_llm_for_summary", summary)
    monkeypatch.setattr(
        escalation, "_build_l2_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("L2 built after expiry")),
    )
    result, level = escalation.summarize_with_escalation(
        "source facts " * 1000, source_tokens=3000, token_budget=100,
        model="primary", fallback_models=["backup"], timeout=10,
    )
    assert calls == ["primary"]
    assert result and level == 3


def test_route_guards_cannot_dispatch_with_stale_remaining_budget(monkeypatch):
    clock = [100.0]
    calls = []
    monkeypatch.setattr(escalation.time, "monotonic", lambda: clock[0])
    breaker = escalation.SummaryCircuitBreaker()
    spend = escalation.SummarySpendGuard()
    reserve = spend.try_record_call

    def contended_reservation():
        clock[0] += 10.0
        return reserve()

    monkeypatch.setattr(spend, "try_record_call", contended_reservation)
    monkeypatch.setattr(
        escalation, "_invoke_summary_llm",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    assert escalation._invoke_summary_llm_chain(
        "source", 100, model="primary", deadline=110.0,
        circuit_breaker=breaker, spend_guard=spend,
    ) is None
    assert calls == []
    assert breaker._failures == {}


def test_deadline_starved_fallback_remains_healthy_next_operation(monkeypatch):
    clock = [100.0]
    calls = []
    monkeypatch.setattr(escalation.time, "monotonic", lambda: clock[0])
    breaker = escalation.SummaryCircuitBreaker(failure_threshold=1)

    def summary(prompt, max_tokens, model="", timeout=None):
        calls.append((model, timeout))
        if model == "primary":
            clock[0] += 9.0
            return None
        if timeout < 2:
            clock[0] += timeout
            return None
        return "healthy backup summary"

    monkeypatch.setattr(escalation, "_call_llm_for_summary", summary)
    first = escalation.summarize_with_escalation(
        "source facts " * 1000, source_tokens=3000, token_budget=100,
        model="primary", fallback_models=["backup"], timeout=10, circuit_breaker=breaker,
    )
    assert first[1] == 3
    assert not breaker.allows("primary")
    assert breaker.allows("backup")
    second = escalation.summarize_with_escalation(
        "source facts " * 1000, source_tokens=3000, token_budget=100,
        model="primary", fallback_models=["backup"], timeout=10, circuit_breaker=breaker,
    )
    assert second == ("healthy backup summary", 1)
    assert calls == [("primary", 10), ("backup", 1), ("backup", 10)]
