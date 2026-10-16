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
    spend = escalation.SummarySpendGuard(max_calls=1)
    reserve = spend.reserve_call

    def contended_reservation():
        clock[0] += 10.0
        return reserve()

    monkeypatch.setattr(spend, "reserve_call", contended_reservation)
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
    assert spend._calls == []
    assert spend._reservations == {}
    assert spend._backoff_until == 0
    assert spend.allows()


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


def test_first_route_timeout_still_opens_its_circuit(monkeypatch):
    clock = [100.0]
    calls = []
    monkeypatch.setattr(escalation.time, "monotonic", lambda: clock[0])
    breaker = escalation.SummaryCircuitBreaker(failure_threshold=1)

    def summary(prompt, max_tokens, model="", timeout=None):
        calls.append((model, timeout))
        if model == "primary":
            clock[0] += timeout
            return None
        return "healthy backup summary"

    monkeypatch.setattr(escalation, "_call_llm_for_summary", summary)
    for _ in range(2):
        escalation.summarize_with_escalation(
            "source facts " * 1000, source_tokens=3000, token_budget=100,
            model="primary", fallback_models=["backup"], timeout=10, circuit_breaker=breaker,
        )
    assert not breaker.allows("primary")
    assert calls == [("primary", 10), ("backup", 10)]


def test_pending_reservation_limits_concurrency_without_false_spend():
    spend = escalation.SummarySpendGuard(max_calls=1)
    token = spend.reserve_call()
    assert token is not None
    assert not spend.allows()
    assert spend.reserve_call() is None
    spend.release_call(token, attempted=False)
    assert spend.allows()
    token = spend.reserve_call()
    spend.release_call(token, attempted=True)
    assert not spend.allows()
    assert spend._reservations == {}
    spend.clear()
    token = spend.reserve_call()
    spend.clear()  # A forced operation starts a new budget, including pending calls.
    spend.release_call(token, attempted=True)
    assert spend.allows()


def test_expired_l1_uses_the_same_l3_cap_as_normal_exhaustion(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(escalation.time, "monotonic", lambda: clock[0])

    def summary(*args, **kwargs):
        clock[0] += 2.0
        return None

    monkeypatch.setattr(escalation, "_call_llm_for_summary", summary)
    request = dict(text="source facts " * 1000, source_tokens=3000,
                   token_budget=100, l3_truncate_tokens=512)
    expired = escalation.summarize_with_escalation(**request, timeout=1)
    normal = escalation.summarize_with_escalation(**request, timeout=None)
    assert expired == normal


def test_out_of_order_completion_does_not_count_expired_reservations(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(escalation.time, "monotonic", lambda: clock[0])
    spend = escalation.SummarySpendGuard(max_calls=2, window_seconds=10)
    old = spend.reserve_call()
    clock[0] = 20.0
    fresh = spend.reserve_call()
    spend.release_call(fresh, attempted=True)
    clock[0] = 30.0
    spend.release_call(old, attempted=True)
    assert spend._calls == [20.0]
    assert spend._backoff_until == 0
    assert spend.allows()


def test_near_full_budget_fallback_timeout_counts_as_route_failure(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(escalation.time, "monotonic", lambda: clock[0])
    breaker = escalation.SummaryCircuitBreaker(failure_threshold=1)

    def summary(prompt, max_tokens, model="", timeout=None):
        clock[0] += 0.1 if model == "primary" else timeout
        return None

    monkeypatch.setattr(escalation, "_call_llm_for_summary", summary)
    escalation.summarize_with_escalation(
        "source facts " * 1000, source_tokens=3000, token_budget=100,
        model="primary", fallback_models=["backup"], timeout=10, circuit_breaker=breaker,
    )
    assert not breaker.allows("primary")
    assert not breaker.allows("backup")


def test_out_of_order_backoff_starts_at_latest_counted_dispatch(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(escalation.time, "monotonic", lambda: clock[0])
    spend = escalation.SummarySpendGuard(max_calls=2, window_seconds=10, backoff_seconds=5)
    old = spend.reserve_call()
    clock[0] = 4.0
    fresh = spend.reserve_call()
    clock[0] = 5.0
    spend.release_call(fresh, attempted=True)
    clock[0] = 6.0
    spend.release_call(old, attempted=True)
    assert spend._backoff_until == 9.0
    assert not spend.allows()
    clock[0] = 9.0
    assert spend.allows()
