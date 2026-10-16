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
    spend.release_call(token)
    assert spend.allows()
    token = spend.reserve_call()
    assert spend.commit_call(token) is not None
    assert not spend.allows()
    assert spend._reservations == {}
    spend.clear()
    token = spend.reserve_call()
    spend.clear()  # A forced operation starts a new budget, including pending calls.
    assert spend.commit_call(token) is None
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


def test_slow_dispatched_call_still_enters_backoff(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(escalation.time, "monotonic", lambda: clock[0])
    spend = escalation.SummarySpendGuard(max_calls=1, window_seconds=10, backoff_seconds=100)
    old = spend.reserve_call()
    assert spend.commit_call(old) is not None
    clock[0] = 11.0  # The provider response is slower than the spend window.
    assert not spend.allows()


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
    assert spend.commit_call(old) is not None
    clock[0] = 4.0
    fresh = spend.reserve_call()
    assert spend.commit_call(fresh) is not None
    clock[0] = 6.0
    # Completion order is irrelevant after dispatch has been charged.
    assert spend._backoff_until == 9.0
    assert not spend.allows()
    clock[0] = 9.0
    assert spend.allows()


def test_commit_lock_cannot_charge_an_expired_dispatch(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(escalation.time, "monotonic", lambda: clock[0])
    spend = escalation.SummarySpendGuard(max_calls=1)
    token = spend.reserve_call()
    real_lock = spend._lock

    class ContendedLock:
        def __enter__(self):
            real_lock.acquire()
            clock[0] = 10.0

        def __exit__(self, *_args):
            real_lock.release()

    monkeypatch.setattr(spend, "_lock", ContendedLock())
    assert spend.commit_call(token, deadline=10.0) is None
    assert spend._calls == []
    assert spend._reservations == {}
    assert spend._backoff_until == 0


def test_chain_accounts_for_slow_call_before_it_returns(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(escalation.time, "monotonic", lambda: clock[0])
    spend = escalation.SummarySpendGuard(max_calls=1, window_seconds=10, backoff_seconds=100)

    def slow_summary(*args, **kwargs):
        assert spend._reservations == {}
        assert spend._backoff_until == 100.0
        clock[0] = 11.0
        return "slow summary"

    monkeypatch.setattr(escalation, "_call_llm_for_summary", slow_summary)
    assert escalation._invoke_summary_llm_chain("source", 100, spend_guard=spend) == "slow summary"
    assert not spend.allows()
