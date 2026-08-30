"""Provider-neutral operation, grounding, and immutable-result tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
import json
from types import SimpleNamespace

import pytest

from hermes_lcm.assertion_store import (
    AssertionCandidate,
    AssertionRelationCandidate,
    AssertionStore,
)
from hermes_lcm.reasoning import (
    compile_evidence_plan,
    execute_plan,
    ground_evidence,
    question_date_as_of_epoch,
    resolve_occurrence_time_with_trust,
    resolve_temporal_window,
    validate_selector_alignment,
    verify_final_answer,
)
from hermes_lcm.store import MessageStore
from hermes_lcm.tools import lcm_compute


def _epoch(day: str) -> float:
    return datetime.fromisoformat(f"{day}T12:00:00+00:00").timestamp()


@pytest.fixture
def evidence_db(tmp_path):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    try:
        yield messages, assertions
    finally:
        assertions.close()
        messages.close()


def _message(messages, content: str, day: str) -> int:
    return messages.append(
        "session-a",
        {"role": "user", "content": content, "timestamp": _epoch(day)},
    )


def _raw(store_id: int, content: str, quote: str, **extra):
    start = content.index(quote)
    return {
        "store_id": store_id,
        "span_start": start,
        "span_end": start + len(quote),
        "quote": quote,
        **extra,
    }


def _ground(messages, assertions, operands, *, question_date="2024-12-31"):
    decision = ground_evidence(
        operands,
        messages=messages,
        assertions=assertions,
        as_of=question_date_as_of_epoch(question_date),
    )
    assert decision.status == "grounded", decision.reason
    return decision.operands


def test_activation_uses_only_question_language_and_fails_closed():
    assert compile_evidence_plan("Tell me about the trip").status == "not_applicable"
    assert compile_evidence_plan("What is the average price?").status == "fallback"
    assert compile_evidence_plan("What happened 5 days ago?").status == "fallback"
    temporal = compile_evidence_plan("What happened 5 days ago?", "2024-03-20")
    assert temporal.status == "planned"
    assert temporal.plan.operation == "date_filter"
    assert temporal.plan.temporal_window.start == date(2024, 3, 15)


def test_planner_uses_explicit_cardinality_and_interval_units():
    how_long_ago = compile_evidence_plan(
        "How long ago did I complete the challenge?",
        "2023-03-20",
    )
    assert how_long_ago.status == "planned"
    assert how_long_ago.plan.operation == "date_interval"
    assert how_long_ago.plan.exact_operands == 1

    interval = compile_evidence_plan(
        "How many weeks had passed since I recovered when I went jogging?",
        "2024-03-20",
    )
    assert interval.status == "planned"
    assert interval.plan.operation == "date_interval"
    assert interval.plan.exact_operands == 2
    assert interval.plan.interval_unit == "week"

    ordered = compile_evidence_plan(
        "What is the order of the three trips from earliest to latest?"
    )
    assert ordered.status == "planned"
    assert ordered.plan.operation == "order"
    assert ordered.plan.exact_operands == 3
    assert ordered.plan.requires_complete_evidence is False

    singular = compile_evidence_plan(
        "What kitchen appliance did I buy 10 days ago?", "2023-03-25"
    )
    assert singular.status == "planned"
    assert singular.plan.operation == "date_filter"
    assert singular.plan.exact_operands is None
    assert singular.plan.requires_complete_evidence is True


@pytest.mark.parametrize(
    ("question_date", "question", "start", "end"),
    [
        ("2024-03-31", "What happened 1 month ago?", "2024-02-29", "2024-03-01"),
        ("2023-03-31", "What happened 1 month ago?", "2023-02-28", "2023-03-01"),
        ("2024-03-18", "What happened last Monday?", "2024-03-11", "2024-03-12"),
        ("2024-03-20", "What happened last month?", "2024-02-01", "2024-03-01"),
    ],
)
def test_temporal_windows_clamp_and_reanchor(question_date, question, start, end):
    window = resolve_temporal_window(question, question_date)
    assert window is not None
    assert (window.start.isoformat(), window.end.isoformat()) == (start, end)


def test_grounding_rejects_unproven_values_labels_keys_and_refs(evidence_db):
    messages, assertions = evidence_db
    content = "Alice read 120 pages of Dune on 2024-03-15."
    store_id = _message(messages, content, "2024-03-15")
    base = _raw(
        store_id,
        content,
        "Alice read 120 pages of Dune on 2024-03-15",
        value=120,
        unit="pages",
        label="Alice",
        key="Alice Dune",
        date="2024-03-15",
    )
    assert ground_evidence(
        [base], messages=messages, assertions=assertions
    ).status == "grounded"

    for mutation, reason in [
        ({"value": 121}, "numeric value"),
        ({"unit": "hours"}, "unit hour"),
        ({"label": "Bob"}, "label"),
        ({"key": "Alice Foundation"}, "canonical key"),
        ({"span_start": 1}, "quote does not match"),
    ]:
        decision = ground_evidence(
            [{**base, **mutation}], messages=messages, assertions=assertions
        )
        assert decision.status == "fallback"
        assert reason in decision.reason


def test_relative_occurrence_time_grounds_without_aliasing_late_observation(evidence_db):
    messages, assertions = evidence_db
    content = "I completed the plank challenge 5 days ago."
    store_id = _message(messages, content, "2023-03-20")
    occurrence = {
        "observed_at": _epoch("2023-03-20"),
        "event_at": _epoch("2023-03-15"),
        "event_date": "2023-03-15",
        "event_time_source": "relative_to_session",
        "session_date": "2023-03-20",
        "precision": "day",
        "policy_version": "occurrence-time-v1",
    }
    decision = ground_evidence(
        [_raw(store_id, content, content, date="2023-03-15", occurrence_time=occurrence)],
        messages=messages,
        assertions=assertions,
        as_of=question_date_as_of_epoch("2023-03-20"),
    )
    assert decision.status == "grounded", decision.reason
    assert decision.operands[0].evidence_date == date(2023, 3, 15)


def test_explicit_occurrence_without_sidecar_is_certified(evidence_db):
    messages, assertions = evidence_db
    content = "I completed the plank challenge on 2023-03-15."
    store_id = messages.append("session-a", {"role": "user", "content": content})
    occurrence = {
        "event_at": _epoch("2023-03-15"),
        "event_date": "2023-03-15",
        "event_time_source": "explicit",
        "precision": "day",
        "policy_version": "occurrence-time-v1",
    }

    decision = ground_evidence(
        [
            _raw(
                store_id,
                content,
                content,
                date="2023-03-15",
                occurrence_time=occurrence,
            )
        ],
        messages=messages,
        assertions=assertions,
        engine=SimpleNamespace(_session_occurrence_dates={}),
    )

    assert decision.status == "grounded", decision.reason
    assert decision.temporal_trust == "not_applicable"
    assert decision.temporal_certified is True
    assert decision.notes == ()


def test_compute_accepts_caller_anchor_only_when_it_agrees_with_sidecar(evidence_db):
    messages, assertions = evidence_db
    content = "I completed the plank challenge 5 days ago."
    store_id = messages.append("session-a", {"role": "user", "content": content})
    occurrence = {
        "event_date": "2023-03-15",
        "event_time_source": "relative_to_session",
        "session_date": "2023-03-20",
    }
    response = json.loads(lcm_compute(
        {
            "question": "How long ago did I complete the plank challenge?",
            "question_date": "2023-03-20",
            "operands": [
                _raw(
                    store_id,
                    content,
                    content,
                    date="2023-03-15",
                    occurrence_time=occurrence,
                )
            ],
        },
        engine=SimpleNamespace(
            _store=messages,
            _assertions=assertions,
            _session_occurrence_dates={"session-a": "2023-03-20"},
        ),
    ))

    assert response["status"] == "computed"
    assert response["trace"]["result_value"] == 5
    assert response["temporal_trust"] == {
        "status": "engine_sidecar",
        "certified": True,
        "notes": [],
    }


@pytest.mark.parametrize(
    ("unit_phrase", "expected_value", "expected_unit"),
    [
        ("days", 794, "day"),
        ("weeks", 113, "week"),
        ("months", 26, "month"),
        ("years", 2, "year"),
        ("calendar days", 794, "day"),
        ("calendar weeks", 113, "week"),
        ("calendar months", 26, "month"),
    ],
)
def test_how_long_ago_answer_honors_explicit_unit(
    evidence_db, unit_phrase, expected_value, expected_unit
):
    messages, assertions = evidence_db
    content = "I completed the plank challenge on January 15, 2021."
    store_id = messages.append(
        "session-a",
        {"role": "user", "content": content, "timestamp": _epoch("2021-01-15")},
    )
    response = json.loads(lcm_compute(
        {
            "question": (
                f"How long ago, in {unit_phrase}, did I complete the plank challenge?"
            ),
            "question_date": "2023-03-20",
            "operands": [
                _raw(store_id, content, content, date="2021-01-15")
            ],
        },
        engine=SimpleNamespace(_store=messages, _assertions=assertions),
    ))

    assert response["status"] == "computed"
    assert response["trace"]["result_value"] == expected_value
    assert response["trace"]["unit"] == expected_unit


@pytest.mark.parametrize("question_date", ["2021-02-28", "2021-03-01"])
def test_how_long_ago_years_clamps_leap_day_anniversary(
    evidence_db, question_date
):
    messages, assertions = evidence_db
    content = "I completed the plank challenge on February 29, 2020."
    store_id = messages.append(
        "session-a",
        {"role": "user", "content": content, "timestamp": _epoch("2020-02-29")},
    )
    response = json.loads(lcm_compute(
        {
            "question": (
                "How long ago, in years, did I complete the plank challenge?"
            ),
            "question_date": question_date,
            "operands": [
                _raw(store_id, content, content, date="2020-02-29")
            ],
        },
        engine=SimpleNamespace(_store=messages, _assertions=assertions),
    ))

    assert response["status"] == "computed"
    assert response["trace"]["result_value"] == 1
    assert response["trace"]["unit"] == "year"


def test_how_long_ago_answer_without_unit_uses_coarsest_fit(evidence_db):
    messages, assertions = evidence_db
    content = "I completed the plank challenge on January 15, 2021."
    store_id = messages.append(
        "session-a",
        {"role": "user", "content": content, "timestamp": _epoch("2021-01-15")},
    )
    response = json.loads(lcm_compute(
        {
            "question": "How long ago did I complete the plank challenge?",
            "question_date": "2023-03-20",
            "operands": [
                _raw(store_id, content, content, date="2021-01-15")
            ],
        },
        engine=SimpleNamespace(_store=messages, _assertions=assertions),
    ))

    assert response["status"] == "computed"
    assert response["trace"]["result"] == "2 years"
    assert response["trace"]["unit"] == "year"


def test_compute_sidecar_overrides_disagreeing_caller_anchor(evidence_db):
    messages, assertions = evidence_db
    content = "I completed the plank challenge 5 days ago."
    store_id = messages.append("session-a", {"role": "user", "content": content})
    response = json.loads(lcm_compute(
        {
            "question": "How long ago did I complete the plank challenge?",
            "question_date": "2023-03-20",
            "operands": [
                _raw(
                    store_id,
                    content,
                    content,
                    date="2023-03-16",
                    occurrence_time={
                        "event_date": "2023-03-16",
                        "event_time_source": "relative_to_session",
                        "session_date": "2023-03-21",
                    },
                )
            ],
        },
        engine=SimpleNamespace(
            _store=messages,
            _assertions=assertions,
            _session_occurrence_dates={"session-a": "2023-03-20"},
        ),
    ))

    assert response["status"] == "computed"
    assert response["trace"]["result_value"] == 5
    assert response["trace"]["evidence_dates"] == ["2023-03-15"]
    assert response["temporal_trust"]["certified"] is True
    assert response["temporal_trust"]["notes"] == [
        "caller session_date 2023-03-21 overridden by engine sidecar 2023-03-20"
    ]


def test_caller_session_date_is_bounded_in_schema_and_trust_note():
    from hermes_lcm.schemas import LCM_COMPUTE

    caller_session_date = "caller-" + ("x" * 500)
    _, trust = resolve_occurrence_time_with_trust(
        "I completed the plank challenge 5 days ago.",
        observed_at=_epoch("2023-03-20"),
        session_date=caller_session_date,
        engine=SimpleNamespace(
            _session_occurrence_dates={"session-a": "2023-03-20"},
        ),
        session_id="session-a",
    )

    note = trust["trust_note"]
    assert caller_session_date not in note
    assert ("caller-" + ("x" * 54) + "...") in note
    session_date_schema = (
        LCM_COMPUTE["parameters"]["properties"]["operands"]["items"]
        ["properties"]["occurrence_time"]["properties"]["session_date"]
    )
    assert session_date_schema["maxLength"] == 64


def test_compute_without_sidecar_marks_temporal_result_low_trust(evidence_db):
    messages, assertions = evidence_db
    content = "I completed the plank challenge 5 days ago."
    store_id = messages.append("session-a", {"role": "user", "content": content})
    response = json.loads(lcm_compute(
        {
            "question": "How long ago did I complete the plank challenge?",
            "question_date": "2023-03-20",
            "operands": [
                _raw(
                    store_id,
                    content,
                    content,
                    date="2023-03-15",
                    occurrence_time={
                        "event_date": "2023-03-15",
                        "event_time_source": "relative_to_session",
                        "session_date": "2023-03-20",
                    },
                )
            ],
        },
        engine=SimpleNamespace(
            _store=messages,
            _assertions=assertions,
            _session_occurrence_dates={},
        ),
    ))

    assert response["status"] == "computed"
    assert response["trace"]["result_value"] == 5
    assert response["temporal_trust"]["status"] == "low_trust"
    assert response["temporal_trust"]["certified"] is False
    assert "sidecar absent" in response["temporal_trust"]["notes"][0]


def test_malformed_sidecar_date_is_low_trust_and_uncertified():
    result, trust = resolve_occurrence_time_with_trust(
        "I completed the plank challenge 5 days ago.",
        observed_at=_epoch("2023-03-20"),
        session_date="2023-03-20",
        engine=SimpleNamespace(
            _session_occurrence_dates={"session-a": "not-a-date"},
        ),
        session_id="session-a",
    )

    assert result["session_date"] is None
    assert not {
        "anchor_trust",
        "temporal_certified",
        "session_date_overridden",
        "trust_note",
    }.intersection(result)
    assert trust["anchor_trust"] == "low_trust"
    assert trust["temporal_certified"] is False
    assert "sidecar invalid" in trust["trust_note"]


def test_session_sidecar_cannot_override_real_host_observation_after_as_of(evidence_db):
    messages, assertions = evidence_db
    content = "I completed the plank challenge 5 days ago."
    store_id = _message(messages, content, "2026-07-19")
    occurrence = {
        "observed_at": _epoch("2023-03-20"),
        "event_at": _epoch("2023-03-15"),
        "event_date": "2023-03-15",
        "event_time_source": "relative_to_session",
        "session_date": "2023-03-20",
        "precision": "day",
        "policy_version": "occurrence-time-v1",
    }
    decision = ground_evidence(
        [_raw(store_id, content, content, date="2023-03-15", occurrence_time=occurrence)],
        messages=messages,
        assertions=assertions,
        as_of=question_date_as_of_epoch("2023-03-20"),
    )
    assert decision.status == "fallback"
    assert "occurrence_time date is not supported" in decision.reason


def test_relative_occurrence_uses_stored_observation_not_caller_session_date(evidence_db):
    messages, assertions = evidence_db
    content = "I completed the plank challenge 5 days ago."
    store_id = _message(messages, content, "2026-07-19")
    occurrence = {
        "observed_at": _epoch("2023-03-20"),
        "event_at": _epoch("2023-03-15"),
        "event_date": "2023-03-15",
        "event_time_source": "relative_to_session",
        "session_date": "2023-03-20",
        "precision": "day",
        "policy_version": "occurrence-time-v1",
    }
    decision = ground_evidence(
        [_raw(store_id, content, content, date="2023-03-15", occurrence_time=occurrence)],
        messages=messages,
        assertions=assertions,
        as_of=question_date_as_of_epoch("2026-07-19"),
    )

    assert decision.status == "fallback"
    assert "occurrence_time date is not supported" in decision.reason


def test_assertion_observation_time_is_not_silently_used_as_event_time(evidence_db):
    messages, assertions = evidence_db
    content = "The status is green."
    store_id = _message(messages, content, "2024-03-01")
    snapshot = assertions.snapshot_source(store_id)
    start = content.index("green")
    result = assertions.publish_source(
        snapshot,
        [AssertionCandidate(
            source_span_start=start,
            source_span_end=start + len("green"),
            subject_key="project:test",
            predicate_key="status.color",
            object_value="green",
            value_text="green",
            kind="status",
            event_at=None,
        )],
    )
    decision = ground_evidence(
        [{"assertion_id": result.assertion_ids[0], "value": "green"}],
        messages=messages,
        assertions=assertions,
    )
    assert decision.status == "grounded", decision.reason
    assert decision.operands[0].evidence_date is None


def test_sum_difference_count_and_mixed_unit_fallback(evidence_db):
    messages, assertions = evidence_db
    first = "Alice spent $30 on Dune."
    second = "Bob spent $18 on Foundation."
    first_id = _message(messages, first, "2024-02-01")
    second_id = _message(messages, second, "2024-02-02")
    operands = _ground(
        messages,
        assertions,
        [
            _raw(first_id, first, "$30", value=30, unit="usd", label="$30"),
            _raw(second_id, second, "$18", value=18, unit="usd", label="$18"),
        ],
    )

    sum_plan = compile_evidence_plan("What was the combined total?").plan
    summed = execute_plan(sum_plan, operands)
    assert summed.trace.result == "$48"

    difference_plan = compile_evidence_plan("What is the difference?").plan
    difference = execute_plan(difference_plan, operands)
    assert difference.trace.result == "$12"

    mixed = replace(operands[1], unit="page")
    assert execute_plan(sum_plan, (operands[0], mixed)).status == "fallback"
    hidden_units = tuple(replace(operand, unit=None) for operand in operands)
    assert execute_plan(sum_plan, hidden_units).status == "fallback"
    partial_time_units = (
        replace(operands[0], unit="hour", value=1),
        replace(operands[1], unit=None, value=30),
    )
    partial = execute_plan(sum_plan, partial_time_units)
    assert partial.status == "fallback"
    assert partial.reason == "every time operand must carry a compatible unit"

    counted = replace(operands[0], key="dune", unit="item")
    counted_again = replace(operands[1], key="dune", unit="item")
    count_plan = compile_evidence_plan("How many distinct items were there?").plan
    count = execute_plan(count_plan, (counted, counted_again))
    assert count.trace.result == "1 item"


def test_sum_preserves_large_integers_and_bounds_decimal_overflow(evidence_db):
    messages, assertions = evidence_db
    first = "The first total is 9007199254740993 items."
    second = "The second total is 1 item."
    first_id = _message(messages, first, "2024-02-01")
    second_id = _message(messages, second, "2024-02-02")
    engine = SimpleNamespace(_store=messages, _assertions=assertions)

    exact = json.loads(lcm_compute(
        {
            "question": "What is the combined total?",
            "evidence_complete": True,
            "operands": [
                _raw(first_id, first, first, value=9007199254740993, unit="item"),
                _raw(second_id, second, second, value=1, unit="item"),
            ],
        },
        engine=engine,
    ))
    assert exact["status"] == "computed"
    assert exact["trace"]["result_value"] == 9007199254740994
    assert exact["trace"]["result"] == "9007199254740994 items"

    comparison = "The comparison total is 9007199254740992 items."
    comparison_id = _message(messages, comparison, "2024-02-03")
    exact_difference = json.loads(lcm_compute(
        {
            "question": "How much more is the first total than the comparison total?",
            "evidence_complete": True,
            "operands": [
                _raw(
                    first_id,
                    first,
                    first,
                    value=9007199254740993,
                    unit="item",
                    label="first total",
                ),
                _raw(
                    comparison_id,
                    comparison,
                    comparison,
                    value=9007199254740992,
                    unit="item",
                    label="comparison total",
                ),
            ],
        },
        engine=engine,
    ))
    assert exact_difference["status"] == "computed"
    assert exact_difference["trace"]["result_value"] == 1
    assert exact_difference["trace"]["result"] == "1 item"

    operands = _ground(
        messages,
        assertions,
        [
            _raw(first_id, first, first, value=9007199254740993, unit="item"),
            _raw(second_id, second, second, value=1, unit="item"),
        ],
    )
    overflowing = tuple(
        replace(operand, value=1e308) for operand in operands
    )
    bounded = execute_plan(
        compile_evidence_plan("What is the combined total?").plan,
        overflowing,
    )
    assert bounded.status == "fallback"
    assert bounded.reason == "numeric result exceeds the bounded decimal range"


def test_planner_populates_requested_result_unit_for_time_arithmetic(evidence_db):
    messages, assertions = evidence_db
    first = "Jogging took 1 hour."
    second = "Yoga took 30 minutes."
    first_id = _message(messages, first, "2024-02-01")
    second_id = _message(messages, second, "2024-02-02")
    engine = SimpleNamespace(_store=messages, _assertions=assertions)
    operands = [
        _raw(first_id, first, first, value=1, unit="hour", label="jogging"),
        _raw(second_id, second, second, value=30, unit="minute", label="yoga"),
    ]

    hours = compile_evidence_plan("What is the combined total in hours?").plan
    minutes = compile_evidence_plan("What is the combined total in minutes?").plan
    how_many_hours = compile_evidence_plan("How many hours was the combined total?").plan
    assert hours.result_unit == "hour"
    assert minutes.result_unit == "minute"
    assert how_many_hours.result_unit == "hour"

    in_hours = json.loads(lcm_compute(
        {
            "question": "What is the combined total in hours?",
            "evidence_complete": True,
            "operands": operands,
        },
        engine=engine,
    ))
    in_minutes = json.loads(lcm_compute(
        {
            "question": "What is the combined total in minutes?",
            "evidence_complete": True,
            "operands": operands,
        },
        engine=engine,
    ))
    difference = json.loads(lcm_compute(
        {
            "question": "How much more time did jogging take than yoga, in minutes?",
            "evidence_complete": True,
            "operands": operands,
        },
        engine=engine,
    ))
    assert in_hours["trace"]["result"] == "1.5 hours"
    assert in_hours["trace"]["result_value"] == 1.5
    assert in_minutes["trace"]["result"] == "90 minutes"
    assert in_minutes["trace"]["result_value"] == 90
    assert difference["trace"]["result"] == "30 minutes"
    assert difference["trace"]["result_value"] == 30


def test_explicit_time_result_unit_wins_over_incidental_currency(evidence_db):
    messages, assertions = evidence_db
    first = "The $10 premium service took 2 hours."
    second = "The $5 basic service took 1 hour."
    first_id = _message(messages, first, "2024-02-01")
    second_id = _message(messages, second, "2024-02-02")
    question = (
        "What is the difference in hours between the $10 premium service "
        "and the $5 basic service?"
    )
    engine = SimpleNamespace(_store=messages, _assertions=assertions)

    plan = compile_evidence_plan(question).plan
    assert plan.result_unit == "hour"

    result = json.loads(lcm_compute(
        {
            "question": question,
            "evidence_complete": True,
            "operands": [
                _raw(first_id, first, first, value=2, unit="hour", label="premium service"),
                _raw(second_id, second, second, value=1, unit="hour", label="basic service"),
            ],
        },
        engine=engine,
    ))
    assert result["status"] == "computed", result
    assert result["trace"]["result"] == "1 hour"


def test_directed_difference_validates_question_order(evidence_db):
    messages, assertions = evidence_db
    first = "Alice spent $30."
    second = "Bob spent $18."
    first_id = _message(messages, first, "2024-02-01")
    second_id = _message(messages, second, "2024-02-02")
    operands = _ground(
        messages,
        assertions,
        [
            _raw(first_id, first, first, value=30, unit="usd", label="Alice"),
            _raw(second_id, second, second, value=18, unit="usd", label="Bob"),
        ],
    )
    question = "How much more did Alice spend than Bob?"
    plan = compile_evidence_plan(question).plan
    assert plan.difference_direction == "first_minus_second"
    assert validate_selector_alignment(question, plan, operands) is None
    assert execute_plan(plan, operands).trace.result == "$12"
    assert "question mention order" in validate_selector_alignment(
        question, plan, tuple(reversed(operands))
    )
    contradicted = (replace(operands[0], value=10), operands[1])
    assert execute_plan(plan, contradicted).status == "fallback"


def test_date_filter_interval_order_and_cardinality(evidence_db):
    messages, assertions = evidence_db
    first = "The PlankChallenge happened on 2023-03-15."
    second = "The launch happened on 2023-03-18."
    first_id = _message(messages, first, "2023-03-15")
    second_id = _message(messages, second, "2023-03-18")
    operands = _ground(
        messages,
        assertions,
        [
            _raw(
                first_id,
                first,
                first,
                label="PlankChallenge",
                date="2023-03-15",
            ),
            _raw(
                second_id,
                second,
                second,
                label="launch",
                date="2023-03-18",
            ),
        ],
        question_date="2023-03-20",
    )
    filtered_plan = compile_evidence_plan(
        "What happened 5 days ago?", "2023-03-20"
    ).plan
    filtered = execute_plan(filtered_plan, operands)
    assert filtered.trace.result == "PlankChallenge"

    interval_plan = compile_evidence_plan(
        "How many days between the PlankChallenge and launch?", "2023-03-20"
    ).plan
    assert interval_plan.operation == "date_interval"
    assert execute_plan(interval_plan, operands).trace.result == "3 days"

    ago_plan = compile_evidence_plan(
        "How many days ago did the launch happen?", "2023-03-20"
    ).plan
    assert execute_plan(ago_plan, (operands[1],)).trace.result == "2 days"

    order_plan = compile_evidence_plan("Put the events in chronological order").plan
    assert execute_plan(order_plan, tuple(reversed(operands))).trace.result == (
        "PlankChallenge -> launch"
    )
    assert execute_plan(interval_plan, operands[:1]).status == "fallback"


def test_temporal_filter_revalidates_exact_cardinality(evidence_db):
    messages, assertions = evidence_db
    rows = [
        ("Alpha happened on 2023-03-15.", "Alpha", "2023-03-15"),
        ("Beta happened on 2023-03-15.", "Beta", "2023-03-15"),
        ("Gamma happened on 2023-03-14.", "Gamma", "2023-03-14"),
    ]
    raw = []
    for content, label, day in rows:
        store_id = _message(messages, content, day)
        raw.append(_raw(store_id, content, content, label=label, date=day))
    engine = SimpleNamespace(_store=messages, _assertions=assertions)
    args = {
        "question": "Which three events happened 5 days ago?",
        "question_date": "2023-03-20",
        "operands": raw,
    }

    filtered = json.loads(lcm_compute(args, engine=engine))
    assert filtered["status"] == "fallback"
    assert filtered["reason"] == "date_filter requires exactly 3 operands"

    # A direct executor control proves a complete in-window population remains valid.
    grounded = _ground(
        messages,
        assertions,
        raw[:2] + [
            _raw(
                _message(messages, "Gamma happened on 2023-03-15.", "2023-03-15"),
                "Gamma happened on 2023-03-15.",
                "Gamma happened on 2023-03-15.",
                label="Gamma",
                date="2023-03-15",
            )
        ],
        question_date="2023-03-20",
    )
    complete = execute_plan(
        compile_evidence_plan(args["question"], args["question_date"]).plan,
        grounded,
    )
    assert complete.status == "computed"
    assert complete.trace.result_value == ("Alpha", "Beta", "Gamma")


def test_order_projects_requested_ordinal_and_preserves_full_order(evidence_db):
    messages, assertions = evidence_db
    rows = [
        ("Alpha happened on 2023-03-10.", "Alpha", "2023-03-10"),
        ("Beta happened on 2023-03-11.", "Beta", "2023-03-11"),
        ("Gamma happened on 2023-03-12.", "Gamma", "2023-03-12"),
    ]
    operands = []
    for content, label, day in rows:
        store_id = _message(messages, content, day)
        operands.append(_raw(store_id, content, content, label=label, date=day))
    engine = SimpleNamespace(_store=messages, _assertions=assertions)

    for question, expected in (
        ("Which event was first?", "Alpha"),
        ("Which event was second?", "Beta"),
        ("Which event was third?", "Gamma"),
        ("Which event was previous?", "Beta"),
        ("Which city did I visit second?", "Beta"),
        ("What restaurant did I visit first?", "Alpha"),
    ):
        result = json.loads(lcm_compute(
            {
                "question": question,
                "question_date": "2023-03-20",
                "evidence_complete": True,
                "operands": list(reversed(operands)),
            },
            engine=engine,
        ))
        assert result["status"] == "computed", (question, result)
        assert result["trace"]["result"] == expected
        assert result["trace"]["result_value"] == [expected]

    full = json.loads(lcm_compute(
        {
            "question": "Put the events in chronological order.",
            "question_date": "2023-03-20",
            "evidence_complete": True,
            "operands": list(reversed(operands)),
        },
        engine=engine,
    ))
    assert full["trace"]["result"] == "Alpha -> Beta -> Gamma"


def test_how_long_ago_uses_one_anchored_operand(evidence_db):
    messages, assertions = evidence_db
    content = "I visited Paris on 2023-03-18."
    store_id = _message(messages, content, "2023-03-18")
    operands = _ground(
        messages,
        assertions,
        [_raw(store_id, content, content, date="2023-03-18")],
        question_date="2023-03-20",
    )

    plan = compile_evidence_plan(
        "How long ago did I visit Paris?",
        "2023-03-20",
    ).plan

    assert plan.operation == "date_interval"
    assert plan.exact_operands == 1
    assert plan.interval_unit == "day"
    assert execute_plan(plan, operands).trace.result == "2 days"


def test_latest_fact_requires_complete_nonconflicting_assertion_state(evidence_db):
    messages, assertions = evidence_db
    first = "My current city is Paris."
    second = "My current city is Berlin."
    first_id = _message(messages, first, "2024-01-01")
    second_id = _message(messages, second, "2024-02-01")
    snapshots = [assertions.snapshot_source(first_id), assertions.snapshot_source(second_id)]
    assertion_ids = []
    for snapshot, city, day in zip(snapshots, ("Paris", "Berlin"), ("2024-01-01", "2024-02-01")):
        start = snapshot.content.index(city)
        result = assertions.publish_source(
            snapshot,
            [AssertionCandidate(
                source_span_start=start,
                source_span_end=start + len(city),
                subject_key="user:self",
                predicate_key="location.city",
                object_value=city,
                value_text=city,
                kind="status",
                event_at=_epoch(day),
            )],
        )
        assertion_ids.append(result.assertion_ids[0])

    grounded = _ground(
        messages,
        assertions,
        [
            {"assertion_id": assertion_ids[0], "value": "Paris", "label": "Paris"},
            {"assertion_id": assertion_ids[1], "value": "Berlin", "label": "Berlin"},
        ],
    )
    plan = compile_evidence_plan("What is my current city?").plan
    decision = execute_plan(plan, grounded)
    assert decision.status == "fallback"
    assert "unresolved conflicting" in decision.reason

    third = "My current city is now Rome, replacing Paris and Berlin."
    third_id = _message(messages, third, "2024-03-01")
    third_snapshot = assertions.snapshot_source(third_id)
    start = third.index("Rome")
    third_candidate = AssertionCandidate(
        source_span_start=start,
        source_span_end=start + len("Rome"),
        subject_key="user:self",
        predicate_key="location.city",
        object_value="Rome",
        value_text="Rome",
        kind="status",
        event_at=_epoch("2024-03-01"),
    )
    third_assertion_id = assertions.assertion_id_for(third_snapshot, third_candidate)
    relation_start = third.index("replacing")
    relations = [
        AssertionRelationCandidate(
            source_span_start=relation_start,
            source_span_end=len(third),
            from_assertion_id=third_assertion_id,
            relation_type="supersedes",
            to_assertion_id=old_id,
        )
        for old_id in assertion_ids
    ]
    assertions.publish_source(
        third_snapshot,
        [third_candidate],
        relations=relations,
    )
    resolved = _ground(
        messages,
        assertions,
        [
            {"assertion_id": assertion_ids[0], "value": "Paris", "label": "Paris"},
            {"assertion_id": assertion_ids[1], "value": "Berlin", "label": "Berlin"},
            {"assertion_id": third_assertion_id, "value": "Rome", "label": "Rome"},
        ],
    )
    latest = execute_plan(plan, resolved)
    assert latest.status == "computed"
    assert latest.trace.result == "Rome"


def test_verifier_preserves_result_entities_units_and_exact_citations(evidence_db):
    messages, assertions = evidence_db
    first = "Alice spent $30."
    second = "Bob spent $18."
    first_id = _message(messages, first, "2024-02-01")
    second_id = _message(messages, second, "2024-02-02")
    operands = _ground(
        messages,
        assertions,
        [
            _raw(first_id, first, first, value=30, unit="usd", label="Alice"),
            _raw(second_id, second, second, value=18, unit="usd", label="Bob"),
        ],
    )
    plan = compile_evidence_plan("How much more did Alice spend than Bob?").plan
    trace = execute_plan(plan, operands).trace
    assert verify_final_answer(trace.answer, trace).status == "verified"
    cited = " ".join(f"[{citation}]" for citation in trace.citations)
    assert verify_final_answer(f"Alice spent $12 more than Bob. {cited}", trace).status == "verified"
    for candidate in (
        f"Alice did not spend $12 more than Bob. {cited}",
        f"Alice never spent $12 more than Bob. {cited}",
        f"Alice spent $13 more than Bob. {cited}",
        f"Alice spent 12 pages more than Bob. {cited}",
        f"Charlie spent $12 more than Bob. {cited}",
        "Alice spent $12 more than Bob. [lcm:999:0-1]",
    ):
        assert verify_final_answer(candidate, trace).status == "fallback"


def test_question_date_boundary_is_utc_end_of_day():
    boundary = question_date_as_of_epoch("2024-02-29")
    assert datetime.fromtimestamp(boundary, tz=timezone.utc).date() == date(2024, 2, 29)
    assert question_date_as_of_epoch("2024-02-30") is None
    assert question_date_as_of_epoch("9999-12-31") is None
    assert question_date_as_of_epoch("9999-12-30") is not None


def test_public_compute_rejects_terminal_question_date_without_overflow(evidence_db):
    messages, assertions = evidence_db
    result = json.loads(lcm_compute(
        {
            "question": "What happened 1 day ago?",
            "question_date": "9999-12-31",
            "operands": [],
        },
        engine=SimpleNamespace(_store=messages, _assertions=assertions),
    ))
    assert result["status"] == "fallback"
    assert result["reason"] == "question_date must be a valid timezone-unambiguous ISO date"


def test_public_compute_tool_reports_stages_and_discards_mutated_candidate(evidence_db):
    messages, assertions = evidence_db
    first = "Alice spent $30."
    second = "Bob spent $18."
    first_id = _message(messages, first, "2024-02-01")
    second_id = _message(messages, second, "2024-02-02")
    args = {
        "question": "How much more did Alice spend than Bob?",
        "question_date": "2024-03-01",
        "operands": [
            _raw(first_id, first, first, value=30, unit="usd", label="Alice"),
            _raw(second_id, second, second, value=18, unit="usd", label="Bob"),
        ],
    }
    engine = SimpleNamespace(_store=messages, _assertions=assertions)
    computed = json.loads(lcm_compute(args, engine=engine))
    assert computed["status"] == "computed"
    assert computed["answer"].startswith("$12 ")
    assert set(computed["provenance"]["stages"]) == {
        "planner",
        "selector",
        "executor",
        "final_answerer",
    }
    assert computed["provenance"]["stages"]["planner"]["provider"] == "none"
    assert computed["provenance"]["stages"]["selector"]["provider"] == (
        "unknown_to_plugin"
    )
    assert (
        computed["provenance"]["stages"]["selector"]["temporal_certified"]
        is None
    )
    assert computed["temporal_trust"]["certified"] is None

    cited = " ".join(f"[{value}]" for value in computed["trace"]["citations"])
    mutated = json.loads(lcm_compute(
        {**args, "candidate_answer": f"Charlie spent $12 more than Bob. {cited}"},
        engine=engine,
    ))
    assert mutated["status"] == "computed"
    assert mutated["candidate_verification"]["status"] == "fallback"
    assert mutated["answer"] == mutated["trace"]["answer"]
    assert mutated["provenance"]["stages"]["final_answerer"]["candidate_used"] is False


def test_public_compute_tool_requires_closed_cardinality(evidence_db):
    messages, assertions = evidence_db
    content = "I visited Paris."
    store_id = _message(messages, content, "2024-02-01")
    response = json.loads(lcm_compute(
        {
            "question": "How many cities did I visit?",
            "operands": [
                _raw(store_id, content, content, key="Paris", label="Paris")
            ],
        },
        engine=SimpleNamespace(_store=messages, _assertions=assertions),
    ))
    assert response["status"] == "fallback"
    assert response["reason"] == "operation requires explicit evidence_complete=true"


def test_count_distinct_rejects_question_subject_as_canonical_key(evidence_db):
    messages, assertions = evidence_db
    paris = "Alice visited Paris."
    rome = "Alice visited Rome."
    paris_id = _message(messages, paris, "2024-02-01")
    rome_id = _message(messages, rome, "2024-02-02")
    args = {
        "question": "How many cities did Alice visit?",
        "evidence_complete": True,
        "operands": [
            _raw(paris_id, paris, paris, key="alice", unit="item"),
            _raw(rome_id, rome, rome, key="alice", unit="item"),
        ],
    }
    engine = SimpleNamespace(_store=messages, _assertions=assertions)

    rejected = json.loads(lcm_compute(args, engine=engine))
    accepted = json.loads(lcm_compute(
        {
            **args,
            "operands": [
                _raw(paris_id, paris, paris, key="paris", unit="item"),
                _raw(rome_id, rome, rome, key="rome", unit="item"),
            ],
        },
        engine=engine,
    ))

    assert rejected["status"] == "fallback"
    assert rejected["reason"] == (
        "count_distinct canonical keys must identify counted entities, "
        "not the question subject"
    )
    assert accepted["status"] == "computed"
    assert accepted["trace"]["result_value"] == 2


def test_named_sum_binds_each_value_to_its_requested_summand(evidence_db):
    messages, assertions = evidence_db
    taxi = "Taxi $20, dinner $50."
    hotel = "Hotel $100, flight $500."
    taxi_id = _message(messages, taxi, "2024-02-01")
    hotel_id = _message(messages, hotel, "2024-02-02")
    question = "What was the total of the taxi and hotel?"
    engine = SimpleNamespace(_store=messages, _assertions=assertions)

    wrong_values = json.loads(lcm_compute(
        {
            "question": question,
            "evidence_complete": True,
            "operands": [
                _raw(taxi_id, taxi, taxi, value=50, unit="usd", label="Taxi"),
                _raw(hotel_id, hotel, hotel, value=500, unit="usd", label="Hotel"),
            ],
        },
        engine=engine,
    ))
    wrong_labels = json.loads(lcm_compute(
        {
            "question": question,
            "evidence_complete": True,
            "operands": [
                _raw(taxi_id, taxi, taxi, value=20, unit="usd", label="dinner"),
                _raw(hotel_id, hotel, hotel, value=100, unit="usd", label="flight"),
            ],
        },
        engine=engine,
    ))
    correct = json.loads(lcm_compute(
        {
            "question": question,
            "evidence_complete": True,
            "operands": [
                _raw(taxi_id, taxi, taxi, value=20, unit="usd", label="Taxi"),
                _raw(hotel_id, hotel, hotel, value=100, unit="usd", label="Hotel"),
            ],
        },
        engine=engine,
    ))

    assert wrong_values["status"] == "fallback"
    assert wrong_labels["status"] == "fallback"
    assert correct["status"] == "computed"
    assert correct["trace"]["result_value"] == 120
