"""V4 role/lifecycle semantics and the production typed-state tool."""

from __future__ import annotations

import json

import pytest

from hermes_lcm.assertion_extraction import parse_assertion_extraction
from hermes_lcm.assertion_state import query_assertion_state
from hermes_lcm.assertion_store import AssertionCandidate, AssertionStore
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore


def _source(messages, assertions, content, observed_at, *, role="user"):
    store_id = messages.append("session-a", {"role": role, "content": content})
    messages._conn.execute(
        "UPDATE messages SET timestamp = ? WHERE store_id = ?",
        (float(observed_at), store_id),
    )
    messages._conn.commit()
    return assertions.snapshot_source(store_id)


def _wire(
    snapshot,
    quote,
    *,
    subject="user:self",
    resolution="self",
    predicate="state",
    value="active",
    kind="status",
    polarity="positive",
    strength=None,
    scope="",
):
    start = snapshot.content.index(quote)
    return {
        "source_span_start": start,
        "source_span_end": start + len(quote),
        "source_quote": quote,
        "subject_key": subject,
        "subject_resolution": resolution,
        "predicate_key": predicate,
        "object_value": value,
        "value_text": str(value),
        "kind": kind,
        "polarity": polarity,
        "strength": strength,
        "scope_key": scope,
        "event_at": None,
        "valid_from": None,
        "valid_to": None,
        "confidence": 1.0,
    }


def _relation(snapshot, quote, relation_type, target_id):
    start = snapshot.content.index(quote)
    return {
        "source_span_start": start,
        "source_span_end": start + len(quote),
        "source_quote": quote,
        "from_index": 0,
        "relation_type": relation_type,
        "to_assertion_id": target_id,
        "evidence": "explicit",
        "confidence": 1.0,
    }


def _payload(snapshot, assertions, relations=()):
    return {
        "schema_version": 1,
        "source_store_id": snapshot.store_id,
        "source_content_sha256": snapshot.content_sha256,
        "assertions": list(assertions),
        "relations": list(relations),
    }


def _publish(store, snapshot, wires, relations=()):
    extraction = parse_assertion_extraction(
        snapshot,
        _payload(snapshot, wires, relations),
        store=store,
    )
    return store.publish_source(
        snapshot,
        extraction.assertions,
        relations=extraction.relations,
    )


@pytest.fixture
def state_db(tmp_path):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    try:
        yield messages, assertions
    finally:
        assertions.close()
        messages.close()


def _recommendation(messages, store, choice, timestamp):
    content = f"I recommend that you choose {choice}."
    snapshot = _source(
        messages,
        store,
        content,
        timestamp,
        role="assistant",
    )
    result = _publish(store, snapshot, [_wire(
        snapshot,
        f"recommend that you choose {choice}",
        subject="user:self",
        resolution="addressee",
        predicate=f"drink.choice.{choice}",
        value=choice,
        kind="recommendation",
    )])
    return snapshot, result.assertion_ids[0]


def test_recommendations_distinguish_accepted_declined_and_unanswered(state_db):
    messages, store = state_db
    _tea, tea_id = _recommendation(messages, store, "tea", 100.0)
    unanswered = query_assertion_state(
        store,
        subject_key="user:self",
        predicate_key="drink.choice.tea",
        kinds=["recommendation"],
    )
    assert unanswered.assertions[0]["semantic_state"] == "unanswered"
    assert unanswered.assertions[0]["attribution"] == "addressee"

    accepted = _source(
        messages,
        store,
        "I accept that recommendation and will choose tea.",
        110.0,
    )
    _publish(
        store,
        accepted,
        [_wire(
            accepted,
            "will choose tea",
            predicate="drink.choice.tea",
            value="tea",
            kind="commitment",
        )],
        [_relation(
            accepted,
            "accept that recommendation",
            "confirms",
            tea_id,
        )],
    )
    accepted_state = query_assertion_state(
        store,
        subject_key="user:self",
        predicate_key="drink.choice.tea",
        kinds=["recommendation"],
    )
    assert accepted_state.assertions[0]["semantic_state"] == "accepted"
    assert len(accepted_state.relations) == 1

    _coffee, coffee_id = _recommendation(messages, store, "coffee", 200.0)
    declined = _source(
        messages,
        store,
        "No, I explicitly decline coffee.",
        210.0,
    )
    _publish(
        store,
        declined,
        [_wire(
            declined,
            "decline coffee",
            predicate="drink.choice.coffee",
            value="coffee",
            kind="status",
            polarity="negative",
        )],
        [_relation(declined, "explicitly decline coffee", "contradicts", coffee_id)],
    )
    declined_state = query_assertion_state(
        store,
        subject_key="user:self",
        predicate_key="drink.choice.coffee",
        kinds=["recommendation"],
    )
    assert declined_state.assertions[0]["semantic_state"] == "declined"
    assert declined_state.conflict_assertion_ids == (coffee_id,)

    _recommendation(messages, store, "water", 300.0)
    still_unanswered = query_assertion_state(
        store,
        subject_key="user:self",
        predicate_key="drink.choice.water",
    )
    assert still_unanswered.assertions[0]["semantic_state"] == "unanswered"


def test_recommendation_disposition_requires_user_state_and_role_alignment(state_db):
    messages, store = state_db
    _recommendation_source, recommendation_id = _recommendation(
        messages, store, "tea", 100.0
    )
    assistant_confirmation = _source(
        messages,
        store,
        "I confirm my own tea recommendation.",
        110.0,
        role="assistant",
    )
    wire = _wire(
        assistant_confirmation,
        "confirm my own tea recommendation",
        subject="user:self",
        resolution="addressee",
        predicate="drink.choice.tea",
        value="tea",
        kind="status",
    )
    with pytest.raises(ValueError, match="explicit user state"):
        parse_assertion_extraction(
            assistant_confirmation,
            _payload(
                assistant_confirmation,
                [wire],
                [_relation(
                    assistant_confirmation,
                    "confirm my own tea recommendation",
                    "confirms",
                    recommendation_id,
                )],
            ),
            store=store,
        )


def test_commitment_fulfillment_is_explicit_typed_and_same_predicate(state_db):
    messages, store = state_db
    promise = _source(messages, store, "I will submit the report.", 100.0)
    promise_result = _publish(store, promise, [_wire(
        promise,
        "submit the report",
        predicate="report.submission",
        value="promised",
        kind="commitment",
    )])
    promise_id = promise_result.assertion_ids[0]
    pending = query_assertion_state(
        store,
        subject_key="user:self",
        predicate_key="report.submission",
        kinds=["commitment"],
    )
    assert pending.assertions[0]["semantic_state"] == "pending"

    completed = _source(messages, store, "I submitted the report.", 200.0)
    _publish(
        store,
        completed,
        [_wire(
            completed,
            "submitted the report",
            predicate="report.submission",
            value="completed",
            kind="action",
        )],
        [_relation(completed, "submitted the report", "fulfills", promise_id)],
    )
    fulfilled = query_assertion_state(
        store,
        subject_key="user:self",
        predicate_key="report.submission",
        kinds=["commitment"],
    )
    assert fulfilled.assertions[0]["semantic_state"] == "fulfilled"
    assert fulfilled.assertions[0]["active"] is False
    assert fulfilled.assertions[0]["source_quote"] == "submit the report"
    assert len(fulfilled.relations) == 1

    unrelated = _source(messages, store, "I washed the car.", 300.0)
    unrelated_wire = _wire(
        unrelated,
        "washed the car",
        predicate="car.washing",
        value="completed",
        kind="action",
    )
    with pytest.raises(ValueError, match="another subject or predicate"):
        parse_assertion_extraction(
            unrelated,
            _payload(
                unrelated,
                [unrelated_wire],
                [_relation(unrelated, "washed the car", "fulfills", promise_id)],
            ),
            store=store,
        )


def test_fulfillment_rejects_non_action_evidence_and_non_commitment_targets(state_db):
    messages, store = state_db
    target = _source(messages, store, "I will drink tea.", 100.0)
    commitment_id = _publish(store, target, [_wire(
        target,
        "drink tea",
        predicate="drink.tea",
        value="promised",
        kind="commitment",
    )]).assertion_ids[0]
    preference = _source(messages, store, "I prefer tea.", 200.0)
    with pytest.raises(ValueError, match="action/event/status"):
        parse_assertion_extraction(
            preference,
            _payload(
                preference,
                [_wire(
                    preference,
                    "prefer tea",
                    predicate="drink.tea",
                    value="tea",
                    kind="preference",
                )],
                [_relation(preference, "prefer tea", "fulfills", commitment_id)],
            ),
            store=store,
        )

    _recommendation_source, recommendation_id = _recommendation(
        messages, store, "coffee", 300.0
    )
    action = _source(messages, store, "I chose coffee.", 400.0)
    with pytest.raises(ValueError, match="targeting a commitment"):
        parse_assertion_extraction(
            action,
            _payload(
                action,
                [_wire(
                    action,
                    "chose coffee",
                    predicate="drink.choice.coffee",
                    value="coffee",
                    kind="action",
                )],
                [_relation(action, "chose coffee", "fulfills", recommendation_id)],
            ),
            store=store,
        )


def test_scoped_third_party_preference_preserves_role_strength_and_exact_span(state_db):
    messages, store = state_db
    snapshot = _source(
        messages,
        store,
        "Alex says he strongly prefers tea in the morning.",
        100.0,
    )
    _publish(store, snapshot, [_wire(
        snapshot,
        "Alex says he strongly prefers tea in the morning",
        subject="person:alex",
        resolution="explicit",
        predicate="drink.preference",
        value="tea",
        kind="preference",
        strength=0.9,
        scope="morning",
    )])
    state = query_assertion_state(
        store,
        subject_key="person:alex",
        predicate_key="drink.preference",
        kinds=["preference"],
        scope_key="morning",
    )
    row = state.assertions[0]
    assert row["speaker_role"] == "user"
    assert row["attribution"] == "third_party"
    assert row["strength"] == 0.9
    assert row["semantic_state"] == "current"
    assert snapshot.content[row["source_span_start"]:row["source_span_end"]] == row[
        "source_quote"
    ]


def test_addressee_resolution_fails_closed_outside_direct_user_assistant_roles(state_db):
    messages, store = state_db
    snapshot = _source(
        messages,
        store,
        "The tool says you should choose tea.",
        100.0,
        role="tool",
    )
    wire = _wire(
        snapshot,
        "you should choose tea",
        subject="user:self",
        resolution="addressee",
        predicate="drink.choice.tea",
        value="tea",
        kind="recommendation",
    )
    with pytest.raises(ValueError, match="addressee identity"):
        parse_assertion_extraction(
            snapshot,
            _payload(snapshot, [wire]),
            store=store,
        )


def test_lcm_query_state_is_production_bounded_and_exact_source_cited(tmp_path):
    disabled = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "disabled.db"),
    ))
    enabled = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "enabled.db"),
        assertions_enabled=True,
    ))
    try:
        assert "lcm_query_state" not in {
            schema["name"] for schema in disabled.get_tool_schemas()
        }
        disabled_result = json.loads(disabled.handle_tool_call(
            "lcm_query_state", {"subject_key": "user:self"}
        ))
        assert disabled_result["status"] == "disabled"

        enabled._session_id = "session-a"
        content = "I prefer tea in the morning."
        store_id = enabled._store.append(
            "session-a", {"role": "user", "content": content}
        )
        snapshot = enabled._assertions.snapshot_source(store_id)
        _publish(enabled._assertions, snapshot, [_wire(
            snapshot,
            "prefer tea in the morning",
            predicate="drink.preference",
            value="tea",
            kind="preference",
            scope="morning",
        )])
        response_text = enabled.handle_tool_call("lcm_query_state", {
            "subject_key": "user:self",
            "predicate_key": "drink.preference",
            "kinds": ["preference"],
            "scope_key": "morning",
            "limit": 500,
        })
        response = json.loads(response_text)
        assert response["status"] == "ok"
        assert response["limit"] == 50
        assert response["limit_clamped_from"] == 500
        assert response["provenance"]["recency_resolution"] == "disabled"
        ref = response["assertions"][0]["source_ref"]
        assert ref["store_id"] == store_id
        assert content[ref["span_start"]:ref["span_end"]] == ref["quote"]
        assert len(ref["content_sha256"]) == 64

        bad_time = json.loads(enabled.handle_tool_call(
            "lcm_query_state",
            {"subject_key": "user:self", "as_of": "2024-01-01"},
        ))
        assert "timezone" in bad_time["error"]
    finally:
        enabled.shutdown()
        disabled.shutdown()


def test_lcm_query_state_response_cap_omits_whole_rows_never_partial_quotes(tmp_path):
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        assertions_enabled=True,
    ))
    engine._session_id = "session-a"
    try:
        for index in range(4):
            content = f"row-{index}:" + chr(ord("a") + index) * 20_000
            store_id = engine._store.append(
                "session-a", {"role": "user", "content": content}
            )
            snapshot = engine._assertions.snapshot_source(store_id)
            engine._assertions.publish_source(snapshot, [AssertionCandidate(
                source_span_start=0,
                source_span_end=len(content),
                subject_key="user:self",
                predicate_key="large.fact",
                object_value=index,
                value_text=str(index),
                kind="fact",
            )])

        response_text = engine.handle_tool_call(
            "lcm_query_state",
            {"subject_key": "user:self", "predicate_key": "large.fact"},
        )
        response = json.loads(response_text)
        assert len(response_text) <= 64_000
        assert response["response_truncated"] is True
        assert response["assertions_omitted_by_response_cap"] >= 1
        assert 0 < len(response["assertions"]) < 4
        for row in response["assertions"]:
            ref = row["source_ref"]
            raw = engine._store.get(ref["store_id"])["content"]
            assert raw[ref["span_start"]:ref["span_end"]] == ref["quote"]
    finally:
        engine.shutdown()
