"""Bounded one-turn retrieval controller and warm evidence-view fixtures."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from hermes_lcm.adaptive_retrieval import (
    MAX_CANDIDATE_REFS,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_TOKENS,
    MAX_REQUIREMENTS,
    MAX_RETRIEVAL_ROUNDS,
    EvidenceRequirement,
    requirements_digest,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
import hermes_lcm.tools as lcm_tools


def _engine(tmp_path, *, enabled=True, name="lcm.db") -> LCMEngine:
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / name),
            adaptive_retrieval_enabled=enabled,
        )
    )
    engine.on_session_start("session-a", conversation_id="conversation-a")
    return engine


def _append(engine: LCMEngine, content: str, *, day="2024-03-01") -> int:
    store_id = engine._store.append("session-a", {
        "role": "user",
        "content": content,
        "source": "cli",
        "conversation_id": "conversation-a",
    })
    timestamp = datetime.fromisoformat(
        f"{day}T12:00:00+00:00"
    ).astimezone(timezone.utc).timestamp()
    engine._store._conn.execute(
        "UPDATE messages SET timestamp=? WHERE store_id=?", (timestamp, store_id)
    )
    engine._store._conn.commit()
    return store_id


def _identity(*, operation="evidence_only", **changes):
    value = {
        "intent_type": "travel memory",
        "operation": operation,
        "subject_key": "user:self",
        "predicate_key": "travel.visit",
        "scope_key": "personal",
        "conversation_id": "conversation-a",
        "time_mode": "none",
        "policy_version": "adaptive-v1",
    }
    value.update(changes)
    return value


def _requirements(*, minimum_refs=1, description="visited places"):
    return [{
        "slot_id": "visits",
        "description": description,
        "minimum_refs": minimum_refs,
    }]


def _call(engine: LCMEngine, **args):
    return json.loads(engine.handle_tool_call("lcm_retrieve", args))


def _start(
    engine: LCMEngine,
    *,
    question="Where did I travel?",
    question_date=None,
    operation="evidence_only",
    minimum_refs=1,
    description="visited places",
    identity_changes=None,
):
    identity = _identity(operation=operation, **(identity_changes or {}))
    args = {
        "action": "start",
        "question": question,
        "identity": identity,
        "requirements": _requirements(
            minimum_refs=minimum_refs, description=description
        ),
    }
    if question_date is not None:
        args["question_date"] = question_date
    return _call(
        engine,
        **args,
    )


def _search(engine: LCMEngine, retrieval_id: str, store_id: int, **extra):
    return _call(
        engine,
        action="search",
        retrieval_id=retrieval_id,
        missing_slot="visits",
        tool="lcm_expand",
        tool_args={"store_id": store_id},
        **extra,
    )


def _operand(evidence: dict, *, key: str):
    return {
        "store_id": evidence["store_id"],
        "span_start": evidence["span_start"],
        "span_end": evidence["span_end"],
        "quote": evidence["quote"],
        "key": key,
        "label": key,
    }


def test_retrieve_schema_matches_strict_runtime_contract():
    from hermes_lcm.schemas import LCM_RETRIEVE

    parameters = LCM_RETRIEVE["parameters"]
    assert parameters["additionalProperties"] is False
    assert "requirements" in parameters["properties"]

    operand = parameters["properties"]["computation"]["properties"]["operands"]["items"]
    assert operand["additionalProperties"] is False
    assert operand["required"] == ["quote"]
    assert {"assertion_id", "store_id", "span_start", "span_end", "quote"} <= set(
        operand["properties"]
    )
    assert {tuple(branch["required"]) for branch in operand["anyOf"]} == {
        ("assertion_id",),
        ("store_id", "span_start", "span_end"),
    }


def test_default_off_and_env_opt_in_are_provider_neutral(monkeypatch, tmp_path):
    monkeypatch.delenv("LCM_ADAPTIVE_RETRIEVAL_ENABLED", raising=False)
    assert LCMConfig().adaptive_retrieval_enabled is False
    assert LCMConfig.from_env().adaptive_retrieval_enabled is False

    disabled = _engine(tmp_path, enabled=False, name="disabled.db")
    try:
        assert disabled._adaptive_retrieval is None
        assert disabled._query_views is None
        assert _call(disabled, action="status", retrieval_id="missing")[
            "status"
        ] == "disabled"
    finally:
        disabled.shutdown()

    def provider_call_forbidden(*args, **kwargs):
        raise AssertionError("controller initialization must not call a provider")

    monkeypatch.setattr(lcm_tools, "resolve_provider", provider_call_forbidden)
    monkeypatch.setenv("LCM_ADAPTIVE_RETRIEVAL_ENABLED", "true")
    config = LCMConfig.from_env()
    assert config.adaptive_retrieval_enabled is True
    enabled = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "enabled.db"),
        adaptive_retrieval_enabled=True,
    ))
    try:
        assert enabled._adaptive_retrieval is not None
        assert enabled._query_views is not None
    finally:
        enabled.shutdown()


def test_exact_slot_closure_compute_finish_and_warm_reuse(tmp_path):
    engine = _engine(tmp_path)
    try:
        paris_id = _append(engine, "I visited Paris.")
        rome_id = _append(engine, "I visited Rome.")
        started = _start(
            engine,
            question="How many distinct cities did I visit?",
            operation="count_distinct",
            minimum_refs=2,
        )
        assert started["status"] == "active"
        assert started["query_view"]["status"] == "miss"
        retrieval_id = started["retrieval_id"]

        first = _search(engine, retrieval_id, paris_id)
        assert first["status"] == "active"
        assert len(first["evidence"]) == 1
        first_ref = first["evidence"][0]["citation"]
        status = _call(
            engine, action="status", retrieval_id=retrieval_id
        )
        assert [item["citation"] for item in status["evidence"]] == [first_ref]

        second = _search(
            engine,
            retrieval_id,
            rome_id,
            resolved_slots=[{
                "slot_id": "visits",
                "evidence_refs": [first_ref],
            }],
        )
        assert second["status"] == "active"
        second_ref = second["evidence"][0]["citation"]
        evidence = {
            first_ref: first["evidence"][0],
            second_ref: second["evidence"][0],
        }

        finished = _call(
            engine,
            action="finish",
            retrieval_id=retrieval_id,
            resolved_slots=[{
                "slot_id": "visits",
                "evidence_refs": [second_ref],
            }],
            selected_refs=[first_ref, second_ref],
            computation={
                "operands": [
                    _operand(evidence[first_ref], key="Paris"),
                    _operand(evidence[second_ref], key="Rome"),
                ]
            },
        )
        assert finished["status"] == "finished"
        assert finished["computation"]["status"] == "computed"
        assert finished["computation"]["trace"]["result_value"] == 2
        assert finished["query_view"]["persistence"]["status"] == "published"

        state = engine._adaptive_retrieval._states[retrieval_id]
        persisted = engine._query_views.lookup(state.identity, record_hit=False)
        assert persisted.status == "hit"
        encoded_manifest = json.dumps(persisted.view["manifest"]).casefold()
        assert "final_answer" not in encoded_manifest
        assert "candidate_answer" not in encoded_manifest
        assert "answer" not in persisted.view["computation_trace"]

        # Reword the literal question but keep the requirement description
        # identical to the original build: warm reuse must survive rephrasing
        # the QUESTION, but requirements_digest() now folds in description
        # (F-PR436-5), so varying the description here would correctly bust
        # the cache -- that is exercised separately in
        # test_cached_view_is_not_reused_across_different_requirement_descriptions.
        warm = _start(
            engine,
            question="How many different cities have I visited?",
            operation="count_distinct",
            minimum_refs=2,
        )
        assert warm["status"] == "ready"
        assert warm["query_view"]["status"] == "hit"
        assert len(warm["evidence"]) == 2
        assert warm["query_view"]["cached_computation_trace"][
            "result_value"
        ] == 2

        different_cardinality = _start(
            engine,
            question="Name a city I visited.",
            operation="count_distinct",
            minimum_refs=1,
        )
        assert different_cardinality["status"] == "active"
        assert different_cardinality["query_view"]["status"] == "miss"
    finally:
        engine.shutdown()


def test_low_trust_computation_persists_and_replays_temporal_trust(tmp_path):
    engine = _engine(tmp_path)
    try:
        store_id = _append(
            engine,
            "I finished the kanban dashboard sprint 5 days ago.",
            day="2024-03-20",
        )
        started = _start(
            engine,
            question="How long ago did I finish the kanban dashboard sprint?",
            question_date="2024-03-20",
            operation="date_interval",
            identity_changes={
                "time_mode": "relative",
                "window_start": "2024-03-15",
                "window_end": "2024-03-16",
            },
        )
        searched = _search(engine, started["retrieval_id"], store_id)
        evidence = searched["evidence"][0]
        citation = evidence["citation"]
        finished = _call(
            engine,
            action="finish",
            retrieval_id=started["retrieval_id"],
            resolved_slots=[{
                "slot_id": "visits",
                "evidence_refs": [citation],
            }],
            selected_refs=[citation],
            computation={
                "operands": [{
                    "store_id": evidence["store_id"],
                    "span_start": evidence["span_start"],
                    "span_end": evidence["span_end"],
                    "quote": evidence["quote"],
                    "date": "2024-03-15",
                    "occurrence_time": {
                        "event_date": "2024-03-15",
                        "event_time_source": "relative_to_session",
                        "session_date": "2024-03-20",
                    },
                }]
            },
        )
        fresh_trust = finished["computation"]["temporal_trust"]
        assert fresh_trust["status"] == "low_trust"
        assert fresh_trust["certified"] is False
        assert finished["query_view"]["persistence"]["status"] == "published"

        warm = _start(
            engine,
            question="When was that kanban dashboard sprint completed?",
            question_date="2024-03-20",
            operation="date_interval",
            identity_changes={
                "time_mode": "relative",
                "window_start": "2024-03-15",
                "window_end": "2024-03-16",
            },
        )
        assert warm["status"] == "ready"
        assert warm["query_view"]["status"] == "hit"
        assert (
            warm["query_view"]["cached_computation_trace"]["temporal_trust"]
            == fresh_trust
        )
    finally:
        engine.shutdown()


def test_persisted_slot_refs_include_only_selected_evidence(tmp_path):
    engine = _engine(tmp_path)
    try:
        paris_id = _append(engine, "I visited Paris.")
        rome_id = _append(engine, "I visited Rome.")
        started = _start(engine)
        found = _call(
            engine,
            action="search",
            retrieval_id=started["retrieval_id"],
            missing_slot="visits",
            tool="lcm_load_session",
            tool_args={"session_id": "session-a", "limit": 8},
        )
        refs_by_store_id = {
            item["store_id"]: item["citation"] for item in found["evidence"]
        }
        first_ref = refs_by_store_id[paris_id]
        second_ref = refs_by_store_id[rome_id]

        finished = _call(
            engine,
            action="finish",
            retrieval_id=started["retrieval_id"],
            resolved_slots=[{
                "slot_id": "visits",
                "evidence_refs": [first_ref, second_ref],
            }],
            selected_refs=[first_ref],
        )
        assert finished["query_view"]["persistence"]["status"] == "published"

        warm = _start(engine)
        assert warm["status"] == "ready"
        assert warm["query_view"]["status"] == "hit"
        assert [item["citation"] for item in warm["evidence"]] == [first_ref]
    finally:
        engine.shutdown()


def test_query_view_cleanup_cannot_replace_the_build_failure(
    tmp_path, monkeypatch
):
    engine = _engine(tmp_path)
    try:
        store_id = _append(engine, "I visited Paris.")
        started = _start(engine)
        found = _search(engine, started["retrieval_id"], store_id)
        exact_ref = found["evidence"][0]["citation"]

        def fail_publish(*args, **kwargs):
            raise ValueError("publish failed")

        def fail_cleanup(*args, **kwargs):
            raise RuntimeError("cleanup failed")

        monkeypatch.setattr(engine._query_views, "publish_ready", fail_publish)
        monkeypatch.setattr(engine._query_views, "mark_failed", fail_cleanup)
        finished = _call(
            engine,
            action="finish",
            retrieval_id=started["retrieval_id"],
            resolved_slots=[{
                "slot_id": "visits",
                "evidence_refs": [exact_ref],
            }],
            selected_refs=[exact_ref],
        )

        persistence = finished["query_view"]["persistence"]
        assert persistence["status"] == "failed"
        assert persistence["reason"] == "publish failed"
    finally:
        engine.shutdown()


def test_requirements_digest_distinguishes_descriptions():
    """requirements_digest() must not collide two requirements that share
    slot_id/minimum_refs but describe different evidence -- description is
    what defines the slot's meaning (F-PR436-5)."""
    ceo = EvidenceRequirement.parse(
        {"slot_id": "role_holder", "description": "the CEO of Acme", "minimum_refs": 1}
    )
    cfo = EvidenceRequirement.parse(
        {"slot_id": "role_holder", "description": "the CFO of Acme", "minimum_refs": 1}
    )
    assert requirements_digest([ceo]) != requirements_digest([cfo])


def test_requirement_description_rejects_surrogate_code_points():
    with pytest.raises(ValueError, match="surrogate code points"):
        EvidenceRequirement.parse({
            "slot_id": "role_holder",
            "description": "the CEO of \ud800Acme",
            "minimum_refs": 1,
        })


def test_max_sized_requirements_validate_and_digest(tmp_path):
    requirements = [
        {
            "slot_id": f"slot{index:02d}" + "x" * (64 - len(f"slot{index:02d}")),
            "description": "d" * 256,
            "minimum_refs": 1,
        }
        for index in range(MAX_REQUIREMENTS)
    ]

    parsed = [EvidenceRequirement.parse(item) for item in requirements]

    assert len({item.slot_id for item in parsed}) == MAX_REQUIREMENTS
    assert len(requirements_digest(parsed)) == 64
    engine = _engine(tmp_path)
    try:
        started = _call(
            engine,
            action="start",
            question="Find the requested evidence.",
            identity=_identity(),
            requirements=requirements,
        )
        assert started["status"] == "active"
    finally:
        engine.shutdown()


def test_cached_view_is_not_reused_across_different_requirement_descriptions(tmp_path):
    """Same identity + same slot_id + same minimum_refs, but a DIFFERENT
    requirement description, must be a cache miss -- otherwise start()
    pre-fills an unrelated question's slot with stale evidence for a
    different meaning (F-PR436-5)."""
    engine = _engine(tmp_path)
    try:
        ceo_id = _append(engine, "Maya is the CEO of Acme.")
        _append(engine, "Ben is the CFO of Acme.")

        ceo_started = _start(
            engine,
            question="Who is the CEO of Acme?",
            description="the CEO of Acme",
        )
        assert ceo_started["query_view"]["status"] == "miss"
        found = _search(engine, ceo_started["retrieval_id"], ceo_id)
        ceo_citation = found["evidence"][0]["citation"]
        finished = _call(
            engine,
            action="finish",
            retrieval_id=ceo_started["retrieval_id"],
            resolved_slots=[{"slot_id": "visits", "evidence_refs": [ceo_citation]}],
            selected_refs=[ceo_citation],
            computation=None,
        )
        assert finished["query_view"]["persistence"]["status"] == "published"

        cfo_started = _start(
            engine,
            question="Who is the CFO of Acme?",
            description="the CFO of Acme",
        )
        assert cfo_started["query_view"]["status"] == "miss"
        assert ceo_citation not in [
            item["citation"] for item in cfo_started.get("evidence", [])
        ]
    finally:
        engine.shutdown()


def test_corpus_advance_requires_bounded_delta_search(tmp_path):
    engine = _engine(tmp_path)
    try:
        paris_id = _append(engine, "I visited Paris.")
        started = _start(engine)
        found = _search(engine, started["retrieval_id"], paris_id)
        citation = found["evidence"][0]["citation"]
        finished = _call(
            engine,
            action="finish",
            retrieval_id=started["retrieval_id"],
            resolved_slots=[{
                "slot_id": "visits",
                "evidence_refs": [citation],
            }],
            selected_refs=[citation],
        )
        assert finished["query_view"]["persistence"]["status"] == "published"

        _append(engine, "I also visited Rome.")
        stale = _start(engine)
        assert stale["status"] == "active"
        assert stale["query_view"]["status"] == "delta_required"
        assert stale["query_view"]["delta_events"]
        assert stale["evidence"] == []
    finally:
        engine.shutdown()


def test_corpus_advance_during_retrieval_rejects_view_publication(tmp_path):
    engine = _engine(tmp_path)
    try:
        paris_id = _append(engine, "I visited Paris.")
        started = _start(engine)
        found = _search(engine, started["retrieval_id"], paris_id)
        citation = found["evidence"][0]["citation"]

        _append(engine, "I also visited Rome.")
        finished = _call(
            engine,
            action="finish",
            retrieval_id=started["retrieval_id"],
            resolved_slots=[{
                "slot_id": "visits",
                "evidence_refs": [citation],
            }],
            selected_refs=[citation],
        )

        assert finished["query_view"]["persistence"]["status"] == "stale_before_publish"
    finally:
        engine.shutdown()


def test_no_progress_terminates_and_incomplete_finish_falls_back(tmp_path):
    engine = _engine(tmp_path)
    try:
        started = _start(engine)
        result = _search(engine, started["retrieval_id"], 999_999_999)
        assert result["status"] == "fallback"
        assert result["rounds"][0]["status"] == "no_progress"
        assert "no exact-evidence progress" in result["termination_reason"]

        finished = _call(
            engine,
            action="finish",
            retrieval_id=started["retrieval_id"],
        )
        assert finished["status"] == "fallback"
        assert "remain open" in finished["termination_reason"]
    finally:
        engine.shutdown()


def test_invalid_computation_returns_evidence_only_fallback(tmp_path):
    engine = _engine(tmp_path)
    try:
        store_id = _append(engine, "I spent $10 on lunch.")
        started = _start(
            engine,
            question="What was the average price?",
            operation="sum",
        )
        found = _search(engine, started["retrieval_id"], store_id)
        evidence = found["evidence"][0]
        citation = evidence["citation"]
        result = _call(
            engine,
            action="finish",
            retrieval_id=started["retrieval_id"],
            resolved_slots=[{
                "slot_id": "visits",
                "evidence_refs": [citation],
            }],
            selected_refs=[citation],
            computation={
                "operands": [{
                    **_operand(evidence, key="$10"),
                    "value": 10,
                    "unit": "usd",
                }]
            },
        )
        assert result["status"] == "fallback"
        assert result["next_path"] == "evidence_only"
        assert result["computation"]["status"] == "fallback"
        assert result["query_view"]["persistence"]["status"] == "skipped"
        assert result["evidence"][0]["citation"] == citation
    finally:
        engine.shutdown()


def test_summary_handle_is_bounded_lead_progress_not_evidence(
    monkeypatch, tmp_path
):
    engine = _engine(tmp_path)
    try:
        def summary_only(args, *, engine):
            return json.dumps({
                "sections": [{
                    "node_id": "node-123",
                    "session_id": "session-a",
                    "summary": "Travel was discussed, but exact sources need expansion.",
                    "expand_hint": "lcm_expand(node_id=node-123)",
                }],
                "provenance": {
                    "provider": "voyage",
                    "model": "voyage-code-3",
                    "api_key": "must-not-escape",
                },
                "metrics": {"query_count": 1},
            })

        monkeypatch.setattr(lcm_tools, "lcm_recent", summary_only)
        started = _start(engine)
        result = _call(
            engine,
            action="search",
            retrieval_id=started["retrieval_id"],
            missing_slot="visits",
            tool="lcm_recent",
            tool_args={"period": "month", "scope": "global"},
        )
        assert result["status"] == "active"
        assert result["evidence"] == []
        assert result["rounds"][0]["status"] == "lead_progress"
        assert result["rounds"][0]["new_lead_count"] >= 1
        assert result["rounds"][0]["tool_provenance"] == {
            "model": "voyage-code-3",
            "provider": "voyage",
        }
        assert result["rounds"][0]["tool_metrics"]["query_count"] == 1
        assert result["leads"][0]["evidence_eligible"] is False
        assert result["leads"][0]["node_id"] == "node-123"
        assert result["requirements"][0]["closed"] is False
    finally:
        engine.shutdown()


def test_round_budget_and_session_ownership_are_enforced(tmp_path):
    engine = _engine(tmp_path)
    try:
        store_ids = [_append(engine, f"I visited City{index}.") for index in range(4)]
        started = _start(engine, minimum_refs=4)
        retrieval_id = started["retrieval_id"]
        resolved = []
        for index in range(MAX_RETRIEVAL_ROUNDS):
            result = _search(
                engine,
                retrieval_id,
                store_ids[index],
                resolved_slots=(
                    [{"slot_id": "visits", "evidence_refs": resolved}]
                    if resolved
                    else []
                ),
            )
            resolved.append(result["evidence"][0]["citation"])
        assert result["budgets"]["exhausted"] is True
        assert result["budgets"]["retrieval_rounds"] == MAX_RETRIEVAL_ROUNDS
        over_budget = _search(
            engine,
            retrieval_id,
            store_ids[-1],
            resolved_slots=[{
                "slot_id": "visits",
                "evidence_refs": resolved,
            }],
        )
        assert over_budget["status"] == "error"
        assert "budget exhausted" in over_budget["error"]

        another = _start(engine)
        engine.on_session_start("session-b", conversation_id="conversation-b")
        foreign = _call(
            engine,
            action="status",
            retrieval_id=another["retrieval_id"],
        )
        assert foreign["status"] == "error"
        assert "another active session" in foreign["error"]
    finally:
        engine.shutdown()


def test_profile_rebind_clears_ephemeral_controller_state(tmp_path):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    engine = LCMEngine(
        config=LCMConfig(adaptive_retrieval_enabled=True),
        hermes_home=str(home_a),
    )
    try:
        engine.on_session_start("session-a", conversation_id="conversation-a")
        started = _start(engine)
        assert engine._adaptive_retrieval._states

        assert engine._rebind_storage_for_home(str(home_b)) is True
        assert engine._adaptive_retrieval._states == {}
        expired = _call(
            engine,
            action="status",
            retrieval_id=started["retrieval_id"],
        )
        assert expired["status"] == "error"
        assert "unknown or expired" in expired["error"]
        assert engine._query_views.db_path == home_b / "lcm.db"
    finally:
        engine.shutdown()


def test_candidate_and_context_caps_apply_before_return(monkeypatch, tmp_path):
    engine = _engine(tmp_path)
    try:
        content = "x" * 2_400
        store_ids = [_append(engine, content) for _ in range(25)]

        def many_exact_refs(args, *, engine):
            return json.dumps({
                "results": [
                    {"store_id": store_id, "content": content, "content_offset": 0}
                    for store_id in store_ids
                ]
            })

        monkeypatch.setattr(lcm_tools, "lcm_query_state", many_exact_refs)
        started = _start(engine, minimum_refs=25)
        result = _call(
            engine,
            action="search",
            retrieval_id=started["retrieval_id"],
            missing_slot="visits",
            tool="lcm_query_state",
            tool_args={"subject_key": "user:self", "limit": 500},
        )
        assert result["budgets"]["candidate_refs"] <= MAX_CANDIDATE_REFS
        assert result["budgets"]["context_tokens"] <= MAX_CONTEXT_TOKENS
        assert result["budgets"]["context_chars"] <= MAX_CONTEXT_CHARS
        assert result["budgets"]["exhausted"] is True
        assert result["rounds"][0]["result_truncated"] is True
        assert result["rounds"][0]["tool_args"]["limit"] == 20
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ({"question_id": "forbidden"}, "unsupported lcm_retrieve arguments"),
        ({"identity": {"intent_type": "travel", "reference_answer": "Paris"}},
         "unsupported fields"),
    ],
)
def test_benchmark_metadata_and_unknown_fields_fail_closed(
    tmp_path, mutation, expected
):
    engine = _engine(tmp_path)
    try:
        args = {
            "action": "start",
            "question": "Where did I travel?",
            "identity": _identity(),
            "requirements": _requirements(),
            **mutation,
        }
        result = _call(engine, **args)
        assert result["status"] == "error"
        assert expected in result["error"]
    finally:
        engine.shutdown()
