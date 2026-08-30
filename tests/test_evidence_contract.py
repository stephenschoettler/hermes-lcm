"""Provider-free V4.6.3 evidence-contract compiler fixtures."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.evidence_compiler import compile_preanswer_evidence
from hermes_lcm.requirements_compiler import (
    _finite_event_key,
    _source_event_clause,
)
from hermes_lcm.store import MessageStore
from hermes_lcm.tools import lcm_compile_evidence
from hermes_lcm.schemas import LCM_COMPILE_EVIDENCE


def _engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    store = MessageStore(config.database_path, ingest_protection_config=config)
    return SimpleNamespace(
        _config=config,
        _store=store,
        _assertions=None,
        _session_occurrence_dates={},
    )


def _append(
    engine,
    content: str,
    *,
    session_id: str = "session-a",
    role: str = "user",
    timestamp: float | None = None,
):
    message = {"role": role, "content": content}
    if timestamp is not None:
        message["timestamp"] = timestamp
    store_id = engine._store.append(session_id, message)
    return {
        "exact_ref": f"lcm:{store_id}:0-{len(content)}",
        "quote": content,
    }


def _compile(engine, question, refs, **kwargs):
    return compile_preanswer_evidence(
        question,
        engine=engine,
        baseline_refs=refs,
        enabled=True,
        **kwargs,
    )


def test_unique_source_asserted_scalar_is_answer_sufficient_without_retrieval(tmp_path):
    engine = _engine(tmp_path)
    source = _append(engine, "You need 15 points to redeem the reward.")
    calls = []
    try:
        result = _compile(
            engine,
            "How many points do I need to redeem the reward?",
            [source],
            retrieve=lambda args: calls.append(args),
        )
    finally:
        engine._store.close()

    assert calls == []
    assert result["status"] == "no_augmentation"
    assert result["state"] == "answer_sufficient"
    assert result["direct_fact"]["value"] == 15
    assert result["direct_fact"]["unit"] == "point"
    assert result["novel_exact_refs"] == []
    assert result["context"] is None
    assert result["provenance"]["selector_calls"] == 0


def test_competing_compatible_scalars_fail_to_unchanged_baseline(tmp_path):
    engine = _engine(tmp_path)
    first = _append(engine, "You need 15 points to redeem the reward.")
    second = _append(
        engine,
        "You need 20 points to redeem the reward.",
        session_id="session-b",
    )
    try:
        result = _compile(
            engine,
            "How many points do I need to redeem the reward?",
            [first, second],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["status"] == "no_augmentation"
    assert result["reason_code"] == "ambiguous_scalar_candidates"
    assert result["context"] is None


def test_adjacent_role_partner_closes_named_scalar_slot(tmp_path):
    engine = _engine(tmp_path)
    lead = _append(engine, "How long is your commute?", role="user")
    _append(engine, "My commute is 35 minutes.", role="assistant")
    try:
        result = _compile(
            engine,
            "How long is my commute?",
            [lead],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "answer_sufficient"
    assert len(result["novel_exact_refs"]) == 1
    assert result["novel_exact_refs"][0].startswith("lcm:2:")
    assert result["metrics"]["session_loads"] == 1
    assert "35 minutes" in result["context"]


def test_fixed_named_sum_computes_unique_operands_with_unit_conversion(tmp_path):
    engine = _engine(tmp_path)
    jogging = _append(engine, "I spent 1 hour jogging.", session_id="jogging")
    yoga = _append(engine, "I spent 30 minutes doing yoga.", session_id="yoga")
    try:
        result = _compile(
            engine,
            "How many minutes total did I spend jogging and doing yoga?",
            [jogging, yoga],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "computation_sufficient", result
    assert result["computation"]["result_value"] == 90
    assert result["computation"]["unit"] == "minute"
    assert len(result["computation"]["citations"]) == 2


def test_instead_of_difference_uses_question_direction(tmp_path):
    engine = _engine(tmp_path)
    bus = _append(engine, "Taking the bus took 30 minutes.", session_id="bus")
    taxi = _append(engine, "Taking a taxi took 50 minutes.", session_id="taxi")
    try:
        result = _compile(
            engine,
            "How much time did I save by taking the bus instead of a taxi?",
            [bus, taxi],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "computation_sufficient"
    assert result["computation"]["result_value"] == 20
    assert result["computation"]["unit"] == "minute"


@pytest.mark.parametrize(
    ("question", "sources"),
    (
        ("How long is my commute?", ("My commute is 35 minutes.",)),
        (
            "How many minutes total did I spend jogging and doing yoga?",
            ("Jogging and yoga had a combined total of 90 minutes.",),
        ),
        (
            "What is the difference in minutes between the bus and taxi?",
            ("The bus and taxi difference is 20 minutes.",),
        ),
        ("What is my favorite color?", ("My favorite color is blue.",)),
    ),
)
def test_question_as_of_excludes_future_baseline_for_every_contract_operation(
    tmp_path,
    question,
    sources,
):
    engine = _engine(tmp_path)
    observed_after_cutoff = datetime(2024, 2, 1, tzinfo=timezone.utc).timestamp()
    refs = [
        _append(
            engine,
            source,
            session_id=f"future-{index}",
            timestamp=observed_after_cutoff,
        )
        for index, source in enumerate(sources)
    ]
    try:
        result = _compile(
            engine,
            question,
            refs,
            question_as_of="2024-01-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "unknown"
    assert result["direct_fact"] is None
    assert result["computation"] is None
    assert result["evidence"] == []


def test_open_cardinality_returns_one_event_count_uncertified(tmp_path):
    engine = _engine(tmp_path)
    source = _append(
        engine,
        "I visited Lisbon during my spring vacation.",
        timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp(),
    )
    try:
        result = _compile(
            engine,
            "How many vacations did I take this year?",
            [source],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "computation_sufficient"
    assert result["finite_coverage"] is False
    assert result["computation"]["result_value"] == 1
    assert result["reason_code"] == "finite_count_uncertified_undated_events"
    assert result["coverage_certificate"]["every_counted_event_dated"] is False
    assert result["coverage_certificate"]["every_counted_event_trusted"] is True
    assert "UNCERTIFIED" in result["context"]


def test_targeted_retrieval_admits_only_positive_slot_coverage(tmp_path):
    engine = _engine(tmp_path)
    baseline = _append(engine, "We discussed the commute yesterday.")
    useful = _append(
        engine,
        "My commute is 35 minutes.",
        session_id="commute-fact",
    )
    irrelevant = _append(
        engine,
        "The train station has a blue roof.",
        session_id="noise",
    )
    calls = []

    def retrieve(args):
        calls.append(args)
        return json.dumps(
            {
                "hits": [
                    {"exact_ref": irrelevant["exact_ref"], "content": irrelevant["quote"]},
                    {"exact_ref": useful["exact_ref"], "content": useful["quote"]},
                ]
            }
        )

    try:
        result = _compile(
            engine,
            "How long is my commute?",
            [baseline],
            retrieve=retrieve,
        )
    finally:
        engine._store.close()

    assert len(calls) == 1
    assert calls[0]["detail"] == "answer_ready"
    assert len(result["novel_exact_refs"]) == 1
    assert result["novel_exact_refs"][0].startswith("lcm:2:")
    assert all(not ref.startswith("lcm:3:") for ref in result["novel_exact_refs"])


def test_auto_tool_mode_calls_same_product_compiler_and_legacy_mode_stays_default(tmp_path):
    engine = _engine(tmp_path)
    source = _append(engine, "You need 15 points to redeem the reward.")
    try:
        auto = json.loads(
            lcm_compile_evidence(
                {
                    "mode": "auto",
                    "question": "How many points do I need to redeem the reward?",
                    "baseline_refs": [source],
                },
                engine=engine,
            )
        )
        legacy = json.loads(
            lcm_compile_evidence(
                {
                    "question": "How many points do I need to redeem the reward?",
                    "baseline_refs": [source],
                },
                engine=engine,
            )
        )
    finally:
        engine._store.close()

    assert auto["state"] == "answer_sufficient"
    assert auto["provenance"]["implementation"] == "compile_preanswer_evidence"
    assert legacy["reason_code"] == "selector_schema_invalid"


def test_bounded_trace_has_no_question_or_secret_payload(tmp_path):
    engine = _engine(tmp_path)
    source = _append(engine, "You need 15 points to redeem the reward.")
    try:
        result = _compile(
            engine,
            "How many points do I need to redeem the reward?",
            [source],
            budgets={"max_added_context_tokens": 850},
        )
    finally:
        engine._store.close()

    encoded_trace = json.dumps(result["trace"], sort_keys=True)
    assert "How many points" not in encoded_trace
    assert "secret" not in encoded_trace.casefold()
    assert len(result["evidence"]) <= 6
    assert result["metrics"]["added_context_tokens"] <= 850
    assert len(result["trace"]["digest_sha256"]) == 64


def test_public_schema_adds_auto_mode_without_changing_default():
    parameters = LCM_COMPILE_EVIDENCE["parameters"]
    assert parameters["properties"]["mode"] == {
        "type": "string",
        "enum": ["proposal", "auto"],
        "default": "proposal",
        "description": (
            "Default proposal mode preserves the legacy wire contract. "
            "Auto mode uses the provider-free requirements compiler."
        ),
    }
    assert parameters["required"] == ["question", "baseline_refs"]
    assert parameters["allOf"][0]["then"] == {"required": ["proposal"]}


def test_store_scan_is_one_bounded_snapshot_and_never_relabels_time(tmp_path):
    engine = _engine(tmp_path)
    first = _append(engine, "First row without a host timestamp.")
    observed = datetime(2025, 1, 2, tzinfo=timezone.utc).timestamp()
    second = _append(engine, "Second row with a host timestamp.", timestamp=observed)
    _append(engine, "Third row.")
    try:
        scan = engine._store.scan_evidence_rows(limit=2)
    finally:
        engine._store.close()

    assert [row["store_id"] for row in scan["rows"]] == [1, 2]
    assert scan["total_rows"] == 3
    assert scan["returned_rows"] == 2
    assert scan["snapshot_max_store_id"] == 3
    assert scan["truncated"] is True
    assert scan["observed_at_missing_rows"] == 2
    assert scan["rows"][0]["observed_at"] is None
    assert scan["rows"][0]["ingested_at"] is not None
    assert scan["rows"][1]["observed_at"] == observed
    assert first["exact_ref"].startswith("lcm:1:")
    assert second["exact_ref"].startswith("lcm:2:")


def test_temporal_event_selects_only_the_resolved_question_day(tmp_path):
    engine = _engine(tmp_path)
    wrong = _append(
        engine,
        "I started the reading challenge.",
        session_id="wrong-day",
    )
    right = _append(
        engine,
        "I completed the Plank Challenge.",
        session_id="right-day",
    )
    engine._session_occurrence_dates.update(
        {"wrong-day": "2024-03-14", "right-day": "2024-03-15"}
    )
    try:
        result = _compile(
            engine,
            "What happened five days ago?",
            [wrong, right],
            question_as_of="2024-03-20",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "answer_sufficient"
    assert result["context"] is None
    assert result["status"] == "no_augmentation"
    assert result["direct_fact"]["value"] == "I completed the Plank Challenge."
    assert result["finite_coverage"] is False
    assert result["direct_fact"]["time_basis"] == "adapter_session_date"


def test_temporal_event_with_wrong_or_unknown_time_falls_back(tmp_path):
    engine = _engine(tmp_path)
    wrong = _append(engine, "I completed the Plank Challenge.", session_id="wrong")
    unknown = _append(engine, "I also started yoga.", session_id="unknown")
    engine._session_occurrence_dates["wrong"] = "2024-03-14"
    try:
        result = _compile(
            engine,
            "What happened five days ago?",
            [wrong, unknown],
            question_as_of="2024-03-20",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["status"] == "no_augmentation"
    assert result["reason_code"] == "event_in_window_not_found"
    assert result["context"] is None


def test_latest_previous_and_future_state_are_time_bounded(tmp_path):
    engine = _engine(tmp_path)
    austin = _append(
        engine,
        "I live in Austin.",
        session_id="old",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp(),
    )
    denver = _append(
        engine,
        "I live in Denver.",
        session_id="current",
        timestamp=datetime(2024, 2, 1, tzinfo=timezone.utc).timestamp(),
    )
    future = _append(
        engine,
        "I live in Boston.",
        session_id="future",
        timestamp=datetime(2024, 4, 1, tzinfo=timezone.utc).timestamp(),
    )
    try:
        latest = _compile(
            engine,
            "Where do I currently live?",
            [austin, denver, future],
            question_as_of="2024-03-01",
            budgets={"max_retrieval_calls": 0},
        )
        previous = _compile(
            engine,
            "Where did I live previously?",
            [austin, denver, future],
            question_as_of="2024-03-01",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert latest["direct_fact"]["value"] == "Denver"
    assert previous["direct_fact"]["value"] == "Austin"
    assert latest["context"] is None
    assert previous["context"] is None


def test_explicit_location_transition_supports_previous_value(tmp_path):
    engine = _engine(tmp_path)
    transition = _append(
        engine,
        "I moved from Austin to Denver.",
        timestamp=datetime(2024, 2, 1, tzinfo=timezone.utc).timestamp(),
    )
    try:
        result = _compile(
            engine,
            "Where did I live previously?",
            [transition],
            question_as_of="2024-03-01",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["direct_fact"]["value"] == "Austin"


def test_adjacent_assistant_advice_is_the_only_admitted_evidence(tmp_path):
    engine = _engine(tmp_path)
    question = _append(engine, "What should I do about dry basil?", role="user")
    advice = _append(
        engine,
        "Water the basil in the morning and keep it near indirect light.",
        role="assistant",
    )
    try:
        result = _compile(
            engine,
            "What advice did you give me about dry basil?",
            [question],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "answer_sufficient"
    assert result["novel_exact_refs"] == [advice["exact_ref"]]
    assert "Water the basil" in result["context"]


def test_date_interval_and_fixed_order_use_unique_exact_operands(tmp_path):
    engine = _engine(tmp_path)
    first = _append(engine, "The first event was on 2024-01-02.", session_id="first")
    second = _append(engine, "The second event was on 2024-01-12.", session_id="second")
    try:
        interval = _compile(
            engine,
            "How many days passed between the first event and the second event?",
            [first, second],
            budgets={"max_retrieval_calls": 0},
        )
        ordered = _compile(
            engine,
            "What is the order of the two events from earliest to latest?",
            [second, first],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert interval["state"] == "computation_sufficient"
    assert interval["computation"]["result_value"] == 10
    assert ordered["state"] == "computation_sufficient"
    assert ordered["computation"]["result_value"][0].startswith("The first event")


def test_ordered_place_labels_ignore_leading_spelled_out_dates(tmp_path):
    engine = _engine(tmp_path)
    paris = _append(
        engine,
        "On June 1, 2024, I visited Paris.",
        session_id="paris",
        timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp(),
    )
    rome = _append(
        engine,
        "On July 2, 2024, I visited Rome.",
        session_id="rome",
        timestamp=datetime(2024, 7, 2, tzinfo=timezone.utc).timestamp(),
    )
    try:
        ordered = _compile(
            engine,
            "Which places did I visit in order?",
            [rome, paris],
            question_as_of="2024-08-01",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert ordered["state"] == "computation_sufficient"
    assert ordered["computation"]["result_value"] == ("Paris", "Rome")


def test_complete_finite_enumeration_counts_distinct_adapter_dated_events(tmp_path):
    engine = _engine(tmp_path)
    bali = _append(engine, "I took a vacation to Bali.", session_id="bali")
    kyoto = _append(engine, "I took a vacation to Kyoto.", session_id="kyoto")
    duplicate = _append(
        engine,
        "I took a vacation to Bali.",
        session_id="bali-repeat",
    )
    engine._session_occurrence_dates.update(
        {
            "bali": "2025-02-01",
            "kyoto": "2025-06-01",
            "bali-repeat": "2025-02-01",
        }
    )
    try:
        result = _compile(
            engine,
            "How many vacations did I take this year?",
            [bali, kyoto, duplicate],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "computation_sufficient"
    assert result["finite_coverage"] is True
    assert result["computation"]["result_value"] == 2
    assert result["coverage_certificate"]["distinct_keys"] == 2
    assert result["coverage_certificate"]["adapter_time_used"] is True


def test_finite_enumeration_rejects_unrelated_action_near_requested_event(tmp_path):
    engine = _engine(tmp_path)
    clause = "On June 1, 2024, I bought clothes for my vacation in Bali."
    assert not _source_event_clause(clause, role="user", unit="vacation")
    assert _source_event_clause(
        "On June 1, 2024, I took a vacation to Bali.",
        role="user",
        unit="vacation",
    )
    source = _append(
        engine,
        clause,
    )
    try:
        result = _compile(
            engine,
            "How many vacations did I take during June?",
            [source],
            question_as_of="2024-06-30",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "unknown"
    assert result["finite_coverage"] is False
    assert result["computation"] is None


def test_finite_enumeration_binds_question_action_to_counted_event(tmp_path):
    assert not _source_event_clause(
        "I bought a concert ticket.",
        role="user",
        unit="concert",
        question="How many concerts did I attend this year?",
    )
    assert _source_event_clause(
        "I attended a concert.",
        role="user",
        unit="concert",
        question="How many concerts did I attend this year?",
    )
    assert _source_event_clause(
        "I bought a book.",
        role="user",
        unit="book",
        question="How many books did I buy this year?",
    )
    assert not _source_event_clause(
        "I bought a vacation package for Bali.",
        role="user",
        unit="vacation",
        question="How many vacations did I take this year?",
    )
    assert _source_event_clause(
        "I took a vacation in Bali.",
        role="user",
        unit="vacation",
        question="How many vacations did I take this year?",
    )

    concert_engine = _engine(tmp_path / "concert")
    ticket = _append(
        concert_engine,
        "I bought an Aurora concert ticket.",
        session_id="ticket",
    )
    attended = _append(
        concert_engine,
        "I attended the Aurora concert.",
        session_id="attended",
    )
    concert_engine._session_occurrence_dates.update(
        {"ticket": "2025-01-01", "attended": "2025-01-02"}
    )
    try:
        concert_result = _compile(
            concert_engine,
            "How many concerts did I attend this year?",
            [ticket, attended],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        concert_engine._store.close()

    assert concert_result["state"] == "computation_sufficient", json.dumps(
        concert_result, indent=2
    )
    assert concert_result["computation"]["result_value"] == 1

    engine = _engine(tmp_path)
    purchased = _append(
        engine,
        "I bought a vacation package for Bali.",
        session_id="purchase",
    )
    taken = _append(
        engine,
        "I took a vacation in Bali.",
        session_id="taken",
    )
    engine._session_occurrence_dates.update(
        {"purchase": "2025-02-01", "taken": "2025-03-01"}
    )
    try:
        result = _compile(
            engine,
            "How many vacations did I take this year?",
            [purchased, taken],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "computation_sufficient"
    assert result["finite_coverage"] is True
    assert result["computation"]["result_value"] == 1


def test_terminal_question_date_fails_closed_in_evidence_path(tmp_path):
    engine = _engine(tmp_path)
    source = _append(engine, "You need 15 points to redeem the reward.")
    try:
        result = _compile(
            engine,
            "How many points do I need to redeem the reward?",
            [source],
            question_as_of="9999-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "unknown"
    assert result["computation"] is None


def test_relative_finite_event_without_sidecar_is_counted_but_uncertified(
    tmp_path,
):
    engine = _engine(tmp_path)
    observed_at = datetime(2025, 6, 2, tzinfo=timezone.utc).timestamp()
    source = _append(
        engine,
        "I took a vacation to Kyoto yesterday.",
        session_id="kyoto",
        timestamp=observed_at,
    )
    try:
        result = _compile(
            engine,
            "How many vacations did I take this year?",
            [source],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "computation_sufficient"
    assert result["computation"]["result_value"] == 1
    assert result["finite_coverage"] is False
    assert result["reason_code"] == "finite_count_uncertified_undated_events"
    assert result["coverage_certificate"]["every_counted_event_dated"] is True
    assert result["coverage_certificate"]["every_counted_event_trusted"] is False
    assert "UNCERTIFIED" in result["context"]


def test_relative_finite_event_without_observed_at_is_counted_but_uncertified(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "hermes_lcm.store.time.time",
        lambda: datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp(),
    )
    engine = _engine(tmp_path)
    source = _append(
        engine,
        "I took a vacation to Kyoto yesterday.",
        session_id="kyoto",
    )
    try:
        result = _compile(
            engine,
            "How many vacations did I take this year?",
            [source],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "computation_sufficient", result
    assert result["computation"]["result_value"] == 1
    assert result["finite_coverage"] is False
    assert result["reason_code"] == "finite_count_uncertified_undated_events"
    assert result["coverage_certificate"]["every_counted_event_dated"] is False
    assert result["coverage_certificate"]["every_counted_event_trusted"] is False
    assert "UNCERTIFIED" in result["context"]


def test_finite_enumeration_distinguishes_same_entity_events_by_date(tmp_path):
    engine = _engine(tmp_path)
    first = _append(engine, "I took a vacation to Bali.", session_id="first")
    second = _append(engine, "I took a vacation to Bali.", session_id="second")
    engine._session_occurrence_dates.update(
        {"first": "2025-02-01", "second": "2025-06-01"}
    )
    try:
        result = _compile(
            engine,
            "How many vacations did I take this year?",
            [first, second],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["finite_coverage"] is True
    assert result["computation"]["result_value"] == 2
    assert result["coverage_certificate"]["distinct_keys"] == 2
    assert result["coverage_certificate"]["every_counted_event_dated"] is True
    assert "UNCERTIFIED" not in result["context"]


def test_finite_event_key_preserves_date_suffix_for_overlong_base():
    names = " ".join(
        f"Place{chr(65 + index // 26)}{chr(65 + index % 26)}"
        for index in range(80)
    )
    quote = f"I took a vacation to {names}."

    first = _finite_event_key(quote, "vacation", "2025-02-01")
    second = _finite_event_key(quote, "vacation", "2025-06-01")

    assert first is not None and second is not None
    assert len(first) == len(second) == 300
    assert first.endswith(" @ 2025-02-01")
    assert second.endswith(" @ 2025-06-01")
    assert first != second


def test_finite_event_key_hashes_distinct_overlong_bases_on_same_date():
    common_names = " ".join(
        f"Place{chr(65 + index // 26)}{chr(65 + index % 26)}"
        for index in range(80)
    )
    first_quote = f"I took a vacation to {common_names} DestinationAlpha."
    second_quote = f"I took a vacation to {common_names} DestinationBeta."

    first = _finite_event_key(first_quote, "vacation", "2025-02-01")
    second = _finite_event_key(second_quote, "vacation", "2025-02-01")

    assert first is not None and second is not None
    assert len(first) == len(second) == 300
    assert first.endswith(" @ 2025-02-01")
    assert second.endswith(" @ 2025-02-01")
    assert first != second


def test_finite_enumeration_returns_dated_and_undated_count_uncertified(tmp_path):
    engine = _engine(tmp_path)
    known = _append(engine, "I took a vacation to Bali.", session_id="known")
    unknown = _append(
        engine,
        "I took a vacation to Kyoto.",
        session_id="unknown",
        timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp(),
    )
    engine._session_occurrence_dates["known"] = "2025-02-01"
    try:
        result = _compile(
            engine,
            "How many vacations did I take this year?",
            [known, unknown],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["finite_coverage"] is False
    assert result["reason_code"] == "finite_count_uncertified_undated_events", result
    assert result["state"] == "computation_sufficient", result
    assert result["computation"]["result_value"] == 2
    assert result["coverage_certificate"]["unknown_time_clauses"] == 1
    assert result["coverage_certificate"]["every_counted_event_dated"] is False
    assert "UNCERTIFIED" in result["context"]


def test_finite_enumeration_collapses_repeated_undated_mentions(tmp_path):
    engine = _engine(tmp_path)
    observed = datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp()
    first = _append(
        engine,
        "I took a vacation to Bali.",
        session_id="first",
        timestamp=observed,
    )
    second = _append(
        engine,
        "I took a vacation to Bali.",
        session_id="second",
        timestamp=observed,
    )
    try:
        result = _compile(
            engine,
            "How many vacations did I take this year?",
            [first, second],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["finite_coverage"] is False
    assert result["reason_code"] == "finite_count_uncertified_undated_events", result
    assert result["computation"] is not None, result
    assert result["computation"]["result_value"] == 1
    assert result["coverage_certificate"]["distinct_keys"] == 1
    assert result["coverage_certificate"]["unknown_time_clauses"] == 2


def test_finite_enumeration_rejects_explicitly_negated_undated_events(tmp_path):
    """Round-6.1 P1: "I took no vacation to Bali" is a NEGATION, not an event.
    The retention path must reject no/none/zero/without negations exactly like
    not/never — a negated statement can never contribute to a count."""
    engine = _engine(tmp_path)
    observed = datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp()
    negated = _append(
        engine,
        "I took no vacation to Bali.",
        session_id="negated",
        timestamp=observed,
    )
    real = _append(
        engine,
        "I took a vacation to Kyoto.",
        session_id="real",
        timestamp=observed,
    )
    try:
        result = _compile(
            engine,
            "How many vacations did I take this year?",
            [negated, real],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["computation"] is not None, result
    assert result["computation"]["result_value"] == 1, result
    assert result["coverage_certificate"]["distinct_keys"] == 1


def test_source_event_negation_binds_to_event_action_or_counted_unit():
    rejected = (
        "I took no vacation to Bali.",
        "I took zero vacations this year.",
        "I took none of my vacations with family.",
        "I went without taking a vacation this year.",
        "I never visited Bali on vacation.",
        "I did not take a vacation to Bali.",
        "I didn't take a vacation to Bali.",
        "I didn t take a vacation to Bali.",
        "We don't take vacations to Bali.",
        "We don t take vacations to Bali.",
        "I doesn't take a vacation to Bali.",
        "I doesn t take a vacation to Bali.",
        "We haven't taken a vacation to Bali.",
        "We haven t taken a vacation to Bali.",
        "I hasn't taken a vacation to Bali.",
        "I hasn t taken a vacation to Bali.",
    )
    admitted = (
        "I took a vacation with Don.",
        "I took a vacation with no checked bags.",
        "I took a vacation without my family.",
        "I took a vacation with no vacations planned afterward.",
        "I took a vacation to Havre.",
        "I took a vacation to Haven.",
    )

    assert all(
        not _source_event_clause(quote, role="user", unit="vacation")
        for quote in rejected
    )
    assert all(
        _source_event_clause(quote, role="user", unit="vacation")
        for quote in admitted
    )


def test_finite_enumeration_counts_available_and_excludes_postdated_event(tmp_path):
    engine = _engine(tmp_path)
    available = _append(
        engine,
        "I took a vacation to Kyoto.",
        session_id="available",
        timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp(),
    )
    postdated = _append(
        engine,
        "I took a vacation to Bali.",
        session_id="postdated",
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp(),
    )
    engine._session_occurrence_dates["available"] = "2025-06-01"
    try:
        result = _compile(
            engine,
            "How many vacations did I take this year?",
            [available, postdated],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["finite_coverage"] is True
    assert result["reason_code"] == "finite_coverage_product_verified"
    assert result["computation"]["result_value"] == 1
    assert result["coverage_certificate"]["unavailable_as_of_clauses"] == 1
    assert result["coverage_certificate"]["distinct_keys"] == 1
    assert postdated["exact_ref"] not in {
        item["exact_ref"] for item in result["evidence"]
    }


def test_finite_enumeration_excludes_future_event_with_available_observed_at(
    tmp_path,
):
    engine = _engine(tmp_path)
    observed = datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp()
    available = _append(
        engine,
        "I took a vacation to Kyoto.",
        session_id="available",
        timestamp=observed,
    )
    postdated = _append(
        engine,
        "I took a vacation to Bali.",
        session_id="postdated",
        timestamp=observed,
    )
    engine._session_occurrence_dates.update(
        {"available": "2025-06-01", "postdated": "2026-01-02"}
    )
    try:
        result = _compile(
            engine,
            "How many vacations did I take this year?",
            [available, postdated],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["finite_coverage"] is True
    assert result["reason_code"] == "finite_coverage_product_verified"
    assert result["computation"]["result_value"] == 1
    assert result["coverage_certificate"]["unavailable_as_of_clauses"] == 1
    assert postdated["exact_ref"] not in {
        item["exact_ref"] for item in result["evidence"]
    }


def test_finite_enumeration_all_unavailable_returns_no_count(tmp_path):
    engine = _engine(tmp_path)
    postdated = _append(
        engine,
        "I took a vacation to Bali.",
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp(),
    )
    try:
        result = _compile(
            engine,
            "How many vacations did I take this year?",
            [postdated],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["computation"] is None
    assert result["reason_code"] == "finite_no_events_in_window"
    assert result["coverage_certificate"]["unavailable_as_of_clauses"] == 1
    assert result["coverage_certificate"]["distinct_keys"] == 0


def test_finite_enumeration_rejects_truncated_raw_scan(tmp_path):
    engine = _engine(tmp_path)
    first = _append(engine, "I took a vacation to Bali.", session_id="bali")
    second = _append(engine, "I took a vacation to Kyoto.", session_id="kyoto")
    engine._session_occurrence_dates.update(
        {"bali": "2025-02-01", "kyoto": "2025-06-01"}
    )
    try:
        result = _compile(
            engine,
            "How many vacations did I take this year?",
            [first, second],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0, "max_scan_rows": 1},
        )
    finally:
        engine._store.close()

    assert result["finite_coverage"] is False
    assert result["reason_code"] == "finite_scan_truncated"
    assert result["coverage_certificate"]["truncated"] is True
    assert result["context"] is None


def test_invalid_and_secret_exact_refs_never_enter_evidence(tmp_path):
    engine = _engine(tmp_path)
    secret = _append(engine, "My api_key=super-secret-value is for the commute app.")
    try:
        result = _compile(
            engine,
            "How long is my commute?",
            [secret, {"exact_ref": "lcm:999:0-10", "quote": "not real"}],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["evidence"] == []
    assert result["context"] is None
    assert result["baseline"]["input_exact_ref_count"] == 2
    assert result["baseline"]["hydrated_exact_ref_count"] == 0


def test_global_hydration_cap_and_full_input_digest_are_honest(tmp_path):
    engine = _engine(tmp_path)
    refs = [
        _append(engine, f"The commute fact candidate {index} is {index} minutes.", session_id=f"s-{index}")
        for index in range(25)
    ]
    calls = []
    try:
        result = _compile(
            engine,
            "How long is my commute?",
            refs,
            retrieve=lambda args: calls.append(args),
        )
    finally:
        engine._store.close()

    assert result["baseline"]["input_exact_ref_count"] == 25
    assert result["baseline"]["hydrated_exact_ref_count"] == 12
    assert result["metrics"]["hydrated_candidates"] == 12
    assert result["metrics"]["candidate_count"] <= 48
    assert calls == []


def test_saturated_baseline_reserves_frontier_for_opposite_role_closure(tmp_path):
    engine = _engine(tmp_path)
    lead = _append(
        engine,
        "Could you remind me how long the ferry commute takes?",
        role="user",
    )
    _append(
        engine,
        "Your ferry commute takes 42 minutes.",
        role="assistant",
    )
    noise = [
        _append(engine, f"Unrelated garden note {index}.", session_id=f"noise-{index}")
        for index in range(11)
    ]
    try:
        result = _compile(
            engine,
            "How long is my ferry commute?",
            [*noise, lead, _append(engine, "Another unrelated note.", session_id="tail")],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "answer_sufficient"
    assert len(result["novel_exact_refs"]) == 1
    assert result["novel_exact_refs"][0].startswith("lcm:2:")
    assert result["metrics"]["hydrated_candidates"] <= 12
    assert result["metrics"]["session_loads"] >= 1


def test_saturated_baseline_can_retrieve_second_named_operand_within_cap(tmp_path):
    engine = _engine(tmp_path)
    walking = _append(engine, "The walking route takes 45 minutes.", session_id="walk")
    cycling = _append(engine, "The cycling route takes 20 minutes.", session_id="cycle")
    noise = [
        _append(engine, f"Unrelated recipe note {index}.", session_id=f"noise-{index}")
        for index in range(12)
    ]
    calls = []

    def retrieve(_args):
        calls.append(True)
        return {"hits": [cycling]}

    try:
        result = _compile(
            engine,
            "How much time do I save by cycling instead of walking?",
            [*noise, walking],
            retrieve=retrieve,
        )
    finally:
        engine._store.close()

    assert calls
    assert result["state"] == "computation_sufficient"
    assert result["computation"]["result_value"] == 25
    assert result["metrics"]["hydrated_candidates"] <= 12


def test_input_ref_outside_active_frontier_is_never_counted_as_novel(tmp_path):
    engine = _engine(tmp_path)
    noise = [
        _append(engine, f"Unrelated travel note {index}.", session_id=f"noise-{index}")
        for index in range(12)
    ]
    cached = _append(
        engine,
        "The ferry commute takes 42 minutes.",
        session_id="cached-rank-13",
    )

    def retrieve(_args):
        return {"hits": [cached]}

    try:
        result = _compile(
            engine,
            "How long is my ferry commute?",
            [*noise, cached],
            retrieve=retrieve,
        )
    finally:
        engine._store.close()

    assert cached["exact_ref"] not in result["novel_exact_refs"]


def test_noun_product_does_not_trigger_arithmetic_fallback(tmp_path):
    engine = _engine(tmp_path)
    source = _append(engine, "You need 80 credits to redeem the free product.")
    try:
        result = _compile(
            engine,
            "How many credits do I need to redeem the free product?",
            [source],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["contract"]["operation"] == "scalar"
    assert result["direct_fact"]["value"] == 80


def test_relative_temporal_wh_clause_uses_explicit_as_of(tmp_path):
    engine = _engine(tmp_path)
    source = _append(
        engine,
        "I held the pottery exhibition at Harbor Hall.",
        session_id="event",
    )
    engine._session_occurrence_dates["event"] = "2025-03-06"
    try:
        result = _compile(
            engine,
            "I mentioned an exhibition two weeks ago. Where was it held?",
            [source],
            question_as_of="2025-03-20",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["contract"]["operation"] == "date_filter"
    assert result["contract"]["answer_kind"] == "place"
    assert result["direct_fact"]["value"] == "Harbor Hall"


def test_direct_stated_total_precedes_operand_arithmetic(tmp_path):
    engine = _engine(tmp_path)
    source = _append(
        engine,
        "The workshop fees were $40 and $65, for a total of $105.",
    )
    try:
        result = _compile(
            engine,
            "How much total money did I spend on the workshop fees?",
            [source],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "answer_sufficient"
    assert result["direct_fact"]["value"] == 105
    assert result["direct_fact"]["unit"] == "usd"
    assert result["context"] is None


def test_three_quoted_operands_compile_three_exact_sum_slots(tmp_path):
    engine = _engine(tmp_path)
    alpha = _append(engine, "Reading 'Alpha' took 1 week.", session_id="alpha")
    beta = _append(engine, "Reading 'Beta' took 2 weeks.", session_id="beta")
    gamma = _append(engine, "Reading 'Gamma' took 3 weeks.", session_id="gamma")
    try:
        result = _compile(
            engine,
            "How many weeks total did I spend on 'Alpha', 'Beta', and 'Gamma'?",
            [alpha, beta, gamma],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert len(result["contract"]["slots"]) == 3
    assert result["state"] == "computation_sufficient"
    assert result["computation"]["result_value"] == 6


def test_unique_highest_anchor_scalar_beats_irrelevant_number_but_tie_fails(tmp_path):
    engine = _engine(tmp_path)
    weak = _append(engine, "The group reservation is for 8 people.", session_id="weak")
    strong = _append(engine, "I directly lead 5 engineers on the platform team.", session_id="strong")
    conflict = _append(engine, "I directly lead 6 engineers on the platform team.", session_id="conflict")
    try:
        selected = _compile(
            engine,
            "How many engineers do I directly lead on the platform team?",
            [weak, strong],
            budgets={"max_retrieval_calls": 0},
        )
        ambiguous = _compile(
            engine,
            "How many engineers do I directly lead on the platform team?",
            [strong, conflict],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert selected["direct_fact"]["value"] == 5
    assert ambiguous["reason_code"] == "ambiguous_scalar_candidates"
    assert ambiguous["context"] is None


def test_latest_ignores_irrelevant_older_conflict_but_frontier_conflict_fails(tmp_path):
    engine = _engine(tmp_path)
    old_time = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    new_time = datetime(2024, 2, 1, tzinfo=timezone.utc).timestamp()
    old_a = _append(engine, "I live in Austin.", session_id="old-a", timestamp=old_time)
    old_b = _append(engine, "I live in Boston.", session_id="old-b", timestamp=old_time)
    current = _append(engine, "I live in Denver.", session_id="current", timestamp=new_time)
    current_conflict = _append(engine, "I live in Seattle.", session_id="new-b", timestamp=new_time)
    try:
        safe = _compile(
            engine,
            "Where do I currently live?",
            [old_a, old_b, current],
            question_as_of="2024-03-01",
            budgets={"max_retrieval_calls": 0},
        )
        conflicted = _compile(
            engine,
            "Where do I currently live?",
            [current, current_conflict],
            question_as_of="2024-03-01",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert safe["direct_fact"]["value"] == "Denver"
    assert conflicted["reason_code"] == "state_conflicted"


def test_finite_scan_ignores_generic_advice_and_uncertifies_unknown_source_event(
    tmp_path,
):
    engine = _engine(tmp_path)
    known = _append(engine, "I attended Jordan's wedding.", session_id="known")
    _append(
        engine,
        "Wedding planning usually includes flowers and invitations.",
        session_id="advice",
        role="assistant",
    )
    unknown = _append(
        engine,
        "I attended Casey's wedding.",
        session_id="unknown",
        timestamp=datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp(),
    )
    engine._session_occurrence_dates["known"] = "2025-02-01"
    try:
        result = _compile(
            engine,
            "How many weddings did I attend this year?",
            [known],
            question_as_of="2025-12-31",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["finite_coverage"] is False
    assert result["reason_code"] == "finite_count_uncertified_undated_events"
    assert result["computation"]["result_value"] == 2
    assert result["coverage_certificate"]["material_clauses"] == 2
    assert result["coverage_certificate"]["unknown_time_clauses"] == 1
    assert unknown["exact_ref"].startswith("lcm:")


def test_increase_from_uses_source_order_and_unique_operand_spans(tmp_path):
    engine = _engine(tmp_path)
    source = _append(
        engine,
        "The commute increased from 20 minutes to 30 minutes.",
    )
    try:
        result = _compile(
            engine,
            "Approximately how much did the commute increase from 20 minutes "
            "to 30 minutes?",
            [source],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["state"] == "computation_sufficient"
    assert result["computation"]["result_value"] == 10
    assert result["computation"]["unit"] == "minute"
    assert len(set(result["computation"]["citations"])) == 2


def test_conflicted_latest_state_never_selects_by_row_order(tmp_path):
    engine = _engine(tmp_path)
    observed = datetime(2025, 1, 2, tzinfo=timezone.utc).timestamp()
    austin = _append(engine, "I live in Austin.", session_id="a", timestamp=observed)
    denver = _append(engine, "I live in Denver.", session_id="b", timestamp=observed)
    try:
        result = _compile(
            engine,
            "Where do I currently live?",
            [austin, denver],
            question_as_of="2025-01-03",
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["status"] == "no_augmentation"
    assert result["reason_code"] == "state_conflicted"
    assert result["context"] is None


def test_second_targeted_query_may_close_a_named_slot_after_no_progress(tmp_path):
    engine = _engine(tmp_path)
    jogging = _append(engine, "I spent 1 hour jogging.", session_id="jogging")
    yoga = _append(engine, "I spent 30 minutes doing yoga.", session_id="yoga")
    noise = _append(engine, "The studio has a blue door.", session_id="noise")
    calls = []

    def retrieve(args):
        calls.append(args)
        selected = noise if len(calls) == 1 else yoga
        return {"hits": [{"exact_ref": selected["exact_ref"]}]}

    try:
        result = _compile(
            engine,
            "How many minutes total did I spend jogging and doing yoga?",
            [jogging],
            retrieve=retrieve,
        )
    finally:
        engine._store.close()

    assert len(calls) == 2
    assert calls[0]["query"] != calls[1]["query"]
    assert result["state"] == "computation_sufficient"
    assert result["computation"]["result_value"] == 90
    assert all(not ref.startswith("lcm:3:") for ref in result["novel_exact_refs"])


def test_retrieval_usage_maps_product_embedding_metrics_without_query_payload(tmp_path):
    engine = _engine(tmp_path)
    baseline = _append(engine, "We discussed the commute yesterday.")
    useful = _append(engine, "My commute is 35 minutes.", session_id="fact")

    def retrieve(_args):
        return {
            "hits": [{"exact_ref": useful["exact_ref"]}],
            "metrics": {
                "embedding_query_calls": 1,
                "embedding_query_tokens": 7,
                "embedding_query_tokens_complete": True,
            },
        }

    try:
        result = _compile(
            engine,
            "How long is my commute?",
            [baseline],
            retrieve=retrieve,
        )
    finally:
        engine._store.close()

    assert result["retrieval"]["provider_calls"] == 1
    assert result["retrieval"]["input_tokens"] == 7
    assert result["retrieval"]["query_tokens_complete"] is True
    assert "commute" not in json.dumps(result["retrieval"]["queries"])


def test_failed_retrieval_marks_query_token_provenance_incomplete(tmp_path):
    engine = _engine(tmp_path)
    baseline = _append(engine, "We discussed the commute yesterday.")

    def retrieve(_args):
        raise RuntimeError("provider unavailable")

    try:
        result = _compile(
            engine,
            "How long is my commute?",
            [baseline],
            retrieve=retrieve,
        )
    finally:
        engine._store.close()

    assert result["retrieval"]["calls"] >= 1
    assert result["retrieval"]["query_tokens_complete"] is False
    assert result["context"] is None


def test_non_conversational_baseline_does_not_trigger_neighbor_expansion(tmp_path):
    engine = _engine(tmp_path)
    statement = _append(engine, "The reward program is changing soon.")
    _append(engine, "You need 15 points to redeem the reward.", role="assistant")
    try:
        result = _compile(
            engine,
            "How many points do I need to redeem the reward?",
            [statement],
            budgets={"max_retrieval_calls": 0},
        )
    finally:
        engine._store.close()

    assert result["status"] == "no_augmentation"
    assert result["metrics"]["session_loads"] == 0
    assert result["novel_exact_refs"] == []


def test_context_budget_exhaustion_is_trace_only(tmp_path):
    engine = _engine(tmp_path)
    question = _append(engine, "What should I do about dry basil?", role="user")
    _append(
        engine,
        "Water the basil in the morning. " + "Keep it in gentle indirect light. " * 30,
        role="assistant",
    )
    try:
        result = _compile(
            engine,
            "What advice did you give me about dry basil?",
            [question],
            budgets={"max_retrieval_calls": 0, "max_added_context_tokens": 64},
        )
    finally:
        engine._store.close()

    assert result["status"] == "no_augmentation"
    assert result["reason_code"] == "context_budget_exhausted"
    assert result["trace"]["truncated"] is True
    assert result["context"] is None
