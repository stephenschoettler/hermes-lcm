"""Contract tests for the sufficiency gate at the preanswer finalize seam."""

from __future__ import annotations

import json

from hermes_lcm.preanswer_evidence import build_preanswer_evidence
from hermes_lcm.store import MessageStore

from types import SimpleNamespace

from hermes_lcm.config import LCMConfig


def _engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    store = MessageStore(config.database_path, ingest_protection_config=config)
    return SimpleNamespace(_config=config, _store=store, _assertions=None)


def _append(engine, content, *, observed_at=None, session_id="session-a"):
    message = {"role": "user", "content": content}
    if observed_at is not None:
        message["timestamp"] = observed_at
    store_id = engine._store.append(session_id, message)
    return {
        "exact_ref": f"lcm:{store_id}:0-{len(content)}",
        "quote": content,
    }


def _result(engine, question, *, refs=(), retrieve=None, **kwargs):
    return build_preanswer_evidence(
        question,
        engine=engine,
        baseline_refs=list(refs),
        retrieve=retrieve,
        **kwargs,
    )


def _taxi_train(engine):
    taxi = _append(engine, "The taxi cost $60.", observed_at=1_710_000_000)
    train = _append(engine, "The train cost $20.", observed_at=1_720_000_000)
    return taxi, train


def test_gate_disabled_is_byte_identical_to_legacy_result(tmp_path):
    engine = _engine(tmp_path)
    taxi, train = _taxi_train(engine)
    try:
        legacy = _result(
            engine,
            "What is the total of the two costs?",
            refs=[taxi, train],
            enabled=True,
        )
        gated_off = _result(
            engine,
            "What is the total of the two costs?",
            refs=[taxi, train],
            enabled=True,
        )
    finally:
        engine._store.close()

    # Two legacy runs differ only in wall-clock metrics.
    assert _without_timing(gated_off) == _without_timing(legacy)
    assert "sufficiency" not in gated_off


_TIMING_KEYS = frozenset({"latency_ms", "sufficiency_gate_latency_ms"})


def _without_timing(result):
    stripped = {k: v for k, v in result.items() if k != "sufficiency"}
    stripped["metrics"] = {
        k: v for k, v in stripped["metrics"].items() if k not in _TIMING_KEYS
    }
    return stripped


def test_gate_adds_only_sufficiency_section_to_sufficient_results(tmp_path):
    engine = _engine(tmp_path)
    taxi, train = _taxi_train(engine)
    try:
        ungated = _result(
            engine,
            "What is the total of the two costs?",
            refs=[taxi, train],
            enabled=True,
        )
        gated = _result(
            engine,
            "What is the total of the two costs?",
            refs=[taxi, train],
            enabled=True,
            sufficiency_gate=True,
        )
    finally:
        engine._store.close()

    # The gate only appends its section; every pre-gate field is identical.
    assert _without_timing(gated) == _without_timing(ungated)


def test_gate_marks_computation_sufficient_without_touching_evidence(tmp_path):
    engine = _engine(tmp_path)
    taxi, train = _taxi_train(engine)
    try:
        result = _result(
            engine,
            "What is the total of the two costs?",
            refs=[taxi, train],
            enabled=True,
            sufficiency_gate=True,
        )
    finally:
        engine._store.close()

    section = result["sufficiency"]
    assert section["state"] == "computation_sufficient"
    assert section["policy_action"] == "answer"
    assert section["disclosure_context"] is None
    assert result["computation"]["result"] == "$80"
    assert result["status"] == "computed"


def test_open_cardinality_partial_discloses_from_stored_fields_only(tmp_path):
    engine = _engine(tmp_path)
    bali = _append(
        engine, "I took a vacation to Bali this year.", observed_at=1_710_000_000
    )
    kyoto = _append(
        engine, "I took a vacation to Kyoto this year.", observed_at=1_720_000_000
    )
    try:
        result = _result(
            engine,
            "How many vacations did I take this year?",
            refs=[bali, kyoto],
            enabled=True,
            question_date="2024-12-31",
            sufficiency_gate=True,
        )
    finally:
        engine._store.close()

    assert result["status"] == "no_augmentation"
    assert result["reason_code"] == "open_cardinality"
    section = result["sufficiency"]
    assert section["state"] == "partial"
    assert section["policy_action"] == "answer_with_disclosure"
    disclosure = section["disclosure_context"]
    assert "<lcm-sufficiency-disclosure>" in disclosure
    assert "state: partial" in disclosure
    assert "open_cardinality" in disclosure
    # The disclosure block is the only authored text; every other value is a
    # stored field.  It must never invent an operand or a result value.
    assert "Bali" not in disclosure and "Kyoto" not in disclosure
    # Partial with empty context renders the disclosure as the context block.
    assert result["context"] == disclosure
    assert result["trace"]["context_sha256"] is not None


def test_no_hit_annotates_instead_of_staying_silent(tmp_path):
    engine = _engine(tmp_path)
    austin = _append(engine, "I used to live in Austin.", observed_at=1_710_000_000)
    calls = []

    def retrieve(args):
        calls.append(args)
        return json.dumps({"hits": []})

    try:
        result = _result(
            engine,
            "Where do I live now?",
            refs=[austin],
            retrieve=retrieve,
            enabled=True,
            sufficiency_gate=True,
        )
    finally:
        engine._store.close()

    assert result["reason_code"] == "no_hit"
    section = result["sufficiency"]
    assert section["state"] == "unknown"
    assert section["policy_action"] == "annotate"
    assert "state: unknown" in result["context"]
    assert "disclose rather than imply sufficiency" in result["context"]


def test_no_claim_paths_stay_unmarked_and_ordinary(tmp_path):
    engine = _engine(tmp_path)
    taxi, train = _taxi_train(engine)
    try:
        disabled = _result(
            engine,
            "What is the total of the two costs?",
            refs=[taxi, train],
            enabled=False,
            sufficiency_gate=True,
        )
        unsupported = _result(
            engine,
            "Tell me about the Atlas project.",
            enabled=True,
            sufficiency_gate=True,
        )
    finally:
        engine._store.close()

    assert disabled["reason_code"] == "feature_disabled"
    assert "sufficiency" not in disabled
    assert unsupported["reason_code"] == "no_supported_operation"
    assert "sufficiency" not in unsupported
    assert unsupported["context"] is None


def test_gate_never_modifies_evidence_computation_or_status(tmp_path):
    engine = _engine(tmp_path)
    taxi, train = _taxi_train(engine)
    try:
        gated = _result(
            engine,
            "What is the total of the two costs?",
            refs=[taxi, train],
            enabled=True,
            sufficiency_gate=True,
        )
    finally:
        engine._store.close()

    assert gated["sufficiency"]["state"] == "computation_sufficient"
    assert gated["evidence"]
    assert gated["computation"] is not None
    assert gated["status"] == "computed"
    # The gate may only append provenance about itself, never rewrite fields.
    assert set(gated["sufficiency"]) == {
        "version",
        "state",
        "policy_action",
        "disclosure_context",
    }