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


def test_delivered_state_outranks_no_claim_reason_code():
    """F1 regression: a delivered state is never overridden by a no-claim code.

    The compiler's ``unknown`` is a BASE state that refusal paths carry with
    no-claim reason codes -- those stay unmarked.  But any other delivered
    state (e.g. ``answer_sufficient``) must outrank the reason-code tables,
    even if a stale/future code collides with the no-claim set.
    """
    from hermes_lcm.sufficiency_gate import apply_sufficiency_gate

    result = {
        "status": "compiled",
        "state": "answer_sufficient",
        "reason_code": "unsupported_operation",  # stale/future collision
        "context": "the taxi and the train together cost $80",
        "trace": {},
        "metrics": {},
    }
    apply_sufficiency_gate(result, enabled=True)
    assert result["sufficiency"]["state"] == "answer_sufficient"
    assert result["sufficiency"]["policy_action"] == "answer"

    # The base-state exception still holds: unknown + no-claim code = unmarked.
    refusal = {
        "status": "no_augmentation",
        "state": "unknown",
        "reason_code": "unsupported_operation",
        "context": None,
        "trace": {},
        "metrics": {},
    }
    apply_sufficiency_gate(refusal, enabled=True)
    assert "sufficiency" not in refusal


def test_gate_failure_is_atomic_and_wrapper_fails_open(
    tmp_path, monkeypatch, caplog
):
    """F2 regression: gate crashes leave no partial mutation, and the
    ``build_preanswer_evidence`` wrapper fails open to the legacy result."""
    import pytest

    import hermes_lcm.preanswer_evidence as pe
    from hermes_lcm import sufficiency_gate as sg

    # (a) Classification crash: zero mutation, zero partial marking.
    broken = {
        "status": "no_augmentation",
        "state": None,
        "reason_code": "no_hit",
        "context": None,
        "trace": {"context_sha256": None},
        "metrics": {"latency_ms": 1.0},
    }

    def _boom_classify(_result):
        raise RuntimeError("classify boom")

    # (a) Classification crash: zero mutation, zero partial marking.  The
    # patch lives in its own scope so (b) can exercise the real classifier.
    with monkeypatch.context() as m:
        m.setattr(sg, "classify_preanswer_result", _boom_classify)
        with pytest.raises(RuntimeError, match="classify boom"):
            sg.apply_sufficiency_gate(broken, enabled=True)
    assert broken == {
        "status": "no_augmentation",
        "state": None,
        "reason_code": "no_hit",
        "context": None,
        "trace": {"context_sha256": None},
        "metrics": {"latency_ms": 1.0},
    }

    # (b) Render crash after classification: still no mutation, because the
    # verdict writes happen on a copy that is only committed on success.  The
    # classifier patch from (a) is scoped to (a) only, so this call actually
    # exercises the render path.
    partial = {
        "status": "no_augmentation",
        "state": "unknown",
        "reason_code": "no_hit",
        "context": None,
        "trace": {"context_sha256": None},
        "metrics": {"latency_ms": 1.0},
    }

    def _boom_render(*_args, **_kwargs):
        raise RuntimeError("render boom")

    with monkeypatch.context() as m:
        m.setattr(sg, "render_disclosure", _boom_render)
        with pytest.raises(RuntimeError, match="render boom"):
            sg.apply_sufficiency_gate(partial, enabled=True)
    assert partial["context"] is None
    assert partial["trace"]["context_sha256"] is None
    assert "sufficiency" not in partial

    # (b2) Missing ``trace`` (the original F2 failure mode): the gate marks
    # the result, skips the trace update, and never partially mutates.
    untraced = {
        "status": "no_augmentation",
        "state": "unknown",
        "reason_code": "no_hit",
        "context": None,
        "metrics": {"latency_ms": 1.0},
    }
    assert sg.apply_sufficiency_gate(untraced, enabled=True) is untraced
    assert untraced["sufficiency"]["state"] == "unknown"
    assert untraced["context"] == untraced["sufficiency"]["disclosure_context"]
    assert "trace" not in untraced

    # (c) The wrapper fails open: a gate crash inside
    # ``build_preanswer_evidence`` still returns the legacy result.  The
    # crash is injected through the gate module's own classification hook
    # (module-identity-proof), and the warning is captured on the logger of
    # the module that actually executes the wrapper.
    engine = _engine(tmp_path)
    taxi, train = _taxi_train(engine)
    try:
        legacy = _result(
            engine,
            "What is the total of the two costs?",
            refs=[taxi, train],
            enabled=True,
        )

        def _boom_apply(*_args, **_kwargs):
            raise RuntimeError("apply boom")

        # Patch AND invoke through the same module instance: the file-level
        # import may predate harness-driven module replacement, so resolving
        # the wrapper via ``pe`` keeps globals and patch target identical.
        monkeypatch.setattr(pe, "apply_sufficiency_gate", _boom_apply)
        import logging as _logging

        prev_disable = _logging.root.manager.disable
        _logging.disable(_logging.NOTSET)
        wrapper_logger = _logging.getLogger(pe.build_preanswer_evidence.__module__)
        prev_level = wrapper_logger.level
        wrapper_logger.setLevel(_logging.WARNING)
        records: list[str] = []

        class _Capture(_logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Capture(level=_logging.WARNING)
        wrapper_logger.addHandler(handler)
        try:
            wrapped = pe.build_preanswer_evidence(
                "What is the total of the two costs?",
                engine=engine,
                baseline_refs=[taxi, train],
                enabled=True,
                sufficiency_gate=True,
            )
        finally:
            wrapper_logger.removeHandler(handler)
            wrapper_logger.setLevel(prev_level)
            _logging.disable(prev_disable)
    finally:
        engine._store.close()

    assert "sufficiency" not in wrapped
    assert _without_timing(wrapped) == _without_timing(legacy)
    assert records and "failed open" in records[0]


def test_gate_keeps_metrics_and_trace_digest_rederivable(tmp_path):
    """F3 regression: metrics are never touched, so the compiler's
    ``trace.digest_sha256`` remains re-derivable from every gated result.

    The digest lives on the requirements-compiler route, so this test drives
    ``compile_preanswer_evidence`` directly and applies the gate to its
    finished result, mirroring the hook wiring.
    """
    from hermes_lcm import requirements_compiler as rc
    from hermes_lcm import sufficiency_gate as sg

    engine = _engine(tmp_path)
    bali = _append(
        engine, "I took a vacation to Bali this year.", observed_at=1_710_000_000
    )
    kyoto = _append(
        engine, "I took a vacation to Kyoto this year.", observed_at=1_720_000_000
    )
    refs = [bali, kyoto]
    try:
        legacy = rc.compile_preanswer_evidence(
            "How many vacations did I take this year?",
            engine=engine,
            baseline_refs=refs,
            question_date="2024-12-31",
            enabled=True,
        )
        gated = rc.compile_preanswer_evidence(
            "How many vacations did I take this year?",
            engine=engine,
            baseline_refs=refs,
            question_date="2024-12-31",
            enabled=True,
        )
        sg.apply_sufficiency_gate(gated, enabled=True)
    finally:
        engine._store.close()

    # Base state unknown stays annotate on this route (evidence lives in the
    # baseline identity, not the delivered evidence list) — and annotate still
    # renders a disclosure context, exactly where gate mutation risked the
    # digest.
    assert gated["state"] == "unknown"
    assert gated["sufficiency"]["state"] == "unknown"
    assert gated["sufficiency"]["policy_action"] == "annotate"
    assert gated["context"] == gated["sufficiency"]["disclosure_context"]
    # metrics are byte-identical between legacy and gated runs (timing aside)
    assert set(gated["metrics"]) == set(legacy["metrics"])
    for key, value in legacy["metrics"].items():
        if key == "latency_ms":
            continue
        assert gated["metrics"][key] == value
    # the gate never added a foreign metric or a mutated context metric
    assert "sufficiency_gate_latency_ms" not in gated["metrics"]
    assert gated["sufficiency"]["gate_latency_ms"] >= 0.0
    # the disclosure size lives in the gate-owned section
    assert gated["sufficiency"]["disclosure_context_chars"] == len(
        gated["sufficiency"]["disclosure_context"]
    )
    # digest_sha256 remains re-derivable from the gated result
    trace = gated["trace"]
    assert trace["digest_sha256"]
    material = {
        "version": gated["version"],
        "status": gated["status"],
        "state": gated["state"],
        "reason_code": gated["reason_code"],
        "contract": {
            "answer_kind": (gated.get("contract") or {}).get("answer_kind"),
            "operation": (gated.get("contract") or {}).get("operation"),
            "coverage_policy": (gated.get("contract") or {}).get("coverage_policy"),
            "slot_count": len((gated.get("contract") or {}).get("slots") or []),
        },
        "baseline": gated["baseline"],
        "closed_requirement_count": len(gated["closed_requirements"]),
        "open_requirement_count": len(gated["open_requirements"]),
        "novel_ref_count": len(gated["novel_exact_refs"]),
        "finite_coverage": gated["finite_coverage"],
        "computation_sha256": gated["computation_sha256"],
        "retrieval_calls": gated["retrieval"]["calls"],
        "metrics": gated["metrics"],
        "truncated": trace["truncated"],
    }
    assert trace["digest_sha256"] == rc._digest(material)


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
        "disclosure_context_chars",
        "gate_latency_ms",
    }