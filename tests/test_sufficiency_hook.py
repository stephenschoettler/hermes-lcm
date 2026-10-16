"""Hook-level contract tests for LCM_PREANSWER_EVIDENCE_MODE=sufficiency_v1.

These reuse the packaging tests' plugin-entrypoint harness so the gate is
exercised through the real ``pre_llm_call`` hook path, not just the module
API.  The sufficiency mode routes through the requirements compiler and adds
the gate verdict on top; ordinary and legacy paths stay byte-identical.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json

from tests.test_packaging_install import (  # noqa: F401  (harness reuse)
    _ensure_agent_context_engine_importable,
    _load_plugin_entrypoint_module,
)


def _sufficiency_module(monkeypatch, tmp_path, name: str):
    _ensure_agent_context_engine_importable(monkeypatch)
    module = _load_plugin_entrypoint_module(name)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_ENABLED", "true")
    monkeypatch.setenv("LCM_PREANSWER_EVIDENCE_MODE", "sufficiency_v1")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    return module


def _ctx_and_hook(module):
    hooks = {}

    class _Ctx:
        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_hook(self, name, callback):
            hooks.setdefault(name, []).append(callback)

    ctx = _Ctx()
    module.register(ctx)
    ctx.engine.on_session_start("active-session", platform="cli")
    return ctx, hooks["pre_llm_call"][0]


def _store_fact(ctx, content, observed_at):
    return ctx.engine._store.append(
        "prior-session", {"role": "user", "content": content, "timestamp": observed_at}
    )


def _recall_hits(hits):
    def handle(tool, args, **_kwargs):
        assert tool == "lcm_recall"
        return json.dumps({"hits": hits})

    return handle


def test_sufficiency_mode_answers_with_gate_verdict_on_validated_fact(
    tmp_path, monkeypatch
):
    module = _sufficiency_module(monkeypatch, tmp_path, "hermes_lcm_packaging_suff_v1_ok")
    ctx, hook = _ctx_and_hook(module)
    fact = "You need 15 points to redeem the reward."
    fact_id = _store_fact(ctx, fact, 100.0)
    ctx.engine.handle_tool_call = _recall_hits(
        [{"exact_ref": f"lcm:{fact_id}:0-{len(fact)}", "content": fact}]
    )
    response = hook(
        session_id="active-session",
        user_message="How many points do I need to redeem the reward?",
        enabled_toolsets=["context_engine"],
    )

    assert "lcm-answer-brief" in response["context"]
    assert "15 point" in response["context"]
    trace = ctx.engine._last_preanswer_evidence_trace
    assert trace["sufficiency"]["state"] == "answer_sufficient"
    assert trace["sufficiency"]["policy_action"] == "answer"
    assert "sufficiency-disclosure" not in response["context"]
    ctx.engine.shutdown()


def test_sufficiency_mode_annotates_when_no_fact_is_found(tmp_path, monkeypatch):
    module = _sufficiency_module(
        monkeypatch, tmp_path, "hermes_lcm_packaging_suff_v1_miss"
    )
    ctx, hook = _ctx_and_hook(module)
    ctx.engine.handle_tool_call = _recall_hits([])
    response = hook(
        session_id="active-session",
        user_message="How many points do I need to redeem the reward?",
        enabled_toolsets=["context_engine"],
    )

    trace = ctx.engine._last_preanswer_evidence_trace
    assert trace["sufficiency"]["state"] == "unknown"
    assert trace["sufficiency"]["policy_action"] == "annotate"
    assert "lcm-sufficiency-disclosure" in response["context"]
    assert "state: unknown" in response["context"]
    ctx.engine.shutdown()


def test_sufficiency_mode_conflict_annotates_instead_of_answering(
    tmp_path, monkeypatch
):
    module = _sufficiency_module(
        monkeypatch, tmp_path, "hermes_lcm_packaging_suff_v1_conflict"
    )
    ctx, hook = _ctx_and_hook(module)
    first = "You need 15 points to redeem the reward."
    second = "You need 40 points to redeem the reward."
    first_id = _store_fact(
        ctx, first, datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    )
    second_id = _store_fact(
        ctx, second, datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp()
    )
    ctx.engine.handle_tool_call = _recall_hits(
        [
            {"exact_ref": f"lcm:{first_id}:0-{len(first)}", "content": first},
            {"exact_ref": f"lcm:{second_id}:0-{len(second)}", "content": second},
        ]
    )
    response = hook(
        session_id="active-session",
        user_message="How many points do I need to redeem the reward?",
        enabled_toolsets=["context_engine"],
    )

    trace = ctx.engine._last_preanswer_evidence_trace
    assert trace["sufficiency"]["state"] == "conflicted"
    assert trace["sufficiency"]["policy_action"] == "annotate"
    assert "lcm-sufficiency-disclosure" in response["context"]
    assert "state: conflicted" in response["context"]
    ctx.engine.shutdown()


def test_sufficiency_mode_no_claim_paths_stay_byte_identical(tmp_path, monkeypatch):
    module = _sufficiency_module(
        monkeypatch, tmp_path, "hermes_lcm_packaging_suff_v1_noop"
    )
    ctx, hook = _ctx_and_hook(module)
    policy = module.get_recall_policy()

    def fail(*_args, **_kwargs):
        raise AssertionError("no product retrieval is allowed")

    ctx.engine.handle_tool_call = fail
    ordinary = hook(
        session_id="active-session",
        user_message="Tell me about the Atlas project",
        enabled_toolsets=["context_engine"],
    )
    disabled_toolset = hook(
        session_id="active-session",
        user_message="How long is my commute?",
        enabled_toolsets=[],
    )

    assert ordinary == {"context": policy}
    assert disabled_toolset == {"context": policy}
    ctx.engine.shutdown()