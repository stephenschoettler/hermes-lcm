from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm import aux_session as aux_session_module
from hermes_lcm import command as command_module
from hermes_lcm import engine as engine_module
from hermes_lcm import reset_state as reset_state_module
from hermes_lcm.access_context import Decision, DenialReason
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


class _Policy:
    def __init__(self, *, reason: DenialReason | None = None) -> None:
        self.reason = reason
        self.calls: list[tuple[str, dict]] = []
        self.audits: list[tuple[DenialReason | None, object]] = []

    def authorize_operation(self, _context, operation, expected_scope):
        self.calls.append((operation, dict(expected_scope)))
        return Decision.deny(self.reason) if self.reason is not None else Decision.allow()

    def resolve_authorized_targets(self, _context, _operation, requested_narrowing):
        return requested_narrowing

    def audit_decision(self, _context, _operation, internal_reason, public_result):
        self.audits.append((internal_reason, public_result))


def _engine(tmp_path: Path) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))
    engine._session_id = "cron-session"
    engine._conversation_id = "cron-conversation"
    engine._foreground_session_id = "foreground-session"
    engine._foreground_conversation_id = "foreground-conversation"
    return engine


def test_tool_denials_project_distinct_internal_reasons_to_one_public_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    try:
        messages: list[str] = []
        policies: list[_Policy] = []
        for reason in (DenialReason.SCOPE_FORBIDDEN, DenialReason.LEASE_STALE):
            policy = _Policy(reason=reason)
            policies.append(policy)
            monkeypatch.setattr(engine_module, "policy_for_engine", lambda _e, p=policy: p)
            monkeypatch.setattr(engine_module, "policy_access_context", lambda _e: None)
            with pytest.raises(engine_module.AuthorizationRequiredError) as exc_info:
                engine.handle_tool_call("lcm_load_session", {"session_id": "hidden"})
            messages.append(str(exc_info.value))

        assert messages[0] == messages[1]
        assert "target_not_found_or_forbidden" in messages[0]
        assert "scope_forbidden" not in messages[0]
        assert "lease_stale" not in messages[1]
        assert [audit[0] for policy in policies for audit in policy.audits] == [
            DenialReason.SCOPE_FORBIDDEN,
            DenialReason.LEASE_STALE,
        ]
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "tool_name",
    ("lcm_recent", "lcm_recall", "lcm_status", "lcm_inspect", "lcm_doctor"),
)
def test_target_free_tool_authorization_uses_foreground_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_name: str
) -> None:
    engine = _engine(tmp_path)
    policy = _Policy()
    monkeypatch.setattr(engine_module, "policy_for_engine", lambda _e: policy)
    monkeypatch.setattr(engine_module, "policy_access_context", lambda _e: None)
    monkeypatch.setattr(
        engine_module.lcm_tools,
        tool_name,
        lambda args, engine: json.dumps(args),
    )
    try:
        engine.handle_tool_call(tool_name, {})
        expected = policy.calls[0][1]
        assert expected["target_scope"] == {
            "session_id": "foreground-session",
            "conversation_id": "foreground-conversation",
        }
        assert expected["caller_session_id"] == "cron-session"
        assert expected["caller_conversation_id"] == "cron-conversation"
    finally:
        engine.shutdown()


def test_reset_state_requires_owner_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    policy = _Policy()
    monkeypatch.setattr(reset_state_module, "policy_for_engine", lambda _e: policy)
    monkeypatch.setattr(reset_state_module, "policy_access_context", lambda _e: None)
    try:
        engine._reset_session_scoped_runtime_state()
        assert [operation for operation, _scope in policy.calls] == ["owner_only"]
        assert policy.calls[0][1]["required_scope"] == "owner_only"
    finally:
        engine.shutdown()


def test_auxiliary_lineage_operations_require_owner_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    policy = _Policy()
    monkeypatch.setattr(aux_session_module, "policy_for_engine", lambda _e: policy)
    monkeypatch.setattr(aux_session_module, "policy_access_context", lambda _e: None)
    try:
        engine._register_auxiliary_session("aux-a")
        engine._deactivate_auxiliary_session("aux-a")
        engine._handoff_auxiliary_session("aux-old", "aux-new")
        engine._unmark_thread_context_auxiliary_session("aux-new")
        assert [operation for operation, _scope in policy.calls] == [
            "owner_only",
            "owner_only",
            "owner_only",
            "owner_only",
        ]
        assert all(scope["required_scope"] == "owner_only" for _op, scope in policy.calls)
    finally:
        engine.shutdown()


def test_rollup_scheduler_uses_policy_resolved_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs: list[tuple[tuple[str, str], object]] = []
    runs: list[str] = []

    class Scheduler:
        def schedule(self, key, job, **_kwargs):
            jobs.append((key, job))
            return True

    class PrivateDag:
        def __init__(self, _path):
            pass

        def close(self):
            pass

    class NarrowingPolicy(_Policy):
        def resolve_authorized_targets(self, _context, _operation, requested_narrowing):
            return {**requested_narrowing, "partition_key": "narrowed-session"}

    carrier = object.__new__(LCMEngine)
    carrier._dag = SimpleNamespace(db_path=str(tmp_path / "rollup.db"))
    carrier._config = SimpleNamespace()
    carrier._summary_circuit_breaker = None
    carrier._summary_spend_guard = None
    carrier._rollup_maintenance_owner = object()
    policy = NarrowingPolicy()
    monkeypatch.setattr(engine_module, "_ROLLUP_MAINTENANCE_SCHEDULER", Scheduler())
    monkeypatch.setattr(engine_module, "policy_for_engine", lambda _e: policy)
    monkeypatch.setattr(engine_module, "policy_access_context", lambda _e: None)
    monkeypatch.setattr(engine_module, "SummaryDAG", PrivateDag)
    monkeypatch.setattr(
        engine_module,
        "run_rollup_maintenance",
        lambda _dag, _config, scope, **_kwargs: runs.append(scope),
    )

    carrier._schedule_rollup_maintenance("wide-session")
    assert jobs and jobs[0][0][1] == "narrowed-session"
    jobs[0][1]()
    assert runs == ["narrowed-session"]


def test_slash_doctor_requires_admin_before_scope_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    calls: list[str] = []
    allowed_policy = _Policy()
    monkeypatch.setattr(command_module, "policy_for_engine", lambda _e: allowed_policy)
    monkeypatch.setattr(command_module, "policy_access_context", lambda _e: None)
    monkeypatch.setattr(command_module, "_doctor_text", lambda _e: calls.append("doctor") or "ok")
    try:
        assert command_module.handle_lcm_command("doctor", engine) == "ok"
        assert calls == ["doctor"]
        assert allowed_policy.calls[0][0] == "admin"
        assert allowed_policy.calls[0][1]["required_scope"] == "admin"

        calls.clear()
        denied_policy = _Policy(reason=DenialReason.SCOPE_FORBIDDEN)
        monkeypatch.setattr(command_module, "policy_for_engine", lambda _e: denied_policy)
        with pytest.raises(command_module.AuthorizationRequiredError, match="target_not_found_or_forbidden"):
            command_module.handle_lcm_command("doctor", engine)
        assert calls == []
    finally:
        engine.shutdown()
