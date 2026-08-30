from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm.access_context import Decision
from hermes_lcm import engine as engine_module
from hermes_lcm import engine_registry
from hermes_lcm import maintenance as maintenance_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

_access_policy = importlib.import_module("hermes_lcm.access_policy")
AuthorizationRequiredError = _access_policy.AuthorizationRequiredError
FailClosedPolicy = _access_policy.FailClosedPolicy
TrustedOwnerPolicy = _access_policy.TrustedOwnerPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]


def _engine(tmp_path: Path) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))
    engine._session_id = "session-a"
    engine._conversation_id = "conversation-a"
    return engine


class _RecordingPolicy:
    def __init__(self, label: str, *, allowed: bool = True) -> None:
        self.label = label
        self.allowed = allowed
        self.calls: list[tuple[str, dict]] = []

    def authorize_operation(self, _context, operation, expected_scope):
        self.calls.append((operation, dict(expected_scope)))
        return Decision.allow() if self.allowed else Decision.deny("context_missing")

    def resolve_authorized_targets(self, _context, _operation, requested_narrowing):
        return requested_narrowing

    def audit_decision(self, *_args):
        return None


def test_default_off_callback_and_maintenance_passthrough(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    try:
        assert isinstance(engine_module.policy_for_engine(engine), TrustedOwnerPolicy)
        engine.on_session_start("session-a", conversation_id="conversation-a")
        assert "session_id" in engine.handle_tool_call("lcm_status", {})
        engine.on_session_end("session-a", [])
        backup = maintenance_module.backup_database(engine)
        assert backup["ok"] is True
        engine.on_session_reset()
    finally:
        engine.shutdown()


def test_tool_dispatch_uses_inventory_authority_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    policy = _RecordingPolicy("authority")
    monkeypatch.setattr(engine_module, "policy_for_engine", lambda _engine: policy)
    monkeypatch.setattr(engine_module, "policy_access_context", lambda _engine: None)
    monkeypatch.setattr(engine_module.lcm_tools, "lcm_compute", lambda *_args, **_kwargs: "compute")
    monkeypatch.setattr(
        engine_module.lcm_tools,
        "lcm_compile_evidence",
        lambda *_args, **_kwargs: "compile",
    )
    try:
        assert engine.handle_tool_call("lcm_compute", {"operands": []}) == "compute"
        assert engine.handle_tool_call("lcm_compile_evidence", {}) == "compile"
        assert [operation for operation, _scope in policy.calls] == ["write", "write"]
        assert all(scope["required_scope"] == "write" for _operation, scope in policy.calls)
    finally:
        engine.shutdown()


def test_clone_for_agent_preserves_teams_wiring(
    tmp_path: Path,
) -> None:
    prototype = _engine(tmp_path)
    prototype.lcm_teams_enabled = True
    prototype.get_lcm_access_context = lambda: None
    clone = None
    try:
        clone = prototype.clone_for_agent()
        assert clone.lcm_teams_enabled is True
        assert callable(clone.get_lcm_access_context)
        assert isinstance(engine_module.policy_for_engine(clone), FailClosedPolicy)
    finally:
        prototype.shutdown()
        if clone is not None:
            clone.shutdown()


def test_backup_maintenance_requires_owner_only_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)

    class OwnerOnlyPolicy(_RecordingPolicy):
        def authorize_operation(self, context, operation, expected_scope):
            self.calls.append((operation, dict(expected_scope)))
            if operation != "owner_only" or expected_scope.get("required_scope") != "owner_only":
                return Decision.deny("scope_forbidden")
            return Decision.allow()

    policy = OwnerOnlyPolicy("owner")
    monkeypatch.setattr(maintenance_module, "policy_for_engine", lambda _engine: policy)
    monkeypatch.setattr(maintenance_module, "policy_access_context", lambda _engine: None)
    try:
        assert maintenance_module.backup_database(engine)["ok"] is True
        assert maintenance_module.rotate_backup_database(engine)["ok"] is True
        assert [operation for operation, _scope in policy.calls] == ["owner_only", "owner_only"]
    finally:
        engine.shutdown()


def test_fail_closed_callbacks_refuse_before_rebind_ingest_or_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    try:
        monkeypatch.setattr(engine_module, "policy_for_engine", lambda _engine: FailClosedPolicy())
        monkeypatch.setattr(engine_module, "policy_access_context", lambda _engine: None)
        rebound: list[str] = []
        monkeypatch.setattr(
            engine,
            "_rebind_storage_for_home",
            lambda home: rebound.append(home),
        )
        with pytest.raises(AuthorizationRequiredError, match="authorize_operation"):
            engine.on_session_start("session-b", hermes_home=str(tmp_path / "other"))
        assert rebound == []

        ingested: list[object] = []
        monkeypatch.setattr(engine, "_ingest_messages", lambda messages: ingested.append(messages))
        with pytest.raises(AuthorizationRequiredError, match="authorize_operation"):
            engine.handle_tool_call("lcm_status", {}, messages=[{"role": "user", "content": "secret"}])
        assert ingested == []

        with pytest.raises(AuthorizationRequiredError, match="authorize_operation"):
            engine.on_session_reset()
    finally:
        engine.shutdown()


def test_rollup_denial_is_not_enqueued(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scheduled: list[object] = []

    class Scheduler:
        def schedule(self, _key, job, **_kwargs):
            scheduled.append(job)
            return True

    carrier = object.__new__(LCMEngine)
    carrier._dag = SimpleNamespace(db_path=str(tmp_path / "rollup.db"))
    carrier._config = SimpleNamespace()
    carrier._summary_circuit_breaker = None
    carrier._summary_spend_guard = None
    carrier._rollup_maintenance_owner = object()
    monkeypatch.setattr(engine_module, "_ROLLUP_MAINTENANCE_SCHEDULER", Scheduler())
    monkeypatch.setattr(engine_module, "policy_for_engine", lambda _engine: FailClosedPolicy())
    monkeypatch.setattr(engine_module, "policy_access_context", lambda _engine: None)

    carrier._schedule_rollup_maintenance("scope-a")
    assert scheduled == []


def test_rollup_enqueue_capture_cross_contamination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs: list[object] = []
    resolutions: list[str] = []
    runs: list[str] = []
    current = {"policy": _RecordingPolicy("A")}

    class Scheduler:
        def schedule(self, _key, job, **_kwargs):
            jobs.append(job)
            return True

    class PrivateDag:
        def __init__(self, _path):
            pass

        def close(self):
            pass

    def resolve(_engine):
        resolutions.append(current["policy"].label)
        return current["policy"]

    carrier = object.__new__(LCMEngine)
    carrier._dag = SimpleNamespace(db_path=str(tmp_path / "rollup.db"))
    carrier._config = SimpleNamespace()
    carrier._summary_circuit_breaker = None
    carrier._summary_spend_guard = None
    carrier._rollup_maintenance_owner = object()
    monkeypatch.setattr(engine_module, "_ROLLUP_MAINTENANCE_SCHEDULER", Scheduler())
    monkeypatch.setattr(engine_module, "policy_for_engine", resolve)
    monkeypatch.setattr(engine_module, "policy_access_context", lambda _engine: None)
    monkeypatch.setattr(engine_module, "SummaryDAG", PrivateDag)
    monkeypatch.setattr(
        engine_module,
        "run_rollup_maintenance",
        lambda _dag, _config, scope, **_kwargs: runs.append(scope),
    )

    carrier._schedule_rollup_maintenance("scope-a")
    current["policy"] = _RecordingPolicy("B")
    carrier._schedule_rollup_maintenance("scope-b")

    # A's closure runs after B has enqueued. There are exactly two caller-side
    # resolutions and no worker-side lookup, so B cannot replace A's decision.
    jobs[0]()
    assert resolutions == ["A", "B"]
    assert runs == ["scope-a"]


def test_resolve_active_engine_denies_wrong_principal_before_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = importlib.import_module("hermes_lcm.__init__")

    class WrongEngine:
        name = "lcm"
        _session_id = "session-b"
        _conversation_id = "conversation-b"

        def ingest(self, _messages):
            return None

    wrong_engine = WrongEngine()
    with engine_registry._ACTIVE_ENGINE_REGISTRY_LOCK:
        engine_registry._ACTIVE_ENGINES_BY_SESSION_ID.clear()
        engine_registry._ACTIVE_ENGINES_BY_SESSION_ID["session-b"] = wrong_engine
    # __init__ resolves the policy helpers lazily via _policy_api(), so patch
    # that seam rather than module-level names that no longer exist.
    monkeypatch.setattr(
        plugin, "_policy_api",
        lambda: ((lambda _engine: FailClosedPolicy()), (lambda _engine: None)),
    )
    try:
        authorized = plugin._authorize_active_engine_resolution(
            object(),
            session_id="session-b",
            conversation_id="conversation-b",
            operation="write",
        )
        resolved = (
            engine_registry.resolve_active_lcm_engine(session_id="session-b")
            if authorized
            else None
        )
        assert resolved is None
    finally:
        with engine_registry._ACTIVE_ENGINE_REGISTRY_LOCK:
            engine_registry._ACTIVE_ENGINES_BY_SESSION_ID.pop("session-b", None)


def test_hooked_modules_resolve_only_through_documented_policy_seam() -> None:
    for relative in (
        "engine.py",
        "__init__.py",
        "aux_session.py",
        "reset_state.py",
        "maintenance.py",
        "scripts/import_lossless_claw.py",
    ):
        path = REPO_ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "policy_for_engine"
        ]
        assert calls, relative
        source = path.read_text(encoding="utf-8")
        assert "lcm_teams_enabled" not in source
        assert "get_lcm_access_context" not in source
