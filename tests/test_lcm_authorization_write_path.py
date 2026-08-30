from __future__ import annotations

import ast
import base64
import importlib
from pathlib import Path

import pytest

from hermes_lcm.access_context import AccessContextV1, Decision
from hermes_lcm.access_context.fixtures import load_fixture
from hermes_lcm import command as command_module
from hermes_lcm import compaction as compaction_module
from hermes_lcm import engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

_access_policy = importlib.import_module("hermes_lcm.access_policy")
AuthorizationRequiredError = _access_policy.AuthorizationRequiredError
FailClosedPolicy = _access_policy.FailClosedPolicy


def _engine(tmp_path: Path, *, rollups: bool = False) -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        temporal_rollups_enabled=rollups,
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1,
        large_output_externalization_path=str(tmp_path / "payloads"),
    )
    engine = LCMEngine(config=config)
    engine._session_id = "session-a"
    engine._conversation_id = "conversation-a"
    engine._foreground_session_id = "session-a"
    engine.threshold_tokens = 10**9
    return engine


def _fail_closed(monkeypatch: pytest.MonkeyPatch, *modules: object) -> None:
    for module in modules:
        monkeypatch.setattr(module, "policy_for_engine", lambda _engine: FailClosedPolicy())
        monkeypatch.setattr(module, "policy_access_context", lambda _engine: None)


def _message_count(engine: LCMEngine) -> int:
    return int(engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])


def _metadata_snapshot(engine: LCMEngine) -> list[tuple[str, str]]:
    return list(engine._store._conn.execute("SELECT key, value FROM metadata ORDER BY key"))


def test_default_off_message_externalize_compaction_and_rollup_passthrough(tmp_path: Path) -> None:
    engine = _engine(tmp_path, rollups=True)
    try:
        payload = "data:text/plain;base64," + base64.b64encode(b"payload" * 100).decode()
        before = _message_count(engine)
        replay = engine._ingest_messages([{"role": "user", "content": payload}])
        stored = engine._store.get_session_messages("session-a")
        assert stored[0]["content"].startswith("[Externalized payload:")
        assert replay[0]["content"] == payload
        assert _message_count(engine) == before + 1
        assert list((tmp_path / "payloads").glob("*.json"))

        # The CompactionMixin carrier is the engine itself; default-off keeps
        # the existing no-op result and does not add another row.
        assert engine.compress(replay) == replay
        assert _message_count(engine) == before + 1

        result = command_module._rollups_rebuild_text(["day", "2026-01-01"], engine)
        assert "LCM temporal rollup rebuild" in result
        rows = engine._store._conn.execute("SELECT COUNT(*) FROM lcm_rollups").fetchone()[0]
        assert rows >= 1
    finally:
        engine.shutdown()


def test_fail_closed_writes_leave_store_sidecar_ledger_and_transaction_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    try:
        _fail_closed(monkeypatch, engine_module, compaction_module)
        before_rows = _message_count(engine)
        before_metadata = _metadata_snapshot(engine)
        with pytest.raises(AuthorizationRequiredError, match="authorize_operation"):
            engine._ingest_messages([{"role": "user", "content": "denied"}])
        assert _message_count(engine) == before_rows
        assert _metadata_snapshot(engine) == before_metadata
        assert engine._store._conn.in_transaction is False
        assert not (tmp_path / "payloads").exists()

        with pytest.raises(AuthorizationRequiredError, match="authorize_operation"):
            engine.compress([{"role": "user", "content": "denied compaction"}])
        assert _message_count(engine) == before_rows
        assert _metadata_snapshot(engine) == before_metadata
        assert engine._store._conn.in_transaction is False
    finally:
        engine.shutdown()


def test_fail_closed_rollup_callsite_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path, rollups=True)
    try:
        _fail_closed(monkeypatch, command_module)
        before = int(engine._store._conn.execute("SELECT COUNT(*) FROM lcm_rollups").fetchone()[0])
        with pytest.raises(AuthorizationRequiredError, match="authorize_operation"):
            command_module._rollups_rebuild_text(["day", "2026-01-01"], engine)
        assert int(engine._store._conn.execute("SELECT COUNT(*) FROM lcm_rollups").fetchone()[0]) == before
        assert engine._store._conn.in_transaction is False
    finally:
        engine.shutdown()


def test_compaction_resolves_policy_from_self_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    events: list[str] = []

    class RecordingPolicy:
        def authorize_operation(self, context, operation, expected_scope):
            events.append("authorize_operation")
            assert expected_scope["kind"] == "compaction"
            assert expected_scope["source_scope"] is None
            return Decision.allow()

        def resolve_authorized_targets(self, context, operation, requested_narrowing):
            events.append("resolve_authorized_targets")
            return requested_narrowing

        def audit_decision(self, *args):
            events.append("audit_decision")

    policy = RecordingPolicy()
    monkeypatch.setattr(compaction_module, "policy_for_engine", lambda carrier: (events.append("carrier") or policy))
    monkeypatch.setattr(compaction_module, "policy_access_context", lambda carrier: None)
    try:
        assert engine.compress([{"role": "user", "content": "carrier"}])
        assert events[:2] == ["carrier", "authorize_operation"]
        assert "resolve_authorized_targets" in events
    finally:
        engine.shutdown()


def test_non_widening_rollup_scope_refuses_fixture_widening(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = load_fixture(Path(__file__).parent / "fixtures/access_context_v1/derivation/rollup-negative.json")
    source = AccessContextV1.from_payload(payload["context"])
    widened = AccessContextV1.from_payload(payload["derived"])
    engine = _engine(tmp_path, rollups=True)

    class WideningPolicy:
        def authorize_operation(self, context, operation, expected_scope):
            return Decision.allow()

        def resolve_authorized_targets(self, context, operation, requested_narrowing):
            return {**requested_narrowing, "derived_scope": widened}

        def audit_decision(self, *args):
            pass

    policy = WideningPolicy()
    monkeypatch.setattr(command_module, "policy_for_engine", lambda engine: policy)
    monkeypatch.setattr(command_module, "policy_access_context", lambda engine: source)
    try:
        with pytest.raises(
            AuthorizationRequiredError, match="target_not_found_or_forbidden"
        ):
            command_module._rollups_rebuild_text(["day", "2026-01-01"], engine)
        assert engine._store._conn.execute("SELECT COUNT(*) FROM lcm_rollups").fetchone()[0] == 0
    finally:
        engine.shutdown()


def test_rollup_rebuild_rejects_partition_narrowing_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path, rollups=True)

    class WrongPartitionPolicy:
        def authorize_operation(self, context, operation, expected_scope):
            assert expected_scope["partition_scope"] == "session-a"
            return Decision.allow()

        def resolve_authorized_targets(self, context, operation, requested_narrowing):
            return {**requested_narrowing, "partition_scope": "other-session"}

        def audit_decision(self, *args):
            pass

    monkeypatch.setattr(command_module, "policy_for_engine", lambda _engine: WrongPartitionPolicy())
    monkeypatch.setattr(command_module, "policy_access_context", lambda _engine: None)
    try:
        with pytest.raises(
            AuthorizationRequiredError, match="target_not_found_or_forbidden"
        ):
            command_module._rollups_rebuild_text(["day", "2026-01-01"], engine)
        assert engine._store._conn.execute("SELECT COUNT(*) FROM lcm_rollups").fetchone()[0] == 0
    finally:
        engine.shutdown()


def test_tool_narrowing_removes_omitted_target_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    policy = type(
        "NarrowingPolicy",
        (),
        {
            "authorize_operation": lambda self, context, operation, expected_scope: Decision.allow(),
            "resolve_authorized_targets": lambda self, context, operation, expected_scope: {
                "target_scope": {"session_id": "narrowed-session"}
            },
            "audit_decision": lambda self, *args: None,
        },
    )()
    monkeypatch.setattr(engine_module, "policy_for_engine", lambda _engine: policy)
    monkeypatch.setattr(engine_module, "policy_access_context", lambda _engine: None)
    monkeypatch.setattr(
        engine_module.lcm_tools,
        "lcm_grep",
        lambda args, engine: args,
    )
    try:
        args = engine.handle_tool_call(
            "lcm_grep",
            {"query": "needle", "session_scope": "all", "session_id": "wide"},
        )
        assert args == {"query": "needle", "session_id": "narrowed-session"}
    finally:
        engine.shutdown()


def test_write_arms_resolve_only_through_documented_policy_seam() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("engine.py", "compaction.py", "command.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "policy_for_engine"
        ]
        assert calls, name
        source = (root / name).read_text(encoding="utf-8")
        assert "lcm_teams_enabled" not in source
        assert "get_lcm_access_context" not in source
