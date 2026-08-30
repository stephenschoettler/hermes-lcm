from __future__ import annotations

import ast
import importlib
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm.access_context import Decision, DenialReason
_access_policy = importlib.import_module("hermes_lcm.access_policy")
AuthorizationRequiredError = _access_policy.AuthorizationRequiredError
FailClosedPolicy = _access_policy.FailClosedPolicy
TrustedOwnerPolicy = _access_policy.TrustedOwnerPolicy
policy_for_engine = _access_policy.policy_for_engine
import hermes_lcm.retrieval_core as retrieval_core
import hermes_lcm.tools as lcm_tools


REPO_ROOT = Path(__file__).resolve().parents[1]


def _engine(tmp_path: Path, *, teams_enabled: bool | None = None) -> SimpleNamespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "read-path.db"
    sqlite3.connect(db_path).close()
    values = {
        "_store": SimpleNamespace(db_path=str(db_path)),
        "_config": SimpleNamespace(),
        "_dag": SimpleNamespace(_conn=None),
    }
    if teams_enabled is not None:
        values["lcm_teams_enabled"] = teams_enabled
    return SimpleNamespace(**values)


def _deadline() -> float:
    return time.monotonic() + 30.0


class _RecordingPolicy:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.stored_scopes: list[dict[str, object]] = []

    def authorize_operation(self, context, operation, expected_scope):
        self.events.append("authorize_operation")
        return Decision.allow()

    def resolve_authorized_targets(self, context, operation, requested_narrowing):
        self.events.append("resolve_authorized_targets")
        return dict(requested_narrowing)

    def authorize_stored_scope(self, context, operation, stored_scope):
        self.events.append("authorize_stored_scope")
        self.stored_scopes.append(dict(stored_scope))
        return Decision.allow()

    def audit_decision(self, context, operation, internal_reason, public_result):
        self.events.append("audit_decision")


def test_default_off_passthrough_for_all_read_arms(tmp_path: Path) -> None:
    """No Teams wiring keeps the existing trusted-owner return values."""
    engine = _engine(tmp_path)
    assert not hasattr(engine, "lcm_teams_enabled")
    assert not hasattr(engine, "get_lcm_access_context")
    knn_result = object()

    class VectorStore:
        _supports_pooling = False

        def __init__(self, *_args, **_kwargs):
            pass

        def knn(self, *_args, **_kwargs):
            return knn_result

        def close(self):
            pass

    provider = SimpleNamespace(model_id="model", provider_id="provider")
    assert (
        retrieval_core.run_knn(
            engine,
            query_vector=[1.0],
            provider=provider,
            knn_limit=2,
            deadline=_deadline(),
            since=None,
            until=None,
            conversation_ids=None,
            source=None,
            vector_store_cls=VectorStore,
        )
        is knn_result
    )
    assert retrieval_core.hydrate_chunk_hits(
        engine, ranked_rows=[], knn_limit=2, deadline=_deadline(), snippet_chars=40
    ) == []
    assert retrieval_core.hydrate_semantic_nodes(
        engine, ranked_rows=[], knn_limit=2, deadline=_deadline()
    ) == []
    assert isinstance(policy_for_engine(engine), TrustedOwnerPolicy)


def test_run_knn_authorizes_before_real_query_and_scan_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real run_knn path gates both query issuance and scan-bound setup."""
    engine = _engine(tmp_path)
    events: list[str] = []
    policy = _RecordingPolicy(events)
    monkeypatch.setattr(retrieval_core, "policy_for_engine", lambda _engine: policy)
    monkeypatch.setattr(retrieval_core, "policy_access_context", lambda _engine: None)
    result = object()

    class VectorStore:
        _supports_pooling = False

        def __init__(self, *_args, bounded_scan_rows=None, **_kwargs):
            events.append("scan_bounds_applied")
            assert bounded_scan_rows == 17

        def knn(self, *_args, **kwargs):
            events.append("knn_query_issued")
            assert kwargs["full_scan"] is True
            assert kwargs["scan_max_rows"] == 23
            assert kwargs["scan_budget_s"] == 0.5
            return result

        def close(self):
            pass

    provider = SimpleNamespace(model_id="model", provider_id="provider")
    returned = retrieval_core.run_knn(
        engine,
        query_vector=[1.0],
        provider=provider,
        knn_limit=2,
        deadline=_deadline(),
        since=None,
        until=None,
        conversation_ids=["session-a"],
        source="source-a",
        vector_store_cls=VectorStore,
        scan_rows=17,
        full_scan=True,
        scan_max_rows=23,
        scan_budget_s=0.5,
    )

    assert returned is result
    assert events.index("authorize_operation") < events.index("scan_bounds_applied")
    assert events.index("resolve_authorized_targets") < events.index("scan_bounds_applied")
    assert events.index("authorize_operation") < events.index("knn_query_issued")
    assert events.index("resolve_authorized_targets") < events.index("knn_query_issued")


def test_fail_closed_read_arms_disclose_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A denying policy stops KNN and both hydration arms before disclosure."""
    engine = _engine(tmp_path)
    policy = FailClosedPolicy()
    monkeypatch.setattr(retrieval_core, "policy_for_engine", lambda _engine: policy)
    monkeypatch.setattr(retrieval_core, "policy_access_context", lambda _engine: None)
    vector_store_calls: list[str] = []

    class VectorStore:
        _supports_pooling = False

        def __init__(self, *_args, **_kwargs):
            vector_store_calls.append("constructed")

        def close(self):
            pass

    provider = SimpleNamespace(model_id="model", provider_id="provider")
    with pytest.raises(AuthorizationRequiredError, match="authorize_operation"):
        retrieval_core.run_knn(
            engine,
            query_vector=[1.0],
            provider=provider,
            knn_limit=2,
            deadline=_deadline(),
            since=None,
            until=None,
            conversation_ids=None,
            source=None,
            vector_store_cls=VectorStore,
        )
    assert vector_store_calls == []

    with pytest.raises(AuthorizationRequiredError, match="authorize_stored_scope"):
        retrieval_core.hydrate_chunk_hits(
            engine,
            ranked_rows=[("7:0", 1.0, "chunk")],
            knn_limit=2,
            deadline=_deadline(),
            snippet_chars=40,
        )
    with pytest.raises(AuthorizationRequiredError, match="authorize_stored_scope"):
        retrieval_core.hydrate_semantic_nodes(
            engine,
            ranked_rows=[(7, 1.0, "summary")],
            knn_limit=2,
            deadline=_deadline(),
        )


class _TrackingConnection:
    def __init__(self) -> None:
        self.closed = False
        self.row_factory = None

    def execute(self, *_args, **_kwargs):
        return []

    def set_progress_handler(self, *_args, **_kwargs):
        pass

    def close(self) -> None:
        self.closed = True


def test_denial_mid_hydration_closes_every_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hydration finally blocks close a connection after a mid-arm denial."""
    engine = _engine(tmp_path)
    policy = FailClosedPolicy()
    monkeypatch.setattr(retrieval_core, "policy_for_engine", lambda _engine: policy)
    monkeypatch.setattr(retrieval_core, "policy_access_context", lambda _engine: None)

    for hydrate, ranked_rows in (
        (retrieval_core.hydrate_chunk_hits, [("7:0", 1.0, "chunk")]),
        (retrieval_core.hydrate_semantic_nodes, [(7, 1.0, "summary")]),
    ):
        connection = _TrackingConnection()
        monkeypatch.setattr(retrieval_core.sqlite3, "connect", lambda *a, _c=connection, **k: _c)
        kwargs = {
            "engine": engine,
            "ranked_rows": ranked_rows,
            "knn_limit": 2,
            "deadline": _deadline(),
        }
        if hydrate is retrieval_core.hydrate_chunk_hits:
            kwargs["snippet_chars"] = 40
        with pytest.raises(AuthorizationRequiredError, match="authorize_stored_scope"):
            hydrate(**kwargs)
        assert connection.closed is True


def test_stored_scope_authorization_precedes_hydration_disclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both hydration arms authorize stored scope before reading content."""
    chunk_events: list[str] = []
    chunk_engine = _engine(tmp_path / "chunk")
    chunk_policy = _RecordingPolicy(chunk_events)
    monkeypatch.setattr(retrieval_core, "policy_for_engine", lambda _engine: chunk_policy)
    monkeypatch.setattr(retrieval_core, "policy_access_context", lambda _engine: None)

    chunk_row = {
        "chunk_id": "7:0",
        "store_id": 7,
        "chunk_index": 0,
        "char_start": 0,
        "char_end": 5,
        "session_id": "session-a",
        "source": "source-a",
        "role": "user",
        "timestamp": 1,
        "content": "hello world",
    }

    class ScopeRow(dict):
        def __getitem__(self, key):
            if key == "content":
                chunk_events.append("content_read")
            return super().__getitem__(key)

    class ChunkConnection:
        row_factory = None

        def execute(self, sql, *_args, **_kwargs):
            if "access_scope" in sql:
                return [ScopeRow({**chunk_row, "access_scope": "principal-a"})]
            if "SELECT" in sql:
                chunk_events.append("content_read")
                return [chunk_row]
            return []

        def set_progress_handler(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(retrieval_core.sqlite3, "connect", lambda *a, **k: ChunkConnection())
    chunk_hits = retrieval_core.hydrate_chunk_hits(
        chunk_engine,
        ranked_rows=[("7:0", 1.0, "chunk")],
        knn_limit=1,
        deadline=_deadline(),
        snippet_chars=40,
    )
    assert chunk_hits[0][0]["snippet"] == "hello"
    assert chunk_policy.stored_scopes[0]["access_scope"] == "principal-a"
    assert chunk_events.index("authorize_stored_scope") < chunk_events.index("content_read")

    semantic_events: list[str] = []
    semantic_engine = _engine(tmp_path / "semantic")
    semantic_policy = _RecordingPolicy(semantic_events)
    monkeypatch.setattr(retrieval_core, "policy_for_engine", lambda _engine: semantic_policy)

    class Dag:
        def get_node(self, node_id):
            semantic_events.append("content_read")
            return SimpleNamespace(node_id=node_id)

    semantic_engine._dag = Dag()

    class SemanticConnection:
        row_factory = None

        def execute(self, *_args, **_kwargs):
            if _args and "access_scope" in _args[0]:
                return [{"node_id": 7, "access_scope": "principal-a"}]
            return []

        def set_progress_handler(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(retrieval_core.sqlite3, "connect", lambda *a, **k: SemanticConnection())
    semantic_hits = retrieval_core.hydrate_semantic_nodes(
        semantic_engine,
        ranked_rows=[(7, 1.0, "summary")],
        knn_limit=1,
        deadline=_deadline(),
    )
    assert semantic_hits[0][0].node_id == 7
    assert semantic_policy.stored_scopes[0]["access_scope"] == "principal-a"
    assert semantic_events.index("authorize_stored_scope") < semantic_events.index("content_read")


def test_enabled_but_unwired_engine_fails_closed_on_every_arm(tmp_path: Path) -> None:
    """Teams enabled without its accessor must never inherit default-off trust."""
    engine = _engine(tmp_path, teams_enabled=True)
    policy = policy_for_engine(engine)
    assert isinstance(policy, FailClosedPolicy)
    assert policy.denial_reason is DenialReason.CONTEXT_MISSING

    class VectorStore:
        _supports_pooling = False

        def __init__(self, *_args, **_kwargs):
            raise AssertionError("KNN must not be reached after denial")

    provider = SimpleNamespace(model_id="model", provider_id="provider")
    with pytest.raises(AuthorizationRequiredError, match="authorize_operation"):
        retrieval_core.run_knn(
            engine,
            query_vector=[1.0],
            provider=provider,
            knn_limit=2,
            deadline=_deadline(),
            since=None,
            until=None,
            conversation_ids=None,
            source=None,
            vector_store_cls=VectorStore,
        )
    with pytest.raises(AuthorizationRequiredError, match="authorize_stored_scope"):
        retrieval_core.hydrate_chunk_hits(
            engine,
            ranked_rows=[("7:0", 1.0, "chunk")],
            knn_limit=1,
            deadline=_deadline(),
            snippet_chars=40,
        )
    with pytest.raises(AuthorizationRequiredError, match="authorize_stored_scope"):
        retrieval_core.hydrate_semantic_nodes(
            engine,
            ranked_rows=[(7, 1.0, "summary")],
            knn_limit=1,
            deadline=_deadline(),
        )


def test_recall_fts_and_chunk_arms_authorize_before_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _engine(tmp_path)
    policy = FailClosedPolicy()
    monkeypatch.setattr(lcm_tools, "policy_for_engine", lambda _engine: policy)
    monkeypatch.setattr(lcm_tools, "policy_access_context", lambda _engine: None)
    with pytest.raises(AuthorizationRequiredError, match="authorize_operation"):
        lcm_tools._lcm_recall_fts_arm(
            engine, "secret", candidate_limit=2, deadline=_deadline()
        )

    monkeypatch.setattr(retrieval_core, "policy_for_engine", lambda _engine: policy)
    monkeypatch.setattr(retrieval_core, "policy_access_context", lambda _engine: None)
    provider = SimpleNamespace(model_id="model", provider_id="provider")
    with pytest.raises(AuthorizationRequiredError, match="authorize_operation"):
        retrieval_core.run_chunk_knn(
            engine,
            query_vector=[1.0],
            provider=provider,
            knn_limit=2,
            deadline=_deadline(),
            since=None,
            until=None,
            conversation_ids=None,
            source=None,
            vector_store_cls=object,
        )


def test_every_read_arm_resolves_only_via_policy_for_engine() -> None:
    """AST guard prevents direct engine reads or permissive policy cascades."""
    source_path = REPO_ROOT / "retrieval_core.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    arm_names = ("run_knn", "hydrate_chunk_hits", "hydrate_semantic_nodes")
    forbidden_engine_attrs = {"lcm_teams_enabled", "get_lcm_access_context"}
    forbidden_policy_names = {
        "resolve_policy",
        "TrustedOwnerPolicy",
        "FailClosedPolicy",
    }

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in arm_names
    }
    assert set(functions) == set(arm_names)
    for arm_name in arm_names:
        arm = functions[arm_name]
        policy_assignments = [
            node
            for node in ast.walk(arm)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "policy"
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
        ]
        assert len(policy_assignments) == 1, arm_name
        assignment = policy_assignments[0]
        value = assignment.value
        assert isinstance(value, ast.Call)
        assert isinstance(value.func, ast.Name)
        assert value.func.id == "policy_for_engine"
        assert len(value.args) == 1
        assert isinstance(value.args[0], ast.Name)
        assert value.args[0].id == "engine"

        direct_engine_reads: list[str] = []
        alternate_policy_resolutions: list[str] = []
        for node in ast.walk(arm):
            if isinstance(node, ast.Attribute):
                if node.attr in forbidden_engine_attrs:
                    direct_engine_reads.append(node.attr)
                if node.attr in forbidden_policy_names:
                    alternate_policy_resolutions.append(node.attr)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in forbidden_policy_names:
                    alternate_policy_resolutions.append(node.func.id)
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in forbidden_engine_attrs
                ):
                    direct_engine_reads.append(str(node.args[1].value))
        assert direct_engine_reads == []
        assert alternate_policy_resolutions == []
