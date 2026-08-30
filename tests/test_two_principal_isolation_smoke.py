"""Two-principal composition smoke for the Teams authorization seam.

This deliberately runs against the built seam, using the policy the engine
actually resolves -- which is now TeamsPolicy, not the #483 placeholder. The
isolation test is no longer xfail.

Read the two together or neither means anything: the isolation body proves B
cannot reach A, and the positive control inside it proves A still can. A policy
that denied everything would pass the first and fail the second.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm.access_context import AccessContextV1, Decision, DenialReason
from hermes_lcm.access_context.denials import PublicDecision
from hermes_lcm import engine as engine_module
from hermes_lcm import maintenance as maintenance_module
from hermes_lcm import retrieval_core
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import externalize_ingest_payload
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.teams import CatalogRevisions, ensure_teams_catalog, set_revisions
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore


SECRET = "principal-A secret message"
SIDECAR_SECRET = "principal-A externalized sidecar secret"
COLLECTION_A = "collection-a"
COLLECTION_B = "collection-b"
SESSION_A = "session-a"
SESSION_B = "session-b"
CONVERSATION_A = "conversation-a"
PERMISSIVE = "hook present but resolves permissive at access_policy/resolution.py:79"


def _context(
    *, principal_id: str, profile_id: str, session_id: str, collection: str
) -> AccessContextV1:
    now = datetime.now(timezone.utc)
    return AccessContextV1.from_host(
        authenticated_transport="smoke-test",
        context_id=f"ctx-{principal_id}",
        request_id=f"req-{principal_id}",
        source_kind="smoke-test",
        deployment_id="deployment-smoke",
        tenant_id="tenant-smoke",
        principal_id=principal_id,
        profile_id=profile_id,
        profile_incarnation=f"inc-{principal_id}",
        session_id=session_id,
        session_owner_principal_id=principal_id,
        conversation_id=CONVERSATION_A if principal_id == "A" else "conversation-b",
        conversation_lane="smoke",
        default_write_collection_id=collection,
        read_policy_ref=f"teams://{collection}",
        grants={"read", "write"},
        policy_revision=1,
        membership_revision=1,
        revocation_epoch=1,
        lease_id=f"lease-{principal_id}",
        lease_generation=1,
        ownership_generation=1,
        issued_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(hours=1),
        narrowing={
            "operation:read",
            "operation:write",
            f"collection:{collection}",
        },
    )


def _engine(
    db_path: Path,
    context: AccessContextV1,
    *,
    teams_enabled: bool,
) -> LCMEngine:
    config = LCMConfig(
        database_path=str(db_path),
        temporal_rollups_enabled=True,
        new_session_retain_depth=-1,
        embeddings_enabled=True,
        embedding_provider="smoke",
        embedding_model="smoke-model",
        # Seed the primary message verbatim; the sidecar below is created
        # explicitly so this smoke can exercise both ordinary and externalized
        # data without turning every fixture row into a placeholder.
        large_output_externalization_enabled=False,
        large_output_externalization_threshold_chars=1,
        large_output_externalization_path=str(db_path.parent / "payloads"),
    )
    engine = LCMEngine(config=config, hermes_home=str(db_path.parent))
    # These are the exact module-constant names in access_policy/resolution.py.
    engine.lcm_teams_enabled = teams_enabled
    if teams_enabled:
        # A real Teams store always has a catalog -- enable_teams creates it --
        # and the catalog OWNS the revisions a context is validated against.
        # Seed it at this context's own revisions, which is what provisioning
        # does: a tenant is created at whatever revisions its control plane has
        # already issued contexts against. Without this the store reads every
        # context as revoked, and the positive control goes red for a reason
        # that has nothing to do with isolation.
        ensure_teams_catalog(engine._store.connection)
        set_revisions(
            engine._store.connection,
            context.tenant_id,
            CatalogRevisions(
                policy_revision=context.policy_revision,
                membership_revision=context.membership_revision,
                revocation_epoch=context.revocation_epoch,
            ),
        )
    engine.get_lcm_access_context = lambda context=context: context
    engine._session_id = context.session_id
    engine._conversation_id = context.conversation_id
    engine._foreground_session_id = context.session_id
    engine._session_platform = context.default_write_collection_id
    return engine


def _seed_a(engine: LCMEngine) -> dict[str, object]:
    store_ids = engine._store.append_batch(
        SESSION_A,
        [
            {"role": "user", "content": SECRET},
            {"role": "assistant", "content": "A-only answer"},
        ],
        source=COLLECTION_A,
        conversation_id=CONVERSATION_A,
    )
    sidecar = externalize_ingest_payload(
        SIDECAR_SECRET,
        role="tool",
        session_id=SESSION_A,
        field_path="result.content",
        config=engine._config,
        hermes_home=engine._hermes_home,
    )
    sidecar_id = None
    sidecar_ref = None
    if sidecar is not None:
        sidecar_ref = sidecar["path"].name
        sidecar_id = engine._store.append(
            SESSION_A,
            {"role": "tool", "content": sidecar["placeholder"]},
            source=COLLECTION_A,
            conversation_id=CONVERSATION_A,
        )

    source_ids = [*store_ids, *([sidecar_id] if sidecar_id is not None else [])]
    summary_time = datetime(2026, 8, 5, tzinfo=timezone.utc).timestamp()
    node_id = engine._dag.add_node(
        SummaryNode(
            session_id=SESSION_A,
            depth=0,
            summary=f"Summary containing {SECRET}",
            token_count=8,
            source_token_count=24,
            source_ids=source_ids,
            source_type="messages",
            earliest_at=summary_time,
            latest_at=summary_time + 1,
            expand_hint="Expand A-only summary",
        )
    )

    vectors = VectorStore(engine._store.db_path, config=engine._config)
    try:
        vectors.register_profile("smoke-model", "smoke", 2, task="summary")
        summary_identity = vectors.capture_identity("smoke-model", provider="smoke")
        vectors.record_embedding(
            str(node_id),
            "summary",
            "smoke-model",
            [1.0, 0.0],
            identity=summary_identity,
        )
        vectors.register_profile("smoke-model", "smoke", 2, task="chunk")
        chunk_identity = EmbeddingIdentity.canonical(
            "smoke", "smoke-model", "", 2, "float32", "little", "chunk"
        )
        vectors.record_chunk_embedding(
            f"{store_ids[0]}:0",
            "smoke-model",
            [1.0, 0.0],
            store_id=store_ids[0],
            chunk_index=0,
            char_start=0,
            char_end=len(SECRET),
            token_estimate=8,
            identity=chunk_identity,
        )
    finally:
        vectors.close()

    rollups = RollupStore(engine._dag.db_path)
    try:
        token = rollups.upsert_building("day", "2026-08-05", SESSION_A)
        rollups.mark_ready(token, f"Rollup containing {SECRET}", 8, [node_id], "smoke")
    finally:
        rollups.close()

    return {
        "store_id": store_ids[0],
        "node_id": node_id,
        "chunk_id": f"{store_ids[0]}:0",
        "sidecar_ref": sidecar_ref,
    }


class _RecordingScheduler:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def schedule(self, *args: object, **kwargs: object) -> bool:
        self.calls.append((args, kwargs))
        return True


class _TargetBoundaryPolicy:
    """Small policy double for the seam-only target-confusion legs.

    The real Teams policy is intentionally permissive until #483.  This
    policy is limited to proving that the boundary asks about the supplied
    target and refuses A's known identifiers for principal B; it is not used
    for the composition smoke or any production path.
    """

    def __init__(self, blocked_values: set[object]) -> None:
        self.blocked_values = blocked_values
        self.calls: list[dict[str, object]] = []
        self.resolutions = 0

    def authorize_operation(self, context, operation, expected_scope):
        scope = dict(expected_scope)
        self.calls.append(scope)
        target = scope.get("target_scope")
        if isinstance(target, dict) and any(
            value in self.blocked_values for value in target.values()
        ):
            return Decision.deny(DenialReason.SCOPE_FORBIDDEN)
        return Decision.allow()

    def resolve_authorized_targets(self, context, operation, requested_narrowing):
        self.resolutions += 1
        return requested_narrowing

    def audit_decision(self, *args):
        return None


def test_tool_boundary_authorizes_caller_supplied_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Target-confusion legs pass through one enforcing boundary."""
    context_a = _context(
        principal_id="A", profile_id="profile-a", session_id=SESSION_A, collection=COLLECTION_A
    )
    context_b = _context(
        principal_id="B", profile_id="profile-b", session_id=SESSION_B, collection=COLLECTION_B
    )
    engine_a = _engine(tmp_path / "boundary.db", context_a, teams_enabled=True)
    engine_b = _engine(tmp_path / "boundary.db", context_b, teams_enabled=True)
    try:
        seed = _seed_a(engine_a)
        policy = _TargetBoundaryPolicy(
            {SESSION_A, COLLECTION_A, seed["store_id"], seed["node_id"], seed["sidecar_ref"]}
        )
        monkeypatch.setattr(engine_module, "policy_for_engine", lambda _engine: policy)

        attempts = (
            ("lcm_load_session", {"session_id": SESSION_A, "limit": 10}),
            ("lcm_expand", {"store_id": seed["store_id"]}),
            ("lcm_expand", {"externalized_ref": seed["sidecar_ref"]}),
            (
                "lcm_grep",
                {"query": SECRET, "session_scope": "session", "session_id": SESSION_A},
            ),
            ("lcm_describe", {"node_id": seed["node_id"]}),
        )
        for name, args in attempts:
            with pytest.raises(engine_module.AuthorizationRequiredError):
                engine_b.handle_tool_call(name, args)

        load_scope = next(
            scope for scope in policy.calls if scope.get("tool_name") == "lcm_load_session"
        )
        assert load_scope["caller_session_id"] == SESSION_B
        assert load_scope["target_scope"] == {"session_id": SESSION_A}
        assert load_scope["session_id"] == SESSION_A
        assert policy.resolutions == 0  # denied before resolution
    finally:
        engine_b.shutdown()
        engine_a.shutdown()


def _target_a(engine: LCMEngine) -> None:
    """Model a host attempting to bind B to A's target session/scope."""
    engine._session_id = SESSION_A
    engine._conversation_id = CONVERSATION_A
    engine._foreground_session_id = SESSION_A
    engine._session_platform = COLLECTION_A


def _run_positive_control(
    engine: LCMEngine,
    seed: dict[str, object],
    scheduler: _RecordingScheduler,
) -> list[str]:
    failures: list[str] = []

    def check(name: str, action) -> object | None:
        try:
            return action()
        except Exception as exc:  # pragma: no cover - rendered in failure report
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            return None

    provider = SimpleNamespace(model_id="smoke-model", provider_id="smoke")
    knn = check(
        "retrieve",
        lambda: retrieval_core.run_knn(
            engine,
            query_vector=[1.0, 0.0],
            provider=provider,
            knn_limit=3,
            deadline=time.monotonic() + 10,
            since=None,
            until=None,
            conversation_ids=None,
            source=COLLECTION_A,
            vector_store_cls=VectorStore,
        ),
    )
    if knn is None or not list(knn):
        failures.append("retrieve: no A result")

    semantic = check(
        "semantic hydration",
        lambda: retrieval_core.hydrate_semantic_nodes(
            engine,
            ranked_rows=[(seed["node_id"], 1.0, "summary")],
            knn_limit=1,
            deadline=time.monotonic() + 10,
        ),
    )
    if not semantic or SECRET not in semantic[0][0].summary:
        failures.append("semantic hydration: A summary unavailable")

    chunks = check(
        "chunk hydration",
        lambda: retrieval_core.hydrate_chunk_hits(
            engine,
            ranked_rows=[(seed["chunk_id"], 1.0, "chunk")],
            knn_limit=1,
            deadline=time.monotonic() + 10,
            snippet_chars=200,
        ),
    )
    if not chunks or SECRET not in chunks[0][0]["snippet"]:
        failures.append("chunk hydration: A content unavailable")

    loaded = check(
        "enumerate/count",
        lambda: json.loads(
            engine.handle_tool_call(
                "lcm_load_session", {"session_id": SESSION_A, "limit": 10}
            )
        ),
    )
    if not loaded or loaded.get("total_messages", 0) < 2:
        failures.append("enumerate/count: A count unavailable")

    expanded = check(
        "expand",
        lambda: json.loads(
            engine.handle_tool_call("lcm_expand", {"store_id": seed["store_id"]})
        ),
    )
    if not expanded or SECRET not in expanded.get("content", ""):
        failures.append("expand: A content unavailable")
    if seed.get("sidecar_ref"):
        sidecar = check(
            "externalized sidecar expand",
            lambda: json.loads(
                engine.handle_tool_call(
                    "lcm_expand", {"externalized_ref": seed["sidecar_ref"]}
                )
            ),
        )
        if not sidecar or SIDECAR_SECRET not in sidecar.get("content", ""):
            failures.append("externalized sidecar expand: A content unavailable")

    before = engine._store.get_session_count(SESSION_A)
    check(
        "write",
        lambda: engine._ingest_messages(
            [{"role": "user", "content": "A positive-control write"}]
        ),
    )
    if engine._store.get_session_count(SESSION_A) <= before:
        failures.append("write: A write did not persist")

    check(
        "session load/resume",
        lambda: engine.on_session_start(
            SESSION_A, conversation_id=CONVERSATION_A, platform=COLLECTION_A
        ),
    )
    if engine.current_session_id != SESSION_A:
        failures.append("session load/resume: A session not bound")

    check("rollup enqueue", lambda: engine._schedule_rollup_maintenance(SESSION_A))
    if not scheduler.calls:
        failures.append("rollup enqueue: A job not accepted")
    recent = check(
        "rollup receive",
        lambda: json.loads(
            engine.handle_tool_call("lcm_recent", {"period": "date:2026-08-05"})
        ),
    )
    if not recent or SECRET not in json.dumps(recent):
        failures.append("rollup receive: A rollup/summary unavailable")

    # maintenance backup is deliberately NOT here. It copies the whole store --
    # every principal's memory into one file -- so under Teams it is an
    # administrative capability, not something principal A inherits by being
    # the one who asked. Asserting A can back up encoded pre-Teams semantics.
    # It is now pinned by test_store_wide_backup_is_admin_only_under_teams,
    # which checks BOTH principals are refused and that Teams-off is unaffected
    # -- a stronger claim than this leg made.

    check("maintenance reset", engine.on_session_reset)
    return failures


def _b_leaks(
    engine: LCMEngine,
    seed: dict[str, object],
    scheduler: _RecordingScheduler,
) -> list[str]:
    """Return concrete B observations; each non-empty entry is a leak."""
    leaks: list[str] = []
    provider = SimpleNamespace(model_id="smoke-model", provider_id="smoke")

    try:
        result = retrieval_core.run_knn(
            engine,
            query_vector=[1.0, 0.0],
            provider=provider,
            knn_limit=3,
            deadline=time.monotonic() + 10,
            since=None,
            until=None,
            conversation_ids=None,
            source=COLLECTION_A,
            vector_store_cls=VectorStore,
        )
        rows = list(result)
        if rows:
            leaks.append(
                f"1 retrieve: B observed {len(rows)} A vector handle(s)/node(s) "
                f"({PERMISSIVE})"
            )
    except Exception:
        pass

    try:
        semantic = retrieval_core.hydrate_semantic_nodes(
            engine,
            ranked_rows=[(seed["node_id"], 1.0, "summary")],
            knn_limit=1,
            deadline=time.monotonic() + 10,
        )
        if semantic:
            leaks.append(
                f"3 semantic expand/hydrate: B observed A content {semantic[0][0].summary!r} "
                f"({PERMISSIVE})"
            )
    except Exception:
        pass

    try:
        chunks = retrieval_core.hydrate_chunk_hits(
            engine,
            ranked_rows=[(seed["chunk_id"], 1.0, "chunk")],
            knn_limit=1,
            deadline=time.monotonic() + 10,
            snippet_chars=200,
        )
        if chunks:
            leaks.append(
            f"3 chunk expand/hydrate: B observed A content {chunks[0][0]['snippet']!r} "
                f"({PERMISSIVE})"
            )
    except Exception:
        pass

    try:
        loaded = json.loads(
            engine.handle_tool_call(
                "lcm_load_session", {"session_id": SESSION_A, "limit": 10}
            )
        )
        if loaded.get("total_messages", 0) or loaded.get("messages"):
            leaks.append(
                f"2 enumerate/count: B observed count={loaded.get('total_messages')} "
                f"and message rows ({PERMISSIVE})"
            )
    except Exception:
        pass

    try:
        expanded = json.loads(
            engine.handle_tool_call("lcm_expand", {"store_id": seed["store_id"]})
        )
        if SECRET in expanded.get("content", ""):
            leaks.append(
                f"3 expand: B observed content={expanded.get('content')!r} "
                f"({PERMISSIVE})"
            )
    except Exception:
        pass
    if seed.get("sidecar_ref"):
        try:
            sidecar = json.loads(
                engine.handle_tool_call(
                    "lcm_expand", {"externalized_ref": seed["sidecar_ref"]}
                )
            )
            if SIDECAR_SECRET in sidecar.get("content", ""):
                leaks.append(
                    f"3 externalized expand: B observed sidecar content={sidecar.get('content')!r} "
                    f"({PERMISSIVE})"
                )
        except Exception:
            pass

    # The remaining legs explicitly attempt to operate on A's bound session
    # rather than supplying a target argument to a tool. Those are the
    # pre-existing Teams-policy placeholder legs until #483.
    _target_a(engine)
    before = engine._store.get_session_count(SESSION_A)
    try:
        engine._ingest_messages([{"role": "user", "content": "B write into A"}])
    except Exception:
        pass
    after = engine._store.get_session_count(SESSION_A)
    if after > before:
        leaks.append(
            f"4 write: B increased A count {before}->{after} "
            f"({PERMISSIVE})"
        )

    try:
        engine.on_session_start(SESSION_A, conversation_id=CONVERSATION_A, platform=COLLECTION_A)
        if engine.current_session_id == SESSION_A:
            leaks.append(
                "5 session load/resume: B bound A session handle "
                f"({PERMISSIVE})"
            )
    except Exception:
        pass

    before_jobs = len(scheduler.calls)
    try:
        engine._schedule_rollup_maintenance(SESSION_A)
    except Exception:
        pass
    if len(scheduler.calls) > before_jobs:
        leaks.append(
            "6 rollup enqueue: B enqueued A scope work "
            f"({PERMISSIVE})"
        )
    try:
        recent = json.loads(
            engine.handle_tool_call("lcm_recent", {"period": "date:2026-08-05"})
        )
        if SECRET in json.dumps(recent):
            leaks.append(
                "6 rollup receive: B received A rollup content "
                f"({PERMISSIVE})"
            )
    except Exception:
        pass

    try:
        backup = maintenance_module.backup_database(engine)
        if backup.get("ok"):
            leaks.append(
                f"7 maintenance backup: B observed backup handle={backup.get('backup_path')} "
                f"({PERMISSIVE})"
            )
    except Exception:
        pass

    before_reset = getattr(engine, "_pending_reset_session_id", "")
    try:
        engine.on_session_reset()
        if engine._pending_reset_session_id == SESSION_A:
            leaks.append(
                f"7 maintenance reset: B changed pending reset state {before_reset!r}->{SESSION_A!r} "
                f"({PERMISSIVE})"
            )
    except Exception:
        pass
    return leaks


def test_two_principal_isolation_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Teams-on composition must isolate B while preserving A's full path."""
    db_path = tmp_path / "teams-on.db"
    context_a = _context(
        principal_id="A", profile_id="profile-a", session_id=SESSION_A, collection=COLLECTION_A
    )
    context_b = _context(
        principal_id="B", profile_id="profile-b", session_id=SESSION_B, collection=COLLECTION_B
    )
    engine_a = _engine(db_path, context_a, teams_enabled=True)
    engine_b = _engine(db_path, context_b, teams_enabled=True)
    scheduler = _RecordingScheduler()
    monkeypatch.setattr(engine_module, "_ROLLUP_MAINTENANCE_SCHEDULER", scheduler)
    try:
        seed = _seed_a(engine_a)
        positive_failures = _run_positive_control(engine_a, seed, scheduler)
        leaks = _b_leaks(engine_b, seed, scheduler)

        forbidden = Decision.deny(
            DenialReason.SCOPE_FORBIDDEN, target_id=seed["store_id"]
        ).public()
        missing = Decision.deny(
            DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN, target_id="missing"
        ).public()
        assert isinstance(forbidden, PublicDecision)
        leg8 = forbidden == missing

        if positive_failures or leaks or not leg8:
            details = [
                "POSITIVE CONTROL: "
                + ("PASS" if not positive_failures else "FAILED: " + "; ".join(positive_failures)),
                "LEG 8 PublicDecision equality: " + ("PASS" if leg8 else "FAILED"),
                *leaks,
            ]
            pytest.fail("\n".join(details))
    finally:
        engine_b.shutdown()
        engine_a.shutdown()


def test_two_principal_positive_control_stays_green(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A's own Teams-on path remains a hard, non-xfailed control."""
    context_a = _context(
        principal_id="A", profile_id="profile-a", session_id=SESSION_A, collection=COLLECTION_A
    )
    engine_a = _engine(tmp_path / "positive.db", context_a, teams_enabled=True)
    scheduler = _RecordingScheduler()
    monkeypatch.setattr(engine_module, "_ROLLUP_MAINTENANCE_SCHEDULER", scheduler)
    try:
        seed = _seed_a(engine_a)
        failures = _run_positive_control(engine_a, seed, scheduler)
        assert not failures, "POSITIVE CONTROL FAILED: " + "; ".join(failures)
    finally:
        engine_a.shutdown()


def test_default_off_single_principal_reaches_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teams-off keeps the existing trusted-owner path intact."""
    context_a = _context(
        principal_id="A", profile_id="profile-a", session_id=SESSION_A, collection=COLLECTION_A
    )
    engine = _engine(tmp_path / "teams-off.db", context_a, teams_enabled=False)
    scheduler = _RecordingScheduler()
    monkeypatch.setattr(engine_module, "_ROLLUP_MAINTENANCE_SCHEDULER", scheduler)
    try:
        seed = _seed_a(engine)
        failures = _run_positive_control(engine, seed, scheduler)
        assert not failures, "DEFAULT-OFF POSITIVE CONTROL FAILED: " + "; ".join(failures)
    finally:
        engine.shutdown()


def test_store_wide_backup_is_admin_only_under_teams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whole-store backup is not any principal's to take.

    It copies every principal's memory into one file, so under Teams it is an
    administrative capability -- #497 gives the connector the `backup.*` family
    and authenticates it separately. The positive control used to assert
    principal A could do it, which encoded pre-Teams semantics: A was not
    "the owner", A was merely the one who asked.

    Asserting BOTH principals are refused is a stronger claim than the leg it
    replaces, which only checked that B was not. And the Teams-off leg is what
    keeps it honest -- without it this would pass just as well if backup were
    broken outright rather than scoped.
    """
    db_path = tmp_path / "backup-admin.db"
    context_a = _context(
        principal_id="A", profile_id="profile-a", session_id=SESSION_A, collection=COLLECTION_A
    )
    context_b = _context(
        principal_id="B", profile_id="profile-b", session_id=SESSION_B, collection=COLLECTION_B
    )
    engine_a = _engine(db_path, context_a, teams_enabled=True)
    engine_b = _engine(db_path, context_b, teams_enabled=True)
    try:
        for name, engine in (("A", engine_a), ("B", engine_b)):
            with pytest.raises(Exception) as excinfo:
                maintenance_module.backup_database(engine)
            assert "authorize" in str(excinfo.value), (
                f"principal {name} was refused, but not by the authorization seam"
            )
    finally:
        engine_b.shutdown()
        engine_a.shutdown()

    # Teams OFF: unchanged. Backup is restricted BY Teams, not broken by it.
    off_path = tmp_path / "backup-teams-off.db"
    engine_off = _engine(off_path, context_a, teams_enabled=False)
    try:
        assert maintenance_module.backup_database(engine_off).get("ok")
    finally:
        engine_off.shutdown()
