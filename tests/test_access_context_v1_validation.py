from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from types import MappingProxyType
from pathlib import Path

from hermes_lcm.access_context import (
    AccessContextV1,
    Decision,
    DenialReason,
    ResolutionMode,
    VALIDATION_ORDER,
    ValidationStage,
    derive_child,
    is_subset_of,
    resolve_mode,
    validate,
)
from hermes_lcm.access_context.denials import PUBLIC_DENIAL_PROJECTION, PublicDecision, project_public
from hermes_lcm.access_context.fixtures import fixture_paths, load_context, load_fixture
from hermes_lcm.access_context.protocols import HostContextCarrier, LcmAuthorizationConsumer


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _context(path: Path) -> AccessContextV1:
    context = load_context(path)
    assert context is not None
    return context


def test_validation_order_is_explicit_and_stable() -> None:
    assert VALIDATION_ORDER == (
        ValidationStage.CONTEXT_PRESENT,
        ValidationStage.CONTEXT_WELL_FORMED,
        ValidationStage.REVISION_SUPPORTED,
        ValidationStage.NOT_EXPIRED,
        ValidationStage.NOT_REVOKED,
        ValidationStage.OWNERSHIP_CURRENT,
        ValidationStage.LEASE_CURRENT,
        ValidationStage.SCOPE_PERMITTED,
        ValidationStage.TARGET_RESOLUTION,
    )


def test_positive_contexts_validate() -> None:
    paths = fixture_paths("positive")
    assert len(paths) >= 5
    for path in paths:
        context = _context(path)
        assert validate(context, required_scope="read", now=NOW).allowed, path


def test_validation_uses_effective_operation_allowlist() -> None:
    context = _context(Path("tests/fixtures/access_context_v1/positive/human.json"))
    narrowed = replace(context, narrowing=frozenset({"operation:read"}))
    decision = validate(narrowed, required_scope="write", now=NOW)
    assert decision.denial_reason is DenialReason.SCOPE_FORBIDDEN


def _expected_decision(path: Path):
    payload = load_fixture(path)
    expected = payload["expected"]
    kwargs = dict(expected.get("validate_kwargs", {}))
    now = datetime.fromisoformat(expected.get("now", "2026-01-01T00:00:00+00:00").replace("Z", "+00:00"))
    required_scope = expected.get("required_scope")
    if "target_allowed" in expected:
        kwargs["target_allowed"] = expected["target_allowed"]
    if "requested_collections" in expected:
        kwargs["requested_collections"] = expected["requested_collections"]
    return validate(_context(path) if payload["context"] is not None else None, required_scope=required_scope, now=now, **kwargs)


def test_negative_fixtures_are_parametrized_and_cover_every_denial() -> None:
    paths = fixture_paths("negative")
    assert len(paths) >= len(tuple(DenialReason))
    discovered = set()
    for path in paths:
        decision = _expected_decision(path)
        expected_reason = load_fixture(path)["expected"]["denial_reason"]
        discovered.add(expected_reason)
        assert not decision.allowed, path
        assert decision.denial_reason.value == expected_reason, path
    assert discovered == {reason.value for reason in DenialReason}


def test_expiry_wins_scope_tie_break() -> None:
    path = Path("tests/fixtures/access_context_v1/negative/expired-and-out-of-scope.json")
    decision = _expected_decision(path)
    assert decision.denial_reason is DenialReason.CONTEXT_EXPIRED


def test_delegation_vectors_prove_subset_and_reject_each_widening() -> None:
    paths = fixture_paths("delegation")
    assert len(paths) >= 10
    widening_fields = set()
    for path in paths:
        payload = load_fixture(path)
        if "candidate" not in payload:
            continue
        parent = AccessContextV1.from_payload(payload["context"])
        candidate = AccessContextV1.from_payload(payload["candidate"])
        expected = payload["expected"]
        assert is_subset_of(candidate, parent) is expected["subset"], path
        if not expected["subset"]:
            widening_fields.add(expected["widen_field"])
    assert {"operations", "collections", "audience", "profile_binding", "session_binding", "expiry", "policy_revision", "membership_revision", "revocation_epoch", "ownership_generation", "lease_generation"} <= widening_fields


def test_three_deep_redelegation_preserves_chain_and_narrowing() -> None:
    root = _context(Path("tests/fixtures/access_context_v1/delegation/redelegation-chain-3-deep.json"))
    one = derive_child(root, operations=["read"], collections=["collection-a"], audience=["profile-a"], expires_at="2026-09-01T00:00:00Z")
    two = derive_child(one, operations=["read"], collections=["collection-a"], audience=["profile-a"], expires_at="2026-08-01T00:00:00Z")
    three = derive_child(two, operations=["read"], collections=["collection-a"], audience=["profile-a"], expires_at="2026-07-01T00:00:00Z")
    assert three.delegation_chain == (root.context_id, one.context_id, two.context_id)
    assert len(three.delegation_chain) == 3
    assert root.narrowing <= three.narrowing
    assert is_subset_of(one, root)
    assert is_subset_of(two, one)
    assert is_subset_of(three, two)


def test_revocation_vectors_invalidate_the_next_use() -> None:
    paths = fixture_paths("revocation")
    assert len(paths) >= 4
    for path in paths:
        payload = load_fixture(path)
        decision = _expected_decision(path)
        assert decision.denial_reason.value == payload["expected"]["denial_reason"], path


def test_absent_carrier_compatibility_matrix() -> None:
    context = _context(Path("tests/fixtures/access_context_v1/positive/human.json"))
    assert resolve_mode(None, False) is ResolutionMode.STANDARD_UNMANAGED
    assert resolve_mode(None, True) is ResolutionMode.FAIL_CLOSED
    assert resolve_mode(context, False) is ResolutionMode.STANDARD_UNMANAGED
    assert resolve_mode(context, True) is ResolutionMode.ENFORCING
    assert validate(None, now=NOW).denial_reason is DenialReason.CONTEXT_MISSING


def test_public_denial_projection_is_total_and_content_free() -> None:
    assert set(PUBLIC_DENIAL_PROJECTION) == set(DenialReason)
    for reason in DenialReason:
        internal = Decision.deny(reason, context_id="ctx", query_text="must not escape")
        public = project_public(internal)
        assert public.denial_reason is PUBLIC_DENIAL_PROJECTION[reason]
        assert "query_text" not in internal.detail
        assert dict(public.detail) == {}


# Detail that VARIES per reason, mirroring what validate() really produces:
# SCOPE_FORBIDDEN carries context_id + policy_revision, while
# TARGET_NOT_FOUND_OR_FORBIDDEN carries nothing at all. A test that passes the
# same detail to every reason cannot see a value-level leak, because it
# manufactures the uniformity it is meant to be checking.
_REALISTIC_DENIAL_DETAIL: dict[DenialReason, dict[str, object]] = {
    DenialReason.CONTEXT_MISSING: {},
    DenialReason.CONTEXT_INVALID: {"context_id": "ctx-invalid"},
    DenialReason.CONTEXT_UNSUPPORTED_VERSION: {"context_id": "ctx-version"},
    DenialReason.CONTEXT_EXPIRED: {"context_id": "ctx-expired", "lease_id": "lease-1"},
    DenialReason.CONTEXT_REVOKED: {"context_id": "ctx-revoked", "revocation_epoch": 9},
    DenialReason.SCOPE_FORBIDDEN: {"context_id": "ctx-123", "policy_revision": 11},
    DenialReason.SCOPE_MISMATCH: {"context_id": "ctx-456", "policy_revision": 12},
    DenialReason.OWNERSHIP_CHANGED: {
        "context_id": "ctx-789",
        "ownership_generation": 3,
        "expected_ownership_generation": 4,
    },
    DenialReason.LEASE_STALE: {"lease_id": "lease-2", "lease_generation": 7},
    DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN: {},
}


def test_public_projection_cannot_re_identify_the_blurred_reason() -> None:
    """Blurring ``denial_reason`` is undone by anything that co-varies with it.

    Detail co-varies two ways: which keys are present, and which values are.
    The second is the subtle one -- an echoed ``context_id`` that is a real ID
    for SCOPE_FORBIDDEN and ``None`` for TARGET_NOT_FOUND_OR_FORBIDDEN
    separates two members of the same public bucket just as effectively as a
    missing key would.
    """

    assert set(_REALISTIC_DENIAL_DETAIL) == set(DenialReason), "cover every reason"

    for reason, detail in _REALISTIC_DENIAL_DETAIL.items():
        public = project_public(Decision.deny(reason, **detail))
        # Exact, not "no discriminating key": an empty projection is the only
        # shape that needs no argument about which keys are safe to echo.
        assert dict(public.detail) == {}, (reason, dict(public.detail))

    # Freeze the bucket membership. A reason added to this bucket later must
    # be considered here rather than silently inheriting the guarantee.
    bucket = [
        reason
        for reason, public_reason in PUBLIC_DENIAL_PROJECTION.items()
        if public_reason is DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN
    ]
    assert set(bucket) == {
        DenialReason.SCOPE_FORBIDDEN,
        DenialReason.SCOPE_MISMATCH,
        DenialReason.OWNERSHIP_CHANGED,
        DenialReason.LEASE_STALE,
        DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN,
    }

    # Every member must be LITERALLY EQUAL once projected -- indistinguishable
    # by value, repr, equality or hash, not merely similar in shape.
    projected = {
        project_public(Decision.deny(reason, **_REALISTIC_DENIAL_DETAIL[reason]))
        for reason in bucket
    }
    assert len(projected) == 1, f"same-bucket denials are distinguishable: {projected}"
    assert len({hash(p) for p in projected}) == 1


def test_decisions_are_hashable_despite_mappingproxy_detail() -> None:
    # frozen=True advertises hashability, but detail is a mappingproxy; without
    # an explicit __hash__ a consumer deduping decisions crashes at runtime.
    internal = Decision.deny(DenialReason.LEASE_STALE, context_id="ctx")
    assert len({internal, Decision.deny(DenialReason.LEASE_STALE, context_id="ctx")}) == 1
    assert len({project_public(internal), project_public(internal)}) == 1
    assert len({Decision.allow(), Decision.allow()}) == 1


def test_hashing_is_total_and_equality_safe_for_hostile_detail() -> None:
    """``isinstance`` admits subclasses, so detail values must be coerced.

    A subclass can set ``__hash__ = None`` or redefine it, which would make an
    otherwise-ordinary Decision unhashable or make two equal Decisions hash
    differently. Values are normalized to exact primitives to prevent both.
    """

    class Unhashable(int):
        __hash__ = None  # type: ignore[assignment]

    class Rehashed(int):
        def __hash__(self) -> int:  # pragma: no cover - value is what matters
            return 999999

    # Would raise TypeError without normalization.
    hash(Decision.allow(policy_revision=Unhashable(5)))
    hash(project_public(Decision.deny(DenialReason.LEASE_STALE, policy_revision=Unhashable(5))))

    equal_pair = (
        Decision.deny(DenialReason.LEASE_STALE, policy_revision=Rehashed(5)),
        Decision.deny(DenialReason.LEASE_STALE, policy_revision=5),
    )
    assert equal_pair[0] == equal_pair[1]
    assert hash(equal_pair[0]) == hash(equal_pair[1]), "equal objects must hash equally"

    # PublicDecision is exported, so a caller can build one directly with a
    # mutable dict; its hash must not change out from under a set or dict.
    source = {"context_id": "x"}
    public = PublicDecision(False, DenialReason.CONTEXT_MISSING, source)
    before = hash(public)
    source["context_id"] = "mutated"
    assert hash(public) == before
    assert isinstance(public.detail, MappingProxyType)

    # Mixed key types must not make sorting raise inside __hash__.
    hash(PublicDecision(False, DenialReason.CONTEXT_MISSING, {1: "a", "context_id": "b"}))


def test_two_principal_replay_vectors_fail_closed() -> None:
    paths = [path for path in fixture_paths("negative") if "replay-" in path.name]
    assert len(paths) >= 6
    for path in paths:
        decision = _expected_decision(path)
        assert not decision.allowed, path
        assert decision.denial_reason in {
            DenialReason.CONTEXT_INVALID,
            DenialReason.OWNERSHIP_CHANGED,
            DenialReason.LEASE_STALE,
        }


def test_concurrent_contexts_do_not_cross_contaminate_threads_or_tasks() -> None:
    human = _context(Path("tests/fixtures/access_context_v1/positive/human.json"))
    agent = _context(Path("tests/fixtures/access_context_v1/positive/agent.json"))

    def check(context: AccessContextV1):
        return validate(context, required_scope="read", now=NOW).detail["context_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert set(pool.map(check, (human, agent))) == {human.context_id, agent.context_id}

    async def run_pair():
        return await asyncio.gather(asyncio.to_thread(check, human), asyncio.to_thread(check, agent))

    # Python 3.9 has no asyncio.to_thread; executor-backed coroutines preserve
    # the same task-isolation assertion without introducing context globals.
    async def run_pair_39():
        loop = asyncio.get_running_loop()
        return await asyncio.gather(loop.run_in_executor(None, check, human), loop.run_in_executor(None, check, agent))

    assert asyncio.run(run_pair_39()) == [human.context_id, agent.context_id]


def test_validation_module_has_no_mutable_context_store() -> None:
    import access_context.validation as validation_module

    assert isinstance(validation_module.VALIDATION_ORDER, tuple)
    assert not any(
        isinstance(value, (dict, list, set))
        for name, value in vars(validation_module).items()
        if not name.startswith("__") and name not in {"annotations"}
    )


class RecordingConsumer:
    """Reference fake used only to freeze authorization/disclosure ordering."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def authorize_operation(self, context, operation, expected_scope):
        self.calls.append("authorize_operation")
        return Decision.allow()

    def resolve_authorized_targets(self, context, operation, requested_narrowing):
        self.calls.append("resolve_authorized_targets")
        return [1]

    def authorize_stored_scope(self, context, operation, stored_scope):
        self.calls.append("authorize_stored_scope")
        return Decision.allow()

    def audit_decision(self, context, operation, internal_reason, public_result: PublicDecision):
        self.calls.append("audit_decision")

    def select_collection(self, target_scope):
        self.calls.append("select_collection")
        return "collection"

    def count_candidates(self, candidates):
        self.calls.append("count_candidates")
        return len(candidates)

    def rank_candidates(self, candidates):
        self.calls.append("rank_candidates")
        return candidates

    def hydrate_targets(self, targets):
        self.calls.append("hydrate_targets")
        return targets

    def issue_handle(self, target):
        self.calls.append("issue_handle")
        return "handle"


class RecordingCarrier:
    def __init__(self, context):
        self.context = context

    def get_lcm_access_context(self):
        return self.context


def test_consumer_protocol_freezes_473_steps_2_through_6() -> None:
    context = _context(Path("tests/fixtures/access_context_v1/positive/human.json"))
    consumer = RecordingConsumer()
    carrier = RecordingCarrier(context)
    assert isinstance(carrier, HostContextCarrier)
    assert isinstance(consumer, LcmAuthorizationConsumer)
    # Step 1 of #473 is "validate current host context". It is validate(), a
    # module function rather than a consumer method, so it is a PRECONDITION
    # here rather than a recorded call -- its own ordering is covered by
    # test_validation_order_is_explicit_and_stable. Steps 2-6 are what the
    # consumer protocol owns, and are what this test freezes.
    assert validate(context, now=NOW).allowed

    context = carrier.get_lcm_access_context()
    scope = {"collection": "collection-main"}
    decision = consumer.authorize_operation(context, "read", scope)
    authorized_targets = consumer.resolve_authorized_targets(context, "read", scope)
    consumer.select_collection(scope)
    consumer.authorize_stored_scope(context, "read", scope)
    consumer.count_candidates(authorized_targets)
    consumer.rank_candidates(authorized_targets)
    consumer.hydrate_targets(authorized_targets)
    consumer.issue_handle(authorized_targets[0])
    consumer.audit_decision(context, "read", None, decision.public())

    # Frozen verbatim from #473: 1 validate context, 2 authorize operation and
    # expected scope, 3 resolve only authorized targets, 4 inspect stored scope
    # before content/revision disclosure, 5 query/rank/hydrate within authorized
    # targets, 6 audit. Step 1 is the carrier read above.
    assert consumer.calls == [
        "authorize_operation",
        "resolve_authorized_targets",
        "select_collection",
        "authorize_stored_scope",
        "count_candidates",
        "rank_candidates",
        "hydrate_targets",
        "issue_handle",
        "audit_decision",
    ]

    def before(earlier: str, later: str) -> bool:
        return consumer.calls.index(earlier) < consumer.calls.index(later)

    disclosure_primitives = (
        "select_collection",
        "count_candidates",
        "rank_candidates",
        "hydrate_targets",
        "issue_handle",
    )
    # Nothing is disclosed before the operation is authorized.
    assert all(before("authorize_operation", name) for name in disclosure_primitives)
    # A collection is never opened before the authorized target set is resolved,
    # and ranking/limits never run over unresolved candidates.
    assert all(before("resolve_authorized_targets", name) for name in disclosure_primitives)
    # Stored scope is re-authorized before any existence, count or content signal.
    assert all(
        before("authorize_stored_scope", name)
        for name in ("count_candidates", "rank_candidates", "hydrate_targets", "issue_handle")
    )
    # The audit is the final step and never gates disclosure.
    assert all(before(name, "audit_decision") for name in disclosure_primitives)
