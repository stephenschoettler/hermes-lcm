from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from hermes_lcm.access_context import AccessContextV1, Decision, DenialReason, LcmAuthorizationConsumer
from hermes_lcm.access_context.denials import PublicDecision
from hermes_lcm.access_context.fixtures import load_context
from hermes_lcm.access_policy import (
    AuthorizationRequiredError,
    FailClosedPolicy,
    TeamsPolicy,
    TrustedOwnerPolicy,
    resolve_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _context(path: str) -> AccessContextV1:
    context = load_context(REPO_ROOT / path)
    assert context is not None
    return context


def test_both_policies_satisfy_consumer_protocol() -> None:
    assert isinstance(TrustedOwnerPolicy(), LcmAuthorizationConsumer)
    assert isinstance(FailClosedPolicy(), LcmAuthorizationConsumer)


def test_resolution_carrier_absent_teams_off_is_trusted_owner() -> None:
    assert isinstance(resolve_policy(None, False), TrustedOwnerPolicy)


def test_resolution_carrier_absent_teams_on_is_fail_closed() -> None:
    policy = resolve_policy(None, True)
    assert isinstance(policy, FailClosedPolicy)
    assert policy.denial_reason is DenialReason.CONTEXT_MISSING


def test_resolution_carrier_present_teams_off_stays_trusted_owner() -> None:
    context = _context("tests/fixtures/access_context_v1/positive/human.json")
    assert isinstance(resolve_policy(context, False), TrustedOwnerPolicy)


def test_resolution_carrier_present_teams_on_resolves_the_enforcing_policy() -> None:
    """Renamed: this used to assert the permissive placeholder.

    A valid Teams context resolving to TrustedOwnerPolicy WAS the #483 gap --
    the test named the placeholder as though it were the contract. It now
    resolves TeamsPolicy, which scopes to the acting principal.
    """
    context = _context("tests/fixtures/access_context_v1/positive/human.json")
    assert isinstance(resolve_policy(context, True, NOW), TeamsPolicy)


def test_resolution_malformed_carrier_fails_closed_without_attribute_error() -> None:
    policy = resolve_policy({"context_id": "decoded-but-not-a-context"}, True, NOW)
    assert isinstance(policy, FailClosedPolicy)
    assert policy.denial_reason is DenialReason.CONTEXT_INVALID


def test_trusted_owner_returns_requested_narrowing_unchanged() -> None:
    narrowing = {"collections": ("collection-a",), "limit": 3}
    result = TrustedOwnerPolicy().resolve_authorized_targets(None, "read", narrowing)
    assert result == narrowing
    assert result is narrowing


def test_fail_closed_denies_every_protocol_operation_with_typed_reason() -> None:
    policy = FailClosedPolicy(DenialReason.CONTEXT_MISSING)
    # Only the two AUTHORIZATION methods return a typed Decision. The
    # disclosure primitives raise instead -- see
    # test_fail_closed_disclosure_primitives_raise_instead_of_returning_a_decision.
    denied = [
        policy.authorize_operation(None, "read", {}),
        policy.authorize_stored_scope(None, "read", {}),
    ]
    targets = policy.resolve_authorized_targets(None, "read", {"limit": 1})

    for decision in denied:
        assert isinstance(decision, Decision)
        assert decision.allowed is False
        assert decision.denial_reason is DenialReason.CONTEXT_MISSING
    # An empty scope MAPPING -- resolve_authorized_targets is declared to
    # return a TargetScope, because every production caller reads it with .get.
    assert targets == {}

    public = denied[0].public()
    assert isinstance(public, PublicDecision)
    policy.audit_decision(None, "read", denied[0].denial_reason, public)
    assert policy.audit_records == [(DenialReason.CONTEXT_MISSING, public)]


def test_invalid_teams_context_resolves_fail_closed_without_ambient_fallback() -> None:
    context = _context("tests/fixtures/access_context_v1/negative/context-invalid.json")
    policy = resolve_policy(context, True, NOW)
    assert isinstance(policy, FailClosedPolicy)
    assert policy.authorize_operation(context, "read", {}).denial_reason is DenialReason.CONTEXT_INVALID


# REMOVED: test_access_policy_imports_are_inert_outside_package_and_its_test.
#
# Written in F1 when access_policy was genuinely inert. F2-F6 then wired it into
# engine/retrieval/compaction/command by design, so its premise is void.
#
# Worth recording WHY it did not fail when that happened: it scanned for `import`
# and `from ... import` AST nodes, while the hook modules were pulling the package
# in via importlib.import_module("access_policy") -- a Call node. The assertion was
# therefore vacuous from F2 onward, passing not because nothing imported the
# package but because it could not see the imports that existed.
#
# The invariant that actually matters now lives in
# tests/test_lcm_authorization_completeness.py: every hook module must import the
# seam RELATIVELY, and absolute imports are rejected.


DISCLOSURE_PRIMITIVES = (
    ("select_collection", ({},)),
    ("count_candidates", ([1, 2],)),
    ("rank_candidates", ([1, 2],)),
    ("hydrate_targets", ([1],)),
    ("issue_handle", (1,)),
)


def test_fail_closed_disclosure_primitives_raise_instead_of_returning_a_decision() -> None:
    """A denying policy must not hand back a truthy object.

    ``isinstance(policy, LcmAuthorizationConsumer)`` cannot catch this: a
    runtime_checkable Protocol checks that the methods EXIST, not that they
    return the declared types. Returning a Decision from methods declared
    ``-> int`` / ``-> Sequence`` type-lies, and because a Decision is truthy,
    ``if policy.select_collection(scope):`` would proceed as though a real
    collection came back -- fail-closed failing open at the call site.
    """

    policy = FailClosedPolicy()
    for name, args in DISCLOSURE_PRIMITIVES:
        with pytest.raises(AuthorizationRequiredError) as excinfo:
            getattr(policy, name)(*args)
        assert excinfo.value.primitive == name
        assert excinfo.value.denial_reason is DenialReason.CONTEXT_MISSING


def test_no_disclosure_primitive_ever_returns_a_truthy_decision() -> None:
    # The specific shape of the bug this guards: a Decision leaking out of a
    # primitive whose declared return type is int/Sequence/collection.
    policy = FailClosedPolicy()
    for name, args in DISCLOSURE_PRIMITIVES:
        try:
            result = getattr(policy, name)(*args)
        except AuthorizationRequiredError:
            continue
        assert not isinstance(result, Decision), f"{name} returned a Decision"


def test_trusted_owner_primitives_keep_their_declared_return_types() -> None:
    # The permissive policy is the one that must stay type-honest, since it is
    # the default path and its results flow straight into real call sites.
    policy = TrustedOwnerPolicy()
    assert policy.count_candidates([1, 2]) == 2
    assert list(policy.rank_candidates([1, 2])) == [1, 2]
    assert list(policy.hydrate_targets([1])) == [1]
    assert policy.issue_handle(7) == 7
