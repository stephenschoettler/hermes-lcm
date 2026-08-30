"""Tests that ask the REAL policy, not a stub that was told to deny.

Every defect the status audit found shared one cause: the test that certified
the behaviour never exercised `TeamsPolicy`. `test_teams_apply_mode_gates`
injected a `_DenyingPolicy`, so it proved the gate PROPAGATES a denial and said
nothing about whether the real policy PRODUCES one -- and it did not. The gate
ran, allowed every principal, and its test stayed green.

These tests take no policy argument. If TeamsPolicy stops denying, they fail.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hermes_lcm.access_context.model import AccessContextV1
from hermes_lcm.access_policy import TeamsPolicy


def _context(principal: str = "carus") -> AccessContextV1:
    now = datetime.now(timezone.utc)
    return AccessContextV1.from_host(
        authenticated_transport="test", context_id="ctx", request_id="req",
        source_kind="human", deployment_id="dep", tenant_id="tenant",
        principal_id=principal, profile_id=principal, profile_incarnation="inc",
        session_id="session-own", session_owner_principal_id=principal,
        conversation_id="conv", conversation_lane="lane",
        read_policy_ref="policy", lease_id="lease",
        issued_at=now - timedelta(minutes=1), expires_at=now + timedelta(hours=1),
    )


@pytest.mark.parametrize(
    "kind",
    ["backup", "assertions_rebuild", "embedding_backfill", "chunk_backfill"],
)
def test_the_real_policy_denies_every_store_wide_operation(kind: str) -> None:
    """These rebuild or copy state spanning EVERY principal.

    The exact scope `command.py:_authorize_apply_mutation` builds -- no session,
    no partition, no target. It carries nothing the owner-of-target loop can
    compare, so unless the kind is recognised as store-wide the policy falls
    through to allow(). It did, for three of these four.
    """
    policy = TeamsPolicy(_context())

    decision = policy.authorize_operation(
        None, "owner_only",
        {"kind": kind, "entry_point": "x", "required_scope": "owner_only"},
    )

    assert not decision.allowed, f"{kind} was permitted to a principal"


def test_a_scope_with_no_recognisable_target_is_the_dangerous_shape() -> None:
    """Documents the fall-through that made the gates inert.

    An unrecognised `kind` with no target keys still reaches allow(). That is
    deliberate -- most operations are not store-wide -- but it means adding a
    new store-wide operation REQUIRES adding it to _STORE_WIDE_KINDS, and
    forgetting is silent. This test exists so that trade-off is visible rather
    than discovered.
    """
    policy = TeamsPolicy(_context())
    decision = policy.authorize_operation(
        None, "write", {"kind": "some_future_operation", "entry_point": "y"}
    )
    assert decision.allowed, (
        "if this ever fails the fall-through changed -- re-read whether every "
        "store-wide operation is now enumerated"
    )


def test_every_store_wide_kind_the_command_layer_uses_is_enumerated() -> None:
    """Bind the two lists together so they cannot drift apart.

    The gates were inert precisely because command.py used kinds the policy had
    never heard of.
    """
    import inspect
    import re

    from hermes_lcm import command as command_module
    from hermes_lcm.access_policy.teams_policy import _STORE_WIDE_KINDS

    used = set()
    for name in (
        "_assertions_rebuild_text",
        "_embedding_backfill_summary_text",
        "_chunk_backfill_text",
    ):
        source = inspect.getsource(getattr(command_module, name))
        used.update(re.findall(r'kind="([a-z_]+)"', source))

    missing = used - set(_STORE_WIDE_KINDS)
    assert not missing, (
        f"command.py gates on kinds the policy does not treat as store-wide: "
        f"{sorted(missing)} -- those gates are inert"
    )
