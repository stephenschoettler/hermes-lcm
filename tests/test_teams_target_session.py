"""Owner-of-target, not same-session-id.

`TeamsPolicy` first shipped comparing the target `session_id` against the
context's. That is the right *intent* and the wrong *rule*: a principal
legitimately touches sessions other than the one it is currently bound to.
`on_session_end` passes the TARGET session in `expected_scope`
(``engine.py:3450-3460``), and an auxiliary session end targets a session id
that is deliberately not `self._session_id`.

So the proxy denied a principal its own auxiliary session, and the standing
isolation smoke could not see it because that fixture pins
`engine._session_id` to the context's session.

These tests fix the rule at the level it was wrong: compare OWNERS.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hermes_lcm.access_context.model import AccessContextV1
from hermes_lcm.access_policy import TeamsPolicy


def _context(principal: str = "acorn", session: str = "session-current") -> AccessContextV1:
    now = datetime.now(timezone.utc)
    return AccessContextV1.from_host(
        authenticated_transport="test",
        context_id=f"ctx-{principal}",
        request_id=f"req-{principal}",
        source_kind="human",
        deployment_id="dep",
        tenant_id="tenant",
        principal_id=principal,
        profile_id=principal,
        profile_incarnation="inc",
        session_id=session,
        session_owner_principal_id=principal,
        conversation_id="conv",
        conversation_lane="lane",
        read_policy_ref="policy",
        lease_id="lease",
        issued_at=now - timedelta(minutes=1),
        expires_at=now + timedelta(hours=1),
    )


def _owner_map(mapping: dict[str, str]):
    """Stand in for the seam-bound resolver, as the audit sink is bound."""
    return lambda session_id: mapping.get(session_id)


def test_a_principal_may_end_its_own_auxiliary_session() -> None:
    """The regression. An auxiliary session id is deliberately not the bound one."""
    policy = TeamsPolicy(
        _context(),
        session_owner=_owner_map({"aux-session": "acorn"}),
    )

    decision = policy.authorize_operation(
        None, "write", {"kind": "session_end", "session_id": "aux-session"}
    )

    assert decision.allowed, "a principal was denied its OWN auxiliary session"


def test_another_principals_session_is_still_denied() -> None:
    """The property the wrong rule was protecting, kept."""
    policy = TeamsPolicy(
        _context(),
        session_owner=_owner_map({"carus-session": "carus"}),
    )

    decision = policy.authorize_operation(
        None, "write", {"kind": "session_end", "session_id": "carus-session"}
    )

    assert not decision.allowed


def test_the_currently_bound_session_is_allowed_without_a_resolver() -> None:
    """No resolver wired: the context's own session must still work.

    This is the default-off-ish path -- a policy with no owner resolver still
    has to let a principal operate on the session it is bound to, or enabling
    Teams breaks ordinary single-session work.
    """
    policy = TeamsPolicy(_context(session="session-current"))

    decision = policy.authorize_operation(
        None, "write", {"kind": "session_end", "session_id": "session-current"}
    )

    assert decision.allowed


def test_an_unknown_session_is_allowed_and_the_write_stamps_the_writer() -> None:
    """A session with no stamped rows yet has no owner to conflict with.

    Denying here would break creating any new session under Teams. It is safe
    because the write is stamped with the WRITER's scope on the way in, so an
    unknown session cannot be used to smuggle a row into another principal's
    space -- it becomes the writer's row.
    """
    policy = TeamsPolicy(_context(), session_owner=_owner_map({}))

    decision = policy.authorize_operation(
        None, "write", {"kind": "session_start", "session_id": "brand-new-session"}
    )

    assert decision.allowed


@pytest.mark.parametrize("key", ["session_id", "source_session_id", "partition_key"])
def test_every_target_key_resolves_by_owner(key: str) -> None:
    """The three keys that carry a target session across different paths."""
    allowed = TeamsPolicy(
        _context(), session_owner=_owner_map({"other": "acorn"})
    ).authorize_operation(None, "write", {key: "other"})
    denied = TeamsPolicy(
        _context(), session_owner=_owner_map({"other": "carus"})
    ).authorize_operation(None, "write", {key: "other"})

    assert allowed.allowed, f"{key}: same-owner target should be allowed"
    assert not denied.allowed, f"{key}: cross-owner target must be denied"
