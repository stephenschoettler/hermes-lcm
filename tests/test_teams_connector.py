"""The connector seam refuses everything until a host wires it (#497).

The failure mode this file exists to prevent is not a connector that rejects a
valid caller -- that is loud, and someone fixes it in minutes. It is a connector
that accepts EVERYONE because nobody configured it, which is silent and survives
to production.

So the first test is the important one: with no credential check wired, every
capability is refused.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_lcm.teams import catalog
from hermes_lcm.teams.connector import (
    Capability,
    ConnectorError,
    ConnectorRequest,
    FailureClass,
    TeamsConnector,
)


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    catalog.ensure_teams_catalog(connection)
    try:
        yield connection
    finally:
        connection.close()


def _request(capability: Capability = Capability.TEAMS_STATUS, **kw) -> ConnectorRequest:
    base = dict(
        request_id="req-1",
        capability=capability,
        acting_principal_id="operator",
        tenant_id="tenant-a",
        payload={"k": "v"},
        credential="secret",
    )
    base.update(kw)
    return ConnectorRequest(**base)


def _ok_handler(_conn, request):
    return {"echo": dict(request.payload)}


# --- the property the phase exists for ------------------------------------

@pytest.mark.parametrize("capability", list(Capability))
def test_an_unwired_connector_refuses_every_capability(conn, capability) -> None:
    connector = TeamsConnector(conn)  # no credential_check -- the default
    assert not connector.is_wired
    with pytest.raises(ConnectorError) as excinfo:
        connector.execute(_request(capability))
    assert excinfo.value.failure is FailureClass.UNAUTHENTICATED


def test_an_unwired_connector_refuses_even_with_handlers_registered(conn) -> None:
    """Registering handlers must not accidentally imply authentication."""
    connector = TeamsConnector(
        conn, handlers={c: _ok_handler for c in Capability}
    )
    with pytest.raises(ConnectorError) as excinfo:
        connector.execute(_request())
    assert excinfo.value.failure is FailureClass.UNAUTHENTICATED


def test_a_credential_check_that_raises_is_a_refusal(conn) -> None:
    """A broken host config must fail closed, never open."""

    def explode(_credential):
        raise RuntimeError("host misconfigured")

    connector = TeamsConnector(conn, credential_check=explode,
                               handlers={Capability.TEAMS_STATUS: _ok_handler})
    with pytest.raises(ConnectorError) as excinfo:
        connector.execute(_request())
    assert excinfo.value.failure is FailureClass.UNAUTHENTICATED


def test_a_rejected_credential_is_unauthenticated(conn) -> None:
    connector = TeamsConnector(conn, credential_check=lambda c: c == "right",
                               handlers={Capability.TEAMS_STATUS: _ok_handler})
    with pytest.raises(ConnectorError) as excinfo:
        connector.execute(_request(credential="wrong"))
    assert excinfo.value.failure is FailureClass.UNAUTHENTICATED


def test_a_wired_connector_serves_an_authenticated_caller(conn) -> None:
    """POSITIVE CONTROL. Refusing everything also passes every test above."""
    connector = TeamsConnector(conn, credential_check=lambda c: c == "secret",
                               handlers={Capability.TEAMS_STATUS: _ok_handler})
    result = connector.execute(_request())
    assert result.status == "ok"
    assert result.data == {"echo": {"k": "v"}}
    assert result.replayed is False


# --- idempotency ----------------------------------------------------------

def test_a_replay_produces_one_effect_and_replays_the_answer(conn) -> None:
    calls: list[int] = []

    def counting(_conn, request):
        calls.append(1)
        return {"n": len(calls)}

    connector = TeamsConnector(conn, credential_check=lambda c: True,
                               handlers={Capability.TEAMS_STATUS: counting})
    first = connector.execute(_request())
    second = connector.execute(_request())

    assert len(calls) == 1, "the handler ran twice; that is not idempotent"
    assert second.replayed is True
    assert second.data == first.data, "a replay must return the ORIGINAL answer"


def test_the_same_id_with_a_different_payload_is_a_conflict(conn) -> None:
    connector = TeamsConnector(conn, credential_check=lambda c: True,
                               handlers={Capability.TEAMS_STATUS: _ok_handler})
    connector.execute(_request())
    with pytest.raises(ConnectorError) as excinfo:
        connector.execute(_request(payload={"k": "CHANGED"}))
    assert excinfo.value.failure is FailureClass.CONFLICT


def test_key_order_is_not_a_conflict(conn) -> None:
    """Two identical bodies differing only in key order are the same request."""
    connector = TeamsConnector(conn, credential_check=lambda c: True,
                               handlers={Capability.TEAMS_STATUS: _ok_handler})
    connector.execute(_request(payload={"a": 1, "b": 2}))
    result = connector.execute(_request(payload={"b": 2, "a": 1}))
    assert result.replayed is True


def test_authentication_precedes_the_idempotency_lookup(conn) -> None:
    """Otherwise an unauthenticated caller can confirm a request id exists."""
    wired = TeamsConnector(conn, credential_check=lambda c: True,
                           handlers={Capability.TEAMS_STATUS: _ok_handler})
    wired.execute(_request())

    unwired = TeamsConnector(conn)
    with pytest.raises(ConnectorError) as excinfo:
        unwired.execute(_request())
    # UNAUTHENTICATED, not a replayed 'ok' -- the existing row must not be
    # observable to a caller who never authenticated.
    assert excinfo.value.failure is FailureClass.UNAUTHENTICATED


# --- shape ----------------------------------------------------------------

def test_an_unbuilt_family_refuses_rather_than_appearing_to_succeed(conn) -> None:
    connector = TeamsConnector(conn, credential_check=lambda c: True)
    with pytest.raises(ConnectorError) as excinfo:
        connector.execute(_request(Capability.MEMBERSHIPS_GRANT))
    assert excinfo.value.failure is FailureClass.NOT_IMPLEMENTED


def test_there_is_no_destructive_principal_delete() -> None:
    """The ratified contract is disable-then-archive, never delete."""
    verbs = {c.value.rsplit(".", 1)[-1] for c in Capability}
    assert "delete" not in verbs
    assert Capability.PRINCIPALS_ARCHIVE in Capability
    assert Capability.PRINCIPALS_SUSPEND in Capability


def test_every_operation_leaves_an_audit_row_without_payload_content(conn) -> None:
    connector = TeamsConnector(conn, credential_check=lambda c: c == "secret",
                               handlers={Capability.TEAMS_STATUS: _ok_handler})
    connector.execute(_request(payload={"secret_note": "carus private text"}))
    with pytest.raises(ConnectorError):
        connector.execute(_request(request_id="req-2", credential="wrong"))

    rows = list(conn.execute(
        "SELECT operation, allowed, denial_reason, detail FROM lcm_teams_audit"
    ))
    assert len(rows) == 2
    assert {r[1] for r in rows} == {0, 1}
    blob = " ".join(str(cell) for row in rows for cell in row)
    assert "carus private text" not in blob, "audit row carries payload content"


def test_the_request_ledger_is_part_of_the_catalog() -> None:
    assert "lcm_teams_requests" in catalog.TEAMS_TABLES
