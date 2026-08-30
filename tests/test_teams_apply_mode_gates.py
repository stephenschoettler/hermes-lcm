"""Apply-mode subcommands that rewrite derived state are authorized.

Three handlers reached their writable stores with NO ``policy_for_engine``
anywhere in the call chain:

- ``/lcm assertions rebuild --apply``
- ``/lcm embed backfill --apply``
- ``/lcm embed backfill --corpus chunks --apply``

Under Teams that let any principal rewrite assertions and embeddings derived
from EVERY principal's memory. The other apply-mode handlers are covered
transitively through ``maintenance.py``; these were not, and nothing failed --
the completeness test only observes handlers that already call the seam, so a
handler that never calls it is invisible to the very test meant to catch this.

They were also reported as missing "the backup every other apply-mode handler
takes". That claim did not survive checking: the three handlers that back up --
rotate, doctor repair, doctor repair schema stamp -- are destructive REPAIRS.
These three are idempotent, incremental rebuilds of DERIVED state, where
re-running is the recovery path, so a full-database copy per batch would cost
real money on a large store and buy nothing. Gate yes, backup no.
"""

from __future__ import annotations

import pytest

from hermes_lcm import command as command_module
from hermes_lcm.access_context import Decision, DenialReason
from hermes_lcm.access_policy import AuthorizationRequiredError


class _DenyingPolicy:
    def authorize_operation(self, _context, _operation, _scope) -> Decision:
        return Decision.deny(DenialReason.SCOPE_FORBIDDEN)

    def audit_decision(self, *_args, **_kwargs) -> None:
        return None


class _AllowingPolicy:
    def __init__(self) -> None:
        self.seen: list[tuple[str, dict]] = []

    def authorize_operation(self, _context, operation, scope) -> Decision:
        self.seen.append((operation, dict(scope)))
        return Decision.allow()

    def audit_decision(self, *_args, **_kwargs) -> None:
        return None


def test_a_denied_principal_cannot_run_an_apply_mutation(monkeypatch) -> None:
    monkeypatch.setattr(command_module, "policy_for_engine", lambda _e: _DenyingPolicy())
    monkeypatch.setattr(command_module, "policy_access_context", lambda _e: None)

    with pytest.raises(AuthorizationRequiredError):
        command_module._authorize_apply_mutation(
            object(), kind="assertions_rebuild", entry_point="rebuild_assertions"
        )


def test_the_gate_requests_owner_authority(monkeypatch) -> None:
    """Not plain `write`: these rewrite state derived from everyone's memory."""
    policy = _AllowingPolicy()
    monkeypatch.setattr(command_module, "policy_for_engine", lambda _e: policy)
    monkeypatch.setattr(command_module, "policy_access_context", lambda _e: None)

    command_module._authorize_apply_mutation(
        object(), kind="embedding_backfill", entry_point="x"
    )

    operation, scope = policy.seen[0]
    assert operation == "owner_only"
    assert scope["required_scope"] == "owner_only"


def test_the_denial_carries_the_public_projection(monkeypatch) -> None:
    monkeypatch.setattr(command_module, "policy_for_engine", lambda _e: _DenyingPolicy())
    monkeypatch.setattr(command_module, "policy_access_context", lambda _e: None)

    with pytest.raises(AuthorizationRequiredError) as excinfo:
        command_module._authorize_apply_mutation(object(), kind="k", entry_point="e")

    assert excinfo.value.denial_reason is DenialReason.TARGET_NOT_FOUND_OR_FORBIDDEN
    assert excinfo.value.denial_reason is not DenialReason.SCOPE_FORBIDDEN


@pytest.mark.parametrize(
    "handler, needle",
    [
        ("_assertions_rebuild_text", "assertions_rebuild"),
        ("_embedding_backfill_summary_text", "embedding_backfill"),
        ("_chunk_backfill_text", "chunk_backfill"),
    ],
)
def test_every_apply_handler_is_gated(handler: str, needle: str) -> None:
    """Structural, so a fourth handler added without a gate is visible.

    Asserting on the source is deliberate: these three were introduced without
    a gate and no behavioural test noticed for as long as they existed.
    """
    import inspect

    source = inspect.getsource(getattr(command_module, handler))
    assert "_authorize_apply_mutation(" in source, f"{handler} is ungated"
    assert needle in source


def test_the_gate_runs_before_the_writable_store_is_opened() -> None:
    """Ordering is the property. Authorizing after the write is theatre."""
    import inspect

    for handler in ("_embedding_backfill_summary_text", "_chunk_backfill_text"):
        source = inspect.getsource(getattr(command_module, handler))
        gate = source.index("_authorize_apply_mutation(")
        store = source.index("VectorStore(")
        assert gate < store, f"{handler}: authorized after opening the writable store"
