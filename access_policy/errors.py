"""Errors raised when a disclosure primitive is reached without authority."""

from __future__ import annotations

from ..access_context.denials import DenialReason


class AuthorizationRequiredError(RuntimeError):
    """A disclosure primitive was called on a denying policy.

    Reaching one of these is a seam bug, not a user-facing denial: the caller
    should have stopped at ``authorize_operation``. Raising keeps that loud.
    Returning a ``Decision`` instead would be worse than useless -- the
    protocol declares ``int`` and ``Sequence`` returns, and a ``Decision`` is
    truthy, so ``if policy.select_collection(scope):`` would proceed as though
    a real collection had been handed back. A fail-closed policy must not fail
    open at the call site.
    """

    def __init__(self, primitive: str, denial_reason: DenialReason) -> None:
        super().__init__(
            f"{primitive} called without authorization ({denial_reason.value}); "
            "authorize_operation must be consulted first"
        )
        self.primitive = primitive
        self.denial_reason = denial_reason
