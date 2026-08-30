"""Neutral default-deny policy for an unusable Teams context."""

from __future__ import annotations

from typing import Any, Sequence

from ..access_context.denials import Decision, DenialReason, PublicDecision
from ..access_context.model import AccessContextV1
from ..access_context.protocols import TargetScope

from .errors import AuthorizationRequiredError


class FailClosedPolicy:
    """Deny every authorization or disclosure operation without a context."""

    def __init__(self, denial_reason: DenialReason = DenialReason.CONTEXT_MISSING) -> None:
        self.denial_reason = DenialReason(denial_reason)
        self.audit_records: list[tuple[DenialReason | None, PublicDecision]] = []

    def _deny(self) -> Decision:
        return Decision.deny(self.denial_reason)

    def authorize_operation(
        self,
        context: AccessContextV1 | None,
        operation: str,
        expected_scope: TargetScope,
    ) -> Decision:
        return self._deny()

    def resolve_authorized_targets(
        self,
        context: AccessContextV1 | None,
        operation: str,
        requested_narrowing: TargetScope,
    ) -> TargetScope:
        # An empty SCOPE, not an empty sequence: callers read this with .get,
        # and the protocol declares a mapping. Both are falsy, so the practical
        # behaviour is unchanged -- this keeps the type honest.
        return {}

    def authorize_stored_scope(
        self,
        context: AccessContextV1 | None,
        operation: str,
        stored_scope: TargetScope,
    ) -> Decision:
        return self._deny()

    def audit_decision(
        self,
        context: AccessContextV1 | None,
        operation: str,
        internal_reason: DenialReason | None,
        public_result: PublicDecision,
    ) -> None:
        self.audit_records.append((internal_reason, public_result))

    # The disclosure primitives RAISE rather than return a Decision. The
    # protocol declares int/Sequence returns here, and a Decision is truthy,
    # so returning one would let `if policy.select_collection(scope):` sail
    # straight through -- a fail-closed policy failing open at the call site.
    def _refuse(self, primitive: str) -> Any:
        raise AuthorizationRequiredError(
            primitive,
            self._deny().public().denial_reason,
        )

    def select_collection(self, target_scope: TargetScope) -> Any:
        return self._refuse("select_collection")

    def count_candidates(self, candidates: Sequence[Any]) -> int:
        return self._refuse("count_candidates")

    def rank_candidates(self, candidates: Sequence[Any]) -> Sequence[Any]:
        return self._refuse("rank_candidates")

    def hydrate_targets(self, targets: Sequence[Any]) -> Sequence[Any]:
        return self._refuse("hydrate_targets")

    def issue_handle(self, target: Any) -> Any:
        return self._refuse("issue_handle")
