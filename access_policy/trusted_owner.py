"""The inert, single-owner authorization policy."""

from __future__ import annotations

from typing import Any, Sequence

from ..access_context.denials import Decision, DenialReason, PublicDecision
from ..access_context.model import AccessContextV1
from ..access_context.protocols import TargetScope


class TrustedOwnerPolicy:
    """Preserve the existing trusted-owner behavior for the default path."""

    def __init__(self, *, teams_enabled: bool = False) -> None:
        # This is routing metadata, not an authorization decision.  Hooks use
        # the policy seam to distinguish default-off compatibility from the
        # Teams-enabled placeholder policy without reading host wiring directly.
        self.teams_enabled = bool(teams_enabled)

    def authorize_operation(
        self,
        context: AccessContextV1 | None,
        operation: str,
        expected_scope: TargetScope,
    ) -> Decision:
        return Decision.allow()

    def resolve_authorized_targets(
        self,
        context: AccessContextV1 | None,
        operation: str,
        requested_narrowing: TargetScope,
    ) -> TargetScope:
        return requested_narrowing

    def authorize_stored_scope(
        self,
        context: AccessContextV1 | None,
        operation: str,
        stored_scope: TargetScope,
    ) -> Decision:
        return Decision.allow()

    def audit_decision(
        self,
        context: AccessContextV1 | None,
        operation: str,
        internal_reason: DenialReason | None,
        public_result: PublicDecision,
    ) -> None:
        return None

    def select_collection(self, target_scope: TargetScope) -> Any:
        return target_scope

    def count_candidates(self, candidates: Sequence[Any]) -> int:
        return len(candidates)

    def rank_candidates(self, candidates: Sequence[Any]) -> Sequence[Any]:
        return candidates

    def hydrate_targets(self, targets: Sequence[Any]) -> Sequence[Any]:
        return targets

    def issue_handle(self, target: Any) -> Any:
        return target
