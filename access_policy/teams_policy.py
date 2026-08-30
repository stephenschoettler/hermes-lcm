"""DRAFT per-principal policy — the experiment the plan calls for first.

Purpose of this draft is to answer ONE question: do the two levers that already
exist -- per-row ``authorize_stored_scope`` and a ``resolve_authorized_targets``
that overrides rather than passes through -- cover every leak probe in the
isolation smoke, or does some read path need an ``access_scope`` predicate added
to its query?

Not the finished policy. Membership, shared collections and delegation all
resolve from the catalog in the real one; this draft decides from the context
alone, which is enough to find out which probes still leak.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ..access_context.denials import Decision, DenialReason, PublicDecision
from ..access_context.model import AccessContextV1
from ..access_context.protocols import TargetScope


# Operations that touch the whole store rather than one principal's targets.
# A backup copies every principal's memory into one file, so it is an
# administrative capability the connector holds, not something a principal
# inherits by being the only one logged in.
#
# The three apply-mode rebuilds belong here for the same reason, and leaving
# them out made their gate INERT: their scope carries no session, partition or
# target key, so the owner-of-target loop below had nothing to compare and
# every principal fell through to allow(). The gate existed, ran, and permitted
# everyone -- and its test used a denying STUB, so it proved the gate propagates
# a denial rather than that this policy produces one.
_STORE_WIDE_KINDS = frozenset(
    {"backup", "assertions_rebuild", "embedding_backfill", "chunk_backfill"}
)


# The OTHER half of store-wide authority, reached by `required_scope` rather than
# by `kind`. `_authorize_doctor_command` (command.py) asks for "admin" with
# kind="slash_command", and nothing below read it -- so the gate ran, allowed
# every principal, and `/lcm doctor clean` handed one principal every other
# principal's session ids, message counts and token totals via
# `scan_session_cleanup_stats`, which takes no filter at all.
#
# "admin" ONLY, deliberately. "owner_only" must NEVER be added here: it is
# overloaded, and on `on_session_reset` it means owner OF THE TARGET -- a
# principal resetting its OWN session, carried with `session_id: self._session_id`.
# Denying the whole word is precisely the conflation that once denied principal A
# its own session load. The two store-wide `owner_only` sites are already covered
# by kind above.
_ADMIN_REQUIRED_SCOPES = frozenset({"admin"})


def principal_of(context: AccessContextV1 | None) -> str:
    """The owner scope a row would be stamped with for this context.

    Must match engine._access_scope_for_storage_session exactly, or reads will
    disagree with writes and the store will look isolated while being broken.
    """

    if context is None:
        return ""
    return str(context.session_owner_principal_id or context.principal_id or "")


class TeamsPolicy:
    """Scope every operation to the acting principal."""

    def __init__(
        self,
        context: AccessContextV1 | None,
        audit_sink: "Callable[..., None] | None" = None,
        session_owner: "Callable[[str], str | None] | None" = None,
    ) -> None:
        self._context = context
        self._audit_sink = audit_sink
        self._session_owner = session_owner
        self.teams_enabled = True

    def _target_owner(self, session_id: str) -> str | None:
        """Who owns a target session, or None when nothing claims it.

        Resolved through a seam-bound callable rather than a database handle,
        the same way the audit sink is bound -- the policy stays pure and
        testable with a plain dict.
        """

        if self._session_owner is None:
            return None
        try:
            owner = self._session_owner(session_id)
        except Exception:  # noqa: BLE001 - an unreadable owner is not a claim
            return None
        return str(owner) if owner else None

    # -- authorization ----------------------------------------------------

    def authorize_operation(
        self,
        context: AccessContextV1 | None,
        operation: str,
        expected_scope: TargetScope,
    ) -> Decision:
        effective = context if context is not None else self._context
        principal = principal_of(effective)
        if not principal:
            return Decision.deny(DenialReason.CONTEXT_INVALID)

        # Store-WIDE operations are not a principal's to perform. `backup_database`
        # copies the entire file, every principal's memory included, so under
        # Teams no principal holds it -- not even the one who happens to be
        # first. It belongs to the connector, which #497 already gives the
        # backup/migration/audit families and which authenticates separately.
        #
        # This is the half of `owner_only` that is store-wide admin. The other
        # half -- owner OF THE TARGET -- is the session checks below. Treating
        # them as one authority is what made a policy that enforced the natural
        # `required <= operation_allowlist` rule deny principal A its own
        # session load: neither principal holds `owner_only`, and no fixture
        # tweak makes that coherent.
        if str(expected_scope.get("kind") or "") in _STORE_WIDE_KINDS:
            return Decision.deny(DenialReason.SCOPE_FORBIDDEN)

        # Administrative authority, requested by scope rather than by kind. It
        # belongs to the connector -- #497 gives it the audit/migration/backup
        # families and authenticates it separately -- never to a principal
        # because it happens to be the one logged in.
        if str(expected_scope.get("required_scope") or "") in _ADMIN_REQUIRED_SCOPES:
            return Decision.deny(DenialReason.SCOPE_FORBIDDEN)

        # Owner OF THE TARGET -- an OWNER comparison, not a session-id one.
        #
        # The first version of this compared the target session id against the
        # context's, which denied a principal its own auxiliary session:
        # on_session_end passes the TARGET session (engine.py:3450-3460), and an
        # auxiliary session id is deliberately not the bound one. Right intent,
        # wrong rule.
        #
        # An UNRESOLVED target is allowed on purpose. A session nothing has
        # claimed yet cannot belong to another principal, and denying would
        # break creating any new session under Teams. It is safe because the
        # write is stamped with the WRITER's scope on the way in, so an unknown
        # session becomes the writer's rather than a way into someone else's.
        #
        # The key differs by path: most carry `session_id`, compression rollover
        # carries `source_session_id`, and rollup scheduling carries the session
        # as `partition_key`.
        for key in ("session_id", "source_session_id", "partition_key"):
            target = expected_scope.get(key)
            if not target or effective is None:
                continue
            target = str(target)
            if target == str(effective.session_id):
                continue
            owner = self._target_owner(target)
            if owner is not None and owner != principal:
                return Decision.deny(DenialReason.SCOPE_FORBIDDEN)

        # Raw identifiers (store_id, node_id) name a row without saying who owns
        # it. The engine resolves the stored owner before authorizing precisely
        # so this comparison is possible; without it, expanding another
        # principal's message by id was allowed and returned their content.
        for owner in expected_scope.get("target_access_scopes") or ():
            if str(owner) != principal:
                return Decision.deny(DenialReason.SCOPE_FORBIDDEN)

        return Decision.allow()

    def resolve_authorized_targets(
        self,
        context: AccessContextV1 | None,
        operation: str,
        requested_narrowing: TargetScope,
    ) -> TargetScope:
        """Replace the caller's corpus with the principal's own.

        Explicitly SET rather than omit: run_knn reads this with
        ``authorized_scope.get("source", source)``, so omitting the key keeps
        the caller's value -- which is the leak, not the fix.
        """

        effective = context if context is not None else self._context
        narrowed = dict(requested_narrowing)
        principal = principal_of(effective)
        if not principal:
            return narrowed

        # Narrow by the OWNER STAMP, not by a collection id.
        #
        # This first shipped setting `source` to default_write_collection_id.
        # On the synthetic smoke that worked, because the fixture seeds rows
        # with the same constant it puts in the context. On REAL data the two
        # are unrelated -- stored `source` values look like
        # "openclaw-lcm:agent:acorn:<uuid>" -- so the filter matched nothing and
        # recall returned an empty corpus to EVERY principal. Isolation by
        # breaking retrieval, which a positive control catches and a leak probe
        # does not.
        #
        # access_scope is the property the rows actually carry, and it is the
        # same value the write path stamps, so read and write agree by
        # construction.
        narrowed["access_scope"] = principal
        return narrowed

    def authorize_stored_scope(
        self,
        context: AccessContextV1 | None,
        operation: str,
        stored_scope: TargetScope,
    ) -> Decision:
        effective = context if context is not None else self._context
        principal = principal_of(effective)
        if not principal:
            return Decision.deny(DenialReason.CONTEXT_INVALID)

        stored = stored_scope.get("access_scope")
        if stored is None:
            # Legacy, unstamped. Allowed for now so a partially-migrated store
            # is not bricked; the real policy has to decide this deliberately.
            return Decision.allow()
        if str(stored) != principal:
            return Decision.deny(DenialReason.SCOPE_FORBIDDEN)
        return Decision.allow()

    def audit_decision(
        self,
        context: AccessContextV1 | None,
        operation: str,
        internal_reason: DenialReason | None,
        public_result: PublicDecision,
    ) -> None:
        """Record what was decided, without recording why in internal terms.

        Volume is deliberately bounded. Every denial is recorded, but only
        NON-READ allows are: `audit_decision` has 39 call sites and reads
        dominate them, so a row per authorization would put an INSERT in the
        hot path of every retrieval. This branch already cost this branch a
        48s->174s regression once, from far less.

        The reason written out is ``public_result``'s, never
        ``internal_reason`` -- see record_audit_event for why that matters when
        an audit export leaves the store.
        """

        if self._audit_sink is None:
            return None
        allowed = bool(getattr(public_result, "allowed", False))
        if allowed and str(operation) == "read":
            return None
        effective = context if context is not None else self._context
        reason = getattr(public_result, "denial_reason", None)
        self._audit_sink(
            tenant_id=str(getattr(effective, "tenant_id", "") or ""),
            principal_id=principal_of(effective),
            operation=str(operation),
            allowed=allowed,
            denial_reason=getattr(reason, "value", None) if reason else None,
        )
        return None

    # -- disclosure primitives (no production call sites; protocol only) ---

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
