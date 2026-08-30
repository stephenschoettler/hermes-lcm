# AccessContextV1 contract
`AccessContextV1` is an additive, inert contract for the future Hermes Agent
host carrier and Hermes-LCM authorization seam. This change defines data,
validation order, denial projection, protocols, shared JSON vectors, and the
authority-path inventory. It does not import the package from the existing
engine, add a feature flag, enforce policy, issue handles, or change a tool.

## Authority and lifetime

The immutable context is derived from authenticated host transport and session
state. Profile, session, conversation, collection, cursor, reference, and
request IDs carry lineage; none grants authority by itself. `narrow()` and
`derive_child()` return new contexts and reject any attempted widening.
Delegation intersects operation, collection, audience/binding, expiry, and
current revision boundaries. The complete delegation chain and narrowing set
are retained for re-delegation inspection.

Validation is deterministic and injected-time only:

`CONTEXT_PRESENT → CONTEXT_WELL_FORMED → REVISION_SUPPORTED → NOT_EXPIRED → NOT_REVOKED → OWNERSHIP_CURRENT → LEASE_CURRENT → SCOPE_PERMITTED → TARGET_RESOLUTION`

Authorization is therefore before collection selection, existence/count,
ranking, hydration, and handle issuance. The consumer protocol exposes those
disclosure primitives so a seam conformance test can record that order.

The authority-path inventory includes the public `lcm_*` tools and the
non-tool paths that can bypass those handlers: store/compaction/rollup and
sidecar writes; maintenance, import, schema, and diagnostics; retrieval and
expansion; auxiliary/lifecycle session state; and host callbacks in
`engine.py`. The `cron` category is represented by the real
`_RollupMaintenanceScheduler`; this repository has no separate OS cron entry
point, so the scheduler note is the honest boundary rather than an invented
function.

## Denials

The internal `Decision` preserves the exact `DenialReason` and a content-free
detail mapping containing IDs or revisions only. `PUBLIC_DENIAL_PROJECTION` is
the single disclosure table: scope, ownership, lease, and target denials are
projected as `target_not_found_or_forbidden`; context lifecycle denials remain
typed.

## Standard single-user compatibility

The contract has no runtime call sites and is default-off by construction. The
carrier matrix is deliberately explicit:

| Host carrier | Teams | Mode |
| --- | --- | --- |
| absent (`None`) | disabled | `STANDARD_UNMANAGED`; standard single-user behavior is entirely unchanged and the contract is not consulted |
| absent (`None`) | enabled | `FAIL_CLOSED`; every Teams-governed path returns `context_missing`, with no unscoped fallback |
| present | disabled | `STANDARD_UNMANAGED`; the context is ignored and carrying one cannot enable Teams |
| present | enabled | `ENFORCING`; normal validation applies |

No context-var, thread-local, crypto, token verifier, policy DSL, collection
catalog, audit store, retry framework, or other hardening is part of this
contract. Enforcement and policy remain the explicitly named follow-up seams.
