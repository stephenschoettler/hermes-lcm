"""Schema and accessors for the Teams catalog.

Created only on an explicit enable. Ordinary startup must leave a store with no
Teams tables at all -- the same discipline the access_scope columns follow, and
the reason a single-user install is unaffected by any of this.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ..db_bootstrap import mark_migration_step_complete


TEAMS_CATALOG_MIGRATION = "teams_catalog_v1"

# Every table shares the lcm_teams prefix so db_bootstrap's feature-family
# allowlist can recognise the whole family with one entry. Without that,
# repair tooling refuses to repair a Teams-enabled store.
TEAMS_TABLES = (
    "lcm_teams_tenants",
    "lcm_teams_principals",
    "lcm_teams_collections",
    "lcm_teams_memberships",
    "lcm_teams_revisions",
    "lcm_teams_audit",
    "lcm_teams_requests",
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS lcm_teams_tenants (
    tenant_id  TEXT PRIMARY KEY,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS lcm_teams_principals (
    principal_id TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    status       TEXT NOT NULL CHECK(status IN ('active', 'suspended')),
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lcm_teams_principals_tenant
    ON lcm_teams_principals(tenant_id, status);

CREATE TABLE IF NOT EXISTS lcm_teams_collections (
    collection_id TEXT PRIMARY KEY,
    tenant_id     TEXT NOT NULL,
    kind          TEXT NOT NULL CHECK(kind IN ('own', 'shared')),
    created_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lcm_teams_collections_tenant
    ON lcm_teams_collections(tenant_id, kind);

CREATE TABLE IF NOT EXISTS lcm_teams_memberships (
    principal_id  TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    grants        TEXT NOT NULL,
    created_at    REAL NOT NULL,
    PRIMARY KEY (principal_id, collection_id)
);

CREATE TABLE IF NOT EXISTS lcm_teams_revisions (
    tenant_id           TEXT PRIMARY KEY,
    policy_revision     INTEGER NOT NULL DEFAULT 0,
    membership_revision INTEGER NOT NULL DEFAULT 0,
    revocation_epoch    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS lcm_teams_audit (
    event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at   REAL NOT NULL,
    tenant_id     TEXT,
    principal_id  TEXT,
    operation     TEXT NOT NULL,
    allowed       INTEGER NOT NULL CHECK(allowed IN (0, 1)),
    denial_reason TEXT,
    detail        TEXT
);

CREATE INDEX IF NOT EXISTS idx_lcm_teams_audit_time
    ON lcm_teams_audit(occurred_at);

-- Connector request ledger. #497 requires that a duplicate management request
-- produce ONE effect, and that the same id carrying a DIFFERENT payload be
-- rejected rather than silently applied on top of the first. Both need the
-- original digest kept, so the id alone is not enough.
--
-- `result_json` stores what the first execution returned, so a replay can be
-- answered from the ledger instead of re-running the effect. That is the whole
-- point: idempotency by replaying the ANSWER, not by re-doing the work and
-- hoping it is harmless.
CREATE TABLE IF NOT EXISTS lcm_teams_requests (
    request_id     TEXT PRIMARY KEY,
    payload_digest TEXT NOT NULL,
    capability     TEXT NOT NULL,
    tenant_id      TEXT,
    principal_id   TEXT,
    recorded_at    REAL NOT NULL,
    result_json    TEXT
);
"""


@dataclass(frozen=True)
class CatalogRevisions:
    """The revisions a context is validated against.

    Held by the catalog rather than supplied by the host: under the narrow-shim
    carrier the host authenticates a principal and nothing more, so a context
    arriving with its own revision numbers proves only that someone wrote them
    into it.
    """

    policy_revision: int = 0
    membership_revision: int = 0
    revocation_epoch: int = 0


def teams_catalog_exists(conn: sqlite3.Connection) -> bool:
    """True when the catalog has been materialised on this store.

    Reads sqlite_master and creates nothing. Every caller of this runs on paths
    that must not mutate a store they are only inspecting.
    """

    placeholders = ",".join("?" for _ in TEAMS_TABLES)
    found = conn.execute(
        f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ({placeholders})",
        TEAMS_TABLES,
    ).fetchone()[0]
    return int(found) == len(TEAMS_TABLES)


def ensure_teams_catalog(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Create the catalog if absent and return the tables it now owns.

    Idempotent: every statement is CREATE ... IF NOT EXISTS, so a second enable
    is a no-op rather than an error. The core numeric schema_version is
    deliberately NOT bumped -- this is an opt-in feature family, and bumping it
    would make every Teams store look newer than a stock build to a downgrade
    check that is entitled to refuse.
    """

    conn.executescript(_SCHEMA)
    mark_migration_step_complete(conn, TEAMS_CATALOG_MIGRATION)
    conn.commit()
    return TEAMS_TABLES


def verify_teams_catalog(conn: sqlite3.Connection) -> list[str]:
    """Return structural defects, without mutating anything."""

    errors: list[str] = []
    for table in TEAMS_TABLES:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if row is None:
            errors.append(f"missing table:{table}")
    return sorted(errors)


def read_revisions(conn: sqlite3.Connection, tenant_id: str) -> CatalogRevisions:
    """Return the tenant's current revisions.

    An unknown tenant reads as all-zero rather than raising. A context that
    genuinely belongs to no tenant will fail the ownership stage on its
    principal, which is a more precise answer than a lookup error here.
    """

    row = conn.execute(
        "SELECT policy_revision, membership_revision, revocation_epoch "
        "FROM lcm_teams_revisions WHERE tenant_id = ?",
        (tenant_id,),
    ).fetchone()
    if row is None:
        return CatalogRevisions()
    return CatalogRevisions(
        policy_revision=int(row[0] or 0),
        membership_revision=int(row[1] or 0),
        revocation_epoch=int(row[2] or 0),
    )


def record_audit_event(
    conn: sqlite3.Connection,
    *,
    occurred_at: float,
    tenant_id: str,
    principal_id: str,
    operation: str,
    allowed: bool,
    denial_reason: str | None = None,
    detail: str | None = None,
) -> None:
    """Append one authorization outcome to the audit trail.

    The denial reason stored here is the PUBLIC projection, never the internal
    one. #497 exposes an ``audit.*`` family, so these rows can leave the store
    and reach a tenant admin; an internal reason distinguishes "forbidden" from
    "does not exist", which is exactly the distinction the public projection
    exists to collapse. An operator debugging a denial has the operation, the
    principal and the timestamp, which is enough to correlate.

    Best-effort by construction: auditing must never be the reason an
    authorized operation fails. A store whose audit table is missing or locked
    still serves its principals.
    """

    try:
        conn.execute(
            "INSERT INTO lcm_teams_audit("
            "occurred_at, tenant_id, principal_id, operation, allowed, "
            "denial_reason, detail) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                float(occurred_at),
                str(tenant_id or ""),
                str(principal_id or ""),
                str(operation or ""),
                1 if allowed else 0,
                str(denial_reason) if denial_reason is not None else None,
                str(detail) if detail is not None else None,
            ),
        )
        conn.commit()
    except sqlite3.Error:
        return


def read_audit_events(
    conn: sqlite3.Connection, *, tenant_id: str | None = None, limit: int = 100
) -> list[dict[str, object]]:
    """Read the audit trail, newest first."""

    if tenant_id is None:
        rows = conn.execute(
            "SELECT occurred_at, tenant_id, principal_id, operation, allowed, "
            "denial_reason, detail FROM lcm_teams_audit "
            "ORDER BY event_id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT occurred_at, tenant_id, principal_id, operation, allowed, "
            "denial_reason, detail FROM lcm_teams_audit WHERE tenant_id = ? "
            "ORDER BY event_id DESC LIMIT ?",
            (str(tenant_id), int(limit)),
        ).fetchall()
    return [
        {
            "occurred_at": row[0],
            "tenant_id": row[1],
            "principal_id": row[2],
            "operation": row[3],
            "allowed": bool(row[4]),
            "denial_reason": row[5],
            "detail": row[6],
        }
        for row in rows
    ]


def set_revisions(
    conn: sqlite3.Connection, tenant_id: str, revisions: CatalogRevisions
) -> None:
    """Write a tenant's revisions outright.

    Provisioning needs this: a tenant is created at whatever revisions its
    control plane already issued contexts against, which is not necessarily
    zero. Bumping from zero would only reach those numbers by accident.
    """

    conn.execute(
        "INSERT INTO lcm_teams_revisions("
        "tenant_id, policy_revision, membership_revision, revocation_epoch"
        ") VALUES(?, ?, ?, ?) "
        "ON CONFLICT(tenant_id) DO UPDATE SET "
        "policy_revision = excluded.policy_revision, "
        "membership_revision = excluded.membership_revision, "
        "revocation_epoch = excluded.revocation_epoch",
        (
            tenant_id,
            int(revisions.policy_revision),
            int(revisions.membership_revision),
            int(revisions.revocation_epoch),
        ),
    )
    conn.commit()


def bump_revision(conn: sqlite3.Connection, tenant_id: str, field: str) -> int:
    """Advance one revision counter and return its new value.

    Revoking access is a revision bump, so this is the operation that makes a
    previously-issued context stale. The field name is checked against the
    known set rather than interpolated blindly.
    """

    if field not in {"policy_revision", "membership_revision", "revocation_epoch"}:
        raise ValueError(f"unknown revision field: {field}")
    conn.execute(
        "INSERT INTO lcm_teams_revisions(tenant_id) VALUES(?) "
        "ON CONFLICT(tenant_id) DO NOTHING",
        (tenant_id,),
    )
    conn.execute(
        f"UPDATE lcm_teams_revisions SET {field} = {field} + 1 WHERE tenant_id = ?",
        (tenant_id,),
    )
    conn.commit()
    return getattr(read_revisions(conn, tenant_id), field)


# ---------------------------------------------------------------------------
# Principals, collections and memberships
#
# These were the accessors the catalog promised and did not have, which is why
# TeamsPolicy decides from the CONTEXT rather than from the catalog: with no way
# to ask "which collections may this principal read", a shared collection could
# not be modelled at all, and the policy could only answer the private case.
#
# Every write bumps the matching revision. That is not bookkeeping -- the narrow
# host carrier deliberately does NOT send revisions, so the catalog is the only
# thing that can say a membership changed, and a stale context is detected by
# comparing against these counters.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    principal_id: str
    tenant_id: str
    status: str


@dataclass(frozen=True)
class Membership:
    principal_id: str
    collection_id: str
    grants: tuple[str, ...]


# ``status`` is CHECK-constrained to these two. Note what is MISSING: there is
# no 'archived'. The ratified contract calls removal "disable-then-archive,
# never destructive delete", so archive needs either a third status or an
# archived_at column, and adding it is a schema migration rather than an
# accessor. Suspend is what exists today, and it is what `principals.suspend`
# maps to; `principals.archive` cannot be honoured until that migration lands.
PRINCIPAL_STATUSES = ("active", "suspended")


def provision_principal(
    conn: sqlite3.Connection,
    *,
    principal_id: str,
    tenant_id: str,
    now: float,
    status: str = "active",
) -> Principal:
    """Create or re-activate a principal. Idempotent by primary key."""

    if status not in PRINCIPAL_STATUSES:
        raise ValueError(f"status must be one of {PRINCIPAL_STATUSES}")
    conn.execute(
        "INSERT INTO lcm_teams_principals(principal_id, tenant_id, status,"
        " created_at, updated_at) VALUES(?,?,?,?,?)"
        " ON CONFLICT(principal_id) DO UPDATE SET status=excluded.status,"
        " updated_at=excluded.updated_at",
        (str(principal_id), str(tenant_id), status, float(now), float(now)),
    )
    conn.commit()
    bump_revision(conn, str(tenant_id), "membership_revision")
    return Principal(str(principal_id), str(tenant_id), status)


def suspend_principal(
    conn: sqlite3.Connection, *, principal_id: str, tenant_id: str, now: float
) -> Principal:
    """Suspend without unstamping anything.

    The principal's rows keep their access_scope, exactly as `disable_teams`
    keeps stamps: attribution is what a later re-provision and every audit
    answer depend on. Suspension also bumps the revocation epoch, so contexts
    already issued to this principal stop validating rather than running to
    their natural expiry.
    """

    conn.execute(
        "UPDATE lcm_teams_principals SET status='suspended', updated_at=?"
        " WHERE principal_id=?",
        (float(now), str(principal_id)),
    )
    conn.commit()
    bump_revision(conn, str(tenant_id), "revocation_epoch")
    return Principal(str(principal_id), str(tenant_id), "suspended")


def read_principal(conn: sqlite3.Connection, principal_id: str) -> Principal | None:
    row = conn.execute(
        "SELECT principal_id, tenant_id, status FROM lcm_teams_principals"
        " WHERE principal_id = ?",
        (str(principal_id),),
    ).fetchone()
    return Principal(str(row[0]), str(row[1]), str(row[2])) if row else None


def create_collection(
    conn: sqlite3.Connection,
    *,
    collection_id: str,
    tenant_id: str,
    kind: str,
    now: float,
) -> str:
    if kind not in ("own", "shared"):
        raise ValueError("kind must be 'own' or 'shared'")
    conn.execute(
        "INSERT OR IGNORE INTO lcm_teams_collections(collection_id, tenant_id,"
        " kind, created_at) VALUES(?,?,?,?)",
        (str(collection_id), str(tenant_id), kind, float(now)),
    )
    conn.commit()
    return str(collection_id)


def grant_membership(
    conn: sqlite3.Connection,
    *,
    principal_id: str,
    collection_id: str,
    grants: "tuple[str, ...] | list[str]",
    tenant_id: str,
    now: float,
) -> Membership:
    """Grant a principal access to a collection. Idempotent."""

    normalized = tuple(sorted({str(g).strip() for g in grants if str(g).strip()}))
    conn.execute(
        "INSERT INTO lcm_teams_memberships(principal_id, collection_id, grants,"
        " created_at) VALUES(?,?,?,?)"
        " ON CONFLICT(principal_id, collection_id) DO UPDATE SET grants=excluded.grants",
        (str(principal_id), str(collection_id), ",".join(normalized), float(now)),
    )
    conn.commit()
    bump_revision(conn, str(tenant_id), "membership_revision")
    return Membership(str(principal_id), str(collection_id), normalized)


def revoke_membership(
    conn: sqlite3.Connection, *, principal_id: str, collection_id: str, tenant_id: str
) -> bool:
    """Remove a grant and bump the revocation epoch.

    #498 requires revocation to block the NEXT operation, not eventually. The
    epoch bump is what does that: a context issued before it stops validating
    immediately, rather than remaining good until its lease expires.
    """

    cursor = conn.execute(
        "DELETE FROM lcm_teams_memberships WHERE principal_id=? AND collection_id=?",
        (str(principal_id), str(collection_id)),
    )
    conn.commit()
    bump_revision(conn, str(tenant_id), "revocation_epoch")
    bump_revision(conn, str(tenant_id), "membership_revision")
    return cursor.rowcount > 0


def read_memberships(
    conn: sqlite3.Connection, principal_id: str
) -> tuple[Membership, ...]:
    rows = conn.execute(
        "SELECT principal_id, collection_id, grants FROM lcm_teams_memberships"
        " WHERE principal_id = ? ORDER BY collection_id",
        (str(principal_id),),
    ).fetchall()
    return tuple(
        Membership(
            str(r[0]),
            str(r[1]),
            tuple(g for g in str(r[2]).split(",") if g),
        )
        for r in rows
    )


def authorized_collections(
    conn: sqlite3.Connection, principal_id: str, *, grant: str = "read"
) -> tuple[str, ...]:
    """The collections this principal may act on -- the question the policy asks.

    A SUSPENDED principal gets nothing, regardless of surviving membership rows.
    Suspension is deliberately non-destructive, so the rows are still there; if
    this read went straight to memberships, suspending someone would remove
    their status and leave their access intact.
    """

    principal = read_principal(conn, principal_id)
    if principal is None or principal.status != "active":
        return ()
    return tuple(
        m.collection_id for m in read_memberships(conn, principal_id) if grant in m.grants
    )
