"""The Teams catalog: who exists, what they may see, and at which revision.

Deliberately provider-neutral. Nothing in this package knows about
ElectricSheep, gbrain, pipedream, or any other caller's vocabulary -- a host
maps its own identity model onto these tables, and the mapping lives in the
host's repository rather than here.

The catalog OWNS the three revisions and the lease that ``AccessContextV1``
carries. Under the ratified narrow-shim carrier the host supplies only an
authenticated principal, tenant and session identity per turn; it never sends a
revision, so nothing here may assume it will.
"""

from .catalog import (
    TEAMS_CATALOG_MIGRATION,
    TEAMS_TABLES,
    CatalogRevisions,
    bump_revision,
    ensure_teams_catalog,
    read_revisions,
    set_revisions,
    teams_catalog_exists,
    verify_teams_catalog,
)

__all__ = [
    "TEAMS_CATALOG_MIGRATION",
    "TEAMS_TABLES",
    "CatalogRevisions",
    "bump_revision",
    "ensure_teams_catalog",
    "read_revisions",
    "set_revisions",
    "teams_catalog_exists",
    "verify_teams_catalog",
]
