"""Shared SQLite bootstrap helpers for hermes-lcm.

This module keeps startup DB initialization in one place so store/DAG use the
same schema-version marker, PRAGMA settings, and FTS repair behavior.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import sqlite3
import threading
import time
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


class SchemaVersionTooNewError(RuntimeError):
    """Raised when a database was written by a newer LCM schema than this build.

    Opening and migrating such a database with older code risks silently
    corrupting data written under semantics this build does not understand, so
    we refuse rather than degrade.
    """


# The core schema ladder stops at 5. Optional embedding tables are NOT part of
# this counter: they are created lazily+idempotently by VectorStore on first use
# (see ``ensure_embedding_tables`` / ``VectorStore._ensure_embedding_schema``) and
# recorded via the named ``embeddings_v1`` migration-state marker instead of a
# numeric bump. This keeps a disabled install at schema_version 5 with no
# embedding tables, fully openable by a base build, and leaves the numeric
# counter free for the temporal train so neither collides on a v6.
SCHEMA_VERSION = 5
SQLITE_BUSY_TIMEOUT_MS = 30_000
_MIN_DISK_SPACE_BYTES = 50 * 1024 * 1024
REQUIRED_CORE_TABLES = (
    "messages",
    "metadata",
    "summary_nodes",
    "lcm_lifecycle_state",
    "lcm_migration_state",
    "messages_fts",
    "nodes_fts",
)


class ExternalContentFtsSpec:
    def __init__(
        self,
        *,
        table_name: str,
        content_table: str,
        content_rowid: str,
        indexed_column: str,
        trigger_sqls: Sequence[str],
    ) -> None:
        self.table_name = table_name
        self.content_table = content_table
        self.content_rowid = content_rowid
        self.indexed_column = indexed_column
        self.trigger_sqls = tuple(trigger_sqls)


def configure_connection(conn: sqlite3.Connection) -> None:
    """Configure SQLite connection for WAL durability and hygiene.

    In a multi-agent deployment (gateway process + CLI sessions + sub-agents),
    every process opens its own sqlite3.Connection pointing at the same
    lcm.db file.  These settings improve committed-write durability and WAL
    hygiene, but do NOT make sibling processes safe from an unexpected process
    death.  Abnormal exit still depends on normal SQLite WAL recovery;
    application-level checkpoints only run during graceful shutdown (see
    ``MessageStore.close()`` etc.).

    Key design decisions:
    - journal_mode=WAL  : writes go to a separate log; readers never block.
    - synchronous=FULL  : fsync both the WAL and the WAL index before every
                          write transaction commit.  WAL + FULL is the only
                          combination SQLite guarantees survives power loss
                          without data loss (NORMAL may lose the WAL index).
    - wal_autocheckpoint=500 : after 500 WAL pages (~2 MB) SQLite will try
                               an automatic passive checkpoint.  This is a
                               best-effort hint — it is silently skipped when
                               another connection holds a read transaction.
                               Under checkpoint starvation WAL can grow well
                               beyond this trigger.
    - journal_size_limit=67108864 (64 MiB) : limits the WAL file size after
                                             a successful checkpoint or reset.
                                             It does NOT force a checkpoint
                                             or cap growth while another
                                             connection holds an old WAL
                                             end mark.
    - mmap_size=268435456 (256 MiB)        : memory-map reads so concurrent
                                              readers cache WAL pages in RAM.
    """
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    _execute_wal_conversion_with_lock_retry(conn)
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA wal_autocheckpoint=500")
    conn.execute("PRAGMA journal_size_limit=67108864")
    conn.execute("PRAGMA mmap_size=268435456")


def _execute_wal_conversion_with_lock_retry(
    conn: sqlite3.Connection,
    *,
    budget_ms: int = SQLITE_BUSY_TIMEOUT_MS,
) -> None:
    """Run ``PRAGMA journal_mode=WAL`` with a bounded lock-contention retry.

    Converting a rollback-journal database to WAL needs the exclusive lock,
    and SQLite can return ``SQLITE_BUSY`` for that upgrade without consulting
    the busy handler when other connections are mid-setup on the same file.
    Concurrent process startup (gateway + CLI + sub-agents) on a not-yet-WAL
    database therefore crashed sporadically with ``database is locked``.
    Once the database is in WAL mode the pragma is a plain read and never
    takes this path, so the retry only matters on first boot after an
    install/upgrade or on a rollback-journal restore.
    """
    deadline = time.monotonic() + budget_ms / 1000.0
    delay_seconds = 0.005
    while True:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
                raise
        time.sleep(delay_seconds)
        delay_seconds = min(delay_seconds * 2, 0.25)


def add_column_if_missing(
    conn: sqlite3.Connection,
    existing_columns: set[str],
    column: str,
    alter_sql: str,
) -> None:
    """Idempotently add a column, tolerating a concurrent process that won the race.

    In the multi-agent deployment (gateway + CLI sessions + sub-agents) every
    process opens its own connection to the same ``lcm.db`` and runs startup
    migrations concurrently.  A plain check-``PRAGMA table_info``-then-``ALTER``
    races: two processes both observe the column as absent (each within its own
    connection snapshot) and both issue ``ALTER TABLE ... ADD COLUMN``.  The loser
    then raised ``sqlite3.OperationalError: duplicate column name``, which
    propagated out of ``_init_db`` and crashed store construction.  Swallowing
    exactly that error makes the migration idempotent under concurrency; any other
    OperationalError still propagates.
    """
    if column in existing_columns:
        return
    try:
        conn.execute(alter_sql)
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


def ensure_metadata_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )


def get_schema_version(conn: sqlite3.Connection) -> int:
    ensure_metadata_table(conn)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    if not row or row[0] is None:
        return 0
    try:
        return int(str(row[0]))
    except (TypeError, ValueError):
        return 0




def read_existing_schema_version(conn: sqlite3.Connection) -> int:
    """Return schema_version without creating or modifying schema objects."""
    metadata_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
    ).fetchone()
    if not metadata_exists:
        return 0
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'schema_version'"
    ).fetchone()
    if not row or row[0] is None:
        return 0
    try:
        return int(str(row[0]))
    except (TypeError, ValueError):
        return 0


def refuse_schema_version_too_new(conn: sqlite3.Connection) -> None:
    """Raise before any startup DDL when a newer build owns the DB."""
    current_version = read_existing_schema_version(conn)
    if current_version <= SCHEMA_VERSION:
        return
    if classify_version_mismatch(conn) == VERSION_MISMATCH_INTERIM_STAMP:
        raise SchemaVersionTooNewError(
            f"LCM database is stamped schema_version {current_version}, but its "
            f"actual schema is the v{SCHEMA_VERSION} shape plus named feature "
            f"markers — the signature of an interim development build that "
            f"recorded a numeric version it never migrated to. There is no newer "
            f"hermes-lcm to upgrade to; do NOT upgrade the plugin. Run "
            f"`/lcm doctor repair schema-stamp` to preview a backup-first reset "
            f"of the stamp to v{SCHEMA_VERSION} (add `apply` to execute)."
        )
    raise SchemaVersionTooNewError(
        f"LCM database schema version {current_version} is newer than this "
        f"build supports (v{SCHEMA_VERSION}). Refusing to open to avoid "
        f"corrupting data written by a newer hermes-lcm. Upgrade the plugin "
        f"or restore a pre-upgrade backup (.db/-wal/-shm)."
    )


# --- interim-build schema-stamp remediation (fix #7) ------------------------
#
# Some databases touched by interim development builds carry a numeric
# ``schema_version`` ahead of this build's ladder even though their actual shape
# is the v5 core plus the named opt-in feature markers (the trains reverted the
# numeric bump in favour of markers). Such a DB is safe to re-stamp back to
# ``SCHEMA_VERSION``; a genuinely newer DB is not. Classification is read-only
# and errs toward ``genuinely_newer`` so an ambiguous shape is never downgraded.

VERSION_MISMATCH_INTERIM_STAMP = "interim_stamp"
VERSION_MISMATCH_GENUINELY_NEWER = "genuinely_newer"

# The exact v5 column contract for every core table that carries one. A missing
# OR an unexpected column both disqualify a DB from the interim-stamp path.
_V5_CORE_TABLE_COLUMNS: dict[str, frozenset[str]] = {
    "messages": frozenset({
        "store_id", "session_id", "source", "conversation_id", "role",
        "content", "tool_call_id", "tool_calls", "tool_name", "timestamp",
        "token_estimate", "pinned",
    }),
    "summary_nodes": frozenset({
        "node_id", "session_id", "depth", "summary", "token_count",
        "source_token_count", "source_ids", "source_type", "created_at",
        "earliest_at", "latest_at", "expand_hint",
    }),
    "metadata": frozenset({"key", "value"}),
    "lcm_migration_state": frozenset({"step_name", "completed_at"}),
    "lcm_lifecycle_state": frozenset({
        "conversation_id", "current_session_id", "last_finalized_session_id",
        "current_frontier_store_id", "last_finalized_frontier_store_id",
        "debt_kind", "debt_size_estimate", "current_bound_at",
        "last_finalized_at", "debt_updated_at", "last_maintenance_attempt_at",
        "last_rollover_at", "last_reset_at", "updated_at",
    }),
}

# Core FTS5 virtual tables: presence is enough — their column layout is owned by
# the FTS5 module, not by this schema contract.
_V5_CORE_PRESENCE_ONLY = ("messages_fts", "nodes_fts")

# Extra tables are tolerated only when they belong to a known opt-in feature
# family (temporal-rollup / embedding / chunk) or are FTS5 shadow tables of the
# core FTS indexes. Anything else means a newer build owns the schema.
_KNOWN_FEATURE_TABLE_PREFIXES = ("lcm_rollup", "lcm_embedding", "lcm_chunk")

# The known opt-in feature families whose derived tables an interim build may
# have created in an EARLY variant (missing later-added columns/tables). Each is
# validated by its own final-shape verifier (resolved at call time in
# :func:`_family_verifier` — the verifiers are defined later in this module); a
# family that fails verification is a rebuildable derived cache that remediation
# drops so the feature's own marker-gated init recreates it in the final shape.
_INTERIM_FEATURE_FAMILIES: tuple[dict[str, str], ...] = (
    {
        "name": "temporal_rollup",
        "prefix": "lcm_rollup",
        "rebuild_hint": "derived rollup cache — rebuild via `/lcm rollups rebuild`",
    },
    {
        "name": "embedding",
        "prefix": "lcm_embedding",
        "rebuild_hint": "derived embedding cache — re-run `/lcm embed backfill --apply`",
    },
    {
        "name": "chunk",
        "prefix": "lcm_chunk",
        "rebuild_hint": "derived chunk cache — re-run `/lcm embed backfill --corpus chunks --apply`",
    },
)


def _family_verifier(prefix: str):
    """Return the final-shape verifier for a feature-family prefix, or ``None``
    when the family has no verifier (its early variants cannot be judged, so it
    is left untouched)."""
    if prefix == "lcm_rollup":
        return verify_temporal_rollup_schema
    if prefix == "lcm_embedding":
        return verify_embedding_schema
    if prefix == "lcm_chunk":
        return verify_chunk_schema
    return None


# Verifier findings that name an *extra* piece this build does not recognise —
# the signature of a genuinely-newer build, never of an early interim variant.
# An interim variant only ever OMITS later-added pieces (reported as ``table:`` /
# ``index:`` / ``column:`` / ``missing object:`` findings); a column the build
# has never heard of means a newer release owns the DB. Such a family must never
# be dropped, and its presence downgrades the whole DB to ``genuinely_newer``.
# (An early column that was later *renamed* surfaces as a collapsed
# ``malformed table:`` finding on the embedding/chunk verifiers — not as
# ``unexpected-column:`` — so it correctly stays on the safe early-variant path.)
_NEWER_BUILD_FINDING_PREFIXES = ("unexpected-column:",)


def _family_reports_newer_shape(findings: Iterable[str]) -> bool:
    """True when any verifier finding is an extra/unknown-piece (newer) signature."""
    return any(str(finding).startswith(_NEWER_BUILD_FINDING_PREFIXES) for finding in findings)


def _user_table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }


def classify_version_mismatch(conn: sqlite3.Connection) -> str:
    """Classify a DB whose stored ``schema_version`` exceeds ``SCHEMA_VERSION``.

    The numeric ``schema_version`` stamp only certifies the *core* schema shape;
    the internal shape of each opt-in feature family (temporal-rollup /
    embedding / chunk) is owned by that feature's own marker-gated init/verify,
    not by the numeric counter. So classification only inspects the core tables
    and the *names* of the extras:

    * :data:`VERSION_MISMATCH_INTERIM_STAMP` — the core v5 shape matches exactly
      AND every extra table matches a known feature-family prefix (or is an FTS5
      shadow table). This is the signature of an interim development build that
      stamped a numeric version it never migrated to; the feature tables may be
      early variants and are remediated separately (see
      :func:`remediate_interim_schema_stamp`).
    * :data:`VERSION_MISMATCH_GENUINELY_NEWER` — any core table/column is missing
      or unexpected, OR any extra table is not a known feature family. A newer
      build genuinely owns this DB.

    Never mutates schema; the safe default is ``genuinely_newer`` so an
    unrecognised shape is never re-stamped/downgraded.
    """
    try:
        tables = _user_table_names(conn)
    except sqlite3.DatabaseError:
        return VERSION_MISMATCH_GENUINELY_NEWER

    # Every v5 core table must be present.
    for table in (*_V5_CORE_TABLE_COLUMNS, *_V5_CORE_PRESENCE_ONLY):
        if table not in tables:
            return VERSION_MISMATCH_GENUINELY_NEWER

    # Core tables must match the v5 column contract exactly — a missing or an
    # unexpected column both mean this is not a v5-shaped DB.
    for table, expected in _V5_CORE_TABLE_COLUMNS.items():
        try:
            actual = {
                str(row[1])
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
        except sqlite3.DatabaseError:
            return VERSION_MISMATCH_GENUINELY_NEWER
        if actual != set(expected):
            return VERSION_MISMATCH_GENUINELY_NEWER

    # Any extra table must belong to a known feature family or be an FTS5 shadow.
    # The feature table's *internal* shape is intentionally NOT checked here — an
    # early-variant feature table is still an interim stamp, repaired on apply.
    allowed = set(_V5_CORE_TABLE_COLUMNS) | set(_V5_CORE_PRESENCE_ONLY)
    for fts in _V5_CORE_PRESENCE_ONLY:
        allowed.update(get_fts_shadow_table_names(fts))
    for table in tables:
        if table in allowed:
            continue
        if any(table.startswith(prefix) for prefix in _KNOWN_FEATURE_TABLE_PREFIXES):
            continue
        return VERSION_MISMATCH_GENUINELY_NEWER

    # A present feature-family table that carries an EXTRA column this build does
    # not recognise is a newer-build signature, not an early interim variant —
    # re-stamping (and dropping the family) would destroy real data. Early
    # variants only ever OMIT later-added pieces, which the verifiers report as
    # distinct "missing" findings; only an ``unexpected-column`` finding flips the
    # classification to genuinely-newer.
    for family in _INTERIM_FEATURE_FAMILIES:
        prefix = family["prefix"]
        if not any(table.startswith(prefix) for table in tables):
            continue
        verify = _family_verifier(prefix)
        if verify is None:
            continue
        try:
            findings = verify(conn)
        except sqlite3.DatabaseError:
            return VERSION_MISMATCH_GENUINELY_NEWER
        if _family_reports_newer_shape(findings):
            return VERSION_MISMATCH_GENUINELY_NEWER

    return VERSION_MISMATCH_INTERIM_STAMP


def _interim_family_drops(conn: sqlite3.Connection) -> list[dict[str, object]]:
    """Return per-family drop plans for early-variant feature tables.

    A feature family whose final-shape verifier reports NO findings is already
    current and is left untouched. A family that fails verification is a
    rebuildable derived cache: its tables (and any family-owned triggers, e.g.
    the rollup invalidation triggers on ``summary_nodes``) are scheduled for a
    drop so the feature's own init recreates the final shape on next use.
    """
    try:
        tables = _user_table_names(conn)
    except sqlite3.DatabaseError:
        return []
    plans: list[dict[str, object]] = []
    for family in _INTERIM_FEATURE_FAMILIES:
        prefix = family["prefix"]
        present = sorted(t for t in tables if t.startswith(prefix))
        if not present:
            continue
        verify = _family_verifier(prefix)
        if verify is None:
            continue
        findings = verify(conn)
        if not findings:
            # Verifier clean — the family is at the final shape; keep it.
            continue
        if _family_reports_newer_shape(findings):
            # An extra/unknown column is a newer-build signature, never an early
            # interim variant — never drop it (defense in depth; the DB is
            # already classified genuinely-newer and remediation has refused).
            continue
        triggers = sorted(
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE ?",
                (prefix + "%",),
            ).fetchall()
        )
        plans.append({
            "family": family["name"],
            "tables": present,
            "triggers": triggers,
            "rebuild_hint": family["rebuild_hint"],
        })
    return plans


def remediate_interim_schema_stamp(
    conn: sqlite3.Connection, *, apply: bool = False
) -> dict[str, object]:
    """Reset an interim-build schema stamp back to ``SCHEMA_VERSION``.

    Dry-run by default: classifies and reports the drop plan without mutating.
    With ``apply=True`` and an ``interim_stamp`` classification it (1) drops any
    early-variant feature-family tables that fail their own final-shape verifier
    — derived caches that each feature's marker-gated init rebuilds — and then
    (2) rewrites ``metadata.schema_version`` to ``SCHEMA_VERSION``. Refuses —
    never mutates — a ``genuinely_newer`` DB. Callers are responsible for taking
    a backup before ``apply`` (see
    :func:`hermes_lcm.maintenance.backup_database`).
    """
    current_version = read_existing_schema_version(conn)
    result: dict[str, object] = {
        "current_version": current_version,
        "target_version": SCHEMA_VERSION,
        "classification": None,
        "applied": False,
        "status": "noop",
        "drop_plan": [],
        "dropped_tables": [],
    }
    if current_version <= SCHEMA_VERSION:
        return result
    classification = classify_version_mismatch(conn)
    result["classification"] = classification
    if classification != VERSION_MISMATCH_INTERIM_STAMP:
        result["status"] = "refused"
        return result
    drop_plan = _interim_family_drops(conn)
    result["drop_plan"] = drop_plan
    if not apply:
        result["status"] = "dry-run"
        return result

    dropped: list[str] = []
    for plan in drop_plan:
        for trigger in plan["triggers"]:  # type: ignore[index]
            conn.execute(f"DROP TRIGGER IF EXISTS {quote_sql_identifier(str(trigger))}")
        for table in plan["tables"]:  # type: ignore[index]
            conn.execute(f"DROP TABLE IF EXISTS {quote_sql_identifier(str(table))}")
            dropped.append(str(table))
    set_schema_version(conn, SCHEMA_VERSION)
    conn.commit()
    result["dropped_tables"] = dropped
    result["applied"] = True
    result["status"] = "ok"
    logger.info(
        "Reset interim schema stamp from v%s to v%s (dropped %d early feature "
        "table(s): %s)",
        current_version,
        SCHEMA_VERSION,
        len(dropped),
        ", ".join(dropped) or "none",
    )
    return result


def ensure_migration_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_migration_state (
            step_name TEXT PRIMARY KEY,
            completed_at REAL NOT NULL
        )
        """
    )


def ensure_lifecycle_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_lifecycle_state (
            conversation_id TEXT PRIMARY KEY,
            current_session_id TEXT,
            last_finalized_session_id TEXT,
            current_frontier_store_id INTEGER NOT NULL DEFAULT 0,
            last_finalized_frontier_store_id INTEGER NOT NULL DEFAULT 0,
            debt_kind TEXT,
            debt_size_estimate INTEGER NOT NULL DEFAULT 0,
            current_bound_at REAL,
            last_finalized_at REAL,
            debt_updated_at REAL,
            last_maintenance_attempt_at REAL,
            last_rollover_at REAL,
            last_reset_at REAL,
            updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_current_session ON lcm_lifecycle_state(current_session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_last_finalized_session ON lcm_lifecycle_state(last_finalized_session_id)"
    )


def ensure_lifecycle_state_columns(conn: sqlite3.Connection) -> None:
    ensure_lifecycle_state_table(conn)
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(lcm_lifecycle_state)").fetchall()
    }
    add_column_if_missing(
        conn, columns, "debt_kind",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_kind TEXT",
    )
    add_column_if_missing(
        conn, columns, "debt_size_estimate",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_size_estimate INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(
        conn, columns, "debt_updated_at",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN debt_updated_at REAL",
    )
    add_column_if_missing(
        conn, columns, "last_maintenance_attempt_at",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN last_maintenance_attempt_at REAL",
    )
    add_column_if_missing(
        conn, columns, "last_rollover_at",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN last_rollover_at REAL",
    )
    add_column_if_missing(
        conn, columns, "last_reset_at",
        "ALTER TABLE lcm_lifecycle_state ADD COLUMN last_reset_at REAL",
    )


def ensure_message_origin_columns(conn: sqlite3.Connection) -> None:
    table_row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
    ).fetchone()
    if not table_row:
        return
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()
    }
    add_column_if_missing(
        conn, columns, "conversation_id",
        "ALTER TABLE messages ADD COLUMN conversation_id TEXT DEFAULT ''",
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_msg_conversation_session ON messages(conversation_id, session_id, store_id)"
    )


def ensure_temporal_rollup_tables(conn: sqlite3.Connection) -> None:
    """Lazily create the opt-in temporal-rollup feature tables.

    These tables are NOT part of the core numeric ``schema_version`` migration:
    they are created idempotently from :class:`RollupStore`'s own init on the
    enabled path (recorded as the named ``temporal_rollups_v1`` migration step),
    so a disabled install leaves ``schema_version`` untouched and stays readable
    by a base build. Keep every statement ``IF NOT EXISTS`` / additive so a
    concurrent enabled process can run this at the same time without racing.

    ``generation`` is an optimistic-concurrency counter bumped on every
    invalidation; ``lease_expires_at`` bounds a ``building`` row so a crashed
    build can be reclaimed. See :mod:`hermes_lcm.rollup_store`.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS lcm_rollups (
            rollup_id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_kind TEXT NOT NULL CHECK (period_kind IN ('day', 'week', 'month')),
            period_start TEXT NOT NULL,
            scope TEXT NOT NULL,
            summary TEXT,
            token_count INTEGER,
            status TEXT NOT NULL DEFAULT 'building'
                CHECK (status IN ('building', 'ready', 'stale', 'failed')),
            built_at TEXT,
            source_fingerprint TEXT,
            error TEXT,
            generation INTEGER NOT NULL DEFAULT 0,
            lease_expires_at TEXT,
            lease_nonce TEXT NOT NULL DEFAULT '',
            failed_at TEXT,
            UNIQUE(period_kind, period_start, scope)
        );

        CREATE TABLE IF NOT EXISTS lcm_rollup_sources (
            rollup_id INTEGER NOT NULL,
            node_id INTEGER NOT NULL,
            PRIMARY KEY(rollup_id, node_id)
        );

        -- The (rollup_id, node_id) PK cannot serve purge's node_id lookup; add a
        -- dedicated index so purging by deleted source node is not a full scan.
        CREATE INDEX IF NOT EXISTS idx_lcm_rollup_sources_node
            ON lcm_rollup_sources(node_id);

        CREATE TABLE IF NOT EXISTS lcm_rollup_invalidations (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id INTEGER,
            scope TEXT NOT NULL,
            covered_start REAL NOT NULL,
            covered_end REAL NOT NULL,
            next_day TEXT,
            operation TEXT NOT NULL CHECK(operation IN ('insert', 'delete', 'update')),
            created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
        );

        CREATE INDEX IF NOT EXISTS idx_lcm_rollup_invalidations_pending
            ON lcm_rollup_invalidations(event_id);

        CREATE INDEX IF NOT EXISTS idx_lcm_rollup_invalidations_scope_event
            ON lcm_rollup_invalidations(scope, event_id);

        CREATE INDEX IF NOT EXISTS idx_lcm_rollup_invalidations_scope_coverage
            ON lcm_rollup_invalidations(scope, covered_start, covered_end, event_id);
        """
    )
    # Backfill the generation/lease columns for a table created by an earlier
    # lazy revision that predates optimistic concurrency.
    rollup_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(lcm_rollups)").fetchall()
    }
    add_column_if_missing(
        conn, rollup_columns, "generation",
        "ALTER TABLE lcm_rollups ADD COLUMN generation INTEGER NOT NULL DEFAULT 0",
    )
    add_column_if_missing(
        conn, rollup_columns, "lease_expires_at",
        "ALTER TABLE lcm_rollups ADD COLUMN lease_expires_at TEXT",
    )
    add_column_if_missing(
        conn, rollup_columns, "lease_nonce",
        "ALTER TABLE lcm_rollups ADD COLUMN lease_nonce TEXT NOT NULL DEFAULT ''",
    )
    add_column_if_missing(
        conn, rollup_columns, "failed_at",
        "ALTER TABLE lcm_rollups ADD COLUMN failed_at TEXT",
    )
    for column, ddl in (
        ("summary", "ALTER TABLE lcm_rollups ADD COLUMN summary TEXT"),
        ("token_count", "ALTER TABLE lcm_rollups ADD COLUMN token_count INTEGER"),
        ("built_at", "ALTER TABLE lcm_rollups ADD COLUMN built_at TEXT"),
        ("source_fingerprint", "ALTER TABLE lcm_rollups ADD COLUMN source_fingerprint TEXT"),
        ("error", "ALTER TABLE lcm_rollups ADD COLUMN error TEXT"),
    ):
        add_column_if_missing(conn, rollup_columns, column, ddl)

    invalidation_columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(lcm_rollup_invalidations)"
        ).fetchall()
    }
    add_column_if_missing(
        conn,
        invalidation_columns,
        "next_day",
        "ALTER TABLE lcm_rollup_invalidations ADD COLUMN next_day TEXT",
    )
    conn.execute(
        """
        UPDATE lcm_rollup_invalidations
        SET covered_start = MIN(covered_start, covered_end),
            covered_end = MAX(covered_start, covered_end)
        WHERE covered_start > covered_end
        """
    )

    def ensure_index(name: str, create_sql: str) -> None:
        existing = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
            (name,),
        ).fetchone()
        def normalize(value: object) -> str:
            return " ".join(str(value or "").lower().split())
        if existing is not None and normalize(existing[0]) == normalize(create_sql):
            return
        if existing is not None:
            conn.execute(f"DROP INDEX {name}")
        conn.execute(create_sql)

    ensure_index(
        "idx_lcm_rollups_ready_period",
        "CREATE INDEX idx_lcm_rollups_ready_period "
        "ON lcm_rollups(scope, period_kind, period_start DESC) "
        "WHERE status = 'ready'",
    )
    ensure_index(
        "idx_lcm_rollups_pending",
        "CREATE INDEX idx_lcm_rollups_pending "
        "ON lcm_rollups(scope, status, failed_at, period_start) "
        "WHERE status IN ('stale', 'failed')",
    )
    ensure_index(
        "idx_lcm_rollups_expired_lease",
        "CREATE INDEX idx_lcm_rollups_expired_lease "
        "ON lcm_rollups(lease_expires_at, rollup_id) "
        "WHERE status = 'building'",
    )
    ensure_index(
        "idx_lcm_rollups_stale_day",
        "CREATE INDEX idx_lcm_rollups_stale_day "
        "ON lcm_rollups(scope, period_start) "
        "WHERE status = 'stale' AND period_kind = 'day'",
    )
    ensure_index(
        "idx_lcm_rollups_stale_aggregate",
        "CREATE INDEX idx_lcm_rollups_stale_aggregate "
        "ON lcm_rollups(scope, period_start, period_kind) "
        "WHERE status = 'stale' AND period_kind IN ('week', 'month')",
    )
    ensure_index(
        "idx_lcm_rollups_failed_day",
        "CREATE INDEX idx_lcm_rollups_failed_day "
        "ON lcm_rollups(scope, failed_at, period_start) "
        "WHERE status = 'failed' AND period_kind = 'day'",
    )
    ensure_index(
        "idx_lcm_rollups_failed_aggregate",
        "CREATE INDEX idx_lcm_rollups_failed_aggregate "
        "ON lcm_rollups(scope, failed_at, period_start, period_kind) "
        "WHERE status = 'failed' AND period_kind IN ('week', 'month')",
    )
    ensure_index(
        "idx_lcm_rollup_invalidations_pending",
        "CREATE INDEX idx_lcm_rollup_invalidations_pending "
        "ON lcm_rollup_invalidations(event_id)",
    )
    ensure_index(
        "idx_lcm_rollup_invalidations_scope_event",
        "CREATE INDEX idx_lcm_rollup_invalidations_scope_event "
        "ON lcm_rollup_invalidations(scope, event_id)",
    )
    ensure_index(
        "idx_lcm_rollup_invalidations_scope_coverage",
        "CREATE INDEX idx_lcm_rollup_invalidations_scope_coverage "
        "ON lcm_rollup_invalidations(scope, covered_start, covered_end, event_id)",
    )
    # Cursor state is keyed per (period_kind, scope) so multiple scopes sharing a
    # database do not clobber one another's build cursor. A pre-scope table (from
    # an earlier revision) only cached vestigial introspection data, so recreate
    # it rather than attempt an unsupported PRIMARY KEY migration.
    state_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lcm_rollup_state'"
    ).fetchone()
    if state_exists:
        state_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(lcm_rollup_state)").fetchall()
        }
        if "scope" not in state_columns:
            conn.execute("DROP TABLE lcm_rollup_state")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lcm_rollup_state (
            period_kind TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT '',
            last_build_cursor TEXT,
            last_built_at TEXT,
            PRIMARY KEY(period_kind, scope)
        )
        """
    )
    ensure_temporal_rollup_invalidation_triggers(conn)


def ensure_temporal_rollup_invalidation_triggers(conn: sqlite3.Connection) -> None:
    """Install transaction-coupled summary mutation outbox triggers when possible."""
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='summary_nodes'"
    ).fetchone() is None:
        return
    trigger_sql = {
        "lcm_rollup_node_insert": """
            CREATE TRIGGER lcm_rollup_node_insert
            AFTER INSERT ON summary_nodes BEGIN
                INSERT INTO lcm_rollup_invalidations(
                    node_id, scope, covered_start, covered_end, operation
                ) VALUES(
                    new.node_id, new.session_id,
                    MIN(COALESCE(new.earliest_at, new.created_at),
                        COALESCE(new.latest_at, new.created_at)),
                    MAX(COALESCE(new.earliest_at, new.created_at),
                        COALESCE(new.latest_at, new.created_at)), 'insert'
                );
            END
        """,
        "lcm_rollup_node_delete": """
            CREATE TRIGGER lcm_rollup_node_delete
            BEFORE DELETE ON summary_nodes BEGIN
                INSERT INTO lcm_rollup_invalidations(
                    node_id, scope, covered_start, covered_end, operation
                ) VALUES(
                    old.node_id, old.session_id,
                    MIN(COALESCE(old.earliest_at, old.created_at),
                        COALESCE(old.latest_at, old.created_at)),
                    MAX(COALESCE(old.earliest_at, old.created_at),
                        COALESCE(old.latest_at, old.created_at)), 'delete'
                );
            END
        """,
        "lcm_rollup_node_update": """
            CREATE TRIGGER lcm_rollup_node_update
            AFTER UPDATE OF session_id, depth, summary, token_count,
                            source_token_count, source_ids, source_type, created_at,
                            earliest_at, latest_at, expand_hint ON summary_nodes BEGIN
                INSERT INTO lcm_rollup_invalidations(
                    node_id, scope, covered_start, covered_end, operation
                ) VALUES(
                    old.node_id, old.session_id,
                    MIN(COALESCE(old.earliest_at, old.created_at),
                        COALESCE(old.latest_at, old.created_at)),
                    MAX(COALESCE(old.earliest_at, old.created_at),
                        COALESCE(old.latest_at, old.created_at)), 'update'
                );
                INSERT INTO lcm_rollup_invalidations(
                    node_id, scope, covered_start, covered_end, operation
                ) VALUES(
                    new.node_id, new.session_id,
                    MIN(COALESCE(new.earliest_at, new.created_at),
                        COALESCE(new.latest_at, new.created_at)),
                    MAX(COALESCE(new.earliest_at, new.created_at),
                        COALESCE(new.latest_at, new.created_at)), 'update'
                );
            END
        """,
    }

    def normalized(sql: object) -> str:
        return re.sub(r"\s+", "", str(sql or "").lower()).rstrip(";")

    for trigger_name, expected_sql in trigger_sql.items():
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
            (trigger_name,),
        ).fetchone()
        if row is not None and normalized(row[0]) != normalized(expected_sql):
            conn.execute(f"DROP TRIGGER {trigger_name}")
            row = None
        if row is None:
            conn.execute(expected_sql)


REQUIRED_TEMPORAL_ROLLUP_TABLES = (
    "lcm_rollups",
    "lcm_rollup_sources",
    "lcm_rollup_state",
    "lcm_rollup_invalidations",
)
REQUIRED_TEMPORAL_ROLLUP_INDEXES = (
    "idx_lcm_rollups_ready_period",
    "idx_lcm_rollups_pending",
    "idx_lcm_rollups_expired_lease",
    "idx_lcm_rollups_stale_day",
    "idx_lcm_rollups_stale_aggregate",
    "idx_lcm_rollups_failed_day",
    "idx_lcm_rollups_failed_aggregate",
    "idx_lcm_rollup_sources_node",
    "idx_lcm_rollup_invalidations_pending",
    "idx_lcm_rollup_invalidations_scope_event",
    "idx_lcm_rollup_invalidations_scope_coverage",
)


def verify_temporal_rollup_schema(conn: sqlite3.Connection) -> list[str]:
    """Return the temporal-rollup tables/indexes that are absent.

    The named ``temporal_rollups_v1`` migration marker records only that the
    feature was once enabled; it is NOT proof the tables still exist. A marker
    can outlive its tables (a crash mid-create, or a DB whose rollup tables were
    dropped), so callers must verify the objects themselves rather than trusting
    the marker (maintainer #387 A3). Returns a list of ``"table:<name>"`` /
    ``"index:<name>"`` entries for every required object that is missing; an
    empty list means the schema is present and consistent.
    """
    missing: list[str] = []
    for name in REQUIRED_TEMPORAL_ROLLUP_TABLES:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (name,),
        ).fetchone()
        if row is None:
            missing.append(f"table:{name}")
    for name in REQUIRED_TEMPORAL_ROLLUP_INDEXES:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name = ?",
            (name,),
        ).fetchone()
        if row is None:
            missing.append(f"index:{name}")
    expected_column_shapes = {
        "lcm_rollups": {
            "rollup_id": ("INTEGER", 0, None, 1),
            "period_kind": ("TEXT", 1, None, 0),
            "period_start": ("TEXT", 1, None, 0),
            "scope": ("TEXT", 1, None, 0),
            "summary": ("TEXT", 0, None, 0),
            "token_count": ("INTEGER", 0, None, 0),
            "status": ("TEXT", 1, "'building'", 0),
            "built_at": ("TEXT", 0, None, 0),
            "source_fingerprint": ("TEXT", 0, None, 0),
            "error": ("TEXT", 0, None, 0),
            "generation": ("INTEGER", 1, "0", 0),
            "lease_expires_at": ("TEXT", 0, None, 0),
            "lease_nonce": ("TEXT", 1, "''", 0),
            "failed_at": ("TEXT", 0, None, 0),
        },
        "lcm_rollup_sources": {
            "rollup_id": ("INTEGER", 1, None, 1),
            "node_id": ("INTEGER", 1, None, 2),
        },
        "lcm_rollup_state": {
            "period_kind": ("TEXT", 1, None, 1),
            "scope": ("TEXT", 1, "''", 2),
            "last_build_cursor": ("TEXT", 0, None, 0),
            "last_built_at": ("TEXT", 0, None, 0),
        },
        "lcm_rollup_invalidations": {
            "event_id": ("INTEGER", 0, None, 1),
            "node_id": ("INTEGER", 0, None, 0),
            "scope": ("TEXT", 1, None, 0),
            "covered_start": ("REAL", 1, None, 0),
            "covered_end": ("REAL", 1, None, 0),
            "next_day": ("TEXT", 0, None, 0),
            "operation": ("TEXT", 1, None, 0),
            "created_at": ("REAL", 1, "strftime('%s','now')", 0),
        },
    }
    for table, expected in expected_column_shapes.items():
        if f"table:{table}" in missing:
            continue
        actual = {
            str(row[1]): (
                str(row[2]).upper(), int(row[3] or 0),
                None if row[4] is None else str(row[4]), int(row[5] or 0),
            )
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for column in sorted(expected.keys() - actual.keys()):
            missing.append(f"column:{table}.{column}")
        for column in sorted(actual.keys() - expected.keys()):
            missing.append(f"unexpected-column:{table}.{column}")
        for column, shape in expected.items():
            if column in actual and actual[column] != shape:
                missing.append(f"column-shape:{table}.{column}")

    # These keys are correctness-bearing, not optional query accelerators.
    for table, expected_pk in (
        ("lcm_rollup_sources", ["rollup_id", "node_id"]),
        ("lcm_rollup_state", ["period_kind", "scope"]),
    ):
        if f"table:{table}" in missing:
            continue
        pk = [
            str(row[1])
            for row in sorted(
                conn.execute(f"PRAGMA table_info({table})").fetchall(),
                key=lambda row: int(row[5] or 0),
            )
            if int(row[5] or 0) > 0
        ]
        if pk != expected_pk:
            missing.append(f"primary-key:{table}")
    if "table:lcm_rollups" not in missing:
        unique_ok = False
        for index in conn.execute("PRAGMA index_list(lcm_rollups)").fetchall():
            if not int(index[2] or 0):
                continue
            columns = [
                str(row[2])
                for row in conn.execute(f"PRAGMA index_info({index[1]})").fetchall()
            ]
            if columns == ["period_kind", "period_start", "scope"]:
                unique_ok = True
                break
        if not unique_ok:
            missing.append("unique:lcm_rollups.period_kind,period_start,scope")

    table_checks = {
        "lcm_rollups": (
            "check(period_kindin('day','week','month'))",
            "check(statusin('building','ready','stale','failed'))",
        ),
        "lcm_rollup_invalidations": (
            "check(operationin('insert','delete','update'))",
        ),
    }
    for table, snippets in table_checks.items():
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        normalized = re.sub(r"\s+", "", str(row[0] if row else "").lower())
        for snippet in snippets:
            if snippet not in normalized:
                missing.append(f"check:{table}")
                break

    expected_indexes = {
        "idx_lcm_rollups_ready_period": (
            "lcm_rollups",
            [("scope", 0), ("period_kind", 0), ("period_start", 1)],
            "where status = 'ready'",
        ),
        "idx_lcm_rollups_pending": (
            "lcm_rollups",
            [("scope", 0), ("status", 0), ("failed_at", 0), ("period_start", 0)],
            "where status in ('stale', 'failed')",
        ),
        "idx_lcm_rollups_expired_lease": (
            "lcm_rollups",
            [("lease_expires_at", 0), ("rollup_id", 0)],
            "where status = 'building'",
        ),
        "idx_lcm_rollups_stale_day": (
            "lcm_rollups",
            [("scope", 0), ("period_start", 0)],
            "where status = 'stale' and period_kind = 'day'",
        ),
        "idx_lcm_rollups_stale_aggregate": (
            "lcm_rollups",
            [("scope", 0), ("period_start", 0), ("period_kind", 0)],
            "where status = 'stale' and period_kind in ('week', 'month')",
        ),
        "idx_lcm_rollups_failed_day": (
            "lcm_rollups",
            [("scope", 0), ("failed_at", 0), ("period_start", 0)],
            "where status = 'failed' and period_kind = 'day'",
        ),
        "idx_lcm_rollups_failed_aggregate": (
            "lcm_rollups",
            [
                ("scope", 0), ("failed_at", 0), ("period_start", 0),
                ("period_kind", 0),
            ],
            "where status = 'failed' and period_kind in ('week', 'month')",
        ),
        "idx_lcm_rollup_sources_node": (
            "lcm_rollup_sources", [("node_id", 0)], ""
        ),
        "idx_lcm_rollup_invalidations_pending": (
            "lcm_rollup_invalidations", [("event_id", 0)], ""
        ),
        "idx_lcm_rollup_invalidations_scope_event": (
            "lcm_rollup_invalidations",
            [("scope", 0), ("event_id", 0)], ""
        ),
        "idx_lcm_rollup_invalidations_scope_coverage": (
            "lcm_rollup_invalidations",
            [
                ("scope", 0), ("covered_start", 0), ("covered_end", 0),
                ("event_id", 0),
            ], ""
        ),
    }
    def normalized_predicate(sql: object) -> str:
        match = re.search(r"\bwhere\b(.+)$", str(sql or ""), re.IGNORECASE | re.DOTALL)
        if match is None:
            return ""
        return re.sub(r"\s+", "", match.group(1).lower()).rstrip(";")

    for name, (table, columns, predicate) in expected_indexes.items():
        if f"index:{name}" in missing:
            continue
        metadata = conn.execute(
            "SELECT tbl_name, sql FROM sqlite_master WHERE type='index' AND name=?",
            (name,),
        ).fetchone()
        index_list_row = next(
            (
                row
                for row in conn.execute(f"PRAGMA index_list({table})").fetchall()
                if str(row[1]) == name
            ),
            None,
        )
        actual_columns = [
            (str(row[2]), int(row[3] or 0))
            for row in conn.execute(f"PRAGMA index_xinfo({name})").fetchall()
            if int(row[5] or 0) == 1 and row[2] is not None
        ]
        expected_predicate = normalized_predicate(predicate)
        actual_predicate = normalized_predicate(metadata[1] if metadata else "")
        expected_partial = int(bool(expected_predicate))
        if (
            metadata is None
            or str(metadata[0]) != table
            or index_list_row is None
            or int(index_list_row[2] or 0) != 0
            or int(index_list_row[4] or 0) != expected_partial
            or actual_columns != columns
            or actual_predicate != expected_predicate
        ):
            missing.append(f"index-shape:{name}")

    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='summary_nodes'"
    ).fetchone() is not None:
        temp = sqlite3.connect(":memory:")
        try:
            temp.executescript(
                """
                CREATE TABLE summary_nodes(
                    node_id INTEGER, session_id TEXT, depth INTEGER, summary TEXT,
                    token_count INTEGER, source_token_count INTEGER,
                    source_ids TEXT, source_type TEXT, created_at REAL,
                    earliest_at REAL, latest_at REAL, expand_hint TEXT
                );
                CREATE TABLE lcm_rollup_invalidations(
                    node_id INTEGER, scope TEXT, covered_start REAL,
                    covered_end REAL, operation TEXT
                );
                """
            )
            ensure_temporal_rollup_invalidation_triggers(temp)
            expected_triggers = {
                str(row[0]): re.sub(r"\s+", "", str(row[1]).lower()).rstrip(";")
                for row in temp.execute(
                    "SELECT name, sql FROM sqlite_master WHERE type='trigger'"
                ).fetchall()
            }
        finally:
            temp.close()
        for name, expected in expected_triggers.items():
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)
            ).fetchone()
            actual = re.sub(
                r"\s+", "", str(row[0] if row else "").lower()
            ).rstrip(";")
            if actual != expected:
                missing.append(f"trigger-shape:{name}")
    return missing
def ensure_embedding_tables(conn: sqlite3.Connection) -> None:
    """Create the opt-in embedding tables idempotently.

    These tables are NOT part of the core ``schema_version`` ladder. They are
    created only when embeddings are actually used (VectorStore construction),
    so an install with embeddings disabled never materializes them and stays at
    schema_version 5, openable by a base build.

    Profiles and vectors are keyed on a canonical *identity* — the sha256 of
    ``(provider, model_name, revision, dim, dtype, byteorder, task)`` — rather
    than on ``model_name`` alone. Re-registering the same model under a
    different provider is therefore a new profile row (no metadata clobber),
    and switching config back to a previously-registered identity reactivates
    that profile with its vectors still valid. ``data_version`` is a durable
    per-identity counter bumped inside every vector write/delete transaction so
    the in-process NumPy matrix cache cannot serve cross-process-stale results.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS lcm_embedding_profile (
            identity_hash TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model_name TEXT NOT NULL,
            revision TEXT NOT NULL DEFAULT '',
            dim INTEGER CHECK(dim BETWEEN 1 AND 4096),
            dtype TEXT NOT NULL DEFAULT 'float32',
            byteorder TEXT NOT NULL DEFAULT 'little',
            task TEXT NOT NULL DEFAULT 'summary',
            registered_at TEXT,
            active INTEGER DEFAULT 1,
            archived_at TEXT NULL,
            data_version INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_lcm_embedding_profile_model
            ON lcm_embedding_profile(model_name, provider);

        CREATE TABLE IF NOT EXISTS lcm_embedding_meta (
            embedded_id TEXT,
            embedded_kind TEXT CHECK(embedded_kind IN ('summary')),
            identity_hash TEXT,
            embedded_at TEXT,
            source_token_count INTEGER,
            archived INTEGER DEFAULT 0,
            PRIMARY KEY(embedded_id, embedded_kind, identity_hash)
        );

        CREATE INDEX IF NOT EXISTS idx_lcm_embedding_meta_identity_embedded_at
            ON lcm_embedding_meta(identity_hash, embedded_at DESC)
            WHERE archived = 0;

        CREATE TABLE IF NOT EXISTS lcm_embedding_vectors (
            embedded_id TEXT,
            identity_hash TEXT,
            vec BLOB NOT NULL,
            PRIMARY KEY(embedded_id, identity_hash)
        );

        CREATE TABLE IF NOT EXISTS lcm_embedding_binary (
            embedded_id TEXT,
            identity_hash TEXT,
            bits BLOB NOT NULL,
            PRIMARY KEY(embedded_id, identity_hash)
        );
        """
    )


# The tables and indexes ``ensure_embedding_tables`` is responsible for. Used to
# VERIFY the schema on VectorStore init rather than trusting the ``embeddings_v1``
# marker alone: the named marker can be present while a table/index is absent
# (e.g. a table was dropped after the marker was written), so init re-ensures and
# confirms these objects exist rather than assuming the marker implies them.
_REQUIRED_EMBEDDING_TABLES = (
    "lcm_embedding_profile",
    "lcm_embedding_meta",
    "lcm_embedding_vectors",
    # Sign-bit prescreen for the two-stage KNN. Present on every embedding
    # schema (empty for float32 identities), populated only for int8 identities.
    "lcm_embedding_binary",
)
_REQUIRED_EMBEDDING_INDEXES = (
    "idx_lcm_embedding_profile_model",
    "idx_lcm_embedding_meta_identity_embedded_at",
)

_EMBEDDING_TABLE_SHAPES: dict[
    str, tuple[tuple[str, str, int, int, str | None], ...]
] = {
    "lcm_embedding_profile": (
        ("identity_hash", "TEXT", 0, 1, None),
        ("provider", "TEXT", 1, 0, None),
        ("model_name", "TEXT", 1, 0, None),
        ("revision", "TEXT", 1, 0, "''"),
        ("dim", "INTEGER", 0, 0, None),
        ("dtype", "TEXT", 1, 0, "'float32'"),
        ("byteorder", "TEXT", 1, 0, "'little'"),
        ("task", "TEXT", 1, 0, "'summary'"),
        ("registered_at", "TEXT", 0, 0, None),
        ("active", "INTEGER", 0, 0, "1"),
        ("archived_at", "TEXT", 0, 0, None),
        ("data_version", "INTEGER", 1, 0, "0"),
    ),
    "lcm_embedding_meta": (
        ("embedded_id", "TEXT", 0, 1, None),
        ("embedded_kind", "TEXT", 0, 2, None),
        ("identity_hash", "TEXT", 0, 3, None),
        ("embedded_at", "TEXT", 0, 0, None),
        ("source_token_count", "INTEGER", 0, 0, None),
        ("archived", "INTEGER", 0, 0, "0"),
    ),
    "lcm_embedding_vectors": (
        ("embedded_id", "TEXT", 0, 1, None),
        ("identity_hash", "TEXT", 0, 2, None),
        ("vec", "BLOB", 1, 0, None),
    ),
    "lcm_embedding_binary": (
        ("embedded_id", "TEXT", 0, 1, None),
        ("identity_hash", "TEXT", 0, 2, None),
        ("bits", "BLOB", 1, 0, None),
    ),
}

_EMBEDDING_INDEX_SHAPES: dict[
    str, tuple[str, tuple[tuple[str, int], ...], str | None]
] = {
    "idx_lcm_embedding_profile_model": (
        "lcm_embedding_profile",
        (("model_name", 0), ("provider", 0)),
        None,
    ),
    "idx_lcm_embedding_meta_identity_embedded_at": (
        "lcm_embedding_meta",
        (("identity_hash", 0), ("embedded_at", 1)),
        "archived=0",
    ),
}

_EMBEDDING_CHECKS = {
    "lcm_embedding_profile": {"dimbetween1and4096"},
    "lcm_embedding_meta": {"embedded_kindin('summary')"},
    "lcm_embedding_vectors": set(),
    "lcm_embedding_binary": set(),
}


def _sql_check_expressions(sql: str) -> set[str]:
    """Extract normalized CHECK bodies while respecting nested parentheses."""
    lowered = sql.lower()
    expressions: set[str] = set()
    offset = 0
    while True:
        start = lowered.find("check(", offset)
        if start < 0:
            return expressions
        body_start = start + len("check(")
        depth = 1
        cursor = body_start
        while cursor < len(lowered) and depth:
            if lowered[cursor] == "(":
                depth += 1
            elif lowered[cursor] == ")":
                depth -= 1
            cursor += 1
        if depth:
            return {"<malformed>"}
        expressions.add(re.sub(r"\s+", "", lowered[body_start:cursor - 1]))
        offset = cursor


def embedding_schema_missing(conn: sqlite3.Connection) -> set[str]:
    """Return the names of required embedding tables/indexes that do not exist.

    An empty set means the embedding schema is fully materialized. A non-empty
    set means the ``embeddings_v1`` marker cannot be trusted on its own and
    ``ensure_embedding_tables`` must (re-)run to repair the gap.
    """
    present = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
        ).fetchall()
    }
    required = set(_REQUIRED_EMBEDDING_TABLES) | set(_REQUIRED_EMBEDDING_INDEXES)
    return required - present


def verify_embedding_schema(conn: sqlite3.Connection) -> list[str]:
    """Return structural embedding-schema errors, not just missing names.

    Feature state is derived and rebuildable, but silently accepting a table
    with the right name and the wrong columns lets profile activation or vector
    publication fail halfway through.  Missing objects are repaired by
    ``ensure_embedding_tables``; incompatible same-name objects are rejected
    before the named migration marker is published.
    """
    errors = [f"missing object: {name}" for name in sorted(embedding_schema_missing(conn))]
    if errors:
        return errors

    for table, expected in _EMBEDDING_TABLE_SHAPES.items():
        actual = tuple(
            (
                str(row[1]),
                str(row[2]).upper(),
                int(row[3]),
                int(row[5]),
                None if row[4] is None else str(row[4]).lower(),
            )
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        )
        if actual != expected:
            errors.append(f"malformed table: {table}")

    # CHECK expressions are not exposed by PRAGMA table_info, so fingerprint
    # their normalized expressions exactly rather than looking for a loose
    # substring. PK order/nullability/types/defaults were checked above.
    table_sql = {
        str(row[0]): " ".join(str(row[1] or "").lower().split())
        for row in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' "
            "AND name IN (%s)" % ",".join("?" for _ in _REQUIRED_EMBEDDING_TABLES),
            _REQUIRED_EMBEDDING_TABLES,
        ).fetchall()
    }
    for table, expected_checks in _EMBEDDING_CHECKS.items():
        sql = table_sql.get(table, "")
        actual_checks = _sql_check_expressions(sql)
        if actual_checks != expected_checks:
            errors.append(f"malformed constraints: {table}")

    for index, (table, expected_columns, predicate) in _EMBEDDING_INDEX_SHAPES.items():
        index_shape = tuple(
            (
                None if row[2] is None else str(row[2]),
                int(row[3]),
                str(row[4]).upper(),
                int(row[5]),
            )
            for row in conn.execute(f"PRAGMA index_xinfo({index})").fetchall()
        )
        expected_shape = tuple(
            (column, desc, "BINARY", 1) for column, desc in expected_columns
        ) + ((None, 0, "BINARY", 0),)
        index_list = {
            str(row[1]): (int(row[2]), int(row[4]))
            for row in conn.execute(f"PRAGMA index_list({table})").fetchall()
        }
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        sql = " ".join(str(row[0] or "").lower().split()) if row else ""
        actual_predicate = None
        if " where " in sql:
            actual_predicate = re.sub(r"\s+", "", sql.split(" where ", 1)[1])
        unique, partial = index_list.get(index, (-1, -1))
        if (
            index_shape != expected_shape
            or unique != 0
            or partial != int(predicate is not None)
            or actual_predicate != predicate
        ):
            errors.append(f"malformed index: {index}")
    return sorted(set(errors))


def ensure_chunk_tables(conn: sqlite3.Connection) -> None:
    """Create the opt-in raw-history chunk tables idempotently.

    Like the embedding tables, these are NOT part of the core
    ``schema_version`` ladder — they are materialized only when the chunk
    corpus is actually used (a chunk-corpus VectorStore is constructed), so an
    install that never runs ``embed backfill --corpus chunks`` stays at
    schema_version 5 with none of them and remains openable by a base build.

    Chunk profiles live in the SHARED ``lcm_embedding_profile`` table under
    ``task='chunk'`` (coexisting with summary profiles), so ``ensure_embedding_tables``
    must have run first. These two tables hold only the per-chunk metadata and
    vectors, keyed on ``(chunk_id, identity_hash)`` where ``chunk_id`` is
    ``store_id:chunk_index``.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS lcm_chunk_meta (
            chunk_id TEXT,
            identity_hash TEXT,
            store_id INTEGER,
            chunk_index INTEGER,
            char_start INTEGER,
            char_end INTEGER,
            token_estimate INTEGER,
            embedded_at TEXT,
            archived INTEGER DEFAULT 0,
            PRIMARY KEY(chunk_id, identity_hash)
        );

        CREATE INDEX IF NOT EXISTS idx_lcm_chunk_meta_identity_embedded_at
            ON lcm_chunk_meta(identity_hash, embedded_at DESC)
            WHERE archived = 0;

        CREATE INDEX IF NOT EXISTS idx_lcm_chunk_meta_store
            ON lcm_chunk_meta(store_id);

        CREATE TABLE IF NOT EXISTS lcm_chunk_vectors (
            chunk_id TEXT,
            identity_hash TEXT,
            vec BLOB NOT NULL,
            PRIMARY KEY(chunk_id, identity_hash)
        );

        CREATE TABLE IF NOT EXISTS lcm_chunk_binary (
            chunk_id TEXT,
            identity_hash TEXT,
            bits BLOB NOT NULL,
            PRIMARY KEY(chunk_id, identity_hash)
        );
        """
    )


# The tables and indexes ``ensure_chunk_tables`` owns. Verified on chunk-corpus
# VectorStore init rather than trusting the ``chunk_vectors_v1`` marker alone —
# the same discipline as the embedding schema: a set marker over a dropped table
# is repaired rather than believed.
_REQUIRED_CHUNK_TABLES = (
    "lcm_chunk_meta",
    "lcm_chunk_vectors",
    # Sign-bit prescreen for the two-stage chunk KNN. Present on every chunk
    # schema (empty for float32 identities), populated only for int8 identities.
    "lcm_chunk_binary",
)
_REQUIRED_CHUNK_INDEXES = (
    "idx_lcm_chunk_meta_identity_embedded_at",
    "idx_lcm_chunk_meta_store",
)

_CHUNK_TABLE_SHAPES: dict[
    str, tuple[tuple[str, str, int, int, str | None], ...]
] = {
    "lcm_chunk_meta": (
        ("chunk_id", "TEXT", 0, 1, None),
        ("identity_hash", "TEXT", 0, 2, None),
        ("store_id", "INTEGER", 0, 0, None),
        ("chunk_index", "INTEGER", 0, 0, None),
        ("char_start", "INTEGER", 0, 0, None),
        ("char_end", "INTEGER", 0, 0, None),
        ("token_estimate", "INTEGER", 0, 0, None),
        ("embedded_at", "TEXT", 0, 0, None),
        ("archived", "INTEGER", 0, 0, "0"),
    ),
    "lcm_chunk_vectors": (
        ("chunk_id", "TEXT", 0, 1, None),
        ("identity_hash", "TEXT", 0, 2, None),
        ("vec", "BLOB", 1, 0, None),
    ),
    "lcm_chunk_binary": (
        ("chunk_id", "TEXT", 0, 1, None),
        ("identity_hash", "TEXT", 0, 2, None),
        ("bits", "BLOB", 1, 0, None),
    ),
}

_CHUNK_INDEX_SHAPES: dict[
    str, tuple[str, tuple[tuple[str, int], ...], str | None]
] = {
    "idx_lcm_chunk_meta_identity_embedded_at": (
        "lcm_chunk_meta",
        (("identity_hash", 0), ("embedded_at", 1)),
        "archived=0",
    ),
    "idx_lcm_chunk_meta_store": (
        "lcm_chunk_meta",
        (("store_id", 0),),
        None,
    ),
}


def chunk_schema_missing(conn: sqlite3.Connection) -> set[str]:
    """Return names of required chunk tables/indexes that do not exist.

    An empty set means the chunk schema is fully materialized; a non-empty set
    means the ``chunk_vectors_v1`` marker cannot be trusted on its own and
    ``ensure_chunk_tables`` must (re-)run to repair the gap.
    """
    present = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
        ).fetchall()
    }
    required = set(_REQUIRED_CHUNK_TABLES) | set(_REQUIRED_CHUNK_INDEXES)
    return required - present


def verify_chunk_schema(conn: sqlite3.Connection) -> list[str]:
    """Return structural chunk-schema errors, mirroring ``verify_embedding_schema``.

    Missing objects are repaired by ``ensure_chunk_tables``; incompatible
    same-name objects are rejected before the ``chunk_vectors_v1`` marker is
    published so a wrong-shaped table cannot masquerade as a valid corpus.
    """
    errors = [f"missing object: {name}" for name in sorted(chunk_schema_missing(conn))]
    if errors:
        return errors

    for table, expected in _CHUNK_TABLE_SHAPES.items():
        actual = tuple(
            (
                str(row[1]),
                str(row[2]).upper(),
                int(row[3]),
                int(row[5]),
                None if row[4] is None else str(row[4]).lower(),
            )
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        )
        if actual != expected:
            errors.append(f"malformed table: {table}")

    for index, (table, expected_columns, predicate) in _CHUNK_INDEX_SHAPES.items():
        index_shape = tuple(
            (
                None if row[2] is None else str(row[2]),
                int(row[3]),
                str(row[4]).upper(),
                int(row[5]),
            )
            for row in conn.execute(f"PRAGMA index_xinfo({index})").fetchall()
        )
        expected_shape = tuple(
            (column, desc, "BINARY", 1) for column, desc in expected_columns
        ) + ((None, 0, "BINARY", 0),)
        index_list = {
            str(row[1]): (int(row[2]), int(row[4]))
            for row in conn.execute(f"PRAGMA index_list({table})").fetchall()
        }
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        sql = " ".join(str(row[0] or "").lower().split()) if row else ""
        actual_predicate = None
        if " where " in sql:
            actual_predicate = re.sub(r"\s+", "", sql.split(" where ", 1)[1])
        unique, partial = index_list.get(index, (-1, -1))
        if (
            index_shape != expected_shape
            or unique != 0
            or partial != int(predicate is not None)
            or actual_predicate != predicate
        ):
            errors.append(f"malformed index: {index}")
    return sorted(set(errors))


def mark_migration_step_complete(conn: sqlite3.Connection, step_name: str) -> None:
    ensure_migration_state_table(conn)
    conn.execute(
        """
        INSERT INTO lcm_migration_state(step_name, completed_at)
        VALUES(?, strftime('%s','now'))
        ON CONFLICT(step_name) DO UPDATE SET completed_at = excluded.completed_at
        """,
        (step_name,),
    )


def set_schema_version(conn: sqlite3.Connection, version: int = SCHEMA_VERSION) -> None:
    ensure_metadata_table(conn)
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(version),),
    )


def get_existing_table_names(conn: sqlite3.Connection, names: Iterable[str]) -> set[str]:
    existing: set[str] = set()
    for name in names:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (name,),
        ).fetchone()
        if row and row[0]:
            existing.add(row[0])
    return existing


def _database_path_for_connection(conn: sqlite3.Connection | None, fallback: str = "") -> str:
    if conn is None:
        return fallback
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except sqlite3.DatabaseError:
        return fallback
    for row in rows:
        if len(row) >= 3 and row[1] == "main" and row[2]:
            return str(row[2])
    return fallback


def inspect_lcm_schema_health(
    conn: sqlite3.Connection | None,
    *,
    database_path: str = "",
    required_tables: Iterable[str] = REQUIRED_CORE_TABLES,
) -> dict[str, object]:
    """Return read-only health metadata for the core hermes-lcm SQLite schema."""
    required = tuple(required_tables)
    resolved_path = _database_path_for_connection(conn, database_path)
    detail: dict[str, object] = {
        "database_path": resolved_path,
        "required_tables": list(required),
        "existing_tables": [],
        "missing_tables": [],
    }
    if conn is None:
        detail["error"] = "LCM store connection is not initialized"
        return detail

    try:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        detail["error"] = str(exc)
        return detail

    existing = sorted(str(row[0]) for row in rows if row and row[0])
    existing_set = set(existing)
    missing = [name for name in required if name not in existing_set]
    detail["existing_tables"] = existing
    detail["missing_tables"] = missing
    return detail


def get_fts_shadow_table_names(table_name: str) -> list[str]:
    return [
        f"{table_name}_data",
        f"{table_name}_idx",
        f"{table_name}_docsize",
        f"{table_name}_config",
    ]


def quote_sql_identifier(identifier: str) -> str:
    if not identifier or not identifier.replace("_", "a").isalnum() or identifier[0].isdigit():
        raise ValueError(f"invalid SQL identifier: {identifier}")
    return f'"{identifier}"'


def _fts_needs_rebuild_structural(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    shadow_tables = get_fts_shadow_table_names(spec.table_name)
    existing_tables = get_existing_table_names(conn, [spec.table_name, *shadow_tables])
    if spec.table_name not in existing_tables:
        return True
    if any(name not in existing_tables for name in shadow_tables):
        return True

    try:
        info = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?",
            (spec.table_name,),
        ).fetchone()
        sql = (info[0] if info else "") or ""
        normalized = sql.lower()
        if "virtual table" not in normalized or "using fts5" not in normalized:
            return True

        columns = conn.execute(
            f"PRAGMA table_info({quote_sql_identifier(spec.table_name)})"
        ).fetchall()
        column_names = {row[1] for row in columns if len(row) > 1}
        if spec.indexed_column not in column_names:
            return True

        content_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_sql_identifier(spec.content_table)}"
        ).fetchone()[0]
        # For an external-content FTS5 table, ``COUNT(*) FROM <fts>`` reads
        # through to the content table (so it can never reveal a lagging index)
        # and is O(index size). The ``<fts>_docsize`` shadow table holds the
        # true indexed-document count and is a cheap ordinary-table count. Its
        # existence is already guaranteed by the shadow-table check above.
        docsize_table = f"{spec.table_name}_docsize"
        fts_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_sql_identifier(docsize_table)}"
        ).fetchone()[0]
        if int(content_count or 0) != int(fts_count or 0):
            return True
    except sqlite3.DatabaseError:
        return True

    return False


INTEGRITY_CHECK_INTERVAL_ENV = "LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS"
DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS = 24.0


def _integrity_check_interval_hours() -> float:
    """Hours between startup FTS deep integrity-checks.

    ``0`` checks on every startup (previous behavior); a negative value never
    checks on startup (relies on structural checks + LIKE fallback + doctor).
    """
    raw = os.environ.get(INTEGRITY_CHECK_INTERVAL_ENV)
    if raw is None:
        return DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
    if not math.isfinite(value):
        # nan/inf would suppress startup checks indefinitely once a marker
        # exists; treat non-finite values as invalid.
        return DEFAULT_INTEGRITY_CHECK_INTERVAL_HOURS
    return value


def _integrity_marker_key(spec: ExternalContentFtsSpec) -> str:
    return f"fts_integrity_checked_at:{spec.table_name}"


def _load_integrity_checked_at(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec
) -> float | None:
    ensure_metadata_table(conn)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (_integrity_marker_key(spec),),
    ).fetchone()
    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return None


def _record_integrity_checked(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> None:
    ensure_metadata_table(conn)
    current = time.time() if now is None else now
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (_integrity_marker_key(spec), str(current)),
    )


def _should_run_integrity_check(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> bool:
    hours = _integrity_check_interval_hours()
    if hours == 0:
        return True
    if hours < 0:
        return False
    last = _load_integrity_checked_at(conn, spec)
    if last is None:
        return True
    current = time.time() if now is None else now
    return (current - last) >= hours * 3600.0


# --- Non-blocking startup integrity scan (issue #6 / #235) -----------------
#
# Even throttled, the O(index-size) FTS5 deep integrity-check still blocked the
# bind on every cache-miss (first bind + each interval expiry): ~2min on a cold
# production DB. When a deep check is due on the startup path we now run only the
# cheap structural check synchronously and dispatch the deep scan to a daemon
# thread that opens its OWN sqlite connection (never the store's — that
# connection is not safe to drive from another thread). The background scan does
# NOT rebuild: on corruption it records a ``fts_integrity_failed:<table>`` marker
# that ``/lcm doctor`` surfaces, pointing operators at the explicit repair path.

BACKGROUND_INTEGRITY_ENV = "LCM_FTS_INTEGRITY_BACKGROUND"

# A ``fts_integrity_scan_started_at`` metadata stamp older than this (seconds) is
# treated as a crashed scan, so a later bind re-dispatches instead of wedging
# forever behind a stamp no live thread will ever clear.
INTEGRITY_SCAN_STALE_SECONDS = 15 * 60.0

# Guards the in-process registry and the one-scan-at-a-time decision below.
_integrity_scan_lock = threading.Lock()
# (db_path, table_name) -> daemon Thread. Exposed so tests can join a dispatched
# scan deterministically; entries are removed when the scan thread exits.
_integrity_scan_threads: dict[tuple[str, str], threading.Thread] = {}


def _background_integrity_enabled() -> bool:
    """Kill-switch: ``LCM_FTS_INTEGRITY_BACKGROUND=false`` restores the exact old
    synchronous integrity-check behavior on the startup path."""
    raw = os.environ.get(BACKGROUND_INTEGRITY_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _integrity_failed_key(spec: ExternalContentFtsSpec) -> str:
    return f"fts_integrity_failed:{spec.table_name}"


def _integrity_scan_started_key(spec: ExternalContentFtsSpec) -> str:
    return f"fts_integrity_scan_started_at:{spec.table_name}"


def _record_integrity_failed(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
    *,
    detail: str,
    now: float | None = None,
) -> None:
    ensure_metadata_table(conn)
    current = time.time() if now is None else now
    payload = json.dumps({"at": current, "detail": str(detail)[:2000]})
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (_integrity_failed_key(spec), payload),
    )


def _clear_integrity_failed(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> None:
    ensure_metadata_table(conn)
    conn.execute("DELETE FROM metadata WHERE key = ?", (_integrity_failed_key(spec),))


def load_integrity_failed(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec
) -> dict[str, object] | None:
    """Return ``{'at': float, 'detail': str}`` when a background scan flagged the
    index as corrupt, else ``None``. Used by ``/lcm doctor`` to surface the flag."""
    ensure_metadata_table(conn)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (_integrity_failed_key(spec),),
    ).fetchone()
    if not row or row[0] is None:
        return None
    try:
        data = json.loads(row[0])
        if isinstance(data, dict):
            return {"at": float(data.get("at") or 0.0), "detail": str(data.get("detail") or "")}
    except (TypeError, ValueError):
        pass
    return {"at": 0.0, "detail": str(row[0])}


def _load_scan_started_at(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec
) -> float | None:
    ensure_metadata_table(conn)
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (_integrity_scan_started_key(spec),),
    ).fetchone()
    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return None


def _record_scan_started(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float
) -> None:
    ensure_metadata_table(conn)
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (_integrity_scan_started_key(spec), str(now)),
    )


def _clear_scan_started(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
    *,
    expected: float | None = None,
) -> None:
    ensure_metadata_table(conn)
    if expected is None:
        conn.execute(
            "DELETE FROM metadata WHERE key = ?", (_integrity_scan_started_key(spec),)
        )
    else:
        # Only clear our own stamp so a newer scan's stamp survives.
        conn.execute(
            "DELETE FROM metadata WHERE key = ? AND value = ?",
            (_integrity_scan_started_key(spec), str(expected)),
        )


def _run_background_integrity_scan(
    db_path: str, spec: ExternalContentFtsSpec, started_at: float
) -> None:
    """Daemon-thread body: deep-check ``spec`` on a private connection.

    Opens its own read/write connection for the scan (the FTS5 integrity-check is
    issued as an INSERT command that rolls back inside a savepoint, so it never
    mutates data, but it does require a writable handle) and a separate brief
    connection to stamp the result. On corruption it flags rather than rebuilds.
    """
    key = (db_path, spec.table_name)
    timeout = SQLITE_BUSY_TIMEOUT_MS / 1000.0
    try:
        scan_conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
        try:
            scan_conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            # Persist the scan-started stamp on this DB so a crash mid-scan is
            # detectable cross-process via the staleness window above.
            _record_scan_started(scan_conn, spec, now=started_at)
            scan_conn.commit()
            result = check_external_content_fts_integrity(scan_conn, spec)
        finally:
            scan_conn.close()

        meta_conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
        try:
            meta_conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            status = result.get("status")
            if status == "pass":
                _record_integrity_checked(meta_conn, spec, now=started_at)
                _clear_integrity_failed(meta_conn, spec)
            elif status == "fail":
                _record_integrity_failed(
                    meta_conn, spec, detail=result.get("detail", ""), now=started_at
                )
                logger.warning(
                    "Background FTS integrity-check found corruption in '%s': %s. "
                    "Run `/lcm doctor repair apply` to rebuild the index.",
                    spec.table_name,
                    result.get("detail", ""),
                )
            # 'unchecked' (e.g. a read-only DB): leave the throttle marker unset
            # so the next bind retries; do not stamp or flag.
            _clear_scan_started(meta_conn, spec, expected=started_at)
            meta_conn.commit()
        finally:
            meta_conn.close()
    except Exception:  # pragma: no cover - defensive
        logger.exception(
            "Background FTS integrity-check for '%s' failed", spec.table_name
        )
        try:
            cleanup = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
            try:
                _clear_scan_started(cleanup, spec, expected=started_at)
                cleanup.commit()
            finally:
                cleanup.close()
        except sqlite3.DatabaseError:
            pass
    finally:
        with _integrity_scan_lock:
            if _integrity_scan_threads.get(key) is threading.current_thread():
                _integrity_scan_threads.pop(key, None)


def _dispatch_background_integrity_scan(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> bool:
    """Try to run the deep FTS integrity-check on a daemon thread.

    Returns ``True`` when the caller should NOT run the check synchronously —
    either a scan was dispatched here or one is already in flight (in-process, or
    in another process per a fresh ``fts_integrity_scan_started_at`` stamp).
    Returns ``False`` to fall back to the synchronous check (e.g. an in-memory or
    anonymous DB that cannot be reopened from another thread).
    """
    db_path = _database_path_for_connection(conn)
    if not db_path or db_path == ":memory:":
        return False
    current = time.time() if now is None else now
    key = (db_path, spec.table_name)
    with _integrity_scan_lock:
        existing = _integrity_scan_threads.get(key)
        if existing is not None and existing.is_alive():
            return True
        started = _load_scan_started_at(conn, spec)
        if started is not None and (current - started) < INTEGRITY_SCAN_STALE_SECONDS:
            # A recent scan (this or another process) owns this table; let it
            # stamp the marker. The bind returns fast without a duplicate scan.
            return True
        # Durably claim the scan cross-process BEFORE starting the thread. The
        # spawned thread stamps ``scan_started_at`` on its own connection, but
        # ``thread.start()`` returns before that stamp is committed — a second
        # process racing ``ensure_external_content_fts`` in that window would read
        # no stamp and dispatch a duplicate deep scan (F6). Writing the stamp here
        # under BEGIN IMMEDIATE closes that window; best-effort (a transient lock
        # just falls back to the thread's own stamp).
        claim_timeout = SQLITE_BUSY_TIMEOUT_MS / 1000.0
        try:
            claim_conn = sqlite3.connect(
                db_path, timeout=claim_timeout, check_same_thread=False
            )
            try:
                claim_conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
                claim_conn.execute("BEGIN IMMEDIATE")
                _record_scan_started(claim_conn, spec, now=current)
                claim_conn.commit()
            finally:
                claim_conn.close()
        except sqlite3.DatabaseError:
            pass

        thread = threading.Thread(
            target=_run_background_integrity_scan,
            args=(db_path, spec, current),
            name=f"lcm-fts-integrity-{spec.table_name}",
            daemon=True,
        )
        _integrity_scan_threads[key] = thread
        thread.start()
        return True


def join_background_integrity_scans(timeout: float | None = None) -> None:
    """Block until in-flight background integrity scans finish.

    Test/diagnostic helper so callers can deterministically observe the marker or
    failure flag a dispatched scan writes."""
    with _integrity_scan_lock:
        threads = list(_integrity_scan_threads.values())
    for thread in threads:
        thread.join(timeout)


def _fts_needs_rebuild(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
    *,
    now: float | None = None,
    throttle: bool = False,
) -> bool:
    if _fts_needs_rebuild_structural(conn, spec):
        return True
    # Structurally sound: the FTS5 integrity-check is O(index size) and was the
    # dominant startup cost on large databases (issue #235). On the startup path
    # (``throttle=True``) skip it when already checked within the interval.
    # Explicit repair (e.g. ``/lcm doctor repair apply``) uses ``throttle=False``
    # so it always runs the deep check and can fix same-row-count drift that the
    # structural checks cannot see.
    if throttle and not _should_run_integrity_check(conn, spec, now=now):
        return False
    # The deep check is due. On the startup path, dispatch it to a background
    # thread so the bind returns immediately (issue #6); the scan flags any
    # corruption via metadata rather than rebuilding here. The kill-switch and
    # non-file DBs fall back to the exact old synchronous behavior below.
    if throttle and _background_integrity_enabled():
        if _dispatch_background_integrity_scan(conn, spec, now=now):
            return False
    result = check_external_content_fts_integrity(conn, spec)
    if result["status"] == "pass":
        _record_integrity_checked(conn, spec, now=now)
    return result["status"] == "fail"


# SQLite/FTS5 error substrings that denote genuine corruption or index drift
# (SQLITE_CORRUPT / SQLITE_NOTADB, and the FTS5 integrity-check's own
# ``checksum mismatch`` for same-row-count stale drift). Everything else a
# writable integrity-check can raise — SQLITE_BUSY / SQLITE_LOCKED "database is
# locked", timeouts — is transient and must classify as ``unchecked``, never
# ``fail`` (which records a corruption flag).
_FTS_CORRUPTION_SIGNATURES = (
    "malformed",
    "disk image",
    "not a database",
    "corrupt",
    "checksum mismatch",
)


def _is_fts_corruption_error(detail: str) -> bool:
    lowered = detail.lower()
    return any(signature in lowered for signature in _FTS_CORRUPTION_SIGNATURES)


def check_external_content_fts_integrity(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
) -> dict[str, str]:
    """Run SQLite's FTS5 integrity-check for an external-content table.

    FTS5 exposes this as a special INSERT command. Wrap it in a savepoint and
    roll it back so diagnostics can verify the index without leaving any state
    behind on the shared connection.
    """

    if _fts_needs_rebuild_structural(conn, spec):
        return {"status": "fail", "detail": "structural repair needed"}

    savepoint = f"lcm_fts_integrity_{spec.table_name}"
    savepoint_sql = quote_sql_identifier(savepoint)
    try:
        conn.execute(f"SAVEPOINT {savepoint_sql}")
        conn.execute(
            f"INSERT INTO {quote_sql_identifier(spec.table_name)}({quote_sql_identifier(spec.table_name)}, rank) VALUES('integrity-check', 1)"
        )
    except sqlite3.DatabaseError as exc:
        try:
            conn.execute(f"ROLLBACK TO {savepoint_sql}")
            conn.execute(f"RELEASE {savepoint_sql}")
        except sqlite3.DatabaseError:
            pass
        detail = str(exc)
        lowered = detail.lower()
        if "readonly" in lowered or "read-only" in lowered:
            return {"status": "unchecked", "detail": detail}
        if _is_fts_corruption_error(detail):
            return {"status": "fail", "detail": detail}
        # A transient lock/busy/timeout — or any other non-corruption error — must
        # NOT be reported as corruption: the background scan would otherwise wedge
        # a false ``fts_integrity_failed`` flag (F3). Only an actual corruption
        # signature (malformed / disk image / not-a-database) fails the check.
        return {"status": "unchecked", "detail": detail}

    try:
        conn.execute(f"ROLLBACK TO {savepoint_sql}")
        conn.execute(f"RELEASE {savepoint_sql}")
    except sqlite3.DatabaseError as exc:
        return {"status": "fail", "detail": str(exc)}

    return {"status": "pass", "detail": "ok"}


def _drop_fts_table(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {quote_sql_identifier(table_name)}")
    for shadow_name in get_fts_shadow_table_names(table_name):
        conn.execute(f"DROP TABLE IF EXISTS {quote_sql_identifier(shadow_name)}")


def _extract_trigger_name(trigger_sql: str) -> str | None:
    match = re.search(
        r"CREATE\s+TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))",
        trigger_sql,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return match.group(1) or match.group(2)


def _drop_fts_triggers(conn: sqlite3.Connection, trigger_sqls: Sequence[str]) -> None:
    for trigger_sql in trigger_sqls:
        trigger_name = _extract_trigger_name(trigger_sql)
        if trigger_name:
            conn.execute(f"DROP TRIGGER IF EXISTS {quote_sql_identifier(trigger_name)}")


def _drop_fts_artifacts(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> None:
    _drop_fts_triggers(conn, spec.trigger_sqls)
    _drop_fts_table(conn, spec.table_name)


def _check_disk_space(db_path: str) -> bool:
    try:
        parent = os.path.dirname(os.path.abspath(db_path)) or "."
        return shutil.disk_usage(parent).free >= _MIN_DISK_SPACE_BYTES
    except (OSError, AttributeError):
        return True


def _fts_missing_triggers(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    expected = {
        trigger_name
        for trigger_name in (_extract_trigger_name(sql) for sql in spec.trigger_sqls)
        if trigger_name
    }
    if not expected:
        return False
    placeholders = ",".join("?" for _ in expected)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ({placeholders})",
        tuple(sorted(expected)),
    ).fetchall()
    existing = {str(row[0]) for row in rows if row and row[0]}
    return bool(expected - existing)


def external_content_fts_needs_repair(conn: sqlite3.Connection, spec: ExternalContentFtsSpec) -> bool:
    return _fts_needs_rebuild_structural(conn, spec) or _fts_missing_triggers(conn, spec)


def repair_external_content_fts(
    conn: sqlite3.Connection,
    spec: ExternalContentFtsSpec,
    *,
    now: float | None = None,
    throttle: bool = False,
) -> dict[str, bool]:
    rebuilt = False
    degraded = False
    if _fts_needs_rebuild(conn, spec, now=now, throttle=throttle):
        db_path = conn.execute("PRAGMA database_list").fetchone()
        if db_path:
            db_file = db_path[2]
            if db_file and not _check_disk_space(db_file):
                logger.warning(
                    "Low disk space for FTS rebuild of '%s' (%d MB needed), degrading to LIKE search",
                    spec.table_name,
                    _MIN_DISK_SPACE_BYTES // (1024 * 1024),
                )
                _drop_fts_artifacts(conn, spec)
                # The corrupt index is gone (degraded to LIKE search); a stale
                # integrity-failed flag would otherwise keep `/lcm doctor`
                # reporting issues-found for an index that no longer exists.
                _clear_integrity_failed(conn, spec)
                conn.commit()
                return {"rebuilt": False, "degraded": True, "triggers_recreated": False}
        _drop_fts_table(conn, spec.table_name)
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE {quote_sql_identifier(spec.table_name)} USING fts5(
                {quote_sql_identifier(spec.indexed_column)},
                content={quote_sql_identifier(spec.content_table)},
                content_rowid={quote_sql_identifier(spec.content_rowid)}
            )
            """
        )
        conn.execute(
            f"INSERT INTO {quote_sql_identifier(spec.table_name)}({quote_sql_identifier(spec.table_name)}) VALUES('rebuild')"
        )
        rebuilt = True

    triggers_were_missing = _fts_missing_triggers(conn, spec)
    for trigger_sql in spec.trigger_sqls:
        conn.execute(trigger_sql)
    if rebuilt:
        # A freshly rebuilt index is known-consistent; record the marker so the
        # next startup can skip the deep integrity-check within the interval.
        _record_integrity_checked(conn, spec, now=now)
    # A completed repair resolves any prior background-scan corruption flag: clear
    # it in the SAME transaction that commits the rebuild so `/lcm doctor` stops
    # reporting issues-found (and the next self-healing scan is not pushed out a
    # full interval). Without this an explicit `repair apply` left the flag stuck.
    _clear_integrity_failed(conn, spec)
    conn.commit()
    return {"rebuilt": rebuilt, "degraded": degraded, "triggers_recreated": triggers_were_missing}


def ensure_external_content_fts(
    conn: sqlite3.Connection, spec: ExternalContentFtsSpec, *, now: float | None = None
) -> None:
    # Startup path: throttle the deep integrity-check. Explicit repair callers
    # use ``repair_external_content_fts(..., throttle=False)`` for a forced check.
    repair_external_content_fts(conn, spec, now=now, throttle=True)


def run_versioned_migrations(conn: sqlite3.Connection) -> None:
    refuse_schema_version_too_new(conn)

    ensure_metadata_table(conn)
    ensure_migration_state_table(conn)

    refuse_schema_version_too_new(conn)
    current_version = get_schema_version(conn)
    if current_version < 2:
        mark_migration_step_complete(conn, "v2_external_content_fts_triggers")
        current_version = 2

    if current_version < 3:
        ensure_lifecycle_state_table(conn)
        mark_migration_step_complete(conn, "v3_lifecycle_state")
        current_version = 3
    else:
        ensure_lifecycle_state_table(conn)

    ensure_lifecycle_state_columns(conn)
    if current_version < 4:
        mark_migration_step_complete(conn, "v4_lifecycle_debt_columns")
        current_version = 4

    ensure_message_origin_columns(conn)
    if current_version < 5:
        mark_migration_step_complete(conn, "v5_message_conversation_id")
        current_version = 5

    # NOTE: the opt-in temporal-rollup tables are deliberately NOT created here.
    # Creating them would advance the core schema for every install (even with
    # the feature off) and make the DB unreadable by a base build. They are
    # created lazily by RollupStore on the enabled path via a NAMED migration
    # step (``temporal_rollups_v1``), independent of this numeric counter.
    # Embedding tables are intentionally NOT created here: they are an opt-in
    # feature materialized lazily by VectorStore (recorded via the named
    # ``embeddings_v1`` marker), so a disabled install stays at v5 with no
    # embedding tables and the numeric counter is free for the temporal train.
    set_schema_version(conn, current_version)
