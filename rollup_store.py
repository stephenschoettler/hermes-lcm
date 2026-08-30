"""SQLite-backed storage for derived temporal summary rollups.

This module is intentionally not wired into the LCM engine yet. It provides
only the durable schema-facing operations used by later temporal-memory work.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, NamedTuple, Optional, Sequence

from .db_bootstrap import (
    configure_connection,
    ensure_temporal_rollup_tables,
    mark_migration_step_complete,
    refuse_schema_version_too_new,
    run_versioned_migrations,
    verify_temporal_rollup_schema,
)
from .sqlite_util import _is_sqlite_locked_error

logger = logging.getLogger(__name__)

# How long a ``building`` row's lease is valid. A build that outlives its lease
# can be reclaimed to ``stale`` by a later maintenance pass; the generation
# compare-and-set in :meth:`RollupStore.mark_ready` still protects correctness
# if the original builder returns late, so a generous lease only costs a
# possible redundant rebuild, never a wrong publish.
_BUILD_LEASE_SECONDS = 900


class RollupBuildToken(NamedTuple):
    """A build lease returned by :meth:`RollupStore.upsert_building`.

    ``generation`` is the row's optimistic-concurrency counter captured at build
    start; :meth:`RollupStore.mark_ready` publishes only if the row's generation
    still equals this token (a compare-and-set), so an invalidation that arrives
    mid-build supersedes the stale builder.
    """

    rollup_id: int
    generation: int
    nonce: str = ""


class RollupStore:
    """SQLite-backed store for temporal rollups and their source nodes."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
        )
        refuse_schema_version_too_new(self._conn)
        configure_connection(self._conn)
        self._conn.row_factory = sqlite3.Row
        run_versioned_migrations(self._conn)
        # The rollup tables are a lazy, opt-in feature: they are NOT part of the
        # core numeric schema_version (see db_bootstrap.run_versioned_migrations).
        # RollupStore is only constructed on the temporal_rollups_enabled path, so
        # creating them here keeps a disabled install at the base schema with no
        # rollup tables while still being idempotent under concurrent construction.
        ensure_temporal_rollup_tables(self._conn)
        # Do NOT trust the named marker alone: it can be present on a DB whose
        # tables were dropped or left partial by a crash mid-create. Verify the
        # required tables+indexes actually exist and re-ensure (idempotent
        # CREATE IF NOT EXISTS makes this safe) before recording the step, so a
        # stale marker can never mask a missing table (maintainer #387 A3).
        missing = verify_temporal_rollup_schema(self._conn)
        if missing:
            ensure_temporal_rollup_tables(self._conn)
            missing = verify_temporal_rollup_schema(self._conn)
            if missing:
                raise RuntimeError(
                    "temporal rollup schema incomplete after ensure: "
                    + ", ".join(missing)
                )
        mark_migration_step_complete(self._conn, "temporal_rollups_v1")
        self._conn.commit()

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        try:
            with self._write_lock, self._conn:
                yield
        except sqlite3.Error as exc:
            if _is_sqlite_locked_error(exc):
                logger.warning("Temporal rollup write blocked by SQLite lock contention")
            raise

    @property
    def connection(self) -> sqlite3.Connection | None:
        """The live connection for read-only diagnostics, or ``None`` after close."""
        return getattr(self, "_conn", None)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _lease_deadline() -> str:
        return (
            datetime.now(timezone.utc) + timedelta(seconds=_BUILD_LEASE_SECONDS)
        ).isoformat()

    def _access_scope_for_partition(self, scope: str) -> str | None:
        """Copy the owner scope from a source row in this rollup partition."""

        for table in ("summary_nodes", "messages"):
            try:
                row = self._conn.execute(
                    f"SELECT access_scope FROM {table} "
                    "WHERE session_id = ? AND access_scope IS NOT NULL "
                    "ORDER BY rowid LIMIT 1",
                    (str(scope),),
                ).fetchone()
            except sqlite3.OperationalError:
                row = None
            if row is not None:
                return str(row[0])
        return None

    def _row_to_rollup(self, row: sqlite3.Row | None) -> dict[str, object] | None:
        if row is None:
            return None
        rollup_id = int(row["rollup_id"])
        source_rows = self._conn.execute(
            "SELECT node_id FROM lcm_rollup_sources WHERE rollup_id = ? ORDER BY node_id",
            (rollup_id,),
        ).fetchall()
        return {
            "rollup_id": rollup_id,
            "period_kind": row["period_kind"],
            "period_start": row["period_start"],
            "scope": row["scope"],
            "summary": row["summary"],
            "token_count": row["token_count"],
            "status": row["status"],
            "built_at": row["built_at"],
            "source_fingerprint": row["source_fingerprint"],
            "error": row["error"],
            "generation": int(row["generation"] or 0),
            "source_node_ids": [int(source_row["node_id"]) for source_row in source_rows],
        }

    def get_rollup(
        self,
        period_kind: str,
        period_start: str,
        scope: str,
    ) -> dict[str, object] | None:
        row = self._conn.execute(
            """
            SELECT *
            FROM lcm_rollups
            WHERE period_kind = ? AND period_start = ? AND scope = ?
            """,
            (period_kind, period_start, scope),
        ).fetchone()
        return self._row_to_rollup(row)

    def upsert_building(
        self, period_kind: str, period_start: str, scope: str
    ) -> RollupBuildToken:
        """Claim a build lease, returning the row id and its NEW generation.

        The claim is itself a compare-and-set: re-claiming an existing row
        advances ``generation`` so two builders racing the same period get
        DISTINCT tokens and only the latest claimant's :meth:`mark_ready` /
        :meth:`mark_failed` succeeds (an older claimant is superseded). A fresh
        row starts at generation 0 (no prior claimant to race). A
        ``lease_expires_at`` is stamped so a crashed builder's row can later be
        reclaimed by :meth:`reclaim_stale_building`.

        The claim deliberately does NOT clear ``lcm_rollup_sources``: the prior
        last-known-good lineage stays queryable until :meth:`mark_ready` swaps it
        (sources-replace + ``status = 'ready'`` in one transaction). Clearing at
        claim opened a purge/deletion race — a concurrent purge-by-node queried
        the affected rollups, found ZERO sources (lineage already cleared) and so
        did not re-stale, letting the in-flight build publish deleted-node
        content as ``ready`` (maintainer #387 A2).
        """
        access_scope = self._access_scope_for_partition(scope)
        with self._write_transaction():
            self._conn.execute(
                """
                INSERT INTO lcm_rollups(
                    period_kind, period_start, scope, access_scope, status, built_at,
                    lease_expires_at, lease_nonce
                )
                VALUES(?, ?, ?, ?, 'building', ?, ?, ?)
                ON CONFLICT(period_kind, period_start, scope) DO UPDATE SET
                    access_scope = COALESCE(excluded.access_scope, lcm_rollups.access_scope),
                    status = 'building',
                    generation = lcm_rollups.generation + 1,
                    built_at = excluded.built_at,
                    lease_expires_at = excluded.lease_expires_at,
                    lease_nonce = excluded.lease_nonce,
                    failed_at = NULL,
                    source_fingerprint = NULL,
                    error = NULL
                """,
                (
                    period_kind, period_start, scope, access_scope, self._now(),
                    self._lease_deadline(), uuid.uuid4().hex,
                ),
            )
            row = self._conn.execute(
                """
                SELECT rollup_id, generation, lease_nonce
                FROM lcm_rollups
                WHERE period_kind = ? AND period_start = ? AND scope = ?
                """,
                (period_kind, period_start, scope),
            ).fetchone()
            rollup_id = int(row["rollup_id"])
        return RollupBuildToken(
            rollup_id, int(row["generation"] or 0), str(row["lease_nonce"] or "")
        )

    def mark_ready(
        self,
        token: RollupBuildToken,
        summary: str,
        token_count: int,
        source_node_ids: Sequence[int],
        fingerprint: str,
    ) -> bool:
        """Publish a completed build iff it was not superseded.

        Returns ``True`` when the row is published, or ``False`` when the build
        was superseded (an invalidation advanced the row's generation past the
        token, or the row was reclaimed) — in which case the newer state is left
        untouched. Raises ``ValueError`` only for a genuinely unknown rollup id.

        The compare-and-set also requires ``status = 'building'``: once a row has
        left ``building`` (published ``ready``, or reclaimed to ``stale``) no
        token-holder may transition it, even at the captured generation
        (maintainer #387 A1). ``mark_ready`` does not itself advance
        ``generation``, so the status clause is what stops a late ``mark_failed``
        from the same token flipping a just-published row.
        """
        unique_source_ids = list(dict.fromkeys(int(node_id) for node_id in source_node_ids))
        with self._write_transaction():
            owned = self._conn.execute(
                """
                SELECT period_kind, period_start, scope FROM lcm_rollups
                WHERE rollup_id = ? AND generation = ? AND lease_nonce = ?
                  AND status = 'building'
                """,
                (int(token.rollup_id), int(token.generation), str(token.nonce)),
            ).fetchone()
            if owned is None:
                exists = self._conn.execute(
                    "SELECT 1 FROM lcm_rollups WHERE rollup_id = ?",
                    (int(token.rollup_id),),
                ).fetchone()
                if exists is None:
                    raise ValueError(f"unknown rollup_id: {token.rollup_id}")
                return False

            start_day = date.fromisoformat(str(owned["period_start"]))
            if str(owned["period_kind"]) == "day":
                end_day = start_day
            elif str(owned["period_kind"]) == "week":
                end_day = start_day + timedelta(days=6)
            else:
                next_month = (
                    start_day.replace(year=start_day.year + 1, month=1, day=1)
                    if start_day.month == 12
                    else start_day.replace(month=start_day.month + 1, day=1)
                )
                end_day = next_month - timedelta(days=1)
            window_start = datetime.combine(
                start_day, datetime.min.time(), tzinfo=timezone.utc
            ).timestamp()
            window_end = datetime.combine(
                end_day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
            ).timestamp()
            pending_mutation = self._conn.execute(
                """
                SELECT 1 FROM lcm_rollup_invalidations
                WHERE scope = ?
                  AND covered_start < ?
                  AND covered_end >= ?
                LIMIT 1
                """,
                (str(owned["scope"]), window_end, window_start),
            ).fetchone()
            if pending_mutation is not None:
                self._conn.execute(
                    """
                    UPDATE lcm_rollups
                    SET status='stale', generation=generation + 1,
                        error='source mutation pending', lease_expires_at=NULL,
                        lease_nonce=''
                    WHERE rollup_id=? AND generation=? AND lease_nonce=?
                      AND status='building'
                    """,
                    (
                        int(token.rollup_id), int(token.generation),
                        str(token.nonce),
                    ),
                )
                return False

            # A first build has no prior lcm_rollup_sources lineage for deletion
            # invalidation to find. Validate the proposed source snapshot in the
            # publication transaction itself so a deleted source can never be
            # committed as ready.
            has_summary_nodes = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='summary_nodes'"
            ).fetchone()
            if unique_source_ids and has_summary_nodes is not None:
                self._conn.execute(
                    "CREATE TEMP TABLE IF NOT EXISTS lcm_rollup_publish_sources "
                    "(node_id INTEGER PRIMARY KEY) WITHOUT ROWID"
                )
                self._conn.execute("DELETE FROM temp.lcm_rollup_publish_sources")
                self._conn.executemany(
                    "INSERT INTO temp.lcm_rollup_publish_sources(node_id) VALUES(?)",
                    ((node_id,) for node_id in unique_source_ids),
                )
                missing = self._conn.execute(
                    """
                    SELECT 1
                    FROM temp.lcm_rollup_publish_sources proposed
                    LEFT JOIN summary_nodes node ON node.node_id = proposed.node_id
                    WHERE node.node_id IS NULL
                    LIMIT 1
                    """
                ).fetchone()
                if missing is not None:
                    self._conn.execute(
                        """
                        UPDATE lcm_rollups
                        SET status='stale', generation=generation + 1,
                            error='source changed during build',
                            lease_expires_at=NULL, lease_nonce=''
                        WHERE rollup_id=? AND generation=? AND lease_nonce=?
                          AND status='building'
                        """,
                        (
                            int(token.rollup_id), int(token.generation),
                            str(token.nonce),
                        ),
                    )
                    return False
            cur = self._conn.execute(
                """
                UPDATE lcm_rollups
                SET summary = ?, token_count = ?, status = 'ready', built_at = ?,
                    source_fingerprint = ?, error = NULL, failed_at = NULL,
                    lease_expires_at = NULL, lease_nonce = ''
                WHERE rollup_id = ? AND generation = ? AND lease_nonce = ?
                  AND status = 'building'
                """,
                (
                    summary,
                    int(token_count),
                    self._now(),
                    fingerprint,
                    int(token.rollup_id),
                    int(token.generation),
                    str(token.nonce),
                ),
            )
            if cur.rowcount == 0:
                exists = self._conn.execute(
                    "SELECT 1 FROM lcm_rollups WHERE rollup_id = ?",
                    (int(token.rollup_id),),
                ).fetchone()
                if exists is None:
                    raise ValueError(f"unknown rollup_id: {token.rollup_id}")
                # Superseded by a newer generation: discard this build's result.
                return False
            self._conn.execute(
                "DELETE FROM lcm_rollup_sources WHERE rollup_id = ?",
                (int(token.rollup_id),),
            )
            self._conn.executemany(
                "INSERT INTO lcm_rollup_sources(rollup_id, node_id) VALUES(?, ?)",
                ((int(token.rollup_id), node_id) for node_id in unique_source_ids),
            )
        return True

    def mark_failed(self, token: RollupBuildToken, error: str) -> bool:
        """Record a build failure only while the caller owns the build lease."""
        with self._write_transaction():
            now = self._now()
            cur = self._conn.execute(
                """
                UPDATE lcm_rollups
                SET status = 'failed', error = ?, failed_at = ?,
                    lease_expires_at = NULL, lease_nonce = ''
                WHERE rollup_id = ? AND generation = ? AND lease_nonce = ?
                  AND status = 'building'
                """,
                (
                    error, now, int(token.rollup_id), int(token.generation),
                    str(token.nonce),
                ),
            )
        return int(cur.rowcount or 0) > 0

    @staticmethod
    def _period_starts_for_day(day: date | str) -> tuple[str, str, str]:
        if isinstance(day, datetime):
            parsed = day.date()
        elif isinstance(day, date):
            parsed = day
        else:
            parsed = date.fromisoformat(str(day))
        week_start = parsed - timedelta(days=parsed.weekday())
        month_start = parsed.replace(day=1)
        return parsed.isoformat(), week_start.isoformat(), month_start.isoformat()

    def mark_stale_for_day(self, day: date | str, scope: str) -> int:
        """Invalidate a day and its containing week + month.

        Every invalidation advances ``generation`` (regardless of the prior
        status) so a build that is in flight for any of these periods is
        superseded and its late ``mark_ready`` becomes a no-op. Rows that do not
        yet exist are seeded as ``stale`` so maintenance builds them.
        """
        day_start, week_start, month_start = self._period_starts_for_day(day)
        access_scope = self._access_scope_for_partition(scope)
        with self._write_transaction():
            cur = self._conn.execute(
                """
                INSERT INTO lcm_rollups(
                    period_kind, period_start, scope, access_scope, status
                )
                VALUES
                    ('day', ?, ?, ?, 'stale'),
                    ('week', ?, ?, ?, 'stale'),
                    ('month', ?, ?, ?, 'stale')
                ON CONFLICT(period_kind, period_start, scope) DO UPDATE SET
                    access_scope = COALESCE(excluded.access_scope, lcm_rollups.access_scope),
                    status = 'stale',
                    generation = lcm_rollups.generation + 1,
                    lease_expires_at = NULL,
                    lease_nonce = ''
                """,
                (
                    day_start, scope, access_scope,
                    week_start, scope, access_scope,
                    month_start, scope, access_scope,
                ),
            )
        return int(cur.rowcount or 0)

    def stale_aggregates_for_day(self, day: date | str, scope: str) -> int:
        """Invalidate only the week + month containing ``day`` (not the day).

        Called after a daily rollup is (re)built so its containing aggregates,
        which may have published against the previous daily, are rebuilt from the
        new daily. Advances ``generation`` so an in-flight aggregate build is
        superseded.
        """
        _day_start, week_start, month_start = self._period_starts_for_day(day)
        access_scope = self._access_scope_for_partition(scope)
        with self._write_transaction():
            cur = self._conn.execute(
                """
                INSERT INTO lcm_rollups(
                    period_kind, period_start, scope, access_scope, status
                )
                VALUES
                    ('week', ?, ?, ?, 'stale'),
                    ('month', ?, ?, ?, 'stale')
                ON CONFLICT(period_kind, period_start, scope) DO UPDATE SET
                    access_scope = COALESCE(excluded.access_scope, lcm_rollups.access_scope),
                    status = 'stale',
                    generation = lcm_rollups.generation + 1,
                    lease_expires_at = NULL,
                    lease_nonce = ''
                """,
                (week_start, scope, access_scope, month_start, scope, access_scope),
            )
        return int(cur.rowcount or 0)

    def upsert_stale(self, period_kind: str, period_start: str, scope: str) -> int:
        """Durably seed a single ``stale`` row for one period.

        Used by ``/lcm rollups rebuild`` to queue every requested target before
        the per-pass build budget is applied, so unattempted targets remain
        durably ``stale`` (not absent) and get built by later maintenance. Does
        not disturb a row that is currently ``building``.
        """
        targets = self._expand_stale_targets([(period_kind, period_start, scope)])
        with self._write_transaction():
            affected = 0
            for target_kind, target_start, target_scope in targets:
                access_scope = self._access_scope_for_partition(target_scope)
                cur = self._conn.execute(
                    """
                    INSERT INTO lcm_rollups(
                        period_kind, period_start, scope, access_scope, status
                    )
                    VALUES(?, ?, ?, ?, 'stale')
                    ON CONFLICT(period_kind, period_start, scope) DO UPDATE SET
                        access_scope = COALESCE(excluded.access_scope, lcm_rollups.access_scope),
                        status = 'stale',
                        generation = lcm_rollups.generation + 1,
                        lease_expires_at = NULL,
                        lease_nonce = ''
                    WHERE lcm_rollups.status != 'building'
                    """,
                    (target_kind, target_start, target_scope, access_scope),
                )
                affected += int(cur.rowcount or 0)
        return affected

    @classmethod
    def _expand_stale_targets(
        cls, targets: Sequence[tuple[str, str, str]]
    ) -> list[tuple[str, str, str]]:
        expanded: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for kind, start, scope in targets:
            values = [(str(kind), str(start), str(scope))]
            if kind == "day":
                _day, week, month = cls._period_starts_for_day(start)
                values.extend(
                    [("week", week, str(scope)), ("month", month, str(scope))]
                )
            for value in values:
                if value not in seen:
                    seen.add(value)
                    expanded.append(value)
        return expanded

    def upsert_stale_many(
        self,
        targets: Sequence[tuple[str, str, str]],
        *,
        authorized_access_scope: str | None = None,
    ) -> int:
        """Durably seed several ``stale`` rows in ONE transaction (all-or-nothing).

        ``/lcm rollups rebuild`` queues every requested target before applying
        the per-pass build budget. Seeding each target in its own transaction let
        a mid-batch failure leave some targets seeded and others absent (e.g. a
        missing month with no row); seeding them together makes the batch
        atomic — a failure on any target rolls the whole seed back so no target
        is left half-queued (maintainer #391 D2). Per-target semantics match
        :meth:`upsert_stale`: a currently-``building`` row is left untouched, and
        a conflicting seed advances ``generation`` to supersede an in-flight
        build.
        """
        rows = self._expand_stale_targets(targets)
        if not rows:
            return 0
        if authorized_access_scope is not None and not str(authorized_access_scope).strip():
            raise ValueError("authorized_access_scope must be non-empty when provided")
        affected = 0
        with self._write_transaction():
            for period_kind, period_start, scope in rows:
                access_scope = (
                    str(authorized_access_scope)
                    if authorized_access_scope is not None
                    else self._access_scope_for_partition(scope)
                )
                cur = self._conn.execute(
                    """
                    INSERT INTO lcm_rollups(
                        period_kind, period_start, scope, access_scope, status
                    )
                    VALUES(?, ?, ?, ?, 'stale')
                    ON CONFLICT(period_kind, period_start, scope) DO UPDATE SET
                        access_scope = COALESCE(excluded.access_scope, lcm_rollups.access_scope),
                        status = 'stale',
                        generation = lcm_rollups.generation + 1,
                        lease_expires_at = NULL,
                        lease_nonce = ''
                    WHERE lcm_rollups.status != 'building'
                    """,
                    (period_kind, period_start, scope, access_scope),
                )
                affected += int(cur.rowcount or 0)
        return affected

    def defer_incomplete(self, token: RollupBuildToken, reason: str) -> bool:
        """Release a claimed aggregate build back to ``stale`` with a reason.

        Called when a claimed aggregate cannot be published because a constituent
        daily is missing/stale/building. Guarded by the build token
        (compare-and-set on ``generation`` for a row this builder still owns), so
        a superseded builder's deferral cannot erase a newer ready aggregate that
        another builder published in the meantime. ``generation`` is left
        unchanged (a deferral is not an input change). Returns ``True`` when the
        owned ``building`` row was released, ``False`` when superseded.
        """
        with self._write_transaction():
            cur = self._conn.execute(
                """
                UPDATE lcm_rollups
                SET status = 'stale', error = ?, lease_expires_at = NULL,
                    lease_nonce = ''
                WHERE rollup_id = ? AND generation = ? AND lease_nonce = ?
                  AND status = 'building'
                """,
                (
                    reason, int(token.rollup_id), int(token.generation),
                    str(token.nonce),
                ),
            )
        return int(cur.rowcount or 0) > 0

    def resolve_no_source(self, token: RollupBuildToken) -> bool:
        """Clear a claimed period that turned out to have no source content.

        A period can go ``stale`` yet have no summary node to build from (all its
        sources were deleted, or a rebuild request seeded a contentless target).
        Such a period must not linger ``stale`` forever consuming a per-pass build
        slot: this deletes the claimed row (and its sources) iff this builder
        still owns it (compare-and-set on ``generation`` AND ``status =
        'building'``, so a row that already left ``building`` at the same
        generation cannot be deleted out from under a newer state — same
        terminal-transition invariant as :meth:`mark_ready`, maintainer #387 A1).
        A later covering publication re-seeds a fresh ``stale`` row. Returns
        ``True`` when cleared, ``False`` when superseded (a newer invalidation is
        left to rebuild).
        """
        with self._write_transaction():
            cur = self._conn.execute(
                "DELETE FROM lcm_rollups WHERE rollup_id = ? AND generation = ? "
                "AND lease_nonce = ? AND status = 'building'",
                (int(token.rollup_id), int(token.generation), str(token.nonce)),
            )
            if int(cur.rowcount or 0) > 0:
                self._conn.execute(
                    "DELETE FROM lcm_rollup_sources WHERE rollup_id = ?",
                    (int(token.rollup_id),),
                )
        return int(cur.rowcount or 0) > 0

    def reclaim_stale_building(self, now: str | None = None, *, limit: int = 256) -> int:
        """Flip expired ``building`` rows back to ``stale`` so a crashed build is
        retried. Advances ``generation`` so the crashed builder, if it ever
        returns, cannot publish over the reclaimed (and possibly re-superseded)
        row.
        """
        cutoff = now or self._now()
        with self._write_transaction():
            cur = self._conn.execute(
                """
                UPDATE lcm_rollups
                SET status = 'stale',
                    generation = generation + 1,
                    lease_expires_at = NULL,
                    lease_nonce = ''
                WHERE rollup_id IN (
                    SELECT rollup_id FROM lcm_rollups
                    WHERE status = 'building'
                      AND lease_expires_at IS NOT NULL
                      AND lease_expires_at < ?
                    ORDER BY lease_expires_at, rollup_id
                    LIMIT ?
                )
                """,
                (cutoff, max(0, int(limit))),
            )
        return int(cur.rowcount or 0)

    def has_pending_invalidations(self, scope: str | None = None) -> bool:
        """Check for durable mutation debt using the pending-event index."""
        if scope is None:
            row = self._conn.execute(
                "SELECT 1 FROM lcm_rollup_invalidations ORDER BY event_id LIMIT 1"
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT 1 FROM lcm_rollup_invalidations WHERE scope=? "
                "ORDER BY event_id LIMIT 1",
                (scope,),
            ).fetchone()
        return row is not None

    def drain_invalidations(
        self, *, event_limit: int = 256, day_budget: int = 256
    ) -> int:
        """Apply a bounded number of invalidated UTC days, resuming durably.

        ``event_limit`` caps event-row enumeration while ``day_budget`` caps the
        total calendar expansion across those events. A long-span event stores
        its next unprocessed day and remains pending until a later pass.
        """
        bounded_events = max(0, int(event_limit))
        remaining_days = max(0, int(day_budget))
        if bounded_events == 0 or remaining_days == 0:
            return 0
        with self._write_transaction():
            rows = self._conn.execute(
                """
                SELECT event_id, scope, access_scope, covered_start, covered_end, next_day
                FROM lcm_rollup_invalidations
                ORDER BY event_id
                LIMIT ?
                """,
                (bounded_events,),
            ).fetchall()
            if not rows:
                return 0
            processed_days = 0
            for row in rows:
                start = datetime.fromtimestamp(
                    float(row["covered_start"]), tz=timezone.utc
                ).date()
                end = datetime.fromtimestamp(
                    float(row["covered_end"]), tz=timezone.utc
                ).date()
                if end < start:
                    start, end = end, start
                current = (
                    date.fromisoformat(str(row["next_day"]))
                    if row["next_day"]
                    else start
                )
                while current <= end and remaining_days > 0:
                    day_start, week_start, month_start = self._period_starts_for_day(
                        current
                    )
                    self._conn.execute(
                        """
                        INSERT INTO lcm_rollups(
                            period_kind, period_start, scope, access_scope, status
                        ) VALUES
                            ('day', ?, ?, ?, 'stale'),
                            ('week', ?, ?, ?, 'stale'),
                            ('month', ?, ?, ?, 'stale')
                        ON CONFLICT(period_kind, period_start, scope) DO UPDATE SET
                            access_scope = COALESCE(excluded.access_scope, lcm_rollups.access_scope),
                            status='stale', generation=lcm_rollups.generation + 1,
                            lease_expires_at=NULL, lease_nonce=''
                        """,
                        (
                            day_start, str(row["scope"]), row["access_scope"],
                            week_start, str(row["scope"]), row["access_scope"],
                            month_start, str(row["scope"]), row["access_scope"],
                        ),
                    )
                    current += timedelta(days=1)
                    processed_days += 1
                    remaining_days -= 1
                if current > end:
                    self._conn.execute(
                        "DELETE FROM lcm_rollup_invalidations WHERE event_id=?",
                        (int(row["event_id"]),),
                    )
                else:
                    self._conn.execute(
                        "UPDATE lcm_rollup_invalidations SET next_day=? WHERE event_id=?",
                        (current.isoformat(), int(row["event_id"])),
                    )
                    break
                if remaining_days == 0:
                    break
        return processed_days

    def ready_rollups_for_window(
        self,
        period_kind: str,
        start: str,
        end: str,
        scope: str,
    ) -> list[dict[str, object]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM lcm_rollups
            WHERE period_kind = ?
              AND period_start >= ?
              AND period_start <= ?
              AND scope = ?
              AND status = 'ready'
            ORDER BY period_start
            """,
            (period_kind, start, end, scope),
        ).fetchall()
        return [self._row_to_rollup(row) for row in rows]

    def get_cursor(self, period_kind: str, scope: str = "") -> str | None:
        row = self._conn.execute(
            "SELECT last_build_cursor FROM lcm_rollup_state WHERE period_kind = ? AND scope = ?",
            (period_kind, scope),
        ).fetchone()
        return str(row["last_build_cursor"]) if row and row["last_build_cursor"] is not None else None

    def set_cursor(
        self,
        period_kind: str,
        cursor: str | None,
        scope: str = "",
        *,
        built_at: str | None = None,
    ) -> None:
        access_scope = self._access_scope_for_partition(scope)
        with self._write_transaction():
            self._conn.execute(
                """
                INSERT INTO lcm_rollup_state(
                    period_kind, scope, access_scope, last_build_cursor, last_built_at
                )
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(period_kind, scope) DO UPDATE SET
                    access_scope = COALESCE(excluded.access_scope, lcm_rollup_state.access_scope),
                    last_build_cursor = excluded.last_build_cursor,
                    last_built_at = excluded.last_built_at
                """,
                (period_kind, scope, access_scope, cursor, built_at or self._now()),
            )

    def purge_rollups_for_sources(self, node_ids: Sequence[int]) -> int:
        """Invalidate rollups built from now-purged source nodes.

        The affected periods are re-seeded ``stale`` (generation advanced, so an
        in-flight build cannot publish purged content) rather than hard-deleted:
        a deleted row at/before the build cursor would leave the cursor advanced
        with no pending row, so the window would never be rebuilt. Re-staling
        leaves a durable pending row that maintenance rebuilds from whatever
        sources remain (or clears via :meth:`resolve_no_source` if none do). The
        rollups' stale source rows are dropped and repopulated on rebuild.
        Returns the number of periods re-staled.
        """
        unique_node_ids = list(dict.fromkeys(int(node_id) for node_id in node_ids))
        if not unique_node_ids:
            return 0
        with self._write_transaction():
            self._conn.execute(
                "CREATE TEMP TABLE IF NOT EXISTS lcm_rollup_purge_nodes "
                "(node_id INTEGER PRIMARY KEY) WITHOUT ROWID"
            )
            self._conn.execute("DELETE FROM temp.lcm_rollup_purge_nodes")
            self._conn.executemany(
                "INSERT INTO temp.lcm_rollup_purge_nodes(node_id) VALUES(?)",
                ((node_id,) for node_id in unique_node_ids),
            )
            cur = self._conn.execute(
                """
                UPDATE lcm_rollups
                SET status = 'stale',
                    generation = generation + 1,
                    summary = NULL,
                    token_count = NULL,
                    source_fingerprint = NULL,
                    error = NULL,
                    lease_expires_at = NULL,
                    lease_nonce = ''
                WHERE rollup_id IN (
                    SELECT source.rollup_id
                    FROM lcm_rollup_sources source
                    JOIN temp.lcm_rollup_purge_nodes purged
                      ON purged.node_id = source.node_id
                )
                """,
            )
            affected = int(cur.rowcount or 0)
            self._conn.execute(
                """
                DELETE FROM lcm_rollup_sources
                WHERE rollup_id IN (
                    SELECT source.rollup_id
                    FROM lcm_rollup_sources source
                    JOIN temp.lcm_rollup_purge_nodes purged
                      ON purged.node_id = source.node_id
                )
                """
            )
        return affected

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass
            conn.close()
            self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass
