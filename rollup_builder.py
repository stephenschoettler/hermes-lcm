"""Flag-gated temporal rollup construction and engine wiring helpers."""

from __future__ import annotations

import hashlib
import json
import logging
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from time import monotonic
from typing import Callable, Sequence

from .config import LCMConfig
from .dag import SummaryDAG
from .escalation import _deterministic_truncate, summarize_with_escalation
from .rollup_periods import CoverageNode, canonical_frontier, load_source_lineage
from .rollup_store import RollupBuildToken, RollupStore
from .sqlite_util import _sqlite_savepoint
from .tokens import count_tokens

logger = logging.getLogger(__name__)

Summarizer = Callable[..., tuple[str, int]]
_FAILED_ROLLUP_BACKOFF = timedelta(seconds=30)
_FRONTIER_WORK_LIMIT = 4_096

_PENDING_ROLLUPS_SQL = """
    SELECT period_kind, period_start
    FROM lcm_rollups
    WHERE scope = ?
      AND status = 'stale'
      AND period_kind = 'day'
    ORDER BY period_start
    LIMIT ?
"""

_PENDING_AGGREGATES_SQL = """
    SELECT period_kind, period_start
    FROM lcm_rollups
    WHERE scope = ?
      AND status = 'stale'
      AND period_kind IN ('week', 'month')
    ORDER BY period_start, period_kind
    LIMIT ?
"""

_FAILED_ROLLUPS_SQL = """
    SELECT period_kind, period_start
    FROM lcm_rollups
    WHERE scope = ?
      AND status = 'failed'
      AND period_kind = 'day'
      AND (failed_at IS NULL OR failed_at <= ?)
    ORDER BY failed_at, period_start, period_kind
    LIMIT ?
"""

_FAILED_AGGREGATES_SQL = """
    SELECT period_kind, period_start
    FROM lcm_rollups
    WHERE scope = ?
      AND status = 'failed'
      AND period_kind IN ('week', 'month')
      AND (failed_at IS NULL OR failed_at <= ?)
    ORDER BY failed_at, period_start, period_kind
    LIMIT ?
"""


class RollupWorkLimitExceeded(RuntimeError):
    """Raised when correctness would require more than one bounded work pass."""


def initialize_rollup_invalidation_outbox(dag: SummaryDAG) -> None:
    """Enable mutation triggers before the feature can publish summary nodes."""
    store = RollupStore(dag.db_path)
    store.close()


def _as_date(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scope_frontier(dag: SummaryDAG, scope: str) -> list[dict[str, object]]:
    """Load a scope frontier without retaining its TEMP-staging snapshot."""
    with dag._db_lock:
        connection = dag.connection
        if connection is None:
            return []
        with _sqlite_savepoint(connection):
            return _scope_frontier_staged(dag, scope)


def _scope_frontier_staged(
    dag: SummaryDAG, scope: str
) -> list[dict[str, object]]:
    """The scope's canonical frontier nodes with normalized interval bounds.

    Loads every summary node for ``scope`` and applies the shared interval-aware
    :func:`canonical_frontier`: a node condensed by a higher-depth parent anywhere
    in the scope is suppressed, so a parent that spans several days stands in for
    the children it covers even when those children land on adjacent days. Each
    SQL enumeration is capped; exceeding the cap fails closed instead of silently
    returning a non-canonical prefix.
    """
    # All temp tables below belong to this SQLite connection, not to a thread.
    # Keep the probe, staging table, ordered read, and transitive-lineage temp
    # walk under one DAG lock so concurrent scopes cannot clear or replace one
    # another's staged IDs.
    with dag._db_lock:
        connection = dag.connection
        if connection is None:
            return []
        # Probe without an expression ORDER BY first. The session index can stop
        # at the sentinel row, so an oversized scope never scans/sorts its corpus.
        id_rows = connection.execute(
            "SELECT node_id FROM summary_nodes WHERE session_id = ? LIMIT ?",
            (scope, _FRONTIER_WORK_LIMIT + 1),
        ).fetchall()
        if len(id_rows) > _FRONTIER_WORK_LIMIT:
            raise RollupWorkLimitExceeded(
                f"scope frontier exceeds bounded work limit ({_FRONTIER_WORK_LIMIT})"
            )
        if not id_rows:
            return []

        connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS lcm_scope_frontier_ids "
            "(node_id INTEGER PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute("DELETE FROM temp.lcm_scope_frontier_ids")
        try:
            connection.executemany(
                "INSERT INTO temp.lcm_scope_frontier_ids(node_id) VALUES(?)",
                ((int(row[0]),) for row in id_rows),
            )
            # Any sort below is over a set already proven to contain at most
            # 4,096 rows. The temp-table join also avoids a dynamic IN-list.
            rows = connection.execute(
                """
                SELECT node.node_id, node.depth, node.summary, node.source_ids,
                       node.source_type,
                       COALESCE(node.earliest_at, node.created_at) AS earliest_at,
                       COALESCE(node.latest_at, node.created_at) AS latest_at
                FROM temp.lcm_scope_frontier_ids wanted
                JOIN summary_nodes node ON node.node_id = wanted.node_id
                ORDER BY COALESCE(node.latest_at, node.created_at), node.node_id
                """
            ).fetchall()
        finally:
            connection.execute("DELETE FROM temp.lcm_scope_frontier_ids")

        candidates: list[CoverageNode] = []
        meta: dict[int, dict[str, object]] = {}
        for row in rows:
            node_id = int(row[0])
            source_type = str(row[4] or "")
            source_node_ids: tuple[int, ...] = ()
            if source_type == "nodes" and row[3]:
                try:
                    source_node_ids = tuple(
                        int(value) for value in json.loads(row[3])
                    )
                except (TypeError, ValueError):
                    source_node_ids = ()
            latest_at = row[6]
            candidates.append(
                CoverageNode(
                    node_id=node_id,
                    depth=int(row[1] or 0),
                    source_node_ids=source_node_ids,
                    earliest_at=row[5],
                    latest_at=latest_at,
                )
            )
            earliest_at = row[5]
            covered_start = float(earliest_at) if earliest_at is not None else None
            covered_end = float(latest_at) if latest_at is not None else None
            if (
                covered_start is not None
                and covered_end is not None
                and covered_end < covered_start
            ):
                covered_start, covered_end = covered_end, covered_start
            meta[node_id] = {
                "summary": str(row[2] or ""),
                "covered_start": covered_start,
                "covered_end": covered_end,
            }
        try:
            source_lineage = load_source_lineage(
                connection,
                [candidate.node_id for candidate in candidates],
                limit=_FRONTIER_WORK_LIMIT,
            )
        except RuntimeError as exc:
            raise RollupWorkLimitExceeded(str(exc)) from exc

        frontier: list[dict[str, object]] = []
        for node in canonical_frontier(candidates, source_lineage=source_lineage):
            info = meta[node.node_id]
            frontier.append(
                {
                    "node_id": node.node_id,
                    "summary": info["summary"],
                    "covered_start": info["covered_start"],
                    "covered_end": info["covered_end"],
                }
            )
        return frontier


def _daily_sources(dag: SummaryDAG, scope: str, day: date) -> list[dict[str, object]]:
    """Return canonical frontier nodes whose full interval intersects ``day``.

    A condensed child and its condensing parent must not both feed rollups: the
    parent already covers the child's lineage. Crucially the parent may land on a
    DIFFERENT day than the child (its span crosses midnight), so the suppression
    is computed over the whole scope's frontier, not just this day's rows — a
    child on Jul15 covered by a parent whose representative day is Jul16 is
    suppressed from Jul15, and the parent feeds every UTC day its full interval
    intersects, so adjacent dailies use one canonical lineage (maintainer #388
    B1 / #389 source-dedup).
    """
    day_start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp()
    day_end = day_start + timedelta(days=1).total_seconds()
    return [
        {"node_id": node["node_id"], "summary": node["summary"]}
        for node in _scope_frontier(dag, scope)
        if node["covered_start"] is not None
        and node["covered_end"] is not None
        and float(node["covered_start"]) < day_end
        and float(node["covered_end"]) >= day_start
    ]


def _summary_controls(config: LCMConfig) -> dict[str, object]:
    return {
        "model": config.summary_model,
        "reasoning_effort": config.summary_reasoning_effort,
        "timeout": config.summary_timeout_ms / 1000.0,
        "l2_budget_ratio": config.l2_budget_ratio,
        "custom_instructions": config.custom_instructions,
        "fallback_models": config.summary_fallback_models,
    }


def _summarize_capped(
    text: str,
    *,
    target_tokens: int,
    max_tokens: int,
    config: LCMConfig,
    summarizer: Summarizer,
    circuit_breaker: object | None,
    spend_guard: object | None,
) -> tuple[str, int]:
    """Use escalation until the result is within the configured hard cap."""
    target = max(1, min(int(target_tokens), int(max_tokens)))
    hard_max = max(1, int(max_tokens))
    candidate = text
    previous_tokens = count_tokens(candidate)

    while True:
        summary, _level = summarizer(
            candidate,
            source_tokens=max(1, previous_tokens),
            token_budget=target,
            l3_truncate_tokens=hard_max,
            circuit_breaker=circuit_breaker,
            spend_guard=spend_guard,
            **_summary_controls(config),
        )
        summary = str(summary)
        summary_tokens = count_tokens(summary)
        if summary_tokens <= hard_max:
            return summary, summary_tokens
        if summary_tokens >= previous_tokens:
            truncated = _deterministic_truncate(summary, hard_max)
            return truncated, count_tokens(truncated)
        candidate = summary
        previous_tokens = summary_tokens


def _mark_failed(
    store: RollupStore, token: "RollupBuildToken | None", exc: Exception
) -> None:
    if token is not None:
        try:
            # Generation-guarded: a superseded builder's late exception must not
            # flip a newer ready/stale row to failed (maintainer #387 blocker 2).
            store.mark_failed(token, f"{type(exc).__name__}: {exc}")
        except Exception:
            logger.debug("LCM temporal rollup failure state could not be persisted", exc_info=True)
    logger.debug("LCM temporal rollup build failed", exc_info=True)


def build_day(
    store: RollupStore,
    dag: SummaryDAG,
    config: LCMConfig,
    scope: str,
    period_date: date | str,
    *,
    summarizer: Summarizer | None = None,
    circuit_breaker: object | None = None,
    spend_guard: object | None = None,
) -> dict[str, object] | None:
    """Build one UTC daily rollup without allowing failures into the caller."""
    token: RollupBuildToken | None = None
    try:
        summarizer = summarizer or summarize_with_escalation
        day = _as_date(period_date)
        store.drain_invalidations(event_limit=256, day_budget=256)
        # Capture the build token (advancing the generation) BEFORE reading the
        # source snapshot, so an invalidation that arrives while we build is
        # guaranteed to supersede this token's mark_ready — a snapshot read
        # before the claim could otherwise be published stale (maintainer #388
        # blocker: capture-token-first).
        token = store.upsert_building("day", day.isoformat(), scope)
        sources = _daily_sources(dag, scope, day)
        if not sources:
            # A stale day with no summary node to build from must resolve, not
            # linger stale forever consuming a per-pass build slot (maintainer
            # #388 blocker: no-source jobs).
            store.resolve_no_source(token)
            return None

        source_ids = sorted(int(source["node_id"]) for source in sources)
        fingerprint = _stable_hash(
            [
                [int(source["node_id"]), hashlib.sha256(str(source["summary"]).encode("utf-8")).hexdigest()]
                for source in sorted(sources, key=lambda source: int(source["node_id"]))
            ]
        )
        text = "\n\n".join(
            f"[Summary node {source['node_id']}]\n{source['summary']}"
            for source in sources
        )
        summary, token_count = _summarize_capped(
            text,
            target_tokens=config.rollup_daily_target_tokens,
            max_tokens=config.rollup_daily_max_tokens,
            config=config,
            summarizer=summarizer,
            circuit_breaker=circuit_breaker,
            spend_guard=spend_guard,
        )
        published = store.mark_ready(token, summary, token_count, source_ids, fingerprint)
        if published:
            # A (re)built daily makes any already-published week/month that
            # covered the previous daily outdated: stale them so they rebuild
            # from the new daily (maintainer #388 blocker 5 — aggregate rebuild).
            store.stale_aggregates_for_day(day, scope)
        return store.get_rollup("day", day.isoformat(), scope)
    except Exception as exc:
        _mark_failed(store, token, exc)
        return None


def _period_window(period_kind: str, period_start: date | str) -> tuple[date, date]:
    start = _as_date(period_start)
    if period_kind == "week":
        start -= timedelta(days=start.weekday())
        return start, start + timedelta(days=6)
    if period_kind == "month":
        start = start.replace(day=1)
        return start, start.replace(day=monthrange(start.year, start.month)[1])
    raise ValueError(f"unsupported aggregate period: {period_kind}")


def _daily_statuses(
    store: RollupStore,
    start: date,
    end: date,
    scope: str,
) -> dict[str, dict[str, object]]:
    connection = store.connection
    if connection is None:
        return {}
    rows = connection.execute(
        """
        SELECT period_start, status, source_fingerprint, summary, token_count, rollup_id
        FROM lcm_rollups
        WHERE period_kind = 'day'
          AND period_start >= ?
          AND period_start <= ?
          AND scope = ?
        ORDER BY period_start
        """,
        (start.isoformat(), end.isoformat(), scope),
    ).fetchall()
    return {
        str(row["period_start"]): {
            "status": str(row["status"]),
            "source_fingerprint": row["source_fingerprint"],
            "summary": row["summary"],
            "token_count": row["token_count"],
            "rollup_id": int(row["rollup_id"]),
        }
        for row in rows
    }


def _days_with_content(
    dag: SummaryDAG,
    scope: str,
    start: date,
    end: date,
) -> set[str]:
    """UTC days in ``[start, end]`` that have canonical-frontier content for ``scope``.

    A rollup consumes the scope's canonical frontier; a day counts as having
    content when a frontier node's full interval intersects it. This MUST use
    the same frontier as :func:`_daily_sources`: a day whose only node is a child
    suppressed by a multi-day parent has NO daily to build, so it must not count
    as content and block the aggregate's completeness gate (maintainer #388 B1).
    A day with no frontier content legitimately has no daily rollup.
    """
    frontier = _scope_frontier(dag, scope)
    result: set[str] = set()
    current = start
    while current <= end:
        day_start = datetime.combine(
            current, datetime.min.time(), tzinfo=timezone.utc
        ).timestamp()
        day_end = day_start + timedelta(days=1).total_seconds()
        if any(
            node["covered_start"] is not None
            and node["covered_end"] is not None
            and float(node["covered_start"]) < day_end
            and float(node["covered_end"]) >= day_start
            for node in frontier
        ):
            result.add(current.isoformat())
        current += timedelta(days=1)
    return result


def _rollup_source_ids(store: RollupStore, rollup_ids: Sequence[int]) -> list[int]:
    if not rollup_ids or store.connection is None:
        return []
    placeholders = ",".join("?" for _ in rollup_ids)
    rows = store.connection.execute(
        f"""
        SELECT DISTINCT node_id
        FROM lcm_rollup_sources
        WHERE rollup_id IN ({placeholders})
        ORDER BY node_id
        """,
        [int(rollup_id) for rollup_id in rollup_ids],
    ).fetchall()
    return [int(row[0]) for row in rows]


def _canonical_aggregate_sources(
    dag: SummaryDAG, source_ids: Sequence[int]
) -> list[dict[str, object]]:
    """Resolve aggregate sources without retaining a TEMP-staging snapshot."""
    with dag._db_lock:
        connection = dag.connection
        if connection is None:
            return []
        with _sqlite_savepoint(connection):
            return _canonical_aggregate_sources_staged(dag, source_ids)


def _canonical_aggregate_sources_staged(
    dag: SummaryDAG, source_ids: Sequence[int]
) -> list[dict[str, object]]:
    """Resolve one bounded canonical node frontier for aggregate publication."""
    unique_ids = list(dict.fromkeys(int(node_id) for node_id in source_ids))
    if len(unique_ids) > _FRONTIER_WORK_LIMIT:
        raise RollupWorkLimitExceeded(
            f"aggregate frontier exceeds bounded work limit ({_FRONTIER_WORK_LIMIT})"
        )
    if not unique_ids:
        return []
    # Aggregate staging and the shared lineage temp walk use the same connection
    # and therefore participate in the same serialization contract as scopes.
    with dag._db_lock:
        connection = dag.connection
        if connection is None:
            return []
        connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS lcm_aggregate_source_ids "
            "(node_id INTEGER PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute("DELETE FROM temp.lcm_aggregate_source_ids")
        try:
            connection.executemany(
                "INSERT INTO temp.lcm_aggregate_source_ids(node_id) VALUES(?)",
                ((node_id,) for node_id in unique_ids),
            )
            rows = connection.execute(
                """
                SELECT node.node_id, node.depth, node.summary, node.source_ids,
                       node.source_type,
                       COALESCE(node.earliest_at, node.created_at),
                       COALESCE(node.latest_at, node.created_at)
                FROM temp.lcm_aggregate_source_ids wanted
                JOIN summary_nodes node ON node.node_id = wanted.node_id
                ORDER BY node.node_id
                """
            ).fetchall()
            source_lineage = load_source_lineage(
                connection, unique_ids, limit=_FRONTIER_WORK_LIMIT
            )
        except RuntimeError as exc:
            raise RollupWorkLimitExceeded(str(exc)) from exc
        finally:
            connection.execute("DELETE FROM temp.lcm_aggregate_source_ids")
    candidates: list[CoverageNode] = []
    summaries: dict[int, str] = {}
    for row in rows:
        parent_sources: tuple[int, ...] = ()
        if str(row[4] or "") == "nodes" and row[3]:
            try:
                parent_sources = tuple(int(value) for value in json.loads(row[3]))
            except (TypeError, ValueError):
                parent_sources = ()
        node_id = int(row[0])
        candidates.append(
            CoverageNode(
                node_id=node_id,
                depth=int(row[1] or 0),
                source_node_ids=parent_sources,
                earliest_at=row[5],
                latest_at=row[6],
            )
        )
        summaries[node_id] = str(row[2] or "")
    return [
        {"node_id": node.node_id, "summary": summaries[node.node_id]}
        for node in canonical_frontier(candidates, source_lineage=source_lineage)
    ]


def _build_aggregate(
    period_kind: str,
    store: RollupStore,
    dag: SummaryDAG,
    config: LCMConfig,
    scope: str,
    period_start: date | str,
    *,
    summarizer: Summarizer | None,
    circuit_breaker: object | None,
    spend_guard: object | None,
) -> dict[str, object] | None:
    token: RollupBuildToken | None = None
    try:
        summarizer = summarizer or summarize_with_escalation
        start, end = _period_window(period_kind, period_start)
        store.drain_invalidations(event_limit=256, day_budget=256)
        # Capture the build token (advancing the generation) BEFORE reading the
        # daily-status snapshot, so a daily (re)build that stales this aggregate
        # while we build supersedes this token's mark_ready — reading dailies
        # before the claim could otherwise publish an aggregate that omits a
        # just-rebuilt daily (maintainer #388 blocker: capture-token-first).
        token = store.upsert_building(period_kind, start.isoformat(), scope)
        statuses = _daily_statuses(store, start, end, scope)

        # Completeness gate (maintainer #388 blocker 5): only publish a ready
        # aggregate when every day that HAS content in the window has a ready
        # daily rollup. A content day that is missing/stale/building blocks the
        # aggregate, which is released back to stale (token-guarded) with a
        # recorded reason so it rebuilds once the daily catches up. Days with no
        # content do not block.
        content_days = _days_with_content(dag, scope, start, end)
        pending_days = sorted(
            day_key
            for day_key in content_days
            if str((statuses.get(day_key) or {}).get("status")) != "ready"
        )
        if pending_days:
            preview = ", ".join(pending_days[:5])
            store.defer_incomplete(
                token,
                f"incomplete: {len(pending_days)} daily rollup(s) not ready ({preview})",
            )
            return None

        days: list[dict[str, object]] = []
        ready: list[tuple[str, dict[str, object]]] = []
        current = start
        while current <= end:
            day_key = current.isoformat()
            row = statuses.get(day_key)
            status = str(row["status"]) if row else "missing"
            fingerprint_value: object = status
            if row and status == "ready":
                fingerprint_value = row.get("source_fingerprint") or _stable_hash(row.get("summary") or "")
                ready.append((day_key, row))
            days.append({"day": day_key, "status": status, "fingerprint": fingerprint_value})
            current += timedelta(days=1)

        if not ready:
            # No ready constituent daily to aggregate: clear the claimed row so
            # it does not linger stale consuming a build slot (a later daily
            # (re)build re-stales the aggregate via stale_aggregates_for_day).
            store.resolve_no_source(token)
            return None

        fingerprint = _stable_hash(days)
        constituent_source_ids = _rollup_source_ids(
            store,
            [int(row["rollup_id"]) for _day_key, row in ready],
        )
        frontier_sources = _canonical_aggregate_sources(dag, constituent_source_ids)
        if not frontier_sources:
            store.defer_incomplete(token, "incomplete: no canonical aggregate sources")
            return None
        source_ids = [int(source["node_id"]) for source in frontier_sources]
        text = "\n\n".join(
            f"[Summary node {source['node_id']}]\n{source['summary']}"
            for source in frontier_sources
        )
        summary, token_count = _summarize_capped(
            text,
            target_tokens=config.rollup_aggregate_max_tokens,
            max_tokens=config.rollup_aggregate_max_tokens,
            config=config,
            summarizer=summarizer,
            circuit_breaker=circuit_breaker,
            spend_guard=spend_guard,
        )
        store.mark_ready(token, summary, token_count, source_ids, fingerprint)
        return store.get_rollup(period_kind, start.isoformat(), scope)
    except Exception as exc:
        _mark_failed(store, token, exc)
        return None


def build_week(
    store: RollupStore,
    dag: SummaryDAG,
    config: LCMConfig,
    scope: str,
    period_start: date | str,
    *,
    summarizer: Summarizer | None = None,
    circuit_breaker: object | None = None,
    spend_guard: object | None = None,
) -> dict[str, object] | None:
    return _build_aggregate(
        "week", store, dag, config, scope, period_start,
        summarizer=summarizer,
        circuit_breaker=circuit_breaker,
        spend_guard=spend_guard,
    )


def build_month(
    store: RollupStore,
    dag: SummaryDAG,
    config: LCMConfig,
    scope: str,
    period_start: date | str,
    *,
    summarizer: Summarizer | None = None,
    circuit_breaker: object | None = None,
    spend_guard: object | None = None,
) -> dict[str, object] | None:
    return _build_aggregate(
        "month", store, dag, config, scope, period_start,
        summarizer=summarizer,
        circuit_breaker=circuit_breaker,
        spend_guard=spend_guard,
    )


# NOTE: there is deliberately no ``mark_stale_after_ingest`` raw-ingest hook.
# Staleness is driven SOLELY by summary-node publication
# (:func:`mark_stale_for_published_summary`, wired at every ``_dag.add_node``
# site). A raw-ingest bind-time mark would stale a period before its covering
# summary exists, letting a rebuild publish ``ready`` from the OLD sources and
# omit the not-yet-published leaf (maintainer #388 P1). Rollups consume
# published summary nodes, so publication is the only correct trigger.


def mark_stale_for_published_summary(
    dag: SummaryDAG,
    scope: str,
    latest_at: float | None,
    created_at: float | None = None,
    *,
    earliest_at: float | None = None,
) -> int:
    """Invalidate the rollups for EVERY UTC day a published summary node covers.

    Rollups consume PUBLISHED summary nodes, not raw messages, so publication is
    the load-bearing staleness signal (maintainer #388 blocker 1): when a summary
    covering days D..D' is published, each of those days and their containing
    week/month must go stale so a later summary cannot leave an older rollup
    ``ready`` and apparently current. A summary whose coverage spans past midnight
    covers more than one day; keying only on ``latest_at`` left the earlier day(s)
    ``ready`` (maintainer #388 B2). The covered span is
    ``[earliest_at, latest_at]`` (each falling back to ``created_at``); the shared
    durable outbox drains the interval in bounded UTC-day chunks.
    """
    store: RollupStore | None = None
    try:
        if not scope:
            return 0
        store = RollupStore(dag.db_path)
        return store.drain_invalidations(event_limit=256, day_budget=256) * 3
    except Exception:
        logger.debug("LCM temporal rollup publication staleness update failed", exc_info=True)
        return 0
    finally:
        if store is not None:
            store.close()


def mark_stale_for_deleted_nodes(dag: SummaryDAG, node_ids: Sequence[int]) -> int:
    """Compatibility wrapper: mutation triggers own deletion invalidation."""
    store: RollupStore | None = None
    try:
        store = RollupStore(dag.db_path)
        before = store.has_pending_invalidations()
        drained = store.drain_invalidations(event_limit=256, day_budget=256)
        return drained if before else 0
    except Exception:
        logger.debug("LCM temporal rollup deletion staleness update failed", exc_info=True)
        return 0
    finally:
        if store is not None:
            store.close()


def run_rollup_maintenance(
    dag: SummaryDAG,
    config: LCMConfig,
    scope: str,
    *,
    circuit_breaker: object | None = None,
    spend_guard: object | None = None,
) -> int:
    """Best-effort bounded maintenance; slow summarizers may leave rollups lagging."""
    store: RollupStore | None = None
    started_at = monotonic()
    try:
        limit = max(0, int(config.rollup_builds_per_pass))
        budget_ms = max(0, int(config.rollup_maintenance_budget_ms))
        connection = dag.connection
        if limit <= 0 or budget_ms <= 0 or connection is None:
            return 0
        store = RollupStore(dag.db_path)
        if (monotonic() - started_at) * 1000 >= budget_ms:
            return 0
        store.drain_invalidations(event_limit=256, day_budget=256)
        if (monotonic() - started_at) * 1000 >= budget_ms:
            return 0
        # Reclaim rows whose build lease expired (a crashed builder left them
        # 'building' forever) back to 'stale' so this pass can retry them
        # (maintainer #388 blocker 2).
        store.reclaim_stale_building(limit=256)
        if (monotonic() - started_at) * 1000 >= budget_ms:
            return 0
        retry_before = (datetime.now(timezone.utc) - _FAILED_ROLLUP_BACKOFF).isoformat()
        stale_rows = connection.execute(
            _PENDING_ROLLUPS_SQL,
            (scope, limit),
        ).fetchall()
        rows = list(stale_rows)
        if len(rows) < limit and (monotonic() - started_at) * 1000 < budget_ms:
            rows.extend(
                connection.execute(
                    _FAILED_ROLLUPS_SQL,
                    (scope, retry_before, limit - len(rows)),
                ).fetchall()
            )
        if len(rows) < limit and (monotonic() - started_at) * 1000 < budget_ms:
            rows.extend(
                connection.execute(
                    _PENDING_AGGREGATES_SQL,
                    (scope, limit - len(rows)),
                ).fetchall()
            )
        if len(rows) < limit and (monotonic() - started_at) * 1000 < budget_ms:
            rows.extend(
                connection.execute(
                    _FAILED_AGGREGATES_SQL,
                    (scope, retry_before, limit - len(rows)),
                ).fetchall()
            )
        if not rows:
            return 0
        builders: dict[str, Callable[..., dict[str, object] | None]] = {
            "day": build_day,
            "week": build_week,
            "month": build_month,
        }
        builds_started = 0
        for row in rows:
            if (monotonic() - started_at) * 1000 >= budget_ms:
                break
            builder = builders[str(row[0])]
            builder(
                store,
                dag,
                config,
                scope,
                str(row[1]),
                circuit_breaker=circuit_breaker,
                spend_guard=spend_guard,
            )
            builds_started += 1
        return builds_started
    except Exception:
        logger.debug("LCM temporal rollup maintenance failed", exc_info=True)
        return 0
    finally:
        if store is not None:
            store.close()
