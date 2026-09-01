from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.rollup_builder as builder_module
import hermes_lcm.rollup_periods as periods_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.rollup_builder import (
    _PENDING_ROLLUPS_SQL,
    build_day,
    build_month,
    build_week,
    run_rollup_maintenance,
)
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.tokens import count_tokens


@pytest.fixture
def rollup_parts(tmp_path):
    db_path = tmp_path / "rollup-builder.db"
    dag = SummaryDAG(db_path)
    store = RollupStore(db_path)
    config = LCMConfig(
        database_path=str(db_path),
        rollup_daily_target_tokens=12,
        rollup_daily_max_tokens=20,
        rollup_aggregate_max_tokens=30,
    )
    try:
        yield store, dag, config
    finally:
        store.close()
        dag.close()


def _timestamp(day: date, hour: int = 12) -> float:
    return datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).timestamp()


def _add_node(
    dag: SummaryDAG,
    scope: str,
    day: date,
    summary: str,
    *,
    depth: int = 0,
    latest_day: date | None = None,
) -> int:
    latest = latest_day or day
    return dag.add_node(
        SummaryNode(
            session_id=scope,
            depth=depth,
            summary=summary,
            token_count=count_tokens(summary),
            source_token_count=count_tokens(summary) * 2,
            source_ids=[depth + 1],
            source_type="messages" if depth == 0 else "nodes",
            created_at=_timestamp(day, 18),
            earliest_at=_timestamp(day, 8),
            latest_at=_timestamp(latest, 22),
        )
    )


def _ready(
    store: RollupStore,
    kind: str,
    start: str,
    scope: str,
    *,
    summary: str,
    source_ids: list[int],
    fingerprint: str,
) -> int:
    store.drain_invalidations(event_limit=256, day_budget=256)
    token = store.upsert_building(kind, start, scope)
    store.mark_ready(
        token,
        summary,
        count_tokens(summary),
        source_ids,
        fingerprint,
    )
    return token.rollup_id


def test_build_day_uses_newest_source_day_and_mocked_summarizer(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-a"
    target_day = date(2026, 7, 15)
    first = _add_node(dag, scope, target_day, "leaf fixture summary")
    second = _add_node(
        dag,
        scope,
        target_day - timedelta(days=1),
        "second fixture summary",
        latest_day=target_day,
    )
    _add_node(dag, scope, target_day - timedelta(days=1), "older summary")
    calls = []

    def summarize(text, **kwargs):
        calls.append((text, kwargs))
        return "deterministic daily rollup", 1

    result = build_day(store, dag, config, scope, target_day, summarizer=summarize)

    assert result is not None
    assert result["summary"] == "deterministic daily rollup"
    assert result["token_count"] == count_tokens("deterministic daily rollup")
    assert result["source_node_ids"] == [first, second]
    assert "leaf fixture summary" in calls[0][0]
    assert "second fixture summary" in calls[0][0]
    assert "older summary" not in calls[0][0]
    assert calls[0][1]["token_budget"] == config.rollup_daily_target_tokens
    assert calls[0][1]["l3_truncate_tokens"] == config.rollup_daily_max_tokens


def test_build_day_honors_target_and_hard_cap_after_oversize_result(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-budget"
    target_day = date(2026, 7, 15)
    config.rollup_daily_target_tokens = 8
    config.rollup_daily_max_tokens = 10
    _add_node(dag, scope, target_day, "source material " * 100)
    calls = []

    def summarize(text, **kwargs):
        calls.append((text, kwargs))
        if len(calls) == 1:
            return "oversize " * 30, 1
        return "bounded fallback", 3

    result = build_day(store, dag, config, scope, target_day, summarizer=summarize)

    assert result is not None
    assert result["summary"] == "bounded fallback"
    assert result["token_count"] <= config.rollup_daily_max_tokens
    assert len(calls) == 2
    assert all(call[1]["token_budget"] == config.rollup_daily_target_tokens for call in calls)
    assert all(call[1]["l3_truncate_tokens"] == config.rollup_daily_max_tokens for call in calls)


def test_week_requires_all_content_dailies_ready_before_publishing(rollup_parts):
    # Both Monday and Tuesday have summary-node content, so a week must not
    # publish while Tuesday's daily is stale; it publishes only once every
    # content day has a ready daily (maintainer #388 blocker 5).
    store, dag, config = rollup_parts
    scope = "session-aggregate"
    monday = date(2026, 7, 13)
    tuesday = monday + timedelta(days=1)
    monday_node = _add_node(dag, scope, monday, "monday node")
    tuesday_node = _add_node(dag, scope, tuesday, "tuesday node")
    _ready(
        store, "day", monday.isoformat(), scope,
        summary="ready monday daily", source_ids=[monday_node], fingerprint="monday-v1",
    )
    _ready(
        store, "day", tuesday.isoformat(), scope,
        summary="stale tuesday daily", source_ids=[tuesday_node], fingerprint="tuesday-v1",
    )
    store.mark_stale_for_day(tuesday, scope)
    seen_text = []

    def summarize(text, **_kwargs):
        seen_text.append(text)
        return f"aggregate version {len(seen_text)}", 1

    # Tuesday is a content day but not ready -> the week is left incomplete.
    blocked = build_week(store, dag, config, scope, monday, summarizer=summarize)
    assert blocked is None
    assert seen_text == []
    incomplete_row = store.get_rollup("week", monday.isoformat(), scope)
    assert incomplete_row["status"] == "stale"
    assert "incomplete" in (incomplete_row["error"] or "")

    # Rebuild Tuesday's daily; now every content day is ready and the week builds.
    rebuilt_daily = build_day(
        store, dag, config, scope, tuesday,
        summarizer=lambda _text, **_kwargs: ("ready tuesday rebuilt", 1),
    )
    assert rebuilt_daily is not None
    built = build_week(store, dag, config, scope, monday, summarizer=summarize)

    assert built is not None
    assert "monday node" in seen_text[0]
    assert "tuesday node" in seen_text[0]
    assert built["source_node_ids"] == [monday_node, tuesday_node]
    assert built["status"] == "ready"


def test_rebuilding_a_daily_stales_its_containing_week_and_month(rollup_parts):
    # A daily rebuild must invalidate any already-published week/month so they
    # never stay ready against an outdated day (maintainer #388 blocker 5).
    store, dag, config = rollup_parts
    scope = "session-cascade"
    monday = date(2026, 7, 13)
    monday_node = _add_node(dag, scope, monday, "monday node")
    _ready(
        store, "week", monday.isoformat(), scope,
        summary="published week", source_ids=[monday_node], fingerprint="week-v1",
    )
    _ready(
        store, "month", date(2026, 7, 1).isoformat(), scope,
        summary="published month", source_ids=[monday_node], fingerprint="month-v1",
    )

    rebuilt = build_day(
        store, dag, config, scope, monday,
        summarizer=lambda _text, **_kwargs: ("fresh monday daily", 1),
    )
    assert rebuilt is not None
    assert store.get_rollup("week", monday.isoformat(), scope)["status"] == "stale"
    assert store.get_rollup("month", date(2026, 7, 1).isoformat(), scope)["status"] == "stale"


def test_month_aggregate_never_queries_dag_when_ready_dailies_exist(rollup_parts, monkeypatch):
    store, dag, config = rollup_parts
    scope = "session-month"
    month_start = date(2026, 7, 1)
    node_id = _add_node(dag, scope, month_start, "first source")
    _ready(
        store,
        "day",
        month_start.isoformat(),
        scope,
        summary="first daily",
        source_ids=[node_id],
        fingerprint="day-one",
    )
    monkeypatch.setattr(
        dag,
        "get_session_nodes",
        lambda *_args, **_kwargs: pytest.fail("aggregate queried DAG nodes"),
    )

    result = build_month(
        store,
        dag,
        config,
        scope,
        month_start,
        summarizer=lambda _text, **_kwargs: ("monthly", 1),
    )

    assert result is not None
    assert result["source_node_ids"] == [node_id]


def test_builder_failure_is_marked_failed_and_does_not_raise(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-failure"
    target_day = date(2026, 7, 15)
    _add_node(dag, scope, target_day, "source summary")

    def fail(_text, **_kwargs):
        raise RuntimeError("summarizer unavailable")

    result = build_day(store, dag, config, scope, target_day, summarizer=fail)

    assert result is None
    failed = store.get_rollup("day", target_day.isoformat(), scope)
    assert failed is not None
    assert failed["status"] == "failed"
    assert "summarizer unavailable" in failed["error"]


def test_empty_builds_have_zero_store_side_effects(rollup_parts):
    store, dag, config = rollup_parts

    assert build_day(store, dag, config, "empty", date(2026, 7, 15)) is None
    assert build_week(store, dag, config, "empty", date(2026, 7, 13)) is None
    assert store.connection.execute("SELECT COUNT(*) FROM lcm_rollups").fetchone()[0] == 0


def test_publication_staleness_and_bounded_bind_maintenance(tmp_path, monkeypatch):
    # Raw ingest ALONE must not stale rollups (maintainer #388 P1): a period is
    # staled only when a covering summary node is PUBLISHED. Then background
    # bind maintenance rebuilds up to rollup_builds_per_pass targets, leaving the rest
    # durably stale for the next pass.
    db_path = tmp_path / "engine-rollups.db"
    config = LCMConfig(
        database_path=str(db_path),
        temporal_rollups_enabled=True,
        rollup_builds_per_pass=2,
    )
    engine = LCMEngine(config=config)
    scope = "temporal-session"
    today = datetime.now(timezone.utc).date()
    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    try:
        engine.on_session_start(scope, conversation_id="temporal-conversation")
        assert engine.drain_rollup_maintenance(timeout=2)
        node_id = _add_node(engine._dag, scope, today, "today source node")
        store = RollupStore(db_path)
        try:
            for kind, start in (
                ("day", today),
                ("week", week_start),
                ("month", month_start),
            ):
                _ready(
                    store,
                    kind,
                    start.isoformat(),
                    scope,
                    summary=f"old {kind}",
                    source_ids=[node_id],
                    fingerprint=f"old-{kind}",
                )

            # Raw ingest does not stale rollups: no covering summary was published.
            engine.ingest([{"role": "user", "content": "raw ingest alone must not stale"}])
            assert [
                store.get_rollup(kind, start.isoformat(), scope)["status"]
                for kind, start in (("day", today), ("week", week_start), ("month", month_start))
            ] == ["ready", "ready", "ready"]

            # Publishing a later summary covering today is the staleness signal.
            later_id = _add_node(engine._dag, scope, today, "later source node")
            engine._invalidate_rollups_for_published_node(engine._dag.get_node(later_id))
            assert [
                store.get_rollup(kind, start.isoformat(), scope)["status"]
                for kind, start in (("day", today), ("week", week_start), ("month", month_start))
            ] == ["stale", "stale", "stale"]

            monkeypatch.setattr(
                builder_module,
                "summarize_with_escalation",
                lambda _text, **_kwargs: ("rebuilt rollup", 1),
            )
            engine._bind_lifecycle_state(scope, conversation_id="temporal-conversation")
            assert engine.drain_rollup_maintenance(timeout=2)

            statuses = [
                store.get_rollup(kind, start.isoformat(), scope)["status"]
                for kind, start in (("day", today), ("week", week_start), ("month", month_start))
            ]
            assert statuses.count("ready") == 2
            assert statuses.count("stale") == 1
            assert statuses[0] == "ready"
        finally:
            store.close()
    finally:
        engine.shutdown()


def test_never_built_stale_day_is_automatically_built(rollup_parts, monkeypatch):
    store, dag, config = rollup_parts
    scope = "session-never-built"
    target_day = date(2026, 7, 15)
    _add_node(dag, scope, target_day, "source for first automatic build")
    store.mark_stale_for_day(target_day, scope)
    config.rollup_builds_per_pass = 1
    monkeypatch.setattr(
        builder_module,
        "summarize_with_escalation",
        lambda _text, **_kwargs: ("first daily rollup", 1),
    )

    assert run_rollup_maintenance(dag, config, scope) == 1
    assert store.get_rollup("day", target_day.isoformat(), scope)["status"] == "ready"


def test_failed_rollup_retry_honors_backoff(rollup_parts, monkeypatch):
    store, dag, config = rollup_parts
    scope = "session-retry"
    old_day = date(2026, 7, 14)
    recent_day = date(2026, 7, 15)
    _add_node(dag, scope, old_day, "old failed source")
    _add_node(dag, scope, recent_day, "recent failed source")
    store.drain_invalidations(event_limit=256, day_budget=256)
    old_token = store.upsert_building("day", old_day.isoformat(), scope)
    recent_token = store.upsert_building("day", recent_day.isoformat(), scope)
    old_id = old_token.rollup_id
    store.mark_failed(old_token, "old failure")
    store.mark_failed(recent_token, "recent failure")
    store.connection.execute(
        "UPDATE lcm_rollups SET failed_at = ? WHERE rollup_id = ?",
        ("2026-07-15T00:00:00+00:00", old_id),
    )
    store.connection.commit()
    config.rollup_builds_per_pass = 1
    monkeypatch.setattr(
        builder_module,
        "summarize_with_escalation",
        lambda _text, **_kwargs: ("retried daily", 1),
    )

    assert run_rollup_maintenance(dag, config, scope) == 1
    assert store.get_rollup("day", old_day.isoformat(), scope)["status"] == "ready"
    assert store.get_rollup("day", recent_day.isoformat(), scope)["status"] == "failed"


def test_maintenance_budget_stops_before_starting_next_build(rollup_parts, monkeypatch):
    store, dag, config = rollup_parts
    scope = "session-budget-stop"
    for offset in range(2):
        target_day = date(2026, 7, 14) + timedelta(days=offset)
        _add_node(dag, scope, target_day, f"source {offset}")
        store.mark_stale_for_day(target_day, scope)
    config.rollup_builds_per_pass = 2
    config.rollup_maintenance_budget_ms = 5
    now = [0.0]
    monkeypatch.setattr(builder_module, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        builder_module,
        "summarize_with_escalation",
        lambda _text, **_kwargs: (now.__setitem__(0, 0.006) or "budgeted daily", 1),
    )

    assert run_rollup_maintenance(dag, config, scope) == 1
    statuses = [
        store.get_rollup("day", (date(2026, 7, 14) + timedelta(days=offset)).isoformat(), scope)["status"]
        for offset in range(2)
    ]
    assert statuses == ["ready", "stale"]


def test_session_start_does_not_wait_for_slow_rollup_provider(tmp_path, monkeypatch):
    db_path = tmp_path / "async-session-start.db"
    config = LCMConfig(
        database_path=str(db_path),
        temporal_rollups_enabled=True,
        rollup_builds_per_pass=1,
    )
    engine = LCMEngine(config=config)
    scope = "slow-rollup-session"
    target_day = date(2026, 7, 15)
    provider_started = threading.Event()
    release_provider = threading.Event()

    def slow_summarizer(_text, **_kwargs):
        provider_started.set()
        if not release_provider.wait(timeout=2):
            raise AssertionError("test did not release slow rollup provider")
        return "completed in background", 1

    try:
        _add_node(engine._dag, scope, target_day, "source for slow rollup")
        store = RollupStore(db_path)
        try:
            store.mark_stale_for_day(target_day, scope)
            monkeypatch.setattr(
                builder_module,
                "summarize_with_escalation",
                slow_summarizer,
            )

            started_at = time.monotonic()
            engine.on_session_start(scope, conversation_id="slow-rollup-conversation")
            bind_elapsed = time.monotonic() - started_at

            assert bind_elapsed < 0.15
            assert provider_started.wait(timeout=1)
            release_provider.set()
            assert engine.drain_rollup_maintenance(timeout=2)
            assert store.get_rollup("day", target_day.isoformat(), scope)["status"] == "ready"
        finally:
            store.close()
    finally:
        release_provider.set()
        engine.shutdown()


def test_rollup_background_maintenance_deduplicates_rapid_scope_binds(
    tmp_path,
    monkeypatch,
):
    config = LCMConfig(
        database_path=str(tmp_path / "deduplicated-maintenance.db"),
        temporal_rollups_enabled=True,
    )
    engine = LCMEngine(config=config)
    scope = "deduplicated-scope"
    first_pass_started = threading.Event()
    release_first_pass = threading.Event()
    call_threads: list[int] = []
    active_calls = 0
    maximum_active_calls = 0
    calls_lock = threading.Lock()

    def controlled_maintenance(dag, _config, called_scope, **_kwargs):
        nonlocal active_calls, maximum_active_calls
        assert dag is not engine._dag
        assert dag.connection is not engine._dag.connection
        assert called_scope == scope
        with calls_lock:
            call_threads.append(threading.get_ident())
            active_calls += 1
            maximum_active_calls = max(maximum_active_calls, active_calls)
            call_number = len(call_threads)
        try:
            if call_number == 1:
                first_pass_started.set()
                if not release_first_pass.wait(timeout=2):
                    raise AssertionError("test did not release first maintenance pass")
            return 0
        finally:
            with calls_lock:
                active_calls -= 1

    monkeypatch.setattr(
        engine_module,
        "run_rollup_maintenance",
        controlled_maintenance,
    )
    try:
        engine.on_session_start(scope, conversation_id="deduplicated-conversation")
        assert first_pass_started.wait(timeout=1)

        for _ in range(20):
            engine._bind_lifecycle_state(
                scope,
                conversation_id="deduplicated-conversation",
            )

        release_first_pass.set()
        assert engine.drain_rollup_maintenance(timeout=2)
        assert len(call_threads) == 2
        assert len(set(call_threads)) == 1
        assert maximum_active_calls == 1
    finally:
        release_first_pass.set()
        engine.shutdown()


def test_rollup_background_scheduler_bounds_distinct_pending_scopes():
    scheduler = engine_module._RollupMaintenanceScheduler(max_pending_jobs=2)
    first_started = threading.Event()
    release_first = threading.Event()
    completed: list[str] = []

    def blocking_first():
        first_started.set()
        if not release_first.wait(timeout=2):
            raise AssertionError("test did not release first scheduler job")
        completed.append("active")

    try:
        assert scheduler.schedule(("db", "active"), blocking_first)
        assert first_started.wait(timeout=1)
        assert scheduler.schedule(
            ("db", "queued-1"),
            lambda: completed.append("queued-1"),
        )
        assert scheduler.schedule(
            ("db", "queued-2"),
            lambda: completed.append("queued-2"),
        )
        assert not scheduler.schedule(
            ("db", "overflow"),
            lambda: completed.append("overflow"),
        )
    finally:
        release_first.set()

    assert scheduler.drain(
        {
            ("db", "active"),
            ("db", "queued-1"),
            ("db", "queued-2"),
        },
        timeout=2,
    )
    assert completed == ["active", "queued-1", "queued-2"]

    # Rejection is bounded deferral, not a poisoned key: a later session bind
    # can offer the same durable stale scope again once capacity is available.
    assert scheduler.schedule(
        ("db", "overflow"),
        lambda: completed.append("overflow"),
    )
    assert scheduler.drain({("db", "overflow")}, timeout=2)
    assert completed == ["active", "queued-1", "queued-2", "overflow"]


def test_rollup_scheduler_reserves_active_follow_up_when_queue_is_full():
    scheduler = engine_module._RollupMaintenanceScheduler(max_pending_jobs=1)
    first_started = threading.Event()
    release_first = threading.Event()
    completed: list[str] = []

    def blocking_first():
        first_started.set()
        if not release_first.wait(timeout=2):
            raise AssertionError("test did not release first scheduler job")
        completed.append("active-1")

    try:
        assert scheduler.schedule(("db", "active"), blocking_first)
        assert first_started.wait(timeout=1)
        assert scheduler.schedule(
            ("db", "queued"),
            lambda: completed.append("queued"),
        )
        assert scheduler.schedule(
            ("db", "active"),
            lambda: completed.append("active-2"),
        )
    finally:
        release_first.set()

    assert scheduler.drain(
        {("db", "active"), ("db", "queued")},
        timeout=2,
    )
    assert completed == ["active-1", "active-2", "queued"]


def test_rollup_scheduler_follow_up_slot_is_single_and_queue_fair():
    scheduler = engine_module._RollupMaintenanceScheduler(max_pending_jobs=1)
    a1_started = threading.Event()
    release_a1 = threading.Event()
    a2_started = threading.Event()
    release_a2 = threading.Event()
    b1_started = threading.Event()
    release_b1 = threading.Event()
    completed: list[str] = []

    def blocking(label, started, release):
        def run():
            started.set()
            if not release.wait(timeout=2):
                raise AssertionError(f"test did not release {label}")
            completed.append(label)

        return run

    try:
        assert scheduler.schedule(
            ("db", "a"),
            blocking("a-1", a1_started, release_a1),
        )
        assert a1_started.wait(timeout=1)
        assert scheduler.schedule(
            ("db", "b"),
            blocking("b-1", b1_started, release_b1),
        )
        assert scheduler.schedule(
            ("db", "a"),
            blocking("a-2", a2_started, release_a2),
        )

        release_a1.set()
        assert a2_started.wait(timeout=1)
        # The immediate follow-up is already consuming the one literal slot;
        # another request is subject to the normal bounded queue instead of
        # accumulating hidden follow-up state.
        assert not scheduler.schedule(
            ("db", "a"),
            lambda: completed.append("a-3"),
        )
        assert scheduler._follow_up_key is None
        assert scheduler._follow_up_job is None

        release_a2.set()
        assert b1_started.wait(timeout=1)
        assert scheduler.schedule(
            ("db", "c"),
            lambda: completed.append("c"),
        )
        assert scheduler.schedule(
            ("db", "b"),
            lambda: completed.append("b-2"),
        )
    finally:
        release_a1.set()
        release_a2.set()
        release_b1.set()

    assert scheduler.drain(
        {("db", "a"), ("db", "b"), ("db", "c")},
        timeout=2,
    )
    assert completed == ["a-1", "a-2", "b-1", "b-2", "c"]
    assert scheduler._follow_up_key is None
    assert scheduler._follow_up_job is None


def test_rollup_scheduler_releases_completed_owner_keys():
    scheduler = engine_module._RollupMaintenanceScheduler(max_pending_jobs=2)
    owner = object()
    completed: list[int] = []

    for index in range(128):
        key = ("db", f"ephemeral-{index}")
        assert scheduler.schedule(
            key,
            lambda index=index: completed.append(index),
            owner=owner,
        )
        assert scheduler.drain_owner(owner, timeout=2)
        assert owner not in scheduler._owned_keys
        assert key not in scheduler._key_owners

    assert completed == list(range(128))


def test_rollup_scheduler_runs_same_key_after_cancelled_job():
    scheduler = engine_module._RollupMaintenanceScheduler(max_pending_jobs=1)
    key = ("db", "cancelled")
    first_started = threading.Event()
    second_ran = threading.Event()

    def cancelled_job():
        first_started.set()
        raise asyncio.CancelledError("forced maintenance cancellation")

    assert scheduler.schedule(key, cancelled_job)
    assert first_started.wait(timeout=1)
    assert scheduler.drain({key}, timeout=1)

    assert scheduler.schedule(key, second_ran.set)
    assert scheduler.drain({key}, timeout=2)
    assert second_ran.is_set()


def test_rollup_scheduler_operator_exclusion_is_bounded_and_defers_maintenance():
    scheduler = engine_module._RollupMaintenanceScheduler(max_pending_jobs=1)
    operator_key = ("db", "operator")
    completed: list[str] = []

    assert scheduler.try_acquire_exclusive(operator_key)
    assert not scheduler.try_acquire_exclusive(operator_key)
    assert not scheduler.try_acquire_exclusive(("db", "second-operator"))
    assert not scheduler.schedule(
        operator_key,
        lambda: completed.append("raced"),
    )
    assert not scheduler.drain({operator_key}, timeout=0)

    scheduler.release_exclusive(operator_key)
    assert scheduler.drain({operator_key}, timeout=0)
    assert scheduler.schedule(
        operator_key,
        lambda: completed.append("after-release"),
    )
    assert scheduler.drain({operator_key}, timeout=2)
    assert completed == ["after-release"]


def test_rollup_background_failure_is_logged_without_breaking_bind(
    tmp_path,
    monkeypatch,
    caplog,
):
    config = LCMConfig(
        database_path=str(tmp_path / "failed-background-maintenance.db"),
        temporal_rollups_enabled=True,
    )
    engine = LCMEngine(config=config)

    def fail_maintenance(*_args, **_kwargs):
        raise RuntimeError("forced background maintenance failure")

    monkeypatch.setattr(engine_module, "run_rollup_maintenance", fail_maintenance)
    caplog.set_level("WARNING", logger="hermes_lcm.engine")
    try:
        engine.on_session_start(
            "failed-maintenance-scope",
            conversation_id="failed-maintenance-conversation",
        )
        assert engine.drain_rollup_maintenance(timeout=2)
        assert engine.bound_session_id == "failed-maintenance-scope"
        assert "background temporal rollup maintenance failed" in caplog.text
        assert "forced background maintenance failure" in caplog.text
    finally:
        engine.shutdown()


def test_engine_shutdown_does_not_wait_for_background_rollup_provider(
    tmp_path,
    monkeypatch,
):
    config = LCMConfig(
        database_path=str(tmp_path / "shutdown-background-maintenance.db"),
        temporal_rollups_enabled=True,
    )
    engine = LCMEngine(config=config)
    maintenance_started = threading.Event()
    release_maintenance = threading.Event()

    def blocking_maintenance(*_args, **_kwargs):
        maintenance_started.set()
        if not release_maintenance.wait(timeout=2):
            raise AssertionError("test did not release background maintenance")
        return 0

    monkeypatch.setattr(
        engine_module,
        "run_rollup_maintenance",
        blocking_maintenance,
    )
    try:
        engine.on_session_start(
            "shutdown-maintenance-scope",
            conversation_id="shutdown-maintenance-conversation",
        )
        assert maintenance_started.wait(timeout=1)

        started_at = time.monotonic()
        engine.shutdown()
        shutdown_elapsed = time.monotonic() - started_at

        assert shutdown_elapsed < 0.15
    finally:
        release_maintenance.set()
        assert engine.drain_rollup_maintenance(timeout=2)


def test_plugin_unload_waits_for_background_rollup_provider(tmp_path, monkeypatch):
    db_path = tmp_path / "unload-background-maintenance.db"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(db_path),
            temporal_rollups_enabled=True,
        )
    )
    maintenance_started = threading.Event()
    release_maintenance = threading.Event()

    def blocking_maintenance(*_args, **_kwargs):
        maintenance_started.set()
        if not release_maintenance.wait(timeout=3):
            raise AssertionError("test did not release background maintenance")
        return 0

    monkeypatch.setattr(
        engine_module,
        "run_rollup_maintenance",
        blocking_maintenance,
    )
    engine.on_session_start(
        "unload-maintenance-scope",
        conversation_id="unload-maintenance-conversation",
    )
    assert maintenance_started.wait(timeout=1)

    unload_done = threading.Event()
    unload_errors = []

    def unload_plugin():
        try:
            engine.shutdown_all_instances()
        except BaseException as exc:  # captured for assertion in the main thread
            unload_errors.append(exc)
        finally:
            unload_done.set()

    unload_thread = threading.Thread(target=unload_plugin)
    unload_thread.start()
    unload_waited = not unload_done.wait(timeout=0.1)
    release_maintenance.set()
    unload_thread.join(timeout=3)
    assert engine.drain_rollup_maintenance(timeout=2)

    assert unload_waited
    assert not unload_thread.is_alive()
    assert unload_errors == []
    db_path.unlink()
    assert not db_path.exists()


def test_pending_maintenance_query_uses_partial_index(rollup_parts):
    store, _dag, _config = rollup_parts
    store.mark_stale_for_day("2026-07-15", "query-plan")

    plan = store.connection.execute(
        "EXPLAIN QUERY PLAN " + _PENDING_ROLLUPS_SQL,
        ("query-plan", 2),
    ).fetchall()

    assert any("idx_lcm_rollups_stale_day" in str(row[3]) for row in plan)


def test_session_reset_stales_rollups_referencing_deleted_nodes(tmp_path):
    db_path = tmp_path / "reset-rollups.db"
    config = LCMConfig(
        database_path=str(db_path),
        temporal_rollups_enabled=True,
        new_session_retain_depth=0,
    )
    engine = LCMEngine(config=config)
    scope = "reset-session"
    try:
        engine.on_session_start(scope, conversation_id="reset-conversation")
        assert engine.drain_rollup_maintenance(timeout=2)
        node_id = _add_node(engine._dag, scope, date(2026, 7, 15), "deleted source")
        store = RollupStore(db_path)
        try:
            _ready(
                store,
                "day",
                "2026-07-15",
                scope,
                summary="summary referencing deleted node",
                source_ids=[node_id],
                fingerprint="deleted-node",
            )

            engine.on_session_reset()

            assert engine._dag.get_session_nodes(scope) == []
            store.drain_invalidations(event_limit=256, day_budget=256)
            assert store.get_rollup("day", "2026-07-15", scope)["status"] == "stale"
        finally:
            store.close()
    finally:
        engine.shutdown()


def test_flag_off_skips_rollup_maintenance(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "flag-off.db"),
        temporal_rollups_enabled=False,
    )
    calls = []
    monkeypatch.setattr(
        engine_module,
        "run_rollup_maintenance",
        lambda *_args, **_kwargs: calls.append("maintenance"),
    )
    engine = LCMEngine(config=config)
    try:
        engine.on_session_start("flag-off-session", conversation_id="flag-off-conversation")
        engine.ingest([{"role": "user", "content": "stored without rollup queries"}])
        engine._bind_lifecycle_state("flag-off-session", conversation_id="flag-off-conversation")
        assert calls == []
    finally:
        engine.shutdown()


def test_publishing_summary_for_ready_day_marks_it_stale(rollup_parts):
    # Publication of a summary covering an already-ready day is the load-bearing
    # staleness signal (maintainer #388 blocker 1): the day (and its week/month)
    # go stale so a later summary cannot leave an older rollup apparently current.
    store, dag, config = rollup_parts
    scope = "session-publish"
    target_day = date(2026, 7, 15)
    node_id = _add_node(dag, scope, target_day, "already summarized")
    for kind, start in (("day", target_day), ("week", date(2026, 7, 13)), ("month", date(2026, 7, 1))):
        _ready(
            store, kind, start.isoformat(), scope,
            summary=f"ready {kind}", source_ids=[node_id], fingerprint=f"{kind}-v1",
        )

    later_node = _add_node(dag, scope, target_day, "a newer summary covering the same day")
    latest_at = _timestamp(target_day, 22)
    from hermes_lcm.rollup_builder import mark_stale_for_published_summary

    assert mark_stale_for_published_summary(dag, scope, latest_at, latest_at) == 3
    assert later_node  # published node exists
    assert store.get_rollup("day", target_day.isoformat(), scope)["status"] == "stale"
    assert store.get_rollup("week", "2026-07-13", scope)["status"] == "stale"
    assert store.get_rollup("month", "2026-07-01", scope)["status"] == "stale"


def test_engine_invalidates_rollups_when_a_node_is_published(tmp_path):
    db_path = tmp_path / "publish-hook.db"
    config = LCMConfig(database_path=str(db_path), temporal_rollups_enabled=True)
    engine = LCMEngine(config=config)
    scope = "publish-hook-session"
    target_day = date(2026, 7, 15)
    try:
        engine.on_session_start(scope, conversation_id="publish-hook-conversation")
        assert engine.drain_rollup_maintenance(timeout=2)
        node_id = _add_node(engine._dag, scope, target_day, "published node")
        store = RollupStore(db_path)
        try:
            _ready(
                store, "day", target_day.isoformat(), scope,
                summary="ready day", source_ids=[node_id], fingerprint="day-v1",
            )
            later_id = _add_node(engine._dag, scope, target_day, "later published node")
            node = engine._dag.get_node(later_id)
            engine._invalidate_rollups_for_published_node(node)
            assert store.get_rollup("day", target_day.isoformat(), scope)["status"] == "stale"
        finally:
            store.close()
    finally:
        engine.shutdown()


def test_maintenance_reclaims_crashed_building_row_and_rebuilds(rollup_parts, monkeypatch):
    # A build that crashed leaves a 'building' row forever; maintenance reclaims
    # it once its lease has expired and rebuilds it (maintainer #388 blocker 2).
    store, dag, config = rollup_parts
    scope = "session-reclaim"
    target_day = date(2026, 7, 15)
    _add_node(dag, scope, target_day, "source for reclaimed build")
    store.upsert_building("day", target_day.isoformat(), scope)
    store.connection.execute(
        "UPDATE lcm_rollups SET lease_expires_at = ? WHERE period_kind = 'day'",
        ("2000-01-01T00:00:00+00:00",),
    )
    store.connection.commit()
    config.rollup_builds_per_pass = 1
    monkeypatch.setattr(
        builder_module,
        "summarize_with_escalation",
        lambda _text, **_kwargs: ("reclaimed rebuild", 1),
    )

    assert run_rollup_maintenance(dag, config, scope) == 1
    assert store.get_rollup("day", target_day.isoformat(), scope)["status"] == "ready"


def test_rollup_builds_per_pass_config_default_and_environment(monkeypatch):
    assert LCMConfig().rollup_builds_per_pass == 2
    assert LCMConfig().rollup_maintenance_budget_ms == 5_000

    monkeypatch.setenv("LCM_ROLLUP_BUILDS_PER_PASS", "5")
    monkeypatch.setenv("LCM_ROLLUP_MAINTENANCE_BUDGET_MS", "750")

    assert LCMConfig.from_env().rollup_builds_per_pass == 5
    assert LCMConfig.from_env().rollup_maintenance_budget_ms == 750


# --- FIXSPEC3 generation-model + staleness + dedup + scope additions -----------


def test_build_day_captures_token_before_reading_sources(rollup_parts, monkeypatch):
    # The build lease must be claimed BEFORE the source snapshot is read, so an
    # invalidation between snapshot and claim cannot escape the generation CAS
    # (maintainer #388 capture-token-first).
    store, dag, config = rollup_parts
    scope = "session-order"
    day = date(2026, 7, 15)
    _add_node(dag, scope, day, "ordered content")
    order: list[str] = []
    real_claim = store.upsert_building
    real_sources = builder_module._daily_sources

    def spy_claim(*args, **kwargs):
        order.append("claim")
        return real_claim(*args, **kwargs)

    def spy_sources(*args, **kwargs):
        order.append("sources")
        return real_sources(*args, **kwargs)

    monkeypatch.setattr(store, "upsert_building", spy_claim)
    monkeypatch.setattr(builder_module, "_daily_sources", spy_sources)

    build_day(store, dag, config, scope, day, summarizer=lambda _t, **_k: ("daily", 1))
    assert order[:2] == ["claim", "sources"]


def test_build_day_supersedes_invalidation_arriving_during_summarize(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-race"
    day = date(2026, 7, 15)
    _add_node(dag, scope, day, "racy content")

    def summarize(_text, **_kwargs):
        # An invalidation lands mid-build; the pre-invalidation token must not
        # publish stale content over it.
        store.mark_stale_for_day(day, scope)
        return "would-be daily", 1

    build_day(store, dag, config, scope, day, summarizer=summarize)
    row = store.get_rollup("day", day.isoformat(), scope)
    assert row["status"] == "stale"
    assert row["summary"] is None


def test_deletion_staleness_bumps_generation_and_supersedes_inflight(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-del"
    day = date(2026, 7, 15)
    node_id = _add_node(dag, scope, day, "to be deleted")
    store.drain_invalidations(event_limit=256, day_budget=256)
    token = store.upsert_building("day", day.isoformat(), scope)
    store.connection.execute(
        "INSERT INTO lcm_rollup_sources(rollup_id, node_id) VALUES(?, ?)",
        (token.rollup_id, node_id),
    )
    store.connection.commit()

    dag.connection.execute("DELETE FROM summary_nodes WHERE node_id=?", (node_id,))
    dag.connection.commit()
    assert builder_module.mark_stale_for_deleted_nodes(dag, [node_id]) == 1
    # The in-flight build cannot publish deleted-node content over the stale row.
    assert store.mark_ready(token, "deleted content", 1, [node_id], "fp") is False
    row = store.get_rollup("day", day.isoformat(), scope)
    assert row["status"] == "stale"
    assert row["generation"] == token.generation + 1


def test_no_source_stale_day_is_resolved_not_left_lingering(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-nosource"
    day = date(2026, 7, 15)
    # A stale day whose only sources were deleted: no summary node remains.
    store.mark_stale_for_day(day, scope)

    assert build_day(store, dag, config, scope, day) is None
    # The day row is cleared, not left stale forever consuming a build slot.
    assert store.get_rollup("day", day.isoformat(), scope) is None


def test_daily_sources_excludes_condensed_children_present_same_day(rollup_parts):
    store, dag, config = rollup_parts
    scope = "session-dedup"
    day = date(2026, 7, 15)
    child = _add_node(dag, scope, day, "child leaf summary")
    parent = dag.add_node(
        SummaryNode(
            session_id=scope,
            depth=1,
            summary="parent condensed summary",
            token_count=count_tokens("parent condensed summary"),
            source_token_count=10,
            source_ids=[child],
            source_type="nodes",
            created_at=_timestamp(day, 18),
            earliest_at=_timestamp(day, 8),
            latest_at=_timestamp(day, 22),
        )
    )
    captured: dict[str, str] = {}

    def summarize(text, **_kwargs):
        captured["text"] = text
        return "daily", 1

    result = build_day(store, dag, config, scope, day, summarizer=summarize)
    assert result["source_node_ids"] == [parent]
    assert "parent condensed summary" in captured["text"]
    assert "child leaf summary" not in captured["text"]


def test_raw_ingest_does_not_prebuild_then_publication_drives_stale(tmp_path, monkeypatch):
    # Item 6 P1: raw ingest must not build/omit a rollup before its summary
    # exists; publication of the covering leaf is the sole staleness signal.
    db_path = tmp_path / "p1.db"
    config = LCMConfig(
        database_path=str(db_path),
        temporal_rollups_enabled=True,
        rollup_builds_per_pass=4,
    )
    engine = LCMEngine(config=config)
    scope = "p1-session"
    day = datetime.now(timezone.utc).date()
    try:
        engine.on_session_start(scope, conversation_id="p1-conv")
        assert engine.drain_rollup_maintenance(timeout=2)
        store = RollupStore(db_path)
        try:
            engine.ingest([{"role": "user", "content": "raw only, no summary yet"}])
            engine._bind_lifecycle_state(scope, conversation_id="p1-conv")
            assert engine.drain_rollup_maintenance(timeout=2)
            assert store.get_rollup("day", day.isoformat(), scope) is None

            node_id = _add_node(engine._dag, scope, day, "published leaf summary")
            monkeypatch.setattr(
                builder_module,
                "summarize_with_escalation",
                lambda _t, **_k: ("rebuilt with leaf", 1),
            )
            engine._invalidate_rollups_for_published_node(engine._dag.get_node(node_id))
            assert store.get_rollup("day", day.isoformat(), scope)["status"] == "stale"

            engine._bind_lifecycle_state(scope, conversation_id="p1-conv")
            assert engine.drain_rollup_maintenance(timeout=2)
            built = store.get_rollup("day", day.isoformat(), scope)
            assert built["status"] == "ready"
            assert node_id in built["source_node_ids"]
        finally:
            store.close()
    finally:
        engine.shutdown()


def test_bypassed_session_skips_rollup_maintenance(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "bypass.db"),
        temporal_rollups_enabled=True,
    )
    calls: list[str] = []
    monkeypatch.setattr(
        engine_module,
        "run_rollup_maintenance",
        lambda *_args, **_kwargs: calls.append("maintenance"),
    )
    engine = LCMEngine(config=config)
    try:
        engine.on_session_start("bypass-session", conversation_id="bypass-conv")
        assert engine.drain_rollup_maintenance(timeout=2)
        calls.clear()

        engine._session_stateless = True
        engine._bind_lifecycle_state("bypass-session", conversation_id="bypass-conv")
        assert engine.drain_rollup_maintenance(timeout=2)
        assert calls == []

        engine._session_stateless = False
        engine._session_ignored = True
        engine._bind_lifecycle_state("bypass-session", conversation_id="bypass-conv")
        assert engine.drain_rollup_maintenance(timeout=2)
        assert calls == []

        engine._session_ignored = False
        engine._bind_lifecycle_state("bypass-session", conversation_id="bypass-conv")
        assert engine.drain_rollup_maintenance(timeout=2)
        assert calls == ["maintenance"]
    finally:
        engine.shutdown()


def test_daily_frontier_does_not_duplicate_multiday_parent_lineage(rollup_parts):
    # Maintainer #388 B1 repro: a child covers Jul15; a parent condenses it
    # spanning Jul15-16. Keying dailies on latest_at put the child in Jul15's
    # source set and the parent in Jul16's, so adjacent dailies duplicated the
    # same covered leaf lineage. The interval-aware canonical frontier must
    # suppress the child (its parent covers it across the day boundary) so the
    # parent feeds both intersected days without duplicating the child lineage.
    store, dag, config = rollup_parts
    scope = "session-b1"
    jul15 = date(2026, 7, 15)
    jul16 = date(2026, 7, 16)
    child = _add_node(dag, scope, jul15, "child leaf covering jul15")
    parent = dag.add_node(
        SummaryNode(
            session_id=scope,
            depth=1,
            summary="parent spanning jul15-16",
            token_count=count_tokens("parent spanning jul15-16"),
            source_token_count=10,
            source_ids=[child],
            source_type="nodes",
            created_at=_timestamp(jul16, 18),
            earliest_at=_timestamp(jul15, 8),
            latest_at=_timestamp(jul16, 22),
        )
    )

    jul15_sources = builder_module._daily_sources(dag, scope, jul15)
    jul16_sources = builder_module._daily_sources(dag, scope, jul16)
    jul15_ids = {source["node_id"] for source in jul15_sources}
    jul16_ids = {source["node_id"] for source in jul16_sources}

    # The child is suppressed everywhere; the canonical parent participates in
    # every day its full interval intersects.
    assert child not in jul15_ids and child not in jul16_ids
    assert jul16_ids == {parent}
    assert jul15_ids == {parent}
    assert builder_module._days_with_content(dag, scope, jul15, jul16) == {
        jul15.isoformat(), jul16.isoformat()
    }


def test_canonical_frontier_collapses_identical_sibling_lineage_independent_of_order():
    child = periods_module.CoverageNode(node_id=1, depth=0)
    higher_id = periods_module.CoverageNode(
        node_id=3, depth=1, source_node_ids=(1,)
    )
    lower_id = periods_module.CoverageNode(
        node_id=2, depth=1, source_node_ids=(1,)
    )

    forward = periods_module.canonical_frontier([child, higher_id, lower_id])
    reverse = periods_module.canonical_frontier([lower_id, higher_id, child])

    assert [node.node_id for node in forward] == [2]
    assert [node.node_id for node in reverse] == [2]


def test_canonical_frontier_uses_transitive_terminal_lineage_and_superset():
    leaf_one = periods_module.CoverageNode(node_id=1, depth=0)
    leaf_two = periods_module.CoverageNode(node_id=2, depth=0)
    grandparent = periods_module.CoverageNode(node_id=4, depth=2)
    lineage = {4: (3, 2), 3: (1,), 2: (), 1: ()}

    frontier = periods_module.canonical_frontier(
        [leaf_one, leaf_two, grandparent], source_lineage=lineage
    )

    assert [node.node_id for node in frontier] == [4]


def test_canonical_frontier_rejects_partial_terminal_lineage_overlap():
    left = periods_module.CoverageNode(
        node_id=4, depth=1, source_node_ids=(1, 2)
    )
    right = periods_module.CoverageNode(
        node_id=5, depth=1, source_node_ids=(2, 3)
    )

    with pytest.raises(
        periods_module.CanonicalFrontierOverlapError,
        match="partially overlap",
    ):
        periods_module.canonical_frontier([left, right])


def test_publication_stales_every_day_a_summary_spans(rollup_parts):
    # Maintainer #388 B2 repro: a newly published summary spanning Jul15-16 left
    # Jul15 ready and only Jul16 went stale. Publication must stale every UTC day
    # the coverage interval intersects (and their week/month).
    store, dag, config = rollup_parts
    scope = "session-b2"
    jul15 = date(2026, 7, 15)
    jul16 = date(2026, 7, 16)
    node = _add_node(dag, scope, jul15, "initial summary")
    for kind, start in (
        ("day", jul15),
        ("day", jul16),
        ("week", date(2026, 7, 13)),
        ("month", date(2026, 7, 1)),
    ):
        _ready(
            store, kind, start.isoformat(), scope,
            summary=f"ready {kind} {start}", source_ids=[node], fingerprint=f"{kind}-{start}",
        )

    from hermes_lcm.rollup_builder import mark_stale_for_published_summary

    published_id = _add_node(
        dag, scope, jul15, "later spanning summary", latest_day=jul16
    )
    published = dag.get_node(published_id)
    mark_stale_for_published_summary(
        dag, scope, published.latest_at, published.created_at,
        earliest_at=published.earliest_at,
    )

    assert store.get_rollup("day", jul15.isoformat(), scope)["status"] == "stale"
    assert store.get_rollup("day", jul16.isoformat(), scope)["status"] == "stale"
    assert store.get_rollup("week", "2026-07-13", scope)["status"] == "stale"
    assert store.get_rollup("month", "2026-07-01", scope)["status"] == "stale"


def test_deleted_node_staleness_covers_more_than_get_session_nodes_limit(rollup_parts):
    # get_session_nodes caps at 1000; the unbounded id capture must return every
    # deleted node so rollups past the cap are still staled (maintainer #388).
    store, dag, _config = rollup_parts
    scope = "session-over-1000"
    for i in range(1001):
        _add_node(dag, scope, date(2026, 7, 15), f"node {i}")

    assert len(dag.get_session_nodes(scope)) == 1000
    all_ids = dag.get_session_node_ids_below_depth(scope, None)
    assert len(all_ids) == 1001
    # Depth filtering also returns the complete set unbounded (all are depth 0).
    assert len(dag.get_session_node_ids_below_depth(scope, 1)) == 1001


def test_aggregate_uses_one_canonical_parent_child_frontier(rollup_parts):
    store, dag, config = rollup_parts
    scope = "aggregate-frontier"
    jul15 = date(2026, 7, 15)
    jul16 = date(2026, 7, 16)
    child = _add_node(dag, scope, jul15, "child content")
    parent = dag.add_node(
        SummaryNode(
            session_id=scope,
            depth=1,
            summary="canonical parent content",
            token_count=3,
            source_token_count=6,
            source_ids=[child],
            source_type="nodes",
            created_at=_timestamp(jul16, 18),
            earliest_at=_timestamp(jul15, 8),
            latest_at=_timestamp(jul16, 22),
        )
    )
    _ready(
        store, "day", jul15.isoformat(), scope,
        summary="daily-child", source_ids=[child], fingerprint="child",
    )
    _ready(
        store, "day", jul16.isoformat(), scope,
        summary="daily-parent", source_ids=[parent], fingerprint="parent",
    )
    seen: list[str] = []
    built = build_week(
        store, dag, config, scope, date(2026, 7, 13),
        summarizer=lambda text, **_kwargs: (seen.append(text) or "week", 1),
    )

    assert built is not None
    assert built["source_node_ids"] == [parent]
    assert "canonical parent content" in seen[0]
    assert "child content" not in seen[0]


def test_aggregate_frontier_suppresses_transitive_child_when_parent_absent(
    rollup_parts,
):
    store, dag, config = rollup_parts
    scope = "aggregate-transitive-frontier"
    jul15 = date(2026, 7, 15)
    jul16 = date(2026, 7, 16)
    child = _add_node(dag, scope, jul15, "transitive child content")
    parent = dag.add_node(
        SummaryNode(
            session_id=scope,
            depth=1,
            summary="intermediate parent not selected by any daily",
            token_count=5,
            source_token_count=8,
            source_ids=[child],
            source_type="nodes",
            created_at=_timestamp(jul15, 18),
            earliest_at=_timestamp(jul15, 8),
            latest_at=_timestamp(jul15, 20),
        )
    )
    grandparent = dag.add_node(
        SummaryNode(
            session_id=scope,
            depth=2,
            summary="canonical grandparent content",
            token_count=4,
            source_token_count=10,
            source_ids=[parent],
            source_type="nodes",
            created_at=_timestamp(jul16, 18),
            earliest_at=_timestamp(jul15, 8),
            latest_at=_timestamp(jul16, 22),
        )
    )
    _ready(
        store, "day", jul15.isoformat(), scope,
        summary="legacy daily child", source_ids=[child], fingerprint="child",
    )
    _ready(
        store, "day", jul16.isoformat(), scope,
        summary="daily grandparent", source_ids=[grandparent], fingerprint="grandparent",
    )
    seen: list[str] = []

    built = build_week(
        store, dag, config, scope, date(2026, 7, 13),
        summarizer=lambda text, **_kwargs: (seen.append(text) or "week", 1),
    )

    assert built is not None
    assert built["source_node_ids"] == [grandparent]
    assert "canonical grandparent content" in seen[0]
    assert "transitive child content" not in seen[0]


def test_aggregate_frontier_fails_closed_on_partial_lineage_overlap(rollup_parts):
    store, dag, config = rollup_parts
    scope = "aggregate-partial-overlap"
    monday = date(2026, 7, 13)
    leaf_ids = [
        _add_node(dag, scope, monday, f"leaf {number}")
        for number in range(3)
    ]
    parents = []
    for source_ids, summary in (
        (leaf_ids[:2], "left parent"),
        (leaf_ids[1:], "right parent"),
    ):
        parents.append(
            dag.add_node(
                SummaryNode(
                    session_id=scope,
                    depth=1,
                    summary=summary,
                    token_count=2,
                    source_token_count=4,
                    source_ids=source_ids,
                    source_type="nodes",
                    created_at=_timestamp(monday, 18),
                    earliest_at=_timestamp(monday, 8),
                    latest_at=_timestamp(monday, 22),
                )
            )
        )
    _ready(
        store,
        "day",
        monday.isoformat(),
        scope,
        summary="legacy overlapping daily",
        source_ids=parents,
        fingerprint="overlap",
    )

    built = build_week(
        store,
        dag,
        config,
        scope,
        monday,
        summarizer=lambda *_args, **_kwargs: ("must not publish", 3),
    )

    assert built is None
    assert store.get_rollup("week", monday.isoformat(), scope)["status"] == "failed"


def test_scope_frontier_never_enumerates_a_nodes_centuries_long_interval(
    rollup_parts,
):
    _store, dag, _config = rollup_parts
    scope = "centuries-long-interval"
    node_id = dag.add_node(
        SummaryNode(
            session_id=scope,
            depth=0,
            summary="one long-lived canonical node",
            token_count=5,
            source_token_count=5,
            source_ids=[1],
            source_type="messages",
            created_at=datetime(2100, 1, 1, tzinfo=timezone.utc).timestamp(),
            earliest_at=datetime(1900, 1, 1, tzinfo=timezone.utc).timestamp(),
            latest_at=datetime(2100, 1, 1, tzinfo=timezone.utc).timestamp(),
        )
    )
    assert not hasattr(periods_module, "covered_days")

    sources = builder_module._daily_sources(dag, scope, date(2026, 7, 15))
    content_days = builder_module._days_with_content(
        dag, scope, date(2026, 7, 1), date(2026, 7, 31)
    )

    assert [source["node_id"] for source in sources] == [node_id]
    assert len(content_days) == 31


def test_source_lineage_walk_stops_near_limit_instead_of_materializing_closure(
    rollup_parts,
):
    _store, dag, _config = rollup_parts
    child_ids = list(range(1, 1_001))
    dag.connection.executemany(
        """
        INSERT INTO summary_nodes(
            node_id, session_id, depth, summary, source_ids, source_type,
            created_at, earliest_at, latest_at
        ) VALUES(?, 'bounded-lineage', 0, 'child', '[1]', 'messages', 1, 1, 1)
        """,
        ((node_id,) for node_id in child_ids),
    )
    root_id = 2_000
    dag.connection.execute(
        """
        INSERT INTO summary_nodes(
            node_id, session_id, depth, summary, source_ids, source_type,
            created_at, earliest_at, latest_at
        ) VALUES(?, 'bounded-lineage', 1, 'wide root', ?, 'nodes', 1, 1, 1)
        """,
        (root_id, json.dumps(child_ids)),
    )
    dag.connection.commit()

    progress_calls = 0

    def progress() -> int:
        nonlocal progress_calls
        progress_calls += 1
        return 0

    dag.connection.set_progress_handler(progress, 1)
    try:
        with pytest.raises(RuntimeError, match="bounded work limit"):
            periods_module.load_source_lineage(
                dag.connection, [root_id], limit=50
            )
    finally:
        dag.connection.set_progress_handler(None, 0)

    # A full 1,001-node recursive closure takes orders of magnitude more VM
    # steps. The edge LIMIT stops after the 50th sentinel visit.
    assert progress_calls < 5_000


def test_scope_frontier_fails_closed_at_sql_work_limit(rollup_parts):
    _store, dag, _config = rollup_parts
    scope = "bounded-frontier"
    rows = [
        (scope, f"node {index}", float(index))
        for index in range(builder_module._FRONTIER_WORK_LIMIT + 1)
    ]
    dag.connection.executemany(
        """
        INSERT INTO summary_nodes(session_id, summary, created_at)
        VALUES(?, ?, ?)
        """,
        rows,
    )
    dag.connection.commit()

    probe_plan = dag.connection.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT node_id FROM summary_nodes WHERE session_id = ? LIMIT ?
        """,
        (scope, builder_module._FRONTIER_WORK_LIMIT + 1),
    ).fetchall()
    plan_text = " ".join(str(row[3]) for row in probe_plan).upper()
    assert "SEARCH SUMMARY_NODES USING" in plan_text
    assert "TEMP B-TREE" not in plan_text

    with pytest.raises(builder_module.RollupWorkLimitExceeded):
        builder_module._scope_frontier(dag, scope)


def test_scope_frontier_serializes_connection_temp_tables_between_threads(
    rollup_parts,
):
    _store, dag, _config = rollup_parts
    day = date(2026, 7, 15)
    node_a = _add_node(dag, "scope-a", day, "A ONLY")
    node_b = _add_node(dag, "scope-b", day, "B ONLY")
    real_connection = dag._conn
    assert real_connection is not None

    a_staged = threading.Event()
    b_staged = threading.Event()
    allow_a_to_read = threading.Event()
    a_done = threading.Event()
    b_staged_before_a_done: list[bool] = []

    class InterleavingConnection:
        """Force the old A-stage/B-stage/A-read corruption schedule."""

        def __getattr__(self, name):
            return getattr(real_connection, name)

        def execute(self, sql, parameters=()):
            return real_connection.execute(sql, parameters)

        def executemany(self, sql, parameters):
            result = real_connection.executemany(sql, parameters)
            if "INSERT INTO temp.lcm_scope_frontier_ids" not in sql:
                return result
            if threading.current_thread().name == "frontier-scope-a":
                a_staged.set()
                if not allow_a_to_read.wait(timeout=2):
                    raise AssertionError("scope A was not released by the test")
            else:
                b_staged_before_a_done.append(not a_done.is_set())
                b_staged.set()
                allow_a_to_read.set()
                if not a_done.wait(timeout=2):
                    raise AssertionError("scope A did not finish before scope B read")
            return result

    results: dict[str, list[dict[str, object]]] = {}
    errors: dict[str, BaseException] = {}

    def read_scope(label: str, scope: str) -> None:
        try:
            results[label] = builder_module._scope_frontier(dag, scope)
        except BaseException as exc:  # surfaced below with the thread label
            errors[label] = exc
        finally:
            if label == "a":
                a_done.set()

    dag._conn = InterleavingConnection()
    thread_a = threading.Thread(
        target=read_scope,
        args=("a", "scope-a"),
        name="frontier-scope-a",
    )
    thread_b = threading.Thread(
        target=read_scope,
        args=("b", "scope-b"),
        name="frontier-scope-b",
    )
    try:
        thread_a.start()
        assert a_staged.wait(timeout=2)
        thread_b.start()
        # With the fixed whole-lifecycle lock B cannot stage until A finishes.
        # Without it B stages immediately and releases A onto B's temp IDs.
        if not b_staged.wait(timeout=0.2):
            allow_a_to_read.set()
        thread_a.join(timeout=2)
        thread_b.join(timeout=2)
    finally:
        allow_a_to_read.set()
        a_done.set()
        thread_a.join(timeout=2)
        thread_b.join(timeout=2)
        dag._conn = real_connection

    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert errors == {}
    assert b_staged_before_a_done == [False]
    assert [(row["node_id"], row["summary"]) for row in results["a"]] == [
        (node_a, "A ONLY")
    ]
    assert [(row["node_id"], row["summary"]) for row in results["b"]] == [
        (node_b, "B ONLY")
    ]


def test_committed_mutation_event_survives_hook_gap_and_reconciles(rollup_parts):
    store, dag, _config = rollup_parts
    scope = "crash-gap"
    day = date(2026, 7, 15)
    original = _add_node(dag, scope, day, "original")
    _ready(
        store, "day", day.isoformat(), scope,
        summary="old ready", source_ids=[original], fingerprint="old",
    )

    _add_node(dag, scope, day, "committed before process crash")
    assert store.has_pending_invalidations(scope)
    store.close()
    reopened = RollupStore(dag.db_path)
    try:
        assert reopened.drain_invalidations(event_limit=256, day_budget=256) == 1
        assert reopened.get_rollup("day", day.isoformat(), scope)["status"] == "stale"
    finally:
        reopened.close()


@pytest.mark.parametrize("staging_helper", ["scope", "aggregate", "lineage"])
def test_dag_temp_staging_releases_snapshot_before_later_write(
    rollup_parts,
    staging_helper,
):
    _store, dag, _config = rollup_parts
    scope = f"transaction-hygiene-{staging_helper}"
    day = date(2026, 7, 15)
    node_id = _add_node(dag, scope, day, "staged source")
    connection = dag.connection
    assert connection is not None
    assert connection.in_transaction is False

    if staging_helper == "scope":
        builder_module._scope_frontier(dag, scope)
    elif staging_helper == "aggregate":
        builder_module._canonical_aggregate_sources(dag, [node_id])
    else:
        periods_module.load_source_lineage(connection, [node_id], limit=10)

    assert connection.in_transaction is False
    with sqlite3.connect(dag.db_path) as independent:
        independent.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
            (f"independent-{staging_helper}", "committed"),
        )

    # A leaked read snapshot makes this fail immediately with
    # SQLITE_BUSY_SNAPSHOT after the independent commit above.
    assert _add_node(dag, scope, day, "write after staging") > node_id


@pytest.mark.parametrize("staging_helper", ["scope", "aggregate", "lineage"])
def test_dag_temp_staging_preserves_caller_owned_transaction(
    rollup_parts,
    staging_helper,
):
    _store, dag, _config = rollup_parts
    scope = f"outer-transaction-{staging_helper}"
    day = date(2026, 7, 15)
    node_id = _add_node(dag, scope, day, "outer transaction source")
    connection = dag.connection
    assert connection is not None

    connection.execute("BEGIN")
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES(?, ?)",
        (f"caller-write-{staging_helper}", "still active"),
    )
    if staging_helper == "scope":
        builder_module._scope_frontier(dag, scope)
    elif staging_helper == "aggregate":
        builder_module._canonical_aggregate_sources(dag, [node_id])
    else:
        periods_module.load_source_lineage(connection, [node_id], limit=10)

    assert connection.in_transaction is True
    assert connection.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (f"caller-write-{staging_helper}",),
    ).fetchone()[0] == "still active"
    connection.rollback()
    assert connection.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (f"caller-write-{staging_helper}",),
    ).fetchone() is None


def test_scope_and_aggregate_staging_release_transaction_on_exception(
    rollup_parts,
    monkeypatch,
):
    _store, dag, _config = rollup_parts
    scope = "staging-exception"
    day = date(2026, 7, 15)
    node_id = _add_node(dag, scope, day, "exception source")
    connection = dag.connection
    assert connection is not None

    def fail_lineage(*_args, **_kwargs):
        raise RuntimeError("forced lineage failure")

    monkeypatch.setattr(builder_module, "load_source_lineage", fail_lineage)
    with pytest.raises(builder_module.RollupWorkLimitExceeded):
        builder_module._scope_frontier(dag, scope)
    assert connection.in_transaction is False

    with pytest.raises(builder_module.RollupWorkLimitExceeded):
        builder_module._canonical_aggregate_sources(dag, [node_id])
    assert connection.in_transaction is False


def test_source_lineage_releases_transaction_on_exception(rollup_parts):
    _store, dag, _config = rollup_parts
    day = date(2026, 7, 15)
    child_id = _add_node(dag, "lineage-exception", day, "child")
    parent_id = dag.add_node(
        SummaryNode(
            session_id="lineage-exception",
            depth=1,
            summary="parent",
            token_count=1,
            source_token_count=2,
            source_ids=[child_id],
            source_type="nodes",
            created_at=_timestamp(day),
            earliest_at=_timestamp(day),
            latest_at=_timestamp(day),
        )
    )
    connection = dag.connection
    assert connection is not None

    with pytest.raises(RuntimeError, match="bounded work limit"):
        periods_module.load_source_lineage(connection, [parent_id], limit=1)

    assert connection.in_transaction is False
