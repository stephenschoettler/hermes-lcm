from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.project_scope import ProjectMetadata
from hermes_lcm.rollup_periods import parse_recent_period
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.schemas import LCM_RECENT
from hermes_lcm.store import MessageStore
from hermes_lcm.tokens import count_tokens
from hermes_lcm import tools as tools_module
from hermes_lcm.tools import (
    _recent_expected_period_starts,
    _recent_has_unready_rollups,
    _recent_ready_rollups,
    lcm_recent,
)


NOW = datetime(2026, 7, 15, 15, 30, tzinfo=timezone.utc)


def test_lcm_recent_schema_advertises_cross_session_scopes():
    parameters = LCM_RECENT["parameters"]
    properties = parameters["properties"]
    assert properties["scope"]["enum"] == ["conversation", "project", "all"]
    assert properties["project_id"] | {"description": None} == {
        "type": "string",
        "minLength": 1,
        "maxLength": 1000,
        "pattern": r"\S",
        "description": None,
    }
    assert "scope='project'" in properties["project_id"]["description"]
    assert {"allOf", "anyOf", "oneOf"}.isdisjoint(parameters)


@pytest.mark.parametrize(
    ("period", "start", "end", "kind", "subday"),
    [
        ("today", "2026-07-15T00:00:00+00:00", "2026-07-16T00:00:00+00:00", "day", False),
        ("yesterday", "2026-07-14T00:00:00+00:00", "2026-07-15T00:00:00+00:00", "day", False),
        ("7d", "2026-07-09T00:00:00+00:00", "2026-07-16T00:00:00+00:00", "day", False),
        ("week", "2026-07-13T00:00:00+00:00", "2026-07-20T00:00:00+00:00", "week", False),
        ("month", "2026-07-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00", "month", False),
        ("date:2026-02-28", "2026-02-28T00:00:00+00:00", "2026-03-01T00:00:00+00:00", "day", False),
        ("last 6h", "2026-07-15T09:30:00+00:00", "2026-07-15T15:30:00+00:00", "day", True),
    ],
)
def test_parse_recent_period_table(period, start, end, kind, subday):
    parsed = parse_recent_period(period, now=NOW)

    assert parsed.start.isoformat() == start
    assert parsed.end.isoformat() == end
    assert parsed.rollup_kind == kind
    assert parsed.subday is subday


@pytest.mark.parametrize(
    "period",
    [
        None,
        "",
        "0d",
        "last 0h",
        "date:2026-02-30",
        "7 days",
        "tomorrow",
        f"{10**30}d",
        f"last {10**30}h",
    ],
)
def test_parse_recent_period_invalid_values_are_clean_errors(period):
    with pytest.raises(ValueError, match="period|day|hour"):
        parse_recent_period(period, now=NOW)


@pytest.mark.parametrize("period", ["today", "week", "month", "1d"])
def test_parse_recent_period_rejects_unrepresentable_upper_day_bounds(period):
    with pytest.raises(ValueError, match="period.*supported date range"):
        parse_recent_period(
            period,
            now=datetime(9999, 12, 31, 12, tzinfo=timezone.utc),
        )


def test_parse_recent_period_rejects_unrepresentable_lower_day_bounds():
    with pytest.raises(ValueError, match="period.*supported date range"):
        parse_recent_period(
            "yesterday",
            now=datetime(1, 1, 1, 12, tzinfo=timezone.utc),
        )


@pytest.fixture
def recent_parts(tmp_path):
    db_path = tmp_path / "recent.db"
    message_store = MessageStore(db_path)
    dag = SummaryDAG(db_path)
    store = RollupStore(db_path)
    config = LCMConfig(database_path=str(db_path), temporal_rollups_enabled=True)
    engine = SimpleNamespace(
        _dag=dag,
        _store=message_store,
        _config=config,
        current_session_id="conversation-a",
    )
    try:
        yield engine, store
    finally:
        store.close()
        dag.close()
        message_store.close()


def _timestamp(day: date, hour: int = 12) -> float:
    return datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).timestamp()


def _add_leaf(dag, session_id, day, summary, *, timestamp=None):
    content_time = timestamp if timestamp is not None else _timestamp(day)
    return dag.add_node(
        SummaryNode(
            session_id=session_id,
            depth=0,
            summary=summary,
            token_count=count_tokens(summary),
            source_token_count=count_tokens(summary),
            source_ids=[1],
            source_type="messages",
            created_at=content_time,
            earliest_at=content_time,
            latest_at=content_time,
        )
    )


def _ready(store, kind, period_start, scope, summary="ready rollup", token_count=17):
    token = store.upsert_building(kind, period_start, scope)
    store.mark_ready(token, summary, token_count, [], "fingerprint")
    return token.rollup_id


def _set_project(engine, session_id, project_id):
    engine._store.set_session_project_metadata(
        session_id,
        ProjectMetadata(project_id, f"/projects/{project_id}", f"/projects/{project_id}"),
    )


def test_lcm_recent_all_returns_bounded_dag_summaries_without_project_metadata(
    recent_parts,
):
    engine, store = recent_parts
    day = date(2026, 7, 15)
    older = _add_leaf(engine._dag, "session-a", day, "older all-session summary", timestamp=_timestamp(day, 8))
    newer = _add_leaf(engine._dag, "session-b", day, "newer all-session summary", timestamp=_timestamp(day, 20))
    tied = _add_leaf(engine._dag, "session-c", day, "tied all-session summary", timestamp=_timestamp(day, 20))
    _add_leaf(engine._dag, "outside-window", date(2026, 7, 14), "outside window")
    _ready(store, "day", day.isoformat(), engine.current_session_id, summary="must not be served")

    result = json.loads(
        lcm_recent(
            {"period": "date:2026-07-15", "scope": "all", "limit": 2},
            engine=engine,
        )
    )

    assert result["mode"] == "dag_summaries"
    assert result["scope"] == "all"
    assert [section["node_id"] for section in result["sections"]] == [tied, newer]
    assert older not in {section["node_id"] for section in result["sections"]}
    assert result["provenance"] == {
        "summaries": [
            {"node_id": tied, "session_id": "session-c"},
            {"node_id": newer, "session_id": "session-b"},
        ]
    }
    assert "rollups_covered" not in result


def test_lcm_recent_project_defaults_to_current_session_project(recent_parts):
    engine, _store = recent_parts
    day = date(2026, 7, 15)
    _set_project(engine, engine.current_session_id, "project-a")
    _set_project(engine, "same-project", "project-a")
    _set_project(engine, "other-project", "project-b")
    current = _add_leaf(engine._dag, engine.current_session_id, day, "current project summary")
    same = _add_leaf(engine._dag, "same-project", day, "same project summary")
    _add_leaf(engine._dag, "other-project", day, "other project summary")
    _add_leaf(engine._dag, "missing-metadata", day, "unscoped summary")

    result = json.loads(
        lcm_recent(
            {"period": "date:2026-07-15", "scope": "project"},
            engine=engine,
        )
    )

    assert result["project_id"] == "project-a"
    assert {section["node_id"] for section in result["sections"]} == {current, same}
    assert {section["session_id"] for section in result["sections"]} == {
        engine.current_session_id,
        "same-project",
    }


def test_lcm_recent_project_accepts_explicit_project_id(recent_parts):
    engine, _store = recent_parts
    day = date(2026, 7, 15)
    _set_project(engine, engine.current_session_id, "project-a")
    _set_project(engine, "other-project", "project-b")
    expected = _add_leaf(engine._dag, "other-project", day, "explicit project summary")
    _add_leaf(engine._dag, engine.current_session_id, day, "current project summary")

    result = json.loads(
        lcm_recent(
            {
                "period": "date:2026-07-15",
                "scope": "project",
                "project_id": "project-b",
            },
            engine=engine,
        )
    )

    assert result["project_id"] == "project-b"
    assert [section["node_id"] for section in result["sections"]] == [expected]


@pytest.mark.parametrize(
    ("project_id", "error"),
    [
        (None, "project_id must be a string"),
        (123, "project_id must be a string"),
        ("   ", "project_id must not be empty"),
        ("p" * 1001, "project_id must be at most 1000 characters"),
    ],
)
def test_lcm_recent_project_rejects_invalid_explicit_id_before_queries(
    recent_parts, monkeypatch, project_id, error
):
    engine, _store = recent_parts

    def fail_query(*_args, **_kwargs):
        raise AssertionError("invalid project_id must be rejected before queries")

    monkeypatch.setattr(tools_module, "_recent_leaf_sections", fail_query)
    monkeypatch.setattr(tools_module, "_recent_ready_rollups", fail_query)

    result = json.loads(
        lcm_recent(
            {"period": "today", "scope": "project", "project_id": project_id},
            engine=engine,
        )
    )

    assert result == {"error": error}


def test_lcm_recent_project_maximum_id_keeps_response_within_char_limit(recent_parts):
    engine, _store = recent_parts
    project_id = "p" * 1000

    encoded = lcm_recent(
        {"period": "today", "scope": "project", "project_id": project_id},
        engine=engine,
    )

    assert json.loads(encoded)["project_id"] == project_id
    assert len(encoded) <= tools_module._LCM_RECENT_MAX_RESPONSE_CHARS


@pytest.mark.parametrize(
    ("scope", "project_id"),
    [("conversation", "other-project"), ("all", 123)],
)
def test_lcm_recent_rejects_project_id_outside_project_scope_before_queries(
    recent_parts, monkeypatch, scope, project_id
):
    engine, _store = recent_parts

    def fail_query(*_args, **_kwargs):
        raise AssertionError("incompatible project_id must be rejected before queries")

    monkeypatch.setattr(tools_module, "_recent_leaf_sections", fail_query)
    monkeypatch.setattr(tools_module, "_recent_ready_rollups", fail_query)

    result = json.loads(
        lcm_recent(
            {"period": "today", "scope": scope, "project_id": project_id},
            engine=engine,
        )
    )

    assert result == {"error": "project_id is only valid when scope=project"}


def test_lcm_recent_project_limit_is_applied_after_stable_lineage_dedup(recent_parts):
    engine, _store = recent_parts
    day = date(2026, 7, 15)
    for session_id in (engine.current_session_id, "parent-session", "independent-session"):
        _set_project(engine, session_id, "project-a")
    child = _add_leaf(engine._dag, engine.current_session_id, day, "covered child")
    independent = _add_leaf(
        engine._dag,
        "independent-session",
        day,
        "independent summary",
        timestamp=_timestamp(day, 6),
    )
    parent = engine._dag.add_node(
        SummaryNode(
            session_id="parent-session",
            depth=1,
            summary="cross-session parent",
            token_count=3,
            source_token_count=4,
            source_ids=[child],
            source_type="nodes",
            created_at=_timestamp(day),
            earliest_at=_timestamp(day),
            latest_at=_timestamp(day),
        )
    )

    result = json.loads(
        lcm_recent(
            {"period": "date:2026-07-15", "scope": "project", "limit": 2},
            engine=engine,
        )
    )

    assert [section["node_id"] for section in result["sections"]] == [parent, independent]
    assert child not in {section["node_id"] for section in result["sections"]}


def test_lcm_recent_project_without_current_project_metadata_is_an_error(recent_parts):
    engine, _store = recent_parts

    result = json.loads(
        lcm_recent({"period": "today", "scope": "project"}, engine=engine)
    )

    assert result == {
        "error": "scope=project requires project metadata for the active session"
    }


def test_lcm_recent_omitted_scope_is_byte_identical_to_explicit_conversation(
    recent_parts,
):
    engine, store = recent_parts
    _ready(store, "day", "2026-07-15", engine.current_session_id)

    omitted = lcm_recent({"period": "date:2026-07-15"}, engine=engine)
    explicit = lcm_recent(
        {"period": "date:2026-07-15", "scope": "conversation"}, engine=engine
    )

    assert omitted == explicit


def test_lcm_recent_serves_ready_rollup_with_provenance(recent_parts):
    engine, store = recent_parts
    rollup_id = _ready(store, "day", "2026-07-15", engine.current_session_id)

    result = json.loads(lcm_recent({"period": "date:2026-07-15"}, engine=engine))

    assert result["mode"] == "rollup"
    assert result["provenance"] == {
        "fallback": False,
        "rollups": [{"rollup_id": rollup_id, "status": "ready"}],
    }
    assert result["sections"][0]["content"].startswith("Tokens: 17\n")


def test_lcm_recent_stale_rollup_falls_back_to_leaf_summaries(recent_parts):
    engine, store = recent_parts
    _add_leaf(engine._dag, engine.current_session_id, date(2026, 7, 15), "leaf fallback")
    store.drain_invalidations()
    _ready(store, "day", "2026-07-15", engine.current_session_id)
    store.mark_stale_for_day("2026-07-15", engine.current_session_id)

    result = json.loads(lcm_recent({"period": "date:2026-07-15"}, engine=engine))

    assert result["mode"] == "leaf_summary_fallback"
    assert result["fallback_reason"] == "rollups_unavailable"
    assert result["provenance"] == {"fallback": True, "rollups": []}
    assert [section["content"] for section in result["sections"]] == ["leaf fallback"]


def test_lcm_recent_fails_closed_while_invalidation_is_pending(recent_parts):
    engine, store = recent_parts
    _ready(store, "day", "2026-07-15", engine.current_session_id)

    # The node insert and durable invalidation event commit together, but no
    # builder has drained the event yet.  The old ready rollup must not be
    # visible in that interval during this window.
    _add_leaf(
        engine._dag,
        engine.current_session_id,
        date(2026, 7, 15),
        "new content awaiting rollup invalidation",
    )

    result = json.loads(lcm_recent({"period": "date:2026-07-15"}, engine=engine))

    assert result["mode"] == "leaf_summary_fallback"
    assert result["fallback_reason"] == "rollups_invalidation_pending"
    assert result["provenance"] == {"fallback": True, "rollups": []}
    assert [section["content"] for section in result["sections"]] == [
        "new content awaiting rollup invalidation"
    ]


def test_recent_window_coverage_requires_every_day_ready(recent_parts):
    # A 2-day window with only one ready daily is incomplete and must fall back
    # for the WHOLE window rather than serve a partial rollup, and a MISSING day
    # (no row at all) must be detected, not only existing non-ready rows
    # (maintainer #389 blocker 1).
    engine, store = recent_parts
    scope = engine.current_session_id
    window = parse_recent_period("2d", now=NOW)  # 2026-07-14 .. 2026-07-16
    assert [d for d in _recent_expected_period_starts(window)] == ["2026-07-14", "2026-07-15"]

    _ready(store, "day", "2026-07-15", scope)  # 2026-07-14 missing entirely
    assert _recent_has_unready_rollups(store, window, scope) is True
    served, reason = _recent_ready_rollups(engine, window, scope)
    assert served == [] and reason == "rollups_unavailable"

    _ready(store, "day", "2026-07-14", scope)  # now fully covered
    assert _recent_has_unready_rollups(store, window, scope) is False
    served, reason = _recent_ready_rollups(engine, window, scope)
    assert reason is None
    assert {row["period_start"] for row in served} == {"2026-07-14", "2026-07-15"}


def test_recent_window_period_enumeration_fails_closed_above_work_cap(recent_parts):
    engine, store = recent_parts
    window = parse_recent_period(
        f"{tools_module._LCM_RECENT_FRONTIER_WORK_LIMIT + 1}d", now=NOW
    )
    statements: list[str] = []
    store.connection.set_trace_callback(statements.append)
    try:
        assert _recent_expected_period_starts(window) == []
        assert _recent_has_unready_rollups(
            store, window, engine.current_session_id
        ) is True
    finally:
        store.connection.set_trace_callback(None)

    # The oversized request is rejected arithmetically; it never enumerates or
    # queries thousands of expected period IDs.
    assert not any("FROM lcm_rollups" in statement for statement in statements)


def test_lcm_recent_fallback_includes_retained_higher_depth_summary(recent_parts):
    # After rotation, a retained higher-depth (carry-forward) summary in-window
    # must be returned by the leaf fallback, not only depth-0 leaves
    # (maintainer #389 blocker 2).
    engine, _store = recent_parts
    engine._config.temporal_rollups_enabled = False  # force the fallback path
    scope = engine.current_session_id
    content_time = _timestamp(date(2026, 7, 15))
    source_id = _add_leaf(
        engine._dag,
        "retained-source-outside-scope",
        date(2026, 7, 15),
        "archived source",
    )
    retained_id = engine._dag.add_node(
        SummaryNode(
            session_id=scope,
            depth=2,
            summary="retained higher-depth carry-forward summary",
            token_count=count_tokens("retained higher-depth carry-forward summary"),
            source_token_count=10,
            source_ids=[source_id],
            source_type="nodes",
            created_at=content_time,
            earliest_at=content_time,
            latest_at=content_time,
        )
    )

    result = json.loads(lcm_recent({"period": "date:2026-07-15"}, engine=engine))

    assert result["mode"] == "leaf_summary_fallback"
    returned_ids = {section["node_id"] for section in result["sections"]}
    assert retained_id in returned_ids


def test_lcm_recent_disabled_flag_falls_back_even_when_ready(recent_parts):
    engine, store = recent_parts
    engine._config.temporal_rollups_enabled = False
    _add_leaf(engine._dag, engine.current_session_id, date(2026, 7, 15), "flag-off leaf")
    _ready(store, "day", "2026-07-15", engine.current_session_id)

    result = json.loads(lcm_recent({"period": "date:2026-07-15"}, engine=engine))

    assert result["fallback_reason"] == "temporal_rollups_disabled"
    assert result["provenance"]["fallback"] is True
    assert result["sections"][0]["kind"] == "leaf_summary"


def test_lcm_recent_subday_window_always_falls_back(recent_parts):
    engine, _store = recent_parts
    recent_time = datetime.now(timezone.utc) - timedelta(minutes=30)
    _add_leaf(
        engine._dag,
        engine.current_session_id,
        recent_time.date(),
        "subday leaf",
        timestamp=recent_time.timestamp(),
    )

    result = json.loads(lcm_recent({"period": "last 2h"}, engine=engine))

    assert result["fallback_reason"] == "subday_window"
    assert result["provenance"]["fallback"] is True
    assert result["sections"][0]["content"] == "subday leaf"


def test_lcm_recent_empty_window_is_a_successful_empty_fallback(recent_parts):
    engine, _store = recent_parts

    result = json.loads(lcm_recent({"period": "date:1999-01-01"}, engine=engine))

    assert "error" not in result
    assert result["provenance"]["fallback"] is True
    assert result["sections"] == []
    assert result["returned_sections"] == 0


def test_lcm_recent_limit_order_and_response_char_bound(recent_parts):
    engine, _store = recent_parts
    engine._config.temporal_rollups_enabled = False
    target_day = date(2026, 7, 15)
    _add_leaf(engine._dag, engine.current_session_id, target_day, "older " * 6000, timestamp=_timestamp(target_day, 8))
    newest_id = _add_leaf(
        engine._dag,
        engine.current_session_id,
        target_day,
        "newer " * 6000,
        timestamp=_timestamp(target_day, 20),
    )
    _add_leaf(engine._dag, engine.current_session_id, target_day, "middle", timestamp=_timestamp(target_day, 12))

    raw = lcm_recent({"period": "date:2026-07-15", "limit": 2}, engine=engine)
    result = json.loads(raw)

    assert len(raw) <= 20_000
    assert result["total_sections"] == 2
    assert len(result["sections"]) <= 2
    assert result["sections"][0]["node_id"] == newest_id
    assert result["truncated"] is True


def test_lcm_recent_conversation_scope_reports_clamped_limit(recent_parts):
    engine, store = recent_parts
    rollup_id = _ready(store, "day", "2026-07-15", engine.current_session_id)

    result = json.loads(
        lcm_recent(
            {"period": "date:2026-07-15", "scope": "conversation", "limit": 500},
            engine=engine,
        )
    )

    assert result["scope"] == "conversation"
    assert result["limit"] == 200
    assert result["limit_clamped_from"] == 500
    assert result["provenance"]["rollups"][0]["rollup_id"] == rollup_id


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({}, "period is required"),
        ({"period": "today", "scope": "workspace"}, "scope must be one of"),
        ({"period": "today", "scope": "global"}, "scope must be one of"),
        ({"period": "today", "limit": 0}, "limit must be a positive integer"),
        ({"period": "today", "limit": True}, "limit must be an integer"),
    ],
)
def test_lcm_recent_argument_validation(recent_parts, args, message):
    engine, _store = recent_parts
    result = json.loads(lcm_recent(args, engine=engine))
    assert message in result["error"]


def test_lcm_recent_reports_max_date_overflow_as_validation_error(recent_parts):
    engine, _store = recent_parts

    result = json.loads(
        lcm_recent({"period": "date:9999-12-31"}, engine=engine)
    )

    assert result == {"error": "period is outside the supported date range"}


def test_recent_rollup_falls_back_when_finalized_session_has_window_content(recent_parts):
    # Rollups are session-scoped, but the leaf fallback spans current +
    # last-finalized session. Rollup mode must not serve current-session-only
    # rollups while the finalized session still holds overlapping window content,
    # or that content would be silently dropped (maintainer #389 blocker 3).
    engine, store = recent_parts
    current = engine.current_session_id
    finalized = "conversation-a-prev"
    engine.current_conversation_id = "conv"
    engine._lifecycle = SimpleNamespace(
        get_by_conversation=lambda _cid: SimpleNamespace(
            current_session_id=current,
            last_finalized_session_id=finalized,
        )
    )
    day = date(2026, 7, 15)
    _ready(store, "day", day.isoformat(), current)  # current fully covers window
    _add_leaf(engine._dag, finalized, day, "finalized session leaf")

    window = parse_recent_period("date:2026-07-15", now=NOW)
    served, reason = _recent_ready_rollups(engine, window, current)
    assert served == []
    assert reason == "rollups_span_multiple_sessions"

    result = json.loads(lcm_recent({"period": "date:2026-07-15"}, engine=engine))
    assert result["mode"] == "leaf_summary_fallback"
    assert result["fallback_reason"] == "rollups_span_multiple_sessions"
    assert "finalized session leaf" in {section["content"] for section in result["sections"]}


def test_recent_fallback_suppresses_child_covered_by_overlapping_parent(recent_parts):
    # Maintainer #389 C1 repro: the leaf fallback selected every overlapping
    # summary with no canonical frontier, so a probe returned BOTH parent-content
    # and child-content — duplicated lineage that consumed the public limit ahead
    # of independent summaries. The interval-aware canonical frontier must
    # suppress the child (contained by an overlapping selected parent) so the
    # limit is not consumed twice.
    engine, _store = recent_parts
    engine._config.temporal_rollups_enabled = False  # force the leaf fallback
    scope = engine.current_session_id
    day = date(2026, 7, 15)
    content_time = _timestamp(day)
    child = _add_leaf(engine._dag, scope, day, "child leaf content")
    independent = _add_leaf(
        engine._dag, scope, day, "independent summary", timestamp=_timestamp(day, 6)
    )
    parent = engine._dag.add_node(
        SummaryNode(
            session_id=scope,
            depth=1,
            summary="parent covering the child",
            token_count=count_tokens("parent covering the child"),
            source_token_count=10,
            source_ids=[child],
            source_type="nodes",
            created_at=content_time,
            earliest_at=content_time,
            latest_at=content_time,
        )
    )

    result = json.loads(lcm_recent({"period": "date:2026-07-15", "limit": 2}, engine=engine))

    assert result["mode"] == "leaf_summary_fallback"
    returned = {section["node_id"] for section in result["sections"]}
    # Only the canonical node is returned for that lineage; the freed slot goes to
    # the independent summary rather than the duplicated child.
    assert parent in returned
    assert independent in returned
    assert child not in returned


def test_recent_fallback_suppresses_transitive_child_when_parent_not_selected(
    recent_parts,
):
    engine, _store = recent_parts
    engine._config.temporal_rollups_enabled = False
    scope = engine.current_session_id
    day = date(2026, 7, 15)
    content_time = _timestamp(day)
    child = _add_leaf(engine._dag, scope, day, "transitive child content")
    parent = engine._dag.add_node(
        SummaryNode(
            # Deliberately outside the requested session set: it remains a DAG
            # lineage intermediate but is not a fallback candidate.
            session_id="lineage-intermediate-outside-scope",
            depth=1,
            summary="unselected intermediate parent",
            token_count=4,
            source_token_count=6,
            source_ids=[child],
            source_type="nodes",
            created_at=content_time,
            earliest_at=content_time,
            latest_at=content_time,
        )
    )
    grandparent = engine._dag.add_node(
        SummaryNode(
            session_id=scope,
            depth=2,
            summary="canonical grandparent content",
            token_count=4,
            source_token_count=8,
            source_ids=[parent],
            source_type="nodes",
            created_at=content_time,
            earliest_at=content_time,
            latest_at=content_time,
        )
    )

    result = json.loads(
        lcm_recent({"period": "date:2026-07-15", "limit": 2}, engine=engine)
    )

    returned = {section["node_id"] for section in result["sections"]}
    assert returned == {grandparent}
    assert child not in returned


def test_recent_fallback_collapses_identical_sibling_lineage(recent_parts):
    engine, _store = recent_parts
    engine._config.temporal_rollups_enabled = False
    scope = engine.current_session_id
    day = date(2026, 7, 15)
    content_time = _timestamp(day)
    child = _add_leaf(engine._dag, scope, day, "shared child")
    sibling_ids = []
    for summary in ("first sibling", "second sibling"):
        sibling_ids.append(
            engine._dag.add_node(
                SummaryNode(
                    session_id=scope,
                    depth=1,
                    summary=summary,
                    token_count=2,
                    source_token_count=4,
                    source_ids=[child],
                    source_type="nodes",
                    created_at=content_time,
                    earliest_at=content_time,
                    latest_at=content_time,
                )
            )
        )

    result = json.loads(
        lcm_recent({"period": "date:2026-07-15", "limit": 10}, engine=engine)
    )

    assert [section["node_id"] for section in result["sections"]] == [
        min(sibling_ids)
    ]


def test_recent_fallback_fails_closed_on_partial_lineage_overlap(recent_parts):
    engine, _store = recent_parts
    engine._config.temporal_rollups_enabled = False
    scope = engine.current_session_id
    day = date(2026, 7, 15)
    content_time = _timestamp(day)
    leaf_ids = [
        _add_leaf(engine._dag, scope, day, f"leaf {number}")
        for number in range(3)
    ]
    for source_ids, summary in (
        (leaf_ids[:2], "left overlap"),
        (leaf_ids[1:], "right overlap"),
    ):
        engine._dag.add_node(
            SummaryNode(
                session_id=scope,
                depth=1,
                summary=summary,
                token_count=2,
                source_token_count=4,
                source_ids=source_ids,
                source_type="nodes",
                created_at=content_time,
                earliest_at=content_time,
                latest_at=content_time,
            )
        )

    result = json.loads(
        lcm_recent({"period": "date:2026-07-15", "limit": 10}, engine=engine)
    )

    assert result["mode"] == "leaf_summary_fallback"
    assert result["sections"] == []


def test_recent_provenance_is_bounded_to_returned_sections(recent_parts):
    # Maintainer #389 C2 repro: 1,000 ready rollups + limit=1 produced a
    # 39,039-char response — all 1,000 provenance rows, zero content sections —
    # blowing the 20,000-char cap because provenance was serialized outside the
    # section/char budget. Provenance must be bound to the sections RETURNED and
    # the total response must respect the cap.
    engine, store = recent_parts
    scope = engine.current_session_id
    today = datetime.now(timezone.utc).date()
    days = [today - timedelta(days=offset) for offset in range(1000)]
    with store.connection:
        store.connection.executemany(
            "INSERT INTO lcm_rollups(period_kind, period_start, scope, summary, "
            "token_count, status, source_fingerprint) "
            "VALUES('day', ?, ?, ?, ?, 'ready', ?)",
            [(d.isoformat(), scope, f"summary for {d}", 5, f"fp-{d}") for d in days],
        )

    raw = lcm_recent({"period": "1000d", "limit": 1}, engine=engine)
    result = json.loads(raw)

    assert result["mode"] == "rollup"
    assert len(raw) <= 20_000
    # Provenance references only the returned section(s), not all 1,000 rollups.
    assert len(result["provenance"]["rollups"]) == result["returned_sections"]
    assert result["returned_sections"] <= 1
    # The bounded aggregate still records how many rollups covered the window.
    assert result["rollups_covered"] == 1000


def test_recent_fallback_includes_summary_overlapping_window_edge(recent_parts):
    # A summary spanning past midnight (latest_at in the NEXT day) still holds
    # this day's content; overlap-based filtering must return it where the old
    # latest_at-only filter dropped it (maintainer #389 blocker: overlap).
    engine, _store = recent_parts
    engine._config.temporal_rollups_enabled = False  # force the leaf fallback
    scope = engine.current_session_id
    day = date(2026, 7, 15)
    spanning_id = engine._dag.add_node(
        SummaryNode(
            session_id=scope,
            depth=0,
            summary="spanning across midnight",
            token_count=count_tokens("spanning across midnight"),
            source_token_count=5,
            source_ids=[1],
            source_type="messages",
            created_at=_timestamp(day, 23),
            earliest_at=_timestamp(day, 23),
            latest_at=_timestamp(date(2026, 7, 16), 1),
        )
    )

    result = json.loads(lcm_recent({"period": "date:2026-07-15"}, engine=engine))
    assert result["mode"] == "leaf_summary_fallback"
    assert spanning_id in {section["node_id"] for section in result["sections"]}


def test_recent_fallback_candidate_work_is_sql_bounded(recent_parts, monkeypatch):
    # T-12 repro: limit=1 previously materialized every matching node before
    # truncating.  The SQL query now fetches at most the work cap plus one
    # sentinel and returns no potentially non-canonical partial frontier.
    engine, _store = recent_parts
    engine._config.temporal_rollups_enabled = False
    scope = engine.current_session_id
    content_time = _timestamp(date(2026, 7, 15))
    rows = [
        SummaryNode(
            session_id=scope,
            depth=0,
            summary=f"independent summary {index}",
            token_count=4,
            source_token_count=4,
            source_ids=[index],
            source_type="messages",
            created_at=content_time,
            earliest_at=content_time,
            latest_at=content_time,
        )
        for index in range(tools_module._LCM_RECENT_FRONTIER_WORK_LIMIT + 1)
    ]
    for node in rows:
        engine._dag.add_node(node)

    probe_plan = engine._dag.connection.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT node_id
        FROM summary_nodes
        WHERE COALESCE(earliest_at, created_at) < ?
          AND COALESCE(latest_at, created_at) >= ?
          AND session_id IN (?)
        LIMIT ?
        """,
        (
            datetime(2026, 7, 16, tzinfo=timezone.utc).timestamp(),
            datetime(2026, 7, 15, tzinfo=timezone.utc).timestamp(),
            scope,
            tools_module._LCM_RECENT_FRONTIER_WORK_LIMIT + 1,
        ),
    ).fetchall()
    plan_text = " ".join(str(row[3]) for row in probe_plan).upper()
    assert "SEARCH SUMMARY_NODES USING" in plan_text
    assert "TEMP B-TREE" not in plan_text

    frontier_calls: list[int] = []
    real_frontier = tools_module.canonical_frontier

    def counted_frontier(candidates):
        frontier_calls.append(len(candidates))
        return real_frontier(candidates)

    monkeypatch.setattr(tools_module, "canonical_frontier", counted_frontier)
    result = json.loads(
        lcm_recent({"period": "date:2026-07-15", "limit": 1}, engine=engine)
    )

    assert result["mode"] == "leaf_summary_fallback"
    assert result["sections"] == []
    assert frontier_calls == []


def test_lcm_recent_all_surfaces_frontier_work_limit(recent_parts, monkeypatch):
    engine, _store = recent_parts
    day = date(2026, 7, 15)
    monkeypatch.setattr(tools_module, "_LCM_RECENT_FRONTIER_WORK_LIMIT", 1)
    _add_leaf(engine._dag, "session-a", day, "first matching summary")
    _add_leaf(engine._dag, "session-b", day, "second matching summary")

    result = json.loads(
        lcm_recent(
            {"period": "date:2026-07-15", "scope": "all", "limit": 1},
            engine=engine,
        )
    )

    assert result == {
        "error": "recent summary frontier exceeds bounded work limit (1); narrow period or scope"
    }


def test_recent_fallback_releases_snapshot_before_later_dag_write(recent_parts):
    engine, _store = recent_parts
    scope = engine.current_session_id
    day = date(2026, 7, 15)
    first_id = _add_leaf(engine._dag, scope, day, "recent staged source")
    connection = engine._dag.connection
    assert connection is not None
    window = parse_recent_period("date:2026-07-15", now=NOW)

    sections = tools_module._recent_leaf_sections(
        engine, window, "conversation", 10
    )

    assert [section["node_id"] for section in sections] == [first_id]
    assert connection.in_transaction is False
    with sqlite3.connect(engine._dag.db_path) as independent:
        independent.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
            ("recent-independent", "committed"),
        )

    # A leaked read snapshot fails here with SQLITE_BUSY_SNAPSHOT.
    assert _add_leaf(engine._dag, scope, day, "recent write after staging") > first_id


def test_recent_fallback_preserves_outer_transaction(recent_parts):
    engine, _store = recent_parts
    scope = engine.current_session_id
    day = date(2026, 7, 15)
    _add_leaf(engine._dag, scope, day, "recent outer source")
    connection = engine._dag.connection
    assert connection is not None
    window = parse_recent_period("date:2026-07-15", now=NOW)

    connection.execute("BEGIN")
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES('recent-caller-write', 'active')"
    )
    tools_module._recent_leaf_sections(engine, window, "conversation", 10)

    assert connection.in_transaction is True
    assert connection.execute(
        "SELECT value FROM metadata WHERE key='recent-caller-write'"
    ).fetchone()[0] == "active"
    connection.rollback()
    assert connection.execute(
        "SELECT value FROM metadata WHERE key='recent-caller-write'"
    ).fetchone() is None


def test_recent_fallback_releases_transaction_on_lineage_exception(
    recent_parts,
    monkeypatch,
):
    engine, _store = recent_parts
    day = date(2026, 7, 15)
    _add_leaf(engine._dag, engine.current_session_id, day, "recent exception source")
    connection = engine._dag.connection
    assert connection is not None
    window = parse_recent_period("date:2026-07-15", now=NOW)

    def fail_lineage(*_args, **_kwargs):
        raise RuntimeError("forced recent lineage failure")

    monkeypatch.setattr(tools_module, "load_source_lineage", fail_lineage)
    assert tools_module._recent_leaf_sections(
        engine, window, "conversation", 10
    ) == []
    assert connection.in_transaction is False
