"""Demand-shaped query-view lifecycle, freshness, and versioning fixtures."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import sqlite3
import threading

import pytest

from hermes_lcm.assertion_store import AssertionCandidate, AssertionStore
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.maintenance import backup_database, rotate_backup_database
from hermes_lcm.query_view_store import (
    QueryViewBuildInProgressError,
    QueryViewIdentity,
    QueryViewStore,
)
from hermes_lcm.store import MessageStore


@pytest.fixture
def view_db(tmp_path):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    views = QueryViewStore(db_path)
    try:
        yield messages, views
    finally:
        views.close()
        messages.close()


def _append(messages: MessageStore, content: str, *, session="session-a") -> int:
    return messages.append(session, {
        "role": "user",
        "content": content,
        "source": "cli",
        "conversation_id": "conversation-a",
    })


def _identity(**changes) -> QueryViewIdentity:
    base = QueryViewIdentity(
        intent_type="preference lookup",
        operation="evidence_only",
        subject_key="user:self",
        predicate_key="drink.preference",
        role_key="user",
        scope_key="personal",
        conversation_id="conversation-a",
        policy_version="query-policy-v1",
    )
    return replace(base, **changes)


def test_requirements_digest_is_part_of_strict_identity(view_db):
    messages, views = view_db
    content = "I prefer jasmine tea."
    store_id = _append(messages, content)
    digest_a = "a" * 64
    digest_b = "b" * 64
    identity = _identity(requirements_digest=digest_a)
    _publish(views, identity, [_dependency(views, store_id, content)])

    assert views.lookup(identity).status == "hit"
    assert views.lookup(_identity(requirements_digest=digest_b)).status == "miss"
    with pytest.raises(ValueError, match="64-character SHA-256"):
        _identity(requirements_digest="not-a-digest").normalized()


def _dependency(views: QueryViewStore, store_id: int, content: str, quote=None):
    exact = content if quote is None else quote
    start = content.index(exact)
    return views.snapshot_dependency(store_id, start, start + len(exact), exact)


def _manifest(*dependencies, open_slots=()):
    return {
        "closed_slots": ["subject", "predicate", "evidence"],
        "open_slots": list(open_slots),
        "operands": [],
        "retrieval_calls": [{"tool": "lcm_recall", "round": 1}],
        "evidence_refs": [dependency.citation for dependency in dependencies],
        "coverage": {"scope": "all", "complete": not open_slots},
    }


def _publish(
    views: QueryViewStore,
    identity: QueryViewIdentity,
    dependencies,
    *,
    completeness="complete",
    trace=None,
    ttl_seconds=3600,
):
    token = views.claim_build(identity)
    assert views.publish_ready(
        token,
        dependencies=dependencies,
        manifest=_manifest(*dependencies, open_slots=("missing",) if completeness == "partial" else ()),
        computation_trace=trace,
        completeness=completeness,
        search_policy_version="search-v1",
        ttl_seconds=ttl_seconds,
    )
    return token


def test_default_off_and_env_opt_in_do_not_call_a_provider(monkeypatch, tmp_path):
    monkeypatch.delenv("LCM_QUERY_VIEWS_ENABLED", raising=False)
    assert LCMConfig().query_views_enabled is False
    assert LCMConfig.from_env().query_views_enabled is False
    monkeypatch.setenv("LCM_QUERY_VIEWS_ENABLED", "true")
    assert LCMConfig.from_env().query_views_enabled is True

    messages = MessageStore(tmp_path / "default-off.db")
    try:
        names = {
            row[0]
            for row in messages._conn.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'lcm_query_%'"
            )
        }
        assert names == set()
    finally:
        messages.close()

    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "default-engine.db"))
    )
    try:
        assert engine._query_views is None
        names = {
            row[0]
            for row in engine._store._conn.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'lcm_query_%'"
            )
        }
        assert names == set()
    finally:
        engine.shutdown()


def test_engine_flag_binds_same_db_rebinds_profiles_and_closes(tmp_path):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    engine = LCMEngine(
        config=LCMConfig(database_path="", query_views_enabled=True),
        hermes_home=str(home_a),
    )
    last_view_store = None
    try:
        first_view_store = engine._query_views
        assert first_view_store is not None
        assert first_view_store.db_path == engine._store.db_path == home_a / "lcm.db"
        content = "I prefer tea."
        store_id = _append(engine._store, content)
        dependency = _dependency(first_view_store, store_id, content)
        _publish(first_view_store, _identity(), [dependency])

        engine.on_session_start(
            "session-b",
            hermes_home=str(home_b),
            platform="cli",
            context_length=200_000,
        )
        assert first_view_store.connection is None
        assert engine._query_views is not None
        assert engine._query_views.db_path == engine._store.db_path == home_b / "lcm.db"
        assert engine._query_views.lookup(_identity(), record_hit=False).status == "miss"

        engine.on_session_start(
            "session-a-return",
            hermes_home=str(home_a),
            platform="cli",
            context_length=200_000,
        )
        last_view_store = engine._query_views
        assert last_view_store is not None
        assert last_view_store.lookup(_identity(), record_hit=False).status == "hit"
    finally:
        engine.shutdown()
    assert last_view_store is not None
    assert last_view_store.connection is None


def test_engine_backup_and_rotation_preserve_query_view_versions(tmp_path):
    db_path = tmp_path / "lcm.db"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(db_path),
            query_views_enabled=True,
        ),
        hermes_home=str(tmp_path),
    )
    try:
        views = engine._query_views
        assert views is not None
        content = "I prefer tea."
        store_id = _append(engine._store, content)
        dependency = _dependency(views, store_id, content)
        _publish(views, _identity(), [dependency])

        timestamped = backup_database(engine)
        rotated = rotate_backup_database(engine)
        assert timestamped["ok"] is True
        assert rotated["ok"] is True
        for result in (timestamped, rotated):
            restored = QueryViewStore(result["backup_path"])
            try:
                lookup = restored.lookup(_identity(), record_hit=False)
                assert lookup.status == "hit"
                assert lookup.view["version"] == 1
                assert lookup.view["dependencies"][0]["source_store_id"] == store_id
            finally:
                restored.close()
    finally:
        engine.shutdown()


def test_legacy_database_migrates_in_place_and_seeds_corpus_state(tmp_path):
    db_path = tmp_path / "legacy.db"
    messages = MessageStore(db_path)
    _append(messages, "Existing raw evidence remains authoritative.")
    messages.close()

    views = QueryViewStore(db_path)
    try:
        snapshot = views.corpus_snapshot()
        assert snapshot.row_count == 1
        assert snapshot.max_store_id == 1
        marker = views._conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name='query_views_v1'"
        ).fetchone()
        assert marker is not None
    finally:
        views.close()


def test_exact_typed_repeat_hits_promotes_and_near_miss_never_reuses(view_db):
    messages, views = view_db
    content = "I prefer tea over coffee."
    store_id = _append(messages, content)
    dependency = _dependency(views, store_id, content, "prefer tea")
    identity = _identity()
    _publish(views, identity, [dependency])

    first = views.lookup(_identity())
    assert first.status == "hit"
    assert first.view["promotion_status"] == "probationary"
    second = views.lookup(_identity())
    assert second.status == "hit"
    assert second.view["promotion_status"] == "promoted"

    for near_miss in (
        _identity(subject_key="person:alex"),
        _identity(predicate_key="vacation.days.available"),
        _identity(operation="latest_fact"),
        _identity(conversation_id="Conversation-A"),
    ):
        assert views.lookup(near_miss).status == "miss"


def test_hit_confirmation_rechecks_generation_after_source_mutation(
    view_db, monkeypatch
):
    messages, views = view_db
    content = "I prefer tea."
    store_id = _append(messages, content)
    dependency = _dependency(views, store_id, content)
    identity = _identity()
    _publish(views, identity, [dependency])
    published_snapshot = views.corpus_snapshot()
    original_snapshot = views.corpus_snapshot
    original_write_transaction = views._write_transaction
    snapshot_calls = 0
    hit_confirmation_hook_reached = False

    def tracked_snapshot():
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original_snapshot()

    @contextmanager
    def mutate_during_hit_confirmation():
        nonlocal hit_confirmation_hook_reached
        # The initial negative-space snapshot completed and found the view
        # current. Mutate only as the hit-confirmation transaction begins.
        assert snapshot_calls == 1
        hit_confirmation_hook_reached = True
        _append(messages, "I prefer coffee.")
        with original_write_transaction():
            yield

    monkeypatch.setattr(views, "corpus_snapshot", tracked_snapshot)
    monkeypatch.setattr(
        views,
        "_write_transaction",
        mutate_during_hit_confirmation,
    )
    result = views.lookup(identity)

    assert hit_confirmation_hook_reached is True
    assert snapshot_calls == 2
    assert result.status == "delta_required"
    assert result.reason == "corpus advanced during hit confirmation"
    assert result.view["status"] == "stale"
    assert result.view["corpus_generation"] == published_snapshot.generation
    assert result.view["hit_count"] == 0


def test_relative_time_reanchors_while_absolute_time_reuses(view_db):
    messages, views = view_db
    content = "I traveled on 2024-03-15."
    store_id = _append(messages, content)
    dependency = _dependency(views, store_id, content)
    relative = _identity(
        intent_type="relative event lookup",
        time_mode="relative",
        question_anchor="2024-03-20",
        window_start="2024-03-15",
        window_end="2024-03-16",
    )
    _publish(views, relative, [dependency])
    assert views.lookup(relative).status == "hit"
    reanchored = replace(
        relative,
        question_anchor="2024-03-21",
        window_start="2024-03-16",
        window_end="2024-03-17",
    )
    assert views.lookup(reanchored).status == "miss"

    absolute = _identity(
        intent_type="absolute event lookup",
        time_mode="absolute",
        window_start="2024-03-15",
        window_end="2024-03-16",
    )
    _publish(views, absolute, [dependency])
    assert views.lookup(absolute).status == "hit"


def test_new_evidence_invalidates_negative_space_and_delta_refresh_versions(view_db):
    messages, views = view_db
    content = "I prefer tea."
    store_id = _append(messages, content)
    dependency = _dependency(views, store_id, content)
    identity = _identity()
    _publish(views, identity, [dependency])
    published_generation = views.lookup(identity, record_hit=False).view["corpus_generation"]

    new_content = "An unrelated project note arrived."
    new_store_id = _append(messages, new_content)
    stale = views.lookup(identity, record_hit=False)
    assert stale.status == "delta_required"
    assert stale.view["status"] == "stale"
    assert [event["store_id"] for event in stale.delta_events] == [new_store_id]
    assert stale.delta_events[0]["generation"] > published_generation

    # The bounded delta search found no relevant correction, so republish the
    # same exact dependency at the advanced coverage watermark.
    refreshed_dependency = _dependency(views, store_id, content)
    _publish(views, identity, [refreshed_dependency])
    refreshed = views.lookup(identity, record_hit=False)
    assert refreshed.status == "hit"
    assert refreshed.view["version"] == 2
    assert refreshed.view["supersedes_version"] == 1
    assert refreshed.view["coverage_store_id"] == new_store_id


@pytest.mark.parametrize("mutation", ["update", "delete"])
def test_exact_positive_source_mutation_fails_closed(view_db, mutation):
    messages, views = view_db
    content = "I prefer tea."
    store_id = _append(messages, content)
    dependency = _dependency(views, store_id, content)
    identity = _identity()
    _publish(views, identity, [dependency])

    if mutation == "update":
        messages._conn.execute(
            "UPDATE messages SET content='I prefer rum.' WHERE store_id=?",
            (store_id,),
        )
    else:
        messages._conn.execute("DELETE FROM messages WHERE store_id=?", (store_id,))
    messages._conn.commit()

    result = views.lookup(identity, record_hit=False)
    assert result.status == "delta_required"
    assert "positive_source" in result.reason or "dependency source" in result.reason
    assert result.view["status"] == "stale"


def test_assertion_dependency_invalidation_fails_closed_without_raw_mutation(tmp_path):
    db_path = tmp_path / "assertion-dependency.db"
    messages = MessageStore(db_path)
    assertions = AssertionStore(db_path)
    views = QueryViewStore(db_path)
    try:
        content = "I prefer tea."
        store_id = _append(messages, content)
        start = content.index("tea")
        published = assertions.publish_source(
            assertions.snapshot_source(store_id),
            [
                AssertionCandidate(
                    source_span_start=start,
                    source_span_end=start + len("tea"),
                    subject_key="user:self",
                    predicate_key="drink.preference",
                    object_value="tea",
                    value_text="tea",
                    kind="preference",
                )
            ],
        )
        dependency = views.snapshot_dependency(
            store_id,
            start,
            start + len("tea"),
            "tea",
            assertion_id=published.assertion_ids[0],
        )
        identity = _identity()
        _publish(views, identity, [dependency])
        assert views.lookup(identity, record_hit=False).status == "hit"

        assert assertions.invalidate_source(store_id, reason="test invalidation") == 1
        stale = views.lookup(identity, record_hit=False)
        assert stale.status == "delta_required"
        assert "assertion" in stale.reason
    finally:
        views.close()
        assertions.close()
        messages.close()


def test_correction_reversal_and_cancellation_are_bounded_delta_events(view_db):
    messages, views = view_db
    content = "I will submit the report."
    store_id = _append(messages, content)
    dependency = _dependency(views, store_id, content)
    identity = _identity(
        intent_type="commitment state",
        predicate_key="report.submission",
        operation="latest_fact",
    )
    _publish(views, identity, [dependency])

    corrections = [
        "Correction: the report is due Friday.",
        "I reverse that decision.",
        "I cancel the report submission.",
    ]
    ids = [_append(messages, value) for value in corrections]
    result = views.lookup(identity, record_hit=False, delta_limit=2)
    assert result.status == "delta_required"
    assert [event["store_id"] for event in result.delta_events] == ids[:2]
    assert result.delta_truncated is True


def test_concurrent_build_tokens_and_mid_build_corpus_changes_cannot_publish(view_db):
    messages, views = view_db
    content = "I prefer tea."
    store_id = _append(messages, content)
    dependency = _dependency(views, store_id, content)
    identity = _identity()
    first = views.claim_build(identity)
    with pytest.raises(QueryViewBuildInProgressError, match="active lease"):
        views.claim_build(identity)
    assert views.publish_ready(
        first,
        dependencies=[dependency],
        manifest=_manifest(dependency),
    )

    refresh = views.claim_build(identity)
    _append(messages, "A row arrived during refresh.")
    assert not views.publish_ready(
        refresh,
        dependencies=[dependency],
        manifest=_manifest(dependency),
    )
    assert views.lookup(identity, record_hit=False).status == "delta_required"


def test_failed_and_expired_build_leases_recover_without_serving_partial_state(view_db):
    messages, views = view_db
    content = "I prefer tea."
    store_id = _append(messages, content)
    dependency = _dependency(views, store_id, content)
    identity = _identity()

    failed = views.claim_build(identity)
    assert views.mark_failed(failed, "bounded test failure") is True
    assert views.lookup(identity, record_hit=False).status == "failed"
    recovered = views.claim_build(identity)
    assert views.publish_ready(
        recovered,
        dependencies=[dependency],
        manifest=_manifest(dependency),
    )
    assert views.lookup(identity, record_hit=False).status == "hit"

    abandoned = views.claim_build(identity)
    assert views.reclaim_expired_builds(now=views._now() + 301) == 1
    assert not views.publish_ready(
        abandoned,
        dependencies=[dependency],
        manifest=_manifest(dependency),
    )
    reclaimed = views.claim_build(identity)
    assert views.publish_ready(
        reclaimed,
        dependencies=[dependency],
        manifest=_manifest(dependency),
    )
    refreshed = views.lookup(identity, record_hit=False)
    assert refreshed.status == "hit"
    assert refreshed.view["version"] == 2


def test_publication_source_cas_serializes_a_concurrent_source_update(
    view_db, monkeypatch
):
    messages, views = view_db
    content = "I prefer tea."
    store_id = _append(messages, content)
    dependency = _dependency(views, store_id, content)
    identity = _identity()
    token = views.claim_build(identity)
    validated = threading.Event()
    release_publish = threading.Event()
    update_finished = threading.Event()
    publish_result: list[bool] = []
    errors: list[BaseException] = []
    original_validate = views._validate_dependency_current

    def paused_validate(item):
        original_validate(item)
        validated.set()
        assert release_publish.wait(timeout=5)

    monkeypatch.setattr(views, "_validate_dependency_current", paused_validate)

    def publish():
        try:
            publish_result.append(
                views.publish_ready(
                    token,
                    dependencies=[dependency],
                    manifest=_manifest(dependency),
                )
            )
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    def update_source():
        conn = sqlite3.connect(messages.db_path, timeout=5)
        try:
            conn.execute(
                "UPDATE messages SET content='I prefer coffee.' WHERE store_id=?",
                (store_id,),
            )
            conn.commit()
        finally:
            conn.close()
            update_finished.set()

    publisher = threading.Thread(target=publish)
    publisher.start()
    assert validated.wait(timeout=5)
    updater = threading.Thread(target=update_source)
    updater.start()
    assert not update_finished.wait(timeout=0.1)
    release_publish.set()
    publisher.join(timeout=5)
    updater.join(timeout=5)

    assert not publisher.is_alive()
    assert not updater.is_alive()
    assert errors == []
    assert publish_result == [True]
    lookup = views.lookup(identity, record_hit=False)
    assert lookup.status == "delta_required"
    assert "positive_source" in lookup.reason or "hash changed" in lookup.reason


def test_partial_views_and_cached_prose_never_serve_as_warm_truth(view_db):
    messages, views = view_db
    content = "I prefer tea."
    store_id = _append(messages, content)
    dependency = _dependency(views, store_id, content)
    identity = _identity()
    _publish(views, identity, [dependency], completeness="partial")
    assert views.lookup(identity, record_hit=False).status == "incomplete"

    token = views.claim_build(identity)
    with pytest.raises(ValueError, match="final prose"):
        views.publish_ready(
            token,
            dependencies=[dependency],
            manifest={**_manifest(dependency), "nested": {"answer": "tea"}},
        )


def test_computation_trace_citations_must_be_exact_and_answer_is_rejected(view_db):
    messages, views = view_db
    content = "Alice spent $30."
    store_id = _append(messages, content)
    dependency = _dependency(views, store_id, content)
    identity = _identity(intent_type="spending total", operation="sum", unit="usd")
    token = views.claim_build(identity)
    with pytest.raises(ValueError, match="non-trace fields"):
        views.publish_ready(
            token,
            dependencies=[dependency],
            manifest=_manifest(dependency),
            computation_trace={"operation": "sum", "result": "$30", "answer": "$30"},
        )
    assert views.mark_failed(token, "invalid trace shape") is True

    token = views.claim_build(identity)
    with pytest.raises(ValueError, match="citations"):
        views.publish_ready(
            token,
            dependencies=[dependency],
            manifest=_manifest(dependency),
            computation_trace={
                "operation": "sum",
                "result": "$30",
                "citations": ["lcm:999:0-1"],
            },
        )
    assert views.mark_failed(token, "invalid trace citations") is True

    token = views.claim_build(identity)
    assert views.publish_ready(
        token,
        dependencies=[dependency],
        manifest=_manifest(dependency),
        computation_trace={
            "operation": "sum",
            "result": "$30",
            "result_value": 30,
            "unit": "usd",
            "citations": [dependency.citation],
            "entities": ["Alice"],
            "evidence_dates": [],
            "steps": ["sum(30) = 30"],
        },
    )


def test_expiry_purge_and_dependency_bound_keep_state_bounded(view_db):
    messages, views = view_db
    content = "I prefer tea."
    store_id = _append(messages, content)
    dependency = _dependency(views, store_id, content)
    identity = _identity()
    _publish(views, identity, [dependency], ttl_seconds=1)
    future = views._now() + 2
    assert views.lookup(identity, now=future, record_hit=False).status == "expired"
    assert views.purge_expired(older_than=future + 1) == 1
    assert views.lookup(identity, record_hit=False).status == "miss"

    many_content = "|".join(f"evidence-{index:02d}" for index in range(41))
    many_id = _append(messages, many_content)
    dependencies = [
        _dependency(views, many_id, many_content, f"evidence-{index:02d}")
        for index in range(41)
    ]
    token = views.claim_build(_identity(predicate_key="bounded.dependencies"))
    with pytest.raises(ValueError, match="between 1 and 40"):
        views.publish_ready(
            token,
            dependencies=dependencies,
            manifest=_manifest(*dependencies),
        )


def test_corpus_event_pruning_preserves_active_negative_space_watermarks(view_db):
    messages, views = view_db
    content = "I prefer tea."
    store_id = _append(messages, content)
    dependency = _dependency(views, store_id, content)
    identity = _identity()
    _publish(views, identity, [dependency])
    for index in range(5):
        _append(messages, f"new evidence {index}")

    assert views.prune_corpus_events(retain=2) == 1
    generations = [
        int(row[0])
        for row in views.connection.execute(
            "SELECT generation FROM lcm_query_corpus_events ORDER BY generation"
        )
    ]
    assert generations == [2, 3, 4, 5, 6]

    future = views._now() + 4_000
    assert views.expire_views(now=future) == 1
    assert views.purge_expired(older_than=future + 1) == 1
    assert views.prune_corpus_events(retain=2) == 3
    generations = [
        int(row[0])
        for row in views.connection.execute(
            "SELECT generation FROM lcm_query_corpus_events ORDER BY generation"
        )
    ]
    assert generations == [5, 6]


def test_malformed_same_name_schema_fails_without_publishing_marker(tmp_path):
    db_path = tmp_path / "malformed.db"
    messages = MessageStore(db_path)
    messages.close()
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE lcm_query_views(view_id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="query-view schema"):
        QueryViewStore(db_path)
    conn = sqlite3.connect(db_path)
    try:
        marker = conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name='query_views_v1'"
        ).fetchone()
        assert marker is None
    finally:
        conn.close()
