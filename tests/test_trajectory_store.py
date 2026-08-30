"""Provider-free contract for the same-database trajectory source store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_lcm.store import MessageStore
from hermes_lcm.trajectory_store import (
    TRAJECTORY_SCHEMA_VERSION,
    CorpusIdentity,
    CorpusIdentityError,
    ExactTrajectoryRefError,
    TrajectoryAssetError,
    TrajectorySource,
    TrajectoryState,
    TrajectorySchemaUnavailableError,
    TrajectoryStore,
    TrajectoryStoreError,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(*, revision: str = "dataset-rev-1") -> CorpusIdentity:
    return CorpusIdentity(
        dataset_name="example/trajectory-benchmark",
        dataset_revision=revision,
        harness_commit="harness-commit-1",
        tier="small",
        domain="enterprise",
        ingest_config_digest="ingest-v1",
    )


def _source(
    asset_root: Path,
    *,
    trajectory_id: str = "trajectory-a",
    ordinal: int = 0,
    state_one_text: str = "Export failed because the storage quota was exhausted.",
) -> TrajectorySource:
    screenshots = []
    for index in range(3):
        screenshot = asset_root / f"state-{index}.png"
        screenshot.write_bytes(b"png" + bytes([index]))
        screenshots.append(screenshot)
    states = (
        TrajectoryState(
            state_index=0,
            step=0,
            url="https://example.test/reports",
            incoming_action=None,
            thoughts="Need to export the report.",
            text="Reports page with an Export button.",
            screenshot_path=screenshots[0],
        ),
        TrajectoryState(
            state_index=1,
            step=1,
            url="https://example.test/reports/export",
            incoming_action="Click Export",
            thoughts="The export should start.",
            text=state_one_text,
            screenshot_path=screenshots[1],
        ),
        TrajectoryState(
            state_index=2,
            step=2,
            url="https://example.test/settings/storage",
            incoming_action="Open storage settings",
            thoughts="Check the quota before retrying.",
            text="Storage is 100% full. Delete an old export before retrying.",
            screenshot_path=screenshots[2],
        ),
    )
    return TrajectorySource(
        trajectory_id=trajectory_id,
        ordinal=ordinal,
        goal="Export the quarterly report",
        start_url="https://example.test/reports",
        outcome="Export failed",
        states=states,
        source_payload={
            "id": trajectory_id,
            "goal": "Export the quarterly report",
            "outcome": "Export failed",
            "states": [
                {
                    "state_index": state.state_index,
                    "step": state.step,
                    "url": state.url,
                    "action": state.incoming_action,
                    "thoughts": state.thoughts,
                    "text": state.text,
                    "screenshot": state.screenshot_path.name,
                }
                for state in states
            ],
        },
    )


def _ranked_source(
    asset_root: Path,
    index: int,
    *,
    with_adjacent: bool,
) -> TrajectorySource:
    states = []
    state_payloads = []
    texts = [f"Priorityneedle ranked evidence {index:02d}."]
    if with_adjacent:
        texts.append(f"Adjacent context for ranked evidence {index:02d}.")
    for state_index, text in enumerate(texts):
        screenshot = asset_root / f"ranked-{index}-{state_index}.png"
        screenshot.write_bytes(f"ranked-{index}-{state_index}".encode())
        state = TrajectoryState(
            state_index=state_index,
            step=state_index,
            url=f"https://example.test/ranked/{index}/{state_index}",
            incoming_action=None if state_index == 0 else "Open details",
            thoughts="Inspect ranked evidence.",
            text=text,
            screenshot_path=screenshot,
        )
        states.append(state)
        state_payloads.append({
            "state_index": state.state_index,
            "step": state.step,
            "url": state.url,
            "action": state.incoming_action,
            "thoughts": state.thoughts,
            "text": state.text,
            "screenshot": screenshot.name,
        })
    trajectory_id = f"ranked-{index:02d}"
    return TrajectorySource(
        trajectory_id=trajectory_id,
        ordinal=index,
        goal="Find ranked evidence",
        start_url=f"https://example.test/ranked/{index}",
        outcome="Evidence found",
        states=tuple(states),
        source_payload={
            "id": trajectory_id,
            "goal": "Find ranked evidence",
            "outcome": "Evidence found",
            "states": state_payloads,
        },
    )


def _ranked_store(tmp_path: Path, *, adjacent_sources: int) -> TrajectoryStore:
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    store = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    trajectory_ids = []
    for index in range(20):
        source = _ranked_source(
            asset_root,
            index,
            with_adjacent=index < adjacent_sources,
        )
        store.insert(source)
        trajectory_ids.append(source.trajectory_id)
    store.finalize(trajectory_ids)
    return store


@pytest.fixture
def trajectory_db(tmp_path: Path):
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    messages = MessageStore(db_path)
    store = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    try:
        yield db_path, asset_root, messages, store
    finally:
        store.close()
        messages.close()


def test_schema_is_optional_same_database_and_uses_named_marker(tmp_path: Path):
    db_path = tmp_path / "lcm.db"
    messages = MessageStore(db_path)
    before = {
        row[0]
        for row in messages._conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'lcm_trajectory%'"
        )
    }
    assert before == set()

    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    store = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    try:
        assert store.db_path.resolve() == db_path.resolve()
        names = {
            row[0]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'lcm_trajectory%'"
            )
        }
        assert {
            "lcm_trajectory_corpora",
            "lcm_trajectory_sources",
            "lcm_trajectory_states",
            "lcm_trajectory_assets",
            "lcm_trajectory_ingest_receipts",
            "lcm_trajectory_transitions",
            "lcm_trajectory_states_fts",
        }.issubset(names)
        assert store.connection.execute(
            "SELECT COUNT(*) FROM lcm_migration_state WHERE step_name = ?",
            ("trajectory_store_v1",),
        ).fetchone()[0] == 1
    finally:
        store.close()
        messages.close()


def test_corpus_identity_is_strict_and_read_only_open_does_not_mutate(tmp_path: Path):
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    store = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    store.insert(_source(asset_root))
    store.finalize(["trajectory-a"])
    store.close()

    with pytest.raises(CorpusIdentityError):
        TrajectoryStore(
            db_path,
            _identity(revision="different-revision"),
            asset_root=asset_root,
        )

    before = db_path.stat().st_mtime_ns
    readonly = TrajectoryStore(
        db_path,
        _identity(),
        asset_root=asset_root,
        read_only=True,
    )
    try:
        assert readonly.status == "complete"
        assert readonly.connection.execute("PRAGMA query_only").fetchone()[0] == 1
    finally:
        readonly.close()
    assert db_path.stat().st_mtime_ns == before


def test_read_only_open_rejects_malformed_current_trajectory_schema(tmp_path: Path):
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    store = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    store.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "ALTER TABLE lcm_trajectory_states "
            "RENAME COLUMN search_text TO malformed_search_text"
        )
        conn.commit()

    with pytest.raises(
        TrajectorySchemaUnavailableError,
        match=r"column:lcm_trajectory_states\.search_text",
    ):
        TrajectoryStore(
            db_path,
            _identity(),
            asset_root=asset_root,
            read_only=True,
        )


def test_newer_trajectory_schema_is_rejected_before_fts_repair(tmp_path: Path):
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    store = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    store.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE lcm_trajectory_corpora SET schema_version=? WHERE singleton=1",
            (TRAJECTORY_SCHEMA_VERSION + 1,),
        )
        conn.execute("DROP TABLE lcm_trajectory_states_fts")
        conn.commit()

    with pytest.raises(CorpusIdentityError):
        TrajectoryStore(db_path, _identity(), asset_root=asset_root)

    with sqlite3.connect(db_path) as conn:
        fts = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE name='lcm_trajectory_states_fts'"
        ).fetchone()
    assert fts is None


def test_malformed_trajectory_schema_raises_before_fts_repair(tmp_path: Path):
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE lcm_trajectory_corpora "
            "(singleton INTEGER PRIMARY KEY CHECK (singleton = 1))"
        )

    with pytest.raises(sqlite3.OperationalError, match="no such column: schema_version"):
        TrajectoryStore(db_path, _identity(), asset_root=asset_root)

    with sqlite3.connect(db_path) as conn:
        fts = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE name='lcm_trajectory_states_fts'"
        ).fetchone()
    assert fts is None


def test_insert_is_idempotent_and_conflicting_source_fails(trajectory_db):
    _db_path, asset_root, _messages, store = trajectory_db
    source = _source(asset_root)
    first = store.insert(source)
    second = store.insert(source)
    assert first.already_current is False
    assert second.already_current is True
    assert store.connection.execute(
        "SELECT COUNT(*) FROM lcm_trajectory_sources"
    ).fetchone()[0] == 1
    assert store.connection.execute(
        "SELECT COUNT(*) FROM lcm_trajectory_states"
    ).fetchone()[0] == 3

    changed = _source(
        asset_root,
        state_one_text="Export failed for an unrelated reason.",
    )
    with pytest.raises(TrajectoryStoreError, match="different digest"):
        store.insert(changed)


def test_destination_action_sequence_and_unknown_times_are_preserved(trajectory_db):
    _db_path, asset_root, _messages, store = trajectory_db
    store.insert(_source(asset_root))
    rows = store.connection.execute(
        """
        SELECT state_index, sequence_ordinal, incoming_action, observed_at, occurred_at
        FROM lcm_trajectory_states
        ORDER BY state_index
        """
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        (0, 0, None, None, None),
        (1, 1, "Click Export", None, None),
        (2, 2, "Open storage settings", None, None),
    ]
    transitions = store.connection.execute(
        """
        SELECT t.sequence_ordinal, pre.state_index, post.state_index,
               t.incoming_action
        FROM lcm_trajectory_transitions t
        JOIN lcm_trajectory_states pre ON pre.state_id = t.pre_state_id
        JOIN lcm_trajectory_states post ON post.state_id = t.post_state_id
        ORDER BY t.sequence_ordinal
        """
    ).fetchall()
    assert [tuple(row) for row in transitions] == [
        (1, 0, 1, "Click Export"),
        (2, 1, 2, "Open storage settings"),
    ]


def test_interrupted_ingest_resumes_at_first_missing_contiguous_receipt(tmp_path: Path):
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    store = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    store.insert(_source(asset_root, trajectory_id="trajectory-a", ordinal=0))
    missing = _source(asset_root, trajectory_id="trajectory-b", ordinal=1)
    bad_state = TrajectoryState(
        **{**missing.states[1].__dict__, "screenshot_path": asset_root / "missing.png"}
    )
    with pytest.raises(TrajectoryAssetError):
        store.insert(
            TrajectorySource(
                **{
                    **missing.__dict__,
                    "states": (missing.states[0], bad_state, missing.states[2]),
                }
            )
        )
    assert store.connection.execute(
        "SELECT ingest_cursor FROM lcm_trajectory_corpora"
    ).fetchone()[0] == 1
    assert store.connection.execute(
        "SELECT COUNT(*) FROM lcm_trajectory_ingest_receipts"
    ).fetchone()[0] == 1
    store.close()

    resumed = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    try:
        resumed.insert(missing)
        resumed.finalize(["trajectory-a", "trajectory-b"])
        assert resumed.connection.execute(
            "SELECT ingest_cursor FROM lcm_trajectory_corpora"
        ).fetchone()[0] == 2
        assert resumed.connection.execute(
            "SELECT COUNT(*) FROM lcm_trajectory_ingest_receipts"
        ).fetchone()[0] == 2
        assert resumed.connection.execute(
            "SELECT COUNT(*) FROM lcm_trajectory_states"
        ).fetchone()[0] == 6
    finally:
        resumed.close()


def test_explicit_source_times_are_separate_from_ingest_time(trajectory_db):
    _db_path, asset_root, _messages, store = trajectory_db
    source = _source(asset_root)
    explicit = TrajectoryState(
        **{
            **source.states[1].__dict__,
            "observed_at": 1_700_000_000.0,
            "observed_at_source": "host_message_timestamp",
            "occurred_at": 1_699_999_000.0,
            "occurred_at_source": "explicit_state_metadata",
        }
    )
    store.insert(
        TrajectorySource(
            **{
                **source.__dict__,
                "states": (source.states[0], explicit, source.states[2]),
            }
        )
    )
    row = store.connection.execute(
        """
        SELECT observed_at, observed_at_source, occurred_at,
               occurred_at_source, ingested_at
        FROM lcm_trajectory_states WHERE state_index = 1
        """
    ).fetchone()
    assert row["observed_at"] == 1_700_000_000.0
    assert row["observed_at_source"] == "host_message_timestamp"
    assert row["occurred_at"] == 1_699_999_000.0
    assert row["occurred_at_source"] == "explicit_state_metadata"
    assert row["ingested_at"] not in {row["observed_at"], row["occurred_at"]}

    invalid = TrajectoryState(
        **{
            **source.states[1].__dict__,
            "observed_at": 1_700_000_001.0,
            "observed_at_source": None,
        }
    )
    with pytest.raises(ValueError, match="requires explicit observed_at_source"):
        store._protected_source(
            TrajectorySource(
                **{
                    **source.__dict__,
                    "trajectory_id": "invalid-time",
                    "ordinal": 1,
                    "states": (source.states[0], invalid, source.states[2]),
                }
            )
        )


def test_bounded_fts_returns_stable_exact_refs_and_late_adjacent_state(trajectory_db):
    _db_path, asset_root, _messages, store = trajectory_db
    store.insert(_source(asset_root))
    corpus_uid = store.finalize(["trajectory-a"])

    hits = store.query(
        "exhausted",
        candidate_limit=12,
        limit=3,
        image_limit=2,
        include_adjacent=True,
    )
    refs = [hit.exact_ref for hit in hits]
    assert len(hits) <= 3
    assert len(refs) == len(set(refs))
    assert all(ref.startswith(f"trajectory://{corpus_uid}/") for ref in refs)
    assert any("storage quota" in hit.text for hit in hits)
    assert any(
        "before retrying" in hit.text and hit.match_kind == "adjacent"
        for hit in hits
    )
    assert sum(hit.screenshot_path is not None for hit in hits) <= 2
    for hit in hits:
        assert store.resolve_exact_ref(hit.exact_ref).exact_ref == hit.exact_ref

    with pytest.raises(ExactTrajectoryRefError):
        store.resolve_exact_ref("trajectory://wrong/trajectory-a/state/1")


def test_unused_adjacency_reserve_backfills_the_full_ranked_limit(tmp_path: Path):
    store = _ranked_store(tmp_path, adjacent_sources=0)
    try:
        ranked = store.query("priorityneedle", limit=16, include_adjacent=False)
        with_reserve = store.query(
            "priorityneedle", limit=16, include_adjacent=True
        )
    finally:
        store.close()

    assert len(ranked) == 16
    assert [hit.exact_ref for hit in with_reserve] == [
        hit.exact_ref for hit in ranked
    ]


def test_partial_adjacency_reserve_backfills_in_rank_order(tmp_path: Path):
    store = _ranked_store(tmp_path, adjacent_sources=1)
    try:
        ranked = store.query("priorityneedle", limit=16, include_adjacent=False)
        with_reserve = store.query(
            "priorityneedle", limit=16, include_adjacent=True
        )
    finally:
        store.close()

    non_adjacent = [
        hit.exact_ref for hit in with_reserve if hit.match_kind != "adjacent"
    ]
    assert len(with_reserve) == 16
    assert sum(hit.match_kind == "adjacent" for hit in with_reserve) == 1
    assert non_adjacent == [hit.exact_ref for hit in ranked[:15]]
    assert non_adjacent[-4:] == [hit.exact_ref for hit in ranked[11:15]]


def test_adjacency_reserve_backfill_preserves_diversity_cap(tmp_path: Path):
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    store = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    target = _ranked_source(asset_root, 0, with_adjacent=True)
    tail_screenshot = asset_root / "ranked-0-2.png"
    tail_screenshot.write_bytes(b"ranked-0-2")
    tail = TrajectoryState(
        state_index=2,
        step=2,
        url="https://example.test/ranked/0/2",
        incoming_action="Open final details",
        thoughts="Inspect lower-ranked evidence.",
        text=f"Priorityneedle {'background ' * 200}",
        screenshot_path=tail_screenshot,
    )
    target_payload = dict(target.source_payload)
    target_payload["states"] = [
        *target_payload["states"],
        {
            "state_index": tail.state_index,
            "step": tail.step,
            "url": tail.url,
            "action": tail.incoming_action,
            "thoughts": tail.thoughts,
            "text": tail.text,
            "screenshot": tail_screenshot.name,
        },
    ]
    target = TrajectorySource(
        trajectory_id=target.trajectory_id,
        ordinal=target.ordinal,
        goal=target.goal,
        start_url=target.start_url,
        outcome=target.outcome,
        states=(*target.states, tail),
        source_payload=target_payload,
    )
    try:
        store.insert(target)
        trajectory_ids = [target.trajectory_id]
        for index in range(1, 4):
            source = _ranked_source(asset_root, index, with_adjacent=False)
            store.insert(source)
            trajectory_ids.append(source.trajectory_id)
        store.finalize(trajectory_ids)

        hits = store.query(
            "priorityneedle",
            limit=6,
            include_adjacent=True,
            diversity_cap=2,
        )
    finally:
        store.close()

    assert sum(hit.trajectory_id == target.trajectory_id for hit in hits) == 2
    assert any(
        hit.trajectory_id == target.trajectory_id and hit.match_kind == "adjacent"
        for hit in hits
    )


def test_full_adjacency_reserve_keeps_the_existing_composition(tmp_path: Path):
    store = _ranked_store(tmp_path, adjacent_sources=11)
    try:
        ranked = store.query("priorityneedle", limit=16, include_adjacent=False)
        with_reserve = store.query(
            "priorityneedle", limit=16, include_adjacent=True
        )
    finally:
        store.close()

    assert len(with_reserve) == 16
    assert [hit.exact_ref for hit in with_reserve[:11]] == [
        hit.exact_ref for hit in ranked[:11]
    ]
    assert all(hit.match_kind == "adjacent" for hit in with_reserve[11:])


def test_query_text_is_a_bounded_exact_excerpt_and_ref_hydrates_full_state(tmp_path: Path):
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    long_text = "prefix " * 600 + "needle exact answer" + " suffix" * 600
    store = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    try:
        store.insert(_source(asset_root, state_one_text=long_text))
        store.finalize(["trajectory-a"])
        hit = store.query("needle exact answer", limit=1, text_char_limit=512)[0]
        assert len(hit.text) == 512
        assert hit.text_truncated is True
        assert hit.text_offset > 0
        assert hit.text == long_text[hit.text_offset : hit.text_offset + 512]
        assert "needle exact answer" in hit.text
        hydrated = store.resolve_exact_ref(hit.exact_ref)
        assert hydrated.text == long_text
        assert hydrated.text_offset == 0
        assert hydrated.text_truncated is False
    finally:
        store.close()


def test_asset_missing_outside_root_and_symlink_escape_fail_closed(tmp_path: Path):
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    store = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    try:
        source = _source(asset_root)
        bad_state = TrajectoryState(
            **{
                **source.states[1].__dict__,
                "screenshot_path": outside,
            }
        )
        with pytest.raises(TrajectoryAssetError):
            store.insert(
                TrajectorySource(
                    **{
                        **source.__dict__,
                        "states": (source.states[0], bad_state, source.states[2]),
                    }
                )
            )

        missing = asset_root / "missing.png"
        missing_state = TrajectoryState(
            **{**source.states[1].__dict__, "screenshot_path": missing}
        )
        with pytest.raises(TrajectoryAssetError):
            store.insert(
                TrajectorySource(
                    **{
                        **source.__dict__,
                        "trajectory_id": "trajectory-missing",
                        "states": (source.states[0], missing_state, source.states[2]),
                    }
                )
            )

        escaped_link = asset_root / "escaped.png"
        escaped_link.symlink_to(outside)
        escaped_state = TrajectoryState(
            **{**source.states[1].__dict__, "screenshot_path": escaped_link}
        )
        with pytest.raises(TrajectoryAssetError):
            store.insert(
                TrajectorySource(
                    **{
                        **source.__dict__,
                        "trajectory_id": "trajectory-symlink",
                        "states": (source.states[0], escaped_state, source.states[2]),
                    }
                )
            )
    finally:
        store.close()


def test_synthetic_secrets_are_redacted_before_sqlite_fts_and_output(tmp_path: Path):
    secret = "sk-proj-super-secret-value-1234567890"
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    store = TrajectoryStore(
        db_path,
        _identity(),
        asset_root=asset_root,
        protect_sensitive=True,
    )
    try:
        source = _source(asset_root, state_one_text=f"api_key={secret}")
        store.insert(source)
        store.finalize(["trajectory-a"])
        serialized = "\n".join(store.connection.iterdump())
        hits = store.query("api key", limit=3)
        assert secret not in serialized
        assert secret not in json.dumps([hit.to_dict() for hit in hits])
        assert "[LCM sensitive redaction:" in serialized
    finally:
        store.close()

    for path in (
        db_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    ):
        if path.exists():
            assert secret.encode() not in path.read_bytes()


def test_query_is_read_only_and_backup_restore_preserves_query_digest(trajectory_db, tmp_path: Path):
    db_path, asset_root, _messages, store = trajectory_db
    store.insert(_source(asset_root))
    store.finalize(["trajectory-a"])
    before_changes = store.connection.total_changes
    first = store.query("storage quota retrying", limit=3)
    assert store.connection.total_changes == before_changes

    backup_path = tmp_path / "restored.db"
    store.backup_to(backup_path)
    with sqlite3.connect(f"file:{backup_path}?mode=ro", uri=True) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"

    restored = TrajectoryStore(
        backup_path,
        _identity(),
        asset_root=asset_root,
        read_only=True,
    )
    try:
        second = restored.query("storage quota retrying", limit=3)
        assert [hit.to_dict() for hit in second] == [hit.to_dict() for hit in first]
        assert restored.query_digest(second) == store.query_digest(first)
    finally:
        restored.close()


def test_missing_fts_artifact_rebuilds_from_exact_state_rows(tmp_path: Path):
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    store = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    store.insert(_source(asset_root))
    store.finalize(["trajectory-a"])
    expected = [hit.to_dict() for hit in store.query("exhausted", limit=3)]
    store.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TRIGGER lcm_trajectory_fts_insert")
        conn.execute("DROP TRIGGER lcm_trajectory_fts_delete")
        conn.execute("DROP TRIGGER lcm_trajectory_fts_update")
        conn.execute("DROP TABLE lcm_trajectory_states_fts")
        conn.commit()

    repaired = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    try:
        assert [hit.to_dict() for hit in repaired.query("exhausted", limit=3)] == expected
        assert repaired.connection.execute(
            "SELECT COUNT(*) FROM lcm_trajectory_states_fts"
        ).fetchone()[0] == 3
    finally:
        repaired.close()


def test_equal_count_stale_fts_is_detected_and_rebuilt(tmp_path: Path):
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    store = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    store.insert(_source(asset_root))
    store.finalize(["trajectory-a"])
    store.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TRIGGER lcm_trajectory_fts_update")
        conn.execute(
            "UPDATE lcm_trajectory_states SET search_text = ? WHERE state_index = 1",
            ("Visible state: replacement-marker",),
        )
        conn.commit()

    repaired = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    try:
        assert repaired.query("replacement marker", limit=1)
        assert repaired.query("exhausted", limit=1) == ()
    finally:
        repaired.close()


def test_corpus_uid_includes_source_and_asset_bytes(tmp_path: Path):
    def build(root: Path, *, text: str, asset_suffix: bytes) -> str:
        root.mkdir()
        assets = root / "assets"
        assets.mkdir()
        source = _source(assets, state_one_text=text)
        source.states[1].screenshot_path.write_bytes(b"png-" + asset_suffix)
        store = TrajectoryStore(root / "lcm.db", _identity(), asset_root=assets)
        try:
            store.insert(source)
            return store.finalize(["trajectory-a"])
        finally:
            store.close()

    first = build(tmp_path / "first", text="same text", asset_suffix=b"first")
    second = build(tmp_path / "second", text="same text", asset_suffix=b"second")
    third = build(tmp_path / "third", text="changed text", asset_suffix=b"first")
    assert len({first, second, third}) == 3


def test_concurrent_insert_and_finalize_preserve_complete_manifest(tmp_path: Path):
    db_path = tmp_path / "lcm.db"
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    first = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    second = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
    source = _source(asset_root)

    def insert_source():
        return first.insert(source)

    def finalize_source():
        try:
            return second.finalize(["trajectory-a"])
        except CorpusIdentityError:
            return None

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            insert_future = pool.submit(insert_source)
            finalize_future = pool.submit(finalize_source)
            insert_future.result()
            finalize_future.result()
        corpus_uid = second.finalize(["trajectory-a"])
        row = second.connection.execute(
            """
            SELECT status, ingest_cursor, trajectory_count, corpus_uid,
                   source_manifest_digest
            FROM lcm_trajectory_corpora
            """
        ).fetchone()
        assert tuple(row[:4]) == ("complete", 1, 1, corpus_uid)
        assert row["source_manifest_digest"]

        duplicate = TrajectoryStore(db_path, _identity(), asset_root=asset_root)
        try:
            assert duplicate.insert(source).already_current is True
        finally:
            duplicate.close()
    finally:
        first.close()
        second.close()
