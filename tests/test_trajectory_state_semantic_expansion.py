"""State-level semantic pool-expansion + backfill (issue #142, Lane S / W3a).

The per-SOURCE semantic index carries one coarse vector per trajectory and so
cannot surface a lexically-invisible answer STATE. These exercise the additive
per-STATE index that can: a target state carries NO query term of its own
(lexically invisible) but is the semantic nearest neighbour of the query, so the
``state_semantic_quota`` knob must pull it INTO the state pool as a quota-capped
additive tail -- while the defaults reproduce the pre-expansion pool and delivery
byte-for-byte, delivery stays unchanged when the ranked pool already fills the
nucleus (additive-only proof), the 5-per-trajectory diversity cap at selection is
preserved, and the backfill itself is resumable/idempotent (skip embedded
states) with a chunked path for over-cap documents.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import threading

import pytest

import hermes_lcm.tokens as token_module
from hermes_lcm.trajectory_store import (
    CorpusIdentity,
    TrajectorySource,
    TrajectoryState,
    TrajectoryStore,
    TrajectoryStoreError,
)


class StateVectorProvider:
    """A tiny 3-D embedder used to steer per-STATE ranking deterministically.

    A state whose text mentions ``alpha-answer`` embeds onto the +x axis (the
    query direction), ``beta-answer`` onto +y, and everything else onto +z, so
    the query (``embed_query`` -> +x) ranks exactly the alpha states first
    regardless of their lexical (BM25) visibility.
    """

    provider_id = "fake"
    model_id = "fake-state-v1"

    def __init__(self) -> None:
        self.last_usage_tokens = 0
        self.document_calls = 0
        self.query_calls = 0
        self.fail_queries = False
        self.usage_tokens_total = 0

    @staticmethod
    def _vector(text: str) -> list[float]:
        folded = str(text).casefold()
        if "alpha-answer" in folded:
            return [1.0, 0.0, 0.0]
        if "beta-answer" in folded:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        self.document_calls += 1
        self.last_usage_tokens = sum(max(1, len(str(t)) // 4) for t in texts)
        self.usage_tokens_total += self.last_usage_tokens
        return [self._vector(t) for t in texts]

    def embed_query(self, text):  # noqa: ARG002
        self.query_calls += 1
        if self.fail_queries:
            raise RuntimeError("simulated query provider failure")
        self.last_usage_tokens = 1
        self.usage_tokens_total += self.last_usage_tokens
        return [1.0, 0.0, 0.0]


class InterruptingStateVectorProvider(StateVectorProvider):
    def __init__(self, *, model_id: str, fail_after: int | None) -> None:
        super().__init__()
        self.model_id = model_id
        self.fail_after = fail_after

    def embed_documents(self, texts):
        if self.fail_after is not None and self.document_calls >= self.fail_after:
            raise RuntimeError("simulated interrupted backfill")
        return super().embed_documents(texts)


class ProbeBudgetProvider(StateVectorProvider):
    def __init__(self, token_limit: int) -> None:
        super().__init__()
        self.token_limit = token_limit
        self.probe_documents: list[str] = []

    def embed_query(self, text):
        self.probe_documents.append(str(text))
        if token_module.count_tokens(str(text)) > self.token_limit:
            raise ValueError("probe exceeded provider token limit")
        return super().embed_query(text)


class RequestBudgetProvider(StateVectorProvider):
    def __init__(self, token_limit: int) -> None:
        super().__init__()
        self.token_limit = token_limit
        self.request_token_counts: list[int] = []

    def embed_documents(self, texts):
        tokens = sum(token_module.count_tokens(str(text)) for text in texts)
        self.request_token_counts.append(tokens)
        if tokens > self.token_limit:
            raise ValueError("request exceeded provider token limit")
        return super().embed_documents(texts)


class CoordinatedStateVectorProvider(StateVectorProvider):
    def __init__(
        self,
        *,
        model_id: str,
        document_vector: list[float],
        started: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.document_vector = document_vector
        self.started = started
        self.release = release

    def embed_documents(self, texts):
        self.document_calls += 1
        if self.started is not None:
            self.started.set()
        if self.release is not None and not self.release.wait(timeout=5):
            raise RuntimeError("timed out waiting to release embedding request")
        self.last_usage_tokens = sum(max(1, len(str(t)) // 4) for t in texts)
        self.usage_tokens_total += self.last_usage_tokens
        return [list(self.document_vector) for _text in texts]


def _identity() -> CorpusIdentity:
    return CorpusIdentity(
        dataset_name="example/state-semantic",
        dataset_revision="rev-state-semantic",
        harness_commit="harness-state-semantic-1",
        tier="small",
        domain="web",
        ingest_config_digest="state-semantic-test-v1",
    )


def _source(asset_root, *, trajectory_id, ordinal, goal, texts) -> TrajectorySource:
    states = []
    for index, text in enumerate(texts):
        screenshot = asset_root / f"{trajectory_id}-{index}.png"
        screenshot.write_bytes(b"png" + hashlib.sha256(text.encode()).digest())
        states.append(TrajectoryState(
            state_index=index,
            step=index,
            url=f"https://example.test/{trajectory_id}/{index}",
            incoming_action=None if index == 0 else f"advance {index}",
            thoughts=f"inspect state {index}",
            text=text,
            screenshot_path=screenshot,
        ))
    return TrajectorySource(
        trajectory_id=trajectory_id,
        ordinal=ordinal,
        goal=goal,
        start_url=f"https://example.test/{trajectory_id}",
        outcome="completed",
        states=tuple(states),
        source_payload={"id": trajectory_id, "goal": goal},
    )


_QUERY = "widget configuration export"


def _state_id(store, trajectory_id: str, state_index: int) -> int:
    row = store._conn.execute(
        """
        SELECT s.state_id FROM lcm_trajectory_states s
        JOIN lcm_trajectory_sources src ON src.source_id = s.source_id
        WHERE src.trajectory_id = ? AND s.state_index = ?
        """,
        (trajectory_id, state_index),
    ).fetchone()
    assert row is not None, f"unknown state {trajectory_id}/{state_index}"
    return int(row[0])


def _pool_state_ids(store) -> set[int]:
    telemetry = store.last_query_telemetry()
    return {int(item["state_id"]) for item in telemetry["state_candidate_pool"]}


def _admitted(store) -> list[dict[str, int]]:
    telemetry = store.last_query_telemetry()
    expansion = telemetry.get("state_semantic_expansion")
    return list(expansion["admitted"]) if expansion else []


def _ranking_bytes(store) -> bytes:
    hits = store.query(_QUERY, image_limit=0, state_semantic_quota=8)
    return json.dumps(
        [hit.to_dict() for hit in hits],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _profile_embedding_rows(store, profile_digest: str) -> list[tuple]:
    return [
        (
            int(row["state_id"]),
            str(row["document_sha256"]),
            bytes(row["vector"]),
            float(row["embedded_at"]),
        )
        for row in store._conn.execute(
            """
            SELECT state_id, document_sha256, vector, embedded_at
            FROM lcm_trajectory_state_embeddings
            WHERE profile_digest = ?
            ORDER BY state_id
            """,
            (profile_digest,),
        ).fetchall()
    ]


def _downgrade_state_embeddings_to_legacy_primary_key(store) -> None:
    """Recreate the pre-#179 table shape without changing stored rows."""
    store._conn.execute("BEGIN IMMEDIATE")
    try:
        store._conn.execute(
            "DROP INDEX lcm_trajectory_state_embeddings_profile"
        )
        store._conn.execute(
            """
            ALTER TABLE lcm_trajectory_state_embeddings
            RENAME TO lcm_trajectory_state_embeddings_legacy_stage
            """
        )
        store._conn.execute(
            """
            CREATE TABLE lcm_trajectory_state_embeddings (
                state_id INTEGER PRIMARY KEY
                    REFERENCES lcm_trajectory_states(state_id) ON DELETE CASCADE,
                profile_digest TEXT NOT NULL
                    REFERENCES lcm_trajectory_state_embedding_profiles(
                        profile_digest
                    )
                    ON DELETE CASCADE,
                document_sha256 TEXT NOT NULL,
                vector BLOB NOT NULL,
                embedded_at REAL NOT NULL
            )
            """
        )
        store._conn.execute(
            """
            INSERT INTO lcm_trajectory_state_embeddings(
                state_id, profile_digest, document_sha256, vector, embedded_at
            )
            SELECT state_id, profile_digest, document_sha256, vector, embedded_at
            FROM lcm_trajectory_state_embeddings_legacy_stage
            """
        )
        store._conn.execute(
            "DROP TABLE lcm_trajectory_state_embeddings_legacy_stage"
        )
        store._conn.execute(
            """
            CREATE INDEX lcm_trajectory_state_embeddings_profile
            ON lcm_trajectory_state_embeddings(profile_digest, state_id)
            """
        )
        store._conn.commit()
    except Exception:
        store._conn.rollback()
        raise


def _build_invisible_semantic_store(tmp_path: Path, *, provider=None):
    """One trajectory whose answer state is lexically INVISIBLE (no query term)
    but is the semantic nearest neighbour (alpha-answer); plus a second lexical
    trajectory so the query pool is non-empty."""
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = provider or StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db",
        _identity(),
        asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=(
            "navigate homepage dashboard overview",         # 0: invisible filler
            "widget configuration export panel form",       # 1: the lexical seed
            "alpha-answer success banner shows view link",  # 2: invisible ANSWER
            "logout footer copyright notice",               # 3: invisible filler
        ),
    ))
    store.insert(_source(
        asset_root,
        trajectory_id="othertask",
        ordinal=1,
        goal="Export the report",
        texts=("report export toolbar button",),
    ))
    store.finalize(["answerpath", "othertask"])
    store.build_state_semantic_index(provider)
    return store


# --- default-off byte-identity ----------------------------------------------

def test_defaults_reproduce_current_bytes(tmp_path):
    store = _build_invisible_semantic_store(tmp_path)
    baseline = store.query(_QUERY, image_limit=0)
    baseline_telemetry = store.last_query_telemetry()
    explicit_off = store.query(_QUERY, image_limit=0, state_semantic_quota=0)
    off_telemetry = store.last_query_telemetry()
    assert [h.exact_ref for h in explicit_off] == [h.exact_ref for h in baseline]
    assert off_telemetry == baseline_telemetry
    # No telemetry key leaks into the default payload (frozen-run byte parity).
    assert "state_semantic_expansion" not in baseline_telemetry


# --- the core recall mechanism ----------------------------------------------

def test_expansion_pulls_lexically_invisible_state_into_pool(tmp_path):
    store = _build_invisible_semantic_store(tmp_path)
    answer = _state_id(store, "answerpath", 2)
    store.query(_QUERY, image_limit=0)
    assert answer not in _pool_state_ids(store), "answer state must start invisible"
    store.query(_QUERY, image_limit=0, state_semantic_quota=8)
    assert answer in _pool_state_ids(store)
    by_state = {entry["state_id"]: entry for entry in _admitted(store)}
    assert answer in by_state
    assert by_state[answer]["rank"] == 1  # the alpha state is the top-ranked


def test_expansion_can_fill_a_pure_lexical_miss(tmp_path):
    """The semantic tail exists for underfill, including an empty FTS pool."""
    store = _build_invisible_semantic_store(tmp_path)
    baseline = store.query("lexically absent zephyr phrase", image_limit=0)
    assert baseline == ()

    expanded = store.query(
        "lexically absent zephyr phrase",
        image_limit=0,
        include_adjacent=False,
        state_semantic_quota=1,
    )

    assert len(expanded) == 1
    assert expanded[0].match_kind == "state_semantic"
    assert _admitted(store)[0]["state_id"] == _state_id(
        store, expanded[0].trajectory_id, expanded[0].state_index
    )


def test_expansion_can_fill_a_termless_lexical_miss(tmp_path):
    store = _build_invisible_semantic_store(tmp_path)
    provider = store.embedding_provider

    baseline = store.query("🚀", image_limit=0)
    baseline_telemetry = store.last_query_telemetry()
    explicit_off = store.query("🚀", image_limit=0, state_semantic_quota=0)
    assert explicit_off == baseline == ()
    assert store.last_query_telemetry() == baseline_telemetry

    query_calls_before = provider.query_calls
    expanded = store.query(
        "🚀",
        image_limit=0,
        include_adjacent=False,
        state_semantic_quota=1,
    )

    assert len(expanded) == 1
    assert expanded[0].match_kind == "state_semantic"
    assert provider.query_calls == query_calls_before + 1
    assert store.last_query_telemetry()["source_candidate_ranks"] == []


def test_quota_caps_admissions(tmp_path):
    """Two lexically-invisible alpha states; a quota of 1 admits exactly one."""
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=(
            "widget configuration export panel form",   # lexical seed
            "alpha-answer first invisible banner",       # invisible alpha
            "alpha-answer second invisible banner",      # invisible alpha
        ),
    ))
    store.finalize(["answerpath"])
    store.build_state_semantic_index(provider)
    store.query(_QUERY, image_limit=0, state_semantic_quota=1)
    admitted = _admitted(store)
    assert len(admitted) == 1


def test_pool_incumbents_are_not_readmitted(tmp_path):
    """A state that already entered the pool lexically is never duplicated, even
    when it is also the semantic top."""
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="allmatch",
        ordinal=0,
        goal="Finish the setup flow",
        # Both matching states are ALSO alpha (lexical + semantic top); only the
        # third is a non-matching invisible alpha the arm can add.
        texts=(
            "widget configuration export alpha-answer intro",
            "widget configuration export alpha-answer detail",
            "alpha-answer plain closing remark",
        ),
    ))
    store.finalize(["allmatch"])
    store.build_state_semantic_index(provider)
    store.query(_QUERY, image_limit=0, state_semantic_quota=8)
    admitted = _admitted(store)
    only = _state_id(store, "allmatch", 2)
    assert [entry["state_id"] for entry in admitted] == [only]
    pool = [
        int(item["state_id"])
        for item in store.last_query_telemetry()["state_candidate_pool"]
    ]
    assert len(pool) == len(set(pool)), "pool must stay duplicate-free"


# --- anti-filler / additive-only controls -----------------------------------

def test_delivery_backfills_when_ranked_pool_fills_nucleus(tmp_path):
    """D-ARCH-2 proof on a full pool: an unused adjacency reserve is backfilled
    from the expanded ranked pool without disturbing incumbent rank order."""
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=provider,
    )
    order = []
    # Enough lexical trajectories to fill the delivered nucleus + backfill.
    for index in range(12):
        trajectory_id = f"lexical-{index:02d}"
        store.insert(_source(
            asset_root,
            trajectory_id=trajectory_id,
            ordinal=index,
            goal="Configure the widget",
            texts=(f"widget configuration export row {index}",),
        ))
        order.append(trajectory_id)
    # One trajectory carrying the lexically-invisible semantic target.
    store.insert(_source(
        asset_root,
        trajectory_id="target",
        ordinal=12,
        goal="Set up the dashboard gadget",
        texts=(
            "widget configuration export panel with all terms",
            "alpha-answer invisible aftermath confirmation",
        ),
    ))
    order.append("target")
    store.finalize(order)
    store.build_state_semantic_index(provider)

    baseline = [h.exact_ref for h in store.query(_QUERY, limit=16, image_limit=0)]
    expanded_hits = store.query(
        _QUERY, limit=16, image_limit=0, state_semantic_quota=16
    )
    expanded = [h.exact_ref for h in expanded_hits]
    # D-ARCH-2 fully utilizes the available ranked pool: all incumbents retain
    # their order and the newly admitted semantic state fills the unused slot.
    assert expanded[:-1] == baseline
    assert [
        (hit.trajectory_id, hit.state_index)
        for hit in expanded_hits
    ] == [
        *((f"lexical-{index:02d}", 0) for index in range(12)),
        ("target", 0),
        ("target", 1),
    ]
    assert len(expanded) == 14
    admitted = _admitted(store)
    assert admitted, "the pool itself must still gain a semantic entry"
    invisible = _state_id(store, "target", 1)
    assert invisible in {entry["state_id"] for entry in admitted}


def test_five_per_trajectory_cap_preserved_at_selection(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=provider,
    )
    # A long trajectory: one lexical seed + seven invisible alpha states, so the
    # arm can admit MORE than five states from a single trajectory to the pool.
    texts = ["widget configuration export summary step"]
    texts += [f"alpha-answer invisible step {index}" for index in range(7)]
    store.insert(_source(
        asset_root,
        trajectory_id="longtask",
        ordinal=0,
        goal="Complete the long procedure",
        texts=tuple(texts),
    ))
    store.finalize(["longtask"])
    store.build_state_semantic_index(provider)
    hits = store.query(
        _QUERY,
        image_limit=0,
        include_adjacent=False,  # nucleus only: isolate the selection cap
        state_semantic_quota=32,
    )
    per_trajectory: dict[str, int] = {}
    for hit in hits:
        per_trajectory[hit.trajectory_id] = per_trajectory.get(hit.trajectory_id, 0) + 1
    assert per_trajectory.get("longtask", 0) <= 5
    # ...even though MORE than five longtask states were admitted to the pool.
    assert len(_admitted(store)) == 7


def test_expanded_states_selected_only_from_the_tail(tmp_path):
    """Semantic-admitted states earn no BM25 rank: every ranked pool row still
    precedes every admitted state-semantic row in the candidate pool."""
    store = _build_invisible_semantic_store(tmp_path)
    store.query(_QUERY, image_limit=0, state_semantic_quota=8)
    telemetry = store.last_query_telemetry()
    pool = [int(item["state_id"]) for item in telemetry["state_candidate_pool"]]
    admitted_ids = {entry["state_id"] for entry in _admitted(store)}
    ranked_positions = [
        index for index, state_id in enumerate(pool) if state_id not in admitted_ids
    ]
    admitted_positions = [
        index for index, state_id in enumerate(pool) if state_id in admitted_ids
    ]
    assert admitted_positions, "admitted states must be visible in the pool"
    assert max(ranked_positions) < min(admitted_positions)


# --- backfill: resumability + inert-without-index ---------------------------

def test_backfill_is_idempotent_and_resumable(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=(
            "widget configuration export panel form",
            "alpha-answer success banner",
            "logout footer copyright notice",
        ),
    ))
    store.finalize(["answerpath"])
    first = store.build_state_semantic_index(provider)
    assert first["states_embedded"] == 3
    assert first["total_states"] == 3
    assert first["provider_calls"] >= 1
    # A re-run embeds nothing (every state already carries a row).
    provider.document_calls = 0
    second = store.build_state_semantic_index(provider)
    assert second["states_embedded"] == 0
    assert second["already_embedded"] == 3
    assert provider.document_calls == 0


def test_legacy_state_embeddings_migrate_idempotently_without_ranking_drift(
    tmp_path,
):
    store = _build_invisible_semantic_store(tmp_path)
    prior = store.active_state_semantic_profile()
    assert prior is not None
    prior_digest = str(prior["profile_digest"])
    _downgrade_state_embeddings_to_legacy_primary_key(store)
    rows_before = _profile_embedding_rows(store, prior_digest)
    rankings_before = _ranking_bytes(store)
    legacy_primary_key = tuple(
        str(row[1])
        for row in store._conn.execute(
            "PRAGMA table_info(lcm_trajectory_state_embeddings)"
        ).fetchall()
        if int(row[5]) > 0
    )
    assert legacy_primary_key == ("state_id",)

    store._ensure_state_semantic_schema()
    store._ensure_state_semantic_schema()

    migrated_primary_key = tuple(
        str(row[1])
        for row in sorted(
            store._conn.execute(
                "PRAGMA table_info(lcm_trajectory_state_embeddings)"
            ).fetchall(),
            key=lambda row: int(row[5]),
        )
        if int(row[5]) > 0
    )
    assert migrated_primary_key == ("profile_digest", "state_id")
    assert _profile_embedding_rows(store, prior_digest) == rows_before
    assert _ranking_bytes(store) == rankings_before


def test_legacy_state_embedding_migration_blocks_same_connection_reader(
    tmp_path,
):
    store = _build_invisible_semantic_store(tmp_path)
    prior = store.active_state_semantic_profile()
    assert prior is not None
    prior_digest = str(prior["profile_digest"])
    prior_dim = int(prior["dim"])
    expected_state_ids = [
        row[0] for row in _profile_embedding_rows(store, prior_digest)
    ]
    _downgrade_state_embeddings_to_legacy_primary_key(store)
    store._state_semantic_cache = None

    table_dropped = threading.Event()
    release_migration = threading.Event()
    reader_started = threading.Event()
    reader_finished = threading.Event()
    reader_results = []
    reader_errors = []
    migration_errors = []

    class PauseAfterLegacyDrop:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, statement, parameters=()):
            cursor = self.connection.execute(statement, parameters)
            normalized = " ".join(statement.upper().split())
            if normalized == "DROP TABLE LCM_TRAJECTORY_STATE_EMBEDDINGS":
                table_dropped.set()
                if not release_migration.wait(timeout=5):
                    raise RuntimeError("timed out waiting to resume migration")
            return cursor

    store._conn = PauseAfterLegacyDrop(store._conn)

    def read_matrix():
        reader_started.set()
        try:
            reader_results.append(
                store._load_state_semantic_matrix(prior_digest, prior_dim)
            )
        except Exception as exc:
            reader_errors.append(exc)
        finally:
            reader_finished.set()

    def migrate():
        try:
            store._ensure_state_semantic_schema()
        except Exception as exc:
            migration_errors.append(exc)

    migration_thread = threading.Thread(target=migrate)
    migration_thread.start()
    assert table_dropped.wait(timeout=2)
    reader_thread = threading.Thread(target=read_matrix)
    reader_thread.start()
    assert reader_started.wait(timeout=2)
    completed_while_table_missing = reader_finished.wait(timeout=0.5)
    release_migration.set()
    try:
        migration_thread.join(timeout=2)
        reader_thread.join(timeout=2)
    finally:
        release_migration.set()

    assert not migration_thread.is_alive()
    reader_thread.join(timeout=2)
    assert not reader_thread.is_alive()
    assert migration_errors == []
    assert not completed_while_table_missing
    assert reader_errors == []
    assert len(reader_results) == 1
    assert reader_results[0][0] == expected_state_ids


def test_legacy_state_embedding_migration_adopts_sole_inactive_digest(tmp_path):
    store = _build_invisible_semantic_store(tmp_path)
    prior = store.active_state_semantic_profile()
    assert prior is not None
    prior_digest = str(prior["profile_digest"])
    _downgrade_state_embeddings_to_legacy_primary_key(store)
    rows_before = _profile_embedding_rows(store, prior_digest)
    store._conn.execute(
        "UPDATE lcm_trajectory_state_embedding_profiles SET active = 0"
    )
    store._conn.commit()

    store._ensure_state_semantic_schema()
    store._ensure_state_semantic_schema()

    assert store.active_state_semantic_profile() is None
    assert _profile_embedding_rows(store, prior_digest) == rows_before


def test_legacy_state_embedding_migration_refuses_ambiguous_inactive_rows(
    tmp_path,
):
    store = _build_invisible_semantic_store(tmp_path)
    prior = store.active_state_semantic_profile()
    assert prior is not None
    _downgrade_state_embeddings_to_legacy_primary_key(store)
    alternate_digest = "alternate-profile"
    store._conn.execute(
        """
        INSERT INTO lcm_trajectory_state_embedding_profiles(
            profile_digest, provider, model_name, dim, document_version,
            source_manifest_digest, state_count, active, created_at
        )
        SELECT ?, provider, ?, dim, document_version, source_manifest_digest,
               state_count, 0, created_at
        FROM lcm_trajectory_state_embedding_profiles
        WHERE profile_digest = ?
        """,
        (alternate_digest, "alternate-model", prior["profile_digest"]),
    )
    store._conn.execute(
        "UPDATE lcm_trajectory_state_embedding_profiles SET active = 0"
    )
    store._conn.execute(
        """
        UPDATE lcm_trajectory_state_embeddings
        SET profile_digest = ?
        WHERE state_id = (
            SELECT MAX(state_id) FROM lcm_trajectory_state_embeddings
        )
        """,
        (alternate_digest,),
    )
    store._conn.commit()

    with pytest.raises(
        TrajectoryStoreError,
        match="multiple profile digests and no active profile.*fresh backfill",
    ):
        store._ensure_state_semantic_schema()

    primary_key = tuple(
        str(row[1])
        for row in store._conn.execute(
            "PRAGMA table_info(lcm_trajectory_state_embeddings)"
        ).fetchall()
        if int(row[5]) > 0
    )
    assert primary_key == ("state_id",)


def test_legacy_state_embedding_migration_refuses_mixed_active_rows(tmp_path):
    store = _build_invisible_semantic_store(tmp_path)
    prior = store.active_state_semantic_profile()
    assert prior is not None
    _downgrade_state_embeddings_to_legacy_primary_key(store)
    alternate_digest = "alternate-profile"
    store._conn.execute(
        """
        INSERT INTO lcm_trajectory_state_embedding_profiles(
            profile_digest, provider, model_name, dim, document_version,
            source_manifest_digest, state_count, active, created_at
        )
        SELECT ?, provider, ?, dim, document_version, source_manifest_digest,
               state_count, 0, created_at
        FROM lcm_trajectory_state_embedding_profiles
        WHERE profile_digest = ?
        """,
        (alternate_digest, "alternate-model", prior["profile_digest"]),
    )
    store._conn.execute(
        """
        UPDATE lcm_trajectory_state_embeddings
        SET profile_digest = ?
        WHERE state_id = (
            SELECT MAX(state_id) FROM lcm_trajectory_state_embeddings
        )
        """,
        (alternate_digest,),
    )
    store._conn.commit()

    with pytest.raises(
        TrajectoryStoreError,
        match="legacy state embeddings do not all match the active profile",
    ):
        store._ensure_state_semantic_schema()

    primary_key = tuple(
        str(row[1])
        for row in store._conn.execute(
            "PRAGMA table_info(lcm_trajectory_state_embeddings)"
        ).fetchall()
        if int(row[5]) > 0
    )
    assert primary_key == ("state_id",)


@pytest.mark.parametrize("fail_after", [0, 1, 2])
def test_interrupted_profile_rebuild_keeps_prior_profile_serving_until_cutover(
    tmp_path, fail_after,
):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    initial = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=initial,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=(
            "widget configuration export panel form",
            "alpha-answer success banner",
            "logout footer copyright notice",
        ),
    ))
    store.finalize(["answerpath"])
    store.build_state_semantic_index(initial)
    prior = store.active_state_semantic_profile()
    assert prior is not None
    prior_digest = str(prior["profile_digest"])
    prior_rows = _profile_embedding_rows(store, prior_digest)
    prior_rankings = _ranking_bytes(store)

    interrupted = InterruptingStateVectorProvider(
        model_id="fake-state-v2", fail_after=fail_after
    )
    with pytest.raises(RuntimeError, match="interrupted backfill"):
        store.build_state_semantic_index(
            interrupted, batch_max_items=1, batch_token_budget=100_000
        )

    serving = store.active_state_semantic_profile()
    assert serving is not None
    assert str(serving["profile_digest"]) == prior_digest
    assert store._conn.execute(
        "SELECT COUNT(*) FROM lcm_trajectory_state_embedding_profiles "
        "WHERE active = 1"
    ).fetchone()[0] == 1
    staged = store._conn.execute(
        """
        SELECT profile_digest, active
        FROM lcm_trajectory_state_embedding_profiles
        WHERE model_name = ?
        """,
        ("fake-state-v2",),
    ).fetchone()
    assert staged is not None and int(staged["active"]) == 0
    assert staged["profile_digest"] != prior_digest
    staged_count = store._conn.execute(
        "SELECT COUNT(*) FROM lcm_trajectory_state_embeddings "
        "WHERE profile_digest = ?",
        (staged["profile_digest"],),
    ).fetchone()[0]
    assert staged_count == fail_after
    assert _profile_embedding_rows(store, prior_digest) == prior_rows
    assert _ranking_bytes(store) == prior_rankings

    interrupted.fail_after = None
    resumed = store.build_state_semantic_index(
        interrupted, batch_max_items=1, batch_token_budget=100_000
    )
    active = store.active_state_semantic_profile()
    assert resumed["already_embedded"] == fail_after
    assert resumed["states_embedded"] == 3 - fail_after
    assert active is not None
    assert active["model_name"] == "fake-state-v2"
    assert active["profile_digest"] == staged["profile_digest"]
    assert store._conn.execute(
        "SELECT COUNT(*) FROM lcm_trajectory_state_embedding_profiles "
        "WHERE active = 1"
    ).fetchone()[0] == 1
    assert _profile_embedding_rows(store, prior_digest) == prior_rows


def test_forced_same_profile_rebuild_refuses_to_mutate_serving_rows(
    tmp_path, monkeypatch,
):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    initial = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=initial,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=(
            "widget configuration export panel form",
            "alpha-answer success banner",
            "logout footer copyright notice",
        ),
    ))
    store.finalize(["answerpath"])
    completed = store.build_state_semantic_index(initial)
    prior_rows = _profile_embedding_rows(store, completed["profile_digest"])
    prior_rankings = _ranking_bytes(store)

    interrupted = InterruptingStateVectorProvider(
        model_id=initial.model_id, fail_after=1
    )
    interrupted.fail_queries = True
    original_active_profile = store.active_state_semantic_profile
    guard_transaction_states = []

    def active_profile_with_transaction_probe():
        guard_transaction_states.append(store._conn.in_transaction)
        return original_active_profile()

    monkeypatch.setattr(
        store,
        "active_state_semantic_profile",
        active_profile_with_transaction_probe,
    )
    with pytest.raises(
        TrajectoryStoreError,
        match="cannot rebuild the active state semantic profile in place",
    ):
        store.build_state_semantic_index(
            interrupted,
            resume=False,
            batch_max_items=1,
            batch_token_budget=100_000,
        )

    assert guard_transaction_states == [True]
    assert not store._conn.in_transaction
    assert interrupted.query_calls == 0
    monkeypatch.setattr(
        store, "active_state_semantic_profile", original_active_profile
    )
    assert interrupted.document_calls == 0
    assert _profile_embedding_rows(store, completed["profile_digest"]) == prior_rows
    assert _ranking_bytes(store) == prior_rankings


def test_forced_distinct_profile_rebuild_replaces_staged_rows_and_cuts_over(
    tmp_path,
):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    initial = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=initial,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=(
            "widget configuration export panel form",
            "alpha-answer success banner",
            "logout footer copyright notice",
        ),
    ))
    store.finalize(["answerpath"])
    store.build_state_semantic_index(initial)
    prior = store.active_state_semantic_profile()
    assert prior is not None
    prior_digest = str(prior["profile_digest"])
    prior_rows = _profile_embedding_rows(store, prior_digest)

    distinct = InterruptingStateVectorProvider(
        model_id="fake-state-v2", fail_after=1
    )
    with pytest.raises(RuntimeError, match="interrupted backfill"):
        store.build_state_semantic_index(
            distinct, batch_max_items=1, batch_token_budget=100_000
        )
    staged = store._conn.execute(
        """
        SELECT profile_digest
        FROM lcm_trajectory_state_embedding_profiles
        WHERE model_name = ?
        """,
        (distinct.model_id,),
    ).fetchone()
    assert staged is not None
    distinct_digest = str(staged["profile_digest"])
    assert len(_profile_embedding_rows(store, distinct_digest)) == 1
    distinct.fail_after = None

    observed_staged_counts = []
    statements = []

    def observe_rebuild_progress(_stats):
        observed_staged_counts.append(store._conn.execute(
            "SELECT COUNT(*) FROM lcm_trajectory_state_embeddings "
            "WHERE profile_digest = ?",
            (distinct_digest,),
        ).fetchone()[0])

    store._conn.set_trace_callback(statements.append)
    try:
        rebuilt = store.build_state_semantic_index(
            distinct,
            resume=False,
            batch_max_items=1,
            batch_token_budget=100_000,
            progress_callback=observe_rebuild_progress,
        )
    finally:
        store._conn.set_trace_callback(None)

    cutover_updates = [
        statement
        for statement in statements
        if "INSERT INTO LCM_TRAJECTORY_STATE_EMBEDDING_PROFILES"
        in statement.upper()
        and "DO UPDATE SET" in statement.upper()
        and "ACTIVE = EXCLUDED.ACTIVE" in statement.upper()
    ]
    active = store.active_state_semantic_profile()
    predecessor = store._conn.execute(
        """
        SELECT active
        FROM lcm_trajectory_state_embedding_profiles
        WHERE profile_digest = ?
        """,
        (prior_digest,),
    ).fetchone()

    assert observed_staged_counts[0] == 0
    assert distinct.query_calls == 2
    assert rebuilt["profile_digest"] == distinct_digest
    assert rebuilt["already_embedded"] == 0
    assert rebuilt["states_embedded"] == 3
    assert active is not None
    assert str(active["profile_digest"]) == distinct_digest
    assert predecessor is not None and int(predecessor["active"]) == 0
    assert len(cutover_updates) == 1
    assert _profile_embedding_rows(store, prior_digest) == prior_rows
    assert len(_profile_embedding_rows(store, distinct_digest)) == 3


def test_slow_overlapping_builder_cannot_rewrite_profile_after_cutover(
    tmp_path,
):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    initial = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=initial,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=(
            "widget configuration export panel form",
            "alpha-answer success banner",
            "logout footer copyright notice",
        ),
    ))
    store.finalize(["answerpath"])
    store.build_state_semantic_index(initial)

    slow_started = threading.Event()
    release_slow = threading.Event()
    slow = CoordinatedStateVectorProvider(
        model_id="fake-state-overlap",
        document_vector=[0.0, 1.0, 0.0],
        started=slow_started,
        release=release_slow,
    )
    fast = CoordinatedStateVectorProvider(
        model_id=slow.model_id,
        document_vector=[1.0, 0.0, 0.0],
    )
    slow_errors = []

    def run_slow_builder():
        try:
            store.build_state_semantic_index(slow)
        except Exception as exc:
            slow_errors.append(exc)

    slow_thread = threading.Thread(target=run_slow_builder)
    slow_thread.start()
    assert slow_started.wait(timeout=2)

    completed = store.build_state_semantic_index(fast)
    serving_rows = _profile_embedding_rows(
        store, completed["profile_digest"]
    )
    release_slow.set()
    slow_thread.join(timeout=5)

    assert not slow_thread.is_alive()
    assert len(slow_errors) == 1
    assert isinstance(slow_errors[0], TrajectoryStoreError)
    assert "became active during backfill" in str(slow_errors[0])
    active = store.active_state_semantic_profile()
    assert active is not None
    assert str(active["profile_digest"]) == completed["profile_digest"]
    assert _profile_embedding_rows(
        store, completed["profile_digest"]
    ) == serving_rows


def test_dimension_probe_is_bounded_by_document_token_budget(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = ProbeBudgetProvider(token_limit=5)
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=("alpha-answer " + ("token " * 100),),
    ))
    store.finalize(["answerpath"])

    stats = store.build_state_semantic_index(
        provider, document_token_budget=5, batch_token_budget=50
    )

    assert stats["states_embedded"] == 1
    assert provider.probe_documents
    assert token_module.count_tokens(provider.probe_documents[0]) <= 5
    assert stats["provider_calls"] == provider.query_calls + provider.document_calls
    assert stats["billed_tokens"] == provider.usage_tokens_total


def test_fallback_chunks_obey_the_shared_token_estimator(tmp_path, monkeypatch):
    monkeypatch.setattr(token_module, "_get_encoder", lambda: None)
    store = object.__new__(TrajectoryStore)
    document = "漢字かな交じり文" * 20

    chunks = store._state_token_chunks(document, token_budget=5)

    assert "".join(chunks) == document
    assert len(chunks) > 1
    assert all(token_module._fallback_token_estimate(chunk) <= 5 for chunk in chunks)


def test_tiktoken_chunks_preserve_multibyte_boundaries(monkeypatch):
    tiktoken = pytest.importorskip("tiktoken")
    encoder = tiktoken.get_encoding("cl100k_base")
    monkeypatch.setattr(token_module, "_get_encoder", lambda: encoder)
    store = object.__new__(TrajectoryStore)
    document = "漢字かな交じり文" * 20
    token_budget = 7

    token_ids = encoder.encode(document)
    naive_rejoin = "".join(
        encoder.decode(token_ids[start:start + token_budget])
        for start in range(0, len(token_ids), token_budget)
    )
    assert "\ufffd" in naive_rejoin

    chunks = store._state_token_chunks(document, token_budget=token_budget)

    assert "".join(chunks) == document
    assert all("\ufffd" not in chunk for chunk in chunks)
    assert len(chunks) > 1
    assert all(len(encoder.encode(chunk)) <= token_budget for chunk in chunks)


def test_exact_budget_document_stays_normal_and_oversize_is_chunked(tmp_path):
    """The exact-budget document remains normal while only the oversized state
    takes the chunk path; both still yield exactly one usable vector."""
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db", _identity(), asset_root=asset_root,
        embedding_provider=provider,
    )
    exact_budget_text = "widget configuration export panel form"
    document_token_budget = token_module.count_tokens(exact_budget_text)
    long_text = "alpha-answer " + ("token " * 400)  # well over the exact fixture
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=(
            exact_budget_text,
            long_text,
        ),
    ))
    store.finalize(["answerpath"])
    stats = store.build_state_semantic_index(
        provider,
        document_token_budget=document_token_budget,
        batch_token_budget=50,
        batch_max_items=4,
    )
    # The budget is derived from the fixture's own tokenizer count, so the
    # exact-budget document stays normal (tokens == budget, not > budget) and
    # only the oversized state is chunked — no word-count/tokenizer confusion.
    assert token_module.count_tokens(exact_budget_text) == document_token_budget
    assert token_module.count_tokens(long_text) > document_token_budget
    assert stats["chunked_states"] == 1
    assert stats["states_embedded"] == 2
    oversize = _state_id(store, "answerpath", 1)
    row = store._conn.execute(
        "SELECT vector FROM lcm_trajectory_state_embeddings WHERE state_id = ?",
        (oversize,),
    ).fetchone()
    assert row is not None and len(bytes(row["vector"])) == stats["dim"] * 4


def test_oversize_chunks_pack_by_item_and_token_budgets(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = RequestBudgetProvider(token_limit=9)
    store = TrajectoryStore(
        tmp_path / "lcm.db",
        _identity(),
        asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(
        _source(
            asset_root,
            trajectory_id="oversize",
            ordinal=0,
            goal="Exercise chunk packing",
            texts=("alpha-answer " + ("token " * 100),),
        )
    )
    store.finalize(["oversize"])

    stats = store.build_state_semantic_index(
        provider,
        document_token_budget=5,
        batch_token_budget=9,
        batch_max_items=4,
    )

    assert stats["states_embedded"] == 1
    assert provider.request_token_counts
    assert max(provider.request_token_counts) <= 9


def test_smaller_batch_budget_routes_normal_document_through_chunks(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = RequestBudgetProvider(token_limit=5)
    store = TrajectoryStore(
        tmp_path / "lcm.db",
        _identity(),
        asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="batch-budget",
        ordinal=0,
        goal="Honor the request cap",
        texts=("alpha-answer " + ("token " * 20),),
    ))
    store.finalize(["batch-budget"])

    stats = store.build_state_semantic_index(
        provider,
        document_token_budget=50,
        batch_token_budget=5,
        batch_max_items=4,
    )

    assert stats["states_embedded"] == 1
    assert stats["chunked_states"] == 1
    assert provider.request_token_counts
    assert max(provider.request_token_counts) <= 5


def test_oversize_progress_can_stop_between_chunks_with_partial_spend(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db",
        _identity(),
        asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(
        _source(
            asset_root,
            trajectory_id="oversize-cap",
            ordinal=0,
            goal="Trip the cap between chunk requests",
            texts=("alpha-answer " + ("token " * 100),),
        )
    )
    store.finalize(["oversize-cap"])
    ledger: list[dict] = []

    class CostCapExceeded(RuntimeError):
        pass

    def record_progress(stats):
        ledger.append(dict(stats))
        if stats["billed_tokens"] > 1:
            raise CostCapExceeded("low test cap exceeded")

    with pytest.raises(CostCapExceeded, match="low test cap"):
        store.build_state_semantic_index(
            provider,
            document_token_budget=5,
            batch_token_budget=5,
            batch_max_items=1,
            progress_callback=record_progress,
        )

    assert provider.document_calls == 1
    assert ledger[-1]["provider_calls"] == provider.query_calls + provider.document_calls
    assert ledger[-1]["billed_tokens"] == provider.usage_tokens_total
    assert ledger[-1]["states_embedded"] == 0


def test_normal_batch_persist_failure_still_ledgers_spend(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db",
        _identity(),
        asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(
        _source(
            asset_root,
            trajectory_id="persist-failure",
            ordinal=0,
            goal="Preserve spend before persistence",
            texts=("alpha-answer account settings",),
        )
    )
    store.finalize(["persist-failure"])
    store._ensure_state_semantic_schema()
    store._conn.execute(
        """
        CREATE TRIGGER fail_state_embedding_insert
        BEFORE INSERT ON lcm_trajectory_state_embeddings
        BEGIN
            SELECT RAISE(FAIL, 'simulated persist failure');
        END
        """
    )
    ledger: list[dict] = []

    with pytest.raises(sqlite3.IntegrityError, match="simulated persist failure"):
        store.build_state_semantic_index(
            provider,
            progress_callback=lambda stats: ledger.append(dict(stats)),
        )

    assert provider.document_calls == 1
    assert ledger[-1]["provider_calls"] == provider.query_calls + provider.document_calls
    assert ledger[-1]["billed_tokens"] == provider.usage_tokens_total
    assert ledger[-1]["states_embedded"] == 0


def test_dimension_probe_progress_can_stop_before_document_request(tmp_path):
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    provider = StateVectorProvider()
    store = TrajectoryStore(
        tmp_path / "lcm.db",
        _identity(),
        asset_root=asset_root,
        embedding_provider=provider,
    )
    store.insert(
        _source(
            asset_root,
            trajectory_id="probe-cap",
            ordinal=0,
            goal="Trip the cap after the dimension probe",
            texts=("alpha-answer account settings",),
        )
    )
    store.finalize(["probe-cap"])
    ledger: list[dict] = []

    class CostCapExceeded(RuntimeError):
        pass

    def record_progress(stats):
        ledger.append(dict(stats))
        raise CostCapExceeded("probe spent the test cap")

    with pytest.raises(CostCapExceeded, match="probe spent the test cap"):
        store.build_state_semantic_index(
            provider,
            progress_callback=record_progress,
        )

    assert provider.query_calls == 1
    assert provider.document_calls == 0
    assert len(ledger) == 1
    assert ledger[0]["provider_calls"] == 1
    assert ledger[0]["billed_tokens"] == 1
    assert ledger[0]["states_embedded"] == 0


def test_state_query_provider_failure_degrades_to_lexical(tmp_path):
    provider = StateVectorProvider()
    store = _build_invisible_semantic_store(tmp_path, provider=provider)
    baseline = store.query(_QUERY, image_limit=0)
    metrics_before = store.semantic_metrics()
    attempts_before = store.semantic_attempt_counters()
    provider.fail_queries = True

    degraded = store.query(
        _QUERY, image_limit=0, state_semantic_quota=8
    )

    assert [hit.exact_ref for hit in degraded] == [
        hit.exact_ref for hit in baseline
    ]
    metrics_after = store.semantic_metrics()
    attempts_after = store.semantic_attempt_counters()
    assert metrics_after["fallbacks"] == metrics_before["fallbacks"] + 1
    assert attempts_after["fallbacks_by_reason"].get("other", 0) == (
        attempts_before["fallbacks_by_reason"].get("other", 0) + 1
    )


def test_state_query_embedding_is_counted_in_semantic_usage(tmp_path):
    provider = StateVectorProvider()
    store = _build_invisible_semantic_store(tmp_path, provider=provider)
    before = store.semantic_metrics()

    store.query(_QUERY, image_limit=0, state_semantic_quota=1)

    after = store.semantic_metrics()
    assert after["query_calls"] == before["query_calls"] + 1
    assert after["query_tokens"] == before["query_tokens"] + 1


def test_state_matrix_cache_refreshes_after_same_profile_rewrite(tmp_path):
    provider = StateVectorProvider()
    store = _build_invisible_semantic_store(tmp_path, provider=provider)
    alpha = _state_id(store, "answerpath", 2)
    beta = _state_id(store, "othertask", 0)
    assert store._semantic_state_ranks("query", 1)[0][0] == alpha

    store._conn.execute(
        """
        UPDATE lcm_trajectory_state_embeddings
        SET vector = CASE state_id
                WHEN ? THEN ?
                WHEN ? THEN ?
                ELSE vector
            END,
            embedded_at = embedded_at + 1000
        WHERE state_id IN (?, ?)
        """,
        (
            alpha, struct.pack("<3f", 0.0, 0.0, 1.0),
            beta, struct.pack("<3f", 1.0, 0.0, 0.0),
            alpha, beta,
        ),
    )
    store._conn.commit()

    assert store._semantic_state_ranks("query", 1)[0][0] == beta


def test_arm_inert_without_provider_or_index(tmp_path):
    """Knob-on but no state index (or no provider) is a no-op, not an error."""
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    # No provider attached and no state backfill performed.
    store = TrajectoryStore(tmp_path / "lcm.db", _identity(), asset_root=asset_root)
    store.insert(_source(
        asset_root,
        trajectory_id="answerpath",
        ordinal=0,
        goal="Update the account settings",
        texts=("widget configuration export panel form", "plain closing remark"),
    ))
    store.finalize(["answerpath"])
    baseline = [h.exact_ref for h in store.query(_QUERY, image_limit=0)]
    with_knob = [
        h.exact_ref
        for h in store.query(_QUERY, image_limit=0, state_semantic_quota=8)
    ]
    assert with_knob == baseline
    assert _admitted(store) == []
