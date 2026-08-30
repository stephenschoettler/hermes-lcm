"""Candidate-composition repair policies (issue #127): lexical floor + arm quota.

These exercise the semantic-"magnet" pathology on a synthetic corpus: a cluster
of semantic-top distractor trajectories monopolises the fused nucleus and
initially displaces the strongest pure-lexical winner (a SOURCE_MISS).
D-ARCH-2's default adjacency-reserve backfill re-admits that winner, while
Policy A (``lexical_floor``) and Policy D (``arm_quota``) still change how the
winner is protected within the composed delivery.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from hermes_lcm.trajectory_store import (
    CorpusIdentity,
    TrajectorySource,
    TrajectoryState,
    TrajectoryStore,
)


class MagnetProvider:
    """Ranks any trajectory whose text mentions 'profile toolbar' as semantically
    top, regardless of the query -- the coarse whole-trajectory 'magnet'."""

    provider_id = "fake"
    model_id = "fake-trajectory-v1"
    dim = 2

    def __init__(self) -> None:
        self.last_usage_tokens = 0

    def embed_documents(self, texts):
        self.last_usage_tokens = sum(max(1, len(str(t)) // 4) for t in texts)
        return [
            [1.0, 0.0] if "profile toolbar" in str(t).casefold() else [0.0, 1.0]
            for t in texts
        ]

    def embed_query(self, text):  # noqa: ARG002
        self.last_usage_tokens = 1
        return [1.0, 0.0]


def _identity() -> CorpusIdentity:
    return CorpusIdentity(
        dataset_name="example/composition",
        dataset_revision="rev-composition",
        harness_commit="harness-composition-1",
        tier="small",
        domain="web",
        ingest_config_digest="composition-test-v1",
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


def _build_magnet_store(tmp_path: Path):
    """A pure-lexical winner ('target') displaced by 12 semantic-top distractors."""
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    store = TrajectoryStore(
        tmp_path / "lcm.db",
        _identity(),
        asset_root=asset_root,
        embedding_provider=MagnetProvider(),
        semantic_top_trajectories=12,
    )
    # Strongest BM25 hit for the query, but NOT semantic-top (no 'profile toolbar').
    store.insert(_source(
        asset_root,
        trajectory_id="target",
        ordinal=0,
        goal="Configure the widget",
        texts=("widget configuration export panel with all three terms",),
    ))
    order = ["target"]
    # 12 distractors: semantic-top ('profile toolbar') + a weak lexical hit ('export').
    for index in range(12):
        trajectory_id = f"distractor-{index:02d}"
        store.insert(_source(
            asset_root,
            trajectory_id=trajectory_id,
            ordinal=index + 1,
            goal="Inspect profile toolbar",
            texts=(f"profile toolbar export note {index}",),
        ))
        order.append(trajectory_id)
    store.finalize(order)
    store.build_semantic_index()
    return store


def _delivered_trajectories(hits):
    return [hit.trajectory_id for hit in hits]


def test_default_backfill_readmits_the_magnet_displacement(tmp_path: Path):
    # D-ARCH-2: unused adjacency reserve backfills the displaced lexical winner
    # after the fused nucleus, preserving the fused rank order.
    store = _build_magnet_store(tmp_path)
    default_hits = store.query(_QUERY, limit=16, image_limit=0)
    default_refs = [hit.exact_ref for hit in default_hits]
    default_trajectories = _delivered_trajectories(default_hits)

    assert default_trajectories[-1] == "target"
    assert all(t.startswith("distractor-") for t in default_trajectories[:-1])
    # The default kwargs are byte-identical to the explicit no-op knob values.
    assert [h.exact_ref for h in store.query(_QUERY, limit=16, image_limit=0, lexical_floor=0)] == default_refs
    assert [h.exact_ref for h in store.query(_QUERY, limit=16, image_limit=0, arm_quota=None)] == default_refs


def test_policy_a_lexical_floor_promotes_the_backfilled_winner(tmp_path: Path):
    # D-ARCH-2 already re-admits the winner by default; Policy A still protects
    # it as the first delivered lexical incumbent rather than a tail backfill.
    store = _build_magnet_store(tmp_path)
    default = _delivered_trajectories(store.query(_QUERY, limit=16, image_limit=0))
    assert default[-1] == "target"

    floored = store.query(_QUERY, limit=16, image_limit=0, lexical_floor=1)
    assert _delivered_trajectories(floored)[0] == "target"
    assert len({hit.exact_ref for hit in floored}) == len(floored)  # no duplicates


def test_policy_d_arm_quota_promotes_the_backfilled_winner(tmp_path: Path):
    # D-ARCH-2 already re-admits the winner by default; Policy D still promotes
    # it through the lexical arm while retaining the semantic arm.
    store = _build_magnet_store(tmp_path)
    default = _delivered_trajectories(store.query(_QUERY, limit=16, image_limit=0))
    assert default[-1] == "target"

    union = store.query(_QUERY, limit=16, image_limit=0, arm_quota=(6, 5))
    trajectories = _delivered_trajectories(union)
    assert trajectories[0] == "target"
    # The semantic arm is still represented (quota union, not lexical-only).
    assert any(t.startswith("distractor-") for t in trajectories)
    assert len({hit.exact_ref for hit in union}) == len(union)


def test_merge_arms_dedups_and_backfills_without_wasting_quota():
    # Deterministic unit check of the arm merge: overlapping arms must not waste a
    # quota slot on a duplicate, and a short arm is backfilled by the other.
    def _rows(ids):
        return [{"state_id": i, "trajectory_id": f"t{i}"} for i in ids]

    arm_lex = _rows([1, 2, 3, 4])
    arm_sem = _rows([2, 3, 5, 6])  # 2,3 overlap with lex
    merged = TrajectoryStore._merge_arms(arm_lex, arm_sem, limit=6, q_lex=2, q_sem=2)
    ids = [row["state_id"] for row in merged]
    # round 1: lex[1,2] then sem[3,5] (2 is a deduped skip, not a wasted slot);
    # round 2: lex[4] (3 deduped) then sem[6].
    assert ids == [1, 2, 3, 5, 4, 6]
    assert len(ids) == len(set(ids))


def test_merge_arms_hybrid_floor_reserves_lexical_incumbents_first():
    # A+D hybrid (issue #127): floor_k reserves the top pure-lexical rows a slot
    # BEFORE the quota round-robin, then the round-robin fills the rest.
    # floor_k=0 (the default) is byte-identical to the pure Policy D merge.
    def _rows(ids):
        return [{"state_id": i, "trajectory_id": f"t{i}"} for i in ids]

    arm_lex = _rows([1, 2, 3, 4])
    arm_sem = _rows([2, 3, 5, 6])  # 2,3 overlap with lex
    hybrid = TrajectoryStore._merge_arms(
        arm_lex, arm_sem, limit=6, q_lex=2, q_sem=2, floor_k=1
    )
    ids = [row["state_id"] for row in hybrid]
    assert ids[0] == 1  # top pure-lexical incumbent reserved by the floor first
    assert ids == [1, 2, 3, 5, 6, 4]
    assert len(ids) == len(set(ids))
    # floor_k=0 (and the omitted default) reproduce the pure Policy D bytes.
    pure_d = [1, 2, 3, 5, 4, 6]
    assert [r["state_id"] for r in TrajectoryStore._merge_arms(
        arm_lex, arm_sem, limit=6, q_lex=2, q_sem=2)] == pure_d
    assert [r["state_id"] for r in TrajectoryStore._merge_arms(
        arm_lex, arm_sem, limit=6, q_lex=2, q_sem=2, floor_k=0)] == pure_d


def test_hybrid_floor_plus_quota_promotes_winner_and_composes(tmp_path: Path):
    # D-ARCH-2 backfills the winner by default; the A+D hybrid composes by
    # protecting it first while the arm quota keeps the semantic arm represented.
    # arm_quota with lexical_floor=0 stays byte-identical to pure Policy D.
    store = _build_magnet_store(tmp_path)
    default = _delivered_trajectories(store.query(_QUERY, limit=16, image_limit=0))
    assert default[-1] == "target"

    hybrid = store.query(_QUERY, limit=16, image_limit=0, lexical_floor=1, arm_quota=(6, 5))
    trajectories = _delivered_trajectories(hybrid)
    assert trajectories[0] == "target"  # lexical incumbent protected by the floor
    assert any(t.startswith("distractor-") for t in trajectories)  # semantic arm kept
    assert len({hit.exact_ref for hit in hybrid}) == len(hybrid)

    pure_d_refs = [h.exact_ref for h in store.query(_QUERY, limit=16, image_limit=0, arm_quota=(6, 5))]
    assert [
        h.exact_ref
        for h in store.query(_QUERY, limit=16, image_limit=0, lexical_floor=0, arm_quota=(6, 5))
    ] == pure_d_refs


def test_policies_are_noops_without_semantic_ranks(tmp_path: Path):
    # No embedding provider -> no semantic boost -> the fused order IS the lexical
    # order, so both policies degenerate to the historical selection.
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    store = TrajectoryStore(
        tmp_path / "lcm.db",
        _identity(),
        asset_root=asset_root,
        embedding_provider=None,
    )
    store.insert(_source(
        asset_root,
        trajectory_id="target",
        ordinal=0,
        goal="Configure the widget",
        texts=("widget configuration export panel",),
    ))
    store.finalize(["target"])
    baseline = [h.exact_ref for h in store.query(_QUERY, limit=8, image_limit=0)]
    assert baseline
    assert [h.exact_ref for h in store.query(_QUERY, limit=8, image_limit=0, lexical_floor=3)] == baseline
    assert [h.exact_ref for h in store.query(_QUERY, limit=8, image_limit=0, arm_quota=(6, 5))] == baseline
