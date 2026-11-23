"""Turn-start reconcile cost must be linear in the replayed/stored rows.

Every gateway turn re-binds the LCM engine to the session, which schedules an
ingest-cursor reconcile against the durable store.  The post-compaction active
context ([scaffold]+[summary]+[already-stored fresh tail]) never matches a
stored suffix at the top cursor, so ``_find_reconciled_cursor_for_store_tail``
scans every cursor down to the scaffold head.  It used to rebuild every
filtered list and recompute every replay identity for ``messages[:cursor]`` at
EACH cursor: O(n^2) identity computations (JSON canonicalisation plus
externalized-payload reads).  Measured before the fix on a synthetic 5k-row
session: 800 replayed rows -> 972,426 identity computations and 47 s; that
shows up as seconds-to-minutes of dead air before each turn on long sessions.

These tests count the real identity computations (not wall time) so they are
deterministic, and they pin the ingest OUTCOME so the speed-up cannot come
from skipping work.
"""
from __future__ import annotations

import json
import random
import sqlite3

import pytest

from hermes_lcm import reconcile as reconcile_mod
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SCAFFOLD = (
    "[Note: This conversation uses Lossless Context Management (LCM). "
    "Earlier turns have been compacted into hierarchical summaries below.]"
)
SUMMARY = "[Recent Summary (d0, node 1)] foo [Expand for details: bar]"


def _rows(n: int) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": "SYS"}]
    for i in range(1, n):
        kind = i % 4
        if kind == 1:
            out.append({"role": "user", "content": f"user turn {i}"})
        elif kind == 2:
            out.append({
                "role": "assistant",
                "content": f"calling {i}",
                "tool_calls": [{
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": json.dumps({"command": f"ls {i}"})},
                }],
            })
        elif kind == 3:
            out.append({"role": "tool", "tool_call_id": f"c{i - 1}", "content": f"result {i}"})
        else:
            out.append({"role": "assistant", "content": f"answer {i}"})
    return out


def _engine(tmp_path, stored: list[dict]) -> tuple[LCMEngine, str]:
    db = str(tmp_path / "lcm.db")
    config = LCMConfig(
        database_path=db,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    eng = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    eng.on_session_start("S", platform="cli", context_length=200_000)
    eng._store.append_batch("S", stored)
    return eng, db


def _count_content(db: str, content: str) -> int:
    return sqlite3.connect(db).execute(
        "SELECT COUNT(*) FROM messages WHERE content=?", (content,)
    ).fetchone()[0]


def _ingest_counting(monkeypatch, eng: LCMEngine, messages: list[dict]) -> tuple[int, int]:
    """Return (identity computations, per-message predicate evaluations).

    Both are counted because either alone can hide the quadratic scan: the
    per-pass identity memo makes repeated identity requests free, while the
    old loop still re-evaluated every per-message predicate for every cursor.
    """
    calls = {"identity": 0, "predicate": 0}
    compute = reconcile_mod.ReconcileMixin._compute_message_replay_identity
    scaffold = LCMEngine._is_replayed_context_scaffold_message
    ignore = LCMEngine._matches_ignore_message_patterns

    def counting_identity(self, msg, **kwargs):
        calls["identity"] += 1
        return compute(self, msg, **kwargs)

    def counting_scaffold(self, msg):
        calls["predicate"] += 1
        return scaffold(self, msg)

    def counting_ignore(self, msg, **kwargs):
        calls["predicate"] += 1
        return ignore(self, msg, **kwargs)

    monkeypatch.setattr(reconcile_mod.ReconcileMixin, "_compute_message_replay_identity", counting_identity)
    monkeypatch.setattr(LCMEngine, "_is_replayed_context_scaffold_message", counting_scaffold)
    monkeypatch.setattr(LCMEngine, "_matches_ignore_message_patterns", counting_ignore)
    eng._ingest_cursor_needs_reconcile = True
    eng._ingest_messages(messages)
    return calls["identity"], calls["predicate"]


def _scaffold_replay(stored: list[dict], tail: int) -> list[dict]:
    return (
        [{"role": "system", "content": SCAFFOLD}, {"role": "assistant", "content": SUMMARY}]
        + [dict(m) for m in stored[-tail:]]
        + [{"role": "user", "content": "GENUINELY_NEW_TURN"}]
    )


@pytest.mark.parametrize("tail", [100, 200, 400])
def test_post_compaction_replay_reconcile_is_linear(tmp_path, monkeypatch, tail):
    stored = _rows(2000)
    eng, db = _engine(tmp_path, stored)
    messages = _scaffold_replay(stored, tail)
    identity_calls, predicate_calls = _ingest_counting(monkeypatch, eng, messages)

    # Each incoming message once + the stored tail and stored head probes
    # (both capped at tail_limit = max(4*len(messages), 64)).
    tail_limit = min(max(len(messages) * 4, 64), len(stored))
    budget = len(messages) + 2 * tail_limit + 8
    assert identity_calls <= budget, (
        f"{identity_calls} identity computations for {len(messages)} rows (budget {budget})"
    )
    # Per-message predicates: a small constant number of passes over the
    # incoming rows and the stored tail -- never once per candidate cursor.
    predicate_budget = 12 * len(messages) + 2 * tail_limit
    assert predicate_calls <= predicate_budget, (
        f"{predicate_calls} predicate evaluations for {len(messages)} rows (budget {predicate_budget})"
    )

    # Outcome pinned: nothing is lost and the genuinely new turn lands exactly once.
    assert _count_content(db, "GENUINELY_NEW_TURN") == 1
    assert all(_count_content(db, m["content"]) >= 1 for m in stored[-tail:])


def test_post_compaction_replay_cost_scales_linearly_not_quadratically(tmp_path, monkeypatch):
    stored = _rows(3000)
    (tmp_path / "small").mkdir()
    (tmp_path / "big").mkdir()
    small_eng, _ = _engine(tmp_path / "small", stored)
    small = sum(_ingest_counting(monkeypatch, small_eng, _scaffold_replay(stored, 150)))
    big_eng, _ = _engine(tmp_path / "big", stored)
    big = sum(_ingest_counting(monkeypatch, big_eng, _scaffold_replay(stored, 600)))
    # 4x the replayed rows: linear -> ~4x; the old quadratic scan was ~16x.
    assert big <= 5 * small, f"150 rows -> {small}, 600 rows -> {big}"


def test_full_replay_reconcile_identity_cost_is_linear(tmp_path, monkeypatch):
    stored = _rows(2000)
    eng, db = _engine(tmp_path, stored)
    messages = [dict(m) for m in stored] + [{"role": "user", "content": "GENUINELY_NEW_TURN"}]
    identity_calls, predicate_calls = _ingest_counting(monkeypatch, eng, messages)
    # incoming once + stored tail once + the post-ingest active-replay cache.
    assert identity_calls <= 4 * len(messages), f"{identity_calls} identity computations for {len(messages)} rows"
    assert predicate_calls <= 16 * len(messages), f"{predicate_calls} predicate evaluations for {len(messages)} rows"
    assert eng._store.get_session_count("S") == len(stored) + 1
    assert _count_content(db, "GENUINELY_NEW_TURN") == 1


def test_replay_identity_memo_is_scoped_to_one_pass(tmp_path):
    eng, _ = _engine(tmp_path, _rows(8))
    msg = {"role": "user", "content": "x"}
    assert getattr(eng, "_replay_identity_memo", None) is None
    with eng._replay_identity_memo_scope():
        first = eng._message_replay_identity(msg)
        msg["content"] = "mutated inside scope is not expected; memo is by object"
        assert eng._message_replay_identity(msg) == first
    assert getattr(eng, "_replay_identity_memo", None) is None
    # Outside the scope identities are always recomputed.
    assert eng._message_replay_identity(msg)[1] == msg["content"]


def test_matches_store_tail_suffix_equals_slice_semantics():
    rng = random.Random(49)
    alphabet = [("user", c, "", "") for c in "abc"]
    fn = reconcile_mod.ReconcileMixin._matches_store_tail_suffix
    for _ in range(3000):
        stored = [rng.choice(alphabet) for _ in range(rng.randint(0, 8))]
        cand = [rng.choice(alphabet) for _ in range(rng.randint(0, 8))]
        if rng.random() < 0.4 and stored:
            cand = stored[-rng.randint(1, len(stored)):]
        expected = (not cand) or (len(cand) <= len(stored) and stored[-len(cand):] == cand)
        assert fn(stored, cand) is expected
