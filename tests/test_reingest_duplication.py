"""Regression tests for the LCM re-ingest duplication bug.

A fresh context-engine instance bound to an existing session receives the
whole Hermes transcript on its first ``ingest()``.  When the incoming replay
does not align exactly with the stored tail (whitespace differences, an earlier
partial replay already in the store, reordered parallel tool results, gateway
notification turns at different positions), the old code fell through to
"persisted ambiguous delta" with ``cursor=0`` and wrote the entire transcript
again.

The fixture is synthetic and scrubbed; it keeps only those structural features.
"""

import json
from pathlib import Path

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "reingest_duplication.json"
SESSION_ID = "reingest-duplication-session"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _copy(messages: list[dict]) -> list[dict]:
    return json.loads(json.dumps(messages))


def _config(tmp_path: Path) -> LCMConfig:
    return LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )


def _start(engine: LCMEngine) -> LCMEngine:
    engine.on_session_start(
        SESSION_ID,
        platform="cli",
        conversation_id="reingest-duplication-conversation",
        context_length=200_000,
    )
    return engine


def _close(engine: LCMEngine) -> None:
    engine._store.close()
    engine._dag.close()
    engine._lifecycle.close()


def _seed(tmp_path: Path, messages: list[dict]) -> int:
    """Seed a durable session through the real ingest path, then restart."""
    engine = _start(LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home")))
    try:
        engine._ingest_messages(_copy(messages))
        return engine._store.get_session_count(SESSION_ID)
    finally:
        _close(engine)


def _replay_engine(tmp_path: Path) -> LCMEngine:
    """A fresh instance on an existing session (gateway restart / cache eviction)."""
    return _start(LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home")))


def _rows(engine: LCMEngine) -> list[dict]:
    return engine._store.get_session_messages(SESSION_ID, limit=10_000)


def _contents(engine: LCMEngine, role: str | None = None) -> list[str]:
    return [
        row["content"]
        for row in _rows(engine)
        if role is None or row["role"] == role
    ]


def _stored_with_earlier_replay() -> list[dict]:
    data = _fixture()
    return _copy(data["stored"]) + _copy(data["stored_duplicate"])


def _reconciliation(engine: LCMEngine) -> dict:
    return engine.get_status()["ingest_reconciliation"]


# -- a. full reproduction ----------------------------------------------------


def test_full_replay_after_restart_persists_no_duplicate_rows(tmp_path):
    data = _fixture()
    stored = _copy(data["stored"]) + _copy(data["stored_duplicate"])
    seeded = _seed(tmp_path, stored)

    engine = _replay_engine(tmp_path)
    try:
        engine._ingest_messages(_copy(data["replay"]))

        assert _reconciliation(engine)["reason"] == "persisted multiset delta"
        reconciliation = _reconciliation(engine)
        assert reconciliation["incoming"] == len(data["replay"])
        assert reconciliation["already_stored"] == len(data["replay"])
        assert reconciliation["persisted"] == 0
        assert engine._store.get_session_count(SESSION_ID) == seeded
        assert engine._ingest_cursor == len(data["replay"])
    finally:
        _close(engine)


# -- b. whitespace-only differences ------------------------------------------


def test_whitespace_only_replay_difference_persists_nothing(tmp_path):
    data = _fixture()
    stored = _copy(data["stored"]) + _copy(data["stored_duplicate"])
    seeded = _seed(tmp_path, stored)

    replay = _copy(data["stored"])
    replay[11]["content"] = "ok"  # stored row is "ok " (live trailing space)

    engine = _replay_engine(tmp_path)
    try:
        engine._ingest_messages(replay)

        assert _reconciliation(engine)["reason"] == "persisted multiset delta"
        assert _reconciliation(engine)["persisted"] == 0
        assert engine._store.get_session_count(SESSION_ID) == seeded
    finally:
        _close(engine)


# -- c. reordered parallel tool results -------------------------------------


def test_reordered_parallel_tool_results_persist_nothing(tmp_path):
    data = _fixture()
    stored = _copy(data["stored"]) + _copy(data["stored_duplicate"])
    seeded = _seed(tmp_path, stored)

    replay = _copy(data["stored"])
    beta = next(i for i, row in enumerate(replay) if row.get("tool_call_id") == "call_beta")
    gamma = next(i for i, row in enumerate(replay) if row.get("tool_call_id") == "call_gamma")
    replay[beta], replay[gamma] = replay[gamma], replay[beta]

    engine = _replay_engine(tmp_path)
    try:
        engine._ingest_messages(replay)

        assert _reconciliation(engine)["reason"] == "persisted multiset delta"
        assert _reconciliation(engine)["persisted"] == 0
        assert engine._store.get_session_count(SESSION_ID) == seeded
    finally:
        _close(engine)


# -- d. stored history already contains an earlier partial replay ------------


def test_earlier_duplicate_replay_in_store_does_not_grow(tmp_path):
    data = _fixture()
    stored = _copy(data["stored"]) + _copy(data["stored_duplicate"])
    seeded = _seed(tmp_path, stored)

    for _ in range(2):
        engine = _replay_engine(tmp_path)
        try:
            engine._ingest_messages(_copy(data["replay"]))
            assert _reconciliation(engine)["reason"] == "persisted multiset delta"
            assert _reconciliation(engine)["persisted"] == 0
            assert engine._store.get_session_count(SESSION_ID) == seeded
        finally:
            _close(engine)


# -- e. genuinely new tail --------------------------------------------------


def test_replay_plus_three_new_messages_persists_exactly_three(tmp_path):
    data = _fixture()
    stored = _copy(data["stored"]) + _copy(data["stored_duplicate"])
    seeded = _seed(tmp_path, stored)

    new_messages = [
        {"role": "user", "content": "new tail question"},
        {"role": "assistant", "content": "new tail answer"},
        {"role": "user", "content": "new tail follow-up"},
    ]
    engine = _replay_engine(tmp_path)
    try:
        engine._ingest_messages(_copy(data["replay"]) + _copy(new_messages))

        reconciliation = _reconciliation(engine)
        assert reconciliation["reason"] == "persisted multiset delta"
        assert reconciliation["persisted"] == len(new_messages)
        assert reconciliation["already_stored"] == len(data["replay"])
        assert engine._store.get_session_count(SESSION_ID) == seeded + len(new_messages)
        assert _contents(engine)[-len(new_messages):] == [m["content"] for m in new_messages]
        assert _contents(engine).count("new tail question") == 1
        assert _contents(engine).count("new tail answer") == 1
        assert _contents(engine).count("new tail follow-up") == 1
        assert engine._ingest_cursor == len(data["replay"]) + len(new_messages)
    finally:
        _close(engine)


# -- f. legitimate repeats --------------------------------------------------


def test_repeated_short_user_message_is_still_stored_twice(tmp_path):
    data = _fixture()
    stored = _copy(data["stored"]) + _copy(data["stored_duplicate"])
    seeded = _seed(tmp_path, stored)
    assert [c.strip() for c in _contents(_replay_engine(tmp_path), role="user")].count("ok") == 1

    replay = _copy(data["replay"]) + [
        {"role": "assistant", "content": "second confirmation reply"},
        {"role": "user", "content": "ok"},
    ]

    engine = _replay_engine(tmp_path)
    try:
        engine._ingest_messages(replay)

        reconciliation = _reconciliation(engine)
        assert reconciliation["reason"] == "persisted multiset delta"
        assert reconciliation["persisted"] == 2
        assert engine._store.get_session_count(SESSION_ID) == seeded + 2
        assert [c.strip() for c in _contents(engine, role="user")].count("ok") == 2
        assert _contents(engine).count("second confirmation reply") == 1
    finally:
        _close(engine)


# -- g. idempotence ---------------------------------------------------------


def test_same_replay_ingested_twice_persists_nothing_the_second_time(tmp_path):
    data = _fixture()
    stored = _copy(data["stored"]) + _copy(data["stored_duplicate"])
    seeded = _seed(tmp_path, stored)

    engine = _replay_engine(tmp_path)
    try:
        engine._ingest_messages(_copy(data["replay"]))
        assert engine._store.get_session_count(SESSION_ID) == seeded

        engine._ingest_messages(_copy(data["replay"]))
        assert engine._store.get_session_count(SESSION_ID) == seeded
    finally:
        _close(engine)

    # A second brand-new instance must reach the same conclusion.
    engine = _replay_engine(tmp_path)
    try:
        engine._ingest_messages(_copy(data["replay"]))
        assert _reconciliation(engine)["persisted"] == 0
        assert engine._store.get_session_count(SESSION_ID) == seeded
    finally:
        _close(engine)


# -- h. compacted session: summary scaffold prefix + durable fresh tail ------


def _summary_scaffold() -> dict:
    return {
        "role": "user",
        "content": (
            "[Recent Summary (d0, node 7)]\n# Session summary: synthetic\n"
            "[Expand for details: lcm_expand(node_id=7)]"
        ),
    }


def test_scaffold_prefix_replay_of_durable_tail_persists_nothing(tmp_path):
    data = _fixture()
    seeded = _seed(tmp_path, _stored_with_earlier_replay())

    engine = _replay_engine(tmp_path)
    try:
        engine._ingest_messages([_summary_scaffold()] + _copy(data["replay"]))

        reconciliation = _reconciliation(engine)
        assert reconciliation["reason"] == "skipped scaffold-only prefix; persisted multiset delta"
        assert reconciliation["persisted"] == 0
        assert engine._store.get_session_count(SESSION_ID) == seeded
    finally:
        _close(engine)


def test_scaffold_prefix_replay_plus_new_messages_persists_only_new(tmp_path):
    data = _fixture()
    seeded = _seed(tmp_path, _stored_with_earlier_replay())
    new = [
        {"role": "user", "content": "scaffold new question"},
        {"role": "assistant", "content": "scaffold new answer"},
    ]

    engine = _replay_engine(tmp_path)
    try:
        engine._ingest_messages([_summary_scaffold()] + _copy(data["replay"]) + _copy(new))

        assert _reconciliation(engine)["persisted"] == 2
        assert engine._store.get_session_count(SESSION_ID) == seeded + 2
        assert _contents(engine)[-2:] == ["scaffold new question", "scaffold new answer"]
        assert not any("Recent Summary" in (c or "") for c in _contents(engine))
    finally:
        _close(engine)
