"""RED spike tests for opt-in async/background compaction.

These tests intentionally describe the desired public/private contract before
implementation exists. They stay xfailed on the design branch so the normal
suite remains green, but strict xfail means each scenario becomes a real gate as
soon as the feature starts landing.
"""

from __future__ import annotations

import json

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

pytestmark = pytest.mark.xfail(
    strict=True,
    reason="async/background compaction with atomic publish is design-only; implementation not landed",
)


def _engine(tmp_path, *, session_id="async-session", conversation_id="async-conversation"):
    config = LCMConfig(
        database_path=str(tmp_path / f"{session_id}.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
    )
    # Future config fields. They are dynamic here so these RED tests can be
    # written before the dataclass grows the real fields.
    config.async_background_compaction_enabled = True
    config.async_background_compaction_worker_enabled = False
    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        conversation_id=conversation_id,
        platform="test",
        context_length=1_000,
    )
    return engine


def _messages(count=10, *, prefix="message"):
    messages = [{"role": "system", "content": "system prompt"}]
    for idx in range(count):
        role = "user" if idx % 2 == 0 else "assistant"
        messages.append(
            {
                "role": role,
                "content": f"{prefix} {idx} " + ("x " * 12),
            }
        )
    return messages


def test_default_disabled_async_compaction_is_inert(tmp_path):
    """Given default config, background prep is disabled and reports zero async debt."""
    config = LCMConfig(
        database_path=str(tmp_path / "disabled.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "disabled-session",
        conversation_id="disabled-conversation",
        platform="test",
        context_length=1_000,
    )
    try:
        messages = _messages()
        engine.ingest(messages)

        result = engine.prepare_background_compaction_once(messages)

        assert result is None or result.state == "disabled"
        status = json.loads(engine.handle_tool_call("lcm_status", {}))
        assert status["async_compaction"]["enabled"] is False
        assert status["async_compaction"]["pending_batches"] == 0
        assert status["async_compaction"]["prepared_batches"] == 0
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    finally:
        engine.shutdown()


def test_pending_summaries_are_invisible_until_atomic_promotion(tmp_path):
    """Given prepared pending leaves, active context/readers ignore them until promotion."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)

        batch = engine.prepare_background_compaction_once(messages)

        assert batch.state == "ready"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
        status = json.loads(engine.handle_tool_call("lcm_status", {}))
        assert status["async_compaction"]["prepared_batches"] == 1
        assert status["dag"]["total_nodes"] == 0
        grep = json.loads(engine.handle_tool_call("lcm_grep", {"query": "message"}))
        assert all(result.get("kind") != "pending_summary" for result in grep.get("results", []))
    finally:
        engine.shutdown()


def test_atomic_promotion_rejects_stale_source_identity(tmp_path):
    """Given source rows changed after prep, promotion rejects without canonical mutation."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        first_source_id = batch.source_ids[0]
        engine._store._conn.execute(
            "UPDATE messages SET content = content || ' reconciled late' WHERE store_id = ?",
            (first_source_id,),
        )
        engine._store._conn.commit()

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "source_identity_mismatch"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
        assert engine.get_async_compaction_status()["rejected_batches"] == 1
    finally:
        engine.shutdown()


def test_atomic_promotion_rejects_live_config_change(tmp_path):
    """Given live config changes after prep, live policy wins over stale persisted metadata."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        engine._config.fresh_tail_count = 6

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "policy_fingerprint_mismatch"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    finally:
        engine.shutdown()


def test_atomic_promotion_rejects_summary_route_change(tmp_path):
    """Given summary model changes after prep, promotion rejects stale route output."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        engine._config.summary_model = "different-summary-model"

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "summary_route_fingerprint_mismatch"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    finally:
        engine.shutdown()


def test_atomic_promotion_rejects_live_threshold_policy_change(tmp_path):
    """Given threshold changes after prep, live config beats persisted batch policy."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        engine._config.context_threshold = 0.75

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "policy_fingerprint_mismatch"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    finally:
        engine.shutdown()


def test_foreground_compaction_race_supersedes_pending_batch(tmp_path, monkeypatch):
    """Given foreground compaction lands first, stale pending work is rejected/superseded."""
    engine = _engine(tmp_path)
    try:
        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **kwargs: ("foreground summary", 0),
        )
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        compacted = engine.compress(messages, current_tokens=engine.threshold_tokens + 1)
        result = engine.promote_prepared_compaction(batch.batch_id, compacted)

        assert engine._dag.get_session_node_count(engine.current_session_id) >= 1
        assert result.promoted is False
        assert result.reason in {"frontier_mismatch", "canonical_source_overlap"}
        async_status = engine.get_async_compaction_status()
        assert async_status["superseded_batches"] + async_status["rejected_batches"] >= 1
    finally:
        engine.shutdown()


def test_summary_failure_backoff_does_not_wedge_foreground_compaction(tmp_path, monkeypatch):
    """Given background summary failure, backoff is visible but foreground can still compact."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)

        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("summary spend backoff open")),
        )
        batch = engine.prepare_background_compaction_once(messages)
        assert batch.state == "failed"
        assert engine.get_async_compaction_status()["failed_batches"] == 1

        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **kwargs: ("foreground recovery summary", 0),
        )
        compacted = engine.compress(messages, current_tokens=engine.threshold_tokens + 1)

        assert compacted != messages
        assert engine._last_compression_status == "compacted"
    finally:
        engine.shutdown()


def test_restart_recovers_or_discards_pending_batches_safely(tmp_path):
    """Given pending/preparing rows at shutdown, restart never treats them as canonical."""
    db_path = tmp_path / "restart.db"
    config = LCMConfig(database_path=str(db_path), fresh_tail_count=2, leaf_chunk_tokens=20)
    config.async_background_compaction_enabled = True

    engine = LCMEngine(config=config)
    engine.on_session_start("restart-session", conversation_id="restart-conversation", context_length=1_000)
    messages = _messages()
    try:
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages, leave_state="preparing")
        assert batch.state == "preparing"
    finally:
        engine.shutdown()

    restarted = LCMEngine(config=config)
    try:
        restarted.on_session_start("restart-session", conversation_id="restart-conversation", context_length=1_000)
        status = restarted.get_async_compaction_status()

        assert restarted._dag.get_session_node_count(restarted.current_session_id) == 0
        assert status["preparing_batches"] == 0
        assert status["pending_batches"] + status["rejected_batches"] + status["failed_batches"] >= 1
    finally:
        restarted.shutdown()


def test_successful_atomic_promotion_is_all_or_nothing(tmp_path):
    """Given a valid ready batch, node insert/frontier advance/batch state commit together."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        old_frontier = engine.get_status()["lifecycle"]["current_frontier_store_id"]

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is True
        assert engine._dag.get_session_node_count(engine.current_session_id) == batch.expected_leaf_count
        lifecycle = engine.get_status()["lifecycle"]
        assert lifecycle["current_frontier_store_id"] > old_frontier
        assert lifecycle["current_frontier_store_id"] == batch.frontier_end_store_id
        assert engine.get_async_compaction_status()["promoted_batches"] == 1
    finally:
        engine.shutdown()


def test_atomic_promotion_rolls_back_partial_publish_failure(tmp_path):
    """Given a mid-promotion failure, no canonical node/frontier/batch half-state remains."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        old_frontier = engine.get_status()["lifecycle"]["current_frontier_store_id"]
        engine._async_compaction_publish_failure_hook = "after_canonical_insert"

        with pytest.raises(RuntimeError, match="injected async promotion failure"):
            engine.promote_prepared_compaction(batch.batch_id, messages)

        lifecycle = engine.get_status()["lifecycle"]
        assert lifecycle["current_frontier_store_id"] == old_frontier
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
        async_status = engine.get_async_compaction_status()
        assert async_status["promoted_batches"] == 0
        assert async_status["prepared_batches"] == 1
    finally:
        engine.shutdown()


def test_status_and_doctor_report_async_compaction_counts(tmp_path):
    """Given mixed async states, status and doctor expose pending/prepared/promoted/rejected counts."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        ready = engine.prepare_background_compaction_once(messages)
        engine.reject_prepared_compaction(ready.batch_id, reason="policy_fingerprint_mismatch")
        engine.prepare_background_compaction_once(messages)

        status = json.loads(engine.handle_tool_call("lcm_status", {}))
        doctor = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert status["async_compaction"]["prepared_batches"] == 1
        assert status["async_compaction"]["rejected_batches"] == 1
        async_checks = [check for check in doctor["checks"] if check["check"].startswith("async_compaction")]
        assert async_checks
        assert any("prepared_batches" in check["detail"] for check in async_checks)
    finally:
        engine.shutdown()
