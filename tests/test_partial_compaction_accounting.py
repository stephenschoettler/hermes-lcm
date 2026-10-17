"""Durable leaf progress must remain visible when later stages fail."""
import pytest
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path, monkeypatch):
    instance = LCMEngine(LCMConfig(database_path=str(tmp_path / "lcm.db")), hermes_home=str(tmp_path / "home"))
    instance.on_session_start("session", conversation_id="lane")
    instance._config.fresh_tail_count = 2
    instance._config.leaf_chunk_tokens = 1
    instance._config.dynamic_leaf_chunk_enabled = False
    instance._config.threshold_full_sweep_enabled = False
    monkeypatch.setattr(instance, "_summarize_leaf_chunk_with_rescue",
                        lambda chunk, **kwargs: (chunk, 1000, "summary of old facts", 1, 0))
    try:
        yield instance
    finally:
        instance.shutdown()


def messages():
    return [{"role": "user", "content": "old fact " * 100},
            {"role": "assistant", "content": "old reply " * 100},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"}]


@pytest.mark.parametrize("stage, method", [("condensation", "_maybe_condense"), ("assembly", "_assemble_context")])
def test_post_leaf_failure_records_partial_and_still_surfaces_error(engine, monkeypatch, stage, method):
    failure = RuntimeError("synthetic post-leaf failure")
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(engine, method, fail)
    before = engine.compression_count
    with pytest.raises(RuntimeError, match="synthetic post-leaf") as captured:
        engine.compress(messages(), current_tokens=100000)
    assert captured.value is failure
    assert len(engine._dag.get_session_nodes("session")) == 1
    assert engine._lifecycle.get_by_conversation("lane").current_frontier_store_id > 0
    assert engine.compression_count == before + 1
    assert engine.last_compression_status == "partial"
    assert engine.last_compression_noop_reason == stage + "_failed_after_leaf_commit"
    assert not engine.last_compression_was_noop


def test_failure_before_leaf_does_not_reuse_previous_partial_status(engine, monkeypatch):
    engine._last_compression_status = "partial"
    engine._last_compression_noop_reason = "old_partial"
    before = engine.compression_count
    def fail(*args, **kwargs):
        raise RuntimeError("before leaf")
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", fail)
    with pytest.raises(RuntimeError, match="before leaf"):
        engine.compress(messages(), current_tokens=100000)
    assert engine.compression_count == before
    assert engine.last_compression_status == "error"
    assert engine._dag.get_session_nodes("session") == []
