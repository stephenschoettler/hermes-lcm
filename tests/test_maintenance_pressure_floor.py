"""Maintenance compaction should require some token pressure.

``should_compress_preflight`` reaches the leaf-candidate check two ways:

* the NON-divergent path gates it behind ``rough >= threshold_tokens``
* the divergent-replay path runs the identical check with no token gate

The divergent path is entered whenever ingest rewrote the replay view, which in
practice means an externalized payload (inline media, base64, a large tool
result). A session that traffics in those therefore requests an opportunistic
maintenance compaction at any size, however small.

``maintenance_min_pressure_ratio`` puts a floor under the two OPPORTUNISTIC arms
only, expressed as a fraction of ``threshold_tokens``. It defaults to 0.0, which
preserves the historical behavior exactly.
"""

from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path: Path, **overrides) -> LCMEngine:
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
        **overrides,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(
        "maintenance-floor-session",
        platform="telegram",
        conversation_id="maintenance-floor-conversation",
        context_length=1_000_000,
    )
    return engine


def test_default_is_disabled_and_preserves_current_behavior(tmp_path):
    """An unconfigured engine must behave exactly as it does today."""
    engine = _engine(tmp_path)
    assert engine._config.maintenance_min_pressure_ratio == 0.0
    engine.threshold_tokens = 750_000
    # every size is permitted while the floor is off
    assert engine._maintenance_pressure_met(1) is True
    assert engine._maintenance_pressure_met(158_262) is True
    assert engine._maintenance_pressure_met(750_000) is True


@pytest.mark.parametrize("observed", [129_088, 158_262, 175_730, 239_896])
def test_small_sessions_defer_when_a_floor_is_set(tmp_path, observed):
    """Sizes well under the threshold must not request maintenance work.

    These four values are real ``requested compress()`` observations from a
    media-heavy session: 17%, 21%, 23% and 32% of a 750,000 threshold.
    """
    engine = _engine(tmp_path, maintenance_min_pressure_ratio=0.5)
    engine.threshold_tokens = 750_000
    assert engine._maintenance_pressure_met(observed) is False


def test_real_pressure_still_compacts(tmp_path):
    """The floor must never suppress a session that genuinely needs work."""
    engine = _engine(tmp_path, maintenance_min_pressure_ratio=0.5)
    engine.threshold_tokens = 750_000
    assert engine._maintenance_pressure_met(422_558) is True
    assert engine._maintenance_pressure_met(750_000) is True
    assert engine._maintenance_pressure_met(1_000_000) is True


def test_floor_boundary_is_inclusive(tmp_path):
    engine = _engine(tmp_path, maintenance_min_pressure_ratio=0.5)
    engine.threshold_tokens = 750_000
    assert engine._maintenance_pressure_met(374_999) is False
    assert engine._maintenance_pressure_met(375_000) is True


def test_unknown_threshold_never_blocks(tmp_path):
    """``threshold_tokens`` of 0 means "not yet known"; do not gate on it."""
    engine = _engine(tmp_path, maintenance_min_pressure_ratio=0.5)
    engine.threshold_tokens = 0
    assert engine._maintenance_pressure_met(1) is True


def test_ratio_is_read_from_the_environment(tmp_path, monkeypatch):
    """The knob follows the same ``LCM_*`` convention as its siblings."""
    monkeypatch.setenv("LCM_MAINTENANCE_MIN_PRESSURE_RATIO", "0.4")
    config = LCMConfig.from_env()
    assert config.maintenance_min_pressure_ratio == pytest.approx(0.4)


def test_both_opportunistic_arms_are_gated():
    """Gate the leaf-candidate arm AND the ignored-backlog arm.

    Gating only one leaves the other free to request compaction at any size, so
    the fix would be half-inert and the symptom would persist.
    """
    import inspect

    from hermes_lcm.compaction import CompactionMixin

    source = inspect.getsource(CompactionMixin.should_compress_preflight)
    assert source.count("_maintenance_pressure_met") >= 2


def test_cleanup_adoption_is_not_gated():
    """Deterministic ingest cleanup must stay available at any size.

    It is already durable and costs no summarizer work; the floor exists to
    defer expensive opportunistic summarization, not to strand a cleanup the
    store has already committed.
    """
    import inspect

    from hermes_lcm.compaction import CompactionMixin

    source = inspect.getsource(CompactionMixin.should_compress_preflight)
    assert source.index("if cleanup_requested:") < source.index(
        "_maintenance_pressure_met"
    )
