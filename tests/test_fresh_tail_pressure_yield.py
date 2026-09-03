"""Fresh-tail pressure yield: compaction must not deadlock inside the count tail.

Regression tests for #441 (same class as #414): a count-protected fresh tail
that covers the session's whole token mass made every compaction attempt no-op
("no eligible raw backlog outside fresh tail" below the count,
"raw backlog outside fresh tail is below leaf chunk threshold" just above it)
while the host reported over-threshold pressure every turn, until the session
died at the provider hard limit.

Semantics under test:

- The yield arms only after ``fresh_tail_pressure_yield_min_observations``
  consecutive blocked entry-point invocations under host-observed pressure
  ("sustained"); a single over-threshold observation does not soften
  ``fresh_tail_count``. Setting the knob to 1 restores first-observation
  yielding.
- The armed tail bound is invocation-scoped: cleared through success,
  exception, nested invocation, and session-reset paths.
- ``should_compress_preflight`` agrees with ``compress`` in the yield states,
  including persisted ignored-placeholder pressure and threshold full-sweep
  mode.
"""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def _fat_user(index, chars=4000):
    return {"role": "user", "content": f"turn {index}: " + ("data " * (chars // 5))}


def _tiny_user(index):
    return {"role": "user", "content": f"small turn {index}"}


def _stub_summarizer(chunk, focus_topic=None, **_kwargs):
    tokens = count_messages_tokens(chunk)
    return chunk, tokens, f"[test summary: {len(chunk)} messages]", 1, 1


def _make_engine(tmp_path, monkeypatch, **config_overrides):
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_pressure_yield.db")
    for key, value in config_overrides.items():
        setattr(config, key, value)
    engine = LCMEngine(config=config)
    engine._session_id = "pressure-yield-session"
    engine.context_length = 200_000
    engine.threshold_tokens = 5_000
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", _stub_summarizer)
    return engine


# ── First-observation mode (min_observations=1): the original deadlock shapes ──


def test_count_tail_covering_everything_yields_under_pressure(tmp_path, monkeypatch):
    # 40 messages, all inside a 128-count tail: on an unfixed engine this is
    # a guaranteed "no eligible raw backlog outside fresh tail" no-op forever.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=1,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)
        assert observed > engine.threshold_tokens

        compressed = engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "compacted"
        assert len(compressed) < len(messages)
        assert count_messages_tokens(compressed) < observed
    finally:
        engine.shutdown()


def test_backlog_below_leaf_chunk_yields_under_pressure(tmp_path, monkeypatch):
    # A couple of tiny messages sit outside the count tail; everything heavy is
    # protected. On an unfixed engine this no-ops with "below leaf chunk
    # threshold" forever.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=10,
        fresh_tail_pressure_yield_min_observations=1,
    )
    try:
        messages = [_tiny_user(0), _tiny_user(1)] + [_fat_user(i) for i in range(10)]
        observed = count_messages_tokens(messages)
        assert observed > engine.threshold_tokens

        compressed = engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "compacted"
        assert len(compressed) < len(messages)
    finally:
        engine.shutdown()


def test_no_pressure_preserves_count_tail_noop(tmp_path, monkeypatch):
    # Same deadlock topology, but the host reports no over-threshold pressure:
    # behavior must stay exactly the pre-fix no-op.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=1,
    )
    try:
        engine.threshold_tokens = 10_000_000
        messages = [_fat_user(i) for i in range(40)]

        compressed = engine.compress(list(messages))

        assert engine._last_compression_status == "noop"
        assert engine._last_compression_noop_reason == (
            "no eligible raw backlog outside fresh tail"
        )
        assert len(compressed) == len(messages)
    finally:
        engine.shutdown()


def test_kill_switch_preserves_noop_under_pressure(tmp_path, monkeypatch):
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_enabled=False,
        fresh_tail_pressure_yield_min_observations=1,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)

        compressed = engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "noop"
        assert engine._last_compression_noop_reason == (
            "no eligible raw backlog outside fresh tail"
        )
        assert len(compressed) == len(messages)
    finally:
        engine.shutdown()


def test_preflight_advertises_compaction_under_deadlock_pressure(tmp_path, monkeypatch):
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=1,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        assert count_messages_tokens(messages) > engine.threshold_tokens

        assert engine.should_compress_preflight(list(messages)) is True
    finally:
        engine.shutdown()


def test_explicit_token_cap_still_wins_when_smaller(tmp_path, monkeypatch):
    # An operator-configured fresh_tail_max_tokens below the derived yield cap
    # keeps bounding the tail; the yield must not loosen it.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_max_tokens=500,
        fresh_tail_pressure_yield_min_observations=1,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)

        compressed = engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "compacted"
        assert len(compressed) < len(messages)
    finally:
        engine.shutdown()


# ── Sustained-pressure semantics (default min_observations) ──────────────────


def test_single_blocked_observation_does_not_yield_at_default(tmp_path, monkeypatch):
    # Default semantics: one over-threshold blocked attempt must NOT soften
    # fresh_tail_count into a token bound. The classic noop is preserved until
    # pressure is sustained.
    engine = _make_engine(tmp_path, monkeypatch, fresh_tail_count=128)
    try:
        assert engine._config.fresh_tail_pressure_yield_min_observations >= 2
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)

        compressed = engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "noop"
        assert engine._last_compression_noop_reason == (
            "no eligible raw backlog outside fresh tail"
        )
        assert len(compressed) == len(messages)
        assert engine._pressure_yield_blocked_streak == 1
    finally:
        engine.shutdown()


def test_sustained_pressure_yields_on_nth_blocked_observation(tmp_path, monkeypatch):
    # Host loop at default settings: preflight is the per-turn entry point.
    # It must decline for min_observations-1 blocked turns, then advertise a
    # compaction that compress actually performs.
    engine = _make_engine(tmp_path, monkeypatch, fresh_tail_count=128)
    try:
        min_observations = engine._config.fresh_tail_pressure_yield_min_observations
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)
        assert observed > engine.threshold_tokens

        verdicts = []
        for _turn in range(min_observations):
            verdicts.append(engine.should_compress_preflight(list(messages)))

        assert verdicts == [False] * (min_observations - 1) + [True]

        compressed = engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "compacted"
        assert len(compressed) < len(messages)
        # Success resets the sustained evidence.
        assert engine._pressure_yield_blocked_streak == 0
    finally:
        engine.shutdown()


def test_pressure_relief_resets_blocked_streak(tmp_path, monkeypatch):
    # Two blocked turns, one calm turn, two blocked turns: the calm turn resets
    # the evidence, so blocked turn #4 overall is only observation 2/3 and must
    # not yield. Only three CONSECUTIVE blocked turns arm the yield.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=3,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        assert count_messages_tokens(messages) > engine.threshold_tokens

        assert engine.should_compress_preflight(list(messages)) is False
        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 2

        # A turn observed under threshold relieves the pressure.
        engine.threshold_tokens = 10_000_000
        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 0
        engine.threshold_tokens = 5_000

        assert engine.should_compress_preflight(list(messages)) is False
        assert engine.should_compress_preflight(list(messages)) is False
        assert engine.should_compress_preflight(list(messages)) is True
    finally:
        engine.shutdown()


# ── Invocation scoping and cleanup of the armed bound ────────────────────────


def test_yield_bound_cleared_after_successful_compress(tmp_path, monkeypatch):
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=1,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)

        engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "compacted"
        assert engine._pressure_yield_tail_token_limit == 0
        # Outside any invocation the tail resolves at its stock geometry:
        # a 128-count tail still covers this whole 40-message list.
        assert engine._fresh_tail_start(messages) == 0
    finally:
        engine.shutdown()


def test_yield_bound_cleared_when_summarizer_raises(tmp_path, monkeypatch):
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=1,
    )

    def _raising_summarizer(chunk, focus_topic=None, **_kwargs):
        raise RuntimeError("summarizer down")

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", _raising_summarizer)
    try:
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)

        with pytest.raises(RuntimeError):
            engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "error"
        assert engine._pressure_yield_tail_token_limit == 0
        assert engine._pressure_yield_scope_depth == 0
    finally:
        engine.shutdown()


def test_nested_invocation_gets_clean_scope_and_restores_outer(tmp_path, monkeypatch):
    engine = _make_engine(tmp_path, monkeypatch, fresh_tail_count=128)
    try:
        with engine._fresh_tail_pressure_yield_invocation():
            engine._pressure_yield_tail_token_limit = 1234
            with engine._fresh_tail_pressure_yield_invocation():
                # The inner (reentrant) invocation must not observe or clear
                # the outer invocation's bound.
                assert engine._pressure_yield_tail_token_limit == 0
                engine._pressure_yield_tail_token_limit = 77
            assert engine._pressure_yield_tail_token_limit == 1234
        assert engine._pressure_yield_tail_token_limit == 0
        assert engine._pressure_yield_scope_depth == 0
    finally:
        engine.shutdown()


def test_session_reset_clears_sustained_evidence_and_bound(tmp_path, monkeypatch):
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=3,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        assert engine.should_compress_preflight(list(messages)) is False
        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 2

        engine._reset_session_scoped_runtime_state()

        assert engine._pressure_yield_blocked_streak == 0
        assert engine._pressure_yield_tail_token_limit == 0
    finally:
        engine.shutdown()


# ── Strict-consecutive streak semantics ──────────────────────────────────────


def test_intervening_eligible_invocation_resets_blocked_streak(tmp_path, monkeypatch):
    # blocked -> eligible -> blocked with min_observations=2: the eligible
    # turn proves the session is not tail-deadlocked, so the two blocked
    # turns are separated events, not sustained pressure. Pre-fix the second
    # blocked turn engaged the yield.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=2,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        assert count_messages_tokens(messages) > engine.threshold_tokens

        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 1

        # Shrinking the count tail exposes plenty of raw backlog: an ordinary
        # eligible preflight, no yield involved.
        engine._config.fresh_tail_count = 4
        assert engine.should_compress_preflight(list(messages)) is True
        assert engine._pressure_yield_blocked_streak == 0
        engine._config.fresh_tail_count = 128

        # Observation 1/2 again: the classic noop must hold.
        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 1
        assert engine._pressure_yield_tail_token_limit == 0
    finally:
        engine.shutdown()


def test_non_tail_blockage_resets_blocked_streak(tmp_path, monkeypatch):
    # An invocation blocked by something other than fresh-tail eligibility
    # (here: the compression boundary cooldown) is not evidence of the tail
    # deadlock and must reset the streak.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=2,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]

        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 1

        with monkeypatch.context() as patch:
            patch.setattr(
                engine, "_compression_boundary_cooldown_active", lambda: True
            )
            assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 0

        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 1
        assert engine._pressure_yield_tail_token_limit == 0
    finally:
        engine.shutdown()


def test_sanitation_only_progress_resets_blocked_streak(tmp_path, monkeypatch):
    # A compress pass that ends "sanitized" (context cleanup, no leaf node)
    # is real progress: stale blocked evidence must not survive it.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=5,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)

        assert engine.should_compress_preflight(list(messages)) is False
        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 2

        real_sanitize = engine._sanitize_active_context_messages

        def _cleaning_sanitize(msgs, **kwargs):
            return real_sanitize(msgs, **kwargs) + [
                {"role": "user", "content": "sanitation marker"}
            ]

        with monkeypatch.context() as patch:
            patch.setattr(
                engine, "_sanitize_active_context_messages", _cleaning_sanitize
            )
            engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "sanitized"
        assert engine._pressure_yield_blocked_streak == 0
    finally:
        engine.shutdown()


def test_forced_overflow_preflight_resets_blocked_streak(tmp_path, monkeypatch):
    # Forced overflow recovery is a context-reducing path in its own right:
    # a preflight that advertises it is not tail-blocked, so the streak
    # restarts.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=2,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]

        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 1

        # An assembly cap far below the observed tokens makes this turn a
        # forced-overflow recovery request, which preflight advertises before
        # it ever consults fresh-tail eligibility.
        saved_cap = engine._config.max_assembly_tokens
        engine._config.max_assembly_tokens = 1_000
        assert engine.should_compress_preflight(list(messages)) is True
        engine._config.max_assembly_tokens = saved_cap
        assert engine._pressure_yield_blocked_streak == 0

        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 1
        assert engine._pressure_yield_tail_token_limit == 0
    finally:
        engine.shutdown()


def test_exception_neither_extends_nor_resets_blocked_streak(tmp_path, monkeypatch):
    # A summarizer failure mid-invocation is not an observation in either
    # direction: the accumulated evidence stays exactly as it was.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=3,
    )

    def _raising_summarizer(chunk, focus_topic=None, **_kwargs):
        raise RuntimeError("summarizer down")

    try:
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)

        assert engine.should_compress_preflight(list(messages)) is False
        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 2

        with monkeypatch.context() as patch:
            patch.setattr(
                engine, "_summarize_leaf_chunk_with_rescue", _raising_summarizer
            )
            with pytest.raises(RuntimeError):
                engine.compress(list(messages), current_tokens=observed)

        # The failed compress counted its own blocked observation before the
        # yield engaged and the summarizer died; the exception exit must not
        # wipe that evidence (the deadlock is still there).
        assert engine._pressure_yield_blocked_streak == 3
        assert engine._pressure_yield_tail_token_limit == 0
    finally:
        engine.shutdown()


# ── Reset authority across nested scopes ─────────────────────────────────────


def test_session_reset_inside_nested_invocation_stays_authoritative(
    tmp_path, monkeypatch
):
    # A reset that runs inside a nested invocation scope must win over the
    # scope-exit restore: pre-fix, leaving the inner scope restored the
    # outer invocation's pre-reset bound (1234 -> reset to 0 -> exit
    # restores 1234).
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=3,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        assert engine.should_compress_preflight(list(messages)) is False
        assert engine.should_compress_preflight(list(messages)) is False
        assert engine._pressure_yield_blocked_streak == 2

        with engine._fresh_tail_pressure_yield_invocation():
            engine._pressure_yield_tail_token_limit = 1234
            with engine._fresh_tail_pressure_yield_invocation():
                engine._reset_session_scoped_runtime_state()
                assert engine._pressure_yield_tail_token_limit == 0
            assert engine._pressure_yield_tail_token_limit == 0
        assert engine._pressure_yield_tail_token_limit == 0
        assert engine._pressure_yield_blocked_streak == 0
        assert engine._pressure_yield_scope_depth == 0
    finally:
        engine.shutdown()


# ── Preflight/compress agreement ─────────────────────────────────────────────


def test_preflight_agrees_with_compress_under_persisted_placeholder_pressure(
    tmp_path, monkeypatch
):
    # A session carrying a persisted ignored-message placeholder used to park in
    # the pre-ingest ambiguous-noop state: preflight declined forever while
    # compress would have engaged the yield. Both sides must now agree.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=1,
    )
    try:
        placeholder_text = LCMEngine._ignored_active_replay_placeholder("ignored content")
        digest = LCMEngine._active_replay_placeholder_digest(placeholder_text)
        assert digest is not None
        engine._generated_ignored_active_replay_placeholder_hashes = {digest}
        messages = [{"role": "assistant", "content": placeholder_text}] + [
            _fat_user(i) for i in range(40)
        ]
        observed = count_messages_tokens(messages)
        assert observed > engine.threshold_tokens

        assert engine.should_compress_preflight(list(messages)) is True

        compressed = engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status in {"compacted", "sanitized"}
        assert len(compressed) < len(messages)
    finally:
        engine.shutdown()


def test_preflight_and_compress_agree_while_disabled_under_placeholder_pressure(
    tmp_path, monkeypatch
):
    # Kill switch on the same topology: both sides must land on the same noop.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_enabled=False,
    )
    try:
        placeholder_text = LCMEngine._ignored_active_replay_placeholder("ignored content")
        digest = LCMEngine._active_replay_placeholder_digest(placeholder_text)
        engine._generated_ignored_active_replay_placeholder_hashes = {digest}
        messages = [{"role": "assistant", "content": placeholder_text}] + [
            _fat_user(i) for i in range(40)
        ]
        observed = count_messages_tokens(messages)

        assert engine.should_compress_preflight(list(messages)) is False
        preflight_status = engine._last_compression_status

        engine.compress(list(messages), current_tokens=observed)

        assert preflight_status == "noop"
        assert engine._last_compression_status in {"noop", "sanitized"}
    finally:
        engine.shutdown()


def test_full_sweep_deadlock_yields_and_compacts(tmp_path, monkeypatch):
    # threshold_full_sweep_enabled must not resurrect the deadlock: a sweep
    # whose "drained" raw prefix is really a tail covering the whole session
    # yields like any other blocked pass. Before this revision, preflight
    # advertised the sweep but compress no-opped forever.
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        threshold_full_sweep_enabled=True,
        fresh_tail_pressure_yield_min_observations=1,
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)
        assert observed > engine.threshold_tokens

        assert engine.should_compress_preflight(list(messages)) is True

        compressed = engine.compress(list(messages), current_tokens=observed)

        assert engine._last_compression_status == "compacted"
        assert len(compressed) < len(messages)
        assert engine._last_threshold_full_sweep["status"] in {"completed", "partial"}
        assert engine._last_threshold_full_sweep["leaf_passes"] >= 1
    finally:
        engine.shutdown()


# ── Rollup/compaction interplay (deterministic, no scheduler timing) ─────────


def test_yielded_compaction_stales_and_rebuilds_rollups_deterministically(
    tmp_path, monkeypatch
):
    # A yield-driven compaction publishes summary nodes like any other pass, so
    # it must feed temporal-rollup staleness the same way: the covered day goes
    # stale on publication and a synchronous maintenance pass rebuilds it. All
    # driven inline — no background scheduler, no wall-clock dependence — so the
    # validation composes with rollup-maintenance scheduling changes (#440).
    from datetime import datetime, timezone

    import hermes_lcm.rollup_builder as builder_module
    from hermes_lcm.rollup_builder import run_rollup_maintenance
    from hermes_lcm.rollup_store import RollupStore

    engine = _make_engine(
        tmp_path,
        monkeypatch,
        fresh_tail_count=128,
        fresh_tail_pressure_yield_min_observations=1,
        temporal_rollups_enabled=True,
    )
    monkeypatch.setattr(
        builder_module,
        "summarize_with_escalation",
        lambda _text, **_kwargs: ("rebuilt rollup", 1),
    )
    try:
        messages = [_fat_user(i) for i in range(40)]
        observed = count_messages_tokens(messages)

        engine.compress(list(messages), current_tokens=observed)
        assert engine._last_compression_status == "compacted"

        nodes = engine._dag.get_session_nodes(engine._session_id)
        assert nodes
        latest = max(node.latest_at or node.created_at for node in nodes)
        day = datetime.fromtimestamp(latest, tz=timezone.utc).date().isoformat()

        store = RollupStore(engine._dag.db_path)
        try:
            row = store.get_rollup("day", day, engine._session_id)
            assert row is not None
            assert row["status"] == "stale"

            built = run_rollup_maintenance(
                engine._dag, engine._config, engine._session_id
            )
            assert built >= 1

            row = store.get_rollup("day", day, engine._session_id)
            assert row["status"] == "ready"
        finally:
            store.close()
    finally:
        engine.shutdown()
