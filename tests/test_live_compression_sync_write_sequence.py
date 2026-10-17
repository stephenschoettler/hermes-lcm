"""Pins what the host's live compression-config sync can move inside an LCM engine.

Hermes core's TUI gateway mirrors a changed ``compression`` section into the active
context engine with three writes
(``tui_gateway/session_compression.py::_apply_live_compression_config()``):

    cc.threshold_tokens_cap = cc._coerce_threshold_tokens_cap(
        compression.get("threshold_tokens", _default_threshold_tokens_cap()))
    cc._threshold_tokens = cc._tail_token_budget = None   # "invalidate the cached trigger"

``_host_live_sync()`` below is that sequence, copied line for line, because it is the
subject: those three fields are the whole surface the host touches on a live config
change. The trigger itself is not among them — ``_set_context_length()`` derives
``threshold_tokens`` from ``effective_context_length * context_threshold``, bounded by
the assembly cap, so an LCM session's trigger is ratio-driven and a sync is write-only.
Whatever ``compression.threshold_tokens`` says, the trigger stays where the ratio put it.

Everything here is stdlib plus pytest: no gateway, no network, no production session.
Each engine gets its own sqlite file and a throwaway ``hermes_home`` under ``tmp_path``.

Two controls keep "0 reads" readable instead of vacuous:

* the assembly-cap control shows this probe can see the trigger move at all,
* the counter-liveness control shows the read counters fire when a field is read.

Without them, a dead counter and an inert field produce exactly the same zeroes.

Companion note: the three writes only work against an engine that exposes the fields
(#595 adds ``threshold_tokens_cap`` / ``_coerce_threshold_tokens_cap`` /
``_threshold_tokens`` / ``_tail_token_budget`` to ``LCMEngine``). Before that the
sequence raises ``AttributeError``, which is why the sync cases below declare that
precondition instead of pretending to measure it. The derivation cases at the end run
either way.
"""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

CONTEXT_LENGTH = 262144
CONTEXT_THRESHOLD = 0.75
# 262144 * 0.75, with no assembly cap in the way: the fixture's clean ratio value.
EXPECTED_TRIGGER = int(CONTEXT_LENGTH * CONTEXT_THRESHOLD)

_SYNC_FIELDS = ("threshold_tokens_cap", "_threshold_tokens", "_tail_token_budget")

# ``_coerce_threshold_tokens_cap`` is the one part of the change that lives on the
# class, so it is the marker this module keys on: the change adds it together with
# the three instance fields below. The guarded tests then assert on the instance
# that the fields are really there, so a half-landed change fails loudly instead of
# skipping past.
_HAS_WRITABLE_SYNC = hasattr(LCMEngine, "_coerce_threshold_tokens_cap")

requires_writable_sync = pytest.mark.skipif(
    not _HAS_WRITABLE_SYNC,
    reason=(
        "LCMEngine does not expose the fields the host's live compression sync writes "
        "(threshold_tokens_cap, _coerce_threshold_tokens_cap, _threshold_tokens, "
        "_tail_token_budget), so the verbatim sequence raises AttributeError — see #595"
    ),
)


def _assert_sync_fields_present():
    """The engine must carry the fields — checked without reading them.

    ``hasattr(engine, field)`` would call the counting properties and turn the
    precondition itself into the read the tests below forbid, so this inspects
    ``LCMEngine.__init__`` instead: every one of the fields is assigned there.
    """
    assigned = set(LCMEngine.__init__.__code__.co_names)
    for field in _SYNC_FIELDS:
        assert field in assigned, f"LCMEngine.__init__ never assigns {field} (see #595)"


def _host_live_sync(engine, threshold_tokens_value):
    """The TUI gateway's live-sync write sequence, verbatim."""
    engine.threshold_tokens_cap = engine._coerce_threshold_tokens_cap(threshold_tokens_value)
    engine._threshold_tokens = engine._tail_token_budget = None


class _CountedEngine(LCMEngine):
    """LCMEngine with read counters on the fields the host sync writes."""

    def __init__(self, *args, **kwargs):
        # Set the counter dict before the base __init__: it assigns the fields
        # below through these properties.
        self.reads = {"cap": 0, "cached_trigger": 0, "tail_budget": 0, "derivations": 0}
        super().__init__(*args, **kwargs)

    @property
    def threshold_tokens_cap(self):
        self.reads["cap"] += 1
        return self.__dict__.get("_probe_cap")

    @threshold_tokens_cap.setter
    def threshold_tokens_cap(self, value):
        self.__dict__["_probe_cap"] = value

    @property
    def _threshold_tokens(self):
        self.reads["cached_trigger"] += 1
        return self.__dict__.get("_probe_cached_trigger")

    @_threshold_tokens.setter
    def _threshold_tokens(self, value):
        self.__dict__["_probe_cached_trigger"] = value

    @property
    def _tail_token_budget(self):
        self.reads["tail_budget"] += 1
        return self.__dict__.get("_probe_tail_budget")

    @_tail_token_budget.setter
    def _tail_token_budget(self, value):
        self.__dict__["_probe_tail_budget"] = value

    def _effective_threshold_tokens(self, context_threshold_tokens):
        self.reads["derivations"] += 1
        return super()._effective_threshold_tokens(context_threshold_tokens)


@pytest.fixture
def engine(tmp_path):
    """A counted engine at a full context window, with no assembly cap in play.

    ``max_assembly_tokens`` / ``reserve_tokens_floor`` are pinned to 0 so the trigger
    is the plain ratio value; the control test raises the cap on purpose.
    """
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_probe.db")
    config.context_threshold = CONTEXT_THRESHOLD
    config.max_assembly_tokens = 0
    config.reserve_tokens_floor = 0
    eng = _CountedEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    eng._set_context_length(
        CONTEXT_LENGTH, source="probe-init", model="probe-model", provider="probe-provider"
    )
    return eng


def _set_context_length(eng):
    return eng._set_context_length(
        CONTEXT_LENGTH, source="probe-rerun", model="probe-model", provider="probe-provider"
    )


# ---------------------------------------------------------------------------
# The characterisation: a live sync is write-only
# ---------------------------------------------------------------------------


@requires_writable_sync
def test_host_live_sync_does_not_move_the_trigger(engine):
    """No value the host writes reaches the trigger, and a re-derivation proves it."""
    _assert_sync_fields_present()
    assert engine.threshold_tokens == EXPECTED_TRIGGER

    for value in (None, 60000, "60000"):
        _host_live_sync(engine, value)
        assert engine.threshold_tokens == EXPECTED_TRIGGER, (
            f"a live sync with threshold_tokens={value!r} moved the trigger"
        )
        _set_context_length(engine)
        assert engine.threshold_tokens == EXPECTED_TRIGGER, (
            f"the next context-length resolution moved the trigger after threshold_tokens={value!r}"
        )


@requires_writable_sync
def test_the_synced_cap_is_never_read_back(engine):
    """The two syncs above leave zero reads behind — and the counters do fire."""
    _assert_sync_fields_present()
    for value in (None, 60000):
        _host_live_sync(engine, value)
    _set_context_length(engine)

    assert engine.reads["cap"] == 0, "something read threshold_tokens_cap after the sync"
    assert engine.reads["cached_trigger"] == 0, "_threshold_tokens was read back"
    assert engine.reads["tail_budget"] == 0, "_tail_token_budget was read back"

    # Counter liveness: the property counts when the field is read. Without this,
    # the three zeroes above would be indistinguishable from dead instrumentation.
    assert engine.threshold_tokens_cap == 60000
    assert engine.reads["cap"] == 1


@requires_writable_sync
def test_a_cleared_context_length_resets_the_synced_fields(engine):
    """The non-positive branch drops the written values together with the trigger."""
    _assert_sync_fields_present()
    _host_live_sync(engine, 60000)
    assert engine._set_context_length(0, source="probe-clear") is True

    assert engine.threshold_tokens == 0
    assert engine.threshold_tokens_cap is None
    assert engine._threshold_tokens is None
    assert engine._tail_token_budget is None


@requires_writable_sync
def test_coerce_matches_the_core_contract():
    """Core's coercion: positive int or None, everything else is "no cap"."""
    coerce = LCMEngine._coerce_threshold_tokens_cap

    assert coerce(60000) == 60000
    assert coerce("60000") == 60000
    assert coerce(1) == 1
    assert coerce(None) is None
    assert coerce(0) is None
    assert coerce(-5) is None
    assert coerce("not-a-number") is None


# ---------------------------------------------------------------------------
# Controls and the derivation invariant: these run with or without #595
# ---------------------------------------------------------------------------


def test_the_trigger_is_ratio_derived(engine):
    """The trigger is ``context_length * context_threshold``, not a synced value."""
    assert engine._effective_assembly_token_cap() is None
    assert engine.threshold_tokens == EXPECTED_TRIGGER


def test_the_assembly_cap_control_moves_the_trigger(engine):
    """Control: a real constraint on the trigger is visible to this probe."""
    assert engine.reads["derivations"] == 1

    engine._config.max_assembly_tokens = 100000
    _set_context_length(engine)

    assert engine.reads["derivations"] == 2, "the re-derivation did not run"
    assert engine.threshold_tokens == 100000, "the control knob did not move the trigger"
