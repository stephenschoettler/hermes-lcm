"""The compaction trigger must always sit above the verbatim fresh-tail floor.

Regression coverage for the compaction-livelock incident: LCM_CONTEXT_THRESHOLD=0.22 was tuned for
a 1M primary window (220K trigger). When a provider fallback moved the session to a 272K route the
same ratio yielded a 59,840-token trigger while the fresh tail alone held 66K-93K verbatim tokens.
The trigger was permanently unreachable, so every preflight re-compacted, logged "insufficient
progress", and eventually latched attempts_exhausted -- 56 of 57 observed no-progress passes were
on fallback routes, zero on the primary.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codex_routing import _FLOOR_HEADROOM, _minimum_viable_threshold


def _trigger(window, ratio):
    return int(window * ratio)


def test_primary_window_is_not_touched():
    """0.22 on a 1M window gives a 220K trigger against an 80K floor -- already viable."""
    minimum = _minimum_viable_threshold(1_000_000, 80_000)
    assert minimum is not None
    assert minimum < 0.22, "the primary route must not be raised"


def test_fallback_window_with_the_same_ratio_is_raised():
    """The exact live failure: 272K window, 0.22 ratio, 59,840 trigger under the floor."""
    window, floor, configured = 272_000, 80_000, 0.22
    assert _trigger(window, configured) < floor, "precondition: this ratio was unsatisfiable"

    minimum = _minimum_viable_threshold(window, floor)

    assert minimum is not None and minimum > configured
    assert _trigger(window, minimum) > floor


@pytest.mark.parametrize("floor", [66_000, 80_000, 92_811])
def test_raised_trigger_clears_every_observed_floor_with_headroom(floor):
    """Observed post-compaction floors ranged 66K-93K. Each must end up strictly below the new
    trigger, with real working room -- a trigger that merely equals the floor still livelocks."""
    window = 272_000
    minimum = _minimum_viable_threshold(window, floor)

    assert minimum is not None
    assert _trigger(window, minimum) >= floor * _FLOOR_HEADROOM


def test_unknown_inputs_do_not_invent_a_threshold():
    """No floor configured / no window known -> leave the operator's ratio alone."""
    assert _minimum_viable_threshold(272_000, 0) is None
    assert _minimum_viable_threshold(0, 80_000) is None
    assert _minimum_viable_threshold(-1, -1) is None


def test_floor_larger_than_the_window_does_not_disable_compaction():
    """A fresh tail that cannot fit is a real error. Returning a ~1.0 ratio would silently switch
    compaction off, which is worse than surfacing it."""
    assert _minimum_viable_threshold(100_000, 200_000) is None


def test_threshold_is_monotonic_in_the_floor():
    """A bigger verbatim floor always needs at least as high a trigger."""
    window = 272_000
    ratios = [_minimum_viable_threshold(window, f) for f in (40_000, 60_000, 80_000, 100_000)]
    assert all(r is not None for r in ratios)
    assert ratios == sorted(ratios)
