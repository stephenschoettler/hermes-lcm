"""Ingest must preserve a caller-supplied per-message ``timestamp``.

A caller re-ingesting messages it has already seen — restoring a durable
transcript, importing a prior session, replaying after a restart — knows each
message's real arrival time and passes it as ``timestamp``. Stamping ingest
wall-clock time over it collapses the entire restored history onto the instant
of the import, which breaks date-range queries and recency ordering.

Live messages carry no ``timestamp`` and must keep getting a fresh per-row
clock reading; ``test_lcm_core.py::test_append_batch_timestamps_are_unique_per_row``
pins that invariant and must stay green.
"""

import time

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.store import MessageStore

REPLAY_BASE = 1_700_000_000.0


@pytest.fixture
def store(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "ts.db"))
    store = MessageStore(config.database_path, ingest_protection_config=config)
    try:
        yield store
    finally:
        store.close()


def _replayed(n, base=REPLAY_BASE):
    return [
        {"role": "user", "content": f"replayed message {i}", "timestamp": base + i}
        for i in range(n)
    ]


def test_append_batch_preserves_supplied_timestamps(store):
    messages = _replayed(4)

    ids = store.append_batch("replay", messages)

    assert [store.get(i)["timestamp"] for i in ids] == [m["timestamp"] for m in messages]


def test_append_preserves_supplied_timestamp(store):
    store_id = store.append(
        "replay", {"role": "user", "content": "one", "timestamp": REPLAY_BASE}
    )

    assert store.get(store_id)["timestamp"] == REPLAY_BASE


def test_replayed_batch_is_not_collapsed_onto_import_time(store):
    """The user-visible symptom: a restored history must not all land at 'now'."""
    before = time.time()

    ids = store.append_batch("replay", _replayed(5))

    stamps = [store.get(i)["timestamp"] for i in ids]
    assert all(ts < before for ts in stamps)
    assert len(set(stamps)) == 5


def test_supplied_timestamps_survive_time_bounds(store):
    """``get_time_bounds`` is a real consumer of these values — journal/date
    range features read the earliest/latest stamps of a set of rows."""
    day = 86_400.0
    ids = store.append_batch(
        "replay",
        [
            {"role": "user", "content": "old", "timestamp": REPLAY_BASE},
            {"role": "user", "content": "newer", "timestamp": REPLAY_BASE + day},
        ],
    )

    earliest, latest = store.get_time_bounds(ids)

    assert earliest == REPLAY_BASE
    assert latest == REPLAY_BASE + day


def test_live_messages_without_timestamp_still_get_now(store):
    """Control: the live-ingest path is unchanged."""
    before = time.time()

    ids = store.append_batch(
        "live", [{"role": "user", "content": f"m{i}"} for i in range(3)]
    )

    after = time.time()
    for store_id in ids:
        assert before <= store.get(store_id)["timestamp"] <= after


def test_live_batch_timestamps_remain_unique(store):
    """Control: preserves the existing per-row-clock invariant for live ingest."""
    ids = store.append_batch(
        "live", [{"role": "user", "content": f"m{i}"} for i in range(50)]
    )

    stamps = [store.get(i)["timestamp"] for i in ids]

    assert len(set(stamps)) == len(stamps)


@pytest.mark.parametrize(
    "bad",
    [None, "", "not-a-number", float("nan"), float("inf"), float("-inf"), 0, -1, [], {}],
)
def test_unusable_timestamp_falls_back_to_now_without_raising(store, bad):
    """A malformed optional field must never fail ingest or write a 1970 epoch."""
    before = time.time()

    store_id = store.append("live", {"role": "user", "content": "x", "timestamp": bad})

    after = time.time()
    assert before <= store.get(store_id)["timestamp"] <= after


def test_integer_and_string_timestamps_are_coerced(store):
    """Transcript formats routinely carry epochs as ints or numeric strings."""
    ids = store.append_batch(
        "replay",
        [
            {"role": "user", "content": "int", "timestamp": int(REPLAY_BASE)},
            {"role": "user", "content": "str", "timestamp": str(REPLAY_BASE)},
        ],
    )

    assert [store.get(i)["timestamp"] for i in ids] == [REPLAY_BASE, REPLAY_BASE]


def test_mixed_batch_preserves_supplied_and_stamps_the_rest(store):
    before = time.time()

    ids = store.append_batch(
        "mixed",
        [
            {"role": "user", "content": "replayed", "timestamp": REPLAY_BASE},
            {"role": "user", "content": "live"},
        ],
    )

    replayed_ts, live_ts = (store.get(i)["timestamp"] for i in ids)
    assert replayed_ts == REPLAY_BASE
    assert live_ts >= before
