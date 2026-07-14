"""Encoder acquisition must never block callers on unbounded network I/O.

tiktoken.get_encoding() downloads its BPE file on first use when it is not
cached on disk; on restricted-egress hosts that request can hang for minutes
(measured ~127s per fresh process in a production deployment) and
_get_encoder() sits on the host's post_llm_call path.  These tests pin the
contract: a slow/hung loader must not stall count_tokens(), and the real
encoder is adopted once the background load completes.
"""

import threading
import time

import pytest

from hermes_lcm import tokens as tokens_mod


@pytest.fixture(autouse=True)
def _reset_encoder_state(monkeypatch):
    monkeypatch.setattr(tokens_mod, "_encoder", None)
    monkeypatch.setattr(tokens_mod, "_encoder_ready", False)
    monkeypatch.setattr(tokens_mod, "_encoder_thread", None)
    monkeypatch.setattr(tokens_mod, "_encoder_generation", 0)
    tokens_mod._count_tokens_cached.cache_clear()
    yield
    tokens_mod._count_tokens_cached.cache_clear()


def test_count_tokens_does_not_block_on_hung_loader(monkeypatch):
    release = threading.Event()

    def hung_loader():
        # Simulates the blocked-egress BPE download: never completes within
        # the test window.
        release.wait(timeout=30)
        raise RuntimeError("loader released without an encoder")

    monkeypatch.setattr(tokens_mod, "_load_encoder", hung_loader)
    monkeypatch.setattr(tokens_mod, "_ENCODER_FIRST_WAIT_S", 0.05)

    start = time.monotonic()
    count = tokens_mod.count_tokens("hello world " * 1000)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"count_tokens blocked for {elapsed:.1f}s on the loader"
    assert count > 0  # estimator result
    release.set()


def test_encoder_adopted_after_background_load(monkeypatch):
    class FakeEncoder:
        def encode(self, text):
            return list(range(42))

    started = threading.Event()

    def slow_loader():
        started.set()
        time.sleep(0.1)
        return FakeEncoder()

    monkeypatch.setattr(tokens_mod, "_load_encoder", slow_loader)
    monkeypatch.setattr(tokens_mod, "_ENCODER_FIRST_WAIT_S", 0.01)

    text = "adoption probe " * 50
    first = tokens_mod.count_tokens(text)  # estimator (loader still sleeping)
    assert started.wait(timeout=2.0)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if tokens_mod.count_tokens(text) == 42:
            break
        time.sleep(0.02)
    assert tokens_mod.count_tokens(text) == 42, "real encoder was never adopted"
    assert first != 42  # the pre-adoption call really used the estimator


def test_adoption_race_stale_estimate_not_served(monkeypatch):
    """An in-flight estimator miss must not poison the post-adoption cache.

    Ordering under test (deterministic, event-driven): a count_tokens() call
    misses the LRU and enters the estimator; the loader then adopts the real
    encoder (and clears the cache); the paused estimator call finishes last,
    inserting its result into the *post-clear* cache. Identical counts made
    after adoption must return the encoder's value, not the stale estimate.
    """

    class FakeEncoder:
        def encode(self, text):
            return list(range(7))

    release_load = threading.Event()
    fallback_entered = threading.Event()
    release_fallback = threading.Event()

    def gated_loader():
        release_load.wait(timeout=10)
        return FakeEncoder()

    real_fallback = tokens_mod._fallback_token_estimate
    probe = "y" * 400

    def gated_fallback(text):
        if text == probe:
            fallback_entered.set()
            release_fallback.wait(timeout=10)
        return real_fallback(text)

    monkeypatch.setattr(tokens_mod, "_load_encoder", gated_loader)
    monkeypatch.setattr(tokens_mod, "_fallback_token_estimate", gated_fallback)
    monkeypatch.setattr(tokens_mod, "_ENCODER_FIRST_WAIT_S", 0.01)

    results = []
    caller = threading.Thread(
        target=lambda: results.append(tokens_mod.count_tokens(probe)),
        daemon=True,
    )
    caller.start()

    # 1. The caller is inside the estimator (pre-adoption LRU miss in flight).
    assert fallback_entered.wait(timeout=5.0)

    # 2. The loader adopts the real encoder while that miss is still pending.
    release_load.set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and tokens_mod._encoder_generation == 0:
        time.sleep(0.01)
    assert tokens_mod._encoder_generation == 1, "encoder was never adopted"

    # 3. The stale estimate lands in the cache *after* adoption's clear.
    release_fallback.set()
    caller.join(timeout=5.0)
    assert not caller.is_alive()
    assert results == [400 // 4 + 1]  # the in-flight call used the estimator

    # 4. Post-adoption counts for the identical text must be exact -- both the
    #    first lookup and a repeat (i.e. the cached value is the encoder's).
    assert tokens_mod.count_tokens(probe) == 7
    assert tokens_mod.count_tokens(probe) == 7


def test_concurrent_callers_single_loader_prompt_returns(monkeypatch):
    """One loader thread total; non-first callers never wait on the load.

    Pins the concurrency contract: concurrent count_tokens() callers against
    a hung loader trigger exactly one _load_encoder() invocation, the first
    caller's wait is bounded, later callers return promptly with estimates,
    and counts after adoption are exact.
    """

    class FakeEncoder:
        def encode(self, text):
            return list(range(7))

    release_load = threading.Event()
    load_calls = []

    def gated_loader():
        load_calls.append(threading.get_ident())
        release_load.wait(timeout=10)
        return FakeEncoder()

    monkeypatch.setattr(tokens_mod, "_load_encoder", gated_loader)
    monkeypatch.setattr(tokens_mod, "_ENCODER_FIRST_WAIT_S", 0.05)

    probe = "z" * 400
    start = time.monotonic()
    first = tokens_mod.count_tokens(probe)  # starts the loader, bounded wait
    first_elapsed = time.monotonic() - start
    assert first_elapsed < 2.0, f"first caller waited {first_elapsed:.1f}s"
    assert first == 400 // 4 + 1

    results = []
    lock = threading.Lock()

    def worker(i):
        t0 = time.monotonic()
        value = tokens_mod.count_tokens(f"{'w' * 396}{i:04d}")
        with lock:
            results.append((value, time.monotonic() - t0))

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=5.0)
        assert not t.is_alive()

    assert len(load_calls) == 1, "loader must be started exactly once"
    assert [v for v, _ in results] == [400 // 4 + 1] * 4  # estimator values
    assert all(elapsed < 1.0 for _, elapsed in results), (
        "non-first callers must not wait on the hung loader"
    )

    release_load.set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and tokens_mod.count_tokens(probe) != 7:
        time.sleep(0.02)
    assert tokens_mod.count_tokens(probe) == 7, "exact counts after adoption"
    assert len(load_calls) == 1


def test_loader_failure_falls_back_to_estimator(monkeypatch):
    def broken_loader():
        raise ImportError("tiktoken not installed")

    monkeypatch.setattr(tokens_mod, "_load_encoder", broken_loader)
    monkeypatch.setattr(tokens_mod, "_ENCODER_FIRST_WAIT_S", 0.5)

    count = tokens_mod.count_tokens("x" * 400)
    assert count == 400 // 4 + 1  # ascii estimator path
    assert tokens_mod._encoder_ready is True
    assert tokens_mod._encoder is None
