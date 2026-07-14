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


def test_loader_failure_falls_back_to_estimator(monkeypatch):
    def broken_loader():
        raise ImportError("tiktoken not installed")

    monkeypatch.setattr(tokens_mod, "_load_encoder", broken_loader)
    monkeypatch.setattr(tokens_mod, "_ENCODER_FIRST_WAIT_S", 0.5)

    count = tokens_mod.count_tokens("x" * 400)
    assert count == 400 // 4 + 1  # ascii estimator path
    assert tokens_mod._encoder_ready is True
    assert tokens_mod._encoder is None
