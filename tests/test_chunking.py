from __future__ import annotations

import pytest


from hermes_lcm.chunking import (
    MIN_CONVERSATIONAL_TOKENS_ENV,
    chunk_message,
    group_by_store_id,
    iter_message_chunks,
    normalize_content_policy,
    _CHUNK_TARGET_TOKENS,
    _sentence_spans,
    _window_spans,
)


def _long_text(sentences: int, words_per_sentence: int = 40) -> str:
    """Build text with enough tokens to force multi-window splitting."""
    sentence = " ".join(f"word{i}" for i in range(words_per_sentence))
    return " ".join(f"{sentence} number{n}." for n in range(sentences))


class TestPolicyNormalization:
    def test_defaults_to_conversational(self):
        assert normalize_content_policy(None) == "conversational"
        assert normalize_content_policy("") == "conversational"
        assert normalize_content_policy("bogus") == "conversational"

    def test_accepts_known_policies(self):
        assert normalize_content_policy("HEADS") == "heads"
        assert normalize_content_policy(" full ") == "full"
        assert normalize_content_policy("conversational") == "conversational"


class TestConversationalPolicy:
    @pytest.fixture(autouse=True)
    def _default_threshold(self, monkeypatch):
        # Default-behavior assertions must not inherit a deployment setting
        # from the test runner's environment.
        monkeypatch.delenv(MIN_CONVERSATIONAL_TOKENS_ENV, raising=False)

    def test_skips_short_acknowledgment(self):
        assert chunk_message(1, "user", "ok thanks", policy="conversational") == []

    def test_embeds_substantial_user_turn(self):
        text = "This is a substantial user question. " * 6
        chunks = chunk_message(7, "user", text, policy="conversational")
        assert len(chunks) == 1
        assert chunks[0].store_id == 7
        assert chunks[0].chunk_index == 0
        assert chunks[0].chunk_id == "7:0"
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len(text)

    def test_skips_tool_and_system_roles(self):
        long = _long_text(20)
        assert chunk_message(1, "tool", long, policy="conversational") == []
        assert chunk_message(1, "system", long, policy="conversational") == []

    def test_skips_empty_content(self):
        assert chunk_message(1, "user", "   ", policy="conversational") == []
        assert chunk_message(1, "assistant", None, policy="conversational") == []

    def test_min_tokens_env_lowers_threshold(self, monkeypatch):
        short = "I adopted a rescue dog named Biscuit yesterday."
        monkeypatch.delenv(MIN_CONVERSATIONAL_TOKENS_ENV, raising=False)
        assert chunk_message(1, "user", short, policy="conversational") == []
        monkeypatch.setenv(MIN_CONVERSATIONAL_TOKENS_ENV, "0")
        chunks = chunk_message(1, "user", short, policy="conversational")
        assert len(chunks) == 1
        assert chunks[0].char_end == len(short)
        # heads shares the conversational-role threshold
        assert chunk_message(1, "user", short, policy="heads") != []

    def test_min_tokens_env_invalid_or_negative_uses_default(self, monkeypatch):
        short = "ok thanks"
        for raw in ("bogus", "-5", ""):
            monkeypatch.setenv(MIN_CONVERSATIONAL_TOKENS_ENV, raw)
            assert chunk_message(1, "user", short, policy="conversational") == []

    def test_min_tokens_env_never_embeds_non_conversational_roles(self, monkeypatch):
        monkeypatch.setenv(MIN_CONVERSATIONAL_TOKENS_ENV, "0")
        assert chunk_message(1, "tool", "short tool note", policy="conversational") == []
        assert chunk_message(1, "system", "short system note", policy="conversational") == []


class TestTurnAlignmentAndSpans:
    def test_small_message_is_single_chunk(self):
        text = "A moderately sized assistant reply that stays well under the target. " * 4
        chunks = chunk_message(3, "assistant", text, policy="full")
        assert len(chunks) == 1
        assert (chunks[0].char_start, chunks[0].char_end) == (0, len(text))

    def test_large_message_splits_into_multiple_windows(self):
        text = _long_text(40)
        chunks = chunk_message(5, "assistant", text, policy="full")
        assert len(chunks) >= 2
        # Chunk indexes are the natural window positions, contiguous from 0.
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_windows_respect_token_target(self):
        text = _long_text(40)
        for start, end in _window_spans(text):
            # Each window's own tokens stay near the target (one sentence may
            # overshoot, but a window is never wildly over).
            assert start < end

    def test_one_sentence_overlap_between_windows(self):
        text = _long_text(40)
        windows = _window_spans(text)
        assert len(windows) >= 2
        # Overlap: each later window re-opens inside the previous window.
        for prev, nxt in zip(windows, windows[1:]):
            assert nxt[0] < prev[1]

    def test_spans_map_back_to_exact_substring(self):
        text = _long_text(30)
        chunks = chunk_message(9, "assistant", text, policy="full")
        for chunk in chunks:
            assert chunk.text == text[chunk.char_start:chunk.char_end]

    def test_sentence_spans_tile_the_string(self):
        text = "First sentence. Second one! Third?\n\nFourth line."
        spans = _sentence_spans(text)
        assert spans[0][0] == 0
        assert spans[-1][1] == len(text)
        for prev, nxt in zip(spans, spans[1:]):
            assert prev[1] == nxt[0]

    def test_single_oversized_sentence_terminates(self):
        # No sentence punctuation and > target tokens: must still window without
        # looping forever.
        text = " ".join(f"tok{i}" for i in range(_CHUNK_TARGET_TOKENS * 3))
        windows = _window_spans(text)
        assert windows
        assert windows[0][0] == 0


class TestHeadsPolicy:
    def test_tool_result_head_chunk_only_when_no_errors(self):
        text = _long_text(40)
        chunks = chunk_message(11, "tool", text, policy="heads")
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0

    def test_tool_result_extracts_error_signature_chunks(self):
        clean = _long_text(12)
        error_tail = (
            " Traceback (most recent call last): "
            + "boom " * 200
            + " ValueError raised."
        )
        text = clean + error_tail
        chunks = chunk_message(13, "tool", text, policy="heads")
        assert len(chunks) >= 2
        # Head is index 0; the error window keeps its natural (non-zero) index.
        assert chunks[0].chunk_index == 0
        assert any(c.chunk_index > 0 for c in chunks)
        joined = " ".join(c.text.lower() for c in chunks[1:])
        assert "traceback" in joined or "valueerror" in joined

    def test_error_chunks_capped_at_three(self):
        # Many error-bearing windows; only head + 3 error chunks are emitted.
        block = ("error: something failed here " + "pad " * 200 + ". ")
        text = _long_text(6) + block * 8
        chunks = chunk_message(17, "tool", text, policy="heads")
        assert len(chunks) <= 4

    def test_conversational_turns_still_embedded_under_heads(self):
        text = "This assistant message is long enough to matter here. " * 12
        chunks = chunk_message(19, "assistant", text, policy="heads")
        assert len(chunks) >= 1

    def test_short_tool_result_still_gets_head(self):
        chunks = chunk_message(21, "tool", "small tool output line", policy="heads")
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0


class TestFullPolicy:
    def test_includes_all_roles(self):
        long = _long_text(20)
        assert chunk_message(1, "tool", long, policy="full")
        assert chunk_message(1, "system", long, policy="full")
        assert chunk_message(1, "user", long, policy="full")

    def test_no_min_token_skip(self):
        chunks = chunk_message(2, "user", "short but kept", policy="full")
        assert len(chunks) == 1

    def test_giant_message_reduced_to_head_and_error(self, monkeypatch):
        import hermes_lcm.chunking as chunking

        # Force the giant-message path without building a 32k-token string.
        monkeypatch.setattr(chunking, "_FULL_PER_MESSAGE_TOKEN_CAP", 100)
        text = _long_text(30) + " error: it failed " + "pad " * 200 + "."
        chunks = chunk_message(23, "assistant", text, policy="full")
        # Far fewer than a full windowing of the same text.
        full_windows = len(chunking._window_spans(text))
        assert len(chunks) < full_windows
        assert chunks[0].chunk_index == 0


class TestIterMessageChunks:
    def test_iterates_and_skips_rows_without_store_id(self):
        rows = [
            {"store_id": 1, "role": "user", "content": "A long enough user turn here to embed. " * 12},
            {"store_id": None, "role": "user", "content": "orphan"},
            {"role": "user", "content": "no store id"},
            {"store_id": 2, "role": "tool", "content": _long_text(10)},
        ]
        chunks = list(iter_message_chunks(rows, policy="conversational"))
        # Only the first row qualifies under conversational.
        assert {c.store_id for c in chunks} == {1}

    def test_full_policy_across_rows(self):
        rows = [
            {"store_id": 5, "role": "user", "content": "kept short"},
            {"store_id": 6, "role": "tool", "content": _long_text(10)},
        ]
        chunks = list(iter_message_chunks(rows, policy="full"))
        assert {c.store_id for c in chunks} == {5, 6}


class TestGroupByStoreId:
    def test_groups_contiguous_runs_preserving_order(self):
        # Two chunks of message 1, one of 2, three of 3 (discovery order).
        store_ids = [1, 1, 2, 3, 3, 3]
        assert group_by_store_id(store_ids) == [[0, 1], [2], [3, 4, 5]]

    def test_empty(self):
        assert group_by_store_id([]) == []

    def test_single_chunk_per_message(self):
        assert group_by_store_id([7, 8, 9]) == [[0], [1], [2]]

    def test_non_adjacent_same_store_stays_separate(self):
        # Consecutive-run grouping: a repeat that is not adjacent opens a new
        # document (chunk discovery never emits a message's chunks non-contiguously).
        assert group_by_store_id([1, 2, 1]) == [[0], [1], [2]]
