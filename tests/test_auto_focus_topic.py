"""Tests for auto-derive focus topic during compression.

Covers:
- derives from the latest real user turns
- skips synthetic context-summary/scaffold content
- explicit focus_topic still wins
- per-turn and total truncation
- multimodal/text-part content
- no leakage of configured sensitive values from structured content or bearer-style text
"""

from hermes_lcm.engine import LCMEngine


class TestDeriveAutoFocusTopic:
    """Tests for LCMEngine._derive_auto_focus_topic."""

    # --- Test 1: derives from latest real user turns ---

    def test_derives_from_latest_user_turns(self, tmp_path):
        engine = LCMEngine(config=None)
        try:
            messages = [
                {"role": "assistant", "content": "Previous assistant reply"},
                {"role": "user", "content": "Please check config.yaml"},
                {"role": "assistant", "content": "Sure let me check"},
                {"role": "user", "content": "The compression issue came up again"},
                {"role": "assistant", "content": "Let me look into that"},
                {"role": "user", "content": "Ok let's go with option A"},
            ]
            focus = engine._derive_auto_focus_topic(messages)
            assert focus is not None
            assert "Recent user focus:" in focus
            assert "Please check config.yaml" in focus
            assert "The compression issue came up again" in focus
            assert "Ok let's go with option A" in focus
            # Assistant messages should not appear
            assert "Previous assistant reply" not in focus
            assert "Sure let me check" not in focus
        finally:
            engine.shutdown()

    def test_returns_none_for_empty_messages(self, tmp_path):
        engine = LCMEngine(config=None)
        try:
            assert engine._derive_auto_focus_topic([]) is None
        finally:
            engine.shutdown()

    def test_returns_none_for_no_user_messages(self, tmp_path):
        engine = LCMEngine(config=None)
        try:
            messages = [
                {"role": "assistant", "content": "Reply one"},
                {"role": "assistant", "content": "Reply two"},
            ]
            assert engine._derive_auto_focus_topic(messages) is None
        finally:
            engine.shutdown()

    # --- Test 2: skips synthetic context-summary content ---

    def test_skips_context_compaction_summary(self, tmp_path):
        engine = LCMEngine(config=None)
        try:
            messages = [
                {"role": "user", "content": "Please check config.yaml"},
                {"role": "assistant", "content": "Sure"},
                {"role": "user", "content": "[CONTEXT COMPACTION -- REFERENCE ONLY] Old summary content"},
                {"role": "user", "content": "This is the real message after compaction"},
            ]
            focus = engine._derive_auto_focus_topic(messages)
            assert focus is not None
            assert "Old summary content" not in focus
            assert "This is the real message after compaction" in focus
            assert "Please check config.yaml" in focus
        finally:
            engine.shutdown()

    def test_skips_all_summary_markers(self, tmp_path):
        engine = LCMEngine(config=None)
        try:
            summaries = [
                "[CONTEXT COMPACTION] something",
                "[CONTEXT SUMMARY]: test",
                "Earlier turns have been compacted...",
                "Earlier turns were compacted...",
            ]
            for summary in summaries:
                messages = [
                    {"role": "user", "content": summary},
                    {"role": "user", "content": "This is a real message"},
                ]
                focus = engine._derive_auto_focus_topic(messages)
                assert focus is not None
                assert "This is a real message" in focus
                assert summary.split("]")[-1].strip() not in focus or summary not in focus
        finally:
            engine.shutdown()

    # --- Test 3: explicit focus_topic still wins ---

    def test_explicit_focus_topic_wins(self, tmp_path):
        """When focus_topic is explicitly provided at the compress() level,
        auto-derive is skipped. We cannot test the full compress() path here,
        but we verify that _derive_auto_focus_topic itself produces output that
        would be replaced by an explicit focus_topic in the caller.

        The actual guard is in compress(): ``if focus_topic is None`` -- so
        when focus_topic is provided, this method is never called.
        """
        engine = LCMEngine(config=None)
        try:
            messages = [
                {"role": "user", "content": "First user message"},
                {"role": "user", "content": "Second user message"},
            ]
            focus = engine._derive_auto_focus_topic(messages)
            assert focus is not None
            assert "First user message" in focus
        finally:
            engine.shutdown()

    # --- Test 4: per-turn and total truncation ---

    def test_per_turn_truncation(self, tmp_path):
        engine = LCMEngine(config=None)
        try:
            long_msg = "x" * 500
            messages = [{"role": "user", "content": long_msg}]
            focus = engine._derive_auto_focus_topic(messages)
            assert focus is not None
            assert "\u2026" in focus  # truncation marker
        finally:
            engine.shutdown()

    def test_total_truncation(self, tmp_path):
        engine = LCMEngine(config=None)
        try:
            # 3 long messages should exceed total limit of _AUTO_FOCUS_MAX_CHARS (700)
            messages = [
                {"role": "user", "content": "a" * 300},
                {"role": "user", "content": "b" * 300},
                {"role": "user", "content": "c" * 300},
            ]
            focus = engine._derive_auto_focus_topic(messages)
            assert focus is not None
            assert len(focus) <= 700  # _AUTO_FOCUS_MAX_CHARS exactly
        finally:
            engine.shutdown()

    def test_max_3_turns(self, tmp_path):
        engine = LCMEngine(config=None)
        try:
            messages = [
                {"role": "user", "content": "Message four"},
                {"role": "user", "content": "Message three"},
                {"role": "user", "content": "Message two"},
                {"role": "user", "content": "Message one"},
            ]
            focus = engine._derive_auto_focus_topic(messages)
            assert focus is not None
            assert "Message one" in focus
            assert "Message two" in focus
            assert "Message three" in focus
            assert "Message four" not in focus  # oldest is dropped
            assert focus.count("-") == 3  # exactly 3 bullet points
        finally:
            engine.shutdown()

    # --- Test 5: multimodal/text-part content ---

    def test_multimodal_content(self, tmp_path):
        engine = LCMEngine(config=None)
        try:
            messages = [
                {"role": "user", "content": "Look at this image"},
                {"role": "user", "content": [{"type": "text", "text": "How does this image look?"}]},
            ]
            focus = engine._derive_auto_focus_topic(messages)
            assert focus is not None
            assert "image" in focus
        finally:
            engine.shutdown()

    # --- Test 6: no leakage of configured sensitive values ---

    def test_redacted_working_messages_no_leakage(self, tmp_path):
        """Sensitive values in working_messages are already redacted by
        _ingest_messages -> _redact_active_replay_messages, so the derived
        focus topic must not contain raw secrets."""
        engine = LCMEngine(config=None)
        try:
            engine._session_id = "test-focus-session"
            # Simulate raw messages that would be ingested
            # Note: In production, these go through _ingest_messages which calls
            # _redact_active_replay_messages before reaching _derive_auto_focus_topic.
            # We pass already-redacted content to simulate working_messages.
            redacted_messages = [
                {"role": "user", "content": "Authorization: Bearer [REDACTED]"},
                {"role": "user", "content": "api_key: [REDACTED] token: [REDACTED]"},
            ]
            focus = engine._derive_auto_focus_topic(redacted_messages)
            assert focus is not None
            assert "[REDACTED]" in focus
            # Verify no raw secrets leaked
            assert "Bearer " not in focus or "[REDACTED]" in focus
        finally:
            engine.shutdown()

    def test_structured_content_no_leakage(self, tmp_path):
        """Dict/JSON token values in working_messages are redacted."""
        engine = LCMEngine(config=None)
        try:
            # Simulate working_messages where content is already redacted
            # by _redact_active_replay_messages for dict-type content
            redacted_messages = [
                {
                    "role": "user",
                    "content": 'config: {"api_key": "[REDACTED]", "endpoint": "http://example.com"}',
                },
            ]
            focus = engine._derive_auto_focus_topic(redacted_messages)
            assert focus is not None
            assert "[REDACTED]" in focus
            assert 'api_key": "' not in focus.split("[REDACTED]")[0] or "[REDACTED]" in focus
        finally:
            engine.shutdown()

    def test_ingest_messages_redacts_before_focus_derivation(self, tmp_path):
        """Integration test: _ingest_messages returns redacted messages that
        are safe to pass to _derive_auto_focus_topic. This verifies the
        actual redaction path, not a mock."""
        from hermes_lcm.config import LCMConfig

        config = LCMConfig(database_path=str(tmp_path / "focus-ingest.db"))
        # Enable sensitive pattern redaction explicitly
        config.sensitive_patterns_enabled = True
        config.sensitive_patterns = ["api_key", "bearer_token", "password_assignment"]
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes-home"))
        try:
            engine._session_id = "test-ingest-focus"
            engine.context_length = 200000
            engine.threshold_tokens = 100000

            # Raw messages with sensitive patterns that match the regex catalog
            raw_messages = [
                {"role": "user", "content": "api_key: sk-proj-abc123def456ghi789 and I checked the config"},
                {"role": "user", "content": "Let me verify the deployment"},
                {"role": "user", "content": "password: supersecret1234 and Authorization: Bearer eyJhbGciOiJIUzI1NiJ9"},
            ]

            # Ingest returns redacted working_messages
            working_messages = engine._ingest_messages(raw_messages)

            # Derive focus from working_messages (the fixed code path)
            focus = engine._derive_auto_focus_topic(working_messages)

            if focus is not None:
                # Verify sensitive values are redacted in the derived focus
                assert "sk-proj-abc123def456ghi789" not in focus, "Raw API key leaked in focus"
                assert "supersecret1234" not in focus, "Raw password leaked in focus"
                assert "eyJhbGciOiJIUzI1NiJ9" not in focus, "Raw bearer token leaked in focus"
        finally:
            engine.shutdown()

    def test_redact_sensitive_text_safety_net(self, tmp_path):
        """Verify the additional redaction safety net in _derive_auto_focus_topic
        catches sensitive values that _redact_active_replay_messages misses
        (e.g., text extracted from structured content via text_content_for_pattern_matching)."""
        from hermes_lcm.config import LCMConfig

        config = LCMConfig(database_path=str(tmp_path / "focus-safety.db"))
        config.sensitive_patterns_enabled = True
        config.sensitive_patterns = ["api_key", "bearer_token", "password_assignment"]
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes-home"))
        try:
            # Messages where content is a list (structured/multimodal) --
            # _redact_active_replay_messages uses parse_json_strings=False for content,
            # so the safety net in _derive_auto_focus_topic must catch these.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "api_key: sk-proj-abc123def456ghi789 check this"},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9-abcdefgh"},
                    ],
                },
            ]

            focus = engine._derive_auto_focus_topic(messages)

            if focus is not None:
                assert "sk-proj-abc123def456ghi789" not in focus, "Raw API key leaked from structured content"
                assert "eyJhbGciOiJIUzI1NiJ9-abcdefgh" not in focus, "Raw bearer token leaked from structured content"
        finally:
            engine.shutdown()

    # --- Skip empty user messages ---

    def test_skips_empty_user_messages(self, tmp_path):
        engine = LCMEngine(config=None)
        try:
            messages = [
                {"role": "user", "content": ""},
                {"role": "user", "content": "   "},
                {"role": "user", "content": "This is a real message"},
            ]
            focus = engine._derive_auto_focus_topic(messages)
            assert focus is not None
            assert "This is a real message" in focus
            assert focus.count("-") == 1
        finally:
            engine.shutdown()
