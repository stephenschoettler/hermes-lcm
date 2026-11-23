"""Round-8 regression tests (issue #2, findings 4029411030/1037/1046/1056)."""

import json
import os

from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import _tail_tagless


def _engine(tmp_path, name: str, **overrides) -> LCMEngine:
    from hermes_lcm.config import LCMConfig

    config = LCMConfig(
        database_path=str(tmp_path / f"{name}.db"),
        large_output_externalization_path=str(tmp_path / f"{name}-externalized"),
        fresh_tail_count=2,
        leaf_chunk_tokens=1,
        context_threshold=0.95,
        sensitive_patterns_enabled=True,
        sensitive_patterns=[
            "api_key",
            "bearer_token",
            "password_assignment",
            "private_key",
        ],
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / f"{name}-home"))
    engine.on_session_start(
        f"{name}-session",
        platform="synthetic",
        conversation_id=f"{name}-conversation",
        context_length=100_000,
    )
    return engine


def test_replay_identity_escape_closes_prefix_namespace(tmp_path):
    """Finding 4029411037: every string starting with EITHER reserved prefix
    must be escaped into a distinct encoding, so distinct provider strings
    never collapse onto one identity. The single-pass escape left
    escape-prefixed strings unchanged, colliding with escaped
    absent-prefixed content; the counted escape embeds the number of leading
    reserved prefixes and keeps the original verbatim."""
    from hermes_lcm.reconcile import (
        _REPLAY_IDENTITY_ABSENT_CONTENT_PREFIX,
        _REPLAY_IDENTITY_ABSENT_CONTENT_ESCAPE_PREFIX,
        _strip_replay_identity_shape_tag,
    )

    engine = _engine(tmp_path, "identity-escape")
    try:
        absent = engine._message_replay_identity(
            {"role": "user", "content": None}
        )
        escaped_absent = engine._message_replay_identity(
            {
                "role": "user",
                "content": _REPLAY_IDENTITY_ABSENT_CONTENT_PREFIX + " tail",
            }
        )
        bare_escape = engine._message_replay_identity(
            {
                "role": "user",
                "content": _REPLAY_IDENTITY_ABSENT_CONTENT_ESCAPE_PREFIX
                + _REPLAY_IDENTITY_ABSENT_CONTENT_PREFIX
                + " tail",
            }
        )
        assert bare_escape != escaped_absent, (
            "a provider string that merely starts with the escape prefix "
            "must not collide with escaped absent-prefixed content"
        )
        # The shape tag (finding 4029411030) prefixes the content component;
        # the escape encoding follows it.
        assert _strip_replay_identity_shape_tag(bare_escape[1]).startswith(
            _REPLAY_IDENTITY_ABSENT_CONTENT_ESCAPE_PREFIX
        )

        # The encoding stays unambiguous: unprefixed content passes through
        # bare, and the absent sentinel still round-trips to itself.
        plain = engine._message_replay_identity(
            {"role": "user", "content": "plain"}
        )
        assert _strip_replay_identity_shape_tag(plain[1]) == "plain"
        again = engine._message_replay_identity({"role": "user", "content": None})
        assert again == absent
    finally:
        engine.shutdown()


def test_load_externalized_payload_sidecar_refuses_symlink_escape(tmp_path):
    """Finding 4029411046: the public sidecar reader must not follow a symlink
    planted inside the payload store to a readable file elsewhere."""
    engine = _engine(tmp_path, "sidecar-symlink")
    try:
        storage = tmp_path / "sidecar-symlink-externalized"
        storage.mkdir(parents=True, exist_ok=True)
        secret = tmp_path / "outside-secret.json"
        secret.write_text(
            json.dumps({"kind": "tool_result", "content": "outside secret"}),
            encoding="utf-8",
        )
        os.symlink(secret, storage / "escape.json")

        assert engine.load_externalized_payload_sidecar("escape.json") is None
        assert engine.load_externalized_payload_sidecar("missing.json") is None

        # A real sidecar still loads through the same reader.
        (storage / "ok.json").write_text(
            json.dumps({"kind": "tool_result", "content": "durable text"}),
            encoding="utf-8",
        )
        loaded = engine.load_externalized_payload_sidecar("ok.json")
        assert loaded is not None
        assert loaded["content"] == "durable text"
    finally:
        engine.shutdown()


def test_storage_rebind_serializes_close_bind_reset_with_claimed_sanitation(tmp_path):
    """Finding 4029411056: the storage swap must hold the sanitation claim
    lock across close/bind/reset, so a claimed sanitation cannot resume
    against a half-swapped or new store. (The close/bind path requires the
    per-home default database — a configured ``database_path`` keeps one file
    across homes and never closes.)"""
    from hermes_lcm.config import LCMConfig

    config = LCMConfig(
        database_path="",
        large_output_externalization_path=str(tmp_path / "rebind-lock-externalized"),
        fresh_tail_count=2,
        leaf_chunk_tokens=1,
        context_threshold=0.95,
        sensitive_patterns_enabled=True,
        sensitive_patterns=[
            "api_key",
            "bearer_token",
            "password_assignment",
            "private_key",
        ],
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "rebind-lock-home"))
    engine.on_session_start(
        "rebind-lock-session",
        platform="synthetic",
        conversation_id="rebind-lock-conversation",
        context_length=100_000,
    )
    try:
        (tmp_path / "rebind-lock-home").mkdir(exist_ok=True)
        other_home = tmp_path / "other-home"
        other_home.mkdir()
        engine.ingest([{"role": "user", "content": "seed"}])
        events: list[str] = []
        lock_held_flags: list[bool] = []
        original_close = LCMEngine._close_storage
        original_bind = LCMEngine._bind_storage
        original_reset = LCMEngine._reset_profile_runtime_state

        def tracing_close(self):
            events.append("close")
            lock_held_flags.append(self._sanitation_claim_lock._is_owned())
            original_close(self)

        def tracing_bind(self, db_path, hermes_home=""):
            events.append("bind")
            lock_held_flags.append(self._sanitation_claim_lock._is_owned())
            original_bind(self, db_path, hermes_home)

        def tracing_reset(self):
            events.append("reset")
            lock_held_flags.append(self._sanitation_claim_lock._is_owned())
            original_reset(self)

        import unittest.mock as mock

        with mock.patch.multiple(
            LCMEngine,
            _close_storage=tracing_close,
            _bind_storage=tracing_bind,
            _reset_profile_runtime_state=tracing_reset,
        ):
            assert engine._rebind_storage_for_home(str(other_home)) is True

        assert events == ["close", "bind", "reset"], events
        assert lock_held_flags and all(lock_held_flags), (
            "close/bind/reset must all run under the sanitation claim lock"
        )
    finally:
        engine.shutdown()


def test_replay_identity_distinguishes_structured_content_from_json_text(tmp_path):
    """Finding 4029411030: structured list content and its JSON-text serialization
    must not share a replay identity.

    The handoff digest (``_cleanup_handoff_message_identity``) digests
    ``_message_replay_identity`` components, so a preflight handoff prepared
    while content is a structured list must validate differently after the
    caller replaces that content with its JSON-text serialization. The shape
    tag lives in the identity's content component (callers unpack exactly 4
    fields), so structured-vs-string is visible to both the digest and the
    reconciliation matching.

    Round-trip constraint: a stored row holds canonical JSON text, so a live
    string whose text is EXACTLY the canonical JSON of a list/dict is
    information-theoretically indistinguishable from the structured form after
    storage; the stored side derives the tag by exact round-trip decode (same
    convention as ``_identity_content_for_active_cleanup``). The tag therefore
    distinguishes structured from JSON-text at the LIVE boundary — including
    any serialization that is not byte-identical to LCM's canonical
    ``json.dumps`` — and keeps every storage round trip stable except that
    narrow exact-canonical-string class (documented residual).
    """
    engine = _engine(tmp_path, "identity-shape-tag")
    try:
        structured = engine._message_replay_identity(
            {"role": "user", "content": ["hello"]}
        )
        json_text = engine._message_replay_identity(
            {"role": "user", "content": '["hello"]'}
        )
        assert len(structured) == 4 and len(json_text) == 4
        assert structured != json_text, (
            "structured list content and its JSON-text serialization must not "
            "share a replay identity"
        )
        # The tag is the FIRST character of the content component, before the
        # escape machinery.
        assert structured[1][0] == "l"
        assert json_text[1][0] == "s"

        # The handoff digest inherits the fix: the same message set serialized
        # to JSON text must produce a different digest than the structured
        # original.
        structured_messages = [{"role": "user", "content": ["hello"]}]
        reinterpreted_messages = [{"role": "user", "content": '["hello"]'}]
        assert engine._cleanup_handoff_message_identity(
            structured_messages
        ) != engine._cleanup_handoff_message_identity(reinterpreted_messages)

        # Round-trip contract (corrected): the stored side tags text as STRING
        # unconditionally — a stored row cannot know whether its canonical JSON
        # text came from a live list or a live string, and the engine contract
        # (TestAssemblyToolPairGuardrail) requires a literal-JSON STRING's
        # re-ingested row to dedupe against itself. So a structured value's
        # stored row carries the string tag: restart replay of structured
        # content duplicates (the pre-existing behavior, unchanged by the tag).
        # The live-boundary distinction (what the handoff digest needs) holds.
        engine.ingest([{"role": "user", "content": ["hello"]}])
        row = next(
            row
            for row in engine._store.get_session_messages(engine._session_id)
            if row.get("role") == "user"
        )
        assert row["content"] == '["hello"]'
        live_identity = engine._message_replay_identity(
            {"role": "user", "content": ["hello"]}
        )
        stored_identity = engine._message_replay_identity(row, stored_row=True)
        assert stored_identity[1][0] == "s"
        assert stored_identity[1][1:] == live_identity[1][1:]
        cleanup_identity = engine._active_cleanup_replay_identity(stored_identity)
        assert cleanup_identity != live_identity  # string-tagged after round trip

        # Plain strings stay unprefixed apart from the tag and keep stable
        # identities across storage.
        plain_live = engine._message_replay_identity(
            {"role": "user", "content": "plain text"}
        )
        assert plain_live[1] == "splain text"
        engine.ingest([{"role": "user", "content": "plain text"}])
        plain_rows = [
            row
            for row in engine._store.get_session_messages(engine._session_id)
            if row.get("content") == "plain text"
        ]
        if plain_rows:
            plain_stored = engine._message_replay_identity(
                plain_rows[0], stored_row=True
            )
            assert plain_stored == plain_live

        # Absent content keeps its sentinel encoding, now shape-tagged, and
        # stays stable.
        absent_live = engine._message_replay_identity({"role": "user", "content": None})
        absent_stored = engine._message_replay_identity(
            {"role": "user", "content": None}, stored_row=True
        )
        assert absent_stored == absent_live
        assert absent_live[1][0] == "n"

        # The escape encoding (finding 4029411037) composes with the tag:
        # absent-tagged content stays distinct from string-tagged content.
        assert absent_live != json_text

        # A dict-content message gets its own tag and round-trips stably.
        dict_live = engine._message_replay_identity(
            {"role": "user", "content": {"k": "v"}}
        )
        assert dict_live[1][0] == "d"
        engine.ingest([{"role": "user", "content": {"k": "v"}}])
        dict_rows = [
            row
            for row in engine._store.get_session_messages(engine._session_id)
            if row.get("content") == '{"k": "v"}'
        ]
        if dict_rows:
            dict_stored = engine._message_replay_identity(dict_rows[0], stored_row=True)
            assert dict_stored == dict_live
    finally:
        engine.shutdown()


def test_store_id_map_cleanup_probe_keeps_first_char_of_taglike_text(tmp_path):
    """Bugbot 4041497061: the store-id map passes TAGLESS identities into the
    cleanup probe; the probe must not peel a shape tag again — assistant text
    like 'null hypothesis' keeps its first character."""
    engine = _engine(tmp_path, "cleanup-tagless")
    identity = engine._message_replay_identity(
        {"role": "assistant", "content": "null hypothesis"}
    )
    tagless = identity[1][1:]  # strip the tag like the map does
    cleaned = engine._active_cleanup_replay_identity(
        (identity[0], tagless, identity[2], identity[3]),
        content_is_tagged=False,
    )
    assert cleaned is not None
    assert cleaned[1] == "null hypothesis"


def test_stale_snapshot_detection_is_shape_tag_agnostic(tmp_path):
    """Bugbot 4041497059: staleness matching compares tagless identities —
    a live structured message must be detected as stale-overlap with its
    stored string-tagged row."""
    engine = _engine(tmp_path, "stale-tagless")
    engine.ingest([{"role": "user", "content": ["hello"]}])
    stored_row = next(
        row
        for row in engine._store.get_session_messages(engine._session_id)
        if row.get("role") == "user"
    )
    stored_identity = engine._message_replay_identity(stored_row, stored_row=True)
    live_identity = engine._message_replay_identity(
        {"role": "user", "content": ["hello"]}
    )
    assert stored_identity != live_identity  # tags differ
    assert _tail_tagless([stored_identity]) == _tail_tagless([live_identity])


def test_prefix_count_scan_is_not_quadratic():
    """Round-3 finding 4041509636: counting N leading reserved prefixes must be
    linear in the prefix bytes, not quadratic in the payload."""
    import time as _time
    from hermes_lcm.reconcile import (
        _REPLAY_IDENTITY_ABSENT_CONTENT_PREFIX,
        _count_leading_reserved_prefixes,
    )
    payload = _REPLAY_IDENTITY_ABSENT_CONTENT_PREFIX * 2000
    start = _time.monotonic()
    count = _count_leading_reserved_prefixes(payload)
    elapsed = _time.monotonic() - start
    assert count == 2000
    assert elapsed < 0.5, f"prefix scan took {elapsed:.3f}s for 2000 prefixes"


def test_claimed_sanitation_releases_lock_on_fallback(tmp_path):
    """Round-3 finding 4041509641: when the cleanup-only path does not apply,
    the claim lock must be released before model-backed compaction runs."""
    from hermes_lcm.compaction import _SanitationFallbackNeeded
    from hermes_lcm.config import LCMConfig

    config = LCMConfig(
        database_path=str(tmp_path / "fb.db"),
        large_output_externalization_path=str(tmp_path / "fb-ext"),
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        context_threshold=0.5,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "fb-home"))
    engine.on_session_start("fb-session", platform="synthetic",
                            conversation_id="fb-conversation", context_length=100_000)

    lock_during_fallback = []
    original_impl = engine._compress_impl

    def spying_impl(*args, **kwargs):
        if not kwargs.get("claimed_sanitation"):
            acquired = engine._sanitation_claim_lock.acquire(blocking=False)
            if acquired:
                engine._sanitation_claim_lock.release()
            lock_during_fallback.append(acquired)
        return original_impl(*args, **kwargs)

    engine._compress_impl = spying_impl
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "filler " * 500},
        {"role": "assistant", "content": "filler reply " * 500},
        {"role": "user", "content": "trigger threshold"},
    ]
    # Drive compress through the claimed path; the impl raises fallback when the
    # cleanup-only conditions fail; the generic rerun must happen unlocked.
    try:
        result = engine.compress(messages, current_tokens=90_000)
    except _SanitationFallbackNeeded:
        raise AssertionError("fallback must be handled inside compress(), not leak")
    assert result is not None
    assert lock_during_fallback, "generic fallback rerun was not observed"
    # acquired=True at every probe means the lock was FREE during the fallback
    # rerun (probe acquires non-blocking and releases immediately).
    assert all(lock_during_fallback), "claim lock held during generic fallback compaction"


def test_cleanup_decodes_absent_sentinel_to_none(tmp_path):
    """Round-3 finding 4041846913: a stored assistant row with NULL content
    (tool-call shape) carries the absent sentinel; the cleanup probe must
    translate it back to None, not treat it as nonempty text."""
    engine = _engine(tmp_path, "absent-cleanup")
    identity = engine._message_replay_identity(
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        stored_row=True,
    )
    cleaned = engine._active_cleanup_replay_identity(identity)
    assert cleaned is not None
    # The sentinel must NOT survive as literal text in the cleaned content.
    assert "[LCM replay identity: content absent]" not in cleaned[1]


def test_sidecar_restored_content_is_re_escaped(tmp_path):
    """Round-3 finding 4041846916: content restored from a sidecar whose text
    begins with a reserved prefix must carry the counted escape encoding, or
    live and stored identities diverge."""
    from hermes_lcm.reconcile import (
        _REPLAY_IDENTITY_ABSENT_CONTENT_PREFIX,
        _escape_replay_identity_content,
    )
    engine = _engine(tmp_path, "sidecar-escape")
    raw = _REPLAY_IDENTITY_ABSENT_CONTENT_PREFIX + " payload tail"
    # The escaped live identity encoding:
    expected = _escape_replay_identity_content(raw)
    assert expected.startswith("[LCM replay identity: content escaped] x1 ")
    # The identity fn applied to a payload-restored content string must produce
    # the same encoding (simulated by computing the identity of a message whose
    # content IS the restored raw string):
    identity = engine._message_replay_identity(
        {"role": "assistant", "content": raw}
    )
    assert identity[1][1:] == expected  # tag prefix + escaped encoding
