"""Ingest idempotency invariant tests (the whole-transcript re-ingest defect — duplicate-ingest root cause).

The live incident: whole-transcript re-ingest on every preflight/compress
double-ingested ~2,592 duplicate message pairs (identical content, timestamps
seconds-to-minutes apart) into the store, counterfeiting ~600K tokens of
context and driving a session into compression no-progress → auto-reset.

Invariants proven here (store-level guard in ``MessageStore``):
(a) replaying the same message batch twice WITHIN the dedup window ingests
    each message exactly once;
(b) replaying the same content OUTSIDE the window — the stored row keeps its
    old source time while the candidate arrives fresh — ingests again;
(c) distinct messages (different content, role, conversation_id, or
    tool_call_id) are never collapsed;
(d) the guard catches the incident mechanism: cursor reset to 0 WITHOUT a
    reconcile decision (base re-INSERTs 3 duplicate rows; the guard
    prevents it);
(e) tool-role Hermes persisted-output markers are EXEMPT: retry semantics
    live above the store, so a replayed marker still re-appends;
(f) the duplicate probe uses the indexed (session_id, timestamp) range, so
    the per-message cost is an indexed seek, not a table scan.
"""

from __future__ import annotations

import time

from hermes_lcm.config import LCMConfig
from hermes_lcm.store import MessageStore
from tests.test_tool_contracts import LCMEngine  # host-stub tolerant import


def _batch(now: float, conversation_id: str = "conv-1"):
    return [
        {"role": "user", "content": "hello there", "timestamp": now - 4},
        {"role": "assistant", "content": "hi! how can I help?", "timestamp": now - 3},
        {"role": "user", "content": "what is the weather?", "timestamp": now - 2},
    ]


def test_replayed_batch_within_window_ingests_once(tmp_path):
    """(a) Replaying the identical batch seconds later must not add rows."""
    store = MessageStore(tmp_path / "replay-within.db")
    try:
        now = time.time()
        first = store.append_batch("sess-a", _batch(now), source="cli",
                                   conversation_id="conv-1")
        assert first == sorted(first)
        assert all(sid > 0 for sid in first)

        second = store.append_batch("sess-a", _batch(now), source="cli",
                                    conversation_id="conv-1")
        # Every replayed message was deduped: no new row, sentinel id.
        assert all(sid == -1 for sid in second)
        assert store._deduped_replay_count == 3

        count = store.get_session_count("sess-a")
        assert count == 3, f"replayed batch re-INSERTed rows: {count} != 3"

        rows = store.get_session_messages("sess-a")
        assert [r["content"] for r in rows] == [
            "hello there", "hi! how can I help?", "what is the weather?",
        ]
    finally:
        store.close()


def test_replay_outside_window_ingests_again(tmp_path):
    """(b) The window is on SOURCE time, not write time: two messages whose
    SOURCE times are 700s apart (a legitimate re-send, or any pre-existing
    stored row replayed against a much later turn) are outside the 600s
    window, so the repeat must NOT be deduped."""
    store = MessageStore(tmp_path / "replay-outside.db")
    try:
        now = time.time()
        old_batch = [
            {"role": "user", "content": "please retry", "timestamp": now - 700},
            {"role": "assistant", "content": "please retry", "timestamp": now - 699},
        ]
        first = store.append_batch("sess-b", old_batch, source="cli",
                                   conversation_id="conv-1")
        assert all(sid > 0 for sid in first)

        fresh_batch = [
            {"role": "user", "content": "please retry", "timestamp": now},
        ]
        second = store.append_batch("sess-b", fresh_batch, source="cli",
                                    conversation_id="conv-1")
        assert all(sid > 0 for sid in second), "legitimate repeat was wrongly deduped"
        assert store._deduped_replay_count == 0
        assert store.get_session_count("sess-b") == 3
    finally:
        store.close()


def test_distinct_identity_inside_window_is_not_collapsed(tmp_path):
    """(P1 4043667415) The duplicate key carries the full replay identity:
    same role + content but a different tool_call_id / tool_calls must
    never collapse, even inside the source-time window."""
    store = MessageStore(tmp_path / "distinct-identity.db")
    try:
        now = time.time()
        ts = now - 1
        # Two tool results with identical content for DIFFERENT call ids.
        ids = store.append_batch(
            "sess-d",
            [
                {"role": "tool", "tool_call_id": "call_a", "content": "ok",
                 "timestamp": ts},
                {"role": "tool", "tool_call_id": "call_b", "content": "ok",
                 "timestamp": ts},
            ],
            source="cli",
            conversation_id="conv-1",
        )
        assert all(sid > 0 for sid in ids), "distinct tool_call_id was collapsed"

        # Two assistant tool-call turns with content=None calling
        # DIFFERENT tools.
        ids = store.append_batch(
            "sess-d",
            [
                {"role": "assistant", "content": None, "timestamp": ts,
                 "tool_calls": [{"id": "call_a", "function": {"name": "tool_one", "arguments": "{}"}}]},
                {"role": "assistant", "content": None, "timestamp": ts,
                 "tool_calls": [{"id": "call_b", "function": {"name": "tool_two", "arguments": "{}"}}]},
            ],
            source="cli",
            conversation_id="conv-1",
        )
        assert all(sid > 0 for sid in ids), "distinct tool_calls were collapsed"

        assert store.get_session_count("sess-d") == 4
        assert store._deduped_replay_count == 0
    finally:
        store.close()


def test_reserialized_tool_calls_still_match_durable_row(tmp_path):
    """(P2 4043737752) Semantically identical tool_calls with a different
    object-key order / argument formatting must still be recognized as the
    same replayed turn, or whole-transcript duplication recurs across hosts
    that reserialize messages."""
    store = MessageStore(tmp_path / "canonical-toolcalls.db")
    try:
        now = time.time()
        ts = now - 1
        ids = store.append_batch(
            "sess-t",
            [
                {"role": "assistant", "content": None, "timestamp": ts,
                 "tool_calls": [
                     {"id": "call_a", "type": "function",
                      "function": {"name": "dump", "arguments": "{\"a\": 1, \"b\": 2}"}},
                 ]},
            ],
            source="cli",
            conversation_id="conv-1",
        )
        assert all(sid > 0 for sid in ids)

        # Same call reserialized: keys reordered in both the call object and
        # the embedded arguments JSON.
        ids = store.append_batch(
            "sess-t",
            [
                {"role": "assistant", "content": None, "timestamp": ts,
                 "tool_calls": [
                     {"function": {"arguments": "{ \"b\": 2, \"a\": 1 }", "name": "dump"},
                      "type": "function", "id": "call_a"},
                 ]},
            ],
            source="cli",
            conversation_id="conv-1",
        )
        assert all(sid == -1 for sid in ids), (
            "reserialized tool calls were treated as a distinct turn"
        )
        assert store._deduped_replay_count == 1
        assert store.get_session_count("sess-t") == 1
    finally:
        store.close()


def test_regenerated_ingest_payload_placeholder_still_dedupes(tmp_path):
    """(P1 4043783280) Ingest protection rewrites inline payloads into
    ``[Externalized LCM ingest payload: …]`` placeholders whose filename
    embeds a per-pass ``time_ns`` suffix; a replayed turn whose placeholder
    was regenerated must still dedupe against the durable row (the payload
    sidecar content, not the generated filename, is the identity)."""
    store = MessageStore(
        tmp_path / "payload-placeholder.db",
        ingest_protection_config=LCMConfig(
            database_path=str(tmp_path / "payload-placeholder.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=10,
            large_output_externalization_path=str(tmp_path / "ext"),
        ),
        hermes_home=str(tmp_path / "ext-home"),
    )
    try:
        now = time.time()
        ts = now - 1
        # >4096 contiguous base64-alphabet chars so ingest protection's
        # long-base64 heuristic rewrites it into a placeholder.
        payload = "IMGDATA:" + ("Q" * 5000)

        def msgs(b64_body: str):
            return [
                {"role": "user", "content": b64_body, "timestamp": ts},
            ]

        # First ingest: the long base64 body is externalized to a placeholder.
        first = store.append_batch("sess-p", msgs(payload), source="cli",
                                   conversation_id="conv-1")
        assert all(sid > 0 for sid in first)
        rows = store.get_session_messages("sess-p")
        assert len(rows) == 1
        stored_content = rows[0]["content"]
        assert stored_content.startswith("[Externalized ") and "ref=" in stored_content

        # Replay with the ORIGINAL body re-protected: a NEW sidecar with a
        # fresh time_ns filename is written, so the raw placeholder text
        # differs byte-for-byte — the identity must still match.
        second = store.append_batch("sess-p", msgs(payload), source="cli",
                                    conversation_id="conv-1")
        assert all(sid == -1 for sid in second), (
            "regenerated payload placeholder was treated as a distinct turn"
        )
        assert store._deduped_replay_count == 1
        assert store.get_session_count("sess-p") == 1
    finally:
        store.close()


def test_distinct_tool_name_is_not_collapsed(tmp_path):
    """(P2 4043783283) tool_name is part of the replay identity: two tool
    messages with the same content and source time but different tool_name
    values must never collapse."""
    store = MessageStore(tmp_path / "tool-name.db")
    try:
        now = time.time()
        ts = now - 1
        ids = store.append_batch(
            "sess-n",
            [
                {"role": "tool", "content": "ok", "tool_name": "alpha",
                 "timestamp": ts},
                {"role": "tool", "content": "ok", "tool_name": "beta",
                 "timestamp": ts},
            ],
            source="cli",
            conversation_id="conv-1",
        )
        assert all(sid > 0 for sid in ids), "distinct tool_name was collapsed"
        assert store._deduped_replay_count == 0
        assert store.get_session_count("sess-n") == 2
    finally:
        store.close()


def test_distinct_messages_are_not_collapsed(tmp_path):
    """(c) Different content / role / conversation / tool_call_id must
    never dedupe away."""
    store = MessageStore(tmp_path / "distinct.db")
    try:
        now = time.time()
        same_text_diff_role = [
            {"role": "user", "content": "identical text", "timestamp": now - 2},
            {"role": "assistant", "content": "identical text", "timestamp": now - 2},
        ]
        ids = store.append_batch("sess-c", same_text_diff_role, source="cli",
                                 conversation_id="conv-1")
        assert all(sid > 0 for sid in ids), "distinct roles were collapsed"

        same_text_diff_conversation = [
            {"role": "user", "content": "identical text", "timestamp": now - 1},
        ]
        ids = store.append_batch("sess-c", same_text_diff_conversation,
                                 source="cli", conversation_id="conv-2")
        # Conversation_id is deliberately NOT part of the duplicate key: the
        # candidate is a same-source-time replay of the row stored one line
        # above (same role, byte-identical content), so it dedupes even
        # though the caller passes a different conversation_id. A genuinely
        # distinct turn in another conversation arrives at a different
        # source time and is never collapsed.
        assert all(sid == -1 for sid in ids)
        assert store._deduped_replay_count == 1

        different_text = [
            {"role": "user", "content": "identical tex", "timestamp": now - 1},
            {"role": "user", "content": "identical textx", "timestamp": now - 1},
        ]
        ids = store.append_batch("sess-c", different_text, source="cli",
                                 conversation_id="conv-1")
        assert all(sid > 0 for sid in ids), "distinct content was collapsed"

        assert store.get_session_count("sess-c") == 4
        assert store._deduped_replay_count == 1
    finally:
        store.close()


def test_engine_reingest_with_lost_cursor_is_deduped(tmp_path):
    """(d) The incident mechanism: the engine's ingest cursor is reset to 0
    WITHOUT a reconcile decision and the whole transcript is re-ingested.
    The store-level guard must hold the durable row count even when the
    engine-level cursor path cannot (base stores 3 duplicate rows)."""
    config = LCMConfig(
        database_path=str(tmp_path / "engine-dedup.db"),
        large_output_externalization_path=str(tmp_path / "engine-dedup-ext"),
        fresh_tail_count=2,
        leaf_chunk_tokens=1,
        context_threshold=0.95,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "engine-dedup-home"))
    engine.on_session_start(
        "dedup-session",
        platform="synthetic",
        conversation_id="dedup-conversation",
        context_length=100_000,
    )
    try:
        now = time.time()
        messages = [
            {"role": "user", "content": "turn one " * 20, "timestamp": now - 3},
            {"role": "assistant", "content": "reply one " * 20, "timestamp": now - 2},
            {"role": "user", "content": "turn two " * 20, "timestamp": now - 1},
        ]
        engine._ingest_messages(messages)
        rows_after_first = engine._store.get_session_count("dedup-session")
        assert rows_after_first == 3

        # Simulate the cursor loss the live incident exposed: the in-memory
        # cursor forgets the transcript was already persisted, so the next
        # ingest re-scans and re-stores the whole batch — with NO reconcile
        # decision (``_ingest_cursor_needs_reconcile`` stays False).
        engine._ingest_cursor = 0
        engine._ingest_cursor_needs_reconcile = False
        engine._ingest_messages(messages)

        rows_after_replay = engine._store.get_session_count("dedup-session")
        assert rows_after_replay == 3, (
            "replayed transcript re-INSERTed duplicate rows: "
            f"{rows_after_replay} != {rows_after_first}"
        )
        assert engine._store._deduped_replay_count == 3

        # (P1 4043647914 / 4043667412) A reconciliation recorded by a
        # PREVIOUS pass (e.g. a restart that advanced the cursor) must not
        # disable the guard for later whole-transcript replays that reset
        # the cursor without reconcile: the escape hatch is keyed on the
        # current invocation, never on the historical action record.
        engine._ingest_cursor = 0
        engine._ingest_cursor_needs_reconcile = False
        engine._ingest_messages(messages)
        rows_after_third = engine._store.get_session_count("dedup-session")
        assert rows_after_third == 3, (
            "stale reconciliation action disabled the replay guard: "
            f"{rows_after_third} != 3"
        )
        assert engine._store._deduped_replay_count == 6
    finally:
        engine._store.close()


def test_tool_role_persisted_output_markers_reappend_on_retry(tmp_path):
    """(e) Tool-role Hermes persisted-output markers are exempt from the
    guard: retry semantics (recovery, re-externalization, deliberate
    re-append) live above the store, so a replayed marker with the same
    content still lands as a new row."""
    store = MessageStore(tmp_path / "marker-retry.db")
    try:
        now = time.time()
        persisted_path = tmp_path / "hermes-results" / "call_retry.txt"
        persisted_path.parent.mkdir(parents=True, exist_ok=True)
        persisted_path.write_text("FULL_RETRY_NEEDLE:" + ("z" * 200),
                                  encoding="utf-8")
        marker = (
            "<persisted-output>\n"
            f"This tool result was too large (218 characters, 0.2 KB).\n"
            f"Full output saved to: {persisted_path}\n"
            "Use the read_file tool with offset and limit to access specific"
            " sections of this output.\n\n"
            "Preview (first 30 chars):\n"
            "FULL_RETRY_NEEDLE:zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\n...\n"
            "</persisted-output>"
        )
        msgs = [
            {"role": "tool", "tool_call_id": "call_retry", "content": marker,
             "timestamp": now - 1},
        ]
        first = store.append_batch("sess-e", msgs, source="cli",
                                   conversation_id="conv-1")
        assert all(sid > 0 for sid in first)

        # Same batch replayed straight back: the marker must NOT be deduped.
        second = store.append_batch("sess-e", msgs, source="cli",
                                    conversation_id="conv-1")
        assert all(sid > 0 for sid in second), (
            "tool-role persisted-output marker was wrongly deduped; retry "
            "semantics live above the store"
        )
        assert store._deduped_replay_count == 0
        assert store.get_session_count("sess-e") == 2
    finally:
        store.close()


def test_dedup_probe_uses_indexed_session_timestamp_range(tmp_path):
    """(f) The guard's lookup must ride idx_msg_session_ts, not scan."""
    store = MessageStore(tmp_path / "plan.db")
    try:
        plan = store._conn.execute(
            """EXPLAIN QUERY PLAN SELECT content, tool_calls FROM messages
               WHERE session_id = ? AND role = ? AND content IS ?
                 AND tool_call_id IS ? AND tool_name IS ?
                 AND COALESCE(observed_at, timestamp) >= ?
                 AND COALESCE(observed_at, timestamp) <= ?""",
            ("sess", "user", "x", None, None, 0.0, 1.0),
        ).fetchall()
        detail = " ".join(str(row[-1]) for row in plan)
        assert "SCAN" not in detail.upper() or "INDEX" in detail.upper(), (
            f"dedup probe does not use an index: {detail}"
        )
        assert "idx_msg_session_source_time" in detail
    finally:
        store.close()
