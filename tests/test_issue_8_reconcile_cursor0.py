"""The reconcile-duplication defect invariant tests: reconcile cursor=0 must not re-ingest the transcript.

The live mechanism (session ``the affected live session``): a long-lived Slack
session whose stored tail was legitimately mutated after ingest — tool outputs
cleared to ``[Old tool output cleared to save context space]`` by the host
prompt-side compressor, sensitive-redaction placeholders, externalization
rewrites — reconciled to cursor=0 on EVERY inbound message, because reconcile's
byte-identity match against mutated stored rows can never succeed. The
reconcile-ran escape hatch (the ingest dedupe-replay guard) then stood the store-level dedupe-replay
guard down BY DESIGN and the whole transcript (~549 rows) was re-persisted per
message (``prior_copies=12`` bursts).

Invariants proven here:

(a) A mutated-tail session replaying its full transcript with reconcile-run
    must NOT duplicate rows: either reconcile advances past the replayed
    prefix, or the batch is deferred to the store guard and the replayed rows
    dedupe. Base re-persists the whole mutated transcript (RED);
    the fix dedupes (GREEN).
(b) Legitimate new content arriving after the mutated tail still appends
    exactly once — the ambiguous-delta append path is preserved.
(c) The deliberate-append encodings (reconcile deciding to append after
    matching a prefix) stay green unmodified — encoded by the existing suites
    (test_lcm_engine restart/rebind tests, test_ingest_protection rebind
    tests) plus an explicit aligned-prefix append case here.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from hermes_lcm.config import LCMConfig

from tests.test_tool_contracts import LCMEngine  # host-stub tolerant import


_CLEARED_PLACEHOLDER = "[Old tool output cleared to save context space]"


def _seed_transcript(engine, session_id, *, turns=60, with_tools=True):
    """Build a realistic transcript (user/assistant + tool rows) and ingest it."""
    now = time.time()
    messages = [{"role": "user", "content": "opening question", "timestamp": now - 200}]
    for i in range(turns):
        messages.append({"role": "assistant", "content": f"answer {i} " + "x" * 40,
                         "timestamp": now - 190 + i * 2})
        if with_tools:
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_seed_{i}",
                "tool_name": "terminal",
                "content": '{"output": "' + f"seed output {i} " + "y" * 120 + '", "exit_code": 0}',
                "timestamp": now - 189 + i * 2,
            })
    messages.append({"role": "user", "content": "final seeded question",
                     "timestamp": now - 100})
    engine._ingest_messages(messages)
    assert engine._store.get_session_count(session_id) > 0
    return messages


def _mutate_stored_tail_tools(engine, session_id, *, clear_from=20):
    """Simulate post-ingest mutation: clear stored tool outputs (lossy)."""
    conn = engine._store._conn
    rows = conn.execute(
        """SELECT store_id FROM messages
           WHERE session_id = ? AND role = 'tool'
           ORDER BY store_id""",
        (session_id,),
    ).fetchall()
    mutated = 0
    for (store_id,) in rows[clear_from:]:
        conn.execute(
            "UPDATE messages SET content = ? WHERE store_id = ?",
            (_CLEARED_PLACEHOLDER, store_id),
        )
        mutated += 1
    conn.commit()
    return mutated


def test_mutated_tail_full_replay_does_not_duplicate_rows(tmp_path):
    """(a) The incident: every turn re-persists the whole transcript.

    Base signature: reconcile lands cursor=0 ("persisted ambiguous delta")
    because the mutated stored tail cannot byte-match the replayed prefix, the
    reconcile-ran escape hatch stands the guard down, and the whole
    transcript (with the cleared placeholders now REPLACING original
    content in the durable store) re-appends per turn.
    """
    db_path = str(tmp_path / "mutated-tail-replay.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "mutated-tail-session"
    now = time.time()

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="slack", conversation_id="mutated-tail-conv",
                            context_length=200000)
    try:
        seed = _seed_transcript(before, session_id)
        mutated = _mutate_stored_tail_tools(before, session_id)
        assert mutated > 0
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    # Restart: reconcile armed over the mutated store, full transcript replayed.
    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="slack", conversation_id="mutated-tail-conv",
                           context_length=200000)
    try:
        replay = list(seed)  # the host replays the whole transcript every turn
        replay.append({"role": "user", "content": "fresh turn after restart",
                       "timestamp": now})
        after._ingest_messages(replay)

        rows_after = after._store.get_session_count(session_id)
        # Exactly ONE new durable row (the fresh turn). The whole-transcript
        # replay must not add duplicates even though the stored tool rows no
        # longer byte-match their replayed originals.
        assert rows_after == rows_before + 1, (
            "mutated-tail full replay re-persisted the transcript: "
            f"{rows_after} != {rows_before} + 1"
        )
        contents = [row["content"] for row in
                    after._store.get_session_messages(session_id)]
        assert contents[-1] == "fresh turn after restart"
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()


def test_mutated_tail_replayed_turns_dedupe_across_repeated_passes(tmp_path):
    """(a, repeated) Every subsequent inbound message replays the transcript
    again; each pass must stay idempotent (no per-turn growth)."""
    db_path = str(tmp_path / "mutated-tail-idempotent.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "mutated-tail-idempotent-session"
    now = time.time()

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="slack", conversation_id="mutated-tail-idempotent-conv",
                            context_length=200000)
    try:
        seed = _seed_transcript(before, session_id)
        _mutate_stored_tail_tools(before, session_id)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="slack", conversation_id="mutated-tail-idempotent-conv",
                           context_length=200000)
    try:
        # The gateway replays the CUMULATIVE transcript each turn: prior
        # fresh turns stay in the list and each pass appends one more.
        cumulative = list(seed)
        counts = []
        for turn in range(3):
            cumulative.append({"role": "user", "content": f"fresh per-turn message {turn}",
                               "timestamp": now + turn})
            after._ingest_messages(cumulative)
            counts.append(after._store.get_session_count(session_id))
        # Each pass adds exactly its one fresh message — the replayed
        # transcript dedupes every time.
        assert counts == [rows_before + 1, rows_before + 2, rows_before + 3], (
            f"per-turn replay duplicated rows: {counts} (baseline {rows_before})"
        )
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()


def test_legitimate_new_turns_after_mutated_tail_still_append(tmp_path):
    """(b) The ambiguous-delta append path must keep working for genuinely new
    content over a mutated tail."""
    db_path = str(tmp_path / "mutated-tail-append.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "mutated-tail-append-session"
    now = time.time()

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="slack", conversation_id="mutated-tail-append-conv",
                            context_length=200000)
    try:
        _seed_transcript(before, session_id)
        _mutate_stored_tail_tools(before, session_id)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="slack", conversation_id="mutated-tail-append-conv",
                           context_length=200000)
    try:
        # A fresh, standalone delta with NO replayed prefix at all — exactly
        # the legitimate case the ambiguous-delta decision exists for.
        fresh_delta = [
            {"role": "user", "content": "legitimate standalone delta", "timestamp": now},
            {"role": "assistant", "content": "legitimate standalone answer", "timestamp": now + 1},
        ]
        after._ingest_messages(fresh_delta)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before + len(fresh_delta)
        contents = [row["content"] for row in
                    after._store.get_session_messages(session_id)]
        assert contents[-2:] == [
            "legitimate standalone delta",
            "legitimate standalone answer",
        ]
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()


def test_new_content_after_full_replay_over_mutated_tail_appends(tmp_path):
    """(b, combined) Replay + fresh suffix in one batch: the replayed portion
    must dedupe AND the fresh suffix must append exactly once."""
    db_path = str(tmp_path / "mutated-tail-replay-plus-delta.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "mutated-tail-replay-plus-delta-session"
    now = time.time()

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="slack", conversation_id="mutated-tail-rpd-conv",
                            context_length=200000)
    try:
        seed = _seed_transcript(before, session_id)
        _mutate_stored_tail_tools(before, session_id)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="slack", conversation_id="mutated-tail-rpd-conv",
                           context_length=200000)
    try:
        batch = list(seed)
        batch.extend([
            {"role": "assistant", "content": "brand new answer", "timestamp": now},
            {"role": "user", "content": "brand new follow-up", "timestamp": now + 1},
        ])
        after._ingest_messages(batch)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before + 2, (
            f"replay+delta appended the wrong number of rows: {rows_after} != {rows_before} + 2"
        )
        contents = [row["content"] for row in
                    after._store.get_session_messages(session_id)]
        assert contents[-2:] == ["brand new answer", "brand new follow-up"]
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()


def test_aligned_prefix_append_after_partial_match_is_preserved(tmp_path):
    """(c) Deliberate-append semantics: when reconcile matches a real prefix
    and only the tail is new, the tail appends (cursor advances) — the escape
    hatch's motivating case must stay green."""
    db_path = str(tmp_path / "aligned-prefix-append.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "aligned-prefix-append-session"
    now = time.time()

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="slack", conversation_id="aligned-prefix-conv",
                            context_length=200000)
    try:
        seed = _seed_transcript(before, session_id)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="slack", conversation_id="aligned-prefix-conv",
                           context_length=200000)
    try:
        # Replay the WHOLE seed (byte-identical, unmutated) plus two new turns:
        # reconcile should match the replayed prefix and advance the cursor,
        # appending ONLY the new turns.
        batch = list(seed)
        batch.extend([
            {"role": "user", "content": "post-restart question", "timestamp": now},
            {"role": "assistant", "content": "post-restart answer", "timestamp": now + 1},
        ])
        after._ingest_messages(batch)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before + 2, (
            f"aligned-prefix replay appended the wrong rows: {rows_after} != {rows_before} + 2"
        )
        contents = [row["content"] for row in
                    after._store.get_session_messages(session_id)]
        assert contents[-2:] == ["post-restart question", "post-restart answer"]
        reconciliation = after._last_ingest_reconciliation
        assert reconciliation["action"] == "advanced cursor", (
            f"aligned replay should advance the cursor, got {reconciliation}"
        )
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()
def test_duplicate_tail_copies_from_prior_bursts_do_not_starve_alignment(tmp_path):
    """review finding: a store already damaged by prior whole-transcript
    bursts holds duplicate copies of every turn; alignment coverage must walk
    DISTINCT stored identities (duplicates neither consume incoming
    occurrences nor count as coverage slots), or `covered == stored_total`
    can never hold on exactly the sessions this fix targets and the
    ambiguous-delta append — the bug — stays in charge."""
    db_path = str(tmp_path / "dup-damaged-replay.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "dup-damaged-session"
    now = time.time()

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="slack", conversation_id="dup-damaged-conv",
                            context_length=200000)
    try:
        seed = _seed_transcript(before, session_id, turns=20)
        # Simulate two prior whole-transcript bursts: byte-identical copies
        # appended straight to the store (the reconcile-ran escape hatch did
        # exactly this on base — the guard was stood down by design).
        import copy as _copy
        store = before._store
        for _burst in range(2):
            store.append_batch(
                session_id,
                _copy.deepcopy(seed),
                source="slack",
                conversation_id="dup-damaged-conv",
                dedupe_replay=False,
            )
        rows_before = before._store.get_session_count(session_id)
        assert rows_before == 3 * len(seed), (
            f"setup: expected 3x copies, got {rows_before} vs {len(seed)}"
        )
        # Host-side prune of the stored tool outputs (tail portion).
        _mutate_stored_tail_tools(before, session_id, clear_from=20)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="slack", conversation_id="dup-damaged-conv",
                           context_length=200000)
    try:
        cumulative = list(seed)
        counts = []
        for turn in range(2):
            cumulative.append({"role": "user", "content": f"fresh turn {turn}",
                               "timestamp": now + turn})
            after._ingest_messages(cumulative)
            counts.append(after._store.get_session_count(session_id))
        assert counts == [rows_before + 1, rows_before + 2], (
            f"dup-damaged full replay re-persisted the transcript: {counts} (baseline {rows_before})"
        )
        contents = [row["content"] for row in
                    after._store.get_session_messages(session_id)]
        assert contents[-2:] == ["fresh turn 0", "fresh turn 1"]
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()
def test_within_burst_repeats_of_covered_turns_are_masked(tmp_path):
    """Live-session variant (the affected live session): a damaged in-memory
    transcript repeats the same turn MANY times within ONE replay batch, and
    those rows carry no observed_at, so the store guard cannot dedupe them.
    When the defer verdict fires, every incoming occurrence of a covered
    turn is masked — not just the alignment-consumed one — or the burst
    re-persists through the guard."""
    db_path = str(tmp_path / "within-burst-repeats.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "within-burst-repeats-session"
    now = time.time()

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="slack", conversation_id="wbr-conv",
                            context_length=200000)
    try:
        seed = _seed_transcript(before, session_id, turns=20)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="slack", conversation_id="wbr-conv",
                           context_length=200000)
    try:
        # A damaged transcript: the seed replayed THREE times inside one
        # batch, plus one genuinely new turn at the end.
        batch = list(seed) + list(seed) + list(seed)
        batch.append({"role": "user", "content": "single fresh turn",
                      "timestamp": now})
        after._ingest_messages(batch)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before + 1, (
            f"within-burst repeats re-persisted the transcript: "
            f"{rows_after} != {rows_before} + 1"
        )
        contents = [row["content"] for row in
                    after._store.get_session_messages(session_id)]
        assert contents[-1] == "single fresh turn"
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()
def test_fresh_turn_reusing_identity_after_replay_is_not_masked(tmp_path):
    """review this fix round: a restart batch = durable transcript + a
    genuinely new turn that REUSES a stored identity (same role/content,
    e.g. a user re-sending an earlier prompt with a FRESH source
    timestamp). The surplus-occurrence mask must not eat it: a surplus
    copy carrying a source time LATER than the stored copy's is new work."""
    db_path = str(tmp_path / "identity-reuse-fresh.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "identity-reuse-fresh-session"
    now = time.time()

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="slack", conversation_id="irf-conv",
                            context_length=200000)
    try:
        seed = _seed_transcript(before, session_id, turns=20)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="slack", conversation_id="irf-conv",
                           context_length=200000)
    try:
        # Full transcript replay + a new user turn that reuses the OPENING
        # prompt's content with a FRESH timestamp (the user re-sent it).
        batch = list(seed)
        batch.append({"role": "user", "content": "opening question",
                      "timestamp": now + 50})
        after._ingest_messages(batch)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before + 1, (
            f"fresh identity-reuse turn was lost: {rows_after} != {rows_before} + 1"
        )
        rows = after._store.get_session_messages(session_id)
        assert rows[-1]["content"] == "opening question"
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()


def test_fresh_retry_sharing_mutated_tool_call_id_is_stored(tmp_path):
    """review this fix round: a stored tool row whose content was replaced by
    a cleared placeholder, then a replay carrying the ORIGINAL result
    followed by a genuinely NEW retry with the SAME tool_call_id but
    distinct content. The mutated fallback must mask only the occurrence
    it consumed — the new retry is new work and must store."""
    db_path = str(tmp_path / "mutated-tcid-retry.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "mutated-tcid-retry-session"
    now = time.time()

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="slack", conversation_id="mtc-conv",
                            context_length=200000)
    try:
        _seed_transcript(before, session_id, turns=20)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="slack", conversation_id="mtc-conv",
                           context_length=200000)
    try:
        # Full transcript replay + a fresh retry of the LAST tool call with
        # distinct content, same tool_call_id. Rebuild the transcript the
        # same way _seed_transcript does (its return value IS the batch).
        # NOTE: _seed_transcript ingests AND returns; here the store already
        # has it, so just reconstruct the message list shape.
        now0 = now - 200
        batch = [{"role": "user", "content": "opening question", "timestamp": now0}]
        for i in range(20):
            batch.append({"role": "assistant", "content": f"answer {i} " + "x" * 40,
                          "timestamp": now - 190 + i * 2})
            batch.append({
                "role": "tool",
                "tool_call_id": f"call_seed_{i}",
                "tool_name": "terminal",
                "content": '{"output": "' + f"seed output {i} " + "y" * 120 + '", "exit_code": 0}',
                "timestamp": now - 189 + i * 2,
            })
        batch.append({"role": "user", "content": "final seeded question",
                      "timestamp": now - 100})
        retry = {
            "role": "tool",
            "tool_call_id": "call_seed_19",
            "tool_name": "terminal",
            "content": '{"output": "retry with fresh content", "exit_code": 0}',
            "timestamp": now,
        }
        batch.append(retry)
        after._ingest_messages(batch)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before + 1, (
            f"fresh retry sharing a mutated tool_call_id was lost: "
            f"{rows_after} != {rows_before} + 1"
        )
        rows = after._store.get_session_messages(session_id)
        assert "retry with fresh content" in rows[-1]["content"]
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()
def test_new_tool_call_turn_not_masked_by_covered_empty_assistant(tmp_path):
    """review this fix round 2: an empty-content assistant WITH tool_calls and a
    call-less empty assistant share (role, '', '') — without the canonical
    tool_calls component in the alignment identity, a covered empty
    assistant masks a genuinely new tool-call turn. The identity now carries
    the canonical tool_calls, so the new turn stores."""
    db_path = str(tmp_path / "toolcall-identity-sep.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "toolcall-identity-sep-session"
    now = time.time()

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="slack", conversation_id="tis-conv",
                            context_length=200000)
    try:
        _seed_transcript(before, session_id, turns=20)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="slack", conversation_id="tis-conv",
                           context_length=200000)
    try:
        # The transcript contains call-less empty assistant rows (from
        # _seed_transcript) — a full replay plus a NEW assistant turn with
        # content=None and a tool_call. Without the tool_calls component
        # the new turn shares the empty assistant's identity and gets
        # masked (lost); with it, the identity is distinct and the row
        # stores.
        seed = []
        seed.append({"role": "user", "content": "opening question", "timestamp": now - 200})
        for i in range(20):
            seed.append({"role": "assistant", "content": f"answer {i} " + "x" * 40,
                         "timestamp": now - 190 + i * 2})
            seed.append({
                "role": "tool",
                "tool_call_id": f"call_seed_{i}",
                "tool_name": "terminal",
                "content": '{"output": "' + f"seed output {i} " + "y" * 120 + '", "exit_code": 0}',
                "timestamp": now - 189 + i * 2,
            })
        seed.append({"role": "user", "content": "final seeded question",
                     "timestamp": now - 100})
        batch = list(seed)
        batch.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_new", "type": "function",
                            "function": {"name": "search", "arguments": "{}"}}],
            "timestamp": now,
        })
        after._ingest_messages(batch)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before + 1, (
            f"new tool-call assistant turn was masked away: {rows_after} != {rows_before} + 1"
        )
        rows = after._store.get_session_messages(session_id)
        assert rows[-1]["tool_calls"] is not None
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()
def test_surplus_mask_handles_iso8601_timestamps(tmp_path):
    """review this fix round 3: a surplus occurrence carrying a timezone-aware
    ISO-8601 timestamp must not crash the alignment (float() ValueError);
    the timestamp is normalized through the same contract the store uses
    (_normalize_observed_at)."""
    db_path = str(tmp_path / "iso-timestamp-replay.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "iso-timestamp-replay-session"

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="slack", conversation_id="iso-conv",
                            context_length=200000)
    try:
        seed = _seed_transcript(before, session_id, turns=10)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="slack", conversation_id="iso-conv",
                           context_length=200000)
    try:
        # Replay with the transcript's timestamps in ISO-8601 form.
        batch = []
        for msg in seed:
            m = dict(msg)
            ts = m.get("timestamp")
            if ts is not None:
                m["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            batch.append(m)
        after._ingest_messages(batch)  # must not raise

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before, (
            f"ISO-timestamp replay re-persisted rows: {rows_after} != {rows_before}"
        )
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()
def test_fresh_repeats_nearly_covering_stored_turns_are_not_masked(tmp_path):
    """review this fix round 5: a genuinely fresh delta that re-sends 9 of 10
    stored turns with FRESH source timestamps plus one new turn must append
    everything — a fresh-timestamp repeat is a deliberate re-send, not a
    replayed occurrence, so coverage stays below the defer bars."""
    db_path = str(tmp_path / "fresh-repeat-delta.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "fresh-repeat-delta-session"
    now = time.time()

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="slack", conversation_id="frd-conv",
                            context_length=200000)
    try:
        seed = _seed_transcript(before, session_id, turns=10)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="slack", conversation_id="frd-conv",
                           context_length=200000)
    try:
        # Re-send 9 of the seed's messages (all but the last) with FRESH
        # timestamps, plus one new turn.
        batch = [dict(m, timestamp=(m.get("timestamp") or now - 50) + 10000)
                 for m in seed[:-1]]
        batch.append({"role": "user", "content": "the one new turn",
                      "timestamp": now + 20000})
        after._ingest_messages(batch)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before + len(batch), (
            f"fresh repeats were dropped as replay: {rows_after} != {rows_before} + {len(batch)}"
        )
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()
def test_stale_system_prefix_snapshot_still_bails_to_no_overlap(tmp_path):
    """review this fix: shrinking the alignment window must not shrink the
    stale-snapshot proof window — a short system-leading stale prefix from
    session start must still hit the stale-no-overlap bail-out (skip the
    prefix, don't re-append it) even when the session has grown far beyond
    len(messages)."""
    db_path = str(tmp_path / "stale-prefix-wide-window.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "stale-prefix-wide-session"

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="cli", conversation_id="spww-conv",
                            context_length=200000)
    try:
        persisted = [{"role": "system", "content": "You are concise."}]
        persisted.extend({"role": "user", "content": f"durable message {i}"}
                         for i in range(200))
        before._ingest_messages(persisted)
        rows_before = before._store.get_session_count(session_id)
        assert rows_before == len(persisted)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="cli", conversation_id="spww-conv",
                           context_length=200000)
    try:
        # A stale runtime snapshot: system row + first few durable rows —
        # much shorter than the session, replayed after restart.
        stale_snapshot = persisted[:5]
        after._ingest_messages(stale_snapshot)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before, (
            f"stale prefix snapshot re-appended: {rows_after} != {rows_before}"
        )
        reconciliation = after._last_ingest_reconciliation
        assert reconciliation["reason"] == "skipped stale no-overlap snapshot", (
            f"stale-prefix bail-out lost: {reconciliation}"
        )
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()
def test_compaction_reset_full_replay_defers_to_alignment(tmp_path):
    """The reconcile-duplication defect live follow-up: after a compaction reset the cursor is 0 AND
    needs_reconcile is False - reconcile never runs, the alignment never
    ran, and the store guard cannot arbitrate the no-observed_at majority of
    a whole-transcript replay. The no-reconcile cursor=0 path must run the
    same alignment+defer gate (measured live: 1814-row bursts continued
    after every gateway restart until this path was covered)."""
    db_path = str(tmp_path / "compaction-reset-replay.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "compaction-reset-replay-session"
    now = time.time()

    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(session_id, platform="slack", conversation_id="crr-conv",
                            context_length=200000)
    try:
        seed = _seed_transcript(engine, session_id, turns=20)
        rows_before = engine._store.get_session_count(session_id)
        # Simulate the compaction reset: cursor to 0 WITHOUT re-arming
        # reconcile (exactly what _reset_compaction_progress does).
        engine._ingest_cursor = 0
        engine._ingest_cursor_needs_reconcile = False

        cumulative = list(seed)
        cumulative.append({"role": "user", "content": "post-reset fresh turn",
                           "timestamp": now})
        engine._ingest_messages(cumulative)

        rows_after = engine._store.get_session_count(session_id)
        assert rows_after == rows_before + 1, (
            f"post-reset whole-transcript replay re-persisted the batch: "
            f"{rows_after} != {rows_before} + 1"
        )
        # Either mechanism satisfies the invariant: the alignment deferred
        # (mask dropped the replayed majority), or the store guard deduped
        # the timestamp-carrying rows. The row-count assertion above is the
        # contract; this just records which mechanism fired.
        if engine._last_ingest_reconciliation["action"] != "deferred to replay guard":
            assert engine._store._deduped_replay_count >= len(seed) - 1, (
                f"neither the alignment nor the guard deduped the replay: "
                f"{engine._last_ingest_reconciliation} / "
                f"{engine._store._deduped_replay_count}"
            )
    finally:
        engine._store.close()
        engine._dag.close()
        engine._lifecycle.close()
def test_reordered_untimestamped_small_batch_appends(tmp_path):
    """review finding: a deliberately REORDERED untimestamped batch whose
    identities already exist in the store is fresh content, not a replay -
    the order-consistency gate (strictly proportional tolerance) must fail
    the defer and append it."""
    db_path = str(tmp_path / "reordered-small-batch.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "reordered-small-batch-session"

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="cli", conversation_id="rsb-conv",
                            context_length=200000)
    try:
        _seed_transcript(before, session_id, turns=10)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="cli", conversation_id="rsb-conv",
                           context_length=200000)
    try:
        # Reordered delta: the seed's LAST three messages in REVERSE order,
        # no timestamps (indistinguishable from replay by time - only
        # order distinguishes them).
        batch = [
            {"role": "user", "content": "final seeded question"},
            {"role": "tool", "tool_call_id": "call_seed_9", "tool_name": "terminal",
             "content": '{"output": "' + "seed output 9 " + "y" * 120 + '", "exit_code": 0}'},
            {"role": "assistant", "content": "answer 9 " + "x" * 40},
        ]
        # Force the no-reconcile cursor=0 path.
        after._ingest_cursor = 0
        after._ingest_cursor_needs_reconcile = False
        after._ingest_messages(batch)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before + len(batch), (
            f"reordered untimestamped batch was discarded as replay: "
            f"{rows_after} != {rows_before} + {len(batch)}"
        )
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()
def test_ignored_row_does_not_inflate_small_batch_gate(tmp_path):
    """review finding: a 2-row deliberate-repeat batch plus one
    ignore-pattern row must take the lossless small-batch path (the gate
    counts EFFECTIVE rows) - the two repeats append."""
    pytest.importorskip("regex", reason="ignore_message_patterns needs the regex module")
    db_path = str(tmp_path / "ignored-inflate.db")
    config = LCMConfig(
        database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100,
        ignore_message_patterns=["^SECRET"],
    )
    session_id = "ignored-inflate-session"

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="cli", conversation_id="ii-conv",
                            context_length=200000)
    try:
        _seed_transcript(before, session_id, turns=10)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="cli", conversation_id="ii-conv",
                           context_length=200000)
    try:
        # 2 effective rows (untimestamped deliberate repeats of the tail -
        # guard-powerless, exactly the review shape) + 1 ignored row.
        batch = [
            {"role": "user", "content": "final seeded question"},
            {"role": "assistant", "content": "answer 9 " + "x" * 40},
            {"role": "user", "content": "SECRET ignored row"},
        ]
        after._ingest_cursor = 0
        after._ingest_cursor_needs_reconcile = False
        after._ingest_messages(batch)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before + 2, (
            f"ignored row inflated the gate and dropped the repeats: "
            f"{rows_after} != {rows_before} + 2"
        )
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()


def test_single_inversion_small_reordered_batch_appends(tmp_path):
    """review finding: a 3-row store replayed as A,C,B (one inversion) is
    reordered fresh content - small batches tolerate ZERO inversions."""
    db_path = str(tmp_path / "single-inversion.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "single-inversion-session"

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="cli", conversation_id="siv-conv",
                            context_length=200000)
    try:
        _seed_transcript(before, session_id, turns=10)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="cli", conversation_id="siv-conv",
                           context_length=200000)
    try:
        # A,C,B (one inversion vs the stored A,B,C order), untimestamped.
        batch = [
            {"role": "assistant", "content": "answer 9 " + "x" * 40},
            {"role": "user", "content": "final seeded question"},
            {"role": "tool", "tool_call_id": "call_seed_9", "tool_name": "terminal",
             "content": '{"output": "' + "seed output 9 " + "y" * 120 + '", "exit_code": 0}'},
        ]
        after._ingest_cursor = 0
        after._ingest_cursor_needs_reconcile = False
        after._ingest_messages(batch)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before + len(batch), (
            f"single-inversion batch was discarded as replay: "
            f"{rows_after} != {rows_before} + {len(batch)}"
        )
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()


def test_untimestamped_retry_sharing_mutated_call_id_stores(tmp_path):
    """review finding: a stored tool row replaced by the cleared-output
    placeholder, replay carrying the original + a new UNTIMESTAMPED retry
    with the same tool_call_id. The mutated fallback masks exactly ONE
    occurrence (the replayed original); the retry stores."""
    db_path = str(tmp_path / "mutated-untimestamped-retry.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "mutated-untimestamped-retry-session"

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="cli", conversation_id="mur-conv",
                            context_length=200000)
    try:
        _seed_transcript(before, session_id, turns=10)
        # Mutate the stored copy of the LAST tool row.
        conn = before._store._conn
        last_tool = conn.execute(
            "SELECT store_id FROM messages WHERE session_id=? AND role='tool' ORDER BY store_id DESC LIMIT 1",
            (session_id,),
        ).fetchone()[0]
        conn.execute("UPDATE messages SET content=? WHERE store_id=?",
                     (_CLEARED_PLACEHOLDER, last_tool))
        conn.commit()
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="cli", conversation_id="mur-conv",
                           context_length=200000)
    try:
        batch = []
        batch.append({"role": "user", "content": "opening question"})
        for i in range(10):
            batch.append({"role": "assistant", "content": f"answer {i} " + "x" * 40})
            batch.append({
                "role": "tool", "tool_call_id": f"call_seed_{i}", "tool_name": "terminal",
                "content": '{"output": "' + f"seed output {i} " + "y" * 120 + '", "exit_code": 0}',
            })
        batch.append({"role": "user", "content": "final seeded question"})
        # The new untimestamped retry sharing the mutated call id.
        batch.append({"role": "tool", "tool_call_id": "call_seed_9", "tool_name": "terminal",
                      "content": '{"output": "retry fresh content", "exit_code": 0}'})
        after._ingest_cursor = 0
        after._ingest_cursor_needs_reconcile = False
        after._ingest_messages(batch)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before + 1, (
            f"untimestamped retry sharing a mutated call id was dropped: "
            f"{rows_after} != {rows_before} + 1"
        )
        rows = after._store.get_session_messages(session_id)
        assert "retry fresh content" in rows[-1]["content"]
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()
def test_scaffold_prefix_advance_with_full_replay_defers(tmp_path):
    """Live 21:56 burst (session the affected live session): reconcile advanced
    cursor=1 past one scaffold row on a 2147-row whole-transcript replay,
    the mask stood down (scaffold-only-prefix path), the guard-off append
    re-persisted everything. A scaffold-prefix advance whose post-cursor
    region provably replays the durable tail must apply the mask."""
    db_path = str(tmp_path / "scaffold-full-replay.db")
    config = LCMConfig(database_path=db_path, fresh_tail_count=4, leaf_chunk_tokens=100)
    session_id = "scaffold-full-replay-session"
    now = time.time()

    before = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    before.on_session_start(session_id, platform="slack", conversation_id="sfr-conv",
                            context_length=200000)
    try:
        seed = _seed_transcript(before, session_id, turns=60)
        rows_before = before._store.get_session_count(session_id)
    finally:
        before._store.close()
        before._dag.close()
        before._lifecycle.close()

    after = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    after.on_session_start(session_id, platform="slack", conversation_id="sfr-conv",
                           context_length=200000)
    try:
        # The live shape: an LCM scaffold system row + the full transcript
        # + a fresh turn. Reconcile matches the scaffold row as a
        # 1-row "prefix" and advances.
        batch = [
            {"role": "system",
             "content": "[Note: This conversation uses Lossless Context Management (LCM). Earlier turns have been compacted into hierarchical summaries below.]"},
        ]
        batch.extend(seed)
        batch.append({"role": "user", "content": "fresh turn after scaffold",
                      "timestamp": now})
        after._ingest_messages(batch)

        rows_after = after._store.get_session_count(session_id)
        assert rows_after == rows_before + 1, (
            f"scaffold-prefix full replay re-persisted the batch: "
            f"{rows_after} != {rows_before} + 1"
        )
        # Either mechanism satisfies the invariant: a full-replay advance
        # (cursor past the whole transcript) or the scaffold-prefix path
        # with the alignment gate deferring the post-cursor region.
        reconciliation = after._last_ingest_reconciliation
        assert reconciliation["action"] == "advanced cursor", (
            f"expected a cursor advance: {reconciliation}"
        )
    finally:
        after._store.close()
        after._dag.close()
        after._lifecycle.close()
