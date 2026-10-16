"""Regression: a compaction must not move the ingest cursor onto a foreign list.

``compaction.py`` used to set ``self._ingest_cursor = len(<the list it returns>)``
in four places.  The cursor is a position inside the list that is ingested
*next*, and the host continues from its own history — a different, usually
longer list.  The too-short cursor then re-appended every already-stored message
between the returned length and the incoming length, once per turn.

Each site gets its own test, because a shared code shape does not imply shared
state conditions.  The sites, by the path that reaches them:

* ``compaction.py`` leaf compaction       -> status ``compacted``
* ``compaction.py`` boundary-cooldown     -> status ``sanitized`` (early return)
* ``compaction.py`` sanitized branch      -> status ``sanitized`` (no leaf node)
* ``compaction.py`` dropped-scaffold      -> status ``noop`` — see the note at the
  end of the file: not reachable from this harness, and why.

The contract asserted for every site:

1. the cursor is not rewritten from the returned list,
2. the reconciliation is armed so the position is re-derived from the store,
3. re-ingesting the same list inserts nothing,
4. ingesting the list plus a new message does not lose that message.
"""
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "regression-compaction-cursor"


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def _engine(
    tmp_path,
    name,
    *,
    threshold=0.10,
    context_length=1_000,
    fresh_tail=2,
    leaf_chunk=20,
):
    config = LCMConfig(
        database_path=str(tmp_path / f"{name}.db"),
        fresh_tail_count=fresh_tail,
        leaf_chunk_tokens=leaf_chunk,
        context_threshold=threshold,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        SESSION,
        conversation_id=f"conversation:{SESSION}",
        platform="test",
        context_length=context_length,
    )
    return engine


def _history(count=12):
    """A history whose messages each carry a distinct replay identity."""
    msgs = [{"role": "system", "content": "system prompt"}]
    for i in range(count):
        if i % 3 == 0:
            msgs.append({"role": "user", "content": f"FRAGE-{i:03d}: Sachstand {i}."})
        elif i % 3 == 1:
            msgs.append({"role": "assistant", "content": f"ANTWORT-{i:03d}: {i} Kartons."})
        else:
            msgs.append(
                {
                    "role": "tool",
                    "tool_name": "terminal",
                    "tool_call_id": f"call_rc_{i:03d}",
                    "content": f"ROT-{i:03d}: 41 Kartons, Siegel unbeschaedigt.",
                }
            )
    return msgs


def _new_message(n=1):
    """A genuinely new event, distinguishable from every stored identity."""
    return [
        {
            "role": "user",
            "content": f"NEU-{i:03d}: Der Wagen steht bereit.",
            "tool_call_id": "",
        }
        for i in range(n)
    ]


def _rows(engine):
    return engine._store.get_session_count(SESSION)


def _duplicates(engine):
    conn = engine._store._conn
    return conn.execute(
        """
        SELECT count(*) FROM (
            SELECT content FROM messages
            WHERE session_id = ? AND tool_call_id IS NOT NULL AND tool_call_id != ''
            GROUP BY tool_call_id HAVING count(*) > 1
        )
        """,
        (SESSION,),
    ).fetchone()[0]


def _assert_handover_is_safe(engine, messages, *, cursor_before, label):
    """The four-part contract, applied after a compaction path has run."""
    assert engine._ingest_cursor_needs_reconcile is True, (
        f"{label}: reconciliation must be armed — the position has to be "
        "re-derived from the store, not read off the returned list"
    )
    assert engine._ingest_cursor == cursor_before, (
        f"{label}: the cursor must not be rewritten from a foreign list"
    )

    before = _rows(engine)
    engine.ingest(messages)
    assert _rows(engine) == before, f"{label}: re-ingesting the same list must insert nothing"
    assert _duplicates(engine) == 0, f"{label}: no duplicate tool results expected"

    before = _rows(engine)
    engine.ingest(list(messages) + _new_message())
    assert _rows(engine) == before + 1, f"{label}: a genuinely new message must be stored"
    assert _duplicates(engine) == 0, f"{label}: no duplicate tool results expected"


# --------------------------------------------------------------------------
# site 1 — leaf compaction (compaction.py, the "compacted" path)
# --------------------------------------------------------------------------
def test_leaf_compaction_does_not_move_the_cursor(tmp_path, monkeypatch):
    """A leaf compaction returns a short list; it must not become the position."""
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("leaf summary", 0),
    )
    engine = _engine(tmp_path, "leaf")
    try:
        messages = _history()
        engine.ingest(messages)
        cursor_before = engine._ingest_cursor

        out = engine.compress(messages, current_tokens=engine.threshold_tokens + 1)
        assert engine._last_compression_status == "compacted", "path not reached"
        assert out != messages, "the leaf path must shorten the context"

        _assert_handover_is_safe(
            engine, messages, cursor_before=cursor_before, label="leaf compaction"
        )
    finally:
        engine.shutdown()


# --------------------------------------------------------------------------
# site 2 — boundary-cooldown cleanup (compaction.py, early return)
# --------------------------------------------------------------------------
def test_boundary_cooldown_cleanup_does_not_move_the_cursor(tmp_path):
    """The cooldown cleanup returns a sanitized list; it must not set the position."""
    engine = _engine(tmp_path, "cooldown")
    try:
        messages = _history()
        engine.ingest(messages)
        cursor_before = engine._ingest_cursor

        engine._preflight_cleanup_only_due_to_boundary_cooldown = True
        out = engine.compress(messages, current_tokens=1_000)
        assert engine._last_compression_status == "sanitized", "path not reached"
        assert out != messages, "the cleanup must change the returned list"

        _assert_handover_is_safe(
            engine, messages, cursor_before=cursor_before, label="boundary cooldown"
        )
    finally:
        engine.shutdown()


# --------------------------------------------------------------------------
# site 3 — sanitized branch without a new leaf node
# --------------------------------------------------------------------------
def test_sanitized_branch_does_not_move_the_cursor(tmp_path, monkeypatch):
    """Sanitizing a context without writing a leaf node must not set the position."""
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("sanitized summary", 0),
    )
    engine = _engine(
        tmp_path,
        "sanitized",
        threshold=0.90,
        context_length=1_000_000,
        fresh_tail=8,
        leaf_chunk=20_000,
    )
    try:
        messages = _history()
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_rc_dangling",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            }
        )
        engine.ingest(messages)
        cursor_before = engine._ingest_cursor

        out = engine.compress(messages, current_tokens=1_000)
        assert engine._last_compression_status == "sanitized", "path not reached"
        assert out != messages, "sanitizing must change the returned list"

        _assert_handover_is_safe(
            engine, messages, cursor_before=cursor_before, label="sanitized branch"
        )
    finally:
        engine.shutdown()


# --------------------------------------------------------------------------
# both failure directions, beyond the per-site contract
# --------------------------------------------------------------------------
def test_repeated_text_as_a_new_event_is_not_suppressed(tmp_path, monkeypatch):
    """A message that repeats stored text word for word is still a new event."""
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("leaf summary", 0),
    )
    engine = _engine(tmp_path, "repeat")
    try:
        messages = _history()
        engine.ingest(messages)
        engine.compress(messages, current_tokens=engine.threshold_tokens + 1)

        before = _rows(engine)
        engine.ingest(messages)
        assert _rows(engine) == before, "the unchanged history must not be re-stored"

        # The same user text, twice more: genuine repeats, not a replay.
        repeat = {"role": "user", "content": "FRAGE-006: Sachstand 6."}
        engine.ingest(list(messages) + [repeat, repeat])
        assert _rows(engine) == before + 2, "genuine repeats must not be suppressed"
    finally:
        engine.shutdown()


def test_restart_during_the_transition_keeps_new_messages(tmp_path, monkeypatch):
    """A compaction followed by a restart must neither duplicate nor lose."""
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("leaf summary", 0),
    )
    engine = _engine(tmp_path, "restart")
    messages = _history()
    engine.ingest(messages)
    engine.compress(messages, current_tokens=engine.threshold_tokens + 1)
    rows_before = _rows(engine)
    engine.shutdown()

    # Restart: a fresh engine on the same database, bound to the same session.
    revived = _engine(tmp_path, "restart")
    try:
        revived.ingest(messages)
        assert _rows(revived) == rows_before, "restart must not duplicate the history"

        revived.ingest(list(messages) + _new_message())
        assert _rows(revived) == rows_before + 1, "restart must not lose a new message"
    finally:
        revived.shutdown()


def test_unreconstructible_history_is_persisted_not_skipped(tmp_path):
    """When the store proves nothing, the safe direction is to persist again.

    This is the documented limit of a positional cursor: if the stored tail
    cannot be matched, reconciliation returns 0 (``persisted ambiguous delta``)
    rather than guessing a skip.  Re-inserting is recoverable; skipping is not.

    All four values below are measured, not assumed.
    """
    engine = _engine(
        tmp_path,
        "ambiguous",
        threshold=0.90,
        context_length=1_000_000,
        fresh_tail=8,
        leaf_chunk=20_000,
    )
    try:
        stored = _history(6)
        engine.ingest(stored)

        # provable: the list is what the store already holds
        assert engine._reconcile_ingest_cursor_from_store(list(stored)) == len(stored)
        # provable: the same list plus one genuinely new message
        assert engine._reconcile_ingest_cursor_from_store(
            list(stored) + _new_message()
        ) == len(stored)

        # unprovable: a rewritten history (summary plus a shortened tail).  The
        # store cannot prove a prefix, so the cursor stays at 0 and the batch is
        # persisted again instead of being skipped.  This is the known
        # limitation documented at the end of this file.
        rewritten = [
            {"role": "assistant", "content": "ZUSAMMENFASSUNG: alles Wichtige."}
        ] + stored[4:]
        assert engine._reconcile_ingest_cursor_from_store(rewritten) == 0
    finally:
        engine.shutdown()


# --------------------------------------------------------------------------
# Known limitation — kept as an explicit xfail, not silently dropped
# --------------------------------------------------------------------------
# A *shortened* history (summary plus already-stored tail) is still stored again:
# reconciliation finds no matching store tail and returns 0, so the whole active
# list is appended a second time.  A positional cursor cannot express "these
# messages are stored, that one is new", and the host supplies no stable
# per-message identity.  The xfail for this case lives in
# ``tests/test_ingest_cursor_reuse.py`` and stays there until the host provides
# a history-replacement signal.  It is a known limitation, not an oversight.
#
#
# --------------------------------------------------------------------------
# The fourth site — why it has no test here
# --------------------------------------------------------------------------
# The dropped-scaffold assignment (status ``noop``) is reached only when the
# active list loses replayed scaffold messages *and* the reassembly comes out
# equal to the already-shortened list, so the ``sanitized`` branch is skipped
# and the ``else`` arm runs with ``dropped_replayed_scaffold_messages`` set.
# Reaching it needs two conditions in the same call: ``candidate_start >
# leading_anchor_count`` after skipping scaffold messages at the leaf-chunk
# start, and then ``fresh_tail_start <= leading_anchor_count``.
#
# A search over 81 input shapes (scaffold kind x insertion index x body length x
# leaf chunk size) did not produce that combination against this harness; the
# sanitizer ran before the scaffold-drop condition, so the ``sanitized`` branch
# always won.  The assignment there is fixed identically, it is covered by the
# same invariant as the three tested sites, and it was verified by reading — but
# it is *not* verified by a test, and this note exists so nobody assumes it is.
#
#
# --------------------------------------------------------------------------
# Not related to this change — a pre-existing process relaunch
# --------------------------------------------------------------------------
# ``tests/test_lcm_engine.py`` aborts early in this environment (exit 1, no test
# report).  It is not caused by these changes: the same run aborts identically on
# the unmodified tree, same exit code and same truncated output.
#
# The cause is outside the plugin: the run is relaunched in a second interpreter
# whose bootstrap re-imports the test runner with the Hermes agent directory
# first on ``sys.path`` and without site processing, so the child cannot import
# ``pytest`` and exits.  The parent's report never appears.  Reproduced both
# under ``pytest`` and by calling one test directly from a plain script, which is
# why this is an environment finding and not a defect in the code under test.

