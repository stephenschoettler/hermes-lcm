"""Regression: the ingest cursor must not be reused across different lists.

Two production defects were reproduced against the real engine, both of which
re-append messages that are already in the store.

1. Overflow recovery (``_finalize_forced_overflow_result``) used to set the
   ingest cursor to ``len(compressed)`` — a length read off a *replacement*
   list — and clear the reconciliation flag in the same step.  The next ingest
   of the full history then re-appended every message between
   ``len(compressed)`` and the incoming length, once per turn.

2. A shortened (rewritten) history — summary plus already-stored tail — is
   still stored again: the reconciliation finds no matching *prefix* and
   returns cursor 0, so the whole active list is appended a second time.
   Documented as ``xfail`` because a correct fix needs a history-replacement
   signal that the host does not currently provide; a positional cursor alone
   cannot express "these messages are stored, that one is new".
"""
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "regression-ingest-cursor"


def _engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    return LCMEngine(config=config, hermes_home=str(tmp_path / "home"))


def _history(n, start=0):
    """A history of ``n`` messages, each with a distinct replay identity."""
    msgs = []
    for i in range(start, start + n):
        if i % 3 == 0:
            msgs.append({"role": "user", "content": f"FRAGE-{i:04d}: Sachstand {i}."})
        elif i % 3 == 1:
            msgs.append(
                {"role": "assistant", "content": f"ANTWORT-{i:04d}: {i} Kartons."}
            )
        else:
            msgs.append(
                {
                    "role": "tool",
                    "tool_name": "terminal",
                    "tool_call_id": f"call_reg_{i:04d}",
                    "content": f"ROT-{i:04d}: 41 Kartons, Siegel unbeschaedigt.",
                }
            )
    return msgs


def _bind(engine, session_id=SESSION):
    engine.on_session_start(
        session_id,
        platform="cli",
        conversation_id=f"conversation:{session_id}",
    )


def _stored(engine, session_id=SESSION):
    return engine._store.get_session_count(session_id)


def _duplicated_ids(engine, session_id=SESSION):
    conn = engine._store._conn
    return conn.execute(
        """
        SELECT count(*) FROM (
            SELECT tool_call_id FROM messages
            WHERE session_id = ? AND tool_call_id IS NOT NULL AND tool_call_id != ''
            GROUP BY tool_call_id HAVING count(*) > 1
        )
        """,
        (session_id,),
    ).fetchone()[0]


# --------------------------------------------------------------------------
# Defect 1 — overflow recovery must not re-apply a replacement list's length
# --------------------------------------------------------------------------
def test_overflow_recovery_does_not_reapply_replacement_list_length(tmp_path):
    """The guardrail's shortened list must not become the ingest position."""
    engine = _engine(tmp_path)
    _bind(engine)
    history = _history(60)
    engine.ingest(history)
    assert _stored(engine) == 60

    # The assembly guardrail swaps the active context for a shorter list.
    engine._finalize_forced_overflow_result(history, history[:10])

    # A cursor describes a position in one specific list, so the replacement
    # length must not be adopted — and the reconciliation must stay armed so
    # the position is re-derived from the store by identity.
    assert engine._ingest_cursor_needs_reconcile is True

    engine.ingest(history)

    assert _stored(engine) == 60, "already-stored messages were appended again"
    assert _duplicated_ids(engine) == 0


def test_overflow_recovery_still_ingests_appended_messages(tmp_path):
    """Re-arming the reconciliation must not swallow genuinely new messages."""
    engine = _engine(tmp_path)
    _bind(engine)
    history = _history(60)
    engine.ingest(history)
    engine._finalize_forced_overflow_result(history, history[:10])

    grown = history + _history(5, start=60)
    engine.ingest(grown)

    assert _stored(engine) == 65, "new messages after the guardrail were lost"
    assert _duplicated_ids(engine) == 0


def test_overflow_recovery_repeats_without_accumulating_rows(tmp_path):
    """The defect repeated once per turn; the fix must hold under repetition."""
    engine = _engine(tmp_path)
    _bind(engine)
    history = _history(60)
    engine.ingest(history)

    for _ in range(3):
        engine._finalize_forced_overflow_result(history, history[:10])
        engine.ingest(history)

    assert _stored(engine) == 60
    assert _duplicated_ids(engine) == 0


# --------------------------------------------------------------------------
# Guard: no text-based de-duplication.  Repeats are real events.
# --------------------------------------------------------------------------
def test_genuinely_repeated_message_is_stored_again(tmp_path):
    """An identical message sent twice is two events and must be kept."""
    engine = _engine(tmp_path)
    _bind(engine)
    history = _history(12)
    engine.ingest(history)
    assert _stored(engine) == 12

    repeat = dict(history[0])
    grown = history + [repeat, _history(1, start=12)[0]]
    engine.ingest(grown)

    assert _stored(engine) == 14, "a repeated message was dropped as a duplicate"


# --------------------------------------------------------------------------
# Defect 2 — a shortened history is re-stored in full (not yet fixed)
# --------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason=(
        "A shortened history (summary + already-stored tail) is appended again: "
        "the reconciliation finds no matching prefix and returns cursor 0. "
        "Fixing this needs a history-replacement signal from the host, not a "
        "longer suffix match — skipping stored messages by suffix alone could "
        "drop genuinely repeated messages."
    ),
)
def test_shortened_history_does_not_restore_stored_tail(tmp_path):
    engine = _engine(tmp_path)
    _bind(engine)
    history = _history(60)
    engine.ingest(history)
    assert _stored(engine) == 60

    _bind(engine)  # rebind, e.g. after a host-side compaction
    shortened = [{"role": "assistant", "content": "[Zusammenfassung]"}] + history[-15:]
    engine.ingest(shortened)

    assert _stored(engine) == 60, "the already-stored tail was written again"
    assert _duplicated_ids(engine) == 0
