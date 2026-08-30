"""The ingest cursor must survive a host handing back a pre-compaction history.

``compress()`` lowers ``_ingest_cursor`` to the length of the compacted list.
A host that runs end-of-session extraction at the compaction boundary then
passes the PRE-compaction history to ``on_session_end`` before the compacted
list is ever ingested. Measured against the lowered cursor every row past it
looks new, so the whole session is appended a second time.
"""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION_ID = "20260101_000000_cursor01"


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig()
    config.fresh_tail_count = 4
    config.leaf_chunk_tokens = 100
    config.database_path = str(tmp_path / "lcm_cursor.db")
    e = LCMEngine(config=config)
    e._session_id = SESSION_ID
    e.context_length = 200000
    e.threshold_tokens = int(200000 * config.context_threshold)
    try:
        yield e
    finally:
        e.shutdown()


def _mock_summarize(prompt, max_tokens, model=""):
    return "Mock summary of conversation.\nExpand for details about: earlier turns"


@pytest.fixture
def mocked_summary(monkeypatch):
    import hermes_lcm.escalation as esc

    monkeypatch.setattr(esc, "_call_llm_for_summary", _mock_summarize)


def _conversation(n_turns=20):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"Question {i}: " + "x" * 200})
        messages.append({"role": "assistant", "content": f"Answer {i}: " + "y" * 200})
    return messages


def _rows(engine):
    return engine._store.get_session_count(SESSION_ID)


def test_pre_compaction_history_after_compress_is_not_stored_again(engine, mocked_summary):
    """The pre-compaction list handed back after compress() appends nothing."""
    messages = _conversation(20)
    engine.compress(messages)
    stored = _rows(engine)
    assert stored == len(messages)

    # What a host's end-of-session extraction passes at the compaction boundary.
    engine._ingest_messages(messages)

    assert _rows(engine) == stored


def test_extended_pre_compaction_history_stores_only_the_new_rows(engine, mocked_summary):
    """The same list plus a new turn appends that turn and nothing else."""
    messages = _conversation(20)
    engine.compress(messages)
    stored = _rows(engine)

    extended = messages + [
        {"role": "user", "content": "Brand new question"},
        {"role": "assistant", "content": "Brand new answer"},
    ]
    engine._ingest_messages(extended)

    assert _rows(engine) == stored + 2


def test_cursor_is_not_advanced_by_another_sessions_history(engine, mocked_summary):
    """The proof is session-scoped: a rebind must not inherit it.

    The identity cache is not cleared when the engine rebinds, so without a
    session stamp a new session whose opening history matches the previous
    one's would have its rows skipped as already accounted for.
    """
    messages = _conversation(6)
    engine._ingest_messages(messages)
    assert _rows(engine) == len(messages)

    other_session = "20260101_000000_cursor02"
    engine.on_session_start(other_session, platform="cli")
    engine._ingest_messages(messages)

    assert engine._store.get_session_count(other_session) == len(messages)


def test_shorter_history_does_not_lower_the_cursor(engine, mocked_summary):
    """A truncated history leaves the cursor where it was."""
    messages = _conversation(6)
    engine._ingest_messages(messages)
    stored = _rows(engine)
    cursor = engine._ingest_cursor

    engine._ingest_messages(messages[:3])

    assert engine._ingest_cursor == cursor
    assert _rows(engine) == stored
