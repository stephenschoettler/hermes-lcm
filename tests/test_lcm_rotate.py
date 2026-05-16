"""Tests for /lcm rotate command surface and engine rotate path."""

import importlib
import os
import re
from pathlib import Path

from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _build_engine(tmp_path, *, fresh_tail_count: int = 3) -> LCMEngine:
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_rotate_test.db")
    config.fresh_tail_count = fresh_tail_count
    hermes_home = tmp_path / "hermes_home"
    engine = LCMEngine(config=config, hermes_home=str(hermes_home))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._lifecycle.bind_session("live-session", conversation_id="live-session")
    engine.context_length = 200000
    engine.threshold_tokens = int(200000 * config.context_threshold)
    return engine


def _seed_messages(engine: LCMEngine, count: int) -> None:
    for index in range(count):
        engine._store.append(
            engine._session_id,
            {"role": "user", "content": f"message-{index}"},
            source="test",
        )
    engine._store._conn.commit()


def _extract_field(result: str, key: str) -> str:
    line = next(
        (line for line in result.splitlines() if line.startswith(f"{key}: ")),
        None,
    )
    assert line is not None, f"expected {key} in output:\n{result}"
    return line.split(": ", 1)[1]


def test_rotate_preview_reports_planned_frontier_without_mutating(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    _seed_messages(engine, count=10)

    result = handle_lcm_command("rotate", engine)

    assert "LCM rotate" in result
    assert "status: preview" in result
    assert "total_message_count: 10" in result
    assert "fresh_tail_count: 3" in result
    assert "pre_tail_message_count: 7" in result
    assert "current_frontier_store_id: 0" in result
    # New frontier should be store_id of last pre-tail message (7).
    assert "new_frontier_store_id: 7" in result
    assert "rotate_backup_path:" in result
    assert "note: read-only preview" in result

    # Confirm no mutation: frontier still 0 in lifecycle state.
    state = engine._lifecycle.get_by_conversation(engine._conversation_id)
    assert state is not None
    assert state.current_frontier_store_id == 0

    # Confirm no backup was written.
    assert not engine.rotate_backup_path().exists()


def test_rotate_preview_reports_noop_when_total_messages_within_tail(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=5)
    _seed_messages(engine, count=3)

    result = handle_lcm_command("rotate", engine)

    assert "status: noop" in result
    assert "reason: no_pre_tail_content" in result
    assert "total_message_count: 3" in result


def test_rotate_apply_advances_frontier_and_writes_rolling_backup(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    _seed_messages(engine, count=10)

    result = handle_lcm_command("rotate apply", engine)

    assert "LCM rotate apply" in result
    assert "status: ok" in result
    assert "previous_frontier_store_id: 0" in result
    assert "new_frontier_store_id: 7" in result
    assert "note: rolling backup overwrites the previous rotate-latest slot" in result
    backup_path = Path(_extract_field(result, "rotate_backup_path"))
    assert backup_path.exists()
    assert backup_path.name.endswith("-rotate-latest.sqlite3")
    # Backup size line uses _fmt_size formatting (e.g. "12.0 KB"); assert
    # it exists and is non-empty rather than pinning the exact representation.
    backup_size_str = _extract_field(result, "rotate_backup_size")
    assert backup_size_str.strip()

    state = engine._lifecycle.get_by_conversation(engine._conversation_id)
    assert state is not None
    assert state.current_frontier_store_id == 7
    # The in-process source-mapping marker must NOT advance here. Pre-tail
    # raw messages may still be present in the host's in-memory active
    # context; advancing the marker would break source_ids lineage on the
    # next in-process compress(). The persisted lifecycle frontier is the
    # bootstrap signal; the in-process marker is only updated by actual
    # compaction inside this process. See the regression at
    # test_rotate_apply_does_not_corrupt_source_lineage_on_next_compress.
    assert engine._last_compacted_store_id == 0


def test_rotate_apply_preserves_raw_messages_for_lossless_recovery(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=2)
    _seed_messages(engine, count=8)

    handle_lcm_command("rotate apply", engine)

    # All 8 raw messages remain in the store after rotate — frontier only
    # changes the bootstrap replay boundary, never deletes raw history.
    assert engine._store.get_session_count(engine._session_id) == 8
    tail = engine._store.get_session_tail(engine._session_id, limit=2)
    assert [row.get("content") for row in tail] == ["message-6", "message-7"]


def test_rotate_apply_rerun_is_idempotent_and_preserves_existing_backup(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    _seed_messages(engine, count=10)

    first = handle_lcm_command("rotate apply", engine)
    assert "status: ok" in first
    first_backup = Path(_extract_field(first, "rotate_backup_path"))
    assert first_backup.exists()
    first_size = first_backup.stat().st_size
    first_mtime = first_backup.stat().st_mtime

    second = handle_lcm_command("rotate apply", engine)
    assert "status: noop" in second
    assert "reason: frontier_already_ahead" in second
    assert (
        "rolling backup was not written so the previous rotate-latest snapshot is preserved"
        in second
    )

    state = engine._lifecycle.get_by_conversation(engine._conversation_id)
    assert state is not None
    assert state.current_frontier_store_id == 7

    # The previous known-good rolling backup must survive a noop rerun. Both
    # mtime and size are unchanged because the noop short-circuit happens
    # before any disk write.
    assert first_backup.exists()
    assert first_backup.stat().st_mtime == first_mtime
    assert first_backup.stat().st_size == first_size


def test_rotate_apply_rolling_backup_overwrites_prior_slot_on_actual_rotate(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    _seed_messages(engine, count=10)

    first = handle_lcm_command("rotate apply", engine)
    assert "status: ok" in first
    first_backup = Path(_extract_field(first, "rotate_backup_path"))
    assert first_backup.exists()
    first_mtime = first_backup.stat().st_mtime

    # Add more messages so the second apply has new content to rotate past
    # the now-advanced frontier (not a noop).
    _seed_messages(engine, count=5)

    # Force a measurable mtime delta — SQLite backup() inside the same second
    # can land on the same mtime granularity on some filesystems.
    older = first_mtime - 5.0
    os.utime(first_backup, (older, older))

    second = handle_lcm_command("rotate apply", engine)
    assert "status: ok" in second
    second_backup = Path(_extract_field(second, "rotate_backup_path"))
    assert second_backup == first_backup, "rotate should reuse the rolling slot, not create a new file"
    assert second_backup.stat().st_mtime > older, "rolling slot mtime should advance on re-rotate"

    # Ensure only one rotate backup exists in the directory — disk usage bounded.
    rotate_backups = list(first_backup.parent.glob("*-rotate-latest.sqlite3"))
    assert rotate_backups == [first_backup]


def test_rotate_refuses_on_ignored_session(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    _seed_messages(engine, count=8)
    engine._session_ignored = True

    result = handle_lcm_command("rotate", engine)

    assert "status: refused" in result
    assert "reason: session_ignored" in result


def test_rotate_refuses_on_stateless_session(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    _seed_messages(engine, count=8)
    engine._session_stateless = True

    result = handle_lcm_command("rotate", engine)

    assert "status: refused" in result
    assert "reason: session_stateless" in result


def test_rotate_apply_refuses_on_ignored_session_without_writing_backup(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    _seed_messages(engine, count=8)
    engine._session_ignored = True

    result = handle_lcm_command("rotate apply", engine)

    assert "status: refused" in result
    assert "reason: session_ignored" in result
    # Backup must not be written when the rotate is refused.
    assert not engine.rotate_backup_path().exists()


def test_rotate_apply_refuses_on_stateless_session_without_writing_backup(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    _seed_messages(engine, count=8)
    engine._session_stateless = True

    result = handle_lcm_command("rotate apply", engine)

    assert "status: refused" in result
    assert "reason: session_stateless" in result
    assert not engine.rotate_backup_path().exists()


def test_rotate_refuses_when_no_session_bound(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    engine._session_id = ""
    engine._conversation_id = ""

    result = handle_lcm_command("rotate", engine)

    assert "status: refused" in result
    assert "reason: no_active_session" in result


def test_rotate_help_rejects_unknown_subcommand(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)

    result = handle_lcm_command("rotate something-else", engine)

    # Soft assertion — exact wording may evolve; verify the help text
    # mentions rotate and rejects the bogus subcommand.
    assert "rotate" in result
    assert "apply" in result


def test_lcm_status_reports_last_rotate_at_after_apply(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    _seed_messages(engine, count=10)

    before = handle_lcm_command("status", engine)
    assert "last_rotate_at: (never)" in before

    handle_lcm_command("rotate apply", engine)

    after = handle_lcm_command("status", engine)
    # Pin the format: UTC ISO-8601 with seconds precision and +00:00 offset.
    assert re.search(
        r"last_rotate_at: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", after
    ), f"expected UTC ISO timestamp in status output:\n{after}"
    assert "last_rotate_at: (never)" not in after
    # Backup size field is rendered with _fmt_size formatting; assert presence
    # and non-empty content, not exact bytes.
    assert "rotate_backup_size: " in after
    assert "rotate_backup_path:" in after


def test_lcm_help_lists_rotate_subcommands(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    result = handle_lcm_command("help", engine)
    assert "/lcm rotate" in result
    assert "/lcm rotate apply" in result


def test_rotate_handles_session_with_exactly_fresh_tail_count_messages(tmp_path):
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    _seed_messages(engine, count=3)

    result = handle_lcm_command("rotate", engine)

    assert "status: noop" in result
    assert "reason: no_pre_tail_content" in result
    assert "total_message_count: 3" in result
    assert "pre_tail_message_count: 0" in result


def test_rotate_empty_tail_branch_returns_noop_shape_without_keyerror(tmp_path):
    """Concurrent deletion can empty the tail after the count check. The
    formatter must render a no-op without KeyError.
    """
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    _seed_messages(engine, count=10)

    # Force the count-vs-tail mismatch the real concurrent-delete race
    # would produce: total_count reports 10, but get_session_tail returns
    # empty. Patch the bound method on this engine's store instance only.
    original_get_session_tail = engine._store.get_session_tail
    try:
        engine._store.get_session_tail = lambda session_id, limit=1000: []  # type: ignore[method-assign]
        result = handle_lcm_command("rotate", engine)
    finally:
        engine._store.get_session_tail = original_get_session_tail  # type: ignore[method-assign]

    assert "status: noop" in result
    assert "reason: empty_tail" in result
    # Required fields must be present so command-layer formatters do not crash.
    assert "pre_tail_message_count: 0" in result
    assert "new_frontier_store_id: 0" in result


def test_rotate_apply_aborts_and_preserves_state_when_backup_write_fails(tmp_path):
    """Backup-first contract: if the rolling backup write fails, rotate apply
    must return status:error and leave lifecycle frontier untouched.
    """
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    _seed_messages(engine, count=10)

    # Pre-create the backup directory as a regular file so mkdir / write fails.
    backup_path = engine.rotate_backup_path()
    backup_dir = backup_path.parent
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    # Replace the would-be backup directory with a regular file to force OSError
    # on mkdir(parents=True, exist_ok=True) inside _rotate_backup_database.
    backup_dir.write_text("not a directory")

    try:
        result = handle_lcm_command("rotate apply", engine)
    finally:
        # Clean up the sentinel file so later tests in the same tmp_path do
        # not interfere — though pytest gives each test its own tmp_path.
        if backup_dir.exists() and backup_dir.is_file():
            backup_dir.unlink()

    assert "LCM rotate apply" in result
    assert "status: error" in result
    assert "error: backup failed:" in result
    assert "note: rotate apply aborted before any lifecycle mutation" in result

    # Critical safety guarantee: lifecycle frontier is unchanged.
    state = engine._lifecycle.get_by_conversation(engine._conversation_id)
    assert state is not None
    assert state.current_frontier_store_id == 0


def test_rotate_apply_reports_stale_lifecycle_state_when_session_drifts(tmp_path):
    """advance_frontier silently returns the unchanged state when the lifecycle
    row's current_session_id no longer matches this engine's session. rotate
    apply must detect that and surface status:refused reason:stale_lifecycle_state
    rather than reporting status:ok with a fabricated frontier.
    """
    engine = _build_engine(tmp_path, fresh_tail_count=3)
    _seed_messages(engine, count=10)

    # Simulate session drift: another flow bound a different session id to
    # this conversation. _bind_lifecycle_state would have updated the row.
    engine._lifecycle.bind_session("other-session", conversation_id=engine._conversation_id)
    # Engine's local _session_id stays on the original "live-session".

    result = handle_lcm_command("rotate apply", engine)

    assert "status: refused" in result
    assert "reason: stale_lifecycle_state" in result

    # The rolling backup WAS written (we passed preflight) — that's expected
    # with the current ordering. Lifecycle frontier should NOT have advanced
    # for "live-session".
    state = engine._lifecycle.get_by_conversation(engine._conversation_id)
    assert state is not None
    assert state.current_frontier_store_id == 0


def test_rotate_apply_does_not_corrupt_source_lineage_on_next_compress(tmp_path, monkeypatch):
    """Regression for the issue Tosko4 surfaced on PR #176.

    After /lcm rotate apply, the in-memory active context still holds the
    pre-rotate raw messages until the host rebuilds it. A normal compress()
    later in the same process must produce a DAG node whose source_ids
    reference the same raw rows it summarized — not just the post-rotate
    tail. Advancing the in-process source-mapping marker on rotate would
    cause _get_store_ids_for_messages to filter out the pre-rotate rows,
    producing a poisoned node (text covers msg-0..msg-7, source_ids = [9]).
    """
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_rotate_lineage.db")
    config.fresh_tail_count = 3
    config.leaf_chunk_tokens = 10
    config.context_threshold = 0.001
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    engine._session_id = "live-session"
    engine._session_platform = "telegram"
    engine._conversation_id = "live-session"
    engine._lifecycle.bind_session("live-session", conversation_id="live-session")
    engine.context_length = 200000
    engine.threshold_tokens = int(200000 * config.context_threshold)

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(10):
        messages.append({"role": "user", "content": f"msg-{i} " + "x" * 200})
    for msg in messages[1:]:
        engine._store.append(engine._session_id, msg, source="test")
    engine._store._conn.commit()

    # Apply rotate. The persisted frontier moves to 8 (store_id of msg-8,
    # the second-to-last message; tail keeps the last 3 store rows). The
    # host's in-memory active context still holds system + msg-0..msg-9.
    rotate_result = engine.rotate_active_session(apply=True)
    assert rotate_result["ok"] is True
    assert rotate_result["noop"] is False
    state = engine._lifecycle.get_by_conversation(engine._conversation_id)
    assert state is not None
    assert state.current_frontier_store_id == rotate_result["new_frontier_store_id"]
    # Critical invariant: the in-process source-mapping marker must NOT
    # have advanced. If it did, _get_store_ids_for_messages would filter
    # out the pre-rotate raw rows on the next compress() and the resulting
    # summary node would have source_ids referencing only post-frontier
    # rows while its text covered pre-frontier content.
    assert engine._last_compacted_store_id == 0

    # Host appends a new assistant turn, simulating Hermes continuing in
    # the same process. The in-memory active context still has all the
    # pre-rotate messages.
    messages.append({"role": "assistant", "content": "ack-msg-9"})
    engine._store.append(engine._session_id, messages[-1], source="test")
    engine._store._conn.commit()

    # Stub the summarizer so the test is deterministic and we can verify
    # exactly which raw messages get compacted.
    lcm_engine_module = importlib.import_module("hermes_lcm.engine")
    monkeypatch.setattr(
        lcm_engine_module,
        "summarize_with_escalation",
        lambda **kwargs: ("Summary of pre-tail messages.\nExpand for details about: msg-0..msg-7", 1),
    )

    engine.compress(messages)

    nodes = engine._dag.get_session_nodes(engine._session_id)
    assert nodes, "compress() should have produced at least one summary node"
    summary_node = nodes[0]

    # source_ids must reference the actual raw rows the summary covers.
    # Empty or post-frontier-only source_ids would mean the lineage was
    # severed by the in-process marker filter.
    assert summary_node.source_ids, (
        f"summary_node.source_ids is empty — _get_store_ids_for_messages "
        f"likely filtered out the pre-rotate rows. Marker was {engine._last_compacted_store_id}, "
        f"compacted msgs first/last: msg-0..msg-7."
    )
    # Source rows must come from the pre-tail range (store_id <= rotate
    # frontier). If source_ids only contained post-frontier rows (the
    # original bug shape: source_ids == [9]), this assertion fails.
    rotate_frontier = rotate_result["new_frontier_store_id"]
    pre_frontier_sources = [
        sid for sid in summary_node.source_ids if sid <= rotate_frontier
    ]
    assert pre_frontier_sources, (
        f"summary covers pre-rotate messages but source_ids "
        f"({summary_node.source_ids}) contains no rows at or below the "
        f"rotate frontier ({rotate_frontier}). Source lineage is severed."
    )


def test_rotate_backup_path_falls_back_to_db_sibling_when_hermes_home_unset(tmp_path):
    """The hermes_home-unset branch in rotate_backup_path puts the rolling
    backup beside the LCM database. Cover the branch so a regression there
    is visible.
    """
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_no_home.db")
    config.fresh_tail_count = 3
    engine = LCMEngine(config=config, hermes_home="")
    try:
        path = engine.rotate_backup_path()
        assert path == tmp_path / "backups" / "lcm" / "lcm_no_home-rotate-latest.sqlite3"
        assert engine.backup_dir() == tmp_path / "backups" / "lcm"
    finally:
        engine.shutdown()
