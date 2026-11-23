from __future__ import annotations

import tempfile


def test_reconcile_memoizes_expensive_message_identities(tmp_path, monkeypatch):
    """Each active message identity is resolved once per reconciliation pass.

    Externalized tool-result identities can read backing files. Recomputing every
    prefix identity for every candidate cursor turns one cold-session reconcile
    into quadratic file I/O and can block the synchronous post-turn hook.
    """
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine
    import hermes_lcm.reconcile as reconcile_module

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
        hermes_home=str(tmp_path / "hermes"),
    )

    host_storage = tmp_path / "hermes-results"
    host_storage.mkdir()
    persisted_markers = []
    for index in range(64):
        full_result = f"FILE_BACKED_RESULT_{index}:" + (str(index % 10) * 256)
        persisted_path = host_storage / f"call-{index}.txt"
        persisted_path.write_text(full_result, encoding="utf-8")
        persisted_markers.append(
            "<persisted-output>\n"
            f"This tool result was too large ({len(full_result):,} characters, 0.3 KB).\n"
            f"Full output saved to: {persisted_path}\n"
            "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
            "Preview (first 30 chars):\n"
            f"{full_result[:30]}\n...\n"
            "</persisted-output>"
        )

    messages = [
        {
            "role": "assistant",
            "content": "[Current user objective preserved from compacted history]\nContinue safely.",
        },
        *[
            {
                "role": "tool",
                "tool_call_id": f"call-{index}",
                "content": persisted_markers[index],
            }
            for index in range(64)
        ],
    ]
    stored_tail_rows = [
        {
            "role": "tool",
            "tool_call_id": f"stored-call-{index}",
            "content": f"unrelated stored tool result {index}",
        }
        for index in range(256)
    ]
    stored_tail = [
        engine._message_replay_identity(row, stored_row=True)
        for row in stored_tail_rows
    ]

    identity_calls = 0
    recovery_calls = 0
    original_identity = engine._message_replay_identity
    original_recover = reconcile_module.recover_hermes_persisted_output_with_file_stat

    def counted_identity(message, *args, **kwargs):
        nonlocal identity_calls
        identity_calls += 1
        return original_identity(message, *args, **kwargs)

    def counted_recover(text):
        nonlocal recovery_calls
        recovery_calls += 1
        return original_recover(text)

    monkeypatch.setattr(engine, "_message_replay_identity", counted_identity)
    monkeypatch.setattr(
        reconcile_module,
        "recover_hermes_persisted_output_with_file_stat",
        counted_recover,
    )

    cursor = engine._find_reconciled_cursor_for_store_tail(
        messages,
        stored_tail,
        stored_tail_rows=stored_tail_rows,
        allow_empty_prefix=True,
        session_count=len(stored_tail),
        raw_session_count=len(stored_tail),
    )

    assert cursor == 1
    print("REPLAY_COUNTS", identity_calls, recovery_calls)
    assert identity_calls <= len(messages)
    assert 64 <= recovery_calls <= 128
