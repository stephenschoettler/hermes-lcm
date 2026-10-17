"""Cross-platform regressions for Windows persisted-output I/O."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from hermes_lcm import externalize as externalize_module
from hermes_lcm.ingest_protection import recover_hermes_persisted_output_with_file_stat


def _persisted_marker(path: Path, original: str, *, preview: str = "A") -> str:
    return "\n".join(
        [
            "<persisted-output>",
            f"This tool result was too large ({len(original)} characters, 0 KB).",
            f"Full output saved to: {path}",
            f"Preview (first {len(preview)} chars):",
            preview,
            "...",
            "</persisted-output>",
        ]
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows directory handles are unsupported by this path")
def test_externalized_payload_write_flushes_file_without_opening_parent_directory(
    tmp_path, monkeypatch
):
    target = tmp_path / "payload.json"
    fsync_calls: list[int] = []
    real_fsync = externalize_module.os.fsync
    real_open = externalize_module.os.open

    def recording_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    def reject_directory_open(path, flags, *args, **kwargs):
        if Path(path) == tmp_path:
            raise AssertionError("Windows parent directory must not be opened for fsync")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(externalize_module.os, "fsync", recording_fsync)
    monkeypatch.setattr(externalize_module.os, "open", reject_directory_open)

    externalize_module._write_externalized_payload(target, {"content": "durable payload"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"content": "durable payload"}
    assert fsync_calls, "the payload file must still be flushed on Windows"


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory fsync contract")
def test_parent_directory_fsync_is_retained_on_posix(tmp_path, monkeypatch):
    opened_flags: list[int] = []
    real_open = externalize_module.os.open

    def recording_open(path, flags, *args, **kwargs):
        opened_flags.append(flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(externalize_module.os, "open", recording_open)

    externalize_module._fsync_directory(tmp_path)

    assert opened_flags
    if hasattr(os, "O_DIRECTORY"):
        assert all(flags & os.O_DIRECTORY for flags in opened_flags)


@pytest.mark.skipif(os.name != "nt", reason="Windows text-mode newline expansion")
def test_crlf_recovery_preserves_original_bare_carriage_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    directory = tmp_path / "hermes-results"
    directory.mkdir()
    original = "A\rB\nC"
    path = directory / "result.txt"
    path.write_bytes(original.replace("\n", "\r\n").encode("utf-8"))

    recovered = recover_hermes_persisted_output_with_file_stat(
        _persisted_marker(path, original)
    )

    assert recovered is not None
    assert recovered[0] == original


@pytest.mark.skipif(os.name != "nt", reason="Windows text-mode newline expansion")
def test_crlf_recovery_rejects_mixed_translated_and_bare_lf(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    directory = tmp_path / "hermes-results"
    directory.mkdir()
    original = "A\nB\nC"
    path = directory / "mixed.txt"
    path.write_bytes(b"A\r\nB\nC")

    recovered = recover_hermes_persisted_output_with_file_stat(
        _persisted_marker(path, original)
    )

    assert recovered is None
