"""Security regressions for legacy externalized-payload JSON readers."""

from __future__ import annotations

import json
import errno
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_lcm.externalize as externalize_module
from hermes_lcm.externalize import (
    externalized_tool_result_has_persisted_output_marker,
    reassign_externalized_payloads,
)


class _Config:
    def __init__(self, storage_dir: Path):
        self.large_output_externalization_path = str(storage_dir)


@pytest.fixture
def payload_store(tmp_path: Path) -> tuple[Path, _Config]:
    storage_dir = tmp_path / "externalized"
    storage_dir.mkdir(mode=0o700)
    return storage_dir, _Config(storage_dir)


def _payload(*, session_id: str = "old-session", content: str = "stored output") -> dict:
    return {
        "kind": "tool_result",
        "tool_call_id": "call-sec01",
        "role": "tool",
        "session_id": session_id,
        "content": content,
        "content_chars": len(content),
        "content_bytes": len(content.encode("utf-8")),
        "persisted_output_markers": [
            {
                "source_path": "/tmp/hermes-results/call-sec01.txt",
                "expected_chars": len(content),
            }
        ],
    }


def _write_payload(path: Path, **overrides) -> dict:
    payload = _payload(**overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_payload_with_duplicate_marker(path: Path, final_marker_value) -> bytes:
    payload = _payload(content="x" * 1024)
    valid_markers = payload.pop("persisted_output_markers")
    raw = (
        json.dumps(payload)[:-1]
        + ', "persisted_output_markers": '
        + json.dumps(valid_markers)
        + ', "persisted_output_markers": '
        + json.dumps(final_marker_value)
        + "}"
    ).encode("utf-8")
    path.write_bytes(raw)
    return raw


def _swap_payload_for_symlink_on_open(
    monkeypatch: pytest.MonkeyPatch,
    payload_path: Path,
    outside_path: Path,
) -> None:
    real_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and dir_fd is not None and Path(path).name == payload_path.name:
            swapped = True
            payload_path.unlink()
            payload_path.symlink_to(outside_path)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(externalize_module.os, "open", swapping_open)


def _replace_payload_with_regular_file_on_open(
    monkeypatch: pytest.MonkeyPatch,
    payload_path: Path,
    replacement_path: Path,
) -> None:
    real_open = os.open
    swapped = False

    def replacing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and dir_fd is not None and Path(path).name == payload_path.name:
            swapped = True
            replacement_path.replace(payload_path)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(externalize_module.os, "open", replacing_open)


def _report_payload_as_foreign_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    real_fstat = os.fstat

    def foreign_owned_fstat(fd):
        opened = real_fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            return opened
        return SimpleNamespace(
            st_mode=opened.st_mode,
            st_dev=opened.st_dev,
            st_ino=opened.st_ino,
            st_size=opened.st_size,
            st_uid=opened.st_uid + 1,
        )

    monkeypatch.setattr(externalize_module.os, "fstat", foreign_owned_fstat)


def _report_storage_directory_as_differently_owned(
    monkeypatch: pytest.MonkeyPatch,
    storage_dir: Path,
) -> None:
    real_lstat = os.lstat
    real_fstat = os.fstat

    def with_different_uid(value):
        return SimpleNamespace(
            st_mode=value.st_mode,
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_size=value.st_size,
            st_uid=value.st_uid + 1,
            st_nlink=value.st_nlink,
        )

    def differently_owned_lstat(path):
        value = real_lstat(path)
        return with_different_uid(value) if Path(path) == storage_dir else value

    def differently_owned_fstat(fd):
        value = real_fstat(fd)
        return with_different_uid(value) if stat.S_ISDIR(value.st_mode) else value

    monkeypatch.setattr(externalize_module.os, "lstat", differently_owned_lstat)
    monkeypatch.setattr(externalize_module.os, "fstat", differently_owned_fstat)


def _fail_if_json_decoded(_value):
    raise AssertionError("oversized payload reached JSON decoding")


def _force_descriptor_relative_stat_failure(
    monkeypatch: pytest.MonkeyPatch,
    error_type=NotImplementedError,
) -> None:
    real_stat = os.stat

    def unsupported_dir_fd_stat(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None:
            raise error_type("descriptor-relative stat is unavailable")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(externalize_module.os, "stat", unsupported_dir_fd_stat)


def _force_descriptor_relative_open_failure(
    monkeypatch: pytest.MonkeyPatch,
    error_type=NotImplementedError,
) -> None:
    real_open = os.open

    def unsupported_dir_fd_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is not None:
            raise error_type("descriptor-relative open is unavailable")
        return real_open(path, flags, mode)

    monkeypatch.setattr(externalize_module.os, "open", unsupported_dir_fd_open)


def test_marker_reader_accepts_owned_regular_payload(payload_store):
    storage_dir, config = payload_store
    payload_path = storage_dir / "marker.json"
    _write_payload(payload_path)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is True


def test_legacy_readers_accept_service_owned_payload_in_differently_owned_store(
    payload_store,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "marker.json"
    _write_payload(payload_path)
    _report_storage_directory_as_differently_owned(monkeypatch, storage_dir)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is True
    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 1


def test_marker_reader_rejects_symlink(payload_store, tmp_path):
    storage_dir, config = payload_store
    outside_path = tmp_path / "outside-marker.json"
    _write_payload(outside_path)
    payload_path = storage_dir / "marker.json"
    payload_path.symlink_to(outside_path)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is False


def test_marker_reader_rejects_hardlink_outside_store(payload_store, tmp_path):
    storage_dir, config = payload_store
    outside_path = tmp_path / "outside-marker.json"
    _write_payload(outside_path)
    payload_path = storage_dir / "marker.json"
    os.link(outside_path, payload_path)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is False


def test_marker_reader_rejects_symlink_swap_race(payload_store, tmp_path, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "marker.json"
    outside_path = tmp_path / "outside-marker.json"
    _write_payload(payload_path)
    _write_payload(outside_path)
    _swap_payload_for_symlink_on_open(monkeypatch, payload_path, outside_path)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is False


def test_marker_reader_rejects_regular_file_swap_race(payload_store, tmp_path, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "marker.json"
    replacement_path = tmp_path / "replacement-marker.json"
    _write_payload(payload_path)
    _write_payload(replacement_path)
    _replace_payload_with_regular_file_on_open(monkeypatch, payload_path, replacement_path)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is False


def test_marker_reader_rejects_non_regular_payload(payload_store):
    storage_dir, config = payload_store
    (storage_dir / "marker.json").mkdir()

    assert externalized_tool_result_has_persisted_output_marker(
        "marker.json",
        config=config,
    ) is False


def test_marker_reader_rejects_foreign_owned_payload(payload_store, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "marker.json"
    _write_payload(payload_path)
    _report_payload_as_foreign_owned(monkeypatch)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is False


def test_marker_reader_avoids_full_json_decode_for_oversize_payload(payload_store, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "marker.json"
    _write_payload(payload_path, content=("x" * 1024) + '\n"\\é')
    monkeypatch.setattr(externalize_module, "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES", 64, raising=False)
    monkeypatch.setattr(externalize_module.json, "loads", _fail_if_json_decoded)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is True


@pytest.mark.parametrize("invalid_kind", [None, 0, [], {}])
def test_oversize_marker_reader_rejects_non_string_kind(
    payload_store,
    monkeypatch,
    invalid_kind,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "invalid-kind-marker.json"
    payload = _payload(content="x" * 1024)
    payload["kind"] = invalid_kind
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is False


def test_oversize_marker_reader_preserves_missing_kind_legacy_default(
    payload_store,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "missing-kind-marker.json"
    payload = _payload(content="x" * 1024)
    payload.pop("kind")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is True


def test_oversize_marker_reader_rejects_non_string_kind_after_content(
    payload_store,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "suffix-kind-marker.json"
    payload = _payload(content="x" * 1024)
    payload.pop("kind")
    payload["kind"] = None
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is False


def test_marker_reader_rejects_corrupted_oversize_content_between_valid_metadata(
    payload_store,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "corrupted-marker.json"
    _write_payload(payload_path, content="x" * 1024)
    raw = payload_path.read_bytes()
    content_start = raw.index(b'"content": "') + len(b'"content": "')
    corrupt_at = content_start + 512
    payload_path.write_bytes(raw[:corrupt_at] + b"\x00" + raw[corrupt_at + 1 :])
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is False


def test_oversize_marker_reader_streams_metadata_beyond_fixed_tail(payload_store, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "long-marker-metadata.json"
    payload = _payload(content="x" * 1024)
    payload["persisted_output_markers"] = [
        {
            "source_path": f"/tmp/hermes-results/{index:04d}-{'m' * 80}.txt",
            "expected_chars": index + 1,
        }
        for index in range(1024)
    ]
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    assert payload_path.stat().st_size > externalize_module._EXTERNALIZED_SEARCH_TAIL_BYTES
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is True


@pytest.mark.parametrize("final_marker_value", [[], "invalid-marker-value"])
def test_oversize_marker_reader_honors_final_duplicate_marker_value(
    payload_store,
    monkeypatch,
    final_marker_value,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "duplicate-marker.json"
    raw = _write_payload_with_duplicate_marker(payload_path, final_marker_value)
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )

    assert json.loads(raw)["persisted_output_markers"] == final_marker_value
    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is False


def test_oversize_marker_and_reassignment_use_bounded_metadata_path(payload_store, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "large-marker.json"
    _write_payload(payload_path, content="x" * 1024)
    monkeypatch.setattr(externalize_module, "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES", 64, raising=False)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is True
    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 1
    reassigned = json.loads(payload_path.read_text(encoding="utf-8"))
    assert reassigned["session_id"] == "new-session"
    assert reassigned["content"] == "x" * 1024


def test_oversize_reassignment_finds_session_id_after_content(payload_store, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "session-after-content.json"
    raw = (
        '{"kind":"tool_result","content":"'
        + ("x" * 1024)
        + '","session_id":"old-session"}'
    ).encode("utf-8")
    payload_path.write_bytes(raw)
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )
    monkeypatch.setattr(externalize_module.json, "loads", _fail_if_json_decoded)

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 1
    monkeypatch.undo()
    reassigned = json.loads(payload_path.read_text(encoding="utf-8"))
    assert reassigned["session_id"] == "new-session"
    assert reassigned["content"] == "x" * 1024


def test_oversize_reassignment_rewrites_only_final_duplicate_session_id(
    payload_store,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "duplicate-session.json"
    raw = (
        '{"session_id":"prefix-session","content":"'
        + ("x" * 1024)
        + '","session_id":"old-session"}'
    ).encode("utf-8")
    payload_path.write_bytes(raw)
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 1
    rewritten = payload_path.read_bytes()
    assert b'"session_id":"prefix-session"' in rewritten
    assert json.loads(rewritten)["session_id"] == "new-session"


def test_oversize_reassignment_honors_nonmatching_final_duplicate_session_id(
    payload_store,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "nonmatching-final-session.json"
    original = (
        '{"session_id":"old-session","content":"'
        + ("x" * 1024)
        + '","session_id":"final-session"}'
    ).encode("utf-8")
    payload_path.write_bytes(original)
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    assert payload_path.read_bytes() == original


def test_oversize_reassignment_rejects_invalid_suffix_after_session_id(
    payload_store,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "invalid-session-suffix.json"
    original = (
        b'{"content":"'
        + (b"x" * 1024)
        + b'","session_id":"old-session","invalid":"\x00"}'
    )
    payload_path.write_bytes(original)
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    assert payload_path.read_bytes() == original
    assert list(storage_dir.glob("*.tmp")) == []


def test_oversize_reassignment_rejects_truncated_payload_before_temp_creation(
    payload_store,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "truncated-reassign.json"
    original = (
        b'{"kind":"tool_result","session_id":"old-session","content":"'
        + (b"x" * 1024)
    )
    payload_path.write_bytes(original)
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    assert payload_path.read_bytes() == original
    assert list(storage_dir.glob("*.tmp")) == []


def test_oversize_reassignment_rejects_sparse_malformed_payload_without_copy(
    payload_store,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "sparse-reassign.json"
    prefix = b'{"kind":"tool_result","session_id":"old-session","content":"'
    with payload_path.open("wb") as handle:
        handle.write(prefix)
        handle.seek((1024 * 1024) - 1)
        handle.write(b"\x00")
    original_stat = payload_path.stat()
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    current_stat = payload_path.stat()
    assert (current_stat.st_dev, current_stat.st_ino, current_stat.st_size) == (
        original_stat.st_dev,
        original_stat.st_ino,
        original_stat.st_size,
    )
    assert payload_path.read_bytes()[: len(prefix)] == prefix
    assert list(storage_dir.glob("*.tmp")) == []


def test_oversize_reassignment_requires_validation_before_temp_open(payload_store, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "unverified-reassign.json"
    _write_payload(payload_path, content="x" * 1024)
    original = payload_path.read_bytes()
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )
    monkeypatch.setattr(
        externalize_module,
        "_stream_externalized_payload_suffix_metadata",
        lambda *_args, **_kwargs: None,
    )
    real_open = os.open

    def reject_temp_open(path, flags, mode=0o777, *, dir_fd=None):
        assert not str(path).endswith(".tmp")
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(externalize_module.os, "open", reject_temp_open)

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    assert payload_path.read_bytes() == original
    assert list(storage_dir.glob("*.tmp")) == []


def test_oversize_reassignment_rejects_same_inode_mutation_after_validation(
    payload_store,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "mutated-after-validation.json"
    _write_payload(payload_path, content="x" * 1024)
    malformed = bytearray(payload_path.read_bytes())
    content_start = malformed.index(b'"content": "') + len(b'"content": "')
    malformed[content_start + 512] = 0
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )
    real_validate = externalize_module._stream_externalized_payload_suffix_metadata
    validation_calls = 0

    def mutate_after_first_validation(*args, **kwargs):
        nonlocal validation_calls
        result = real_validate(*args, **kwargs)
        validation_calls += 1
        if validation_calls == 1:
            with payload_path.open("r+b") as handle:
                handle.write(malformed)
                handle.truncate()
        return result

    monkeypatch.setattr(
        externalize_module,
        "_stream_externalized_payload_suffix_metadata",
        mutate_after_first_validation,
    )

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
        json.loads(payload_path.read_text(encoding="utf-8"))
    assert list(storage_dir.glob("*.tmp")) == []


def test_oversize_reassignment_rejects_append_after_validation(
    payload_store,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "appended-after-validation.json"
    _write_payload(payload_path, content="x" * 1024)
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )
    real_validate = externalize_module._stream_externalized_payload_suffix_metadata
    validation_calls = 0

    def append_after_first_validation(*args, **kwargs):
        nonlocal validation_calls
        result = real_validate(*args, **kwargs)
        validation_calls += 1
        if validation_calls == 1:
            with payload_path.open("ab") as handle:
                handle.write(b" appended-after-validation")
        return result

    monkeypatch.setattr(
        externalize_module,
        "_stream_externalized_payload_suffix_metadata",
        append_after_first_validation,
    )

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    with pytest.raises(json.JSONDecodeError):
        json.loads(payload_path.read_text(encoding="utf-8"))
    assert list(storage_dir.glob("*.tmp")) == []


@pytest.mark.parametrize("error_type", [TypeError, NotImplementedError])
def test_legacy_readers_use_safe_fallback_when_descriptor_relative_stat_is_unsupported(
    payload_store,
    monkeypatch,
    error_type,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "marker.json"
    _write_payload(payload_path)
    _force_descriptor_relative_stat_failure(monkeypatch, error_type)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is True
    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 1
    assert json.loads(payload_path.read_text(encoding="utf-8"))["session_id"] == "new-session"


@pytest.mark.parametrize("error_type", [TypeError, NotImplementedError])
def test_legacy_readers_use_safe_fallback_when_descriptor_relative_open_is_unsupported(
    payload_store,
    monkeypatch,
    error_type,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "marker.json"
    _write_payload(payload_path)
    _force_descriptor_relative_open_failure(monkeypatch, error_type)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is True
    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 1
    assert json.loads(payload_path.read_text(encoding="utf-8"))["session_id"] == "new-session"


def test_oversize_legacy_readers_use_capability_fallback(payload_store, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "large-marker.json"
    _write_payload(payload_path, content="x" * 1024)
    _force_descriptor_relative_stat_failure(monkeypatch)
    monkeypatch.setattr(
        externalize_module,
        "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES",
        64,
        raising=False,
    )

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is True
    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 1
    reassigned = json.loads(payload_path.read_text(encoding="utf-8"))
    assert reassigned["session_id"] == "new-session"
    assert reassigned["content"] == "x" * 1024


def test_legacy_reader_capability_fallback_rejects_symlink(payload_store, tmp_path, monkeypatch):
    storage_dir, config = payload_store
    outside_path = tmp_path / "outside-marker.json"
    _write_payload(outside_path)
    payload_path = storage_dir / "marker.json"
    payload_path.symlink_to(outside_path)
    _force_descriptor_relative_stat_failure(monkeypatch)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is False
    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0


def test_legacy_reader_capability_fallback_rejects_payload_swap(
    payload_store,
    tmp_path,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "marker.json"
    outside_path = tmp_path / "outside-marker.json"
    _write_payload(payload_path)
    _write_payload(outside_path)
    _force_descriptor_relative_stat_failure(monkeypatch)
    real_open = os.open
    swapped = False

    def swapping_fallback_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and dir_fd is None and Path(path) == payload_path:
            swapped = True
            payload_path.unlink()
            payload_path.symlink_to(outside_path)
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(externalize_module.os, "open", swapping_fallback_open)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is False
    assert swapped is True


def test_legacy_reader_does_not_fallback_for_unrelated_io_error(payload_store, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "marker.json"
    _write_payload(payload_path)
    real_stat = os.stat

    def denied_dir_fd_stat(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None:
            raise PermissionError(errno.EACCES, "permission denied")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(externalize_module.os, "stat", denied_dir_fd_stat)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is False
    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    assert json.loads(payload_path.read_text(encoding="utf-8"))["session_id"] == "old-session"


def test_fallback_reassignment_revalidates_payload_before_replace(
    payload_store,
    tmp_path,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "reassign.json"
    original = _write_payload(payload_path)
    held_original = tmp_path / "held-original.json"
    replacement_path = tmp_path / "replacement.json"
    replacement = _write_payload(
        replacement_path,
        session_id="replacement-session",
        content="replacement sentinel",
    )
    _force_descriptor_relative_stat_failure(monkeypatch)
    real_replace = externalize_module._replace_externalized_payload
    swapped = False

    def swapping_replace(path, payload, *args, **kwargs):
        nonlocal swapped
        payload_path.replace(held_original)
        replacement_path.replace(payload_path)
        swapped = True
        return real_replace(path, payload, *args, **kwargs)

    monkeypatch.setattr(externalize_module, "_replace_externalized_payload", swapping_replace)

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    assert swapped is True
    assert json.loads(payload_path.read_text(encoding="utf-8")) == replacement
    assert json.loads(held_original.read_text(encoding="utf-8")) == original
    assert list(storage_dir.glob("*.tmp")) == []


def test_reassignment_preserves_content_and_atomic_replacement(payload_store):
    storage_dir, config = payload_store
    payload_path = storage_dir / "reassign.json"
    original = _write_payload(payload_path)

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 1

    reassigned = json.loads(payload_path.read_text(encoding="utf-8"))
    assert reassigned["session_id"] == "new-session"
    assert reassigned["content"] == original["content"]
    assert list(storage_dir.glob("*.tmp")) == []


def test_reassignment_post_read_store_directory_swap_does_not_escape(payload_store, tmp_path, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "reassign.json"
    original = _write_payload(payload_path)
    held_storage_dir = tmp_path / "held-externalized"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_path = outside_dir / payload_path.name
    _write_payload(outside_path, session_id="outside-session", content="outside sentinel")
    outside_bytes = outside_path.read_bytes()
    real_replace = externalize_module._replace_externalized_payload
    swapped = False

    def swapping_replace(path, payload, *args, **kwargs):
        nonlocal swapped
        assert path == payload_path
        storage_dir.rename(held_storage_dir)
        storage_dir.symlink_to(outside_dir, target_is_directory=True)
        swapped = True
        return real_replace(path, payload, *args, **kwargs)

    monkeypatch.setattr(externalize_module, "_replace_externalized_payload", swapping_replace)

    moved = reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    )

    assert swapped is True
    assert outside_path.read_bytes() == outside_bytes
    held_payload = json.loads((held_storage_dir / payload_path.name).read_text(encoding="utf-8"))
    assert moved in {0, 1}
    assert held_payload["session_id"] == ("new-session" if moved == 1 else original["session_id"])
    assert held_payload["content"] == original["content"]
    assert list(held_storage_dir.glob("*.tmp")) == []


def test_reassignment_revalidates_payload_identity_immediately_before_replace(
    payload_store,
    tmp_path,
    monkeypatch,
):
    storage_dir, config = payload_store
    payload_path = storage_dir / "reassign.json"
    original = _write_payload(payload_path)
    held_original = tmp_path / "held-original.json"
    replacement_path = tmp_path / "replacement.json"
    replacement = _write_payload(
        replacement_path,
        session_id="replacement-session",
        content="replacement sentinel",
    )
    real_replace = externalize_module._replace_externalized_payload
    swapped = False

    def swapping_replace(path, payload, *args, **kwargs):
        nonlocal swapped
        payload_path.replace(held_original)
        replacement_path.replace(payload_path)
        swapped = True
        return real_replace(path, payload, *args, **kwargs)

    monkeypatch.setattr(externalize_module, "_replace_externalized_payload", swapping_replace)

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    assert swapped is True
    assert json.loads(payload_path.read_text(encoding="utf-8")) == replacement
    assert json.loads(held_original.read_text(encoding="utf-8")) == original
    assert list(storage_dir.glob("*.tmp")) == []


def test_reassignment_reader_rejects_symlink(payload_store, tmp_path):
    storage_dir, config = payload_store
    outside_path = tmp_path / "outside-reassign.json"
    outside_payload = _write_payload(outside_path)
    payload_path = storage_dir / "reassign.json"
    payload_path.symlink_to(outside_path)

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    assert json.loads(outside_path.read_text(encoding="utf-8")) == outside_payload


def test_reassignment_reader_rejects_hardlink_outside_store(payload_store, tmp_path):
    storage_dir, config = payload_store
    outside_path = tmp_path / "outside-reassign.json"
    outside_payload = _write_payload(outside_path)
    payload_path = storage_dir / "reassign.json"
    os.link(outside_path, payload_path)

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    assert json.loads(outside_path.read_text(encoding="utf-8")) == outside_payload


def test_reassignment_reader_rejects_symlink_swap_race(payload_store, tmp_path, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "reassign.json"
    outside_path = tmp_path / "outside-reassign.json"
    _write_payload(payload_path)
    outside_payload = _write_payload(outside_path)
    _swap_payload_for_symlink_on_open(monkeypatch, payload_path, outside_path)

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    assert json.loads(outside_path.read_text(encoding="utf-8")) == outside_payload


def test_reassignment_reader_rejects_regular_file_swap_race(payload_store, tmp_path, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "reassign.json"
    replacement_path = tmp_path / "replacement-reassign.json"
    _write_payload(payload_path)
    replacement_payload = _write_payload(replacement_path)
    _replace_payload_with_regular_file_on_open(monkeypatch, payload_path, replacement_path)

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
    assert json.loads(payload_path.read_text(encoding="utf-8")) == replacement_payload


def test_reassignment_reader_rejects_non_regular_payload(payload_store):
    storage_dir, config = payload_store
    (storage_dir / "reassign.json").mkdir()

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0


def test_reassignment_reader_rejects_foreign_owned_payload(payload_store, monkeypatch):
    storage_dir, config = payload_store
    _write_payload(storage_dir / "reassign.json")
    _report_payload_as_foreign_owned(monkeypatch)

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0


def test_reassignment_streams_oversize_without_full_json_decode(payload_store, monkeypatch):
    storage_dir, config = payload_store
    _write_payload(storage_dir / "reassign.json", content="x" * 1024)
    monkeypatch.setattr(externalize_module, "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES", 64, raising=False)
    monkeypatch.setattr(externalize_module.json, "loads", _fail_if_json_decoded)

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 1


def test_reassignment_skips_malformed_oversize_payload(payload_store, monkeypatch):
    storage_dir, config = payload_store
    (storage_dir / "malformed.json").write_bytes(b"\xff" * 1024)
    monkeypatch.setattr(externalize_module, "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES", 64, raising=False)

    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0
