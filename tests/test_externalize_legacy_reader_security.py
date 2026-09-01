"""Security regressions for legacy externalized-payload JSON readers."""

from __future__ import annotations

import json
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


def _fail_if_json_decoded(_value):
    raise AssertionError("oversized payload reached JSON decoding")


def test_marker_reader_accepts_owned_regular_payload(payload_store):
    storage_dir, config = payload_store
    payload_path = storage_dir / "marker.json"
    _write_payload(payload_path)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is True


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


def test_marker_reader_decodes_only_bounded_metadata_for_oversize_payload(payload_store, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "marker.json"
    _write_payload(payload_path, content=("x" * 1024) + '\n"\\é')
    monkeypatch.setattr(externalize_module, "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES", 64, raising=False)
    real_loads = json.loads
    decoded_sizes = []

    def bounded_loads(value):
        decoded_sizes.append(len(value))
        assert len(value) <= externalize_module._EXTERNALIZED_SEARCH_TAIL_BYTES + 1
        return real_loads(value)

    monkeypatch.setattr(externalize_module.json, "loads", bounded_loads)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is True
    assert decoded_sizes


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


def test_legacy_readers_fail_closed_when_descriptor_relative_stat_is_unsupported(payload_store, monkeypatch):
    storage_dir, config = payload_store
    payload_path = storage_dir / "marker.json"
    _write_payload(payload_path)
    real_stat = os.stat

    def unsupported_dir_fd_stat(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None:
            raise NotImplementedError("descriptor-relative stat is unavailable")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(externalize_module.os, "stat", unsupported_dir_fd_stat)

    assert externalized_tool_result_has_persisted_output_marker(
        payload_path.name,
        config=config,
    ) is False
    assert reassign_externalized_payloads(
        "old-session",
        "new-session",
        config=config,
    ) == 0


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
