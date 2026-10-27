"""Real temp-file/SQLite cases; unexecuted staging, no provider or live history."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm import externalize as ext
from hermes_lcm import store as store_module
from hermes_lcm.store import MessageStore


@pytest.fixture
def config(tmp_path):
    return SimpleNamespace(
        large_output_externalization_path=str(tmp_path / "payloads"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=1,
    )


def freeze_clock(monkeypatch):
    monkeypatch.setattr(ext.time, "time_ns", lambda: 1)


@pytest.mark.parametrize("public_function", ["externalize_ingest_payload", "maybe_externalize_payload"])
def test_same_payload_different_sessions_do_not_collide(tmp_path, monkeypatch, config, public_function):
    freeze_clock(monkeypatch)
    call = getattr(ext, public_function)
    results = [call("same synthetic payload", session_id=session, config=config, hermes_home=str(tmp_path))
               for session in ("synthetic-a", "synthetic-b")]
    assert all(result is not None for result in results)
    assert results[0]["path"] != results[1]["path"]
    assert [json.loads(Path(result["path"]).read_text(encoding="utf-8"))["session_id"]
            for result in results] == ["synthetic-a", "synthetic-b"]


def test_replacement_does_not_touch_preexisting_clock_temp(tmp_path, monkeypatch):
    freeze_clock(monkeypatch)
    target = tmp_path / "payload.json"
    target.write_text(json.dumps({"content": "old"}), encoding="utf-8")
    foreign = tmp_path / "payload.json.1.tmp"
    foreign.write_text("unrelated preexisting temp", encoding="utf-8")
    ext._replace_externalized_payload(target, {"content": "new"})
    assert json.loads(target.read_text(encoding="utf-8"))["content"] == "new"
    assert foreign.read_text(encoding="utf-8") == "unrelated preexisting temp"


def test_streamed_reassignment_does_not_touch_preexisting_clock_temp(tmp_path, monkeypatch, config):
    freeze_clock(monkeypatch)
    content = "synthetic-content-" * 256
    result = ext.externalize_ingest_payload(content, session_id="old", config=config, hermes_home=str(tmp_path))
    assert result is not None
    path = Path(result["path"])
    foreign = path.with_name(path.name + ".1.tmp")
    foreign.write_text("unrelated preexisting temp", encoding="utf-8")
    monkeypatch.setattr(ext, "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES", 128)
    assert ext._reassign_oversize_externalized_payload(path, old_session_id="old", new_session_id="new", storage_dir=path.parent)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "new" and payload["content"] == content
    assert foreign.read_text(encoding="utf-8") == "unrelated preexisting temp"


def test_batch_order_with_frozen_wall_clock(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module.time, "time", lambda: 1234.0)
    store = MessageStore(tmp_path / "synthetic.db")
    try:
        ids = store.append_batch("synthetic", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
        rows = store._conn.execute("SELECT timestamp FROM messages ORDER BY store_id").fetchall()
        assert len(ids) == 2 and rows[0][0] < rows[1][0]
    finally:
        store.close()


def _directory_fd_or_skip(path):
    if os.name == "nt" or os.open not in os.supports_dir_fd:
        pytest.skip("directory-relative branch requires native dir_fd support")
    return os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))


@pytest.mark.parametrize("use_dir_fd", [False, True])
def test_selected_uuid_collision_preserves_foreign_temp_and_target(tmp_path, monkeypatch, use_dir_fd):
    target = tmp_path / "payload.json"
    original = b'{"content": "original target"}'
    target.write_bytes(original)
    suffix = "f" * 16
    monkeypatch.setattr(ext, "_unique_file_suffix", lambda: suffix)
    foreign = target.with_name(target.name + "." + suffix + ".tmp")
    sentinel = b"foreign selected-UUID collision sentinel"
    foreign.write_bytes(sentinel)
    directory_fd = _directory_fd_or_skip(tmp_path) if use_dir_fd else None
    try:
        with pytest.raises(FileExistsError):
            ext._replace_externalized_payload(target, {"content": "new"}, dir_fd=directory_fd)
        assert target.read_bytes() == original
        assert foreign.read_bytes() == sentinel
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


@pytest.mark.parametrize("use_dir_fd", [False, True])
def test_streamed_selected_uuid_collision_preserves_foreign_temp_and_target(tmp_path, monkeypatch, config, use_dir_fd):
    result = ext.externalize_ingest_payload("synthetic-content-" * 256, session_id="old", config=config, hermes_home=str(tmp_path))
    assert result is not None
    target = Path(result["path"])
    original = target.read_bytes()
    suffix = "e" * 16
    monkeypatch.setattr(ext, "_unique_file_suffix", lambda: suffix)
    foreign = target.with_name(target.name + "." + suffix + ".tmp")
    sentinel = b"foreign streamed selected-UUID collision sentinel"
    foreign.write_bytes(sentinel)
    monkeypatch.setattr(ext, "_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES", 128)
    real_open = ext._open_legacy_externalized_payload

    def choose_branch(path, *, storage_dir):
        opened = real_open(path, storage_dir=storage_dir)
        assert opened is not None
        fd, directory_fd, metadata = opened
        if use_dir_fd and directory_fd is None:
            os.close(fd)
            pytest.skip("directory-relative stream branch unavailable")
        if not use_dir_fd and directory_fd is not None:
            os.close(directory_fd)
            directory_fd = None
        return fd, directory_fd, metadata

    monkeypatch.setattr(ext, "_open_legacy_externalized_payload", choose_branch)
    with pytest.raises(FileExistsError):
        ext._reassign_oversize_externalized_payload(target, old_session_id="old", new_session_id="new", storage_dir=target.parent)
    assert target.read_bytes() == original
    assert foreign.read_bytes() == sentinel


@pytest.mark.parametrize("use_dir_fd", [False, True])
def test_cleanup_retains_a_foreign_entry_replacing_created_temp(tmp_path, use_dir_fd):
    owned = tmp_path / "owned.tmp"
    foreign = tmp_path / "foreign.tmp"
    owned.write_bytes(b"created by operation")
    identity = owned.stat()
    foreign.write_bytes(b"foreign replacement sentinel")
    # The foreign inode was allocated while owned still existed, avoiding inode reuse.
    directory_fd = _directory_fd_or_skip(tmp_path) if use_dir_fd else None
    os.replace(foreign, owned)
    try:
        expected = (identity.st_dev, identity.st_ino)
        if use_dir_fd:
            ext._unlink_partial_payload_at(owned.name, dir_fd=directory_fd, expected_identity=expected)
        else:
            ext._unlink_partial_payload(owned, expected_identity=expected)
        assert owned.read_bytes() == b"foreign replacement sentinel"
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
