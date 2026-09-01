"""Helpers for optional large-payload externalization.

Externalization started as a pre-compaction serializer guard for oversized tool
outputs. The same durable payload format is also used by the ingest path so
obvious oversized content can be kept out of SQLite/FTS while still being
recoverable through the LCM inspection and expansion tools.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import logging
import math
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, BinaryIO, Dict

DEFAULT_LARGE_OUTPUT_DIRNAME = "lcm-large-outputs"
_EXTERNALIZED_REF_RE = re.compile(
    r"\[(?:Externalized|GC'd externalized) (?:tool output|payload):.*?;\s*ref=([^;\]\s]+)\]"
)
_EXTERNALIZED_SEARCH_HEADER_BYTES = 64 * 1024
_EXTERNALIZED_SEARCH_TAIL_BYTES = 64 * 1024
# Legacy readers need the whole JSON document, but must never allocate from an
# attacker-controlled file size without a ceiling. This is deliberately larger
# than normal provider-visible payloads while keeping one read memory-bounded.
_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES = 64 * 1024 * 1024


def _placeholder_metadata(value: Any) -> str:
    text = str(value or "?")
    safe = re.sub(r"[^A-Za-z0-9_.:/-]+", "-", text).strip("-")
    return (safe or "?")[:120]


logger = logging.getLogger(__name__)


def _tool_call_stub(tool_call_id: str) -> str:
    return (tool_call_id or "tool-result").replace("/", "-").replace(":", "-")[:48]


def _safe_stub(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", (value or "").strip())
    return (text or fallback)[:48]


def _content_digest_prefix(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()[:12]


def _preview_sha256(preview_prefix: Any) -> str:
    if not preview_prefix:
        return ""
    return hashlib.sha256(str(preview_prefix).encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    dir_fd = os.open(path, flags)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _missing_directory_components(path: Path) -> list[Path]:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return list(reversed(missing))


_WARNED_EXTERNALIZATION_PATHS: set[str] = set()


def _warn_externalization_path_outside_base(path: Path, allowed_base: Path) -> None:
    key = str(path)
    if key in _WARNED_EXTERNALIZATION_PATHS:
        return
    _WARNED_EXTERNALIZATION_PATHS.add(key)
    logger.warning(
        "LCM externalized-payload path %s is outside the hermes_home base %s; "
        "set LCM_HERMES_BASE_DIR to enforce strict containment",
        path,
        allowed_base,
    )


def get_large_output_storage_dir(config, hermes_home: str = "", *, create: bool) -> Path:
    configured = getattr(config, "large_output_externalization_path", "") or ""
    if configured:
        path = Path(configured).expanduser().resolve()
        # Check containment for configured paths when LCM_HERMES_BASE_DIR is set
        env_base = os.environ.get("LCM_HERMES_BASE_DIR")
        if env_base:
            allowed_base = Path(env_base).expanduser().resolve()
            try:
                path.relative_to(allowed_base)
            except ValueError:
                raise ValueError(f"Path {path} is not within allowed base {allowed_base}")
        elif hermes_home:
            # No explicit base configured: hermes_home is the natural default
            # containment root. A configured path may legitimately point to
            # another volume, so warn (once) rather than break a running
            # deployment; set LCM_HERMES_BASE_DIR to enforce strictly.
            allowed_base = Path(hermes_home).expanduser().resolve()
            try:
                path.relative_to(allowed_base)
            except ValueError:
                _warn_externalization_path_outside_base(path, allowed_base)
    else:
        base = Path(hermes_home).expanduser().resolve() if hermes_home else Path("~/.hermes").expanduser().resolve()
        path = base / DEFAULT_LARGE_OUTPUT_DIRNAME
        # Check containment within allowed base for default/hermes_home-based paths
        # Only enforced when LCM_HERMES_BASE_DIR is explicitly set
        env_base = os.environ.get("LCM_HERMES_BASE_DIR")
        if env_base:
            allowed_base = Path(env_base).expanduser().resolve()
            try:
                path.relative_to(allowed_base)
            except ValueError:
                raise ValueError(f"Path {path} is not within allowed base {allowed_base}")
    if create:
        missing_dirs = _missing_directory_components(path)
        path.mkdir(parents=True, exist_ok=True)
        for created_dir in missing_dirs:
            _fsync_directory(created_dir.parent)
        try:
            path.chmod(0o700)
        except OSError as exc:
            logger.warning("Could not restrict LCM externalized payload directory permissions for %s: %s", path, exc)
    return path


def _unlink_partial_payload(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove partial LCM externalized payload %s: %s", path, exc)


def _unlink_partial_payload_at(name: str, *, dir_fd: int) -> None:
    try:
        os.unlink(name, dir_fd=dir_fd)
    except FileNotFoundError:
        pass
    except (OSError, TypeError, NotImplementedError) as exc:
        logger.warning("Could not remove partial LCM externalized payload %s: %s", name, exc)


def _write_externalized_payload(path: Path, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except OSError:
        _unlink_partial_payload(path)
        raise
    finally:
        if fd >= 0:
            os.close(fd)


def _write_externalized_payload_at(name: str, payload: Dict[str, Any], *, dir_fd: int) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(name, flags, 0o600, dir_fd=dir_fd)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(dir_fd)
    except (OSError, TypeError, NotImplementedError):
        _unlink_partial_payload_at(name, dir_fd=dir_fd)
        raise
    finally:
        if fd >= 0:
            os.close(fd)


def _revalidate_payload_name(
    name: str,
    *,
    dir_fd: int,
    expected_identity: tuple[int, int],
) -> None:
    current = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != expected_identity:
        raise OSError("externalized payload directory entry changed before replacement")


def _replace_externalized_payload(
    path: Path,
    payload: Dict[str, Any],
    *,
    dir_fd: int | None = None,
    expected_identity: tuple[int, int] | None = None,
) -> None:
    tmp_path = path.with_name(f"{path.name}.{time.time_ns():x}.tmp")
    if dir_fd is not None:
        try:
            _write_externalized_payload_at(tmp_path.name, payload, dir_fd=dir_fd)
            if expected_identity is not None:
                _revalidate_payload_name(
                    path.name,
                    dir_fd=dir_fd,
                    expected_identity=expected_identity,
                )
            os.replace(tmp_path.name, path.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            os.fsync(dir_fd)
        except (OSError, TypeError, NotImplementedError):
            _unlink_partial_payload_at(tmp_path.name, dir_fd=dir_fd)
            raise
        return
    try:
        _write_externalized_payload(tmp_path, payload)
        tmp_path.replace(path)
        _fsync_directory(path.parent)
    except OSError:
        _unlink_partial_payload(tmp_path)
        raise


def _persisted_output_marker_entry_from_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not metadata:
        return None
    source_path = metadata.get("persisted_output_source_path")
    expected_chars = metadata.get("persisted_output_expected_chars")
    preview_sha256 = metadata.get("persisted_output_preview_sha256") or _preview_sha256(
        metadata.get("persisted_output_preview_prefix")
    )
    redacted_preview_sha256 = metadata.get("persisted_output_redacted_preview_sha256")
    file_size = metadata.get("persisted_output_file_size")
    file_mtime_ns = metadata.get("persisted_output_file_mtime_ns")
    file_ctime_ns = metadata.get("persisted_output_file_ctime_ns")

    if source_path is None or expected_chars is None:
        return None
    try:
        expected_chars = int(expected_chars)
    except (TypeError, ValueError):
        return None
    source_path = str(source_path)
    if not source_path:
        return None
    entry = {
        "source_path": source_path,
        "expected_chars": expected_chars,
    }
    if preview_sha256:
        entry["preview_sha256"] = str(preview_sha256)
    if redacted_preview_sha256:
        entry["redacted_preview_sha256"] = str(redacted_preview_sha256)
    if file_size is not None:
        try:
            entry["file_size"] = int(file_size)
        except (TypeError, ValueError):
            pass
    if file_mtime_ns is not None:
        try:
            entry["file_mtime_ns"] = int(file_mtime_ns)
        except (TypeError, ValueError):
            pass
    if file_ctime_ns is not None:
        try:
            entry["file_ctime_ns"] = int(file_ctime_ns)
        except (TypeError, ValueError):
            pass
    return entry


def _persisted_output_marker_entries(
    payload: Dict[str, Any],
    *,
    include_legacy_preview_prefix: bool = False,
) -> list[Dict[str, Any]]:
    entries: list[Dict[str, Any]] = []
    seen: set[tuple[str, int, str, str, int | None, int | None, int | None]] = set()
    try:
        from .ingest_protection import _has_lossy_sensitive_redaction
    except Exception:
        _has_lossy_sensitive_redaction = None  # type: ignore[assignment]
    payload_content_has_lossy_redaction = bool(
        _has_lossy_sensitive_redaction
        and _has_lossy_sensitive_redaction(str(payload.get("content") or ""))
    )

    def add(
        source_path: Any,
        expected_chars: Any,
        preview_prefix: Any = None,
        preview_sha256: Any = None,
        redacted_preview_sha256: Any = None,
        file_size: Any = None,
        file_mtime_ns: Any = None,
        file_ctime_ns: Any = None,
    ) -> None:
        if source_path is None or expected_chars is None:
            return
        try:
            chars = int(expected_chars)
        except (TypeError, ValueError):
            return
        source = str(source_path)
        if not source:
            return
        preview_digest = "" if payload_content_has_lossy_redaction else str(preview_sha256 or "")
        if not preview_digest and preview_prefix and not payload_content_has_lossy_redaction:
            preview_digest = _preview_sha256(preview_prefix)
        redacted_preview_digest = str(redacted_preview_sha256 or "")
        try:
            size = int(file_size) if file_size is not None else None
        except (TypeError, ValueError):
            size = None
        try:
            mtime_ns = int(file_mtime_ns) if file_mtime_ns is not None else None
        except (TypeError, ValueError):
            mtime_ns = None
        try:
            ctime_ns = int(file_ctime_ns) if file_ctime_ns is not None else None
        except (TypeError, ValueError):
            ctime_ns = None
        key = (source, chars, preview_digest, redacted_preview_digest, size, mtime_ns, ctime_ns)
        if key in seen:
            return
        seen.add(key)
        entry = {"source_path": source, "expected_chars": chars}
        if preview_digest:
            entry["preview_sha256"] = preview_digest
        if redacted_preview_digest:
            entry["redacted_preview_sha256"] = redacted_preview_digest
        if size is not None:
            entry["file_size"] = size
        if mtime_ns is not None:
            entry["file_mtime_ns"] = mtime_ns
        if ctime_ns is not None:
            entry["file_ctime_ns"] = ctime_ns
        if include_legacy_preview_prefix and preview_prefix:
            entry["legacy_preview_prefix"] = str(preview_prefix)
        entries.append(entry)

    add(
        payload.get("persisted_output_source_path"),
        payload.get("persisted_output_expected_chars"),
        payload.get("persisted_output_preview_prefix"),
        payload.get("persisted_output_preview_sha256"),
        payload.get("persisted_output_redacted_preview_sha256"),
        payload.get("persisted_output_file_size"),
        payload.get("persisted_output_file_mtime_ns"),
        payload.get("persisted_output_file_ctime_ns"),
    )
    markers = payload.get("persisted_output_markers")
    if isinstance(markers, list):
        for marker in markers:
            if not isinstance(marker, dict):
                continue
            add(
                marker.get("source_path"),
                marker.get("expected_chars"),
                marker.get("preview_prefix"),
                marker.get("preview_sha256"),
                marker.get("redacted_preview_sha256"),
                marker.get("file_size"),
                marker.get("file_mtime_ns"),
                marker.get("file_ctime_ns"),
            )
    return entries


def _safe_persisted_output_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    marker = _persisted_output_marker_entry_from_metadata(metadata)
    if marker is None:
        return {}
    safe = {
        "persisted_output_source_path": marker["source_path"],
        "persisted_output_expected_chars": marker["expected_chars"],
    }
    if marker.get("preview_sha256"):
        safe["persisted_output_preview_sha256"] = marker["preview_sha256"]
    if marker.get("redacted_preview_sha256"):
        safe["persisted_output_redacted_preview_sha256"] = marker["redacted_preview_sha256"]
    if marker.get("file_size") is not None:
        safe["persisted_output_file_size"] = marker["file_size"]
    if marker.get("file_mtime_ns") is not None:
        safe["persisted_output_file_mtime_ns"] = marker["file_mtime_ns"]
    if marker.get("file_ctime_ns") is not None:
        safe["persisted_output_file_ctime_ns"] = marker["file_ctime_ns"]
    return safe


def _redacted_legacy_preview_sha256(marker: Dict[str, Any], config) -> str:
    legacy_preview_prefix = marker.get("legacy_preview_prefix")
    if not legacy_preview_prefix:
        return ""
    try:
        from .ingest_protection import _has_lossy_sensitive_redaction, redact_sensitive_value
    except Exception:
        return ""
    redacted_preview = redact_sensitive_value(
        str(legacy_preview_prefix),
        config,
        parse_json_strings=False,
    )
    if _has_lossy_sensitive_redaction(str(redacted_preview)):
        return ""
    return _preview_sha256(redacted_preview)


def _persisted_output_marker_matches_preview_digest(
    marker: Dict[str, Any],
    preview_sha256: str,
    *,
    config,
    allow_redacted_preview_match: bool = True,
) -> bool:
    if not preview_sha256:
        return False
    if preview_sha256 == str(marker.get("preview_sha256") or ""):
        return True
    if not allow_redacted_preview_match:
        return False
    if preview_sha256 == str(marker.get("redacted_preview_sha256") or ""):
        return True
    return preview_sha256 == _redacted_legacy_preview_sha256(marker, config)


def _marker_file_not_newer_than_payload(marker: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    """Return true when a live marker file still matches the durable payload era."""
    source_path = marker.get("source_path")
    created_at = payload.get("created_at")
    if not source_path or created_at is None:
        return False
    try:
        current_stat = Path(str(source_path)).stat()
        marker_size = marker.get("file_size")
        marker_mtime_ns = marker.get("file_mtime_ns")
        marker_ctime_ns = marker.get("file_ctime_ns")
        if marker_size is None or marker_mtime_ns is None or marker_ctime_ns is None:
            return False
        return (
            int(current_stat.st_size) == int(marker_size)
            and int(current_stat.st_mtime_ns) == int(marker_mtime_ns)
            and int(current_stat.st_ctime_ns) == int(marker_ctime_ns)
        )
    except (OSError, TypeError, ValueError):
        return False


def _sanitize_persisted_output_marker_metadata(payload: Dict[str, Any]) -> bool:
    """Remove legacy raw persisted-output previews from durable metadata.

    Older payloads stored marker preview text as ``preview_prefix``. That preview
    is enough to leak secrets when the recovered content itself has been
    redacted, so durable metadata now stores only SHA-256 proofs. This helper is
    intentionally tolerant of old payloads and rewrites them opportunistically
    when they are touched again.
    """
    changed = False
    try:
        from .ingest_protection import _has_lossy_sensitive_redaction
    except Exception:
        _has_lossy_sensitive_redaction = None  # type: ignore[assignment]
    payload_content_has_lossy_redaction = bool(
        _has_lossy_sensitive_redaction
        and _has_lossy_sensitive_redaction(str(payload.get("content") or ""))
    )
    entries = _persisted_output_marker_entries(payload)

    if "persisted_output_preview_prefix" in payload:
        payload.pop("persisted_output_preview_prefix", None)
        changed = True

    marker_list = payload.get("persisted_output_markers")
    if entries and marker_list != entries:
        payload["persisted_output_markers"] = entries
        changed = True

    first_digest = next((entry.get("preview_sha256") for entry in entries if entry.get("preview_sha256")), "")
    if first_digest and payload.get("persisted_output_preview_sha256") != first_digest:
        payload["persisted_output_preview_sha256"] = first_digest
        changed = True
    elif not first_digest and payload_content_has_lossy_redaction and "persisted_output_preview_sha256" in payload:
        payload.pop("persisted_output_preview_sha256", None)
        changed = True

    first_redacted_digest = next(
        (entry.get("redacted_preview_sha256") for entry in entries if entry.get("redacted_preview_sha256")),
        "",
    )
    if first_redacted_digest and payload.get("persisted_output_redacted_preview_sha256") != first_redacted_digest:
        payload["persisted_output_redacted_preview_sha256"] = first_redacted_digest
        changed = True

    first_file_size = next((entry.get("file_size") for entry in entries if entry.get("file_size") is not None), None)
    if first_file_size is not None and payload.get("persisted_output_file_size") != first_file_size:
        payload["persisted_output_file_size"] = first_file_size
        changed = True

    first_file_mtime_ns = next((entry.get("file_mtime_ns") for entry in entries if entry.get("file_mtime_ns") is not None), None)
    if first_file_mtime_ns is not None and payload.get("persisted_output_file_mtime_ns") != first_file_mtime_ns:
        payload["persisted_output_file_mtime_ns"] = first_file_mtime_ns
        changed = True

    first_file_ctime_ns = next((entry.get("file_ctime_ns") for entry in entries if entry.get("file_ctime_ns") is not None), None)
    if first_file_ctime_ns is not None and payload.get("persisted_output_file_ctime_ns") != first_file_ctime_ns:
        payload["persisted_output_file_ctime_ns"] = first_file_ctime_ns
        changed = True

    return changed


def _merge_persisted_output_marker_metadata(payload: Dict[str, Any], metadata: Dict[str, Any] | None) -> bool:
    changed = _sanitize_persisted_output_marker_metadata(payload)
    marker = _persisted_output_marker_entry_from_metadata(metadata)
    if marker is not None:
        try:
            from .ingest_protection import _has_lossy_sensitive_redaction
        except Exception:
            _has_lossy_sensitive_redaction = None  # type: ignore[assignment]
        if _has_lossy_sensitive_redaction and _has_lossy_sensitive_redaction(str(payload.get("content") or "")):
            marker.pop("preview_sha256", None)
    if marker is None:
        return changed
    entries = _persisted_output_marker_entries(payload)
    key = (
        marker["source_path"],
        marker["expected_chars"],
        marker.get("preview_sha256", ""),
        marker.get("redacted_preview_sha256", ""),
        marker.get("file_size"),
        marker.get("file_mtime_ns"),
        marker.get("file_ctime_ns"),
    )
    if any(
        (
            entry["source_path"],
            entry["expected_chars"],
            entry.get("preview_sha256", ""),
            entry.get("redacted_preview_sha256", ""),
            entry.get("file_size"),
            entry.get("file_mtime_ns"),
            entry.get("file_ctime_ns"),
        ) == key
        for entry in entries
    ):
        return changed
    entries.append(marker)
    payload["persisted_output_markers"] = entries
    payload.setdefault("persisted_output_source_path", marker["source_path"])
    payload.setdefault("persisted_output_expected_chars", marker["expected_chars"])
    if marker.get("preview_sha256"):
        payload.setdefault("persisted_output_preview_sha256", marker["preview_sha256"])
    if marker.get("redacted_preview_sha256"):
        payload.setdefault("persisted_output_redacted_preview_sha256", marker["redacted_preview_sha256"])
    if marker.get("file_size") is not None:
        payload.setdefault("persisted_output_file_size", marker["file_size"])
    if marker.get("file_mtime_ns") is not None:
        payload.setdefault("persisted_output_file_mtime_ns", marker["file_mtime_ns"])
    if marker.get("file_ctime_ns") is not None:
        payload.setdefault("persisted_output_file_ctime_ns", marker["file_ctime_ns"])
    return True


def resolve_large_output_storage_dir(config, hermes_home: str = "") -> Path:
    return get_large_output_storage_dir(config, hermes_home=hermes_home, create=True)


def _externalized_summary(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ref": path.name,
        "kind": payload.get("kind", "tool_result"),
        "tool_call_id": payload.get("tool_call_id", ""),
        "role": payload.get("role", ""),
        "session_id": payload.get("session_id", ""),
        "field_path": payload.get("field_path", ""),
        "content_chars": payload.get("content_chars", len(payload.get("content", ""))),
        "content_bytes": payload.get("content_bytes", len((payload.get("content", "") or "").encode("utf-8"))),
        "created_at": payload.get("created_at"),
    }


def _same_file_identity(left, right) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _same_owner(left, right) -> bool:
    left_uid = getattr(left, "st_uid", None)
    right_uid = getattr(right, "st_uid", None)
    return left_uid is None or right_uid is None or left_uid == right_uid


def _has_single_link(file_stat) -> bool:
    return getattr(file_stat, "st_nlink", 1) == 1


def _open_legacy_externalized_payload(
    path: Path,
    *,
    storage_dir: Path,
) -> tuple[int, int, os.stat_result] | None:
    """Open one legacy sidecar through verified directory-relative descriptors.

    The pre-open ``stat`` and post-open ``fstat`` identity check closes the
    regular-file-to-symlink replacement window even where ``O_NOFOLLOW`` is not
    available. Opening relative to the already verified directory descriptor
    keeps the object lookup contained to the configured payload store. The
    caller owns both returned descriptors.
    """
    if path.parent != storage_dir or not path.name or Path(path.name).name != path.name:
        return None

    dir_fd = -1
    payload_fd = -1
    try:
        expected_dir = os.lstat(storage_dir)
        if stat.S_ISLNK(expected_dir.st_mode) or not stat.S_ISDIR(expected_dir.st_mode):
            return None
        dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        dir_flags |= getattr(os, "O_BINARY", 0)
        dir_fd = os.open(storage_dir, dir_flags)
        opened_dir = os.fstat(dir_fd)
        if not stat.S_ISDIR(opened_dir.st_mode) or not _same_file_identity(expected_dir, opened_dir):
            return None

        expected_payload = os.stat(path.name, dir_fd=dir_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(expected_payload.st_mode)
            or not _same_owner(expected_payload, opened_dir)
            or not _has_single_link(expected_payload)
        ):
            return None

        if expected_payload.st_size < 0:
            return None

        payload_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        payload_flags |= getattr(os, "O_BINARY", 0)
        payload_fd = os.open(path.name, payload_flags, dir_fd=dir_fd)
        opened_payload = os.fstat(payload_fd)
        if (
            not stat.S_ISREG(opened_payload.st_mode)
            or not _same_file_identity(expected_payload, opened_payload)
            or not _same_owner(opened_payload, opened_dir)
            or not _has_single_link(opened_payload)
            or opened_payload.st_size < 0
        ):
            return None

        result = (payload_fd, dir_fd, opened_payload)
        payload_fd = -1
        dir_fd = -1
        return result
    except (OSError, TypeError, NotImplementedError):
        return None
    finally:
        if payload_fd >= 0:
            os.close(payload_fd)
        if dir_fd >= 0:
            os.close(dir_fd)


def _read_legacy_externalized_payload_json_with_directory(
    path: Path,
    *,
    storage_dir: Path,
) -> tuple[Dict[str, Any], int, os.stat_result] | None:
    """Read one legacy sidecar fully, subject to the allocation ceiling."""
    opened = _open_legacy_externalized_payload(path, storage_dir=storage_dir)
    if opened is None:
        return None
    payload_fd, dir_fd, opened_payload = opened
    try:
        max_bytes = max(1, int(_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES))
        if opened_payload.st_size > max_bytes:
            return None

        with os.fdopen(payload_fd, "rb") as handle:
            payload_fd = -1
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes:
            return None
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, dict):
            return None
        result = (decoded, dir_fd, opened_payload)
        dir_fd = -1
        return result
    except (OSError, TypeError, UnicodeDecodeError, ValueError, json.JSONDecodeError, NotImplementedError):
        return None
    finally:
        if payload_fd >= 0:
            os.close(payload_fd)
        if dir_fd >= 0:
            os.close(dir_fd)


def _read_legacy_externalized_payload_json(
    path: Path,
    *,
    storage_dir: Path,
) -> Dict[str, Any] | None:
    opened = _read_legacy_externalized_payload_json_with_directory(path, storage_dir=storage_dir)
    if opened is None:
        return None
    payload, dir_fd, _payload_stat = opened
    try:
        return payload
    finally:
        os.close(dir_fd)


def _build_externalized_placeholder(summary: Dict[str, Any]) -> str:
    kind = _placeholder_metadata(summary.get("kind", "tool_result") or "tool_result")
    if kind != "tool_result":
        role = _placeholder_metadata(summary.get("role") or "?")
        return (
            f"[Externalized payload: kind={kind}; role={role}; "
            f"chars={summary.get('content_chars', 0)}; bytes={summary.get('content_bytes', 0)}; "
            f"ref={summary.get('ref', '')}]"
        )
    return (
        f"[Externalized tool output: tool_call_id={_placeholder_metadata(summary.get('tool_call_id') or '?')}; "
        f"chars={summary.get('content_chars', 0)}; bytes={summary.get('content_bytes', 0)}; ref={summary.get('ref', '')}]"
    )


def build_transcript_gc_placeholder(summary: Dict[str, Any]) -> str:
    kind = _placeholder_metadata(summary.get("kind", "tool_result") or "tool_result")
    if kind != "tool_result":
        role = _placeholder_metadata(summary.get("role") or "?")
        return (
            f"[GC'd externalized payload: kind={kind}; role={role}; "
            f"chars={summary.get('content_chars', 0)}; ref={summary.get('ref', '')}]"
        )
    return (
        f"[GC'd externalized tool output: tool_call_id={_placeholder_metadata(summary.get('tool_call_id') or '?')}; "
        f"chars={summary.get('content_chars', 0)}; ref={summary.get('ref', '')}]"
    )


def extract_externalized_ref(text: str) -> str | None:
    refs = extract_externalized_refs(text)
    return refs[0] if refs else None


def extract_externalized_refs(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    refs: list[str] = []
    for match in _EXTERNALIZED_REF_RE.finditer(text):
        ref = match.group(1).strip()
        if not ref or "/" in ref or "\\" in ref or Path(ref).name != ref or ref in refs:
            continue
        refs.append(ref)
    return refs


def is_externalized_placeholder(text: str) -> bool:
    """Return true for a compact legacy externalized payload/tool marker."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > 512:
        return False
    return bool(_EXTERNALIZED_REF_RE.fullmatch(stripped))


def load_externalized_payload(ref: str, *, config, hermes_home: str = "") -> Dict[str, Any] | None:
    if not ref or Path(ref).name != ref:
        return None
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    if not storage_dir.exists() or not storage_dir.is_dir():
        return None
    path = storage_dir / ref
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    summary = _externalized_summary(path, payload)
    summary["content"] = payload.get("content", "")
    return summary


def _decode_json_string_fragment(raw: bytes) -> str:
    """Decode a possibly truncated JSON string without reading beyond its bound."""
    candidate = raw
    while candidate:
        try:
            value = json.loads(b'"' + candidate + b'"')
        except (UnicodeDecodeError, json.JSONDecodeError):
            candidate = candidate[:-1]
            continue
        return value if isinstance(value, str) else ""
    return ""


def _numeric_json_field(data: bytes, field: str) -> int | float | None:
    pattern = re.compile(rb'"' + re.escape(field.encode("ascii")) + rb'"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)')
    match = pattern.search(data)
    if match is None:
        return None
    text = match.group(1).decode("ascii")
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        return None


def _normalized_content_size(value: int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
    try:
        normalized = int(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return normalized if normalized >= 0 else None


def _parse_top_level_json_string_fields_before_content(text: str) -> tuple[dict[str, Any], int | None]:
    decoder = json.JSONDecoder()
    fields: dict[str, Any] = {}
    length = len(text)
    index = 0

    def skip_json_whitespace(pos: int) -> int:
        while pos < length and text[pos] in " \t\n\r":
            pos += 1
        return pos

    index = skip_json_whitespace(index)
    if index >= length or text[index] != "{":
        return fields, None
    index += 1

    while True:
        index = skip_json_whitespace(index)
        if index >= length or text[index] == "}":
            return fields, None
        if text[index] != '"':
            return fields, None
        try:
            key, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return fields, None
        if not isinstance(key, str):
            return fields, None
        index = skip_json_whitespace(index)
        if index >= length or text[index] != ":":
            return fields, None
        index += 1
        index = skip_json_whitespace(index)
        if key == "content":
            return fields, index if index < length and text[index] == '"' else None
        if index >= length:
            return fields, None
        try:
            value, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return fields, None
        if isinstance(value, str):
            fields[key] = value
        elif key == "kind":
            # Preserve an explicitly invalid kind so oversized readers cannot
            # mistake it for the supported missing-kind legacy shape.
            fields[key] = value
        elif key == "session_id":
            fields.pop(key, None)
        index = skip_json_whitespace(index)
        if index >= length:
            return fields, None
        if text[index] == ",":
            index += 1
            continue
        if text[index] == "}":
            return fields, None
        return fields, None


def _inspect_top_level_json_string_fields_before_content(text: str) -> tuple[dict[str, Any], bool]:
    fields, content_quote_index = _parse_top_level_json_string_fields_before_content(text)
    return fields, content_quote_index is not None


def _read_externalized_payload_metadata_prefix_from_handle(
    handle: BinaryIO,
    *,
    max_read_bytes: int,
) -> tuple[str, dict[str, Any], bool, bool]:
    """Read through the opening quote of the top-level content string."""
    prefix = bytearray()
    text_parts: list[str] = []
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    prefix_truncated = False
    read_limit = max(1, int(max_read_bytes))
    fields: dict[str, Any] = {}

    while len(prefix) < read_limit:
        chunk = handle.read(min(4096, read_limit - len(prefix)))
        if not chunk:
            break
        prefix.extend(chunk)
        try:
            decoded = decoder.decode(chunk, final=False)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if decoded:
            text_parts.append(decoded)
        prefix_text = "".join(text_parts)
        fields, content_quote_index = _parse_top_level_json_string_fields_before_content(prefix_text)
        if content_quote_index is not None:
            metadata_prefix_text = prefix_text[: content_quote_index + 1]
            return metadata_prefix_text, fields, True, False

    prefix_truncated = len(prefix) >= read_limit and bool(handle.read(1))
    if not prefix_truncated:
        try:
            final_text = decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if final_text:
            text_parts.append(final_text)
    return "".join(text_parts), fields, False, prefix_truncated


def read_externalized_payload_metadata_prefix(
    path: Path,
    *,
    max_read_bytes: int,
) -> tuple[str, bool, bool]:
    with path.open("rb") as handle:
        prefix_text, _fields, content_key_seen, prefix_truncated = (
            _read_externalized_payload_metadata_prefix_from_handle(
                handle,
                max_read_bytes=max_read_bytes,
            )
        )
    return prefix_text, content_key_seen, prefix_truncated


def read_externalized_payload_search_prefix(
    ref: str,
    *,
    config,
    hermes_home: str = "",
    max_content_bytes: int = 512_000,
) -> Dict[str, Any]:
    """Read bounded metadata and the first bytes of one payload's content.

    This deliberately does not deserialize the full sidecar. It reads at most a
    small header, ``max_content_bytes`` from the JSON content string, and a
    small tail for size metadata.
    """
    if (
        not ref
        or Path(ref).name != ref
        or "/" in ref
        or "\\" in ref
        or not ref.endswith(".json")
    ):
        return {"status": "invalid_ref", "ref": ref}
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    path = storage_dir / ref
    try:
        expected_stat = os.lstat(path)
    except FileNotFoundError:
        return {"status": "missing", "ref": ref}
    except OSError:
        return {"status": "unreadable", "ref": ref}
    if stat.S_ISLNK(expected_stat.st_mode):
        return {"status": "symlink", "ref": ref}
    if not stat.S_ISREG(expected_stat.st_mode):
        return {"status": "missing", "ref": ref}

    content_limit = max(1, min(int(max_content_bytes), 512_000))
    fd = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_BINARY", 0)
        fd = os.open(path, flags)
        opened_stat = os.fstat(fd)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or (opened_stat.st_dev, opened_stat.st_ino)
            != (expected_stat.st_dev, expected_stat.st_ino)
        ):
            return {"status": "unreadable", "ref": ref}
        file_size = opened_stat.st_size
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            metadata_prefix_text, metadata_fields, content_key_seen, _prefix_truncated = (
                _read_externalized_payload_metadata_prefix_from_handle(
                    handle,
                    max_read_bytes=_EXTERNALIZED_SEARCH_HEADER_BYTES,
                )
            )
            if not content_key_seen:
                return {"status": "unsupported_payload", "ref": ref}
            handle.seek(len(metadata_prefix_text.encode("utf-8")))
            raw = bytearray()
            escaped = False
            closed = False
            while len(raw) < content_limit:
                chunk = handle.read(min(64 * 1024, content_limit - len(raw)))
                if not chunk:
                    break
                for byte in chunk:
                    if not escaped and byte == 0x22:
                        closed = True
                        break
                    raw.append(byte)
                    if escaped:
                        escaped = False
                    elif byte == 0x5C:
                        escaped = True
                if closed:
                    break
            handle.seek(max(0, file_size - _EXTERNALIZED_SEARCH_TAIL_BYTES))
            tail = handle.read(_EXTERNALIZED_SEARCH_TAIL_BYTES)
    except (OSError, ValueError):
        return {"status": "unreadable", "ref": ref}
    finally:
        if fd >= 0:
            os.close(fd)

    content = _decode_json_string_fragment(bytes(raw))
    original_bytes = _numeric_json_field(tail, "content_bytes")
    original_chars = _numeric_json_field(tail, "content_chars")
    created_at = _numeric_json_field(tail, "created_at")
    try:
        created_at_float = float(created_at or 0)
    except (OverflowError, ValueError):
        created_at_float = 0.0
    if not math.isfinite(created_at_float):
        created_at_float = 0.0
    return {
        "status": "ok",
        "ref": ref,
        "kind": metadata_fields.get("kind") or "tool_result",
        "tool_call_id": metadata_fields.get("tool_call_id", ""),
        "role": metadata_fields.get("role", ""),
        "session_id": metadata_fields.get("session_id", ""),
        "content": content,
        "content_scanned_bytes": len(raw),
        "original_content_bytes": _normalized_content_size(original_bytes),
        "original_content_chars": _normalized_content_size(original_chars),
        "created_at": created_at_float,
        "scan_truncated": not closed,
    }


class _StreamingJSONReader:
    """Small bounded-memory byte reader for validating oversized sidecars."""

    def __init__(self, handle: BinaryIO, *, start: int) -> None:
        handle.seek(start)
        self._handle = handle
        self._buffer = b""
        self._index = 0

    def peek(self) -> int | None:
        if self._index >= len(self._buffer):
            self._buffer = self._handle.read(64 * 1024)
            self._index = 0
            if not self._buffer:
                return None
        return self._buffer[self._index]

    def read(self) -> int | None:
        byte = self.peek()
        if byte is not None:
            self._index += 1
        return byte


def _stream_json_skip_whitespace(reader: _StreamingJSONReader) -> None:
    while (byte := reader.peek()) is not None and byte in b" \t\n\r":
        reader.read()


def _stream_json_is_digit(byte: int | None) -> bool:
    return byte is not None and ord("0") <= byte <= ord("9")


def _stream_json_read_string(
    reader: _StreamingJSONReader,
    *,
    capture_limit: int = 4096,
) -> str | None:
    if reader.read() != ord('"'):
        raise ValueError("invalid_json_string")
    raw = bytearray()
    truncated = False
    escaped = False
    unicode_digits_remaining = 0
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    while True:
        byte = reader.read()
        if byte is None:
            raise ValueError("truncated_json_string")
        if unicode_digits_remaining:
            if byte not in b"0123456789abcdefABCDEF":
                raise ValueError("invalid_json_unicode_escape")
            unicode_digits_remaining -= 1
        elif escaped:
            if byte == ord("u"):
                unicode_digits_remaining = 4
            elif byte not in b'"\\/bfnrt':
                raise ValueError("invalid_json_escape")
            escaped = False
        elif byte == ord("\\"):
            escaped = True
        elif byte == ord('"'):
            decoder.decode(b"", final=True)
            if truncated:
                return None
            value, end = json.JSONDecoder().raw_decode(
                (b'"' + bytes(raw) + b'"').decode("utf-8")
            )
            if end != len(raw) + 2 or not isinstance(value, str):
                raise ValueError("invalid_json_string")
            return value
        elif byte < 0x20:
            raise ValueError("invalid_json_control_character")
        decoder.decode(bytes((byte,)), final=False)
        if len(raw) < capture_limit:
            raw.append(byte)
        else:
            truncated = True


def _stream_json_read_number(reader: _StreamingJSONReader) -> str | None:
    captured = bytearray()
    truncated = False

    def consume() -> int:
        nonlocal truncated
        byte = reader.read()
        if byte is None:
            raise ValueError("truncated_json_number")
        if len(captured) < 128:
            captured.append(byte)
        else:
            truncated = True
        return byte

    if reader.peek() == ord("-"):
        consume()
    first = reader.peek()
    if first == ord("0"):
        consume()
        if _stream_json_is_digit(reader.peek()):
            raise ValueError("invalid_json_number")
    elif first is not None and ord("1") <= first <= ord("9"):
        consume()
        while _stream_json_is_digit(reader.peek()):
            consume()
    else:
        raise ValueError("invalid_json_number")
    if reader.peek() == ord("."):
        consume()
        if not _stream_json_is_digit(reader.peek()):
            raise ValueError("invalid_json_number")
        while _stream_json_is_digit(reader.peek()):
            consume()
    if reader.peek() in (ord("e"), ord("E")):
        consume()
        if reader.peek() in (ord("+"), ord("-")):
            consume()
        if not _stream_json_is_digit(reader.peek()):
            raise ValueError("invalid_json_number")
        while _stream_json_is_digit(reader.peek()):
            consume()
    return None if truncated else captured.decode("ascii")


def _stream_json_consume_literal(reader: _StreamingJSONReader, literal: bytes) -> None:
    for expected in literal:
        if reader.read() != expected:
            raise ValueError("invalid_json_literal")


def _stream_json_skip_value(reader: _StreamingJSONReader, *, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("json_nesting_too_deep")
    _stream_json_skip_whitespace(reader)
    byte = reader.peek()
    if byte == ord('"'):
        _stream_json_read_string(reader, capture_limit=0)
        return
    if byte == ord("{"):
        reader.read()
        _stream_json_skip_whitespace(reader)
        if reader.peek() == ord("}"):
            reader.read()
            return
        while True:
            _stream_json_read_string(reader, capture_limit=0)
            _stream_json_skip_whitespace(reader)
            if reader.read() != ord(":"):
                raise ValueError("invalid_json_object")
            _stream_json_skip_value(reader, depth=depth + 1)
            _stream_json_skip_whitespace(reader)
            separator = reader.read()
            if separator == ord("}"):
                return
            if separator != ord(","):
                raise ValueError("invalid_json_object")
            _stream_json_skip_whitespace(reader)
    if byte == ord("["):
        reader.read()
        _stream_json_skip_whitespace(reader)
        if reader.peek() == ord("]"):
            reader.read()
            return
        while True:
            _stream_json_skip_value(reader, depth=depth + 1)
            _stream_json_skip_whitespace(reader)
            separator = reader.read()
            if separator == ord("]"):
                return
            if separator != ord(","):
                raise ValueError("invalid_json_array")
            _stream_json_skip_whitespace(reader)
    if byte in (ord("-"), *range(ord("0"), ord("9") + 1)):
        _stream_json_read_number(reader)
        return
    if byte == ord("t"):
        _stream_json_consume_literal(reader, b"true")
        return
    if byte == ord("f"):
        _stream_json_consume_literal(reader, b"false")
        return
    if byte == ord("n"):
        _stream_json_consume_literal(reader, b"null")
        return
    raise ValueError("invalid_json_value")


def _stream_json_read_marker_object(reader: _StreamingJSONReader) -> Dict[str, Any] | None:
    if reader.read() != ord("{"):
        raise ValueError("invalid_marker_object")
    marker: Dict[str, Any] = {}
    _stream_json_skip_whitespace(reader)
    if reader.peek() == ord("}"):
        reader.read()
        return None
    while True:
        key = _stream_json_read_string(reader, capture_limit=256)
        _stream_json_skip_whitespace(reader)
        if reader.read() != ord(":"):
            raise ValueError("invalid_marker_object")
        _stream_json_skip_whitespace(reader)
        if key == "source_path" and reader.peek() == ord('"'):
            marker["source_path"] = _stream_json_read_string(reader)
        elif key == "expected_chars" and reader.peek() == ord('"'):
            marker["expected_chars"] = _stream_json_read_string(reader, capture_limit=128)
        elif key == "expected_chars" and reader.peek() in (
            ord("-"),
            *range(ord("0"), ord("9") + 1),
        ):
            marker["expected_chars"] = _stream_json_read_number(reader)
        else:
            _stream_json_skip_value(reader, depth=1)
        _stream_json_skip_whitespace(reader)
        separator = reader.read()
        if separator == ord("}"):
            break
        if separator != ord(","):
            raise ValueError("invalid_marker_object")
        _stream_json_skip_whitespace(reader)
    return _persisted_output_marker_entry_from_metadata(
        {
            "persisted_output_source_path": marker.get("source_path"),
            "persisted_output_expected_chars": marker.get("expected_chars"),
        }
    )


def _stream_json_read_marker_array(reader: _StreamingJSONReader) -> Dict[str, Any] | None:
    if reader.read() != ord("["):
        raise ValueError("invalid_marker_array")
    first_valid: Dict[str, Any] | None = None
    _stream_json_skip_whitespace(reader)
    if reader.peek() == ord("]"):
        reader.read()
        return None
    while True:
        if reader.peek() == ord("{"):
            marker = _stream_json_read_marker_object(reader)
            if first_valid is None and marker is not None:
                first_valid = marker
        else:
            _stream_json_skip_value(reader, depth=1)
        _stream_json_skip_whitespace(reader)
        separator = reader.read()
        if separator == ord("]"):
            return first_valid
        if separator != ord(","):
            raise ValueError("invalid_marker_array")
        _stream_json_skip_whitespace(reader)


def _stream_externalized_content_end(handle: BinaryIO, *, content_start: int) -> int | None:
    if content_start < 0:
        return None
    handle.seek(content_start)
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    escaped = False
    unicode_digits_remaining = 0
    while True:
        chunk_start = handle.tell()
        chunk = handle.read(64 * 1024)
        if not chunk:
            return None
        for index, byte in enumerate(chunk):
            if unicode_digits_remaining:
                if byte not in b"0123456789abcdefABCDEF":
                    return None
                unicode_digits_remaining -= 1
            elif escaped:
                if byte == ord("u"):
                    unicode_digits_remaining = 4
                elif byte not in b'"\\/bfnrt':
                    return None
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                decoder.decode(chunk[:index], final=True)
                return chunk_start + index + 1
            elif byte < 0x20:
                return None
        decoder.decode(chunk, final=False)


def _stream_externalized_payload_suffix_metadata(
    handle: BinaryIO,
    *,
    content_start: int,
) -> Dict[str, Any] | None:
    """Validate the complete sidecar and retain only bounded marker metadata."""
    content_end = _stream_externalized_content_end(handle, content_start=content_start)
    if content_end is None:
        return None
    reader = _StreamingJSONReader(handle, start=content_end)
    metadata: Dict[str, Any] = {}
    try:
        _stream_json_skip_whitespace(reader)
        if reader.peek() == ord("}"):
            reader.read()
        else:
            if reader.read() != ord(","):
                return None
            while True:
                _stream_json_skip_whitespace(reader)
                key = _stream_json_read_string(reader, capture_limit=256)
                _stream_json_skip_whitespace(reader)
                if reader.read() != ord(":"):
                    return None
                _stream_json_skip_whitespace(reader)
                if key == "persisted_output_markers" and reader.peek() == ord("["):
                    marker = _stream_json_read_marker_array(reader)
                    if marker is not None:
                        metadata["persisted_output_markers"] = [marker]
                elif key == "persisted_output_source_path" and reader.peek() == ord('"'):
                    metadata[key] = _stream_json_read_string(reader)
                elif key == "persisted_output_expected_chars" and reader.peek() == ord('"'):
                    metadata[key] = _stream_json_read_string(reader, capture_limit=128)
                elif key == "persisted_output_expected_chars" and reader.peek() in (
                    ord("-"),
                    *range(ord("0"), ord("9") + 1),
                ):
                    metadata[key] = _stream_json_read_number(reader)
                elif key == "kind":
                    if reader.peek() == ord('"'):
                        metadata[key] = _stream_json_read_string(reader, capture_limit=256)
                    else:
                        metadata[key] = None
                        _stream_json_skip_value(reader, depth=1)
                else:
                    _stream_json_skip_value(reader, depth=1)
                _stream_json_skip_whitespace(reader)
                separator = reader.read()
                if separator == ord("}"):
                    break
                if separator != ord(","):
                    return None
        _stream_json_skip_whitespace(reader)
        if reader.peek() is not None:
            return None
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return metadata


def _read_oversize_externalized_payload_metadata(
    path: Path,
    *,
    storage_dir: Path,
) -> Dict[str, Any] | None:
    """Stream-validate an oversized sidecar and retain bounded marker metadata."""
    opened = _open_legacy_externalized_payload(path, storage_dir=storage_dir)
    if opened is None:
        return None
    payload_fd, dir_fd, payload_stat = opened
    try:
        max_bytes = max(1, int(_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES))
        if payload_stat.st_size <= max_bytes:
            return None
        with os.fdopen(payload_fd, "rb") as handle:
            payload_fd = -1
            prefix, fields, content_key_seen, prefix_truncated = (
                _read_externalized_payload_metadata_prefix_from_handle(
                    handle,
                    max_read_bytes=_EXTERNALIZED_SEARCH_HEADER_BYTES,
                )
            )
            if not content_key_seen or prefix_truncated:
                return None
            suffix = _stream_externalized_payload_suffix_metadata(
                handle,
                content_start=len(prefix.encode("utf-8")),
            )
            if suffix is None:
                return None
            return {**fields, **suffix}
    except (OSError, TypeError, UnicodeDecodeError, ValueError, NotImplementedError):
        return None
    finally:
        if payload_fd >= 0:
            os.close(payload_fd)
        os.close(dir_fd)


def _replace_session_id_in_metadata_prefix(
    prefix: str,
    *,
    old_session_id: str,
    new_session_id: str,
) -> str | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r'"session_id"\s*:\s*', prefix):
        try:
            value, end = decoder.raw_decode(prefix, match.end())
        except json.JSONDecodeError:
            continue
        if value != old_session_id:
            continue
        replacement = json.dumps(new_session_id, ensure_ascii=False)
        return prefix[: match.end()] + replacement + prefix[end:]
    return None


def _reassign_oversize_externalized_payload(
    path: Path,
    *,
    old_session_id: str,
    new_session_id: str,
    storage_dir: Path,
) -> bool:
    """Rewrite only the bounded session-id prefix and stream the payload body."""
    opened = _open_legacy_externalized_payload(path, storage_dir=storage_dir)
    if opened is None:
        return False
    payload_fd, dir_fd, payload_stat = opened
    tmp_name = f"{path.name}.{time.time_ns():x}.tmp"
    tmp_fd = -1
    try:
        max_bytes = max(1, int(_LEGACY_EXTERNALIZED_PAYLOAD_READ_MAX_BYTES))
        if payload_stat.st_size <= max_bytes:
            return False
        with os.fdopen(payload_fd, "rb") as source:
            payload_fd = -1
            prefix, fields, content_key_seen, prefix_truncated = (
                _read_externalized_payload_metadata_prefix_from_handle(
                    source,
                    max_read_bytes=_EXTERNALIZED_SEARCH_HEADER_BYTES,
                )
            )
            if (
                not content_key_seen
                or prefix_truncated
                or fields.get("session_id") != old_session_id
            ):
                return False
            if _stream_externalized_payload_suffix_metadata(
                source,
                content_start=len(prefix.encode("utf-8")),
            ) is None:
                return False
            rewritten_prefix = _replace_session_id_in_metadata_prefix(
                prefix,
                old_session_id=old_session_id,
                new_session_id=new_session_id,
            )
            if rewritten_prefix is None:
                return False

            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            flags |= getattr(os, "O_CLOEXEC", 0)
            tmp_fd = os.open(tmp_name, flags, 0o600, dir_fd=dir_fd)
            with os.fdopen(tmp_fd, "wb") as destination:
                tmp_fd = -1
                prefix_bytes = prefix.encode("utf-8")
                destination.write(rewritten_prefix.encode("utf-8"))
                source.seek(len(prefix_bytes))
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())

        _revalidate_payload_name(
            path.name,
            dir_fd=dir_fd,
            expected_identity=(payload_stat.st_dev, payload_stat.st_ino),
        )
        os.replace(tmp_name, path.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        os.fsync(dir_fd)
        return True
    except (OSError, TypeError, UnicodeDecodeError, ValueError, NotImplementedError):
        _unlink_partial_payload_at(tmp_name, dir_fd=dir_fd)
        raise
    finally:
        if tmp_fd >= 0:
            os.close(tmp_fd)
        if payload_fd >= 0:
            os.close(payload_fd)
        os.close(dir_fd)


def externalized_tool_result_has_persisted_output_marker(ref: str, *, config, hermes_home: str = "") -> bool:
    if not ref or Path(ref).name != ref:
        return False
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    if not storage_dir.exists() or not storage_dir.is_dir():
        return False
    path = storage_dir / ref
    payload = _read_legacy_externalized_payload_json(path, storage_dir=storage_dir)
    if payload is None:
        payload = _read_oversize_externalized_payload_metadata(path, storage_dir=storage_dir)
    if payload is None:
        return False
    if payload.get("kind", "tool_result") != "tool_result":
        return False
    return bool(_persisted_output_marker_entries(payload))


def reassign_externalized_payloads(
    old_session_id: str,
    new_session_id: str,
    *,
    config,
    hermes_home: str = "",
) -> int:
    """Move externalized payload session metadata across a logical session boundary."""
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return 0
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    if not storage_dir.exists() or not storage_dir.is_dir():
        return 0

    moved = 0
    for path in storage_dir.glob("*.json"):
        opened = _read_legacy_externalized_payload_json_with_directory(path, storage_dir=storage_dir)
        if opened is None:
            try:
                if _reassign_oversize_externalized_payload(
                    path,
                    old_session_id=old_session_id,
                    new_session_id=new_session_id,
                    storage_dir=storage_dir,
                ):
                    moved += 1
            except (OSError, TypeError, UnicodeDecodeError, ValueError, NotImplementedError) as exc:
                logger.warning("Externalized payload session reassignment skipped for %s: %s", path.name, exc)
            continue
        payload, dir_fd, payload_stat = opened
        try:
            if (payload.get("session_id") or "") != old_session_id:
                continue
            payload["session_id"] = new_session_id
            try:
                _replace_externalized_payload(
                    path,
                    payload,
                    dir_fd=dir_fd,
                    expected_identity=(payload_stat.st_dev, payload_stat.st_ino),
                )
            except (OSError, TypeError, NotImplementedError) as exc:
                logger.warning("Externalized payload session reassignment skipped for %s: %s", path.name, exc)
                continue
            moved += 1
        finally:
            os.close(dir_fd)
    return moved


def find_externalized_payload_for_message(
    content: str,
    *,
    tool_call_id: str = "",
    session_id: str = "",
    kind: str | None = "tool_result",
    role: str = "",
    config,
    hermes_home: str = "",
) -> Dict[str, Any] | None:
    if not content:
        return None
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    if not storage_dir.exists() or not storage_dir.is_dir():
        return None

    digest_prefix = _content_digest_prefix(content)
    candidates = sorted(storage_dir.glob(f"*_{digest_prefix}_*.json"))
    fallback_match = None
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if kind is not None and payload.get("kind", "tool_result") != kind:
            continue
        if (payload.get("tool_call_id") or "") != (tool_call_id or ""):
            continue
        payload_role = payload.get("role") or ""
        if role and payload_role and payload_role != role:
            continue
        if payload.get("content") != content:
            continue
        summary = _externalized_summary(path, payload)
        payload_session_id = (payload.get("session_id") or "")
        if session_id:
            if payload_session_id == session_id:
                return summary
            continue
        if fallback_match is None:
            fallback_match = summary
    return fallback_match


def find_externalized_tool_result_content_for_call(
    *,
    tool_call_id: str,
    session_id: str = "",
    expected_chars: int | None = None,
    persisted_output_source_path: str | None = None,
    persisted_output_preview_sha256: str | None = None,
    require_persisted_output_file_not_newer: bool = False,
    allow_redacted_preview_match: bool = True,
    require_missing_file_generation_metadata: bool = False,
    persisted_output_file_size: int | None = None,
    persisted_output_file_mtime_ns: int | None = None,
    persisted_output_file_ctime_ns: int | None = None,
    config,
    hermes_home: str = "",
) -> str | None:
    """Return durable externalized tool-result content for a matching marker.

    This is used only for replay identity recovery when Hermes' temporary
    persisted-output file has already been cleaned up but LCM previously stored
    the recovered full tool output durably. A reused tool-call id alone is not
    sufficient proof; marker-specific metadata captured before redaction must
    match when provided.
    """
    if not tool_call_id:
        return None
    if (expected_chars is not None or persisted_output_source_path) and not persisted_output_preview_sha256:
        return None
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    if not storage_dir.exists() or not storage_dir.is_dir():
        return None
    for path in sorted(storage_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("kind", "tool_result") != "tool_result":
            continue
        if (payload.get("tool_call_id") or "") != tool_call_id:
            continue
        if session_id and (payload.get("session_id") or "") != session_id:
            continue
        payload_role = payload.get("role") or ""
        if payload_role and payload_role != "tool":
            continue
        content = payload.get("content")
        if not isinstance(content, str):
            continue
        if (
            expected_chars is not None
            or persisted_output_source_path
            or persisted_output_preview_sha256
            or require_persisted_output_file_not_newer
        ):
            marker_matches = False
            for marker in _persisted_output_marker_entries(payload, include_legacy_preview_prefix=True):
                if expected_chars is not None and marker.get("expected_chars") != expected_chars:
                    continue
                if persisted_output_source_path and marker.get("source_path") != persisted_output_source_path:
                    continue
                if require_missing_file_generation_metadata and (
                    marker.get("file_size") is not None
                    or marker.get("file_mtime_ns") is not None
                    or marker.get("file_ctime_ns") is not None
                ):
                    continue
                if persisted_output_file_size is not None and marker.get("file_size") != persisted_output_file_size:
                    continue
                if persisted_output_file_mtime_ns is not None and marker.get("file_mtime_ns") != persisted_output_file_mtime_ns:
                    continue
                if persisted_output_file_ctime_ns is not None and marker.get("file_ctime_ns") != persisted_output_file_ctime_ns:
                    continue
                if (
                    persisted_output_preview_sha256
                    and not _persisted_output_marker_matches_preview_digest(
                        marker,
                        persisted_output_preview_sha256,
                        config=config,
                        allow_redacted_preview_match=allow_redacted_preview_match,
                    )
                ):
                    continue
                if require_persisted_output_file_not_newer and not _marker_file_not_newer_than_payload(marker, payload):
                    continue
                marker_matches = True
                break
            if not marker_matches:
                continue
        return content
    return None


def externalize_ingest_payload(
    content: str,
    *,
    role: str = "",
    session_id: str = "",
    field_path: str = "",
    config,
    hermes_home: str = "",
    kind: str = "ingest_payload",
) -> Dict[str, Any] | None:
    if not content:
        return None
    try:
        storage_dir = resolve_large_output_storage_dir(config, hermes_home=hermes_home)
    except OSError as exc:
        logger.warning("LCM ingest payload externalization skipped (non-blocking): %s", exc)
        return None

    digest_prefix = _content_digest_prefix(content)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    unique_suffix = f"{time.time_ns():x}"
    kind_stub = _safe_stub(kind, "ingest_payload")
    field_stub = re.sub(r"[^A-Za-z0-9_.-]+", "-", field_path or "payload")[:48]
    filename = f"{timestamp}_{kind_stub}_{field_stub}_{digest_prefix}_{unique_suffix}.json"
    path = storage_dir / filename
    payload = {
        "kind": kind,
        "role": role,
        "session_id": session_id,
        "field_path": field_path,
        "content": content,
        "content_chars": len(content),
        "content_bytes": len(content.encode("utf-8")),
        "created_at": time.time(),
    }
    try:
        _write_externalized_payload(path, payload)
    except OSError as exc:
        logger.warning("LCM ingest payload externalization skipped (non-blocking): %s", exc)
        return None

    summary = _externalized_summary(path, payload)
    placeholder = (
        f"[Externalized LCM ingest payload: kind={_placeholder_metadata(summary.get('kind') or kind)}; "
        f"field={_placeholder_metadata(summary.get('field_path') or '?')}; chars={summary.get('content_chars', 0)}; "
        f"bytes={summary.get('content_bytes', 0)}; ref={summary.get('ref', '')}]"
    )
    return {
        "placeholder": placeholder,
        "path": path,
        "payload": payload,
    }


def maybe_externalize_tool_output(
    content: str,
    *,
    tool_call_id: str = "",
    session_id: str = "",
    config,
    hermes_home: str = "",
    force: bool = False,
) -> Dict[str, Any] | None:
    return maybe_externalize_payload(
        content,
        kind="tool_result",
        tool_call_id=tool_call_id,
        session_id=session_id,
        role="tool",
        config=config,
        hermes_home=hermes_home,
        force=force,
    )


def maybe_externalize_payload(
    content: str,
    *,
    kind: str = "raw_payload",
    tool_call_id: str = "",
    session_id: str = "",
    role: str = "",
    config,
    hermes_home: str = "",
    force: bool = False,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """Externalize one normalized payload if configured.

    Returns a dict with a compact placeholder and the durable JSON payload path,
    or ``None`` when disabled, below threshold and not forced, or storage is
    unavailable. On storage failure callers should keep the original content so
    there is no silent data loss.
    """
    if not getattr(config, "large_output_externalization_enabled", False):
        return None

    threshold = max(1, int(getattr(config, "large_output_externalization_threshold_chars", 0) or 0))
    if not content or (len(content) <= threshold and not force):
        return None

    try:
        storage_dir = resolve_large_output_storage_dir(config, hermes_home=hermes_home)
    except OSError as exc:
        logger.warning("Large payload externalization skipped (non-blocking): %s", exc)
        return None

    existing = find_externalized_payload_for_message(
        content,
        tool_call_id=tool_call_id,
        session_id=session_id,
        kind=kind,
        role=role,
        config=config,
        hermes_home=hermes_home,
    )
    if existing is not None:
        existing_path = storage_dir / existing["ref"]
        existing_payload = None
        if metadata:
            try:
                existing_payload = json.loads(existing_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing_payload = None
            if existing_payload is not None and _merge_persisted_output_marker_metadata(existing_payload, metadata):
                try:
                    _replace_externalized_payload(existing_path, existing_payload)
                    existing = _externalized_summary(existing_path, existing_payload)
                except OSError as exc:
                    logger.warning("Large payload metadata update skipped (non-blocking): %s", exc)
        return {
            "placeholder": _build_externalized_placeholder(existing),
            "path": existing_path,
            "payload": existing,
        }

    digest_prefix = _content_digest_prefix(content)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    unique_suffix = f"{time.time_ns():x}"
    if kind == "tool_result":
        # Keep the original filename shape for compatibility with existing
        # externalized tool-output stores and tests.
        filename = f"{timestamp}_{_tool_call_stub(tool_call_id)}_{digest_prefix}_{unique_suffix}.json"
    else:
        filename = (
            f"{timestamp}_{_safe_stub(kind, 'payload')}_"
            f"{_safe_stub(role, 'message')}_{digest_prefix}_{unique_suffix}.json"
        )
    path = storage_dir / filename

    payload = {
        "kind": kind,
        "tool_call_id": tool_call_id,
        "role": role,
        "session_id": session_id,
        "content": content,
        "content_chars": len(content),
        "content_bytes": len(content.encode("utf-8")),
        "created_at": time.time(),
    }
    if metadata:
        payload.update(_safe_persisted_output_metadata(metadata))
        _merge_persisted_output_marker_metadata(payload, metadata)
    try:
        _write_externalized_payload(path, payload)
    except OSError as exc:
        logger.warning("Large payload externalization skipped (non-blocking): %s", exc)
        return None

    placeholder = _build_externalized_placeholder(
        {
            "kind": kind,
            "tool_call_id": tool_call_id,
            "role": role,
            "content_chars": payload["content_chars"],
            "content_bytes": payload["content_bytes"],
            "ref": path.name,
        }
    )
    return {
        "placeholder": placeholder,
        "path": path,
        "payload": payload,
    }
