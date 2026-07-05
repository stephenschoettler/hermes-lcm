"""Helpers for optional large-payload externalization.

Externalization started as a pre-compaction serializer guard for oversized tool
outputs. The same durable payload format is also used by the ingest path so
obvious oversized content can be kept out of SQLite/FTS while still being
recoverable through the LCM inspection and expansion tools.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

DEFAULT_LARGE_OUTPUT_DIRNAME = "lcm-large-outputs"
_EXTERNALIZED_REF_RE = re.compile(
    r"\[(?:Externalized|GC'd externalized) (?:tool output|payload):.*?;\s*ref=([^;\]\s]+)\]"
)


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


def _replace_externalized_payload(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.{time.time_ns():x}.tmp")
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


def externalized_tool_result_has_persisted_output_marker(ref: str, *, config, hermes_home: str = "") -> bool:
    if not ref or Path(ref).name != ref:
        return False
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    if not storage_dir.exists() or not storage_dir.is_dir():
        return False
    path = storage_dir / ref
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
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
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (payload.get("session_id") or "") != old_session_id:
            continue
        payload["session_id"] = new_session_id
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            _write_externalized_payload(tmp_path, payload)
            tmp_path.replace(path)
            _fsync_directory(path.parent)
        except OSError as exc:
            logger.warning("Externalized payload session reassignment skipped for %s: %s", path.name, exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        moved += 1
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
) -> Dict[str, Any] | None:
    return maybe_externalize_payload(
        content,
        kind="tool_result",
        tool_call_id=tool_call_id,
        session_id=session_id,
        role="tool",
        config=config,
        hermes_home=hermes_home,
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
