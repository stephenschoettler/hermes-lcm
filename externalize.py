"""Helpers for optional large tool-output externalization during pre-compaction serialization."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict

DEFAULT_LARGE_OUTPUT_DIRNAME = "lcm-large-outputs"
_EXTERNALIZED_REF_RE = re.compile(r"ref=([^;\]\s]+)")
logger = logging.getLogger(__name__)


def _tool_call_stub(tool_call_id: str) -> str:
    return (tool_call_id or "tool-result").replace("/", "-").replace(":", "-")[:48]


def _content_digest_prefix(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()[:12]


def get_large_output_storage_dir(config, hermes_home: str = "", *, create: bool) -> Path:
    configured = getattr(config, "large_output_externalization_path", "") or ""
    if configured:
        path = Path(configured).expanduser()
    else:
        base = Path(hermes_home).expanduser() if hermes_home else Path("~/.hermes").expanduser()
        path = base / DEFAULT_LARGE_OUTPUT_DIRNAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_large_output_storage_dir(config, hermes_home: str = "") -> Path:
    return get_large_output_storage_dir(config, hermes_home=hermes_home, create=True)


def _externalized_summary(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ref": path.name,
        "kind": payload.get("kind", "tool_result"),
        "tool_call_id": payload.get("tool_call_id", ""),
        "session_id": payload.get("session_id", ""),
        "content_chars": payload.get("content_chars", len(payload.get("content", ""))),
        "content_bytes": payload.get("content_bytes", len((payload.get("content", "") or "").encode("utf-8"))),
        "created_at": payload.get("created_at"),
    }


def _build_externalized_placeholder(summary: Dict[str, Any]) -> str:
    return (
        f"[Externalized tool output: tool_call_id={summary.get('tool_call_id') or '?'}; "
        f"chars={summary.get('content_chars', 0)}; bytes={summary.get('content_bytes', 0)}; ref={summary.get('ref', '')}]"
    )


def build_transcript_gc_placeholder(summary: Dict[str, Any]) -> str:
    return (
        f"[GC'd externalized tool output: tool_call_id={summary.get('tool_call_id') or '?'}; "
        f"chars={summary.get('content_chars', 0)}; ref={summary.get('ref', '')}]"
    )


def extract_externalized_ref(text: str) -> str | None:
    if not text:
        return None
    match = _EXTERNALIZED_REF_RE.search(text)
    if not match:
        return None
    ref = match.group(1).strip()
    if not ref or Path(ref).name != ref:
        return None
    return ref


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


def find_externalized_payload_for_message(
    content: str,
    *,
    tool_call_id: str = "",
    session_id: str = "",
    config,
    hermes_home: str = "",
) -> Dict[str, Any] | None:
    if not content:
        return None
    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    if not storage_dir.exists() or not storage_dir.is_dir():
        return None

    digest_prefix = _content_digest_prefix(content)
    tool_stub = _tool_call_stub(tool_call_id)
    candidates = sorted(storage_dir.glob(f"*_{tool_stub}_{digest_prefix}_*.json"))
    fallback_match = None
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("kind") != "tool_result":
            continue
        if (payload.get("tool_call_id") or "") != (tool_call_id or ""):
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


def maybe_externalize_tool_output(
    content: str,
    *,
    tool_call_id: str = "",
    session_id: str = "",
    config,
    hermes_home: str = "",
) -> Dict[str, Any] | None:
    if not getattr(config, "large_output_externalization_enabled", False):
        return None

    threshold = max(1, int(getattr(config, "large_output_externalization_threshold_chars", 0) or 0))
    if not content or len(content) <= threshold:
        return None

    try:
        storage_dir = resolve_large_output_storage_dir(config, hermes_home=hermes_home)
    except OSError as exc:
        logger.warning("Large tool-output externalization skipped (non-blocking): %s", exc)
        return None

    existing = find_externalized_payload_for_message(
        content,
        tool_call_id=tool_call_id,
        session_id=session_id,
        config=config,
        hermes_home=hermes_home,
    )
    if existing is not None:
        return {
            "placeholder": _build_externalized_placeholder(existing),
            "path": storage_dir / existing["ref"],
            "payload": existing,
        }

    digest_prefix = _content_digest_prefix(content)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    unique_suffix = f"{time.time_ns():x}"
    tool_stub = _tool_call_stub(tool_call_id)
    filename = f"{timestamp}_{tool_stub}_{digest_prefix}_{unique_suffix}.json"
    path = storage_dir / filename

    payload = {
        "kind": "tool_result",
        "tool_call_id": tool_call_id,
        "session_id": session_id,
        "content": content,
        "content_chars": len(content),
        "content_bytes": len(content.encode("utf-8")),
        "created_at": time.time(),
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Large tool-output externalization skipped (non-blocking): %s", exc)
        return None

    placeholder = _build_externalized_placeholder(
        {
            "tool_call_id": tool_call_id,
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
