"""Plugin and git runtime-identity helpers for LCM status/doctor surfaces.

Isolated from ``engine.py`` (WS5 seam): resolving the plugin's own name/version
from its manifest and probing best-effort git identity for source checkouts are
a cohesive provenance concern that feeds the ``plugin_*`` fields of the runtime
status/doctor payload. The manifest-version cache lives here alongside the
functions that own it. ``engine.py`` imports the entry points it calls.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PLUGIN_ROOT = Path(__file__).resolve().parent
_PLUGIN_METADATA: dict[str, str] | None = None


def _strip_metadata_scalar(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _plugin_metadata() -> dict[str, str]:
    """Return plugin identity from the loaded code tree.

    Always re-read the manifest from disk when available so status tools reflect
    hot-updated plugin checkouts even in long-lived Hermes processes.
    """
    global _PLUGIN_METADATA

    metadata = {"name": "hermes-lcm", "version": "unknown"}
    manifest = _PLUGIN_ROOT / "plugin.yaml"
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            key, sep, raw_value = line.partition(":")
            if not sep:
                continue
            key = key.strip()
            if key in {"name", "version"}:
                metadata[key] = _strip_metadata_scalar(raw_value)
        _PLUGIN_METADATA = metadata
        return dict(metadata)
    except OSError:
        logger.debug("LCM plugin manifest not readable at %s", manifest)

    if _PLUGIN_METADATA is not None:
        return dict(_PLUGIN_METADATA)
    return dict(metadata)


def _git_runtime_identity(root: Path) -> dict[str, Any]:
    """Best-effort git identity for source checkouts.

    Packaged installs may not have a `.git` directory. In that case the fields
    stay empty instead of turning status/doctor into a git dependency.
    """

    if not (root / ".git").exists():
        return {
            "plugin_git_commit": "",
            "plugin_git_branch": "",
            "plugin_git_dirty": None,
            "plugin_git_remote": "",
        }

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("LCM git identity probe failed at %s: %s", root, exc)
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    dirty_output = _git("status", "--porcelain")
    return {
        "plugin_git_commit": _git("rev-parse", "HEAD") or "",
        "plugin_git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "",
        "plugin_git_dirty": None if dirty_output is None else bool(dirty_output),
        "plugin_git_remote": _git("config", "--get", "remote.origin.url") or "",
    }
