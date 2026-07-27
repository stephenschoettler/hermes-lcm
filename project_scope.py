"""Project identity resolution without Hermes core dependencies."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

_GIT_TIMEOUT_SECONDS = 1.5


@dataclass(frozen=True)
class ProjectMetadata:
    project_id: str
    project_root: str
    cwd: str


def _normalize_path(value: str) -> str:
    return os.path.normcase(
        os.path.realpath(os.path.abspath(os.path.expanduser(value)))
    )


def _git(cwd: str, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", cwd, *args],
            check=False,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def resolve_project_metadata(cwd: str | None) -> ProjectMetadata:
    """Resolve a stable project from an explicit cwd, folding Git worktrees."""
    raw_cwd = str(cwd or "").strip()
    if not raw_cwd:
        return ProjectMetadata(project_id="", project_root="", cwd="")

    normalized_cwd = _normalize_path(raw_cwd)
    common_dir = _git(
        normalized_cwd,
        "rev-parse",
        "--path-format=absolute",
        "--git-common-dir",
    )
    if common_dir:
        normalized_common_dir = _normalize_path(common_dir)
        if os.path.basename(normalized_common_dir) == ".git":
            project_root = os.path.dirname(normalized_common_dir)
        else:
            project_root = _git(normalized_cwd, "rev-parse", "--show-toplevel")
    else:
        project_root = ""

    normalized_root = _normalize_path(project_root) if project_root else normalized_cwd
    return ProjectMetadata(
        project_id=normalized_root,
        project_root=normalized_root,
        cwd=normalized_cwd,
    )
