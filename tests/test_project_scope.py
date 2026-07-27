from __future__ import annotations

import os
import subprocess

from hermes_lcm.project_scope import ProjectMetadata, resolve_project_metadata


def _git(*args: str, cwd) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_git_worktree_uses_common_repository_root_as_project_identity(tmp_path):
    repo = tmp_path / "repo"
    worktree = tmp_path / "linked-worktree"
    repo.mkdir()
    _git("init", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test User", cwd=repo)
    _git("commit", "--allow-empty", "-m", "init", cwd=repo)
    _git("worktree", "add", "-b", "linked", str(worktree), cwd=repo)

    metadata = resolve_project_metadata(str(worktree / "."))

    common_root = os.path.realpath(repo)
    assert metadata == ProjectMetadata(
        project_id=common_root,
        project_root=common_root,
        cwd=os.path.realpath(worktree),
    )


def test_non_git_cwd_falls_back_to_normalized_cwd(tmp_path):
    cwd = tmp_path / "folder"
    cwd.mkdir()

    metadata = resolve_project_metadata(str(cwd / ".." / "folder"))

    normalized = os.path.realpath(cwd)
    assert metadata == ProjectMetadata(
        project_id=normalized,
        project_root=normalized,
        cwd=normalized,
    )


def test_missing_cwd_has_no_implicit_ambient_project():
    assert resolve_project_metadata(None) == ProjectMetadata(
        project_id="",
        project_root="",
        cwd="",
    )
    assert resolve_project_metadata("  ") == ProjectMetadata(
        project_id="",
        project_root="",
        cwd="",
    )


def test_git_subprocess_failure_falls_back_to_normalized_cwd(tmp_path, monkeypatch):
    cwd = tmp_path / "repo-shaped-folder"
    cwd.mkdir()

    def fail(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=1.5)

    monkeypatch.setattr(subprocess, "run", fail)

    normalized = os.path.realpath(cwd)
    assert resolve_project_metadata(str(cwd)) == ProjectMetadata(
        project_id=normalized,
        project_root=normalized,
        cwd=normalized,
    )
