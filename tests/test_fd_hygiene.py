"""Regression tests for bounded SQLite file-descriptor usage."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _db_fd_count(db_path: Path) -> int:
    """Count this process's descriptors for the database and its WAL files."""
    if os.name != "posix":
        pytest.skip("/proc/self/fd is POSIX-only")
    return sum(
        1
        for fd in Path("/proc/self/fd").iterdir()
        if db_path.name in os.path.realpath(fd)
    )


def test_cloned_engines_keep_lcm_fd_count_bounded(tmp_path: Path):
    """Clones must not retain one SQLite connection set per agent."""
    db_path = tmp_path / "lcm.db"
    prototype = LCMEngine(
        config=LCMConfig(database_path=str(db_path)),
        hermes_home=str(tmp_path / "hermes"),
    )
    clones = []
    try:
        baseline = _db_fd_count(db_path)
        for index in range(50):
            clone = prototype.clone_for_agent()
            clones.append(clone)
            clone.on_session_start(
                f"session-{index}",
                platform="test",
                conversation_id=f"conversation-{index}",
            )
            clone.ingest([{"role": "user", "content": f"message-{index}"}])

        # A clone has independent session state, but storage is shared. Allow
        # the small fixed set of WAL/shm descriptors owned by that storage.
        assert _db_fd_count(db_path) <= baseline + 3
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 50
    finally:
        for clone in clones:
            clone.shutdown()
        prototype.shutdown()