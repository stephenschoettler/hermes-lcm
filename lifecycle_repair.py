"""Targeted recovery for the ``lcm_lifecycle_state`` bookkeeping table.

Recurring corruption reports (#643, #681) share one signature: page damage
confined to ``lcm_lifecycle_state`` and its two indexes while every content
table (``messages``, ``summary_nodes``, both FTS indexes, chunks/embeddings)
stays readable.  The table is pure bookkeeping -- current session binding,
frontier markers and compaction debt -- so losing its rows costs at most a
re-bind, while leaving the damaged b-tree in place keeps the engine failing on
every lifecycle read/write (the gateway then silently degrades to the built-in
compressor).

This module detects that specific condition and rebuilds the table in place:

* detection is table-scoped (``PRAGMA quick_check(<table>)``), so the common
  healthy path never scans the whole multi-hundred-megabyte file;
* a rebuild only happens when *every* error reported by the full
  ``integrity_check`` points at the lifecycle table or its indexes -- anything
  touching a content table is left to the operator;
* the rebuild is the same DDL that :mod:`db_bootstrap` uses, followed by
  ``VACUUM`` because ``DROP TABLE`` leaves the old b-tree pages on the
  freelist and a bare rebuild then fails ``integrity_check`` with
  ``Page N: never used``.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

LIFECYCLE_TABLE = "lcm_lifecycle_state"

#: Objects whose damage this module is allowed to repair.
_LIFECYCLE_OBJECT = re.compile(r"lcm_lifecycle_state|idx_lcm_lifecycle_", re.I)
#: Content-table names; any error naming one of these vetoes a rebuild.
_FOREIGN_OBJECT = re.compile(
    r"(messages|summary_nodes|metadata|lcm_chunk_\w*|lcm_embedding_\w*|lcm_migration_state)",
    re.I,
)
_TREE_REF = re.compile(r"\bTree (\d+)\b")
#: SQLite messages that mean "this file/table is structurally broken".
_CORRUPTION_MARKERS = (
    "malformed",
    "not a database",
    "btreeinitpage",
    "file is encrypted",
    "database disk image",
)

LIFECYCLE_DDL = """
DROP TABLE IF EXISTS lcm_lifecycle_state;
CREATE TABLE lcm_lifecycle_state (
    conversation_id TEXT PRIMARY KEY,
    current_session_id TEXT,
    last_finalized_session_id TEXT,
    current_frontier_store_id INTEGER NOT NULL DEFAULT 0,
    last_finalized_frontier_store_id INTEGER NOT NULL DEFAULT 0,
    debt_kind TEXT,
    debt_size_estimate INTEGER NOT NULL DEFAULT 0,
    current_bound_at REAL,
    last_finalized_at REAL,
    debt_updated_at REAL,
    last_maintenance_attempt_at REAL,
    last_rollover_at REAL,
    last_reset_at REAL,
    updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_current_session
    ON lcm_lifecycle_state(current_session_id);
CREATE INDEX IF NOT EXISTS idx_lcm_lifecycle_last_finalized_session
    ON lcm_lifecycle_state(last_finalized_session_id);
"""


def is_corruption_error(exc: BaseException) -> bool:
    """True when *exc* is SQLite reporting structural damage, not a logic error."""
    text = str(exc).lower()
    return any(marker in text for marker in _CORRUPTION_MARKERS)


def table_check(conn: sqlite3.Connection) -> list[str]:
    """``PRAGMA quick_check`` scoped to the lifecycle table.

    A damaged b-tree root makes SQLite raise instead of returning rows, so the
    exception text is folded into the result rather than propagated.
    """
    try:
        rows = conn.execute(f"PRAGMA quick_check('{LIFECYCLE_TABLE}')").fetchall()
    except sqlite3.DatabaseError as exc:
        return [str(exc)]
    return [str(row[0]) for row in rows]


def classify(errors: list[str], conn: sqlite3.Connection) -> tuple[bool, list[str]]:
    """Split integrity errors into "lifecycle only" and "anything else".

    ``Rowid N out of order`` style errors carry no table name, only a b-tree
    root page, so the page is resolved through ``sqlite_master``.
    """
    owners: dict[int, str] = {}
    try:
        owners = {
            int(rootpage): name
            for name, rootpage in conn.execute(
                "SELECT name, rootpage FROM sqlite_master WHERE rootpage > 0"
            ).fetchall()
        }
    except sqlite3.DatabaseError:
        # Unreadable schema: cannot prove the damage is confined, refuse repair.
        return False, list(errors)

    foreign: list[str] = []
    for error in errors:
        if error.strip() == "ok":
            continue
        tree = _TREE_REF.search(error)
        owner = owners.get(int(tree.group(1))) if tree else None
        if owner is not None:
            if _LIFECYCLE_OBJECT.search(owner):
                continue
            foreign.append(error)
            continue
        if _LIFECYCLE_OBJECT.search(error) and not _FOREIGN_OBJECT.search(error):
            continue
        foreign.append(error)
    return (not foreign), foreign


def rebuild(conn: sqlite3.Connection) -> dict[str, Any]:
    """Recreate the lifecycle table (and its indexes) in place, then compact.

    Returns ``{"error": ...}`` instead of raising when SQLite cannot even drop
    the damaged b-tree: that happens when pages are physically unreadable (as
    opposed to logically inconsistent), and in that state any in-process repair
    is unsafe -- the caller falls back to the operator procedure.
    """
    rows_before = _row_count(conn)
    try:
        conn.executescript(LIFECYCLE_DDL)
        conn.commit()
        # DROP TABLE leaves the old pages on the freelist; without the VACUUM
        # the next integrity_check reports "Page N: never used".
        conn.execute("VACUUM")
    except sqlite3.DatabaseError as exc:
        return {"rows_before": rows_before, "rows_after": None, "error": str(exc)}
    return {"rows_before": rows_before, "rows_after": _row_count(conn)}


def heal_if_corrupt(conn: sqlite3.Connection) -> dict[str, Any]:
    """Rebuild the lifecycle table when -- and only when -- it is the sole damage.

    Returns a dict with ``checked``/``healed``/``reason`` keys; never raises for
    a corruption it declines to repair, so callers can surface it as a doctor
    finding instead of crashing the engine.
    """
    errors = table_check(conn)
    if errors == ["ok"]:
        return {"checked": True, "healed": False, "reason": "healthy", "errors": []}

    full = _full_integrity(conn)
    lifecycle_only, foreign = classify(full, conn)
    if not lifecycle_only:
        # A damaged b-tree root makes SQLite raise instead of describing the
        # damage ("database disk image is malformed"), which classifies as
        # foreign. Before giving up, prove the content tables are readable on
        # their own: if they are, the damage really is lifecycle-only.
        if not _content_tables_readable(conn):
            return {
                "checked": True,
                "healed": False,
                "reason": "damage outside lcm_lifecycle_state",
                "errors": (foreign or errors)[:5],
            }
    stats = rebuild(conn)
    if stats.get("error"):
        return {
            "checked": True,
            "healed": False,
            "reason": f"rebuild failed: {stats['error'][:200]}",
            "errors": errors[:5],
            **stats,
        }
    after = _full_integrity(conn)
    return {
        "checked": True,
        "healed": after == ["ok"],
        "reason": "rebuilt" if after == ["ok"] else "rebuilt but table still damaged",
        "errors": [] if after == ["ok"] else after[:5],
        **stats,
    }


def _content_tables_readable(conn: sqlite3.Connection) -> bool:
    """True when every content table present in the file passes its own check."""
    try:
        present = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    except sqlite3.DatabaseError:
        return False
    candidates = ("messages", "summary_nodes", "lcm_chunk_meta", "lcm_embedding_meta")
    for table in candidates:
        if table not in present:
            continue
        try:
            rows = conn.execute(f"PRAGMA quick_check('{table}')").fetchall()
        except sqlite3.DatabaseError:
            return False
        if [str(row[0]) for row in rows] != ["ok"]:
            return False
    return True


def _full_integrity(conn: sqlite3.Connection) -> list[str]:
    try:
        return [str(row[0]) for row in conn.execute("PRAGMA integrity_check").fetchall()]
    except sqlite3.DatabaseError as exc:
        return [str(exc)]


def _row_count(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {LIFECYCLE_TABLE}").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.DatabaseError:
        return -1
