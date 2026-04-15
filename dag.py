"""Summary DAG — hierarchical compaction graph.

Each node is a summary of source material (raw messages or lower-depth
summaries). Nodes form a directed acyclic graph where edges point from
a summary to its sources.

Depth semantics:
  D0 — leaf summaries of raw messages (minutes timescale)
  D1 — condensation of D0 nodes (hours)
  D2 — condensation of D1 nodes (days)
  D3+ — further condensation (weeks/months)
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db_bootstrap import (
    ExternalContentFtsSpec,
    configure_connection,
    ensure_external_content_fts,
    run_versioned_migrations,
)

logger = logging.getLogger(__name__)


@dataclass
class SummaryNode:
    """A single node in the summary DAG."""
    node_id: int = 0
    session_id: str = ""
    depth: int = 0
    summary: str = ""
    token_count: int = 0
    source_token_count: int = 0  # total tokens of source material
    source_ids: List[int] = field(default_factory=list)  # store_ids or node_ids
    source_type: str = "messages"  # "messages" or "nodes"
    created_at: float = 0.0
    expand_hint: str = ""  # "Expand for details about: ..."


class SummaryDAG:
    """SQLite-backed DAG of summary nodes."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(str(self.db_path), timeout=5.0, check_same_thread=False)
        configure_connection(self._conn)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS summary_nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                depth INTEGER NOT NULL DEFAULT 0,
                summary TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                source_token_count INTEGER DEFAULT 0,
                source_ids TEXT NOT NULL DEFAULT '[]',
                source_type TEXT NOT NULL DEFAULT 'messages',
                created_at REAL NOT NULL,
                expand_hint TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_session_depth
                ON summary_nodes(session_id, depth, created_at);

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        ensure_external_content_fts(
            self._conn,
            ExternalContentFtsSpec(
                table_name="nodes_fts",
                content_table="summary_nodes",
                content_rowid="node_id",
                indexed_column="summary",
                trigger_sqls=(
                    """
                    CREATE TRIGGER IF NOT EXISTS nodes_fts_insert
                        AFTER INSERT ON summary_nodes BEGIN
                        INSERT INTO nodes_fts(rowid, summary)
                            VALUES (new.node_id, new.summary);
                    END;
                    """,
                    """
                    CREATE TRIGGER IF NOT EXISTS nodes_fts_delete
                        AFTER DELETE ON summary_nodes BEGIN
                        INSERT INTO nodes_fts(nodes_fts, rowid, summary)
                            VALUES('delete', old.node_id, old.summary);
                    END;
                    """,
                ),
            ),
        )
        run_versioned_migrations(self._conn)
        self._conn.commit()

    # -- Write --------------------------------------------------------------

    def add_node(self, node: SummaryNode) -> int:
        """Insert a summary node and return its node_id."""
        cur = self._conn.execute(
            """INSERT INTO summary_nodes
               (session_id, depth, summary, token_count, source_token_count,
                source_ids, source_type, created_at, expand_hint)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node.session_id,
                node.depth,
                node.summary,
                node.token_count,
                node.source_token_count,
                json.dumps(node.source_ids),
                node.source_type,
                node.created_at or time.time(),
                node.expand_hint,
            ),
        )
        self._conn.commit()
        node.node_id = cur.lastrowid
        return node.node_id

    def delete_below_depth(self, session_id: str, min_depth: int) -> int:
        """Delete all nodes for a session with depth < min_depth.

        Returns the number of deleted nodes. Used during session reset
        to retain only high-level summaries across sessions.
        """
        cur = self._conn.execute(
            """DELETE FROM summary_nodes
               WHERE session_id = ? AND depth < ?""",
            (session_id, min_depth),
        )
        deleted = cur.rowcount
        if deleted:
            self._conn.commit()
        return deleted

    def delete_session_nodes(self, session_id: str) -> int:
        """Delete all nodes for a session. Returns count deleted."""
        cur = self._conn.execute(
            "DELETE FROM summary_nodes WHERE session_id = ?",
            (session_id,),
        )
        deleted = cur.rowcount
        if deleted:
            self._conn.commit()
        return deleted

    def reassign_session_nodes(self, old_session_id: str, new_session_id: str) -> int:
        """Move all nodes from one session_id to another.

        Used for /new carry-over where retained summaries should become part of
        the fresh session while preserving node IDs and node-to-node links.
        """
        cur = self._conn.execute(
            "UPDATE summary_nodes SET session_id = ? WHERE session_id = ?",
            (new_session_id, old_session_id),
        )
        moved = cur.rowcount
        if moved:
            self._conn.commit()
        return moved

    # -- Read ---------------------------------------------------------------

    def get_node(self, node_id: int) -> Optional[SummaryNode]:
        row = self._conn.execute(
            "SELECT * FROM summary_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def get_session_nodes(self, session_id: str,
                          depth: int | None = None,
                          limit: int = 1000) -> List[SummaryNode]:
        """Get nodes for a session, optionally filtered by depth."""
        if depth is not None:
            rows = self._conn.execute(
                """SELECT * FROM summary_nodes
                   WHERE session_id = ? AND depth = ?
                   ORDER BY created_at LIMIT ?""",
                (session_id, depth, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM summary_nodes
                   WHERE session_id = ?
                   ORDER BY depth, created_at LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def count_at_depth(self, session_id: str, depth: int) -> int:
        """Count nodes at a specific depth for a session."""
        row = self._conn.execute(
            """SELECT COUNT(*) FROM summary_nodes
               WHERE session_id = ? AND depth = ?""",
            (session_id, depth),
        ).fetchone()
        return row[0] if row else 0

    def get_uncondensed_at_depth(self, session_id: str, depth: int,
                                  limit: int = 100) -> List[SummaryNode]:
        """Get nodes at a depth that haven't been condensed yet.

        A node is 'uncondensed' if it's not referenced as a source by
        any higher-depth node.
        """
        rows = self._conn.execute(
            """SELECT n.* FROM summary_nodes n
               WHERE n.session_id = ? AND n.depth = ?
               AND n.node_id NOT IN (
                   SELECT json_each.value FROM summary_nodes p,
                   json_each(p.source_ids)
                   WHERE p.session_id = ? AND p.depth > ? AND p.source_type = 'nodes'
               )
               ORDER BY n.created_at LIMIT ?""",
            (session_id, depth, session_id, depth, limit),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    # -- Search -------------------------------------------------------------

    def search(self, query: str, session_id: str | None = None,
               limit: int = 20) -> List[SummaryNode]:
        """FTS5 search across all summary nodes."""
        try:
            if session_id:
                rows = self._conn.execute(
                    """SELECT n.* FROM nodes_fts fts
                       JOIN summary_nodes n ON n.node_id = fts.rowid
                       WHERE nodes_fts MATCH ? AND n.session_id = ?
                       ORDER BY rank LIMIT ?""",
                    (query, session_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT n.* FROM nodes_fts fts
                       JOIN summary_nodes n ON n.node_id = fts.rowid
                       WHERE nodes_fts MATCH ?
                       ORDER BY rank LIMIT ?""",
                    (query, limit),
                ).fetchall()
            return [self._row_to_node(r) for r in rows]
        except sqlite3.DatabaseError:
            like = f"%{query.strip().lower()}%"
            if session_id:
                rows = self._conn.execute(
                    """SELECT * FROM summary_nodes
                       WHERE session_id = ? AND LOWER(COALESCE(summary, '')) LIKE ?
                       ORDER BY created_at LIMIT ?""",
                    (session_id, like, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT * FROM summary_nodes
                       WHERE LOWER(COALESCE(summary, '')) LIKE ?
                       ORDER BY created_at LIMIT ?""",
                    (like, limit),
                ).fetchall()
            return [self._row_to_node(r) for r in rows]

    # -- DAG traversal ------------------------------------------------------

    def get_source_nodes(self, node: SummaryNode) -> List[SummaryNode]:
        """Get the immediate child nodes of a summary node."""
        if node.source_type != "nodes" or not node.source_ids:
            return []
        placeholders = ",".join("?" * len(node.source_ids))
        rows = self._conn.execute(
            f"""SELECT * FROM summary_nodes
                WHERE node_id IN ({placeholders})
                ORDER BY created_at""",
            node.source_ids,
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def describe_subtree(self, node_id: int) -> Dict[str, Any]:
        """Return metadata about a node's subtree without loading content."""
        node = self.get_node(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}

        children = []
        if node.source_type == "nodes":
            for child_node in self.get_source_nodes(node):
                children.append({
                    "node_id": child_node.node_id,
                    "depth": child_node.depth,
                    "token_count": child_node.token_count,
                    "source_token_count": child_node.source_token_count,
                    "expand_hint": child_node.expand_hint,
                })

        return {
            "node_id": node.node_id,
            "depth": node.depth,
            "token_count": node.token_count,
            "source_token_count": node.source_token_count,
            "source_type": node.source_type,
            "num_sources": len(node.source_ids),
            "expand_hint": node.expand_hint,
            "children": children,
        }

    # -- Helpers ------------------------------------------------------------

    def _row_to_node(self, row) -> SummaryNode:
        return SummaryNode(
            node_id=row[0],
            session_id=row[1],
            depth=row[2],
            summary=row[3],
            token_count=row[4],
            source_token_count=row[5],
            source_ids=json.loads(row[6]) if row[6] else [],
            source_type=row[7],
            created_at=row[8],
            expand_hint=row[9] or "",
        )

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
