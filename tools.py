"""Tool handlers for LCM — the code that runs when the LLM calls each tool."""

import json
import logging
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import LCMEngine

logger = logging.getLogger(__name__)

# The engine instance is set by the engine on init
_engine: "LCMEngine | None" = None


def set_engine(engine: "LCMEngine") -> None:
    global _engine
    _engine = engine


def lcm_grep(args: Dict[str, Any], **kwargs) -> str:
    """Search across the full DAG and raw messages."""
    if _engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "No query provided"})

    limit = args.get("limit", 10)
    session_id = _engine._session_id

    results = []

    # Search raw messages
    try:
        msg_hits = _engine._store.search(query, session_id=session_id, limit=limit)
        for hit in msg_hits:
            results.append({
                "type": "message",
                "depth": "raw",
                "store_id": hit["store_id"],
                "role": hit["role"],
                "snippet": hit.get("snippet", hit.get("content", "")[:200]),
            })
    except Exception as e:
        logger.debug("Message search failed: %s", e)

    # Search summary nodes
    try:
        node_hits = _engine._dag.search(query, session_id=session_id, limit=limit)
        for node in node_hits:
            results.append({
                "type": "summary",
                "depth": f"d{node.depth}",
                "node_id": node.node_id,
                "snippet": node.summary[:300],
                "token_count": node.token_count,
                "expand_hint": node.expand_hint,
            })
    except Exception as e:
        logger.debug("Node search failed: %s", e)

    # Sort by relevance (raw messages first, then by depth ascending)
    results.sort(key=lambda r: (0 if r["type"] == "message" else 1, r.get("depth", "")))

    return json.dumps({
        "query": query,
        "total_results": len(results),
        "results": results[:limit],
    })


def lcm_describe(args: Dict[str, Any], **kwargs) -> str:
    """Inspect a node's subtree or get session DAG overview."""
    if _engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    node_id = args.get("node_id")
    session_id = _engine._session_id

    if node_id is not None:
        info = _engine._dag.describe_subtree(node_id)
        return json.dumps(info)

    # Session overview: count nodes at each depth
    overview = {
        "session_id": session_id,
        "store_message_count": _engine._store.get_session_count(session_id),
        "depths": {},
    }

    for depth in range(10):  # check up to d9
        count = _engine._dag.count_at_depth(session_id, depth)
        if count == 0 and depth > 0:
            break
        if count > 0:
            nodes = _engine._dag.get_session_nodes(session_id, depth=depth, limit=100)
            overview["depths"][f"d{depth}"] = {
                "count": count,
                "total_tokens": sum(n.token_count for n in nodes),
                "total_source_tokens": sum(n.source_token_count for n in nodes),
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "token_count": n.token_count,
                        "expand_hint": n.expand_hint,
                    }
                    for n in nodes[:20]  # cap at 20 per depth for display
                ],
            }

    return json.dumps(overview)


def lcm_expand(args: Dict[str, Any], **kwargs) -> str:
    """Expand a summary node to its source content."""
    if _engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    node_id = args.get("node_id")
    if node_id is None:
        return json.dumps({"error": "node_id is required"})

    max_tokens = args.get("max_tokens", 4000)

    node = _engine._dag.get_node(node_id)
    if not node:
        return json.dumps({"error": f"Node {node_id} not found"})

    if node.source_type == "messages":
        # Expand to raw messages
        messages = []
        from .tokens import count_tokens
        budget_used = 0
        for sid in node.source_ids:
            stored = _engine._store.get(sid)
            if not stored:
                continue
            content = stored.get("content", "")
            msg_tokens = count_tokens(content)
            if budget_used + msg_tokens > max_tokens and messages:
                messages.append({
                    "note": f"Truncated — {len(node.source_ids) - len(messages)} more messages available",
                })
                break
            messages.append({
                "store_id": stored["store_id"],
                "role": stored["role"],
                "content": content[:2000] if len(content) > 2000 else content,
            })
            budget_used += msg_tokens

        return json.dumps({
            "node_id": node_id,
            "depth": node.depth,
            "source_type": "messages",
            "expanded": messages,
        })

    elif node.source_type == "nodes":
        # Expand to child summaries
        children = _engine._dag.get_source_nodes(node)
        return json.dumps({
            "node_id": node_id,
            "depth": node.depth,
            "source_type": "nodes",
            "expanded": [
                {
                    "node_id": c.node_id,
                    "depth": c.depth,
                    "summary": c.summary[:1000],
                    "token_count": c.token_count,
                    "expand_hint": c.expand_hint,
                }
                for c in children
            ],
        })

    return json.dumps({"error": f"Unknown source_type: {node.source_type}"})
