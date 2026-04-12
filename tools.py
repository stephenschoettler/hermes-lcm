"""Tool handlers for LCM — the code that runs when the LLM calls each tool."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import LCMEngine

logger = logging.getLogger(__name__)


def _require_engine(kwargs: Dict[str, Any]) -> "LCMEngine | None":
    engine = kwargs.get("engine")
    return engine if engine is not None else None


def _get_session_node(engine: "LCMEngine", node_id: int):
    node = engine._dag.get_node(node_id)
    if node is None or node.session_id != engine._session_id:
        return None
    return node


def lcm_grep(args: Dict[str, Any], **kwargs) -> str:
    """Search across the full DAG and raw messages for the current session."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "No query provided"})

    limit = args.get("limit", 10)
    session_id = engine._session_id
    results = []

    try:
        msg_hits = engine._store.search(query, session_id=session_id, limit=limit)
        for hit in msg_hits:
            results.append(
                {
                    "type": "message",
                    "depth": "raw",
                    "store_id": hit["store_id"],
                    "role": hit["role"],
                    "snippet": hit.get("snippet", hit.get("content", "")[:200]),
                }
            )
    except Exception as exc:
        logger.debug("Message search failed: %s", exc)

    try:
        node_hits = engine._dag.search(query, session_id=session_id, limit=limit)
        for node in node_hits:
            results.append(
                {
                    "type": "summary",
                    "depth": f"d{node.depth}",
                    "node_id": node.node_id,
                    "snippet": node.summary[:300],
                    "token_count": node.token_count,
                    "expand_hint": node.expand_hint,
                }
            )
    except Exception as exc:
        logger.debug("Node search failed: %s", exc)

    results.sort(key=lambda result: (0 if result["type"] == "message" else 1, result.get("depth", "")))
    return json.dumps({"query": query, "total_results": len(results), "results": results[:limit]})


def lcm_describe(args: Dict[str, Any], **kwargs) -> str:
    """Inspect a node's subtree or get session DAG overview."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    node_id = args.get("node_id")
    session_id = engine._session_id

    if node_id is not None:
        node = _get_session_node(engine, node_id)
        if node is None:
            return json.dumps({"error": f"Node {node_id} not found in current session"})
        info = engine._dag.describe_subtree(node_id)
        return json.dumps(info)

    all_nodes = engine._dag.get_session_nodes(session_id)
    overview = {
        "session_id": session_id,
        "store_message_count": engine._store.get_session_count(session_id),
        "depths": {},
    }

    for depth in sorted({node.depth for node in all_nodes}):
        nodes = [node for node in all_nodes if node.depth == depth]
        overview["depths"][f"d{depth}"] = {
            "count": len(nodes),
            "total_tokens": sum(node.token_count for node in nodes),
            "total_source_tokens": sum(node.source_token_count for node in nodes),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "token_count": node.token_count,
                    "expand_hint": node.expand_hint,
                }
                for node in nodes[:20]
            ],
        }

    return json.dumps(overview)


def lcm_expand(args: Dict[str, Any], **kwargs) -> str:
    """Expand a summary node to its source content."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    node_id = args.get("node_id")
    if node_id is None:
        return json.dumps({"error": "node_id is required"})

    node = _get_session_node(engine, node_id)
    if node is None:
        return json.dumps({"error": f"Node {node_id} not found in current session"})

    max_tokens = args.get("max_tokens", 4000)

    if node.source_type == "messages":
        from .tokens import count_tokens

        messages = []
        budget_used = 0
        for store_id in node.source_ids:
            stored = engine._store.get(store_id)
            if not stored or stored.get("session_id") != engine._session_id:
                continue
            content = stored.get("content", "")
            msg_tokens = count_tokens(content)
            if budget_used + msg_tokens > max_tokens and messages:
                messages.append(
                    {
                        "note": f"Truncated — {len(node.source_ids) - len(messages)} more messages available",
                    }
                )
                break
            messages.append(
                {
                    "store_id": stored["store_id"],
                    "role": stored["role"],
                    "content": content[:2000] if len(content) > 2000 else content,
                }
            )
            budget_used += msg_tokens

        return json.dumps(
            {
                "node_id": node_id,
                "depth": node.depth,
                "source_type": "messages",
                "expanded": messages,
            }
        )

    if node.source_type == "nodes":
        children = [child for child in engine._dag.get_source_nodes(node) if child.session_id == engine._session_id]
        return json.dumps(
            {
                "node_id": node_id,
                "depth": node.depth,
                "source_type": "nodes",
                "expanded": [
                    {
                        "node_id": child.node_id,
                        "depth": child.depth,
                        "summary": child.summary[:1000],
                        "token_count": child.token_count,
                        "expand_hint": child.expand_hint,
                    }
                    for child in children
                ],
            }
        )

    return json.dumps({"error": f"Unknown source_type: {node.source_type}"})
