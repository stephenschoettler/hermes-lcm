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


def _expand_message_sources(engine: "LCMEngine", node, max_tokens: int) -> list[dict[str, Any]]:
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
    return messages


def _expand_child_nodes(engine: "LCMEngine", node) -> list[dict[str, Any]]:
    children = [child for child in engine._dag.get_source_nodes(node) if child.session_id == engine._session_id]
    return [
        {
            "node_id": child.node_id,
            "depth": child.depth,
            "summary": child.summary[:1000],
            "token_count": child.token_count,
            "expand_hint": child.expand_hint,
        }
        for child in children
    ]


def _collect_context_blocks_for_node(engine: "LCMEngine", node, max_tokens: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {
            "type": "summary",
            "node_id": node.node_id,
            "depth": node.depth,
            "summary": node.summary,
            "expand_hint": node.expand_hint,
            "token_count": node.token_count,
        }
    ]

    if node.source_type == "messages":
        messages = _expand_message_sources(engine, node, max_tokens=max_tokens)
        if messages:
            blocks.append(
                {
                    "type": "messages",
                    "node_id": node.node_id,
                    "messages": messages,
                }
            )
    elif node.source_type == "nodes":
        children = _expand_child_nodes(engine, node)
        if children:
            blocks.append(
                {
                    "type": "child_nodes",
                    "node_id": node.node_id,
                    "children": children,
                }
            )

    return blocks


def _synthesize_expansion_answer(
    *,
    prompt: str,
    context_blocks: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    timeout: float,
) -> str:
    from agent.auxiliary_client import call_llm

    system_prompt = (
        "You answer questions using expanded LCM retrieval context. "
        "Be concise, factual, and grounded in the provided context. "
        "If the context is insufficient, say so plainly."
    )
    user_prompt = (
        f"QUESTION:\n{prompt}\n\n"
        "EXPANDED CONTEXT:\n"
        f"{json.dumps(context_blocks, ensure_ascii=False, indent=2)}"
    )
    call_kwargs = {
        "task": "compression",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if model:
        call_kwargs["model"] = model
    response = call_llm(**call_kwargs)
    content = response.choices[0].message.content
    if not isinstance(content, str):
        content = str(content) if content else ""
    return content.strip()


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
        messages = _expand_message_sources(engine, node, max_tokens=max_tokens)
        return json.dumps(
            {
                "node_id": node_id,
                "depth": node.depth,
                "source_type": "messages",
                "expanded": messages,
            }
        )

    if node.source_type == "nodes":
        children = _expand_child_nodes(engine, node)
        return json.dumps(
            {
                "node_id": node_id,
                "depth": node.depth,
                "source_type": "nodes",
                "expanded": children,
            }
        )

    return json.dumps({"error": f"Unknown source_type: {node.source_type}"})


def lcm_expand_query(args: Dict[str, Any], **kwargs) -> str:
    """Answer a question by expanding matching summaries or explicit node ids."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return json.dumps({"error": "prompt is required"})

    max_tokens = int(args.get("max_tokens", 2000))
    max_results = int(args.get("max_results", 5))
    query = str(args.get("query") or "").strip()
    raw_node_ids = args.get("node_ids") or []

    nodes = []
    if raw_node_ids:
        for node_id in raw_node_ids:
            node = _get_session_node(engine, int(node_id))
            if node is not None:
                nodes.append(node)
    elif query:
        nodes = engine._dag.search(query, session_id=engine._session_id, limit=max_results)
    else:
        return json.dumps({"error": "Provide either query or node_ids"})

    if not nodes:
        return json.dumps(
            {
                "prompt": prompt,
                "query": query,
                "answer": "No matching summaries found in the current session.",
                "node_ids": [],
                "matches": [],
            }
        )

    context_blocks = []
    for node in nodes[:max_results]:
        context_blocks.extend(_collect_context_blocks_for_node(engine, node, max_tokens=max_tokens))

    model = engine._config.expansion_model or engine._config.summary_model or ""
    timeout = engine._config.expansion_timeout_ms / 1000
    answer = _synthesize_expansion_answer(
        prompt=prompt,
        context_blocks=context_blocks,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    return json.dumps(
        {
            "prompt": prompt,
            "query": query,
            "answer": answer,
            "model": model,
            "node_ids": [node.node_id for node in nodes[:max_results]],
            "matches": [
                {
                    "node_id": node.node_id,
                    "depth": node.depth,
                    "summary": node.summary[:300],
                    "expand_hint": node.expand_hint,
                }
                for node in nodes[:max_results]
            ],
        }
    )
