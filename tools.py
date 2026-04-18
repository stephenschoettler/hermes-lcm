"""Tool handlers for LCM — the code that runs when the LLM calls each tool."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, TYPE_CHECKING

from .externalize import (
    extract_externalized_ref,
    find_externalized_payload_for_message,
    load_externalized_payload,
)
from .extraction import sanitize_pre_compaction_content
from .search_query import AGE_DECAY_RATE, normalize_search_sort

if TYPE_CHECKING:
    from .engine import LCMEngine


logger = logging.getLogger(__name__)


def _combined_result_sort_key(result: dict[str, Any], sort: str) -> tuple:
    sort_timestamp = float(result.get("_sort_ts") or 0.0)
    rank = result.get("_sort_rank")
    rank_value = float(rank) if rank is not None else float("inf")
    directness = float(result.get("_sort_directness") or 0.0)
    type_bias = 0 if result.get("type") == "message" else 1
    role = result.get("role")
    if role == "user":
        role_bias = 0
    elif role == "assistant":
        role_bias = 1
    elif role == "tool":
        role_bias = 2
    else:
        role_bias = 1

    effective_directness = directness if result.get("type") == "message" else (directness * 0.8)

    if sort == "relevance":
        return (rank_value, -effective_directness, role_bias, -sort_timestamp, type_bias)

    if sort == "hybrid":
        age_hours = max(0.0, (time.time() - sort_timestamp) / 3600.0)
        blended = rank_value / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("inf")
        summary_override = int(result.get("_hybrid_summary_override") or 0)
        return (-summary_override, blended, -effective_directness, role_bias, -sort_timestamp, type_bias)

    if result.get("type") == "message":
        return (-sort_timestamp, type_bias, role_bias, rank_value, 0.0, float("inf"))
    return (-sort_timestamp, type_bias, 0, rank_value, 0.0, role_bias)


def _require_engine(kwargs: Dict[str, Any]) -> "LCMEngine | None":
    engine = kwargs.get("engine")
    return engine if engine is not None else None


def _get_session_node(engine: "LCMEngine", node_id: int):
    node = engine._dag.get_node(node_id)
    if node is None or node.session_id != engine._session_id:
        return None
    return node


def _get_externalized_payload(engine: "LCMEngine", ref: str) -> dict[str, Any] | None:
    payload = load_externalized_payload(ref, config=engine._config, hermes_home=engine._hermes_home)
    if payload is None:
        return None
    payload_session_id = payload.get("session_id") or ""
    if payload_session_id and payload_session_id != engine._session_id:
        return None
    return payload


def _expand_message_sources(engine: "LCMEngine", node, max_tokens: int) -> list[dict[str, Any]]:
    from .tokens import count_tokens

    stored_by_id = engine._store.get_batch(node.source_ids)

    messages = []
    budget_used = 0
    for store_id in node.source_ids:
        stored = stored_by_id.get(store_id)
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
        expanded = {
            "store_id": stored["store_id"],
            "role": stored["role"],
            "content": content[:2000] if len(content) > 2000 else content,
        }
        if stored.get("role") == "tool":
            ref = extract_externalized_ref(content)
            if ref:
                externalized = _get_externalized_payload(engine, ref)
                if externalized is not None:
                    externalized.pop("content", None)
                    expanded["externalized"] = externalized
            if "externalized" not in expanded:
                lookup_candidates = [content]
                sanitized_content = sanitize_pre_compaction_content(content)
                if sanitized_content != content:
                    lookup_candidates.insert(0, sanitized_content)
                for candidate in lookup_candidates:
                    externalized = find_externalized_payload_for_message(
                        candidate,
                        tool_call_id=stored.get("tool_call_id", ""),
                        session_id=stored.get("session_id", ""),
                        config=engine._config,
                        hermes_home=engine._hermes_home,
                    )
                    if externalized is not None:
                        expanded["externalized"] = externalized
                        break
        messages.append(expanded)
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
    """Search raw messages + summaries using independent session/source filters.

    ``session_scope`` decides which sessions are eligible.
    ``source`` then filters raw rows directly and summaries by descendant source
    lineage within that eligible set.
    """
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "No query provided"})

    limit = args.get("limit", 10)
    sort = normalize_search_sort(args.get("sort"))
    source_limit = max(limit * 4, limit, 20)
    session_scope = str(args.get("session_scope", "current")).lower()
    source = str(args.get("source") or "").strip() or None
    session_id = None if session_scope == "all" else engine._session_id
    results = []

    try:
        msg_hits = engine._store.search(
            query,
            session_id=session_id,
            limit=source_limit,
            sort=sort,
            source=source,
        )
        for hit in msg_hits:
            results.append(
                {
                    "type": "message",
                    "depth": "raw",
                    "store_id": hit["store_id"],
                    "session_id": hit["session_id"],
                    "source": hit.get("source") or "",
                    "role": hit["role"],
                    "snippet": hit.get("snippet", hit.get("content", "")[:200]),
                    "_sort_ts": hit.get("timestamp", 0),
                    "_sort_rank": hit.get("search_rank"),
                    "_sort_directness": hit.get("_directness_score") or 0.0,
                }
            )
    except Exception as exc:
        logger.warning("Message search failed: %s", exc)

    try:
        node_hits = engine._dag.search(
            query,
            session_id=session_id,
            limit=source_limit,
            sort=sort,
            source=source,
        )
        for node in node_hits:
            results.append(
                {
                    "type": "summary",
                    "depth": f"d{node.depth}",
                    "node_id": node.node_id,
                    "session_id": node.session_id,
                    "snippet": node.summary[:300],
                    "token_count": node.token_count,
                    "expand_hint": node.expand_hint,
                    "earliest_at": node.earliest_at,
                    "latest_at": node.latest_at,
                    "_sort_ts": node.latest_at or node.created_at,
                    "_sort_rank": node.search_rank,
                    "_sort_directness": node.search_directness or 0.0,
                }
            )
    except Exception as exc:
        logger.warning("Node search failed: %s", exc)

    if sort == "hybrid":
        max_message_directness = max(
            (float(result.get("_sort_directness") or 0.0) for result in results if result.get("type") == "message"),
            default=0.0,
        )
        for result in results:
            if result.get("type") == "summary":
                result["_hybrid_summary_override"] = 1 if float(result.get("_sort_directness") or 0.0) >= (max_message_directness + 8.0) else 0

    results.sort(key=lambda result: _combined_result_sort_key(result, sort))
    for result in results:
        result.pop("_sort_ts", None)
        result.pop("_sort_rank", None)
        result.pop("_sort_directness", None)
        result.pop("_hybrid_summary_override", None)
    return json.dumps(
        {
            "query": query,
            "sort": sort,
            "session_scope": session_scope,
            "source": source,
            "total_results": len(results),
            "results": results[:limit],
        }
    )


def lcm_describe(args: Dict[str, Any], **kwargs) -> str:
    """Inspect a summary node's subtree or get session DAG overview."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    externalized_ref = str(args.get("externalized_ref") or "").strip()
    if externalized_ref:
        payload = _get_externalized_payload(engine, externalized_ref)
        if payload is None:
            return json.dumps({"error": f"Externalized payload {externalized_ref} not found in current session"})
        return json.dumps(
            {
                "externalized_ref": externalized_ref,
                "kind": payload.get("kind", "tool_result"),
                "tool_call_id": payload.get("tool_call_id", ""),
                "session_id": payload.get("session_id", ""),
                "content_chars": payload.get("content_chars", 0),
                "content_bytes": payload.get("content_bytes", 0),
                "created_at": payload.get("created_at"),
                "content_preview": (payload.get("content") or "")[:500],
            }
        )

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

    externalized_ref = str(args.get("externalized_ref") or "").strip()
    if externalized_ref:
        payload = _get_externalized_payload(engine, externalized_ref)
        if payload is None:
            return json.dumps({"error": f"Externalized payload {externalized_ref} not found in current session"})
        return json.dumps(
            {
                "externalized_ref": externalized_ref,
                "source_type": "externalized_payload",
                "kind": payload.get("kind", "tool_result"),
                "tool_call_id": payload.get("tool_call_id", ""),
                "session_id": payload.get("session_id", ""),
                "content_chars": payload.get("content_chars", 0),
                "content_bytes": payload.get("content_bytes", 0),
                "content": payload.get("content", ""),
            }
        )

    node_id = args.get("node_id")
    if node_id is None:
        return json.dumps({"error": "node_id or externalized_ref is required"})

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


def lcm_status(args: Dict[str, Any], **kwargs) -> str:
    """Quick health overview of the LCM engine for the current session."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    session_id = engine._session_id
    if not session_id:
        return json.dumps({"error": "No active session"})

    # Store stats
    store_messages = engine._store.get_session_count(session_id)
    store_tokens = engine._store.get_session_token_total(session_id)

    # DAG stats by depth
    all_nodes = engine._dag.get_session_nodes(session_id)
    depths: dict[int, dict] = {}
    for node in all_nodes:
        d = depths.setdefault(node.depth, {"count": 0, "tokens": 0, "source_tokens": 0})
        d["count"] += 1
        d["tokens"] += node.token_count
        d["source_tokens"] += node.source_token_count

    total_dag_tokens = sum(d["tokens"] for d in depths.values())
    total_source_tokens = sum(d["source_tokens"] for d in depths.values())
    compression_ratio = round(total_source_tokens / total_dag_tokens, 1) if total_dag_tokens > 0 else 0
    lifecycle = engine.get_status().get("lifecycle")

    return json.dumps({
        "session_id": session_id,
        "compression_count": engine.compression_count,
        "context_length": engine.context_length,
        "threshold_tokens": engine.threshold_tokens,
        "last_prompt_tokens": engine.last_prompt_tokens,
        "store": {
            "messages": store_messages,
            "estimated_tokens": store_tokens,
        },
        "dag": {
            "total_nodes": len(all_nodes),
            "total_tokens": total_dag_tokens,
            "compression_ratio": f"{compression_ratio}:1",
            "depths": {
                f"d{depth}": info for depth, info in sorted(depths.items())
            },
        },
        "config": {
            "fresh_tail_count": engine._config.fresh_tail_count,
            "leaf_chunk_tokens": engine._config.leaf_chunk_tokens,
            "dynamic_leaf_chunk_enabled": engine._config.dynamic_leaf_chunk_enabled,
            "dynamic_leaf_chunk_max": engine._config.dynamic_leaf_chunk_max,
            "cache_friendly_condensation_enabled": engine._config.cache_friendly_condensation_enabled,
            "cache_friendly_min_debt_groups": engine._config.cache_friendly_min_debt_groups,
            "deferred_maintenance_enabled": engine._config.deferred_maintenance_enabled,
            "deferred_maintenance_max_passes": engine._config.deferred_maintenance_max_passes,
            "context_threshold": engine._config.context_threshold,
            "max_depth": engine._config.incremental_max_depth,
            "condensation_fanin": engine._config.condensation_fanin,
            "summary_model": engine._config.summary_model or "(auxiliary)",
            "expansion_model": engine._config.expansion_model or "(summary model)",
        },
        "session_filters": {
            "ignored": engine._session_ignored,
            "stateless": engine._session_stateless,
        },
        "lifecycle": lifecycle,
    })


def lcm_doctor(args: Dict[str, Any], **kwargs) -> str:
    """Run diagnostics on the LCM database and configuration."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    checks: list[dict] = []
    session_id = engine._session_id

    # 1. Database integrity
    try:
        result = engine._store._conn.execute("PRAGMA integrity_check").fetchone()
        ok = result and result[0] == "ok"
        checks.append({
            "check": "database_integrity",
            "status": "pass" if ok else "fail",
            "detail": result[0] if result else "no response",
        })
    except Exception as e:
        checks.append({
            "check": "database_integrity",
            "status": "fail",
            "detail": str(e),
        })

    # 2. FTS index sync
    try:
        msg_count = engine._store._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        fts_count = engine._store._conn.execute(
            "SELECT COUNT(*) FROM messages_fts"
        ).fetchone()[0]
        checks.append({
            "check": "fts_index_sync",
            "status": "pass" if fts_count >= msg_count else "warn",
            "detail": f"{fts_count} FTS rows, {msg_count} session messages",
        })
    except Exception as e:
        checks.append({
            "check": "fts_index_sync",
            "status": "fail",
            "detail": str(e),
        })

    # 3. Orphaned DAG nodes (nodes referencing store_ids that don't exist)
    try:
        all_nodes = engine._dag.get_session_nodes(session_id)
        orphaned = 0
        for node in all_nodes:
            if node.source_type == "messages":
                for sid in node.source_ids:
                    stored = engine._store.get(sid)
                    if stored is None:
                        orphaned += 1
                        break
        checks.append({
            "check": "orphaned_dag_nodes",
            "status": "pass" if orphaned == 0 else "warn",
            "detail": f"{orphaned} nodes reference missing store messages" if orphaned else "all nodes have valid sources",
        })
    except Exception as e:
        checks.append({
            "check": "orphaned_dag_nodes",
            "status": "fail",
            "detail": str(e),
        })

    # 4. Configuration validation
    config_warnings = []
    c = engine._config
    if c.fresh_tail_count < 2:
        config_warnings.append("fresh_tail_count < 2 may cause aggressive compaction")
    if c.context_threshold > 0.95:
        config_warnings.append("context_threshold > 0.95 leaves very little headroom")
    if c.context_threshold < 0.3:
        config_warnings.append("context_threshold < 0.3 triggers compaction very early")
    if c.condensation_fanin < 2:
        config_warnings.append("condensation_fanin < 2 creates excessive depth growth")
    if c.incremental_max_depth == 0:
        config_warnings.append("incremental_max_depth=0 disables condensation entirely")

    checks.append({
        "check": "config_validation",
        "status": "pass" if not config_warnings else "warn",
        "detail": config_warnings if config_warnings else "all settings within normal ranges",
    })

    # 5. Context pressure
    if engine.context_length > 0:
        usage_pct = round(engine.last_prompt_tokens / engine.context_length * 100, 1) if engine.context_length else 0
        threshold_pct = round(c.context_threshold * 100, 1)
        checks.append({
            "check": "context_pressure",
            "status": "pass" if usage_pct < threshold_pct else "warn",
            "detail": f"{usage_pct}% used, compaction triggers at {threshold_pct}%",
        })

    overall = "healthy"
    if any(ch["status"] == "fail" for ch in checks):
        overall = "unhealthy"
    elif any(ch["status"] == "warn" for ch in checks):
        overall = "warnings"

    return json.dumps({
        "overall": overall,
        "checks": checks,
    })
