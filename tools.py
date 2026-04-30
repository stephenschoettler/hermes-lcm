"""Tool handlers for LCM — the code that runs when the LLM calls each tool."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from .externalize import (
    extract_externalized_ref,
    find_externalized_payload_for_message,
    load_externalized_payload,
)
from .extraction import sanitize_pre_compaction_content
from .model_routing import apply_lcm_model_route
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


def _state_db_path_for_engine(engine: "LCMEngine") -> Path:
    hermes_home = getattr(engine, "_hermes_home", "") or ""
    if hermes_home:
        return Path(hermes_home).expanduser() / "state.db"
    db_path = Path(getattr(engine._store, "db_path", Path.home() / ".hermes" / "lcm.db"))
    return db_path.parent / "state.db"


def _has_lifecycle_fragmentation(stats: dict[str, Any]) -> bool:
    direct_mismatch_keys = (
        "lifecycle_current_missing_in_lcm_any",
        "lifecycle_last_finalized_missing_in_lcm_any",
        "lifecycle_current_missing_in_state",
        "lifecycle_last_finalized_missing_in_state",
        "lcm_message_sessions_missing_in_state",
        "lcm_node_sessions_missing_in_state",
    )
    lifecycle_rows = int(stats.get("lifecycle_rows", 0) or 0)
    missing_lifecycle_reference_keys = (
        "message_sessions_without_lifecycle_reference",
        "node_sessions_without_lifecycle_reference",
    )
    return (
        any(int(stats.get(key, 0) or 0) > 0 for key in direct_mismatch_keys)
        or (
            lifecycle_rows > 0
            and any(int(stats.get(key, 0) or 0) > 0 for key in missing_lifecycle_reference_keys)
        )
        or (bool(stats.get("state_db_checked")) and bool(stats.get("state_db_error")))
    )


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


def _truncate_text_to_token_budget(text: str, max_tokens: int) -> tuple[str, bool]:
    from .tokens import count_tokens

    if max_tokens <= 0 or not text:
        return "", bool(text)

    if count_tokens(text) <= max_tokens:
        return text, False

    low = 0
    high = len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid]
        if count_tokens(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best, True


def _parse_int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_non_negative_int(value: Any, default: int) -> int:
    return max(0, _parse_int_value(value, default))


def _parse_positive_int(value: Any, default: int) -> int:
    return max(1, _parse_int_value(value, default))


def _slice_content_for_response(content: str, max_tokens: int, content_offset: int = 0) -> dict[str, Any]:
    content = content or ""
    content_offset = min(max(0, content_offset), len(content))
    sliced, _ = _truncate_text_to_token_budget(content[content_offset:], max_tokens)
    if not sliced and content_offset < len(content):
        # A tiny token budget can fail to fit even the next character. Return one
        # character anyway so callers make deterministic, lossless cursor progress
        # instead of receiving has_more=true with the same content_offset forever.
        sliced = content[content_offset:content_offset + 1]
    next_content_offset = content_offset + len(sliced)
    has_more = next_content_offset < len(content)
    return {
        "content": sliced,
        "content_chars": len(content),
        "content_offset": content_offset,
        "content_returned_chars": len(sliced),
        "content_truncated": has_more,
        "next_content_offset": next_content_offset if has_more else 0,
        "has_more": has_more,
    }


def _full_content_slice(content: str, content_offset: int = 0) -> dict[str, Any]:
    content = content or ""
    content_offset = min(max(0, content_offset), len(content))
    sliced = content[content_offset:]
    return {
        "content": sliced,
        "content_chars": len(content),
        "content_offset": content_offset,
        "content_returned_chars": len(sliced),
        "content_truncated": False,
        "next_content_offset": 0,
        "has_more": False,
    }


def _is_compact_externalized_marker(content: str, ref: str | None) -> bool:
    if not ref or not content:
        return False
    if len(content) > 512:
        return False
    return content.startswith("[Externalized tool output:") or content.startswith("[GC'd externalized tool output:")


def _pagination_payload(
    *,
    total_sources: int,
    source_offset: int,
    content_offset: int,
    source_limit: int,
    returned_sources: int,
    next_source_offset: int | None,
    next_content_offset: int,
    has_more: bool,
) -> dict[str, Any]:
    if not has_more:
        next_source_offset = None
        next_content_offset = 0
    remaining_sources = 0
    if has_more and next_source_offset is not None:
        remaining_sources = max(0, total_sources - next_source_offset)
    return {
        "source_offset": source_offset,
        "content_offset": content_offset,
        "source_limit": source_limit,
        "returned_sources": returned_sources,
        "total_sources": total_sources,
        "next_source_offset": next_source_offset,
        "next_content_offset": next_content_offset,
        "has_more": has_more,
        "remaining_sources": remaining_sources,
    }


def _expand_message_sources(
    engine: "LCMEngine",
    node,
    max_tokens: int,
    *,
    source_offset: int = 0,
    source_limit: int | None = None,
    content_offset: int = 0,
    hydrate_externalized_content: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .tokens import count_tokens

    total_sources = len(node.source_ids)
    source_offset = min(max(0, source_offset), total_sources)
    remaining_source_count = max(0, total_sources - source_offset)
    if source_limit is None:
        source_limit = remaining_source_count
    else:
        source_limit = min(max(0, source_limit), remaining_source_count)
    content_offset = max(0, content_offset)
    source_ids = node.source_ids[source_offset:source_offset + source_limit]
    stored_by_id = engine._store.get_batch(source_ids)

    messages: list[dict[str, Any]] = []
    budget_used = 0
    next_source_offset: int | None = source_offset
    next_content_offset = content_offset
    has_more = source_offset < total_sources

    for relative_index, store_id in enumerate(source_ids):
        source_index = source_offset + relative_index
        remaining_tokens = max_tokens - budget_used
        if remaining_tokens <= 0:
            next_source_offset = source_index
            next_content_offset = 0
            has_more = True
            break
        stored = stored_by_id.get(store_id)
        if not stored or stored.get("session_id") != engine._session_id:
            next_source_offset = source_index + 1
            next_content_offset = 0
            has_more = next_source_offset < total_sources
            continue
        transcript_content = stored.get("content", "")
        content = transcript_content
        content_source = "message"
        externalized = None
        ref = extract_externalized_ref(transcript_content)
        if ref:
            externalized = _get_externalized_payload(engine, ref)
        if hydrate_externalized_content and externalized is not None:
            content = externalized.get("content", "")
            content_source = "externalized_payload"
        effective_content_offset = content_offset if source_index == source_offset else 0
        if not hydrate_externalized_content and _is_compact_externalized_marker(content, ref):
            sliced = _full_content_slice(content, effective_content_offset)
        else:
            sliced = _slice_content_for_response(content, remaining_tokens, effective_content_offset)
        expanded = {
            "store_id": stored["store_id"],
            "source_index": source_index,
            "role": stored["role"],
            "content": sliced["content"],
            "content_chars": sliced["content_chars"],
            "content_offset": sliced["content_offset"],
            "content_returned_chars": sliced["content_returned_chars"],
            "content_truncated": sliced["content_truncated"],
            "next_content_offset": sliced["next_content_offset"],
            "content_source": content_source,
        }
        if content_source == "externalized_payload":
            expanded["transcript_content"] = transcript_content
        if stored.get("role") == "tool":
            if externalized is not None:
                externalized_summary = dict(externalized)
                externalized_summary.pop("content", None)
                expanded["externalized"] = externalized_summary
            if "externalized" not in expanded:
                lookup_candidates = [transcript_content]
                sanitized_content = sanitize_pre_compaction_content(transcript_content)
                if sanitized_content != transcript_content:
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
        budget_used += count_tokens(sliced["content"])
        if sliced["has_more"]:
            next_source_offset = source_index
            next_content_offset = sliced["next_content_offset"]
            has_more = True
            break
        next_source_offset = source_index + 1
        next_content_offset = 0
        has_more = next_source_offset < total_sources
    else:
        has_more = (source_offset + source_limit) < total_sources
        next_source_offset = source_offset + source_limit if has_more else None
        next_content_offset = 0

    pagination = _pagination_payload(
        total_sources=total_sources,
        source_offset=source_offset,
        content_offset=content_offset,
        source_limit=source_limit,
        returned_sources=len(messages),
        next_source_offset=next_source_offset,
        next_content_offset=next_content_offset,
        has_more=has_more,
    )
    return messages, pagination


def _expand_child_nodes(
    engine: "LCMEngine",
    node,
    max_tokens: int | None = None,
    *,
    source_offset: int = 0,
    source_limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .tokens import count_tokens

    total_sources = len(node.source_ids)
    source_offset = min(max(0, source_offset), total_sources)
    remaining_source_count = max(0, total_sources - source_offset)
    if source_limit is None:
        source_limit = remaining_source_count
    else:
        source_limit = min(max(0, source_limit), remaining_source_count)
    selected_source_ids = node.source_ids[source_offset:source_offset + source_limit]
    children: list[tuple[int, Any]] = []
    for relative_index, child_id in enumerate(selected_source_ids):
        child = engine._dag.get_node(child_id)
        if child is None or child.session_id != engine._session_id:
            continue
        children.append((source_offset + relative_index, child))

    expanded: list[dict[str, Any]] = []
    budget_used = 0
    next_source_offset: int | None = None
    has_more = (source_offset + source_limit) < total_sources
    for source_index, child in children:
        summary = child.summary
        summary_truncated = False
        if max_tokens is not None:
            remaining_tokens = max_tokens - budget_used
            if remaining_tokens <= 0:
                next_source_offset = source_index
                has_more = True
                break
            summary, summary_truncated = _truncate_text_to_token_budget(summary, remaining_tokens)
        expanded.append(
            {
                "node_id": child.node_id,
                "source_index": source_index,
                "depth": child.depth,
                "summary": summary[:1000] if max_tokens is None else summary,
                "summary_truncated": summary_truncated or (max_tokens is None and len(child.summary) > 1000),
                "token_count": child.token_count,
                "source_token_count": child.source_token_count,
                "expand_hint": child.expand_hint,
            }
        )
        budget_used += count_tokens(summary)
        if summary_truncated:
            next_source_offset = source_index + 1
            has_more = next_source_offset < total_sources
            break
        next_source_offset = source_index + 1

    if has_more and next_source_offset is None:
        next_source_offset = source_offset + source_limit

    return expanded, _pagination_payload(
        total_sources=total_sources,
        source_offset=source_offset,
        content_offset=0,
        source_limit=source_limit,
        returned_sources=len(expanded),
        next_source_offset=next_source_offset,
        next_content_offset=0,
        has_more=has_more,
    )


def _collect_context_blocks_for_node(
    engine: "LCMEngine",
    node,
    max_tokens: int,
    *,
    hydrate_externalized_content: bool = False,
) -> list[dict[str, Any]]:
    from .tokens import count_tokens

    summary, summary_truncated = _truncate_text_to_token_budget(node.summary, max_tokens)
    blocks: list[dict[str, Any]] = [
        {
            "type": "summary",
            "node_id": node.node_id,
            "depth": node.depth,
            "summary": summary,
            "summary_truncated": summary_truncated,
            "expand_hint": node.expand_hint,
            "token_count": node.token_count,
        }
    ]
    remaining_tokens = max(0, max_tokens - count_tokens(summary))

    if node.source_type == "messages":
        messages, pagination = _expand_message_sources(
            engine,
            node,
            max_tokens=remaining_tokens,
            hydrate_externalized_content=hydrate_externalized_content,
        )
        if messages or pagination.get("has_more"):
            block = {
                "type": "messages",
                "node_id": node.node_id,
                "messages": messages,
                "pagination": pagination,
            }
            blocks.append(block)
    elif node.source_type == "nodes":
        children, pagination = _expand_child_nodes(engine, node, max_tokens=remaining_tokens)
        if children or pagination.get("has_more"):
            blocks.append(
                {
                    "type": "child_nodes",
                    "node_id": node.node_id,
                    "children": children,
                    "pagination": pagination,
                }
            )

    return blocks


def _context_content_token_count(blocks: list[dict[str, Any]]) -> int:
    from .tokens import count_tokens

    total = 0
    for block in blocks:
        if block.get("type") == "summary":
            total += count_tokens(str(block.get("summary") or ""))
        elif block.get("type") == "messages":
            for message in block.get("messages", []):
                total += count_tokens(str(message.get("content") or ""))
                total += count_tokens(str(message.get("transcript_content") or ""))
        elif block.get("type") == "child_nodes":
            total += sum(count_tokens(str(child.get("summary") or "")) for child in block.get("children", []))
    return total


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
    apply_lcm_model_route(call_kwargs, model)
    response = call_llm(**call_kwargs)
    content = response.choices[0].message.content
    if not isinstance(content, str):
        content = str(content) if content else ""
    from .escalation import _strip_reasoning_blocks
    return _strip_reasoning_blocks(content).strip()


def lcm_grep(args: Dict[str, Any], **kwargs) -> str:
    """Search raw messages + summaries in the active session with optional source filtering."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "No query provided"})

    limit = args.get("limit", 10)
    sort = normalize_search_sort(args.get("sort"))
    source_limit = max(limit * 4, limit, 20)
    requested_session_scope = str(args.get("session_scope", "current")).lower()
    session_scope = "current"
    source = str(args.get("source") or "").strip() or None
    if requested_session_scope != "current":
        logger.warning("Ignoring unsupported session_scope=%s for lcm_grep", requested_session_scope)
    session_id = engine._session_id
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
    response = {
        "query": query,
        "sort": sort,
        "session_scope": session_scope,
        "source": source,
        "total_results": len(results),
        "results": results[:limit],
    }
    if requested_session_scope != "current":
        response["ignored_session_scope"] = requested_session_scope
        response["scope_note"] = "lcm_grep is current-session only; use session_search for broad cross-session recall."
    return json.dumps(response)


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
    max_tokens = _parse_positive_int(args.get("max_tokens", 4000), 4000)
    source_offset = _parse_non_negative_int(args.get("source_offset", 0), 0)
    source_limit_arg = args.get("source_limit")
    source_limit = _parse_positive_int(source_limit_arg, 0) if source_limit_arg is not None else None
    content_offset = _parse_non_negative_int(args.get("content_offset", 0), 0)
    if externalized_ref:
        payload = _get_externalized_payload(engine, externalized_ref)
        if payload is None:
            return json.dumps({"error": f"Externalized payload {externalized_ref} not found in current session"})
        content = payload.get("content", "")
        sliced = _slice_content_for_response(content, max_tokens, content_offset)
        return json.dumps(
            {
                "externalized_ref": externalized_ref,
                "source_type": "externalized_payload",
                "kind": payload.get("kind", "tool_result"),
                "tool_call_id": payload.get("tool_call_id", ""),
                "session_id": payload.get("session_id", ""),
                "content_chars": payload.get("content_chars", len(content)),
                "content_bytes": payload.get("content_bytes", 0),
                "content": sliced["content"],
                "content_offset": sliced["content_offset"],
                "content_returned_chars": sliced["content_returned_chars"],
                "content_truncated": sliced["content_truncated"],
                "next_content_offset": sliced["next_content_offset"],
                "has_more": sliced["has_more"],
            }
        )

    node_id = args.get("node_id")
    if node_id is None:
        return json.dumps({"error": "node_id or externalized_ref is required"})

    node = _get_session_node(engine, node_id)
    if node is None:
        return json.dumps({"error": f"Node {node_id} not found in current session"})

    if node.source_type == "messages":
        messages, pagination = _expand_message_sources(
            engine,
            node,
            max_tokens=max_tokens,
            source_offset=source_offset,
            source_limit=source_limit,
            content_offset=content_offset,
        )
        return json.dumps(
            {
                "node_id": node_id,
                "depth": node.depth,
                "source_type": "messages",
                "expanded": messages,
                "pagination": pagination,
            }
        )

    if node.source_type == "nodes":
        children, pagination = _expand_child_nodes(
            engine,
            node,
            max_tokens=max_tokens,
            source_offset=source_offset,
            source_limit=source_limit,
        )
        return json.dumps(
            {
                "node_id": node_id,
                "depth": node.depth,
                "source_type": "nodes",
                "expanded": children,
                "pagination": pagination,
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

    def _parse_int_arg(name: str, default: int) -> tuple[int | None, str | None]:
        raw_value = args.get(name, default)
        try:
            return int(raw_value), None
        except (TypeError, ValueError):
            return None, f"{name} must be an integer"

    max_tokens, max_tokens_error = _parse_int_arg("max_tokens", 2000)
    if max_tokens_error:
        return json.dumps({"error": max_tokens_error})
    max_tokens = max(1, max_tokens)
    context_default = max(max_tokens, int(getattr(engine._config, "expansion_context_tokens", 32_000) or 32_000))
    context_max_tokens, context_max_tokens_error = _parse_int_arg("context_max_tokens", context_default)
    if context_max_tokens_error:
        return json.dumps({"error": context_max_tokens_error})
    context_max_tokens = max(1, context_max_tokens)

    max_results, max_results_error = _parse_int_arg("max_results", 5)
    if max_results_error:
        return json.dumps({"error": max_results_error})

    query = str(args.get("query") or "").strip()
    raw_node_ids = args.get("node_ids") or []

    nodes = []
    if raw_node_ids:
        for node_id in raw_node_ids:
            try:
                parsed_node_id = int(node_id)
            except (TypeError, ValueError):
                return json.dumps({"error": "node_ids must contain only integers"})
            node = _get_session_node(engine, parsed_node_id)
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
    context_budget_used = 0
    for node in nodes[:max_results]:
        remaining_context_tokens = max(0, context_max_tokens - context_budget_used)
        node_blocks = _collect_context_blocks_for_node(
            engine,
            node,
            max_tokens=remaining_context_tokens,
            hydrate_externalized_content=True,
        )
        context_blocks.extend(node_blocks)
        context_budget_used += _context_content_token_count(node_blocks)

    context_pagination = []
    for block in context_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "summary" and block.get("summary_truncated"):
            context_pagination.append(
                {
                    "node_id": block.get("node_id"),
                    "type": "summary",
                    "summary_truncated": True,
                    "expand_args": {"node_id": block.get("node_id")},
                }
            )
            continue

        if block_type == "child_nodes":
            for child in block.get("children", []):
                if child.get("summary_truncated"):
                    child_node_id = child.get("node_id")
                    context_pagination.append(
                        {
                            "node_id": block.get("node_id"),
                            "type": "child_summary",
                            "child_node_id": child_node_id,
                            "source_index": child.get("source_index"),
                            "summary_truncated": True,
                            "expand_args": {"node_id": child_node_id},
                        }
                    )

        pagination = block.get("pagination")
        if not pagination or not pagination.get("has_more"):
            continue

        item = {
            "node_id": block.get("node_id"),
            "type": block_type,
            "pagination": pagination,
        }
        if block_type == "messages":
            truncated_message = next(
                (message for message in block.get("messages", []) if message.get("content_truncated")),
                None,
            )
            if truncated_message:
                item["source_index"] = truncated_message.get("source_index")
                item["content_source"] = truncated_message.get("content_source")
                externalized = truncated_message.get("externalized") or {}
                externalized_ref = externalized.get("ref")
                if externalized_ref:
                    item["externalized_ref"] = externalized_ref
                    item["tool_call_id"] = externalized.get("tool_call_id")
                if truncated_message.get("content_source") == "externalized_payload" and externalized_ref:
                    item["expand_args"] = {
                        "externalized_ref": externalized_ref,
                        "content_offset": pagination.get("next_content_offset") or 0,
                    }
                else:
                    item["expand_args"] = {
                        "node_id": block.get("node_id"),
                        "source_offset": pagination.get("next_source_offset") or 0,
                        "content_offset": pagination.get("next_content_offset") or 0,
                    }
            else:
                item["expand_args"] = {
                    "node_id": block.get("node_id"),
                    "source_offset": pagination.get("next_source_offset") or 0,
                    "content_offset": pagination.get("next_content_offset") or 0,
                }
        elif block_type == "child_nodes":
            item["expand_args"] = {
                "node_id": block.get("node_id"),
                "source_offset": pagination.get("next_source_offset") or 0,
            }
        context_pagination.append(item)

    context_truncated = any(
        bool(item.get("summary_truncated")) or bool(item.get("pagination", {}).get("has_more"))
        for item in context_pagination
    )

    selected_nodes = nodes[:max_results]
    matches = [
        {
            "node_id": node.node_id,
            "depth": node.depth,
            "summary": node.summary[:300],
            "expand_hint": node.expand_hint,
        }
        for node in selected_nodes
    ]
    node_ids = [node.node_id for node in selected_nodes]

    def _degraded_payload(reason: str, *, include_timeout: bool = False) -> str:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "query": query,
            "error": reason,
            "degraded": True,
            "model": model,
            "max_tokens": max_tokens,
            "context_max_tokens": context_max_tokens,
            "context_truncated": context_truncated,
            "context_pagination": context_pagination,
            "node_ids": node_ids,
            "matches": matches,
        }
        if include_timeout:
            payload["timeout_seconds"] = timeout
        return json.dumps(payload)

    model = engine._config.expansion_model or engine._config.summary_model or ""
    timeout = engine._config.expansion_timeout_ms / 1000
    try:
        answer = _synthesize_expansion_answer(
            prompt=prompt,
            context_blocks=context_blocks,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except TimeoutError:
        logger.warning("LCM expand_query synthesis timed out after %.3fs", timeout)
        return _degraded_payload(
            f"lcm_expand_query synthesis timed out after {timeout:.3g}s",
            include_timeout=True,
        )

    answer = str(answer).strip() if answer is not None else ""
    if not answer:
        logger.warning("LCM expand_query synthesis returned an empty answer")
        return _degraded_payload("lcm_expand_query synthesis returned an empty answer")

    return json.dumps(
        {
            "prompt": prompt,
            "query": query,
            "answer": answer,
            "model": model,
            "max_tokens": max_tokens,
            "context_max_tokens": context_max_tokens,
            "context_truncated": context_truncated,
            "context_pagination": context_pagination,
            "node_ids": node_ids,
            "matches": matches,
        }
    )


def lcm_status(args: Dict[str, Any], **kwargs) -> str:
    """Quick health overview of the LCM engine for the current session."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    session_id = engine._session_id
    if not session_id:
        return json.dumps({
            "error": "No active session",
            "runtime_identity": engine.get_runtime_identity(),
        })

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
    full_status = engine.get_status()
    lifecycle = full_status.get("lifecycle")
    lifecycle_fragmentation = full_status.get("lifecycle_fragmentation")
    source_lineage = full_status.get("source_lineage")
    runtime_identity = full_status.get("runtime_identity")

    return json.dumps({
        "session_id": session_id,
        "compression_count": engine.compression_count,
        "context_length": engine.context_length,
        "threshold_tokens": engine.threshold_tokens,
        "last_prompt_tokens": engine.last_prompt_tokens,
        "last_input_tokens": engine.last_input_tokens,
        "last_output_tokens": engine.last_output_tokens,
        "last_cache_read_tokens": engine.last_cache_read_tokens,
        "last_cache_write_tokens": engine.last_cache_write_tokens,
        "last_reasoning_tokens": engine.last_reasoning_tokens,
        "cache_metrics_available": engine.cache_metrics_available,
        "cache_read_ratio": round(engine.cache_read_ratio, 4),
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
        "source_lineage": source_lineage,
        "runtime_identity": runtime_identity,
        "lifecycle": lifecycle,
        "lifecycle_fragmentation": lifecycle_fragmentation,
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
            """
            SELECT COUNT(*)
            FROM messages_fts
            JOIN messages ON messages_fts.rowid = messages.store_id
            WHERE messages.session_id = ?
            """,
            (session_id,),
        ).fetchone()[0]
        checks.append({
            "check": "fts_index_sync",
            "status": "pass" if fts_count >= msg_count else "warn",
            "detail": f"{fts_count} session FTS rows, {msg_count} session messages",
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

    # 5. Source-lineage hygiene
    try:
        source_stats = engine._store.get_source_stats()
        checks.append({
            "check": "source_lineage_hygiene",
            "status": "pass",
            "detail": {
                **source_stats,
                "normalization_mode": "backcompat-normalization",
            },
        })
    except Exception as e:
        checks.append({
            "check": "source_lineage_hygiene",
            "status": "fail",
            "detail": str(e),
        })

    # 6. Lifecycle/session fragmentation
    try:
        lifecycle_fragmentation = engine._lifecycle.get_fragmentation_stats(
            state_db_path=_state_db_path_for_engine(engine)
        )
        checks.append({
            "check": "lifecycle_fragmentation",
            "status": "warn" if _has_lifecycle_fragmentation(lifecycle_fragmentation) else "pass",
            "detail": lifecycle_fragmentation,
        })
    except Exception as e:
        checks.append({
            "check": "lifecycle_fragmentation",
            "status": "fail",
            "detail": str(e),
        })

    # 7. Context pressure
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
        "runtime_identity": engine.get_runtime_identity(),
        "checks": checks,
    })
