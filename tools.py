"""Tool handlers for LCM — the code that runs when the LLM calls each tool."""

from __future__ import annotations

import codecs
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from .externalize import (
    extract_externalized_ref,
    extract_externalized_refs,
    find_externalized_payload_for_message,
    get_large_output_storage_dir,
    load_externalized_payload,
)
from .diagnostics import (
    _has_lifecycle_fragmentation,
    _state_db_path_for_engine,
    doctor_guidance_for_checks,
)
from .dag import build_nodes_fts_spec
from .db_bootstrap import check_external_content_fts_integrity, inspect_lcm_schema_health
from .extraction import sanitize_pre_compaction_content
from .ingest_protection import (
    externalized_payload_stats,
    extract_ingest_externalized_refs,
    restore_ingest_payload_placeholders,
    scan_externalized_payload_integrity,
    scan_sqlite_payload_risks,
    sensitive_pattern_status,
)
from .model_routing import apply_lcm_model_route
from .presets import preset_status_payload
from .search_query import AGE_DECAY_RATE, normalize_search_sort
from .session_patterns import build_session_match_keys, compile_session_pattern
from .store import build_message_fts_spec

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
    if node is None or node.session_id != engine.current_session_id:
        return None
    return node


def _get_externalized_payload(
    engine: "LCMEngine",
    ref: str,
    *,
    allowed_session_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    payload = load_externalized_payload(ref, config=engine._config, hermes_home=engine._hermes_home)
    if payload is None:
        return None
    payload_session_id = payload.get("session_id") or ""
    allowed = allowed_session_ids or {engine.current_session_id}
    if payload_session_id and payload_session_id not in allowed:
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


def _parse_optional_float(value: Any, name: str) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    try:
        return float(value), None
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must be a number"


def _parse_optional_timestamp(value: Any, name: str) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, f"{name} must be a Unix timestamp or timezone-aware ISO 8601 string"
    if isinstance(value, (int, float)):
        try:
            return float(value), None
        except (TypeError, ValueError, OverflowError):
            return None, f"{name} must be a Unix timestamp or timezone-aware ISO 8601 string"
    text = str(value).strip()
    if not text:
        return None, f"{name} must not be empty"
    try:
        return float(text), None
    except (TypeError, ValueError, OverflowError):
        pass
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        return None, f"{name} must be a Unix timestamp or timezone-aware ISO 8601 string"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None, f"{name} ISO timestamp must include a timezone offset or Z"
    return parsed.timestamp(), None


def _parse_grep_role(value: Any) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    role = str(value or "").strip()
    valid_roles = {"system", "user", "assistant", "tool", "unknown"}
    if role not in valid_roles:
        return None, "role must be one of: system, user, assistant, tool, unknown"
    return role, None


def _parse_strict_int(value: Any, name: str) -> tuple[int | None, str | None]:
    try:
        if isinstance(value, bool):
            raise ValueError
        return int(value), None
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must be an integer"


_LCM_GREP_VALID_SCOPES = frozenset({"current", "all", "session"})
_LCM_GREP_HARD_LIMIT_CAP = 200
_LCM_LOAD_SESSION_DEFAULT_LIMIT = 100
_LCM_LOAD_SESSION_HARD_LIMIT_CAP = 200
_LCM_LOAD_SESSION_DEFAULT_MAX_CONTENT_CHARS = 4000
_LCM_LOAD_SESSION_HARD_MAX_CONTENT_CHARS = 20_000
_LCM_INSPECT_DEFAULT_LIMIT = 20
_LCM_INSPECT_HARD_LIMIT_CAP = 200
_LCM_INSPECT_REF_SCAN_MESSAGE_LIMIT = 10_000
_LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES = 16_384


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


def _query_terms_for_match_window(query: str | None) -> list[str]:
    if not query:
        return []
    terms: list[str] = []
    normalized_query = " ".join(re.findall(r"\w+", query))
    if normalized_query:
        terms.append(normalized_query)

    def add_term(term: str) -> None:
        term = term.strip()
        if not term:
            return
        terms.append(term)
        parts = [part for part in re.split(r"[^\w]+", term) if part]
        if len(parts) > 1:
            terms.append(" ".join(parts))
        terms.extend(part for part in parts if len(part) >= 2)

    for quoted in re.findall(r'"([^"]+)"', query):
        add_term(quoted)
    for token in re.findall(r"[\w][\w:-]*\*?", query):
        token = token.rstrip("*").strip()
        if not token or token.upper() in {"AND", "OR", "NOT", "NEAR"}:
            continue
        if ":" in token:
            token = token.rsplit(":", 1)[-1]
        if len(token) >= 2:
            add_term(token)
    seen: set[str] = set()
    unique: list[str] = []
    for term in sorted(terms, key=len, reverse=True):
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(term)
    return unique


def _content_offset_for_query_match(content: str, query: str | None) -> int:
    folded = content.casefold()
    for term in _query_terms_for_match_window(query):
        index = folded.find(term.casefold())
        if index >= 0:
            return index
    return 0


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


def _restore_ingest_placeholder_for_lookup(
    content: str,
    ref: str | None,
    payload: dict[str, Any] | None,
    *,
    config,
    hermes_home: str,
    session_id: str,
) -> str | None:
    if not content or not ref or not payload or payload.get("kind") != "ingest_payload":
        return None
    restored = restore_ingest_payload_placeholders(
        content,
        config=config,
        hermes_home=hermes_home,
        session_id=session_id,
    )
    return restored if restored != content else None


def _is_compact_externalized_marker(content: str, ref: str | None) -> bool:
    if not ref or not content:
        return False
    if len(content) > 512:
        return False
    return (
        content.startswith("[Externalized tool output:")
        or content.startswith("[GC'd externalized tool output:")
        or content.startswith("[Externalized payload:")
        or content.startswith("[GC'd externalized payload:")
        or "[Externalized LCM ingest payload:" in content
    )


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
        if not stored:
            next_source_offset = source_index + 1
            next_content_offset = 0
            has_more = next_source_offset < total_sources
            continue
        transcript_content = stored.get("content", "")
        content = transcript_content
        content_source = "message"
        externalized = None
        ref_payload = None
        ingest_refs = extract_ingest_externalized_refs(transcript_content)
        ref = ingest_refs[0] if ingest_refs else extract_externalized_ref(transcript_content)
        if ref:
            ref_payload = _get_externalized_payload(
                engine,
                ref,
                allowed_session_ids={engine.current_session_id, stored.get("session_id", "")},
            )
            if ref_payload is not None and ref_payload.get("kind") != "ingest_payload":
                externalized = ref_payload
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
            "session_id": stored.get("session_id", ""),
            "source": stored.get("source") or "",
            "from_current_session": stored.get("session_id", "") == engine.current_session_id,
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
                restored_ingest_content = _restore_ingest_placeholder_for_lookup(
                    transcript_content,
                    ref,
                    ref_payload,
                    config=engine._config,
                    hermes_home=engine._hermes_home,
                    session_id=stored.get("session_id", ""),
                )
                if restored_ingest_content is not None:
                    lookup_candidates.insert(0, restored_ingest_content)
                    sanitized_restored = sanitize_pre_compaction_content(restored_ingest_content)
                    if sanitized_restored != restored_ingest_content:
                        lookup_candidates.insert(0, sanitized_restored)
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
        if child is None or child.session_id != engine.current_session_id:
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


def _bounded_source_path_payload(source_path: list[dict[str, int]]) -> dict[str, Any]:
    path_tail = source_path[-8:]
    payload: dict[str, Any] = {
        "source_path": path_tail,
        "source_path_depth": len(source_path),
    }
    if len(path_tail) < len(source_path):
        payload["source_path_truncated"] = True
    return payload


def _collect_descendant_evidence_blocks(
    engine: "LCMEngine",
    node,
    max_tokens: int,
    *,
    hydrate_externalized_content: bool = False,
    visited_node_ids: set[int] | None = None,
    source_path: list[dict[str, int]] | None = None,
    remaining_node_visits: list[int] | None = None,
) -> list[dict[str, Any]]:
    if max_tokens <= 0 or node.source_type != "nodes":
        return []
    if visited_node_ids is None:
        visited_node_ids = set()
    if source_path is None:
        source_path = []
    if remaining_node_visits is None:
        # Budget and cycle detection are the primary limits. Keep a high,
        # budget-scaled guard so corrupt zero-token DAGs cannot make expansion
        # walk an unbounded number of nodes while normal deep summaries still
        # reach their leaf evidence.
        remaining_node_visits = [max(64, int(max_tokens) * 4)]
    if remaining_node_visits[0] <= 0:
        return []

    blocks: list[dict[str, Any]] = []
    budget_used = 0
    root_node_id = int(node.node_id)
    stack: list[tuple[Any, list[dict[str, int]], set[int], int]] = [
        (node, source_path, {*visited_node_ids, root_node_id}, 0)
    ]

    while stack and budget_used < max_tokens and remaining_node_visits[0] > 0:
        current, current_path, current_visited, source_index = stack.pop()
        if source_index >= len(current.source_ids):
            continue

        stack.append((current, current_path, current_visited, source_index + 1))
        child_id = current.source_ids[source_index]
        child = engine._dag.get_node(child_id)
        if child is None or child.session_id != engine.current_session_id:
            continue
        child_node_id = int(child.node_id)
        if child_node_id in current_visited:
            continue

        remaining_node_visits[0] -= 1
        child_path = [*current_path, {"node_id": int(current.node_id), "source_index": source_index}]
        remaining_tokens = max(0, max_tokens - budget_used)
        if child.source_type == "messages":
            messages, pagination = _expand_message_sources(
                engine,
                child,
                max_tokens=remaining_tokens,
                hydrate_externalized_content=hydrate_externalized_content,
            )
            if messages or pagination.get("has_more"):
                block = {
                    "type": "child_messages",
                    "parent_node_id": current.node_id,
                    "node_id": child.node_id,
                    "depth": child.depth,
                    "source_index": source_index,
                    **_bounded_source_path_payload(child_path),
                    "messages": messages,
                    "pagination": pagination,
                }
                blocks.append(block)
                budget_used += _context_content_token_count([block])
            continue

        if child.source_type == "nodes":
            children, pagination = _expand_child_nodes(engine, child, max_tokens=remaining_tokens)
            if children or pagination.get("has_more"):
                block = {
                    "type": "descendant_child_nodes",
                    "parent_node_id": current.node_id,
                    "node_id": child.node_id,
                    "depth": child.depth,
                    "source_index": source_index,
                    **_bounded_source_path_payload(child_path),
                    "children": children,
                    "pagination": pagination,
                }
                blocks.append(block)
                budget_used += _context_content_token_count([block])
            if budget_used < max_tokens and remaining_node_visits[0] > 0:
                stack.append((child, child_path, {*current_visited, child_node_id}, 0))
    return blocks


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
        used_tokens = _context_content_token_count(blocks)
        descendant_tokens = max(0, max_tokens - used_tokens)
        if descendant_tokens > 0:
            blocks.extend(
                _collect_descendant_evidence_blocks(
                    engine,
                    node,
                    max_tokens=descendant_tokens,
                    hydrate_externalized_content=hydrate_externalized_content,
                )
            )

    return blocks


def _collect_raw_match_context_block(
    engine: "LCMEngine",
    rows: list[dict[str, Any]],
    max_tokens: int,
    *,
    query: str | None = None,
    exclude_store_ids: set[int] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    from .tokens import count_tokens

    exclude_store_ids = exclude_store_ids or set()
    messages: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    budget_used = 0
    has_more = False
    next_store_id: int | None = None
    for row in rows:
        store_id = row.get("store_id")
        if store_id in exclude_store_ids:
            continue
        remaining_tokens = max(0, max_tokens - budget_used)
        if remaining_tokens <= 0:
            has_more = True
            next_store_id = store_id if isinstance(store_id, int) else None
            break
        content = str(row.get("content") or "")
        match_offset = _content_offset_for_query_match(content, query)
        content_slice = _slice_content_for_response(content, remaining_tokens, content_offset=match_offset)
        content = content_slice["content"]
        item = {
            "store_id": store_id,
            "session_id": row.get("session_id") or "",
            "source": row.get("source") or "",
            "role": row.get("role"),
            "timestamp": row.get("timestamp", 0),
            **content_slice,
            "content_source": "raw_search_hit",
            "search_rank": row.get("search_rank"),
        }
        if row.get("tool_call_id"):
            item["tool_call_id"] = row.get("tool_call_id")
        if match_offset:
            item["match_window_offset"] = match_offset
        if row.get("tool_calls"):
            item["tool_calls_omitted"] = True
        if row.get("tool_name"):
            item["tool_name"] = row.get("tool_name")
        messages.append(item)
        matches.append(
            {
                "store_id": store_id,
                "role": row.get("role"),
                "snippet": row.get("snippet") or content[:300],
                "search_rank": row.get("search_rank"),
            }
        )
        budget_used += count_tokens(content)
        if content_slice["has_more"]:
            has_more = True
            break

    if not messages and not has_more:
        return None, matches
    block = {
        "type": "raw_messages",
        "messages": messages,
        "pagination": {
            "has_more": has_more,
            "returned_sources": len(messages),
            "total_sources": len(rows),
            "next_store_id": next_store_id,
        },
    }
    return block, matches


def _collect_store_ids_from_context_blocks(blocks: list[dict[str, Any]]) -> set[int]:
    store_ids: set[int] = set()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for message in block.get("messages", []) or []:
            store_id = message.get("store_id")
            if isinstance(store_id, int):
                store_ids.add(store_id)
    return store_ids


def _context_content_token_count(blocks: list[dict[str, Any]]) -> int:
    from .tokens import count_tokens

    total = 0
    for block in blocks:
        if block.get("type") == "summary":
            total += count_tokens(str(block.get("summary") or ""))
        if "source_path" in block:
            total += count_tokens(
                json.dumps(
                    {
                        "source_path": block.get("source_path") or [],
                        "source_path_depth": block.get("source_path_depth"),
                        "source_path_truncated": block.get("source_path_truncated", False),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        if block.get("type") in {"messages", "child_messages", "raw_messages"}:
            for message in block.get("messages", []):
                total += count_tokens(str(message.get("content") or ""))
                total += count_tokens(str(message.get("transcript_content") or ""))
        elif block.get("type") in {"child_nodes", "descendant_child_nodes"}:
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


def _parse_load_session_roles(value: Any) -> tuple[list[str], str | None]:
    if value is None:
        return [], None
    if not isinstance(value, list):
        return [], "roles must be an array of strings"
    roles: list[str] = []
    seen: set[str] = set()
    for item in value:
        role = str(item or "").strip()
        if not role:
            return [], "roles must contain only non-empty strings"
        if role not in seen:
            roles.append(role)
            seen.add(role)
    return roles, None


def _slice_loaded_content(content: Any, max_content_chars: int) -> dict[str, Any]:
    text = content or ""
    sliced = text[:max_content_chars]
    has_more = len(sliced) < len(text)
    return {
        "content": sliced,
        "content_chars": len(text),
        "content_returned_chars": len(sliced),
        "content_truncated": has_more,
        "next_content_offset": len(sliced) if has_more else 0,
    }


def _serialize_loaded_message(engine: "LCMEngine", row: dict[str, Any], max_content_chars: int) -> dict[str, Any]:
    stored_session_id = row.get("session_id", "")
    content_slice = _slice_loaded_content(row.get("content", "") or "", max_content_chars)
    item: dict[str, Any] = {
        "store_id": row.get("store_id"),
        "session_id": stored_session_id,
        "source": row.get("source") or "",
        "role": row.get("role"),
        "timestamp": row.get("timestamp", 0),
        "content": content_slice["content"],
        "content_chars": content_slice["content_chars"],
        "content_returned_chars": content_slice["content_returned_chars"],
        "content_truncated": content_slice["content_truncated"],
        "next_content_offset": content_slice["next_content_offset"],
        "from_current_session": bool(engine.current_session_id) and stored_session_id == engine.current_session_id,
    }
    if row.get("tool_call_id"):
        item["tool_call_id"] = row.get("tool_call_id")
    if row.get("tool_calls"):
        item["tool_calls"] = row.get("tool_calls")
    if row.get("tool_name"):
        item["tool_name"] = row.get("tool_name")
    return item


def lcm_load_session(args: Dict[str, Any], **kwargs) -> str:
    """Load an ordered, bounded raw-message page for one explicit session_id."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    session_id = str(args.get("session_id") or "").strip()
    if not session_id:
        return json.dumps({"error": "session_id is required"})

    raw_limit_arg = args.get("limit", _LCM_LOAD_SESSION_DEFAULT_LIMIT)
    parsed_limit, limit_error = _parse_strict_int(raw_limit_arg, "limit")
    if limit_error:
        return json.dumps({"error": limit_error})
    if parsed_limit is None or parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_LOAD_SESSION_HARD_LIMIT_CAP)

    raw_max_content_chars = args.get("max_content_chars", _LCM_LOAD_SESSION_DEFAULT_MAX_CONTENT_CHARS)
    max_content_chars, max_content_error = _parse_strict_int(raw_max_content_chars, "max_content_chars")
    if max_content_error:
        return json.dumps({"error": max_content_error})
    if max_content_chars is None or max_content_chars <= 0:
        return json.dumps({"error": "max_content_chars must be a positive integer"})
    requested_max_content_chars = max_content_chars
    max_content_chars = min(max_content_chars, _LCM_LOAD_SESSION_HARD_MAX_CONTENT_CHARS)

    after_store_id, cursor_error = _parse_strict_int(args.get("after_store_id", 0), "after_store_id")
    if cursor_error:
        return json.dumps({"error": cursor_error})
    if after_store_id is None or after_store_id < 0:
        return json.dumps({"error": "after_store_id must be a non-negative integer"})

    roles, roles_error = _parse_load_session_roles(args.get("roles"))
    if roles_error:
        return json.dumps({"error": roles_error})

    time_from, time_from_error = _parse_optional_float(args.get("time_from"), "time_from")
    if time_from_error:
        return json.dumps({"error": time_from_error})
    time_to, time_to_error = _parse_optional_float(args.get("time_to"), "time_to")
    if time_to_error:
        return json.dumps({"error": time_to_error})
    if time_from is not None and time_to is not None and time_to < time_from:
        return json.dumps({"error": "time_to must be greater than or equal to time_from"})

    total_messages = engine._store.count_session_load_messages(
        session_id,
        roles=roles or None,
        time_from=time_from,
        time_to=time_to,
    )
    rows = engine._store.load_session_page(
        session_id,
        after_store_id=after_store_id,
        limit=limit + 1,
        roles=roles or None,
        time_from=time_from,
        time_to=time_to,
    )
    page_rows = rows[:limit]
    has_more = len(rows) > limit
    next_cursor = page_rows[-1]["store_id"] if has_more and page_rows else None

    response: dict[str, Any] = {
        "session_id": session_id,
        "limit": limit,
        "max_content_chars": max_content_chars,
        "after_store_id": after_store_id,
        "total_messages": total_messages,
        "returned_messages": len(page_rows),
        "messages": [_serialize_loaded_message(engine, row, max_content_chars) for row in page_rows],
        "next_cursor": next_cursor,
        "has_more": has_more,
    }
    if roles:
        response["roles"] = roles
    if time_from is not None:
        response["time_from"] = time_from
    if time_to is not None:
        response["time_to"] = time_to
    if requested_limit > _LCM_LOAD_SESSION_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    if requested_max_content_chars > _LCM_LOAD_SESSION_HARD_MAX_CONTENT_CHARS:
        response["max_content_chars_clamped_from"] = requested_max_content_chars
    return json.dumps(response)


def lcm_grep(args: Dict[str, Any], **kwargs) -> str:
    """Search raw messages + summaries with optional cross-session scoping.

    Default scope is the current session, preserving historical behavior and returning
    both raw-message and summary-node hits. Callers may explicitly request
    ``session_scope='all'`` (every session in the local LCM database) or
    ``session_scope='session'`` (a single ``session_id``); broader scopes return
    raw-message hits only and exist for bounded archive recovery over rows already
    present in ``lcm.db``. ``limit`` is clamped to ``_LCM_GREP_HARD_LIMIT_CAP``
    regardless of input.
    """
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    query = args.get("query", "").strip()
    if not query:
        return json.dumps({"error": "No query provided"})

    raw_limit_arg = args.get("limit", 10)
    parsed_limit = _parse_int_value(raw_limit_arg, 10)
    if parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_GREP_HARD_LIMIT_CAP)
    sort = normalize_search_sort(args.get("sort"))
    source_limit = max(limit * 4, limit, 20)

    requested_session_scope = str(args.get("session_scope", "current")).lower()
    raw_session_id_arg = args.get("session_id")
    explicit_session_id = (
        str(raw_session_id_arg).strip() if raw_session_id_arg is not None else ""
    )
    source = str(args.get("source") or "").strip() or None
    conversation_id = str(args.get("conversation_id") or "").strip() or None
    role, role_error = _parse_grep_role(args.get("role"))
    if role_error:
        return json.dumps({"error": role_error})
    time_from, time_from_error = _parse_optional_timestamp(args.get("time_from"), "time_from")
    if time_from_error:
        return json.dumps({"error": time_from_error})
    time_to, time_to_error = _parse_optional_timestamp(args.get("time_to"), "time_to")
    if time_to_error:
        return json.dumps({"error": time_to_error})
    if time_from is not None and time_to is not None and time_to < time_from:
        return json.dumps({"error": "time_to must be greater than or equal to time_from"})
    raw_message_filter_active = (
        role is not None
        or time_from is not None
        or time_to is not None
        or conversation_id is not None
    )

    if requested_session_scope == "current":
        if explicit_session_id:
            return json.dumps({
                "error": "session_id is only valid with session_scope=session",
            })
        # MessageStore.search and SummaryDAG.search treat session_id="" as a
        # literal scoped filter, so an unbound engine searching scope=current
        # returns zero results rather than leaking cross-session matches.
        # Read current_session_id (the foreground view) so a cron-style side
        # channel that briefly owns engine._session_id does not redirect the
        # default search scope away from the operator's real conversation.
        search_session_id: str | None = engine.current_session_id
        session_scope = "current"
    elif requested_session_scope == "all":
        if explicit_session_id:
            return json.dumps({
                "error": "session_id is not used with session_scope=all",
            })
        search_session_id = None
        session_scope = "all"
    elif requested_session_scope == "session":
        if not explicit_session_id:
            return json.dumps({
                "error": "session_scope=session requires session_id",
            })
        search_session_id = explicit_session_id
        session_scope = "session"
    else:
        # Preserve historical behavior for unknown scopes: route through the
        # current-session path and report. The data-layer empty-string scoping
        # contract keeps an unbound engine from leaking cross-session matches
        # here too.
        search_session_id = engine.current_session_id
        session_scope = "current"
        logger.warning(
            "Ignoring unsupported session_scope=%s for lcm_grep",
            requested_session_scope,
        )

    current_session_id = engine.current_session_id
    has_current_session = bool(current_session_id)
    results: list[Dict[str, Any]] = []

    try:
        msg_hits = engine._store.search(
            query,
            session_id=search_session_id,
            limit=source_limit,
            sort=sort,
            source=source,
            conversation_id=conversation_id,
            role=role,
            time_from=time_from,
            time_to=time_to,
        )
        for hit in msg_hits:
            timestamp_value = hit.get("timestamp", 0) or 0
            results.append(
                {
                    "type": "message",
                    "depth": "raw",
                    "store_id": hit["store_id"],
                    "session_id": hit["session_id"],
                    "source": hit.get("source") or "",
                    "conversation_id": hit.get("conversation_id") or "",
                    "role": hit["role"],
                    "timestamp": timestamp_value,
                    "snippet": hit.get("snippet", hit.get("content", "")[:200]),
                    "from_current_session": has_current_session and hit["session_id"] == current_session_id,
                    "_sort_ts": timestamp_value,
                    "_sort_rank": hit.get("search_rank"),
                    "_sort_directness": hit.get("_directness_score") or 0.0,
                }
            )
    except Exception as exc:
        logger.warning("Message search failed: %s", exc)

    # Summary-node search is intentionally current-session only. Cross-session
    # DAG expansion is deferred; returning summary hits without an expansion
    # contract would push this tool toward a memory-system shape rather than
    # a plugin-local archive search. Raw-message hits remain expandable across
    # sessions via lcm_expand(store_id=...).
    if session_scope == "current" and not raw_message_filter_active:
        try:
            node_hits = engine._dag.search(
                query,
                session_id=search_session_id,
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
                        "from_current_session": True,
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

    response: Dict[str, Any] = {
        "query": query,
        "sort": sort,
        "session_scope": session_scope,
        "source": source,
        "conversation_id": conversation_id,
        "limit": limit,
        "total_results": len(results),
        "results": results[:limit],
    }
    if role is not None:
        response["role"] = role
    if time_from is not None:
        response["time_from"] = time_from
    if time_to is not None:
        response["time_to"] = time_to
    if raw_message_filter_active:
        response["summary_results_omitted"] = True
    if session_scope == "session":
        response["session_id"] = explicit_session_id
    if requested_limit > _LCM_GREP_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    if requested_session_scope not in _LCM_GREP_VALID_SCOPES:
        response["ignored_session_scope"] = requested_session_scope
        response["scope_note"] = (
            "Unsupported session_scope; stayed on current. "
            "Valid values: current, all, session."
        )
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
                "role": payload.get("role", ""),
                "session_id": payload.get("session_id", ""),
                "field_path": payload.get("field_path", ""),
                "content_chars": payload.get("content_chars", 0),
                "content_bytes": payload.get("content_bytes", 0),
                "created_at": payload.get("created_at"),
                "content_preview": (payload.get("content") or "")[:500],
            }
        )

    node_id = args.get("node_id")
    session_id = engine.current_session_id

    if node_id is not None:
        node = _get_session_node(engine, node_id)
        if node is None:
            return json.dumps({"error": f"Node {node_id} not found in current session"})
        info = engine._dag.describe_subtree(node_id)
        return json.dumps(info)

    depth_stats = engine._dag.get_session_depth_stats(session_id)
    depth_samples = engine._dag.get_session_depth_samples(
        session_id,
        per_depth_limit=20,
        depths=list(depth_stats),
    )
    overview = {
        "session_id": session_id,
        "store_message_count": engine._store.get_session_count(session_id),
        "depths": {},
    }

    for depth, stats in sorted(depth_stats.items()):
        nodes = depth_samples.get(depth, [])
        overview["depths"][f"d{depth}"] = {
            "count": stats["count"],
            "total_tokens": stats["tokens"],
            "total_source_tokens": stats["source_tokens"],
            "nodes": [
                {
                    "node_id": node.node_id,
                    "token_count": node.token_count,
                    "expand_hint": node.expand_hint,
                }
                for node in nodes
            ],
        }

    return json.dumps(overview)


def lcm_expand(args: Dict[str, Any], **kwargs) -> str:
    """Expand a summary node, externalized payload, or raw message to its content.

    Mode selection (exactly one is required):
    - ``externalized_ref``: open a stored externalized payload by ref filename (current session only)
    - ``store_id``: fetch a single raw message by store_id; works across sessions
    - ``node_id``: expand a summary node to its source content (current session only)

    Only ``store_id`` mode accepts an arbitrary cross-session target. ``node_id``
    stays current-session scoped, but carried-over current-session nodes may
    reference raw source rows that still belong to the previous session.
    """
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    externalized_ref = str(args.get("externalized_ref") or "").strip()
    raw_store_id_arg = args.get("store_id")
    raw_node_id_arg = args.get("node_id")

    modes_provided: list[str] = []
    if externalized_ref:
        modes_provided.append("externalized_ref")
    if raw_store_id_arg is not None:
        modes_provided.append("store_id")
    if raw_node_id_arg is not None:
        modes_provided.append("node_id")

    if len(modes_provided) > 1:
        return json.dumps({
            "error": (
                "Provide only one of node_id, externalized_ref, store_id "
                f"(got {', '.join(modes_provided)})"
            ),
        })
    if not modes_provided:
        return json.dumps({
            "error": "node_id, externalized_ref, or store_id is required",
        })

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
                "role": payload.get("role", ""),
                "session_id": payload.get("session_id", ""),
                "field_path": payload.get("field_path", ""),
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

    if raw_store_id_arg is not None:
        try:
            store_id = int(raw_store_id_arg)
        except (TypeError, ValueError, OverflowError):
            return json.dumps({"error": "store_id must be an integer"})
        stored = engine._store.get(store_id)
        if stored is None:
            return json.dumps({"error": f"Message store_id {store_id} not found"})
        transcript_content = stored.get("content", "") or ""
        sliced = _slice_content_for_response(transcript_content, max_tokens, content_offset)
        engine_session_id = engine.current_session_id
        stored_session_id = stored.get("session_id", "")
        result: Dict[str, Any] = {
            "store_id": store_id,
            "source_type": "raw_message",
            "session_id": stored_session_id,
            "source": stored.get("source") or "",
            "conversation_id": stored.get("conversation_id") or "",
            "role": stored.get("role"),
            "timestamp": stored.get("timestamp", 0),
            "tool_call_id": stored.get("tool_call_id") or "",
            "from_current_session": bool(engine_session_id) and stored_session_id == engine_session_id,
            "content": sliced["content"],
            "content_chars": sliced["content_chars"],
            "content_offset": sliced["content_offset"],
            "content_returned_chars": sliced["content_returned_chars"],
            "content_truncated": sliced["content_truncated"],
            "next_content_offset": sliced["next_content_offset"],
            "has_more": sliced["has_more"],
        }
        # Surface externalized-payload metadata when the row references one. Content
        # is not hydrated by default, mirroring the existing _expand_message_sources
        # default. Externalized lookup remains session-scoped (per the existing
        # _get_externalized_payload contract); cross-session rows surface only the
        # ref string, with a hint pointing at the same-session expansion path.
        ref_values = [transcript_content]
        if stored.get("tool_calls"):
            try:
                ref_values.append(json.dumps(stored.get("tool_calls"), ensure_ascii=False, sort_keys=True))
            except (TypeError, ValueError):
                ref_values.append(str(stored.get("tool_calls")))
        refs: list[str] = []
        for value in ref_values:
            if not isinstance(value, str):
                continue
            for found_ref in extract_ingest_externalized_refs(value):
                if found_ref not in refs:
                    refs.append(found_ref)
            legacy_ref = extract_externalized_ref(value)
            if legacy_ref and legacy_ref not in refs:
                refs.append(legacy_ref)
        if refs:
            result["externalized_refs"] = refs
            result["externalized_ref"] = refs[0]
            if bool(engine_session_id) and stored_session_id == engine_session_id:
                payload_summaries = []
                for ref in refs:
                    payload = _get_externalized_payload(engine, ref)
                    if payload is None:
                        continue
                    payload_summary = dict(payload)
                    payload_summary.pop("content", None)
                    payload_summaries.append(payload_summary)
                if payload_summaries:
                    result["externalized_payloads"] = payload_summaries
                    result["externalized"] = payload_summaries[0]
            else:
                result["externalized_note"] = (
                    "Externalized payload metadata is session-scoped; "
                    "cross-session ref is surfaced for traceability only and cannot be expanded in this version."
                )
        return json.dumps(result)

    node_id = raw_node_id_arg

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
    max_results = max(1, int(max_results or 5))

    query = str(args.get("query") or "").strip()
    raw_node_ids = args.get("node_ids") or []

    nodes = []
    raw_results: list[dict[str, Any]] = []
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
        nodes = engine._dag.search(query, session_id=engine.current_session_id, limit=max_results)
        raw_results = engine._store.search(query, session_id=engine.current_session_id, limit=max_results)
    else:
        return json.dumps({"error": "Provide either query or node_ids"})

    if not nodes and not raw_results:
        return json.dumps(
            {
                "prompt": prompt,
                "query": query,
                "answer": "No matching summaries or raw messages found in the current session.",
                "node_ids": [],
                "matches": [],
                "raw_matches": [],
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

    raw_matches: list[dict[str, Any]] = []
    if raw_results:
        seen_store_ids = _collect_store_ids_from_context_blocks(context_blocks)
        remaining_context_tokens = max(0, context_max_tokens - context_budget_used)
        raw_block, raw_matches = _collect_raw_match_context_block(
            engine,
            raw_results,
            max_tokens=remaining_context_tokens,
            query=query,
            exclude_store_ids=seen_store_ids,
        )
        if raw_block is not None:
            context_blocks.append(raw_block)
            context_budget_used += _context_content_token_count([raw_block])

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

        if block_type in {"child_nodes", "descendant_child_nodes"}:
            for child in block.get("children", []):
                if child.get("summary_truncated"):
                    child_node_id = child.get("node_id")
                    context_pagination.append(
                        {
                            "node_id": block.get("node_id"),
                            "type": "child_summary" if block_type == "child_nodes" else "descendant_child_summary",
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
        if block_type in {"messages", "child_messages"}:
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
        elif block_type == "raw_messages":
            truncated_message = next(
                (message for message in block.get("messages", []) if message.get("content_truncated")),
                None,
            )
            if truncated_message:
                item["store_id"] = truncated_message.get("store_id")
                item["content_source"] = truncated_message.get("content_source")
                item["expand_args"] = {
                    "store_id": truncated_message.get("store_id"),
                    "content_offset": truncated_message.get("next_content_offset") or 0,
                }
            elif pagination.get("next_store_id"):
                item["store_id"] = pagination.get("next_store_id")
                item["expand_args"] = {"store_id": pagination.get("next_store_id")}
        elif block_type in {"child_nodes", "descendant_child_nodes"}:
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
            "raw_matches": raw_matches,
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
            "raw_matches": raw_matches,
        }
    )


def _summary_quality_stats(engine: "LCMEngine", session_id: str) -> dict[str, Any]:
    """Return read-only summary compression quality diagnostics for one session."""
    conn = engine._dag._conn
    if conn is None:
        raise RuntimeError("LCM DAG connection is not initialized")
    rows = conn.execute(
        """
        SELECT node_id, session_id, depth, token_count, source_token_count
        FROM summary_nodes
        WHERE session_id = ? AND source_token_count > 0
        ORDER BY
            CASE WHEN token_count <= 0 THEN 1 ELSE 0 END DESC,
            CASE WHEN token_count > 0
                 THEN CAST(source_token_count AS REAL) / token_count
                 ELSE source_token_count
            END DESC
        LIMIT 5
        """,
        (session_id,),
    ).fetchall()
    totals = conn.execute(
        """
        SELECT
            COUNT(*),
            COALESCE(SUM(source_token_count), 0),
            COALESCE(SUM(token_count), 0),
            SUM(CASE WHEN source_token_count >= 100000
                      AND token_count < 500 THEN 1 ELSE 0 END),
            SUM(CASE WHEN token_count > 0
                      AND CAST(source_token_count AS REAL) / token_count >= 400
                     THEN 1 ELSE 0 END)
        FROM summary_nodes
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    total_nodes = int(totals[0] or 0)
    total_source_tokens = int(totals[1] or 0)
    total_summary_tokens = int(totals[2] or 0)
    tiny_large_source_nodes = int(totals[3] or 0)
    extreme_ratio_nodes = int(totals[4] or 0)
    overall_ratio = (
        round(total_source_tokens / total_summary_tokens, 1)
        if total_summary_tokens > 0
        else 0.0
    )
    worst_nodes = []
    for node_id, session_id, depth, token_count, source_token_count in rows:
        ratio = (
            round(float(source_token_count) / float(token_count), 1)
            if token_count and token_count > 0
            else None
        )
        worst_nodes.append({
            "node_id": int(node_id),
            "session_id": session_id,
            "depth": int(depth),
            "source_token_count": int(source_token_count or 0),
            "token_count": int(token_count or 0),
            "compression_ratio": ratio,
        })
    return {
        "total_nodes": total_nodes,
        "session_id": session_id,
        "total_source_tokens": total_source_tokens,
        "total_summary_tokens": total_summary_tokens,
        "overall_compression_ratio": overall_ratio,
        "extreme_ratio_threshold": 400,
        "tiny_large_source_threshold": {
            "source_token_count_min": 100000,
            "token_count_max": 500,
        },
        "extreme_ratio_nodes": extreme_ratio_nodes,
        "tiny_large_source_nodes": tiny_large_source_nodes,
        "worst_nodes": worst_nodes,
        "recommendation": (
            "Inspect worst_nodes with lcm_expand; tiny summaries for very large sources often indicate degraded fallback summarization."
            if extreme_ratio_nodes or tiny_large_source_nodes
            else "summary compression ratios are within the diagnostic thresholds"
        ),
    }


def _matched_session_patterns(session_keys: list[str], patterns: list[str]) -> list[str]:
    """Return configured session glob patterns that match the supplied keys."""
    matched: list[str] = []
    for pattern in patterns:
        try:
            compiled = compile_session_pattern(pattern)
        except re.error:
            continue
        if any(compiled.match(key) for key in session_keys if key):
            matched.append(pattern)
    return matched


def _inspect_externalized_refs_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        sources = [value]
    else:
        try:
            sources = [json.dumps(value, ensure_ascii=False)]
        except (TypeError, ValueError):
            sources = [str(value)]

    refs: list[str] = []
    for source in sources:
        for ref in extract_ingest_externalized_refs(source) + extract_externalized_refs(source):
            if ref not in refs:
                refs.append(ref)
    return refs


def _inspect_message_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Return row metadata only; never include raw message content."""
    content = row.get("content") or ""
    item: dict[str, Any] = {
        "store_id": row.get("store_id"),
        "session_id": row.get("session_id") or "",
        "source": row.get("source") or "",
        "conversation_id": row.get("conversation_id") or "",
        "role": row.get("role") or "unknown",
        "timestamp": row.get("timestamp", 0),
        "token_estimate": row.get("token_estimate", 0),
        "content_chars": len(content),
    }
    if row.get("tool_call_id"):
        item["tool_call_id"] = row.get("tool_call_id")
    if row.get("tool_name"):
        item["tool_name"] = row.get("tool_name")

    refs: list[str] = []
    for value in (row.get("content"), row.get("tool_calls")):
        for ref in _inspect_externalized_refs_from_value(value):
            if ref not in refs:
                refs.append(ref)
    if refs:
        item["externalized_refs"] = refs
    return item


def _inspect_lifecycle_state(engine: "LCMEngine", session_id: str, conversation_id: str) -> dict[str, Any] | None:
    state = None
    if conversation_id:
        state = engine._lifecycle.get_by_conversation(conversation_id)
    if state is None and session_id:
        state = engine._lifecycle.get_by_session(session_id)
    if state is None:
        return None
    return {
        "conversation_id": state.conversation_id,
        "current_session_id": state.current_session_id,
        "last_finalized_session_id": state.last_finalized_session_id,
        "current_frontier_store_id": state.current_frontier_store_id,
        "last_finalized_frontier_store_id": state.last_finalized_frontier_store_id,
        "debt_kind": state.debt_kind,
        "debt_size_estimate": state.debt_size_estimate,
        "current_bound_at": state.current_bound_at,
        "last_finalized_at": state.last_finalized_at,
        "debt_updated_at": state.debt_updated_at,
        "last_maintenance_attempt_at": state.last_maintenance_attempt_at,
        "last_rollover_at": state.last_rollover_at,
        "last_reset_at": state.last_reset_at,
        "updated_at": state.updated_at,
    }


def _inspect_highest_compacted_source_store_id(engine: "LCMEngine", session_id: str) -> int:
    highest = 0
    rows = engine._dag._conn.execute(
        """
        SELECT source_ids
        FROM summary_nodes
        WHERE session_id = ? AND source_type = 'messages'
        """,
        (session_id,),
    ).fetchall()
    for (raw_source_ids,) in rows:
        try:
            source_ids = json.loads(raw_source_ids or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for source_id in source_ids:
            try:
                highest = max(highest, int(source_id))
            except (TypeError, ValueError, OverflowError):
                continue
    return highest


def _inspect_top_level_json_string_fields_before_content(text: str) -> tuple[dict[str, str], bool]:
    decoder = json.JSONDecoder()
    fields: dict[str, str] = {}
    length = len(text)
    index = 0

    def skip_json_whitespace(pos: int) -> int:
        while pos < length and text[pos] in " \t\n\r":
            pos += 1
        return pos

    index = skip_json_whitespace(index)
    if index >= length or text[index] != "{":
        return fields, False
    index += 1

    while True:
        index = skip_json_whitespace(index)
        if index >= length or text[index] == "}":
            return fields, False
        if text[index] != '"':
            return fields, False
        try:
            key, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return fields, False
        if not isinstance(key, str):
            return fields, False
        index = skip_json_whitespace(index)
        if index >= length or text[index] != ":":
            return fields, False
        index += 1
        index = skip_json_whitespace(index)
        if key == "content":
            return fields, index < length and text[index] == '"'
        if index >= length:
            return fields, False
        try:
            value, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return fields, False
        if isinstance(value, str):
            fields[key] = value
        elif key == "session_id":
            fields.pop(key, None)
        index = skip_json_whitespace(index)
        if index >= length:
            return fields, False
        if text[index] == ",":
            index += 1
            continue
        if text[index] == "}":
            return fields, False
        return fields, False


def _read_externalized_payload_metadata_prefix(path: Path) -> tuple[str, bool, bool]:
    """Read bounded JSON metadata before the externalized payload body.

    Returns ``(prefix_text, content_string_seen, prefix_truncated)``. The content
    string body is intentionally not consumed; ``lcm_inspect`` reports bounded
    metadata only and leaves full JSON/body validation to explicit expansion.
    """
    prefix = bytearray()
    text_parts: list[str] = []
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    prefix_truncated = False
    with path.open("rb") as handle:
        while len(prefix) < _LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES:
            byte = handle.read(1)
            if not byte:
                break
            prefix.extend(byte)
            try:
                decoded = decoder.decode(byte, final=False)
            except UnicodeDecodeError as exc:
                raise ValueError("invalid_payload") from exc
            if decoded:
                text_parts.append(decoded)
            prefix_text = "".join(text_parts)
            _, content_key_seen = _inspect_top_level_json_string_fields_before_content(prefix_text)
            if content_key_seen:
                return prefix_text, True, False
        prefix_truncated = len(prefix) >= _LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES and bool(handle.read(1))
    if not prefix_truncated:
        try:
            final_text = decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ValueError("invalid_payload") from exc
        if final_text:
            text_parts.append(final_text)
    return "".join(text_parts), False, prefix_truncated


def _inspect_externalized_payload_metadata(engine: "LCMEngine", ref: str, session_id: str) -> dict[str, Any]:
    if not ref or Path(ref).name != ref:
        return {"readable": False, "error": "invalid_ref"}
    try:
        storage_dir = get_large_output_storage_dir(
            engine._config,
            hermes_home=engine._hermes_home,
            create=False,
        )
        path = storage_dir / ref
        if not path.exists():
            return {"readable": False, "error": "missing"}
        if not path.is_file():
            return {"readable": False, "error": "not_a_file"}
        metadata_prefix_text, content_key_seen, prefix_truncated = _read_externalized_payload_metadata_prefix(path)
    except FileNotFoundError:
        return {"readable": False, "error": "missing"}
    except (OSError, ValueError) as exc:
        return {"readable": False, "error": str(exc)}

    metadata_fields, _content_key_seen = _inspect_top_level_json_string_fields_before_content(metadata_prefix_text)
    payload_session_id = metadata_fields.get("session_id", "")
    if payload_session_id and payload_session_id != session_id:
        return {"readable": False, "error": "session_mismatch"}
    if not payload_session_id:
        return {"readable": False, "error": "session_metadata_unavailable"}
    if not content_key_seen:
        error = "metadata_prefix_truncated" if prefix_truncated else "invalid_payload"
        return {"readable": False, "error": error}

    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"readable": False, "error": "missing"}
    except OSError as exc:
        return {"readable": False, "error": str(exc)}

    metadata: dict[str, Any] = {
        "readable": True,
        "file_size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
        "payload_validation": "metadata_prefix",
    }
    if payload_session_id:
        metadata["payload_session_id"] = payload_session_id
    return metadata


def _inspect_externalized_refs(engine: "LCMEngine", session_id: str, limit: int) -> dict[str, Any]:
    message_total = engine._store.get_session_count(session_id)
    rows = engine._store.load_session_page(session_id, limit=_LCM_INSPECT_REF_SCAN_MESSAGE_LIMIT)
    scan_truncated = message_total > len(rows)
    items: list[dict[str, Any]] = []
    total_known = 0
    seen: set[tuple[int, str]] = set()
    for row in rows:
        refs: list[str] = []
        for value in (row.get("content"), row.get("tool_calls")):
            for ref in _inspect_externalized_refs_from_value(value):
                if ref not in refs:
                    refs.append(ref)
        for ref in refs:
            key = (int(row.get("store_id") or 0), ref)
            if key in seen:
                continue
            seen.add(key)
            total_known += 1
            if len(items) >= limit:
                continue
            metadata = _inspect_externalized_payload_metadata(engine, ref, session_id)
            item: dict[str, Any] = {
                "externalized_ref": ref,
                "store_id": row.get("store_id"),
                "session_id": row.get("session_id") or "",
                "source": row.get("source") or "",
                "conversation_id": row.get("conversation_id") or "",
                "role": row.get("role") or "unknown",
                "timestamp": row.get("timestamp", 0),
                "readable": metadata.get("readable") is True,
            }
            if row.get("tool_call_id"):
                item["tool_call_id"] = row.get("tool_call_id")
            item.update(metadata)
            items.append(item)

    return {
        "total_known": total_known,
        "total_known_exact": not scan_truncated,
        "scanned_messages": len(rows),
        "scan_truncated": scan_truncated,
        "returned": len(items),
        "has_more": total_known > len(items) or scan_truncated,
        "items": items,
    }


def lcm_inspect(args: Dict[str, Any], **kwargs) -> str:
    """Return a read-only metadata inventory of the current LCM session."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    raw_limit_arg = args.get("limit", _LCM_INSPECT_DEFAULT_LIMIT)
    parsed_limit, limit_error = _parse_strict_int(raw_limit_arg, "limit")
    if limit_error:
        return json.dumps({"error": limit_error})
    if parsed_limit is None or parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_INSPECT_HARD_LIMIT_CAP)

    session_id = engine.current_session_id
    conversation_id = engine.current_conversation_id
    if not session_id:
        full_status = engine.get_status()
        return json.dumps({
            "error": "No active session",
            "read_only": True,
            "runtime_identity": full_status.get("runtime_identity") or engine.get_runtime_identity(),
            "ingest_protection": full_status.get("ingest_protection"),
        })

    full_status = engine.get_status()
    runtime_identity = full_status.get("runtime_identity") or engine.get_runtime_identity()
    lifecycle = _inspect_lifecycle_state(engine, session_id, conversation_id)

    store_totals_row = engine._store._conn.execute(
        """
        SELECT COUNT(*), MIN(store_id), MAX(store_id), COALESCE(SUM(token_estimate), 0)
        FROM messages
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    message_total = int(store_totals_row[0] or 0) if store_totals_row else 0
    min_store_id = store_totals_row[1] if store_totals_row else None
    max_store_id = store_totals_row[2] if store_totals_row else None
    estimated_tokens = int(store_totals_row[3] or 0) if store_totals_row else 0
    fresh_tail_count = max(0, int(engine._config.fresh_tail_count or 0))
    fresh_tail_limit = min(limit, fresh_tail_count) if fresh_tail_count > 0 else 0
    fresh_tail_items = [
        _inspect_message_metadata(row)
        for row in engine._store.get_session_tail(session_id, fresh_tail_limit)
    ]

    depth_stats = engine._dag.get_session_depth_stats(session_id)
    total_dag_nodes = sum(info["count"] for info in depth_stats.values())
    total_dag_tokens = sum(info["tokens"] for info in depth_stats.values())
    total_dag_source_tokens = sum(info["source_tokens"] for info in depth_stats.values())
    latest_node_rows = engine._dag._conn.execute(
        """
        SELECT node_id, session_id, depth, token_count, source_token_count,
               source_type, created_at, earliest_at, latest_at, expand_hint
        FROM summary_nodes
        WHERE session_id = ?
        ORDER BY created_at DESC, node_id DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    latest_nodes = [
        {
            "node_id": int(row[0]),
            "session_id": row[1],
            "depth": int(row[2]),
            "token_count": int(row[3] or 0),
            "source_token_count": int(row[4] or 0),
            "source_type": row[5],
            "created_at": row[6],
            "earliest_at": row[7],
            "latest_at": row[8],
            "expand_hint_available": bool(row[9]),
            "expand_hint_chars": len(row[9] or ""),
        }
        for row in latest_node_rows
    ]

    highest_compacted_source_store_id = _inspect_highest_compacted_source_store_id(engine, session_id)
    lifecycle_current_frontier = int((lifecycle or {}).get("current_frontier_store_id") or 0)
    lifecycle_finalized_frontier = int((lifecycle or {}).get("last_finalized_frontier_store_id") or 0)
    runtime_last_compacted = int(getattr(engine, "_last_compacted_store_id", 0) or 0)

    platform = engine.current_session_platform
    session_keys = build_session_match_keys(session_id, platform=platform)
    ignore_patterns = list(engine._config.ignore_session_patterns or [])
    stateless_patterns = list(engine._config.stateless_session_patterns or [])

    response: dict[str, Any] = {
        "read_only": True,
        "session_id": session_id,
        "conversation_id": conversation_id,
        "limit": limit,
        "runtime_identity": runtime_identity,
        "lineage": {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "session_platform": platform,
            "side_channel_active": engine.side_channel_active,
            "bound_session_id": getattr(engine, "_session_id", ""),
            "bound_conversation_id": getattr(engine, "_conversation_id", ""),
            "lifecycle": lifecycle,
            "source_lineage": full_status.get("source_lineage"),
        },
        "messages": {
            "total": message_total,
            "estimated_tokens": estimated_tokens,
            "min_store_id": min_store_id,
            "max_store_id": max_store_id,
            "fresh_tail_count": fresh_tail_count,
            "pre_tail_message_count": max(0, message_total - fresh_tail_count),
            "fresh_tail": {
                "returned": len(fresh_tail_items),
                "items": fresh_tail_items,
            },
        },
        "compaction": {
            "last": {
                "status": full_status.get("last_compression_status", "idle"),
                "noop_reason": full_status.get("last_compression_noop_reason", ""),
                "condensation_suppressed_reason": full_status.get("condensation_suppressed_reason", ""),
                "compression_count": engine.compression_count,
                "last_prompt_tokens": engine.last_prompt_tokens,
                "threshold_tokens": engine.threshold_tokens,
            },
            "frontier": {
                "runtime_last_compacted_store_id": runtime_last_compacted,
                "highest_compacted_source_store_id": highest_compacted_source_store_id,
                "lifecycle_current_frontier_store_id": lifecycle_current_frontier,
                "lifecycle_last_finalized_frontier_store_id": lifecycle_finalized_frontier,
            },
        },
        "dag": {
            "total_nodes": total_dag_nodes,
            "total_tokens": total_dag_tokens,
            "total_source_tokens": total_dag_source_tokens,
            "depths": {f"d{depth}": info for depth, info in sorted(depth_stats.items())},
            "latest_nodes": latest_nodes,
        },
        "externalized_refs": _inspect_externalized_refs(engine, session_id, limit),
        "ingest_protection": full_status.get("ingest_protection"),
        "filters": {
            "session_keys": session_keys,
            "ignored": engine.current_session_ignored,
            "stateless": engine.current_session_stateless,
            "ignore_session_patterns": ignore_patterns,
            "stateless_session_patterns": stateless_patterns,
            "matched_ignore_session_patterns": _matched_session_patterns(session_keys, ignore_patterns),
            "matched_stateless_session_patterns": _matched_session_patterns(session_keys, stateless_patterns),
            "ignore_message_patterns": list(engine._config.ignore_message_patterns or []),
            "ignored_message_count": full_status.get("ignored_message_count", 0),
        },
    }
    if requested_limit > _LCM_INSPECT_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    return json.dumps(response)


def lcm_status(args: Dict[str, Any], **kwargs) -> str:
    """Quick health overview of the LCM engine for the current session."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    # Read the foreground view so a side-channel session that briefly owns
    # engine._session_id (cron tick inside the gateway process, debug probe,
    # etc.) does not divert lcm_status away from the operator's real
    # conversation. Falls back to the bound id when no foreground has ever
    # been bound, so cron-only or stateless-only deployments still report
    # something usable.
    session_id = engine.current_session_id
    if not session_id:
        return json.dumps({
            "error": "No active session",
            "runtime_identity": engine.get_runtime_identity(),
        })

    # Store stats
    store_messages = engine._store.get_session_count(session_id)
    store_tokens = engine._store.get_session_token_total(session_id)

    # DAG stats by depth
    depths = engine._dag.get_session_depth_stats(session_id)

    total_dag_tokens = sum(d["tokens"] for d in depths.values())
    total_source_tokens = sum(d["source_tokens"] for d in depths.values())
    total_dag_nodes = sum(d["count"] for d in depths.values())
    compression_ratio = round(total_source_tokens / total_dag_tokens, 1) if total_dag_tokens > 0 else 0
    full_status = engine.get_status()
    lifecycle = full_status.get("lifecycle")
    lifecycle_fragmentation = full_status.get("lifecycle_fragmentation")
    source_lineage = full_status.get("source_lineage")
    runtime_identity = full_status.get("runtime_identity")
    ingest_reconciliation = full_status.get("ingest_reconciliation")
    config_sources = full_status.get("config_sources") or {}
    config_source_warnings = full_status.get("config_source_warnings") or []
    ignored_config_yaml_lcm_keys = full_status.get("ignored_config_yaml_lcm_keys") or []

    # Filter classification for the session lcm_status is reporting on.
    # The engine encapsulates the foreground vs bound divergence; this tool
    # just reads the property contract.
    side_channel_active = engine.side_channel_active

    return json.dumps({
        "session_id": session_id,
        "compression_count": engine.compression_count,
        "last_compression_status": full_status.get("last_compression_status", "idle"),
        "last_compression_noop_reason": full_status.get("last_compression_noop_reason", ""),
        "model": full_status.get("model", ""),
        "provider": full_status.get("provider", ""),
        "raw_context_length": full_status.get("raw_context_length", engine.context_length),
        "context_length": engine.context_length,
        "effective_context_length_cap": full_status.get("effective_context_length_cap"),
        "effective_context_length_reason": full_status.get("effective_context_length_reason", ""),
        "context_length_source": full_status.get("context_length_source", ""),
        "configured_context_threshold": full_status.get("configured_context_threshold", engine._config.context_threshold),
        "context_threshold": full_status.get("context_threshold", engine._config.context_threshold),
        "context_threshold_source": full_status.get("context_threshold_source", ""),
        "context_threshold_autoraised": full_status.get("context_threshold_autoraised"),
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
            "total_nodes": total_dag_nodes,
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
            "critical_budget_pressure_ratio": engine._config.critical_budget_pressure_ratio,
            "context_threshold": engine._config.context_threshold,
            "max_depth": engine._config.incremental_max_depth,
            "condensation_fanin": engine._config.condensation_fanin,
            "summary_model": engine._config.summary_model or "(auxiliary)",
            "summary_timeout_ms": engine._config.summary_timeout_ms,
            "summary_spend_max_calls": engine._config.summary_spend_max_calls,
            "summary_spend_window_seconds": engine._config.summary_spend_window_seconds,
            "summary_spend_backoff_seconds": engine._config.summary_spend_backoff_seconds,
            "expansion_model": engine._config.expansion_model or "(summary model)",
        },
        "config_sources": config_sources,
        "config_source_warnings": config_source_warnings,
        "ignored_config_yaml_lcm_keys": ignored_config_yaml_lcm_keys,
        "session_filters": {
            "ignored": engine.current_session_ignored,
            "stateless": engine.current_session_stateless,
            "ignore_session_patterns": full_status.get("ignore_session_patterns", []),
            "ignore_session_patterns_source": full_status.get("ignore_session_patterns_source", "default"),
            "stateless_session_patterns": full_status.get("stateless_session_patterns", []),
            "stateless_session_patterns_source": full_status.get("stateless_session_patterns_source", "default"),
            "ignore_message_patterns": full_status.get("ignore_message_patterns", []),
            "ignore_message_patterns_source": full_status.get("ignore_message_patterns_source", "default"),
            "ignored_message_count": full_status.get("ignored_message_count", 0),
            "side_channel_active": side_channel_active,
            **(
                {"side_channel_session_id": engine._session_id}
                if side_channel_active
                else {}
            ),
        },
        "source_lineage": source_lineage,
        "ingest_protection": full_status.get("ingest_protection", sensitive_pattern_status(engine._config)),
        "preset_suggestion": preset_status_payload(engine),
        "ingest_reconciliation": ingest_reconciliation,
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
    # Diagnose the foreground session, not whatever side-channel session
    # currently owns engine._session_id. Falls back to the bound id when no
    # foreground has ever been bound.
    session_id = engine.current_session_id

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

    # Ingest health: a swallowed persistence error means turns were not
    # durably stored, silently breaking the lossless guarantee. Surface it.
    ingest_failures = int(getattr(engine, "_ingest_failure_count", 0) or 0)
    consecutive_failures = int(getattr(engine, "_consecutive_ingest_failures", 0) or 0)
    if consecutive_failures > 0:
        ingest_status = "fail"
    elif ingest_failures > 0:
        ingest_status = "warn"
    else:
        ingest_status = "pass"
    checks.append({
        "check": "ingest_health",
        "status": ingest_status,
        "detail": {
            "total_failures": ingest_failures,
            "consecutive_failures": consecutive_failures,
            "last_error": getattr(engine, "_last_ingest_error", "") or "",
            "last_error_time": getattr(engine, "_last_ingest_error_time", 0) or 0,
        } if ingest_failures else "no ingest failures recorded",
    })

    # ignore_message_patterns drops discard raw content that is never persisted.
    # A non-zero count is worth surfacing so an over-broad pattern is noticed.
    dropped = int(getattr(engine, "_ignore_pattern_dropped_count", 0) or 0)
    checks.append({
        "check": "ignore_pattern_drops",
        "status": "warn" if dropped else "pass",
        "detail": (
            f"{dropped} message(s) dropped by ignore_message_patterns and not "
            "persisted; verify the pattern is not matching substantive turns"
            if dropped
            else "no messages dropped by ignore_message_patterns"
        ),
    })

    try:
        conn = engine._store._conn
        if conn is None:
            raise RuntimeError("LCM store connection is not initialized")
        schema_health = inspect_lcm_schema_health(
            conn,
            database_path=str(engine._store.db_path),
        )
        missing_tables = schema_health.get("missing_tables")
        has_missing = isinstance(missing_tables, list) and bool(missing_tables)
        checks.append({
            "check": "schema_core_tables",
            "status": "fail" if has_missing or schema_health.get("error") else "pass",
            "detail": schema_health,
        })
    except Exception as e:
        checks.append({
            "check": "schema_core_tables",
            "status": "fail",
            "detail": str(e),
        })

    # 1b. FTS5 integrity, separated from generic SQLite integrity so malformed
    # inverted indexes point at the exact table and repair path.
    for check_name, conn, spec in (
        ("messages_fts_integrity", engine._store._conn, build_message_fts_spec()),
        ("nodes_fts_integrity", engine._dag._conn, build_nodes_fts_spec()),
    ):
        try:
            fts_integrity = check_external_content_fts_integrity(conn, spec)
            status = fts_integrity["status"]
            checks.append({
                "check": check_name,
                "status": "warn" if status == "unchecked" else status,
                "detail": fts_integrity if status == "unchecked" else fts_integrity["detail"],
            })
        except Exception as e:
            checks.append({
                "check": check_name,
                "status": "fail",
                "detail": str(e),
            })

    # 2. SQLite storage posture and payload diagnostics
    try:
        journal_mode_row = engine._store._conn.execute("PRAGMA journal_mode").fetchone()
        quick_check_row = engine._store._conn.execute("PRAGMA quick_check").fetchone()
        db_path = Path(engine._store.db_path)
        wal_path = Path(str(db_path) + "-wal")
        checks.append({
            "check": "sqlite_storage",
            "status": "pass" if quick_check_row and quick_check_row[0] == "ok" else "fail",
            "detail": {
                "database_path": str(db_path),
                "database_exists": db_path.exists(),
                "journal_mode": journal_mode_row[0] if journal_mode_row else "unknown",
                "quick_check": quick_check_row[0] if quick_check_row else "unknown",
                "database_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
                "wal_size_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
            },
        })
        payload_risks = scan_sqlite_payload_risks(engine._store._conn)
        externalized_stats = externalized_payload_stats(engine._config, hermes_home=engine._hermes_home)
        externalized_integrity = scan_externalized_payload_integrity(
            engine._store._conn,
            engine._config,
            hermes_home=engine._hermes_home,
        )
        suspicious_count = (
            len(payload_risks["suspicious_data_uri_content_rows"])
            + len(payload_risks["suspicious_data_uri_tool_calls_rows"])
            + len(payload_risks["suspicious_base64_like_rows"])
            + len(payload_risks["suspicious_repetitive_assistant_rows"])
            + len(payload_risks["heartbeat_noise_rows"])
        )
        missing_externalized_refs = int(externalized_integrity.get("externalized_payload_refs_missing", 0) or 0)
        checks.append({
            "check": "payload_storage",
            "status": "warn" if suspicious_count or missing_externalized_refs else "pass",
            "detail": {
                **payload_risks,
                **externalized_stats,
                **externalized_integrity,
            },
        })
    except Exception as e:
        checks.append({
            "check": "payload_storage",
            "status": "fail",
            "detail": str(e),
        })

    try:
        protection = sensitive_pattern_status(engine._config)
        protection_status = "pass"
        if protection["enabled"] and not protection["active_patterns"]:
            protection_status = "warn"
        elif protection["unknown_patterns"]:
            protection_status = "warn"
        checks.append({
            "check": "sensitive_pattern_handling",
            "status": protection_status,
            "detail": protection,
        })
    except Exception as e:
        checks.append({
            "check": "sensitive_pattern_handling",
            "status": "fail",
            "detail": str(e),
        })

    # 3. FTS index sync
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

    try:
        summary_quality = _summary_quality_stats(engine, session_id)
        degraded_count = (
            summary_quality.get("extreme_ratio_nodes", 0)
            + summary_quality.get("tiny_large_source_nodes", 0)
        )
        checks.append({
            "check": "summary_quality",
            "status": "warn" if degraded_count else "pass",
            "detail": summary_quality,
        })
    except Exception as e:
        checks.append({
            "check": "summary_quality",
            "status": "fail",
            "detail": str(e),
        })

    # 4. Configuration validation
    config_warnings = []
    c = engine._config
    if c.fresh_tail_count < 2:
        config_warnings.append("fresh_tail_count < 2 may cause aggressive compaction")
    runtime_context_threshold = float(getattr(engine, "context_threshold", c.context_threshold))
    if runtime_context_threshold > 0.95:
        config_warnings.append("runtime context_threshold > 0.95 leaves very little headroom")
    if runtime_context_threshold < 0.3:
        config_warnings.append("runtime context_threshold < 0.3 triggers compaction very early")
    if c.condensation_fanin < 2:
        config_warnings.append("condensation_fanin < 2 creates excessive depth growth")
    if c.incremental_max_depth == 0:
        config_warnings.append("incremental_max_depth=0 disables condensation entirely")
    for warning in getattr(c, "config_source_warnings", []) or []:
        config_warnings.append(warning)
    for key in getattr(c, "ignored_config_yaml_lcm_keys", []) or []:
        config_warnings.append(
            f"config.yaml lcm.{key} is not a supported LCM config.yaml key and was ignored; use the matching LCM_* env var if this setting is intentional"
        )

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
        runtime_threshold = float(getattr(engine, "context_threshold", c.context_threshold))
        threshold_pct = round(runtime_threshold * 100, 1)
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
        "guidance": doctor_guidance_for_checks(checks),
    })
