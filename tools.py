"""Tool handlers for LCM — the code that runs when the LLM calls each tool."""

from __future__ import annotations

import copy
import json
import logging
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from .externalize import (
    _inspect_top_level_json_string_fields_before_content as _externalized_top_level_fields_before_content,
    extract_externalized_ref,
    extract_externalized_refs,
    find_externalized_payload_for_message,
    get_large_output_storage_dir,
    load_externalized_payload,
    read_externalized_payload_metadata_prefix,
    read_externalized_payload_search_prefix,
)
from .embedding_provider import VoyageError, default_chunk_model, resolve_provider
from .diagnostics import (
    _has_lifecycle_fragmentation,
    _state_db_path_for_engine,
    doctor_guidance_for_checks,
)
from .dag import build_nodes_fts_spec
from .db_bootstrap import (
    check_external_content_fts_integrity,
    inspect_lcm_schema_health,
    load_integrity_failed,
)
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
from .rollup_periods import (
    CoverageNode,
    RecentPeriodWindow,
    canonical_frontier,
    load_source_lineage,
    parse_recent_period,
)
from .retrieval_core import (
    _hit_identity,
    _lcm_grep_confidence,
    _lcm_grep_deadline_error,
    _resolve_semantic_conversation_scope,
    _shape_message_hit,
    _shape_summary_hit,
    hydrate_chunk_hits,
    hydrate_semantic_nodes,
    rrf_fuse,
    run_chunk_knn,
    run_knn,
)
from .rollup_store import RollupStore
from .search_query import AGE_DECAY_RATE, normalize_search_sort
from .session_patterns import build_session_match_keys, compile_session_pattern
from .sqlite_util import _sqlite_savepoint
from .store import build_message_fts_spec
from .vector_store import VectorStore

if TYPE_CHECKING:
    from .engine import LCMEngine


logger = logging.getLogger(__name__)


def _combined_result_sort_key(result: dict[str, Any], sort: str) -> tuple:
    sort_timestamp = float(result.get("_sort_ts") or 0.0)
    rank = result.get("_sort_rank")
    rank_value = float(rank) if rank is not None else float("inf")
    directness = float(result.get("_sort_directness") or 0.0)
    type_bias = 0 if result.get("type") == "message" else 1
    # BM25 ranks and payload byte offsets are incomparable. Keep the existing
    # history ordering in tier 0; sidecars follow in tier 1 and use byte offset
    # only as their native in-tier tie-break.
    rank_tier = 1 if result.get("type") == "externalized" else 0
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
        return (rank_tier, rank_value, -effective_directness, role_bias, -sort_timestamp, type_bias)

    if sort == "hybrid":
        age_hours = max(0.0, (time.time() - sort_timestamp) / 3600.0)
        blended = rank_value / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("inf")
        summary_override = int(result.get("_hybrid_summary_override") or 0)
        return (
            -summary_override,
            rank_tier,
            blended,
            -effective_directness,
            role_bias,
            -sort_timestamp,
            type_bias,
        )

    if result.get("type") == "message":
        return (rank_tier, -sort_timestamp, type_bias, role_bias, rank_value, 0.0, float("inf"))
    return (rank_tier, -sort_timestamp, type_bias, 0, rank_value, 0.0, role_bias)

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
_LCM_GREP_VALID_CONTENT_SCOPES = frozenset({"history", "externalized", "both"})
_LCM_GREP_HARD_LIMIT_CAP = 200
_LCM_GREP_EXTERNALIZED_FILE_CAP = 256
_LCM_GREP_EXTERNALIZED_DISCOVERY_CAP = 4096
_LCM_GREP_EXTERNALIZED_METADATA_READ_BYTES = 64 * 1024
_LCM_GREP_EXTERNALIZED_CONTENT_BYTES = 512_000
_LCM_GREP_EXTERNALIZED_DOCUMENT_TAIL_BYTES = 64 * 1024
_LCM_GREP_RESPONSE_CHAR_CAP = 64_000
_LCM_RECENT_DEFAULT_LIMIT = 10
_LCM_RECENT_HARD_LIMIT_CAP = 200
_LCM_RECENT_FRONTIER_WORK_LIMIT = 4096
_LCM_GREP_HYBRID_CANDIDATE_CAP = 500
_LCM_GREP_SEMANTIC_SNIPPET_CHARS = 300
_LCM_GREP_RRF_K = 60
# -- lcm_recall (cross-conversation forever-memory recall) --
_LCM_RECALL_DEFAULT_LIMIT = 8
_LCM_RECALL_LIMIT_CAP = 25
_LCM_RECALL_DEFAULT_SCOPE_BIAS = 0.5
_LCM_RECALL_SNIPPET_CHARS = 300
_LCM_RECALL_RESPONSE_CHAR_CAP = 64_000
_LCM_RECALL_VALID_INCLUDE = frozenset({"all", "summaries", "verbatim"})
# Recency boost half-life (30 days) and its floor: a memory's rank_score is
# multiplied by 2**(-age/half_life), clamped so age never zeroes an otherwise
# strong hit — it only nudges toward newer memories.
_LCM_RECALL_RECENCY_HALF_LIFE_S = 30 * 24 * 3600.0
_LCM_RECALL_RECENCY_FLOOR = 0.5
_LCM_RECALL_RRF_K = 60
_LCM_LOAD_SESSION_DEFAULT_LIMIT = 100
_LCM_LOAD_SESSION_HARD_LIMIT_CAP = 200
_LCM_LOAD_SESSION_DEFAULT_MAX_CONTENT_CHARS = 4000
_LCM_LOAD_SESSION_HARD_MAX_CONTENT_CHARS = 20_000
_LCM_RECENT_MAX_RESPONSE_CHARS = _LCM_LOAD_SESSION_HARD_MAX_CONTENT_CHARS
_LCM_INSPECT_DEFAULT_LIMIT = 20
_LCM_INSPECT_HARD_LIMIT_CAP = 200
_LCM_INSPECT_REF_SCAN_MESSAGE_LIMIT = 10_000
_LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES = 16_384
_LCM_INSPECT_MAX_RESPONSE_CHARS = 20_000
_OPERATOR_TEXT_FIELD_MAX_CHARS = 1_000


def _bounded_operator_field(value: object) -> tuple[str, bool]:
    text = str(value or "")
    if len(text) <= _OPERATOR_TEXT_FIELD_MAX_CHARS:
        return text, False
    suffix = "..."
    return text[: _OPERATOR_TEXT_FIELD_MAX_CHARS - len(suffix)] + suffix, True


def _bound_operator_strings(value: Any) -> tuple[Any, int]:
    """Return a JSON-compatible copy with every free-text field bounded."""
    if isinstance(value, str):
        bounded, truncated = _bounded_operator_field(value)
        return bounded, int(truncated)
    if isinstance(value, list):
        result: list[Any] = []
        truncated_fields = 0
        for item in value:
            bounded, count = _bound_operator_strings(item)
            result.append(bounded)
            truncated_fields += count
        return result, truncated_fields
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        truncated_fields = 0
        for key, item in value.items():
            bounded, count = _bound_operator_strings(item)
            result[key] = bounded
            truncated_fields += count
        return result, truncated_fields
    return value, 0


def _bounded_inspect_json(response: dict[str, Any]) -> str:
    """Serialize ``lcm_inspect`` under one final response-size invariant."""
    payload, truncated_fields = _bound_operator_strings(response)
    rollup_truncated_fields = (
        (payload.get("temporal_rollups") or {}).get("truncated_fields") or []
    )
    total_truncated_fields = truncated_fields + len(rollup_truncated_fields)
    payload["char_limit"] = _LCM_INSPECT_MAX_RESPONSE_CHARS
    payload["truncated"] = bool(total_truncated_fields)
    if total_truncated_fields:
        payload["truncated_field_count"] = total_truncated_fields
    encoded = json.dumps(payload, ensure_ascii=False)
    if len(encoded) <= _LCM_INSPECT_MAX_RESPONSE_CHARS:
        return encoded

    # If cardinality rather than one text field exceeds the cap, keep whole
    # top-level sections in a deterministic priority order.  Never cut encoded
    # JSON mid-token; omitted sections are reported explicitly.
    priority = [
        "read_only",
        "session_id",
        "conversation_id",
        "limit",
        "temporal_rollups",
        "runtime_identity",
        "lineage",
        "messages",
        "compaction",
        "dag",
        "externalized_refs",
        "ingest_protection",
        "filters",
        "limit_clamped_from",
    ]
    compact: dict[str, Any] = {
        "char_limit": _LCM_INSPECT_MAX_RESPONSE_CHARS,
        "truncated": True,
        "truncation": {
            "reason": "response_char_limit",
            "omitted_top_level_sections": [],
        },
    }
    if total_truncated_fields:
        compact["truncated_field_count"] = total_truncated_fields
    retained: list[str] = []
    omitted: list[str] = []
    ordered_keys = priority + [key for key in payload if key not in priority]
    for key in dict.fromkeys(ordered_keys):
        if key in {"char_limit", "truncated", "truncated_field_count"}:
            continue
        if key not in payload:
            continue
        compact[key] = payload[key]
        if len(json.dumps(compact, ensure_ascii=False)) <= _LCM_INSPECT_MAX_RESPONSE_CHARS - 1_000:
            retained.append(key)
        else:
            compact.pop(key)
            omitted.append(key)
    compact["truncation"]["omitted_top_level_sections"] = omitted
    encoded = json.dumps(compact, ensure_ascii=False)
    while len(encoded) > _LCM_INSPECT_MAX_RESPONSE_CHARS and retained:
        key = retained.pop()
        compact.pop(key, None)
        omitted.append(key)
        compact["truncation"]["omitted_top_level_sections"] = omitted
        encoded = json.dumps(compact, ensure_ascii=False)
    return encoded
_TEMPORAL_ROLLUP_PERIOD_KINDS = ("day", "week", "month")
_TEMPORAL_ROLLUP_STATUSES = ("ready", "stale", "building", "failed")


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


def _recent_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _recent_rollup_bounds(window: RecentPeriodWindow) -> tuple[str, str]:
    start = window.start.date().isoformat()
    if window.rollup_kind == "day":
        end = (window.end - timedelta(microseconds=1)).date().isoformat()
        return start, end
    return start, start


def _recent_expected_period_starts(window: RecentPeriodWindow) -> list[str]:
    """Every rollup ``period_start`` the window requires to be served in rollup
    mode. For a day window this is every calendar day in ``[start, end)``; for a
    week/month window it is the single containing aggregate's start.
    """
    if window.rollup_kind == "day":
        first = window.start.date()
        last = (window.end - timedelta(microseconds=1)).date()
        day_count = (last - first).days + 1
        if day_count > _LCM_RECENT_FRONTIER_WORK_LIMIT:
            return []
        days: list[str] = []
        current = first
        while current <= last:
            days.append(current.isoformat())
            current += timedelta(days=1)
        return days
    return [window.start.date().isoformat()]


def _recent_has_unready_rollups(
    store: RollupStore,
    window: RecentPeriodWindow,
    scope: str,
) -> bool:
    """True when the window is not fully covered by ``ready`` rollups.

    Detects MISSING days (no row at all), not only existing non-ready rows: every
    period the window requires must have a ``ready`` rollup, otherwise the whole
    window falls back (maintainer #389 blocker 1).
    """
    connection = store.connection
    if connection is None:
        return True
    if window.rollup_kind == "day":
        first = window.start.date()
        last = (window.end - timedelta(microseconds=1)).date()
        expected_count = (last - first).days + 1
    else:
        expected_count = 1
    if expected_count <= 0 or expected_count > _LCM_RECENT_FRONTIER_WORK_LIMIT:
        return True
    start, end = _recent_rollup_bounds(window)
    ready_row = connection.execute(
        """
        SELECT COUNT(*)
        FROM lcm_rollups
        WHERE period_kind = ?
          AND period_start >= ?
          AND period_start <= ?
          AND scope = ?
          AND status = 'ready'
        """,
        (window.rollup_kind, start, end, scope),
    ).fetchone()
    return int(ready_row[0] or 0) != expected_count


def _session_has_window_content(
    engine: "LCMEngine",
    window: RecentPeriodWindow,
    session_id: str,
) -> bool:
    """True when ``session_id`` has a summary node whose covered span overlaps
    the window (earliest < end AND latest >= start), matching the leaf-fallback's
    overlap semantics rather than newest-timestamp-only.
    """
    connection = engine._dag.connection
    if connection is None or not session_id:
        return False
    try:
        with engine._dag._db_lock:
            row = connection.execute(
                """
                SELECT 1
                FROM summary_nodes
                WHERE session_id = ?
                  AND COALESCE(earliest_at, created_at) < ?
                  AND COALESCE(latest_at, created_at) >= ?
                LIMIT 1
                """,
                (session_id, window.end.timestamp(), window.start.timestamp()),
            ).fetchone()
    except Exception:  # pragma: no cover - defensive read-only degradation
        return False
    return row is not None


def _recent_ready_rollups(
    engine: "LCMEngine",
    window: RecentPeriodWindow,
    scope: str,
) -> tuple[list[dict[str, object]], str | None]:
    if window.subday:
        return [], "subday_window"
    if not engine._config.temporal_rollups_enabled:
        return [], "temporal_rollups_disabled"

    # Rollups are session-scoped, but the leaf fallback spans the whole
    # conversation family (current + last-finalized session). Only serve rollups
    # when the rollup scope solely covers the window: if another session in that
    # span holds overlapping content (post-rotation retained lineage), fall back
    # to leaf sections, which span the same sessions — so rollup mode never drops
    # a finalized session's content (maintainer #389: match the fallback span).
    for other in _recent_conversation_scope_session_ids(engine):
        if other != scope and _session_has_window_content(engine, window, other):
            return [], "rollups_span_multiple_sessions"

    store: RollupStore | None = None
    try:
        store = RollupStore(engine._dag.db_path)
        # A summary mutation and its invalidation event commit atomically.  Do
        # not serve a previously-ready rollup while that durable event is still
        # waiting for bounded maintenance to reconcile the affected periods.
        if store.has_pending_invalidations(scope):
            return [], "rollups_invalidation_pending"
        start, end = _recent_rollup_bounds(window)
        if _recent_has_unready_rollups(store, window, scope):
            return [], "rollups_unavailable"
        rollups = store.ready_rollups_for_window(
            window.rollup_kind,
            start,
            end,
            scope,
        )
        if not rollups:
            return [], "rollups_unavailable"
        return rollups, None
    except Exception:
        logger.debug("LCM recent rollup read failed; using leaf summaries", exc_info=True)
        return [], "rollups_unavailable"
    finally:
        if store is not None:
            store.close()


def _recent_conversation_scope_session_ids(engine: "LCMEngine") -> list[str]:
    """Session ids that make up the current conversation family for fallback.

    Rotation reassigns retained higher-depth/carry-forward summaries into the
    new session, but a just-finalized session may still hold retained lineage, so
    the fallback spans the current session plus the conversation's finalized
    session (maintainer #389 blocker 2). Scope identity is otherwise session-based
    (see the operator guide) because summary nodes carry no conversation key.
    """
    ids: list[str] = []
    current = str(engine.current_session_id or "")
    if current:
        ids.append(current)
    lifecycle = getattr(engine, "_lifecycle", None)
    conversation_id = getattr(engine, "current_conversation_id", None)
    if lifecycle is not None and conversation_id:
        try:
            state = lifecycle.get_by_conversation(conversation_id)
        except Exception:  # pragma: no cover - defensive read-only degradation
            state = None
        if state is not None:
            for sid in (state.current_session_id, state.last_finalized_session_id):
                if sid and str(sid) not in ids:
                    ids.append(str(sid))
    return ids or [current]


def _recent_leaf_sections(
    engine: "LCMEngine",
    window: RecentPeriodWindow,
    requested_scope: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Load fallback nodes without retaining their TEMP-staging snapshot."""
    with engine._dag._db_lock:
        connection = engine._dag.connection
        if connection is None:
            return []
        with _sqlite_savepoint(connection):
            return _recent_leaf_sections_staged(
                engine,
                window,
                requested_scope,
                limit,
            )


def _recent_leaf_sections_staged(
    engine: "LCMEngine",
    window: RecentPeriodWindow,
    requested_scope: str,
    limit: int,
) -> list[dict[str, Any]]:
    connection = engine._dag.connection
    if connection is None:
        return []
    # Include retained higher-depth/carry-forward summaries, not just depth-0
    # current-session leaves, mirroring how lcm_grep/describe select across
    # depths and retained lineage (maintainer #389 blocker 2).
    # Include any summary whose covered span INTERSECTS the window, not only
    # those whose newest timestamp lands inside it: a summary spanning several
    # days (earliest before the window, latest inside, or vice versa) still holds
    # window content and must be returned (maintainer #389: overlap, not
    # latest_at). Overlap = earliest < window.end AND latest >= window.start.
    where = [
        "COALESCE(earliest_at, created_at) < ?",
        "COALESCE(latest_at, created_at) >= ?",
    ]
    params: list[object] = [window.end.timestamp(), window.start.timestamp()]
    if requested_scope == "conversation":
        session_ids = _recent_conversation_scope_session_ids(engine)
        placeholders = ",".join("?" for _ in session_ids)
        where.append(f"session_id IN ({placeholders})")
        params.extend(session_ids)
    # Probe one sentinel row beyond the work cap without an expression ORDER BY.
    # The session index can stop at the sentinel instead of scanning/sorting the
    # full matching corpus; ordering happens only after this set is proven small.
    probe_params = [*params, _LCM_RECENT_FRONTIER_WORK_LIMIT + 1]
    try:
        with engine._dag._db_lock:
            id_rows = connection.execute(
                f"""
                SELECT node_id
                FROM summary_nodes
                WHERE {' AND '.join(where)}
                LIMIT ?
                """,
                probe_params,
            ).fetchall()
            if len(id_rows) > _LCM_RECENT_FRONTIER_WORK_LIMIT:
                logger.warning(
                    "LCM recent fallback exceeded the %d-node canonical frontier "
                    "bound; returning no partial frontier",
                    _LCM_RECENT_FRONTIER_WORK_LIMIT,
                )
                return []
            if not id_rows:
                return []

            connection.execute(
                "CREATE TEMP TABLE IF NOT EXISTS lcm_recent_frontier_ids "
                "(node_id INTEGER PRIMARY KEY) WITHOUT ROWID"
            )
            connection.execute("DELETE FROM temp.lcm_recent_frontier_ids")
            try:
                connection.executemany(
                    "INSERT INTO temp.lcm_recent_frontier_ids(node_id) VALUES(?)",
                    ((int(row[0]),) for row in id_rows),
                )
                rows = connection.execute(
                    """
                    SELECT node.node_id, node.session_id, node.summary,
                           node.token_count, node.depth, node.source_ids,
                           node.source_type,
                           COALESCE(node.earliest_at, node.created_at) AS earliest_at,
                           COALESCE(node.latest_at, node.created_at) AS latest_at
                    FROM temp.lcm_recent_frontier_ids wanted
                    JOIN summary_nodes node ON node.node_id = wanted.node_id
                    ORDER BY COALESCE(node.latest_at, node.created_at) DESC,
                             node.node_id DESC
                    """
                ).fetchall()
            finally:
                connection.execute("DELETE FROM temp.lcm_recent_frontier_ids")

            source_lineage = load_source_lineage(
                connection,
                [int(row[0]) for row in rows],
                limit=_LCM_RECENT_FRONTIER_WORK_LIMIT,
            )
    except Exception:
        logger.debug(
            "LCM recent fallback or transitive lineage read failed closed",
            exc_info=True,
        )
        return []

    candidates: list[CoverageNode] = []
    by_id: dict[int, "object"] = {}
    for row in rows:
        node_id = int(row[0])
        source_type = str(row[6] or "")
        source_node_ids: tuple[int, ...] = ()
        if source_type == "nodes" and row[5]:
            try:
                source_node_ids = tuple(int(value) for value in json.loads(row[5]))
            except (TypeError, ValueError):
                source_node_ids = ()
        candidates.append(
            CoverageNode(
                node_id=node_id,
                depth=int(row[4] or 0),
                source_node_ids=source_node_ids,
                earliest_at=row[7],
                latest_at=row[8],
            )
        )
        by_id[node_id] = row

    # Suppress any child whose coverage is contained by an overlapping selected
    # parent, THEN apply the limit — so duplicated lineage cannot consume the
    # public budget twice (maintainer #389 C1). ``rows`` is already ordered
    # newest-first, and canonical_frontier preserves that order.
    try:
        frontier_rows = [
            by_id[node.node_id]
            for node in canonical_frontier(
                candidates, source_lineage=source_lineage
            )
        ][:limit]
    except Exception:
        logger.debug("LCM recent canonical frontier failed closed", exc_info=True)
        return []
    return [
        {
            "kind": "leaf_summary",
            "node_id": int(row[0]),
            "session_id": str(row[1]),
            "token_count": int(row[3] or 0),
            "earliest_at": row[7],
            "latest_at": row[8],
            "content": str(row[2] or ""),
            "content_truncated": False,
        }
        for row in frontier_rows
    ]


def _recent_rollup_sections(rollups: list[dict[str, object]]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for rollup in sorted(rollups, key=lambda row: str(row["period_start"]), reverse=True):
        token_count = int(rollup.get("token_count") or 0)
        sections.append(
            {
                "kind": "rollup",
                "rollup_id": int(rollup["rollup_id"]),
                "period_kind": str(rollup["period_kind"]),
                "period_start": str(rollup["period_start"]),
                "status": str(rollup["status"]),
                "token_count": token_count,
                "content": f"Tokens: {token_count}\n{rollup.get('summary') or ''}",
                "content_truncated": False,
            }
        )
    return sections


def _bounded_recent_json(response: dict[str, Any], sections: list[dict[str, Any]]) -> str:
    response["sections"] = []
    response["total_sections"] = len(sections)
    response["returned_sections"] = 0
    response["truncated"] = False
    provenance = response.setdefault("provenance", {})

    def encode() -> str:
        # Provenance is bound to the sections actually RETURNED, not to every
        # candidate rollup: a large ready-rollup set must not serialize thousands
        # of provenance rows outside the char budget (maintainer #389 C2). Because
        # this runs inside every fit check, the per-section provenance entry is
        # counted against the cap in lockstep with its section.
        provenance["rollups"] = [
            {"rollup_id": int(section["rollup_id"]), "status": str(section["status"])}
            for section in response["sections"]
            if section.get("kind") == "rollup"
        ]
        return json.dumps(response, ensure_ascii=False)

    for section in sections:
        response["sections"].append(section)
        response["returned_sections"] = len(response["sections"])
        if len(encode()) <= _LCM_RECENT_MAX_RESPONSE_CHARS:
            continue

        response["sections"].pop()
        response["returned_sections"] = len(response["sections"])
        content = str(section.get("content") or "")
        low, high = 0, len(content)
        best: dict[str, Any] | None = None
        while low <= high:
            midpoint = (low + high) // 2
            candidate = dict(section)
            candidate["content"] = content[:midpoint]
            candidate["content_truncated"] = midpoint < len(content)
            response["sections"].append(candidate)
            response["returned_sections"] = len(response["sections"])
            fits = len(encode()) <= _LCM_RECENT_MAX_RESPONSE_CHARS
            response["sections"].pop()
            response["returned_sections"] = len(response["sections"])
            if fits:
                best = candidate
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best is not None:
            response["sections"].append(best)
            response["returned_sections"] = len(response["sections"])
        response["truncated"] = True
        break

    if response["returned_sections"] < response["total_sections"]:
        response["truncated"] = True
    return encode()


def lcm_recent(args: Dict[str, Any], **kwargs) -> str:
    """Serve conversation rollups or fall back; cross-session rollups are future work."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    try:
        window = parse_recent_period(args.get("period"))
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    requested_scope = str(args.get("scope", "conversation")).strip().lower()
    if requested_scope != "conversation":
        return json.dumps({"error": "scope must be one of: conversation"})

    parsed_limit, limit_error = _parse_strict_int(
        args.get("limit", _LCM_RECENT_DEFAULT_LIMIT),
        "limit",
    )
    if limit_error:
        return json.dumps({"error": limit_error})
    if parsed_limit is None or parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_RECENT_HARD_LIMIT_CAP)

    rollup_scope = engine.current_session_id
    rollups, fallback_reason = _recent_ready_rollups(engine, window, rollup_scope)
    fallback = not rollups
    if fallback:
        sections = _recent_leaf_sections(engine, window, requested_scope, limit)
    else:
        sections = _recent_rollup_sections(rollups)[:limit]

    response: dict[str, Any] = {
        "period": window.period,
        "scope": requested_scope,
        "window": {
            "start": _recent_iso(window.start),
            "end": _recent_iso(window.end),
        },
        "limit": limit,
        "char_limit": _LCM_RECENT_MAX_RESPONSE_CHARS,
        "mode": "leaf_summary_fallback" if fallback else "rollup",
        # ``provenance.rollups`` is filled by _bounded_recent_json from the
        # sections actually returned (bounded by limit + char cap);
        # ``rollups_covered`` is the O(1) aggregate count of ready rollups the
        # window matched, so operators still see "N covered, showing M"
        # (maintainer #389 C2).
        "provenance": {"fallback": fallback},
        "rollups_covered": len(rollups),
    }
    if fallback_reason is not None:
        response["fallback_reason"] = fallback_reason
    if requested_limit > _LCM_RECENT_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    return _bounded_recent_json(response, sections)


def _lcm_grep_full_text(args: Dict[str, Any], **kwargs) -> str:
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
    limit_cap = int(kwargs.get("_limit_cap", _LCM_GREP_HARD_LIMIT_CAP))
    limit = min(requested_limit, limit_cap)
    sort = normalize_search_sort(args.get("sort"))
    source_limit = max(limit * 4, limit, 20)

    content_scope = str(args.get("content_scope") or "history").strip().lower()
    if content_scope not in _LCM_GREP_VALID_CONTENT_SCOPES:
        return json.dumps({"error": "content_scope must be one of: history, externalized, both"})
    raw_externalized_refs = args.get("externalized_refs")
    if raw_externalized_refs is not None and content_scope == "history":
        return json.dumps({"error": "externalized_refs requires content_scope=externalized or both"})
    externalized_refs: list[str] | None = None
    if raw_externalized_refs is not None:
        if not isinstance(raw_externalized_refs, list):
            return json.dumps({"error": "externalized_refs must be an array of ref filenames"})
        if len(raw_externalized_refs) > _LCM_GREP_EXTERNALIZED_FILE_CAP:
            return json.dumps({"error": f"externalized_refs is limited to {_LCM_GREP_EXTERNALIZED_FILE_CAP} refs"})
        externalized_refs = []
        for value in raw_externalized_refs:
            ref = str(value or "").strip()
            if (
                not ref
                or Path(ref).name != ref
                or "/" in ref
                or "\\" in ref
                or not ref.endswith(".json")
            ):
                return json.dumps({"error": f"Invalid externalized ref: {ref or '<empty>'}"})
            if ref not in externalized_refs:
                externalized_refs.append(ref)

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
    externalized_filter_active = raw_message_filter_active or source is not None

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

    searches_externalized = content_scope in {"externalized", "both"}
    if searches_externalized and session_scope != "current":
        return json.dumps({
            "error": "Externalized payload search supports session_scope=current only",
        })
    if searches_externalized and not engine.current_session_id:
        return json.dumps({"error": "Externalized payload search requires an active session"})

    current_session_id = engine.current_session_id
    has_current_session = bool(current_session_id)
    results: list[Dict[str, Any]] = []

    if content_scope in {"history", "both"}:
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
                results.append(
                    _shape_message_hit(
                        hit,
                        current_session_id=current_session_id,
                        has_current_session=has_current_session,
                    )
                )
        except Exception as exc:
            logger.warning("Message search failed: %s", exc)

    # Summary-node search is intentionally current-session only. Cross-session
    # DAG expansion is deferred; returning summary hits without an expansion
    # contract would push this tool toward a memory-system shape rather than
    # a plugin-local archive search. Raw-message hits remain expandable across
    # sessions via lcm_expand(store_id=...).
    if content_scope in {"history", "both"} and session_scope == "current" and not raw_message_filter_active:
        try:
            node_hits = engine._dag.search(
                query,
                session_id=search_session_id,
                limit=source_limit,
                sort=sort,
                source=source,
            )
            for node in node_hits:
                results.append(_shape_summary_hit(node))
        except Exception as exc:
            logger.warning("Node search failed: %s", exc)

    externalized_scan: dict[str, Any] | None = None
    externalized_results_omitted = False
    if searches_externalized and externalized_filter_active:
        # Externalized sidecars are tool/ingest payloads, not raw messages, and
        # carry no role/timestamp/source/conversation lane comparable to history
        # rows. Suppress sidecar search whenever one of those filters is active
        # rather than leak unscoped payloads. Source remains valid for summary
        # search above, so it intentionally does not affect summary omission.
        externalized_results_omitted = True
    elif searches_externalized:
        try:
            storage_dir = get_large_output_storage_dir(
                engine._config,
                hermes_home=engine._hermes_home,
                create=False,
            )
        except (OSError, ValueError) as exc:
            return json.dumps({
                "error": f"Externalized payload storage is unavailable: {exc}",
            })
        scan_counts = {
            "candidate_files": 0,
            "discovery_files": 0,
            "discovery_truncated": False,
            "scanned_files": 0,
            "matched_files": 0,
            "rejected_symlink": 0,
            "rejected_invalid_or_unreadable": 0,
            "rejected_session_mismatch": 0,
        }
        if externalized_refs is None:
            discovered_refs = []
            try:
                for entries_seen, path in enumerate(storage_dir.iterdir(), start=1):
                    if entries_seen > _LCM_GREP_EXTERNALIZED_DISCOVERY_CAP:
                        scan_counts["discovery_truncated"] = True
                        break
                    if not path.name.endswith(".json"):
                        continue
                    discovered_refs.append(path.name)
            except OSError:
                discovered_refs = []
            # Bound directory consumption before sorting so one busy payload
            # directory cannot make a search materialize every sidecar name.
            discovered_refs.sort(reverse=True)
            candidate_refs = []
            for ref in discovered_refs:
                scan_counts["discovery_files"] += 1
                path = storage_dir / ref
                if path.is_symlink():
                    scan_counts["rejected_symlink"] += 1
                    continue
                metadata = _inspect_externalized_payload_metadata(
                    engine,
                    ref,
                    engine.current_session_id,
                    max_read_bytes=_LCM_GREP_EXTERNALIZED_METADATA_READ_BYTES,
                )
                if metadata.get("readable"):
                    try:
                        payload = read_externalized_payload_search_prefix(
                            ref,
                            config=engine._config,
                            hermes_home=engine._hermes_home,
                            # Re-open with no-follow/inode checks after strict
                            # metadata validation; the candidate scan reads content.
                            max_content_bytes=1,
                        )
                    except (OSError, ValueError) as exc:
                        return json.dumps({
                            "error": f"Externalized payload storage is unavailable: {exc}",
                        })
                    status = payload.get("status")
                    if status == "symlink":
                        scan_counts["rejected_symlink"] += 1
                    elif status != "ok":
                        scan_counts["rejected_invalid_or_unreadable"] += 1
                    elif payload.get("session_id") != engine.current_session_id:
                        scan_counts["rejected_session_mismatch"] += 1
                    else:
                        candidate_refs.append(ref)
                        if len(candidate_refs) >= _LCM_GREP_EXTERNALIZED_FILE_CAP:
                            break
                elif metadata.get("error") == "session_mismatch":
                    scan_counts["rejected_session_mismatch"] += 1
                else:
                    scan_counts["rejected_invalid_or_unreadable"] += 1
        else:
            candidate_refs = externalized_refs
        scan_counts["candidate_files"] = len(candidate_refs)

        externalized_matches: list[dict[str, Any]] = []
        for ref in candidate_refs:
            if externalized_refs is not None:
                path = storage_dir / ref
                if path.is_symlink():
                    scan_counts["rejected_symlink"] += 1
                    return json.dumps({"error": f"Externalized ref is a symlink: {ref}"})
                metadata = _inspect_externalized_payload_metadata(
                    engine,
                    ref,
                    engine.current_session_id,
                    max_read_bytes=_LCM_GREP_EXTERNALIZED_METADATA_READ_BYTES,
                    require_valid_document_tail=True,
                )
                if not metadata.get("readable"):
                    if metadata.get("error") == "session_mismatch":
                        scan_counts["rejected_session_mismatch"] += 1
                        return json.dumps({"error": f"Externalized ref is not owned by the active session: {ref}"})
                    scan_counts["rejected_invalid_or_unreadable"] += 1
                    return json.dumps({"error": f"Externalized ref is not readable: {ref}"})
            try:
                payload = read_externalized_payload_search_prefix(
                    ref,
                    config=engine._config,
                    hermes_home=engine._hermes_home,
                    max_content_bytes=_LCM_GREP_EXTERNALIZED_CONTENT_BYTES,
                )
            except (OSError, ValueError) as exc:
                return json.dumps({
                    "error": f"Externalized payload storage is unavailable: {exc}",
                })
            status = payload.get("status")
            if status == "symlink":
                scan_counts["rejected_symlink"] += 1
                if externalized_refs is not None:
                    return json.dumps({"error": f"Externalized ref is a symlink: {ref}"})
                continue
            if status != "ok":
                scan_counts["rejected_invalid_or_unreadable"] += 1
                if externalized_refs is not None:
                    return json.dumps({"error": f"Externalized ref is not readable: {ref}"})
                continue
            scan_counts["scanned_files"] += 1
            if payload.get("session_id") != engine.current_session_id:
                scan_counts["rejected_session_mismatch"] += 1
                if externalized_refs is not None:
                    return json.dumps({"error": f"Externalized ref is not owned by the active session: {ref}"})
                continue
            content = str(payload.get("content") or "")
            match = re.search(re.escape(query), content, flags=re.IGNORECASE)
            if match is None:
                continue
            byte_position = len(content[: match.start()].encode("utf-8"))
            line = content.count("\n", 0, match.start()) + 1
            snippet_start = max(0, match.start() - 120)
            snippet_end = min(len(content), match.end() + 180)
            item = {
                "type": "externalized",
                "depth": "payload",
                "ref": ref,
                "tool_call_id": payload.get("tool_call_id") or "",
                "snippet": content[snippet_start:snippet_end],
                "line": line,
                "byte_position": byte_position,
                "original_content_bytes": payload.get("original_content_bytes"),
                "original_content_chars": payload.get("original_content_chars"),
                "scan_truncated": bool(payload.get("scan_truncated")),
                "content_scanned_bytes": payload.get("content_scanned_bytes", 0),
                "from_current_session": True,
                "_sort_ts": payload.get("created_at") or 0,
                "_sort_rank": byte_position,
                "_sort_directness": 10.0,
            }
            externalized_matches.append(item)
            scan_counts["matched_files"] += 1

        # Search every file in the bounded candidate set before truncating.
        # Discovery order is not a ranking signal: relevance and hybrid use the
        # payload's native byte position, while recency uses its timestamp.
        externalized_matches.sort(key=lambda result: _combined_result_sort_key(result, sort))
        response_chars = 0
        for item in externalized_matches:
            item_chars = len(json.dumps(item, ensure_ascii=False))
            if response_chars + item_chars > _LCM_GREP_RESPONSE_CHAR_CAP:
                break
            response_chars += item_chars
            results.append(item)
        externalized_scan = {
            **scan_counts,
            "file_limit": _LCM_GREP_EXTERNALIZED_FILE_CAP,
            "discovery_limit": _LCM_GREP_EXTERNALIZED_DISCOVERY_CAP,
            "content_bytes_per_file": _LCM_GREP_EXTERNALIZED_CONTENT_BYTES,
            "response_char_limit": _LCM_GREP_RESPONSE_CHAR_CAP,
            "active_session_only": True,
        }

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
        "content_scope": content_scope,
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
    if externalized_results_omitted:
        response["externalized_results_omitted"] = True
    if session_scope == "session":
        response["session_id"] = explicit_session_id
    if requested_limit > limit_cap:
        response["limit_clamped_from"] = requested_limit
    if requested_session_scope not in _LCM_GREP_VALID_SCOPES:
        response["ignored_session_scope"] = requested_session_scope
        response["scope_note"] = (
            "Unsupported session_scope; stayed on current. "
            "Valid values: current, all, session."
        )
    if externalized_refs is not None:
        response["externalized_refs"] = externalized_refs
    if externalized_scan is not None:
        response["externalized_scan"] = externalized_scan
    return json.dumps(response)


# A hard-timed semantic operation cannot kill the worker thread it abandons
# (Python has no thread cancellation), so bound how many can be live at once.
# A worker releases its slot when it eventually finishes; once every slot is
# held by a stuck worker, further requests degrade to full-text immediately
# instead of spawning unbounded threads under repeated timeouts.
_LCM_SEMANTIC_MAX_WORKERS = 4
_lcm_semantic_worker_slots = threading.BoundedSemaphore(_LCM_SEMANTIC_MAX_WORKERS)
# FTS fallback has its own bounded lane so abandoned semantic calls cannot
# consume the capacity required to degrade safely.
_LCM_FULL_TEXT_MAX_WORKERS = 4
_lcm_full_text_worker_slots = threading.BoundedSemaphore(_LCM_FULL_TEXT_MAX_WORKERS)


class _WorkerCapacityError(RuntimeError):
    """Raised when no bounded worker slot is available."""


def _run_within_deadline(
    fn,
    *,
    remaining_s: float,
    name: str,
    worker_slots: threading.BoundedSemaphore | None = None,
):
    """Run ``fn`` in a bounded daemon worker, raising if the deadline lapses.

    ``remaining_s`` is the time left in the operation's single absolute
    deadline. On timeout the worker is abandoned but keeps its slot until it
    finishes, so stuck workers cannot accumulate without bound.
    """
    remaining_s = float(remaining_s)
    if remaining_s <= 0:
        raise TimeoutError("semantic latency budget exhausted")
    slots = _lcm_semantic_worker_slots if worker_slots is None else worker_slots
    if not slots.acquire(blocking=False):
        raise _WorkerCapacityError(f"{name} worker capacity is exhausted")
    outcome: list[tuple[bool, Any]] = []

    def invoke() -> None:
        try:
            outcome.append((True, fn()))
        except BaseException as exc:  # noqa: BLE001 - forwarded to caller
            outcome.append((False, exc))
        finally:
            slots.release()

    worker = threading.Thread(target=invoke, name=name, daemon=True)
    try:
        worker.start()
    except BaseException:
        slots.release()
        raise
    worker.join(remaining_s)
    if worker.is_alive():
        raise TimeoutError(f"{name} exceeded the semantic latency budget")
    succeeded, value = outcome[0]
    if not succeeded:
        raise value
    return value


def _lcm_grep_full_text_with_deadline(
    args: Dict[str, Any],
    *,
    engine: "LCMEngine",
    deadline: float,
    limit_cap: int = _LCM_GREP_HARD_LIMIT_CAP,
) -> dict[str, Any]:
    """Run FTS on independent read connections under the request deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return _lcm_grep_deadline_error(
            str(args.get("mode") or "semantic").lower(), "full_text"
        )

    def invoke() -> dict[str, Any]:
        message_conn: sqlite3.Connection | None = None
        dag_conn: sqlite3.Connection | None = None
        expired = [False]

        def require_remaining(stage: str) -> float:
            stage_remaining = deadline - time.monotonic()
            if stage_remaining <= 0:
                raise TimeoutError(f"lcm full-text deadline exhausted before {stage}")
            return stage_remaining

        def interrupt_if_expired() -> int:
            if time.monotonic() >= deadline:
                expired[0] = True
                return 1
            return 0

        try:
            require_remaining("database path resolution")
            db_path = Path(engine._store.db_path).resolve()
            require_remaining("database path resolution")
            uri = f"{db_path.as_uri()}?mode=ro"
            require_remaining("database URI construction")
            message_conn = sqlite3.connect(
                uri,
                uri=True,
                timeout=max(0.001, require_remaining("message connection")),
            )
            require_remaining("message connection")
            require_remaining("DAG connection")
            dag_conn = sqlite3.connect(
                uri,
                uri=True,
                timeout=max(0.001, require_remaining("DAG connection")),
            )
            require_remaining("DAG connection")
            for conn in (message_conn, dag_conn):
                require_remaining("connection setup")
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA query_only=ON")
                require_remaining("connection setup")
                conn.set_progress_handler(interrupt_if_expired, 1000)
                require_remaining("connection setup")
            read_engine = copy.copy(engine)
            read_store = copy.copy(engine._store)
            read_dag = copy.copy(engine._dag)
            read_store._conn = message_conn
            read_store._db_lock = threading.RLock()
            read_dag._conn = dag_conn
            read_dag._db_lock = threading.RLock()
            read_engine._store = read_store
            read_engine._dag = read_dag
            require_remaining("full-text search")
            payload = json.loads(
                _lcm_grep_full_text(
                    args,
                    engine=read_engine,
                    _limit_cap=limit_cap,
                )
            )
            if expired[0] or time.monotonic() >= deadline:
                return _lcm_grep_deadline_error(
                    str(args.get("mode") or "semantic").lower(), "full_text"
                )
            return payload
        finally:
            if message_conn is not None:
                message_conn.close()
            if dag_conn is not None:
                dag_conn.close()

    try:
        return _run_within_deadline(
            invoke,
            remaining_s=remaining,
            name="lcm-full-text",
            worker_slots=_lcm_full_text_worker_slots,
        )
    except (_WorkerCapacityError, TimeoutError):
        return _lcm_grep_deadline_error(
            str(args.get("mode") or "semantic").lower(), "full_text"
        )
    except Exception as exc:
        return {
            "error": f"full-text fallback failed: {exc}",
            "mode": str(args.get("mode") or "semantic").lower(),
        }


def _lcm_grep_embed_query(
    provider: Any, query: str, *, remaining_s: float
) -> list[float]:
    """Embed one query within the operation's remaining absolute budget."""
    def invoke() -> list[float]:
        interactive = getattr(provider, "embed_query_interactive", None)
        if callable(interactive):
            return interactive(query, timeout=max(0.001, remaining_s))
        return provider.embed_query(query)

    vector = _run_within_deadline(
        invoke, remaining_s=remaining_s, name="lcm-query-embed"
    )
    return [float(value) for value in vector]


def _lcm_grep_resolve_provider(
    engine: "LCMEngine", *, deadline: float | None = None
) -> Any:
    config = engine._config
    cache_key = (
        str(getattr(config, "embedding_provider", "") or "").strip().lower(),
        str(getattr(config, "embedding_model", "") or "").strip(),
    )
    cached = getattr(engine, "_lcm_embedding_provider_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("provider resolution deadline exhausted")
    provider = resolve_provider(config)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("provider resolution deadline exhausted")
    engine._lcm_embedding_provider_cache = (cache_key, provider)
    return provider


def _resolve_recall_provider(
    engine: "LCMEngine",
    *,
    deadline: float | None = None,
    provider_override: str | None = None,
) -> Any:
    """Resolve the embedding provider for a recall query.

    ``provider_override`` (SPEC F proactive injection) lets the injection path
    embed its query with a different provider than interactive search — e.g. a
    local fastembed provider for the offline path even when search uses voyage.
    It caches under its own engine slot so it never evicts the main
    interactive-search provider cache (no per-turn thrash between the two).
    Empty override falls straight through to the shared resolver, so normal
    tool calls are byte-identical.
    """
    override = str(provider_override or "").strip()
    if not override:
        return _lcm_grep_resolve_provider(engine, deadline=deadline)
    config = engine._config
    cache_key = (
        override.lower(),
        str(getattr(config, "embedding_model", "") or "").strip(),
    )
    cached = getattr(engine, "_lcm_proactive_provider_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("provider resolution deadline exhausted")
    override_config = copy.copy(config)
    override_config.embedding_provider = override
    provider = resolve_provider(override_config)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("provider resolution deadline exhausted")
    engine._lcm_proactive_provider_cache = (cache_key, provider)
    return provider


def _resolve_recall_chunk_provider(
    engine: "LCMEngine", summary_provider: Any, *, deadline: float | None = None
) -> Any:
    """Resolve the provider/model identity registered for the chunk corpus."""
    chunk_model = default_chunk_model(
        summary_provider.provider_id, summary_provider.model_id
    )
    if chunk_model == summary_provider.model_id:
        return summary_provider
    cache_key = (str(summary_provider.provider_id).lower(), chunk_model)
    cached = getattr(engine, "_lcm_chunk_provider_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("chunk provider resolution deadline exhausted")
    chunk_config = copy.copy(engine._config)
    chunk_config.embedding_provider = summary_provider.provider_id
    chunk_config.embedding_model = chunk_model
    provider = resolve_provider(chunk_config)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("chunk provider resolution deadline exhausted")
    engine._lcm_chunk_provider_cache = (cache_key, provider)
    return provider


def _lcm_grep_semantic(
    args: Dict[str, Any],
    *,
    engine: "LCMEngine",
    deadline: float,
    candidate_limit: int | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    mode = str(args.get("mode") or "semantic").lower()
    if time.monotonic() >= deadline:
        return _lcm_grep_deadline_error(mode, "semantic_entry")
    query = str(args.get("query", "")).strip()
    if not query:
        return {"error": "No query provided"}

    raw_limit_arg = args.get("limit", 10)
    parsed_limit = _parse_int_value(raw_limit_arg, 10)
    if parsed_limit <= 0:
        return {"error": "limit must be a positive integer"}
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_GREP_HARD_LIMIT_CAP)
    knn_limit = candidate_limit if candidate_limit is not None else limit

    requested_session_scope = str(args.get("session_scope", "current")).lower()
    raw_session_id_arg = args.get("session_id")
    explicit_session_id = (
        str(raw_session_id_arg).strip() if raw_session_id_arg is not None else ""
    )
    if requested_session_scope == "current":
        if explicit_session_id:
            return {"error": "session_id is only valid with session_scope=session"}
        session_scope = "current"
        search_session_id: str | None = engine.current_session_id
    elif requested_session_scope == "all":
        if explicit_session_id:
            return {"error": "session_id is not used with session_scope=all"}
        session_scope = "all"
        search_session_id = None
    elif requested_session_scope == "session":
        if not explicit_session_id:
            return {"error": "session_scope=session requires session_id"}
        session_scope = "session"
        search_session_id = explicit_session_id
    else:
        session_scope = "current"
        search_session_id = engine.current_session_id
        logger.warning(
            "Ignoring unsupported session_scope=%s for semantic lcm_grep",
            requested_session_scope,
        )

    source = str(args.get("source") or "").strip() or None
    conversation_id = str(args.get("conversation_id") or "").strip() or None
    role, role_error = _parse_grep_role(args.get("role"))
    if role_error:
        return {"error": role_error}
    time_from, time_from_error = _parse_optional_timestamp(args.get("time_from"), "time_from")
    if time_from_error:
        return {"error": time_from_error}
    time_to, time_to_error = _parse_optional_timestamp(args.get("time_to"), "time_to")
    if time_to_error:
        return {"error": time_to_error}
    if time_from is not None and time_to is not None and time_to < time_from:
        return {"error": "time_to must be greater than or equal to time_from"}

    def degraded(reason: str) -> dict[str, Any]:
        if not allow_fallback:
            return {
                "mode": mode,
                "degraded_to_fts": True,
                "degraded_reason": reason,
                "coverage": "none",
                "results": [],
            }
        fts_args = dict(args)
        fts_args.pop("mode", None)
        payload = _lcm_grep_full_text_with_deadline(
            fts_args,
            engine=engine,
            deadline=deadline,
        )
        if "error" not in payload:
            payload["mode"] = mode
            payload["degraded_to_fts"] = True
            payload["degraded_reason"] = reason
            payload["coverage"] = "none"
        return payload

    if not bool(getattr(engine._config, "embeddings_enabled", False)):
        return degraded("semantic retrieval is disabled")

    # role is a raw-message dimension; a summary vector has no single role, so
    # it cannot be enforced over embedded summaries. Rather than silently
    # ignoring it, degrade to full_text — which does enforce role — so a
    # role=user query never returns assistant/tool summaries.
    if role is not None:
        return degraded("role filtering is only supported by full_text retrieval")

    # The advertised lcm_grep contract (schemas.LCM_GREP) returns RAW message
    # hits only for broader scopes and for time/conversation filters; full_text
    # honors this by omitting summary hits in exactly these cases. Embedded
    # summaries have no single lane and are cross-session/unexpandable, so the
    # semantic arm degrades to the raw full_text path rather than emit summary
    # hits that violate the contract (and, for conversation_id, leak
    # wrong-lane summaries from a session that carries multiple conversations).
    if session_scope != "current":
        return degraded("broader scopes return raw-message hits only")
    if time_from is not None or time_to is not None:
        return degraded("time-scoped queries return raw-message hits only")
    if conversation_id is not None:
        return degraded("conversation-scoped queries return raw-message hits only")

    # content_scope is a payload-search dimension owned by the full-text arm.
    # Externalized payloads are never embedded (embedded_kind='summary'), so a
    # semantic request scoped to them has no vector corpus to search. Degrade
    # to full_text — which implements bounded externalized payload scanning —
    # rather than silently returning history-only semantic hits that ignore
    # the requested scope. In hybrid mode this surfaces as the full-text-arm
    # result (which honors content_scope) plus the explicit degraded marker.
    content_scope = str(args.get("content_scope") or "history").strip().lower()
    if content_scope != "history":
        return degraded(
            "content_scope beyond history is served by full_text retrieval"
        )

    # Scope the (current-session) summaries to the active session id. With the
    # raw-only degradations above, conversation_id is always None here, so this
    # resolves to the current session set.
    knn_conversation_ids = _resolve_semantic_conversation_scope(
        engine, search_session_id=search_session_id, conversation_id=conversation_id
    )
    if time.monotonic() >= deadline:
        return _lcm_grep_deadline_error(mode, "scope_resolution")

    try:
        provider = _run_within_deadline(
            lambda: _lcm_grep_resolve_provider(engine, deadline=deadline),
            remaining_s=deadline - time.monotonic(),
            name="lcm-provider-resolution",
        )
    except _WorkerCapacityError as exc:
        return degraded(f"semantic capacity exhausted: {exc}")
    except TimeoutError:
        return _lcm_grep_deadline_error(mode, "provider_resolution")
    except Exception as exc:
        return degraded(f"embedding provider unavailable: {exc}")
    if provider is None:
        return degraded("embedding provider is not configured")

    try:
        query_vector = _lcm_grep_embed_query(
            provider, query, remaining_s=deadline - time.monotonic()
        )
    except VoyageError as exc:
        if exc.kind == "auth":
            return {
                "error": f"Embedding provider authentication failed; {exc}",
                "mode": mode,
            }
        return degraded(f"query embedding failed: {exc}")
    except _WorkerCapacityError as exc:
        return degraded(f"semantic capacity exhausted: {exc}")
    except Exception as exc:
        return degraded(f"query embedding failed: {exc}")

    def _run_knn() -> Any:
        # VectorStore is resolved through this module's namespace so tests that
        # monkeypatch ``tools.VectorStore`` continue to govern the KNN backend.
        return run_knn(
            engine,
            query_vector=query_vector,
            provider=provider,
            knn_limit=knn_limit,
            deadline=deadline,
            since=time_from,
            until=time_to,
            conversation_ids=knn_conversation_ids,
            source=source,
            vector_store_cls=VectorStore,
        )

    try:
        knn_results = _run_within_deadline(
            _run_knn,
            remaining_s=deadline - time.monotonic(),
            name="lcm-knn",
        )
        coverage = knn_results.coverage
        ranked_rows = list(knn_results)
    except _WorkerCapacityError as exc:
        return degraded(f"semantic capacity exhausted: {exc}")
    except TimeoutError as exc:
        return degraded(f"semantic vector search exceeded the latency budget: {exc}")
    except Exception as exc:
        return degraded(f"semantic vector search failed: {exc}")

    if coverage == "none":
        return degraded("semantic vectors are unavailable (coverage=none)")
    if not ranked_rows:
        return degraded("semantic retrieval returned no vector candidates")

    try:
        hydrated_nodes = _run_within_deadline(
            lambda: hydrate_semantic_nodes(
                engine,
                ranked_rows=ranked_rows,
                knn_limit=knn_limit,
                deadline=deadline,
            ),
            remaining_s=deadline - time.monotonic(),
            name="lcm-result-hydration",
        )
    except _WorkerCapacityError as exc:
        return degraded(f"semantic capacity exhausted: {exc}")
    except TimeoutError:
        return _lcm_grep_deadline_error(mode, "result_resolution")
    except Exception as exc:
        return degraded(f"semantic result hydration failed: {exc}")

    current_session_id = engine.current_session_id
    has_current_session = bool(current_session_id)
    results: list[dict[str, Any]] = []
    for node, score in hydrated_nodes:
        if time.monotonic() >= deadline:
            return _lcm_grep_deadline_error(mode, "result_resolution")
        # conversation/role/source/time filters are enforced inside knn() before
        # the top-k cap, so no eligible lower-ranked vector was dropped for an
        # ineligible top hit; nothing further to post-filter here.
        confidence = _lcm_grep_confidence(score)
        result = {
            "type": "summary",
            "depth": f"d{node.depth}",
            "node_id": node.node_id,
            "session_id": node.session_id,
            "snippet": node.summary[:_LCM_GREP_SEMANTIC_SNIPPET_CHARS],
            "token_count": node.token_count,
            "expand_hint": node.expand_hint,
            "earliest_at": node.earliest_at,
            "latest_at": node.latest_at,
            "from_current_session": has_current_session and node.session_id == current_session_id,
            "score": score,
            "cosine_score": score,
            "confidence": confidence,
            "confidence_band": confidence,
        }
        results.append(result)
        if len(results) >= knn_limit:
            break

    if not results:
        return degraded("semantic vector candidates could not be resolved")

    response: dict[str, Any] = {
        "query": query,
        "mode": "semantic",
        "sort": normalize_search_sort(args.get("sort")),
        "session_scope": session_scope,
        "source": source,
        "conversation_id": conversation_id,
        "limit": limit,
        "total_results": len(results),
        "results": results[:limit] if candidate_limit is None else results,
        "coverage": coverage,
        "degraded_to_fts": False,
    }
    if role is not None:
        response["role"] = role
    if time_from is not None:
        response["time_from"] = time_from
    if time_to is not None:
        response["time_to"] = time_to
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
    return response


def _lcm_grep_hybrid(
    args: Dict[str, Any], *, engine: "LCMEngine", deadline: float
) -> dict[str, Any]:
    if time.monotonic() >= deadline:
        return _lcm_grep_deadline_error("hybrid", "hybrid_entry")
    requested_limit = _parse_int_value(args.get("limit", 10), 10)
    if requested_limit <= 0:
        return {"error": "limit must be a positive integer"}
    limit = min(requested_limit, _LCM_GREP_HARD_LIMIT_CAP)
    candidate_limit = min(
        _LCM_GREP_HYBRID_CANDIDATE_CAP,
        max(50, limit * 3),
    )

    fts_args = dict(args)
    fts_args["mode"] = "hybrid"
    fts_args["limit"] = candidate_limit
    fts = _lcm_grep_full_text_with_deadline(
        fts_args,
        engine=engine,
        deadline=deadline,
        limit_cap=_LCM_GREP_HYBRID_CANDIDATE_CAP,
    )
    if "error" in fts:
        return fts

    def degraded_to_fts(reason: str, *, coverage: str = "none") -> dict[str, Any]:
        response = dict(fts)
        response["mode"] = "hybrid"
        response["limit"] = limit
        response["total_results"] = len(fts.get("results", []))
        response["results"] = list(fts.get("results", []))[:limit]
        response["degraded_to_fts"] = True
        response["degraded_reason"] = reason
        response["coverage"] = coverage
        if requested_limit > _LCM_GREP_HARD_LIMIT_CAP:
            response["limit_clamped_from"] = requested_limit
        else:
            response.pop("limit_clamped_from", None)
        return response

    if time.monotonic() >= deadline:
        return degraded_to_fts(
            "semantic arm skipped because the request deadline was exhausted"
        )

    semantic_args = dict(args)
    semantic_args["mode"] = "hybrid"
    semantic_args["limit"] = candidate_limit
    semantic = _lcm_grep_semantic(
        semantic_args,
        engine=engine,
        deadline=deadline,
        candidate_limit=candidate_limit,
        allow_fallback=False,
    )
    if "error" in semantic:
        if semantic.get("timeout") is True:
            return degraded_to_fts(
                "semantic arm exceeded the request deadline",
                coverage=str(semantic.get("coverage", "none")),
            )
        return semantic
    if semantic.get("degraded_to_fts"):
        return degraded_to_fts(
            str(semantic.get("degraded_reason", "semantic arm unavailable")),
            coverage=str(semantic.get("coverage", "none")),
        )

    if time.monotonic() >= deadline:
        return _lcm_grep_deadline_error("hybrid", "fusion")
    # FTS is arm 0, semantic is arm 1; rrf_fuse merges by hit identity and
    # accumulates 1/(k+rank). Arm-specific metadata (which arm, confidence,
    # snippet provenance) is grep-presentation and stays here.
    ordered = rrf_fuse(
        [fts.get("results", []), semantic.get("results", [])],
        k=_LCM_GREP_RRF_K,
    )
    semantic_by_identity = {
        _hit_identity(hit): hit for hit in semantic.get("results", [])
    }
    for entry in ordered:
        ranks = entry["ranks"]
        if 0 in ranks and 1 in ranks:
            # The semantic form carries score/confidence while the FTS form carries
            # its exact house snippet/provenance. Preserve the latter as the base.
            sem_hit = semantic_by_identity[_hit_identity(entry["hit"])]
            entry["hit"].setdefault("semantic_snippet", sem_hit.get("snippet", ""))

    results: list[dict[str, Any]] = []
    for entry in ordered[:limit]:
        if time.monotonic() >= deadline:
            return _lcm_grep_deadline_error("hybrid", "fusion")
        ranks = entry["ranks"]
        hit = dict(entry["hit"])
        hit["score"] = float(entry["rrf_score"])
        hit["rrf_score"] = float(entry["rrf_score"])
        if 0 in ranks:
            hit["fts_rank"] = ranks[0]
        if 1 in ranks:
            sem_hit = semantic_by_identity[_hit_identity(entry["hit"])]
            hit["semantic_rank"] = ranks[1]
            hit["semantic_score"] = sem_hit.get("score")
            hit["confidence"] = sem_hit.get("confidence")
            hit["confidence_band"] = sem_hit.get("confidence_band")
        results.append(hit)

    response = dict(fts)
    response["mode"] = "hybrid"
    response["limit"] = limit
    response["total_results"] = len(ordered)
    response["results"] = results
    response["coverage"] = semantic.get("coverage", "none")
    response["degraded_to_fts"] = False
    response["fusion"] = "rrf"
    response["rrf_k"] = _LCM_GREP_RRF_K
    if requested_limit > _LCM_GREP_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    else:
        response.pop("limit_clamped_from", None)
    return response


def lcm_grep(args: Dict[str, Any], **kwargs) -> str:
    """Search LCM history using full-text, semantic, or RRF hybrid retrieval."""
    request_started = time.monotonic()
    mode = str(args.get("mode") or "full_text").strip().lower()
    if mode == "full_text":
        return _lcm_grep_full_text(args, **kwargs)

    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})
    timeout_s = max(
        0.001,
        float(getattr(engine._config, "embedding_query_timeout_s", 3.0)),
    )
    deadline = request_started + timeout_s
    if time.monotonic() >= deadline:
        return json.dumps(_lcm_grep_deadline_error(mode, "tool_entry"))
    if mode == "semantic":
        return json.dumps(_lcm_grep_semantic(args, engine=engine, deadline=deadline))
    if mode == "hybrid":
        return json.dumps(_lcm_grep_hybrid(args, engine=engine, deadline=deadline))
    return json.dumps({
        "error": "mode must be one of: full_text, semantic, hybrid",
    })


def _lcm_recall_recency_boost(timestamp: Any, *, now: float) -> float:
    """Half-life recency multiplier in ``[floor, 1.0]`` (newer => closer to 1)."""
    try:
        ts = float(timestamp or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    if ts <= 0:
        return _LCM_RECALL_RECENCY_FLOOR
    age = max(0.0, now - ts)
    boost = 2.0 ** (-(age / _LCM_RECALL_RECENCY_HALF_LIFE_S))
    return max(_LCM_RECALL_RECENCY_FLOOR, boost)


def _lcm_recall_summary_expand_hint(hit: dict[str, Any]) -> str:
    # Cross-session summary/DAG expansion is deferred (lcm_expand node_id is
    # current-session only), so a cross-session summary points at the session
    # loader instead of a node handle it cannot expand.
    if hit.get("from_current_session"):
        return f"lcm_expand(node_id={hit.get('node_id')})"
    return f"lcm_load_session(session_id='{hit.get('session_id') or ''}')"


def _lcm_recall_excerpt_expand_hint(hit: dict[str, Any]) -> str:
    offset = int(hit.get("content_offset") or 0)
    return f"lcm_expand(store_id={hit.get('store_id')}, content_offset={offset})"


def _lcm_recall_bounded_reason(
    arm: str, scanned: int | None, total: int | None
) -> str:
    """Degraded-reasons text for a ``coverage='bounded'`` arm (SCAN-1).

    Names the arm and the scanned/total ratio so a caller can see that the arm
    scored only the most-recent slice of the corpus (older archived vectors were
    excluded by the recency-bounded candidate scan), rather than the truncation
    being silent.
    """
    if scanned is not None and total is not None:
        return (
            f"{arm} arm coverage bounded: scored the {scanned} most-recent of "
            f"{total} vectors (older vectors excluded)"
        )
    return (
        f"{arm} arm coverage bounded: scored only the most-recent slice of the "
        "corpus (older vectors excluded)"
    )


def _lcm_recall_approx_reason(arm: str) -> str:
    """Disclosure text for a ``coverage='full_approx'`` arm (FIX 2).

    The two-stage path reaches the WHOLE corpus but stage-1 Hamming keeps only
    the M=mult*k lowest-distance survivors before the exact rescore, so its top-k
    is an approximate (recall@M) result rather than the exact top-k the
    exact-scan 'full' coverage gives. Surfaced like 'bounded' so a caller can see
    the ranking is approximate rather than assuming exhaustive exactness.
    """
    return (
        f"{arm} arm coverage full_approx: whole corpus reached via the binary "
        "prescreen, but top-k is approximate (stage-1 keeps only the closest "
        "survivors before exact rescore)"
    )


def _lcm_recall_fts_arm(
    engine: "LCMEngine", query: str, *, candidate_limit: int, deadline: float
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """FTS arm: raw messages across ALL sessions (no conversation filter)."""
    payload = _lcm_grep_full_text_with_deadline(
        {
            "query": query,
            "mode": "recall",
            "session_scope": "all",
            "limit": candidate_limit,
        },
        engine=engine,
        deadline=deadline,
        limit_cap=_LCM_GREP_HYBRID_CANDIDATE_CAP,
    )
    if "error" in payload:
        return [], payload
    hits: list[dict[str, Any]] = []
    for row in payload.get("results", []):
        store_id = row.get("store_id")
        if store_id is None:
            continue
        hit = {
            "kind": "message_excerpt",
            "store_id": store_id,
            "session_id": row.get("session_id"),
            "source": row.get("source") or "",
            "role": row.get("role"),
            "timestamp": row.get("timestamp") or 0,
            "content_offset": 0,
            "snippet": (row.get("snippet") or "")[:_LCM_RECALL_SNIPPET_CHARS],
            "from_current_session": bool(row.get("from_current_session")),
        }
        hit["expand_hint"] = _lcm_recall_excerpt_expand_hint(hit)
        hits.append(hit)
    return hits, None


def _lcm_recall_summary_arm(
    engine: "LCMEngine",
    *,
    query_vector: list[float],
    provider: Any,
    candidate_limit: int,
    deadline: float,
) -> tuple[list[dict[str, Any]], str]:
    """Summary KNN arm: embedded summaries across ALL sessions (no filter)."""
    knn_results = _run_within_deadline(
        lambda: run_knn(
            engine,
            query_vector=query_vector,
            provider=provider,
            knn_limit=candidate_limit,
            deadline=deadline,
            since=None,
            until=None,
            conversation_ids=None,
            source=None,
            vector_store_cls=VectorStore,
            scan_rows=max(1, int(getattr(engine._config, "recall_scan_rows", 25_000))),
        ),
        remaining_s=deadline - time.monotonic(),
        name="lcm-recall-summary-knn",
    )
    coverage = knn_results.coverage
    ranked_rows = list(knn_results)
    if coverage == "none" or not ranked_rows:
        return [], coverage, knn_results.scanned, knn_results.total
    nodes = _run_within_deadline(
        lambda: hydrate_semantic_nodes(
            engine,
            ranked_rows=ranked_rows,
            knn_limit=candidate_limit,
            deadline=deadline,
        ),
        remaining_s=deadline - time.monotonic(),
        name="lcm-recall-summary-hydrate",
    )
    current = engine.current_session_id
    hits: list[dict[str, Any]] = []
    for node, _score in nodes:
        hit = {
            "kind": "summary",
            "node_id": node.node_id,
            "session_id": node.session_id,
            "timestamp": node.latest_at or node.created_at or 0,
            "snippet": (node.summary or "")[:_LCM_RECALL_SNIPPET_CHARS],
            "from_current_session": bool(current) and node.session_id == current,
        }
        hit["expand_hint"] = _lcm_recall_summary_expand_hint(hit)
        hits.append(hit)
    return hits, coverage, knn_results.scanned, knn_results.total


def _lcm_recall_chunk_arm(
    engine: "LCMEngine",
    *,
    query_vector: list[float],
    provider: Any,
    candidate_limit: int,
    deadline: float,
) -> tuple[list[dict[str, Any]], str]:
    """Chunk KNN arm: verbatim chunk vectors across ALL sessions (no filter)."""
    knn_results = _run_within_deadline(
        lambda: run_chunk_knn(
            engine,
            query_vector=query_vector,
            provider=provider,
            knn_limit=candidate_limit,
            deadline=deadline,
            since=None,
            until=None,
            conversation_ids=None,
            source=None,
            vector_store_cls=VectorStore,
            scan_rows=max(1, int(getattr(engine._config, "recall_scan_rows", 25_000))),
        ),
        remaining_s=deadline - time.monotonic(),
        name="lcm-recall-chunk-knn",
    )
    coverage = knn_results.coverage
    ranked_rows = list(knn_results)
    if coverage == "none" or not ranked_rows:
        return [], coverage, knn_results.scanned, knn_results.total
    raw_hits = _run_within_deadline(
        lambda: hydrate_chunk_hits(
            engine,
            ranked_rows=ranked_rows,
            knn_limit=candidate_limit,
            deadline=deadline,
            snippet_chars=_LCM_RECALL_SNIPPET_CHARS,
        ),
        remaining_s=deadline - time.monotonic(),
        name="lcm-recall-chunk-hydrate",
    )
    current = engine.current_session_id
    hits: list[dict[str, Any]] = []
    for hit, _score in raw_hits:
        hit["from_current_session"] = bool(current) and hit.get("session_id") == current
        hit["expand_hint"] = _lcm_recall_excerpt_expand_hint(hit)
        hits.append(hit)
    return hits, coverage, knn_results.scanned, knn_results.total


def _lcm_recall_rerank(
    provider: Any,
    query: str,
    ordered: list[dict[str, Any]],
    *,
    window: int,
    deadline: float,
    config: Any,
) -> tuple[list[dict[str, Any]], str]:
    """Optionally REORDER the top ``window`` fused candidates in ONE API call.

    Default OFF. This is a pure rank-reorder WITHIN the window: the reranker's
    relevance scores decide the intra-window order but are never spliced onto the
    ``rrf_score``/``_final_score`` scale (the two live on incompatible scales, so a
    ~0-1 voyage score would otherwise dominate the ~0.05-max RRF score). Entries
    outside the window keep their incoming (post scope/recency prior) order.

    ``ordered`` MUST already carry the scope/recency prior so the window reflects
    the true top-N rather than the raw RRF order (RERANK-1). Any failure (no
    provider, non-voyage, network, deadline) skips silently back to the incoming
    order with a ``skipped: <reason>`` status.
    """
    if not bool(getattr(config, "rerank_enabled", False)):
        return ordered, "disabled"
    if (
        provider is None
        or getattr(provider, "provider_id", "") != "voyage"
        or not hasattr(provider, "rerank")
    ):
        return ordered, "skipped: rerank requires the voyage provider"
    head = ordered[:window]
    if not head:
        return ordered, "skipped: no candidates to rerank"
    documents = [str(entry["hit"].get("snippet") or "") for entry in head]
    try:
        ranked = _run_within_deadline(
            lambda: provider.rerank(
                query,
                documents,
                top_k=len(documents),
                timeout=max(0.001, deadline - time.monotonic()),
            ),
            remaining_s=deadline - time.monotonic(),
            name="lcm-recall-rerank",
        )
    except Exception as exc:  # noqa: BLE001 - any failure => skip rerank
        return ordered, f"skipped: {exc}"
    if not ranked:
        return ordered, "skipped: empty rerank result"
    reordered: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, _relevance in ranked:
        if 0 <= index < len(head) and index not in seen:
            reordered.append(head[index])
            seen.add(index)
    for index, entry in enumerate(head):
        if index not in seen:
            reordered.append(entry)
    reordered.extend(ordered[window:])
    return reordered, "applied"


def lcm_recall(args: Dict[str, Any], **kwargs) -> str:
    """Search the agent's entire memory (all conversations, all time) by meaning.

    Fuses three arms over the whole local database — FTS raw messages, embedded
    summary KNN, and verbatim chunk KNN — with RRF, then applies a soft
    scope/recency prior and (optionally) a cross-encoder rerank. The current
    conversation is only ever a ranking BOOST, never a hard filter.
    """
    request_started = time.monotonic()
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    query = str(args.get("query", "")).strip()
    if not query:
        return json.dumps({"error": "No query provided"})

    parsed_limit = _parse_int_value(args.get("limit", _LCM_RECALL_DEFAULT_LIMIT), _LCM_RECALL_DEFAULT_LIMIT)
    if parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_RECALL_LIMIT_CAP)

    scope_bias, scope_bias_error = _parse_optional_float(args.get("scope_bias"), "scope_bias")
    if scope_bias_error:
        return json.dumps({"error": scope_bias_error})
    if scope_bias is None:
        scope_bias = _LCM_RECALL_DEFAULT_SCOPE_BIAS
    scope_bias = max(0.0, min(1.0, float(scope_bias)))

    include = str(args.get("include") or "all").strip().lower()
    if include not in _LCM_RECALL_VALID_INCLUDE:
        return json.dumps({"error": "include must be one of: all, summaries, verbatim"})

    # lcm_recall fans out three arms + fusion/hydration/rerank, so it uses its own
    # (larger) budget rather than lcm_grep's single-arm query deadline (sprint-opt-2).
    timeout_s = max(0.001, float(getattr(engine._config, "recall_query_timeout_s", 8.0)))
    deadline = request_started + timeout_s

    candidate_limit = min(_LCM_GREP_HYBRID_CANDIDATE_CAP, max(50, limit * 4))
    rerank_window = min(50, max(1, limit * 4))

    embeddings_enabled = bool(getattr(engine._config, "embeddings_enabled", False))
    # FTS runs for verbatim/all normally, but ALSO whenever embeddings are
    # disabled -- including for include='summaries', whose only vector arm is dead
    # in that state. Without this, include='summaries' + embeddings-off returns
    # zero hits instead of degrading to the full-text arm (F4-degrade-to-fts).
    run_fts = include in {"all", "verbatim"} or not embeddings_enabled
    run_summary = include in {"all", "summaries"}
    run_chunk = include in {"all", "verbatim"}

    arm_hits: dict[str, list[dict[str, Any]]] = {}
    coverage: dict[str, str] = {}
    degraded_reasons: list[str] = []
    timed_out = False
    provider: Any = None

    # -- FTS arm (the default-on value: works with embeddings disabled) --
    if run_fts:
        try:
            hits, fts_error = _lcm_recall_fts_arm(
                engine, query, candidate_limit=candidate_limit, deadline=deadline
            )
        except (_WorkerCapacityError, TimeoutError) as exc:
            hits, fts_error = [], {"error": str(exc)}
        if fts_error is not None:
            coverage["fts"] = "none"
            degraded_reasons.append("full-text arm unavailable")
            timed_out = timed_out or bool(fts_error.get("timeout"))
        else:
            arm_hits["fts"] = hits
            coverage["fts"] = "ok"

    # -- Vector arms. Local/same-model corpora share one query embedding;
    # Voyage's context chunk corpus resolves and embeds with its own model. --
    if run_summary or run_chunk:
        if not embeddings_enabled:
            degraded_reasons.append("semantic retrieval is disabled")
            if run_summary:
                coverage["summary"] = "disabled"
            if run_chunk:
                coverage["chunk"] = "disabled"
        elif time.monotonic() >= deadline:
            timed_out = True
        else:
            query_vector: list[float] | None = None
            chunk_provider: Any = None
            chunk_query_vector: list[float] | None = None
            provider_override = kwargs.get("provider_override")
            try:
                provider = _run_within_deadline(
                    lambda: _resolve_recall_provider(
                        engine,
                        deadline=deadline,
                        provider_override=provider_override,
                    ),
                    remaining_s=deadline - time.monotonic(),
                    name="lcm-provider-resolution",
                )
                if provider is None:
                    degraded_reasons.append("embedding provider is not configured")
                elif run_summary:
                    query_vector = _lcm_grep_embed_query(
                        provider, query, remaining_s=deadline - time.monotonic()
                    )
            except VoyageError as exc:
                provider = None
                degraded_reasons.append(f"query embedding failed: {exc}")
            except TimeoutError:
                provider = None
                timed_out = True
            except Exception as exc:  # noqa: BLE001 - degrade, never bare-error the whole tool
                provider = None
                degraded_reasons.append(f"embedding provider unavailable: {exc}")

            if run_chunk and provider is not None:
                try:
                    chunk_provider = _run_within_deadline(
                        lambda: _resolve_recall_chunk_provider(
                            engine, provider, deadline=deadline
                        ),
                        remaining_s=deadline - time.monotonic(),
                        name="lcm-chunk-provider-resolution",
                    )
                    if chunk_provider is None:
                        degraded_reasons.append(
                            "chunk embedding provider is not configured"
                        )
                    elif (
                        query_vector is not None
                        and chunk_provider.provider_id == provider.provider_id
                        and chunk_provider.model_id == provider.model_id
                    ):
                        chunk_query_vector = query_vector
                    else:
                        chunk_query_vector = _lcm_grep_embed_query(
                            chunk_provider,
                            query,
                            remaining_s=deadline - time.monotonic(),
                        )
                except VoyageError as exc:
                    degraded_reasons.append(f"chunk query embedding failed: {exc}")
                except TimeoutError:
                    timed_out = True
                except Exception as exc:  # noqa: BLE001
                    degraded_reasons.append(
                        f"chunk embedding provider unavailable: {exc}"
                    )

            if query_vector is not None:
                if run_summary:
                    try:
                        hits, cov, scanned, total = _lcm_recall_summary_arm(
                            engine,
                            query_vector=query_vector,
                            provider=provider,
                            candidate_limit=candidate_limit,
                            deadline=deadline,
                        )
                        arm_hits["summary"] = hits
                        coverage["summary"] = cov
                        if cov == "none":
                            degraded_reasons.append("summary vectors are unavailable")
                        elif cov == "bounded":
                            degraded_reasons.append(
                                _lcm_recall_bounded_reason("summary", scanned, total)
                            )
                        elif cov == "full_approx":
                            degraded_reasons.append(
                                _lcm_recall_approx_reason("summary")
                            )
                    except TimeoutError:
                        timed_out = True
                        coverage["summary"] = "none"
                    except Exception as exc:  # noqa: BLE001
                        coverage["summary"] = "none"
                        degraded_reasons.append(f"summary arm failed: {exc}")
            if chunk_query_vector is not None:
                if run_chunk:
                    try:
                        hits, cov, scanned, total = _lcm_recall_chunk_arm(
                            engine,
                            query_vector=chunk_query_vector,
                            provider=chunk_provider,
                            candidate_limit=candidate_limit,
                            deadline=deadline,
                        )
                        arm_hits["chunk"] = hits
                        coverage["chunk"] = cov
                        if cov == "none":
                            degraded_reasons.append("chunk vectors are unavailable")
                        elif cov == "bounded":
                            degraded_reasons.append(
                                _lcm_recall_bounded_reason("chunk", scanned, total)
                            )
                        elif cov == "full_approx":
                            degraded_reasons.append(
                                _lcm_recall_approx_reason("chunk")
                            )
                    except TimeoutError:
                        timed_out = True
                        coverage["chunk"] = "none"
                    except Exception as exc:  # noqa: BLE001
                        coverage["chunk"] = "none"
                        degraded_reasons.append(f"chunk arm failed: {exc}")

    # -- RRF fusion over the arms that produced hits (order fixes base-hit win) --
    # Per-arm weights down-weight the weak FTS arm so the 3-arm hybrid is never
    # dragged below its best arm (measured −21 R@5 on LongMemEval for naive
    # equal-weight fusion). Weights echo into provenance below.
    arm_order = [name for name in ("fts", "summary", "chunk") if arm_hits.get(name)]
    configured_arm_weights = getattr(engine._config, "recall_arm_weights", None) or {}
    arm_weights = [float(configured_arm_weights.get(name, 1.0)) for name in arm_order]
    ordered = rrf_fuse(
        [arm_hits[name] for name in arm_order],
        k=_LCM_RECALL_RRF_K,
        weights=arm_weights,
    )

    # Merge chunk-arm provenance onto a message whose store_id also surfaced via
    # FTS. The chunk list is best-first, so keep the BEST-ranked chunk per store
    # (setdefault, not a plain comprehension which keeps the WORST last entry --
    # F1-chunk-dedupe-wrong-span). When both arms carry the message, the merged
    # hit takes the better-ranked arm's snippet AND its offsets together so the
    # displayed preview and the expand handle describe the SAME span (DEDUPE-1);
    # a mismatched snippet/offset pair pointed at two different parts of the
    # message. arm_order fixes the fused base object to the earliest arm, so we
    # compare per-arm ranks explicitly rather than trusting insertion order.
    chunk_by_store: dict[Any, dict[str, Any]] = {}
    for chunk_hit in arm_hits.get("chunk", []):
        chunk_by_store.setdefault(chunk_hit.get("store_id"), chunk_hit)
    chunk_arm_index = arm_order.index("chunk") if "chunk" in arm_order else None
    fts_arm_index = arm_order.index("fts") if "fts" in arm_order else None
    for entry in ordered:
        hit = entry["hit"]
        if hit.get("kind") != "message_excerpt":
            continue
        chunk_hit = chunk_by_store.get(hit.get("store_id"))
        if chunk_hit is None:
            continue
        ranks = entry["ranks"]
        chunk_rank = ranks.get(chunk_arm_index) if chunk_arm_index is not None else None
        fts_rank = ranks.get(fts_arm_index) if fts_arm_index is not None else None
        # Adopt the chunk arm's span+snippet when the base hit lacks a span (a
        # pure-FTS base) OR the chunk arm ranked this message at least as well as
        # the FTS arm. Lower rank number == better.
        chunk_wins = not hit.get("chunk_span") or (
            chunk_rank is not None
            and (fts_rank is None or chunk_rank <= fts_rank)
        )
        if chunk_wins:
            hit["chunk_span"] = chunk_hit.get("chunk_span")
            hit["content_offset"] = chunk_hit.get("content_offset", 0)
            hit["snippet"] = chunk_hit.get("snippet") or hit.get("snippet")
            hit["expand_hint"] = _lcm_recall_excerpt_expand_hint(hit)

    # -- Scope-prior + recency rescoring. Applied BEFORE the rerank window is
    #    selected so an item the prior lifts into the true top-N is the one
    #    actually reranked (RERANK-1). Boost eligibility spans the whole
    #    conversation-scope session set, not a single session-id equality, so a
    #    mid-conversation session rotation keeps the boost (SCOPE-1). --
    now = time.time()
    scope_session_ids = set(_recent_conversation_scope_session_ids(engine))
    for entry in ordered:
        hit = entry["hit"]
        rank_score = float(entry.get("rrf_score", 0.0))
        is_current = 1.0 if hit.get("session_id") in scope_session_ids else 0.0
        recency = _lcm_recall_recency_boost(hit.get("timestamp"), now=now)
        entry["_final_score"] = rank_score * (1.0 + scope_bias * is_current) * recency
    ordered.sort(
        key=lambda entry: (
            -float(entry["_final_score"]),
            -float(entry.get("rrf_score", 0.0)),
            _hit_identity(entry["hit"]),
        )
    )

    # -- Optional rerank stage (default OFF): a pure rank-REORDER within the top
    #    window of the post-prior order (no score splicing onto the RRF scale). --
    ordered, rerank_status = _lcm_recall_rerank(
        provider, query, ordered, window=rerank_window, deadline=deadline, config=engine._config
    )

    # -- Response shaping (char-capped) --
    hits_out: list[dict[str, Any]] = []
    response_chars = 0
    for entry in ordered:
        hit = entry["hit"]
        arms = sorted({arm_order[index] for index in entry["ranks"].keys()})
        item: dict[str, Any] = {
            "kind": hit.get("kind"),
            "session_id": hit.get("session_id"),
            "timestamp": hit.get("timestamp") or 0,
            "snippet": (hit.get("snippet") or "")[:_LCM_RECALL_SNIPPET_CHARS],
            "score": round(float(entry["_final_score"]), 6),
            "expand_hint": hit.get("expand_hint"),
            "from_current_session": bool(hit.get("from_current_session")),
            "arms": arms,
        }
        if hit.get("kind") == "summary":
            item["node_id"] = hit.get("node_id")
        else:
            item["store_id"] = hit.get("store_id")
            if hit.get("chunk_span"):
                item["chunk_span"] = hit["chunk_span"]
        item_chars = len(json.dumps(item, ensure_ascii=False))
        if hits_out and response_chars + item_chars > _LCM_RECALL_RESPONSE_CHAR_CAP:
            break
        response_chars += item_chars
        hits_out.append(item)
        if len(hits_out) >= limit:
            break

    degraded = bool(degraded_reasons)
    response: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "scope_bias": scope_bias,
        "include": include,
        "total_results": len(hits_out),
        "hits": hits_out,
        "provenance": {
            "arms_run": arm_order,
            "arm_weights": {name: arm_weights[i] for i, name in enumerate(arm_order)},
            "coverage": coverage,
            "rerank": rerank_status,
            "ordering": (
                "rrf-fusion -> scope/recency prior -> rerank reorder (top window); "
                "the reported score is the scope/recency-adjusted RRF score, and "
                "rerank (when applied) only permutes the top window without "
                "replacing that score"
            ),
        },
        "degraded": degraded,
    }
    if degraded:
        response["degraded_reason"] = "; ".join(dict.fromkeys(degraded_reasons))
    if timed_out:
        response["timeout"] = True
    if requested_limit > _LCM_RECALL_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
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
    conn = engine._dag.connection
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
    rows = engine._dag.connection.execute(
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
    return _externalized_top_level_fields_before_content(text)


def _read_externalized_payload_metadata_prefix(
    path: Path,
    *,
    max_read_bytes: int = _LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES,
) -> tuple[str, bool, bool]:
    """Read bounded JSON metadata before the externalized payload body.

    Returns ``(prefix_text, content_string_seen, prefix_truncated)``. The content
    string body is intentionally not consumed; ``lcm_inspect`` reports bounded
    metadata only and leaves full JSON/body validation to explicit expansion.
    """
    return read_externalized_payload_metadata_prefix(
        path,
        max_read_bytes=max_read_bytes,
    )


def _validate_externalized_payload_json_tail(
    path: Path,
    metadata_prefix_text: str,
    *,
    max_tail_bytes: int = _LCM_GREP_EXTERNALIZED_DOCUMENT_TAIL_BYTES,
) -> dict[str, Any] | None:
    """Validate the closing JSON structure using only a bounded tail window."""
    metadata_prefix = metadata_prefix_text.encode("utf-8")
    try:
        file_size = path.stat().st_size
        tail_offset = max(len(metadata_prefix), file_size - max_tail_bytes)
        with path.open("rb") as handle:
            handle.seek(tail_offset)
            tail = handle.read(max_tail_bytes + 1)
    except OSError:
        return None
    if len(tail) > max_tail_bytes:
        return None

    for index, byte in enumerate(tail):
        if byte != ord('"'):
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and tail[cursor] == ord("\\"):
            backslashes += 1
            cursor -= 1
        if backslashes % 2:
            continue
        try:
            payload = json.loads((metadata_prefix + b'"' + tail[index + 1 :]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _inspect_externalized_payload_metadata(
    engine: "LCMEngine",
    ref: str,
    session_id: str,
    *,
    max_read_bytes: int = _LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES,
    require_valid_document_tail: bool = False,
) -> dict[str, Any]:
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
        metadata_prefix_text, content_key_seen, prefix_truncated = _read_externalized_payload_metadata_prefix(
            path,
            max_read_bytes=max_read_bytes,
        )
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

    if require_valid_document_tail:
        full_payload = _validate_externalized_payload_json_tail(path, metadata_prefix_text)
        if full_payload is None:
            return {"readable": False, "error": "invalid_payload"}
        if (full_payload.get("session_id") or "") != session_id:
            return {"readable": False, "error": "session_mismatch"}

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
        "payload_validation": "document_tail" if require_valid_document_tail else "metadata_prefix",
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


def _temporal_rollups_status(engine: "LCMEngine") -> dict[str, Any]:
    """Return the cheap, read-only temporal-rollup operator status payload."""
    enabled = bool(engine._config.temporal_rollups_enabled)
    scope = engine.current_session_id or ""
    payload: dict[str, Any] = {
        "enabled": enabled,
        "scope": scope,
        "counts": {
            kind: {status: 0 for status in _TEMPORAL_ROLLUP_STATUSES}
            for kind in _TEMPORAL_ROLLUP_PERIOD_KINDS
        },
        "oldest_stale_age_seconds": None,
        "last_build_cursors": {kind: None for kind in _TEMPORAL_ROLLUP_PERIOD_KINDS},
        "last_built_at": {kind: None for kind in _TEMPORAL_ROLLUP_PERIOD_KINDS},
        "last_error": None,
    }
    # Disabled deployments deliberately avoid even the metadata reads. This
    # keeps the optional feature inert while preserving a stable status shape.
    if not enabled or not scope:
        return payload

    conn = engine._dag.connection
    if conn is None:
        return payload
    try:
        rows = conn.execute(
            """
            SELECT period_kind, status, COUNT(*)
            FROM lcm_rollups INDEXED BY sqlite_autoindex_lcm_rollups_1
            WHERE period_kind IN ('day', 'week', 'month') AND scope = ?
            GROUP BY period_kind, status
            """,
            (scope,),
        ).fetchall()
        for period_kind, status, count in rows:
            kind_key = str(period_kind)
            status_key = str(status)
            if kind_key in payload["counts"] and status_key in payload["counts"][kind_key]:
                payload["counts"][kind_key][status_key] = int(count or 0)

        oldest_stale = conn.execute(
            """
            SELECT period_start
            FROM lcm_rollups INDEXED BY sqlite_autoindex_lcm_rollups_1
            WHERE period_kind IN ('day', 'week', 'month')
              AND scope = ? AND status = 'stale'
            ORDER BY period_start
            LIMIT 1
            """,
            (scope,),
        ).fetchone()
        if oldest_stale and oldest_stale[0]:
            stale_start = datetime.combine(
                datetime.fromisoformat(str(oldest_stale[0])).date(),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
            payload["oldest_stale_age_seconds"] = max(
                0, int((datetime.now(timezone.utc) - stale_start).total_seconds())
            )

        cursor_rows = conn.execute(
            """
            SELECT period_kind, last_build_cursor, last_built_at
            FROM lcm_rollup_state
            WHERE period_kind IN ('day', 'week', 'month') AND scope = ?
            """,
            (scope,),
        ).fetchall()
        for period_kind, cursor, built_at in cursor_rows:
            kind_key = str(period_kind)
            if kind_key in payload["last_build_cursors"]:
                payload["last_build_cursors"][kind_key] = cursor
                payload["last_built_at"][kind_key] = built_at

        error_row = conn.execute(
            """
            SELECT substr(error, 1, ?)
            FROM lcm_rollups
            WHERE scope = ? AND error IS NOT NULL AND error != ''
            ORDER BY rollup_id DESC
            LIMIT 1
            """,
            (_OPERATOR_TEXT_FIELD_MAX_CHARS + 1, scope),
        ).fetchone()
        if error_row:
            last_error, was_truncated = _bounded_operator_field(error_row[0])
            payload["last_error"] = last_error
            if was_truncated:
                payload.setdefault("truncated_fields", []).append("last_error")
    except Exception as exc:  # pragma: no cover - defensive legacy-schema degradation
        logger.debug("LCM temporal rollup status query failed", exc_info=True)
        query_error, was_truncated = _bounded_operator_field(
            f"{type(exc).__name__}: {exc}"
        )
        payload["query_error"] = query_error
        if was_truncated:
            payload.setdefault("truncated_fields", []).append("query_error")
    return payload


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
        return _bounded_inspect_json({
            "error": "No active session",
            "read_only": True,
            "runtime_identity": full_status.get("runtime_identity") or engine.get_runtime_identity(),
            "ingest_protection": full_status.get("ingest_protection"),
            "temporal_rollups": _temporal_rollups_status(engine),
        })

    full_status = engine.get_status()
    runtime_identity = full_status.get("runtime_identity") or engine.get_runtime_identity()
    lifecycle = _inspect_lifecycle_state(engine, session_id, conversation_id)

    store_totals_row = engine._store.connection.execute(
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
    fresh_tail_rows, fresh_tail_boundary = engine._get_session_fresh_tail(session_id)
    fresh_tail_display_rows = fresh_tail_rows[-limit:]
    fresh_tail_items = [
        _inspect_message_metadata(row)
        for row in fresh_tail_display_rows
    ]

    depth_stats = engine._dag.get_session_depth_stats(session_id)
    total_dag_nodes = sum(info["count"] for info in depth_stats.values())
    total_dag_tokens = sum(info["tokens"] for info in depth_stats.values())
    total_dag_source_tokens = sum(info["source_tokens"] for info in depth_stats.values())
    latest_node_rows = engine._dag.connection.execute(
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
            "fresh_tail_max_tokens": engine._config.fresh_tail_max_tokens,
            "effective_fresh_tail_count": len(fresh_tail_rows),
            "effective_fresh_tail_tokens": fresh_tail_boundary.tokens,
            "pre_tail_message_count": max(0, message_total - len(fresh_tail_rows)),
            "fresh_tail": {
                "returned": len(fresh_tail_items),
                "token_limited": fresh_tail_boundary.token_limited,
                "tool_group_extended": fresh_tail_boundary.tool_group_extended,
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
        "temporal_rollups": _temporal_rollups_status(engine),
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
    return _bounded_inspect_json(response)


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
        "total_compactions": full_status.get("total_compactions", 0),
        "total_compactions_scope": full_status.get(
            "total_compactions_scope", "current_conversation"
        ),
        "last_compression_status": full_status.get("last_compression_status", "idle"),
        "last_compression_noop_reason": full_status.get("last_compression_noop_reason", ""),
        "threshold_full_sweep": full_status.get("threshold_full_sweep"),
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
            "fresh_tail_max_tokens": engine._config.fresh_tail_max_tokens,
            "leaf_chunk_tokens": engine._config.leaf_chunk_tokens,
            "dynamic_leaf_chunk_enabled": engine._config.dynamic_leaf_chunk_enabled,
            "dynamic_leaf_chunk_max": engine._config.dynamic_leaf_chunk_max,
            "cache_friendly_condensation_enabled": engine._config.cache_friendly_condensation_enabled,
            "cache_friendly_min_debt_groups": engine._config.cache_friendly_min_debt_groups,
            "deferred_maintenance_enabled": engine._config.deferred_maintenance_enabled,
            "deferred_maintenance_max_passes": engine._config.deferred_maintenance_max_passes,
            "critical_budget_pressure_ratio": engine._config.critical_budget_pressure_ratio,
            "threshold_full_sweep_enabled": engine._config.threshold_full_sweep_enabled,
            "summary_prefix_target_tokens": engine._config.summary_prefix_target_tokens,
            "threshold_full_sweep_max_passes": 12,
            "threshold_full_sweep_max_seconds": 120,
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
        "proactive_recall": {
            "enabled": bool(getattr(engine._config, "proactive_recall_enabled", False)),
            "min_score": getattr(engine._config, "proactive_recall_min_score", 0.0),
            "budget_tokens": getattr(engine._config, "proactive_recall_budget_tokens", 0),
            "provider_override": getattr(engine._config, "proactive_recall_provider", "") or "",
            "injected": int(getattr(engine, "_proactive_recall_injected_count", 0) or 0),
            "skipped": int(getattr(engine, "_proactive_recall_skipped_count", 0) or 0),
            "timeout": int(getattr(engine, "_proactive_recall_timeout_count", 0) or 0),
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
        result = engine._store.connection.execute("PRAGMA integrity_check").fetchone()
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
        conn = engine._store.connection
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
        ("messages_fts_integrity", engine._store.connection, build_message_fts_spec()),
        ("nodes_fts_integrity", engine._dag.connection, build_nodes_fts_spec()),
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
        # A prior non-blocking background integrity scan records a persisted
        # ``fts_integrity_failed:<table>`` flag when it finds corruption
        # without rebuilding. Surface it even when this run's live check is
        # throttled/unchecked, mirroring the /lcm doctor text path.
        try:
            failed_flag = load_integrity_failed(conn, spec)
        except Exception:  # pragma: no cover - defensive
            failed_flag = None
        if failed_flag:
            checks.append({
                "check": f"{check_name}_background_flag",
                "status": "fail",
                "detail": {
                    "flagged_at": failed_flag.get("at"),
                    "detail": failed_flag.get("detail"),
                    "guidance": "background integrity scan flagged this index; run `/lcm doctor repair apply`",
                },
            })

    # 2. SQLite storage posture and payload diagnostics
    try:
        journal_mode_row = engine._store.connection.execute("PRAGMA journal_mode").fetchone()
        quick_check_row = engine._store.connection.execute("PRAGMA quick_check").fetchone()
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
        payload_risks = scan_sqlite_payload_risks(engine._store.connection)
        externalized_stats = externalized_payload_stats(engine._config, hermes_home=engine._hermes_home)
        externalized_integrity = scan_externalized_payload_integrity(
            engine._store.connection,
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
        msg_count = engine._store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        fts_count = engine._store.connection.execute(
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
