"""Provider-neutral bounded retrieval state for one continuous answer turn.

The answerer remains the only semantic selector.  This module validates its
typed intent, tracks named evidence gaps, dispatches only existing LCM tools,
normalizes exact raw/assertion refs, enforces hard budgets, and optionally
persists a query-derived evidence view.  The controller contains no model or
provider client and never stores final prose; dispatched retrieval tools retain
their own declared provider behavior and provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
import math
import re
import threading
import time
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence
import uuid

from .query_view_store import (
    CorpusSnapshot,
    QueryViewBuildInProgressError,
    QueryViewIdentity,
    QueryViewStore,
)
from .reasoning import temporal_trust_wire
from .tokens import count_tokens


ADAPTIVE_RETRIEVAL_VERSION = "adaptive-retrieval-v1"
MAX_RETRIEVAL_ROUNDS = 3
MAX_CANDIDATE_REFS = 40
MAX_SEARCH_LEADS = 24
MAX_CONTEXT_TOKENS = 7_500
MAX_CONTEXT_CHARS = 40_000
MAX_REQUIREMENTS = 12
MAX_REQUIREMENT_SLOT_ID_CHARS = 64
MAX_REQUIREMENT_DESCRIPTION_CHARS = 256
MAX_ACTIVE_RETRIEVALS = 32
RETRIEVAL_TTL_SECONDS = 15 * 60
MAX_EVIDENCE_CHARS = 2_400
MAX_QUESTION_CHARS = 4_096
MAX_TOOL_ARGS_CHARS = 2_048
MAX_LEAD_FIELD_CHARS = 512

_SLOT_ID_RE = re.compile(
    rf"^[a-z][a-z0-9_.-]{{0,{MAX_REQUIREMENT_SLOT_ID_CHARS - 1}}}$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_ARGUMENT_KEYS = frozenset({
    "question_id",
    "questionid",
    "ground_truth",
    "groundtruth",
    "reference_answer",
    "referenceanswer",
    "audit_label",
    "auditlabel",
    "judge_output",
    "question_type",
})
_ALLOWED_RETRIEVAL_TOOLS = frozenset({
    "lcm_recall",
    "lcm_recent",
    "lcm_query_state",
    "lcm_load_session",
    "lcm_expand",
})
_TOOL_ARGUMENT_KEYS: dict[str, frozenset[str]] = {
    "lcm_recall": frozenset({"query", "limit", "scope_bias", "include", "detail"}),
    "lcm_recent": frozenset({"period", "scope", "limit"}),
    "lcm_query_state": frozenset({
        "subject_key", "predicate_key", "kinds", "scope_key", "speaker_role",
        "as_of", "limit",
    }),
    "lcm_load_session": frozenset({
        "session_id", "limit", "max_content_chars", "after_store_id", "roles",
        "time_from", "time_to",
    }),
    "lcm_expand": frozenset({
        "node_id", "externalized_ref", "store_id", "max_tokens", "source_offset",
        "source_limit", "content_offset",
    }),
}

Dispatch = Callable[[str, dict[str, Any]], str]


def _canonical_json(value: Any, *, field_name: str, max_chars: int) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite JSON") from exc
    if len(encoded) > max_chars:
        raise ValueError(f"{field_name} exceeds {max_chars} characters")
    return encoded


# Derived from MAX_REQUIREMENTS, the declared slot/description limits, and
# MAX_CANDIDATE_REFS. NUL models the widest canonical JSON escape; one additional
# max-sized item provides headroom if the identity shape gains small metadata.
_REQUIREMENT_IDENTITY_ITEM_MAX_CHARS = len(
    json.dumps(
        {
            "slot_id": "s" * MAX_REQUIREMENT_SLOT_ID_CHARS,
            "description": "\0" * MAX_REQUIREMENT_DESCRIPTION_CHARS,
            "minimum_refs": MAX_CANDIDATE_REFS,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
)
_REQUIREMENTS_DIGEST_MAX_CHARS = (
    2
    + MAX_REQUIREMENTS * _REQUIREMENT_IDENTITY_ITEM_MAX_CHARS
    + (MAX_REQUIREMENTS - 1)
    + _REQUIREMENT_IDENTITY_ITEM_MAX_CHARS
)


def _normalize_question_date(value: Any) -> str:
    text = str(value or "").strip().replace("/", "-")
    if not text:
        return ""
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError("question_date must be an ISO calendar date") from exc


def _validate_argument_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if normalized in _FORBIDDEN_ARGUMENT_KEYS:
                raise ValueError(f"unsupported runtime argument key: {key}")
            _validate_argument_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_argument_keys(item)


@dataclass(frozen=True)
class EvidenceRequirement:
    slot_id: str
    description: str
    minimum_refs: int = 1

    @classmethod
    def parse(cls, value: Any) -> "EvidenceRequirement":
        if not isinstance(value, Mapping):
            raise ValueError("each evidence requirement must be an object")
        unknown = set(value) - {"slot_id", "description", "minimum_refs"}
        if unknown:
            raise ValueError(
                "evidence requirement contains unsupported fields: "
                + ", ".join(sorted(str(item) for item in unknown))
            )
        slot_id = str(value.get("slot_id") or "").strip().casefold()
        if not _SLOT_ID_RE.fullmatch(slot_id):
            raise ValueError(
                "slot_id must start with a letter and contain only lowercase "
                "letters, digits, dots, underscores, or hyphens"
            )
        description = " ".join(str(value.get("description") or "").split())
        if (
            not description
            or len(description) > MAX_REQUIREMENT_DESCRIPTION_CHARS
        ):
            raise ValueError(
                "requirement description must contain "
                f"1..{MAX_REQUIREMENT_DESCRIPTION_CHARS} characters"
            )
        if any(0xD800 <= ord(char) <= 0xDFFF for char in description):
            raise ValueError(
                "requirement description must not contain surrogate code points"
            )
        raw_minimum = value.get("minimum_refs", 1)
        if isinstance(raw_minimum, bool):
            raise ValueError("minimum_refs must be an integer")
        try:
            minimum = int(raw_minimum)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("minimum_refs must be an integer") from exc
        if not 1 <= minimum <= MAX_CANDIDATE_REFS:
            raise ValueError(
                f"minimum_refs must be between 1 and {MAX_CANDIDATE_REFS}"
            )
        return cls(slot_id, description, minimum)

    def public_dict(self, refs: Sequence[str]) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "description": self.description,
            "minimum_refs": self.minimum_refs,
            "evidence_refs": list(refs),
            "closed": len(refs) >= self.minimum_refs,
        }


def requirements_digest(requirements: Sequence[EvidenceRequirement]) -> str:
    identity = [
        {
            "slot_id": item.slot_id,
            "description": item.description,
            "minimum_refs": item.minimum_refs,
        }
        for item in sorted(requirements, key=lambda item: item.slot_id)
    ]
    return hashlib.sha256(
        _canonical_json(
            identity,
            field_name="requirements",
            max_chars=_REQUIREMENTS_DIGEST_MAX_CHARS,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ExactEvidence:
    citation: str
    store_id: int
    span_start: int
    span_end: int
    quote: str
    content_sha256: str
    session_id: str
    conversation_id: str
    source: str
    role: str
    timestamp: float
    assertion_id: str = ""
    context_tokens: int = 0
    context_chars: int = 0

    def public_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "citation": self.citation,
            "store_id": self.store_id,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "quote": self.quote,
            "content_sha256": self.content_sha256,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "source": self.source,
            "role": self.role,
            "timestamp": self.timestamp,
        }
        if self.assertion_id:
            payload["assertion_id"] = self.assertion_id
        return payload


@dataclass(frozen=True)
class SearchLead:
    lead_id: str
    payload: dict[str, Any]
    context_tokens: int
    context_chars: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "lead_id": self.lead_id,
            "evidence_eligible": False,
            **self.payload,
        }


@dataclass(frozen=True)
class RetrievalRound:
    round_number: int
    missing_slot: str
    tool: str
    tool_args: dict[str, Any]
    new_refs: tuple[str, ...]
    new_leads: tuple[str, ...]
    latency_ms: float
    status: str
    tool_provenance: dict[str, Any] = field(default_factory=dict)
    tool_metrics: dict[str, Any] = field(default_factory=dict)
    result_truncated: bool = False

    def public_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_number,
            "missing_slot": self.missing_slot,
            "tool": self.tool,
            "tool_args": self.tool_args,
            "new_refs": list(self.new_refs),
            "new_ref_count": len(self.new_refs),
            "new_lead_count": len(self.new_leads),
            "latency_ms": self.latency_ms,
            "status": self.status,
            "tool_provenance": self.tool_provenance,
            "tool_metrics": self.tool_metrics,
            "result_truncated": self.result_truncated,
        }


@dataclass
class RetrievalState:
    retrieval_id: str
    question: str
    question_date: str
    identity: QueryViewIdentity
    requirements: tuple[EvidenceRequirement, ...]
    owner_session_id: str
    owner_conversation_id: str
    created_at: float
    updated_at: float
    candidates: dict[str, ExactEvidence] = field(default_factory=dict)
    leads: dict[str, SearchLead] = field(default_factory=dict)
    slot_refs: dict[str, list[str]] = field(default_factory=dict)
    rounds: list[RetrievalRound] = field(default_factory=list)
    status: Literal["active", "ready", "fallback", "finished", "abandoned"] = "active"
    termination_reason: str = ""
    context_tokens: int = 0
    context_chars: int = 0
    budget_exhausted: bool = False
    view_lookup_status: str = "miss"
    view_delta_events: tuple[dict[str, Any], ...] = ()
    view_delta_truncated: bool = False
    cached_trace: dict[str, Any] | None = None
    view_corpus_snapshot: CorpusSnapshot | None = None


def _parse_requirements(raw: Any) -> tuple[EvidenceRequirement, ...]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_REQUIREMENTS:
        raise ValueError(
            f"requirements must contain between 1 and {MAX_REQUIREMENTS} items"
        )
    parsed = tuple(EvidenceRequirement.parse(item) for item in raw)
    ids = [item.slot_id for item in parsed]
    if len(ids) != len(set(ids)):
        raise ValueError("requirement slot_id values must be unique")
    if sum(item.minimum_refs for item in parsed) > MAX_CANDIDATE_REFS:
        raise ValueError("combined minimum_refs exceeds the candidate-ref budget")
    return parsed


def _parse_identity(
    raw: Any,
    *,
    question_date: str,
    digest: str,
) -> QueryViewIdentity:
    if not isinstance(raw, Mapping):
        raise ValueError("identity must be an object")
    allowed = {
        "intent_type",
        "operation",
        "subject_key",
        "predicate_key",
        "role_key",
        "scope_key",
        "conversation_id",
        "unit",
        "distinct_policy",
        "time_mode",
        "question_anchor",
        "window_start",
        "window_end",
        "policy_version",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            "identity contains unsupported fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    values = {key: raw[key] for key in allowed if key in raw}
    time_mode = str(values.get("time_mode") or "none").strip().casefold()
    if time_mode == "relative" and not values.get("question_anchor"):
        values["question_anchor"] = question_date
    identity = QueryViewIdentity(**values, requirements_digest=digest).normalized()
    if time_mode == "relative" and identity.question_anchor != question_date:
        raise ValueError("relative identity anchor must equal question_date")
    return identity


def _bounded_tool_args(tool: str, raw: Any) -> dict[str, Any]:
    if tool not in _ALLOWED_RETRIEVAL_TOOLS:
        raise ValueError(
            "tool must be one of: " + ", ".join(sorted(_ALLOWED_RETRIEVAL_TOOLS))
        )
    if not isinstance(raw, Mapping):
        raise ValueError("tool_args must be an object")
    unknown = set(raw) - _TOOL_ARGUMENT_KEYS[tool]
    if unknown:
        raise ValueError(
            f"{tool} arguments contain unsupported fields: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    _validate_argument_keys(raw)
    args = dict(raw)
    _canonical_json(args, field_name="tool_args", max_chars=MAX_TOOL_ARGS_CHARS)

    def bounded_int(key: str, default: int, maximum: int) -> int:
        value = args.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"{tool} {key} must be an integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{tool} {key} must be an integer") from exc
        return min(maximum, max(1, parsed))

    if tool == "lcm_recall":
        args["detail"] = "answer_ready"
        args["limit"] = bounded_int("limit", 8, 8)
    elif tool == "lcm_recent":
        args["limit"] = bounded_int("limit", 8, 8)
    elif tool == "lcm_query_state":
        args["limit"] = bounded_int("limit", 20, 20)
    elif tool == "lcm_load_session":
        args["limit"] = bounded_int("limit", 8, 8)
        args["max_content_chars"] = bounded_int(
            "max_content_chars", MAX_EVIDENCE_CHARS, MAX_EVIDENCE_CHARS
        )
    elif tool == "lcm_expand":
        args["max_tokens"] = bounded_int("max_tokens", 1_800, 1_800)
    _canonical_json(args, field_name="tool_args", max_chars=MAX_TOOL_ARGS_CHARS)
    return args


def _candidate_from_item(
    item: Mapping[str, Any],
    *,
    engine: Any,
) -> ExactEvidence | None:
    source_ref = item.get("source_ref")
    ref = source_ref if isinstance(source_ref, Mapping) else item
    raw_store_id = ref.get("store_id")
    if isinstance(raw_store_id, bool) or raw_store_id is None:
        return None
    try:
        store_id = int(raw_store_id)
    except (TypeError, ValueError, OverflowError):
        return None
    row = engine._store.get(store_id)
    if row is None:
        return None
    content = str(row.get("content") or "")
    explicit_span = "span_start" in ref or "span_end" in ref
    quote_value = ref.get("quote") if explicit_span else item.get("content")
    quote = str(quote_value or "")
    if not quote:
        return None
    if len(quote) > MAX_EVIDENCE_CHARS:
        quote = quote[:MAX_EVIDENCE_CHARS]
    try:
        start = int(ref.get("span_start")) if explicit_span else int(item.get("content_offset", 0))
    except (TypeError, ValueError, OverflowError):
        return None
    end = start + len(quote)
    if explicit_span:
        try:
            declared_end = int(ref.get("span_end"))
        except (TypeError, ValueError, OverflowError):
            return None
        if declared_end - start < len(quote):
            return None
        end = start + len(quote)
    if start < 0 or end <= start or end > len(content) or content[start:end] != quote:
        if explicit_span:
            return None
        first = content.find(quote)
        if first < 0 or content.find(quote, first + 1) >= 0:
            return None
        start = first
        end = first + len(quote)
    assertion_id = str(item.get("assertion_id") or "").strip().casefold()
    if assertion_id and not _SHA256_RE.fullmatch(assertion_id):
        return None
    citation = f"lcm:{store_id}:{start}-{end}"
    public_stub = {
        "citation": citation,
        "store_id": store_id,
        "span_start": start,
        "span_end": end,
        "quote": quote,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "session_id": row.get("session_id") or "",
        "conversation_id": row.get("conversation_id") or "",
        "source": row.get("source") or "",
        "role": row.get("role") or "unknown",
        "timestamp": row.get("timestamp") or 0,
    }
    encoded_stub = _canonical_json(
        public_stub, field_name="evidence", max_chars=8_192
    )
    return ExactEvidence(
        citation=citation,
        store_id=store_id,
        span_start=start,
        span_end=end,
        quote=quote,
        content_sha256=str(public_stub["content_sha256"]),
        session_id=str(row.get("session_id") or ""),
        conversation_id=str(row.get("conversation_id") or ""),
        source=str(row.get("source") or ""),
        role=str(row.get("role") or "unknown"),
        timestamp=float(row.get("timestamp") or 0),
        assertion_id=assertion_id,
        context_tokens=count_tokens(encoded_stub),
        context_chars=len(encoded_stub),
    )


def _extract_exact_evidence(payload: Any, *, engine: Any) -> list[ExactEvidence]:
    found: dict[str, ExactEvidence] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            candidate = _candidate_from_item(value, engine=engine)
            if candidate is not None:
                found.setdefault(candidate.citation, candidate)
            for key, item in value.items():
                if key != "source_ref":
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return list(found.values())


def _extract_search_leads(
    payload: Any,
    *,
    exact_store_ids: set[int],
) -> list[SearchLead]:
    """Return bounded drill-down handles; leads can never close evidence slots."""
    found: dict[str, SearchLead] = {}
    locator_keys = (
        "node_id",
        "store_id",
        "session_id",
        "externalized_ref",
        "expand_hint",
        "next_cursor",
    )
    metadata_keys = (
        "kind",
        "role",
        "source",
        "timestamp",
        "content_offset",
        "source_offset",
        "snippet",
        "summary",
    )

    def bounded_value(value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, str):
            return value[:MAX_LEAD_FIELD_CHARS]
        return None

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            lead: dict[str, Any] = {}
            for key in locator_keys + metadata_keys:
                if key in value:
                    bounded = bounded_value(value[key])
                    if bounded not in (None, ""):
                        lead[key] = bounded
            expand_args = value.get("expand_args")
            if isinstance(expand_args, Mapping):
                bounded_args = {
                    str(key): bounded_value(item)
                    for key, item in expand_args.items()
                    if str(key) in _TOOL_ARGUMENT_KEYS["lcm_expand"]
                    and bounded_value(item) not in (None, "")
                }
                if bounded_args:
                    lead["expand_args"] = bounded_args
            has_locator = any(key in lead for key in locator_keys) or bool(
                lead.get("expand_args")
            )
            raw_store_id = lead.get("store_id")
            exact_only = (
                raw_store_id is not None
                and not any(
                    key in lead
                    for key in (
                        "node_id", "externalized_ref", "expand_hint", "next_cursor"
                    )
                )
                and not lead.get("expand_args")
            )
            if has_locator and not (
                exact_only
                and isinstance(raw_store_id, int)
                and raw_store_id in exact_store_ids
            ):
                canonical = _canonical_json(
                    lead, field_name="search_lead", max_chars=4_096
                )
                lead_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                found.setdefault(
                    lead_id,
                    SearchLead(
                        lead_id=lead_id,
                        payload=lead,
                        context_tokens=count_tokens(canonical),
                        context_chars=len(canonical),
                    ),
                )
            for key, item in value.items():
                if key not in {"source_ref", "expand_args"}:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return list(found.values())


def _bounded_tool_metadata(value: Any, *, depth: int = 0) -> Any:
    """Copy safe response provenance/metrics without forwarding unbounded data."""
    if depth > 4:
        return None
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda item: str(item))[:32]:
            key = str(raw_key)
            normalized = key.casefold().replace("-", "_")
            if normalized in {"answer", "final_answer", "response", "prose"} or any(
                marker in normalized
                for marker in (
                    "api_key",
                    "secret",
                    "credential",
                    "authorization",
                    "access_token",
                    "refresh_token",
                )
            ):
                continue
            bounded = _bounded_tool_metadata(value[raw_key], depth=depth + 1)
            if bounded is not None:
                result[key[:128]] = bounded
        return result
    if isinstance(value, (list, tuple)):
        result_list: list[Any] = []
        for item in value[:32]:
            bounded = _bounded_tool_metadata(item, depth=depth + 1)
            if bounded is not None:
                result_list.append(bounded)
        return result_list
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:MAX_LEAD_FIELD_CHARS]
    return None


def _tool_metadata_payload(value: Any) -> dict[str, Any]:
    bounded = _bounded_tool_metadata(value)
    if not isinstance(bounded, dict):
        return {}
    try:
        _canonical_json(bounded, field_name="tool_metadata", max_chars=2_048)
        return bounded
    except ValueError:
        compact = {
            key: bounded[key]
            for key in (
                "transport",
                "provider",
                "model",
                "coverage",
                "fallback",
                "degraded",
                "detail",
            )
            if key in bounded
        }
        compact["truncated"] = True
        try:
            _canonical_json(compact, field_name="tool_metadata", max_chars=2_048)
        except ValueError:
            return {"truncated": True}
        return compact


def _evidence_from_view_dependency(value: Mapping[str, Any]) -> ExactEvidence:
    citation = (
        f"lcm:{int(value['source_store_id'])}:"
        f"{int(value['span_start'])}-{int(value['span_end'])}"
    )
    quote = str(value["quote"])
    public_stub = {
        "citation": citation,
        "store_id": int(value["source_store_id"]),
        "span_start": int(value["span_start"]),
        "span_end": int(value["span_end"]),
        "quote": quote,
        "content_sha256": str(value["source_content_sha256"]),
        "session_id": str(value["source_session_id"]),
        "conversation_id": str(value["source_conversation_id"]),
        "source": str(value["source_name"]),
        "role": str(value["source_role"]),
        "timestamp": float(value["source_timestamp"]),
    }
    if value.get("assertion_id"):
        public_stub["assertion_id"] = str(value["assertion_id"])
    encoded_stub = _canonical_json(
        public_stub, field_name="evidence", max_chars=8_192
    )
    return ExactEvidence(
        citation=citation,
        store_id=int(value["source_store_id"]),
        span_start=int(value["span_start"]),
        span_end=int(value["span_end"]),
        quote=quote,
        content_sha256=str(value["source_content_sha256"]),
        session_id=str(value["source_session_id"]),
        conversation_id=str(value["source_conversation_id"]),
        source=str(value["source_name"]),
        role=str(value["source_role"]),
        timestamp=float(value["source_timestamp"]),
        assertion_id=str(value.get("assertion_id") or ""),
        context_tokens=count_tokens(encoded_stub),
        context_chars=len(encoded_stub),
    )


class AdaptiveRetrievalRegistry:
    """Thread-safe ephemeral controller state scoped to one engine/profile."""

    def __init__(self, query_views: QueryViewStore | None = None):
        self._query_views = query_views
        self._states: dict[str, RetrievalState] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _now() -> float:
        return time.time()

    def clear(self) -> None:
        with self._lock:
            self._states.clear()

    close = clear

    def _purge(self, now: float) -> None:
        expired = [
            key
            for key, state in self._states.items()
            if state.updated_at + RETRIEVAL_TTL_SECONDS <= now
        ]
        for key in expired:
            self._states.pop(key, None)
        if len(self._states) < MAX_ACTIVE_RETRIEVALS:
            return
        oldest = min(self._states.values(), key=lambda state: state.updated_at)
        self._states.pop(oldest.retrieval_id, None)

    @staticmethod
    def _requirements_payload(state: RetrievalState) -> list[dict[str, Any]]:
        return [
            requirement.public_dict(state.slot_refs.get(requirement.slot_id, ()))
            for requirement in state.requirements
        ]

    @staticmethod
    def _budget_payload(state: RetrievalState) -> dict[str, Any]:
        return {
            "retrieval_rounds": len(state.rounds),
            "retrieval_round_limit": MAX_RETRIEVAL_ROUNDS,
            "candidate_refs": len(state.candidates),
            "candidate_ref_limit": MAX_CANDIDATE_REFS,
            "search_leads": len(state.leads),
            "search_lead_limit": MAX_SEARCH_LEADS,
            "context_tokens": state.context_tokens,
            "context_token_limit": MAX_CONTEXT_TOKENS,
            "context_chars": state.context_chars,
            "context_char_limit": MAX_CONTEXT_CHARS,
            "exhausted": state.budget_exhausted,
        }

    def _state_payload(
        self,
        state: RetrievalState,
        *,
        evidence: Iterable[ExactEvidence] = (),
        leads: Iterable[SearchLead] = (),
    ) -> dict[str, Any]:
        return {
            "status": state.status,
            "retrieval_id": state.retrieval_id,
            "requirements": self._requirements_payload(state),
            "evidence": [item.public_dict() for item in evidence],
            "leads": [item.public_dict() for item in leads],
            "rounds": [item.public_dict() for item in state.rounds],
            "budgets": self._budget_payload(state),
            "termination_reason": state.termination_reason,
            "query_view": {
                "status": state.view_lookup_status,
                "delta_events": list(state.view_delta_events),
                "delta_truncated": state.view_delta_truncated,
                "cached_computation_trace": state.cached_trace,
            },
            "provenance": {
                "controller": {
                    "transport": "deterministic_local",
                    "provider": "none",
                    "model": "none",
                    "version": ADAPTIVE_RETRIEVAL_VERSION,
                }
            },
        }

    def _get(self, retrieval_id: Any, *, engine: Any) -> RetrievalState:
        key = str(retrieval_id or "").strip()
        state = self._states.get(key)
        if state is None:
            raise ValueError("unknown or expired retrieval_id")
        session_id = str(getattr(engine, "current_session_id", "") or "")
        conversation_id = str(getattr(engine, "current_conversation_id", "") or "")
        if state.owner_session_id and session_id and state.owner_session_id != session_id:
            raise ValueError("retrieval_id belongs to another active session")
        if (
            state.owner_conversation_id
            and conversation_id
            and state.owner_conversation_id != conversation_id
        ):
            raise ValueError("retrieval_id belongs to another conversation")
        if state.updated_at + RETRIEVAL_TTL_SECONDS <= self._now():
            self._states.pop(key, None)
            raise ValueError("retrieval_id expired")
        return state

    def start(
        self,
        *,
        question: Any,
        question_date: Any,
        identity: Any,
        requirements: Any,
        engine: Any,
    ) -> dict[str, Any]:
        text = " ".join(str(question or "").split())
        if not text or len(text) > MAX_QUESTION_CHARS:
            raise ValueError(
                f"question must contain 1..{MAX_QUESTION_CHARS} characters"
            )
        normalized_date = _normalize_question_date(question_date)
        parsed_requirements = _parse_requirements(requirements)
        digest = requirements_digest(parsed_requirements)
        parsed_identity = _parse_identity(
            identity,
            question_date=normalized_date,
            digest=digest,
        )
        owner_session_id = str(getattr(engine, "current_session_id", "") or "")
        owner_conversation_id = str(
            getattr(engine, "current_conversation_id", "") or ""
        )
        if (
            parsed_identity.conversation_id
            and owner_conversation_id
            and parsed_identity.conversation_id != owner_conversation_id
        ):
            raise ValueError("identity conversation_id does not match the active conversation")

        now = self._now()
        with self._lock:
            self._purge(now)
            state = RetrievalState(
                retrieval_id=uuid.uuid4().hex,
                question=text,
                question_date=normalized_date,
                identity=parsed_identity,
                requirements=parsed_requirements,
                owner_session_id=owner_session_id,
                owner_conversation_id=owner_conversation_id,
                created_at=now,
                updated_at=now,
                slot_refs={item.slot_id: [] for item in parsed_requirements},
                view_corpus_snapshot=(
                    self._query_views.corpus_snapshot()
                    if self._query_views is not None
                    else None
                ),
            )
            if self._query_views is not None:
                lookup = self._query_views.lookup(parsed_identity, record_hit=False)
                state.view_lookup_status = lookup.status
                state.view_delta_events = lookup.delta_events
                state.view_delta_truncated = lookup.delta_truncated
                if lookup.status == "hit" and lookup.view is not None:
                    manifest = lookup.view.get("manifest") or {}
                    coverage = manifest.get("coverage") or {}
                    raw_slot_refs = coverage.get("slot_refs") or {}
                    dependencies = lookup.view.get("dependencies") or []
                    candidates = {
                        item.citation: item
                        for item in (
                            _evidence_from_view_dependency(value)
                            for value in dependencies
                            if isinstance(value, Mapping)
                        )
                    }
                    candidate_tokens = sum(
                        item.context_tokens for item in candidates.values()
                    )
                    candidate_chars = sum(
                        item.context_chars for item in candidates.values()
                    )
                    valid = (
                        coverage.get("requirements_digest") == digest
                        and len(candidates) <= MAX_CANDIDATE_REFS
                        and candidate_tokens <= MAX_CONTEXT_TOKENS
                        and candidate_chars <= MAX_CONTEXT_CHARS
                    )
                    for requirement in parsed_requirements:
                        refs = raw_slot_refs.get(requirement.slot_id, [])
                        if not isinstance(refs, list) or any(
                            str(ref) not in candidates for ref in refs
                        ):
                            valid = False
                            break
                        state.slot_refs[requirement.slot_id] = [str(ref) for ref in refs]
                        if len(state.slot_refs[requirement.slot_id]) < requirement.minimum_refs:
                            valid = False
                            break
                    if valid:
                        confirmed = self._query_views.lookup(
                            parsed_identity, record_hit=True
                        )
                        if confirmed.status == "hit":
                            state.candidates = candidates
                            state.context_tokens = candidate_tokens
                            state.context_chars = candidate_chars
                            state.status = "ready"
                            state.view_lookup_status = "hit"
                            state.cached_trace = lookup.view.get(
                                "computation_trace"
                            )
                        else:
                            valid = False
                            state.view_lookup_status = confirmed.status
                    if not valid:
                        state.slot_refs = {
                            item.slot_id: [] for item in parsed_requirements
                        }
                        if (
                            candidate_tokens > MAX_CONTEXT_TOKENS
                            or candidate_chars > MAX_CONTEXT_CHARS
                            or len(candidates) > MAX_CANDIDATE_REFS
                        ):
                            state.view_lookup_status = "miss"
                            state.termination_reason = (
                                "cached query view exceeds current controller bounds"
                            )
                elif lookup.status == "delta_required":
                    state.termination_reason = "query view requires bounded delta retrieval"
            self._states[state.retrieval_id] = state
            return self._state_payload(
                state,
                evidence=(state.candidates.values() if state.status == "ready" else ()),
            )

    @staticmethod
    def _apply_resolutions(state: RetrievalState, raw: Any) -> None:
        if raw in (None, []):
            return
        if not isinstance(raw, list) or len(raw) > MAX_REQUIREMENTS:
            raise ValueError("resolved_slots must be a bounded array")
        requirements = {item.slot_id: item for item in state.requirements}
        for item in raw:
            if not isinstance(item, Mapping) or set(item) - {"slot_id", "evidence_refs"}:
                raise ValueError(
                    "resolved_slots items require only slot_id and evidence_refs"
                )
            slot_id = str(item.get("slot_id") or "").strip().casefold()
            if slot_id not in requirements:
                raise ValueError(f"unknown evidence slot: {slot_id}")
            refs = item.get("evidence_refs")
            if not isinstance(refs, list) or not refs or len(refs) > MAX_CANDIDATE_REFS:
                raise ValueError("resolved slot evidence_refs must be a non-empty array")
            target = state.slot_refs[slot_id]
            for raw_ref in refs:
                citation = str(raw_ref or "").strip()
                if citation not in state.candidates:
                    raise ValueError(
                        f"resolved slot references unobserved evidence: {citation}"
                    )
                if citation not in target:
                    target.append(citation)

    @staticmethod
    def _open_slot_ids(state: RetrievalState) -> list[str]:
        return [
            item.slot_id
            for item in state.requirements
            if len(state.slot_refs[item.slot_id]) < item.minimum_refs
        ]

    def search(
        self,
        *,
        retrieval_id: Any,
        missing_slot: Any,
        tool: Any,
        tool_args: Any,
        resolved_slots: Any,
        engine: Any,
        dispatch: Dispatch,
    ) -> dict[str, Any]:
        with self._lock:
            state = self._get(retrieval_id, engine=engine)
            if state.status != "active":
                raise ValueError(f"retrieval is not active: {state.status}")
            self._apply_resolutions(state, resolved_slots)
            slot_id = str(missing_slot or "").strip().casefold()
            if slot_id not in self._open_slot_ids(state):
                raise ValueError("missing_slot must name a currently open requirement")
            if state.budget_exhausted or len(state.rounds) >= MAX_RETRIEVAL_ROUNDS:
                state.budget_exhausted = True
                state.termination_reason = "retrieval round budget exhausted"
                raise ValueError(state.termination_reason)
            tool_name = str(tool or "").strip()
            bounded_args = _bounded_tool_args(tool_name, tool_args)

            started = time.perf_counter()
            try:
                raw_result = dispatch(tool_name, bounded_args)
                payload = json.loads(raw_result)
                if not isinstance(payload, Mapping):
                    raise ValueError("retrieval tool returned non-object JSON")
                tool_error = str(payload.get("error") or "")
                tool_provenance = _tool_metadata_payload(
                    payload.get("provenance") or {}
                )
                tool_metrics = _tool_metadata_payload(payload.get("metrics") or {})
                extracted = [] if tool_error else _extract_exact_evidence(payload, engine=engine)
                extracted_leads = (
                    []
                    if tool_error
                    else _extract_search_leads(
                        payload,
                        exact_store_ids={item.store_id for item in extracted},
                    )
                )
            except Exception as exc:
                payload = {}
                tool_error = str(exc)
                extracted = []
                extracted_leads = []
                tool_provenance = {}
                tool_metrics = {}
            latency_ms = round((time.perf_counter() - started) * 1_000.0, 3)

            new_items: list[ExactEvidence] = []
            result_truncated = False
            for candidate in extracted:
                if candidate.citation in state.candidates:
                    continue
                if len(state.candidates) >= MAX_CANDIDATE_REFS:
                    result_truncated = True
                    state.budget_exhausted = True
                    break
                if state.context_tokens + candidate.context_tokens > MAX_CONTEXT_TOKENS:
                    result_truncated = True
                    state.budget_exhausted = True
                    break
                if state.context_chars + candidate.context_chars > MAX_CONTEXT_CHARS:
                    result_truncated = True
                    state.budget_exhausted = True
                    break
                state.candidates[candidate.citation] = candidate
                state.context_tokens += candidate.context_tokens
                state.context_chars += candidate.context_chars
                new_items.append(candidate)

            new_leads: list[SearchLead] = []
            for lead in extracted_leads:
                if lead.lead_id in state.leads:
                    continue
                if len(state.leads) >= MAX_SEARCH_LEADS:
                    result_truncated = True
                    state.budget_exhausted = True
                    break
                if state.context_tokens + lead.context_tokens > MAX_CONTEXT_TOKENS:
                    result_truncated = True
                    state.budget_exhausted = True
                    break
                if state.context_chars + lead.context_chars > MAX_CONTEXT_CHARS:
                    result_truncated = True
                    state.budget_exhausted = True
                    break
                state.leads[lead.lead_id] = lead
                state.context_tokens += lead.context_tokens
                state.context_chars += lead.context_chars
                new_leads.append(lead)

            if new_items:
                round_status = "exact_evidence"
            elif new_leads:
                round_status = "lead_progress"
            else:
                round_status = "no_progress"
            state.rounds.append(
                RetrievalRound(
                    round_number=len(state.rounds) + 1,
                    missing_slot=slot_id,
                    tool=tool_name,
                    tool_args=bounded_args,
                    new_refs=tuple(item.citation for item in new_items),
                    new_leads=tuple(item.lead_id for item in new_leads),
                    latency_ms=latency_ms,
                    status=round_status,
                    tool_provenance=tool_provenance,
                    tool_metrics=tool_metrics,
                    result_truncated=result_truncated,
                )
            )
            state.updated_at = self._now()
            if not new_items and not new_leads:
                state.status = "fallback"
                state.termination_reason = (
                    f"retrieval made no exact-evidence progress: {tool_error}"
                    if tool_error
                    else "retrieval made no exact-evidence progress"
                )
            elif len(state.rounds) >= MAX_RETRIEVAL_ROUNDS:
                state.budget_exhausted = True
            return self._state_payload(state, evidence=new_items, leads=new_leads)

    @staticmethod
    def _operand_citation(operand: Any, state: RetrievalState) -> str:
        if not isinstance(operand, Mapping):
            raise ValueError("each computation operand must be an object")
        assertion_id = str(operand.get("assertion_id") or "").strip().casefold()
        if assertion_id:
            matches = [
                item.citation
                for item in state.candidates.values()
                if item.assertion_id == assertion_id
            ]
            if len(matches) != 1:
                raise ValueError("computation assertion_id is not one observed exact ref")
            return matches[0]
        try:
            return (
                f"lcm:{int(operand['store_id'])}:"
                f"{int(operand['span_start'])}-{int(operand['span_end'])}"
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("raw computation operands require an exact LCM ref") from exc

    def _persist_view(
        self,
        state: RetrievalState,
        selected: Sequence[ExactEvidence],
        *,
        compute_args: Mapping[str, Any] | None,
        computation: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if self._query_views is None:
            return {"status": "disabled"}
        if state.view_lookup_status == "hit":
            return {"status": "reused"}
        if compute_args is not None and (
            computation is None or computation.get("status") != "computed"
        ):
            return {"status": "skipped", "reason": "computation did not validate"}
        token = None
        try:
            token = self._query_views.claim_build(
                state.identity,
                corpus_snapshot=state.view_corpus_snapshot,
            )
            dependencies = [
                self._query_views.snapshot_dependency(
                    item.store_id,
                    item.span_start,
                    item.span_end,
                    item.quote,
                    assertion_id=item.assertion_id,
                )
                for item in selected
            ]
            trace = None
            if computation is not None and isinstance(computation.get("trace"), Mapping):
                raw_trace = computation["trace"]
                trace = {
                    key: raw_trace[key]
                    for key in (
                        "operation", "result", "result_value", "unit", "citations",
                        "entities", "evidence_dates", "steps",
                    )
                    if key in raw_trace
                }
                raw_trust = computation.get("temporal_trust")
                if isinstance(raw_trust, Mapping):
                    raw_notes = raw_trust.get("notes")
                    trace["temporal_trust"] = temporal_trust_wire(
                        raw_trust.get("status"),
                        raw_trust.get("certified"),
                        raw_notes if isinstance(raw_notes, list) else (),
                    )
            selected_refs = {item.citation for item in selected}
            manifest = {
                "closed_slots": sorted(state.slot_refs),
                "open_slots": [],
                "operands": list(compute_args.get("operands", [])) if compute_args else [],
                "retrieval_calls": [item.public_dict() for item in state.rounds],
                "evidence_refs": [item.citation for item in selected],
                "coverage": {
                    "complete": True,
                    "requirements_digest": state.identity.requirements_digest,
                    "slot_refs": {
                        key: [
                            ref for ref in value if ref in selected_refs
                        ]
                        for key, value in sorted(state.slot_refs.items())
                    },
                    "retrieval_rounds": len(state.rounds),
                    "candidate_refs": len(state.candidates),
                    "context_tokens": state.context_tokens,
                    "context_chars": state.context_chars,
                },
            }
            published = self._query_views.publish_ready(
                token,
                dependencies=dependencies,
                manifest=manifest,
                computation_trace=trace,
                completeness="complete",
                search_policy_version=ADAPTIVE_RETRIEVAL_VERSION,
            )
            return {"status": "published" if published else "stale_before_publish"}
        except QueryViewBuildInProgressError:
            return {"status": "building_elsewhere"}
        except Exception as exc:
            if token is not None:
                try:
                    self._query_views.mark_failed(token, str(exc))
                except Exception:
                    # Cleanup is best-effort and must never replace the build
                    # failure that selected this response contract.
                    pass
            return {"status": "failed", "reason": str(exc)[:500]}

    def finish(
        self,
        *,
        retrieval_id: Any,
        resolved_slots: Any,
        selected_refs: Any,
        computation: Any,
        engine: Any,
        dispatch: Dispatch,
    ) -> dict[str, Any]:
        with self._lock:
            state = self._get(retrieval_id, engine=engine)
            if state.status in {"finished", "abandoned"}:
                raise ValueError(f"retrieval is already {state.status}")
            self._apply_resolutions(state, resolved_slots)
            open_slots = self._open_slot_ids(state)
            if open_slots:
                state.status = "fallback"
                state.termination_reason = (
                    "required evidence slots remain open: " + ", ".join(open_slots)
                )
                state.updated_at = self._now()
                return self._state_payload(state)

            if selected_refs in (None, []):
                refs = list(
                    dict.fromkeys(
                        citation
                        for requirement in state.requirements
                        for citation in state.slot_refs[requirement.slot_id]
                    )
                )
            elif isinstance(selected_refs, list):
                refs = list(dict.fromkeys(str(value or "").strip() for value in selected_refs))
            else:
                raise ValueError("selected_refs must be an array")
            if not 1 <= len(refs) <= MAX_CANDIDATE_REFS:
                raise ValueError("selected_refs must contain between 1 and 40 exact refs")
            if any(ref not in state.candidates for ref in refs):
                raise ValueError("selected_refs contains unobserved evidence")
            selected_set = set(refs)
            for requirement in state.requirements:
                if len(selected_set.intersection(state.slot_refs[requirement.slot_id])) < requirement.minimum_refs:
                    raise ValueError(
                        f"selected_refs does not satisfy slot {requirement.slot_id}"
                    )
            selected = [state.candidates[ref] for ref in refs]

            compute_args: dict[str, Any] | None = None
            compute_result: dict[str, Any] | None = None
            if computation not in (None, {}):
                if not isinstance(computation, Mapping):
                    raise ValueError("computation must be an object")
                _validate_argument_keys(computation)
                unknown = set(computation) - {"operands", "candidate_answer"}
                if unknown:
                    raise ValueError(
                        "computation contains unsupported fields: "
                        + ", ".join(sorted(str(item) for item in unknown))
                    )
                operands = computation.get("operands")
                if not isinstance(operands, list):
                    raise ValueError("computation operands must be an array")
                for operand in operands:
                    citation = self._operand_citation(operand, state)
                    if citation not in selected_set:
                        raise ValueError(
                            "computation operand is not one of the selected exact refs"
                        )
                    candidate = state.candidates[citation]
                    if str(operand.get("quote") or "") != candidate.quote:
                        raise ValueError("computation operand quote changed after retrieval")
                compute_args = {
                    "question": state.question,
                    "question_date": state.question_date or None,
                    "evidence_complete": True,
                    "operands": operands,
                }
                if "candidate_answer" in computation:
                    compute_args["candidate_answer"] = computation["candidate_answer"]
                raw_compute = dispatch("lcm_compute", compute_args)
                parsed_compute = json.loads(raw_compute)
                if not isinstance(parsed_compute, Mapping):
                    raise ValueError("lcm_compute returned non-object JSON")
                compute_result = dict(parsed_compute)

            view_result = self._persist_view(
                state,
                selected,
                compute_args=compute_args,
                computation=compute_result,
            )
            compute_failed = (
                compute_args is not None
                and (
                    compute_result is None
                    or compute_result.get("status") != "computed"
                )
            )
            if compute_failed:
                state.status = "fallback"
                state.termination_reason = str(
                    compute_result.get("reason")
                    if compute_result is not None
                    else "computation did not return a validated result"
                )[:1_000]
            else:
                state.status = "finished"
                state.termination_reason = ""
            state.updated_at = self._now()
            payload = self._state_payload(state, evidence=selected)
            payload["computation"] = compute_result
            payload["query_view"]["persistence"] = view_result
            if compute_failed:
                payload["next_path"] = "evidence_only"
            return payload

    def status(self, *, retrieval_id: Any, engine: Any) -> dict[str, Any]:
        with self._lock:
            state = self._get(retrieval_id, engine=engine)
            state.updated_at = self._now()
            return self._state_payload(
                state,
                evidence=state.candidates.values(),
                leads=state.leads.values(),
            )

    def abandon(self, *, retrieval_id: Any, engine: Any) -> dict[str, Any]:
        with self._lock:
            state = self._get(retrieval_id, engine=engine)
            state.status = "abandoned"
            state.termination_reason = "abandoned by answerer"
            state.updated_at = self._now()
            payload = self._state_payload(state)
            self._states.pop(state.retrieval_id, None)
            return payload
