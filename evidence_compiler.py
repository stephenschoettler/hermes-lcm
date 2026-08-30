"""Bounded, source-grounded evidence compilation for V4.6.

The compiler is product-owned and host-neutral.  A semantic selector may propose
facets and exact evidence, but immutable product code validates every proposal
against the message store.  The compiler returns structured evidence and an
optional canonical computation trace; it never returns final prose.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import time
from typing import Any, Callable, Mapping, Sequence

from .evidence_pack import build_evidence_pack, normalize_question_date
from .query_view_store import QueryViewBuildInProgressError, QueryViewIdentity
from .reasoning import compile_evidence_plan, normalize_unit


EVIDENCE_COMPILER_VERSION = "evidence-compiler-v1"
SELECTOR_SCHEMA_VERSION = "evidence-selector-v1"
_EXACT_REF_RE = re.compile(r"^lcm:(?P<store_id>[1-9]\d*):(?P<start>\d+)-(?P<end>\d+)$")
_FACET_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CLAIM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_NUMBER_RE = re.compile(r"(?<![\w.])-?(?:\d+(?:,\d{3})*|\d*\.\d+)(?!\w)")
_ALLOWED_OPERATIONS = {
    "none",
    "scalar_fact",
    "date_interval",
    "date_filter",
    "count_distinct",
    "sum",
    "difference",
    "order",
    "latest_fact",
}
_PROPOSAL_KEYS = {
    "version",
    "requested_facets",
    "selections",
    "missing_facets",
    "usage",
}
_SELECTION_KEYS = {
    "claim_id",
    "facet",
    "exact_ref",
    "quote",
    "entity",
    "date",
    "value",
    "unit",
    "distinct_key",
    "label",
    "role",
    "source",
}
_USAGE_KEYS = {
    "provider",
    "model",
    "effort",
    "calls",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "cost_usd",
}
_SECRET_RE = re.compile(
    r"(?:Bearer\s+[A-Za-z0-9._~-]{8,}|\b(?:sk|pa)-[A-Za-z0-9_-]{8,}|"
    r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+)",
    re.IGNORECASE,
)
_MAX_PROPOSED_FACETS = 12


def _enumerated_operand_count(question: str) -> int | None:
    """Return a code-proven small operand count from explicit named lists."""
    normalized = " ".join(str(question or "").split())
    quoted = re.findall(r"['\"]([^'\"]{1,200})['\"]", normalized)
    if len(quoted) >= 2:
        return min(8, len(quoted))
    match = re.search(
        r"\b(?:total|combined|altogether|sum)\b.*?\bof\b\s+(.+?)"
        r"(?=\b(?:i|we)\s+(?:have|had|did|do|spent|used|earned|made)\b|\?)",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"\bhow many\s+(?:hours?|minutes?|days?|weeks?|months?)\s+of\s+"
            r"(.+?)(?=\b(?:did|do|was|were|have|has)\b|\?)",
            normalized,
            re.IGNORECASE,
        )
    if not match:
        return None
    phrase = match.group(1)
    if not re.search(r"\b(?:and|plus)\b", phrase, re.IGNORECASE):
        return None
    parts = [
        item.strip(" ,")
        for item in re.split(r"\s*(?:,|\band\b|\bplus\b)\s*", phrase, flags=re.IGNORECASE)
        if item.strip(" ,")
    ]
    return len(parts) if 2 <= len(parts) <= 8 else None


def _is_scalar_fact_question(question: str) -> bool:
    normalized = " ".join(str(question or "").casefold().split())
    return bool(
        re.search(
            r"\bhow (?:many|much)\b.*\b(?:do|did|will|would|can)\s+i\s+"
            r"(?:need|lead|view|earn|redeem|manage|owe|have left)\b",
            normalized,
        )
        or re.search(
            r"\bwhat (?:was|is)\b.*\b(?:increase|decrease|change|balance|count|number)\b",
            normalized,
        )
    )


@dataclass(frozen=True)
class CompilerBudgets:
    max_input_refs: int = 40
    max_selections: int = 40
    max_selector_chars: int = 8_192
    max_quote_chars: int = 2_400
    max_retrieval_calls: int = 1
    max_novel_refs: int = 8
    response_char_cap: int = 64_000

    def as_dict(self) -> dict[str, int]:
        return {
            "max_input_refs": self.max_input_refs,
            "max_selections": self.max_selections,
            "max_selector_chars": self.max_selector_chars,
            "max_quote_chars": self.max_quote_chars,
            "max_retrieval_calls": self.max_retrieval_calls,
            "max_novel_refs": self.max_novel_refs,
            "response_char_cap": self.response_char_cap,
        }


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return min(maximum, max(minimum, parsed))


def _budgets(value: Any) -> CompilerBudgets:
    raw = value if isinstance(value, Mapping) else {}
    return CompilerBudgets(
        max_input_refs=_bounded_int(
            raw.get("max_input_refs"), default=40, minimum=1, maximum=50
        ),
        max_selections=_bounded_int(
            raw.get("max_selections"), default=40, minimum=1, maximum=50
        ),
        max_selector_chars=_bounded_int(
            raw.get("max_selector_chars"), default=8_192, minimum=1_024, maximum=64_000
        ),
        max_quote_chars=_bounded_int(
            raw.get("max_quote_chars"), default=2_400, minimum=64, maximum=2_400
        ),
        max_retrieval_calls=_bounded_int(
            raw.get("max_retrieval_calls"), default=1, minimum=0, maximum=2
        ),
        max_novel_refs=_bounded_int(
            raw.get("max_novel_refs"), default=8, minimum=1, maximum=8
        ),
        response_char_cap=_bounded_int(
            raw.get("response_char_cap"), default=64_000, minimum=4_096, maximum=64_000
        ),
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _word_number(value: str) -> int | None:
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    if value.isdigit():
        return int(value)
    return words.get(value)


def _explicit_cardinality(question: str) -> int | None:
    match = re.search(
        r"\b(?:the|these|those|of\s+the)\s+"
        r"(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?:events?|items?|vacations?|holidays?|trips?|costs?|values?|"
        r"invoices?|decisions?|steps?|people|persons?)\b",
        question.casefold(),
    )
    return _word_number(match.group(1)) if match else None


def _facet(name: str, *, required: bool = True) -> dict[str, Any]:
    return {"name": name, "required": required}


def derive_evidence_request(question: Any, question_date: Any = None) -> dict[str, Any]:
    """Derive a bounded generic request without benchmark metadata."""
    text = str(question or "").strip()
    if not text or len(text) > 4_000:
        raise ValueError("question must contain between 1 and 4000 characters")
    normalized_date, date_error = normalize_question_date(question_date)
    if date_error:
        raise ValueError(date_error)
    as_of = normalized_date.day.isoformat() if normalized_date else None
    plan_decision = compile_evidence_plan(text, as_of)
    plan = plan_decision.plan if plan_decision.status == "planned" else None
    operation = plan.operation if plan is not None else "none"
    normalized = text.casefold()

    enumerated = _enumerated_operand_count(text)
    if (
        operation == "count_distinct"
        and enumerated is not None
        and re.search(r"\bhow many\s+(?:hours?|minutes?|days?|weeks?|months?)\s+of\b", normalized)
    ):
        operation = "sum"
    elif operation == "count_distinct" and _is_scalar_fact_question(text):
        operation = "scalar_fact"

    facets: list[dict[str, Any]] = []

    def add(name: str) -> None:
        if name not in {item["name"] for item in facets}:
            facets.append(_facet(name))

    if re.search(
        r"\b(decid(?:e|ed|ing)|decision|approved?|chosen?|choose)\b", normalized
    ):
        add("decision")
    if re.search(r"\b(why|rationale|reason|because)\b", normalized):
        add("rationale")
    if re.search(r"\b(owner|owns?|responsible|accountable)\b", normalized):
        add("owner")
    if re.search(r"\b(deadline|due|due date|when is .* due)\b", normalized):
        add("deadline")
    if re.search(
        r"\b(how do we|how should we|workflow|procedure|runbook|steps?)\b", normalized
    ):
        add("procedure")
    if re.search(
        r"\b(gotcha|failure|fails?|failed|remedy|workaround|pitfall)\b", normalized
    ):
        add("gotcha")
    if re.search(
        r"\b(premise|available|availability|has no|does not have|does .* have)\b",
        normalized,
    ):
        add("premise")

    if operation == "scalar_fact":
        add("value")
    elif operation == "count_distinct":
        add("events")
    elif operation in {"sum", "difference", "date_interval", "order"}:
        add("operands")
    elif operation == "latest_fact" or re.search(
        r"\b(current|currently|latest|now|most recent)\b", normalized
    ):
        add("current_state")
    elif operation == "date_filter" or re.search(
        r"\b(happened|ago|yesterday|event)\b", normalized
    ):
        add("event")
    elif re.search(r"\b(prefer|preference|policy)\b", normalized):
        add("preference")
    if not facets:
        add("answer")

    expected = (
        plan.exact_operands
        if plan is not None and plan.exact_operands is not None
        else _explicit_cardinality(text)
    )
    if operation == "sum" and expected is None:
        expected = enumerated
    if operation == "scalar_fact":
        expected = 1
    exhaustive = bool(
        operation in {"count_distinct", "scalar_fact"}
        or re.search(r"\b(all|every|complete list|exhaustive)\b", normalized)
        or expected is not None
        and operation
        in {"count_distinct", "scalar_fact", "sum", "difference", "order", "date_interval"}
    )
    return {
        "version": EVIDENCE_COMPILER_VERSION,
        "question": text,
        "as_of": as_of,
        "facets": facets,
        "facet_source": "deterministic",
        "deterministic_facets": [item["name"] for item in facets],
        "operation": operation,
        "exhaustive": exhaustive,
        "expected_cardinality": expected,
        "temporal_window": (
            plan.temporal_window.as_dict()
            if plan is not None and plan.temporal_window is not None
            else None
        ),
    }


def _normalize_baseline_refs(
    refs: Sequence[Any], *, limit: int
) -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    exact_refs: list[str] = []
    for raw in list(refs)[:limit]:
        if isinstance(raw, str):
            item = {"exact_ref": raw.strip()}
        elif isinstance(raw, Mapping):
            item = dict(raw)
            item["exact_ref"] = str(item.get("exact_ref") or "").strip()
        else:
            continue
        exact_ref = item["exact_ref"]
        if not _EXACT_REF_RE.fullmatch(exact_ref) or exact_ref in exact_refs:
            continue
        exact_refs.append(exact_ref)
        normalized.append(item)
    return normalized, exact_refs


def _usage(value: Any) -> tuple[dict[str, Any], str | None]:
    if value in (None, {}):
        return {}, None
    if not isinstance(value, Mapping) or set(value) - _USAGE_KEYS:
        return {}, "selector usage is invalid"
    result: dict[str, Any] = {}
    for key in ("provider", "model", "effort"):
        if key in value:
            text = str(value[key] or "").strip()
            if not text or len(text) > 200:
                return {}, f"selector usage {key} is invalid"
            result[key] = text
    for key in ("calls", "input_tokens", "output_tokens"):
        if key in value:
            if isinstance(value[key], bool):
                return {}, f"selector usage {key} is invalid"
            try:
                number = int(value[key])
            except (TypeError, ValueError, OverflowError):
                return {}, f"selector usage {key} is invalid"
            if number < 0:
                return {}, f"selector usage {key} is invalid"
            result[key] = number
    for key in ("latency_ms", "cost_usd"):
        if key in value:
            try:
                number = float(value[key])
            except (TypeError, ValueError, OverflowError):
                return {}, f"selector usage {key} is invalid"
            if not math.isfinite(number) or number < 0:
                return {}, f"selector usage {key} is invalid"
            result[key] = number
    return result, None


def _validate_proposal(
    raw: Any,
    *,
    request: Mapping[str, Any],
    limits: CompilerBudgets,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    try:
        encoded = _canonical_json(raw)
    except (TypeError, ValueError):
        return None, "selector_schema_invalid", None
    proposal_digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if len(encoded) > limits.max_selector_chars:
        return None, "selector_budget_exhausted", proposal_digest
    if _SECRET_RE.search(encoded):
        return None, "selector_secret_rejected", proposal_digest
    if not isinstance(raw, Mapping) or set(raw) - _PROPOSAL_KEYS:
        return None, "selector_schema_invalid", proposal_digest
    if raw.get("version") != SELECTOR_SCHEMA_VERSION:
        return None, "selector_schema_invalid", proposal_digest
    operation = str(request.get("operation") or "")
    if operation not in _ALLOWED_OPERATIONS:
        return None, "selector_schema_invalid", proposal_digest
    selections = raw.get("selections")
    missing = raw.get("missing_facets")
    proposed_facets = raw.get("requested_facets", [])
    if not isinstance(selections, list) or len(selections) > limits.max_selections:
        return None, "selector_schema_invalid", proposal_digest
    if (
        not isinstance(proposed_facets, list)
        or len(proposed_facets) > _MAX_PROPOSED_FACETS
    ):
        return None, "selector_schema_invalid", proposal_digest
    normalized_proposed: list[str] = []
    for item in proposed_facets:
        name = str(item or "").strip().casefold()
        if (
            not _FACET_RE.fullmatch(name)
            or name == "answer"
            or name in normalized_proposed
        ):
            return None, "selector_schema_invalid", proposal_digest
        normalized_proposed.append(name)
    deterministic = [str(item["name"]) for item in request["facets"]]
    if normalized_proposed and deterministic == ["answer"]:
        effective = list(normalized_proposed)
        facet_source = "semantic_proposal"
    else:
        effective = list(deterministic)
        for name in normalized_proposed:
            if name not in effective:
                effective.append(name)
        facet_source = (
            "deterministic_plus_semantic" if normalized_proposed else "deterministic"
        )
    if not isinstance(missing, list) or len(missing) > len(effective):
        return None, "selector_schema_invalid", proposal_digest
    requested = set(effective)
    normalized_missing: list[str] = []
    for item in missing:
        name = str(item or "").strip().casefold()
        if not _FACET_RE.fullmatch(name) or name not in requested:
            return None, "selector_schema_invalid", proposal_digest
        if name not in normalized_missing:
            normalized_missing.append(name)
    claim_ids: set[str] = set()
    normalized_selections: list[dict[str, Any]] = []
    for selection in selections:
        if not isinstance(selection, Mapping) or set(selection) - _SELECTION_KEYS:
            return None, "selector_schema_invalid", proposal_digest
        claim_id = str(selection.get("claim_id") or "").strip()
        facet = str(selection.get("facet") or "").strip().casefold()
        exact_ref = str(selection.get("exact_ref") or "").strip()
        quote = str(selection.get("quote") or "")
        if (
            not _CLAIM_RE.fullmatch(claim_id)
            or claim_id in claim_ids
            or not _FACET_RE.fullmatch(facet)
            or facet not in requested
            or not _EXACT_REF_RE.fullmatch(exact_ref)
            or not quote
            or len(quote) > limits.max_quote_chars
        ):
            return None, "selector_schema_invalid", proposal_digest
        claim_ids.add(claim_id)
        normalized = dict(selection)
        normalized.update(
            {
                "claim_id": claim_id,
                "facet": facet,
                "exact_ref": exact_ref,
                "quote": quote,
            }
        )
        normalized_selections.append(normalized)
    normalized_usage, usage_error = _usage(raw.get("usage"))
    if usage_error:
        return None, "selector_schema_invalid", proposal_digest
    return (
        {
            "version": SELECTOR_SCHEMA_VERSION,
            "operation": operation,
            "requested_facets": normalized_proposed,
            "effective_facets": [_facet(name) for name in effective],
            "facet_source": facet_source,
            "selections": normalized_selections,
            "missing_facets": normalized_missing,
            "usage": normalized_usage,
        },
        None,
        proposal_digest,
    )


def _normalized_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[\w]+", value.casefold(), re.UNICODE))


def _contains_text(quote: str, value: Any) -> bool:
    text = " ".join(str(value or "").split()).casefold()
    return bool(text) and text in " ".join(quote.split()).casefold()


def _contains_key(quote: str, value: Any) -> bool:
    quote_tokens = set(_normalized_tokens(quote))
    key_tokens = _normalized_tokens(str(value or ""))
    return bool(key_tokens) and all(token in quote_tokens for token in key_tokens)


def _contains_numeric_value(quote: str, value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    if not math.isfinite(float(value)):
        return False
    numbers = []
    for match in _NUMBER_RE.finditer(quote):
        try:
            numbers.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue
    return float(value) in numbers


def _contains_unit(quote: str, value: Any) -> bool:
    unit = normalize_unit(value)
    if not unit:
        return False
    normalized = quote.casefold()
    if unit == "usd":
        return "$" in quote or bool(re.search(r"\b(?:usd|dollars?)\b", normalized))
    aliases = {
        "hour": r"\b(?:hours?|hrs?)\b",
        "minute": r"\b(?:minutes?|mins?)\b",
        "day": r"\bdays?\b",
        "week": r"\bweeks?\b",
        "month": r"\bmonths?\b",
        "page": r"\bpages?\b",
        "item": r"\bitems?\b",
        "event": r"\bevents?\b",
    }
    return bool(re.search(aliases.get(unit, rf"\b{re.escape(unit)}s?\b"), normalized))


def _candidate(selection: Mapping[str, Any]) -> dict[str, Any]:
    candidate = {
        "exact_ref": selection["exact_ref"],
        "quote": selection["quote"],
    }
    for source, target in (
        ("value", "value"),
        ("unit", "unit"),
        ("distinct_key", "key"),
        ("label", "label"),
    ):
        if selection.get(source) is not None:
            candidate[target] = selection[source]
    return candidate


def _validate_claim(
    selection: Mapping[str, Any],
    *,
    question: str,
    question_date: str | None,
    engine: Any,
    allowed_refs: set[str],
    limits: CompilerBudgets,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    exact_ref = str(selection["exact_ref"])
    quote = str(selection["quote"])
    if exact_ref not in allowed_refs:
        errors.append("exact_ref_not_in_bounded_evidence")
    if len(quote) > limits.max_quote_chars:
        errors.append("quote_budget_exceeded")
    entity = selection.get("entity")
    if entity is not None and not _contains_text(quote, entity):
        errors.append("entity_not_in_exact_quote")
    value = selection.get("value")
    if value is not None:
        if isinstance(value, (dict, list, tuple, set, bool)):
            errors.append("value_type_invalid")
        elif isinstance(value, (int, float)):
            if not _contains_numeric_value(quote, value):
                errors.append("value_not_in_exact_quote")
        elif not _contains_text(quote, value):
            errors.append("value_not_in_exact_quote")
    if selection.get("unit") is not None and not _contains_unit(
        quote, selection["unit"]
    ):
        errors.append("unit_not_in_exact_quote")
    if selection.get("distinct_key") is not None and not _contains_key(
        quote, selection["distinct_key"]
    ):
        errors.append("distinct_key_not_in_exact_quote")
    if selection.get("label") is not None and not _contains_text(
        quote, selection["label"]
    ):
        errors.append("label_not_in_exact_quote")
    if errors:
        return None, errors

    raw = build_evidence_pack(
        {
            "question": question,
            "question_date": question_date,
            "baseline_refs": [_candidate(selection)],
            "budgets": {
                "max_refs": 1,
                "max_quote_chars": limits.max_quote_chars,
                "max_retrieval_calls": 0,
            },
        },
        engine=engine,
    )
    payload = json.loads(raw)
    evidence = payload.get("evidence") if isinstance(payload, Mapping) else None
    if not isinstance(evidence, list) or not evidence:
        reasons = payload.get("rejections") if isinstance(payload, Mapping) else None
        if isinstance(reasons, list):
            errors.extend(
                str(item.get("reason_code") or "claim_grounding_failed")
                for item in reasons
                if isinstance(item, Mapping)
            )
        return None, errors or ["claim_grounding_failed"]
    item = dict(evidence[0])
    if selection.get("role") is not None and str(item.get("role")) != str(
        selection["role"]
    ):
        errors.append("role_not_grounded")
    if selection.get("source") is not None and str(item.get("source")) != str(
        selection["source"]
    ):
        errors.append("source_not_grounded")
    proposed_date = str(selection.get("date") or "").strip()
    if proposed_date:
        occurrence = item.get("occurrence_time")
        observation = item.get("observation_time")
        grounded_dates = {
            str(occurrence.get("event_date") or "")
            if isinstance(occurrence, Mapping)
            else "",
            str(observation.get("date") or "")
            if isinstance(observation, Mapping)
            else "",
        }
        if proposed_date not in grounded_dates:
            errors.append("date_not_grounded")
    if errors:
        return None, errors
    facets = dict(item.get("facets") or {})
    if entity is not None:
        facets["entity"] = str(entity)
    item.update(
        {
            "claim_id": selection["claim_id"],
            "facet": selection["facet"],
            "facets": facets,
        }
    )
    return item, []


def _operational_candidates(
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_facet: dict[str, list[Mapping[str, Any]]] = {}
    for item in evidence:
        by_facet.setdefault(str(item.get("facet") or ""), []).append(item)

    def refs(names: Sequence[str]) -> list[str]:
        return list(
            dict.fromkeys(
                str(item["exact_ref"])
                for name in names
                for item in by_facet.get(name, [])
            )
        )

    candidates: list[dict[str, Any]] = []
    if by_facet.get("decision"):
        candidates.append(
            {"kind": "decision", "exact_refs": refs(["decision", "rationale"])}
        )
    if by_facet.get("owner") or by_facet.get("deadline"):
        candidates.append(
            {"kind": "commitment", "exact_refs": refs(["owner", "deadline"])}
        )
    for facet, kind in (
        ("procedure", "workflow"),
        ("gotcha", "gotcha"),
        ("premise", "state"),
        ("current_state", "state"),
        ("preference", "preference"),
    ):
        if by_facet.get(facet):
            candidates.append({"kind": kind, "exact_refs": refs([facet])})
    return candidates


def _retrieval_delta(
    retrieve: Callable[[dict[str, Any]], Any] | None,
    *,
    question: str,
    missing_facets: Sequence[str],
    seen_refs: Sequence[str],
    limits: CompilerBudgets,
) -> dict[str, Any]:
    result = {
        "calls": 0,
        "status": "not_called",
        "facet": None,
        "novel_exact_refs": [],
        "usage": {},
    }
    if not missing_facets:
        return result
    if limits.max_retrieval_calls == 0:
        result["status"] = "budget_exhausted"
        return result
    if retrieve is None:
        result["status"] = "unavailable"
        return result
    facet = missing_facets[0]
    result["facet"] = facet
    started = time.perf_counter()
    try:
        raw = retrieve(
            {
                "query": f"{facet.replace('_', ' ')} {question}"[:1_000],
                "facet": facet,
                "include": "verbatim",
                "detail": "answer_ready",
                "limit": limits.max_novel_refs,
                "seen_refs": list(seen_refs),
            }
        )
        payload = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(payload, Mapping) or payload.get("error"):
            raise ValueError("retrieval_error")
    except Exception:
        result.update(
            {
                "calls": 1,
                "status": "error",
                "latency_ms": round((time.perf_counter() - started) * 1_000.0, 3),
            }
        )
        return result
    seen = set(seen_refs)
    novel: list[str] = []
    hits = payload.get("hits")
    for hit in hits if isinstance(hits, list) else []:
        if not isinstance(hit, Mapping):
            continue
        exact_ref = str(hit.get("exact_ref") or "").strip()
        if (
            _EXACT_REF_RE.fullmatch(exact_ref)
            and exact_ref not in seen
            and exact_ref not in novel
        ):
            novel.append(exact_ref)
            if len(novel) >= limits.max_novel_refs:
                break
    metrics = payload.get("metrics")
    metric_map = metrics if isinstance(metrics, Mapping) else {}
    result.update(
        {
            "calls": 1,
            "status": "novel" if novel else "no_progress",
            "novel_exact_refs": novel,
            "usage": {
                key: metric_map[key]
                for key in (
                    "embedding_query_calls",
                    "embedding_query_tokens",
                    "embedding_query_tokens_complete",
                    "embedding_queries",
                )
                if key in metric_map
            },
            "latency_ms": round((time.perf_counter() - started) * 1_000.0, 3),
        }
    )
    return result


def _base(
    *,
    reason_code: str,
    limits: CompilerBudgets,
    baseline_refs: Sequence[str],
    started: float,
) -> dict[str, Any]:
    return {
        "version": EVIDENCE_COMPILER_VERSION,
        "status": "fallback",
        "state": "unknown",
        "reason_code": reason_code,
        "request": None,
        "evidence": [],
        "historical_evidence": [],
        "missing_facets": [],
        "finite_coverage": False,
        "computation": None,
        "computation_sha256": None,
        "operational_candidates": [],
        "rejections": [],
        "retrieval": {
            "calls": 0,
            "status": "not_called",
            "facet": None,
            "novel_exact_refs": [],
            "usage": {},
        },
        "selector": {
            "calls": 0,
            "status": "not_called",
            "proposal_sha256": None,
            "usage": {},
        },
        "baseline": {
            "exact_ref_count": len(baseline_refs),
            "exact_refs_sha256": _digest(list(baseline_refs)),
        },
        "budgets": limits.as_dict(),
        "trace": {},
        "trace_sha256": None,
        "metrics": {
            "selected_claims": 0,
            "exact_span_valid": 0,
            "rejected_claims": 0,
            "latency_ms": round((time.perf_counter() - started) * 1_000.0, 3),
        },
        "provenance": {
            "storage": "same_lcm_db",
            "raw_messages_authoritative": True,
            "selector_is_proposal": True,
            "final_prose_cached": False,
            "persisted": False,
        },
        "persistence": {
            "requested": False,
            "status": "not_requested",
            "view_id": None,
            "reason_code": "",
        },
    }


def _finish(result: dict[str, Any], *, started: float) -> dict[str, Any]:
    trace = {
        "version": EVIDENCE_COMPILER_VERSION,
        "request_sha256": _digest(result.get("request")),
        "selector_sha256": result["selector"].get("proposal_sha256"),
        "selected_claims": result["metrics"]["selected_claims"],
        "rejected_claims": result["metrics"]["rejected_claims"],
        "state": result["state"],
        "missing_facets": list(result["missing_facets"]),
        "retrieval_status": result["retrieval"]["status"],
        "evidence_refs_sha256": _digest(
            [item["exact_ref"] for item in result["evidence"]]
        ),
        "computation_sha256": result.get("computation_sha256"),
    }
    result["trace"] = trace
    result["trace_sha256"] = _digest(trace)
    result["metrics"]["latency_ms"] = round(
        (time.perf_counter() - started) * 1_000.0, 3
    )
    try:
        encoded = _canonical_json(result)
    except (TypeError, ValueError):
        return _base(
            reason_code="response_serialization_failed",
            limits=_budgets(result.get("budgets")),
            baseline_refs=[],
            started=started,
        )
    if len(encoded) > int(result["budgets"]["response_char_cap"]):
        fallback = _base(
            reason_code="response_truncated",
            limits=_budgets(result.get("budgets")),
            baseline_refs=[],
            started=started,
        )
        fallback["trace"] = {
            "version": EVIDENCE_COMPILER_VERSION,
            "truncated": True,
        }
        fallback["trace_sha256"] = _digest(fallback["trace"])
        return fallback
    return result


def _view_identity(
    request: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> QueryViewIdentity:
    operation = str(request.get("operation") or "none")
    if operation == "none":
        operation = "evidence_only"
    temporal = request.get("temporal_window")
    as_of = str(request.get("as_of") or "")
    question = str(request.get("question") or "")
    values: dict[str, Any] = {
        "intent_type": str(candidates[0].get("kind") or "evidence_compiler"),
        "operation": operation,
        "requirements_digest": _digest(request),
        "policy_version": EVIDENCE_COMPILER_VERSION,
    }
    if isinstance(temporal, Mapping) and temporal.get("start") and temporal.get("end"):
        relative = bool(
            as_of
            and re.search(
                r"\b(?:ago|today|yesterday|last\s+(?:day|week|month|year))\b",
                question.casefold(),
            )
        )
        values.update(
            {
                "time_mode": "relative" if relative else "absolute",
                "question_anchor": as_of if relative else "",
                "window_start": str(temporal["start"]),
                "window_end": str(temporal["end"]),
            }
        )
    return QueryViewIdentity(**values).normalized()


def _persist_compiled_view(result: dict[str, Any], *, engine: Any) -> None:
    result["persistence"]["requested"] = True
    candidates = result.get("operational_candidates")
    evidence = result.get("evidence")
    request = result.get("request")
    if not candidates or not evidence or not isinstance(request, Mapping):
        result["persistence"].update(
            {"status": "not_eligible", "reason_code": "no_high_value_grounded_state"}
        )
        return
    store = getattr(engine, "_query_views", None)
    if store is None:
        result["persistence"].update(
            {"status": "unavailable", "reason_code": "query_view_store_unavailable"}
        )
        return
    token = None
    try:
        identity = _view_identity(request, candidates)
        dependencies = [
            store.snapshot_dependency(
                int(item["store_id"]),
                int(item["span_start"]),
                int(item["span_end"]),
                str(item["quote"]),
                assertion_id=str(item.get("assertion_id") or ""),
            )
            for item in evidence
        ]
        token = store.claim_build(identity)
        closed = sorted(
            {str(item.get("facet") or "") for item in evidence if item.get("facet")}
        )
        manifest = {
            "closed_slots": closed,
            "open_slots": list(result.get("missing_facets") or []),
            "operands": [
                {
                    "exact_ref": str(item["exact_ref"]),
                    "facet": str(item.get("facet") or ""),
                    "facets": dict(item.get("facets") or {}),
                }
                for item in evidence
            ],
            "retrieval_calls": int(result.get("retrieval", {}).get("calls") or 0),
            "evidence_refs": [dependency.citation for dependency in dependencies],
            "coverage": {
                "state": str(result.get("state") or "unknown"),
                "finite": bool(result.get("finite_coverage")),
            },
        }
        completeness = (
            "complete"
            if result.get("state")
            in {"answer_sufficient", "finite_coverage", "computation_sufficient"}
            else "partial"
        )
        published = store.publish_ready(
            token,
            dependencies=dependencies,
            manifest=manifest,
            completeness=completeness,
            search_policy_version=EVIDENCE_COMPILER_VERSION,
        )
    except QueryViewBuildInProgressError:
        result["persistence"].update(
            {"status": "busy", "reason_code": "query_view_build_in_progress"}
        )
        return
    except Exception as exc:
        if token is not None:
            try:
                store.mark_failed(token, str(exc))
            except Exception:
                # Cleanup is best-effort and must never replace the publish
                # failure that selected this response contract.
                pass
        result["persistence"].update(
            {"status": "error", "reason_code": "query_view_publish_failed"}
        )
        return
    result["persistence"].update(
        {
            "status": "published" if published else "stale",
            "view_id": identity.view_id,
            "reason_code": "" if published else "corpus_changed_during_publish",
        }
    )
    result["provenance"]["persisted"] = bool(published)


def prepare_evidence_selector(
    question: Any,
    *,
    baseline_refs: Sequence[Any] = (),
    question_date: Any = None,
    budgets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the deterministic, code-owned selector envelope."""
    limits = _budgets(budgets)
    normalized_refs, exact_refs = _normalize_baseline_refs(
        baseline_refs, limit=limits.max_input_refs
    )
    request = derive_evidence_request(question, question_date)
    selector_request = {
        "version": SELECTOR_SCHEMA_VERSION,
        "question": request["question"],
        "as_of": request["as_of"],
        "facets": request["facets"],
        "operation": request["operation"],
        "exhaustive": request["exhaustive"],
        "expected_cardinality": request["expected_cardinality"],
        "baseline_evidence": normalized_refs,
        "budgets": {
            "max_selections": limits.max_selections,
            "max_quote_chars": limits.max_quote_chars,
        },
    }
    return {
        "version": SELECTOR_SCHEMA_VERSION,
        "request": request,
        "selector_request": selector_request,
        "baseline_exact_refs": exact_refs,
        "baseline_exact_refs_sha256": _digest(exact_refs),
        "budgets": limits.as_dict(),
    }


def compile_evidence(
    question: Any,
    *,
    engine: Any,
    baseline_refs: Sequence[Any] = (),
    question_date: Any = None,
    selector: Callable[[dict[str, Any]], Any] | None = None,
    retrieve: Callable[[dict[str, Any]], Any] | None = None,
    enabled: bool = False,
    persist_view: bool = False,
    budgets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a bounded exact-evidence brief, or fail to the ordinary path."""
    started = time.perf_counter()
    limits = _budgets(budgets)
    normalized_refs, exact_refs = _normalize_baseline_refs(
        baseline_refs, limit=limits.max_input_refs
    )
    result = _base(
        reason_code="feature_disabled",
        limits=limits,
        baseline_refs=exact_refs,
        started=started,
    )
    if not enabled:
        return _finish(result, started=started)
    if selector is None:
        result["reason_code"] = "selector_unavailable"
        return _finish(result, started=started)
    try:
        prepared = prepare_evidence_selector(
            question,
            baseline_refs=normalized_refs,
            question_date=question_date,
            budgets=budgets,
        )
    except ValueError as exc:
        result["reason_code"] = str(exc)
        return _finish(result, started=started)
    request = prepared["request"]
    result["request"] = request
    selector_request = prepared["selector_request"]
    selector_started = time.perf_counter()
    try:
        raw_proposal = selector(selector_request)
    except Exception:
        result["reason_code"] = "selector_error"
        result["selector"] = {
            "calls": 1,
            "status": "error",
            "proposal_sha256": None,
            "usage": {},
            "latency_ms": round((time.perf_counter() - selector_started) * 1_000.0, 3),
        }
        return _finish(result, started=started)
    proposal, proposal_error, proposal_digest = _validate_proposal(
        raw_proposal, request=request, limits=limits
    )
    result["selector"] = {
        "calls": 1,
        "status": "invalid" if proposal_error else "valid",
        "proposal_sha256": proposal_digest,
        "usage": proposal.get("usage", {}) if proposal else {},
        "latency_ms": round((time.perf_counter() - selector_started) * 1_000.0, 3),
    }
    if proposal_error or proposal is None:
        result["reason_code"] = proposal_error or "selector_schema_invalid"
        return _finish(result, started=started)
    request["facets"] = proposal["effective_facets"]
    request["facet_source"] = proposal["facet_source"]

    allowed_refs = set(exact_refs)
    validated: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for selection in proposal["selections"]:
        item, errors = _validate_claim(
            selection,
            question=request["question"],
            question_date=request["as_of"],
            engine=engine,
            allowed_refs=allowed_refs,
            limits=limits,
        )
        if errors:
            rejections.extend(
                {
                    "claim_id": selection["claim_id"],
                    "facet": selection["facet"],
                    "reason_code": error,
                }
                for error in dict.fromkeys(errors)
            )
            continue
        validated.append(item)  # type: ignore[arg-type]
        candidates.append(_candidate(selection))
    result["rejections"] = rejections
    result["metrics"].update(
        {
            "selected_claims": len(validated),
            "exact_span_valid": len(validated),
            "rejected_claims": len({item["claim_id"] for item in rejections}),
        }
    )

    required = [item["name"] for item in request["facets"] if item["required"]]
    covered = {str(item["facet"]) for item in validated}
    missing = list(
        dict.fromkeys(
            [name for name in required if name not in covered]
            + list(proposal["missing_facets"])
        )
    )
    result["missing_facets"] = missing
    result["retrieval"] = _retrieval_delta(
        retrieve,
        question=request["question"],
        missing_facets=missing,
        seen_refs=exact_refs,
        limits=limits,
    )

    computation = None
    selection = None
    combined_evidence = list(validated)
    if candidates:
        packed = json.loads(
            build_evidence_pack(
                {
                    "question": request["question"],
                    "question_date": request["as_of"],
                    "baseline_refs": candidates,
                    "budgets": {
                        "max_refs": limits.max_input_refs,
                        "max_quote_chars": limits.max_quote_chars,
                        "max_retrieval_calls": 0,
                    },
                },
                engine=engine,
            )
        )
        computation = packed.get("computation")
        selection = packed.get("selection")

    historical: list[dict[str, Any]] = []
    conflicted = False
    if request["operation"] == "latest_fact" and validated:
        selected_refs = (
            selection.get("exact_refs") if isinstance(selection, Mapping) else None
        )
        if isinstance(selected_refs, list) and len(selected_refs) == 1:
            chosen = set(str(value) for value in selected_refs)
            historical = [item for item in validated if item["exact_ref"] not in chosen]
            combined_evidence = [
                item for item in validated if item["exact_ref"] in chosen
            ]
        else:
            conflicted = len(validated) > 1

    expected = request.get("expected_cardinality")
    # Caller-evidence certification uses exact_ref uniqueness; engine finite-scan
    # certification uses dedupe_key uniqueness. These are intentionally different surfaces.
    finite_coverage = bool(
        request["exhaustive"]
        and isinstance(expected, int)
        and expected > 0
        and len({item["exact_ref"] for item in validated}) == expected
        and not missing
        and not rejections
    )
    if request["operation"] == "count_distinct" and expected is None:
        computation = None
    if request["operation"] == "scalar_fact":
        computation = None
    if request["exhaustive"] and not finite_coverage:
        computation = None
    if conflicted:
        state = "conflicted"
        reason = "unresolved_conflict"
    elif computation is not None:
        state = "computation_sufficient"
        reason = "validated_canonical_computation"
    elif finite_coverage:
        state = "finite_coverage"
        reason = "fixed_cardinality_product_verified"
    elif request["exhaustive"] and combined_evidence:
        state = "partial"
        reason = "finite_coverage_unproven"
    elif combined_evidence and not missing:
        state = "answer_sufficient"
        reason = "named_facets_grounded"
    elif combined_evidence:
        state = "partial"
        reason = "named_facets_partial"
    else:
        state = "unknown"
        reason = "no_validated_evidence"

    result.update(
        {
            "status": "compiled" if state != "unknown" else "fallback",
            "state": state,
            "reason_code": reason,
            "evidence": combined_evidence,
            "historical_evidence": historical,
            "finite_coverage": finite_coverage,
            "computation": computation,
            "computation_sha256": _digest(computation) if computation else None,
            "operational_candidates": _operational_candidates(validated),
        }
    )
    if persist_view:
        _persist_compiled_view(result, engine=engine)
    return _finish(result, started=started)


def compile_preanswer_evidence(*args, **kwargs):
    """Invoke the deterministic V4.6.3 requirements compiler.

    The lazy import preserves plugin bootstrap order and keeps the legacy
    proposal compiler byte-compatible.
    """
    from .requirements_compiler import compile_preanswer_evidence as _compile

    return _compile(*args, **kwargs)
