"""Deterministic, baseline-first pre-answer evidence compilation.

This module owns the V4.6.3 requirements path.  It performs no model call and
never has access to benchmark identity or answer data.  Unsupported, ambiguous,
or non-improving work returns an unchanged-baseline decision.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import hashlib
import json
import math
import re
import time
from typing import Any, Callable, Mapping, Sequence

from .answer_contract import (
    AnswerContract,
    EvidenceSlot,
    compile_answer_contract,
)
from .occurrence_time import resolve_occurrence_time
from .reasoning import (
    EvidencePlan,
    execute_plan,
    ground_evidence,
    normalize_unit,
    question_date_as_of_epoch,
)


REQUIREMENTS_COMPILER_VERSION = "evidence-contract-compiler-v1"
_EXACT_REF_RE = re.compile(
    r"^lcm:(?P<store_id>[1-9]\d*):(?P<start>\d+)-(?P<end>\d+)$"
)
_DIGIT_NUMBER_RE = re.compile(
    r"(?<![\w.])-?(?:\d+(?:,\d{3})*|\d*\.\d+)(?!\w)"
)
_WORD_NUMBER_VALUES = {
    "zero": 0,
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
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}
_WORD_NUMBER_RE = re.compile(
    r"\b(?:" + "|".join(_WORD_NUMBER_VALUES) + r")\b", re.IGNORECASE
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_SECRET_RE = re.compile(
    r"(?:Bearer\s+[A-Za-z0-9._~-]{8,}|\b(?:sk|pa)-[A-Za-z0-9_-]{8,}|"
    r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+)",
    re.IGNORECASE,
)
_DATE_TEXT_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?)\b",
    re.IGNORECASE,
)
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%B %d, %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%b %d %Y",
)
_TIME_FACTORS = {
    "minute": 1.0,
    "hour": 60.0,
    "day": 1_440.0,
    "week": 10_080.0,
}
_MEASUREMENT_UNITS = frozenset({
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "year",
    "usd",
    "point",
    "page",
})


@dataclass(frozen=True)
class RequirementsBudgets:
    max_candidates: int = 48
    max_hydrated_candidates: int = 12
    max_novel_refs: int = 6
    max_added_context_tokens: int = 850
    max_retrieval_calls: int = 2
    max_quote_chars: int = 1_200
    max_scan_rows: int = 4_096

    def as_dict(self) -> dict[str, int]:
        return {
            "max_candidates": self.max_candidates,
            "max_hydrated_candidates": self.max_hydrated_candidates,
            "max_novel_refs": self.max_novel_refs,
            "max_added_context_tokens": self.max_added_context_tokens,
            "max_retrieval_calls": self.max_retrieval_calls,
            "max_quote_chars": self.max_quote_chars,
            "max_scan_rows": self.max_scan_rows,
        }


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return min(maximum, max(minimum, parsed))


def _budgets(raw: Any) -> RequirementsBudgets:
    value = raw if isinstance(raw, Mapping) else {}
    return RequirementsBudgets(
        max_candidates=_bounded_int(
            value.get("max_candidates"), default=48, minimum=1, maximum=48
        ),
        max_hydrated_candidates=_bounded_int(
            value.get("max_hydrated_candidates"), default=12, minimum=1, maximum=12
        ),
        max_novel_refs=_bounded_int(
            value.get("max_novel_refs"), default=6, minimum=1, maximum=6
        ),
        max_added_context_tokens=_bounded_int(
            value.get("max_added_context_tokens"), default=850, minimum=64, maximum=850
        ),
        max_retrieval_calls=_bounded_int(
            value.get("max_retrieval_calls"), default=2, minimum=0, maximum=2
        ),
        max_quote_chars=_bounded_int(
            value.get("max_quote_chars"), default=1_200, minimum=64, maximum=1_200
        ),
        max_scan_rows=_bounded_int(
            value.get("max_scan_rows"), default=4_096, minimum=1, maximum=4_096
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


def _tokens(value: Any) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(str(value or ""))}


def _estimate_tokens(value: str) -> int:
    return max(1, math.ceil(len(value) / 4)) if value else 0


def _raw_quote(raw: Any) -> str:
    if not isinstance(raw, Mapping):
        return ""
    return str(raw.get("quote") or raw.get("content") or "")


def _rank_baseline_inputs(
    values: Sequence[Any], contract: AnswerContract | None
) -> list[Any]:
    """Prioritize already-retrieved exact evidence before bounded hydration.

    This ranking can choose what to validate, but it cannot make an evidence
    claim. Exact validation and ambiguity checks still happen after hydration.
    Stable input order breaks equal scores.
    """
    if contract is None:
        return list(values)

    unit_forms = set(_unit_forms(contract.requested_unit))
    ranked: list[tuple[int, int, Any]] = []
    for index, raw in enumerate(values):
        quote = _raw_quote(raw)
        tokens = _tokens(quote)
        anchor_overlap = len(tokens.intersection(contract.anchors))
        score = anchor_overlap * 20
        if unit_forms and any(form in quote.casefold() for form in unit_forms):
            score += 12
        if any(slot.anchor and _tokens(slot.anchor).issubset(tokens) for slot in contract.slots):
            score += 30
        if contract.operation in {"scalar", "sum", "difference"} and (
            _DIGIT_NUMBER_RE.search(quote) or _WORD_NUMBER_RE.search(quote)
        ):
            score += 6
        if contract.temporal_window is not None and (
            _DATE_TEXT_RE.search(quote) or anchor_overlap
        ):
            score += 6
        if quote.rstrip().endswith("?"):
            score += 4
        ranked.append((-score, index, raw))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked]


def _canonical_unit(raw: Any) -> str | None:
    text = str(raw or "").strip().casefold().replace("_", " ")
    if not text:
        return None
    irregular = {
        "people": "person",
        "persons": "person",
        "properties": "property",
        "vacations": "vacation",
        "holidays": "holiday",
        "clothes": "clothing_item",
        "clothing items": "clothing_item",
    }
    if text in irregular:
        return irregular[text]
    normalized = normalize_unit(text)
    if normalized != text:
        return normalized
    if text.endswith("ies") and len(text) > 4:
        return text[:-3] + "y"
    if text.endswith("s") and not text.endswith("ss") and len(text) > 3:
        return text[:-1]
    return text.replace(" ", "_")


def _compatible_unit(actual: str | None, requested: str | None) -> bool:
    if requested is None:
        return actual is not None
    if actual == requested:
        return True
    return actual in _TIME_FACTORS and requested in _TIME_FACTORS


def _source_session_date(
    engine: Any, row: Mapping[str, Any]
) -> tuple[str | None, str]:
    session_id = str(row.get("session_id") or "")
    sidecar = getattr(engine, "_session_occurrence_dates", {}) or {}
    raw = str(sidecar.get(session_id) or "").strip()
    if raw:
        try:
            return (
                date.fromisoformat(raw[:10].replace("/", "-")).isoformat(),
                "benchmark_session_date",
            )
        except ValueError:
            return None, "unknown"
    observed = row.get("observed_at")
    if observed is None:
        return None, "unknown"
    try:
        return (
            datetime.fromtimestamp(float(observed), tz=timezone.utc).date().isoformat(),
            "host_message_timestamp",
        )
    except (TypeError, ValueError, OverflowError, OSError):
        return None, "unknown"


def _normalize_ref(raw: Any, *, engine: Any, origin: str) -> dict[str, Any] | None:
    if isinstance(raw, str):
        candidate = {"exact_ref": raw}
    elif isinstance(raw, Mapping):
        candidate = dict(raw)
    else:
        return None
    exact_ref = str(candidate.get("exact_ref") or "").strip()
    if not exact_ref and candidate.get("store_id") is not None:
        try:
            store_id = int(candidate["store_id"])
            offset = max(0, int(candidate.get("content_offset") or 0))
        except (TypeError, ValueError, OverflowError):
            return None
        quote = str(candidate.get("content") or candidate.get("quote") or "")
        exact_ref = f"lcm:{store_id}:{offset}-{offset + len(quote)}"
    match = _EXACT_REF_RE.fullmatch(exact_ref)
    if match is None:
        return None
    store_id = int(match.group("store_id"))
    declared_start = int(match.group("start"))
    declared_end = int(match.group("end"))
    row = engine._store.get(store_id)
    if row is None:
        return None
    content = str(row.get("content") or "")
    if declared_start < 0 or declared_end <= declared_start or declared_end > len(content):
        return None
    window = content[declared_start:declared_end]
    proposed = str(candidate.get("quote") or candidate.get("content") or "")
    if proposed:
        first = window.find(proposed)
        if first < 0 or window.find(proposed, first + 1) >= 0:
            return None
        start = declared_start + first
        end = start + len(proposed)
        quote = proposed
    else:
        start, end, quote = declared_start, declared_end, window
    if not quote or _SECRET_RE.search(quote):
        return None
    observed = row.get("observed_at")
    try:
        observed_epoch = float(observed) if observed is not None else None
        if observed_epoch is not None and not math.isfinite(observed_epoch):
            observed_epoch = None
    except (TypeError, ValueError, OverflowError):
        observed_epoch = None
    session_date, session_date_source = _source_session_date(engine, row)
    occurrence = resolve_occurrence_time(
        quote,
        observed_at=observed_epoch or 0.0,
        session_date=session_date,
    )
    return {
        "exact_ref": f"lcm:{store_id}:{start}-{end}",
        "store_id": store_id,
        "span_start": start,
        "span_end": end,
        "quote": quote,
        "content": content,
        "session_id": str(row.get("session_id") or ""),
        "role": str(row.get("role") or "unknown"),
        "observed_at": observed_epoch,
        "ingested_at": row.get("ingested_at"),
        "session_date": session_date,
        "session_date_source": session_date_source,
        "occurrence_time": occurrence,
        "origin": origin,
    }


def _number_value(raw: str) -> int | float | None:
    normalized = raw.casefold().replace(",", "")
    if normalized in _WORD_NUMBER_VALUES:
        return _WORD_NUMBER_VALUES[normalized]
    try:
        value = float(normalized)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return int(value) if value.is_integer() else value


def _date_value(raw: str) -> date | None:
    normalized = " ".join(str(raw or "").strip().split())
    for pattern in _DATE_FORMATS:
        try:
            return datetime.strptime(normalized, pattern).date()
        except ValueError:
            continue
    return None


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    for match in re.finditer(r"(?:[.!?;,]|\band\b|\bplus\b)", text[:start], re.IGNORECASE):
        left = match.end()
    right = len(text)
    boundary = re.search(r"(?:[.!?;,]|\band\b|\bplus\b)", text[end:], re.IGNORECASE)
    if boundary:
        right = end + boundary.start()
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return left, right


def _unit_near_number(
    quote: str,
    start: int,
    end: int,
    *,
    requested: str | None,
) -> str | None:
    before = quote[max(0, start - 3) : start]
    after = quote[end : end + 40]
    if "$" in before or re.match(r"\s*(?:usd|dollars?)\b", after, re.IGNORECASE):
        return "usd"
    if requested:
        forms = {
            requested.replace("_", " "),
            requested.replace("_", " ") + "s",
        }
        if requested == "person":
            forms.update({"people", "persons"})
        if requested == "property":
            forms.add("properties")
        for form in sorted(forms, key=len, reverse=True):
            if re.match(rf"\s*{re.escape(form)}\b", after, re.IGNORECASE):
                return requested
    match = re.match(
        r"\s*(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|pages?|"
        r"points?|items?|events?|people|persons?|properties|[A-Za-z][A-Za-z-]{2,})\b",
        after,
        re.IGNORECASE,
    )
    return _canonical_unit(match.group(1)) if match else None


def _numeric_phrase_bounds(
    quote: str,
    start: int,
    end: int,
    *,
    unit: str | None,
) -> tuple[int, int]:
    """Return the smallest exact span that still grounds value and unit.

    A whole sentence can contain more than one operand (for example, "from
    20 minutes to 30 minutes").  Grounding the whole sentence twice would
    create duplicate evidence references, while grounding only the digits
    would discard the unit.  This keeps each operand independently citable.
    """

    phrase_start = start
    dollar = quote[max(0, start - 3) : start]
    dollar_index = dollar.rfind("$")
    if unit == "usd" and dollar_index >= 0:
        absolute = max(0, start - 3) + dollar_index
        if quote[absolute + 1 : start].isspace() or absolute + 1 == start:
            phrase_start = absolute

    phrase_end = end
    suffix = re.match(
        r"\s*(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|pages?|"
        r"points?|items?|events?|people|persons?|properties|usd|dollars?|"
        r"[A-Za-z][A-Za-z-]{2,})\b",
        quote[end : end + 40],
        re.IGNORECASE,
    )
    if suffix and _canonical_unit(suffix.group(1)) == unit:
        phrase_end = end + suffix.end()
    return phrase_start, phrase_end


def _anchor_score(contract: AnswerContract, quote: str, slot: EvidenceSlot | None) -> int:
    haystack = _tokens(quote)
    score = len(haystack.intersection(contract.anchors))
    if slot and slot.anchor:
        anchor_tokens = _tokens(slot.anchor)
        if anchor_tokens and anchor_tokens.issubset(haystack):
            score += 4
    return score


def _numeric_candidates(
    source: Mapping[str, Any], contract: AnswerContract
) -> list[dict[str, Any]]:
    quote = str(source["quote"])
    mentions: list[tuple[int, int, int | float]] = []
    for match in _DIGIT_NUMBER_RE.finditer(quote):
        value = _number_value(match.group(0))
        if value is not None:
            mentions.append((match.start(), match.end(), value))
    for match in _WORD_NUMBER_RE.finditer(quote):
        value = _number_value(match.group(0))
        if value is not None:
            mentions.append((match.start(), match.end(), value))
    mentions.sort(key=lambda item: (item[0], item[1]))

    output: list[dict[str, Any]] = []
    for start, end, value in mentions:
        unit = _unit_near_number(
            quote, start, end, requested=contract.requested_unit
        )
        clause_start, clause_end = _clause_bounds(quote, start, end)
        clause = quote[clause_start:clause_end]
        if not clause or clause.rstrip().endswith("?"):
            continue
        phrase_start, phrase_end = _numeric_phrase_bounds(
            quote, start, end, unit=unit
        )
        phrase = quote[phrase_start:phrase_end]
        absolute_start = int(source["span_start"]) + phrase_start
        absolute_end = int(source["span_start"]) + phrase_end
        candidate = {
            **source,
            "exact_ref": f"lcm:{source['store_id']}:{absolute_start}-{absolute_end}",
            "span_start": absolute_start,
            "span_end": absolute_end,
            "quote": phrase,
            "value": value,
            "unit": unit,
            "slot_names": [],
            "anchor_score": 0,
        }
        for slot in contract.slots:
            if slot.value_type != "number":
                continue
            score = _anchor_score(contract, clause, slot)
            compatible = _compatible_unit(unit, slot.unit)
            if (
                not compatible
                and slot.anchor
                and score >= 4
                and slot.unit is None
                and unit is None
            ):
                compatible = True
            if (
                not compatible
                and slot.anchor
                and score >= 6
                and slot.unit not in _MEASUREMENT_UNITS
                and unit not in _MEASUREMENT_UNITS
            ):
                # Entity-count wording varies (for example an adjective may
                # precede the head noun). Strong named-facet grounding can
                # bind the count, but measurement dimensions never relax.
                compatible = True
            if not compatible:
                continue
            if slot.anchor and score < 4:
                continue
            if not slot.anchor and contract.anchors and score < 1:
                continue
            candidate["slot_names"].append(slot.name)
            candidate["anchor_score"] = max(candidate["anchor_score"], score)
        if (
            contract.operation in {"sum", "difference"}
            and re.search(
                r"\b(?:total|combined|altogether|in all|sum|difference|saved?|savings?)\b",
                clause,
                re.IGNORECASE,
            )
            and _compatible_unit(unit, contract.requested_unit)
            and _anchor_score(contract, quote, None) >= 1
        ):
            candidate["slot_names"].append("answer")
            candidate["anchor_score"] = max(
                candidate["anchor_score"], _anchor_score(contract, quote, None) + 8
            )
        candidate["anchor_score"] = max(
            int(candidate["anchor_score"]),
            int(source.get("paired_anchor_score") or 0),
        )
        if candidate["slot_names"]:
            candidate["slot_names"] = list(dict.fromkeys(candidate["slot_names"]))
            output.append(candidate)
    return output


def _declarative_value(source: Mapping[str, Any], contract: AnswerContract) -> str | None:
    quote = " ".join(str(source["quote"]).split())
    if not quote or quote.endswith("?"):
        return None
    transition = (
        re.search(
            r"\bfrom\s+([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3})\s+"
            r"to\s+([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3})",
            quote,
        )
        if contract.answer_kind == "place"
        else None
    )
    if transition and contract.operation in {"latest", "previous"}:
        return (
            transition.group(2).strip()
            if contract.operation == "latest"
            else transition.group(1).strip()
        )
    if (
        contract.operation != "date_filter"
        and _anchor_score(contract, quote, contract.slots[0]) < 1
    ):
        return None
    kind = contract.answer_kind
    if kind == "time":
        match = _DATE_TEXT_RE.search(quote)
        return match.group(0) if match else None
    if kind == "person":
        names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", quote)
        blocked = {"I", "The", "A", "An", "My", "We", "It"}
        values = [name for name in names if name not in blocked]
        return values[-1] if values else None
    if kind == "place":
        if transition:
            return transition.group(2).strip()
        match = re.search(
            r"\b(?:in|at|to|near|from)\s+([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3})",
            quote,
        )
        return match.group(1).strip() if match else None
    if kind == "advice":
        return quote if str(source.get("role")) == "assistant" else None
    return quote


def _text_candidate(
    source: Mapping[str, Any], contract: AnswerContract
) -> dict[str, Any] | None:
    slot = contract.slots[0]
    if slot.expected_role != "any" and str(source.get("role")) != slot.expected_role:
        return None
    value = _declarative_value(source, contract)
    if value is None:
        return None
    transition_explicit = bool(
        contract.answer_kind == "place"
        and re.search(
            r"\bfrom\s+[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3}\s+"
            r"to\s+[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3}",
            str(source["quote"]),
        )
    )
    return {
        **source,
        "value": value,
        "unit": None,
        "slot_names": [slot.name],
        "anchor_score": max(
            _anchor_score(contract, str(source["quote"]), slot),
            int(source.get("paired_anchor_score") or 0),
        ),
        "transition_explicit": transition_explicit,
    }


def _date_candidates(
    source: Mapping[str, Any], contract: AnswerContract
) -> list[dict[str, Any]]:
    quote = str(source["quote"])
    output: list[dict[str, Any]] = []
    for match in _DATE_TEXT_RE.finditer(quote):
        parsed = _date_value(match.group(0))
        if parsed is None:
            continue
        start = int(source["span_start"]) + match.start()
        end = int(source["span_start"]) + match.end()
        occurrence = resolve_occurrence_time(
            match.group(0),
            observed_at=source.get("observed_at") or 0.0,
            session_date=source.get("session_date"),
        )
        output.append(
            {
                **source,
                "exact_ref": f"lcm:{source['store_id']}:{start}-{end}",
                "span_start": start,
                "span_end": end,
                "quote": match.group(0),
                "value": match.group(0),
                "date": parsed.isoformat(),
                "occurrence_time": occurrence,
                "unit": None,
                "slot_names": [slot.name for slot in contract.slots],
                "anchor_score": _anchor_score(contract, quote, None),
            }
        )
    return output


def _ordered_event_candidate(
    source: Mapping[str, Any], contract: AnswerContract
) -> dict[str, Any] | None:
    quote = " ".join(str(source["quote"]).split())
    if not quote or quote.endswith("?"):
        return None
    occurrence = source.get("occurrence_time")
    event_date = (
        str(occurrence.get("event_date") or "")
        if isinstance(occurrence, Mapping)
        else ""
    )
    if not event_date:
        mention = _DATE_TEXT_RE.search(quote)
        parsed = _date_value(mention.group(0)) if mention else None
        event_date = parsed.isoformat() if parsed else ""
    if not event_date:
        return None
    key = _finite_event_key(quote, contract.requested_unit or "event")
    if key is None:
        return None
    label = None
    if contract.answer_kind == "place":
        place = re.search(
            r"\b(?:arrived\s+in|flew\s+to|stayed\s+in|travelled\s+to|"
            r"traveled\s+to|visited|went\s+to)\s+"
            r"([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3})\b",
            quote,
        )
        if place is not None:
            label = place.group(1).strip()
    quote_without_dates = _DATE_TEXT_RE.sub(" ", quote)
    names = [
        name
        for name in re.findall(
            r"\b[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,3}\b",
            quote_without_dates,
        )
        if name not in {"I", "The", "A", "An", "My", "We", "On", "In"}
    ]
    label = label or (names[-1] if names else quote[:300])
    slot_names = []
    for slot in contract.slots:
        if slot.anchor and not _tokens(slot.anchor).issubset(_tokens(quote)):
            continue
        slot_names.append(slot.name)
    if not slot_names:
        return None
    return {
        **source,
        "value": label,
        "label": label[:300],
        "key": key,
        "date": event_date,
        "unit": None,
        "slot_names": slot_names,
        "anchor_score": _anchor_score(contract, quote, None),
    }


def _extract_candidates(
    sources: Sequence[Mapping[str, Any]], contract: AnswerContract
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for source in sources:
        if contract.operation in {"scalar", "sum", "difference"}:
            candidates = _numeric_candidates(source, contract)
        elif contract.operation == "date_interval":
            candidates = _date_candidates(source, contract)
        elif contract.operation == "order":
            ordered = _ordered_event_candidate(source, contract)
            candidates = [ordered] if ordered is not None else []
        else:
            text_candidate = _text_candidate(source, contract)
            candidates = [text_candidate] if text_candidate is not None else []
        for candidate in candidates:
            key = (
                candidate["exact_ref"],
                candidate.get("value"),
                candidate.get("unit"),
                candidate.get("key"),
                tuple(candidate.get("slot_names") or ()),
            )
            if key not in seen:
                seen.add(key)
                output.append(candidate)
    return output


def _candidate_inventory(
    sources: Sequence[Mapping[str, Any]],
    contract: AnswerContract,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    return _temporally_eligible_candidates(
        _extract_candidates(sources, contract)[:limit], contract
    )


def _open_slots(
    candidates: Sequence[Mapping[str, Any]], contract: AnswerContract
) -> list[EvidenceSlot]:
    if not contract.slots:
        return []
    if all(slot.anchor is None for slot in contract.slots):
        unique_refs = {
            str(candidate.get("exact_ref") or "")
            for candidate in candidates
            if candidate.get("exact_ref")
        }
        return list(contract.slots[len(unique_refs) :])
    return [
        slot
        for slot in contract.slots
        if not any(slot.name in candidate.get("slot_names", []) for candidate in candidates)
    ]


def _normalized_numeric(value: int | float, unit: str | None, requested: str | None):
    number = float(value)
    if requested in _TIME_FACTORS and unit in _TIME_FACTORS:
        number = number * _TIME_FACTORS[unit] / _TIME_FACTORS[requested]
        unit = requested
    return (round(number, 9), unit)


def _select_scalar(
    candidates: Sequence[Mapping[str, Any]], contract: AnswerContract
) -> tuple[dict[str, Any] | None, str]:
    compatible = [
        candidate
        for candidate in candidates
        if "answer" in candidate.get("slot_names", [])
        and isinstance(candidate.get("value"), (int, float))
        and not isinstance(candidate.get("value"), bool)
    ]
    if not compatible:
        return None, "scalar_not_found"
    best_score = max(int(item.get("anchor_score") or 0) for item in compatible)
    compatible = [
        item
        for item in compatible
        if int(item.get("anchor_score") or 0) == best_score
    ]
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for candidate in compatible:
        key = _normalized_numeric(
            candidate["value"], candidate.get("unit"), contract.requested_unit
        )
        grouped.setdefault(key, []).append(candidate)
    if len(grouped) != 1:
        return None, "ambiguous_scalar_candidates"
    (value, unit), values = next(iter(grouped.items()))
    chosen = max(
        values,
        key=lambda item: (
            int(item.get("anchor_score") or 0),
            item.get("origin") != "baseline",
            -int(item["store_id"]),
        ),
    )
    return {
        **chosen,
        "value": int(value) if float(value).is_integer() else value,
        "unit": unit,
    }, "unique_source_asserted_scalar"


def _select_text(
    candidates: Sequence[Mapping[str, Any]], contract: AnswerContract
) -> tuple[dict[str, Any] | None, str]:
    compatible = [candidate for candidate in candidates if "answer" in candidate.get("slot_names", [])]
    if not compatible:
        return None, "answer_fact_not_found"
    best_score = max(int(item.get("anchor_score") or 0) for item in compatible)
    compatible = [
        item
        for item in compatible
        if int(item.get("anchor_score") or 0) == best_score
    ]
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in compatible:
        key = " ".join(str(candidate.get("value") or "").casefold().split())
        if key:
            groups.setdefault(key, []).append(candidate)
    if len(groups) != 1:
        return None, "ambiguous_fact_candidates"
    values = next(iter(groups.values()))
    return dict(max(values, key=lambda item: int(item.get("anchor_score") or 0))), "unique_source_fact"


def _select_temporal_event(
    candidates: Sequence[Mapping[str, Any]], contract: AnswerContract
) -> tuple[dict[str, Any] | None, str]:
    eligible = [
        candidate
        for candidate in candidates
        if "answer" in candidate.get("slot_names", [])
        and candidate.get("temporal_match_basis") in {
            "occurred_at",
            "adapter_session_date",
        }
    ]
    if not eligible:
        return None, "event_in_window_not_found"
    # Input order preserves the product rank.  Anchor score may improve that
    # order, but grammar never turns the chosen event into finite coverage.
    best_score = max(int(item.get("anchor_score") or 0) for item in eligible)
    chosen = next(
        item
        for item in eligible
        if int(item.get("anchor_score") or 0) == best_score
    )
    return dict(chosen), "best_supported_event_in_window"


def _candidate_time(candidate: Mapping[str, Any]) -> tuple[float | None, str]:
    occurrence = candidate.get("occurrence_time")
    if isinstance(occurrence, Mapping) and occurrence.get("event_at") is not None:
        try:
            return float(occurrence["event_at"]), "occurred_at"
        except (TypeError, ValueError, OverflowError):
            pass
    session_date = str(candidate.get("session_date") or "")
    if (
        session_date
        and candidate.get("session_date_source") == "benchmark_session_date"
    ):
        try:
            day = date.fromisoformat(session_date)
            return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp(), "adapter_session_date"
        except ValueError:
            pass
    observed = candidate.get("observed_at")
    if observed is not None:
        try:
            value = float(observed)
            if math.isfinite(value):
                return value, "observed_at"
        except (TypeError, ValueError, OverflowError):
            pass
    return None, "unknown"


def _candidate_event_day(candidate: Mapping[str, Any]) -> tuple[date | None, str]:
    occurrence = candidate.get("occurrence_time")
    if isinstance(occurrence, Mapping):
        raw = str(occurrence.get("event_date") or "")
        if raw:
            try:
                return date.fromisoformat(raw), "occurred_at"
            except ValueError:
                pass
    if candidate.get("session_date_source") == "benchmark_session_date":
        raw = str(candidate.get("session_date") or "")
        if raw:
            try:
                return date.fromisoformat(raw), "adapter_session_date"
            except ValueError:
                pass
    return None, "unknown"


def _available_as_of(candidate: Mapping[str, Any], contract: AnswerContract) -> bool:
    if not contract.question_as_of:
        return True
    boundary = question_date_as_of_epoch(contract.question_as_of)
    if boundary is None:
        return False
    event_day, _ = _candidate_event_day(candidate)
    if event_day is not None and event_day > date.fromisoformat(contract.question_as_of):
        return False
    # ours: availability is enforced for every operation (the temporal-only
    # early return was deliberately removed); theirs: observed_at leads and the
    # check falls through when the candidate carries no availability signal.
    observed = candidate.get("observed_at")
    if observed is not None:
        try:
            available = float(observed)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(available):
            return False
    elif candidate.get("session_date_source") == "benchmark_session_date":
        raw = str(candidate.get("session_date") or "")
        try:
            available = datetime.combine(
                date.fromisoformat(raw), datetime.min.time(), tzinfo=timezone.utc
            ).timestamp()
        except ValueError:
            return False
    else:
        return True
    return available <= boundary


def _temporally_eligible_candidates(
    candidates: Sequence[Mapping[str, Any]], contract: AnswerContract
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        if not _available_as_of(candidate, contract):
            continue
        if contract.temporal_window is not None:
            event_day, basis = _candidate_event_day(candidate)
            if event_day is None:
                continue
            if not (
                contract.temporal_window.start
                <= event_day
                < contract.temporal_window.end
            ):
                continue
            candidate = {**candidate, "temporal_match_basis": basis}
        eligible.append(dict(candidate))
    return eligible


def _select_state(
    candidates: Sequence[Mapping[str, Any]], contract: AnswerContract
) -> tuple[dict[str, Any] | None, str]:
    dated: list[tuple[float, str, Mapping[str, Any]]] = []
    for candidate in candidates:
        if "answer" not in candidate.get("slot_names", []):
            continue
        timestamp, basis = _candidate_time(candidate)
        if timestamp is not None:
            dated.append((timestamp, basis, candidate))
    if not dated:
        return None, "state_time_unknown"
    dated.sort(key=lambda item: (item[0], int(item[2]["store_id"])))
    if contract.operation == "previous":
        explicit_transitions = [
            item for item in dated if item[2].get("transition_explicit") is True
        ]
        if explicit_transitions:
            timestamp, basis, chosen = explicit_transitions[-1]
            return {
                **chosen,
                "selection_time": timestamp,
                "selection_time_basis": basis,
            }, "previous_state_selected_from_explicit_transition"

    by_time: dict[float, list[tuple[float, str, Mapping[str, Any]]]] = {}
    for item in dated:
        by_time.setdefault(item[0], []).append(item)
    frontier: list[tuple[float, str, Mapping[str, Any]]] = []
    seen_values: set[str] = set()
    for timestamp in sorted(by_time, reverse=True):
        bucket = by_time[timestamp]
        values: dict[str, list[tuple[float, str, Mapping[str, Any]]]] = {}
        for item in bucket:
            key = " ".join(str(item[2].get("value") or "").casefold().split())
            if key:
                values.setdefault(key, []).append(item)
        novel_values = [key for key in values if key not in seen_values]
        if len(novel_values) > 1:
            return None, "state_conflicted"
        if not novel_values:
            continue
        key = novel_values[0]
        chosen_item = min(
            values[key],
            key=lambda item: (
                item[2].get("origin") != "baseline",
                int(item[2]["store_id"]),
            ),
        )
        frontier.append(chosen_item)
        seen_values.add(key)
        if contract.operation == "latest" or len(frontier) == 2:
            break

    index = 0 if contract.operation == "latest" else 1
    if len(frontier) <= index:
        return None, "previous_state_not_proven" if index else "latest_state_not_proven"
    timestamp, basis, chosen = frontier[index]
    return {
        **chosen,
        "selection_time": timestamp,
        "selection_time_basis": basis,
    }, "latest_state_selected" if index == 0 else "previous_state_selected"


def _slot_operands(
    candidates: Sequence[Mapping[str, Any]], contract: AnswerContract
) -> tuple[list[dict[str, Any]] | None, str]:
    slots = list(contract.slots)

    def semantic_value(item: Mapping[str, Any], slot: EvidenceSlot):
        value = item.get("value")
        if slot.value_type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None
            return _normalized_numeric(value, item.get("unit"), contract.requested_unit)
        if slot.value_type == "date":
            parsed = str(item.get("date") or "")
            event_key = str(item.get("key") or "")
            return (event_key, parsed) if parsed and event_key else ((parsed,) if parsed else None)
        text = " ".join(str(value or "").casefold().split())
        return (text,) if text else None

    if slots and all(slot.anchor is None for slot in slots):
        compatible: list[Mapping[str, Any]] = []
        seen_candidates: set[tuple[Any, ...]] = set()
        for item in candidates:
            semantic = semantic_value(item, slots[0])
            if semantic is None:
                continue
            key = semantic if contract.operation == "order" else (
                str(item.get("exact_ref") or ""),
                semantic,
            )
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            compatible.append(item)
        if len(compatible) < len(slots):
            return None, f"missing_operand_{len(compatible) + 1}"
        if len(compatible) != len(slots):
            return None, "ambiguous_unanchored_operands"
        ordered = sorted(
            compatible,
            key=lambda item: (int(item["store_id"]), int(item["span_start"])),
        )
        return [dict(item) for item in ordered], "unique_fixed_operands"

    selected: list[dict[str, Any]] = []
    used_refs: set[str] = set()
    for slot in slots:
        options = [
            candidate
            for candidate in candidates
            if slot.name in candidate.get("slot_names", [])
            and semantic_value(candidate, slot) is not None
        ]
        if not options:
            return None, f"missing_{slot.name}"
        best_score = max(int(item.get("anchor_score") or 0) for item in options)
        best = [
            item
            for item in options
            if int(item.get("anchor_score") or 0) == best_score
        ]
        values = {semantic_value(item, slot) for item in best}
        if len(values) != 1:
            return None, f"ambiguous_{slot.name}"
        available = [
            item for item in best if str(item["exact_ref"]) not in used_refs
        ]
        if not available:
            return None, f"non_unique_{slot.name}"
        chosen = min(
            available,
            key=lambda item: (
                item.get("origin") != "baseline",
                int(item["store_id"]),
                int(item["span_start"]),
            ),
        )
        selected.append(dict(chosen))
        used_refs.add(str(chosen["exact_ref"]))
    return selected, "unique_fixed_operands"


def _compute(
    question: str,
    contract: AnswerContract,
    candidates: Sequence[Mapping[str, Any]],
    *,
    engine: Any,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    operands, reason = _slot_operands(candidates, contract)
    if operands is None:
        return None, [], reason
    operation = contract.operation
    if operation not in {"sum", "difference", "date_interval", "order"}:
        return None, [], "operation_not_computable"
    plan = EvidencePlan(
        operation=operation,  # type: ignore[arg-type]
        minimum_operands=len(contract.slots),
        maximum_operands=len(contract.slots),
        exact_operands=len(contract.slots),
        temporal_window=contract.temporal_window,
        question_anchor=(
            date.fromisoformat(contract.question_as_of)
            if contract.question_as_of
            else None
        ),
        interval_unit=(
            contract.requested_unit
            if contract.requested_unit in {"day", "week", "month"}
            else "day"
        ),
        difference_direction=contract.difference_direction,
        order_direction=contract.order_direction,
        requires_complete_evidence=False,
        result_unit=contract.requested_unit,
    )
    raw_operands = []
    for slot, candidate in zip(contract.slots, operands):
        raw = {
            "store_id": candidate["store_id"],
            "span_start": candidate["span_start"],
            "span_end": candidate["span_end"],
            "quote": candidate["quote"],
            "value": candidate.get("value"),
            "unit": candidate.get("unit"),
            "occurrence_time": candidate.get("occurrence_time"),
        }
        if candidate.get("date"):
            raw["date"] = candidate["date"]
        if candidate.get("label"):
            raw["label"] = candidate["label"]
        elif slot.anchor and slot.anchor.casefold() in str(candidate["quote"]).casefold():
            raw["label"] = slot.anchor
        raw_operands.append(raw)
    grounding = ground_evidence(
        raw_operands,
        messages=engine._store,
        assertions=getattr(engine, "_assertions", None),
        as_of=question_date_as_of_epoch(contract.question_as_of),
        session_dates=getattr(engine, "_session_occurrence_dates", None),
    )
    if grounding.status != "grounded":
        return None, operands, f"grounding_failed:{grounding.reason}"
    computed = execute_plan(plan, grounding.operands)
    if computed.status != "computed" or computed.trace is None:
        return None, operands, f"computation_failed:{computed.reason}"
    return computed.trace.as_dict(), operands, "validated_canonical_computation"


def _input_exact_refs(values: Sequence[Any]) -> list[str]:
    refs: list[str] = []
    for raw in values:
        if isinstance(raw, str):
            value = raw
        elif isinstance(raw, Mapping):
            value = str(raw.get("exact_ref") or "")
        else:
            value = ""
        refs.append(value)
    return refs


def _input_ref_ranges(values: Sequence[Any]) -> dict[int, list[tuple[int, int]]]:
    ranges: dict[int, list[tuple[int, int]]] = {}
    for exact_ref in _input_exact_refs(values):
        match = _EXACT_REF_RE.fullmatch(exact_ref)
        if match is None:
            continue
        ranges.setdefault(int(match.group("store_id")), []).append(
            (int(match.group("start")), int(match.group("end")))
        )
    return ranges


def _covered_by_input_ref(
    candidate: Mapping[str, Any], ranges: Mapping[int, Sequence[tuple[int, int]]]
) -> bool:
    try:
        store_id = int(candidate["store_id"])
        start = int(candidate["span_start"])
        end = int(candidate["span_end"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return any(left <= start and end <= right for left, right in ranges.get(store_id, ()))


def _baseline_identity(
    sources: Sequence[Mapping[str, Any]], incoming: Sequence[Any]
) -> dict[str, Any]:
    refs = [str(source["exact_ref"]) for source in sources]
    incoming_refs = _input_exact_refs(incoming)
    return {
        "input_exact_ref_count": len(incoming_refs),
        "input_exact_refs_sha256": _digest(incoming_refs),
        "hydrated_exact_ref_count": len(refs),
        "hydrated_exact_refs_sha256": _digest(refs),
        # Backward-compatible aliases for older report readers.
        "exact_ref_count": len(refs),
        "exact_refs_sha256": _digest(refs),
    }


def _base_result(
    *,
    contract: AnswerContract | None,
    limits: RequirementsBudgets,
    baseline: Sequence[Mapping[str, Any]],
    baseline_input: Sequence[Any],
    started: float,
    status: str = "no_augmentation",
    state: str = "unknown",
    reason_code: str = "not_applicable",
) -> dict[str, Any]:
    return {
        "version": REQUIREMENTS_COMPILER_VERSION,
        "status": status,
        "state": state,
        "reason_code": reason_code,
        "context": None,
        "contract": contract.as_dict() if contract else None,
        "closed_requirements": [],
        "open_requirements": [slot.name for slot in contract.slots] if contract else [],
        "baseline": _baseline_identity(baseline, baseline_input),
        "direct_fact": None,
        "evidence": [],
        "novel_exact_refs": [],
        "finite_coverage": False,
        "coverage_certificate": None,
        "computation": None,
        "computation_sha256": None,
        "retrieval": {
            "calls": 0,
            "queries": [],
            "provider_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "query_tokens_complete": True,
        },
        "budgets": limits.as_dict(),
        "metrics": {
            "candidate_count": 0,
            "hydrated_candidates": len(baseline),
            "admitted_novel_refs": 0,
            "session_loads": 0,
            "added_context_tokens": 0,
            "latency_ms": round((time.perf_counter() - started) * 1_000.0, 3),
        },
        "trace": {
            "version": REQUIREMENTS_COMPILER_VERSION,
            "digest_sha256": None,
            "context_sha256": None,
            "truncated": False,
        },
        "provenance": {
            "implementation": "compile_preanswer_evidence",
            "transport": "deterministic_local",
            "provider": "none",
            "model": "none",
            "selector_calls": 0,
            "storage": "same_lcm_db",
            "baseline_bytes_changed": False,
            "final_prose_cached": False,
        },
    }


def _finish(result: dict[str, Any], *, started: float) -> dict[str, Any]:
    context = result.get("context")
    result["metrics"]["latency_ms"] = round(
        (time.perf_counter() - started) * 1_000.0, 3
    )
    result["metrics"]["added_context_tokens"] = (
        _estimate_tokens(context) if isinstance(context, str) else 0
    )
    result["trace"]["context_sha256"] = (
        hashlib.sha256(context.encode("utf-8")).hexdigest()
        if isinstance(context, str)
        else None
    )
    trace_material = {
        "version": result["version"],
        "status": result["status"],
        "state": result["state"],
        "reason_code": result["reason_code"],
        "contract": {
            "answer_kind": (result.get("contract") or {}).get("answer_kind"),
            "operation": (result.get("contract") or {}).get("operation"),
            "coverage_policy": (result.get("contract") or {}).get("coverage_policy"),
            "slot_count": len((result.get("contract") or {}).get("slots") or []),
        },
        "baseline": result["baseline"],
        "closed_requirement_count": len(result["closed_requirements"]),
        "open_requirement_count": len(result["open_requirements"]),
        "novel_ref_count": len(result["novel_exact_refs"]),
        "finite_coverage": result["finite_coverage"],
        "computation_sha256": result["computation_sha256"],
        "retrieval_calls": result["retrieval"]["calls"],
        "metrics": result["metrics"],
        "truncated": result["trace"]["truncated"],
    }
    result["trace"]["digest_sha256"] = _digest(trace_material)
    return result


def _adjacent_sources(
    baseline: Sequence[Mapping[str, Any]],
    *,
    contract: AnswerContract,
    engine: Any,
    limits: RequirementsBudgets,
    capacity: int,
) -> tuple[list[dict[str, Any]], int]:
    if capacity <= 0:
        return [], 0
    output: list[dict[str, Any]] = []
    seen: set[str] = {str(item["exact_ref"]) for item in baseline}
    loads = 0
    ranked_anchors = sorted(
        baseline,
        key=lambda item: (
            -_anchor_score(
                contract,
                str(item.get("content") or item.get("quote") or ""),
                None,
            ),
            int(item["store_id"]),
        ),
    )
    for anchor in ranked_anchors:
        if len(output) >= capacity:
            break
        session_id = str(anchor.get("session_id") or "")
        if not session_id:
            continue
        anchor_content = str(anchor.get("content") or anchor.get("quote") or "").strip()
        anchor_score = _anchor_score(contract, anchor_content, None)
        question_shaped = bool(
            anchor_content.endswith("?")
            or re.match(
                r"^(?:who|what|when|where|which|how|why|do|did|does|is|are|was|were|can|could|would|should)\b",
                anchor_content,
                re.IGNORECASE,
            )
        )
        if not question_shaped and anchor_score < 2:
            continue
        rows = engine._store.load_session_window(
            session_id,
            anchor_store_id=int(anchor["store_id"]),
            before=1,
            after=1,
        )
        loads += 1
        anchor_role = str(anchor.get("role") or "")
        for row in rows:
            store_id = int(row["store_id"])
            if store_id == int(anchor["store_id"]):
                continue
            role = str(row.get("role") or "unknown")
            if anchor_role in {"user", "assistant"} and role == anchor_role:
                continue
            content = str(row.get("content") or "")[: limits.max_quote_chars]
            raw = {
                "exact_ref": f"lcm:{store_id}:0-{len(content)}",
                "quote": content,
            }
            hydrated = _normalize_ref(raw, engine=engine, origin="adjacent_role_partner")
            if hydrated and hydrated["exact_ref"] not in seen:
                hydrated["paired_anchor_score"] = anchor_score + 8
                marginal = _candidate_inventory([hydrated], contract, limit=limits.max_candidates)
                if not marginal:
                    continue
                seen.add(hydrated["exact_ref"])
                output.append(hydrated)
                if len(output) >= capacity:
                    break
    return output[:capacity], loads


def _active_baseline_frontier(
    baseline: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    contract: AnswerContract,
    limit: int,
) -> list[dict[str, Any]]:
    """Keep every material baseline source, then the best closure nuclei."""
    material_store_ids = {int(item["store_id"]) for item in candidates}
    retained = [dict(item) for item in baseline if int(item["store_id"]) in material_store_ids]
    if len(retained) >= limit:
        return retained[:limit]
    seen = {str(item["exact_ref"]) for item in retained}
    nuclei = sorted(
        baseline,
        key=lambda item: (
            -_anchor_score(
                contract,
                str(item.get("content") or item.get("quote") or ""),
                None,
            ),
            int(item["store_id"]),
        ),
    )
    # Preserve at most half of the active set as non-candidate nuclei so that
    # named closure always has room for an opposite-role or retrieval delta.
    nucleus_limit = max(1, limit // 2)
    for item in nuclei:
        if len(retained) >= limit or len(retained) >= len(material_store_ids) + nucleus_limit:
            break
        ref = str(item["exact_ref"])
        if ref not in seen:
            retained.append(dict(item))
            seen.add(ref)
    return retained


def _query_for(
    contract: AnswerContract,
    *,
    slot: EvidenceSlot | None = None,
    secondary: bool = False,
) -> str:
    terms = [slot.anchor] if slot is not None and slot.anchor else list(contract.anchors)
    if secondary:
        terms.extend(contract.anchors)
    if contract.requested_unit:
        terms.append(contract.requested_unit.replace("_", " "))
    if contract.temporal_window is not None:
        terms.extend(
            [
                contract.temporal_window.start.isoformat(),
                contract.temporal_window.end.isoformat(),
            ]
        )
    return " ".join(dict.fromkeys(term for term in terms if term))[:500]


def _merge_usage(target: dict[str, Any], payload: Mapping[str, Any]) -> None:
    provenance = payload.get("provenance")
    sources = [payload, provenance] if isinstance(provenance, Mapping) else [payload]
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in ("provider_calls", "input_tokens", "output_tokens"):
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                target[key] += max(0, int(value))
        value = source.get("cost_usd")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            target["cost_usd"] += max(0.0, float(value))
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        target["query_tokens_complete"] = False
        return
    calls = metrics.get("embedding_query_calls")
    if isinstance(calls, (int, float)) and not isinstance(calls, bool):
        target["provider_calls"] += max(0, int(calls))
    tokens = metrics.get("embedding_query_tokens")
    if isinstance(tokens, (int, float)) and not isinstance(tokens, bool):
        target["input_tokens"] += max(0, int(tokens))
    if metrics.get("embedding_query_tokens_complete") is not True:
        target["query_tokens_complete"] = False


def _retrieve_sources(
    contract: AnswerContract,
    *,
    retrieve: Callable[[dict[str, Any]], Any],
    engine: Any,
    seen_refs: Sequence[str],
    open_slots: Sequence[EvidenceSlot],
    capacity: int,
    limits: RequirementsBudgets,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    if capacity <= 0:
        return []
    output: list[dict[str, Any]] = []
    seen = set(seen_refs)
    queries: list[str] = []
    slots = list(open_slots) or [None]
    for index in range(limits.max_retrieval_calls):
        slot = slots[min(index, len(slots) - 1)]
        query = _query_for(contract, slot=slot, secondary=index == 1)
        if not query or query in queries:
            continue
        queries.append(query)
        args = {
            "query": query,
            "include": "verbatim",
            "detail": "answer_ready",
            "limit": min(limits.max_hydrated_candidates, capacity),
            "scope_bias": 0.0,
            "seen_refs": sorted(seen),
            "include_occurrence_time": True,
        }
        result["retrieval"]["calls"] += 1
        result["retrieval"]["queries"].append(
            {"query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest()}
        )
        try:
            raw = retrieve(args)
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            result["retrieval"]["query_tokens_complete"] = False
            continue
        if not isinstance(payload, Mapping):
            result["retrieval"]["query_tokens_complete"] = False
            continue
        _merge_usage(result["retrieval"], payload)
        hits = payload.get("hits")
        for hit in hits if isinstance(hits, list) else []:
            hydrated = _normalize_ref(hit, engine=engine, origin="targeted_retrieval")
            if hydrated is None or hydrated["exact_ref"] in seen:
                continue
            marginal = _temporally_eligible_candidates(
                _extract_candidates([hydrated], contract), contract
            )
            open_names = {item.name for item in open_slots}
            if not marginal or (
                open_names
                and not any(
                    open_names.intersection(candidate.get("slot_names") or [])
                    for candidate in marginal
                )
            ):
                continue
            seen.add(hydrated["exact_ref"])
            output.append(hydrated)
            if len(output) >= capacity:
                return output
    return output


def _unit_forms(unit: str | None) -> tuple[str, ...]:
    if not unit:
        return ()
    base = unit.replace("_", " ")
    irregular = {
        "person": ("person", "people", "persons"),
        "property": ("property", "properties"),
        "clothing_item": ("clothing", "clothes", "clothing item", "clothing items"),
    }
    return irregular.get(unit, (base, f"{base}s"))


def _material_clauses(content: str, unit: str | None) -> list[tuple[int, int, str]]:
    forms = _unit_forms(unit)
    if not content or not forms:
        return []
    matcher = re.compile(
        r"\b(?:" + "|".join(re.escape(form) for form in forms) + r")\b",
        re.IGNORECASE,
    )
    clauses: list[tuple[int, int, str]] = []
    for match in matcher.finditer(content):
        start = max(
            content.rfind(".", 0, match.start()),
            content.rfind("!", 0, match.start()),
            content.rfind("?", 0, match.start()),
            content.rfind("\n", 0, match.start()),
        ) + 1
        endings = [
            value
            for marker in (".", "!", "?", "\n")
            if (value := content.find(marker, match.end())) >= 0
        ]
        end = min(endings) + 1 if endings else len(content)
        while start < end and content[start].isspace():
            start += 1
        while end > start and content[end - 1].isspace():
            end -= 1
        clause = content[start:end]
        if clause and not any(existing[0] == start and existing[1] == end for existing in clauses):
            clauses.append((start, end, clause))
    return clauses


def _finite_event_key(
    quote: str,
    unit: str | None,
    resolved_date: str | None = None,
) -> str | None:
    if not unit:
        return None
    blocked = {
        "A",
        "An",
        "During",
        "I",
        "In",
        "My",
        "On",
        "The",
        "This",
        "We",
    }
    names = [
        name.casefold()
        for name in re.findall(
            r"\b[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,2}\b", quote
        )
        if name not in blocked
    ]
    if names:
        base = " ".join([unit.replace("_", " "), *dict.fromkeys(names)])
    else:
        explicit = _DATE_TEXT_RE.search(quote)
        if explicit is None:
            return None
        base = f"{unit.replace('_', ' ')} {explicit.group(0).casefold()}"
    def bounded_key(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        marker = f" #{digest}"
        return f"{value[:limit - len(marker)]}{marker}"

    if resolved_date:
        suffix = f" @ {resolved_date}"
        return f"{bounded_key(base, 300 - len(suffix))}{suffix}"
    return bounded_key(base, 300)


def _question_event_verbs(question: str | None) -> str | None:
    normalized = " ".join(str(question or "").casefold().split())
    actions = (
        (r"\battend(?:ed|ing)?\b", r"attended"),
        (r"\bvisit(?:ed|ing)?\b", r"visited"),
        (r"\b(?:go|went|going)\s+to\b", r"went to"),
        (r"\b(?:take|took|taking)\b", r"took"),
        (r"\bview(?:ed|ing)?\b", r"viewed"),
        (r"\badd(?:ed|ing)?\b", r"added"),
        (r"\b(?:buy|bought|buying)\b", r"bought"),
        (r"\breturn(?:ed|ing)?\s+from\b", r"returned from"),
        (r"\bparticipat(?:e|ed|ing)\s+in\b", r"participated in"),
        (r"\bcomplet(?:e|ed|ing)\b", r"completed"),
        (r"\bjoin(?:ed|ing)?\b", r"joined"),
        (r"\btravel(?:ed|led|ing)?\s+to\b", r"traveled to"),
    )
    matched = [source for pattern, source in actions if re.search(pattern, normalized)]
    return "(?:" + "|".join(matched) + ")" if matched else None


def _source_event_clause(
    quote: str,
    *,
    role: Any,
    unit: str | None,
    question: str | None = None,
) -> bool:
    """Return true only for a first-person, source-stated event occurrence."""
    if str(role or "").casefold() != "user" or not unit:
        return False
    if quote.rstrip().endswith("?"):
        return False
    normalized = " ".join(quote.casefold().split())
    if re.search(
        r"\b(?:might|may|plan(?:ning)? to|want to|hope to|would|could|should)\b",
        normalized,
    ):
        return False
    forms = "|".join(re.escape(form) for form in _unit_forms(unit))
    actions = (
        r"attend(?:ed)?|visit(?:ed)?|go(?:ne)? to|went to|take|took|taken|"
        r"view(?:ed)?|add(?:ed)?|buy|bought|return(?:ed)? from|"
        r"participat(?:e|ed) in|complet(?:e|ed)|join(?:ed)?|"
        r"travel(?:led|ed)? to"
    )
    contraction = r"(?:didn|don|doesn|haven|hasn)(?:['’]|\s+)t"
    if re.search(
        rf"\b(?:i|we)\s+(?:(?:did|do|does|have|has)\s+not|"
        rf"{contraction}|never)\s+(?:{actions})\b",
        normalized,
    ):
        return False
    if re.search(
        rf"\b(?:i|we)\s+(?:{actions})\s+(?:no|zero)\s+(?:{forms})\b|"
        rf"\b(?:i|we)\s+(?:{actions})\s+none\s+of\s+"
        rf"(?:(?:my|our|the)\s+)?(?:{forms})\b|"
        rf"\b(?:i|we)\s+(?:went|returned|travelled|traveled)\s+without\s+"
        rf"(?:(?:a|an|any|the)\s+)?(?:{forms})\b|"
        rf"\b(?:i|we)\s+(?:went|returned|travelled|traveled)\s+without\s+"
        rf"(?:attending|visiting|going to|taking|viewing|adding|"
        rf"buying|returning from|participating in|completing|joining|"
        rf"travelling to|traveling to)\s+(?:(?:a|an|any|the)\s+)?"
        rf"(?:{forms})\b",
        normalized,
    ):
        return False
    # ours: the positive match binds to the verbs the question actually asked
    # about (unit-specific for vacation-like units); theirs: the negation guards
    # above scope "never/didn't/no/none/without" to the event assertion.
    event_verbs = r"(?:took|went on|returned from|completed)"
    if unit not in {"vacation", "holiday", "trip"}:
        event_verbs = _question_event_verbs(question) or (
            r"(?:attended|visited|went to|took|returned from|participated in|"
            r"completed|joined|traveled to)"
        )
    if re.search(
        rf"\b(?:i|we)\s+{event_verbs}\s+"
        rf"(?:(?!(?:for|during|before|after|about|because)\b)[\w'-]+\s+){{0,3}}"
        rf"(?:{forms})\b",
        normalized,
        re.IGNORECASE,
    ):
        return True
    # A first-person action placed *inside* the counted event asserts that the
    # event happened, whatever the action's own verb was: "I visited Lisbon
    # during my spring vacation" is a vacation. "during" is containment and is
    # the only preposition in the blocked list above that carries it -- "for my
    # vacation" is a purpose, not an occurrence, so "I bought clothes for my
    # vacation" stays out of the count.
    return bool(
        re.search(
            rf"\b(?:i|we)\s+[\w'-]+(?:\s+[\w'-]+){{0,4}}\s+during\s+"
            rf"(?:my|our)\s+(?:[\w'-]+\s+){{0,2}}(?:{forms})\b",
            normalized,
            re.IGNORECASE,
        )
    )


def _finite_enumeration(
    question: str,
    contract: AnswerContract,
    *,
    engine: Any,
    scan_limit: int = 4096,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None, str]:
    if (
        contract.coverage_policy != "source_asserted_or_finite_enumeration"
        or contract.temporal_window is None
        or contract.requested_unit is None
    ):
        return None, [], None, "finite_enumeration_not_applicable"
    scan = engine._store.scan_evidence_rows(limit=scan_limit)
    certificate = {
        "policy": "whole_corpus_bounded_exact_rows_v1",
        "snapshot_max_store_id": int(scan.get("snapshot_max_store_id") or 0),
        "total_rows": int(scan.get("total_rows") or 0),
        "scanned_rows": int(scan.get("returned_rows") or 0),
        "truncated": bool(scan.get("truncated")),
        "material_clauses": 0,
        "unknown_time_clauses": 0,
        "unavailable_as_of_clauses": 0,
        "ungrounded_key_clauses": 0,
        "distinct_keys": 0,
        "time_bases": [],
        "adapter_time_used": False,
        "every_counted_event_trusted": False,
        "every_counted_event_dated": False,
    }
    if certificate["truncated"]:
        return None, [], certificate, "finite_scan_truncated"

    candidates: list[dict[str, Any]] = []
    time_bases: set[str] = set()
    for row in scan.get("rows") or []:
        content = str(row.get("content") or "")
        for start, end, clause in _material_clauses(content, contract.requested_unit):
            if not _source_event_clause(
                clause,
                role=row.get("role"),
                unit=contract.requested_unit,
                question=question,
            ):
                continue
            certificate["material_clauses"] += 1
            hydrated = _normalize_ref(
                {
                    "exact_ref": f"lcm:{int(row['store_id'])}:{start}-{end}",
                    "quote": clause,
                },
                engine=engine,
                origin="finite_corpus_scan",
            )
            if hydrated is None:
                certificate["ungrounded_key_clauses"] += 1
                continue
            if not _available_as_of(hydrated, contract):
                certificate["unavailable_as_of_clauses"] += 1
                continue
            event_day, basis = _candidate_event_day(hydrated)
            if event_day is None:
                certificate["unknown_time_clauses"] += 1
                resolved_date = None
            else:
                time_bases.add(basis)
                if not (
                    contract.temporal_window.start
                    <= event_day
                    < contract.temporal_window.end
                ):
                    continue
                resolved_date = event_day.isoformat()
            grounding_key = _finite_event_key(clause, contract.requested_unit)
            dedupe_key = _finite_event_key(
                clause,
                contract.requested_unit,
                resolved_date,
            )
            if grounding_key is None or dedupe_key is None:
                certificate["ungrounded_key_clauses"] += 1
                continue
            candidates.append(
                {
                    **hydrated,
                    "key": grounding_key,
                    # ours names the undated key "grounding_key"; theirs names
                    # the dated key "dedupe_key". Both names are published so
                    # every consumer on either side resolves.
                    "grounding_key": grounding_key,
                    "dedupe_key": dedupe_key,
                    "unit": contract.requested_unit,
                    "value": None,
                    "date": resolved_date,
                    "time_basis": basis,
                }
            )

    certificate["time_bases"] = sorted(time_bases)
    certificate["adapter_time_used"] = "adapter_session_date" in time_bases
    if certificate["material_clauses"] == 0:
        return None, [], certificate, "finite_no_material_events"
    if certificate["ungrounded_key_clauses"]:
        return None, [], certificate, "finite_event_keys_unproven"

    # Engine finite-scan certification uses dedupe_key uniqueness; caller-evidence
    # certification uses exact_ref uniqueness. These are intentionally different surfaces.
    by_key: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        by_key.setdefault(str(candidate["dedupe_key"]), candidate)
    operands = list(by_key.values())
    certificate["distinct_keys"] = len(operands)
    certificate["every_counted_event_dated"] = all(
        bool(item.get("date")) for item in operands
    )
    if not operands:
        return None, [], certificate, "finite_no_events_in_window"
    raw_operands = [
        {
            "store_id": item["store_id"],
            "span_start": item["span_start"],
            "span_end": item["span_end"],
            "quote": item["quote"],
            "key": item["grounding_key"],
            "unit": item["unit"],
            "occurrence_time": item["occurrence_time"],
        }
        for item in operands
    ]
    grounding = ground_evidence(
        raw_operands,
        messages=engine._store,
        assertions=getattr(engine, "_assertions", None),
        as_of=question_date_as_of_epoch(contract.question_as_of),
        session_dates=getattr(engine, "_session_occurrence_dates", None),
        engine=engine,
    )
    if grounding.status != "grounded":
        return None, operands, certificate, f"finite_grounding_failed:{grounding.reason}"
    certificate["every_counted_event_trusted"] = (
        grounding.temporal_certified is not False
    )
    computed_operands = tuple(
        replace(grounded, key=str(candidate["dedupe_key"]))
        for grounded, candidate in zip(grounding.operands, operands, strict=True)
    )
    plan = EvidencePlan(
        operation="count_distinct",
        minimum_operands=len(operands),
        maximum_operands=len(operands),
        exact_operands=len(operands),
        requires_complete_evidence=True,
    )
    computed = execute_plan(plan, computed_operands)
    if computed.status != "computed" or computed.trace is None:
        return None, operands, certificate, f"finite_computation_failed:{computed.reason}"
    certificate["certificate_sha256"] = _digest(certificate)
    reason = (
        "finite_coverage_product_verified"
        if (
            certificate["every_counted_event_dated"]
            and certificate["every_counted_event_trusted"]
        )
        else "finite_count_uncertified_undated_events"
    )
    return computed.trace.as_dict(), operands, certificate, reason


def _render_fact(candidate: Mapping[str, Any], contract: AnswerContract) -> str:
    value = candidate.get("value")
    unit = candidate.get("unit")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if unit == "usd":
            stated = f"${value:g}"
        else:
            stated = f"{value:g}" + (f" {unit}" if unit else "")
    else:
        stated = str(value or "").strip()
    return "\n".join(
        [
            f'<lcm-answer-brief version="{REQUIREMENTS_COMPILER_VERSION}">',
            "Product-validated source-stated answer evidence:",
            f"- source value: {stated}",
            f"- exact evidence: [{candidate['exact_ref']}] {candidate['quote']}",
            f"- answer kind: {contract.answer_kind}",
            "Use only this cited evidence; do not infer exhaustive coverage.",
            "</lcm-answer-brief>",
        ]
    )


def _render_computation(
    computation: Mapping[str, Any], *, uncertified: bool = False
) -> str:
    refs = [str(item) for item in computation.get("citations") or []]
    lines = [
        f'<lcm-answer-brief version="{REQUIREMENTS_COMPILER_VERSION}">',
        "Product-validated canonical computation:",
        f"- result: {computation.get('result')}",
        f"- exact operands: {', '.join(refs)}",
    ]
    if uncertified:
        lines.append(
            "- certification: UNCERTIFIED because at least one counted event has "
            "no trusted occurrence date; this count is not certified as exhaustive "
            "for the requested time window."
        )
    lines.extend(
        [
            (
                "Use the canonical result unchanged; preserve the UNCERTIFIED "
                "disclosure and do not add or alter operands."
                if uncertified
                else "Use the canonical result unchanged; do not add or alter operands."
            ),
            "</lcm-answer-brief>",
        ]
    )
    return "\n".join(lines)


def _deliver(
    result: dict[str, Any],
    *,
    state: str,
    reason_code: str,
    context: str,
    evidence: Sequence[Mapping[str, Any]],
    novel_refs: Sequence[str],
    computation: Mapping[str, Any] | None = None,
) -> None:
    max_tokens = int(result["budgets"]["max_added_context_tokens"])
    if _estimate_tokens(context) > max_tokens:
        result["reason_code"] = "context_budget_exhausted"
        result["trace"]["truncated"] = True
        return
    result.update(
        {
            "status": "compiled",
            "state": state,
            "reason_code": reason_code,
            "context": context,
            "evidence": [
                {
                    key: item[key]
                    for key in (
                        "exact_ref",
                        "quote",
                        "role",
                        "session_date",
                        "origin",
                    )
                    if item.get(key) is not None
                }
                for item in evidence[: int(result["budgets"]["max_novel_refs"])]
            ],
            "novel_exact_refs": list(novel_refs)[: int(result["budgets"]["max_novel_refs"])],
            "closed_requirements": [
                slot["name"] for slot in (result.get("contract") or {}).get("slots") or []
            ],
            "open_requirements": [],
            "computation": dict(computation) if computation else None,
            "computation_sha256": _digest(computation) if computation else None,
        }
    )


def _record_baseline_fact(
    result: dict[str, Any],
    candidate: Mapping[str, Any],
    contract: AnswerContract,
    *,
    render_context: bool,
    time_basis_field: str | None = None,
) -> None:
    direct_fact = {
        "value": candidate["value"],
        "unit": candidate.get("unit"),
        "exact_ref": candidate["exact_ref"],
    }
    if time_basis_field is not None:
        direct_fact["time_basis"] = candidate.get(time_basis_field)
    if render_context:
        _deliver(
            result,
            state="answer_sufficient",
            reason_code="baseline_already_answer_sufficient",
            context=_render_fact(candidate, contract),
            evidence=[candidate],
            novel_refs=(),
        )
        result["direct_fact"] = direct_fact
        return
    result.update(
        {
            "state": "answer_sufficient",
            "reason_code": "baseline_already_answer_sufficient",
            "direct_fact": direct_fact,
            "closed_requirements": ["answer"],
            "open_requirements": [],
        }
    )


def compile_preanswer_evidence(
    question: Any,
    *,
    engine: Any,
    baseline_refs: Sequence[Any] = (),
    question_as_of: Any = None,
    question_date: Any = None,
    retrieve: Callable[[dict[str, Any]], Any] | None = None,
    enabled: bool = False,
    render_baseline_context: bool = False,
    budgets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile one bounded exact evidence brief or return an unchanged baseline."""
    started = time.perf_counter()
    limits = _budgets(budgets)
    as_of = question_as_of if question_as_of is not None else question_date
    decision = compile_answer_contract(question, as_of)
    contract = decision.contract

    baseline_input = list(baseline_refs)
    ranked_baseline_input = _rank_baseline_inputs(baseline_input, contract)
    baseline: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in ranked_baseline_input[: limits.max_candidates]:
        hydrated = _normalize_ref(raw, engine=engine, origin="baseline")
        if hydrated is not None and hydrated["exact_ref"] not in seen:
            seen.add(hydrated["exact_ref"])
            baseline.append(hydrated)
            if len(baseline) >= limits.max_hydrated_candidates:
                break
    result = _base_result(
        contract=contract,
        limits=limits,
        baseline=baseline,
        baseline_input=baseline_input,
        started=started,
        reason_code=decision.reason_code,
    )
    if not enabled:
        result["reason_code"] = "feature_disabled"
        return _finish(result, started=started)
    if decision.status != "planned" or contract is None:
        result["reason_code"] = decision.reason_code
        return _finish(result, started=started)

    baseline_candidates = _candidate_inventory(
        baseline, contract, limit=limits.max_candidates
    )
    result["metrics"]["candidate_count"] = len(baseline_candidates)

    if contract.operation == "scalar":
        direct, reason = _select_scalar(baseline_candidates, contract)
        if direct is not None:
            _record_baseline_fact(
                result,
                direct,
                contract,
                render_context=render_baseline_context,
            )
            return _finish(result, started=started)
    elif contract.operation in {"sum", "difference", "date_interval", "order"}:
        direct, _ = _select_scalar(baseline_candidates, contract)
        if direct is not None:
            _record_baseline_fact(
                result,
                direct,
                contract,
                render_context=render_baseline_context,
            )
            return _finish(result, started=started)
        computation, operands, reason = _compute(
            str(question), contract, baseline_candidates, engine=engine
        )
        if computation is not None:
            _deliver(
                result,
                state="computation_sufficient",
                reason_code=reason,
                context=_render_computation(computation),
                evidence=operands,
                novel_refs=(),
                computation=computation,
            )
            return _finish(result, started=started)
    elif contract.operation in {"latest", "previous"}:
        chosen, reason = _select_state(baseline_candidates, contract)
        if chosen is not None:
            _record_baseline_fact(
                result,
                chosen,
                contract,
                render_context=render_baseline_context,
                time_basis_field="selection_time_basis",
            )
            return _finish(result, started=started)
    elif contract.operation == "date_filter":
        chosen, reason = _select_temporal_event(baseline_candidates, contract)
        if chosen is not None:
            _record_baseline_fact(
                result,
                chosen,
                contract,
                render_context=render_baseline_context,
                time_basis_field="temporal_match_basis",
            )
            return _finish(result, started=started)
    else:
        direct, reason = _select_text(baseline_candidates, contract)
        if direct is not None:
            _record_baseline_fact(
                result,
                direct,
                contract,
                render_context=render_baseline_context,
            )
            return _finish(result, started=started)

    active_baseline = _active_baseline_frontier(
        baseline,
        baseline_candidates,
        contract=contract,
        limit=limits.max_hydrated_candidates,
    )
    adjacent, loads = _adjacent_sources(
        baseline,
        contract=contract,
        engine=engine,
        limits=limits,
        capacity=max(0, limits.max_hydrated_candidates - len(active_baseline)),
    )
    result["metrics"]["session_loads"] = loads
    novel_sources = list(adjacent)
    all_sources = [*active_baseline, *novel_sources]
    candidates = _candidate_inventory(
        all_sources, contract, limit=limits.max_candidates
    )

    def attempt():
        if contract.operation == "scalar":
            fact, why = _select_scalar(candidates, contract)
            return fact, None, [fact] if fact else [], why
        if contract.operation in {"sum", "difference", "date_interval", "order"}:
            fact, direct_why = _select_scalar(candidates, contract)
            if fact is not None:
                return fact, None, [fact], direct_why
            computation, operands, why = _compute(
                str(question), contract, candidates, engine=engine
            )
            return None, computation, operands, why
        if contract.operation in {"latest", "previous"}:
            fact, why = _select_state(candidates, contract)
            return fact, None, [fact] if fact else [], why
        if contract.operation == "date_filter":
            fact, why = _select_temporal_event(candidates, contract)
            return fact, None, [fact] if fact else [], why
        fact, why = _select_text(candidates, contract)
        return fact, None, [fact] if fact else [], why

    direct, computation, selected, reason = attempt()
    if (
        direct is None
        and computation is None
        and retrieve is not None
        and limits.max_retrieval_calls
    ):
        capacity = max(0, limits.max_hydrated_candidates - len(all_sources))
        input_seen_refs = [
            exact_ref
            for exact_ref in _input_exact_refs(baseline_input)
            if _EXACT_REF_RE.fullmatch(exact_ref)
        ]
        retrieved = _retrieve_sources(
            contract,
            retrieve=retrieve,
            engine=engine,
            seen_refs=[*input_seen_refs, *[item["exact_ref"] for item in all_sources]],
            open_slots=_open_slots(candidates, contract),
            capacity=capacity,
            limits=limits,
            result=result,
        )
        novel_sources.extend(retrieved)
        all_sources.extend(retrieved)
        candidates = _candidate_inventory(
            all_sources, contract, limit=limits.max_candidates
        )
        direct, computation, selected, reason = attempt()

    coverage_certificate = None
    if (
        direct is None
        and computation is None
        and contract.operation == "scalar"
        and contract.coverage_policy == "source_asserted_or_finite_enumeration"
        and contract.temporal_window is not None
        and contract.requested_unit is not None
    ):
        computation, selected, coverage_certificate, reason = _finite_enumeration(
            str(question),
            contract,
            engine=engine,
            scan_limit=limits.max_scan_rows,
        )
        if coverage_certificate is not None:
            result["coverage_certificate"] = coverage_certificate
        if computation is not None:
            result["finite_coverage"] = bool(
                coverage_certificate
                and coverage_certificate.get("every_counted_event_dated")
                and coverage_certificate.get("every_counted_event_trusted")
            )

    result["metrics"].update(
        {
            "candidate_count": len(candidates),
            "hydrated_candidates": len(all_sources),
        }
    )
    if direct is None and computation is None:
        result["reason_code"] = reason
        return _finish(result, started=started)

    baseline_ranges = _input_ref_ranges(baseline_input)
    selected_items = [item for item in selected if isinstance(item, Mapping)]
    novel_selected_all = [
        str(item["exact_ref"])
        for item in selected_items
        if not _covered_by_input_ref(item, baseline_ranges)
    ]
    novel_selected_all = list(dict.fromkeys(novel_selected_all))
    if len(novel_selected_all) > limits.max_novel_refs:
        result["reason_code"] = "novel_ref_budget_exhausted"
        return _finish(result, started=started)
    novel_selected = novel_selected_all
    result["metrics"]["admitted_novel_refs"] = len(novel_selected)
    if computation is not None:
        _deliver(
            result,
            state="computation_sufficient",
            reason_code=reason,
            context=_render_computation(
                computation,
                uncertified=(
                    coverage_certificate is not None
                    and not result["finite_coverage"]
                ),
            ),
            evidence=selected_items,
            novel_refs=novel_selected,
            computation=computation,
        )
    elif direct is not None and novel_selected:
        result["direct_fact"] = {
            "value": direct["value"],
            "unit": direct.get("unit"),
            "exact_ref": direct["exact_ref"],
            **(
                {"time_basis": direct["selection_time_basis"]}
                if direct.get("selection_time_basis")
                else {}
            ),
        }
        _deliver(
            result,
            state="answer_sufficient",
            reason_code=reason,
            context=_render_fact(direct, contract),
            evidence=[direct],
            novel_refs=novel_selected,
        )
    else:
        result.update(
            {
                "state": "answer_sufficient",
                "reason_code": "baseline_already_answer_sufficient",
                "direct_fact": {
                    "value": direct["value"] if direct else None,
                    "unit": direct.get("unit") if direct else None,
                    "exact_ref": direct["exact_ref"] if direct else None,
                },
                "closed_requirements": [slot.name for slot in contract.slots],
                "open_requirements": [],
            }
        )
    return _finish(result, started=started)
