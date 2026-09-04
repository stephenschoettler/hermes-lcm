"""AMR (auditable-memory-records 0.1) citation verification for assertions.

Read-only companion to :mod:`assertion_store`: it verifies stored quote
anchors against the authoritative message content and reports the four AMR
§6 outcomes. It never mutates state and never resolves disagreement between
records — verification answers only "does this citation still hold?".

Outcome semantics (AMR §6 — collapsing these into a boolean is the classic
mistake; ``anchor_tampered`` and ``source_drifted`` look identical to a
naive "citation invalid" check and mean opposite things):

- ``ok`` — stored hash matches the stored quote, and the quote is still at
  its span in the unchanged source.
- ``anchor_tampered`` — the record's own citation is internally
  inconsistent (stored hash disagrees with the stored quote, or the
  span/quote pair disagrees with a provably unchanged source).
- ``source_drifted`` — the record is honest, but the source changed
  underneath it.
- ``source_missing`` — the cited source cannot be resolved.

Level-1 marking is carried by the closed ``epistemic`` vocabulary; ``None``
is the unmarked state and is never coerced to ``fact`` (absent is not
fact). This module also exposes the per-implementation AMR declaration.
"""

from __future__ import annotations

from typing import Any, Sequence

from .assertion_store import AssertionStore
from .db_bootstrap import (
    ASSERTION_QUOTE_HASH_ALGORITHM,
    _normalize_quote_for_hash,
    _quote_hash,
)

AMR_SPEC_DECLARATION = "auditable_memory: 0.1"
AMR_CONFORMANCE_LEVEL = "1-marked"

OUTCOME_OK = "ok"
OUTCOME_ANCHOR_TAMPERED = "anchor_tampered"
OUTCOME_SOURCE_DRIFTED = "source_drifted"
OUTCOME_SOURCE_MISSING = "source_missing"


def _content_sha256(content: str) -> str:
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def verify_assertion_citations(
    store: AssertionStore,
    *,
    assertion_ids: Sequence[str] | None = None,
    source_store_id: int | None = None,
    include_invalidated: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Verify stored quote anchors, returning one outcome row per assertion.

    Read-only. Results are ordered by assertion_id for determinism. The
    check recomputes the AMR §5 hash under the stored hash's own algorithm
    prefix; unrecognized prefixes are reported as ``anchor_tampered``
    rather than guessed.
    """
    bounded_limit = int(limit)
    if not 1 <= bounded_limit <= 500:
        raise ValueError("limit must be between 1 and 500")

    where = ["1"]
    args: list[Any] = []
    if assertion_ids is not None:
        if not assertion_ids:
            return []
        normalized = [str(value).strip().lower() for value in assertion_ids]
        placeholders = ", ".join("?" for _ in normalized)
        where.append(f"a.assertion_id IN ({placeholders})")
        args.extend(normalized)
    if source_store_id is not None:
        where.append("a.source_store_id = ?")
        args.append(int(source_store_id))
    if not include_invalidated:
        where.append("s.invalidated_at IS NULL")
    args.append(bounded_limit)

    rows = store.connection.execute(
        f"""
        SELECT a.assertion_id, a.source_store_id, a.source_span_start,
               a.source_span_end, a.source_quote, a.source_quote_hash,
               s.source_content_sha256, s.invalidated_at, m.content
          FROM lcm_assertions AS a
          JOIN lcm_assertion_sources AS s
            ON s.source_store_id = a.source_store_id
           AND s.extraction_version = a.extraction_version
           AND s.source_content_sha256 = a.source_content_sha256
          LEFT JOIN messages AS m ON m.store_id = a.source_store_id
         WHERE {' AND '.join(where)}
         ORDER BY a.assertion_id ASC
         LIMIT ?
        """,
        args,
    ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        stored_hash = row["source_quote_hash"]
        stored_quote = str(row["source_quote"] or "")
        content = row["content"]
        detail = ""
        partial = False

        if content is None:
            outcome = OUTCOME_SOURCE_MISSING
            detail = "source message row is gone"
        else:
            current_content = str(content)
            content_unchanged = (
                _content_sha256(current_content)
                == str(row["source_content_sha256"])
            )
            if stored_hash is None or str(stored_hash).strip() == "":
                # AMR §6.1: without an anchor hash only the drift axis is
                # decidable; anchor_tampered MUST NOT be reported and the
                # partial check is signalled.
                partial = True
                if not content_unchanged:
                    outcome = OUTCOME_SOURCE_DRIFTED
                    detail = "source content changed; no stored hash to check integrity"
                else:
                    start = int(row["source_span_start"])
                    end = int(row["source_span_end"])
                    if 0 <= start < end <= len(current_content) and (
                        current_content[start:end] == stored_quote
                    ):
                        outcome = OUTCOME_OK
                        detail = "partial check: span verified, no stored hash"
                    else:
                        outcome = OUTCOME_SOURCE_DRIFTED
                        detail = "partial check: span moved, no stored hash"
            else:
                partial = False
                stored_hash = str(stored_hash)
                algorithm = stored_hash.split(":", 1)[0] if ":" in stored_hash else ""
                if algorithm != ASSERTION_QUOTE_HASH_ALGORITHM:
                    outcome = OUTCOME_ANCHOR_TAMPERED
                    detail = f"unrecognized hash algorithm: {algorithm or 'bare digest'}"
                elif _quote_hash(stored_quote) != stored_hash:
                    outcome = OUTCOME_ANCHOR_TAMPERED
                    detail = "stored hash disagrees with the stored quote"
                elif not content_unchanged:
                    present = _normalized_contains(current_content, stored_quote)
                    outcome = OUTCOME_SOURCE_DRIFTED
                    detail = (
                        "source content changed; quote "
                        + ("still present" if present else "no longer present")
                    )
                else:
                    start = int(row["source_span_start"])
                    end = int(row["source_span_end"])
                    if 0 <= start < end <= len(current_content) and (
                        current_content[start:end] == stored_quote
                    ):
                        outcome = OUTCOME_OK
                        detail = "hash and span verified against unchanged source"
                    else:
                        moved = _normalized_offset(current_content, stored_quote)
                        outcome = OUTCOME_ANCHOR_TAMPERED
                        detail = (
                            "span/quote inconsistent with unchanged source"
                            + (
                                f"; quote found at offset {moved}"
                                if moved is not None
                                else "; quote absent from source"
                            )
                        )

        results.append({
            "assertion_id": str(row["assertion_id"]),
            "source_store_id": int(row["source_store_id"]),
            "outcome": outcome,
            "partial": partial,
            "detail": detail,
            "stored_quote_hash": stored_hash,
        })
    return results


def _normalized_contains(content: str, quote: str) -> bool:
    return _normalize_quote_for_hash(quote) in _normalize_quote_for_hash(content)


def _normalized_offset(content: str, quote: str) -> int | None:
    normalized_content = _normalize_quote_for_hash(content)
    normalized_quote = _normalize_quote_for_hash(quote)
    offset = normalized_content.find(normalized_quote)
    return offset if offset >= 0 else None


def verify_relation_citations(
    store: AssertionStore,
    *,
    relation_ids: Sequence[str] | None = None,
    source_store_id: int | None = None,
    include_invalidated: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Same four-outcome verification for typed-relation quote anchors."""
    bounded_limit = int(limit)
    if not 1 <= bounded_limit <= 500:
        raise ValueError("limit must be between 1 and 500")

    where = ["1"]
    args: list[Any] = []
    if relation_ids is not None:
        if not relation_ids:
            return []
        normalized = [str(value).strip().lower() for value in relation_ids]
        placeholders = ", ".join("?" for _ in normalized)
        where.append(f"r.relation_id IN ({placeholders})")
        args.extend(normalized)
    if source_store_id is not None:
        where.append("r.source_store_id = ?")
        args.append(int(source_store_id))
    if not include_invalidated:
        where.append("s.invalidated_at IS NULL")
    args.append(bounded_limit)

    rows = store.connection.execute(
        f"""
        SELECT r.relation_id, r.source_store_id, r.source_span_start,
               r.source_span_end, r.source_quote, r.source_quote_hash,
               s.source_content_sha256, s.invalidated_at, m.content
          FROM lcm_assertion_relations AS r
          JOIN lcm_assertion_sources AS s
            ON s.source_store_id = r.source_store_id
           AND s.extraction_version = r.extraction_version
           AND s.source_content_sha256 = r.source_content_sha256
          LEFT JOIN messages AS m ON m.store_id = r.source_store_id
         WHERE {' AND '.join(where)}
         ORDER BY r.relation_id ASC
         LIMIT ?
        """,
        args,
    ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        stored_hash = row["source_quote_hash"]
        stored_quote = str(row["source_quote"] or "")
        content = row["content"]
        if content is None:
            results.append({
                "relation_id": str(row["relation_id"]),
                "source_store_id": int(row["source_store_id"]),
                "outcome": OUTCOME_SOURCE_MISSING,
                "partial": False,
                "detail": "source message row is gone",
                "stored_quote_hash": stored_hash,
            })
            continue
        current_content = str(content)
        content_unchanged = (
            _content_sha256(current_content) == str(row["source_content_sha256"])
        )
        if stored_hash is None or str(stored_hash).strip() == "":
            partial = True
            if content_unchanged:
                start = int(row["source_span_start"])
                end = int(row["source_span_end"])
                outcome = (
                    OUTCOME_OK
                    if 0 <= start < end <= len(current_content)
                    and current_content[start:end] == stored_quote
                    else OUTCOME_SOURCE_DRIFTED
                )
                detail = "partial check: no stored hash"
            else:
                outcome = OUTCOME_SOURCE_DRIFTED
                detail = "source content changed; no stored hash to check integrity"
        else:
            partial = False
            stored_hash = str(stored_hash)
            algorithm = stored_hash.split(":", 1)[0] if ":" in stored_hash else ""
            if algorithm != ASSERTION_QUOTE_HASH_ALGORITHM:
                outcome = OUTCOME_ANCHOR_TAMPERED
                detail = f"unrecognized hash algorithm: {algorithm or 'bare digest'}"
            elif _quote_hash(stored_quote) != stored_hash:
                outcome = OUTCOME_ANCHOR_TAMPERED
                detail = "stored hash disagrees with the stored quote"
            elif not content_unchanged:
                outcome = OUTCOME_SOURCE_DRIFTED
                detail = "source content changed"
            else:
                start = int(row["source_span_start"])
                end = int(row["source_span_end"])
                if 0 <= start < end <= len(current_content) and (
                    current_content[start:end] == stored_quote
                ):
                    outcome = OUTCOME_OK
                    detail = "hash and span verified against unchanged source"
                else:
                    outcome = OUTCOME_ANCHOR_TAMPERED
                    detail = "span/quote inconsistent with unchanged source"
        results.append({
            "relation_id": str(row["relation_id"]),
            "source_store_id": int(row["source_store_id"]),
            "outcome": outcome,
            "partial": partial,
            "detail": detail,
            "stored_quote_hash": stored_hash,
        })
    return results
