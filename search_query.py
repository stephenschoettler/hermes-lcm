"""Helpers for search query handling across FTS and LIKE fallback paths."""

from __future__ import annotations

import re
from typing import List

_CJK_RE = re.compile(
    r"["
    r"\u3400-\u4dbf"
    r"\u4e00-\u9fff"
    r"\u3000-\u303f"
    r"\u3040-\u30ff"
    r"\uac00-\ud7af"
    r"\uff00-\uffef"
    r"]"
)
_EMOJI_RE = re.compile(
    r"["
    r"\u2600-\u27bf"
    r"\U0001F300-\U0001FAFF"
    r"]"
)
_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"')
_BOOLEAN_OPERATORS = {"AND", "OR", "NOT", "NEAR"}
_RISKY_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9][\-:/][A-Za-z0-9]")
_SPLIT_PUNCT_RE = re.compile(r"[-:/]+")
_STRIP_EDGE_PUNCT = "()[]{}.,;"


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def contains_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text or ""))


def contains_risky_fts_ascii(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if raw.count('"') % 2:
        return True
    text_without_phrases = _QUOTED_PHRASE_RE.sub(" ", raw)
    return bool(_RISKY_FTS_TOKEN_RE.search(text_without_phrases))


def requires_like_fallback(query: str) -> bool:
    return contains_cjk(query) or contains_emoji(query) or contains_risky_fts_ascii(query)


def _token_variants(token: str) -> List[str]:
    cleaned = (token or "").strip().strip(_STRIP_EDGE_PUNCT)
    if not cleaned:
        return []
    if cleaned.upper() in _BOOLEAN_OPERATORS:
        return []

    variants = [cleaned]
    if _SPLIT_PUNCT_RE.search(cleaned):
        parts = [part for part in _SPLIT_PUNCT_RE.split(cleaned) if part]
        if len(parts) > 1:
            variants.extend(parts)

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant not in seen:
            deduped.append(variant)
            seen.add(variant)
    return deduped


def extract_search_terms(query: str) -> List[str]:
    text = (query or "").strip()
    if not text:
        return []

    terms: list[str] = []
    for phrase in _QUOTED_PHRASE_RE.findall(text):
        cleaned = phrase.strip()
        if cleaned:
            terms.append(cleaned)

    text_without_phrases = _QUOTED_PHRASE_RE.sub(" ", text)
    for token in text_without_phrases.split():
        terms.extend(_token_variants(token))

    if not terms:
        fallback_text = text.strip().strip(_STRIP_EDGE_PUNCT)
        if fallback_text:
            terms.append(fallback_text)

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term not in seen:
            deduped.append(term)
            seen.add(term)
    return deduped


def escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def count_term_matches(text: str, term: str) -> int:
    haystack = (text or "")
    needle = (term or "")
    if not haystack or not needle:
        return 0
    return haystack.lower().count(needle.lower())


def build_snippet(text: str, terms: List[str], width: int = 80) -> str:
    content = (text or "")
    if not content:
        return ""
    lowered = content.lower()
    for term in terms:
        if not term:
            continue
        idx = lowered.find(term.lower())
        if idx >= 0:
            start = max(0, idx - width // 2)
            end = min(len(content), idx + len(term) + width // 2)
            snippet = content[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
            return snippet
    return content[:width] + ("..." if len(content) > width else "")
