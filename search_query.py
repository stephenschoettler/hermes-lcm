"""Helpers for search query handling across FTS and LIKE fallback paths."""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, List

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
_CURLY_QUOTED_PHRASE_RE = re.compile(r"“([^”]+)”")
_BOOLEAN_OPERATORS = {"AND", "OR", "NOT", "NEAR"}
_RISKY_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9][\-:/][A-Za-z0-9]")
_SPLIT_PUNCT_RE = re.compile(r"[-:/]+")
_STRIP_EDGE_PUNCT = "\"'()[]{}.,;"
_TRAILING_SENTENCE_PUNCT = "?!"
_PROSE_TERM_LIMIT = 12
_PROSE_SYMBOL_SLOTS = 2
_PROSE_STOPWORDS = frozenset({
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "did", "do", "does", "find", "for", "from", "had", "has",
    "have", "how", "i", "in", "is", "it", "me",
    "my", "near", "not", "of", "on", "or", "our", "please", "recall", "remember",
    "s", "say", "said", "should", "tell", "that", "the", "their", "them", "there",
    "these", "they", "this", "those", "to", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
})
_PROSE_LEAD_WORDS = frozenset({
    "can", "could", "did", "do", "does", "find", "how", "please", "recall",
    "remember", "tell", "what", "when", "where", "which", "who", "why", "would",
})
# Tokenizer debris from possessives and contractions ("dog's" -> [dog, s],
# "didn't" -> [didn, t]). None of these is ever a subject term, and none may
# inflate classification counts. The n't stems are the closed set of tokens
# that are not standalone English words.
_PROSE_FRAGMENTS = frozenset({
    "s", "t", "d", "m", "ll", "ve", "re",
    "didn", "doesn", "isn", "wasn", "aren", "weren", "hasn", "haven", "hadn",
    "wouldn", "couldn", "shouldn", "needn", "mustn", "mightn", "shan", "oughtn",
})
# Stopwords that can never be the SUBJECT of a memory search (pronouns,
# articles, prepositions, verbs of asking). The subject-fallback may return a
# stopword like ``will`` (a noun in possessive-heavy queries) but never one of
# these — a lone framing token as the sole disjunction term broadens recall to
# any row containing a common word.
_PROSE_NEVER_SUBJECT = frozenset({
    "a", "an", "and", "about", "are", "as", "at", "be", "been", "but",
    "by", "for", "from", "had", "has", "have", "i", "in", "is", "it", "me",
    "my", "near", "not", "of", "on", "or", "our", "say", "said", "that",
    "the", "their", "them", "there", "these", "they", "this", "those", "to",
    "us", "was", "we", "were", "with", "you", "your",
})
# Characters that are special in FTS5 QUERY SYNTAX (as opposed to characters
# FTS5 simply cannot spell in a bareword). Only these have to go on the LIKE
# path, which has no query grammar of its own.
_FTS5_SPECIAL_CHARS = frozenset('"()*^-:{}.')  # FTS5 query grammar


def _like_safe_char(char: str) -> str:
    """Map one unquoted character to its LIKE-safe form.

    LIKE is a substring match, so the only thing it needs removed is the
    operator punctuation a user typed FOR the index (quoted phrases, prefix
    ``*``). Everything else is signal it can match on — emoji above all, which
    the FTS term form must drop because unicode61 does not index it, and which
    LIKE is therefore the ONLY way to find.
    """
    return " " if char in _FTS5_SPECIAL_CHARS else char


def _fts5_safe_char(char: str) -> str:
    """Map one unquoted character to its FTS5 bareword-safe form.

    An FTS5 bareword accepts only alphanumerics; every other character is
    either query syntax (``"()*^-:{}.``), a string delimiter (``'``), or a
    plain syntax error (``? , & $ ! % = < ;`` ...). A raw natural-language
    question therefore fails ``MATCH`` outright and used to fall through to the
    LIKE full-scan, which blows the recall deadline at scale and returns
    nothing (F31 §3). Substituting a separator instead lets a question reach
    the index in its term form; the default unicode61 tokenizer splits the
    INDEXED text on exactly the same boundary, so no term is lost by the
    substitution.

    ``str.isalnum()`` alone is NOT that boundary: a combining mark is not
    alphanumeric, so a decomposed ``naïve`` (``nai`` + U+0308) split into
    ``nai ve`` while unicode61 indexes the word as ``naive`` — zero rows where
    the raw query matched. Marks therefore stay inside the token, and
    ``sanitize_fts5_query`` composes the query first.
    """
    if char.isalnum() or char.isspace():
        return char
    return char if unicodedata.category(char).startswith("M") else " "


def _lower_operator_tokens(text: str) -> str:
    return " ".join(
        token.lower() if token in _BOOLEAN_OPERATORS else token
        for token in text.split(" ")
    )


def _neutralize_bare_operators(sanitized: str) -> str:
    """Lowercase bare AND/OR/NOT/NEAR outside quoted phrases.

    FTS5 operators are only operators in UPPERCASE, so lowercasing turns them
    back into ordinary barewords. Without this a raw question SILENTLY ACQUIRES
    boolean semantics it never asked for: ``Portland, OR hotel`` sanitized to
    ``Portland OR hotel`` and broadened to a disjunction, while a leading
    ``NOT ready`` was a syntax error that dumped the query onto the LIKE
    full-scan. Raw and deliberate queries share one unmarked entry point, so the
    raw reading has to be the safe one. Quoted phrases are left alone: an
    explicit ``"NEAR"`` is already a literal, not an operator.
    """
    out: list[str] = []
    last = 0
    for match in _QUOTED_PHRASE_RE.finditer(sanitized):
        out.append(_lower_operator_tokens(sanitized[last:match.start()]))
        out.append(match.group(0))
        last = match.end()
    out.append(_lower_operator_tokens(sanitized[last:]))
    return "".join(out)


def sanitize_fts5_query(query: str, *, allow_operators: bool = False) -> str:
    """Reduce a query to FTS5-safe terms, preserving balanced phrase quotes.

    Composed (NFC) first so a decomposed accent is one alphanumeric character
    rather than a base plus a combining mark, which is what unicode61 folds and
    indexes. The LIKE path deliberately does NOT normalize: it is a literal
    substring match against stored bytes, so it must not re-spell the query.

    ``allow_operators`` is the explicit marker for a query a CALLER composed as
    FTS5 syntax (the benchmark harness joins its barewords with ``OR``). It
    keeps bare AND/OR/NOT/NEAR intact. It must never be set for text that came
    from a user or an agent: the default assumes raw prose, which is the only
    safe reading when the two cannot be told apart.
    """
    composed = unicodedata.normalize("NFC", query or "")
    sanitized = _sanitize_query(composed, _fts5_safe_char)
    return sanitized if allow_operators else _neutralize_bare_operators(sanitized)


def should_use_fts_prose_mode(
    query: str,
    sanitized: str | None = None,
) -> bool:
    """Distinguish conversational prose from compact keyword query shapes."""
    raw = query or ""
    safe = sanitize_fts5_query(raw) if sanitized is None else sanitized
    # A retained ASCII phrase or a balanced smart-quoted phrase is an explicit
    # precision signal. Keep it conjunctive even when the query ends in "?".
    # BALANCED-ONLY by contract (both quote styles): a lone curly quote is far
    # more often a typing accident inside a real question than precision
    # intent, so it must not force a prose-shaped query onto the narrow route.
    if (
        _QUOTED_PHRASE_RE.search(safe)
        or _CURLY_QUOTED_PHRASE_RE.search(raw)
    ):
        return False

    # Classification retains ordinary lowercase conjunctions. The scoring
    # extractor intentionally drops boolean-looking tokens, which previously
    # hid "and"/"or" from the prose stopword threshold.
    # Possessive/contraction debris ("s", "didn", "t") is not a word for
    # classification counts.
    terms = [
        term
        for term in _WORD_RE.findall(safe)
        if term.casefold() not in _PROSE_FRAGMENTS
    ]
    # unicode61 drops these symbols, but in a raw prose request they can be the
    # subject itself (``Find ©?``). Count them for classification without
    # putting them into the FTS query; the prose LIKE extractor retains them.
    terms.extend(
        char
        for char in raw
        if contains_unindexed_unicode_symbol(char)
    )
    if len(terms) < 2:
        return False
    lowered = [term.casefold() for term in terms]
    stopword_count = sum(term in _PROSE_STOPWORDS for term in lowered)
    starts_like_prose = lowered[0] in _PROSE_LEAD_WORDS
    return (
        (
            "?" in raw
            and (
                stopword_count >= 1
                or starts_like_prose
                or len(terms) >= 6
            )
        )
        or starts_like_prose
        or (len(terms) >= 6 and stopword_count >= 2)
    )


def _extract_bounded_prose_terms(sanitized: str) -> List[str]:
    """Drop conversational framing while retaining one subject fallback."""
    candidates = extract_search_terms(
        sanitized,
        normalize_sentence_punctuation=True,
    )
    eligible_terms: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        folded = term.casefold()
        if (
            folded in _PROSE_STOPWORDS
            or folded in _PROSE_FRAGMENTS
            or folded in seen
        ):
            continue
        seen.add(folded)
        eligible_terms.append(term)

    # Reserve a bounded number of slots for the unindexable routing signals
    # that sent the query to LIKE. When the ordinary cap is full, the first
    # symbols evict the lowest-priority (latest) ordinary terms.
    symbol_terms = [
        term
        for term in eligible_terms
        if contains_unindexed_unicode_symbol(term)
    ][:_PROSE_SYMBOL_SLOTS]
    ordinary_limit = _PROSE_TERM_LIMIT - len(symbol_terms)
    ordinary_terms = [
        term
        for term in eligible_terms
        if not contains_unindexed_unicode_symbol(term)
    ][:ordinary_limit]
    selected = set(ordinary_terms) | set(symbol_terms)
    prose_terms = [term for term in eligible_terms if term in selected]

    if prose_terms:
        return prose_terms

    # A broad stoplist can also contain the grammatical subject (for example,
    # ``will``). Prefer the last token that could plausibly BE a subject —
    # never a lead word, tokenizer fragment, or pure framing word. If nothing
    # but framing survives, return no terms: build_fts5_match_query then falls
    # back to the sanitized conjunctive form instead of broadening recall on a
    # lone framing token.
    for term in reversed(candidates):
        folded = term.casefold()
        if (
            folded not in _PROSE_LEAD_WORDS
            and folded not in _PROSE_FRAGMENTS
            and folded not in _PROSE_NEVER_SUBJECT
        ):
            return [term]
    return []


def extract_prose_search_terms(query: str, sanitized: str | None = None) -> List[str]:
    """Return the bounded signal terms shared by FTS and LIKE prose routes."""
    safe = sanitize_fts5_query(query) if sanitized is None else sanitized
    return _extract_bounded_prose_terms(safe)


def build_fts5_match_query(
    query: str,
    *,
    allow_operators: bool = False,
    prose_mode: bool = False,
) -> str:
    """Build the query passed to FTS5 while preserving the default byte path.

    Flag-off and caller-composed queries are exactly the existing sanitized
    form. Flag-on raw prose drops conversational framing, deduplicates terms,
    caps work at ``_PROSE_TERM_LIMIT``, and lets FTS5/BM25 rank a disjunction.
    Compact keyword queries and balanced quoted queries remain conjunctive.
    """
    sanitized = sanitize_fts5_query(query, allow_operators=allow_operators)
    if (
        not prose_mode
        or allow_operators
        or not should_use_fts_prose_mode(query, sanitized)
    ):
        return sanitized

    prose_terms = extract_prose_search_terms(query, sanitized)
    if not prose_terms:
        # An empty disjunction would be an FTS5 MATCH syntax error; keep the
        # pre-prose sanitized form as the fallback exactly as flag-off does.
        return sanitized
    return " OR ".join(prose_terms)


def sanitize_like_query(query: str) -> str:
    """Strip FTS5 syntax operators, preserving every other character.

    The LIKE path's sanitization has to be WEAKER than the FTS one: a character
    the index cannot spell is still a character LIKE can match. Sharing the FTS
    term form here dropped emoji from the fallback that exists to find them
    (``launch 🚀`` searched only ``%launch%``).
    """
    return _sanitize_query(query, _like_safe_char)


def _sanitize_query(query: str, replace: Callable[[str], str]) -> str:
    """Walk ``query`` outside balanced phrase quotes, mapping chars via ``replace``."""
    if not query:
        return ""

    result: list[str] = []
    quote_buffer: list[str] = []
    in_quote = False
    for char in query:
        if char == '"':
            if in_quote:
                result.append('"')
                result.extend(quote_buffer)
                result.append('"')
                quote_buffer = []
                in_quote = False
            else:
                if result and not result[-1].isspace():
                    result.append(" ")
                in_quote = True
                quote_buffer = []
            continue
        if in_quote:
            quote_buffer.append(char)
            continue
        result.append(replace(char))
    if in_quote and quote_buffer:
        result.extend(replace(char) for char in "".join(quote_buffer))
    return " ".join("".join(result).split())


_WORD_RE = re.compile(r"[\w-]+", re.UNICODE)


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def contains_emoji(text: str) -> bool:
    return bool(_EMOJI_RE.search(text or ""))


def contains_unindexed_unicode_symbol(text: str) -> bool:
    """Whether raw text carries a non-ASCII symbol unicode61 will not index.

    ASCII punctuation keeps the established tokenizer-parity path (notably
    hyphens and ``$`` in prose). This catches the sibling finding's stripped
    Unicode signal such as ©, €, and ™ without broadening ordinary punctuation
    back onto the full-table LIKE scan.
    """
    return any(
        ord(char) > 0x7F and unicodedata.category(char).startswith("S")
        for char in (text or "")
    )


def contains_risky_fts_ascii(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if raw.count('"') % 2:
        return True
    text_without_phrases = _QUOTED_PHRASE_RE.sub(" ", raw)
    return bool(_RISKY_FTS_TOKEN_RE.search(text_without_phrases))


def requires_like_fallback(
    query: str,
    sanitized: str | None = None,
    *,
    preserve_unicode_symbols: bool = True,
) -> bool:
    """Whether ``query`` must be answered by the LIKE scan instead of the index.

    The test is what SANITIZATION LOSES, not what the FTS5 query grammar cannot
    spell. A compound token (``art-related``, ``api:v2``, ``a/b``) sanitizes to
    ordinary terms the index answers perfectly well, so routing it to the
    full-table LIKE scan just re-imports the scaling ceiling this branch is
    fixing — 6 of the 50 fixed Phase 1B questions carry a hyphen. The risky-ASCII
    check therefore runs against the SANITIZED form, which is what actually
    reaches ``MATCH``.

    Genuine losses stay on LIKE: unicode61 does not segment CJK; it drops emoji
    and non-ASCII Unicode symbols from the index entirely; and a query that
    sanitizes to nothing has no terms left to match. Those character classes
    are tested against the RAW query because sanitization is exactly what
    removes them.
    """
    raw = query or ""
    safe = sanitize_fts5_query(raw) if sanitized is None else (sanitized or "")
    if not safe.strip():
        return True
    if (
        contains_cjk(raw)
        or contains_emoji(raw)
        or (
            preserve_unicode_symbols
            and contains_unindexed_unicode_symbol(raw)
        )
    ):
        return True
    return contains_risky_fts_ascii(safe)


def _token_variants(
    token: str,
    *,
    normalize_sentence_punctuation: bool = False,
) -> List[str]:
    cleaned = (token or "").strip().strip(_STRIP_EDGE_PUNCT)
    if normalize_sentence_punctuation:
        without_sentence_punct = cleaned.strip(_TRAILING_SENTENCE_PUNCT)
        if without_sentence_punct:
            cleaned = without_sentence_punct
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


def extract_search_terms(
    query: str,
    *,
    normalize_sentence_punctuation: bool = False,
) -> List[str]:
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
        terms.extend(
            _token_variants(
                token,
                normalize_sentence_punctuation=normalize_sentence_punctuation,
            )
        )

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


def extract_quoted_phrases(query: str) -> List[str]:
    return [phrase.strip() for phrase in _QUOTED_PHRASE_RE.findall(query or "") if phrase.strip()]


def escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def count_term_matches(text: str, term: str) -> int:
    haystack = (text or "")
    needle = (term or "")
    if not haystack or not needle:
        return 0
    return haystack.lower().count(needle.lower())


def compute_directness_score(text: str, terms: List[str], phrases: List[str] | None = None) -> float:
    content = text or ""
    if not content:
        return 0.0

    unique_hits = 0
    total_hits = 0
    non_phrase_unique_hits = 0
    non_phrase_total_hits = 0
    normalized_phrases = {(phrase or "").strip().lower() for phrase in (phrases or []) if (phrase or "").strip()}
    for term in terms:
        matches = count_term_matches(content, term)
        if matches > 0:
            unique_hits += 1
            total_hits += matches
            if term.strip().lower() not in normalized_phrases:
                non_phrase_unique_hits += 1
                non_phrase_total_hits += matches

    phrase_hits = 0
    lowered = content.lower()
    for phrase in phrases or []:
        if phrase and phrase.lower() in lowered:
            phrase_hits += 1

    repetition_penalty = max(0, total_hits - unique_hits)
    non_phrase_repetition_penalty = max(0, non_phrase_total_hits - non_phrase_unique_hits)
    score = float((unique_hits * 5) + (phrase_hits * 8))
    if not phrases:
        score -= min(repetition_penalty, 6)
    else:
        score -= min(non_phrase_repetition_penalty, 6)

    if phrases:
        for phrase in phrases:
            normalized_phrase = (phrase or "").strip().lower()
            if not normalized_phrase:
                continue
            phrase_occurrences = lowered.count(normalized_phrase)
            if phrase_occurrences <= 1:
                continue
            segments = re.split(re.escape(normalized_phrase), lowered)
            gap_unique_counts = []
            for segment in segments:
                segment_tokens = [
                    token.lower()
                    for token in _WORD_RE.findall(segment)
                    if any(char.isalpha() for char in token)
                ]
                gap_unique_counts.append(len(set(segment_tokens)))
            interior_gap_counts = gap_unique_counts[1:-1]
            tail_gap_count = gap_unique_counts[-1] if gap_unique_counts else 0
            extra_occurrences = phrase_occurrences - 1
            score -= extra_occurrences * 0.5
            score -= sum(1.5 for count in interior_gap_counts if 0 < count <= 4)
            if all(count == 0 for count in interior_gap_counts) and tail_gap_count <= 2:
                score -= min(extra_occurrences, 3) * 1.0

    return score


def _is_precise_query_shape(terms: List[str], phrases: List[str] | None = None) -> bool:
    if len(terms) == 1:
        return True
    return len(phrases or []) == 1 and len(terms) <= 2


def should_widen_candidate_fetch(terms: List[str], phrases: List[str] | None = None) -> bool:
    return _is_precise_query_shape(terms, phrases)


def should_apply_directness_rank_adjustment(terms: List[str], phrases: List[str] | None = None) -> bool:
    return _is_precise_query_shape(terms, phrases)


def compute_directness_rank_bonus_upper_bound(terms: List[str], phrases: List[str] | None = None) -> float:
    return float((len(terms) * 5) + (len(phrases or []) * 8))


def compute_search_fetch_limit(limit: int, terms: List[str], phrases: List[str] | None = None) -> int:
    base = max(limit * 5, limit, 20)
    if should_widen_candidate_fetch(terms, phrases):
        return max(base, limit * 10, 50)
    return base


def compute_like_fallback_fetch_limit(limit: int, terms: List[str], phrases: List[str] | None = None) -> int:
    """Bound LIKE fallback candidate rows before Python-side scoring/sorting."""
    return compute_search_fetch_limit(limit, terms, phrases)


def compute_search_candidate_cap(limit: int) -> int:
    """Return a hard upper bound on candidate rows inspected per search call."""
    return min(max(limit * 20, limit, 500), 5_000)


AGE_DECAY_RATE = 0.001


def normalize_search_sort(sort: str | None) -> str:
    """Normalize sort parameter to one of: recency, relevance, hybrid."""
    normalized = (sort or "recency").strip().lower()
    return normalized if normalized in {"recency", "relevance", "hybrid"} else "recency"


def resolve_prose_sort(
    sort: str | None,
    fts_prose_mode: bool,
    query: str,
    *,
    allow_operators: bool = False,
) -> str | None:
    """Promote implicit prose searches to relevance without changing defaults."""
    if (
        sort is None
        and fts_prose_mode
        and not allow_operators
        and should_use_fts_prose_mode(query)
    ):
        return "relevance"
    return sort


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
