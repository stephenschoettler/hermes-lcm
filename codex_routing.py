"""Codex OAuth route detection and effective context-window caps.

Isolated from ``engine.py`` (WS5 seam) so the Codex-specific routing policy —
which model slugs are on the ChatGPT Codex OAuth route and what effective
context window that route enforces — lives in one cohesive place. These are
pure helpers with no engine state; ``engine.py`` imports them and keeps its own
policy constants (for example the gpt-5.5 compaction threshold).
"""

from __future__ import annotations

# ChatGPT Codex OAuth exposes provider-enforced context windows that can be
# materially lower than the same model slug on direct OpenAI/OpenRouter routes.
# Hermes Agent resolves these from chatgpt.com/backend-api/codex/models, with
# this table as its fallback. LCM sees only the host-advertised context_length;
# when that value was explicitly overridden above the real Codex OAuth window,
# we still have to budget against the effective provider window or compaction
# fires too late and provider requests can overflow.
# Hermes-side marker for explicitly selected, host-validated long-context aliases.
_CODEX_CONTEXT_VARIANT_SUFFIX = "-900k"
_CODEX_CONTEXT_VARIANT_CAP = 900_000


_CODEX_OAUTH_CONTEXT_CAPS: dict[str, int] = {
    "gpt-5.1-codex-max": 272_000,
    "gpt-5.1-codex-mini": 272_000,
    "gpt-5.3-codex-spark": 128_000,
    "gpt-5.3-codex": 272_000,
    "gpt-5.2-codex": 272_000,
    "gpt-5.4-mini": 272_000,
    "gpt-5.5": 272_000,
    "gpt-5.4": 272_000,
    "gpt-5.2": 272_000,
    "gpt-5.6": 372_000,
    "gpt-5": 272_000,
}


def _bare_model_slug(model: str | None) -> str:
    return (model or "").strip().lower().rsplit("/", 1)[-1]


def _is_openai_codex_route(provider: str | None) -> bool:
    return (provider or "").strip().lower() == "openai-codex"


def _is_host_verified_codex_context_variant(model: str | None) -> bool:
    """Return whether Hermes recognises *model* as a valid context variant.

    Newer Hermes hosts expose ``is_codex_context_variant`` as the single source
    of truth for ``-900k`` eligibility.  Older hosts do not, so fail closed and
    retain LCM's conservative Codex OAuth cap rather than trusting the suffix
    alone.
    """
    try:
        from agent.model_metadata import is_codex_context_variant
    except (ImportError, AttributeError):
        return False

    try:
        return bool(is_codex_context_variant(model))
    except Exception:
        return False


def _codex_oauth_context_cap(model: str | None, provider: str | None) -> int | None:
    """Return LCM's best-known Codex OAuth effective context cap.

    This intentionally mirrors Hermes Agent's hardcoded fallback policy, not the
    direct OpenAI model catalog. A host-provided context_length may be a user
    override or stale cache entry; Codex OAuth still enforces these lower route
    windows. Explicit long-context variants validated by Hermes use their named
    900K ceiling so LCM preserves the host-resolved window without accepting a
    stale value above the provider limit.
    """
    if not _is_openai_codex_route(provider):
        return None
    bare_model = _bare_model_slug(model)
    if not bare_model:
        return None
    if (
        bare_model.endswith(_CODEX_CONTEXT_VARIANT_SUFFIX)
        and _is_host_verified_codex_context_variant(model)
    ):
        return _CODEX_CONTEXT_VARIANT_CAP
    for slug, cap in sorted(
        _CODEX_OAUTH_CONTEXT_CAPS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if slug in bare_model:
            return cap
    return None


def _is_codex_gpt55_route(model: str | None, provider: str | None) -> bool:
    """Return True for gpt-5.5 on ChatGPT Codex OAuth, mirroring Hermes core."""
    if not _is_openai_codex_route(provider):
        return False
    bare_model = _bare_model_slug(model)
    return (
        bare_model == "gpt-5.5"
        or bare_model.startswith("gpt-5.5-")
        or bare_model.startswith("gpt-5.5.")
    )
