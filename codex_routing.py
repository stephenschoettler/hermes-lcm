"""Codex OAuth route detection and effective context-window caps.

Isolated from ``engine.py`` (WS5 seam) so the Codex-specific routing policy —
which model slugs are on the ChatGPT Codex OAuth route and what effective
context window that route enforces — lives in one cohesive place. These are
pure helpers with no engine state; ``engine.py`` imports them and keeps its own
policy constants (for example the gpt-5.5 compaction threshold).
"""

from __future__ import annotations

import math

# Only these exact normalized bare slugs have a proven 900k Codex OAuth route.
# Keep them separate from the family fallbacks below so suffixes and synthetic
# aliases cannot inherit the larger window.
_CODEX_OAUTH_EXACT_CONTEXT_CAPS: dict[str, int] = {
    "gpt-5.6-terra-900k": 900_000,
    "gpt-5.6-sol-900k": 900_000,
    "gpt-5.6-luna-900k": 900_000,
}

# ChatGPT Codex OAuth exposes provider-enforced context windows that can be
# materially lower than the same model slug on direct OpenAI/OpenRouter routes.
# Hermes Agent resolves these from chatgpt.com/backend-api/codex/models, with
# this table as its fallback. LCM sees only the host-advertised context_length;
# when that value was explicitly overridden above the real Codex OAuth window,
# we still have to budget against the effective provider window or compaction
# fires too late and provider requests can overflow.
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

# Multiplier applied to the fresh-tail floor when deriving a viable trigger. 1.35 leaves ~35%
# working room between the floor and the trigger, so a pass that compacts everything above the
# tail makes real progress instead of immediately re-triggering.
_FLOOR_HEADROOM = 1.35


def _bare_model_slug(model: str | None) -> str:
    return (model or "").strip().lower().rsplit("/", 1)[-1]


def _is_openai_codex_route(provider: str | None) -> bool:
    return (provider or "").strip().lower() == "openai-codex"


def _codex_oauth_context_cap(model: str | None, provider: str | None) -> int | None:
    """Return LCM's best-known Codex OAuth effective context cap.

    This intentionally mirrors Hermes Agent's hardcoded fallback policy, not the
    direct OpenAI model catalog. A host-provided context_length may be a user
    override or stale cache entry; Codex OAuth still enforces these lower route
    windows.
    """
    if not _is_openai_codex_route(provider):
        return None
    bare_model = _bare_model_slug(model)
    if not bare_model:
        return None
    exact_cap = _CODEX_OAUTH_EXACT_CONTEXT_CAPS.get(bare_model)
    if exact_cap is not None:
        return exact_cap
    for slug, cap in sorted(
        _CODEX_OAUTH_CONTEXT_CAPS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if slug in bare_model:
            return cap
    return None


def _minimum_viable_threshold(
    context_length: int,
    fresh_tail_floor_tokens: int,
    *,
    headroom: float = _FLOOR_HEADROOM,
) -> float | None:
    """Smallest context_threshold that compaction can actually satisfy on this route.

    Compaction can never reduce the active context below the fresh tail, which is kept verbatim.
    If ``context_length * context_threshold`` lands at or below that floor, every preflight pass
    re-triggers, makes "insufficient progress", and the engine eventually latches
    ``attempts_exhausted`` -- burning minutes per turn while never clearing the trigger.

    Returns the ratio that puts the trigger ``headroom`` above the floor, or None when the
    configured ratio is already safe (or the inputs are unknown).
    """
    if context_length <= 0 or fresh_tail_floor_tokens <= 0:
        return None
    # The engine derives its trigger with int(context_length * ratio), which truncates. Ceil the
    # target token count first and nudge the ratio up by one ULP-ish step so the truncated trigger
    # still lands at or above the target -- otherwise the guard can come back one token short and
    # the livelock survives the fix.
    target_tokens = math.ceil(fresh_tail_floor_tokens * headroom)
    if target_tokens >= context_length:
        # The floor alone fills the window: raising the trigger cannot fix this, and clamping to
        # ~1.0 would disable compaction entirely. Leave it to the caller's own reporting.
        return None
    needed = target_tokens / context_length
    while int(context_length * needed) < target_tokens:
        needed = math.nextafter(needed, 1.0)
    return needed


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
