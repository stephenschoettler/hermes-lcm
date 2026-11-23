"""LCM model override routing helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRoute:
    provider: str | None
    model: str


ProviderResolver = Callable[[str], bool]

# Model slug prefixes that map to NVIDIA NIM (when no :free suffix).
_NVIDIA_NIM_PREFIXES = frozenset({"nvidia", "deepseek-ai"})

# When a model slug doesn't match any known provider prefix, use this
# auxiliary provider by default.
_DEFAULT_AUXILIARY_PROVIDER = "openrouter"

# Conservative allowlist for built-in provider registry fallbacks.
_PROVIDER_PREFIXES = frozenset({"cerebras", "nvidia"})


def _provider_route_is_resolvable(provider: str) -> bool:
    """Return whether Hermes can route an explicit auxiliary provider."""
    provider = (provider or "").strip().lower()
    if not provider:
        return False
    if provider.startswith("custom:"):
        provider = provider.split(":", 1)[1].strip()
        if not provider:
            return False

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        if provider in PROVIDER_REGISTRY:
            return provider in _PROVIDER_PREFIXES
    except Exception:
        pass

    try:
        from hermes_cli.runtime_provider import _get_named_custom_provider

        if _get_named_custom_provider(provider):
            return True
    except Exception:
        pass

    return False


def parse_lcm_model_override(
    value: str | None,
    *,
    provider_resolver: ProviderResolver | None = None,
) -> ModelRoute:
    """Parse an LCM model override into explicit provider/model routing.

    - Models ending in ``:free`` route to OpenRouter with the suffix preserved.
    - Models with ``nvidia/`` or ``deepseek-ai/`` prefixes (no ``:free`` suffix)
      route to NVIDIA NIM, keeping the full original model string intact.
    - Everything else passes through unchanged.
    """
    model = (value or "").strip()
    if not model:
        return ModelRoute(provider=None, model="")

    # OpenRouter free-tier models keep their :free suffix — do NOT strip it.
    if model.endswith(":free"):
        return ModelRoute(provider="openrouter", model=model)

    # NVIDIA NIM routes (nvidia/ and deepseek-ai/ prefixes, no :free).
    for prefix in _NVIDIA_NIM_PREFIXES:
        if model.startswith(f"{prefix}/"):
            return ModelRoute(provider="nvidia", model=model)

    return ModelRoute(provider=None, model=model)


def apply_lcm_model_route(call_kwargs: dict, model: str | None) -> None:
    """Apply parsed LCM provider/model overrides to Hermes auxiliary kwargs."""
    route = parse_lcm_model_override(model)
    if route.provider:
        call_kwargs["provider"] = route.provider
    if route.model:
        call_kwargs["model"] = route.model
    if model and route.model:
        logger.debug(
            "LCM auxiliary model override routed: raw=%r provider=%s model=%s",
            model,
            route.provider or "(task default)",
            route.model,
        )
