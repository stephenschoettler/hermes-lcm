"""LCM model override routing helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRoute:
    provider: str | None
    model: str


ProviderResolver = Callable[[str], bool]


# Conservative allowlist for built-in provider registry fallbacks. Non-canonical
# named custom providers from the Hermes config are safe to split; registry-only
# providers still need an allowlist because many provider IDs are also valid
# OpenRouter model namespaces, for example ``anthropic/...``, ``google/...``
# and ``x-ai/...``. ``custom:`` prefixes are intentionally not split here until
# the Hermes auxiliary resolver supports them end-to-end.
_PROVIDER_PREFIXES = frozenset({"cerebras"})


def _provider_route_is_resolvable(provider: str) -> bool:
    """Return whether Hermes can route an explicit auxiliary provider."""
    if provider.startswith("custom:"):
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

    Values whose first path segment is resolvable by the Hermes host are split
    into ``provider=<prefix>`` and ``model=<rest>``. The default resolver only
    treats non-canonical named custom providers (plus conservative registry
    allowlist entries) as resolvable so OpenRouter-style model slugs and
    canonical built-in provider names remain model-only overrides.
    """
    model = (value or "").strip()
    if not model:
        return ModelRoute(provider=None, model="")

    provider, sep, rest = model.partition("/")
    provider = provider.strip().lower()
    rest = rest.strip()
    can_resolve_provider = provider_resolver or _provider_route_is_resolvable
    if sep and rest and can_resolve_provider(provider):
        return ModelRoute(provider=provider, model=rest)

    return ModelRoute(provider=None, model=model)


def apply_lcm_model_route(call_kwargs: dict, model: str | None) -> None:
    """Apply parsed LCM provider/model overrides to Hermes auxiliary kwargs."""
    route = parse_lcm_model_override(model)
    if route.provider:
        call_kwargs["provider"] = route.provider
    if route.model:
        call_kwargs["model"] = route.model
