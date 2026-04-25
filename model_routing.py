"""LCM model override routing helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRoute:
    provider: str | None
    model: str


ProviderResolver = Callable[[str], bool]


# Conservative allowlist: only split prefixes explicitly supported by LCM
# overrides. Do not infer from arbitrary provider names, because many Hermes
# provider IDs are also valid OpenRouter model namespaces, for example
# ``anthropic/...``, ``deepseek/...``, ``google/...``, and ``x-ai/...``.
_PROVIDER_PREFIXES = frozenset({"cerebras"})


def _provider_route_is_resolvable(provider: str) -> bool:
    """Return whether Hermes can route an explicit auxiliary provider."""
    try:
        from hermes_cli.runtime_provider import _get_named_custom_provider

        if _get_named_custom_provider(provider):
            return True
    except Exception:
        pass

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        return provider in PROVIDER_REGISTRY
    except Exception:
        return False


def parse_lcm_model_override(
    value: str | None,
    *,
    provider_resolver: ProviderResolver | None = None,
) -> ModelRoute:
    """Parse an LCM model override into explicit provider/model routing.

    Values whose first path segment is both allowlisted and resolvable by the
    Hermes host are split into ``provider=<prefix>`` and ``model=<rest>``.
    Other values, even when they contain ``/``, remain model-only overrides.
    """
    model = (value or "").strip()
    if not model:
        return ModelRoute(provider=None, model="")

    provider, sep, rest = model.partition("/")
    provider = provider.strip().lower()
    rest = rest.strip()
    can_resolve_provider = provider_resolver or _provider_route_is_resolvable
    if sep and provider in _PROVIDER_PREFIXES and rest and can_resolve_provider(provider):
        return ModelRoute(provider=provider, model=rest)

    return ModelRoute(provider=None, model=model)


def apply_lcm_model_route(call_kwargs: dict, model: str | None) -> None:
    """Apply parsed LCM provider/model overrides to Hermes auxiliary kwargs."""
    route = parse_lcm_model_override(model)
    if route.provider:
        call_kwargs["provider"] = route.provider
    if route.model:
        call_kwargs["model"] = route.model
