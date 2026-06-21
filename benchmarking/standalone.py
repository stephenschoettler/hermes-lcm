"""Standalone checkout helpers for deterministic benchmark/release scripts."""

from __future__ import annotations

import importlib
import sys
import types


def ensure_agent_context_engine_importable() -> None:
    """Provide the minimal Hermes ContextEngine API when Hermes Agent is absent.

    Release benchmark and stress scripts exercise hermes-lcm directly from a
    plugin checkout. They only need the ContextEngine base class for inheritance;
    they do not need a full Hermes Agent install or live plugin host.
    """
    try:
        importlib.import_module("agent.context_engine")
        return
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing not in {"agent", "agent.context_engine"}:
            raise

    agent_module = sys.modules.get("agent")
    if agent_module is None:
        agent_module = types.ModuleType("agent")
        agent_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["agent"] = agent_module

    context_module = types.ModuleType("agent.context_engine")

    class ContextEngine:
        """Small fallback compatible with Hermes Agent's ContextEngine ABC."""

        last_prompt_tokens: int = 0
        last_completion_tokens: int = 0
        last_total_tokens: int = 0
        threshold_tokens: int = 0
        context_length: int = 0
        compression_count: int = 0
        threshold_percent: float = 0.75
        protect_first_n: int = 3
        protect_last_n: int = 6

        @property
        def name(self) -> str:
            return self.__class__.__name__

        def update_from_response(self, usage):  # pragma: no cover - fallback default
            self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            self.last_total_tokens = int(usage.get("total_tokens", 0) or 0)

        def should_compress(self, prompt_tokens=None) -> bool:  # pragma: no cover - fallback default
            return False

        def compress(self, messages, current_tokens=None, focus_topic=None):  # pragma: no cover - fallback default
            return messages

        def should_compress_preflight(self, messages) -> bool:
            return False

        def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
            return False

        def has_content_to_compress(self, messages) -> bool:
            return True

        def on_session_start(self, session_id: str, **kwargs) -> None:
            return None

        def on_session_end(self, session_id: str, messages) -> None:
            return None

        def on_session_reset(self) -> None:
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0
            self.last_total_tokens = 0
            self.compression_count = 0

        def get_tool_schemas(self):
            return []

        def handle_tool_call(self, name: str, args, **kwargs) -> str:
            import json

            return json.dumps({"error": f"Unknown context engine tool: {name}"})

        def get_status(self):
            usage_percent = min(100, self.last_prompt_tokens / self.context_length * 100) if self.context_length else 0
            return {
                "last_prompt_tokens": self.last_prompt_tokens,
                "threshold_tokens": self.threshold_tokens,
                "context_length": self.context_length,
                "usage_percent": usage_percent,
                "compression_count": self.compression_count,
            }

        def update_model(
            self,
            model: str,
            context_length: int,
            base_url: str = "",
            api_key: str = "",
            provider: str = "",
            api_mode: str = "",
        ) -> None:
            self.context_length = context_length
            self.threshold_tokens = int(context_length * self.threshold_percent)

    setattr(context_module, "ContextEngine", ContextEngine)
    sys.modules["agent.context_engine"] = context_module
    setattr(agent_module, "context_engine", context_module)
