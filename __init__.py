"""Hermes LCM Plugin — Lossless Context Management.

Replaces the built-in ContextCompressor with a DAG-based context engine
that persists every message and provides structured retrieval tools.

Based on the LCM paper by Ehrlich & Blackman (Voltropy PBC, Feb 2026).
"""

import logging
import os

logger = logging.getLogger(__name__)


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def register(ctx):
    """Plugin entry point — register the LCM context engine and tools."""
    from .config import LCMConfig
    from .engine import LCMEngine

    config = LCMConfig.from_env()

    # Resolve hermes_home for profile-scoped storage
    hermes_home = ""
    try:
        from hermes_cli.config import get_hermes_home
        hermes_home = str(get_hermes_home())
    except Exception:
        import os
        hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))

    engine = LCMEngine(config=config, hermes_home=hermes_home)

    # Register as the context engine (replaces ContextCompressor)
    ctx.register_context_engine(engine)

    # Register LCM retrieval/diagnostic tools as Hermes agent tools.
    # These must also be enabled per-platform via platform_toolsets
    # (add "context_engine" to the platform's toolset list).
    from .schemas import (
        LCM_GREP,
        LCM_LOAD_SESSION,
        LCM_DESCRIBE,
        LCM_EXPAND,
        LCM_EXPAND_QUERY,
        LCM_STATUS,
        LCM_DOCTOR,
    )

    _LCM_TOOL_DEFS = [
        (LCM_GREP, "🔍"),
        (LCM_LOAD_SESSION, "📋"),
        (LCM_DESCRIBE, "📊"),
        (LCM_EXPAND, "🔎"),
        (LCM_EXPAND_QUERY, "❓"),
        (LCM_STATUS, "💚"),
        (LCM_DOCTOR, "🏥"),
    ]

    def _wrap_handler(tool_name: str, lcm_engine: LCMEngine):
        """Return a handler that routes tool calls through the LCM engine.

        Preserves the engine's message-ingest step (current-turn search)
        and dispatches to the correct tool implementation.
        """
        def handler(args: dict, **kwargs) -> str:
            return lcm_engine.handle_tool_call(tool_name, args, **kwargs)
        return handler

    register_tool_fn = getattr(ctx, "register_tool", None)
    if callable(register_tool_fn):
        for schema, emoji in _LCM_TOOL_DEFS:
            tname = schema.get("name", "")
            if tname:
                try:
                    register_tool_fn(
                        name=tname,
                        toolset="context_engine",
                        schema=schema,
                        handler=_wrap_handler(tname, engine),
                        description=schema.get("description", ""),
                        emoji=emoji,
                    )
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "LCM tool registration for %s failed: %s", tname, exc
                    )
    else:
        logger.info("ctx.register_tool unavailable — LCM tools registered as context engine only")

    register_command = getattr(ctx, "register_command", None)
    slash_enabled = _env_flag_enabled("LCM_ENABLE_SLASH_COMMAND", default=False)
    if callable(register_command) and slash_enabled:
        from .command import handle_lcm_command

        register_command(
            "lcm",
            lambda raw_args: handle_lcm_command(raw_args, engine),
            description="LCM status and diagnostics",
        )
    elif callable(register_command):
        logger.info("LCM slash command registration disabled (set LCM_ENABLE_SLASH_COMMAND=1 to enable /lcm)")
    else:
        logger.info("LCM slash command registration unavailable on this Hermes host; continuing without /lcm")

    logger.info("LCM plugin loaded — lossless context management active")