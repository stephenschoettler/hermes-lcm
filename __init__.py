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


def _make_wrapped_handler(tool_name: str, engine):
    """Route a registered lcm_* tool through the engine dispatch path."""
    def _wrapped(args: dict, **kwargs) -> str:
        return engine.handle_tool_call(tool_name, args, **kwargs)
    return _wrapped


def _host_forwards_registered_tool_messages(ctx) -> bool:
    """Return whether ctx.register_tool handlers receive active messages.

    Hermes Agent's current registry dispatch passes task_id/user_task to
    plugin tools, but not the active conversation messages list. Registering
    duplicate lcm_* tool names on that host makes the model call the registry
    handler instead of the native context-engine dispatch branch, so LCM loses
    current-turn ingest before lcm_grep/lcm_expand style recovery.

    Keep plugin-side tool registration opt-in until a host explicitly
    advertises that registered context-engine handlers receive messages.
    """
    capability = getattr(ctx, "context_engine_tool_handlers_receive_messages", False)
    if callable(capability):
        try:
            capability = capability()
        except Exception:
            return False
    return bool(capability)


def register(ctx):
    """Plugin entry point — register the LCM context engine and tools."""
    from .config import LCMConfig
    from .engine import LCMEngine, resolve_active_lcm_engine
    from .schemas import (
        LCM_GREP,
        LCM_LOAD_SESSION,
        LCM_DESCRIBE,
        LCM_EXPAND,
        LCM_EXPAND_QUERY,
        LCM_STATUS,
        LCM_INSPECT,
        LCM_DOCTOR,
    )

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

    # Subscribe to the host's explicit subagent lifecycle events when available.
    # These carry the child_session_id/parent_session_id linkage directly, so LCM
    # can identify a subagent session from the host's own signal instead of
    # walking the call stack and reading private agent attributes. Hosts without
    # a plugin hook bus simply skip this and fall back to the legacy frame walk.
    register_hook = getattr(ctx, "register_hook", None)
    if callable(register_hook):
        from .aux_session import record_subagent_start, record_subagent_stop
        try:
            register_hook("subagent_start", lambda **payload: record_subagent_start(payload))
            register_hook("subagent_stop", lambda **payload: record_subagent_stop(payload))
        except Exception as exc:
            logger.info(
                "LCM explicit subagent-lineage hooks unavailable on this Hermes "
                "host; auxiliary detection uses the legacy frame-walk fallback: %s",
                exc,
            )

    # Register tools via the plugin registry only on hosts that preserve the
    # active messages=... contract for registered context-engine tools.
    # Older/current Hermes hosts already expose lcm_* correctly through the
    # native context-engine schema/dispatch path (Path B). Registering duplicate
    # names through the plugin registry (Path A) on message-blind hosts would
    # shadow Path B and lose current-turn ingest, so the Path B fallback is the
    # expected healthy behavior there.
    _TOOLS = [
        ("lcm_grep", LCM_GREP, "🔍"),
        ("lcm_load_session", LCM_LOAD_SESSION, "📋"),
        ("lcm_describe", LCM_DESCRIBE, "📊"),
        ("lcm_expand", LCM_EXPAND, "🔎"),
        ("lcm_expand_query", LCM_EXPAND_QUERY, "❓"),
        ("lcm_status", LCM_STATUS, "💚"),
        ("lcm_inspect", LCM_INSPECT, "🧭"),
        ("lcm_doctor", LCM_DOCTOR, "🏥"),
    ]
    register_tool = getattr(ctx, "register_tool", None)
    if callable(register_tool) and _host_forwards_registered_tool_messages(ctx):
        for name, schema, emoji in _TOOLS:
            try:
                register_tool(
                    name=name,
                    toolset="context_engine",
                    schema=schema,
                    handler=_make_wrapped_handler(name, engine),
                    description=schema.get("description", ""),
                    emoji=emoji,
                )
            except Exception as exc:
                logger.warning(
                    "LCM plugin-registry tool registration for %s did not complete; "
                    "LCM tools remain available through context-engine schemas: %s",
                    name,
                    exc,
                )
    elif callable(register_tool):
        logger.info(
            "LCM tools are available through context-engine schemas "
            "(expected Path B fallback on this Hermes host). Standalone "
            "plugin-registry tool registration (Path A) requires message-aware "
            "handlers and is not required here."
        )
    else:
        logger.info(
            "LCM tools are available through context-engine schemas (Path B); "
            "plugin-registry tool registration is unavailable on this Hermes "
            "host and is not required."
        )

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

    # Register a post_llm_call hook so every completed turn is persisted to
    # the durable store, regardless of whether compression triggers.  Without
    # this, short WebUI conversations (which never expire and may never hit
    # the compression threshold) are invisible to LCM forever.
    #
    # The hook fires once per turn after the tool-calling loop completes and
    # receives conversation_history including the assistant response.  The
    # existing _ingest_messages cursor prevents duplicates if compress() runs
    # later the same turn.
    try:
        from hermes_cli.plugins import get_plugin_manager as _get_pm
        _mgr = _get_pm()

        def _on_post_llm_call(**kwargs):
            history = kwargs.get("conversation_history")
            if not history:
                return
            active_engine = kwargs.get("context_compressor")
            if not (
                active_engine is not None
                and getattr(active_engine, "name", None) == "lcm"
                and hasattr(active_engine, "ingest")
            ):
                active_engine = None

            session_id = str(kwargs.get("session_id") or "")
            conversation_id = str(
                kwargs.get("conversation_id")
                or kwargs.get("gateway_session_key")
                or ""
            )
            platform = str(kwargs.get("platform") or "")

            if active_engine is None:
                active_engine = resolve_active_lcm_engine(
                    session_id=session_id,
                    conversation_id=conversation_id,
                ) or engine

            try:
                # Session identity is authoritative for rebinding. Older hosts
                # can deliver stale lane metadata alongside the correct active
                # session id; rebinding a clone on conversation_id mismatch
                # alone would move it away from the runtime it is serving.
                if (
                    session_id
                    and str(getattr(active_engine, "current_session_id", "") or "") != session_id
                ):
                    active_engine.on_session_start(
                        session_id,
                        platform=platform,
                        conversation_id=conversation_id or None,
                    )
                active_engine.ingest(history)
            except Exception as exc:
                logger.debug("LCM post_llm_call ingest error: %s", exc)

        _mgr._hooks.setdefault("post_llm_call", []).append(_on_post_llm_call)
        logger.debug("LCM registered post_llm_call hook for per-turn ingest")
    except Exception as exc:
        logger.debug("LCM could not register post_llm_call hook: %s", exc)

    logger.info("LCM plugin loaded — lossless context management active")
