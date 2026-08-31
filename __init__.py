"""Hermes LCM Plugin — Lossless Context Management.

Replaces the built-in ContextCompressor with a DAG-based context engine
that persists every message and provides structured retrieval tools.

Based on the LCM paper by Ehrlich & Blackman (Voltropy PBC, Feb 2026).
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def get_recall_policy() -> str:
    """Load the canonical product policy without making bare imports package-dependent."""
    from .guidance import get_recall_policy as _get_recall_policy

    return _get_recall_policy()


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


def _engine_bound_session_id(engine) -> str:
    """Return the lifecycle/ingest session bound on an LCM engine.

    ``current_session_id`` is an operator-facing foreground view and can differ
    from the bound ingest session while an auxiliary side channel is active.
    Post-turn ingest rebinding must use the bound id or it can append a resumed
    foreground turn to a stale auxiliary child.
    """
    return str(
        getattr(engine, "bound_session_id", "")
        or getattr(engine, "_session_id", "")
        or ""
    )


def _ensure_engine_bound_to_session(
    active_engine,
    session_id: str,
    *,
    platform: str = "",
    conversation_id: str = "",
) -> None:
    session_id = str(session_id or "")
    if session_id and _engine_bound_session_id(active_engine) != session_id:
        active_engine.on_session_start(
            session_id,
            platform=platform,
            conversation_id=conversation_id or None,
        )


def _hook_question_date(payload: dict) -> object:
    """Return an explicit turn anchor without inventing event time."""
    explicit = payload.get("question_date") or payload.get("question_as_of")
    if explicit:
        return explicit
    history = payload.get("conversation_history")
    if not isinstance(history, list):
        return None
    user_message = str(payload.get("user_message") or "")
    for message in reversed(history):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if user_message and isinstance(content, str) and content != user_message:
            continue
        timestamp = message.get("timestamp")
        if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
            try:
                return datetime.fromtimestamp(
                    float(timestamp), tz=timezone.utc
                ).date().isoformat()
            except (OverflowError, OSError, ValueError):
                return None
        if isinstance(timestamp, str) and timestamp.strip():
            return timestamp.strip()
    return None


def _answer_ready_baseline(active_engine, question: str, payload: dict):
    """Return caller-supplied exact refs or create one bounded product baseline."""
    baseline_refs = payload.get("baseline_refs")
    if isinstance(baseline_refs, (list, tuple)):
        return tuple(baseline_refs)
    raw = active_engine.handle_tool_call(
        "lcm_recall",
        {
            "query": question,
            "include": "verbatim",
            "detail": "answer_ready",
            "limit": 25,
            "scope_bias": 0.0,
            "include_occurrence_time": True,
        },
    )
    recalled = json.loads(raw) if isinstance(raw, str) else raw
    hits = recalled.get("hits") if isinstance(recalled, dict) else None
    candidates = []
    for hit in hits if isinstance(hits, list) else []:
        if not isinstance(hit, dict):
            continue
        exact_ref = str(hit.get("exact_ref") or "").strip()
        quote = str(hit.get("content") or hit.get("snippet") or "")
        if exact_ref and quote:
            candidates.append({"exact_ref": exact_ref, "quote": quote})
        elif hit.get("store_id") is not None and quote:
            candidates.append(
                {
                    "store_id": hit.get("store_id"),
                    "content_offset": hit.get("content_offset", 0),
                    "content": quote,
                }
            )
    return tuple(candidates)


def _effective_preanswer_mode(config) -> str:
    if not bool(getattr(config, "preanswer_evidence_enabled", False)):
        return "off"
    raw = str(getattr(config, "preanswer_evidence_mode", "") or "").strip().casefold()
    if not raw:
        return "legacy_selective"
    return raw if raw in {"off", "legacy_selective", "requirements_v1"} else "off"


def _pre_llm_context(active_engine, recall_policy: str, payload: dict) -> dict:
    """Keep ordinary baseline bytes, or add one bounded product-owned delta."""
    enabled_toolsets = payload.get("enabled_toolsets")
    context_engine_enabled = not (
        isinstance(enabled_toolsets, (list, tuple, set, frozenset))
        and "context_engine" not in enabled_toolsets
    )
    config = getattr(active_engine, "_config", None)
    mode = _effective_preanswer_mode(config)
    if mode == "off" or not context_engine_enabled:
        return {"context": recall_policy}
    try:
        question = str(payload.get("user_message") or "").strip()
        if not question:
            return {"context": recall_policy}
        question_date = _hook_question_date(payload)

        if mode == "requirements_v1":
            from .answer_contract import compile_answer_contract
            from .evidence_compiler import compile_preanswer_evidence

            contract = compile_answer_contract(question, question_date)
            if contract.status != "planned":
                return {"context": recall_policy}
            baseline_was_internal = not isinstance(
                payload.get("baseline_refs"), (list, tuple)
            )
            baseline_refs = _answer_ready_baseline(active_engine, question, payload)
            result = compile_preanswer_evidence(
                question,
                engine=active_engine,
                baseline_refs=baseline_refs,
                question_as_of=question_date,
                retrieve=lambda recall_args: active_engine.handle_tool_call(
                    "lcm_recall", recall_args
                ),
                enabled=True,
                render_baseline_context=baseline_was_internal,
            )
            try:
                active_engine._last_preanswer_evidence_trace = result
            except Exception:
                pass
            context = result.get("context") if isinstance(result, dict) else None
            if not isinstance(context, str) or not context:
                return {"context": recall_policy}
            return {"context": f"{recall_policy}\n\n{context}"}

        from .selective_recall import (
            build_selective_session_bundle,
            route_selective_recall,
        )

        route = route_selective_recall(question, question_date)
        if route["route"] == "ordinary":
            # This branch deliberately performs no recall, session load, or
            # auxiliary-model call.  The ordinary policy bytes stay identical.
            return {"context": recall_policy}

        # Official Hermes hook payloads do not currently include answer-ready
        # refs.  Create that bounded baseline in product code when absent; the
        # benchmark bridge supplies its frozen cached baseline and skips this
        # retrieval, preserving those search bytes exactly.
        baseline_refs = _answer_ready_baseline(active_engine, question, payload)

        result = build_selective_session_bundle(
            question,
            engine=active_engine,
            baseline_refs=baseline_refs,
            question_date=question_date,
            enabled=True,
        )
        compiler_result = None
        selector_usage = None
        if bool(getattr(config, "selective_compiler_enabled", False)):
            try:
                from .selective_compiler import (
                    call_selective_auxiliary_selector,
                    compile_selective_evidence,
                    prepare_selective_compiler,
                )

                compiler_refs = []
                for raw in [*baseline_refs, *(result.get("evidence") or [])]:
                    if not isinstance(raw, dict):
                        continue
                    exact_ref = str(raw.get("exact_ref") or "").strip()
                    quote = str(raw.get("quote") or raw.get("content") or "")
                    if exact_ref and quote:
                        compiler_refs.append(
                            {
                                "exact_ref": exact_ref,
                                "quote": quote,
                                "date": raw.get("date"),
                            }
                        )
                prepared = prepare_selective_compiler(
                    question,
                    baseline_refs=compiler_refs,
                    question_date=question_date,
                    engine=active_engine,
                )
                if prepared["status"] == "selector_required":
                    proposal, selector_usage = call_selective_auxiliary_selector(
                        prepared,
                        model=str(
                            getattr(config, "selective_compiler_model", "") or ""
                        ),
                        timeout_seconds=8.0,
                    )
                    compiler_result = compile_selective_evidence(
                        question,
                        engine=active_engine,
                        compiler_refs=prepared["compiler_refs"],
                        selector_proposal=proposal,
                        question_date=question_date,
                        enabled=True,
                    )
            except Exception as exc:
                logger.warning("LCM selective compiler failed open: %s", exc)
    except Exception as exc:  # pragma: no cover - outer host safety net
        logger.warning("LCM pre-answer evidence failed open: %s", exc)
        return {"context": recall_policy}
    try:
        active_engine._last_preanswer_evidence_trace = (
            {
                **compiler_result,
                "session_bundle": result,
                "selector_usage": selector_usage,
            }
            if isinstance(compiler_result, dict)
            else result
        )
    except Exception:
        pass
    augmentations = []
    session_context = result.get("context") if isinstance(result, dict) else None
    compiler_context = (
        compiler_result.get("context") if isinstance(compiler_result, dict) else None
    )
    for value in (session_context, compiler_context):
        if isinstance(value, str) and value:
            augmentations.append(value)
    if not augmentations:
        return {"context": recall_policy}
    return {"context": f"{recall_policy}\n\n" + "\n\n".join(augmentations)}


def _session_context_value(name: str) -> str:
    """Read task-local host session metadata with legacy env compatibility."""
    try:
        from gateway.session_context import get_session_env
    except ImportError:
        return str(os.environ.get(name, "") or "")
    try:
        return str(get_session_env(name, "") or "")
    except Exception:
        # Once a concurrent host exposes task-local session context, never fall
        # back to process-global env after a read failure: it may name another
        # lane. Unbound is safer than cross-session command dispatch.
        logger.debug("LCM plugin command could not read %s", name, exc_info=True)
        return ""


def _command_engine_for_current_session(engine, resolve_active_lcm_engine):
    """Resolve the runtime serving the current plugin-command invocation.

    Gateway hosts bind task-local session/lane metadata before dispatching a
    plugin slash command. Prefer the already-active AIAgent clone registered for
    that lane. If no clone exists yet, keep the process-wide prototype's genuine
    ``(unbound)`` cold-start status rather than mutating shared runtime state.
    """
    session_id = _session_context_value("HERMES_SESSION_ID")
    conversation_id = _session_context_value("HERMES_SESSION_KEY")
    if session_id or conversation_id:
        active_engine = resolve_active_lcm_engine(
            session_id=session_id,
            conversation_id=conversation_id,
            allow_foreground=True,
        )
        if active_engine is not None:
            return active_engine
    return engine


def _make_command_handler(handle_lcm_command, engine, resolve_active_lcm_engine):
    def _handler(raw_args: str):
        return handle_lcm_command(
            raw_args,
            _command_engine_for_current_session(
                engine,
                resolve_active_lcm_engine,
            ),
        )

    return _handler


def register(ctx):
    """Plugin entry point — register the LCM context engine and tools."""
    from .config import LCMConfig
    from .engine import LCMEngine, resolve_active_lcm_engine
    from .schemas import (
        LCM_GREP,
        LCM_RECALL,
        LCM_QUERY_STATE,
        LCM_COMPUTE,
        LCM_COMPILE_EVIDENCE,
        LCM_EVIDENCE_PACK,
        LCM_RETRIEVE,
        LCM_RECENT,
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

    # Ship the same recall contract through both Hermes plugin skill
    # registration (explicit qualified loads) and the installer's ordinary
    # profile skill link (normal discovery). Older hosts simply lack this
    # capability and keep their existing schema-driven behavior.
    skill_root = Path(__file__).resolve().parent / "skills" / "hermes-lcm"
    register_skill = getattr(ctx, "register_skill", None)
    if callable(register_skill):
        try:
            register_skill(
                "hermes-lcm",
                skill_root,
                description=(
                    "Use, configure, diagnose, and recall exact evidence "
                    "with the Hermes-LCM lossless context plugin."
                ),
            )
        except Exception as exc:
            logger.warning(
                "LCM bundled skill registration did not complete; normal "
                "profile skill discovery may still be available: %s",
                exc,
            )

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

        # Hermes invokes this hook after the context engine has received
        # on_session_start(). Resolve through LCM's own registry so merely
        # loading the plugin cannot inject guidance when another context
        # engine is serving the turn. Capture one validated policy value for
        # deterministic, byte-stable injection across eligible turns.
        try:
            recall_policy = get_recall_policy()

            def _on_pre_llm_call(**payload):
                session_id = str(payload.get("session_id") or "")
                conversation_id = str(
                    payload.get("conversation_id")
                    or payload.get("gateway_session_key")
                    or ""
                )
                active_engine = resolve_active_lcm_engine(
                    session_id=session_id,
                    conversation_id=conversation_id,
                )
                if active_engine is None or getattr(active_engine, "name", None) != "lcm":
                    return None
                return _pre_llm_context(active_engine, recall_policy, payload)

            register_hook("pre_llm_call", _on_pre_llm_call)
        except Exception as exc:
            logger.warning(
                "LCM recall-policy hook registration did not complete; "
                "tool schemas remain available: %s",
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
        ("lcm_recall", LCM_RECALL, "🧠"),
        ("lcm_query_state", LCM_QUERY_STATE, "🧾"),
        ("lcm_compute", LCM_COMPUTE, "🧮"),
        ("lcm_compile_evidence", LCM_COMPILE_EVIDENCE, "🧷"),
        ("lcm_evidence_pack", LCM_EVIDENCE_PACK, "📦"),
        ("lcm_retrieve", LCM_RETRIEVE, "🧭"),
        ("lcm_recent", LCM_RECENT, "🕒"),
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
            _make_command_handler(
                handle_lcm_command,
                engine,
                resolve_active_lcm_engine,
            ),
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
                _ensure_engine_bound_to_session(
                    active_engine,
                    session_id,
                    platform=platform,
                    conversation_id=conversation_id,
                )
                active_engine.ingest(history)
            except Exception as exc:
                logger.debug("LCM post_llm_call ingest error: %s", exc)

        _mgr._hooks.setdefault("post_llm_call", []).append(_on_post_llm_call)
        logger.debug("LCM registered post_llm_call hook for per-turn ingest")
    except Exception as exc:
        logger.debug("LCM could not register post_llm_call hook: %s", exc)

    logger.info("LCM plugin loaded — lossless context management active")
