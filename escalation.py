"""Three-level summarization escalation.

Level 1 (Normal):    LLM summary preserving details
Level 2 (Aggressive): LLM bullet-point summary at half the token budget
Level 3 (Fallback):   Deterministic truncation — no LLM, guaranteed convergence

Each level checks if Tokens(summary) < Tokens(source). If not, escalates.
"""

from __future__ import annotations

import inspect
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import tokens as _token_module
from .model_routing import apply_lcm_model_route
from .tokens import count_tokens

logger = logging.getLogger(__name__)


# Strip inline reasoning blocks emitted by thinking models (MiniMax-M2.7,
# GLM-5.1, Qwen QwQ, DeepSeek R1, etc.) before persisting summary text.
# Without this, the reasoning content — which often quotes the summarizer
# system prompt verbatim — gets stored as the summary and later confuses
# lcm_expand_query, which feeds the summary back to the model as context.
# Tags mirror the set handled in hermes-agent run_agent.py.
_THINK_BLOCK_RE = re.compile(
    r"<(?P<tag>think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\s*>"
    r".*?"
    r"</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)

# Matches the *start* of a reasoning block with no required close. Applied to
# text after closed <think>...</think> pairs have been stripped: if what
# remains still begins with a reasoning marker, the model emitted an *unclosed*
# block (typically because it ran into max_tokens before the closing tag), and
# the leftover raw reasoning must not be persisted as the summary. Covers the
# angle-tag family plus pipe-delimited (<|think|>), bracket ([think]), and
# prose-header (``Thinking Process:`` / ``Chain of thought:``) shapes.
_REASONING_START_RE = re.compile(
    r"^\s*(?:"
    r"<\s*(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)(?:\s[^>]*)?>"
    r"|<\|\s*(?:start_of_)?(?:think|thinking|reasoning|thought)\s*\|>"
    r"|\[\s*(?:think|thinking|reasoning|thought)\s*\]"
    r"|(?:#{1,6}\s*)?(?:thinking|reasoning|thought)\s+process\s*:"
    r"|(?:#{1,6}\s*)?chain[-\s]+of[-\s]+thought\s*:"
    r")",
    re.IGNORECASE,
)

_DEFAULT_ROUTE_KEY = "<task-default>"


@dataclass
class SummaryCircuitBreaker:
    """In-process circuit breaker for summary model routes.

    The breaker is intentionally small and process-local. It prevents a hot
    compression loop from repeatedly hitting a failing auxiliary route while
    preserving deterministic L3 truncation as the final convergence fallback.
    """

    failure_threshold: int = 2
    cooldown_seconds: int = 300
    _failures: dict[str, int] = field(default_factory=dict)
    _open_until: dict[str, float] = field(default_factory=dict)

    def _key(self, model: str | None) -> str:
        return (model or "").strip() or _DEFAULT_ROUTE_KEY

    def allows(self, model: str | None, *, now: float | None = None) -> bool:
        key = self._key(model)
        current_time = time.monotonic() if now is None else now
        opened_until = self._open_until.get(key, 0.0)
        if opened_until <= current_time:
            if key in self._open_until:
                self._open_until.pop(key, None)
            return True
        return False

    def record_success(self, model: str | None) -> None:
        key = self._key(model)
        self._failures.pop(key, None)
        self._open_until.pop(key, None)

    def record_failure(self, model: str | None, *, now: float | None = None) -> None:
        key = self._key(model)
        failures = self._failures.get(key, 0) + 1
        self._failures[key] = failures
        threshold = max(1, int(self.failure_threshold or 1))
        if failures >= threshold:
            current_time = time.monotonic() if now is None else now
            cooldown = max(0, int(self.cooldown_seconds or 0))
            self._open_until[key] = current_time + cooldown
            logger.warning(
                "LCM summary route circuit opened for %s after %d failure(s); cooldown=%ss",
                key,
                failures,
                cooldown,
            )


@dataclass
class SummarySpendGuard:
    """In-process sliding-window rate limiter for summarizer calls.

    The circuit breaker reacts to *failures*. This guards the orthogonal case:
    a pathologically looping compaction that succeeds every time but burns
    auxiliary-model spend without bound. When the call budget for the window is
    exhausted it opens a backoff during which the escalation path falls back to
    deterministic L3 truncation (no spend, still converges). A forced/manual
    compaction calls clear() so operator-driven repair is never blocked.
    """

    max_calls: int = 24
    window_seconds: float = 600.0
    backoff_seconds: float = 1800.0
    _calls: list[float] = field(default_factory=list)
    _backoff_until: float = 0.0

    def _prune(self, current_time: float) -> None:
        cutoff = current_time - self.window_seconds
        if self._calls and self._calls[0] < cutoff:
            self._calls = [t for t in self._calls if t >= cutoff]

    def allows(self, *, now: float | None = None) -> bool:
        if self.max_calls <= 0:
            return True
        current_time = time.monotonic() if now is None else now
        if current_time < self._backoff_until:
            return False
        self._prune(current_time)
        return len(self._calls) < self.max_calls

    def record_call(self, *, now: float | None = None) -> None:
        if self.max_calls <= 0:
            return
        current_time = time.monotonic() if now is None else now
        self._prune(current_time)
        self._calls.append(current_time)
        if len(self._calls) >= self.max_calls and self._backoff_until <= current_time:
            self._backoff_until = current_time + max(0.0, self.backoff_seconds)
            # Backoff is the penalty; start the window fresh so the guard allows
            # again once it elapses rather than double-blocking on the old count.
            self._calls.clear()
            logger.warning(
                "LCM summary spend guard tripped: %d calls within %ss; "
                "backing off summarizer for %ss (deterministic fallback active)",
                self.max_calls,
                self.window_seconds,
                self.backoff_seconds,
            )

    def clear(self) -> None:
        self._calls.clear()
        self._backoff_until = 0.0


def _strip_reasoning_blocks(text: str) -> str:
    """Remove <think>/<thinking>/<reasoning>/<thought>/<REASONING_SCRATCHPAD>
    blocks from ``text``. Idempotent and safe on text without any tags."""
    if not text or "<" not in text:
        return text
    return _THINK_BLOCK_RE.sub("", text)


def _sanitize_reasoning_summary(text: str) -> str:
    """Return a summary safe to persist, or ``""`` when the model returned only
    reasoning.

    ``_strip_reasoning_blocks`` removes *closed* ``<think>...</think>`` pairs,
    but a reasoning model that runs into ``max_tokens`` before emitting the
    closing tag leaves an *unclosed* block the paired-tag regex cannot match.
    The leftover raw reasoning — which often quotes the summarizer system prompt
    verbatim — would then be accepted as the summary purely because it is shorter
    than the source. When the stripped remainder is empty, or still begins with
    an (unclosed) reasoning marker, treat the result as unusable and return
    ``""`` so the caller escalates to the next model / L2 / deterministic
    fallback instead of persisting reasoning as the summary.
    """
    if not isinstance(text, str):
        return ""
    stripped = _strip_reasoning_blocks(text).strip()
    if not stripped or _REASONING_START_RE.match(stripped):
        return ""
    return stripped


def _call_llm_for_summary(prompt: str, max_tokens: int,
                           model: str = "", timeout: float | None = None) -> Optional[str]:
    """Call the Hermes auxiliary LLM for summarization."""
    try:
        from agent.auxiliary_client import call_llm
        call_kwargs = {
            "task": "compression",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }
        apply_lcm_model_route(call_kwargs, model)
        if timeout is not None:
            call_kwargs["timeout"] = timeout
        response = call_llm(**call_kwargs)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            content = str(content) if content else ""
        sanitized = _sanitize_reasoning_summary(content)
        if content.strip() and not sanitized:
            logger.warning(
                "LCM summary discarded reasoning-only output (model=%s); escalating",
                model or "<default>",
            )
        return sanitized
    except Exception as e:
        logger.warning("LLM summarization failed: %s", e)
        return None


def _invoke_summary_llm(prompt: str, max_tokens: int, model: str = "", timeout: float | None = None) -> Optional[str]:
    kwargs = {"model": model} if model else {}
    if timeout is not None:
        try:
            sig = inspect.signature(_call_llm_for_summary)
            if "timeout" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                kwargs["timeout"] = timeout
        except Exception:
            pass
    return _call_llm_for_summary(prompt, max_tokens, **kwargs)


def _normalized_focus_topic(focus_topic: str, max_chars: int = 160) -> str:
    """Return a single-line, bounded focus topic for prompt injection."""
    normalized = " ".join(str(focus_topic or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 1)].rstrip() + "…"


# Historical section headings — mirror upstream hermes-agent constants so that
# the summariser has consistent structural anchors for grouping stale content.
# These headings act as summariser guidance, not an enforced active-context
# contract: _assemble_context() passes node.summary through as ordinary content,
# so headings influence LLM attention rather than being hard reference-only
# markers.  The practical effect is that LLMs naturally down-weight content
# under "Historical" headings, but no code path enforces the boundary.
# (hermes-agent issue #9631: iterative compaction kept completed topics alive.
#  PR #44687 adds auto-derive focus topic; PR #44454 salvaged #44345/#41650
#  and introduced HISTORICAL_*_HEADING constants [8f8cad7ec / d5e2fbf24]
#  for structural demote of stale/completed topics.)
_HISTORICAL_HEADING_MARKERS = (
    "## Historical Task Snapshot",
    "## Historical In-Progress State",
    "## Historical Pending User Asks",
    "## Historical Remaining Work",
)


def _build_l1_focus_brief(focus_topic: str) -> str:
    """Build L1 focus guidance with explicit demote instructions for stale topics.

    Mirrors upstream hermes-agent PR #44687 (auto-derive focus topic) and
    PR #44454 (historical heading constants + stale-task demotion) to prevent
    iterative compaction from keeping completed topics alive and overriding
    the current active topic (issue #9631).
    """
    topic = _normalized_focus_topic(focus_topic)
    if not topic:
        return ""
    markers = " / ".join(f"'{m}'" for m in _HISTORICAL_HEADING_MARKERS)
    return (
        "Focus brief:\n"
        f"Primary focus: {topic}\n"
        "Preserve concrete decisions, constraints, files, commands, identifiers, and current state for this focus.\n"
        "Spend roughly 60-70% of the summary token budget on the focus when relevant.\n"
        "\n"
        "Demote old / completed topics:\n"
        "If the summary contains tasks, questions, or remaining work that are no longer active in the latest turns,\n"
        f"mark them under one of these historical headings: {markers}.\n"
        "Frame them as STALE context — the agent must NOT resume that work unless the latest user message\n"
        "explicitly asks for it. If fully resolved, reduce to a one-line bullet or omit.\n"
        "Exception: active blockers or handoff state should NOT be demoted even if they are absent from the\n"
        "latest turns. Keep blockers and pending handoffs outside historical headings so the agent can still act on them.\n"
    )


def _build_l2_focus_brief(focus_topic: str) -> str:
    """Build L2 focus guidance with explicit demote instructions for stale topics.

    Mirrors upstream hermes-agent PR #44687 (auto-focus) and PR #44454
    (historical heading constants + stale-task demotion).
    """
    topic = _normalized_focus_topic(focus_topic)
    if not topic:
        return ""
    markers = " / ".join(f"'{m}'" for m in _HISTORICAL_HEADING_MARKERS)
    return (
        "Focus brief:\n"
        f"Primary focus: {topic}\n"
        "Prefer bullets that preserve decisions, blockers, files, commands, identifiers, and current state for this focus.\n"
        "Keep other active tasks only when they are current blockers or handoff state.\n"
        "\n"
        "Demote old / completed topics:\n"
        f"Place non-current work under: {markers}.\n"
        "These sections are STALE — the agent must not act on them unless the latest user message explicitly\n"
        "requests it. Reduce resolved topics to one-liners or drop.\n"
        "Exception: active blockers and pending handoff state should NOT be demoted even when absent from recent\n"
        "turns. Keep them outside historical headings so the agent retains awareness of unresolved constraints.\n"
    )


def _summary_model_chain(primary_model: str = "", fallback_models: list[str] | tuple[str, ...] | None = None) -> list[str]:
    chain: list[str] = []
    for model in [primary_model, *(fallback_models or [])]:
        normalized = (model or "").strip()
        if normalized not in chain:
            chain.append(normalized)
    if not chain:
        chain.append("")
    return chain


def _invoke_summary_llm_chain(
    prompt: str,
    max_tokens: int,
    *,
    model: str = "",
    fallback_models: list[str] | tuple[str, ...] | None = None,
    timeout: float | None = None,
    circuit_breaker: SummaryCircuitBreaker | None = None,
    spend_guard: "SummarySpendGuard | None" = None,
    accepts_result: Callable[[str], bool] | None = None,
) -> Optional[str]:
    chain = _summary_model_chain(model, fallback_models)
    skipped = 0
    for candidate_model in chain:
        if circuit_breaker is not None and not circuit_breaker.allows(candidate_model):
            skipped += 1
            logger.warning(
                "LCM summary route skipped by open circuit: %s",
                candidate_model or _DEFAULT_ROUTE_KEY,
            )
            continue
        # Check the spend guard per-route so a mid-chain trip stops the
        # remaining fallbacks instead of over-spending by up to len(chain)-1.
        if spend_guard is not None and not spend_guard.allows():
            logger.warning(
                "LCM summary spend guard active; skipping LLM summarization and "
                "deferring to deterministic fallback"
            )
            break
        if spend_guard is not None:
            spend_guard.record_call()
        try:
            result = _invoke_summary_llm(
                prompt,
                max_tokens,
                model=candidate_model,
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning("LLM summarization failed: %s", exc)
            result = None
        if result and (accepts_result is None or accepts_result(result)):
            if circuit_breaker is not None:
                circuit_breaker.record_success(candidate_model)
            return result
        if circuit_breaker is not None:
            circuit_breaker.record_failure(candidate_model)
    if skipped == len(chain):
        logger.warning("LCM summary fallback chain exhausted: all routes are temporarily open")
    return None


def _build_l1_prompt(text: str, token_budget: int, depth: int,
                     focus_topic: str = "", custom_instructions: str = "") -> str:
    """Level 1: preserve details."""
    depth_guidance = {
        0: "Preserve decisions, rationale, constraints, active tasks, file paths, commands, and specific values.",
        1: "Distill into arc-level outcomes: what evolved, what was decided, current state. Drop per-turn detail.",
        2: "Capture durable narrative: decisions in effect, completed milestones, timeline. Drop process detail.",
    }
    guidance = depth_guidance.get(depth, depth_guidance[2])

    focus_guidance = _build_l1_focus_brief(focus_topic)

    custom_block = ""
    if custom_instructions:
        custom_block = f"\nAdditional instructions:\n{custom_instructions}\n"

    return f"""Summarize this conversation segment for future turns.
{guidance}
Remove repetition and conversational filler.
End with: "Expand for details about: <what was compressed>"
{focus_guidance}{custom_block}

Target ~{token_budget} tokens.

CONTENT:
{text}"""


def _build_l2_prompt(text: str, token_budget: int,
                     focus_topic: str = "", custom_instructions: str = "") -> str:
    """Level 2: aggressive bullet points."""
    focus_guidance = _build_l2_focus_brief(focus_topic)

    custom_block = ""
    if custom_instructions:
        custom_block = f"\nAdditional instructions:\n{custom_instructions}\n"

    return f"""Compress this into bullet points. Maximum {token_budget} tokens.
Keep only: decisions made, files changed, errors hit, current state.
Drop all reasoning, alternatives considered, and process detail.
{focus_guidance}{custom_block}

CONTENT:
{text}"""


_L3_TRUNCATION_MARKER = (
    "\n\n[...deterministic truncation — details available via lcm_expand...]\n\n"
)


def _truncate_text_to_tokens(text: str, max_tokens: int, *, from_end: bool = False) -> str:
    """Truncate ``text`` to at most ``max_tokens`` tokens for L3 fallback."""
    if max_tokens <= 0 or not text:
        return ""
    enc = _token_module._get_encoder()
    if enc is not None:
        try:
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            kept = tokens[-max_tokens:] if from_end else tokens[:max_tokens]
            return enc.decode(kept)
        except Exception:
            pass
    if count_tokens(text) <= max_tokens:
        return text
    length = len(text)
    non_ascii = 0 if text.isascii() else sum(1 for ch in text if ord(ch) > 127)
    ratio = (non_ascii / length) if length else 0.0
    if ratio >= 0.5:
        divisor = 1.5
    elif ratio >= 0.2:
        divisor = 2.5
    else:
        divisor = _token_module._CHARS_PER_TOKEN
    char_budget = max(1, int(max_tokens * divisor))
    # The estimate is approximate; correct any overshoot in a few bounded steps
    # so the returned slice never exceeds the token budget.
    for _ in range(8):
        candidate = text[-char_budget:] if from_end else text[:char_budget]
        estimated = count_tokens(candidate)
        if estimated <= max_tokens or char_budget <= 1:
            return candidate
        char_budget = max(1, int(char_budget * max_tokens / estimated) - 1)
    return text[-char_budget:] if from_end else text[:char_budget]


def _deterministic_truncate(text: str, max_tokens: int) -> str:
    """Level 3: no LLM, just truncate deterministically.

    Keeps the first and last portions to preserve start context and most recent
    state. Guaranteed to converge. Budgeted in *tokens* via the tiktoken encoder
    (not a flat chars*4 estimate), so the result honours ``max_tokens`` even for
    CJK / dense scripts, where chars*4 overshoots ~2-4x and would defeat the very
    budget L3 exists to guarantee.
    """
    if count_tokens(text) <= max_tokens:
        return text

    marker_tokens = count_tokens(_L3_TRUNCATION_MARKER)
    if max_tokens <= marker_tokens + 4:
        # Budget too small to afford the head/tail marker; single head cut.
        return _truncate_text_to_tokens(text, max_tokens)

    def assemble(body_tokens: int) -> str:
        head_tokens = body_tokens // 2
        tail_tokens = body_tokens - head_tokens
        head = _truncate_text_to_tokens(text, head_tokens)
        tail = _truncate_text_to_tokens(text, tail_tokens, from_end=True)
        return head + _L3_TRUNCATION_MARKER + tail

    # ``count_tokens`` is exact with tiktoken, but the no-tiktoken fallback is
    # intentionally a script-density estimate and is not additive: counting the
    # CJK head, ASCII marker, and CJK tail separately can fit while the combined
    # string exceeds ``max_tokens``. Binary search the body budget against the
    # final assembled result so L3 is bounded under both counters.
    best = _L3_TRUNCATION_MARKER
    low = 0
    high = max_tokens - marker_tokens
    while low <= high:
        body_tokens = (low + high) // 2
        candidate = assemble(body_tokens)
        if count_tokens(candidate) <= max_tokens:
            best = candidate
            low = body_tokens + 1
        else:
            high = body_tokens - 1
    return best


def summarize_with_escalation(
    text: str,
    source_tokens: int,
    token_budget: int,
    depth: int = 0,
    model: str = "",
    timeout: float | None = None,
    l2_budget_ratio: float = 0.50,
    l3_truncate_tokens: int = 512,
    focus_topic: str = "",
    custom_instructions: str = "",
    fallback_models: list[str] | tuple[str, ...] | None = None,
    circuit_breaker: SummaryCircuitBreaker | None = None,
    spend_guard: "SummarySpendGuard | None" = None,
) -> tuple[str, int]:
    """Run 3-level escalation. Returns (summary, level_used).

    Guarantees convergence: level 3 is deterministic and always produces
    output shorter than the source.
    """
    # Level 1: detailed summary
    l1_prompt = _build_l1_prompt(text, token_budget, depth,
                                 focus_topic=focus_topic,
                                 custom_instructions=custom_instructions)
    l1_result = _invoke_summary_llm_chain(
        l1_prompt,
        token_budget * 2,
        model=model,
        fallback_models=fallback_models,
        timeout=timeout,
        circuit_breaker=circuit_breaker,
        spend_guard=spend_guard,
        accepts_result=lambda result: count_tokens(result) < source_tokens,
    )

    if l1_result:
        logger.debug("L1 summarization succeeded (%d tokens)", count_tokens(l1_result))
        return l1_result, 1

    # Level 2: aggressive bullets at reduced budget
    l2_budget = int(token_budget * l2_budget_ratio)
    l2_prompt = _build_l2_prompt(text, l2_budget,
                                 focus_topic=focus_topic,
                                 custom_instructions=custom_instructions)
    l2_result = _invoke_summary_llm_chain(
        l2_prompt,
        l2_budget * 2,
        model=model,
        fallback_models=fallback_models,
        timeout=timeout,
        circuit_breaker=circuit_breaker,
        spend_guard=spend_guard,
        accepts_result=lambda result: count_tokens(result) < source_tokens,
    )

    if l2_result:
        logger.debug("L2 summarization succeeded (%d tokens)", count_tokens(l2_result))
        return l2_result, 2

    # Level 3: deterministic truncation — guaranteed convergence
    l3_result = _deterministic_truncate(text, l3_truncate_tokens)
    logger.debug("L3 deterministic truncation (%d tokens)", count_tokens(l3_result))
    return l3_result, 3
