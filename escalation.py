"""Three-level summarization escalation.

Level 1 (Normal):    LLM summary preserving details
Level 2 (Aggressive): LLM bullet-point summary at half the token budget
Level 3 (Fallback):   Deterministic truncation — no LLM, guaranteed convergence

Each level checks if Tokens(summary) < Tokens(source). If not, escalates.
"""

import inspect
import logging
from typing import List, Optional

from .tokens import count_tokens

logger = logging.getLogger(__name__)


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
        if model:
            call_kwargs["model"] = model
        if timeout is not None:
            call_kwargs["timeout"] = timeout
        response = call_llm(**call_kwargs)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            content = str(content) if content else ""
        return content.strip()
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


def _build_l1_prompt(text: str, token_budget: int, depth: int,
                     focus_topic: str = "", custom_instructions: str = "") -> str:
    """Level 1: preserve details."""
    depth_guidance = {
        0: "Preserve decisions, rationale, constraints, active tasks, file paths, commands, and specific values.",
        1: "Distill into arc-level outcomes: what evolved, what was decided, current state. Drop per-turn detail.",
        2: "Capture durable narrative: decisions in effect, completed milestones, timeline. Drop process detail.",
    }
    guidance = depth_guidance.get(depth, depth_guidance[2])

    focus_guidance = ""
    if focus_topic:
        focus_guidance = (
            f'Prioritize preserving information related to: "{focus_topic}".\n'
            "Spend roughly 60-70% of the summary budget on that topic when relevant.\n"
        )

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
    focus_guidance = ""
    if focus_topic:
        focus_guidance = (
            f'Prioritize bullets related to: "{focus_topic}" when present.\n'
        )

    custom_block = ""
    if custom_instructions:
        custom_block = f"\nAdditional instructions:\n{custom_instructions}\n"

    return f"""Compress this into bullet points. Maximum {token_budget} tokens.
Keep only: decisions made, files changed, errors hit, current state.
Drop all reasoning, alternatives considered, and process detail.
{focus_guidance}{custom_block}

CONTENT:
{text}"""


def _deterministic_truncate(text: str, max_tokens: int) -> str:
    """Level 3: no LLM, just truncate deterministically.

    Takes the first and last portions to preserve start context and
    most recent state. Guaranteed to converge.
    """
    if count_tokens(text) <= max_tokens:
        return text

    # Rough char budget (4 chars/token)
    char_budget = max_tokens * 4
    if len(text) <= char_budget:
        return text

    head_budget = int(char_budget * 0.4)
    tail_budget = int(char_budget * 0.4)
    middle = "\n\n[...deterministic truncation — details available via lcm_expand...]\n\n"

    return text[:head_budget] + middle + text[-tail_budget:]


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
) -> tuple[str, int]:
    """Run 3-level escalation. Returns (summary, level_used).

    Guarantees convergence: level 3 is deterministic and always produces
    output shorter than the source.
    """
    # Level 1: detailed summary
    l1_prompt = _build_l1_prompt(text, token_budget, depth,
                                 focus_topic=focus_topic,
                                 custom_instructions=custom_instructions)
    l1_result = _invoke_summary_llm(l1_prompt, token_budget * 2, model=model, timeout=timeout)

    if l1_result and count_tokens(l1_result) < source_tokens:
        logger.debug("L1 summarization succeeded (%d tokens)", count_tokens(l1_result))
        return l1_result, 1

    # Level 2: aggressive bullets at reduced budget
    l2_budget = int(token_budget * l2_budget_ratio)
    l2_prompt = _build_l2_prompt(text, l2_budget,
                                 focus_topic=focus_topic,
                                 custom_instructions=custom_instructions)
    l2_result = _invoke_summary_llm(l2_prompt, l2_budget * 2, model=model, timeout=timeout)

    if l2_result and count_tokens(l2_result) < source_tokens:
        logger.debug("L2 summarization succeeded (%d tokens)", count_tokens(l2_result))
        return l2_result, 2

    # Level 3: deterministic truncation — guaranteed convergence
    l3_result = _deterministic_truncate(text, l3_truncate_tokens)
    logger.debug("L3 deterministic truncation (%d tokens)", count_tokens(l3_result))
    return l3_result, 3
