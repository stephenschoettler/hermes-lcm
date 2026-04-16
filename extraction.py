"""Pre-compaction extraction — extract decisions and commitments before summarization.

Best-effort: failures never block compaction. Extracted content is written to
daily note files so key decisions survive even if the DAG summary loses nuance.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract decisions, commitments, outcomes, and rules from this conversation segment.

Format as a flat list of bullet points. Each bullet should be self-contained and understandable
without the surrounding conversation. Include:
- Decisions made (what was chosen, and why if stated)
- Commitments (who will do what)
- Outcomes (what happened as a result of an action)
- Rules or constraints discovered

Skip: greetings, meta-discussion, reasoning that led nowhere, repeated information.
If there is nothing worth extracting, respond with exactly: NOTHING_TO_EXTRACT

CONTENT:
{text}"""


def _call_extraction_llm(prompt: str, model: str = "",
                          timeout: float | None = None) -> Optional[str]:
    """Call the Hermes auxiliary LLM for extraction."""
    try:
        from agent.auxiliary_client import call_llm
        call_kwargs = {
            "task": "extraction",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 2000,
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
        logger.debug("Extraction LLM call failed: %s", e)
        return None


def extract_before_compaction(
    serialized_messages: str,
    output_path: str,
    session_id: str = "",
    model: str = "",
    timeout: float | None = None,
) -> bool:
    """Extract decisions from messages about to be compacted and write to a daily file.

    Returns True if extraction succeeded, False otherwise.
    Never raises — failures are logged and swallowed.
    """
    try:
        prompt = EXTRACTION_PROMPT.format(text=serialized_messages)
        result = _call_extraction_llm(prompt, model=model, timeout=timeout)

        if not result or result.strip() == "NOTHING_TO_EXTRACT":
            logger.debug("Pre-compaction extraction: nothing to extract")
            return True

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = output_dir / f"{date_str}.md"

        header = f"\n\n## Extraction — {datetime.now().strftime('%H:%M')}"
        if session_id:
            header += f" ({session_id})"
        header += "\n\n"

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(result)
            f.write("\n")

        logger.info("Pre-compaction extraction written to %s", file_path)
        return True

    except Exception as e:
        logger.warning("Pre-compaction extraction failed (non-blocking): %s", e)
        return False
