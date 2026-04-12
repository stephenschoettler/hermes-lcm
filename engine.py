"""LCM Engine — Lossless Context Management.

Implements the ContextEngine ABC. Replaces the built-in ContextCompressor
with a DAG-based summarization system that preserves every message.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

from .config import LCMConfig
from .dag import SummaryDAG, SummaryNode
from .escalation import summarize_with_escalation
from .schemas import LCM_DESCRIBE, LCM_EXPAND, LCM_GREP
from .store import MessageStore
from .tokens import count_message_tokens, count_messages_tokens, count_tokens
from . import tools as lcm_tools

logger = logging.getLogger(__name__)


class LCMEngine(ContextEngine):
    """Lossless Context Management engine.

    Architecture:
      1. Every message is persisted verbatim in an immutable MessageStore
      2. When context pressure builds, older messages outside the fresh tail
         are summarized into leaf nodes (D0) in a SummaryDAG
      3. When enough nodes accumulate at a depth, they're condensed into
         higher-depth nodes (D1, D2, ...)
      4. The agent gets tools (lcm_grep, lcm_describe, lcm_expand) to
         search and drill into compacted history
      5. Active context = system prompt + DAG summaries + fresh tail
    """

    def __init__(self, config: LCMConfig | None = None,
                 hermes_home: str = ""):
        self._config = config or LCMConfig.from_env()
        self._hermes_home = hermes_home

        # Resolve DB path
        if self._config.database_path:
            db_path = Path(self._config.database_path)
        elif hermes_home:
            db_path = Path(hermes_home) / "lcm.db"
        else:
            db_path = Path.home() / ".hermes" / "lcm.db"

        self._store = MessageStore(db_path)
        self._dag = SummaryDAG(db_path)

        self._session_id: str = ""

        # Track which store_ids have been ingested into the DAG
        self._last_compacted_store_id: int = 0

        # Cursor: index in the current messages list up to which all
        # messages have been persisted.  After compress() shortens the
        # list, the cursor resets to len(compressed) so that only
        # genuinely new messages (appended after compaction) get ingested.
        self._ingest_cursor: int = 0

        # State required by ContextEngine ABC and run_agent.py compatibility
        self.model = ""
        self.base_url = ""
        self.api_key = ""
        self.provider = ""
        self.context_length = 0
        self.threshold_tokens = 0
        self.threshold_percent = self._config.context_threshold
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        # run_agent.py reads these for preflight checks
        self.protect_first_n = 3
        self.protect_last_n = self._config.fresh_tail_count
        # run_agent.py reads these for context probing
        self._context_probed = False
        self._context_probe_persistable = False
        self.quiet_mode = False
        self.summary_model = self._config.summary_model

    @property
    def name(self) -> str:
        return "lcm"

    # -- ContextEngine required methods ------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if self.threshold_tokens <= 0:
            return False
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages):
        """Pre-flight check — also ingests messages into the store."""
        if self._session_id and messages:
            try:
                self._ingest_messages(messages)
            except Exception as e:
                logger.debug("Ingest during preflight: %s", e)
        from .tokens import count_messages_tokens
        rough = count_messages_tokens(messages)
        return rough >= self.threshold_tokens

    def compress(self, messages: List[Dict[str, Any]],
                 current_tokens: int = None) -> List[Dict[str, Any]]:
        """Main compaction entry point.

        1. Ingest any new messages into the store
        2. Identify messages outside the fresh tail
        3. Summarize them into DAG leaf nodes
        4. Check if condensation is needed
        5. Assemble new active context: summaries + fresh tail
        """
        if not messages:
            return messages

        prompt_tokens = current_tokens or self.last_prompt_tokens

        # Step 1: Ingest new messages into the immutable store
        self._ingest_messages(messages)

        # Step 2: Identify fresh tail boundary
        n = len(messages)
        fresh_tail_start = max(0, n - self._config.fresh_tail_count)

        # Protect system prompt (always index 0)
        if fresh_tail_start <= 1:
            # Not enough messages to compact
            return messages

        # Step 3: Identify messages to compact (between system prompt and fresh tail)
        # Skip system prompt (index 0), compact indices 1..fresh_tail_start
        to_compact = messages[1:fresh_tail_start]

        if not to_compact:
            return messages

        # Calculate source tokens
        source_tokens = count_messages_tokens(to_compact)
        if source_tokens < self._config.leaf_chunk_tokens:
            # Not enough to justify compaction
            return messages

        # Step 4: Serialize and summarize
        serialized = self._serialize_messages(to_compact)
        token_budget = max(2000, int(source_tokens * 0.20))
        token_budget = min(token_budget, 12000)

        summary_text, level = summarize_with_escalation(
            text=serialized,
            source_tokens=source_tokens,
            token_budget=token_budget,
            depth=0,
            model=self._config.summary_model,
            l2_budget_ratio=self._config.l2_budget_ratio,
            l3_truncate_tokens=self._config.l3_truncate_tokens,
        )

        # Step 5: Create DAG node
        # Collect store_ids for source tracking
        source_store_ids = self._get_store_ids_for_messages(to_compact)

        node = SummaryNode(
            session_id=self._session_id,
            depth=0,
            summary=summary_text,
            token_count=count_tokens(summary_text),
            source_token_count=source_tokens,
            source_ids=source_store_ids,
            source_type="messages",
            created_at=time.time(),
            expand_hint=self._extract_expand_hint(summary_text),
        )
        self._dag.add_node(node)
        self._last_compacted_store_id = max(source_store_ids) if source_store_ids else 0

        # Step 6: Check if condensation is needed
        self._maybe_condense()

        # Step 7: Assemble new active context
        compressed = self._assemble_context(messages[0], messages[fresh_tail_start:])
        self.compression_count += 1
        # Reset cursor to the length of the compressed context so that
        # only messages appended *after* this point get ingested next time.
        self._ingest_cursor = len(compressed)

        logger.info(
            "LCM compaction #%d: %d messages → %d (L%d, %d→%d tokens, %d DAG nodes)",
            self.compression_count, n, len(compressed), level,
            source_tokens, count_tokens(summary_text),
            len(self._dag.get_session_nodes(self._session_id)),
        )

        return compressed

    # -- ContextEngine optional methods ------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        if "hermes_home" in kwargs:
            self._hermes_home = kwargs["hermes_home"]
        # Pick up context_length from kwargs if provided
        if "context_length" in kwargs:
            self.context_length = kwargs["context_length"]
            self.threshold_tokens = int(
                self.context_length * self._config.context_threshold
            )

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        # Ensure all messages are persisted
        self._ingest_messages(messages)

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._last_compacted_store_id = 0
        self._ingest_cursor = 0
        self._context_probed = False
        self._context_probe_persistable = False

        # Retain DAG nodes across sessions based on config.
        #   -1  → keep all nodes
        #    0  → delete everything
        #    N  → keep nodes at depth >= N (e.g. 2 keeps d2+)
        retain = self._config.new_session_retain_depth
        if self._session_id and retain != -1:
            if retain == 0:
                self._dag.delete_session_nodes(self._session_id)
            else:
                self._dag.delete_below_depth(self._session_id, retain)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [LCM_GREP, LCM_DESCRIBE, LCM_EXPAND]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        # Ingest live messages if passed (enables current-turn search)
        messages = kwargs.get("messages")
        
        if messages and self._session_id:
            try:
                self._ingest_messages(messages)
            except Exception as e:
                logger.debug("Ingest during tool call failed: %s", e)

        handlers = {
            "lcm_grep": lcm_tools.lcm_grep,
            "lcm_describe": lcm_tools.lcm_describe,
            "lcm_expand": lcm_tools.lcm_expand,
        }
        handler = handlers.get(name)
        if handler:
            return handler(args, engine=self)
        return json.dumps({"error": f"Unknown LCM tool: {name}"})

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        if self._session_id:
            status["engine"] = "lcm"
            status["store_messages"] = self._store.get_session_count(self._session_id)
            status["dag_nodes"] = len(self._dag.get_session_nodes(self._session_id))
        return status

    def update_model(self, model: str, context_length: int,
                     base_url: str = "", api_key: str = "",
                     provider: str = "") -> None:
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self._config.context_threshold)

    # -- Internal: message ingestion ---------------------------------------

    def _ingest_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Persist new messages to the store.

        Uses a cursor to track which portion of the current messages list
        has already been persisted.  After compress() shortens the list,
        the cursor is reset to len(compressed), so only messages appended
        after compaction are ingested — regardless of how the store count
        compares to the current list length.
        """
        if not self._session_id:
            logger.debug("Ingest skipped: no session_id")
            return

        n = len(messages)
        cursor = self._ingest_cursor
        logger.debug(
            "Ingest: session=%s cursor=%d incoming=%d",
            self._session_id, cursor, n,
        )

        new_messages = messages[cursor:] if cursor < n else []

        if not new_messages:
            return

        estimates = [count_message_tokens(m) for m in new_messages]
        self._store.append_batch(self._session_id, new_messages, estimates)
        self._ingest_cursor = n
        logger.debug("Ingested %d messages into LCM store", len(new_messages))

    def _get_store_ids_for_messages(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Map message list back to store_ids via content matching.

        Uses (role, content) to match against stored messages.  Robust
        across compaction cycles — synthetic summary messages that aren't
        in the store are silently skipped.
        """
        from collections import defaultdict

        all_stored = self._store.get_session_messages(self._session_id)
        # Build ordered lookup: (role, content) → [store_id, ...]
        lookup: dict[tuple, list[int]] = defaultdict(list)
        for s in all_stored:
            key = (s.get("role", ""), s.get("content") or "")
            lookup[key].append(s["store_id"])

        ids = []
        consumed: set[int] = set()
        for msg in messages:
            key = (msg.get("role", ""), msg.get("content") or "")
            for sid in lookup.get(key, []):
                if sid not in consumed:
                    ids.append(sid)
                    consumed.add(sid)
                    break

        return ids

    # -- Internal: summarization -------------------------------------------

    def _serialize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Serialize messages into labeled text for the summarizer."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""

            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                if len(content) > 3000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            if role == "assistant":
                if len(content) > 3000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = fn.get("arguments", "")
                            if len(args) > 500:
                                args = args[:400] + "..."
                            tc_parts.append(f"  {name}({args})")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            if len(content) > 3000:
                content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
            parts.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(parts)

    # -- Internal: condensation --------------------------------------------

    def _maybe_condense(self) -> None:
        """Check if any depth level has enough nodes for condensation."""
        max_depth = self._config.incremental_max_depth
        if max_depth == 0:
            return  # condensation disabled

        # When max_depth is -1 (unlimited), derive the upper bound from
        # the deepest existing node + 1, so condensation can always
        # create the next depth level.
        if max_depth < 0:
            all_nodes = self._dag.get_session_nodes(self._session_id)
            upper = (max(n.depth for n in all_nodes) + 1) if all_nodes else 1
        else:
            upper = max_depth

        for depth in range(upper):
            uncondensed = self._dag.get_uncondensed_at_depth(
                self._session_id, depth
            )
            if len(uncondensed) < self._config.condensation_fanin:
                continue

            # Take the first fanin nodes and condense
            to_condense = uncondensed[:self._config.condensation_fanin]
            combined_text = "\n\n---\n\n".join(n.summary for n in to_condense)
            source_tokens = sum(n.token_count for n in to_condense)
            token_budget = max(1000, int(source_tokens * 0.40))

            summary_text, level = summarize_with_escalation(
                text=combined_text,
                source_tokens=source_tokens,
                token_budget=token_budget,
                depth=depth + 1,
                model=self._config.summary_model,
                l2_budget_ratio=self._config.l2_budget_ratio,
                l3_truncate_tokens=self._config.l3_truncate_tokens,
            )

            node = SummaryNode(
                session_id=self._session_id,
                depth=depth + 1,
                summary=summary_text,
                token_count=count_tokens(summary_text),
                source_token_count=source_tokens,
                source_ids=[n.node_id for n in to_condense],
                source_type="nodes",
                created_at=time.time(),
                expand_hint=self._extract_expand_hint(summary_text),
            )
            self._dag.add_node(node)

            logger.info(
                "LCM condensation: d%d × %d → d%d (L%d, %d→%d tokens)",
                depth, len(to_condense), depth + 1, level,
                source_tokens, count_tokens(summary_text),
            )

    # -- Internal: context assembly ----------------------------------------

    def _assemble_context(self, system_msg: Dict[str, Any],
                          tail_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build the active context from DAG summaries + fresh tail.

        Structure:
          [system prompt (with LCM note)]
          [highest-depth summary nodes first, then lower]
          [fresh tail messages]
        """
        result = []

        # System prompt with LCM annotation
        sys_msg = system_msg.copy()
        if self.compression_count == 0:
            sys_content = sys_msg.get("content", "")
            sys_msg["content"] = (
                sys_content
                + "\n\n[Note: This conversation uses Lossless Context Management (LCM). "
                "Earlier turns have been compacted into hierarchical summaries below. "
                "Use lcm_grep to search history, lcm_describe to inspect the DAG, "
                "and lcm_expand to recover original details from any summary.]"
            )
        result.append(sys_msg)

        # Collect DAG summaries — highest depth first for context hierarchy
        all_nodes = self._dag.get_session_nodes(self._session_id)
        if all_nodes:
            # Group by depth, take the most recent uncondensed at each level
            # For active context, we want the highest-level summaries
            # that haven't been condensed into even higher levels
            depths = sorted(set(n.depth for n in all_nodes), reverse=True)
            summary_parts = []
            for d in depths:
                uncondensed = self._dag.get_uncondensed_at_depth(self._session_id, d)
                for node in uncondensed:
                    depth_label = {
                        0: "Recent",
                        1: "Session Arc",
                        2: "Durable",
                    }.get(d, f"Depth-{d}")
                    summary_parts.append(
                        f"[{depth_label} Summary (d{d}, node {node.node_id})]\n"
                        f"{node.summary}\n"
                        f"[Expand for details: {node.expand_hint}]"
                    )

            if summary_parts:
                combined = "\n\n---\n\n".join(summary_parts)
                # Choose role to avoid consecutive same-role
                last_role = result[-1].get("role", "system")
                summary_role = "assistant" if last_role != "assistant" else "user"
                result.append({"role": summary_role, "content": combined})

        # Fresh tail
        result.extend(tail_messages)

        return result

    # -- Internal: helpers -------------------------------------------------

    @staticmethod
    def _extract_expand_hint(summary: str) -> str:
        """Extract the 'Expand for details about:' line from a summary."""
        marker = "Expand for details about:"
        idx = summary.rfind(marker)
        if idx >= 0:
            hint = summary[idx + len(marker):].strip()
            # Take first line only
            return hint.split("\n")[0].strip()
        return ""

    # -- Lifecycle ---------------------------------------------------------

    def shutdown(self):
        self._store.close()
        self._dag.close()
