"""LCM configuration with defaults and env var overrides."""

import os
from dataclasses import dataclass, field


@dataclass
class LCMConfig:
    """All tunables for the LCM engine."""

    # -- Fresh tail: recent messages never compacted ---
    fresh_tail_count: int = 64

    # -- Compaction thresholds ---
    # Max source tokens in a leaf chunk before summarization triggers
    leaf_chunk_tokens: int = 20_000
    # Fraction of context window that triggers compaction (0.0–1.0)
    context_threshold: float = 0.75
    # Max condensation depth (-1 = unlimited, 0 = leaf only)
    incremental_max_depth: int = 1
    # How many same-depth summaries trigger condensation
    condensation_fanin: int = 4

    # -- Escalation ---
    # L2 bullet budget as fraction of L1
    l2_budget_ratio: float = 0.50
    # L3 deterministic truncate token limit
    l3_truncate_tokens: int = 512

    # -- Assembly guardrails ---
    # Hard cap for the assembled active context (0 = disabled)
    max_assembly_tokens: int = 0
    # Reserve this many tokens from the model context window before assembly
    # (0 = disabled). Effective cap becomes context_length - reserve_tokens_floor.
    reserve_tokens_floor: int = 0

    # -- Models ---
    summary_model: str = ""       # empty = use Hermes auxiliary model

    # -- Storage ---
    database_path: str = ""       # empty = auto (~/.hermes/lcm.db)

    # -- Session carry-over ---
    # Depth retained after /new (-1 = all, 0 = nothing, 2 = keep d2+)
    new_session_retain_depth: int = 2

    @classmethod
    def from_env(cls) -> "LCMConfig":
        """Build config from environment variables (LCM_ prefix)."""
        c = cls()
        _int = lambda key, default: int(os.environ.get(key, default))
        _float = lambda key, default: float(os.environ.get(key, default))
        _str = lambda key, default: os.environ.get(key, default)

        c.fresh_tail_count = _int("LCM_FRESH_TAIL_COUNT", c.fresh_tail_count)
        c.leaf_chunk_tokens = _int("LCM_LEAF_CHUNK_TOKENS", c.leaf_chunk_tokens)
        c.context_threshold = _float("LCM_CONTEXT_THRESHOLD", c.context_threshold)
        c.incremental_max_depth = _int("LCM_INCREMENTAL_MAX_DEPTH", c.incremental_max_depth)
        c.condensation_fanin = _int("LCM_CONDENSATION_FANIN", c.condensation_fanin)
        c.l2_budget_ratio = _float("LCM_L2_BUDGET_RATIO", c.l2_budget_ratio)
        c.l3_truncate_tokens = _int("LCM_L3_TRUNCATE_TOKENS", c.l3_truncate_tokens)
        c.max_assembly_tokens = _int("LCM_MAX_ASSEMBLY_TOKENS", c.max_assembly_tokens)
        c.reserve_tokens_floor = _int("LCM_RESERVE_TOKENS_FLOOR", c.reserve_tokens_floor)
        c.summary_model = _str("LCM_SUMMARY_MODEL", c.summary_model)
        c.database_path = _str("LCM_DATABASE_PATH", c.database_path)
        c.new_session_retain_depth = _int("LCM_NEW_SESSION_RETAIN_DEPTH", c.new_session_retain_depth)

        return c
