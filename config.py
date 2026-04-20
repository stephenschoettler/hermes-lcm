"""LCM configuration with defaults and env var overrides."""
import os
from dataclasses import dataclass, field


def _parse_pattern_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_env(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _parse_float_env(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _parse_bool_env(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


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
    # When enabled, leaf compaction may use a larger working chunk size based on backlog pressure
    dynamic_leaf_chunk_enabled: bool = False
    # Upper bound for the working dynamic leaf chunk threshold
    dynamic_leaf_chunk_max: int = 40_000
    # When enabled, suppress follow-on condensation after a leaf pass unless
    # debt/pressure says the extra churn is worth it
    cache_friendly_condensation_enabled: bool = False
    # Minimum number of same-depth fanin groups before one follow-on
    # condensation pass is allowed in cache-friendly mode
    cache_friendly_min_debt_groups: int = 2
    # When enabled, turns can persist raw-backlog maintenance debt and use
    # later bounded catch-up passes to reduce it.
    deferred_maintenance_enabled: bool = False
    # Maximum extra leaf passes a debt-triggered later turn may spend on
    # catch-up work.
    deferred_maintenance_max_passes: int = 4

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

    # -- Session filtering ---
    # Sessions to exclude from LCM storage entirely.
    ignore_session_patterns: list[str] = field(default_factory=list)
    # Sessions that may read carried-over LCM state but never write new data.
    stateless_session_patterns: list[str] = field(default_factory=list)
    # Diagnostics: where each pattern list came from.
    ignore_session_patterns_source: str = "default"
    stateless_session_patterns_source: str = "default"

    # -- Summary instructions ---
    # Custom instructions injected into all summarization prompts
    custom_instructions: str = ""

    # -- Pre-compaction extraction ---
    # Extract decisions/commitments to files before compaction
    extraction_enabled: bool = False
    # Model for extraction (empty = fall back to summary_model)
    extraction_model: str = ""
    # Directory for daily extraction files (empty = auto: ~/.hermes/lcm-extractions/)
    extraction_output_path: str = ""

    # -- Large tool-output externalization ---
    # When enabled, oversized tool results are written to plugin-managed storage
    # and replaced with compact references in pre-compaction serializer input.
    large_output_externalization_enabled: bool = False
    # Character threshold above which tool results are externalized.
    large_output_externalization_threshold_chars: int = 12_000
    # Explicit storage directory for externalized payloads (empty = auto under hermes home).
    large_output_externalization_path: str = ""
    # When enabled, already-externalized summarized tool-result transcript rows may
    # be rewritten to compact GC placeholders after successful leaf compaction.
    large_output_transcript_gc_enabled: bool = False

    # -- Models ---
    summary_model: str = ""       # empty = use Hermes auxiliary model
    expansion_model: str = ""     # empty = fall back to summary_model / Hermes auxiliary model

    # -- Timeouts ---
    summary_timeout_ms: int = 60_000
    expansion_timeout_ms: int = 120_000

    # -- Storage ---
    database_path: str = ""       # empty = auto (~/.hermes/lcm.db)

    # -- Session carry-over ---
    # Depth retained after /new (-1 = all, 0 = nothing, 2 = keep d2+)
    new_session_retain_depth: int = 2
    # Safety gate: destructive `/lcm doctor clean apply` workflow is disabled by default.
    doctor_clean_apply_enabled: bool = False

    @classmethod
    def from_env(cls) -> "LCMConfig":
        """Build config from environment variables (LCM_ prefix)."""
        c = cls()
        _int = _parse_int_env
        _float = _parse_float_env
        _str = lambda key, default: os.environ.get(key, default)

        c.fresh_tail_count = _int("LCM_FRESH_TAIL_COUNT", c.fresh_tail_count)
        c.leaf_chunk_tokens = _int("LCM_LEAF_CHUNK_TOKENS", c.leaf_chunk_tokens)
        c.context_threshold = _float("LCM_CONTEXT_THRESHOLD", c.context_threshold)
        c.incremental_max_depth = _int("LCM_INCREMENTAL_MAX_DEPTH", c.incremental_max_depth)
        c.condensation_fanin = _int("LCM_CONDENSATION_FANIN", c.condensation_fanin)
        c.dynamic_leaf_chunk_enabled = _parse_bool_env(
            "LCM_DYNAMIC_LEAF_CHUNK_ENABLED", c.dynamic_leaf_chunk_enabled
        )
        c.dynamic_leaf_chunk_max = _int("LCM_DYNAMIC_LEAF_CHUNK_MAX", c.dynamic_leaf_chunk_max)
        c.cache_friendly_condensation_enabled = _parse_bool_env(
            "LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED",
            c.cache_friendly_condensation_enabled,
        )
        c.cache_friendly_min_debt_groups = _int(
            "LCM_CACHE_FRIENDLY_MIN_DEBT_GROUPS",
            c.cache_friendly_min_debt_groups,
        )
        c.deferred_maintenance_enabled = _parse_bool_env(
            "LCM_DEFERRED_MAINTENANCE_ENABLED",
            c.deferred_maintenance_enabled,
        )
        c.deferred_maintenance_max_passes = _int(
            "LCM_DEFERRED_MAINTENANCE_MAX_PASSES",
            c.deferred_maintenance_max_passes,
        )
        c.l2_budget_ratio = _float("LCM_L2_BUDGET_RATIO", c.l2_budget_ratio)
        c.l3_truncate_tokens = _int("LCM_L3_TRUNCATE_TOKENS", c.l3_truncate_tokens)
        c.max_assembly_tokens = _int("LCM_MAX_ASSEMBLY_TOKENS", c.max_assembly_tokens)
        c.reserve_tokens_floor = _int("LCM_RESERVE_TOKENS_FLOOR", c.reserve_tokens_floor)
        c.custom_instructions = _str("LCM_CUSTOM_INSTRUCTIONS", c.custom_instructions)
        c.extraction_enabled = _parse_bool_env("LCM_EXTRACTION_ENABLED", c.extraction_enabled)
        c.extraction_model = _str("LCM_EXTRACTION_MODEL", c.extraction_model)
        c.extraction_output_path = _str("LCM_EXTRACTION_OUTPUT_PATH", c.extraction_output_path)
        c.large_output_externalization_enabled = _parse_bool_env(
            "LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED",
            c.large_output_externalization_enabled,
        )
        c.large_output_externalization_threshold_chars = _int(
            "LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS",
            c.large_output_externalization_threshold_chars,
        )
        c.large_output_externalization_path = _str(
            "LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH",
            c.large_output_externalization_path,
        )
        c.large_output_transcript_gc_enabled = _parse_bool_env(
            "LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED",
            c.large_output_transcript_gc_enabled,
        )
        c.summary_model = _str("LCM_SUMMARY_MODEL", c.summary_model)
        c.expansion_model = _str("LCM_EXPANSION_MODEL", c.expansion_model)
        c.summary_timeout_ms = _int("LCM_SUMMARY_TIMEOUT_MS", c.summary_timeout_ms)
        c.expansion_timeout_ms = _int("LCM_EXPANSION_TIMEOUT_MS", c.expansion_timeout_ms)
        c.database_path = _str("LCM_DATABASE_PATH", c.database_path)
        c.new_session_retain_depth = _int("LCM_NEW_SESSION_RETAIN_DEPTH", c.new_session_retain_depth)
        c.doctor_clean_apply_enabled = _parse_bool_env(
            "LCM_DOCTOR_CLEAN_APPLY_ENABLED",
            c.doctor_clean_apply_enabled,
        )

        raw_ignore = os.environ.get("LCM_IGNORE_SESSION_PATTERNS")
        if raw_ignore is not None:
            c.ignore_session_patterns = _parse_pattern_list(raw_ignore)
            c.ignore_session_patterns_source = "env"

        raw_stateless = os.environ.get("LCM_STATELESS_SESSION_PATTERNS")
        if raw_stateless is not None:
            c.stateless_session_patterns = _parse_pattern_list(raw_stateless)
            c.stateless_session_patterns_source = "env"

        return c
