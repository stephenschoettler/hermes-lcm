"""LCM configuration with defaults and env var overrides."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional fallback for minimal installs
    yaml = None


def _parse_pattern_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_env_with_source(
    key: str,
    default: int,
    *,
    default_source: str = "default",
) -> tuple[int, str, str | None]:
    raw = os.environ.get(key)
    if raw is None:
        return default, default_source, None
    try:
        return int(raw), f"env:{key}", None
    except (TypeError, ValueError):
        return default, default_source, f"invalid env {key}={raw!r} ignored"


def _parse_float_env_with_source(
    key: str,
    default: float,
    *,
    default_source: str = "default",
) -> tuple[float, str, str | None]:
    raw = os.environ.get(key)
    if raw is None:
        return default, default_source, None
    try:
        return float(raw), f"env:{key}", None
    except (TypeError, ValueError):
        return default, default_source, f"invalid env {key}={raw!r} ignored"


def _config_bool_disabled(value) -> bool:
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            return True
        try:
            return float(normalized) == 0
        except ValueError:
            return False
    return False


def _hermes_config_path() -> Path:
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    return home / "config.yaml"


def _load_hermes_config_yaml() -> dict[str, Any]:
    cfg_path = _hermes_config_path()
    try:
        text = cfg_path.read_text()
    except Exception:
        return {}
    if yaml is not None:
        try:
            loaded = yaml.safe_load(text) or {}
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        key, raw_value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root
        value = raw_value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            continue
        value = value.strip("'\"")
        lowered = value.lower()
        if lowered in {"true", "yes", "on"}:
            parsed: Any = True
        elif lowered in {"false", "no", "off"}:
            parsed = False
        else:
            try:
                parsed = float(value) if "." in value else int(value)
            except ValueError:
                parsed = value
        parent[key] = parsed
    return root


def _supported_lcm_config_yaml_keys() -> frozenset[str]:
    return frozenset({"context_threshold"} | {spec.name for spec in ENV_FIELD_SPECS})


def _ignored_lcm_config_yaml_keys(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg if cfg is not None else _load_hermes_config_yaml()
    lcm_section = cfg.get("lcm") if isinstance(cfg, dict) else None
    if not isinstance(lcm_section, dict):
        return []
    supported_keys = _supported_lcm_config_yaml_keys()
    return sorted(
        str(key)
        for key in lcm_section
        if str(key) not in supported_keys
    )


def _hermes_compression_threshold(default: float) -> float:
    """Read lcm.context_threshold or Hermes compression.threshold from config.yaml.

    Priority when no ``LCM_CONTEXT_THRESHOLD`` env var is set:
      1. ``lcm.context_threshold`` (LCM-specific override in config.yaml)
      2. ``compression.threshold`` (Hermes global setting, unless compression disabled)

    Hermes gateways may load ``~/.hermes/config.yaml`` without exporting every
    setting into the process environment. The ``lcm.context_threshold`` key lets
    operators tune LCM compaction independently of the Hermes compression setting.
    Disabled Hermes compression should not leak its threshold into LCM.
    """
    value, _source = _hermes_compression_threshold_with_source(default)
    return value


def _hermes_compression_threshold_with_source(
    default: float,
    cfg: dict[str, Any] | None = None,
) -> tuple[float, str]:
    cfg = cfg if cfg is not None else _load_hermes_config_yaml()
    try:
        lcm_section = cfg.get("lcm") or {}
        if isinstance(lcm_section, dict):
            lcm_val = lcm_section.get("context_threshold")
            if lcm_val is not None:
                return float(lcm_val), "config_yaml:lcm.context_threshold"
        compression = cfg.get("compression") or {}
        if not isinstance(compression, dict):
            return default, "default"
        if _config_bool_disabled(compression.get("enabled")):
            return default, "default"
        comp_val = compression.get("threshold")
        if comp_val is not None:
            return float(comp_val), "config_yaml:compression.threshold"
    except Exception:
        return default, "default"
    return default, "default"


def _hermes_auxiliary_compression_timeout_ms(default: int) -> int:
    """Read Hermes auxiliary.compression.timeout when no LCM override is present.

    Hermes uses seconds for the auxiliary compression timeout, while LCM stores
    the summary timeout in milliseconds. Aligning the default keeps LCM summary
    calls from timing out earlier than the host compression route unless
    ``LCM_SUMMARY_TIMEOUT_MS`` is explicitly configured.
    """
    value, _source = _hermes_auxiliary_compression_timeout_ms_with_source(default)
    return value


def _hermes_auxiliary_compression_timeout_ms_with_source(
    default: int,
    cfg: dict[str, Any] | None = None,
) -> tuple[int, str]:
    cfg = cfg if cfg is not None else _load_hermes_config_yaml()
    try:
        auxiliary = cfg.get("auxiliary") or {}
        if not isinstance(auxiliary, dict):
            return default, "default"
        compression = auxiliary.get("compression") or {}
        if not isinstance(compression, dict):
            return default, "default"
        value = compression.get("timeout")
        if value is None:
            return default, "default"
        return int(float(value) * 1000), "config_yaml:auxiliary.compression.timeout"
    except Exception:
        return default, "default"


def _hermes_codex_gpt55_autoraise_with_source(
    default: bool,
    cfg: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    cfg = cfg if cfg is not None else _load_hermes_config_yaml()
    try:
        compression = cfg.get("compression") or {}
        if not isinstance(compression, dict):
            return default, "default"
        value = compression.get("codex_gpt55_autoraise")
        if value is None:
            return default, "default"
        return (not _config_bool_disabled(value)), "config_yaml:compression.codex_gpt55_autoraise"
    except Exception:
        return default, "default"


@dataclass(frozen=True)
class _EnvFieldSpec:
    """One scalar ``LCM_*`` environment override: which config field it sets,
    its environment variable, and the Python type used to parse it."""

    name: str
    env_key: str
    py_type: type


# Single source of truth for the scalar LCM_* env overrides. ``from_env`` applies
# the non-source-tracked entries uniformly, and ``presets`` derives its
# preset-field lookups from the same list so the field/env/type mapping is not
# duplicated. Order mirrors the historical ``from_env`` order for readability.
ENV_FIELD_SPECS: tuple[_EnvFieldSpec, ...] = (
    _EnvFieldSpec("fresh_tail_count", "LCM_FRESH_TAIL_COUNT", int),
    _EnvFieldSpec("leaf_chunk_tokens", "LCM_LEAF_CHUNK_TOKENS", int),
    _EnvFieldSpec("context_threshold", "LCM_CONTEXT_THRESHOLD", float),
    _EnvFieldSpec("incremental_max_depth", "LCM_INCREMENTAL_MAX_DEPTH", int),
    _EnvFieldSpec("condensation_fanin", "LCM_CONDENSATION_FANIN", int),
    _EnvFieldSpec("dynamic_leaf_chunk_enabled", "LCM_DYNAMIC_LEAF_CHUNK_ENABLED", bool),
    _EnvFieldSpec("dynamic_leaf_chunk_max", "LCM_DYNAMIC_LEAF_CHUNK_MAX", int),
    _EnvFieldSpec("cache_friendly_condensation_enabled", "LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED", bool),
    _EnvFieldSpec("cache_friendly_min_debt_groups", "LCM_CACHE_FRIENDLY_MIN_DEBT_GROUPS", int),
    _EnvFieldSpec("deferred_maintenance_enabled", "LCM_DEFERRED_MAINTENANCE_ENABLED", bool),
    _EnvFieldSpec("deferred_maintenance_max_passes", "LCM_DEFERRED_MAINTENANCE_MAX_PASSES", int),
    _EnvFieldSpec("critical_budget_pressure_ratio", "LCM_CRITICAL_BUDGET_PRESSURE_RATIO", float),
    _EnvFieldSpec("l2_budget_ratio", "LCM_L2_BUDGET_RATIO", float),
    _EnvFieldSpec("l3_truncate_tokens", "LCM_L3_TRUNCATE_TOKENS", int),
    _EnvFieldSpec("max_assembly_tokens", "LCM_MAX_ASSEMBLY_TOKENS", int),
    _EnvFieldSpec("reserve_tokens_floor", "LCM_RESERVE_TOKENS_FLOOR", int),
    _EnvFieldSpec("custom_instructions", "LCM_CUSTOM_INSTRUCTIONS", str),
    _EnvFieldSpec("extraction_enabled", "LCM_EXTRACTION_ENABLED", bool),
    _EnvFieldSpec("extraction_model", "LCM_EXTRACTION_MODEL", str),
    _EnvFieldSpec("extraction_output_path", "LCM_EXTRACTION_OUTPUT_PATH", str),
    _EnvFieldSpec("sensitive_patterns_enabled", "LCM_SENSITIVE_PATTERNS_ENABLED", bool),
    _EnvFieldSpec("large_output_externalization_enabled", "LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED", bool),
    _EnvFieldSpec("large_output_externalization_threshold_chars", "LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS", int),
    _EnvFieldSpec("large_output_externalization_path", "LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH", str),
    _EnvFieldSpec("large_output_active_replay_stubbing_enabled", "LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED", bool),
    _EnvFieldSpec("large_output_active_replay_stub_threshold_tokens", "LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS", int),
    _EnvFieldSpec("large_output_transcript_gc_enabled", "LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED", bool),
    _EnvFieldSpec("summary_model", "LCM_SUMMARY_MODEL", str),
    _EnvFieldSpec("summary_circuit_breaker_failure_threshold", "LCM_SUMMARY_CIRCUIT_BREAKER_FAILURE_THRESHOLD", int),
    _EnvFieldSpec("summary_circuit_breaker_cooldown_seconds", "LCM_SUMMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS", int),
    _EnvFieldSpec("summary_spend_max_calls", "LCM_SUMMARY_SPEND_MAX_CALLS", int),
    _EnvFieldSpec("summary_spend_window_seconds", "LCM_SUMMARY_SPEND_WINDOW_SECONDS", float),
    _EnvFieldSpec("summary_spend_backoff_seconds", "LCM_SUMMARY_SPEND_BACKOFF_SECONDS", float),
    _EnvFieldSpec("expansion_model", "LCM_EXPANSION_MODEL", str),
    _EnvFieldSpec("expansion_context_tokens", "LCM_EXPANSION_CONTEXT_TOKENS", int),
    _EnvFieldSpec("summary_timeout_ms", "LCM_SUMMARY_TIMEOUT_MS", int),
    _EnvFieldSpec("expansion_timeout_ms", "LCM_EXPANSION_TIMEOUT_MS", int),
    _EnvFieldSpec("database_path", "LCM_DATABASE_PATH", str),
    _EnvFieldSpec("new_session_retain_depth", "LCM_NEW_SESSION_RETAIN_DEPTH", int),
    _EnvFieldSpec("doctor_clean_apply_enabled", "LCM_DOCTOR_CLEAN_APPLY_ENABLED", bool),
    _EnvFieldSpec("empty_lifecycle_gc_enabled", "LCM_EMPTY_LIFECYCLE_GC_ENABLED", bool),
    _EnvFieldSpec("empty_lifecycle_gc_threshold", "LCM_EMPTY_LIFECYCLE_GC_THRESHOLD", int),
)

def _coerce_yaml_value(field: str, raw: Any, py_type: type):
    try:
        if py_type is bool:
            if isinstance(raw, bool):
                return raw, None
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                if raw == 1:
                    return True, None
                if raw == 0:
                    return False, None
            if isinstance(raw, str):
                normalized = raw.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    return True, None
                if normalized in {"0", "false", "no", "off"}:
                    return False, None
            return None, f"invalid config_yaml:lcm.{field}={raw!r} ignored"
        if py_type is int:
            if isinstance(raw, bool) or not isinstance(raw, (int, str)):
                return None, f"invalid config_yaml:lcm.{field}={raw!r} ignored"
            return int(raw), None
        if py_type is float:
            if isinstance(raw, bool):
                return None, f"invalid config_yaml:lcm.{field}={raw!r} ignored"
            return float(raw), None
        if py_type is str:
            if not isinstance(raw, str):
                return None, f"invalid config_yaml:lcm.{field}={raw!r} ignored"
            return raw, None
    except (TypeError, ValueError):
        return None, f"invalid config_yaml:lcm.{field}={raw!r} ignored"
    return None, f"unsupported config_yaml:lcm.{field} type {py_type!r} ignored"


def _lcm_yaml_value_with_source(
    field: str,
    py_type: type,
    default,
    *,
    default_source: str = "default",
    cfg: dict[str, Any] | None = None,
) -> tuple[Any, str, str | None]:
    cfg = cfg if cfg is not None else _load_hermes_config_yaml()
    lcm_section = cfg.get("lcm") if isinstance(cfg, dict) else None
    if not isinstance(lcm_section, dict) or field not in lcm_section:
        return default, default_source, None
    value, warning = _coerce_yaml_value(field, lcm_section.get(field), py_type)
    if warning:
        return default, default_source, warning
    return value, f"config_yaml:lcm.{field}", None


def _parse_env_spec_with_source(
    key: str,
    py_type: type,
    default,
    *,
    default_source: str = "default",
) -> tuple[Any, str, str | None]:
    raw = os.environ.get(key)
    if raw is None:
        return default, default_source, None
    value, warning = _coerce_yaml_value(key, raw, py_type)
    if warning:
        return default, default_source, f"invalid env {key}={raw!r} ignored"
    return value, f"env:{key}", None

# Fields whose YAML/env reading needs a computed default; ``from_env`` handles
# these explicitly, so the uniform scalar loop skips them.
_SOURCE_TRACKED_ENV_FIELDS = frozenset({
    "context_threshold",
    "summary_timeout_ms",
})

# Fields exposed as runtime preset overrides (consumed by presets.py).
_PRESET_ENV_FIELDS = frozenset({
    "context_threshold",
    "fresh_tail_count",
    "leaf_chunk_tokens",
    "condensation_fanin",
    "incremental_max_depth",
})


@dataclass
class LCMConfig:
    """All tunables for the LCM engine."""

    # -- Fresh tail: recent messages never compacted ---
    fresh_tail_count: int = 32

    # -- Compaction thresholds ---
    # Max source tokens in a leaf chunk before summarization triggers
    leaf_chunk_tokens: int = 20_000
    # Fraction of context window that triggers compaction (0.0–1.0)
    context_threshold: float = 0.35
    # Mirror Hermes Agent's Codex gpt-5.5 route-specific threshold auto-raise
    # when LCM is inheriting the host compression threshold. Explicit LCM
    # threshold overrides remain authoritative.
    codex_gpt55_autoraise_enabled: bool = True
    # Max condensation depth (-1 = unlimited, 0 = leaf only)
    incremental_max_depth: int = 3
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
    # Disabled at 0.0. When set, only bypass cache-friendly/deferred polite
    # gates once prompt pressure reaches this fraction of the context window.
    critical_budget_pressure_ratio: float = 0.0

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

    # -- Session and message filtering ---
    # Sessions to exclude from LCM storage entirely.
    ignore_session_patterns: list[str] = field(default_factory=list)
    # Sessions that may read carried-over LCM state but never write new data.
    stateless_session_patterns: list[str] = field(default_factory=list)
    # Per-message regex patterns; matching messages are skipped before LCM storage.
    ignore_message_patterns: list[str] = field(default_factory=list)
    # Diagnostics: where each pattern list came from.
    ignore_session_patterns_source: str = "default"
    stateless_session_patterns_source: str = "default"
    ignore_message_patterns_source: str = "default"

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

    # -- Sensitive-pattern handling ---
    # Disabled by default. When enabled, named patterns redact matching secrets
    # before LCM storage, FTS indexing, summarization, or externalization.
    sensitive_patterns_enabled: bool = False
    # Named pattern catalog entries to apply when sensitive handling is enabled.
    sensitive_patterns: list[str] = field(
        default_factory=lambda: ["api_key", "bearer_token", "password_assignment", "private_key"]
    )
    # Diagnostics: where the sensitive pattern list came from.
    sensitive_patterns_source: str = "default"

    # -- Large tool-output externalization ---
    # When enabled, oversized tool results are written to plugin-managed storage
    # and replaced with compact references in pre-compaction serializer input.
    large_output_externalization_enabled: bool = False
    # Character threshold above which tool results are externalized.
    large_output_externalization_threshold_chars: int = 12_000
    # Explicit storage directory for externalized payloads (empty = auto under hermes home).
    large_output_externalization_path: str = ""
    # Replace eligible textual tool results with durable compact refs in
    # provider-visible replay. Current-turn ingest is intercepted immediately;
    # historical assembly separately respects the protected fresh tail. This
    # remains opt-in and requires large-output externalization.
    large_output_active_replay_stubbing_enabled: bool = False
    # Token-aware active-replay threshold. The character threshold above still
    # controls ordinary ingest externalization; this threshold controls when a
    # provider-visible textual tool result is replaced by its durable ref.
    large_output_active_replay_stub_threshold_tokens: int = 25_000
    # When enabled, already-externalized summarized tool-result transcript rows may
    # be rewritten to compact GC placeholders after successful leaf compaction.
    large_output_transcript_gc_enabled: bool = False

    # -- Models ---
    summary_model: str = ""       # empty = use Hermes auxiliary model
    # Optional fallback summary models tried after summary_model/task default.
    summary_fallback_models: list[str] = field(default_factory=list)
    # Consecutive failed summary calls before a route is skipped temporarily.
    summary_circuit_breaker_failure_threshold: int = 2
    # Seconds to skip an open summary route before allowing a retry.
    summary_circuit_breaker_cooldown_seconds: int = 300
    # Sliding-window cap for paid/auxiliary summarizer calls before falling
    # back to deterministic L3 truncation. 0 disables the spend guard.
    summary_spend_max_calls: int = 24
    # Window, in seconds, over which summary spend calls are counted.
    summary_spend_window_seconds: float = 600.0
    # Backoff, in seconds, after the spend window is exhausted.
    summary_spend_backoff_seconds: float = 1800.0
    expansion_model: str = ""     # empty = fall back to summary_model / Hermes auxiliary model
    # Serialized summary/raw/child-source/externalized context budget fed to lcm_expand_query's auxiliary LLM before it returns a bounded answer.
    expansion_context_tokens: int = 32_000

    # -- Timeouts ---
    summary_timeout_ms: int = 60_000
    expansion_timeout_ms: int = 120_000

    # -- Storage ---
    database_path: str = ""       # empty = HERMES_HOME/lcm.db; LCM_DATABASE_PATH may override

    # -- Session carry-over ---
    # Depth retained after /new (-1 = all, 0 = nothing, 2 = keep d2+)
    new_session_retain_depth: int = 2
    # Safety gate: destructive `/lcm doctor clean apply` workflow is disabled by default.
    doctor_clean_apply_enabled: bool = False

    # -- Lifecycle GC ---
    # Enables automatic pruning of lifecycle rows for sessions that never
    # ingested any messages or nodes (gateway restart orphans, ephemeral
    # cron ticks, etc.).  Runs at session-start when the lifecycle table
    # exceeds ``empty_lifecycle_gc_threshold`` rows.
    empty_lifecycle_gc_enabled: bool = True
    # Number of lifecycle rows at which the GC pass fires.  Default 200
    # so fresh installs skip the work until enough churn has occurred.
    empty_lifecycle_gc_threshold: int = 200
    # Age guard for automatic lifecycle GC. Startup GC must not delete
    # recently-bound empty rows because another live engine may not have
    # ingested its first message yet. Set to 0 only in trusted/test
    # environments that intentionally want immediate empty-row pruning.
    empty_lifecycle_gc_max_age_hours: float | None = 24.0

    # -- Diagnostics ---
    # Field-level provenance for values loaded through from_env(). Manual
    # LCMConfig(...) instances leave this empty and status treats them as manual/default.
    config_sources: dict[str, str] = field(default_factory=dict)
    config_source_warnings: list[str] = field(default_factory=list)
    ignored_config_yaml_lcm_keys: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "LCMConfig":
        """Build config from environment variables (LCM_ prefix)."""
        c = cls()
        hermes_config = _load_hermes_config_yaml()
        config_sources: dict[str, str] = {}
        config_source_warnings: list[str] = []

        def _record(field: str, source: str, *warnings: str | None) -> None:
            config_sources[field] = source
            config_source_warnings.extend(warning for warning in warnings if warning)

        c.ignored_config_yaml_lcm_keys = _ignored_lcm_config_yaml_keys(hermes_config)

        # Fields with host-config fallbacks stay explicit; the uniform scalar
        # loop below handles all ordinary env > YAML > default fields.
        context_default, context_source = _hermes_compression_threshold_with_source(
            c.context_threshold, hermes_config
        )
        c.context_threshold, source, warning = _parse_float_env_with_source(
            "LCM_CONTEXT_THRESHOLD",
            context_default,
            default_source=context_source,
        )
        _record("context_threshold", source, warning)
        c.codex_gpt55_autoraise_enabled, source = _hermes_codex_gpt55_autoraise_with_source(
            c.codex_gpt55_autoraise_enabled, hermes_config
        )
        _record("codex_gpt55_autoraise_enabled", source)
        summary_timeout_default, summary_timeout_source = _hermes_auxiliary_compression_timeout_ms_with_source(
            c.summary_timeout_ms, hermes_config
        )
        yaml_default, yaml_source, yaml_warning = _lcm_yaml_value_with_source(
            "summary_timeout_ms",
            int,
            summary_timeout_default,
            default_source=summary_timeout_source,
            cfg=hermes_config,
        )
        c.summary_timeout_ms, source, warning = _parse_int_env_with_source(
            "LCM_SUMMARY_TIMEOUT_MS",
            yaml_default,
            default_source=yaml_source,
        )
        _record("summary_timeout_ms", source, yaml_warning, warning)

        # Every other scalar LCM_* override is applied uniformly from the spec.
        # Precedence is env > lcm.<field> in config.yaml > dataclass default.
        for spec in ENV_FIELD_SPECS:
            if spec.name in _SOURCE_TRACKED_ENV_FIELDS:
                continue
            yaml_default, yaml_source, yaml_warning = _lcm_yaml_value_with_source(
                spec.name,
                spec.py_type,
                getattr(c, spec.name),
                cfg=hermes_config,
            )
            value, source, warning = _parse_env_spec_with_source(
                spec.env_key,
                spec.py_type,
                yaml_default,
                default_source=yaml_source,
            )
            setattr(c, spec.name, value)
            _record(spec.name, source, yaml_warning, warning)

        # Pattern-list overrides carry a source sidecar and stay explicit.
        raw_sensitive_patterns = os.environ.get("LCM_SENSITIVE_PATTERNS")
        if raw_sensitive_patterns is not None:
            c.sensitive_patterns = _parse_pattern_list(raw_sensitive_patterns)
            c.sensitive_patterns_source = "env"
        raw_summary_fallback_models = os.environ.get("LCM_SUMMARY_FALLBACK_MODELS")
        if raw_summary_fallback_models is not None:
            c.summary_fallback_models = _parse_pattern_list(raw_summary_fallback_models)

        raw_max_age = os.environ.get("LCM_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS")
        if raw_max_age is not None:
            try:
                c.empty_lifecycle_gc_max_age_hours = float(raw_max_age)
            except (TypeError, ValueError):
                pass

        raw_ignore = os.environ.get("LCM_IGNORE_SESSION_PATTERNS")
        if raw_ignore is not None:
            c.ignore_session_patterns = _parse_pattern_list(raw_ignore)
            c.ignore_session_patterns_source = "env"

        raw_stateless = os.environ.get("LCM_STATELESS_SESSION_PATTERNS")
        if raw_stateless is not None:
            c.stateless_session_patterns = _parse_pattern_list(raw_stateless)
            c.stateless_session_patterns_source = "env"

        raw_ignore_messages = os.environ.get("LCM_IGNORE_MESSAGE_PATTERNS")
        if raw_ignore_messages is not None:
            c.ignore_message_patterns = _parse_pattern_list(raw_ignore_messages)
            c.ignore_message_patterns_source = "env"

        c.config_sources = config_sources
        c.config_source_warnings = config_source_warnings
        return c
