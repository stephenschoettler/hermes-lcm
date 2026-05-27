"""Deterministic LCM replay runner for benchmark-driven preset tuning."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .metrics import canary_present, count_active_canaries, normalize_message_content
from .types import LCMPolicy, ReplayFixture, ReplayMetrics, SummaryFailureMode, _summary_failure_mode


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CANARY_LINE_RE = re.compile(r"\b(CANARY_[A-Z0-9_]+)\s*=\s*([A-Z0-9_:-]+)")


def _ensure_hermes_lcm_package() -> None:
    """Make this source checkout importable as `hermes_lcm` without plugin registration."""
    if "hermes_lcm" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "hermes_lcm",
        _REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(_REPO_ROOT)],
    )
    if spec is None:
        raise RuntimeError("could not create hermes_lcm package spec")
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(_REPO_ROOT)]
    module.__package__ = "hermes_lcm"
    sys.modules["hermes_lcm"] = module


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return safe or "replay"


def _policy_run_key(policy: LCMPolicy) -> str:
    return f"{_safe_name(policy.name)}__v{_safe_name(policy.policy_version)}"


def deterministic_summary(
    *,
    text: str,
    source_tokens: int,
    token_budget: int,
    depth: int,
    max_canaries: int = 1,
    **_: Any,
) -> tuple[str, int]:
    """Return a deterministic summary that preserves bounded canary facts."""
    preserved: list[str] = []
    seen: set[str] = set()
    for match in _CANARY_LINE_RE.finditer(text):
        canary_id, value = match.groups()
        if canary_id in seen:
            continue
        seen.add(canary_id)
        preserved.append(f"{canary_id} = {value}")
        if len(preserved) >= max_canaries:
            break
    lines = [
        "Deterministic benchmark summary.",
        f"source_tokens={source_tokens}",
        f"token_budget={token_budget}",
        f"depth={depth}",
    ]
    if preserved:
        lines.append("Preserved canaries:")
        lines.extend(f"- {line}" for line in preserved)
    return "\n".join(lines), 1


@contextmanager
def patched_deterministic_summarizer(max_canaries: int = 1) -> Iterator[None]:
    """Patch LCM summarization inside a fail-closed deterministic context."""
    _ensure_hermes_lcm_package()
    import hermes_lcm.engine as lcm_engine

    original = lcm_engine.summarize_with_escalation

    def _stub(**kwargs: Any) -> tuple[str, int]:
        return deterministic_summary(max_canaries=max_canaries, **kwargs)

    lcm_engine.summarize_with_escalation = _stub
    try:
        yield
    finally:
        lcm_engine.summarize_with_escalation = original


def _config_from_policy(policy: LCMPolicy, database_path: Path):
    _ensure_hermes_lcm_package()
    from hermes_lcm.config import LCMConfig

    return LCMConfig(
        fresh_tail_count=policy.fresh_tail_count,
        leaf_chunk_tokens=policy.leaf_chunk_tokens,
        context_threshold=policy.context_threshold,
        incremental_max_depth=policy.incremental_max_depth,
        condensation_fanin=policy.condensation_fanin,
        dynamic_leaf_chunk_enabled=policy.dynamic_leaf_chunk_enabled,
        database_path=str(database_path),
    )


def _new_engine(policy: LCMPolicy, run_dir: Path):
    _ensure_hermes_lcm_package()
    from hermes_lcm.engine import LCMEngine

    policy_key = _policy_run_key(policy)
    db_path = run_dir / f"{policy_key}.lcm.db"
    hermes_home = run_dir / "hermes-home"
    hermes_home.mkdir(parents=True, exist_ok=True)
    config = _config_from_policy(policy, db_path)
    engine = LCMEngine(config=config, hermes_home=str(hermes_home))
    session_id = f"bench-{policy_key}"
    conversation_id = f"bench-{policy_key}"
    engine.on_session_start(
        session_id,
        platform="benchmark",
        conversation_id=conversation_id,
        model="benchmark-model",
        provider="benchmark",
        context_length=policy.context_length,
    )
    engine.update_model(
        model="benchmark-model",
        context_length=policy.context_length,
        provider="benchmark",
    )
    return engine


def _grep_canary(engine: Any, canary: Any) -> bool:
    query = canary.expected_query or canary.id
    try:
        grep_payload = json.loads(engine.handle_tool_call("lcm_grep", {"query": query, "limit": 5}))
    except Exception:
        return False
    for hit in grep_payload.get("results", []):
        hit_text = "\n".join(
            str(hit.get(key, "")) for key in ("snippet", "content", "summary")
        )
        if canary_present(hit_text, canary):
            return True
        store_id = hit.get("store_id")
        if store_id is None:
            continue
        try:
            expanded = json.loads(
                engine.handle_tool_call(
                    "lcm_expand",
                    {"store_id": int(store_id), "max_tokens": 100_000},
                )
            )
        except Exception:
            continue
        expanded_text = normalize_message_content(expanded.get("content"))
        if canary_present(expanded_text, canary):
            return True
    return False


def _count_retrieval_canaries(engine: Any, fixture: ReplayFixture) -> int:
    return sum(1 for canary in fixture.canaries if _grep_canary(engine, canary))


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def _summary_profile(fixture: ReplayFixture) -> tuple[int, SummaryFailureMode]:
    profile = fixture.benchmark_profile
    level = int(profile.get("summary_level", 1)) if profile else 1
    mode = _summary_failure_mode(profile.get("summary_failure_mode")) if profile else SummaryFailureMode.NONE
    return level, mode


def run_replay(
    fixture: ReplayFixture,
    policy: LCMPolicy,
    *,
    output_dir: str | Path,
    max_summary_canaries: int = 1,
) -> ReplayMetrics:
    """Replay one fixture against one LCM policy with deterministic summarization."""
    _ensure_hermes_lcm_package()
    from hermes_lcm.tokens import count_messages_tokens

    root_dir = Path(output_dir)
    run_dir = root_dir / f"{_safe_name(fixture.name)}__{_policy_run_key(policy)}"
    if run_dir.exists() and not run_dir.is_dir():
        raise ValueError(f"Refusing to reuse benchmark run path that is not a directory: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"Refusing to reuse non-empty benchmark run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    engine = _new_engine(policy, run_dir)
    start = time.perf_counter()
    failures: list[str] = []
    messages = [dict(message) for message in fixture.messages]
    summary_level, summary_failure_mode = _summary_profile(fixture)
    compressed_messages = messages
    compaction_attempts = 0
    before = count_messages_tokens(messages)
    try:
        should_compress = engine.should_compress_preflight(messages)
        if should_compress:
            compaction_attempts = 1
            with patched_deterministic_summarizer(max_canaries=max_summary_canaries):
                compressed_messages = engine.compress(messages, current_tokens=before)
        after = count_messages_tokens(compressed_messages)
        active_canaries = count_active_canaries(compressed_messages, fixture.canaries)
        retrieval_canaries = _count_retrieval_canaries(engine, fixture)
        tail_count = min(max(policy.fresh_tail_count, 0), len(compressed_messages))
        fresh_tail = compressed_messages[-tail_count:] if tail_count else []
        fresh_tail_tokens = count_messages_tokens(fresh_tail)
        estimated_next_turn_tokens = count_messages_tokens(messages[-2:]) if messages else 0
        headroom_tokens = engine.threshold_tokens - after
        status = engine.get_status()
        metrics = ReplayMetrics(
            policy_name=policy.name,
            policy_version=policy.policy_version,
            fixture_name=fixture.name,
            fixture_tags=list(fixture.tags),
            summary_level=summary_level,
            summary_failure_mode=summary_failure_mode,
            prompt_tokens_before=before,
            prompt_tokens_after=after,
            threshold_tokens=engine.threshold_tokens,
            compression_count=engine.compression_count,
            compaction_attempts=compaction_attempts,
            post_compaction_headroom_tokens=headroom_tokens,
            post_compaction_headroom_ratio=_ratio(headroom_tokens, engine.threshold_tokens),
            fresh_tail_message_count=tail_count,
            fresh_tail_tokens=fresh_tail_tokens,
            fresh_tail_pressure_ratio=_ratio(fresh_tail_tokens, engine.threshold_tokens),
            estimated_next_turn_tokens=estimated_next_turn_tokens,
            repeated_compaction_risk=bool(compaction_attempts and headroom_tokens <= estimated_next_turn_tokens),
            active_canaries_found=active_canaries,
            retrieval_canaries_found=retrieval_canaries,
            total_canaries=len(fixture.canaries),
            active_canary_recall=_ratio(active_canaries, len(fixture.canaries)) if fixture.canaries else 1.0,
            retrieval_canary_recall=_ratio(retrieval_canaries, len(fixture.canaries)) if fixture.canaries else 1.0,
            failures=failures,
            database_path=str(engine._store.db_path),
            hermes_home=str(engine._hermes_home),
            active_message_count=len(compressed_messages),
            store_messages=int(status.get("store_messages", 0) or 0),
            dag_nodes=int(status.get("dag_nodes", 0) or 0),
            elapsed_ms=round((time.perf_counter() - start) * 1000, 3),
        )
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        after = count_messages_tokens(compressed_messages)
        tail_count = min(max(policy.fresh_tail_count, 0), len(compressed_messages))
        fresh_tail = compressed_messages[-tail_count:] if tail_count else []
        fresh_tail_tokens = count_messages_tokens(fresh_tail)
        estimated_next_turn_tokens = count_messages_tokens(messages[-2:]) if messages else 0
        headroom_tokens = engine.threshold_tokens - after
        metrics = ReplayMetrics(
            policy_name=policy.name,
            policy_version=policy.policy_version,
            fixture_name=fixture.name,
            fixture_tags=list(fixture.tags),
            summary_level=summary_level,
            summary_failure_mode=summary_failure_mode,
            prompt_tokens_before=before,
            prompt_tokens_after=after,
            threshold_tokens=engine.threshold_tokens,
            compression_count=engine.compression_count,
            compaction_attempts=compaction_attempts,
            post_compaction_headroom_tokens=headroom_tokens,
            post_compaction_headroom_ratio=_ratio(headroom_tokens, engine.threshold_tokens),
            fresh_tail_message_count=tail_count,
            fresh_tail_tokens=fresh_tail_tokens,
            fresh_tail_pressure_ratio=_ratio(fresh_tail_tokens, engine.threshold_tokens),
            estimated_next_turn_tokens=estimated_next_turn_tokens,
            repeated_compaction_risk=bool(compaction_attempts and headroom_tokens <= estimated_next_turn_tokens),
            active_canaries_found=0,
            retrieval_canaries_found=0,
            total_canaries=len(fixture.canaries),
            active_canary_recall=0.0 if fixture.canaries else 1.0,
            retrieval_canary_recall=0.0 if fixture.canaries else 1.0,
            failures=failures,
            database_path=str(engine._store.db_path),
            hermes_home=str(engine._hermes_home),
            active_message_count=len(compressed_messages),
            elapsed_ms=round((time.perf_counter() - start) * 1000, 3),
        )
    finally:
        engine.shutdown()

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def run_replays(
    fixtures: list[ReplayFixture],
    policies: list[LCMPolicy],
    *,
    output_dir: str | Path,
) -> list[ReplayMetrics]:
    metrics: list[ReplayMetrics] = []
    root = Path(output_dir)
    for fixture in fixtures:
        for policy in policies:
            metrics.append(run_replay(fixture, policy, output_dir=root))
    return metrics
