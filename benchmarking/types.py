"""Serializable types for deterministic LCM benchmark replays."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class SummaryFailureMode(str, Enum):
    NONE = "none"
    LLM_TIMEOUT_THEN_TRUNCATE = "llm_timeout_then_truncate"
    LLM_REFUSAL_THEN_TRUNCATE = "llm_refusal_then_truncate"
    EMPTY_SUMMARY_THEN_TRUNCATE = "empty_summary_then_truncate"


def _summary_failure_mode(value: Any) -> SummaryFailureMode:
    if isinstance(value, SummaryFailureMode):
        return value
    return SummaryFailureMode(str(value or SummaryFailureMode.NONE.value))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(frozen=True)
class LCMPolicy:
    name: str
    context_length: int
    context_threshold: float
    fresh_tail_count: int
    leaf_chunk_tokens: int
    condensation_fanin: int = 4
    incremental_max_depth: int = 1
    dynamic_leaf_chunk_enabled: bool = False
    target_after_compaction: float | None = None
    min_turns_between_compactions: int = 0
    policy_version: str = "1"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LCMPolicy":
        return cls(
            name=str(data["name"]),
            context_length=int(data["context_length"]),
            context_threshold=float(data["context_threshold"]),
            fresh_tail_count=int(data["fresh_tail_count"]),
            leaf_chunk_tokens=int(data["leaf_chunk_tokens"]),
            condensation_fanin=int(data.get("condensation_fanin", 4)),
            incremental_max_depth=int(data.get("incremental_max_depth", 1)),
            dynamic_leaf_chunk_enabled=_as_bool(data.get("dynamic_leaf_chunk_enabled", False)),
            target_after_compaction=(
                None
                if data.get("target_after_compaction") is None
                else float(data["target_after_compaction"])
            ),
            min_turns_between_compactions=int(data.get("min_turns_between_compactions", 0)),
            policy_version=str(data.get("policy_version", "1")),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class Canary:
    id: str
    value: str
    expected_query: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Canary":
        canary_id = str(data["id"])
        return cls(
            id=canary_id,
            value=str(data["value"]),
            expected_query=str(data.get("expected_query") or canary_id),
        )


@dataclass(frozen=True)
class ReplayFixture:
    name: str
    messages: list[dict[str, Any]]
    canaries: list[Canary] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    benchmark_profile: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "messages": list(self.messages),
            "canaries": [canary.to_dict() for canary in self.canaries],
            "tags": list(self.tags),
        }
        if self.benchmark_profile:
            data["benchmark_profile"] = dict(self.benchmark_profile)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplayFixture":
        return cls(
            name=str(data["name"]),
            messages=[dict(message) for message in data["messages"]],
            canaries=[Canary.from_dict(item) for item in data.get("canaries", [])],
            tags=[str(tag) for tag in data.get("tags", [])],
            benchmark_profile=dict(data.get("benchmark_profile", {})),
        )


@dataclass
class ReplayMetrics:
    policy_name: str
    fixture_name: str
    prompt_tokens_before: int
    prompt_tokens_after: int
    threshold_tokens: int
    compression_count: int
    compaction_attempts: int
    post_compaction_headroom_tokens: int
    active_canaries_found: int
    retrieval_canaries_found: int
    total_canaries: int
    failures: list[str] = field(default_factory=list)
    policy_version: str = "1"
    fixture_tags: list[str] = field(default_factory=list)
    summary_level: int = 1
    summary_failure_mode: SummaryFailureMode = SummaryFailureMode.NONE
    post_compaction_headroom_ratio: float = 0.0
    fresh_tail_message_count: int = 0
    fresh_tail_tokens: int = 0
    fresh_tail_pressure_ratio: float = 0.0
    estimated_next_turn_tokens: int = 0
    repeated_compaction_risk: bool = False
    active_canary_recall: float = 0.0
    retrieval_canary_recall: float = 0.0
    database_path: str = ""
    hermes_home: str = ""
    active_message_count: int = 0
    store_messages: int = 0
    dag_nodes: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["summary_failure_mode"] = _summary_failure_mode(self.summary_failure_mode).value
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReplayMetrics":
        return cls(
            policy_name=str(data["policy_name"]),
            fixture_name=str(data["fixture_name"]),
            prompt_tokens_before=int(data["prompt_tokens_before"]),
            prompt_tokens_after=int(data["prompt_tokens_after"]),
            threshold_tokens=int(data["threshold_tokens"]),
            compression_count=int(data["compression_count"]),
            compaction_attempts=int(data["compaction_attempts"]),
            post_compaction_headroom_tokens=int(data["post_compaction_headroom_tokens"]),
            active_canaries_found=int(data["active_canaries_found"]),
            retrieval_canaries_found=int(data["retrieval_canaries_found"]),
            total_canaries=int(data["total_canaries"]),
            failures=[str(item) for item in data.get("failures", [])],
            policy_version=str(data.get("policy_version", "1")),
            fixture_tags=[str(item) for item in data.get("fixture_tags", [])],
            summary_level=int(data.get("summary_level", 1)),
            summary_failure_mode=_summary_failure_mode(data.get("summary_failure_mode", SummaryFailureMode.NONE)),
            post_compaction_headroom_ratio=float(data.get("post_compaction_headroom_ratio", 0.0)),
            fresh_tail_message_count=int(data.get("fresh_tail_message_count", 0)),
            fresh_tail_tokens=int(data.get("fresh_tail_tokens", 0)),
            fresh_tail_pressure_ratio=float(data.get("fresh_tail_pressure_ratio", 0.0)),
            estimated_next_turn_tokens=int(data.get("estimated_next_turn_tokens", 0)),
            repeated_compaction_risk=_as_bool(data.get("repeated_compaction_risk", False)),
            active_canary_recall=float(data.get("active_canary_recall", 0.0)),
            retrieval_canary_recall=float(data.get("retrieval_canary_recall", 0.0)),
            database_path=str(data.get("database_path", "")),
            hermes_home=str(data.get("hermes_home", "")),
            active_message_count=int(data.get("active_message_count", 0)),
            store_messages=int(data.get("store_messages", 0)),
            dag_nodes=int(data.get("dag_nodes", 0)),
            elapsed_ms=float(data.get("elapsed_ms", 0.0)),
        )
