"""Deterministic hermes-lcm stress release checks."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
STRESS_CHECK_VERSION = "1"
DEFAULT_SCENARIOS = (
    "multi_cycle_canary_recall",
    "redaction_and_externalization_boundaries",
    "cross_session_scope_and_pagination",
    "query_fuzz_no_crash",
    "concurrent_read_write_smoke",
)


@dataclass(frozen=True)
class StressTier:
    name: str
    multi_turns: int
    multi_compress_every: int
    multi_sample_indexes: tuple[int, ...]
    scope_turns_per_session: int
    fuzz_repetitions: int
    concurrent_turns: int
    reader_threads: int


TIERS = {
    "smoke": StressTier(
        name="smoke",
        multi_turns=36,
        multi_compress_every=12,
        multi_sample_indexes=(0, 1, 17, 35),
        scope_turns_per_session=8,
        fuzz_repetitions=2,
        concurrent_turns=24,
        reader_threads=2,
    ),
    "release": StressTier(
        name="release",
        multi_turns=180,
        multi_compress_every=12,
        multi_sample_indexes=(0, 1, 2, 17, 44, 88, 123, 179),
        scope_turns_per_session=24,
        fuzz_repetitions=8,
        concurrent_turns=90,
        reader_threads=4,
    ),
}


class StressRun:
    def __init__(self, *, output_dir: Path, tier: StressTier, plugin_dir: Path | None = None) -> None:
        self.output_dir = output_dir.resolve()
        self.tier = tier
        self.plugin_dir = (plugin_dir or _REPO_ROOT).resolve()
        self.sandbox_dir = self.output_dir / "sandbox"
        self.results_dir = self.output_dir / "results"
        self.logs_dir = self.output_dir / "logs"
        self.failures: list[dict[str, Any]] = []
        self.results: dict[str, Any] = {
            "stress_check_version": STRESS_CHECK_VERSION,
            "tier": tier.name,
            "run_dir": str(self.output_dir),
            "plugin_dir": str(self.plugin_dir),
            "started_at_utc": _utc_timestamp(),
            "environment": _environment_metadata(),
            "source": _source_metadata(self.plugin_dir),
            "cases": {},
            "failures": self.failures,
        }
        self._original_summarize: Any = None
        self._original_synthesize: Any = None
        self._engine_mod: Any = None
        self._tools_mod: Any = None

    def prepare(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        (self.sandbox_dir / "home").mkdir(parents=True, exist_ok=True)
        (self.sandbox_dir / "hermes-home").mkdir(parents=True, exist_ok=True)

    def record(self, case: str, key: str, value: Any) -> None:
        self.results.setdefault("cases", {}).setdefault(case, {})[key] = value

    def fail(self, case: str, bug_id: str, title: str, details: dict[str, Any]) -> None:
        item = {"case": case, "bug_id": bug_id, "title": title, "details": details}
        self.failures.append(item)
        self.results.setdefault("cases", {}).setdefault(case, {}).setdefault("failures", []).append(item)

    @contextlib.contextmanager
    def patched_summarizers(self) -> Iterator[None]:
        _ensure_hermes_lcm_package(self.plugin_dir, reload_submodules=True)
        import hermes_lcm.engine as engine_mod
        import hermes_lcm.tools as tools_mod

        self._engine_mod = engine_mod
        self._tools_mod = tools_mod
        self._original_summarize = engine_mod.summarize_with_escalation
        self._original_synthesize = tools_mod._synthesize_expansion_answer
        engine_mod.summarize_with_escalation = deterministic_summary
        tools_mod._synthesize_expansion_answer = deterministic_expand_answer
        try:
            yield
        finally:
            engine_mod.summarize_with_escalation = self._original_summarize
            tools_mod._synthesize_expansion_answer = self._original_synthesize

    def make_engine(self, case: str, **overrides: Any):
        _ensure_hermes_lcm_package(self.plugin_dir)
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        hermes_home = self.sandbox_dir / case / "home"
        db_path = self.sandbox_dir / case / "lcm.db"
        hermes_home.mkdir(parents=True, exist_ok=True)
        cfg = LCMConfig(
            database_path=str(db_path),
            fresh_tail_count=overrides.pop("fresh_tail_count", 8),
            leaf_chunk_tokens=overrides.pop("leaf_chunk_tokens", 240),
            context_threshold=overrides.pop("context_threshold", 0.50),
            incremental_max_depth=overrides.pop("incremental_max_depth", 3),
            condensation_fanin=overrides.pop("condensation_fanin", 3),
            dynamic_leaf_chunk_enabled=overrides.pop("dynamic_leaf_chunk_enabled", True),
            dynamic_leaf_chunk_max=overrides.pop("dynamic_leaf_chunk_max", 600),
            max_assembly_tokens=overrides.pop("max_assembly_tokens", 0),
            reserve_tokens_floor=overrides.pop("reserve_tokens_floor", 0),
            **overrides,
        )
        engine = LCMEngine(config=cfg, hermes_home=str(hermes_home))
        engine.on_session_start(
            f"stress-{case}",
            platform="cli",
            conversation_id=f"conv-{case}",
            hermes_home=str(hermes_home),
        )
        engine.update_model("stress-model", 4_000, provider="benchmark")
        return engine

    def call_tool(self, engine: Any, name: str, args: dict[str, Any], messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        raw = engine.handle_tool_call(name, args, messages=messages or [])
        try:
            loaded = json.loads(raw)
        except Exception:
            return {"_raw": raw, "_json_error": traceback.format_exc()}
        return loaded if isinstance(loaded, dict) else {"_value": loaded}

    def run_case(self, case_name: str, fn: Callable[["StressRun"], None]) -> None:
        start = time.perf_counter()
        try:
            fn(self)
            self.record(case_name, "ok", not bool(self.results.get("cases", {}).get(case_name, {}).get("failures")))
        except Exception as exc:
            self.fail(
                case_name,
                f"{case_name}.unhandled_exception",
                f"Unhandled exception in stress case {case_name}",
                {"exception": repr(exc), "traceback": traceback.format_exc()},
            )
            self.record(case_name, "ok", False)
        finally:
            self.record(case_name, "elapsed_s", round(time.perf_counter() - start, 4))

    def finalize(self) -> dict[str, Any]:
        self.results["finished_at_utc"] = _utc_timestamp()
        self.results["failure_count"] = len(self.failures)
        results_path = self.results_dir / "stress-results.json"
        summary_path = self.output_dir / "stress-summary.md"
        self.results["results_path"] = str(results_path)
        self.results["summary_path"] = str(summary_path)
        write_stress_summary(summary_path, self.results)
        self.results["artifact_hashes"] = {
            "results/stress-results.json_without_artifact_hashes": _canonical_results_sha256(self.results),
            str(summary_path.relative_to(self.output_dir)): _file_sha256(summary_path),
        }
        results_path.write_text(json.dumps(self.results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.results


def run_stress_check(
    *,
    output_dir: str | Path,
    tier: str = "release",
    scenarios: Iterable[str] | None = None,
    plugin_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run deterministic stress scenarios and write release-check artifacts."""
    selected_tier = TIERS[tier]
    selected = tuple(DEFAULT_SCENARIOS if scenarios is None else scenarios)
    unknown = [name for name in selected if name not in _SCENARIO_FUNCTIONS]
    if unknown:
        raise ValueError(f"unknown stress scenario(s): {', '.join(unknown)}")
    runner = StressRun(
        output_dir=Path(output_dir),
        tier=selected_tier,
        plugin_dir=Path(plugin_dir) if plugin_dir is not None else None,
    )
    runner.results["scenario_names"] = list(selected)
    with _sandboxed_environment(runner.sandbox_dir):
        runner.prepare()
        with runner.patched_summarizers():
            for name in selected:
                runner.run_case(name, _SCENARIO_FUNCTIONS[name])
        return runner.finalize()


def write_stress_summary(path: str | Path, results: dict[str, Any]) -> None:
    output_path = Path(path)
    lines = [
        "# hermes-lcm stress release-check summary",
        "",
        f"stress_check_version: {results.get('stress_check_version')}",
        f"tier: {results.get('tier')}",
        f"failure_count: {results.get('failure_count')}",
        f"source_commit: {results.get('source', {}).get('head')}",
        f"source_dirty: {results.get('source', {}).get('dirty')}",
        f"run_dir: {results.get('run_dir')}",
        "",
        "## Cases",
        "",
    ]
    for name, payload in sorted(results.get("cases", {}).items()):
        ok = payload.get("ok")
        elapsed = payload.get("elapsed_s")
        lines.append(f"- {name}: ok={ok} elapsed_s={elapsed}")
    failures = results.get("failures") or []
    lines.extend(["", "## Failures", ""])
    if failures:
        for failure in failures:
            lines.append(f"- {failure.get('case')}: {failure.get('bug_id')} - {failure.get('title')}")
    else:
        lines.append("- none")
    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- results: {results.get('results_path')}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def deterministic_summary(*, text: str, source_tokens: int, token_budget: int, depth: int, **_: Any) -> tuple[str, int]:
    canaries: list[str] = []
    for match in re.finditer(r"(CANARY_[A-Z0-9_]+)\s*=\s*([A-Z0-9_:-]+)", text):
        pair = f"{match.group(1)}={match.group(2)}"
        if pair not in canaries:
            canaries.append(pair)
    tool_ids: list[str] = []
    for match in re.finditer(r"tool_call_id['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9_.:-]+)", text):
        if match.group(1) not in tool_ids:
            tool_ids.append(match.group(1))
    lines = [
        f"Deterministic stress summary depth={depth} source_tokens={source_tokens} budget={token_budget}",
        "CANARIES: " + (", ".join(canaries[:80]) if canaries else "none"),
        "TOOL_IDS: " + (", ".join(tool_ids[:80]) if tool_ids else "none"),
    ]
    return "\n".join(lines), 1


def deterministic_expand_answer(*, prompt: str, context_blocks: list[dict[str, Any]], model: str, max_tokens: int, timeout: float) -> str:
    del prompt, model, max_tokens, timeout
    serialized = json.dumps(context_blocks, ensure_ascii=False)
    canaries = sorted(set(re.findall(r"CANARY_[A-Z0-9_]+\s*=\s*[A-Z0-9_:-]+", serialized)))
    return "Deterministic expansion answer. " + ("; ".join(canaries[:20]) if canaries else "No canaries found.")


def _clear_hermes_lcm_submodules(pkg: str = "hermes_lcm") -> None:
    package = sys.modules.get(pkg)
    prefix = f"{pkg}."
    for name in list(sys.modules):
        if not name.startswith(prefix):
            continue
        sys.modules.pop(name, None)
        if package is not None:
            child_name = name[len(prefix):].split(".", 1)[0]
            if hasattr(package, child_name):
                delattr(package, child_name)


def _ensure_hermes_lcm_package(plugin_dir: Path, *, reload_submodules: bool = False) -> None:
    pkg = "hermes_lcm"
    if pkg in sys.modules:
        module = sys.modules[pkg]
        module_path = Path(getattr(module, "__path__", [plugin_dir])[0]).resolve()
        if module_path == plugin_dir.resolve():
            if reload_submodules:
                _clear_hermes_lcm_submodules(pkg)
            return
        for name in list(sys.modules):
            if name == pkg or name.startswith(f"{pkg}."):
                sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        pkg,
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    if spec is None:
        raise RuntimeError(f"cannot create hermes_lcm package spec for {plugin_dir}")
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(plugin_dir)]
    mod.__package__ = pkg
    sys.modules[pkg] = mod
    # Do not execute __init__.py; it expects a Hermes plugin context.


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


_ENV_KEYS = (
    "HOME",
    "HERMES_HOME",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
)


@contextlib.contextmanager
def _sandboxed_environment(sandbox_dir: Path) -> Iterator[None]:
    original = {key: os.environ.get(key) for key in _ENV_KEYS}
    os.environ["HOME"] = str(sandbox_dir / "home")
    os.environ["HERMES_HOME"] = str(sandbox_dir / "hermes-home")
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["OPENROUTER_API_KEY"] = ""
    os.environ["ANTHROPIC_API_KEY"] = ""
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _source_metadata(repo_dir: Path) -> dict[str, Any]:
    return {
        "repo_dir": str(repo_dir),
        "head": _git(repo_dir, "rev-parse", "HEAD"),
        "head_short": _git(repo_dir, "rev-parse", "--short=12", "HEAD"),
        "branch": _git(repo_dir, "branch", "--show-current"),
        "dirty": bool(_git(repo_dir, "status", "--porcelain")),
        "status_short": _git(repo_dir, "status", "--short", "--branch"),
    }


def _git(cwd: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_results_sha256(results: dict[str, Any]) -> str:
    payload = dict(results)
    payload.pop("artifact_hashes", None)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return hashlib.sha256(data).hexdigest()


def _db_counts(db_path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    con = sqlite3.connect(db_path)
    try:
        for table in ["messages", "summary_nodes", "messages_fts", "nodes_fts", "lcm_lifecycle_state"]:
            try:
                out[table] = int(con.execute(f"select count(*) from {table}").fetchone()[0])
            except sqlite3.Error:
                out[table] = -1
    finally:
        con.close()
    return out


def _db_text_dump(db_path: Path) -> str:
    con = sqlite3.connect(db_path)
    parts: list[str] = []
    try:
        for table in ["messages", "summary_nodes"]:
            try:
                for row in con.execute(f"select * from {table}"):
                    parts.append(repr(row))
            except sqlite3.Error:
                pass
    finally:
        con.close()
    return "\n".join(parts)


def _no_orphan_tool_results(messages: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    calls: set[str] = set()
    result_ids: list[str] = []
    for msg in messages:
        if msg.get("role") == "assistant":
            for tool_call in msg.get("tool_calls") or []:
                if isinstance(tool_call, dict) and (tool_call.get("id") or tool_call.get("tool_call_id")):
                    calls.add(str(tool_call.get("id") or tool_call.get("tool_call_id")))
        if msg.get("role") == "tool":
            tool_call_id = str(msg.get("tool_call_id") or "")
            if tool_call_id:
                result_ids.append(tool_call_id)
    orphans = [tool_call_id for tool_call_id in result_ids if tool_call_id not in calls]
    return not orphans, orphans


def _case_multi_cycle_canary_recall(run: StressRun) -> None:
    _ensure_hermes_lcm_package(run.plugin_dir)
    from hermes_lcm.tokens import count_messages_tokens

    case = "multi_cycle_canary_recall"
    engine = run.make_engine(case, fresh_tail_count=10, leaf_chunk_tokens=220, condensation_fanin=2)
    messages: list[dict[str, Any]] = [{"role": "system", "content": "System anchor for LCM stress."}]
    expected: dict[str, str] = {}
    compressed_lengths: list[int] = []
    try:
        for i in range(run.tier.multi_turns):
            cid = f"CANARY_LONG_{i:04d}"
            val = f"VALUE_LONG_{i:04d}"
            expected[cid] = val
            messages.append({"role": "user", "content": f"turn={i} {cid} = {val} details alpha beta gamma " + ("x" * 80)})
            if i % 7 == 0:
                tcid = f"call-{i}"
                messages.append({
                    "role": "assistant",
                    "content": f"I will inspect {cid}.",
                    "tool_calls": [{"id": tcid, "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"canary": cid})}}],
                })
                messages.append({"role": "tool", "tool_call_id": tcid, "name": "lookup", "content": json.dumps({"result": val, "canary": cid})})
            else:
                messages.append({"role": "assistant", "content": f"ack {cid} {val}"})
            if i % run.tier.multi_compress_every == run.tier.multi_compress_every - 1:
                messages = engine.compress(messages, current_tokens=max(4_100, count_messages_tokens(messages)))
                compressed_lengths.append(len(messages))
                ok, orphans = _no_orphan_tool_results(messages)
                if not ok:
                    run.fail(case, "active_context_orphan_tool_result", "Compressed active context contains tool results without matching assistant tool_calls", {"turn": i, "orphans": orphans[:20], "active_roles": [m.get("role") for m in messages]})
        messages = engine.compress(messages, current_tokens=max(4_100, count_messages_tokens(messages)))
        engine.on_session_end(engine.current_session_id, messages)

        missed: list[dict[str, Any]] = []
        expanded_missed: list[dict[str, Any]] = []
        for index in run.tier.multi_sample_indexes:
            cid = f"CANARY_LONG_{index:04d}"
            grep = run.call_tool(engine, "lcm_grep", {"query": cid, "limit": 5, "sort": "relevance"})
            hay = json.dumps(grep, ensure_ascii=False)
            if cid not in hay or expected[cid] not in hay:
                missed.append({"canary": cid, "grep": grep})
                continue
            store_ids: list[int] = []
            for result in grep.get("results", []) + grep.get("matches", []) + grep.get("data", []):
                if isinstance(result, dict) and result.get("store_id"):
                    store_ids.append(int(result["store_id"]))
            if not store_ids:
                continue
            found_expanded = False
            expanded_samples: list[dict[str, Any]] = []
            for store_id in store_ids[:5]:
                expanded = run.call_tool(engine, "lcm_expand", {"store_id": store_id, "max_tokens": 500})
                expanded_samples.append({"store_id": store_id, "expand": expanded})
                ehay = json.dumps(expanded, ensure_ascii=False)
                if cid in ehay and expected[cid] in ehay:
                    found_expanded = True
                    break
            if not found_expanded:
                expanded_missed.append({"canary": cid, "store_ids": store_ids[:5], "expand_samples": expanded_samples})
        if missed:
            run.fail(case, "grep_canary_recall_miss", "lcm_grep failed to recover planted canaries from compacted session", {"missed": missed})
        if expanded_missed:
            run.fail(case, "expand_canary_recall_miss", "lcm_expand failed to recover raw planted canary content from grep result", {"missed": expanded_missed})

        status = run.call_tool(engine, "lcm_status", {})
        doctor = run.call_tool(engine, "lcm_doctor", {})
        run.record(case, "compressed_lengths", compressed_lengths[-10:])
        run.record(case, "status", status)
        run.record(case, "doctor", doctor)
        run.record(case, "db_counts", _db_counts(Path(engine._store.db_path)))
        if "error" in doctor:
            run.fail(case, "doctor_error_after_stress", "lcm_doctor returned an error after normal stress compaction", {"doctor": doctor})
    finally:
        engine.shutdown()


def _case_redaction_and_externalization_boundaries(run: StressRun) -> None:
    case = "redaction_and_externalization_boundaries"
    secret_values = [
        "sk-tes...cdef",
        "Bearer abcdef1234567890SECRETXYZ",
        "correct horse battery staple",
        "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCSTRESSKEY\n-----END PRIVATE KEY-----",
    ]
    large_blob = base64.b64encode(("LCM-LARGE-PAYLOAD-" * 800).encode()).decode()
    data_url = "data:image/png;base64," + large_blob
    engine = run.make_engine(
        case,
        sensitive_patterns_enabled=True,
        sensitive_patterns=["api_key", "bearer_token", "password_assignment", "private_key"],
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=500,
        large_output_transcript_gc_enabled=True,
        fresh_tail_count=4,
        leaf_chunk_tokens=100,
    )
    try:
        messages = [
            {"role": "system", "content": "System anchor."},
            {"role": "user", "content": f"api_key = {secret_values[0]} and authorization: {secret_values[1]}"},
            {"role": "assistant", "content": "running command", "tool_calls": [{"id": "secret-call", "type": "function", "function": {"name": "login", "arguments": json.dumps({"password": secret_values[2], "client_secret": secret_values[0]})}}]},
            {"role": "tool", "tool_call_id": "secret-call", "name": "login", "content": json.dumps({"private_key": secret_values[3], "blob": data_url, "token": secret_values[1]})},
            {"role": "user", "content": "CANARY_SECRET_0001 = VALUE_SECRET_0001 " + ("padding " * 100)},
            {"role": "assistant", "content": "ack CANARY_SECRET_0001 VALUE_SECRET_0001"},
        ]
        compressed = engine.compress(messages, current_tokens=4_500)
        engine.on_session_end(engine.current_session_id, compressed)

        db_path = Path(engine._store.db_path)
        dump = _db_text_dump(db_path)
        try:
            db_file_bytes = db_path.read_bytes()
        except Exception:
            db_file_bytes = b""
        leaked: list[dict[str, str]] = []
        for secret in secret_values + [large_blob[:2000], data_url[:2000]]:
            if secret and secret in dump:
                leaked.append({"where": "sqlite_rows", "secret_prefix": secret[:80]})
            if secret and secret.encode() in db_file_bytes:
                leaked.append({"where": "sqlite_file_bytes", "secret_prefix": secret[:80]})
        ext_dir = Path(engine._hermes_home) / "lcm-large-outputs"
        ext_files = sorted(str(p) for p in ext_dir.glob("*.json")) if ext_dir.exists() else []
        ext_text = "\n".join(Path(p).read_text(errors="ignore") for p in ext_files)
        for secret in secret_values:
            if secret in ext_text:
                leaked.append({"where": "externalized_payload_file", "secret_prefix": secret[:80]})
        if leaked:
            run.fail(case, "sensitive_or_large_payload_leak", "Sensitive or oversized payload material was persisted raw across storage boundaries", {"leaked": leaked, "externalized_files": ext_files[:5]})
        grep_secret = run.call_tool(engine, "lcm_grep", {"query": secret_values[0], "limit": 10})
        grep_secret_results_text = json.dumps(grep_secret.get("results", []), ensure_ascii=False)
        if secret_values[0] in grep_secret_results_text:
            run.fail(case, "grep_returns_raw_secret", "lcm_grep returned a raw secret after sensitive-pattern redaction was enabled", {"grep": grep_secret})
        grep_canary = run.call_tool(engine, "lcm_grep", {"query": "CANARY_SECRET_0001", "limit": 5})
        if "CANARY_SECRET_0001" not in json.dumps(grep_canary, ensure_ascii=False):
            run.fail(case, "redaction_broke_nonsecret_recall", "Sensitive redaction/externalization broke ordinary canary recall", {"grep": grep_canary})
        run.record(case, "externalized_files", ext_files)
        run.record(case, "db_counts", _db_counts(db_path))
        run.record(case, "grep_secret", grep_secret)
        run.record(case, "grep_canary", grep_canary)
    finally:
        engine.shutdown()


def _case_cross_session_scope_and_pagination(run: StressRun) -> None:
    case = "cross_session_scope_and_pagination"
    engine = run.make_engine(case, fresh_tail_count=3, leaf_chunk_tokens=80)
    try:
        sessions = ["scope-a", "scope-b"]
        for sid in sessions:
            engine.on_session_start(sid, platform="cli", conversation_id="conv-scope", hermes_home=str(Path(engine._hermes_home)))
            messages = [{"role": "system", "content": "scope system"}]
            prefix = sid.upper().replace("-", "_")
            for i in range(run.tier.scope_turns_per_session):
                messages.append({"role": "user", "content": f"{sid} CANARY_{prefix}_{i:03d} = VALUE_{prefix}_{i:03d} " + ("scope " * 40)})
                messages.append({"role": "assistant", "content": f"ack {sid} {i}"})
            compressed = engine.compress(messages, current_tokens=4_500)
            engine.on_session_end(sid, compressed)
        engine.on_session_start("scope-b", platform="cli", conversation_id="conv-scope", hermes_home=str(Path(engine._hermes_home)))

        current_a = run.call_tool(engine, "lcm_grep", {"query": "CANARY_SCOPE_A_000", "limit": 5})
        all_a = run.call_tool(engine, "lcm_grep", {"query": "CANARY_SCOPE_A_000", "limit": 5, "session_scope": "all"})
        explicit_a = run.call_tool(engine, "lcm_grep", {"query": "CANARY_SCOPE_A_000", "limit": 5, "session_scope": "session", "session_id": "scope-a"})
        load_a_1 = run.call_tool(engine, "lcm_load_session", {"session_id": "scope-a", "limit": 7, "max_content_chars": 80})
        cursor = load_a_1.get("next_cursor") or 0
        load_a_2 = run.call_tool(engine, "lcm_load_session", {"session_id": "scope-a", "limit": 7, "after_store_id": cursor, "max_content_chars": 80})

        if "CANARY_SCOPE_A_000" in json.dumps(current_a.get("results", []), ensure_ascii=False):
            run.fail(case, "current_scope_cross_session_leak", "lcm_grep current scope returned another session's raw content", {"current_result": current_a})
        if "CANARY_SCOPE_A_000" not in json.dumps(all_a.get("results", []), ensure_ascii=False):
            run.fail(case, "all_scope_missing_cross_session_hit", "lcm_grep session_scope=all failed to find another session's raw content", {"all_result": all_a})
        if "CANARY_SCOPE_A_000" not in json.dumps(explicit_a.get("results", []), ensure_ascii=False):
            run.fail(case, "explicit_session_scope_missing_hit", "lcm_grep session_scope=session failed to find the requested session content", {"explicit_result": explicit_a})
        rows1 = load_a_1.get("messages") or load_a_1.get("rows") or []
        rows2 = load_a_2.get("messages") or load_a_2.get("rows") or []
        if not rows1 or not rows2:
            run.fail(case, "load_session_pagination_empty", "lcm_load_session pagination returned empty pages for a populated session", {"page1": load_a_1, "page2": load_a_2})
        else:
            ids1 = [row.get("store_id") for row in rows1 if isinstance(row, dict)]
            ids2 = [row.get("store_id") for row in rows2 if isinstance(row, dict)]
            if set(ids1) & set(ids2):
                run.fail(case, "load_session_pagination_overlap", "lcm_load_session after_store_id pagination repeated rows", {"ids1": ids1, "ids2": ids2, "cursor": cursor})
        run.record(case, "current_a", current_a)
        run.record(case, "all_a", all_a)
        run.record(case, "explicit_a", explicit_a)
        run.record(case, "load_page_1", load_a_1)
        run.record(case, "load_page_2", load_a_2)
        run.record(case, "db_counts", _db_counts(Path(engine._store.db_path)))
    finally:
        engine.shutdown()


def _case_query_fuzz_no_crash(run: StressRun) -> None:
    _ensure_hermes_lcm_package(run.plugin_dir)
    from hermes_lcm.tokens import count_messages_tokens

    case = "query_fuzz_no_crash"
    engine = run.make_engine(case, fresh_tail_count=4, leaf_chunk_tokens=80)
    try:
        messages = [{"role": "system", "content": "fuzz system"}]
        weird_terms = [
            "alpha-beta", "owner/repo#123", "path:/tmp/x", "quoted \"term\"", "emoji 🚀", "中文片段", "C++", "foo:bar", "NEAR/5", "a*b?c", "sk-abc[redacted]", "line\nbreak",
        ]
        for i, term in enumerate(weird_terms * run.tier.fuzz_repetitions):
            messages.append({"role": "user", "content": f"FUZZ_{i:03d} term={term} CANARY_FUZZ_{i:03d} = VALUE_FUZZ_{i:03d}"})
            messages.append({"role": "assistant", "content": f"ack fuzz {i} {term}"})
        messages = engine.compress(messages, current_tokens=max(4_500, count_messages_tokens(messages)))
        errors: list[dict[str, Any]] = []
        queries = weird_terms + ["\"quoted term\"", "owner/repo#123 OR 中文片段", "C++ NOT java", "***", "((((", "role:user", "CANARY_FUZZ_001", "FUZZ_001 term=owner/repo#123"]
        for query in queries:
            for sort in ["recency", "relevance", "hybrid", "not-a-sort"]:
                res = run.call_tool(engine, "lcm_grep", {"query": query, "limit": 10, "sort": sort})
                serialized = json.dumps(res, ensure_ascii=False)
                if "Traceback" in serialized or ("error" in res and "must" not in str(res.get("error")) and "query" not in str(res.get("error")).lower()):
                    errors.append({"query": query, "sort": sort, "result": res})
        if errors:
            run.fail(case, "grep_query_fuzz_errors", "lcm_grep returned internal errors for punctuation/unicode query fuzzing", {"errors": errors[:20]})
        run.record(case, "query_count", len(queries) * 4)
        run.record(case, "sample_result", run.call_tool(engine, "lcm_grep", {"query": "CANARY_FUZZ_001", "limit": 3}))
    finally:
        engine.shutdown()


def _case_concurrent_read_write_smoke(run: StressRun) -> None:
    case = "concurrent_read_write_smoke"
    engine = run.make_engine(case, fresh_tail_count=6, leaf_chunk_tokens=120, dynamic_leaf_chunk_enabled=True)
    messages: list[dict[str, Any]] = [{"role": "system", "content": "concurrency system"}]
    lock = threading.Lock()
    thread_errors: list[dict[str, Any]] = []
    stop = False

    def reader(idx: int) -> None:
        nonlocal stop
        while not stop:
            try:
                with lock:
                    snapshot = list(messages)
                res = run.call_tool(engine, "lcm_grep", {"query": "CANARY_CONCURRENT", "limit": 5}, messages=snapshot)
                serialized = json.dumps(res).lower()
                if "database is locked" in serialized or "traceback" in serialized:
                    thread_errors.append({"reader": idx, "result": res})
            except Exception:
                thread_errors.append({"reader": idx, "traceback": traceback.format_exc()})
            time.sleep(0.002)

    threads = [threading.Thread(target=reader, args=(i,), daemon=True) for i in range(run.tier.reader_threads)]
    for thread in threads:
        thread.start()
    try:
        for i in range(run.tier.concurrent_turns):
            with lock:
                messages.append({"role": "user", "content": f"CANARY_CONCURRENT_{i:03d} = VALUE_CONCURRENT_{i:03d} " + ("load " * 60)})
                messages.append({"role": "assistant", "content": f"ack concurrent {i}"})
                if i % 10 == 9:
                    messages[:] = engine.compress(messages, current_tokens=4_500)
    finally:
        stop = True
        for thread in threads:
            thread.join(timeout=2)
    try:
        engine.on_session_end(engine.current_session_id, messages)
        if thread_errors:
            run.fail(case, "concurrent_read_write_errors", "Concurrent read/write smoke produced lock or internal errors", {"errors": thread_errors[:20]})
        final = run.call_tool(engine, "lcm_grep", {"query": "CANARY_CONCURRENT_000", "limit": 5})
        if "CANARY_CONCURRENT_000" not in json.dumps(final, ensure_ascii=False):
            run.fail(case, "concurrent_old_canary_missing", "Old canary missing after concurrent read/write stress", {"grep": final})
        run.record(case, "thread_errors_count", len(thread_errors))
        run.record(case, "final_old_canary", final)
        run.record(case, "db_counts", _db_counts(Path(engine._store.db_path)))
    finally:
        engine.shutdown()


_SCENARIO_FUNCTIONS: dict[str, Callable[[StressRun], None]] = {
    "multi_cycle_canary_recall": _case_multi_cycle_canary_recall,
    "redaction_and_externalization_boundaries": _case_redaction_and_externalization_boundaries,
    "cross_session_scope_and_pagination": _case_cross_session_scope_and_pagination,
    "query_fuzz_no_crash": _case_query_fuzz_no_crash,
    "concurrent_read_write_smoke": _case_concurrent_read_write_smoke,
}
