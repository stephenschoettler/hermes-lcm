from __future__ import annotations

import ast
import collections
import json
import re
from dataclasses import dataclass
from pathlib import Path

from hermes_lcm.engine import LCM_TOOL_TARGET_BINDINGS
from hermes_lcm.scope_storage import enumerate_scope_writers


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "access_context" / "inventory.json"
MUTATION_TARGETS_PATH = REPO_ROOT / "tests" / "data" / "lcm_mutation_targets.json"

_SQL_MUTATION_METHODS = {"execute", "executemany", "executescript"}
_SQL_MUTATION_VERBS = {
    "ALTER",
    "CREATE",
    "DELETE",
    "DROP",
    "INSERT",
    "REPLACE",
    "UPDATE",
}
_GATE_PRIMITIVES = {
    "policy_for_engine",
    "_authorize_apply_mutation",
    "_authorize_doctor_command",
}
_RESERVED_TARGET_WORDS = {
    "AS",
    "IF",
    "INTO",
    "NOT",
    "OF",
    "ON",
    "SET",
    "TABLE",
}
_EXCLUDED_CALL_NAMES = _SQL_MUTATION_METHODS | {
    "__enter__",
    "__exit__",
    "add",
    "append",
    "close",
    "commit",
    "fetchall",
    "fetchone",
    "get",
    "items",
    "join",
    "keys",
    "lower",
    "rollback",
    "setdefault",
    "sort",
    "startswith",
    "strip",
    "upper",
    "values",
}
_ENGINE_ATTRIBUTE_TYPES = {
    "_assertions": "AssertionStore",
    "_dag": "SummaryDAG",
    "_query_views": "QueryViewStore",
    "_store": "MessageStore",
    "_trajectory": "TrajectoryStore",
    "_vectors": "VectorStore",
}

# A string-literal mutation can name its target directly. These are the bounded
# dynamic identifier expressions used by the current tree. The key deliberately
# includes the source function and expression: a new dynamic target is not
# silently generalized and fails extraction until its finite target set is
# reviewed here.
_DYNAMIC_MUTATION_TARGETS: dict[tuple[str, str, str], tuple[str, ...]] = {
    (
        "command.py",
        "_ensure_inflight_table",
        "old_table",
    ): ("lcm_embedding_backfill_inflight_legacy",),
    (
        "command.py",
        "_ensure_inflight_table",
        "name",
    ): (
        "idx_lcm_embedding_inflight_identity_state",
        "idx_lcm_embedding_inflight_maintenance",
    ),
    (
        "command.py",
        "_delete_clean_candidates_atomically",
        "lifecycle_scope",
    ): ("temp_lcm_delete_lifecycle_scope",),
    (
        "dag.py",
        "SummaryDAG.stage_delete_session_scope",
        "_DELETE_SESSION_SCOPE_TABLE",
    ): ("lcm_delete_session_scope",),
    (
        "db_bootstrap.py",
        "check_external_content_fts_integrity",
        "quote_sql_identifier(spec.table_name)",
    ): ("messages_fts", "nodes_fts"),
    (
        "db_bootstrap.py",
        "_drop_fts_table",
        "quote_sql_identifier(table_name)",
    ): ("messages_fts", "nodes_fts"),
    (
        "db_bootstrap.py",
        "_drop_fts_table",
        "quote_sql_identifier(shadow_name)",
    ): (
        "messages_fts_config",
        "messages_fts_data",
        "messages_fts_docsize",
        "messages_fts_idx",
        "nodes_fts_config",
        "nodes_fts_data",
        "nodes_fts_docsize",
        "nodes_fts_idx",
    ),
    (
        "db_bootstrap.py",
        "repair_external_content_fts",
        "quote_sql_identifier(spec.table_name)",
    ): ("messages_fts", "nodes_fts"),
    (
        "db_bootstrap.py",
        "_drop_fts_triggers",
        "quote_sql_identifier(trigger_name)",
    ): (
        "msg_fts_delete",
        "msg_fts_insert",
        "msg_fts_update",
        "nodes_fts_delete",
        "nodes_fts_insert",
    ),
    (
        "db_bootstrap.py",
        "ensure_temporal_rollup_tables.ensure_index",
        "name",
    ): ("idx_*",),
    (
        "db_bootstrap.py",
        "ensure_temporal_rollup_invalidation_triggers",
        "trigger_name",
    ): (
        "lcm_rollup_node_delete",
        "lcm_rollup_node_insert",
        "lcm_rollup_node_update",
    ),
    (
        "db_bootstrap.py",
        "remediate_interim_schema_stamp",
        "quote_sql_identifier(str(trigger))",
    ): (
        "lcm_assertion_source_insert_guard",
        "lcm_rollup_node_delete",
        "lcm_rollup_node_insert",
        "lcm_rollup_node_update",
    ),
    (
        "db_bootstrap.py",
        "remediate_interim_schema_stamp",
        "quote_sql_identifier(str(table))",
    ): (
        "lcm_assertion_relations",
        "lcm_assertion_sources",
        "lcm_assertions",
        "lcm_chunk_binary",
        "lcm_chunk_meta",
        "lcm_chunk_vectors",
        "lcm_embedding_binary",
        "lcm_embedding_meta",
        "lcm_embedding_profile",
        "lcm_embedding_vectors",
        "lcm_rollup_invalidations",
        "lcm_rollup_sources",
        "lcm_rollup_state",
        "lcm_rollups",
    ),
    (
        "rollup_periods.py",
        "_load_source_lineage_staged",
        "table",
    ): ("lcm_lineage_current", "lcm_lineage_frontier", "lcm_lineage_seen"),
    (
        "scope_storage.py",
        "ensure_scope_columns",
        "table",
    ): (
        "lcm_chunk_binary",
        "lcm_chunk_meta",
        "lcm_chunk_vectors",
        "lcm_embedding_binary",
        "lcm_embedding_meta",
        "lcm_embedding_vectors",
        "lcm_rollup_invalidations",
        "lcm_rollup_state",
        "lcm_rollups",
        "messages",
        "summary_nodes",
    ),
    (
        "scope_storage.py",
        "_backfill_session_table",
        "table",
    ): ("messages", "summary_nodes"),
    (
        "scope_storage.py",
        "_backfill_joined_table",
        "table",
    ): (
        "lcm_chunk_binary",
        "lcm_chunk_meta",
        "lcm_chunk_vectors",
        "lcm_embedding_binary",
        "lcm_embedding_meta",
        "lcm_embedding_vectors",
    ),
    (
        "scope_storage.py",
        "_backfill_rollup_table",
        "table",
    ): ("lcm_rollup_invalidations", "lcm_rollup_state", "lcm_rollups"),
    (
        "vector_store.py",
        "VectorStore._temp_id_table",
        "table",
    ): ("_lcm_id_scratch_*",),
}


@dataclass(frozen=True)
class HookCall:
    site: str
    line: int
    node: ast.Call
    function: ast.AST | None


@dataclass(frozen=True)
class MutationSite:
    module: str
    function: str
    line: int
    target: str | None
    unresolved: str | None = None

    @property
    def location(self) -> str:
        return f"{self.module}:{self.function}:{self.line}"


@dataclass
class FunctionRecord:
    key: str
    module: str
    qualname: str
    name: str
    class_name: str | None
    node: ast.FunctionDef | ast.AsyncFunctionDef
    calls: set[str]
    local_gates: set[str]


class _HookVisitor(ast.NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.stack: list[str] = []
        self.functions: list[ast.AST] = []
        self.calls: list[HookCall] = []

    def _site(self) -> str:
        suffix = ".".join(self.stack)
        return f"{self.module}:{suffix}" if suffix else self.module

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.stack.append(node.name)
        self.functions.append(node)
        self.generic_visit(node)
        self.functions.pop()
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "policy_for_engine"
        ):
            self.calls.append(
                HookCall(self._site(), node.lineno, node, self.functions[-1] if self.functions else None)
            )
        self.generic_visit(node)


def _inventory_payload() -> list[dict[str, object]]:
    raw = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, list) and raw
    assert all(isinstance(entry, dict) for entry in raw)
    return raw


def _source_files() -> tuple[Path, ...]:
    # "benchmarking" joins the list its two siblings were already on: it is a
    # separate evaluation harness with its OWN sqlite databases (an embedding
    # cache keyed by content hash), not the LCM store, and no principal can
    # reach it. The -ing spelling was simply missed when the others were added,
    # which stayed invisible until an upstream merge put a mutating file there.
    excluded = {"tests", "bench", "benchmarks", "benchmarking", "__pycache__"}
    return tuple(
        sorted(
            path
            for path in REPO_ROOT.rglob("*.py")
            if not any(part.startswith(".venv") or part in excluded for part in path.parts)
        )
    )


def test_scope_bearing_writers_populate_access_scope() -> None:
    writers = enumerate_scope_writers(REPO_ROOT)
    violations = [
        writer.name for writer in writers if not writer.populates_access_scope
    ]
    assert not violations, (
        "scope-bearing writers missing access_scope: " + ", ".join(violations)
    )


def _hook_calls() -> tuple[HookCall, ...]:
    calls: list[HookCall] = []
    for path in _source_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        visitor = _HookVisitor(relative)
        visitor.visit(tree)
        calls.extend(visitor.calls)
    return tuple(calls)


def _function_records() -> dict[str, FunctionRecord]:
    records: dict[str, FunctionRecord] = {}
    for path in _source_files():
        module = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=module)
        stack: list[str] = []
        classes: list[str] = []

        class Collector(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                stack.append(node.name)
                classes.append(node.name)
                self.generic_visit(node)
                classes.pop()
                stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._function(node)

            def _function(
                self, node: ast.FunctionDef | ast.AsyncFunctionDef
            ) -> None:
                stack.append(node.name)
                qualname = ".".join(stack)
                key = f"{module}:{qualname}"
                records[key] = FunctionRecord(
                    key=key,
                    module=module,
                    qualname=qualname,
                    name=node.name,
                    class_name=classes[-1] if classes else None,
                    node=node,
                    calls=set(),
                    local_gates=set(),
                )
                self.generic_visit(node)
                stack.pop()

        Collector().visit(tree)
    return records


class _DirectCallVisitor(ast.NodeVisitor):
    """Visit one function body without crediting nested lexical scopes."""

    def __init__(self, *, include_lambdas: bool = False) -> None:
        self.calls: list[ast.Call] = []
        self.include_lambdas = include_lambdas

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        if self.include_lambdas:
            self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return


def _direct_calls(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    include_lambdas: bool = False,
) -> tuple[ast.Call, ...]:
    visitor = _DirectCallVisitor(include_lambdas=include_lambdas)
    for statement in node.body:
        visitor.visit(statement)
    return tuple(visitor.calls)


def _sql_template(
    node: ast.AST,
) -> tuple[str, tuple[ast.AST, ...]] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, ()
    if not isinstance(node, ast.JoinedStr) or not node.values:
        return None
    first = node.values[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return None
    expressions: list[ast.AST] = []
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            parts.append(f"__LCM_EXPR_{len(expressions)}__")
            expressions.append(value.value)
        else:
            return None
    return "".join(parts), tuple(expressions)


def _mutation_offsets(sql: str) -> tuple[int, ...]:
    """Find mutation verbs outside SQL strings and comments."""

    offsets: list[int] = []
    index = 0
    quote: str | None = None
    line_comment = False
    block_comment = False
    while index < len(sql):
        char = sql[index]
        pair = sql[index:index + 2]
        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if pair == "*/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote is not None:
            if quote == "]":
                if char == "]":
                    quote = None
            elif char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if pair == "--":
            line_comment = True
            index += 2
            continue
        if pair == "/*":
            block_comment = True
            index += 2
            continue
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            continue
        if char == "[":
            quote = "]"
            index += 1
            continue
        if char.isalpha():
            match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", sql[index:])
            assert match is not None
            word = match.group(0).upper()
            if word in _SQL_MUTATION_VERBS:
                offsets.append(index)
            index += len(match.group(0))
            continue
        index += 1
    return tuple(offsets)


def _word(sql: str, offset: int) -> tuple[str, int] | None:
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", sql[offset:])
    if match is None:
        return None
    return match.group(1).upper(), offset + match.end()


def _consume(sql: str, offset: int, expected: str) -> int | None:
    parsed = _word(sql, offset)
    if parsed is None or parsed[0] != expected:
        return None
    return parsed[1]


def _identifier_component(sql: str, offset: int) -> tuple[str, int] | None:
    offset += len(sql[offset:]) - len(sql[offset:].lstrip())
    if offset >= len(sql):
        return None
    opener = sql[offset]
    closer = {('"'): '"', "`": "`", "[": "]"}.get(opener)
    if closer is not None:
        cursor = offset + 1
        while cursor < len(sql):
            if sql[cursor] == closer:
                if closer != "]" and cursor + 1 < len(sql) and sql[cursor + 1] == closer:
                    cursor += 2
                    continue
                return sql[offset + 1:cursor], cursor + 1
            cursor += 1
        return None
    match = re.match(
        r"(?:__LCM_EXPR_[0-9]+__|[A-Za-z_][A-Za-z0-9_$]*)",
        sql[offset:],
    )
    if match is None:
        return None
    return match.group(0), offset + match.end()


def _identifier(sql: str, offset: int) -> tuple[str, int] | None:
    first = _identifier_component(sql, offset)
    if first is None:
        return None
    name, cursor = first
    spaced = cursor + len(sql[cursor:]) - len(sql[cursor:].lstrip())
    if spaced < len(sql) and sql[spaced] == ".":
        second = _identifier_component(sql, spaced + 1)
        if second is None:
            return None
        name, cursor = second
    return name, cursor


def _mutation_target(sql: str) -> str | None:
    parsed = _word(sql, 0)
    if parsed is None:
        return None
    verb, offset = parsed
    if verb == "INSERT":
        maybe_or = _word(sql, offset)
        if maybe_or and maybe_or[0] == "OR":
            resolution = _word(sql, maybe_or[1])
            if resolution is None:
                return None
            offset = resolution[1]
        offset = _consume(sql, offset, "INTO") or -1
    elif verb == "REPLACE":
        offset = _consume(sql, offset, "INTO") or -1
    elif verb == "UPDATE":
        maybe_or = _word(sql, offset)
        if maybe_or and maybe_or[0] == "OR":
            resolution = _word(sql, maybe_or[1])
            if resolution is None:
                return None
            offset = resolution[1]
    elif verb == "DELETE":
        offset = _consume(sql, offset, "FROM") or -1
    elif verb == "ALTER":
        offset = _consume(sql, offset, "TABLE") or -1
    elif verb == "DROP":
        kind = _word(sql, offset)
        if kind is None or kind[0] not in {"INDEX", "TABLE", "TRIGGER", "VIEW"}:
            return None
        offset = kind[1]
        maybe_if = _word(sql, offset)
        if maybe_if and maybe_if[0] == "IF":
            offset = _consume(sql, maybe_if[1], "EXISTS") or -1
    elif verb == "CREATE":
        modifier = _word(sql, offset)
        if modifier and modifier[0] in {"TEMP", "TEMPORARY", "UNIQUE"}:
            offset = modifier[1]
            modifier = _word(sql, offset)
        if modifier and modifier[0] == "VIRTUAL":
            offset = modifier[1]
            modifier = _word(sql, offset)
        if modifier is None or modifier[0] not in {
            "INDEX",
            "TABLE",
            "TRIGGER",
            "VIEW",
        }:
            return None
        offset = modifier[1]
        maybe_if = _word(sql, offset)
        if maybe_if and maybe_if[0] == "IF":
            offset = _consume(sql, maybe_if[1], "NOT") or -1
            offset = _consume(sql, offset, "EXISTS") if offset >= 0 else -1
            offset = offset or -1
    else:
        return None
    if offset < 0:
        return None
    target = _identifier(sql, offset)
    if target is None or target[0].upper() in _RESERVED_TARGET_WORDS:
        return None
    return target[0]


def _dynamic_targets(
    module: str,
    function: str,
    expression: ast.AST,
) -> tuple[str, ...] | None:
    rendered = ast.unparse(expression)
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return (expression.value,)
    return _DYNAMIC_MUTATION_TARGETS.get((module, function, rendered))


def test_mutation_target_parser_covers_supported_sqlite_forms() -> None:
    cases = {
        "CREATE TABLE IF NOT EXISTS messages(id INTEGER)": "messages",
        "CREATE UNIQUE INDEX IF NOT EXISTS [idx_messages] ON messages(id)": "idx_messages",
        "CREATE TEMP TABLE `scratch_rows`(id INTEGER)": "scratch_rows",
        "INSERT OR REPLACE INTO [messages](id) VALUES(1)": "messages",
        'UPDATE "messages" SET content = NULL': "messages",
        "DELETE FROM temp.[scratch_rows]": "scratch_rows",
        'ALTER TABLE "messages" ADD COLUMN note TEXT': "messages",
        "DROP TABLE IF EXISTS `messages`": "messages",
    }
    for sql, expected in cases.items():
        assert _mutation_target(sql) == expected


def _mutation_sites(
    records: dict[str, FunctionRecord] | None = None,
) -> tuple[MutationSite, ...]:
    records = records or _function_records()
    sites: list[MutationSite] = []
    for record in records.values():
        for call in _direct_calls(record.node, include_lambdas=True):
            if (
                not isinstance(call.func, ast.Attribute)
                or call.func.attr not in _SQL_MUTATION_METHODS
                or not call.args
            ):
                continue
            template = _sql_template(call.args[0])
            if template is None:
                continue
            sql, expressions = template
            offsets = _mutation_offsets(sql)
            for offset in offsets:
                fragment = sql[offset:]
                target = _mutation_target(fragment)
                if target is None:
                    if not sql[:offset].strip(" \t\r\n;"):
                        sites.append(
                            MutationSite(
                                record.module,
                                record.qualname,
                                call.lineno,
                                None,
                                "mutation SQL begins with an unresolvable target",
                            )
                        )
                    continue
                marker = re.fullmatch(r"__LCM_EXPR_([0-9]+)__", target)
                if marker is None:
                    sites.append(
                        MutationSite(
                            record.module,
                            record.qualname,
                            call.lineno,
                            target.lower(),
                        )
                    )
                    continue
                expression_index = int(marker.group(1))
                resolved = _dynamic_targets(
                    record.module,
                    record.qualname,
                    expressions[expression_index],
                )
                if not resolved:
                    sites.append(
                        MutationSite(
                            record.module,
                            record.qualname,
                            call.lineno,
                            None,
                            "dynamic target "
                            + ast.unparse(expressions[expression_index]),
                        )
                    )
                    continue
                sites.extend(
                    MutationSite(
                        record.module,
                        record.qualname,
                        call.lineno,
                        resolved_target.lower(),
                    )
                    for resolved_target in resolved
                )
    return tuple(sites)


def _mutation_target_payload() -> tuple[
    tuple[dict[str, str], ...],
    tuple[dict[str, str], ...],
]:
    raw = json.loads(MUTATION_TARGETS_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), "mutation target map must be an object"
    target_entries = raw.get("targets")
    waivers = raw.get("handler_waivers", [])
    assert isinstance(target_entries, list) and target_entries
    assert isinstance(waivers, list)
    normalized_targets: list[dict[str, str]] = []
    for entry in target_entries:
        assert isinstance(entry, dict)
        table = entry.get("table")
        classification = entry.get("class")
        reason = entry.get("reason")
        assert isinstance(table, str) and table.strip()
        assert classification in {"principal_data", "infra"}
        assert isinstance(reason, str) and reason.strip()
        normalized_targets.append(
            {"table": table.lower(), "class": classification, "reason": reason}
        )
    tables = [entry["table"] for entry in normalized_targets]
    assert len(tables) == len(set(tables)), "duplicate mutation target classifications"
    normalized_waivers: list[dict[str, str]] = []
    for waiver in waivers:
        assert isinstance(waiver, dict)
        module = waiver.get("module")
        handler = waiver.get("handler")
        reason = waiver.get("reason")
        assert isinstance(module, str) and module in {"command.py", "tools.py"}
        assert isinstance(handler, str) and handler.strip()
        assert isinstance(reason, str) and reason.strip()
        normalized_waivers.append(
            {"module": module, "handler": handler, "reason": reason}
        )
    return tuple(normalized_targets), tuple(normalized_waivers)


def _target_class(
    target: str,
    entries: tuple[dict[str, str], ...],
) -> str | None:
    exact = [entry for entry in entries if entry["table"] == target]
    patterns = [
        entry
        for entry in entries
        if entry["table"].endswith("*")
        and target.startswith(entry["table"][:-1])
    ]
    matches = exact or patterns
    assert len(matches) <= 1, f"overlapping target classifications for {target}"
    return matches[0]["class"] if matches else None


def test_every_db_mutation_target_is_resolved_and_classified() -> None:
    entries, _waivers = _mutation_target_payload()
    sites = _mutation_sites()
    assert sites, "no literal DB mutation sites discovered"
    unresolved = sorted(
        f"{site.location} ({site.unresolved})"
        for site in sites
        if site.target is None
    )
    assert not unresolved, "unresolved DB mutation targets:\n" + "\n".join(unresolved)
    unclassified = sorted(
        f"{site.location} -> {site.target}"
        for site in sites
        if site.target is not None and _target_class(site.target, entries) is None
    )
    assert not unclassified, "unclassified DB mutation targets:\n" + "\n".join(unclassified)


def _import_bindings(module: str) -> dict[str, tuple[str, str]]:
    path = REPO_ROOT / module
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=module)
    current_parts = list(Path(module).with_suffix("").parts)
    if current_parts and current_parts[-1] == "__init__":
        current_parts.pop()
    else:
        current_parts = current_parts[:-1]
    bindings: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            keep = max(0, len(current_parts) - (node.level - 1))
            parts = current_parts[:keep]
        else:
            parts = []
        if node.module:
            parts.extend(node.module.split("."))
        imported_module = "/".join(parts) + ".py"
        if not (REPO_ROOT / imported_module).exists():
            package_module = "/".join(parts + ["__init__"]) + ".py"
            if (REPO_ROOT / package_module).exists():
                imported_module = package_module
        for alias in node.names:
            bindings[alias.asname or alias.name] = (imported_module, alias.name)
    return bindings


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _annotation_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
        and child.id not in {"None", "Optional"}
    }


def _local_types(record: FunctionRecord) -> dict[str, set[str]]:
    types: dict[str, set[str]] = collections.defaultdict(set)
    arguments = (
        list(record.node.args.posonlyargs)
        + list(record.node.args.args)
        + list(record.node.args.kwonlyargs)
    )
    if record.node.args.vararg is not None:
        arguments.append(record.node.args.vararg)
    if record.node.args.kwarg is not None:
        arguments.append(record.node.args.kwarg)
    for argument in arguments:
        types[argument.arg].update(_annotation_names(argument.annotation))

    class Visitor(ast.NodeVisitor):
        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name):
                types[node.target.id].update(_annotation_names(node.annotation))
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            if isinstance(node.value, ast.Call):
                constructor = _call_name(node.value)
                if constructor:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if constructor == "_require_engine":
                                types[target.id].add("LCMEngine")
                            elif (
                                constructor == "getattr"
                                and len(node.value.args) >= 2
                                and isinstance(node.value.args[1], ast.Constant)
                                and isinstance(node.value.args[1].value, str)
                            ):
                                attribute_type = _ENGINE_ATTRIBUTE_TYPES.get(
                                    node.value.args[1].value
                                )
                                if attribute_type:
                                    types[target.id].add(attribute_type)
                            else:
                                types[target.id].add(constructor)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

    visitor = Visitor()
    for statement in record.node.body:
        visitor.visit(statement)
    if "engine" in types or any(
        argument.arg == "engine"
        for argument in (
            list(record.node.args.posonlyargs)
            + list(record.node.args.args)
            + list(record.node.args.kwonlyargs)
        )
    ):
        types["engine"].add("LCMEngine")
    return types


def _populate_call_graph(records: dict[str, FunctionRecord]) -> None:
    by_name: dict[str, set[str]] = collections.defaultdict(set)
    by_module_name: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    by_class_name: dict[tuple[str, str, str], set[str]] = collections.defaultdict(set)
    by_simple_class_name: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for record in records.values():
        by_name[record.name].add(record.key)
        by_module_name[(record.module, record.name)].add(record.key)
        if record.class_name:
            by_class_name[(record.module, record.class_name, record.name)].add(record.key)
            by_simple_class_name[(record.class_name, record.name)].add(record.key)
    imports = {module: _import_bindings(module) for module in {r.module for r in records.values()}}
    for record in records.values():
        local_types = _local_types(record)
        direct_gate_calls = {
            name
            for call in _direct_calls(record.node)
            if (name := _call_name(call)) in _GATE_PRIMITIVES
        }
        record.local_gates.update(direct_gate_calls)
        for call in _direct_calls(record.node, include_lambdas=True):
            name = _call_name(call)
            if not name:
                continue
            if name in _EXCLUDED_CALL_NAMES:
                continue
            candidates: set[str] = set()
            if isinstance(call.func, ast.Name):
                imported = imports[record.module].get(name)
                if imported is not None:
                    candidates.update(by_module_name[imported])
                candidates.update(by_module_name[(record.module, name)])
            elif isinstance(call.func, ast.Attribute):
                owner = call.func.value
                if isinstance(owner, ast.Name) and owner.id in {"self", "cls"} and record.class_name:
                    candidates.update(
                        by_class_name[(record.module, record.class_name, name)]
                    )
                elif isinstance(owner, ast.Name):
                    candidates.update(by_class_name[(record.module, owner.id, name)])
                    imported = imports[record.module].get(owner.id)
                    if imported is not None:
                        candidates.update(
                            by_class_name[(imported[0], imported[1], name)]
                        )
                    for class_name in local_types.get(owner.id, ()):
                        candidates.update(
                            by_simple_class_name[(class_name, name)]
                        )
                elif (
                    isinstance(owner, ast.Attribute)
                    and isinstance(owner.value, ast.Name)
                    and "LCMEngine" in local_types.get(owner.value.id, ())
                ):
                    attribute_type = _ENGINE_ATTRIBUTE_TYPES.get(owner.attr)
                    if attribute_type:
                        candidates.update(
                            by_simple_class_name[(attribute_type, name)]
                        )
            if not candidates:
                # Bare names resolve through Python's lexical/module bindings.
                # An untyped attribute is intentionally not fanned out across
                # unrelated classes; that was the structurally-blind
                # prototype's false-credit mechanism in another form.
                if isinstance(call.func, ast.Name):
                    candidates.update(by_name[name])
            record.calls.update(candidates)


def _entry_points(records: dict[str, FunctionRecord]) -> dict[str, FunctionRecord]:
    return {
        key: record
        for key, record in records.items()
        if (
            record.module == "command.py"
            and record.class_name is None
            and record.name.startswith("_")
            and record.name.endswith("_text")
        )
        or (
            record.module == "tools.py"
            and record.class_name is None
            and record.name.startswith("lcm_")
        )
    }


def test_handlers_reaching_principal_data_call_a_gate_locally() -> None:
    records = _function_records()
    _populate_call_graph(records)
    entries, waivers = _mutation_target_payload()
    mutation_targets_by_function: dict[str, set[str]] = collections.defaultdict(set)
    for site in _mutation_sites(records):
        if site.target is not None:
            mutation_targets_by_function[f"{site.module}:{site.function}"].add(site.target)
    handlers = _entry_points(records)
    assert len([r for r in handlers.values() if r.module == "command.py"]) >= 32
    assert len([r for r in handlers.values() if r.module == "tools.py"]) >= 15
    waiver_keys = {(w["module"], w["handler"]) for w in waivers}
    unknown_waivers = sorted(
        f"{module}:{handler}"
        for module, handler in waiver_keys
        if f"{module}:{handler}" not in handlers
    )
    assert not unknown_waivers, f"handler waivers no longer resolve: {unknown_waivers}"

    violations: list[str] = []
    handler_keys = set(handlers)
    for handler in handlers.values():
        reached: set[str] = set()
        pending = [handler.key]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            reached.update(mutation_targets_by_function.get(current, ()))
            for callee in records[current].calls:
                # Other named handlers are separate authorization boundaries.
                if callee not in handler_keys or callee == handler.key:
                    pending.append(callee)
        principal_targets = sorted(
            target
            for target in reached
            if _target_class(target, entries) == "principal_data"
        )
        if (
            principal_targets
            and not handler.local_gates
            and (handler.module, handler.name) not in waiver_keys
        ):
            violations.append(
                f"{handler.module}:{handler.name} reaches "
                + ", ".join(principal_targets)
                + " without a lexically-local gate call"
            )
    assert not violations, "ungated principal-data handlers:\n" + "\n".join(violations)


def test_hook_sites_resolve_only_through_access_policy_seam() -> None:
    calls = _hook_calls()
    assert calls
    calls_by_module: dict[str, list[HookCall]] = {}
    for call in calls:
        calls_by_module.setdefault(call.site.split(":", 1)[0], []).append(call)

    for module, module_calls in calls_by_module.items():
        path = REPO_ROOT / module
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=module)
        # The seam must be imported RELATIVELY, as every other local module is.
        # An absolute import (importlib.import_module("access_policy") or a bare
        # `from access_policy import ...`) loads a SECOND copy of the package when
        # the plugin is loaded as `hermes_lcm`, so a caller's
        # `except AuthorizationRequiredError` would not catch what the engine
        # raises -- and it breaks package import entirely when the plugin
        # directory is not on sys.path. Proven on a real production store.
        relative_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.level >= 1
            and (
                (node.module or "").endswith("access_policy")
                or any(a.name == "access_policy" for a in node.names)
            )
        ] + [
            # A standalone script cannot use a relative import, so the
            # package-qualified name is the correct equivalent for it.
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and str(node.args[0].value) == "hermes_lcm.access_policy"
        ]
        assert relative_imports, (
            f"{module} must import access_policy relatively (found none); "
            "an absolute import creates a duplicate package under hermes_lcm"
        )
        absolute_imports = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and (node.module or "").startswith("access_policy")
            )
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).startswith("access_policy")
            )
        ]
        assert not absolute_imports, (
            f"{module} imports access_policy ABSOLUTELY; this duplicates the "
            "package under hermes_lcm and breaks strict package import"
        )

        for call in module_calls:
            assert isinstance(call.node.func, ast.Name)
            assert call.node.func.id == "policy_for_engine"
            if call.function is None:
                continue
            for node in ast.walk(call.function):
                if isinstance(node, ast.Call):
                    assert not (
                        isinstance(node.func, ast.Name)
                        and node.func.id in {"resolve_policy", "TrustedOwnerPolicy", "FailClosedPolicy"}
                    ), f"{call.site} resolves policy outside policy_for_engine"
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id == "getattr"
                        and len(node.args) >= 2
                        and isinstance(node.args[1], ast.Constant)
                        and node.args[1].value in {"lcm_teams_enabled", "get_lcm_access_context"}
                    ):
                        raise AssertionError(f"{call.site} reads policy wiring directly")
                if isinstance(node, ast.Attribute) and node.attr in {
                    "lcm_teams_enabled",
                    "get_lcm_access_context",
                    "resolve_policy",
                    "TrustedOwnerPolicy",
                    "FailClosedPolicy",
                }:
                    raise AssertionError(f"{call.site} reads policy wiring directly")


def test_tool_authority_paths_are_discovered_from_source() -> None:
    tree = ast.parse((REPO_ROOT / "tools.py").read_text(encoding="utf-8"), filename="tools.py")
    source_tools = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("lcm_")
    }
    inventory_tools = {
        str(entry["entry_point"])
        for entry in _inventory_payload()
        if entry.get("module") == "tools.py" and str(entry.get("entry_point", "")).startswith("lcm_")
    }
    assert source_tools == inventory_tools, (
        f"source lcm_* handlers differ from inventory: "
        f"source-only={sorted(source_tools - inventory_tools)}, "
        f"inventory-only={sorted(inventory_tools - source_tools)}"
    )


def test_tool_target_bindings_cover_source_and_inventory_both_directions() -> None:
    tree = ast.parse((REPO_ROOT / "tools.py").read_text(encoding="utf-8"), filename="tools.py")
    source_tools = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("lcm_")
    }
    mapping_tools = set(LCM_TOOL_TARGET_BINDINGS)
    assert mapping_tools == source_tools, (
        "target binding map differs from source lcm_* handlers: "
        f"source-only={sorted(source_tools - mapping_tools)}, "
        f"mapping-only={sorted(mapping_tools - source_tools)}"
    )

    inventory_entries = {
        str(entry["entry_point"]): entry
        for entry in _inventory_payload()
        if entry.get("module") == "tools.py" and str(entry.get("entry_point", "")).startswith("lcm_")
    }
    assert set(inventory_entries) == mapping_tools
    for tool_name in sorted(source_tools):
        entry = inventory_entries[tool_name]
        binding = entry.get("target_binding")
        assert isinstance(binding, dict), f"{tool_name} must declare target_binding"
        expected = LCM_TOOL_TARGET_BINDINGS[tool_name]
        expected_args = list(expected.get("args", ()))
        assert binding.get("args") == expected_args, f"{tool_name} target args drifted"
        target_free = bool(expected.get("target_free", False))
        assert bool(binding.get("target_free", False)) is target_free
        if target_free:
            reason = binding.get("reason")
            assert isinstance(reason, str) and reason.strip(), (
                f"{tool_name} target_free entries require a non-empty reason"
            )
        else:
            assert not binding.get("reason"), f"{tool_name} target-bound entries cannot carry a target-free reason"
