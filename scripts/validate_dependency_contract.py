#!/usr/bin/env python3
"""Validate the versioned Hermes-LCM host-owned dependency contract."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONTRACT = REPO_ROOT / "dependency-contract.json"
_CONTRACT_VERSION_RE = re.compile(r"^[1-9]\d*\.\d+\.\d+$")
_HOST_VERSION_RE = re.compile(r"^>=\d+\.\d+,<\d+$")


def _literal_dynamic_import(node: ast.Call) -> str | None:
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    module = node.args[0].value
    if not isinstance(module, str) or not module:
        return None
    function = node.func
    if isinstance(function, ast.Name) and function.id == "__import__":
        return module
    if (
        isinstance(function, ast.Attribute)
        and function.attr == "import_module"
        and isinstance(function.value, ast.Name)
        and function.value.id == "importlib"
    ):
        return module
    return None


def _python_files(repo_root: Path, globs: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in globs:
        files.update(path for path in repo_root.glob(pattern) if path.is_file())
    return sorted(files)


def collect_external_imports(
    repo_root: Path,
    globs: Iterable[str],
    *,
    local_imports: set[str],
) -> dict[str, list[str]]:
    """Return non-stdlib, non-local top-level imports and source references."""
    local_modules = set(local_imports)
    local_modules.update(path.stem for path in repo_root.glob("*.py"))
    external: dict[str, set[str]] = {}
    for path in _python_files(repo_root, globs):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative_path = path.relative_to(repo_root)
        for node in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    imported_modules.append(node.module)
            elif isinstance(node, ast.Call):
                dynamic_import = _literal_dynamic_import(node)
                if dynamic_import:
                    imported_modules.append(dynamic_import)
            for imported_module in imported_modules:
                top_level = imported_module.split(".", 1)[0]
                if top_level in sys.stdlib_module_names or top_level in local_modules:
                    continue
                reference = f"{relative_path}:{getattr(node, 'lineno', 0)}"
                external.setdefault(top_level, set()).add(reference)
    return {
        module: sorted(references)
        for module, references in sorted(external.items())
    }


def _bound_target_names(target: ast.AST) -> set[str]:
    return {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }


class _DirectScopeBindingCollector(ast.NodeVisitor):
    """Collect bindings owned by one function scope, excluding nested scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()
        self.global_names: set[str] = set()
        self.nonlocal_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(alias.asname or alias.name for alias in node.names if alias.name != "*")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)

    def _visit_comprehension(self, node: ast.AST) -> None:
        generators = node.generators
        for generator in generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension


class _AliasScope:
    def __init__(
        self,
        kind: str,
        *,
        bindings: dict[str, str | None] | None = None,
        global_names: set[str] | None = None,
        nonlocal_names: set[str] | None = None,
    ) -> None:
        self.kind = kind
        self.bindings = bindings or {}
        self.global_names = global_names or set()
        self.nonlocal_names = nonlocal_names or set()


class _ModuleAliasUseCollector(ast.NodeVisitor):
    """Resolve attribute uses rooted at module imports without crossing shadows."""

    _FUNCTION_SCOPE_KINDS = {"function", "lambda", "comprehension"}

    def __init__(
        self,
        tree: ast.AST,
        *,
        local_modules: set[str],
        reference_path: Path,
    ) -> None:
        self.local_modules = local_modules
        self.reference_path = reference_path
        self.external: dict[str, dict[str, set[str]]] = {}
        self.scopes = [_AliasScope("module")]
        self.parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }

    def _set_binding(self, name: str, imported_api: str | None) -> None:
        scope = self.scopes[-1]
        if name in scope.global_names:
            self.scopes[0].bindings[name] = imported_api
            return
        if name in scope.nonlocal_names:
            for outer in reversed(self.scopes[:-1]):
                if outer.kind != "class" and name in outer.bindings:
                    outer.bindings[name] = imported_api
                    return
        scope.bindings[name] = imported_api

    def _resolve_binding(self, name: str) -> str | None:
        origin_is_function = self.scopes[-1].kind in self._FUNCTION_SCOPE_KINDS
        for index in range(len(self.scopes) - 1, -1, -1):
            scope = self.scopes[index]
            if name in scope.global_names:
                return self.scopes[0].bindings.get(name)
            if origin_is_function and scope.kind == "class":
                continue
            if name in scope.bindings:
                return scope.bindings[name]
        return None

    def _push_function_scope(self, node: ast.AST, arguments: ast.arguments) -> None:
        collector = _DirectScopeBindingCollector()
        body = node.body if isinstance(node.body, list) else [node.body]
        for statement in body:
            collector.visit(statement)
        argument_names = {
            argument.arg
            for argument in (
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            )
        }
        if arguments.vararg:
            argument_names.add(arguments.vararg.arg)
        if arguments.kwarg:
            argument_names.add(arguments.kwarg.arg)
        local_names = (collector.names | argument_names) - collector.global_names - collector.nonlocal_names
        self.scopes.append(
            _AliasScope(
                "lambda" if isinstance(node, ast.Lambda) else "function",
                bindings={name: None for name in local_names},
                global_names=collector.global_names,
                nonlocal_names=collector.nonlocal_names,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top_level = alias.name.split(".", 1)[0]
            bound_name = alias.asname or top_level
            imported_api = alias.name if alias.asname else top_level
            if top_level in sys.stdlib_module_names or top_level in self.local_modules:
                self._set_binding(bound_name, None)
            else:
                self._set_binding(bound_name, imported_api)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        top_level = module.split(".", 1)[0]
        is_external = (
            node.level == 0
            and bool(top_level)
            and top_level not in sys.stdlib_module_names
            and top_level not in self.local_modules
        )
        for alias in node.names:
            if alias.name != "*":
                imported_api = f"{module}.{alias.name}" if is_external else None
                self._set_binding(alias.asname or alias.name, imported_api)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)
        self._set_binding(node.name, None)
        self._push_function_scope(node, node.args)
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        self._push_function_scope(node, node.args)
        self.visit(node.body)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._set_binding(node.name, None)
        self.scopes.append(_AliasScope("class"))
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.visit(node.target)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._set_binding(node.id, None)

    def _visit_for(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self.visit(node.target)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    visit_For = _visit_for
    visit_AsyncFor = _visit_for

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self._set_binding(node.name, None)
        for statement in node.body:
            self.visit(statement)

    def _visit_branch(
        self,
        statements: Iterable[ast.stmt],
        initial_bindings: dict[str, str | None],
    ) -> dict[str, str | None]:
        self.scopes[-1].bindings = dict(initial_bindings)
        for statement in statements:
            self.visit(statement)
        return dict(self.scopes[-1].bindings)

    @staticmethod
    def _merge_possible_bindings(
        branches: Iterable[dict[str, str | None]],
    ) -> dict[str, str | None]:
        branch_list = list(branches)
        merged: dict[str, str | None] = {}
        for name in set().union(*(branch.keys() for branch in branch_list)):
            imported_apis = {
                branch.get(name)
                for branch in branch_list
                if isinstance(branch.get(name), str)
            }
            merged[name] = next(iter(imported_apis)) if len(imported_apis) == 1 else None
        return merged

    def visit_Try(self, node: ast.Try | ast.TryStar) -> None:
        initial_bindings = dict(self.scopes[-1].bindings)
        try_bindings = self._visit_branch(node.body, initial_bindings)
        normal_bindings = self._visit_branch(node.orelse, try_bindings)
        branch_bindings = [normal_bindings]
        for handler in node.handlers:
            self.scopes[-1].bindings = dict(initial_bindings)
            self.visit(handler)
            branch_bindings.append(dict(self.scopes[-1].bindings))
        self.scopes[-1].bindings = self._merge_possible_bindings(branch_bindings)
        for statement in node.finalbody:
            self.visit(statement)

    visit_TryStar = visit_Try

    def _visit_comprehension(self, node: ast.AST) -> None:
        local_names = {
            name
            for generator in node.generators
            for name in _bound_target_names(generator.target)
        }
        self.scopes.append(
            _AliasScope("comprehension", bindings={name: None for name in local_names})
        )
        for generator in node.generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.scopes.pop()

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def visit_Attribute(self, node: ast.Attribute) -> None:
        parent = self.parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.value is node:
            self.generic_visit(node)
            return
        attributes: list[str] = []
        value: ast.AST = node
        while isinstance(value, ast.Attribute):
            attributes.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            imported_api = self._resolve_binding(value.id)
            if imported_api is not None:
                canonical_api = ".".join((imported_api, *reversed(attributes)))
                top_level = canonical_api.split(".", 1)[0]
                reference = f"{self.reference_path}:{node.lineno}"
                self.external.setdefault(top_level, {}).setdefault(
                    canonical_api,
                    set(),
                ).add(reference)
        self.generic_visit(node)


def collect_external_imported_apis(
    repo_root: Path,
    globs: Iterable[str],
    *,
    local_imports: set[str],
) -> dict[str, dict[str, list[str]]]:
    """Return external modules and directly imported symbols by package root."""
    local_modules = set(local_imports)
    local_modules.update(path.stem for path in repo_root.glob("*.py"))
    external: dict[str, dict[str, set[str]]] = {}
    for path in _python_files(repo_root, globs):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative_path = path.relative_to(repo_root)
        for node in ast.walk(tree):
            imported_apis: list[str] = []
            if isinstance(node, ast.Import):
                imported_apis.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for alias in node.names:
                    imported_apis.append(
                        node.module if alias.name == "*" else f"{node.module}.{alias.name}"
                    )
            elif isinstance(node, ast.Call):
                dynamic_import = _literal_dynamic_import(node)
                if dynamic_import:
                    imported_apis.append(dynamic_import)
            for imported_api in imported_apis:
                top_level = imported_api.split(".", 1)[0]
                if top_level in sys.stdlib_module_names or top_level in local_modules:
                    continue
                reference = f"{relative_path}:{getattr(node, 'lineno', 0)}"
                external.setdefault(top_level, {}).setdefault(imported_api, set()).add(reference)
        alias_uses = _ModuleAliasUseCollector(
            tree,
            local_modules=local_modules,
            reference_path=relative_path,
        )
        alias_uses.visit(tree)
        for module, imported_apis in alias_uses.external.items():
            for imported_api, references in imported_apis.items():
                external.setdefault(module, {}).setdefault(imported_api, set()).update(
                    references
                )
    return {
        module: {
            imported_api: sorted(references)
            for imported_api, references in sorted(imported_apis.items())
        }
        for module, imported_apis in sorted(external.items())
    }


def _imported_api_is_declared(observed: str, declared: set[str]) -> bool:
    return observed in declared or any(api.startswith(observed + ".") for api in declared)


def _load_contract(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, [f"contract not found: {path}"]
    except json.JSONDecodeError as exc:
        return None, [f"contract is not valid JSON: {exc}"]
    if not isinstance(content, dict):
        return None, ["contract root must be an object"]
    return content, []


def _validate_python_matrix(repo_root: Path, contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    supported_versions = contract.get("supported_versions")
    if not isinstance(supported_versions, dict):
        return []
    versions = supported_versions.get("python")
    if not isinstance(versions, list) or not versions or not all(
        isinstance(version, str) and re.fullmatch(r"3\.\d+", version)
        for version in versions
    ):
        return ["supported_versions.python must be a non-empty list of Python 3.x minors"]
    matrix_path_value = contract.get("evidence", {}).get("python_ci_matrix")
    if not isinstance(matrix_path_value, str) or not matrix_path_value:
        return ["evidence.python_ci_matrix must name the CI workflow"]
    matrix_path = repo_root / matrix_path_value
    try:
        matrix = matrix_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [f"Python CI matrix file not found: {matrix_path_value}"]
    match = re.search(r"python-version:\s*\[([^\]]+)\]", matrix)
    if match is None:
        return [f"could not locate an inline python-version matrix in {matrix_path_value}"]
    matrix_versions = re.findall(r"[\"'](3\.\d+)[\"']", match.group(1))
    if matrix_versions != versions:
        errors.append(
            "supported Python versions do not match CI matrix: "
            f"contract={versions!r} ci={matrix_versions!r}"
        )
    return errors


def validate_contract(repo_root: Path, contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    contract_version = contract.get("contract_version")
    if not isinstance(contract_version, str) or not _CONTRACT_VERSION_RE.fullmatch(
        contract_version
    ):
        errors.append("contract_version must be a non-zero semantic version")
    if contract.get("boundary") != "host-owned":
        errors.append("boundary must be host-owned")

    ownership = contract.get("ownership")
    if not isinstance(ownership, dict):
        errors.append("ownership must be an object")
    else:
        for key in ("dependency_resolver", "update_owner", "update_trigger"):
            if not isinstance(ownership.get(key), str) or not ownership[key].strip():
                errors.append(f"ownership.{key} must be a non-empty string")

    supported_versions = contract.get("supported_versions")
    if not isinstance(supported_versions, dict):
        errors.append("supported_versions must be an object")
    else:
        host_versions = supported_versions.get("hermes_agent")
        if not isinstance(host_versions, str) or not _HOST_VERSION_RE.fullmatch(
            host_versions
        ):
            errors.append("supported_versions.hermes_agent must use >=major.minor,<major")
        policy = supported_versions.get("optional_dependency_policy")
        if not isinstance(policy, str) or not policy.strip():
            errors.append("supported_versions.optional_dependency_policy is required")
    errors.extend(_validate_python_matrix(repo_root, contract))

    runtime_scan = contract.get("runtime_scan")
    if not isinstance(runtime_scan, dict):
        return [*errors, "runtime_scan must be an object"]
    globs = runtime_scan.get("globs")
    local_imports = runtime_scan.get("local_imports")
    if not isinstance(globs, list) or not globs or not all(
        isinstance(pattern, str) and pattern for pattern in globs
    ):
        errors.append("runtime_scan.globs must be a non-empty string list")
        globs = []
    if not isinstance(local_imports, list) or not all(
        isinstance(module, str) and module for module in local_imports
    ):
        errors.append("runtime_scan.local_imports must be a string list")
        local_imports = []

    declared = contract.get("external_imports")
    if not isinstance(declared, dict) or not declared:
        return [*errors, "external_imports must be a non-empty object"]
    for module, dependency in declared.items():
        if not isinstance(module, str) or not module:
            errors.append("external_imports keys must be non-empty module names")
            continue
        if not isinstance(dependency, dict):
            errors.append(f"external_imports.{module} must be an object")
            continue
        for key in ("distribution", "scope", "version_policy"):
            if not isinstance(dependency.get(key), str) or not dependency[key].strip():
                errors.append(f"external_imports.{module}.{key} must be non-empty")
        if dependency.get("availability") not in {"required", "optional"}:
            errors.append(
                f"external_imports.{module}.availability must be required or optional"
            )
        imported_api = dependency.get("imported_api")
        if not isinstance(imported_api, list) or not imported_api or not all(
            isinstance(api, str) and api for api in imported_api
        ):
            errors.append(
                f"external_imports.{module}.imported_api must be a non-empty string list"
            )

    if globs:
        observed = collect_external_imports(
            repo_root,
            globs,
            local_imports=set(local_imports),
        )
        undeclared = sorted(set(observed) - set(declared))
        stale = sorted(set(declared) - set(observed))
        for module in undeclared:
            errors.append(
                f"undeclared external import {module!r}: {', '.join(observed[module])}"
            )
        for module in stale:
            errors.append(f"declared external import {module!r} is not observed")
        observed_apis = collect_external_imported_apis(
            repo_root,
            globs,
            local_imports=set(local_imports),
        )
        for module in sorted(set(observed_apis) & set(declared)):
            dependency = declared[module]
            if not isinstance(dependency, dict):
                continue
            imported_api = dependency.get("imported_api")
            if not isinstance(imported_api, list) or not all(
                isinstance(api, str) and api for api in imported_api
            ):
                continue
            declared_apis = set(imported_api)
            for observed_api, references in observed_apis[module].items():
                if _imported_api_is_declared(observed_api, declared_apis):
                    continue
                errors.append(
                    f"undeclared imported API {observed_api!r}: {', '.join(references)}"
                )
    return errors


def environment_report(contract: dict[str, Any]) -> list[dict[str, str]]:
    """Report local availability literally; this is not a vulnerability scan."""
    report: list[dict[str, str]] = []
    for module, dependency in sorted(contract["external_imports"].items()):
        distribution = dependency["distribution"]
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed-or-source-checkout"
        report.append(
            {
                "module": module,
                "distribution": distribution,
                "availability": dependency["availability"],
                "module_status": "available" if available else "unavailable",
                "installed_version": version,
            }
        )
    return report


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--report-environment",
        action="store_true",
        help="Report installed module/distribution versions without claiming CVE status.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    contract, errors = _load_contract(args.contract)
    if contract is not None:
        errors.extend(validate_contract(REPO_ROOT, contract))
    if errors:
        for error in errors:
            print(f"dependency contract error: {error}", file=sys.stderr)
        return 1
    assert contract is not None
    print(
        "dependency contract valid: "
        f"version {contract['contract_version']}; "
        f"{len(contract['external_imports'])} external imports declared; "
        f"{len(contract['supported_versions']['python'])} Python versions supported"
    )
    if args.report_environment:
        print("environment availability (not a CVE scan):")
        for item in environment_report(contract):
            print(json.dumps(item, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
