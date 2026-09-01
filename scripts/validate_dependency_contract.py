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
