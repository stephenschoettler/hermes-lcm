import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "dependency-contract.json"
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_dependency_contract.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "dependency_contract_validator",
        VALIDATOR_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependency_contract_validator_accepts_repository():
    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "dependency contract valid: version 1.0.1" in result.stdout
    assert "9 external imports declared" in result.stdout


def test_dependency_scanner_fails_closed_on_static_and_literal_dynamic_imports(tmp_path):
    validator = _load_validator()
    (tmp_path / "sample.py").write_text(
        "import importlib\n"
        "import requests\n"
        "importlib.import_module('newpkg.api')\n",
        encoding="utf-8",
    )

    imports = validator.collect_external_imports(
        tmp_path,
        ["*.py"],
        local_imports=set(),
    )

    assert {"newpkg", "requests"}.issubset(imports)


def test_dependency_scanner_does_not_treat_generated_host_stub_as_local(tmp_path):
    validator = _load_validator()
    (tmp_path / "sample.py").write_text(
        "from agent.context_engine import ContextEngine\n",
        encoding="utf-8",
    )
    agent_stub = tmp_path / "agent"
    agent_stub.mkdir()
    (agent_stub / "__init__.py").write_text("", encoding="utf-8")
    (agent_stub / "context_engine.py").write_text(
        "class ContextEngine:\n    pass\n",
        encoding="utf-8",
    )

    imports = validator.collect_external_imports(
        tmp_path,
        ["*.py"],
        local_imports=set(),
    )

    assert imports["agent"] == ["sample.py:1"]


def test_contract_records_host_ownership_versions_and_update_owner():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["schema_version"] == 1
    assert contract["contract_version"] == "1.0.1"
    assert contract["boundary"] == "host-owned"
    assert contract["ownership"]["dependency_resolver"] == "Hermes Agent host environment"
    assert contract["ownership"]["update_owner"] == "Hermes-LCM maintainers"
    assert contract["supported_versions"]["python"] == ["3.11", "3.12", "3.13", "3.14"]
    assert contract["supported_versions"]["hermes_agent"] == ">=0.16,<1"
    assert contract["runtime_scan"]["local_imports"] == ["hermes_lcm", "benchmarking"]
    assert set(contract["external_imports"]) == {
        "agent",
        "fastembed",
        "gateway",
        "hermes_cli",
        "huggingface_hub",
        "numpy",
        "regex",
        "tiktoken",
        "yaml",
    }
    assert contract["external_imports"]["agent"]["availability"] == "required"
    assert all(
        dependency["version_policy"]
        for dependency in contract["external_imports"].values()
    )


def test_dependency_contract_validation_is_wired_into_ci_and_release_gate():
    ci_workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    release_validator = (
        REPO_ROOT / "scripts" / "validate_release.sh"
    ).read_text(encoding="utf-8")

    assert "python scripts/validate_dependency_contract.py" in ci_workflow
    assert (
        'run_gate "dependency contract" "$PYTHON_BIN" '
        "scripts/validate_dependency_contract.py --report-environment"
        in release_validator
    )
