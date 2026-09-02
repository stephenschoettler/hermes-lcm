import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "dependency-contract.json"
ASSURANCE_DOC_PATH = REPO_ROOT / "docs" / "dependency-assurance.md"
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
    assert "dependency contract valid: version 1.0.3" in result.stdout
    assert "9 external imports declared" in result.stdout


def test_dependency_assurance_documentation_matches_contract_version():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assurance_doc = ASSURANCE_DOC_PATH.read_text(encoding="utf-8")

    assert f"`{contract['contract_version']}` supports:" in assurance_doc


def test_dependency_contract_validator_rejects_non_object_supported_versions(tmp_path):
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["supported_versions"] = ["3.11"]
    malformed_contract_path = tmp_path / "dependency-contract.json"
    malformed_contract_path.write_text(json.dumps(contract), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), "--contract", str(malformed_contract_path)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert (
        "dependency contract error: supported_versions must be an object"
        in result.stderr
    )
    assert "Traceback" not in result.stderr


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


def test_dependency_contract_validator_rejects_imported_api_drift(tmp_path, monkeypatch):
    validator = _load_validator()
    (tmp_path / "sample.py").write_text(
        "from fastembed import NewEmbedding\n",
        encoding="utf-8",
    )
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["runtime_scan"] = {
        "globs": ["*.py"],
        "local_imports": [],
        "excluded": [],
    }
    contract["external_imports"] = {
        "fastembed": contract["external_imports"]["fastembed"],
    }
    monkeypatch.setattr(validator, "_validate_python_matrix", lambda *_args: [])

    errors = validator.validate_contract(tmp_path, contract)

    assert errors == [
        "undeclared imported API 'fastembed.NewEmbedding': sample.py:1"
    ]


def _validate_sample_imports(tmp_path, monkeypatch, source, *modules):
    validator = _load_validator()
    (tmp_path / "sample.py").write_text(source, encoding="utf-8")
    repository_contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract = {
        **repository_contract,
        "runtime_scan": {
            "globs": ["*.py"],
            "local_imports": [],
            "excluded": [],
        },
        "external_imports": {
            module: repository_contract["external_imports"][module]
            for module in modules
        },
    }
    monkeypatch.setattr(validator, "_validate_python_matrix", lambda *_args: [])
    return validator.validate_contract(tmp_path, contract)


def test_dependency_contract_accepts_declared_module_alias_attribute_uses(
    tmp_path,
    monkeypatch,
):
    errors = _validate_sample_imports(
        tmp_path,
        monkeypatch,
        "import numpy as _np\n"
        "import yaml\n"
        "_np.asarray([1, 2, 3])\n"
        "yaml.safe_load('value: 1')\n",
        "numpy",
        "yaml",
    )

    assert errors == []


@pytest.mark.parametrize(
    "source",
    [
        "from agent import context_engine as ce\nce.ContextEngine()\n",
        "from agent import context_engine\ncontext_engine.ContextEngine()\n",
    ],
)
def test_dependency_contract_accepts_declared_direct_import_attribute_uses(
    tmp_path,
    monkeypatch,
    source,
):
    errors = _validate_sample_imports(
        tmp_path,
        monkeypatch,
        source,
        "agent",
    )

    assert errors == []


@pytest.mark.parametrize(
    ("source", "module", "expected_api", "line"),
    [
        ("import numpy as _np\n_np.matrix([1])\n", "numpy", "numpy.matrix", 2),
        (
            "import numpy as _np\n_np.linalg.norm([1])\n",
            "numpy",
            "numpy.linalg.norm",
            2,
        ),
        (
            "import tiktoken as tokenizer\n"
            "tokenizer.encoding_for_model('unsupported')\n",
            "tiktoken",
            "tiktoken.encoding_for_model",
            2,
        ),
        (
            "try:\n"
            "    import yaml\n"
            "except Exception:\n"
            "    yaml = None\n"
            "yaml.unsafe_load('value: 1')\n",
            "yaml",
            "yaml.unsafe_load",
            5,
        ),
    ],
)
def test_dependency_contract_rejects_undeclared_module_alias_attribute_uses(
    tmp_path,
    monkeypatch,
    source,
    module,
    expected_api,
    line,
):
    errors = _validate_sample_imports(
        tmp_path,
        monkeypatch,
        source,
        module,
    )

    assert errors == [
        f"undeclared imported API {expected_api!r}: sample.py:{line}"
    ]


@pytest.mark.parametrize(
    ("source", "module", "expected_api", "line"),
    [
        (
            "from agent import context_engine as ce\nce.Unsupported()\n",
            "agent",
            "agent.context_engine.Unsupported",
            2,
        ),
        (
            "from agent import context_engine\ncontext_engine.Unsupported()\n",
            "agent",
            "agent.context_engine.Unsupported",
            2,
        ),
        (
            "from agent import context_engine as ce\nce.ContextEngine.unsupported()\n",
            "agent",
            "agent.context_engine.ContextEngine.unsupported",
            2,
        ),
        (
            "from fastembed import TextEmbedding as Embedding\n"
            "Embedding.unsupported()\n",
            "fastembed",
            "fastembed.TextEmbedding.unsupported",
            2,
        ),
        (
            "try:\n"
            "    from agent import context_engine as ce\n"
            "except ImportError:\n"
            "    ce = None\n"
            "ce.Unsupported()\n",
            "agent",
            "agent.context_engine.Unsupported",
            5,
        ),
    ],
)
def test_dependency_contract_rejects_undeclared_direct_import_attribute_uses(
    tmp_path,
    monkeypatch,
    source,
    module,
    expected_api,
    line,
):
    errors = _validate_sample_imports(
        tmp_path,
        monkeypatch,
        source,
        module,
    )

    assert errors == [
        f"undeclared imported API {expected_api!r}: sample.py:{line}"
    ]


def test_dependency_contract_does_not_treat_shadowed_aliases_as_module_uses(
    tmp_path,
    monkeypatch,
):
    errors = _validate_sample_imports(
        tmp_path,
        monkeypatch,
        "import numpy as _np\n"
        "def local_value(_np):\n"
        "    return _np.unsupported_local_attribute()\n"
        "_np.asarray([1, 2, 3])\n",
        "numpy",
    )

    assert errors == []


def test_dependency_contract_does_not_treat_rebound_module_name_as_api_use(
    tmp_path,
    monkeypatch,
):
    errors = _validate_sample_imports(
        tmp_path,
        monkeypatch,
        "import yaml\n"
        "yaml = object()\n"
        "yaml.unsupported_local_attribute()\n",
        "yaml",
    )

    assert errors == []


@pytest.mark.parametrize(
    "source",
    [
        "from agent import context_engine as ce\n"
        "def local_value(ce):\n"
        "    return ce.unsupported_local_attribute()\n"
        "ce.ContextEngine()\n",
        "from agent import context_engine as ce\n"
        "ce = object()\n"
        "ce.unsupported_local_attribute()\n",
    ],
)
def test_dependency_contract_does_not_treat_shadowed_or_rebound_direct_imports_as_api_uses(
    tmp_path,
    monkeypatch,
    source,
):
    errors = _validate_sample_imports(
        tmp_path,
        monkeypatch,
        source,
        "agent",
    )

    assert errors == []


def test_dependency_contract_does_not_treat_stdlib_direct_import_as_api_use(
    tmp_path,
    monkeypatch,
):
    errors = _validate_sample_imports(
        tmp_path,
        monkeypatch,
        "from pathlib import Path as LocalPath\n"
        "LocalPath.unsupported_local_attribute()\n"
        "import yaml\n"
        "yaml.safe_load('value: 1')\n",
        "yaml",
    )

    assert errors == []


def test_dependency_contract_does_not_treat_local_direct_import_as_api_use(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "localmod.py").write_text("class Client:\n    pass\n", encoding="utf-8")

    errors = _validate_sample_imports(
        tmp_path,
        monkeypatch,
        "from localmod import Client as LocalClient\n"
        "LocalClient.unsupported_local_attribute()\n"
        "import yaml\n"
        "yaml.safe_load('value: 1')\n",
        "yaml",
    )

    assert errors == []


def test_contract_records_host_ownership_versions_and_update_owner():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["schema_version"] == 1
    assert contract["contract_version"] == "1.0.3"
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
    assert {
        "regex.compile",
        "regex.DOTALL",
        "regex.IGNORECASE",
        "regex.MULTILINE",
        "regex.VERBOSE",
        "regex.error",
    }.issubset(contract["external_imports"]["regex"]["imported_api"])
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
