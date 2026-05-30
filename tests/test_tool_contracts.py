"""Public lcm_* tool contract drift guards."""

import ast
import importlib
import inspect
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import hermes_lcm.schemas as schemas
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig


def _import_lcm_engine():
    try:
        return getattr(importlib.import_module("hermes_lcm.engine"), "LCMEngine")
    except ModuleNotFoundError as exc:
        if exc.name not in {"agent", "agent.context_engine"}:
            raise
        agent_module = sys.modules.get("agent")
        if agent_module is None:
            agent_module = ModuleType("agent")
            agent_module.__path__ = []
            sys.modules["agent"] = agent_module

        context_engine_module = ModuleType("agent.context_engine")

        class ContextEngine:
            pass

        setattr(context_engine_module, "ContextEngine", ContextEngine)
        sys.modules["agent.context_engine"] = context_engine_module
        setattr(agent_module, "context_engine", context_engine_module)
        sys.modules.pop("hermes_lcm.engine", None)
        return getattr(importlib.import_module("hermes_lcm.engine"), "LCMEngine")


LCMEngine = _import_lcm_engine()


REPO_ROOT = Path(__file__).resolve().parent.parent


def _manifest_tool_names() -> list[str]:
    lines = (REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8").splitlines()
    tool_names: list[str] = []
    in_provides_tools = False

    for line in lines:
        stripped = line.strip()
        if stripped == "provides_tools:":
            in_provides_tools = True
            continue
        if in_provides_tools:
            if not line.startswith("  "):
                break
            if stripped.startswith("- "):
                tool_names.append(stripped.removeprefix("- ").strip())

    return tool_names


def _schema_by_tool_name() -> dict[str, dict]:
    public_schemas = {}
    for symbol, value in vars(schemas).items():
        if not symbol.startswith("LCM_"):
            continue
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            continue
        name = value["name"]
        if not name.startswith("lcm_"):
            continue
        expected_symbol = name.upper()
        assert symbol == expected_symbol, f"{symbol} should be exported as {expected_symbol}"
        public_schemas[name] = value
    return public_schemas


def _engine_tool_schemas(tmp_path) -> list[dict]:
    config = LCMConfig(database_path=str(tmp_path / "contract.db"))
    engine = LCMEngine(config=config)
    try:
        return engine.get_tool_schemas()
    finally:
        engine.shutdown()


def _dispatch_tool_names() -> list[str]:
    source = textwrap.dedent(inspect.getsource(LCMEngine.handle_tool_call))
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "handlers" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            raise AssertionError("handle_tool_call handlers must be a dict literal")
        tool_names = []
        for key in node.value.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise AssertionError("handle_tool_call handler keys must be string literals")
            tool_names.append(key.value)
        return tool_names

    raise AssertionError("handle_tool_call must define a handlers dispatch map")


def _assert_unique(surface: str, tool_names: list[str]) -> None:
    duplicates = sorted({name for name in tool_names if tool_names.count(name) > 1})
    assert duplicates == [], f"duplicate tools in {surface}: {duplicates}"


def test_public_tool_names_are_synchronized_across_contract_surfaces(tmp_path):
    manifest_tools = _manifest_tool_names()
    schema_by_name = _schema_by_tool_name()
    engine_schemas = _engine_tool_schemas(tmp_path)
    engine_tools = [schema["name"] for schema in engine_schemas]
    dispatch_tools = _dispatch_tool_names()

    _assert_unique("plugin.yaml provides_tools", manifest_tools)
    _assert_unique("schemas.py", list(schema_by_name))
    _assert_unique("LCMEngine.get_tool_schemas", engine_tools)
    _assert_unique("LCMEngine.handle_tool_call", dispatch_tools)

    assert manifest_tools == engine_tools
    assert manifest_tools == dispatch_tools
    assert set(manifest_tools) == set(schema_by_name)

    engine_schema_by_name = {schema["name"]: schema for schema in engine_schemas}
    assert engine_schema_by_name == {name: schema_by_name[name] for name in manifest_tools}


def test_engine_dispatch_handles_every_declared_public_tool(tmp_path, monkeypatch):
    manifest_tools = _manifest_tool_names()
    config = LCMConfig(database_path=str(tmp_path / "dispatch.db"))
    engine = LCMEngine(config=config)
    calls = []

    def make_fake_handler(expected_tool_name):
        def fake_handler(args, *, engine):
            calls.append((expected_tool_name, args, engine))
            return f"handled:{expected_tool_name}"

        return fake_handler

    for tool_name in manifest_tools:
        monkeypatch.setattr(lcm_tools, tool_name, make_fake_handler(tool_name))

    try:
        for tool_name in manifest_tools:
            args = {"sentinel": tool_name}
            assert engine.handle_tool_call(tool_name, args) == f"handled:{tool_name}"
            assert calls[-1] == (tool_name, args, engine)

        unknown = engine.handle_tool_call("lcm_missing", {})
        assert "Unknown LCM tool: lcm_missing" in unknown
    finally:
        engine.shutdown()


def test_schema_module_exports_only_declared_public_lcm_tool_schemas():
    manifest_tools = _manifest_tool_names()
    schema_by_name = _schema_by_tool_name()

    assert set(schema_by_name) == set(manifest_tools)
    for tool_name in manifest_tools:
        schema = schema_by_name[tool_name]
        assert schema["name"] == tool_name
        assert schema["parameters"]["type"] == "object"
        assert isinstance(schema["parameters"].get("properties"), dict)
        assert isinstance(schema["parameters"].get("required"), list)
