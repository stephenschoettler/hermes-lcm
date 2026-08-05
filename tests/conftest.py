"""Test configuration for hermes-lcm plugin tests.

Patches the plugin modules so they can be imported both as a package
(relative imports during plugin loading) and directly during testing.
"""
import sys
import importlib
from pathlib import Path

# Find the Hermes agent module (agent.context_engine)
# Try common locations where the agent module might be
potential_roots = [
    str(Path(__file__).resolve().parent.parent.parent.parent),  # coder profile root
    str(Path(__file__).resolve().parent.parent.parent.parent.parent),  # one level up
    "/home/hermes/hermes-agent-qwen-pr",  # explicit hermes-agent location
    "/home/hermes/hermes-agent",  # alternative hermes-agent location
]

repo_root = None
for root in potential_roots:
    agent_module = Path(root) / "agent" / "context_engine.py"
    if agent_module.exists():
        repo_root = root
        break

if repo_root is None:
    # Fallback: try to find agent module by searching upward
    current = Path(__file__).resolve()
    for _ in range(6):  # Search up to 6 levels up
        parent = current.parent
        agent_check = parent / "agent" / "context_engine.py"
        if agent_check.exists():
            repo_root = str(parent)
            break
        current = parent

if repo_root and repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Register the plugin directory as a proper package
plugin_dir = Path(__file__).resolve().parent.parent
pkg_name = "hermes_lcm"

if pkg_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(plugin_dir / "__init__.py"),
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(plugin_dir)]
    mod.__package__ = pkg_name
    sys.modules[pkg_name] = mod
    # Don't exec the module (it tries to register with ctx)
    # Just make submodules importable

    # Register each submodule
    for py_file in plugin_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        sub_name = f"{pkg_name}.{py_file.stem}"
        if sub_name not in sys.modules:
            sub_spec = importlib.util.spec_from_file_location(
                sub_name, str(py_file),
                submodule_search_locations=[],
            )
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sub_mod.__package__ = pkg_name
            sys.modules[sub_name] = sub_mod
            setattr(mod, py_file.stem, sub_mod)
            try:
                sub_spec.loader.exec_module(sub_mod)
            except Exception:
                pass  # some modules may fail (e.g. engine needs agent)
