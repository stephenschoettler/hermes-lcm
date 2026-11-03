"""Test configuration for hermes-lcm plugin tests.

Patches the plugin modules so they can be imported both as a package
(relative imports during plugin loading) and directly during testing.
"""
import importlib
import os
from pathlib import Path
import sys
import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_test_hermes_home(tmp_path_factory):
    """Ensure running tests never touches or pollutes the host ~/.hermes database."""
    temp_home = tmp_path_factory.mktemp("hermes_test_home")
    old_home = os.environ.get("HERMES_HOME")
    old_db = os.environ.get("LCM_DATABASE_PATH")
    os.environ["HERMES_HOME"] = str(temp_home)
    os.environ["LCM_DATABASE_PATH"] = str(temp_home / "lcm.db")
    yield temp_home
    if old_home is not None:
        os.environ["HERMES_HOME"] = old_home
    else:
        os.environ.pop("HERMES_HOME", None)
    if old_db is not None:
        os.environ["LCM_DATABASE_PATH"] = old_db
    else:
        os.environ.pop("LCM_DATABASE_PATH", None)


# Make the repo root importable (for agent.context_engine etc.)
repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if repo_root not in sys.path:
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
