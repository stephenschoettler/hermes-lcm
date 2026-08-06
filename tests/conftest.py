"""Test configuration for hermes-lcm plugin tests.

Patches the plugin modules so they can be imported both as a package
(relative imports during plugin loading) and directly during testing.
"""
import os
import sys
import importlib
from pathlib import Path

# The standalone CI bootstrap places its hermes-agent stub in this checkout.
# A host integration run may explicitly point at a real hermes-agent checkout;
# never discover one by searching unrelated parent directories.
plugin_dir = Path(__file__).resolve().parent.parent
repo_root = Path(os.environ.get("HERMES_AGENT_ROOT", plugin_dir)).resolve()
if not (repo_root / "agent" / "context_engine.py").is_file():
    if "HERMES_AGENT_ROOT" in os.environ:
        raise RuntimeError(
            "HERMES_AGENT_ROOT must contain agent/context_engine.py: "
            f"{repo_root}"
        )
else:
    sys.path.insert(0, str(repo_root))

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
