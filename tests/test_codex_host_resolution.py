"""Opt-in integration with a real Hermes checkout, never the installed plugin.

Run with HERMES_LCM_TEST_HOST=/path/to/hermes-agent and that host's Python.
Only HTTP responses are faked; config loading and resolver precedence are real.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    "model, override, live, raw, expected, probes",
    [
        ("gpt-5.6-sol", None, 272_000, 1_000_000, 272_000, 1),
        ("gpt-5.6-sol", None, 900_000, 1_000_000, 900_000, 1),
        ("gpt-5.6-sol", 1_000_000, 272_000, 1_000_000, 1_000_000, 0),
        ("gpt-5.6-sol", 128_000, 272_000, 1_000_000, 128_000, 0),
        ("openai/gpt-5.6-sol-900k", 800_000, 272_000, 1_000_000, 800_000, 0),
        ("gpt-5.6-sol", None, 900_000, 128_000, 128_000, 1),
        ("gpt-5.6-terra-900k", None, 272_000, 1_000_000, 900_000, 1),
        ("gpt-5.6-sol-900k", None, 272_000, 1_000_000, 900_000, 1),
        ("gpt-5.6-luna-900k", None, 272_000, 1_000_000, 900_000, 1),
        ("gpt-5.6-luna-900k", None, 128_000, 1_000_000, 128_000, 1),
    ],
)
def test_real_host_resolution(tmp_path, model, override, live, raw, expected, probes):
    host = os.environ.get("HERMES_LCM_TEST_HOST")
    if not host:
        pytest.skip("set HERMES_LCM_TEST_HOST to a real Hermes source checkout")
    host = Path(host).resolve()
    assert (host / "agent" / "model_metadata.py").is_file()
    plugin = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    hermes_home.mkdir(parents=True)
    config: dict[str, object] = {"plugins": {"enabled": []}}
    if override is not None:
        config["model_overrides"] = {"openai-codex": {model: {"context_window": override}}}
    # JSON is also YAML; no host imports or config reads in the parent process.
    (hermes_home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    env = {
        "HOME": str(home), "HERMES_HOME": str(hermes_home),
        "XDG_CACHE_HOME": str(home / ".cache"), "PYTHONPATH": str(host),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "PATH": os.defpath,
    }
    if os.name == "nt" and "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    script = textwrap.dedent('''
        import json
        import os
        from pathlib import Path
        import socket
        import sys
        from types import ModuleType, SimpleNamespace

        plugin, host, model, live, raw, expected, probes = json.loads(sys.argv[1])
        def deny_network(*args, **kwargs):
            raise AssertionError("unexpected network access in host integration test")
        socket.socket.connect = deny_network
        socket.create_connection = deny_network
        package = ModuleType("hermes_lcm")
        package.__path__ = [plugin]
        sys.modules["hermes_lcm"] = package

        import agent.model_metadata as metadata
        from hermes_lcm import codex_routing
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine
        assert Path(metadata.__file__).resolve() == Path(host) / "agent" / "model_metadata.py"
        assert Path(codex_routing.__file__).resolve() == Path(plugin) / "codex_routing.py"
        calls = []
        def fake_get(url, **kwargs):
            assert url == metadata.CODEX_MODELS_CATALOG_URL
            assert kwargs["headers"]["Authorization"] == "Bearer synthetic-route-token"
            calls.append(url)
            slug = model.rsplit("/", 1)[-1].removesuffix("-900k")
            return SimpleNamespace(status_code=200, json=lambda: {
                "models": [{"slug": slug, "context_window": live}]
            })
        metadata.requests.get = fake_get
        engine = LCMEngine(config=LCMConfig(
            database_path=str(Path(os.environ["HERMES_HOME"]) / "test.db")
        ))
        try:
            engine.update_model(
                model=model, provider="openai-codex", api_key="synthetic-route-token",
                base_url="https://custom.invalid/v1", context_length=raw,
            )
            assert engine.context_length == expected, (engine.context_length, expected)
            assert engine.raw_context_length == raw
            assert len(calls) == probes, calls
            print(json.dumps({"context": engine.context_length, "http_probes": len(calls)}))
        finally:
            engine.shutdown()
    ''')
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps([
            str(plugin), str(host), model, live, raw, expected, probes,
        ])],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"context": expected, "http_probes": probes}
