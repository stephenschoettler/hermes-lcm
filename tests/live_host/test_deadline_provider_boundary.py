"""Exercise real host imports and route preparation without a provider substitute."""

import importlib.util
from importlib._bootstrap import _get_module_lock
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


def _run_case(case, tmp_path):
    core = Path(os.environ["HERMES_LCM_HOST_ROOT"]).resolve()
    assert (core / "agent/auxiliary_client.py").is_file()
    allowed = {"PATH", "SYSTEMROOT", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "TEMP", "TMP", "HOME"}
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    environment.update(HERMES_HOME=str(tmp_path), HERMES_DISABLE_LAZY_INSTALLS="1")
    result = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).resolve()), case, str(core), str(tmp_path)],
        text=True, capture_output=True, env=environment, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["ok"] and receipt["provider_connect_attempts"] == 0
    assert receipt["spend_calls"] == receipt["reservations"] == receipt["backoff_until"] == 0
    assert receipt["route_failures"] == {}


@pytest.mark.skipif(not os.environ.get("HERMES_LCM_HOST_ROOT"), reason="requires a real Hermes host checkout")
def test_import_lock_expiry_does_not_start_or_charge_provider(tmp_path):
    _run_case("import", tmp_path)


@pytest.mark.skipif(not os.environ.get("HERMES_LCM_HOST_ROOT"), reason="requires a real Hermes host checkout")
def test_route_logging_lock_expiry_does_not_start_or_charge_provider(tmp_path):
    _run_case("route", tmp_path)


def _execute_case(case, core, directory):
    sys.path.insert(0, str(core))
    plugin = Path(__file__).resolve().parents[2]
    package = importlib.util.spec_from_file_location(
        "hermes_lcm", plugin / "__init__.py", submodule_search_locations=[str(plugin)],
    )
    module = importlib.util.module_from_spec(package)
    sys.modules["hermes_lcm"] = module
    package.loader.exec_module(module)
    from hermes_lcm import escalation, model_routing

    connects = []

    def audit(event, arguments):
        if event == "socket.connect":
            connects.append(str(arguments[1]))
            raise RuntimeError("An expired live-host check must never open a provider connection")

    sys.addaudithook(audit)
    ready = threading.Event()
    release = threading.Event()
    if case == "import":
        assert "agent.auxiliary_client" not in sys.modules
        lock = _get_module_lock("agent.auxiliary_client")
        handler = None
    elif case == "route":
        from agent.auxiliary_client import call_llm

        assert callable(call_llm)
        handler = logging.FileHandler(directory / "route.log", encoding="utf-8")
        lock = handler.lock
        prior_level = model_routing.logger.level
        model_routing.logger.setLevel(logging.DEBUG)
        model_routing.logger.addHandler(handler)
    else:
        raise ValueError(case)

    def hold_real_lock():
        lock.acquire()
        try:
            ready.set()
            assert release.wait(5), "The live-host request did not finish"
        finally:
            lock.release()

    thread = threading.Thread(target=hold_real_lock)
    thread.start()
    assert ready.wait(5)
    timer = threading.Timer(0.25, release.set)
    timer.start()
    guard = escalation.SummarySpendGuard(max_calls=1)
    breaker = escalation.SummaryCircuitBreaker()
    started = time.monotonic()
    try:
        result = escalation._invoke_summary_llm_chain(
            "Preserve the operation deadline when preparing this summary.", 32,
            model="openai/gpt-4.1-mini", timeout=0.05, deadline=started + 0.05,
            spend_guard=guard, circuit_breaker=breaker,
        )
    finally:
        release.set()
        timer.cancel()
        thread.join(timeout=5)
        assert not thread.is_alive()
        if handler is not None:
            model_routing.logger.removeHandler(handler)
            model_routing.logger.setLevel(prior_level)
            handler.close()
    assert result is None
    assert time.monotonic() - started >= 0.20
    receipt = {
        "ok": not connects and not guard._calls and not guard._reservations
              and guard._backoff_until == 0 and not breaker._failures,
        "case": case, "provider_connect_attempts": len(connects),
        "spend_calls": len(guard._calls), "reservations": len(guard._reservations),
        "backoff_until": guard._backoff_until, "route_failures": breaker._failures,
    }
    print(json.dumps(receipt))
    assert receipt["ok"], receipt


if __name__ == "__main__":
    _execute_case(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
