import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.model_routing import apply_lcm_reasoning_effort


@pytest.mark.parametrize(
    "field,env_name,yaml_key",
    [
        ("summary_reasoning_effort", "LCM_SUMMARY_REASONING_EFFORT", "summary_reasoning_effort"),
        ("expansion_reasoning_effort", "LCM_EXPANSION_REASONING_EFFORT", "expansion_reasoning_effort"),
    ],
)
def test_reasoning_effort_yaml_env_precedence(tmp_path, monkeypatch, field, env_name, yaml_key):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"lcm:\n  {yaml_key}: low\n")
    assert getattr(LCMConfig.from_env(), field) == "low"
    monkeypatch.setenv(env_name, "high")
    config = LCMConfig.from_env()
    assert getattr(config, field) == "high"
    assert config.config_sources[field] == "env"


@pytest.mark.parametrize(
    "env_name", ["LCM_SUMMARY_REASONING_EFFORT", "LCM_EXPANSION_REASONING_EFFORT"]
)
def test_invalid_nonempty_reasoning_effort_fails_loudly(monkeypatch, env_name):
    monkeypatch.setenv(env_name, "turbo")
    with pytest.raises(ValueError, match="reasoning effort"):
        LCMConfig.from_env()


def test_direct_invalid_reasoning_effort_fails_loudly():
    with pytest.raises(ValueError, match="unsupported LCM reasoning effort"):
        apply_lcm_reasoning_effort({}, "turbo")


def test_empty_reasoning_effort_preserves_request_defaults():
    kwargs = {"extra_body": {"existing": True}}
    apply_lcm_reasoning_effort(kwargs, "")
    assert kwargs == {"extra_body": {"existing": True}}


@pytest.mark.parametrize(
    "effort,expected",
    [
        ("none", {"enabled": False, "effort": "none"}),
        ("minimal", {"effort": "minimal"}),
        ("xhigh", {"effort": "xhigh"}),
    ],
)
def test_reasoning_effort_request_payload(effort, expected):
    kwargs = {"extra_body": {"existing": True}}
    apply_lcm_reasoning_effort(kwargs, effort)
    assert kwargs == {
        "extra_body": {"existing": True},
        "reasoning_config": expected,
    }
