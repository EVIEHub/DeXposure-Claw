import pytest
from dexposure_agent.config import AgentConfig
from pydantic import ValidationError


def test_default_config():
    cfg = AgentConfig()
    assert cfg.tau_data == 0.7
    assert cfg.tau_conf == 0.6
    assert cfg.z_threshold == 2.0
    assert cfg.rolling_window == 42
    assert cfg.pi_min == 0.2
    assert cfg.lambda_tail == 0.5
    assert cfg.mc_samples == 50
    assert cfg.top_k == 5
    assert cfg.horizons == [1, 4, 8]


def test_custom_config():
    cfg = AgentConfig(tau_data=0.5, mc_samples=100)
    assert cfg.tau_data == 0.5
    assert cfg.mc_samples == 100


def test_invalid_tau_data():
    with pytest.raises(ValueError):
        AgentConfig(tau_data=1.5)


def test_invalid_tau_data_negative():
    with pytest.raises(ValueError):
        AgentConfig(tau_data=-0.1)


@pytest.mark.parametrize("horizons", [[], [0], [1, 1], [2, 4]])
def test_invalid_horizons(horizons):
    with pytest.raises(ValueError):
        AgentConfig(horizons=horizons)


def test_unknown_config_is_rejected():
    with pytest.raises(ValidationError):
        AgentConfig(unknown_knob=1)


def test_nonfinite_config_is_rejected():
    with pytest.raises(ValidationError):
        AgentConfig(tau_data=float("nan"))
