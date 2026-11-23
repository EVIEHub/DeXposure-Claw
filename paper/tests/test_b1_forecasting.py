import pytest

from experiments.b1_risk_forecasting import run_b1


def test_forecasting_rejects_invalid_horizon_before_loading_data(tmp_path):
    with pytest.raises(ValueError, match="unique subset"):
        run_b1("m1_persistence_rules", data_dir=str(tmp_path), horizons=[1, 1])


def test_forecasting_rejects_silently_ignored_configuration(tmp_path):
    with pytest.raises(TypeError, match="unsupported configuration"):
        run_b1(
            "m1_persistence_rules",
            data_dir=str(tmp_path),
            unsupported_knob=1,
        )
