import numpy as np
import pytest

from experiments.b3_uncertainty_calibration import (
    _compute_crps_sample,
    run_b3,
)


def test_crps_is_zero_for_perfect_point_samples():
    assert _compute_crps_sample(np.array([2.0, 2.0, 2.0]), 2.0) == 0.0


def test_calibration_rejects_invalid_coverage_before_loading_data(tmp_path):
    with pytest.raises(ValueError, match="strictly between"):
        run_b3(
            "m5_fm_rules",
            data_dir=str(tmp_path),
            target_coverage=1.0,
        )


def test_calibration_rejects_silently_ignored_configuration(tmp_path):
    with pytest.raises(TypeError, match="unsupported configuration"):
        run_b3(
            "m5_fm_rules",
            data_dir=str(tmp_path),
            unsupported_knob=1,
        )
