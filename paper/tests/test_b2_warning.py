import math

import pytest

from experiments.b2_early_warning import _f1_score, run_b2


def test_warning_f1_is_undefined_when_recall_is_undefined():
    assert math.isnan(_f1_score(0.0, float("nan")))


def test_warning_rejects_invalid_budget_before_loading_data(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        run_b2(
            "h1_weighted_degree",
            data_dir=str(tmp_path),
            alert_budgets=[0],
        )


def test_warning_rejects_silently_ignored_configuration(tmp_path):
    with pytest.raises(TypeError, match="unsupported configuration"):
        run_b2(
            "h1_weighted_degree",
            data_dir=str(tmp_path),
            unsupported_knob=1,
        )
