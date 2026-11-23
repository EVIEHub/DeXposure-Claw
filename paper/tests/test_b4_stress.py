from __future__ import annotations

import math

import pytest

from experiments.b4_stress_test import _jaccard, run_b4


def test_empty_target_overlap_is_undefined():
    assert math.isnan(_jaccard(set(), set()))


def test_nonempty_target_overlap_is_jaccard():
    assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    ("scenarios", "top_k", "horizon"),
    [([], 10, 4), (["S1", "S1"], 10, 4), (["S9"], 10, 4), (["S1"], 0, 4), (["S1"], 10, 3)],
)
def test_invalid_b4_configuration_fails_before_data_loading(
    tmp_path, scenarios, top_k, horizon
):
    with pytest.raises(ValueError):
        run_b4(
            "m1_persistence_rules",
            data_dir=str(tmp_path / "missing"),
            scenarios=scenarios,
            top_k=top_k,
            horizon=horizon,
        )
