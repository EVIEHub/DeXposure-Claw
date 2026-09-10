import json
from types import SimpleNamespace

import pytest

from experiments.b5_decision_quality import (
    _bind_authoritative_pi_min,
    _jaccard,
    detect_truly_stressed_protocols,
    run_b5,
)
from experiments.b6_robustness import run_b6


def test_b5_rejects_silently_ignored_configuration(tmp_path):
    with pytest.raises(TypeError, match="unsupported configuration"):
        run_b5(
            "m1_persistence_rules",
            data_dir=str(tmp_path),
            unsupported_knob=1,
        )


def test_b5_empty_target_stability_is_undefined():
    assert _jaccard(set(), set()) != _jaccard(set(), set())


def test_ground_truth_uses_active_denominator_and_positive_ranked_drops():
    current = {
        "nodes": {f"p{i}": {} for i in range(100)},
        "edges": [
            {"source": f"p{i}", "target": f"p{i}", "weight": 1.0}
            for i in range(100)
        ],
    }
    future = {
        "nodes": {f"p{i}": {} for i in range(100)},
        "edges": [
            {"source": f"p{i}", "target": f"p{i}", "weight": 0.0 if i < 20 else 1.0}
            for i in range(100)
        ],
    }
    assert detect_truly_stressed_protocols(current, future, pct=0.05) == {
        "p0",
        "p1",
        "p10",
        "p11",
        "p12",
    }


def test_ground_truth_skips_malformed_or_unknown_edges():
    current = {
        "nodes": {"known": {}},
        "edges": [
            {"source": "known", "target": "known", "weight": 10.0},
            {"source": None, "target": "known", "weight": 100.0},
            {"source": "unknown", "target": "known", "weight": 100.0},
        ],
    }
    future = {"nodes": {"known": {}}, "edges": []}
    assert detect_truly_stressed_protocols(current, future, pct=0.05) == {"known"}


def test_b5_aligns_prediction_label_uncertainty_and_artifacts(
    monkeypatch,
    tmp_path,
    sample_graph,
):
    from experiments import b3_uncertainty_calibration, b5_decision_quality
    from experiments import predict_helper

    dates = [f"2025-01-{day:02d}" for day in range(1, 11)]
    snapshots = {date: sample_graph.model_copy(update={"date": date}) for date in dates}

    class FakeLoader:
        def __init__(self, data_dir):
            self.dates = dates

        def iter_test_with_baselines(self, test_split, baseline_window):
            baseline = [{"N1": 0.0}]
            yield snapshots[dates[0]], baseline
            yield snapshots[dates[1]], baseline

        def load(self):
            return list(snapshots.values())

        def load_single(self, date):
            return snapshots[date]

    observed: dict[str, list[int]] = {
        "calibration": [],
        "prediction": [],
        "monitor": [],
        "scenario": [],
    }

    def fake_calibrate(loader, test_split, horizon):
        observed["calibration"].append(horizon)
        return 0.1

    def fake_predict(method_id, graph, horizon, config):
        observed["prediction"].append(horizon)
        return graph

    def fake_monitor(**kwargs):
        observed["monitor"].append(kwargs["horizon"])
        return SimpleNamespace(alerts=[])

    def fake_scenario(pred_graph, mc_samples, config, horizon):
        observed["scenario"].append(horizon)
        return SimpleNamespace(ranked_losses=[])

    monkeypatch.setattr(b5_decision_quality, "SnapshotLoader", FakeLoader)
    monkeypatch.setattr(
        b3_uncertainty_calibration,
        "_calibrate_mc_sigma",
        fake_calibrate,
    )
    monkeypatch.setattr(predict_helper, "predict_graph", fake_predict)
    monkeypatch.setattr(b5_decision_quality, "run_monitor", fake_monitor)
    monkeypatch.setattr(
        b5_decision_quality,
        "_build_scenario_summary",
        fake_scenario,
    )

    results = run_b5(
        "m1_persistence_rules",
        data_dir="unused",
        test_split="2025-01~2025-01",
        horizon=8,
        results_dir=str(tmp_path),
        mc_samples=2,
    )

    assert observed == {
        "calibration": [8],
        "prediction": [8, 8],
        "monitor": [8, 8],
        "scenario": [8, 8],
    }
    assert results[0].prediction_horizon == 8
    assert results[0].label_horizon == 8
    assert (tmp_path / "b5_decision_h8__m1_persistence_rules.json").is_file()
    weekly_path = tmp_path / "b5_weekly_h8__m1_persistence_rules.json"
    weekly = json.loads(weekly_path.read_text())
    assert weekly["horizon"] == 8
    assert weekly["prediction_horizon"] == weekly["label_horizon"] == 8
    assert all(
        week["prediction_horizon"] == week["label_horizon"] == 8
        for week in weekly["weeks"]
    )


@pytest.mark.parametrize("horizon", [0, 2, 4.0, 16, True])
def test_b5_rejects_unsupported_horizon_before_loading_data(tmp_path, horizon):
    with pytest.raises(ValueError, match="horizon must be one of"):
        run_b5(
            "m1_persistence_rules",
            data_dir=str(tmp_path),
            horizon=horizon,
        )


def test_b5_authoritative_pi_min_is_required_and_bound(monkeypatch):
    monkeypatch.setenv("DEXPOSURE_AUTHORITATIVE", "1")
    monkeypatch.setenv("DEXPOSURE_FM_PI_MIN", "0.35")
    assert _bind_authoritative_pi_min({}) == {"pi_min": 0.35}
    with pytest.raises(ValueError, match="conflicts"):
        _bind_authoritative_pi_min({"pi_min": 0.2})


def test_b5_authoritative_pi_min_missing_fails_closed(monkeypatch):
    monkeypatch.setenv("DEXPOSURE_AUTHORITATIVE", "1")
    monkeypatch.delenv("DEXPOSURE_FM_PI_MIN", raising=False)
    with pytest.raises(ValueError, match="DEXPOSURE_FM_PI_MIN"):
        _bind_authoritative_pi_min({})


def test_b6_rejects_unknown_regime_before_loading_data(tmp_path):
    with pytest.raises(ValueError, match="Unknown robustness regimes"):
        run_b6(
            "m1_persistence_rules",
            data_dir=str(tmp_path),
            regimes=["not_implemented"],
        )


def test_b6_rejects_silently_ignored_configuration(tmp_path):
    with pytest.raises(TypeError, match="unsupported configuration"):
        run_b6(
            "m1_persistence_rules",
            data_dir=str(tmp_path),
            unsupported_knob=1,
        )
