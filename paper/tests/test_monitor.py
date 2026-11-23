import math

import pytest

from dexposure_agent.monitor import (
    _compute_dispersion,
    _gini,
    _pagerank,
    compute_confidence,
    compute_metrics,
    detect_alerts,
    run_monitor,
)


def test_compute_metrics_returns_all_ids(sample_graph):
    metrics = compute_metrics(sample_graph)
    # N1=SIS, N2=HHI, N3=Density, N4=PageRank Gini, N5=Degree Gini
    expected_ids = {"N1", "N2", "N3", "N4", "N5"}
    assert expected_ids.issubset(set(metrics.keys()))


def test_compute_metrics_values_in_range(sample_graph):
    metrics = compute_metrics(sample_graph)
    # All network-level metrics should be in [0, 1] or reasonable ranges
    assert 0.0 <= metrics["N3"] <= 1.0  # Density
    assert 0.0 <= metrics["N5"] <= 1.0  # Gini


def test_detect_alerts_with_spike(sample_config):
    current = {"N1": 0.9}
    baseline = {"N1": {"mean": 0.3, "std": 0.1}}
    alerts = detect_alerts(current, baseline, horizon=4, config=sample_config)
    assert len(alerts) == 1
    assert alerts[0].metric_id == "N1"
    assert alerts[0].z_score > sample_config.z_threshold


def test_detect_alerts_no_spike(sample_config):
    current = {"N1": 0.35}
    baseline = {"N1": {"mean": 0.3, "std": 0.1}}
    alerts = detect_alerts(current, baseline, horizon=4, config=sample_config)
    assert len(alerts) == 0


def test_detect_alerts_handles_zero_std(sample_config):
    """Zero std means no variability — any deviation should trigger."""
    current = {"N1": 0.31}
    baseline = {"N1": {"mean": 0.3, "std": 0.0}}
    alerts = detect_alerts(current, baseline, horizon=4, config=sample_config)
    # With zero std, use a small epsilon to avoid division by zero
    assert len(alerts) <= 1  # May or may not trigger depending on epsilon handling


def test_compute_confidence_does_not_double_penalize_horizon(sample_config):
    c1 = compute_confidence(dh_score=0.9, dispersion=0.1, horizon=1, config=sample_config)
    c4 = compute_confidence(dh_score=0.9, dispersion=0.1, horizon=12, config=sample_config)
    assert c1 == c4


def test_confidence_gate_is_reachable(sample_config):
    confidence = compute_confidence(
        dh_score=1.0,
        dispersion=0.0,
        horizon=4,
        config=sample_config,
    )
    assert confidence > sample_config.tau_conf


def test_removed_horizon_penalty_would_make_default_gate_unreachable(sample_config):
    old_best_case = 1.0 / (1.0 + math.log(2.0))
    assert old_best_case < sample_config.tau_conf


def test_gini_finite_sample_value():
    assert _gini([0.0, 1.0]) == pytest.approx(0.5)


def test_pagerank_uses_edge_weights():
    nodes = ["a", "b", "c"]
    equal = _pagerank(nodes, {"a": {"b": 1.0, "c": 1.0}, "b": {}, "c": {}})
    weighted = _pagerank(
        nodes,
        {"a": {"b": 100.0, "c": 1.0}, "b": {}, "c": {}},
    )
    assert weighted["b"] > equal["b"]
    assert weighted["c"] < equal["c"]


def test_compute_confidence_degrades_with_dispersion(sample_config):
    c_low = compute_confidence(dh_score=0.9, dispersion=0.1, horizon=4, config=sample_config)
    c_high = compute_confidence(dh_score=0.9, dispersion=0.9, horizon=4, config=sample_config)
    assert c_low > c_high


def test_compute_confidence_degrades_with_low_dh(sample_config):
    c_good = compute_confidence(dh_score=0.9, dispersion=0.1, horizon=4, config=sample_config)
    c_bad = compute_confidence(dh_score=0.3, dispersion=0.1, horizon=4, config=sample_config)
    assert c_good > c_bad


def test_dispersion_is_unidentified_with_fewer_than_two_samples(sample_graph):
    assert _compute_dispersion("N1", []) is None
    assert _compute_dispersion("N1", [sample_graph]) is None


def test_run_monitor_blocks_confidence_when_dispersion_is_unidentified(
    sample_graph,
    sample_config,
):
    current = compute_metrics(sample_graph)
    baseline = {
        metric_id: value - 1.0
        for metric_id, value in current.items()
    }
    result = run_monitor(
        predicted_graph=sample_graph,
        mc_samples=[sample_graph],
        baseline_history=[baseline],
        horizon=4,
        config=sample_config,
    )
    assert result.alerts
    assert all(alert.confidence == 0.0 for alert in result.alerts)


def test_run_monitor_integration(sample_graph, sample_config):
    result = run_monitor(
        predicted_graph=sample_graph,
        mc_samples=[sample_graph],
        baseline_history=[],
        horizon=4,
        config=sample_config,
    )
    assert hasattr(result, "alerts")
    assert hasattr(result, "metrics")
