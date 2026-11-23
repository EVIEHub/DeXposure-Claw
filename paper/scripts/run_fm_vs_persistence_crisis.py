#!/usr/bin/env python3
"""FM vs Persistence on crisis periods.

Core question: Does DeXposure-FM outperform persistence during crises?

Experiments:
1. b1_forecast-crisis: Risk metric forecasting (MAE, RankCorr, TrendCons) on crisis periods
   - m5_fm_rules (FM) vs m1_persistence_rules (persistence) on Terra/Luna, FTX, SVB
2. b5_decision-crisis: Horizon-aligned decision quality
   - m5_fm_rules vs m1_persistence_rules on the same periods, reported separately by horizon

Usage:
    DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 python3 scripts/run_fm_vs_persistence_crisis.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats as scipy_stats

_SCRIPT_DIR = Path(__file__).resolve().parent
_AGENT_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _AGENT_DIR.parent
sys.path.insert(0, str(_AGENT_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from dexposure_agent.config import AgentConfig
from dexposure_agent.data_health import compute_data_health
from dexposure_agent.data_loader import SnapshotLoader
from dexposure_agent.decision import generate_tickets
from dexposure_agent.monitor import compute_metrics, run_monitor, _pagerank
from dexposure_agent.scenario import run_scenarios
from dexposure_agent.types import Edge, GraphSnapshot
from experiments.b3_uncertainty_calibration import _calibrate_mc_sigma
from experiments.predict_helper import predict_graph
from experiments.b5_decision_quality import _detect_truly_stressed_protocols
from experiments.exp_logger import json_safe

DATA_DIR = str(_REPO_ROOT / "data")
RESULTS_DIR = str(_AGENT_DIR / "results" / "run_fm_vs_persistence_crisis")

CRISIS_SPLITS = {
    "terra_luna": ("2022-04~2022-07", "Terra/Luna collapse"),
    "ftx":        ("2022-10~2023-01", "FTX collapse"),
    "svb":        ("2023-01~2023-05", "SVB/USDC depeg"),
    "calm_2025":  ("2025-01~2025-08", "2025 calm baseline"),
}
# Restrict to a subset via env, e.g. CRISIS_SPLIT_KEYS=terra_luna,ftx,svb
# (needed on the cloud box where the data subset ends in 2023-06).
_split_keys = os.environ.get("CRISIS_SPLIT_KEYS")
if _split_keys:
    _wanted = {k.strip() for k in _split_keys.split(",") if k.strip()}
    CRISIS_SPLITS = {k: v for k, v in CRISIS_SPLITS.items() if k in _wanted}

METHODS = ["m5_fm_rules", "m1_persistence_rules"]
HORIZONS = [
    int(value)
    for value in os.environ.get("DEXPOSURE_HORIZONS", "1,4,8").split(",")
    if value.strip()
]
METRIC_IDS = ["N1", "N2", "N3", "N4", "N5"]


def _frozen_config() -> AgentConfig:
    """Load the one FM threshold used by both crisis comparison arms."""
    raw_pi_min = os.environ.get("DEXPOSURE_FM_PI_MIN")
    if raw_pi_min is None or not raw_pi_min.strip():
        raise RuntimeError(
            "authoritative crisis comparison requires DEXPOSURE_FM_PI_MIN"
        )
    try:
        pi_min = float(raw_pi_min)
    except ValueError as exc:
        raise RuntimeError("DEXPOSURE_FM_PI_MIN must be a finite number") from exc
    if not np.isfinite(pi_min) or not 0.0 <= pi_min <= 1.0:
        raise RuntimeError("DEXPOSURE_FM_PI_MIN must be finite and within [0, 1]")
    return AgentConfig(horizons=HORIZONS, pi_min=pi_min)


# ──────────────────────────────────────────────────────────────────────────────
# b1_forecast-crisis: Risk metric forecasting
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class B1CrisisResult:
    method: str
    crisis: str
    horizon: int
    pagerank_mae: float = float("nan")
    hhi_mae: float = float("nan")
    density_mae: float = float("nan")
    gini_mae: float = float("nan")
    rank_correlation: float = float("nan")
    trend_consistency: float = float("nan")
    n_snapshots: int = 0
    pi_min: float = float("nan")
    # Per-week edge weight prediction error (FM-specific)
    mean_edge_weight_mae: float = float("nan")
    mean_new_edge_count: float = float("nan")


def _compute_edge_weight_mae(pred: GraphSnapshot, gt: GraphSnapshot) -> float:
    """Compute MAE of edge weights between predicted and ground truth graphs."""
    pred_edges = {(e.source, e.target): e.weight for e in pred.edges}
    gt_edges = {(e.source, e.target): e.weight for e in gt.edges}
    common = set(pred_edges.keys()) & set(gt_edges.keys())
    if not common:
        return float("nan")
    errors = [abs(pred_edges[k] - gt_edges[k]) for k in common]
    return float(np.mean(errors))


def _count_new_edges(pred: GraphSnapshot, current: GraphSnapshot) -> int:
    """Count edges in prediction that don't exist in current graph."""
    current_edges = {(e.source, e.target) for e in current.edges}
    pred_edges = {(e.source, e.target) for e in pred.edges}
    return len(pred_edges - current_edges)


def _future_snapshot(
    date: str,
    horizon: int,
    all_dates: list[str],
    date_to_snap: dict[str, GraphSnapshot],
) -> GraphSnapshot | None:
    """Return the ground-truth snapshot exactly ``horizon`` weeks after ``date``."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    try:
        future_idx = all_dates.index(date) + horizon
    except ValueError:
        return None
    if future_idx >= len(all_dates):
        return None
    return date_to_snap.get(all_dates[future_idx])


def run_b1_crisis() -> list[B1CrisisResult]:
    """Run b1_forecast risk forecasting on crisis and calm periods, comparing m5_fm_rules vs m1_persistence_rules."""
    loader = SnapshotLoader(data_dir=DATA_DIR)
    all_dates = loader.dates
    all_snaps = loader.load()
    date_to_snap = {s.date: s for s in all_snaps}
    config = _frozen_config()

    results = []

    for crisis_name, (test_split, desc) in CRISIS_SPLITS.items():
        logger.info(f"b1_forecast-crisis: {crisis_name} ({desc}) split={test_split}")
        test_snaps = loader.load(date_range=test_split)
        logger.info(f"  Loaded {len(test_snaps)} test snapshots")

        for method in METHODS:
            for h in HORIZONS:
                pr_errors, hhi_errors, density_errors, gini_errors = [], [], [], []
                rank_corrs = []
                edge_maes, new_edge_counts = [], []
                prev_pred_metrics, prev_gt_metrics = None, None
                trend_correct, trend_total = 0, 0
                n_valid = 0

                for snap_t in test_snaps:
                    gt = _future_snapshot(snap_t.date, h, all_dates, date_to_snap)
                    if gt is None:
                        continue

                    pred = predict_graph(method, snap_t, horizon=h, config=config)

                    # Compute network-level metrics
                    pred_m = compute_metrics(pred)
                    gt_m = compute_metrics(gt)

                    pr_errors.append(abs(pred_m["N1"] - gt_m["N1"]))
                    hhi_errors.append(abs(pred_m["N2"] - gt_m["N2"]))
                    density_errors.append(abs(pred_m["N3"] - gt_m["N3"]))
                    gini_errors.append(abs(pred_m["N5"] - gt_m["N5"]))

                    # Edge-level accuracy
                    edge_maes.append(_compute_edge_weight_mae(pred, gt))
                    new_edge_counts.append(_count_new_edges(pred, snap_t))

                    # Rank correlation (node-level PageRank)
                    pred_nodes = list(pred.nodes.keys())
                    gt_nodes = list(gt.nodes.keys())
                    pred_adj = {n: {} for n in pred_nodes}
                    for e in pred.edges:
                        if e.source in pred_adj:
                            pred_adj[e.source][e.target] = pred_adj[e.source].get(e.target, 0) + e.weight
                    gt_adj = {n: {} for n in gt_nodes}
                    for e in gt.edges:
                        if e.source in gt_adj:
                            gt_adj[e.source][e.target] = gt_adj[e.source].get(e.target, 0) + e.weight
                    pred_pr = _pagerank(pred_nodes, pred_adj) if pred_nodes else {}
                    gt_pr = _pagerank(gt_nodes, gt_adj) if gt_nodes else {}

                    common_nodes = sorted(set(pred_pr.keys()) & set(gt_pr.keys()))
                    if len(common_nodes) >= 3:
                        rho, _ = scipy_stats.spearmanr(
                            [pred_pr[n] for n in common_nodes],
                            [gt_pr[n] for n in common_nodes]
                        )
                        if not np.isnan(rho):
                            rank_corrs.append(float(rho))

                    # Trend consistency
                    if prev_pred_metrics is not None and prev_gt_metrics is not None:
                        for mid in METRIC_IDS:
                            pd_delta = pred_m.get(mid, 0) - prev_pred_metrics.get(mid, 0)
                            gd_delta = gt_m.get(mid, 0) - prev_gt_metrics.get(mid, 0)
                            if (pd_delta >= 0) == (gd_delta >= 0):
                                trend_correct += 1
                            trend_total += 1

                    prev_pred_metrics = pred_m
                    prev_gt_metrics = gt_m
                    n_valid += 1

                valid_edge_maes = [x for x in edge_maes if not np.isnan(x)]

                r = B1CrisisResult(
                    method=method,
                    crisis=crisis_name,
                    horizon=h,
                    pagerank_mae=float(np.mean(pr_errors)) if pr_errors else float("nan"),
                    hhi_mae=float(np.mean(hhi_errors)) if hhi_errors else float("nan"),
                    density_mae=float(np.mean(density_errors)) if density_errors else float("nan"),
                    gini_mae=float(np.mean(gini_errors)) if gini_errors else float("nan"),
                    rank_correlation=float(np.mean(rank_corrs)) if rank_corrs else float("nan"),
                    trend_consistency=trend_correct / trend_total if trend_total > 0 else float("nan"),
                    n_snapshots=n_valid,
                    pi_min=config.pi_min,
                    mean_edge_weight_mae=float(np.mean(valid_edge_maes)) if valid_edge_maes else float("nan"),
                    mean_new_edge_count=float(np.mean(new_edge_counts)) if new_edge_counts else float("nan"),
                )
                results.append(r)
                logger.info(
                    f"  {method} h={h}: RankCorr={r.rank_correlation:.4f} "
                    f"TrendCons={r.trend_consistency:.4f} HHI_MAE={r.hhi_mae:.4f} "
                    f"EdgeMAE={r.mean_edge_weight_mae:.4f} NewEdges={r.mean_new_edge_count:.1f} "
                    f"n={n_valid}"
                )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# b5_decision-crisis: Horizon-aligned decision quality
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class B5CrisisResult:
    method: str
    crisis: str
    prediction_horizon: int
    label_horizon: int
    mc_sigma: float
    n_weeks: int = 0
    n_alerts: int = 0
    ticket_precision: float = float("nan")
    false_intervention_rate: float = float("nan")
    target_stability: float = float("nan")
    audit_completeness: float = float("nan")
    n_tickets: int = 0
    n_intervention: int = 0
    pi_min: float = float("nan")


def _jaccard(a, b):
    if not a and not b:
        return float("nan")
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _mc_samples(graph, n, sigma, rng):
    samples = []
    for _ in range(n):
        edges = []
        for e in graph.edges:
            noise = rng.normal(0, sigma * abs(e.weight)) if e.weight else 0
            edges.append(Edge(source=e.source, target=e.target, weight=max(0, e.weight + noise)))
        samples.append(graph.model_copy(update={"edges": edges}))
    return samples


def run_b5_crisis() -> list[B5CrisisResult]:
    """Compare decision quality separately for each prediction/label horizon."""
    loader = SnapshotLoader(data_dir=DATA_DIR)
    all_dates = loader.dates
    all_snaps = loader.load()
    date_to_snap = {s.date: s for s in all_snaps}
    config = _frozen_config()

    results = []

    for crisis_name, (test_split, desc) in CRISIS_SPLITS.items():
        logger.info(f"b5_decision-crisis: {crisis_name} ({desc}) split={test_split}")
        test_pairs = list(loader.iter_test_with_baselines(test_split, baseline_window=config.rolling_window))
        logger.info(f"  Loaded {len(test_pairs)} test weeks")

        for h in HORIZONS:
            mc_sigma = _calibrate_mc_sigma(loader, test_split, horizon=h)
            seed_bytes = hashlib.sha256(f"{crisis_name}:{h}".encode()).digest()

            for method in METHODS:
                logger.info(f"  Running {method} at aligned horizon h={h}")
                t0 = time.time()
                # Match Monte Carlo draws across methods within a crisis/horizon pair.
                rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "big"))

                r = B5CrisisResult(
                    method=method,
                    crisis=crisis_name,
                    prediction_horizon=h,
                    label_horizon=h,
                    mc_sigma=mc_sigma,
                    pi_min=config.pi_min,
                )
                ticket_ok, false_intv, target_sets, audit_covs = [], [], [], []

                for snap_t, baseline_history in test_pairs:
                    gt = _future_snapshot(snap_t.date, h, all_dates, date_to_snap)
                    if gt is None:
                        continue

                    t_idx = all_dates.index(snap_t.date)
                    previous = date_to_snap.get(all_dates[t_idx - 1]) if t_idx > 0 else None
                    dh = compute_data_health(snap_t, config, previous_graph=previous)

                    pred = predict_graph(method, snap_t, horizon=h, config=config)
                    mc = _mc_samples(pred, config.mc_samples, mc_sigma, rng)
                    monitor = run_monitor(
                        predicted_graph=pred,
                        mc_samples=mc,
                        baseline_history=baseline_history,
                        horizon=h,
                        config=config,
                        dh_score=dh.score,
                    )
                    alerts = monitor.alerts
                    r.n_alerts += len(alerts)
                    scenario_summary = run_scenarios(pred, mc, config, horizon=h)
                    decision = generate_tickets(alerts, scenario_summary, dh, config)

                    targets, intv_targets = set(), set()
                    for ticket in decision.tickets:
                        targets.update(ticket.targets)
                        r.n_tickets += 1
                        if ticket.action in ("Recommend-Reduce", "Contingency"):
                            intv_targets.update(ticket.targets)
                            r.n_intervention += 1
                    target_sets.append(targets)

                    stressed = _detect_truly_stressed_protocols(snap_t, gt)
                    ticket_ok.extend(target in stressed for target in targets)
                    false_intv.extend(target not in stressed for target in intv_targets)
                    if stressed:
                        audit_covs.append(len(stressed & targets) / len(stressed))
                    r.n_weeks += 1

                r.ticket_precision = (
                    float(np.mean(ticket_ok)) if ticket_ok else float("nan")
                )
                r.false_intervention_rate = (
                    float(np.mean(false_intv)) if false_intv else float("nan")
                )
                r.audit_completeness = (
                    float(np.mean(audit_covs)) if audit_covs else float("nan")
                )

                stability = [
                    _jaccard(target_sets[i], target_sets[i - 1])
                    for i in range(1, len(target_sets))
                ]
                stability = [value for value in stability if np.isfinite(value)]
                r.target_stability = (
                    float(np.mean(stability)) if stability else float("nan")
                )

                elapsed = time.time() - t0
                logger.info(
                    f"  {method} h={h}: alerts={r.n_alerts} "
                    f"prec={r.ticket_precision:.3f} FIR={r.false_intervention_rate:.3f} "
                    f"audit={r.audit_completeness:.4f} "
                    f"stab={r.target_stability:.3f} sigma={mc_sigma:.4f} ({elapsed:.0f}s)"
                )
                results.append(r)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 65)
    logger.info("FM vs PERSISTENCE ON CRISIS PERIODS")
    logger.info("=" * 65)

    t_start = time.time()
    logger.info("Frozen FM pi_min={} (DEXPOSURE_FM_PI_MIN)", os.environ.get("DEXPOSURE_FM_PI_MIN"))

    # b1_forecast-crisis
    logger.info("\n" + "=" * 65)
    logger.info("PART 1: b1_forecast Risk Forecasting — m5_fm_rules vs m1_persistence_rules on crisis periods")
    logger.info("=" * 65)
    b1_results = run_b1_crisis()

    # b5_decision-crisis
    logger.info("\n" + "=" * 65)
    logger.info("PART 2: b5_decision Horizon-aligned Decision Quality — m5_fm_rules vs m1_persistence_rules")
    logger.info("=" * 65)
    b5_results = run_b5_crisis()

    # Save
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    b1_data = [asdict(r) for r in b1_results]
    (out_dir / "b1_crisis.json").write_text(
        json.dumps(json_safe(b1_data), indent=2, default=str, allow_nan=False)
    )

    b5_data = [asdict(r) for r in b5_results]
    (out_dir / "b5_crisis.json").write_text(
        json.dumps(json_safe(b5_data), indent=2, default=str, allow_nan=False)
    )

    total = time.time() - t_start
    logger.info(f"\nALL DONE in {total:.0f}s ({total/60:.1f} min)")

    # Print b1_forecast summary
    print("\n=== b1_forecast: FM vs Persistence on Crisis Periods ===")
    print(f"{'Crisis':<12} {'Method':<6} {'h':>3} {'RankCorr':>9} {'TrendCons':>10} {'HHI_MAE':>8} {'EdgeMAE':>8} {'NewEdge':>8} {'n':>3}")
    print("-" * 75)
    for r in b1_results:
        print(f"{r.crisis:<12} {r.method:<6} {r.horizon:>3} {r.rank_correlation:>9.4f} "
              f"{r.trend_consistency:>10.4f} {r.hhi_mae:>8.4f} "
              f"{r.mean_edge_weight_mae:>8.4f} {r.mean_new_edge_count:>8.1f} {r.n_snapshots:>3}")

    # Print b5_decision summary
    print("\n=== b5_decision: Horizon-aligned Decision Quality on Crisis ===")
    print(f"{'Crisis':<12} {'Method':<6} {'h':>3} {'Alerts':>7} {'Prec':>6} {'FIR':>6} {'Audit':>7} {'Stab':>6} {'#Intv':>6}")
    print("-" * 66)
    for r in b5_results:
        print(f"{r.crisis:<12} {r.method:<6} {r.prediction_horizon:>3} "
              f"{r.n_alerts:>7} {r.ticket_precision:>6.3f} "
              f"{r.false_intervention_rate:>6.3f} {r.audit_completeness:>7.4f} "
              f"{r.target_stability:>6.3f} {r.n_intervention:>6}")


if __name__ == "__main__":
    main()
