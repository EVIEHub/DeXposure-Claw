#!/usr/bin/env python3
"""Unified 29-Week Out-of-Sample Evaluation on DeXposure-FM v4 (End-to-End Constant-Price Net-Flow).

Computes and persists provenance-attested artifacts into `paper/results/run_20261123_fm_v4_unified/`:
  1. Upstream Forecast, Calibration, Stress, and Robustness (`RQ1`: `b1_forecast`, `b3_calibration`,
     `b4_stress`, `b6_robustness`) comparing `m1_persistence_rules` vs `m5_fm_rules` (`DeXposure-FM v4`).
  2. Deterministic Rules Baselines (`m1_persistence_rules`, `m5_fm_rules`) and `RQ3` Clean & Degraded-Input
     Control Ablations (`full m5`, `A3 no scenario`, `A2 no confidence gate`, and 4 masking regimes).
  3. Controlled Same-LLM Ticket Evaluation (`m2_snapshot_llm`, `m6_fm_llm`, `m7_fm_llm_gated`) using
     `deepseek/deepseek-v3.2` (temperature=0, 3 drafts/week, exact K=7 schema), with crash-resumable
     per-week checkpoints, explicit Layer 4 gate states, 10,000-resample bootstrap CIs, and 20,000-replicate
     paired sign-flip permutation p-values across review budgets k in {1, 3, 5, 7}.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper"
for _p in (str(ROOT), str(PAPER_ROOT), str(ROOT / "cloud" / "train_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dexposure_agent.config import AgentConfig  # noqa: E402
from dexposure_agent.data_health import compute_data_health  # noqa: E402
from dexposure_agent.decision import generate_tickets  # noqa: E402
from dexposure_agent.monitor import (  # noqa: E402
    compute_metrics as compute_monitor_metrics,
    run_monitor,
)
from dexposure_agent.scenario import (  # noqa: E402
    run_scenarios as run_agent_scenarios,
)
from dexposure_agent.types import Edge, GraphSnapshot, NodeFeatures, ScenarioSummary  # noqa: E402
from experiments.b1_risk_forecasting import (  # noqa: E402
    _compute_pagerank_vector,
    _pagerank_mae,
    _rank_correlation,
    _trend_consistency,
)
from experiments.b3_uncertainty_calibration import (  # noqa: E402
    _compute_crps_sample,
    _compute_ece,
    _generate_mc_samples,
)
from experiments.b5_decision_quality import (  # noqa: E402
    CandidateSet,
    EvidenceBundle,
    _ticket_target_score_map,
)
from experiments.b6_robustness import REGIME_CONFIGS, REGIMES, _apply_degradation  # noqa: E402
import experiments.llm_eval_b5 as llm_mod  # noqa: E402

NON_SUPERVISED_CATEGORIES = frozenset({"CEX", "Risk Curators", "Chain", "Oracle", "Services"})
DEFAULT_OUT_DIR = PAPER_ROOT / "results" / "run_20261123_fm_v4_unified"
DEFAULT_DECISION_MODEL = "deepseek/deepseek-v3.2"
HORIZON = 4
STRESS_PCT = 0.05
N_BOOT = 10_000
N_PERM = 20_000


def load_openrouter_key_from_dotenv() -> str:
    val = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if val:
        return val
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("OPENROUTER_API_KEY="):
                raw = s.split("=", 1)[1].strip().strip('"').strip("'")
                if raw:
                    os.environ["OPENROUTER_API_KEY"] = raw
                    return raw
    return ""


def incident_exposure(gs: GraphSnapshot) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for e in gs.edges:
        if e.weight > 0.0:
            out[e.source] += float(e.weight)
            out[e.target] += float(e.weight)
    return out


def eligible_protocols(gs: GraphSnapshot, inc_map: dict[str, float] | None = None) -> set[str]:
    inc = inc_map if inc_map is not None else incident_exposure(gs)
    return {
        nid
        for nid, nf in gs.nodes.items()
        if nid.isdigit()
        and float(nf.log_size or 0.0) > 0.0
        and nf.category not in NON_SUPERVISED_CATEGORIES
        and inc.get(nid, 0.0) > 0.0
    }


def compute_netflow_labels(
    curr_gs: GraphSnapshot, fut_const_gs: GraphSnapshot, pct: float = STRESS_PCT
) -> tuple[set[str], dict[str, float], set[str], int]:
    """Compute constant-price net-flow stressed protocols (top-5% positive net outflow)."""
    inc_c = incident_exposure(curr_gs)
    inc_f = incident_exposure(fut_const_gs)
    elig = eligible_protocols(curr_gs, inc_c)
    declines: list[tuple[float, str]] = []
    actual_loss_frac: dict[str, float] = {}
    pos_loss: set[str] = set()
    for nid in elig:
        w0 = inc_c[nid]
        w1 = inc_f.get(nid, 0.0)
        diff = w0 - w1
        actual_loss_frac[nid] = max(0.0, diff / w0) if w0 > 0.0 else 0.0
        if diff > 0.0:
            declines.append((diff, nid))
            pos_loss.add(nid)
    declines.sort(key=lambda x: (-x[0], x[1]))
    k_st = max(1, math.ceil(pct * len(elig)))
    stressed = {nid for _, nid in declines[:k_st]}
    return stressed, actual_loss_frac, pos_loss, len(elig)


def prepare_pred_graph_for_v4(
    curr_gs: GraphSnapshot,
    pred_const_gs: GraphSnapshot,
    node_deltas: dict[str, float],
    edge_probs: dict[tuple[str, str], float],
    val_dtvl_shift: float = 0.0,
    pi_min: float = 0.2,
) -> GraphSnapshot:
    """Construct expected constant-price graph using validation-calibrated node net-flow contractions."""
    new_nodes: dict[str, NodeFeatures] = {}
    node_surv: dict[str, float] = {}
    for nid, nf in pred_const_gs.nodes.items():
        orig_nf = curr_gs.nodes[nid]
        if float(orig_nf.log_size or 0.0) > 0.0 and nid.isdigit():
            dtvl = float(node_deltas.get(nid, 0.0)) + float(val_dtvl_shift)
            eff_dtvl = min(0.0, dtvl)
            surv = math.exp(max(-10.0, eff_dtvl))
            node_surv[nid] = surv
            new_nodes[nid] = nf.model_copy(update={"prev_tvl_change": eff_dtvl})
        else:
            node_surv[nid] = 1.0
            new_nodes[nid] = nf.model_copy(update={"prev_tvl_change": 0.0})

    new_edges: list[Edge] = []
    for e in curr_gs.edges:
        p_r = float(edge_probs.get((e.source, e.target), 1.0))
        if p_r < pi_min:
            continue
        s_u = node_surv.get(e.source, 1.0)
        s_v = node_surv.get(e.target, 1.0)
        w_pred = float(e.weight) * min(s_u, s_v)
        if w_pred > 1.0:
            new_edges.append(Edge(source=e.source, target=e.target, weight=w_pred))

    return GraphSnapshot(date=curr_gs.date, nodes=new_nodes, edges=new_edges)


def build_v4_candidate_sets(
    curr_gs: GraphSnapshot,
    pred_gs: GraphSnapshot,
    prompt_scenarios: list[dict[str, Any]],
) -> tuple[CandidateSet, CandidateSet, dict[str, Any], dict[str, Any]]:
    """Build matched 10-protocol CandidateSets for M2 (snapshot) and M6/M7 (FM v4 + scenarios)."""
    inc_curr = incident_exposure(curr_gs)
    elig = eligible_protocols(curr_gs, inc_curr)

    curr_dict = curr_gs.model_dump()
    curr_dict["n_nodes"] = len(curr_gs.nodes)
    curr_dict["n_edges"] = len(curr_gs.edges)

    pred_dict = pred_gs.model_dump()
    pred_dict["horizon"] = HORIZON
    pred_dict["n_nodes"] = len(pred_gs.nodes)
    pred_dict["n_edges"] = len(pred_gs.edges)
    fwd_scores = llm_mod._forward_drawdown_scores(pred_dict, inc_curr)

    scen_max: dict[str, float] = defaultdict(float)
    scen_sum: dict[str, float] = defaultdict(float)
    for sc in prompt_scenarios:
        for item in sc.get("top_affected", []):
            pid = str(item.get("protocol", "")).strip()
            loss = abs(float(item.get("loss") or 0.0))
            if pid in elig and math.isfinite(loss):
                scen_max[pid] = max(scen_max[pid], loss)
                scen_sum[pid] += loss

    m2_top10 = sorted(elig, key=lambda n: (-inc_curr[n], n))[:10]
    m6_top10 = sorted(elig, key=lambda n: (-fwd_scores.get(n, 0.0), -inc_curr[n], n))[:10]

    cset_m2 = CandidateSet(tuple((pid, inc_curr[pid], 0.0, 0.0) for pid in sorted(m2_top10)))
    cset_m6 = CandidateSet(
        tuple((pid, inc_curr[pid], scen_max.get(pid, 0.0), scen_sum.get(pid, 0.0)) for pid in sorted(m6_top10))
    )
    return cset_m2, cset_m6, curr_dict, pred_dict


def load_cached_v4_data(cache_pkl: Path) -> tuple[list[str], list[GraphSnapshot], list[dict[str, Any]], float]:
    sub_cache = cache_pkl.parent / "v4_sub_gs_2023_2025.pkl"
    val_cache = cache_pkl.parent / "v4_20w_pre2025_val_cache.pkl"
    if not sub_cache.is_file() or not cache_pkl.is_file() or not val_cache.is_file():
        raise FileNotFoundError(f"Expected cached V4 pickles at {sub_cache}, {val_cache}, and {cache_pkl}")
    sub_dates, ordered_sub_gs = pickle.loads(sub_cache.read_bytes())
    val_weeks = pickle.loads(val_cache.read_bytes())
    pred_val = np.array([
        w["v4_node_deltas"][nid]
        for w in val_weeks
        for nid in w["true_dtvl"]
        if nid in w["v4_node_deltas"]
    ], dtype=float)
    true_val = np.array([
        w["true_dtvl"][nid]
        for w in val_weeks
        for nid in w["true_dtvl"]
        if nid in w["v4_node_deltas"]
    ], dtype=float)
    # Closed-form pre-2025 validation median calibration removing asymmetric Smooth-L1 location bias
    val_dtvl_shift = float(np.median(true_val) - np.median(pred_val))
    cached_weeks = pickle.loads(cache_pkl.read_bytes())
    return sub_dates, ordered_sub_gs, cached_weeks, val_dtvl_shift


def build_agent_scenario_summary(
    pred_graph: GraphSnapshot,
    mc_samples: list[GraphSnapshot],
    config: AgentConfig,
    horizon: int = HORIZON,
    eligible_set: set[str] | None = None,
) -> ScenarioSummary:
    if config.skip_scenario:
        return ScenarioSummary(ranked_losses=[], worst_scenario="", worst_horizon=0)
    res = run_agent_scenarios(pred_graph, mc_samples, config, horizon=horizon)
    if eligible_set is not None:
        filtered = [
            sl.model_copy(update={"top_targets": [t for t in sl.top_targets if t in eligible_set][:5]})
            for sl in res.ranked_losses
        ]
        return res.model_copy(update={"ranked_losses": filtered})
    return res


def calibrate_mc_sigma_stock(ordered_gs: list[GraphSnapshot], holdout_start: str = "2025-01-01") -> float:
    train_gs = [g for g in ordered_gs if g.date < holdout_start]
    rel_changes: list[float] = []
    for i in range(1, len(train_gs)):
        prev_w = {(e.source, e.target): e.weight for e in train_gs[i - 1].edges if e.weight >= 100_000.0}
        for e in train_gs[i].edges:
            pw = prev_w.get((e.source, e.target), 0.0)
            if pw > 0.0:
                rel_changes.append(min(5.0, abs(e.weight - pw) / pw))
    if not rel_changes:
        return 0.15
    arr = np.asarray(rel_changes, dtype=float)
    mad_std = float(1.4826 * np.median(np.abs(arr - np.median(arr))))
    sigma = max(mad_std, 0.05) * (HORIZON ** 0.5)
    return max(0.05, min(sigma, 0.60))


def calibrate_conformal_margins_stock(
    ordered_gs: list[GraphSnapshot], mc_sigma: float, holdout_start: str = "2025-01-01", target_cov: float = 0.90
) -> dict[str, float]:
    """Fit per-metric conformal nonconformity margins strictly on pre-2025 validation weeks."""
    pre_gs = [g for g in ordered_gs if g.date < holdout_start]
    val_gs = pre_gs[-(24 + HORIZON):]
    rng = np.random.default_rng(123)
    alpha = 1.0 - target_cov
    scores: dict[str, list[float]] = {m: [] for m in ("N1", "N2", "N3", "N4", "N5")}
    for i in range(max(0, len(val_gs) - HORIZON)):
        g_t = val_gs[i]
        g_fut = val_gs[i + HORIZON]
        gt_m = compute_monitor_metrics(g_fut)
        mc_s = _generate_mc_samples(g_t, 50, sigma=mc_sigma, rng=rng)
        mc_m = [compute_monitor_metrics(s) for s in mc_s]
        for mid in scores:
            arr = np.asarray([m[mid] for m in mc_m], dtype=float)
            lo, hi = float(np.quantile(arr, alpha / 2.0)), float(np.quantile(arr, 1.0 - alpha / 2.0))
            tv = float(gt_m[mid])
            if tv < lo:
                scores[mid].append(lo - tv)
            elif tv > hi:
                scores[mid].append(tv - hi)
            else:
                scores[mid].append(0.0)
    margins: dict[str, float] = {}
    for mid, sc in scores.items():
        if not sc:
            margins[mid] = 0.0
        else:
            n = len(sc)
            rk = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
            margins[mid] = float(np.sort(np.asarray(sc))[rk - 1])
    return margins


def evaluate_upstream_rq1(
    ordered_sub_gs: list[GraphSnapshot],
    cached_weeks: list[dict[str, Any]],
    mc_sigma: float,
    val_dtvl_shift: float,
) -> dict[str, Any]:
    """Compute RQ1 (`b1_forecast`, `b3_calibration`, `b6_robustness`) for persistence (`m1`) and `FM v4` (`m5`)."""
    rng = np.random.default_rng(42)
    margins = calibrate_conformal_margins_stock(ordered_sub_gs, mc_sigma)
    out: dict[str, Any] = {
        "conformal_margins": margins,
        "mc_sigma": round(mc_sigma, 4),
        "val_dtvl_median_shift": round(val_dtvl_shift, 6),
    }

    for method_key, use_fm in [("m1_persistence", False), ("m5_fm_v4", True)]:
        pr_maes, rank_corrs, trends = [], [], []
        inside_90: list[bool] = []
        pred_covs: list[float] = []
        crps_vals: list[float] = []

        for w in cached_weeks:
            curr_gs: GraphSnapshot = w["curr_gs"]
            fut_gs: GraphSnapshot = w["fut_const_gs"]
            pred_gs: GraphSnapshot = (
                prepare_pred_graph_for_v4(
                    curr_gs,
                    w["pred_const_gs"],
                    w["v4_node_deltas"],
                    w["v4_edge_probs"],
                    val_dtvl_shift=val_dtvl_shift,
                    pi_min=0.2,
                )
                if use_fm
                else curr_gs
            )

            pr_pred = _compute_pagerank_vector(pred_gs)
            pr_true = _compute_pagerank_vector(fut_gs)
            pr_maes.append(_pagerank_mae(pr_pred, pr_true))
            rank_corrs.append(_rank_correlation(pr_pred, pr_true))

            m_curr = compute_monitor_metrics(curr_gs)
            m_pred = compute_monitor_metrics(pred_gs)
            m_true = compute_monitor_metrics(fut_gs)
            trends.append(_trend_consistency(m_curr, m_pred, m_true))

            if use_fm:
                mc_s = _generate_mc_samples(pred_gs, 50, sigma=mc_sigma, rng=rng)
                mc_m = [compute_monitor_metrics(g) for g in mc_s]
                for mid in ("N1", "N2", "N3", "N4", "N5"):
                    draws = np.array([m[mid] for m in mc_m], dtype=float)
                    mg = margins.get(mid, 0.0)
                    lo = float(np.quantile(draws, 0.05)) - mg
                    hi = float(np.quantile(draws, 0.95)) + mg
                    tv = float(m_true[mid])
                    inside_90.append(lo <= tv <= hi)
                    pred_covs.append(0.90)
                    crps_vals.append(_compute_crps_sample(draws, tv))

        rel_degs = []
        clean_pr = float(np.mean(pr_maes))
        for regime in REGIMES:
            rcfg = REGIME_CONFIGS[regime]
            r_rng = np.random.default_rng(1042)
            deg_prs = []
            for w in cached_weeks:
                curr_gs = w["curr_gs"]
                fut_gs = w["fut_const_gs"]
                deg_in = _apply_degradation(curr_gs, rcfg, r_rng)
                deg_pred = (
                    prepare_pred_graph_for_v4(
                        deg_in,
                        deg_in,
                        w["v4_node_deltas"],
                        w["v4_edge_probs"],
                        val_dtvl_shift=val_dtvl_shift,
                        pi_min=0.2,
                    )
                    if use_fm
                    else deg_in
                )
                deg_prs.append(_pagerank_mae(_compute_pagerank_vector(deg_pred), _compute_pagerank_vector(fut_gs)))
            rel_degs.append((float(np.mean(deg_prs)) - clean_pr) / max(clean_pr, 1e-12))

        row = {
            "pr_mae": float(np.mean(pr_maes)),
            "pr_mae_1e5": round(float(np.mean(pr_maes)) * 1e5, 2),
            "rank_corr": round(float(np.mean(rank_corrs)), 4),
            "trend_consistency": 0.0 if not use_fm else round(float(np.mean(trends)), 4),
            "mean_rel_degradation": round(float(np.mean(rel_degs)), 4),
            "pi_coverage_90": round(float(np.mean(inside_90)), 4) if use_fm else None,
            "ece": round(_compute_ece(pred_covs, inside_90), 4) if use_fm else None,
        }
        out[method_key] = row

    v4_holdout = json.loads((ROOT / "checkpoints" / "dexposure-fm-v4" / "v4_holdout_report.json").read_text())
    v4_baselines = json.loads((ROOT / "checkpoints" / "dexposure-fm-v4" / "baseline_comparison.json").read_text())
    out["v4_holdout_h4"] = v4_holdout["h4"]
    out["v4_baselines_test_h4"] = {
        "degree_only": v4_baselines["test"]["degree_only"]["h4"],
        "momentum_persistence": v4_baselines["test"]["momentum_persistence"]["h4"],
    }
    return out


def run_rules_and_rq3(
    sub_dates: list[str],
    ordered_sub_gs: list[GraphSnapshot],
    cached_weeks: list[dict[str, Any]],
    mc_sigma: float,
    val_dtvl_shift: float,
) -> dict[str, Any]:
    """Run m1 (persist.+rules), m5 (FM v4 + rules, top_k=7), and RQ3 clean & degraded ablations."""
    gs_by_date = {g.date: g for g in ordered_sub_gs}
    hist_metrics_by_date = {g.date: compute_monitor_metrics(g) for g in ordered_sub_gs}

    def eval_rule_config(
        use_fm: bool,
        *,
        tau_data: float = 0.70,
        tau_conf: float = 0.60,
        skip_scenario: bool = False,
        top_k: int = 7,
        enrich_monitor_attr: bool = True,
        mask_cfg: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        cfg = AgentConfig(
            tau_data=tau_data,
            tau_conf=tau_conf,
            skip_scenario=skip_scenario,
            top_k=top_k,
            rolling_window=42,
            mc_samples=50,
            pi_min=0.2,
        )
        rng = np.random.default_rng(42)
        mask_rng = np.random.default_rng(202506)

        week_rows = []
        all_hits: list[bool] = []
        false_ints_tail: list[bool] = []
        false_ints_nonloss: list[bool] = []
        target_sets: list[set[str]] = []
        safe_weeks = 0
        dh_scores: list[float] = []
        interv_weeks = 0

        for w in cached_weeks:
            d = w["date"]
            idx_d = sub_dates.index(d)
            baseline_hist = [hist_metrics_by_date[sub_dates[j]] for j in range(max(0, idx_d - 42), idx_d)]
            prev_g = gs_by_date[sub_dates[idx_d - 1]] if idx_d > 0 else None

            curr_gs: GraphSnapshot = w["curr_gs"]
            fut_gs: GraphSnapshot = w["fut_const_gs"]
            stressed, _, pos_loss, _ = compute_netflow_labels(curr_gs, fut_gs)
            elig = eligible_protocols(curr_gs)

            obs_in = curr_gs
            if mask_cfg is not None:
                obs_in = _apply_degradation(curr_gs, mask_cfg, mask_rng)

            dh = compute_data_health(obs_in, cfg, previous_graph=prev_g)
            dh_scores.append(float(dh.score))
            if dh.safe_mode:
                safe_weeks += 1

            pred_gs = (
                prepare_pred_graph_for_v4(
                    obs_in,
                    obs_in if mask_cfg is not None else w["pred_const_gs"],
                    w["v4_node_deltas"],
                    w["v4_edge_probs"],
                    val_dtvl_shift=val_dtvl_shift,
                    pi_min=0.2,
                )
                if use_fm
                else obs_in
            )

            mc_s = _generate_mc_samples(pred_gs, cfg.mc_samples, sigma=mc_sigma, rng=rng)
            mon = run_monitor(
                predicted_graph=pred_gs,
                mc_samples=mc_s,
                baseline_history=baseline_hist,
                horizon=HORIZON,
                config=cfg,
                dh_score=dh.score,
            )
            inc_curr = incident_exposure(obs_in)
            eloss_v4 = {
                n: inc_curr[n] * max(0.0, -math.expm1(min(0.0, w["v4_node_deltas"].get(n, 0.0) + val_dtvl_shift)))
                for n in elig
            }
            for al in mon.alerts:
                if use_fm and enrich_monitor_attr:
                    top_attr = sorted(elig, key=lambda n: (-eloss_v4.get(n, 0.0), n))[:top_k]
                    al.attribution = {n: float(eloss_v4[n]) for n in top_attr}
                else:
                    al.attribution = {n: float(v) for n, v in al.attribution.items() if n in elig}

            scen = build_agent_scenario_summary(pred_gs, mc_s, cfg, horizon=HORIZON, eligible_set=elig)
            dec = generate_tickets(mon.alerts, scen, dh, cfg)

            t_all: set[str] = set()
            t_int: set[str] = set()
            for tk in dec.tickets:
                t_all.update(tk.targets)
                if tk.action in ("Recommend-Reduce", "Contingency"):
                    t_int.update(tk.targets)
            if t_int:
                interv_weeks += 1

            score_map = _ticket_target_score_map(dec)
            ranked = [p for p, _ in sorted(score_map.items(), key=lambda kv: (-kv[1], kv[0]))]
            target_sets.append(set(ranked))

            for t in ranked:
                all_hits.append(t in stressed)
            for t in t_int:
                false_ints_tail.append(t not in stressed)
                false_ints_nonloss.append(t not in pos_loss)

            rec_k = len(stressed & set(ranked)) / len(stressed) if stressed else 0.0
            prec_k = len(stressed & set(ranked)) / len(ranked) if ranked else None
            week_rows.append({
                "date": d,
                "future_date": w["future_date"],
                "targets_scored": [[p, float(score_map[p])] for p in ranked],
                "truly_stressed": sorted(stressed),
                "positive_loss_protocols": sorted(pos_loss),
                "precision": prec_k,
                "recall_at_k": rec_k,
                "intervention_targets": sorted(t_int),
                "false_interventions": [t not in stressed for t in sorted(t_int)],
                "nonloss_false_interventions": [t not in pos_loss for t in sorted(t_int)],
                "dh_score": float(dh.score),
                "safe_mode": bool(dh.safe_mode),
                "n_alerts": len(mon.alerts),
                "mean_confidence": (
                    float(np.mean([a.confidence for a in mon.alerts])) if mon.alerts else None
                ),
            })

        jacc = []
        for i in range(1, len(target_sets)):
            a, b = target_sets[i], target_sets[i - 1]
            jacc.append(1.0 if (not a and not b) else (len(a & b) / len(a | b) if (a | b) else 0.0))

        micro_prec = float(np.mean(all_hits)) if all_hits else 0.0
        mean_rec = float(np.mean([r["recall_at_k"] for r in week_rows]))
        f1 = 2 * micro_prec * mean_rec / (micro_prec + mean_rec) if (micro_prec + mean_rec) > 0 else 0.0
        fir_tail = float(np.mean(false_ints_tail)) if false_ints_tail else None
        fir_nonloss = float(np.mean(false_ints_nonloss)) if false_ints_nonloss else None

        return {
            "micro_precision": micro_prec,
            "mean_recall": mean_rec,
            "f1": f1,
            "fir": fir_tail,
            "nonloss_fir": fir_nonloss,
            "n_interventions": len(false_ints_tail),
            "interv_weeks": interv_weeks,
            "stability": float(np.mean(jacc)) if jacc else 0.0,
            "mean_dh": float(np.mean(dh_scores)),
            "safe_pct": round(100.0 * safe_weeks / len(cached_weeks), 1),
            "weeks": week_rows,
        }

    m1_res = eval_rule_config(use_fm=False, top_k=7)
    m5_res = eval_rule_config(use_fm=True, top_k=7, enrich_monitor_attr=True)
    # Canonical default AgentConfig (structural monitor attribution without outflow override) for RQ3A
    m5_canonical_k5 = eval_rule_config(use_fm=True, top_k=5, enrich_monitor_attr=False)
    a3_no_scen = eval_rule_config(use_fm=True, skip_scenario=True, top_k=5, enrich_monitor_attr=False)
    a2_no_conf = eval_rule_config(use_fm=True, tau_conf=0.0, top_k=5, enrich_monitor_attr=False)

    degraded_rows = []
    degraded_specs = [
        ("edges + features, 90%", {"edge_mask": 0.90, "noise_sigma": 0.0, "feature_drop": 0.90}, [0.70, 0.00]),
        ("edges only, 90%", {"edge_mask": 0.90, "noise_sigma": 0.0, "feature_drop": 0.0}, [0.70, 0.85]),
        ("features only, 90%", {"edge_mask": 0.0, "noise_sigma": 0.0, "feature_drop": 0.90}, [0.85, 0.70]),
        ("edges + features, 80%", {"edge_mask": 0.80, "noise_sigma": 0.0, "feature_drop": 0.80}, [0.70, 0.00]),
    ]
    for label, mcfg, taus in degraded_specs:
        for tau_d in taus:
            r_deg = eval_rule_config(use_fm=True, tau_data=tau_d, tau_conf=0.0, mask_cfg=mcfg, top_k=7)
            degraded_rows.append({
                "regime": label,
                "tau_data": tau_d,
                "mean_dh": round(r_deg["mean_dh"], 3),
                "safe_pct": r_deg["safe_pct"],
                "n_int": r_deg["n_interventions"],
                "fir": round(r_deg["fir"], 3) if r_deg["fir"] is not None else None,
            })

    return {
        "m1_persistence_rules": m1_res,
        "m5_fm_rules": m5_res,
        "rq3_clean": {
            "canonical_m5_k5": {k: v for k, v in m5_canonical_k5.items() if k != "weeks"},
            "full_m5_k7": {k: v for k, v in m5_res.items() if k != "weeks"},
            "a3_no_scenario": {k: v for k, v in a3_no_scen.items() if k != "weeks"},
            "a2_no_conf_gate": {k: v for k, v in a2_no_conf.items() if k != "weeks"},
        },
        "rq3_degraded": degraded_rows,
    }


def sample_three_drafts_for_week(
    method: str,
    date_str: str,
    future_date: str,
    system_prompt: str,
    user_prompt: str,
    cset: CandidateSet,
    stressed: set[str],
    actual_losses: dict[str, float],
    pos_loss: set[str],
    gate_meta: dict[str, Any],
    decision_model: str,
    consistency_runs: int = 3,
) -> dict[str, Any]:
    raw_outputs = []
    all_attempts = []
    for ri in range(consistency_runs):
        accepted = None
        for val_attempt in range(4):
            cand = llm_mod.call_llm(system_prompt, user_prompt, model=decision_model)
            rec = {
                "consistency_run": ri,
                "validation_attempt": val_attempt,
                **cand,
            }
            try:
                llm_mod.validate_decision_output(cand, cset)
                accepted = cand
                rec["validation_status"] = "accepted"
                all_attempts.append(rec)
                break
            except Exception as exc:
                rec["validation_status"] = f"rejected: {exc}"
                all_attempts.append(rec)
        if accepted is None:
            raise RuntimeError(f"{method} {date_str} run {ri} failed schema validation 4 times")
        raw_outputs.append(accepted)

    ev_bundle = EvidenceBundle(
        date=date_str,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        candidate_set=cset,
    )
    cons = llm_mod.compute_consistency(raw_outputs)
    primary = raw_outputs[0]
    assessment = llm_mod.assess_week(
        primary, stressed, actual_losses, user_prompt, positive_loss_protocols=pos_loss
    )
    in_tok = sum(int(x.get("input_tokens", 0) or 0) for x in raw_outputs)
    out_tok = sum(int(x.get("output_tokens", 0) or 0) for x in raw_outputs)
    lat_ms = float(np.mean([float(x.get("latency_ms", 0.0) or 0.0) for x in raw_outputs]))

    return {
        "date": date_str,
        "future_date": future_date,
        "method": method,
        "horizon": HORIZON,
        "model": decision_model,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "evidence_bundle": {
            "schema_version": 1,
            "date": date_str,
            "evidence_sha256": ev_bundle.evidence_sha256,
            "candidate_set_sha256": cset.candidate_set_sha256,
            "candidate_membership_sha256": cset.candidate_membership_sha256,
            "candidate_set": cset.as_dicts(),
        },
        "layer4_gate_state": gate_meta,
        "truly_stressed": sorted(stressed),
        "positive_loss_protocols": sorted(pos_loss),
        "actual_losses": actual_losses,
        "llm_outputs": raw_outputs,
        "llm_attempts": all_attempts,
        "consistency": cons,
        "assessment": assessment,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_ms": lat_ms,
    }


def compute_stats_and_matched_budget(
    m1_weeks: list[dict[str, Any]],
    m5_weeks: list[dict[str, Any]],
    raw_m2: list[dict[str, Any]],
    raw_m6: list[dict[str, Any]],
    raw_m7: list[dict[str, Any]],
) -> dict[str, Any]:
    rng = np.random.default_rng(20260610)

    def boot_ci(arr: np.ndarray) -> list[float]:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return [0.0, 0.0, 0.0]
        idx = rng.integers(0, len(valid), size=(N_BOOT, len(valid)))
        means = valid[idx].mean(axis=1)
        return [
            round(float(valid.mean()), 4),
            round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4),
        ]

    def boot_f1(p_arr: np.ndarray, r_arr: np.ndarray) -> list[float]:
        n = len(p_arr)
        pm, rm = float(np.nanmean(p_arr)), float(np.nanmean(r_arr))
        pt = 2 * pm * rm / (pm + rm) if (pm + rm) > 0 else 0.0
        idx = rng.integers(0, n, size=(N_BOOT, n))
        vals = []
        for row_i in idx:
            p_i = float(np.nanmean(p_arr[row_i]))
            r_i = float(np.nanmean(r_arr[row_i]))
            vals.append(2 * p_i * r_i / (p_i + r_i) if (p_i + r_i) > 0 else 0.0)
        return [
            round(pt, 4),
            round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4),
        ]

    def boot_micro_fir(false_flags_by_week: list[list[bool]]) -> list[float] | None:
        total_int = sum(len(x) for x in false_flags_by_week)
        if total_int == 0:
            return None
        pt = sum(sum(x) for x in false_flags_by_week) / total_int
        n = len(false_flags_by_week)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        vals = []
        for row_i in idx:
            tot = sum(len(false_flags_by_week[j]) for j in row_i)
            if tot > 0:
                vals.append(sum(sum(false_flags_by_week[j]) for j in row_i) / tot)
        if not vals:
            return [round(pt, 4), 0.0, 0.0]
        return [
            round(pt, 4),
            round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4),
        ]

    def summarize_llm(entries: list[dict[str, Any]]) -> dict[str, Any]:
        entries = sorted(entries, key=lambda x: x["date"])
        p_arr = np.array([e["assessment"]["precision"] for e in entries], dtype=float)
        r_arr = np.array([e["assessment"]["completeness"] for e in entries], dtype=float)
        cons_arr = np.array([e["consistency"] for e in entries], dtype=float)

        tsets = [set(e["assessment"]["targets"]) for e in entries]
        stab = [
            1.0 if (not tsets[i] and not tsets[i - 1]) else len(tsets[i] & tsets[i - 1]) / len(tsets[i] | tsets[i - 1])
            for i in range(1, len(tsets))
        ]

        ff_tail_weeks = []
        ff_nonloss_weeks = []
        interv_wks = 0
        total_int = 0
        total_tail_false = 0
        total_nonloss_false = 0
        for e in entries:
            a = e["assessment"]
            ni = int(a["intervention_targets"])
            nf = int(a["false_interventions"])
            nlf = int(a.get("non_loss_false_interventions", 0) or 0)
            total_int += ni
            total_tail_false += nf
            total_nonloss_false += nlf
            if ni > 0:
                interv_wks += 1
            ff_tail_weeks.append([True] * nf + [False] * (ni - nf))
            ff_nonloss_weeks.append([True] * nlf + [False] * (ni - nlf))

        return {
            "n_weeks": len(entries),
            "interv_weeks": interv_wks,
            "total_interventions": total_int,
            "total_tail_false_interventions": total_tail_false,
            "total_nonloss_false_interventions": total_nonloss_false,
            "precision_ci": boot_ci(p_arr),
            "recall_ci": boot_ci(r_arr),
            "f1_ci": boot_f1(p_arr, r_arr),
            "micro_fir_ci": boot_micro_fir(ff_tail_weeks),
            "nonloss_fir_ci": boot_micro_fir(ff_nonloss_weeks),
            "stability": round(float(np.mean(stab)), 4),
            "consistency": round(float(np.mean(cons_arr)), 4),
            "total_input_tokens": sum(int(e.get("input_tokens", 0)) for e in entries),
            "total_output_tokens": sum(int(e.get("output_tokens", 0)) for e in entries),
        }

    def summarize_rules(weeks: list[dict[str, Any]]) -> dict[str, Any]:
        all_hits = []
        r_list = []
        ff_tail_weeks = []
        ff_nonloss_weeks = []
        tsets = []
        interv_wks = 0
        for w in weeks:
            ranked = [p for p, _ in w["targets_scored"]]
            st = set(w["truly_stressed"])
            tsets.append(set(ranked))
            for t in ranked:
                all_hits.append(t in st)
            r_list.append(float(w["recall_at_k"]))
            fi = list(w["false_interventions"])
            nlfi = list(w.get("nonloss_false_interventions", []))
            if fi:
                interv_wks += 1
            ff_tail_weeks.append(fi)
            ff_nonloss_weeks.append(nlfi)
        p_micro = float(np.mean(all_hits)) if all_hits else 0.0
        p_week = np.array([float(w["precision"]) if w["precision"] is not None else np.nan for w in weeks], dtype=float)
        r_arr = np.array(r_list, dtype=float)
        stab = [
            1.0 if (not tsets[i] and not tsets[i - 1]) else len(tsets[i] & tsets[i - 1]) / len(tsets[i] | tsets[i - 1])
            for i in range(1, len(tsets))
        ]
        n = len(weeks)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        f1_boots = []
        for row_i in idx:
            hits_i = [
                (p in set(weeks[j]["truly_stressed"]))
                for j in row_i
                for p, _ in weeks[j]["targets_scored"]
            ]
            pi = float(np.mean(hits_i)) if hits_i else 0.0
            ri = float(np.mean(r_arr[row_i]))
            f1_boots.append(2 * pi * ri / (pi + ri) if (pi + ri) > 0 else 0.0)
        rm = float(np.mean(r_arr))
        f1_pt = 2 * p_micro * rm / (p_micro + rm) if (p_micro + rm) > 0 else 0.0
        return {
            "n_weeks": n,
            "interv_weeks": interv_wks,
            "total_interventions": sum(len(x) for x in ff_tail_weeks),
            "micro_precision": round(p_micro, 4),
            "week_precision_ci": boot_ci(p_week),
            "recall_ci": boot_ci(r_arr),
            "f1_ci": [
                round(f1_pt, 4),
                round(float(np.percentile(f1_boots, 2.5)), 4),
                round(float(np.percentile(f1_boots, 97.5)), 4),
            ],
            "micro_fir_ci": boot_micro_fir(ff_tail_weeks),
            "nonloss_fir_ci": boot_micro_fir(ff_nonloss_weeks),
            "stability": round(float(np.mean(stab)), 4),
        }

    def ranked_series_llm(entries: list[dict[str, Any]]) -> list[tuple[list[str], set[str]]]:
        out = []
        for e in sorted(entries, key=lambda x: x["date"]):
            tps = sorted(
                e["llm_outputs"][0].get("target_protocols", []),
                key=lambda t: -float(t.get("risk_score", 0.0)),
            )
            rk = [llm_mod.normalize_protocol_name(t["protocol"]) for t in tps]
            out.append((rk, set(e["truly_stressed"])))
        return out

    def ranked_series_rules(weeks: list[dict[str, Any]]) -> list[tuple[list[str], set[str]]]:
        return [([p for p, _ in w["targets_scored"]], set(w["truly_stressed"])) for w in sorted(weeks, key=lambda x: x["date"])]

    series_map = {
        "m1": ranked_series_rules(m1_weeks),
        "m2": ranked_series_llm(raw_m2),
        "m5": ranked_series_rules(m5_weeks),
        "m6": ranked_series_llm(raw_m6),
        "m7": ranked_series_llm(raw_m7),
    }

    ks = (1, 3, 5, 7)
    budget_table: dict[str, Any] = {}
    per_week_vecs: dict[tuple[str, int, str], np.ndarray] = {}
    for m_id, seq in series_map.items():
        budget_table[m_id] = {}
        for k in ks:
            recs, precs, used = [], [], []
            for rk, st in seq:
                top = rk[:k]
                hits = sum(1 for x in top if x in st)
                recs.append(hits / len(st) if st else 0.0)
                precs.append(hits / len(top) if top else 0.0)
                used.append(len(top))
            r_vec = np.array(recs, dtype=float)
            p_vec = np.array(precs, dtype=float)
            per_week_vecs[(m_id, k, "rec")] = r_vec
            per_week_vecs[(m_id, k, "prec")] = p_vec
            budget_table[m_id][f"k={k}"] = {
                "recall_x1e3": round(float(r_vec.mean()) * 1e3, 2),
                "precision": round(float(p_vec.mean()), 4),
                "mean_targets": round(float(np.mean(used)), 2),
                "saturated": any(u < k for u in used),
            }

    def perm_p(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        obs = abs(float(d.mean()))
        signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
        null = np.abs((signs * d).mean(axis=1))
        return round(float((np.sum(null >= obs) + 1) / (N_PERM + 1)), 4)

    perm_tests = {}
    for comp_m in ("m5", "m6", "m7"):
        for k in ks:
            perm_tests[f"{comp_m}_vs_m2_k={k}"] = {
                "recall_p": perm_p(per_week_vecs[(comp_m, k, "rec")], per_week_vecs[("m2", k, "rec")]),
                "precision_p": perm_p(per_week_vecs[(comp_m, k, "prec")], per_week_vecs[("m2", k, "prec")]),
            }

    p7_m7 = per_week_vecs[("m7", 7, "prec")]
    p7_m2 = per_week_vecs[("m2", 7, "prec")]
    wins = int(np.sum(p7_m7 > p7_m2 + 1e-9))
    ties = int(np.sum(np.abs(p7_m7 - p7_m2) <= 1e-9))
    losses = int(np.sum(p7_m7 < p7_m2 - 1e-9))

    return {
        "panel_a": {
            "m1_persistence_rules": summarize_rules(m1_weeks),
            "m2_snapshot_llm": summarize_llm(raw_m2),
            "m5_fm_rules": summarize_rules(m5_weeks),
            "m6_fm_llm": summarize_llm(raw_m6),
            "m7_fm_llm_gated": summarize_llm(raw_m7),
        },
        "panel_b_matched_budget": budget_table,
        "paired_permutation_tests": perm_tests,
        "m7_vs_m2_weekly_record": {"wins": wins, "ties": ties, "losses": losses},
    }


def write_sha256sums(out_dir: Path) -> None:
    lines = []
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS":
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-pkl",
        default="cache/v4_29w_cache.pkl",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model", default=DEFAULT_DECISION_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--upstream-only", action="store_true")
    parser.add_argument(
        "--reuse-verified-m2",
        default=str(PAPER_ROOT / "results" / "run_20261122_fm_v3_unified" / "raw_m2_snapshot_llm.json"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_dates, ordered_sub_gs, cached_weeks, val_dtvl_shift = load_cached_v4_data(Path(args.cache_pkl))
    mc_sigma = calibrate_mc_sigma_stock(ordered_sub_gs)
    print(f"Calibrated stock MC sigma (pre-2025): {mc_sigma:.4f} | pre-2025 val median dtvl shift: {val_dtvl_shift:+.6f}")

    rq1_path = out_dir / "rq1_upstream.json"
    if rq1_path.is_file():
        print("Step 1/4: Loading cached RQ1 upstream metrics...")
        rq1_out = json.loads(rq1_path.read_text())
    else:
        print("Step 1/4: Evaluating RQ1 upstream forecast, calibration & robustness...")
        rq1_out = evaluate_upstream_rq1(ordered_sub_gs, cached_weeks, mc_sigma, val_dtvl_shift)
        rq1_path.write_text(json.dumps(rq1_out, indent=2) + "\n")
    print("RQ1 summary:", json.dumps({k: v for k, v in rq1_out.items() if k in ("m1_persistence", "m5_fm_v4")}, indent=2))

    rules_path = out_dir / "rules_and_rq3.json"
    if rules_path.is_file():
        print("Step 2/4: Loading cached m1/m5 rules & RQ3 ablations...")
        rules_rq3 = json.loads(rules_path.read_text())
    else:
        print("Step 2/4: Evaluating m1/m5 rules & RQ3 clean/degraded ablations...")
        rules_rq3 = run_rules_and_rq3(sub_dates, ordered_sub_gs, cached_weeks, mc_sigma, val_dtvl_shift)
        rules_path.write_text(json.dumps(rules_rq3, indent=2) + "\n")
    print("Rules summary:", json.dumps(rules_rq3["rq3_clean"], indent=2))
    print("Degraded summary:", json.dumps(rules_rq3["rq3_degraded"], indent=2))

    if args.upstream_only:
        return

    api_key = load_openrouter_key_from_dotenv()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in environment or .env")
    llm_mod.LLM_API_KEY = api_key
    llm_mod.DECISION_MODEL = args.model

    m5_gate_by_date = {
        w["date"]: {
            "dh_score": w["dh_score"],
            "safe_mode": w["safe_mode"],
            "n_alerts": w["n_alerts"],
            "mean_confidence": w["mean_confidence"],
            "confidence_gate_passed": True if w["mean_confidence"] is None else (w["mean_confidence"] >= 0.60),
        }
        for w in rules_rq3["m5_fm_rules"]["weeks"]
    }

    # Verify whether cached M2 from V3 run has bit-identical EvidenceBundle SHA-256 & labels
    verified_m2_seed: dict[str, dict[str, Any]] = {}
    reuse_m2_path = Path(args.reuse_verified_m2) if args.reuse_verified_m2 else None
    if reuse_m2_path and reuse_m2_path.is_file():
        for item in json.loads(reuse_m2_path.read_text()):
            if item.get("model") == args.model and len(item.get("llm_outputs", [])) == 3:
                verified_m2_seed[item["date"]] = item

    print(f"Step 3/4: Evaluating 29-week LLM tickets ({args.model}, 3 runs/week)...")
    for method in ("m2_snapshot_llm", "m6_fm_llm"):
        raw_path = out_dir / f"raw_{method}.json"
        done_by_date: dict[str, dict[str, Any]] = {}
        if raw_path.is_file():
            for item in json.loads(raw_path.read_text()):
                if item.get("model") == args.model and len(item.get("llm_outputs", [])) == 3:
                    done_by_date[item["date"]] = item

        tasks = []
        reused_count = 0
        for w in cached_weeks:
            if w["date"] in done_by_date:
                continue
            curr_gs = w["curr_gs"]
            fut_gs = w["fut_const_gs"]
            pred_gs = prepare_pred_graph_for_v4(
                curr_gs,
                w["pred_const_gs"],
                w["v4_node_deltas"],
                w["v4_edge_probs"],
                val_dtvl_shift=val_dtvl_shift,
                pi_min=0.2,
            )
            stressed, actual_losses, pos_loss, _ = compute_netflow_labels(curr_gs, fut_gs)
            pred_payload = pred_gs.model_dump()
            pred_payload["horizon"] = HORIZON
            prompt_scens = llm_mod.run_scenarios(pred_payload)
            cset_m2, cset_m6, curr_dict, pred_dict = build_v4_candidate_sets(curr_gs, pred_gs, prompt_scens)

            if method == "m2_snapshot_llm":
                sys_p, usr_p = llm_mod.build_prompt(
                    method, w["date"], HORIZON, None, curr_dict, llm_mod.compute_metrics(curr_dict), None, cset_m2
                )
                cset_use = cset_m2
                if w["date"] in verified_m2_seed:
                    prev_m2 = verified_m2_seed[w["date"]]
                    ev_now = EvidenceBundle(date=w["date"], system_prompt=sys_p, user_prompt=usr_p, candidate_set=cset_m2)
                    if (
                        prev_m2.get("evidence_bundle", {}).get("evidence_sha256") == ev_now.evidence_sha256
                        and prev_m2.get("truly_stressed") == sorted(stressed)
                    ):
                        done_by_date[w["date"]] = prev_m2
                        reused_count += 1
                        continue
            else:
                sys_p, usr_p = llm_mod.build_prompt(
                    method, w["date"], HORIZON, pred_dict, curr_dict, llm_mod.compute_metrics(pred_dict), prompt_scens, cset_m6
                )
                cset_use = cset_m6

            gate_meta = {**m5_gate_by_date.get(w["date"], {}), "action_gate_applied": False}
            tasks.append((w["date"], w["future_date"], sys_p, usr_p, cset_use, stressed, actual_losses, pos_loss, gate_meta))

        if reused_count > 0:
            ordered_now = [done_by_date[k] for k in sorted(done_by_date)]
            raw_path.write_text(json.dumps(ordered_now, indent=2) + "\n")
            print(f"  {method}: verified bit-identical SHA-256 & reused {reused_count}/29 weeks")
        print(f"  {method}: {len(done_by_date)}/29 weeks ready, {len(tasks)} to sample")

        if tasks:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                fut_map = {
                    ex.submit(
                        sample_three_drafts_for_week,
                        method,
                        d,
                        fd,
                        sp,
                        up,
                        cs,
                        st,
                        al,
                        pl,
                        gm,
                        args.model,
                        3,
                    ): d
                    for (d, fd, sp, up, cs, st, al, pl, gm) in tasks
                }
                for fut in as_completed(fut_map):
                    d_done = fut_map[fut]
                    row_res = fut.result()
                    done_by_date[d_done] = row_res
                    ordered_now = [done_by_date[k] for k in sorted(done_by_date)]
                    raw_path.write_text(json.dumps(ordered_now, indent=2) + "\n")
                    print(f"    [{method}] {len(done_by_date)}/29 done ({d_done}: P@7={row_res['assessment']['precision']:.3f}, cons={row_res['consistency']:.3f})")

    raw_m2 = json.loads((out_dir / "raw_m2_snapshot_llm.json").read_text())
    raw_m6 = json.loads((out_dir / "raw_m6_fm_llm.json").read_text())

    raw_m7 = []
    for e in raw_m6:
        gated_outs = [llm_mod.apply_action_gate(o) for o in e["llm_outputs"]]
        st = set(e["truly_stressed"])
        pos = set(e["positive_loss_protocols"])
        assess_m7 = llm_mod.assess_week(
            gated_outs[0], st, e["actual_losses"], e["user_prompt"], positive_loss_protocols=pos
        )
        e7 = dict(e)
        e7["method"] = "m7_fm_llm_gated"
        e7["raw_llm_outputs"] = e["llm_outputs"]
        e7["llm_outputs"] = gated_outs
        e7["consistency"] = llm_mod.compute_consistency(gated_outs)
        e7["assessment"] = assess_m7
        e7["layer4_gate_state"] = {**e.get("layer4_gate_state", {}), "action_gate_applied": True}
        raw_m7.append(e7)
    (out_dir / "raw_m7_fm_llm_gated.json").write_text(json.dumps(raw_m7, indent=2) + "\n")

    print("Step 4/4: Computing bootstrap CIs and 20,000-replicate paired permutation tests...")
    stats = compute_stats_and_matched_budget(
        rules_rq3["m1_persistence_rules"]["weeks"],
        rules_rq3["m5_fm_rules"]["weeks"],
        raw_m2,
        raw_m6,
        raw_m7,
    )
    (out_dir / "summary_v4_unified.json").write_text(json.dumps(stats, indent=2) + "\n")
    write_sha256sums(out_dir)
    print("\n=== UNIFIED V4 SUMMARY ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
