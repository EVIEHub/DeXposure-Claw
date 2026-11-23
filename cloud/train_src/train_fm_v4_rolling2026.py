#!/usr/bin/env python3
"""Rolling Pre-2026 DeXposure-FM v4 Training, Calibration & 2026 Head-to-Head Evaluation.

Advances the exact `DeXposure-FM v4` training & calibration protocol by 1 year:
  - Original V4 (`pre-2025`): `holdout_start = "2025-01-01"`, `validation_weeks = 24`
  - Rolling V4 (`pre-2026`):  `holdout_start = "2026-01-01"`, `validation_weeks = 24`
      * Train: `2020-03-23 .. 2025-07-14` (278 Mondays; all targets strictly <= 2025-07-14)
      * Val:   `2025-07-21 .. 2025-12-29` (24 Mondays; all targets strictly <= 2025-12-29)
      * Test:  `2026-01-05 .. 2026-11-23` (47 Mondays; h=1: 46w, h=4: 43w, h=8: 39w, h=12: 35w)

Runs the complete GPU campaign in a single job:
  1. Verifies SHA-256 of the stitched 349-week NetFlowStockSnapshot bundle and spot oracle.
  2. Constructs leakage-free multi-horizon Net-Flow samples (`278` train / `24` val / `47` test).
  3. Trains `DeXposureFMV4` (`GraphPFN` + monotonic Net-Flow edge & protocol heads) for up to
     40 epochs (`patience=8`) with resumable per-epoch checkpoints and comprehensive logging.
  4. Fits monotonicity-preserving validation hazard calibration + closed-form validation median
     log-contraction shifts `beta_h^{val,2025}` strictly on the 24 pre-2026 validation weeks.
  5. Writes complete release checkpoints (`dexposure-fm-v4.pt`, `dexposure-fm-h{1,4,8,12}.pt`,
     `feature_schema.json`, `calibration_summary.json`, `v4_holdout_report.json`, `SHA256SUMS`)
     into `checkpoints/dexposure-fm-v4-rolling2025/` (never touching `checkpoints/dexposure-fm-v4/`).
  6. Evaluates BOTH `Frozen Pre-2025 FM-v4` AND `Rolling Pre-2026 FM-v4` (plus Degree, Momentum `m1`,
     and Snapshot Exposure `m2` baselines) head-to-head on the exact same 2026 holdout across all
     upstream multi-horizon metrics and downstream `DeXposure-Bench` Top-k supervisory targeting
     (with 10,000-resample bootstrap CIs and 20,000-replicate paired permutation p-values).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import shutil
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "paper"), str(ROOT / "cloud"), str(ROOT / "cloud" / "train_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dexposure_agent.fm_v4 import (  # noqa: E402
    DeXposureFMV4,
    HistoricalSpotPriceOracle,
    NetFlowStockSnapshot,
    V4_CORE_NODE_FEATURES,
    V4_EDGE_STATE_FEATURES,
    V4_HORIZONS,
    load_pretrained_graphpfn,
    revalue_snapshot_at_ref_date,
    v4_contract_hash,
)
from train_fm_v4 import (  # noqa: E402
     atomic_torch_save,
    build_joint_netflow_samples,
    collect_raw_outputs,
    composite_validation_score,
    compute_multi_task_loss,
    evaluate_heuristic_baselines,
    evaluate_joint_model,
    fit_monotonic_calibration_on_val,
    log,
    read_json,
    samples_fingerprint,
    sha256,
    write_json,
)

BUNDLE_PATH = ROOT / "output" / "stitched_349w_v4_bundle.pkl.gz"
BUNDLE_META_PATH = ROOT / "output" / "stitched_349w_v4_bundle.meta.json"
SPOT_2026_PATH = ROOT / "paper" / "dexposure_agent" / "resources" / "historical_spot_prices_v4_2026.json"
BASE_GRAPHPFN_CKPT = ROOT / "checkpoints" / "graphpfn-v1.ckpt"
FROZEN_PRE2025_CKPT = ROOT / "checkpoints" / "dexposure-fm-v4" / "dexposure-fm-v4.pt"

PLAN_PATH = ROOT / "cloud" / "fm_v4_plan.json"
ROLLING_TRAIN_DIR = ROOT / "output" / "fm_v4_rolling2025_train"
ROLLING_RELEASE_DIR = ROOT / "checkpoints" / "dexposure-fm-v4-rolling2025"
COMPARISON_OUT_DIR = ROOT / "paper" / "results" / "run_20261123_v4_rolling2026_comparison"

NON_SUPERVISED_CATEGORIES = frozenset({"CEX", "Risk Curators", "Chain", "Oracle", "Services"})
PRE2025_VAL_MEDIAN_SHIFT_H4 = 0.080181


def compute_validation_median_shifts(
    model: Any,
    val_samples: list[Any],
    horizons: list[int],
    device: Any,
) -> dict[str, float]:
    raw_val = collect_raw_outputs(model, val_samples, device)
    shifts: dict[str, float] = {}
    for k, h in enumerate(horizons):
        tr_l, pr_l = [], []
        for row in raw_val:
            s = row["sample"]
            if not bool(s.valid_horizons[k]):
                continue
            nmk = s.node_valid_mask[:, k]
            if nmk.any():
                tr_l.append(s.y_delta_log_tvl[nmk, k])
                pr_l.append(row["pred_delta_log_tvl"][nmk, k])
        if tr_l:
            tc = np.concatenate(tr_l)
            pc = np.concatenate(pr_l)
            shifts[f"h{h}"] = round(float(np.median(tc) - np.median(pc)), 6)
        else:
            shifts[f"h{h}"] = 0.0
    return shifts


def compute_stressed_set_v4(
    snap_t: NetFlowStockSnapshot,
    snap_th: NetFlowStockSnapshot,
    oracle: HistoricalSpotPriceOracle,
    stress_pct: float = 0.05,
) -> tuple[set[str], dict[str, float], dict[str, float]]:
    obs_exp: dict[str, float] = defaultdict(float)
    for (u, v), w in snap_t.edge_map.items():
        obs_exp[u] += w
        obs_exp[v] += w

    _, _, out_c, in_c = revalue_snapshot_at_ref_date(snap_th, snap_t.date, oracle)
    real_exp = {nid: float(out_c.get(nid, 0.0) + in_c.get(nid, 0.0)) for nid in (set(obs_exp) | set(out_c) | set(in_c))}

    losses: list[tuple[str, float]] = []
    outflow_map: dict[str, float] = {}
    for i, nid in enumerate(snap_t.node_ids):
        if snap_t.is_conduit[i] or not nid.isdigit() or snap_t.categories[i] in NON_SUPERVISED_CATEGORIES:
            continue
        w0 = float(obs_exp.get(nid, 0.0))
        if w0 <= 0.0:
            continue
        delta = w0 - float(real_exp.get(nid, 0.0))
        outflow_map[nid] = delta
        losses.append((nid, delta))

    k_top = max(1, math.ceil(stress_pct * len(losses))) if losses else 0
    pos_losses = sorted([x for x in losses if x[1] > 0.0], key=lambda kv: (-kv[1], kv[0]))
    return {nid for nid, _ in pos_losses[:k_top]}, dict(obs_exp), outflow_map


def paired_stats(diff_arr: np.ndarray, n_boot: int = 10_000, n_perm: int = 20_000, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(diff_arr)
    obs = float(np.mean(diff_arr))
    boots = rng.choice(diff_arr, size=(n_boot, n), replace=True).mean(axis=1)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * diff_arr[None, :]).mean(axis=1)
    p_two = float((np.sum(np.abs(perm_means) >= abs(obs) - 1e-12) + 1) / (n_perm + 1))
    w = int((diff_arr > 1e-9).sum())
    t = int((np.abs(diff_arr) <= 1e-9).sum())
    l = int((diff_arr < -1e-9).sum())
    return {
        "mean_diff": round(obs, 4),
        "ci95_low": round(float(ci_lo), 4),
        "ci95_high": round(float(ci_hi), 4),
        "perm_p_two_sided": round(p_two, 6),
        "weekly_record_wtl": f"{w}W-{t}T-{l}L",
    }


def evaluate_2026_downstream_head_to_head(
    snapshots: list[NetFlowStockSnapshot],
    date_to_idx: dict[str, int],
    oracle: HistoricalSpotPriceOracle,
    raw_frozen: list[dict[str, Any]],
    raw_rolling: list[dict[str, Any]],
    horizons: list[int],
    rolling_val_shift_h4: float,
) -> dict[str, Any]:
    k_h4 = horizons.index(4)
    rows_frozen = [r for r in raw_frozen if bool(r["sample"].valid_horizons[k_h4])]
    rows_rolling = [r for r in raw_rolling if bool(r["sample"].valid_horizons[k_h4])]
    assert len(rows_frozen) == len(rows_rolling), "Mismatched resolved 2026 h=4 weeks"

    method_names = (
        "m2_snapshot_exposure",
        "m1_momentum_outflow",
        "m5_fm_v4_frozen_pre2025",
        "m5_fm_v4_rolling_pre2026",
        "m5_fm_v4_rolling_hybrid",
    )
    hits: dict[str, dict[int, list[float]]] = {m: {k: [] for k in (1, 3, 5, 7)} for m in method_names}
    captured_usd: dict[str, float] = defaultdict(float)
    tot_pos_outflow = 0.0

    for rf, rr in zip(rows_frozen, rows_rolling):
        s = rf["sample"]
        assert s.date_t == rr["sample"].date_t
        t_i = date_to_idx[s.date_t]
        snap_t = snapshots[t_i]
        snap_th = snapshots[t_i + 4]

        stressed_set, obs_exp, realized_outflows = compute_stressed_set_v4(snap_t, snap_th, oracle)
        tot_pos_outflow += sum(v for v in realized_outflows.values() if v > 0.0)

        sup_indices = [
            i
            for i, nid in enumerate(snap_t.node_ids)
            if not snap_t.is_conduit[i]
            and nid.isdigit()
            and snap_t.categories[i] not in NON_SUPERVISED_CATEGORIES
            and obs_exp.get(nid, 0.0) > 0.0
        ]

        # m2: static observed exposure W_t(u) (TVL >= $25M)
        m2_cands = [i for i in sup_indices if float(snap_t.tvl_usd[i]) >= 25_000_000.0] or sup_indices
        m2_rank = [snap_t.node_ids[i] for i in sorted(m2_cands, key=lambda i: (-obs_exp[snap_t.node_ids[i]], snap_t.node_ids[i]))]

        # m1: 4w constant-price trailing momentum outflow
        m1_sc = {
            snap_t.node_ids[i]: obs_exp[snap_t.node_ids[i]] * max(0.0, 1.0 - math.exp(min(0.0, float(snap_t.features[i, 14]))))
            for i in sup_indices
        }
        m1_rank = [snap_t.node_ids[i] for i in sorted(sup_indices, key=lambda i: (-m1_sc[snap_t.node_ids[i]], snap_t.node_ids[i]))]

        # Frozen pre-2025 m5
        dtvl_f = rf["pred_delta_log_tvl"][:, k_h4]
        mf_sc = {
            snap_t.node_ids[i]: obs_exp[snap_t.node_ids[i]] * max(0.0, 1.0 - math.exp(float(dtvl_f[i]) + PRE2025_VAL_MEDIAN_SHIFT_H4))
            for i in sup_indices
        }
        mf_rank = [snap_t.node_ids[i] for i in sorted(sup_indices, key=lambda i: (-mf_sc[snap_t.node_ids[i]], snap_t.node_ids[i]))]

        # Rolling pre-2026 m5 (exact same parameter-free Layer 2 expected outflow formula using pre-2026 val shift!)
        dtvl_r = rr["pred_delta_log_tvl"][:, k_h4]
        p_str_r = rr["stress_probs"][:, k_h4]
        mr_sc = {
            snap_t.node_ids[i]: obs_exp[snap_t.node_ids[i]] * max(0.0, 1.0 - math.exp(float(dtvl_r[i]) + rolling_val_shift_h4))
            for i in sup_indices
        }
        mr_rank = [snap_t.node_ids[i] for i in sorted(sup_indices, key=lambda i: (-mr_sc[snap_t.node_ids[i]], snap_t.node_ids[i]))]

        # Rolling hybrid risk-expected outflow: `w_t(u) * (0.5 * p_stress + 0.5 * expected_contraction)`
        mh_sc = {
            snap_t.node_ids[i]: obs_exp[snap_t.node_ids[i]]
            * (0.5 * float(p_str_r[i]) + 0.5 * max(0.0, 1.0 - math.exp(float(dtvl_r[i]) + rolling_val_shift_h4)))
            for i in sup_indices
        }
        mh_rank = [snap_t.node_ids[i] for i in sorted(sup_indices, key=lambda i: (-mh_sc[snap_t.node_ids[i]], snap_t.node_ids[i]))]

        ranks = {
            "m2_snapshot_exposure": m2_rank,
            "m1_momentum_outflow": m1_rank,
            "m5_fm_v4_frozen_pre2025": mf_rank,
            "m5_fm_v4_rolling_pre2026": mr_rank,
            "m5_fm_v4_rolling_hybrid": mh_rank,
        }
        for m_name, r_list in ranks.items():
            for k in (1, 3, 5, 7):
                topk = r_list[:k]
                hits[m_name][k].append(sum(1 for u in topk if u in stressed_set) / max(1, len(topk)))
            captured_usd[m_name] += sum(max(0.0, realized_outflows.get(u, 0.0)) for u in r_list[:7])

    summary: dict[str, Any] = {
        "n_resolved_h4_weeks": len(rows_rolling),
        "date_range": [rows_rolling[0]["sample"].date_t, rows_rolling[-1]["sample"].date_t],
        "total_realized_positive_outflow_usd": round(tot_pos_outflow, 2),
        "methods": {},
        "paired_comparisons": {
            "rolling_pre2026_vs_frozen_pre2025": {},
            "rolling_pre2026_vs_m2_snapshot": {},
            "rolling_hybrid_vs_frozen_pre2025": {},
            "rolling_hybrid_vs_m2_snapshot": {},
        },
    }
    for m_name in method_names:
        cap = float(captured_usd[m_name])
        summary["methods"][m_name] = {
            **{f"P@{k}": round(float(np.mean(hits[m_name][k])), 4) for k in (1, 3, 5, 7)},
            "captured_outflow_top7_usd": round(cap, 2),
            "captured_outflow_top7_share": round(cap / max(1.0, tot_pos_outflow), 4),
        }

    for k in (1, 3, 5, 7):
        arr_r = np.asarray(hits["m5_fm_v4_rolling_pre2026"][k])
        arr_h = np.asarray(hits["m5_fm_v4_rolling_hybrid"][k])
        arr_f = np.asarray(hits["m5_fm_v4_frozen_pre2025"][k])
        arr_m2 = np.asarray(hits["m2_snapshot_exposure"][k])
        summary["paired_comparisons"]["rolling_pre2026_vs_frozen_pre2025"][f"P@{k}"] = paired_stats(arr_r - arr_f)
        summary["paired_comparisons"]["rolling_pre2026_vs_m2_snapshot"][f"P@{k}"] = paired_stats(arr_r - arr_m2)
        summary["paired_comparisons"]["rolling_hybrid_vs_frozen_pre2025"][f"P@{k}"] = paired_stats(arr_h - arr_f)
        summary["paired_comparisons"]["rolling_hybrid_vs_m2_snapshot"][f"P@{k}"] = paired_stats(arr_h - arr_m2)

    return summary


def run_campaign(holdout_start: str = "2026-01-01", val_weeks: int = 24) -> None:
    import torch
    from lib.graphpfn.model import GraphPFN

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for rolling FM-v4 training")

    t_campaign = time.time()
    gpu_props = torch.cuda.get_device_properties(0)
    log(f"GPU_DETECTED name={gpu_props.name!r} total_mib={gpu_props.total_memory // (1024 * 1024)}")

    base_plan = read_json(PLAN_PATH)
    plan = dict(base_plan)
    plan["experiment_kind"] = "dexposure_fm_v4_rolling_pre2026"
    plan["holdout_start"] = holdout_start
    plan["validation_weeks"] = val_weeks

    seed = int(plan["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for d in (ROLLING_TRAIN_DIR, ROLLING_RELEASE_DIR, COMPARISON_OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    finetuned_dir = ROLLING_TRAIN_DIR / "finetuned"
    trial_dir = ROLLING_TRAIN_DIR / "trials" / "v4_rolling_pre2026"
    finetuned_dir.mkdir(parents=True, exist_ok=True)
    trial_dir.mkdir(parents=True, exist_ok=True)

    meta = read_json(BUNDLE_META_PATH)
    actual_bundle_sha = sha256(BUNDLE_PATH)
    if actual_bundle_sha != meta["bundle_sha256"]:
        raise ValueError(f"Bundle SHA-256 mismatch: {actual_bundle_sha} != {meta['bundle_sha256']}")
    oracle = HistoricalSpotPriceOracle(SPOT_2026_PATH)
    if oracle.sha256 != meta["spot_oracle_sha256"]:
        raise ValueError("Spot oracle SHA-256 mismatch against bundle metadata")

    log(f"LOADING_STITCHED_BUNDLE {BUNDLE_PATH.name} (sha256={actual_bundle_sha[:16]}...)")
    with gzip.open(BUNDLE_PATH, "rb") as gz:
        bundle = pickle.load(gz)
    dates: list[str] = list(bundle["dates"])
    categories: list[str] = list(bundle["categories"])
    stock_snaps: list[NetFlowStockSnapshot] = list(bundle["snapshots"])
    log(f"BUNDLE_LOADED weeks={len(stock_snaps)} ({dates[0]} .. {dates[-1]})")

    pre_holdout = [d for d in dates if d < holdout_start]
    test_dates = [d for d in dates if d >= holdout_start]
    train_dates = pre_holdout[:-val_weeks]
    val_dates = pre_holdout[-val_weeks:]
    split = {"train": train_dates, "val": val_dates, "test": test_dates}
    write_json(ROLLING_TRAIN_DIR / "split.json", split)
    write_json(ROLLING_TRAIN_DIR / "plan.json", plan)

    log(
        f"SPLIT_READY train={len(train_dates)} ({train_dates[0]}..{train_dates[-1]}) | "
        f"val={len(val_dates)} ({val_dates[0]}..{val_dates[-1]}) | "
        f"test={len(test_dates)} ({test_dates[0]}..{test_dates[-1]})"
    )

    schema = {
        "schema_version": 4,
        "model_family": "dexposure_fm_v4",
        "training_split_regime": "rolling_pre2026",
        "settlement_basis": "constant_price_net_flow_ex_eth_btc_beta",
        "categories": categories,
        "core_node_features": list(V4_CORE_NODE_FEATURES),
        "edge_state_features": list(V4_EDGE_STATE_FEATURES),
        "feature_dim": len(V4_CORE_NODE_FEATURES) + len(categories),
        "edge_feature_dim": len(V4_EDGE_STATE_FEATURES),
        "horizons": list(V4_HORIZONS),
        "spot_oracle_sha256": oracle.sha256,
        "preprocessing_contract_hash": v4_contract_hash(categories, oracle.sha256),
    }
    for d in (ROLLING_RELEASE_DIR, finetuned_dir):
        write_json(d / "feature_schema.json", schema)

    date_to_idx = {d: i for i, d in enumerate(dates)}
    train_indices = [date_to_idx[d] for d in train_dates]
    val_indices = [date_to_idx[d] for d in val_dates]
    test_indices = [date_to_idx[d] for d in test_dates]
    horizons = [int(h) for h in plan["horizons"]]
    neg_ratio = int(plan["neg_ratio"])

    log("BUILDING_LEAKAGE_FREE_MULTI_HORIZON_SAMPLES")
    t_s = time.time()
    train_samples = build_joint_netflow_samples(
        stock_snaps, train_indices, max(train_indices), horizons, neg_ratio, seed, oracle
    )
    val_samples = build_joint_netflow_samples(
        stock_snaps, val_indices, max(val_indices), horizons, neg_ratio, seed + 1, oracle
    )
    test_samples = build_joint_netflow_samples(
        stock_snaps, test_indices, max(test_indices), horizons, neg_ratio, seed + 2, oracle
    )
    write_json(
        ROLLING_TRAIN_DIR / "pair_hashes.json",
        {
            "train": samples_fingerprint(train_samples),
            "val": samples_fingerprint(val_samples),
            "test": samples_fingerprint(test_samples),
            "counts": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
        },
    )
    log(
        f"SAMPLES_BUILT in {time.time() - t_s:.1f}s | "
        f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}"
    )

    baselines = {
        "val": evaluate_heuristic_baselines(val_samples, horizons),
        "test": evaluate_heuristic_baselines(test_samples, horizons),
    }
    write_json(ROLLING_RELEASE_DIR / "baseline_comparison.json", baselines)

    device = torch.device("cuda")
    params = plan["parameters"]
    encoder = load_pretrained_graphpfn(BASE_GRAPHPFN_CKPT, device)
    model = DeXposureFMV4(
        encoder=encoder,
        embed_dim=int(encoder.tfm.embed_dim),
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["dropout"]),
    ).to(device)

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.link_scorer.parameters()) + list(model.node_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": float(params["encoder_lr"])},
            {"params": head_params, "lr": float(params["head_lr"])},
        ],
        weight_decay=float(params["weight_decay"]),
    )
    epochs = int(plan["epochs"])
    patience = int(plan["early_stop_patience"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    resume_path = trial_dir / "resume_v4_rolling.partial.pt"
    best_raw_path = trial_dir / "best_raw_v4_rolling.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score = (-1e9, -1e9)
    best_epoch = 0

    if resume_path.is_file():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1
        best_score = tuple(saved["best_score"])
        best_epoch = int(saved["best_epoch"])
        log(f"RESUMED_FROM_CHECKPOINT next_epoch={start_epoch} best_epoch={best_epoch}")

    rng_perm = np.random.default_rng(seed + 100)
    for epoch in range(start_epoch, epochs + 1):
        t_ep = time.time()
        model.train()
        order = rng_perm.permutation(len(train_samples)).tolist()
        ep_losses: dict[str, list[float]] = {
            "total": [], "retain": [], "exist": [], "edge_delta": [], "stress": [], "node_delta": []
        }

        for idx in order:
            s = train_samples[idx]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = compute_multi_task_loss(model, s, params, device, use_amp=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            for k_l, v_l in parts.items():
                ep_losses[k_l].append(v_l)

        scheduler.step()
        _, val_v4 = evaluate_joint_model(model, val_samples, horizons, device)
        val_comp = composite_validation_score(val_v4, horizons)
        mean_train = {k_l: float(np.mean(v_l)) for k_l, v_l in ep_losses.items()}

        row = {
            "epoch": epoch,
            "elapsed_sec": round(time.time() - t_ep, 2),
            "train_loss": mean_train,
            "val_composite_score": list(val_comp),
            "val_h1_retain_auroc": val_v4["h1"]["active_edge_retention_80pct"]["auroc"],
            "val_h4_retain_auroc": val_v4["h4"]["active_edge_retention_80pct"]["auroc"],
            "val_h4_stress_auroc": val_v4["h4"]["node_stress_10pct"]["auroc"],
            "val_h4_frac_mae": val_v4["h4"]["active_edge_fractional_retention"]["mae"],
            "val_h4_node_mae": val_v4["h4"]["node_log_netflow_mae"],
        }
        history.append(row)
        write_json(trial_dir / "history.json", history)

        improved = val_comp > best_score
        if improved:
            best_score = val_comp
            best_epoch = epoch
            atomic_torch_save({"epoch": epoch, "score": list(val_comp), "model": model.state_dict()}, best_raw_path)

        atomic_torch_save(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_score": list(best_score),
                "history": history,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            },
            resume_path,
        )
        log(
            f"EPOCH {epoch:02d}/{epochs:02d} ({row['elapsed_sec']:.1f}s) "
            f"loss={mean_train['total']:.4f} val_score={val_comp[0]:.4f} "
            f"val_ret_auc(h1={row['val_h1_retain_auroc']:.3f}, h4={row['val_h4_retain_auroc']:.3f}) "
            f"val_stress_auc(h4={row['val_h4_stress_auroc']:.3f}) "
            f"val_frac_mae(h4={row['val_h4_frac_mae']:.3f}) "
            f"{'*BEST*' if improved else f'wait={epoch - best_epoch}/{patience}'}"
        )
        if epoch - best_epoch >= patience:
            log(f"EARLY_STOPPING at epoch={epoch} (best_epoch={best_epoch})")
            break

    resume_path.unlink(missing_ok=True)
    completed_epochs = len(history)
    best_raw = torch.load(best_raw_path, map_location=device, weights_only=False)
    model.load_state_dict(best_raw["model"], strict=True)
    best_raw_path.unlink(missing_ok=True)

    log("FITTING_PRE2026_VALIDATION_MONOTONIC_CALIBRATION")
    calib_summary = fit_monotonic_calibration_on_val(model, val_samples, device)
    calib_summary["calibration_split"] = "pre2026_validation_only_2025H2"
    val_median_shifts = compute_validation_median_shifts(model, val_samples, horizons, device)
    calib_summary["val_delta_log_tvl_median_shifts"] = val_median_shifts
    write_json(ROLLING_RELEASE_DIR / "calibration_summary.json", calib_summary)
    log(
        f"CALIBRATION_COMPLETE retain_scales={calib_summary['retain_hazard_scales']} "
        f"stress_scales={calib_summary['stress_hazard_scales']} val_median_shifts={val_median_shifts}"
    )

    _, val_calibrated_v4 = evaluate_joint_model(model, val_samples, horizons, device)

    # Persist release checkpoints (`dexposure-fm-v4.pt` + `dexposure-fm-h{1,4,8,12}.pt`)
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    provenance_common = {
        "model_family": "dexposure_fm_v4",
        "training_split_regime": "rolling_pre2026",
        "settlement_basis": "constant_price_net_flow_ex_eth_btc_beta",
        "joint_horizons": horizons,
        "seed": seed,
        "epochs": epochs,
        "completed_epochs": completed_epochs,
        "selected_epoch": best_epoch,
        "stitched_bundle_sha256": actual_bundle_sha,
        "base_checkpoint_sha256": sha256(BASE_GRAPHPFN_CKPT),
        "spot_oracle_sha256": oracle.sha256,
        "split": split,
        "calibration": calib_summary,
    }
    unified_payload = {
        "artifact_status": "complete",
        "model_family": "dexposure_fm_v4",
        "schema_version": 4,
        "training_split_regime": "rolling_pre2026",
        "settlement_basis": "constant_price_net_flow_ex_eth_btc_beta",
        "horizons": horizons,
        "hidden_dim": int(params["hidden_dim"]),
        "model": cpu_state,
        "provenance": provenance_common,
    }
    atomic_torch_save(unified_payload, ROLLING_RELEASE_DIR / "dexposure-fm-v4.pt")
    for h in horizons:
        h_payload = {**unified_payload, "horizon": h}
        atomic_torch_save(h_payload, ROLLING_RELEASE_DIR / f"dexposure-fm-h{h}.pt")
        write_json(
            ROLLING_RELEASE_DIR / f"run_config_h{h}.json",
            {
                "artifact_status": "complete",
                "experiment_kind": plan["experiment_kind"],
                "model_family": "dexposure_fm_v4",
                "forecast_horizon_weeks": h,
                "holdout_start": holdout_start,
                "validation_weeks": val_weeks,
                "selected_epoch": best_epoch,
                "completed_epochs": completed_epochs,
                "parameters": params,
                "calibration": calib_summary,
            },
        )

    log("EVALUATING_ROLLING_PRE2026_AND_FROZEN_PRE2025_HEAD_TO_HEAD_ON_2026_HOLDOUT")
    task1_rolling, v4_rolling_2026 = evaluate_joint_model(model, test_samples, horizons, device)
    write_json(ROLLING_RELEASE_DIR / "task1_metrics.json", {"model": "DeXposure-FM-v4-rolling-pre2026", **task1_rolling})
    write_json(ROLLING_RELEASE_DIR / "v4_holdout_report.json", v4_rolling_2026)

    # Load frozen pre-2025 model on GPU for exact same-sample comparison
    frozen_payload = torch.load(FROZEN_PRE2025_CKPT, map_location=device, weights_only=False)
    frozen_state = frozen_payload["model"] if isinstance(frozen_payload, dict) and "model" in frozen_payload else frozen_payload
    frozen_enc = GraphPFN(edge_head=False)
    frozen_model = DeXposureFMV4(
        encoder=frozen_enc,
        embed_dim=int(frozen_enc.tfm.embed_dim),
        hidden_dim=int(frozen_payload.get("hidden_dim", 256)) if isinstance(frozen_payload, dict) else 256,
        dropout=0.0,
    ).to(device)
    frozen_model.load_state_dict(frozen_state, strict=True)
    frozen_model.eval()

    _, v4_frozen_2026 = evaluate_joint_model(frozen_model, test_samples, horizons, device)
    raw_frozen_2026 = collect_raw_outputs(frozen_model, test_samples, device)
    raw_rolling_2026 = collect_raw_outputs(model, test_samples, device)

    downstream_h2h = evaluate_2026_downstream_head_to_head(
        snapshots=stock_snaps,
        date_to_idx=date_to_idx,
        oracle=oracle,
        raw_frozen=raw_frozen_2026,
        raw_rolling=raw_rolling_2026,
        horizons=horizons,
        rolling_val_shift_h4=float(val_median_shifts["h4"]),
    )

    for h in horizons:
        mf = v4_frozen_2026[f"h{h}"]
        mr = v4_rolling_2026[f"h{h}"]
        log(
            f"2026_H2H h={h:2d}w | ret80_auc: {mf['active_edge_retention_80pct']['auroc']:.4f} -> {mr['active_edge_retention_80pct']['auroc']:.4f} "
            f"| stress_auc: {mf['node_stress_10pct']['auroc']:.4f} -> {mr['node_stress_10pct']['auroc']:.4f} "
            f"| netflow_rho: {mf['node_netflow_spearman']:.4f} -> {mr['node_netflow_spearman']:.4f} "
            f"| whale_top7: {100*mf['walk_forward_whale_top7_hit_rate']:.1f}% -> {100*mr['walk_forward_whale_top7_hit_rate']:.1f}% "
            f"| alpha_top7: {100*mf['walk_forward_alpha_top7_hit_rate']:.1f}% -> {100*mr['walk_forward_alpha_top7_hit_rate']:.1f}%"
        )

    # Write SHA256SUMS for release directory
    (ROLLING_RELEASE_DIR / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(p)}  {p.name}\n"
            for p in sorted(ROLLING_RELEASE_DIR.iterdir())
            if p.is_file() and p.name != "SHA256SUMS"
        ),
        encoding="utf-8",
    )

    comparison_report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpu_name": gpu_props.name,
        "elapsed_seconds": round(time.time() - t_campaign, 2),
        "split_counts": {k: len(v) for k, v in split.items()},
        "split_ranges": {k: [v[0], v[-1]] for k, v in split.items()},
        "selected_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "rolling_calibration": calib_summary,
        "rolling_validation_calibrated": val_calibrated_v4,
        "upstream_2026_frozen_pre2025": v4_frozen_2026,
        "upstream_2026_rolling_pre2026": v4_rolling_2026,
        "upstream_2026_baselines": baselines["test"],
        "downstream_2026_queue_head_to_head": downstream_h2h,
        "release_checkpoint_sha256": sha256(ROLLING_RELEASE_DIR / "dexposure-fm-v4.pt"),
        "frozen_pre2025_checkpoint_sha256": sha256(FROZEN_PRE2025_CKPT),
    }
    out_report_path = COMPARISON_OUT_DIR / "summary_v4_rolling2026_vs_frozen.json"
    write_json(out_report_path, comparison_report)
    write_json(ROLLING_TRAIN_DIR / "result.json", comparison_report)
    log(f"CAMPAIGN_COMPLETE in {time.time() - t_campaign:.1f}s -> {out_report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-start", default="2026-01-01")
    parser.add_argument("--val-weeks", type=int, default=24)
    args = parser.parse_args()
    run_campaign(holdout_start=args.holdout_start, val_weeks=args.val_weeks)


if __name__ == "__main__":
    main()
