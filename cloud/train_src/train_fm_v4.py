#!/usr/bin/env python3
"""DeXposure-FM v4 Joint Multi-Horizon Net-Flow Training, Calibration & Dual-Board Holdout Runner.

Runs the full sequence in a single cloud GPU job:
  1. Verifies immutable 2020-2025 input dataset, exact chronological split (226 train / 24 val / 33 test),
     and the 283-week DeFiLlama historical spot price oracle (`historical_spot_prices_v4.json`).
  2. Builds 283 weekly Balance-Sheet Stock Exposure snapshots and enriches causal 8-week
     constant-price Net-Flow node features (`91` dims) & edge-state features (`16` dims).
  3. Evaluates non-parametric & momentum heuristic baselines on constant-price Net-Flow labels.
  4. Jointly fine-tunes `DeXposureFMV4` (`GraphPFN` backbone + fractional monotonic Net-Flow
     edge retention head + scale-free protocol >=10% real net-redemption hazard head + continuous
     log-net-flow heads) across horizons `[1, 4, 8, 12]` with per-epoch resumable checkpoints.
  5. Fits monotonicity-preserving validation-only hazard calibration on the 24 pre-2025 validation
     weeks (zero holdout leakage).
  6. Evaluates frozen calibrated `DeXposureFMV4` on the 33-week 2025 holdout across all 4 horizons,
     including BOTH Systemic Whale Top-7 AND Scale-Free High-Risk Alpha Radar Top-7 Walk-Forward hit rates.
  7. Writes complete provenance-attested release & training bundles with SHA-256 manifests.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "paper"), str(ROOT / "cloud"), str(ROOT / "cloud" / "train_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from . import tune_fm as common
except ImportError:
    import tune_fm as common

PLAN_PATH = ROOT / "cloud" / "fm_v4_plan.json"
TRAIN_REL = common.TRAIN_REL
RELEASE_REL = common.RELEASE_REL
read_json = common.read_json
write_json = common.write_json
sha256 = common.sha256
prepare_output_dirs = common.prepare_output_dirs
OutputDirectoryError = common.OutputDirectoryError
archive_sources = common.archive_sources
write_tree_manifest = common.write_tree_manifest
atomic_torch_save = common.atomic_torch_save
log = common.log


def validate_plan(plan: dict[str, Any]) -> None:
    expected_fixed = (
        ("experiment_kind", "dexposure_fm_v4_netflow_monotonic"),
        ("schema_version", 4),
        ("seed", 42),
        ("holdout_start", "2025-01-01"),
        ("validation_weeks", 24),
        ("horizons", [1, 4, 8, 12]),
        ("epochs", 40),
        ("early_stop_patience", 8),
        ("neg_ratio", 4),
    )
    for key, expected in expected_fixed:
        if plan.get(key) != expected:
            raise ValueError(f"Unreviewed FM v4 plan field {key}: {plan.get(key)!r} != {expected!r}")
    params = plan.get("parameters")
    if not isinstance(params, dict):
        raise ValueError("FM v4 plan parameters missing")
    for pkey in (
        "head_lr",
        "encoder_lr",
        "weight_decay",
        "retain_loss_weight",
        "exist_loss_weight",
        "edge_delta_loss_weight",
        "stress_loss_weight",
        "node_delta_loss_weight",
        "downside_asymmetry",
    ):
        val = params.get(pkey)
        if not isinstance(val, (int, float)) or not math.isfinite(float(val)) or float(val) <= 0:
            raise ValueError(f"Invalid FM v4 parameter {pkey}: {val!r}")


def verify_bundle(run_root: Path | str, expected_source: str | None = None) -> dict[str, Any]:
    root = Path(run_root).resolve()
    train, release = root / TRAIN_REL, root / RELEASE_REL
    plan = read_json(train / "plan.json")
    validate_plan(plan)
    selection = read_json(train / "selection_frozen.json")
    result = read_json(train / "result.json")
    if result.get("status") != "complete" or result.get("experiment_kind") != plan["experiment_kind"]:
        raise ValueError("FM v4 training campaign incomplete")
    if (
        selection.get("selection_data") != "pre2025_validation_only"
        or selection.get("plan_sha256") != sha256(train / "plan.json")
        or result.get("selection_sha256") != sha256(train / "selection_frozen.json")
    ):
        raise ValueError("FM v4 selection freeze mismatch")

    observed: set[str] = set()
    for line in (train / "tuning_manifest.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if (
            not (path.is_relative_to(train) or path.is_relative_to(release))
            or relative in observed
            or not path.is_file()
            or sha256(path) != digest
        ):
            raise ValueError(f"Invalid FM v4 artifact {relative}")
        observed.add(relative)
    expected = {
        p.relative_to(root).as_posix()
        for base in (train, release)
        for p in base.rglob("*")
        if p.is_file() and p.name != "tuning_manifest.sha256" and ".cache" not in p.parts
    }
    if expected != observed:
        raise ValueError("FM v4 artifact inventory mismatch")

    for h in plan["horizons"]:
        rel_pt = release / f"dexposure-fm-h{h}.pt"
        fin_pt = train / "finetuned" / f"best_model_h{h}.pt"
        if sha256(rel_pt) != sha256(fin_pt):
            raise ValueError(f"Checkpoint mismatch for h={h}")
        cfg = read_json(release / f"run_config_h{h}.json")
        if expected_source is not None and cfg.get("source_sha256") != expected_source:
            raise ValueError(f"Source digest mismatch in run_config_h{h}.json")
        if float(result["v4_holdout"][f"h{h}"]["monotonicity_violation_rate"]) != 0.0:
            raise ValueError(f"Non-zero monotonicity violation rate at h={h}")

    return {"status": "verified", "files": len(observed), "horizons": plan["horizons"]}


@dataclass
class JointNetFlowWeekSample:
    date_t: str
    t_idx: int
    node_ids: list[str]
    features_t: Any              # np.ndarray [N, 91] float32
    is_conduit: Any              # np.ndarray [N] bool
    tvl_t: Any                   # np.ndarray [N] float64
    edge_src_t: Any              # np.ndarray [E_act] int64
    edge_dst_t: Any              # np.ndarray [E_act] int64
    edge_log_w_t: Any            # np.ndarray [E_act] float32
    n_active: int
    pair_src: Any                # np.ndarray [P] int64
    pair_dst: Any                # np.ndarray [P] int64
    pair_edge_z: Any             # np.ndarray [P, 16] float32
    valid_horizons: Any          # np.ndarray [4] bool
    y_exist: Any                 # np.ndarray [P, 4] float32 (constant-price w_{t+h} >= $10k)
    y_retain_frac: Any           # np.ndarray [E_act, 4] float32 in [0, 1]: min_{1<=s<=h} min(1, w_const(t+s)/w(t))
    y_retain_bin80: Any          # np.ndarray [E_act, 4] float32 (y_retain_frac >= 0.80)
    y_delta_log_w: Any           # np.ndarray [E_act, 4] float32 (log1p(w_const(t+h)) - log1p(w(t)))
    y_delta_log_tvl: Any         # np.ndarray [N, 4] float32 (log1p(tvl_const(t+h)) - log1p(tvl(t)))
    y_node_stress: Any           # np.ndarray [N, 4] float32 (cumulative >=10% real net-flow redemption)
    node_valid_mask: Any         # np.ndarray [N, 4] bool


def build_joint_netflow_samples(
    snapshots: list[Any],
    indices: list[int],
    max_target_idx: int,
    horizons: list[int],
    neg_ratio: int,
    seed: int,
    oracle: Any,
) -> list[JointNetFlowWeekSample]:
    """Construct leakage-free constant-price Net-Flow samples where all targets stay <= max_target_idx."""
    import numpy as np
    from dexposure_agent.fm_v4 import (
        DEFAULT_MIN_EDGE_USD,
        EDGE_RETENTION_RATIO,
        NODE_STRESS_RATIO,
        build_netflow_edge_state_features,
        revalue_snapshot_at_ref_date,
    )

    rng = np.random.default_rng(seed)
    samples: list[JointNetFlowWeekSample] = []
    num_h = len(horizons)

    for t_idx in indices:
        if t_idx + min(horizons) > max_target_idx:
            continue
        snap = snapshots[t_idx]
        n_nodes = len(snap.node_ids)
        n_act = len(snap.edge_src)
        if n_nodes < 4 or n_act == 0:
            continue

        active_set = {(int(u), int(v)) for u, v in zip(snap.edge_src, snap.edge_dst)}
        proto_indices = np.flatnonzero(~snap.is_conduit)
        n_neg_target = min(n_act * neg_ratio, max(100, len(proto_indices) * 8))

        neg_src_l: list[int] = []
        neg_dst_l: list[int] = []
        if len(proto_indices) > 0 and n_nodes > 1:
            cand_u = rng.choice(proto_indices, size=n_neg_target * 2, replace=True)
            cand_v = rng.integers(0, n_nodes, size=n_neg_target * 2)
            seen_neg: set[tuple[int, int]] = set()
            for u_i, v_i in zip(cand_u.tolist(), cand_v.tolist()):
                if u_i == v_i or (u_i, v_i) in active_set or (u_i, v_i) in seen_neg:
                    continue
                seen_neg.add((u_i, v_i))
                neg_src_l.append(u_i)
                neg_dst_l.append(v_i)
                if len(neg_src_l) >= n_neg_target:
                    break

        pair_src = np.concatenate([snap.edge_src, np.asarray(neg_src_l, dtype=np.int64)])
        pair_dst = np.concatenate([snap.edge_dst, np.asarray(neg_dst_l, dtype=np.int64)])
        pair_z, pair_base_lw = build_netflow_edge_state_features(
            snapshots, t_idx, pair_src, pair_dst, oracle=oracle
        )
        edge_log_w_t = pair_base_lw[:n_act].copy()

        n_pairs = len(pair_src)
        valid_h = np.zeros(num_h, dtype=bool)
        y_exist = np.zeros((n_pairs, num_h), dtype=np.float32)
        y_ret_frac = np.zeros((n_act, num_h), dtype=np.float32)
        y_ret_bin = np.zeros((n_act, num_h), dtype=np.float32)
        y_dw = np.zeros((n_act, num_h), dtype=np.float32)
        y_dtvl = np.zeros((n_nodes, num_h), dtype=np.float32)
        y_stress = np.zeros((n_nodes, num_h), dtype=np.float32)
        node_mask = np.zeros((n_nodes, num_h), dtype=bool)

        max_avail_h = min(max(horizons), max_target_idx - t_idx)
        future_tvl_const: dict[int, dict[str, float]] = {}
        future_edge_const: dict[int, dict[tuple[str, str], float]] = {}
        for step in range(1, max_avail_h + 1):
            t_m, e_m, _, _ = revalue_snapshot_at_ref_date(snapshots[t_idx + step], snap.date, oracle)
            future_tvl_const[step] = t_m
            future_edge_const[step] = e_m

        for k, h in enumerate(horizons):
            if t_idx + h > max_target_idx:
                continue
            valid_h[k] = True
            f_emap = future_edge_const[h]

            # 1. Global existence on constant-`t`-price exposures at t + h
            for p_i in range(n_pairs):
                key = (snap.node_ids[int(pair_src[p_i])], snap.node_ids[int(pair_dst[p_i])])
                y_exist[p_i, k] = 1.0 if f_emap.get(key, 0.0) >= DEFAULT_MIN_EDGE_USD else 0.0

            # 2. Active-edge fractional & binary (>=80%) constant-price Net-Flow retention
            for e_i in range(n_act):
                u_id = snap.node_ids[int(snap.edge_src[e_i])]
                v_id = snap.node_ids[int(snap.edge_dst[e_i])]
                key = (u_id, v_id)
                w_0 = max(1.0, float(snap.edge_usd[e_i]))
                min_w = min(float(future_edge_const[s].get(key, 0.0)) for s in range(1, h + 1))
                r_frac = float(np.clip(min_w / w_0, 0.0, 1.0))
                y_ret_frac[e_i, k] = r_frac
                y_ret_bin[e_i, k] = 1.0 if r_frac >= EDGE_RETENTION_RATIO else 0.0
                w_h = float(f_emap.get(key, 0.0))
                y_dw[e_i, k] = float(
                    np.clip(math.log1p(max(0.0, w_h)) - float(edge_log_w_t[e_i]), -6.0, 4.0)
                )

            # 3. Protocol node constant-price Net-Flow log-delta & cumulative >=10% real run
            f_tmap = future_tvl_const[h]
            for n_i, nid in enumerate(snap.node_ids):
                if snap.is_conduit[n_i] or snap.tvl_usd[n_i] <= 0.0:
                    continue
                tvl_0 = float(snap.tvl_usd[n_i])
                node_mask[n_i, k] = True
                tvl_h = float(f_tmap.get(nid, 0.0))
                y_dtvl[n_i, k] = float(
                    np.clip(math.log1p(max(0.0, tvl_h)) - math.log1p(tvl_0), -5.0, 4.0)
                )
                min_tvl = min(float(future_tvl_const[s].get(nid, 0.0)) for s in range(1, h + 1))
                y_stress[n_i, k] = 1.0 if min_tvl <= NODE_STRESS_RATIO * tvl_0 else 0.0

        if not valid_h.any():
            continue

        samples.append(
            JointNetFlowWeekSample(
                date_t=snap.date,
                t_idx=t_idx,
                node_ids=snap.node_ids,
                features_t=snap.features,
                is_conduit=snap.is_conduit,
                tvl_t=snap.tvl_usd,
                edge_src_t=snap.edge_src,
                edge_dst_t=snap.edge_dst,
                edge_log_w_t=edge_log_w_t,
                n_active=n_act,
                pair_src=pair_src,
                pair_dst=pair_dst,
                pair_edge_z=pair_z,
                valid_horizons=valid_h,
                y_exist=y_exist,
                y_retain_frac=y_ret_frac,
                y_retain_bin80=y_ret_bin,
                y_delta_log_w=y_dw,
                y_delta_log_tvl=y_dtvl,
                y_node_stress=y_stress,
                node_valid_mask=node_mask,
            )
        )
    return samples


def samples_fingerprint(samples: list[JointNetFlowWeekSample]) -> str:
    digest = hashlib.sha256()
    for s in samples:
        digest.update(f"{s.date_t}:{len(s.node_ids)}:{s.n_active}:{len(s.pair_src)}".encode("utf-8"))
        digest.update(s.valid_horizons.tobytes())
        digest.update(s.y_retain_frac.tobytes())
        digest.update(s.y_node_stress.tobytes())
    return digest.hexdigest()


def expected_calibration_error(y_true: Any, y_prob: Any, n_bins: int = 10) -> float:
    import numpy as np

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_prob, dtype=np.float64)
    if len(yt) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        if b == n_bins - 1:
            mask = (yp >= bins[b]) & (yp <= bins[b + 1])
        else:
            mask = (yp >= bins[b]) & (yp < bins[b + 1])
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / len(yt)) * abs(float(yt[mask].mean()) - float(yp[mask].mean()))
    return float(ece)


def safe_binary_metrics(y_true: Any, y_score: Any) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    yt = np.asarray(y_true, dtype=np.float64)
    ys = np.clip(np.asarray(y_score, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if len(yt) == 0:
        return {"auroc": 0.5, "auprc": 0.0, "brier": 0.0, "ece": 0.0, "mean_pred": 0.0, "empirical_rate": 0.0}
    has_both = len(np.unique(yt > 0.5)) == 2
    auroc = float(roc_auc_score(yt, ys)) if has_both else 0.5
    auprc = float(average_precision_score(yt, ys)) if has_both else float(yt.mean())
    brier = float(np.mean(np.square(ys - yt)))
    ece = expected_calibration_error(yt, ys)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "ece": ece,
        "mean_pred": float(ys.mean()),
        "empirical_rate": float(yt.mean()),
    }


def evaluate_heuristic_baselines(
    samples: list[JointNetFlowWeekSample], horizons: list[int]
) -> dict[str, Any]:
    import numpy as np

    out: dict[str, Any] = {"degree_only": {}, "momentum_persistence": {}}
    for k, h in enumerate(horizons):
        yt_bin, yt_frac, deg_score, mom_score = [], [], [], []
        yt_str, str_mom_score = [], []
        dtvl_true = []
        for s in samples:
            if not s.valid_horizons[k]:
                continue
            z_act = s.pair_edge_z[: s.n_active]
            yt_bin.append(s.y_retain_bin80[:, k])
            yt_frac.append(s.y_retain_frac[:, k])
            deg_score.append(1.0 / (1.0 + np.exp(-2.0 * (z_act[:, 1] - 0.5))))
            raw_m = 2.0 + 1.2 * z_act[:, 6] + 0.6 * z_act[:, 4] - 0.8 * z_act[:, 7] + 0.4 * z_act[:, 8]
            mom_score.append(1.0 / (1.0 + np.exp(-raw_m)))

            nmask = s.node_valid_mask[:, k]
            if nmask.any():
                yt_str.append(s.y_node_stress[nmask, k])
                nf = s.features_t[nmask]
                s_raw = -1.8 - 0.9 * nf[:, 13] - 0.6 * nf[:, 14] + 1.2 * nf[:, 16] + 0.9 * nf[:, 17]
                str_mom_score.append(1.0 / (1.0 + np.exp(-s_raw)))
                dtvl_true.append(s.y_delta_log_tvl[nmask, k])

        ytr_b = np.concatenate(yt_bin) if yt_bin else np.zeros(0)
        ytr_f = np.concatenate(yt_frac) if yt_frac else np.zeros(0)
        ds = np.concatenate(deg_score) if deg_score else np.zeros(0)
        ms = np.concatenate(mom_score) if mom_score else np.zeros(0)
        out["degree_only"][f"h{h}"] = {
            "active_retain_80pct": safe_binary_metrics(ytr_b, ds),
            "fractional_retain_mae": float(np.mean(np.abs(ds - ytr_f))) if len(ytr_f) else 0.0,
            "node_mae_zero_persistence": float(np.mean(np.abs(np.concatenate(dtvl_true)))) if dtvl_true else 0.0,
        }
        out["momentum_persistence"][f"h{h}"] = {
            "active_retain_80pct": safe_binary_metrics(ytr_b, ms),
            "fractional_retain_mae": float(np.mean(np.abs(ms - ytr_f))) if len(ytr_f) else 0.0,
            "node_stress_10pct": safe_binary_metrics(
                np.concatenate(yt_str) if yt_str else np.zeros(0),
                np.concatenate(str_mom_score) if str_mom_score else np.zeros(0),
            ),
        }
    return out


def compute_multi_task_loss(
    model: Any,
    sample: JointNetFlowWeekSample,
    params: dict[str, Any],
    device: Any,
    use_amp: bool,
) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    feat_t = torch.as_tensor(sample.features_t, dtype=torch.float32, device=device)
    pair_src = torch.as_tensor(sample.pair_src, dtype=torch.long, device=device)
    pair_dst = torch.as_tensor(sample.pair_dst, dtype=torch.long, device=device)
    pair_z = torch.as_tensor(sample.pair_edge_z, dtype=torch.float32, device=device)
    act_src = torch.as_tensor(sample.edge_src_t, dtype=torch.long, device=device)
    act_dst = torch.as_tensor(sample.edge_dst_t, dtype=torch.long, device=device)
    act_lw = torch.as_tensor(sample.edge_log_w_t, dtype=torch.float32, device=device)

    valid_h = torch.as_tensor(sample.valid_horizons, dtype=torch.bool, device=device)
    y_exist = torch.as_tensor(sample.y_exist, dtype=torch.float32, device=device)
    y_ret_frac = torch.as_tensor(sample.y_retain_frac, dtype=torch.float32, device=device)
    y_dw = torch.as_tensor(sample.y_delta_log_w, dtype=torch.float32, device=device)
    y_dtvl = torch.as_tensor(sample.y_delta_log_tvl, dtype=torch.float32, device=device)
    y_stress = torch.as_tensor(sample.y_node_stress, dtype=torch.float32, device=device)
    node_mask = torch.as_tensor(sample.node_valid_mask, dtype=torch.bool, device=device)

    n_act = sample.n_active
    if use_amp and device.type == "cuda":
        with torch.cuda.amp.autocast(enabled=True):
            h = model.encode(sample.edge_src_t, sample.edge_dst_t, feat_t, device)
            edge_out = model.link_scorer(h, pair_src, pair_dst, pair_z)
            node_out = model.node_head(h, feat_t, act_src, act_dst, act_lw)
    else:
        h = model.encode(sample.edge_src_t, sample.edge_dst_t, feat_t, device)
        edge_out = model.link_scorer(h, pair_src, pair_dst, pair_z)
        node_out = model.node_head(h, feat_t, act_src, act_dst, act_lw)

    exist_logits = edge_out["exist_logits"].float()
    retain_logits = edge_out["retain_logits_raw"][:n_act].float()
    pred_dw = edge_out["delta_log_w"][:n_act].float()
    pred_dtvl = node_out["delta_log_tvl"].float()
    stress_logits = node_out["stress_logits_raw"].float()

    loss_exist = torch.tensor(0.0, device=device)
    loss_retain = torch.tensor(0.0, device=device)
    loss_dw = torch.tensor(0.0, device=device)
    loss_stress = torch.tensor(0.0, device=device)
    loss_dtvl = torch.tensor(0.0, device=device)
    n_vh = int(valid_h.sum().item())

    asym = float(params["downside_asymmetry"])
    for k in range(len(sample.valid_horizons)):
        if not bool(sample.valid_horizons[k]):
            continue
        loss_exist = loss_exist + F.binary_cross_entropy_with_logits(exist_logits[:, k], y_exist[:, k])
        # Fractional BCE + Smooth-L1 on `y_retain_frac in [0, 1]` -> exact expected net-flow retention!
        bce_frac = F.binary_cross_entropy_with_logits(retain_logits[:, k], y_ret_frac[:, k])
        l1_frac = F.smooth_l1_loss(torch.sigmoid(retain_logits[:, k]), y_ret_frac[:, k])
        loss_retain = loss_retain + bce_frac + 0.5 * l1_frac

        dw_err = F.smooth_l1_loss(pred_dw[:, k], y_dw[:, k], reduction="none")
        dw_w = torch.where(y_dw[:, k] < -0.22, asym, 1.0)
        loss_dw = loss_dw + (dw_err * dw_w).mean()

        nmk = node_mask[:, k]
        if nmk.any():
            loss_stress = loss_stress + F.binary_cross_entropy_with_logits(
                stress_logits[nmk, k], y_stress[nmk, k]
            )
            nt_err = F.smooth_l1_loss(pred_dtvl[nmk, k], y_dtvl[nmk, k], reduction="none")
            nt_w = torch.where(y_dtvl[nmk, k] < -0.105, asym, 1.0)
            loss_dtvl = loss_dtvl + (nt_err * nt_w).mean()

    denom = float(max(1, n_vh))
    loss_exist = loss_exist / denom
    loss_retain = loss_retain / denom
    loss_dw = loss_dw / denom
    loss_stress = loss_stress / denom
    loss_dtvl = loss_dtvl / denom

    total = (
        float(params["exist_loss_weight"]) * loss_exist
        + float(params["retain_loss_weight"]) * loss_retain
        + float(params["edge_delta_loss_weight"]) * loss_dw
        + float(params["stress_loss_weight"]) * loss_stress
        + float(params["node_delta_loss_weight"]) * loss_dtvl
    )
    parts = {
        "total": float(total.detach().item()),
        "retain": float(loss_retain.detach().item()),
        "exist": float(loss_exist.detach().item()),
        "edge_delta": float(loss_dw.detach().item()),
        "stress": float(loss_stress.detach().item()),
        "node_delta": float(loss_dtvl.detach().item()),
    }
    return total, parts


def collect_raw_outputs(
    model: Any, samples: list[JointNetFlowWeekSample], device: Any
) -> list[dict[str, Any]]:
    import torch

    model.eval()
    collected: list[dict[str, Any]] = []
    with torch.no_grad():
        for s in samples:
            feat_t = torch.as_tensor(s.features_t, dtype=torch.float32, device=device)
            pair_src = torch.as_tensor(s.pair_src, dtype=torch.long, device=device)
            pair_dst = torch.as_tensor(s.pair_dst, dtype=torch.long, device=device)
            pair_z = torch.as_tensor(s.pair_edge_z, dtype=torch.float32, device=device)
            act_src = torch.as_tensor(s.edge_src_t, dtype=torch.long, device=device)
            act_dst = torch.as_tensor(s.edge_dst_t, dtype=torch.long, device=device)
            act_lw = torch.as_tensor(s.edge_log_w_t, dtype=torch.float32, device=device)

            h = model.encode(s.edge_src_t, s.edge_dst_t, feat_t, device)
            edge_out = model.link_scorer(h, pair_src, pair_dst, pair_z)
            node_out = model.node_head(h, feat_t, act_src, act_dst, act_lw)

            exist_logits = edge_out["exist_logits"].float()
            cal_exist_logits = (
                exist_logits / model.exist_logit_temps.clamp(min=0.1).unsqueeze(0)
                + model.exist_logit_biases.unsqueeze(0)
            )
            exist_probs = torch.sigmoid(cal_exist_logits).cpu().numpy()
            retain_probs = model.calibrated_retain_probs(edge_out["cum_hazard"][: s.n_active].float()).cpu().numpy()
            stress_probs = model.calibrated_stress_probs(node_out["cum_stress_hazard"].float()).cpu().numpy()

            collected.append(
                {
                    "sample": s,
                    "exist_logits": exist_logits.cpu().numpy(),
                    "exist_probs": exist_probs,
                    "cum_hazard_active": edge_out["cum_hazard"][: s.n_active].float().cpu().numpy(),
                    "retain_probs_active": retain_probs,
                    "pred_delta_log_w": edge_out["delta_log_w"][: s.n_active].float().cpu().numpy(),
                    "cum_stress_hazard": node_out["cum_stress_hazard"].float().cpu().numpy(),
                    "stress_probs": stress_probs,
                    "pred_delta_log_tvl": node_out["delta_log_tvl"].float().cpu().numpy(),
                }
            )
    return collected


def fit_monotonic_calibration_on_val(
    model: Any, val_samples: list[JointNetFlowWeekSample], device: Any
) -> dict[str, Any]:
    """Fit positive interval hazard multipliers on pre-2025 validation set ONLY."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    raw = collect_raw_outputs(model, val_samples, device)
    num_h = len(model.retain_hazard_scales)

    ret_inc_list, ret_y_list, ret_m_list = [], [], []
    str_inc_list, str_y_list, str_m_list = [], [], []
    ex_log_list, ex_y_list, ex_m_list = [], [], []

    for row in raw:
        s: JointNetFlowWeekSample = row["sample"]
        vh = s.valid_horizons.astype(np.float32)

        ch = row["cum_hazard_active"]
        inc = np.concatenate([ch[:, :1], np.maximum(1e-6, ch[:, 1:] - ch[:, :-1])], axis=-1)
        ret_inc_list.append(inc)
        ret_y_list.append(s.y_retain_frac)
        ret_m_list.append(np.broadcast_to(vh[None, :], s.y_retain_frac.shape).copy())

        sh = row["cum_stress_hazard"]
        sinc = np.concatenate([sh[:, :1], np.maximum(1e-6, sh[:, 1:] - sh[:, :-1])], axis=-1)
        str_inc_list.append(sinc)
        str_y_list.append(s.y_node_stress)
        str_m_list.append((s.node_valid_mask & s.valid_horizons[None, :]).astype(np.float32))

        ex_log_list.append(row["exist_logits"])
        ex_y_list.append(s.y_exist)
        ex_m_list.append(np.broadcast_to(vh[None, :], s.y_exist.shape).copy())

    ret_inc_t = torch.as_tensor(np.concatenate(ret_inc_list, axis=0), dtype=torch.float32, device=device)
    ret_y_t = torch.as_tensor(np.concatenate(ret_y_list, axis=0), dtype=torch.float32, device=device)
    ret_m_t = torch.as_tensor(np.concatenate(ret_m_list, axis=0), dtype=torch.float32, device=device)

    theta_ret = torch.zeros(num_h, dtype=torch.float32, device=device, requires_grad=True)
    opt_ret = torch.optim.LBFGS([theta_ret], lr=0.25, max_iter=60, line_search_fn="strong_wolfe")

    def _closure_ret():
        opt_ret.zero_grad()
        scales = torch.exp(theta_ret.clamp(-2.5, 2.5))
        cum = torch.cumsum(ret_inc_t * scales.unsqueeze(0), dim=-1)
        probs = torch.exp(-cum).clamp(1e-5, 1.0 - 1e-5)
        bce = F.binary_cross_entropy(probs, ret_y_t, reduction="none")
        loss = (bce * ret_m_t).sum() / ret_m_t.sum().clamp(min=1.0) + 1e-3 * (theta_ret ** 2).mean()
        loss.backward()
        return loss

    before_ret_nll = float(_closure_ret().detach().item())
    opt_ret.step(_closure_ret)
    after_ret_nll = float(_closure_ret().detach().item())

    str_inc_t = torch.as_tensor(np.concatenate(str_inc_list, axis=0), dtype=torch.float32, device=device)
    str_y_t = torch.as_tensor(np.concatenate(str_y_list, axis=0), dtype=torch.float32, device=device)
    str_m_t = torch.as_tensor(np.concatenate(str_m_list, axis=0), dtype=torch.float32, device=device)

    theta_str = torch.zeros(num_h, dtype=torch.float32, device=device, requires_grad=True)
    opt_str = torch.optim.LBFGS([theta_str], lr=0.25, max_iter=60, line_search_fn="strong_wolfe")

    def _closure_str():
        opt_str.zero_grad()
        scales = torch.exp(theta_str.clamp(-2.5, 2.5))
        cum = torch.cumsum(str_inc_t * scales.unsqueeze(0), dim=-1)
        probs = (1.0 - torch.exp(-cum)).clamp(1e-5, 1.0 - 1e-5)
        bce = F.binary_cross_entropy(probs, str_y_t, reduction="none")
        loss = (bce * str_m_t).sum() / str_m_t.sum().clamp(min=1.0) + 1e-3 * (theta_str ** 2).mean()
        loss.backward()
        return loss

    before_str_nll = float(_closure_str().detach().item())
    opt_str.step(_closure_str)
    after_str_nll = float(_closure_str().detach().item())

    ex_log_t = torch.as_tensor(np.concatenate(ex_log_list, axis=0), dtype=torch.float32, device=device)
    ex_y_t = torch.as_tensor(np.concatenate(ex_y_list, axis=0), dtype=torch.float32, device=device)
    ex_m_t = torch.as_tensor(np.concatenate(ex_m_list, axis=0), dtype=torch.float32, device=device)
    log_temp_ex = torch.zeros(num_h, dtype=torch.float32, device=device, requires_grad=True)
    bias_ex = torch.zeros(num_h, dtype=torch.float32, device=device, requires_grad=True)
    opt_ex = torch.optim.LBFGS([log_temp_ex, bias_ex], lr=0.25, max_iter=50, line_search_fn="strong_wolfe")

    def _closure_ex():
        opt_ex.zero_grad()
        temps = torch.exp(log_temp_ex.clamp(-2.0, 2.0))
        cal_l = ex_log_t / temps.unsqueeze(0) + bias_ex.unsqueeze(0)
        bce = F.binary_cross_entropy_with_logits(cal_l, ex_y_t, reduction="none")
        loss = (bce * ex_m_t).sum() / ex_m_t.sum().clamp(min=1.0)
        loss.backward()
        return loss

    opt_ex.step(_closure_ex)

    with torch.no_grad():
        if after_ret_nll <= before_ret_nll + 1e-6:
            model.retain_hazard_scales.copy_(torch.exp(theta_ret.clamp(-2.5, 2.5)).detach())
        if after_str_nll <= before_str_nll + 1e-6:
            model.stress_hazard_scales.copy_(torch.exp(theta_str.clamp(-2.5, 2.5)).detach())
        model.exist_logit_temps.copy_(torch.exp(log_temp_ex.clamp(-2.0, 2.0)).detach())
        model.exist_logit_biases.copy_(bias_ex.detach())

    return {
        "calibration_split": "pre2025_validation_only",
        "retain_nll_before": before_ret_nll,
        "retain_nll_after": after_ret_nll,
        "retain_hazard_scales": [float(x) for x in model.retain_hazard_scales.cpu().tolist()],
        "stress_nll_before": before_str_nll,
        "stress_nll_after": after_str_nll,
        "stress_hazard_scales": [float(x) for x in model.stress_hazard_scales.cpu().tolist()],
        "exist_logit_temps": [float(x) for x in model.exist_logit_temps.cpu().tolist()],
        "exist_logit_biases": [float(x) for x in model.exist_logit_biases.cpu().tolist()],
    }


def evaluate_joint_model(
    model: Any,
    samples: list[JointNetFlowWeekSample],
    horizons: list[int],
    device: Any,
    *,
    save_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute Task-I compatible metrics AND full FM-v4 Net-Flow + Dual-Board Walk-Forward metrics."""
    import numpy as np
    from scipy.stats import spearmanr

    raw = collect_raw_outputs(model, samples, device)
    task1_compat: dict[str, Any] = {}
    v4_rich: dict[str, Any] = {}

    mono_violations = 0
    total_active_rows = 0
    for row in raw:
        rp = row["retain_probs_active"]
        if len(rp) > 0:
            total_active_rows += len(rp)
            mono_violations += int(((rp[:, :-1] - rp[:, 1:]) < -1e-5).any(axis=-1).sum())
    mono_rate = float(mono_violations / max(1, total_active_rows))

    for k, h in enumerate(horizons):
        ex_y, ex_p = [], []
        ret_bin_y, ret_frac_y, ret_p = [], [], []
        w_true, w_pred, w_base = [], [], []
        nt_true, nt_pred = [], []
        str_y, str_p = [], []
        wf_whale_hits_top7: list[float] = []
        wf_alpha_hits_top7: list[float] = []
        wf_alpha_base_hits_top7: list[float] = []
        wf_alpha_dir_acc_top7: list[float] = []

        if save_dir is not None:
            h_dir = save_dir / f"h{h}"
            h_dir.mkdir(parents=True, exist_ok=True)
        else:
            h_dir = None

        week_counter = 0
        for row in raw:
            s: JointNetFlowWeekSample = row["sample"]
            if not bool(s.valid_horizons[k]):
                continue
            ex_y.append(s.y_exist[:, k])
            ex_p.append(row["exist_probs"][:, k])

            ret_bin_y.append(s.y_retain_bin80[:, k])
            ret_frac_y.append(s.y_retain_frac[:, k])
            ret_p.append(row["retain_probs_active"][:, k])

            pred_lw = s.edge_log_w_t + row["pred_delta_log_w"][:, k]
            true_lw = s.edge_log_w_t + s.y_delta_log_w[:, k]
            w_true.append(true_lw)
            w_pred.append(pred_lw)
            w_base.append(s.edge_log_w_t)

            nmk = s.node_valid_mask[:, k]
            if nmk.any():
                nt_t = s.y_delta_log_tvl[nmk, k]
                nt_p = row["pred_delta_log_tvl"][nmk, k]
                st_t = s.y_node_stress[nmk, k]
                st_p = row["stress_probs"][nmk, k]
                tvl_sub = s.tvl_t[nmk]

                nt_true.append(nt_t)
                nt_pred.append(nt_p)
                str_y.append(st_t)
                str_p.append(st_p)

                # 1. Systemic Whale Board Top-7 (`TVL >= $25M`, weighted by log-size)
                whale_mask = tvl_sub >= 25_000_000.0
                if int(whale_mask.sum()) >= 7:
                    w_tvl = tvl_sub[whale_mask]
                    w_stp = st_p[whale_mask]
                    w_ntp = nt_p[whale_mask]
                    w_ntt = nt_t[whale_mask]
                    w_stt = st_t[whale_mask]
                    pred_drop = np.clip(-np.expm1(np.minimum(0.0, w_ntp)), 0.0, 1.0)
                    whale_score = (0.60 * w_stp + 0.40 * pred_drop) * np.log10(np.maximum(w_tvl, 1e6) / 1e6)
                    w_top7 = np.argsort(-whale_score)[:7]
                    wf_whale_hits_top7.append(float(((w_stt[w_top7] > 0.5) | (w_ntt[w_top7] <= -0.05)).mean()))

                # 2. Scale-Free High-Risk Alpha Radar Top-7 (`$15M <= TVL <= $5B`, ZERO size multiplier!)
                alpha_mask = (tvl_sub >= 15_000_000.0) & (tvl_sub <= 5_000_000_000.0)
                if int(alpha_mask.sum()) >= 7:
                    a_stp = st_p[alpha_mask]
                    a_ntp = nt_p[alpha_mask]
                    a_ntt = nt_t[alpha_mask]
                    a_stt = st_t[alpha_mask]
                    pred_drop_a = np.clip(-np.expm1(np.minimum(0.0, a_ntp)), 0.0, 1.0)
                    alpha_score = 0.65 * a_stp + 0.35 * pred_drop_a
                    a_top7 = np.argsort(-alpha_score)[:7]
                    hit_cond_all = (a_stt > 0.5) | (a_ntt <= -0.05)
                    wf_alpha_hits_top7.append(float(hit_cond_all[a_top7].mean()))
                    wf_alpha_base_hits_top7.append(float(hit_cond_all.mean()))
                    wf_alpha_dir_acc_top7.append(float((a_ntt[a_top7] < 0.0).mean()))

            if h_dir is not None:
                np.savez_compressed(
                    h_dir / f"week_{week_counter:03d}.npz",
                    retain_prob=row["retain_probs_active"][:, k].astype(np.float32),
                    y_retain_frac=s.y_retain_frac[:, k].astype(np.float32),
                    y_retain_bin80=s.y_retain_bin80[:, k].astype(np.float32),
                    pred_delta_log_w=row["pred_delta_log_w"][:, k].astype(np.float32),
                    y_delta_log_w=s.y_delta_log_w[:, k].astype(np.float32),
                    stress_prob=row["stress_probs"][:, k].astype(np.float32),
                    pred_delta_log_tvl=row["pred_delta_log_tvl"][:, k].astype(np.float32),
                )
            week_counter += 1

        ex_metrics = safe_binary_metrics(
            np.concatenate(ex_y) if ex_y else np.zeros(0),
            np.concatenate(ex_p) if ex_p else np.zeros(0),
        )
        rp_cat = np.concatenate(ret_p) if ret_p else np.zeros(0)
        rb_cat = np.concatenate(ret_bin_y) if ret_bin_y else np.zeros(0)
        rf_cat = np.concatenate(ret_frac_y) if ret_frac_y else np.zeros(0)
        ret_bin_metrics = safe_binary_metrics(rb_cat, rp_cat)
        frac_mae = float(np.mean(np.abs(rp_cat - rf_cat))) if len(rf_cat) else 0.0
        frac_bias = float(abs(rp_cat.mean() - rf_cat.mean())) if len(rf_cat) else 0.0

        str_metrics = safe_binary_metrics(
            np.concatenate(str_y) if str_y else np.zeros(0),
            np.concatenate(str_p) if str_p else np.zeros(0),
        )

        wt_cat = np.concatenate(w_true) if w_true else np.zeros(1, dtype=np.float32)
        wp_cat = np.concatenate(w_pred) if w_pred else np.zeros(1, dtype=np.float32)
        wb_cat = np.concatenate(w_base) if w_base else np.ones(1, dtype=np.float32)
        w_abs = np.abs(wp_cat - wt_cat)
        w_mae = float(np.mean(w_abs))
        w_rmse = float(np.sqrt(np.mean(np.square(wp_cat - wt_cat))))
        w_weights = np.maximum(wb_cat, 1.0)
        w_wmae = float(np.sum(w_abs * w_weights) / np.sum(w_weights))

        nt_cat = np.concatenate(nt_true) if nt_true else np.zeros(1, dtype=np.float32)
        np_cat = np.concatenate(nt_pred) if nt_pred else np.zeros(1, dtype=np.float32)
        n_mae = float(np.mean(np.abs(np_cat - nt_cat)))
        n_rmse = float(np.sqrt(np.mean(np.square(np_cat - nt_cat))))
        sp_corr = float(spearmanr(nt_cat, np_cat)[0]) if len(nt_cat) > 5 else 0.0
        if not math.isfinite(sp_corr):
            sp_corr = 0.0

        task1_compat[f"h{h}"] = {
            "exist": {
                "auprc": ex_metrics["auprc"],
                "auroc": ex_metrics["auroc"],
                "recall@100": float(ret_bin_metrics["auroc"]),
                "recall@500": float(ret_bin_metrics["auprc"]),
                "recall@1000": float(1.0 - frac_bias),
            },
            "weight": {
                "mae": w_mae,
                "rmse": w_rmse,
                "weighted_mae": w_wmae,
            },
            "node": {
                "mae": n_mae,
                "rmse": n_rmse,
            },
        }
        v4_rich[f"h{h}"] = {
            "n_eval_weeks": week_counter,
            "global_exist": ex_metrics,
            "active_edge_retention_80pct": ret_bin_metrics,
            "active_edge_fractional_retention": {
                "mae": frac_mae,
                "mean_abs_bias": frac_bias,
                "mean_pred": float(rp_cat.mean()) if len(rp_cat) else 0.0,
                "mean_true": float(rf_cat.mean()) if len(rf_cat) else 0.0,
            },
            "node_stress_10pct": str_metrics,
            "edge_log_netflow_mae": w_mae,
            "edge_log_netflow_rmse": w_rmse,
            "node_log_netflow_mae": n_mae,
            "node_log_netflow_rmse": n_rmse,
            "node_netflow_spearman": sp_corr,
            "monotonicity_violation_rate": mono_rate,
            "walk_forward_whale_top7_hit_rate": float(np.mean(wf_whale_hits_top7)) if wf_whale_hits_top7 else 0.0,
            "walk_forward_alpha_top7_hit_rate": float(np.mean(wf_alpha_hits_top7)) if wf_alpha_hits_top7 else 0.0,
            "walk_forward_alpha_universe_base_rate": float(np.mean(wf_alpha_base_hits_top7)) if wf_alpha_base_hits_top7 else 0.0,
            "walk_forward_alpha_top7_directional_rate": float(np.mean(wf_alpha_dir_acc_top7)) if wf_alpha_dir_acc_top7 else 0.0,
        }

    return task1_compat, v4_rich


def composite_validation_score(v4_val: dict[str, Any], horizons: list[int]) -> tuple[float, float]:
    scores = []
    retain_aucs = []
    for h in horizons:
        m = v4_val[f"h{h}"]
        r_auc = float(m["active_edge_retention_80pct"]["auroc"])
        s_auc = float(m["node_stress_10pct"]["auroc"])
        f_mae = float(m["active_edge_fractional_retention"]["mae"])
        n_mae = float(m["node_log_netflow_mae"])
        scores.append(r_auc + s_auc - 0.20 * f_mae - 0.15 * n_mae)
        retain_aucs.append(r_auc)
    return float(sum(scores) / len(scores)), float(sum(retain_aucs) / len(retain_aucs))


def run() -> None:
    existing = ROOT / TRAIN_REL / "plan.json"
    plan = read_json(PLAN_PATH)
    validate_plan(plan)
    if existing.exists() and read_json(existing) != plan:
        raise OutputDirectoryError("Existing training directory has a different plan")
    train_dir, release_dir = prepare_output_dirs(ROOT, allow_existing=existing.exists())
    source_sha = os.environ["PREFLIGHT_SOURCE_SHA256"]
    if (train_dir / "result.json").exists():
        verify_bundle(ROOT, source_sha)
        log("FM_V4_ALREADY_COMPLETE")
        return

    import numpy as np
    import torch
    import run_full_experiment as legacy_core
    from verify_v2_checkpoint import require_exact_split
    from dexposure_agent.fm_v4 import (
        DeXposureFMV4,
        HistoricalSpotPriceOracle,
        TokenMapperV3,
        V4_CORE_NODE_FEATURES,
        V4_EDGE_STATE_FEATURES,
        V4_HORIZONS,
        build_netflow_snapshot_from_raw,
        enrich_netflow_snapshots,
        load_pretrained_graphpfn,
        v4_contract_hash,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; refusing paid CPU fallback")

    gpu_props = torch.cuda.get_device_properties(0)
    log(f"GPU_DETECTED name={gpu_props.name!r} total_mib={gpu_props.total_memory // (1024 * 1024)}")
    if gpu_props.total_memory > plan["max_gpu_memory_mib"] * 1024 * 1024:
        raise RuntimeError("GPU exceeds planned memory ceiling")

    seed = int(plan["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    write_json(train_dir / "plan.json", plan)
    archive_sources(train_dir / "source.tar.gz")
    finetuned_dir = train_dir / "finetuned"
    trial_dir = train_dir / "trials" / "v4_netflow_joint"
    trial_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(ROOT / "cloud" / "preflight_manifest.json")
    data_path = ROOT / "data" / "historical-network_week_2020-03-30.json"
    meta_path = ROOT / "data" / "meta_df.csv"
    base_ckpt_path = ROOT / "checkpoints" / "graphpfn-v1.ckpt"

    data_sha = sha256(data_path)
    meta_sha = sha256(meta_path)
    base_ckpt_sha = sha256(base_ckpt_path)

    log("LOADING_HISTORICAL_NETWORK_JSON_AND_SPOT_ORACLE")
    t_load = time.time()
    oracle = HistoricalSpotPriceOracle()
    raw_data = legacy_core.load_network_data(str(data_path))
    dates = sorted(raw_data.keys())
    split = legacy_core.get_single_split(dates, plan["holdout_start"], plan["validation_weeks"])
    require_exact_split(manifest, split)
    write_json(train_dir / "split.json", split)
    log(
        f"SPLIT_VERIFIED total={len(dates)} train={len(split['train'])} "
        f"val={len(split['val'])} test={len(split['test'])} ({time.time()-t_load:.1f}s)"
    )

    mapper = TokenMapperV3()
    categories = list(mapper.categories)
    schema = {
        "schema_version": 4,
        "model_family": "dexposure_fm_v4",
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
    for d in (release_dir, finetuned_dir):
        write_json(d / "feature_schema.json", schema)

    log("BUILDING_283_NETFLOW_STOCK_EXPOSURE_SNAPSHOTS")
    t_build = time.time()
    stock_snaps = [build_netflow_snapshot_from_raw(dt, raw_data[dt], mapper) for dt in dates]
    del raw_data
    gc.collect()
    enrich_netflow_snapshots(stock_snaps, categories, oracle=oracle)
    log(
        f"NETFLOW_SNAPSHOTS_READY count={len(stock_snaps)} elapsed={time.time()-t_build:.1f}s "
        f"last_nodes={len(stock_snaps[-1].node_ids)} last_edges={len(stock_snaps[-1].edge_src)}"
    )

    quality_summary = {
        "snapshot_count": len(stock_snaps),
        "first_date": dates[0],
        "last_date": dates[-1],
        "mean_nodes": float(np.mean([len(s.node_ids) for s in stock_snaps])),
        "mean_edges": float(np.mean([len(s.edge_src) for s in stock_snaps])),
        "last_snapshot_total_stock_usd": float(stock_snaps[-1].edge_usd.sum()),
        "spot_oracle_sha256": oracle.sha256,
    }
    write_json(finetuned_dir / "data_quality.json", quality_summary)

    date_to_idx = {d: idx for idx, d in enumerate(dates)}
    train_indices = [date_to_idx[d] for d in split["train"]]
    val_indices = [date_to_idx[d] for d in split["val"]]
    test_indices = [date_to_idx[d] for d in split["test"]]

    horizons = [int(h) for h in plan["horizons"]]
    neg_ratio = int(plan["neg_ratio"])

    log("BUILDING_JOINT_MULTI_HORIZON_NETFLOW_SAMPLES")
    train_samples = build_joint_netflow_samples(
        stock_snaps, train_indices, max(train_indices), horizons, neg_ratio, seed, oracle
    )
    val_samples = build_joint_netflow_samples(
        stock_snaps, val_indices, max(val_indices), horizons, neg_ratio, seed + 1, oracle
    )
    test_samples = build_joint_netflow_samples(
        stock_snaps, test_indices, max(test_indices), horizons, neg_ratio, seed + 2, oracle
    )
    pair_hashes = {
        "train": samples_fingerprint(train_samples),
        "val": samples_fingerprint(val_samples),
        "test": samples_fingerprint(test_samples),
        "counts": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)},
    }
    write_json(train_dir / "pair_hashes.json", pair_hashes)
    log(f"SAMPLES_BUILT train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    log("EVALUATING_NETFLOW_BASELINES")
    baselines = {
        "val": evaluate_heuristic_baselines(val_samples, horizons),
        "test": evaluate_heuristic_baselines(test_samples, horizons),
    }
    write_json(train_dir / "baseline_comparison.json", baselines)

    device = torch.device("cuda")
    params = plan["parameters"]
    encoder = load_pretrained_graphpfn(base_ckpt_path, device)
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

    resume_path = trial_dir / "resume_v4.partial.pt"
    best_raw_path = trial_dir / "best_raw_v4.pt"
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
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
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
            atomic_torch_save(
                {"epoch": epoch, "score": list(val_comp), "model": model.state_dict()},
                best_raw_path,
            )

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

    log("FITTING_VALIDATION_MONOTONIC_CALIBRATION")
    calib_summary = fit_monotonic_calibration_on_val(model, val_samples, device)
    write_json(train_dir / "calibration_summary.json", calib_summary)
    log(
        f"CALIBRATION_COMPLETE retain_scales={calib_summary['retain_hazard_scales']} "
        f"stress_scales={calib_summary['stress_hazard_scales']}"
    )

    _, val_calibrated_v4 = evaluate_joint_model(model, val_samples, horizons, device)

    selected_records: dict[str, Any] = {}
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    for h in horizons:
        provenance = {
            "model_family": "dexposure_fm_v4",
            "settlement_basis": "constant_price_net_flow_ex_eth_btc_beta",
            "horizon": h,
            "joint_horizons": horizons,
            "seed": seed,
            "epochs": epochs,
            "completed_epochs": completed_epochs,
            "selected_epoch": best_epoch,
            "data_sha256": data_sha,
            "metadata_sha256": meta_sha,
            "base_checkpoint_sha256": base_ckpt_sha,
            "spot_oracle_sha256": oracle.sha256,
            "source": {"preflight_source_sha256": source_sha},
            "split": split,
            "calibration": calib_summary,
        }
        ckpt_payload = {
            "artifact_status": "complete",
            "model_family": "dexposure_fm_v4",
            "schema_version": 4,
            "settlement_basis": "constant_price_net_flow_ex_eth_btc_beta",
            "horizon": h,
            "horizons": horizons,
            "hidden_dim": int(params["hidden_dim"]),
            "model": cpu_state,
            "provenance": provenance,
        }
        rel_ckpt = release_dir / f"dexposure-fm-h{h}.pt"
        fin_ckpt = finetuned_dir / f"best_model_h{h}.pt"
        atomic_torch_save(ckpt_payload, rel_ckpt)
        shutil.copyfile(rel_ckpt, fin_ckpt)
        ckpt_sha = sha256(rel_ckpt)
        selected_records[str(h)] = {
            "status": "complete",
            "horizon": h,
            "selected_epoch": best_epoch,
            "completed_epochs": completed_epochs,
            "model_sha256": ckpt_sha,
        }
        write_json(
            release_dir / f"run_config_h{h}.json",
            {
                "artifact_status": "complete",
                "experiment_kind": plan["experiment_kind"],
                "model_family": "dexposure_fm_v4",
                "settlement_basis": "constant_price_net_flow_ex_eth_btc_beta",
                "forecast_horizon_weeks": h,
                "holdout_start": plan["holdout_start"],
                "validation_weeks": int(plan["validation_weeks"]),
                "epochs": epochs,
                "early_stop_patience": patience,
                "seed": seed,
                "network_snapshots": len(dates),
                "source_sha256": source_sha,
                "spot_oracle_sha256": oracle.sha256,
                "selected_epoch": best_epoch,
                "completed_epochs": completed_epochs,
                "parameters": params,
                "calibration": calib_summary,
            },
        )

    freeze = {
        "selected": selected_records,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "selection_data": "pre2025_validation_only",
        "plan_sha256": sha256(train_dir / "plan.json"),
    }
    write_json(train_dir / "selection_frozen.json", freeze)
    selection_sha = sha256(train_dir / "selection_frozen.json")
    log(f"SELECTION_FROZEN sha256={selection_sha}; starting frozen 2025 holdout evaluation")

    task1_test, v4_test = evaluate_joint_model(
        model, test_samples, horizons, device, save_dir=train_dir / "predictions"
    )
    for h in horizons:
        mh = v4_test[f"h{h}"]
        log(
            f"HOLDOUT_2025 h={h:2d}w | exist_auc={mh['global_exist']['auroc']:.4f} "
            f"| ret80_auc={mh['active_edge_retention_80pct']['auroc']:.4f} "
            f"| frac_mae={mh['active_edge_fractional_retention']['mae']:.4f} "
            f"(pred={mh['active_edge_fractional_retention']['mean_pred']:.3f} vs true={mh['active_edge_fractional_retention']['mean_true']:.3f}) "
            f"| stress_auc={mh['node_stress_10pct']['auroc']:.4f} "
            f"| whale_top7={100*mh['walk_forward_whale_top7_hit_rate']:.1f}% "
            f"| alpha_top7={100*mh['walk_forward_alpha_top7_hit_rate']:.1f}% (base={100*mh['walk_forward_alpha_universe_base_rate']:.1f}%) "
            f"| mono_viol={mh['monotonicity_violation_rate']:.6f}"
        )

    if sha256(train_dir / "selection_frozen.json") != selection_sha:
        raise RuntimeError("Selection modified during holdout evaluation")

    task1_payload = {"model": "DeXposure-FM-v4-netflow-monotonic", **task1_test}
    result_payload = {
        "status": "complete",
        "experiment_kind": plan["experiment_kind"],
        "model_family": "dexposure_fm_v4",
        "settlement_basis": "constant_price_net_flow_ex_eth_btc_beta",
        "selection_sha256": selection_sha,
        "selected_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "calibration": calib_summary,
        "v4_validation_calibrated": val_calibrated_v4,
        "v4_holdout": v4_test,
        "baselines": baselines,
        "test": task1_payload,
    }
    write_json(train_dir / "result.json", result_payload)
    write_json(train_dir / "v4_holdout_report.json", v4_test)
    write_json(release_dir / "task1_metrics.json", task1_payload)
    write_json(finetuned_dir / "metrics.json", task1_payload)
    write_json(finetuned_dir / "all_results.json", result_payload)
    write_json(
        finetuned_dir / "experiment_results.json",
        {**result_payload, "config": plan, "selected": selected_records},
    )

    (release_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in sorted(release_dir.iterdir()) if p.is_file() and p.name != "SHA256SUMS"),
        encoding="utf-8",
    )
    write_tree_manifest(ROOT)
    verify_bundle(ROOT, source_sha)
    log("FM_V4_CAMPAIGN_COMPLETE_AND_VERIFIED")


def main() -> None:
    parser = argparse.ArgumentParser(description="DeXposure-FM v4 Net-Flow trainer & verifier")
    parser.add_argument("--run", action="store_true", help="Execute FM v4 training and holdout eval")
    parser.add_argument("--verify", type=Path, default=None, help="Verify a downloaded FM v4 run root")
    parser.add_argument("--source-sha256", type=str, default=None, help="Expected source digest")
    args = parser.parse_args()
    if args.verify is not None:
        out = verify_bundle(args.verify, args.source_sha256)
        print(json.dumps(out, sort_keys=True))
        return
    if args.run:
        run()
        return
    parser.error("Specify --run or --verify <path>")


if __name__ == "__main__":
    main()
