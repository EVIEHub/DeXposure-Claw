"""DeXposure-FM v3: Stock-Exposure & Calibrated Monotonic Drawdown Foundation Model.

Fixes the three foundational flaws of V2:
  1. Balance-Sheet Stock Graph (`creditor holder -> debtor issuer/conduit`, weight = actual USD
     stock holding `v_{p,t}^q`) instead of sign-flipping weekly flow diffs `|\\Delta S|`.
  2. Explicit 12-dim edge-state + rolling momentum features alongside GraphPFN (`LimiX-16M`)
     structural embeddings, solving the `0.508` active-edge conditional AUROC bottleneck.
  3. Single unified multi-horizon (`h in {1, 4, 8, 12}`) architecture with strictly non-decreasing
     cumulative withdrawal / stress hazards (`p_retain(1w) >= p_retain(4w) >= p_retain(8w) >= p_retain(12w)`
     by mathematical construction) and validation-only Platt calibration (`pos_weight = 1.0` on active
     retention, zero prior distortion).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_DISABLE_GRAPHBOLT", "1")

from dexposure_agent.types import Edge, GraphSnapshot, NodeFeatures

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPING_BUNDLE = Path(__file__).resolve().parent / "resources" / "token_mapping_v3.json"
DEFAULT_V3_CKPT_DIR = Path(
    os.environ.get(
        "DEXPOSURE_FM_V3_CKPT_DIR",
        str(_PROJECT_ROOT / "checkpoints" / "dexposure-fm-v3"),
    )
)

V3_HORIZONS: tuple[int, ...] = (1, 4, 8, 12)
HORIZON_TO_IDX: dict[int, int] = {h: idx for idx, h in enumerate(V3_HORIZONS)}

# Economic retention / stress thresholds
EDGE_RETENTION_RATIO: float = 0.80  # Position retains >= 80% of USD stock (i.e. <20% withdrawal)
NODE_STRESS_RATIO: float = 0.90     # Protocol TVL drops to <= 90% of current TVL (>=10% contraction)

# Canonical stock graph scale constants (identical in 2020-2025 training & 2026 live serving)
DEFAULT_MIN_PROTOCOL_TVL_USD: float = 500_000.0
DEFAULT_MAX_PROTOCOLS: int = 1200
DEFAULT_MIN_EDGE_USD: float = 10_000.0
DEFAULT_MIN_CONDUIT_STOCK_USD: float = 500_000.0
DEFAULT_MAX_CONDUITS: int = 350

DISALLOWED_ISSUER_CATEGORIES: frozenset[str] = frozenset({"Chain", "Oracle", "Services"})

V3_CORE_NODE_FEATURES: tuple[str, ...] = (
    "log_tvl",
    "log_out_stock",
    "log_in_stock",
    "out_deg_scale",
    "in_deg_scale",
    "out_deg_pct",
    "in_deg_pct",
    "asset_hhi",
    "liability_hhi",
    "net_leverage_ratio",
    "conduit_share_held",
    "protocol_share_held",
    "is_conduit",
    "tvl_mom_1w",
    "tvl_mom_4w",
    "tvl_mom_8w",
    "tvl_vol_4w",
    "tvl_drawdown_4w",
    "out_stock_mom_1w",
    "out_stock_mom_4w",
    "in_stock_mom_1w",
    "in_stock_mom_4w",
    "active_age_8w",
    "out_edge_retention_1w",
)

V3_EDGE_STATE_FEATURES: tuple[str, ...] = (
    "is_active_t",
    "log_w_t",
    "share_of_holder",
    "share_of_issuer",
    "delta_log_w_1w",
    "delta_log_w_4w",
    "edge_age_4w",
    "edge_vol_4w",
    "holder_tvl_mom_1w",
    "holder_tvl_mom_4w",
    "issuer_inflow_mom_1w",
    "issuer_inflow_mom_4w",
)


def v3_contract_hash(categories: list[str] | tuple[str, ...]) -> str:
    payload = {
        "schema_version": 3,
        "horizons": list(V3_HORIZONS),
        "edge_retention_ratio": EDGE_RETENTION_RATIO,
        "node_stress_ratio": NODE_STRESS_RATIO,
        "min_protocol_tvl_usd": DEFAULT_MIN_PROTOCOL_TVL_USD,
        "max_protocols": DEFAULT_MAX_PROTOCOLS,
        "min_edge_usd": DEFAULT_MIN_EDGE_USD,
        "min_conduit_stock_usd": DEFAULT_MIN_CONDUIT_STOCK_USD,
        "core_node_features": list(V3_CORE_NODE_FEATURES),
        "edge_state_features": list(V3_EDGE_STATE_FEATURES),
        "categories": list(categories),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TokenMapperV3:
    """Cleaned token-to-issuer/conduit mapper with TVL plausibility protection."""

    def __init__(self, bundle_path: Path | str = DEFAULT_MAPPING_BUNDLE) -> None:
        self.bundle_path = Path(bundle_path)
        raw = json.loads(self.bundle_path.read_text(encoding="utf-8"))
        self.categories: list[str] = list(raw["categories"])
        self.force_conduit: dict[str, str] = dict(raw["force_conduit"])
        self.verified_protocol_issuers: dict[str, str] = dict(raw["verified_protocol_issuers"])
        self.token_to_id: dict[str, str] = dict(raw["token_to_id"])
        self.id_to_info: dict[str, dict[str, str]] = {
            str(k): {"name": str(v.get("name", k)), "category": str(v.get("category", "Unknown"))}
            for k, v in raw["id_to_info"].items()
        }

    @staticmethod
    def clean_symbol(raw_symbol: str) -> str:
        sym = str(raw_symbol or "").split("(")[0].strip().upper()
        if sym.startswith("COINGECKO:"):
            sym = sym.split(":", 1)[1].strip()
        return sym

    def register_live_protocols(self, protocols: list[dict[str, Any]]) -> None:
        """Register live DeFiLlama metadata while enforcing TVL & category guards."""
        sorted_protos = sorted(
            protocols,
            key=lambda p: -float(p.get("tvl") or 0.0) if isinstance(p, dict) else 0.0,
        )
        for p in sorted_protos:
            if not isinstance(p, dict):
                continue
            pid = str(p.get("id") or "").strip()
            if not pid:
                continue
            name = str(p.get("name") or pid).strip()
            cat = str(p.get("category") or "Unknown").strip()
            tvl = float(p.get("tvl") or 0.0)
            if cat not in self.categories:
                cat_mapped = self.id_to_info.get(pid, {}).get("category", "Unknown")
            else:
                cat_mapped = cat
            self.id_to_info.setdefault(pid, {"name": name, "category": cat_mapped})
            sym = self.clean_symbol(str(p.get("symbol") or ""))
            if (
                sym
                and sym != "-"
                and sym not in self.force_conduit
                and sym not in self.verified_protocol_issuers
                and sym not in self.token_to_id
            ):
                is_legacy = name.endswith(" V1") or name.endswith(" Arc")
                if cat not in DISALLOWED_ISSUER_CATEGORIES and not is_legacy and tvl >= 10_000_000.0:
                    self.token_to_id[sym] = pid
                else:
                    self.token_to_id[sym] = sym

    def resolve_token_issuer(
        self,
        raw_symbol: str,
        *,
        protocol_tvls: dict[str, float] | None = None,
        token_totals: dict[str, float] | None = None,
    ) -> str | None:
        sym = self.clean_symbol(raw_symbol)
        if not sym or sym in {"UNKNOWN", "OTHER", "LP", "NAN", "-"} or "LP" in sym.split("-"):
            return None
        if sym == "USDE":
            return "USDE"
        if sym in self.force_conduit:
            return self.force_conduit[sym]
        if sym in self.verified_protocol_issuers:
            target = self.verified_protocol_issuers[sym]
            if protocol_tvls is None or protocol_tvls.get(target, 0.0) >= DEFAULT_MIN_PROTOCOL_TVL_USD:
                return target
            return sym

        target = self.token_to_id.get(sym, sym)
        if target in {"UNKNOWN", "OTHER", "LP"}:
            return None
        if target.isdigit():
            info = self.id_to_info.get(target, {})
            nm = info.get("name", "")
            cat = info.get("category", "Unknown")
            if cat in DISALLOWED_ISSUER_CATEGORIES or nm.endswith(" V1") or nm.endswith(" Arc"):
                return sym
            if protocol_tvls is not None:
                ptvl = float(protocol_tvls.get(target, 0.0))
                held = float((token_totals or {}).get(sym, 0.0))
                if ptvl < DEFAULT_MIN_PROTOCOL_TVL_USD or (held >= 10_000_000.0 and ptvl < 0.10 * held):
                    return sym
            return target
        return target.upper()

    def protocol_name(self, node_id: str) -> str:
        nid = str(node_id)
        info = self.id_to_info.get(nid)
        if info and info.get("name"):
            return info["name"]
        return nid

    def protocol_category(self, node_id: str) -> str:
        nid = str(node_id)
        info = self.id_to_info.get(nid)
        if info and info.get("category") and info["category"] in self.categories:
            return info["category"]
        return "Unknown"


@dataclass
class StockSnapshotData:
    """Rich stock-exposure snapshot representation shared by training and live inference."""

    date: str
    node_ids: list[str]
    node_index: dict[str, int]
    tvl_usd: np.ndarray          # [N] float64, 0.0 for pure token conduits
    is_conduit: np.ndarray       # [N] bool
    categories: list[str]        # [N] str
    edge_src: np.ndarray         # [E] int64 (creditor holder)
    edge_dst: np.ndarray         # [E] int64 (debtor issuer / conduit)
    edge_usd: np.ndarray         # [E] float64 (stock USD exposure)
    edge_map: dict[tuple[str, str], float]
    features: np.ndarray | None = None  # [N, F] float32 after enrich_stock_snapshots


def build_stock_snapshot_from_holdings(
    date_str: str,
    protocol_holdings: dict[str, dict[str, float]],
    mapper: TokenMapperV3,
    *,
    explicit_sizes: dict[str, float] | None = None,
    min_protocol_tvl_usd: float = DEFAULT_MIN_PROTOCOL_TVL_USD,
    max_protocols: int = DEFAULT_MAX_PROTOCOLS,
    min_edge_usd: float = DEFAULT_MIN_EDGE_USD,
    min_conduit_stock_usd: float = DEFAULT_MIN_CONDUIT_STOCK_USD,
) -> StockSnapshotData:
    """Construct a canonical balance-sheet Stock Exposure Graph (`holder -> issuer/conduit`)."""
    proto_tvl: dict[str, float] = {}
    cleaned_holdings: dict[str, dict[str, float]] = {}

    for raw_pid, comp in protocol_holdings.items():
        pid = str(raw_pid).strip()
        if not pid or pid in {"UNKNOWN", "OTHER", "LP"}:
            continue
        tok_map: dict[str, float] = defaultdict(float)
        for raw_sym, raw_val in (comp or {}).items():
            val = float(raw_val or 0.0)
            if math.isfinite(val) and val > 0.0:
                sym = mapper.clean_symbol(raw_sym)
                if sym:
                    tok_map[sym] += val
        comp_sum = sum(tok_map.values())
        size_val = float((explicit_sizes or {}).get(pid, 0.0) or 0.0)
        tvl = max(size_val, comp_sum)
        # Exclude pure symbol conduits that have 0 TVL in raw historical snapshots
        if not pid.isdigit() and size_val <= 0.0 and comp_sum <= 0.0:
            continue
        if tvl >= min_protocol_tvl_usd:
            proto_tvl[pid] = tvl
            cleaned_holdings[pid] = dict(tok_map)

    # Rank active protocols deterministically by descending TVL
    ranked_pids = [
        pid
        for pid, _ in sorted(proto_tvl.items(), key=lambda kv: (-kv[1], kv[0]))[:max_protocols]
    ]
    active_proto_set = set(ranked_pids)
    active_proto_tvl = {pid: proto_tvl[pid] for pid in ranked_pids}

    token_totals: dict[str, float] = defaultdict(float)
    for pid in ranked_pids:
        for sym, val in cleaned_holdings.get(pid, {}).items():
            token_totals[sym] += val

    raw_pair_usd: dict[tuple[str, str], float] = defaultdict(float)
    conduit_inbound_usd: dict[str, float] = defaultdict(float)
    conduit_holders: dict[str, set[str]] = defaultdict(set)

    for pid in ranked_pids:
        for sym, val in cleaned_holdings.get(pid, {}).items():
            if val < min_edge_usd:
                continue
            target = mapper.resolve_token_issuer(
                sym,
                protocol_tvls=active_proto_tvl,
                token_totals=token_totals,
            )
            if not target or target == pid:
                continue
            if target in active_proto_set:
                raw_pair_usd[(pid, target)] += val
            elif not target.isdigit():
                raw_pair_usd[(pid, target)] += val
                conduit_inbound_usd[target] += val
                conduit_holders[target].add(pid)
            else:
                # Numeric protocol below active TVL threshold -> keep exposure via symbol conduit
                raw_pair_usd[(pid, sym)] += val
                conduit_inbound_usd[sym] += val
                conduit_holders[sym].add(pid)

    # Keep asset conduits that either connect >=2 distinct protocols or hold >= $10M reserve value
    candidate_conduits = [
        (c, tot)
        for c, tot in conduit_inbound_usd.items()
        if c not in active_proto_set
        and tot >= min_conduit_stock_usd
        and (len(conduit_holders[c]) >= 2 or tot >= 10_000_000.0)
    ]
    candidate_conduits.sort(key=lambda kv: (-kv[1], kv[0]))
    valid_conduits = {c for c, _ in candidate_conduits[:DEFAULT_MAX_CONDUITS]}
    node_ids = sorted(active_proto_set | valid_conduits)
    node_index = {nid: idx for idx, nid in enumerate(node_ids)}

    tvl_arr = np.zeros(len(node_ids), dtype=np.float64)
    conduit_arr = np.zeros(len(node_ids), dtype=bool)
    cats: list[str] = []
    for idx, nid in enumerate(node_ids):
        if nid in active_proto_set:
            tvl_arr[idx] = active_proto_tvl[nid]
            conduit_arr[idx] = False
            cats.append(mapper.protocol_category(nid))
        else:
            tvl_arr[idx] = 0.0
            conduit_arr[idx] = True
            cats.append("Unknown")

    edge_src_list: list[int] = []
    edge_dst_list: list[int] = []
    edge_usd_list: list[float] = []
    edge_map: dict[tuple[str, str], float] = {}

    for (u, v), w_usd in sorted(raw_pair_usd.items()):
        if w_usd < min_edge_usd or u not in node_index or v not in node_index or u == v:
            continue
        edge_src_list.append(node_index[u])
        edge_dst_list.append(node_index[v])
        edge_usd_list.append(float(w_usd))
        edge_map[(u, v)] = float(w_usd)

    return StockSnapshotData(
        date=date_str,
        node_ids=node_ids,
        node_index=node_index,
        tvl_usd=tvl_arr,
        is_conduit=conduit_arr,
        categories=cats,
        edge_src=np.asarray(edge_src_list, dtype=np.int64),
        edge_dst=np.asarray(edge_dst_list, dtype=np.int64),
        edge_usd=np.asarray(edge_usd_list, dtype=np.float64),
        edge_map=edge_map,
    )


def build_stock_snapshot_from_raw(
    date_str: str,
    raw_snap: dict[str, Any],
    mapper: TokenMapperV3,
    **kwargs: Any,
) -> StockSnapshotData:
    """Build StockSnapshotData from a raw `{nodes: [{id, size, composition}], ...}` dict."""
    holdings: dict[str, dict[str, float]] = {}
    sizes: dict[str, float] = {}
    for node in raw_snap.get("nodes", []):
        nid = str(node.get("id") or "").strip()
        if not nid:
            continue
        sz = float(node.get("size") or 0.0)
        comp = node.get("composition") or {}
        if sz > 0.0 or comp:
            sizes[nid] = max(0.0, sz)
            holdings[nid] = {str(k): float(v or 0.0) for k, v in comp.items() if float(v or 0.0) > 0.0}
    return build_stock_snapshot_from_holdings(
        date_str,
        holdings,
        mapper,
        explicit_sizes=sizes,
        **kwargs,
    )


def stock_snapshot_to_graph_snapshot(snap: StockSnapshotData) -> GraphSnapshot:
    """Convert StockSnapshotData into the Agent pipeline's `GraphSnapshot` schema."""
    n = len(snap.node_ids)
    out_edges_by_node: list[list[float]] = [[] for _ in range(n)]
    for s, w in zip(snap.edge_src, snap.edge_usd):
        if float(w) > 0.0:
            out_edges_by_node[int(s)].append(float(w))

    nodes_dict: dict[str, NodeFeatures] = {}
    for idx, nid in enumerate(snap.node_ids):
        weights = out_edges_by_node[idx]
        tot_out = sum(weights)
        if tot_out > 0.0 and weights:
            shares = [w / tot_out for w in weights]
            num_toks = len(weights)
            max_sh = max(0.0, min(1.0, max(shares)))
            ent = max(0.0, -sum(p * math.log(p) for p in shares if p > 0.0))
        else:
            num_toks = 0
            max_sh = 0.0
            ent = 0.0
        tvl = float(snap.tvl_usd[idx])
        log_sz = math.log1p(max(0.0, tvl))
        change_1w = float(snap.features[idx, 13]) if snap.features is not None else 0.0
        cat_str = snap.categories[idx].strip() if snap.categories[idx].strip() else "Unknown"
        nodes_dict[nid] = NodeFeatures(
            log_size=log_sz,
            num_tokens=num_toks,
            max_share=max_sh,
            entropy=ent,
            category=cat_str,
            prev_tvl_change=change_1w,
        )

    edges_list = [
        Edge(
            source=snap.node_ids[int(s)],
            target=snap.node_ids[int(d)],
            weight=float(w),
        )
        for s, d, w in zip(snap.edge_src, snap.edge_dst, snap.edge_usd)
    ]
    return GraphSnapshot(date=snap.date, nodes=nodes_dict, edges=edges_list)


def _rank_percentiles(values: np.ndarray) -> np.ndarray:
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    ranks[values <= 0] = 0.0
    return ranks


def enrich_stock_snapshots(
    snapshots: list[StockSnapshotData],
    categories_vocab: list[str] | tuple[str, ...],
) -> list[StockSnapshotData]:
    """Populate scale-invariant topological + rolling balance-sheet node features `[N, F]`."""
    cat_to_idx = {c: i for i, c in enumerate(categories_vocab)}
    unk_idx = cat_to_idx.get("Unknown", 0)
    num_core = len(V3_CORE_NODE_FEATURES)
    num_cats = len(categories_vocab)
    feat_dim = num_core + num_cats

    # Precompute per-snapshot node stats maps for fast rolling lookbacks
    per_snap_tvl_log: list[dict[str, float]] = []
    per_snap_out_log: list[dict[str, float]] = []
    per_snap_in_log: list[dict[str, float]] = []

    for snap in snapshots:
        n = len(snap.node_ids)
        out_usd = np.zeros(n, dtype=np.float64)
        in_usd = np.zeros(n, dtype=np.float64)
        if len(snap.edge_src) > 0:
            np.add.at(out_usd, snap.edge_src, snap.edge_usd)
            np.add.at(in_usd, snap.edge_dst, snap.edge_usd)
        per_snap_tvl_log.append({
            nid: math.log1p(max(0.0, float(snap.tvl_usd[i])))
            for i, nid in enumerate(snap.node_ids)
        })
        per_snap_out_log.append({
            nid: math.log1p(max(0.0, float(out_usd[i])))
            for i, nid in enumerate(snap.node_ids)
        })
        per_snap_in_log.append({
            nid: math.log1p(max(0.0, float(in_usd[i])))
            for i, nid in enumerate(snap.node_ids)
        })

    for t, snap in enumerate(snapshots):
        n = len(snap.node_ids)
        feats = np.zeros((n, feat_dim), dtype=np.float32)
        if n == 0:
            snap.features = feats
            continue

        out_deg = np.zeros(n, dtype=np.float64)
        in_deg = np.zeros(n, dtype=np.float64)
        out_usd = np.zeros(n, dtype=np.float64)
        in_usd = np.zeros(n, dtype=np.float64)
        out_sq = np.zeros(n, dtype=np.float64)
        in_sq = np.zeros(n, dtype=np.float64)
        conduit_held = np.zeros(n, dtype=np.float64)
        proto_held = np.zeros(n, dtype=np.float64)

        if len(snap.edge_src) > 0:
            np.add.at(out_deg, snap.edge_src, 1.0)
            np.add.at(in_deg, snap.edge_dst, 1.0)
            np.add.at(out_usd, snap.edge_src, snap.edge_usd)
            np.add.at(in_usd, snap.edge_dst, snap.edge_usd)
            np.add.at(out_sq, snap.edge_src, np.square(snap.edge_usd))
            np.add.at(in_sq, snap.edge_dst, np.square(snap.edge_usd))
            dst_is_cond = snap.is_conduit[snap.edge_dst]
            np.add.at(conduit_held, snap.edge_src[dst_is_cond], snap.edge_usd[dst_is_cond])
            np.add.at(proto_held, snap.edge_src[~dst_is_cond], snap.edge_usd[~dst_is_cond])

        log_tvl = np.log1p(np.maximum(0.0, snap.tvl_usd)).astype(np.float32)
        log_out = np.log1p(np.maximum(0.0, out_usd)).astype(np.float32)
        log_in = np.log1p(np.maximum(0.0, in_usd)).astype(np.float32)

        # Scale-invariant degree features (no max-degree covariateshift!)
        out_deg_scale = (np.log1p(out_deg) / 6.0).astype(np.float32)
        in_deg_scale = (np.log1p(in_deg) / 8.0).astype(np.float32)
        out_deg_pct = _rank_percentiles(out_deg)
        in_deg_pct = _rank_percentiles(in_deg)

        asset_hhi = np.where(out_usd > 0.0, out_sq / np.maximum(out_usd * out_usd, 1e-12), 0.0).clip(0.0, 1.0)
        liab_hhi = np.where(in_usd > 0.0, in_sq / np.maximum(in_usd * in_usd, 1e-12), 0.0).clip(0.0, 1.0)
        denom_base = np.maximum(snap.tvl_usd, np.maximum(out_usd, 1.0))
        net_lev = np.clip(in_usd / denom_base, 0.0, 10.0) / 10.0
        cond_share = np.where(out_usd > 0.0, conduit_held / np.maximum(out_usd, 1.0), 0.0).clip(0.0, 1.0)
        prot_share = np.where(out_usd > 0.0, proto_held / np.maximum(out_usd, 1.0), 0.0).clip(0.0, 1.0)

        feats[:, 0] = log_tvl / 25.0
        feats[:, 1] = log_out / 25.0
        feats[:, 2] = log_in / 25.0
        feats[:, 3] = out_deg_scale
        feats[:, 4] = in_deg_scale
        feats[:, 5] = out_deg_pct
        feats[:, 6] = in_deg_pct
        feats[:, 7] = asset_hhi.astype(np.float32)
        feats[:, 8] = liab_hhi.astype(np.float32)
        feats[:, 9] = net_lev.astype(np.float32)
        feats[:, 10] = cond_share.astype(np.float32)
        feats[:, 11] = prot_share.astype(np.float32)
        feats[:, 12] = snap.is_conduit.astype(np.float32)

        prev_edge_map = snapshots[t - 1].edge_map if t >= 1 else None
        out_retained_usd = defaultdict(float)
        if prev_edge_map is not None:
            for (u, v), w_now in snap.edge_map.items():
                w_prev = prev_edge_map.get((u, v), 0.0)
                if w_prev > 0.0:
                    out_retained_usd[u] += min(w_now, w_prev)

        for i, nid in enumerate(snap.node_ids):
            lt_now = float(log_tvl[i])
            lo_now = float(log_out[i])
            li_now = float(log_in[i])

            lt_1 = per_snap_tvl_log[t - 1].get(nid, lt_now) if t >= 1 else lt_now
            lt_4 = per_snap_tvl_log[t - 4].get(nid, lt_1) if t >= 4 else lt_1
            lt_8 = per_snap_tvl_log[t - 8].get(nid, lt_4) if t >= 8 else lt_4

            lo_1 = per_snap_out_log[t - 1].get(nid, lo_now) if t >= 1 else lo_now
            lo_4 = per_snap_out_log[t - 4].get(nid, lo_1) if t >= 4 else lo_1
            li_1 = per_snap_in_log[t - 1].get(nid, li_now) if t >= 1 else li_now
            li_4 = per_snap_in_log[t - 4].get(nid, li_1) if t >= 4 else li_1

            # 4-week rolling series for volatility and max drawdown
            hist_vals = [
                per_snap_tvl_log[k].get(nid)
                for k in range(max(0, t - 4), t + 1)
                if nid in per_snap_tvl_log[k]
            ]
            if len(hist_vals) >= 2:
                diffs = [hist_vals[j] - hist_vals[j - 1] for j in range(1, len(hist_vals))]
                vol_4w = float(np.std(diffs))
                peak = max(hist_vals)
                dd_4w = max(0.0, peak - lt_now)
            else:
                vol_4w = 0.0
                dd_4w = 0.0

            lookback_len = min(8, t + 1)
            age_cnt = sum(1 for k in range(t - lookback_len + 1, t + 1) if nid in per_snap_tvl_log[k])
            age_8w = float(age_cnt / max(1, lookback_len))

            if prev_edge_map is not None and out_usd[i] > 0.0:
                ret_1w = float(min(1.0, out_retained_usd.get(nid, 0.0) / out_usd[i]))
            else:
                ret_1w = 1.0

            feats[i, 13] = float(np.clip(lt_now - lt_1, -3.0, 3.0))
            feats[i, 14] = float(np.clip(lt_now - lt_4, -4.0, 4.0))
            feats[i, 15] = float(np.clip(lt_now - lt_8, -5.0, 5.0))
            feats[i, 16] = float(np.clip(vol_4w, 0.0, 3.0))
            feats[i, 17] = float(np.clip(dd_4w, 0.0, 5.0))
            feats[i, 18] = float(np.clip(lo_now - lo_1, -3.0, 3.0))
            feats[i, 19] = float(np.clip(lo_now - lo_4, -4.0, 4.0))
            feats[i, 20] = float(np.clip(li_now - li_1, -3.0, 3.0))
            feats[i, 21] = float(np.clip(li_now - li_4, -4.0, 4.0))
            feats[i, 22] = age_8w
            feats[i, 23] = ret_1w

            cidx = cat_to_idx.get(snap.categories[i], unk_idx)
            feats[i, num_core + cidx] = 1.0

        snap.features = np.nan_to_num(feats, nan=0.0, posinf=3.0, neginf=-3.0)

    return snapshots


def build_edge_state_features(
    snapshots: list[StockSnapshotData],
    t_idx: int,
    pair_src: np.ndarray,
    pair_dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build explicit 12-dim edge-state matrix `[P, 12]` + baseline `log1p(w_t)` `[P]`."""
    snap = snapshots[t_idx]
    assert snap.features is not None, "Call enrich_stock_snapshots before build_edge_state_features"
    n_nodes = len(snap.node_ids)
    n_pairs = len(pair_src)
    z = np.zeros((n_pairs, len(V3_EDGE_STATE_FEATURES)), dtype=np.float32)
    base_log_w = np.zeros(n_pairs, dtype=np.float32)
    if n_pairs == 0:
        return z, base_log_w

    out_usd = np.zeros(n_nodes, dtype=np.float64)
    in_usd = np.zeros(n_nodes, dtype=np.float64)
    if len(snap.edge_src) > 0:
        np.add.at(out_usd, snap.edge_src, snap.edge_usd)
        np.add.at(in_usd, snap.edge_dst, snap.edge_usd)
    holder_denom = np.maximum(snap.tvl_usd, np.maximum(out_usd, 1.0))
    issuer_denom = np.maximum(in_usd, np.maximum(snap.tvl_usd, 1.0))

    hist_maps = [
        snapshots[k].edge_map if k >= 0 else {}
        for k in (t_idx - 4, t_idx - 3, t_idx - 2, t_idx - 1, t_idx)
    ]  # indices: 0=t-4, 1=t-3, 2=t-2, 3=t-1, 4=t

    node_feats = snap.features
    for idx in range(n_pairs):
        u_i = int(pair_src[idx])
        v_i = int(pair_dst[idx])
        key = (snap.node_ids[u_i], snap.node_ids[v_i])

        w_now = float(hist_maps[4].get(key, 0.0))
        is_act = 1.0 if w_now > 0.0 else 0.0
        lw_now = math.log1p(max(0.0, w_now))
        base_log_w[idx] = lw_now

        lw_seq = [math.log1p(max(0.0, float(hm.get(key, 0.0)))) for hm in hist_maps]
        act_seq = [1.0 if float(hm.get(key, 0.0)) > 0.0 else 0.0 for hm in hist_maps]

        lw_1 = lw_seq[3] if t_idx >= 1 else lw_now
        lw_4 = lw_seq[0] if t_idx >= 4 else lw_1
        avail_steps = min(4, t_idx)
        if avail_steps >= 2:
            step_diffs = [lw_seq[4 - j] - lw_seq[3 - j] for j in range(avail_steps)]
            vol_4w = float(np.std(step_diffs))
        else:
            vol_4w = 0.0
        age_4w = float(sum(act_seq[4 - avail_steps : 5]) / (avail_steps + 1))

        share_h = float(min(1.0, w_now / holder_denom[u_i])) if w_now > 0.0 else 0.0
        share_i = float(min(1.0, w_now / issuer_denom[v_i])) if w_now > 0.0 else 0.0

        z[idx, 0] = is_act
        z[idx, 1] = float(lw_now / 25.0)
        z[idx, 2] = share_h
        z[idx, 3] = share_i
        z[idx, 4] = float(np.clip(lw_now - lw_1, -4.0, 4.0))
        z[idx, 5] = float(np.clip(lw_now - lw_4, -5.0, 5.0))
        z[idx, 6] = age_4w
        z[idx, 7] = float(np.clip(vol_4w, 0.0, 4.0))
        z[idx, 8] = float(node_feats[u_i, 13])   # holder_tvl_mom_1w
        z[idx, 9] = float(node_feats[u_i, 14])   # holder_tvl_mom_4w
        z[idx, 10] = float(node_feats[v_i, 20])  # issuer_inflow_mom_1w
        z[idx, 11] = float(node_feats[v_i, 21])  # issuer_inflow_mom_4w

    return np.nan_to_num(z, nan=0.0, posinf=4.0, neginf=-4.0), base_log_w


# ===========================================================================
# PyTorch Model Architecture: DeXposureFMV3
# ===========================================================================


def _encode_with_graphpfn(
    encoder: nn.Module,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    import dgl
    from lib.graphpfn.model import GraphPFNLayerWrapper
    from lib.limix.model.layer import MultiheadAttention as MHA

    num_nodes = int(features.shape[0])
    if num_nodes < 2:
        embed_dim = int(encoder.tfm.embed_dim)
        return torch.zeros((num_nodes, embed_dim), dtype=torch.float32, device=device)

    src_cpu = torch.as_tensor(edge_src, dtype=torch.int64)
    dst_cpu = torch.as_tensor(edge_dst, dtype=torch.int64)
    # Add reverse message-passing edges so issuer shock context flows back to holders in GraphPFN
    bidir_src_cpu = torch.cat([src_cpu, dst_cpu], dim=0)
    bidir_dst_cpu = torch.cat([dst_cpu, src_cpu], dim=0)
    graph = dgl.graph((bidir_src_cpu, bidir_dst_cpu), num_nodes=num_nodes)
    if device.type == "cuda":
        graph._cuda_edges = (  # type: ignore[attr-defined]
            bidir_src_cpu.to(device=device, dtype=torch.long),
            bidir_dst_cpu.to(device=device, dtype=torch.long),
        )

    train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    train_mask[-1] = False
    n_train = int(train_mask.sum().item())
    y_train = torch.zeros(n_train, dtype=torch.float32, device=device)
    features = features.to(device=device, dtype=torch.float32)
    tfm_features = torch.cat([features[train_mask], features[~train_mask]], dim=0)

    for module in encoder.modules():
        if isinstance(module, GraphPFNLayerWrapper):
            module.train_mask = train_mask
            module.graph = graph
        if isinstance(module, MHA):
            module.batched = False

    if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
        sdpa_backends = (
            [torch.nn.attention.SDPBackend.MATH]
            if device.type == "cpu"
            else [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        )
        with torch.nn.attention.sdpa_kernel(sdpa_backends):
            out = encoder.tfm.forward(
                x=tfm_features.unsqueeze(0),
                y=y_train.unsqueeze(0),
                eval_pos=n_train,
                task_type="reg",
                checkpointing=encoder.training,
            )
    else:
        if device.type == "cpu":
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                out = encoder.tfm.forward(
                    x=tfm_features.unsqueeze(0),
                    y=y_train.unsqueeze(0),
                    eval_pos=n_train,
                    task_type="reg",
                    checkpointing=encoder.training,
                )
        else:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=True, enable_mem_efficient=True
            ):
                out = encoder.tfm.forward(
                    x=tfm_features.unsqueeze(0),
                    y=y_train.unsqueeze(0),
                    eval_pos=n_train,
                    task_type="reg",
                    checkpointing=encoder.training,
                )

    inv_order = torch.argsort((~train_mask).float(), stable=True)
    order = torch.argsort(inv_order, stable=True)
    if "encoder_out_full" in out:
        feature_embeds = out["encoder_out_full"][:, :, :-1, :]
        return feature_embeds.mean(dim=2).squeeze(0)[order]
    return out["encoder_embed"].squeeze(0)[order]


class MonotonicStockEdgeHead(nn.Module):
    """Joint multi-horizon stock edge scorer with guaranteed cross-horizon monotonicity."""

    def __init__(
        self,
        embed_dim: int,
        edge_feat_dim: int = len(V3_EDGE_STATE_FEATURES),
        hidden_dim: int = 256,
        num_horizons: int = len(V3_HORIZONS),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = embed_dim * 4 + edge_feat_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # 1. Global link existence logits (active + new link formation vs random negatives)
        self.exist_head = nn.Linear(hidden_dim, num_horizons)
        # 2. Strictly non-negative withdrawal hazard increments -> monotonic p_retain(h)
        self.hazard_head = nn.Linear(hidden_dim, num_horizons)
        nn.init.constant_(self.hazard_head.bias, -2.2)  # ~90% initial 1w retention prior
        # 3. Continuous multi-horizon log-stock residual: log1p(w_{t+h}) - log1p(w_t)
        self.delta_weight_head = nn.Linear(hidden_dim, num_horizons)

    def forward(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hu = h[src]
        hv = h[dst]
        pair_repr = torch.cat([hu, hv, hu * hv, torch.abs(hu - hv), edge_state], dim=-1)
        feat = self.trunk(pair_repr)
        exist_logits = self.exist_head(feat)
        # Non-negative increments guarantee cumulative hazard Lambda(h_1) <= Lambda(h_4) <= ...
        hazard_inc = F.softplus(self.hazard_head(feat)) + 1e-5
        cum_hazard = torch.cumsum(hazard_inc, dim=-1)
        retain_prob_raw = torch.exp(-cum_hazard).clamp(1e-5, 1.0 - 1e-5)
        retain_logits_raw = torch.logit(retain_prob_raw, eps=1e-5)
        delta_log_w = self.delta_weight_head(feat)
        return {
            "exist_logits": exist_logits,
            "retain_logits_raw": retain_logits_raw,
            "cum_hazard": cum_hazard,
            "delta_log_w": delta_log_w,
        }


class MonotonicNeighborNodeHead(nn.Module):
    """Joint multi-horizon protocol TVL drawdown & monotonic stress probability head."""

    def __init__(
        self,
        embed_dim: int,
        node_mom_dim: int = 12,
        hidden_dim: int = 256,
        num_horizons: int = len(V3_HORIZONS),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = embed_dim * 3 + node_mom_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        self.delta_tvl_head = nn.Linear(hidden_dim // 2, num_horizons)
        self.stress_hazard_head = nn.Linear(hidden_dim // 2, num_horizons)
        nn.init.constant_(self.stress_hazard_head.bias, -2.0)

    def forward(
        self,
        h: torch.Tensor,
        node_features: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_log_w: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        num_nodes, embed_dim = h.shape
        device = h.device
        asset_ctx = torch.zeros((num_nodes, embed_dim), dtype=h.dtype, device=device)
        liab_ctx = torch.zeros((num_nodes, embed_dim), dtype=h.dtype, device=device)

        if edge_src.numel() > 0:
            w = edge_log_w.clamp(min=0.1)
            # Outgoing stock edges u -> v: holder u's asset portfolio exposure to v
            out_denom = torch.zeros(num_nodes, dtype=h.dtype, device=device)
            out_denom.index_add_(0, edge_src, w.to(h.dtype))
            w_out = (w.to(h.dtype) / out_denom[edge_src].clamp(min=1e-6)).unsqueeze(-1)
            asset_ctx.index_add_(0, edge_src, h[edge_dst] * w_out)

            # Incoming stock edges u -> v: issuer v's holder base concentration
            in_denom = torch.zeros(num_nodes, dtype=h.dtype, device=device)
            in_denom.index_add_(0, edge_dst, w.to(h.dtype))
            w_in = (w.to(h.dtype) / in_denom[edge_dst].clamp(min=1e-6)).unsqueeze(-1)
            liab_ctx.index_add_(0, edge_dst, h[edge_src] * w_in)

        # Slice rolling momentum & balance-sheet ratios from node_features[:, 12:24]
        mom_slice = node_features[:, 12:24].to(dtype=h.dtype, device=device)
        cat_repr = torch.cat([h, asset_ctx, liab_ctx, mom_slice], dim=-1)
        feat = self.trunk(cat_repr)
        delta_tvl = self.delta_tvl_head(feat)
        stress_inc = F.softplus(self.stress_hazard_head(feat)) + 1e-5
        cum_stress_hazard = torch.cumsum(stress_inc, dim=-1)
        surv_prob_raw = torch.exp(-cum_stress_hazard).clamp(1e-5, 1.0 - 1e-5)
        stress_prob_raw = (1.0 - surv_prob_raw).clamp(1e-5, 1.0 - 1e-5)
        stress_logits_raw = torch.logit(stress_prob_raw, eps=1e-5)
        return {
            "delta_log_tvl": delta_tvl,
            "stress_logits_raw": stress_logits_raw,
            "cum_stress_hazard": cum_stress_hazard,
        }


class DeXposureFMV3(nn.Module):
    """Unified multi-horizon Stock-Exposure & Drawdown Foundation Model."""

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 192,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.link_scorer = MonotonicStockEdgeHead(
            embed_dim=embed_dim,
            edge_feat_dim=len(V3_EDGE_STATE_FEATURES),
            hidden_dim=hidden_dim,
            num_horizons=len(V3_HORIZONS),
            dropout=dropout,
        )
        self.node_head = MonotonicNeighborNodeHead(
            embed_dim=embed_dim,
            node_mom_dim=12,
            hidden_dim=hidden_dim,
            num_horizons=len(V3_HORIZONS),
            dropout=dropout,
        )
        # Monotonicity-preserving positive hazard scales `alpha_k > 0` fit on validation set:
        # calibrated cumulative hazard = cumsum(alpha_k * delta_hazard_k), preserving
        # p_retain(1w) >= p_retain(4w) >= p_retain(8w) >= p_retain(12w) strictly!
        self.register_buffer("retain_hazard_scales", torch.ones(len(V3_HORIZONS), dtype=torch.float32))
        self.register_buffer("stress_hazard_scales", torch.ones(len(V3_HORIZONS), dtype=torch.float32))
        self.register_buffer("exist_logit_temps", torch.ones(len(V3_HORIZONS), dtype=torch.float32))
        self.register_buffer("exist_logit_biases", torch.zeros(len(V3_HORIZONS), dtype=torch.float32))

    def encode(
        self,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        features: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        return _encode_with_graphpfn(self.encoder, edge_src, edge_dst, features, device)

    def calibrated_retain_probs(self, cum_hazard: torch.Tensor) -> torch.Tensor:
        """Apply non-negative per-interval scales so cross-horizon monotonicity NEVER breaks."""
        inc = torch.cat([cum_hazard[:, :1], cum_hazard[:, 1:] - cum_hazard[:, :-1]], dim=-1).clamp(min=1e-6)
        scales = self.retain_hazard_scales.clamp(min=0.05, max=20.0)
        cal_cum = torch.cumsum(inc * scales.unsqueeze(0), dim=-1)
        return torch.exp(-cal_cum).clamp(1e-5, 1.0 - 1e-5)

    def calibrated_stress_probs(self, cum_stress_hazard: torch.Tensor) -> torch.Tensor:
        inc = torch.cat(
            [cum_stress_hazard[:, :1], cum_stress_hazard[:, 1:] - cum_stress_hazard[:, :-1]],
            dim=-1,
        ).clamp(min=1e-6)
        scales = self.stress_hazard_scales.clamp(min=0.05, max=20.0)
        cal_cum = torch.cumsum(inc * scales.unsqueeze(0), dim=-1)
        return (1.0 - torch.exp(-cal_cum)).clamp(1e-5, 1.0 - 1e-5)


def load_pretrained_graphpfn(checkpoint_path: Path | str, device: torch.device) -> nn.Module:
    from lib.graphpfn.model import GraphPFN

    encoder = GraphPFN(edge_head=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in base GraphPFN checkpoint: {unexpected[:5]}")
    encoder.to(device)
    return encoder


# ===========================================================================
# Live & Offline Predictor Wrapper: FMV3Predictor
# ===========================================================================


class FMV3Predictor:
    """Calibrated multi-horizon Stock-Exposure predictor with single-pass encoder memoization."""

    model_family = "dexposure_fm_v3"

    def __init__(
        self,
        checkpoint_dir: Path | str = DEFAULT_V3_CKPT_DIR,
        device: str | None = None,
        pi_min: float = 0.0,
        mapping_bundle: Path | str = DEFAULT_MAPPING_BUNDLE,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pi_min = float(pi_min)
        self.mapper = TokenMapperV3(mapping_bundle)

        schema_path = self.checkpoint_dir / "feature_schema.json"
        unified_ckpt = self.checkpoint_dir / "dexposure-fm-v3.pt"
        h1_ckpt = self.checkpoint_dir / "dexposure-fm-h1.pt"
        self._ckpt_path = unified_ckpt if unified_ckpt.is_file() else h1_ckpt
        self._available = schema_path.is_file() and self._ckpt_path.is_file()

        if self._available:
            self._schema = json.loads(schema_path.read_text(encoding="utf-8"))
            self._categories = list(self._schema["categories"])
        else:
            self._schema = {}
            self._categories = list(self.mapper.categories)

        self._model: DeXposureFMV3 | None = None
        self._memo_cache: dict[str, dict[str, Any]] = {}

        # Observable outputs populated on each `.predict(...)` call
        self.last_edge_probs: dict[tuple[str, str], float] = {}
        self.last_edge_delta_log_w: dict[tuple[str, str], float] = {}
        self.last_node_deltas: dict[str, float] = {}
        self.last_node_stress_probs: dict[str, float] = {}
        self.last_pred_snap: GraphSnapshot | None = None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def available_horizons(self) -> list[int]:
        return list(V3_HORIZONS) if self._available else []

    @property
    def checkpoint_identity(self) -> dict[str, Any]:
        ckpt_sha = None
        if self._ckpt_path.is_file():
            digest = hashlib.sha256()
            with self._ckpt_path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(chunk)
            ckpt_sha = digest.hexdigest()
        return {
            "model_family": self.model_family,
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_file": self._ckpt_path.name,
            "checkpoint_sha256": ckpt_sha,
            "schema_version": self._schema.get("schema_version", 3),
            "pi_min": self.pi_min,
        }

    def _ensure_loaded(self) -> DeXposureFMV3:
        if self._model is not None:
            return self._model
        if not self._available:
            raise FileNotFoundError(f"DeXposure-FM v3 checkpoint missing in {self.checkpoint_dir}")
        from lib.graphpfn.model import GraphPFN

        payload = torch.load(self._ckpt_path, map_location=self.device, weights_only=False)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        encoder = GraphPFN(edge_head=False)
        hidden_dim = int(payload.get("hidden_dim", 256)) if isinstance(payload, dict) else 256
        model = DeXposureFMV3(
            encoder=encoder,
            embed_dim=int(encoder.tfm.embed_dim),
            hidden_dim=hidden_dim,
            dropout=0.0,
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    def clear_cache(self) -> None:
        self._memo_cache.clear()

    def _graph_snapshot_to_stock(self, gs: GraphSnapshot) -> StockSnapshotData:
        node_ids = sorted(gs.nodes.keys())
        node_index = {nid: i for i, nid in enumerate(node_ids)}
        tvl_arr = np.zeros(len(node_ids), dtype=np.float64)
        cond_arr = np.zeros(len(node_ids), dtype=bool)
        cats: list[str] = []
        for i, nid in enumerate(node_ids):
            nf = gs.nodes[nid]
            tvl = math.expm1(max(0.0, float(nf.log_size or 0.0)))
            tvl_arr[i] = tvl
            cond_arr[i] = tvl <= 0.0 and not nid.isdigit()
            cats.append(nf.category if nf.category in self._categories else "Unknown")

        edge_agg: dict[tuple[str, str], float] = defaultdict(float)
        for e in gs.edges:
            if e.source in node_index and e.target in node_index and e.source != e.target and e.weight > 0:
                edge_agg[(e.source, e.target)] += float(e.weight)

        src_l, dst_l, w_l = [], [], []
        edge_map: dict[tuple[str, str], float] = {}
        for (u, v), w in sorted(edge_agg.items()):
            src_l.append(node_index[u])
            dst_l.append(node_index[v])
            w_l.append(w)
            edge_map[(u, v)] = w

        return StockSnapshotData(
            date=gs.date,
            node_ids=node_ids,
            node_index=node_index,
            tvl_usd=tvl_arr,
            is_conduit=cond_arr,
            categories=cats,
            edge_src=np.asarray(src_l, dtype=np.int64),
            edge_dst=np.asarray(dst_l, dtype=np.int64),
            edge_usd=np.asarray(w_l, dtype=np.float64),
            edge_map=edge_map,
        )

    def _run_joint_forward(
        self,
        current_graph: GraphSnapshot,
        history_graphs: list[GraphSnapshot] | None = None,
    ) -> dict[str, Any]:
        cache_key = (
            f"{current_graph.date}:{len(current_graph.nodes)}:{len(current_graph.edges)}:"
            f"{len(history_graphs or [])}"
        )
        if cache_key in self._memo_cache:
            return self._memo_cache[cache_key]

        model = self._ensure_loaded()
        seq_gs = list(history_graphs or [])
        if not seq_gs or seq_gs[-1].date != current_graph.date:
            seq_gs.append(current_graph)

        stock_seq = [self._graph_snapshot_to_stock(g) for g in seq_gs]
        enrich_stock_snapshots(stock_seq, self._categories)
        curr_stock = stock_seq[-1]
        t_idx = len(stock_seq) - 1

        edge_z, base_log_w = build_edge_state_features(
            stock_seq, t_idx, curr_stock.edge_src, curr_stock.edge_dst
        )

        with torch.no_grad():
            feat_t = torch.as_tensor(curr_stock.features, dtype=torch.float32, device=self.device)
            h = model.encode(curr_stock.edge_src, curr_stock.edge_dst, feat_t, self.device)

            src_t = torch.as_tensor(curr_stock.edge_src, dtype=torch.long, device=self.device)
            dst_t = torch.as_tensor(curr_stock.edge_dst, dtype=torch.long, device=self.device)
            ez_t = torch.as_tensor(edge_z, dtype=torch.float32, device=self.device)
            bw_t = torch.as_tensor(base_log_w, dtype=torch.float32, device=self.device)

            if src_t.numel() > 0:
                edge_out = model.link_scorer(h, src_t, dst_t, ez_t)
                retain_probs = model.calibrated_retain_probs(edge_out["cum_hazard"]).cpu().numpy()
                delta_log_w = edge_out["delta_log_w"].cpu().numpy()
            else:
                retain_probs = np.zeros((0, len(V3_HORIZONS)), dtype=np.float32)
                delta_log_w = np.zeros((0, len(V3_HORIZONS)), dtype=np.float32)

            node_out = model.node_head(h, feat_t, src_t, dst_t, bw_t)
            delta_log_tvl = node_out["delta_log_tvl"].cpu().numpy()
            stress_probs = model.calibrated_stress_probs(node_out["cum_stress_hazard"]).cpu().numpy()

        bundle = {
            "stock": curr_stock,
            "retain_probs": retain_probs,   # [E, 4] strictly monotonic across h=1,4,8,12
            "delta_log_w": delta_log_w,     # [E, 4]
            "delta_log_tvl": delta_log_tvl, # [N, 4]
            "stress_probs": stress_probs,   # [N, 4] strictly monotonic across h=1,4,8,12
        }
        # Keep at most 4 cached snapshots in RAM
        if len(self._memo_cache) >= 4:
            self._memo_cache.clear()
        self._memo_cache[cache_key] = bundle
        return bundle

    def predict(
        self,
        current_graph: GraphSnapshot,
        horizon: int = 4,
        *,
        history_graphs: list[GraphSnapshot] | None = None,
    ) -> GraphSnapshot:
        """Predict future Stock Exposure Graph at `horizon in {1, 4, 8, 12}`."""
        if horizon not in HORIZON_TO_IDX:
            raise ValueError(f"Unsupported FM v3 horizon {horizon}; expected {V3_HORIZONS}")
        h_idx = HORIZON_TO_IDX[horizon]
        joint = self._run_joint_forward(current_graph, history_graphs=history_graphs)
        curr_stock: StockSnapshotData = joint["stock"]
        retain_col = joint["retain_probs"][:, h_idx] if len(curr_stock.edge_src) > 0 else np.zeros(0)
        dw_col = joint["delta_log_w"][:, h_idx] if len(curr_stock.edge_src) > 0 else np.zeros(0)
        dtvl_col = joint["delta_log_tvl"][:, h_idx]
        stress_col = joint["stress_probs"][:, h_idx]

        edge_probs: dict[tuple[str, str], float] = {}
        edge_dw: dict[tuple[str, str], float] = {}
        pred_edges: list[Edge] = []

        for idx in range(len(curr_stock.edge_src)):
            u = curr_stock.node_ids[int(curr_stock.edge_src[idx])]
            v = curr_stock.node_ids[int(curr_stock.edge_dst[idx])]
            w_obs = float(curr_stock.edge_usd[idx])
            p_ret = float(np.clip(retain_col[idx], 1e-4, 0.9999))
            dw = float(np.clip(dw_col[idx], -4.0, 2.5))
            edge_probs[(u, v)] = p_ret
            edge_dw[(u, v)] = dw

            if p_ret >= self.pi_min:
                w_pred = w_obs * p_ret
                if w_pred > 1.0:
                    pred_edges.append(Edge(source=u, target=v, weight=float(w_pred)))

        node_deltas: dict[str, float] = {}
        node_stress: dict[str, float] = {}
        pred_nodes: dict[str, NodeFeatures] = {}
        for i, nid in enumerate(curr_stock.node_ids):
            orig_nf = current_graph.nodes[nid]
            if curr_stock.is_conduit[i]:
                node_deltas[nid] = 0.0
                node_stress[nid] = 0.0
                pred_nodes[nid] = orig_nf.model_copy()
            else:
                dtvl = float(np.clip(dtvl_col[i], -3.5, 2.5))
                p_str = float(np.clip(stress_col[i], 1e-4, 0.9999))
                node_deltas[nid] = dtvl
                node_stress[nid] = p_str
                new_log_sz = max(0.0, float(orig_nf.log_size) + dtvl)
                pred_nodes[nid] = orig_nf.model_copy(update={"log_size": new_log_sz})

        self.last_edge_probs = edge_probs
        self.last_edge_delta_log_w = edge_dw
        self.last_node_deltas = node_deltas
        self.last_node_stress_probs = node_stress
        pred_snap = GraphSnapshot(date=current_graph.date, nodes=pred_nodes, edges=pred_edges)
        self.last_pred_snap = pred_snap
        return pred_snap
