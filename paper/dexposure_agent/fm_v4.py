"""DeXposure-FM v4: Price-Beta-Stripped Net-Flow & Scale-Free Stress Foundation Model.

Advances beyond `DeXposure-FM v3` (Option B):
  1. **Benchmark-Deflated Constant-Price Net Flow (Training & Serving Parity)**:
     Strips spot `ETH`, `BTC`, and major crypto price swings from both causal lookback
     momentum features (`[t-8w .. t]`) and multi-horizon supervision targets (`[t+1w .. t+12w]`),
     while preserving 100% of physical token redemptions, stablecoin depegs, and LST/BTC-wrapper
     relative depegs.
  2. **Explicit Numeraire Basket Composition Features**:
     Augments node features (`28` core + `63` category = `91` dims) and edge-state features
     (`16` dims) with `USD_STABLE` / `ETH_BETA` / `BTC_BETA` / `OTHER_CRYPTO` balance-sheet shares.
  3. **Calibrated Fractional Net-Flow Retention (`r_{e,h} in [0, 1]`) & Monotonic Run Hazard**:
     Trains the monotonic cumulative hazard edge head directly on fractional constant-price
     retention `min_{1<=s<=h} min(1, w_e^{(t)}(t+s) / w_e(t))` so `w_e * (1 - p_e)` equals the
     true expected dollar net outflow (eliminating binary `<80%` wipeout exaggeration) while
     guaranteeing `p_e(1w) >= p_e(4w) >= p_e(8w) >= p_e(12w)` on 100% of edges.
  4. **Scale-Free Protocol Run Probability (`p_stress`) & Net-Flow Log-Delta (`delta_log_tvl`)**:
     Directly powers both the Systemic Dollar-Exposure Main Queue (`M7`) and the Scale-Free
     High-Risk Alpha Radar Sub-Leaderboard (`去规模化高危排雷副榜`).
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

from dexposure_agent.fm_v3 import (
    DEFAULT_MAPPING_BUNDLE,
    DEFAULT_MAX_CONDUITS,
    DEFAULT_MAX_PROTOCOLS,
    DEFAULT_MIN_CONDUIT_STOCK_USD,
    DEFAULT_MIN_EDGE_USD,
    DEFAULT_MIN_PROTOCOL_TVL_USD,
    EDGE_RETENTION_RATIO,
    HORIZON_TO_IDX,
    NODE_STRESS_RATIO,
    TokenMapperV3,
    V3_HORIZONS,
    _encode_with_graphpfn,
    _rank_percentiles,
    load_pretrained_graphpfn,
)
from dexposure_agent.types import Edge, GraphSnapshot, NodeFeatures

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPOT_PRICES_BUNDLE = Path(__file__).resolve().parent / "resources" / "historical_spot_prices_v4.json"
DEFAULT_V4_CKPT_DIR = Path(
    os.environ.get(
        "DEXPOSURE_FM_V4_CKPT_DIR",
        str(_PROJECT_ROOT / "checkpoints" / "dexposure-fm-v4"),
    )
)

V4_HORIZONS: tuple[int, ...] = V3_HORIZONS
NUMERAIRE_BASKETS: tuple[str, ...] = ("USD_STABLE", "ETH_BETA", "BTC_BETA", "OTHER_CRYPTO")
BASKET_TO_IDX: dict[str, int] = {b: idx for idx, b in enumerate(NUMERAIRE_BASKETS)}

V4_CORE_NODE_FEATURES: tuple[str, ...] = (
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
    "netflow_tvl_mom_1w",
    "netflow_tvl_mom_4w",
    "netflow_tvl_mom_8w",
    "netflow_tvl_vol_4w",
    "netflow_tvl_drawdown_4w",
    "netflow_out_mom_1w",
    "netflow_out_mom_4w",
    "netflow_in_mom_1w",
    "netflow_in_mom_4w",
    "active_age_8w",
    "netflow_out_retention_1w",
    "basket_usd_share",
    "basket_eth_share",
    "basket_btc_share",
    "basket_other_share",
)

V4_EDGE_STATE_FEATURES: tuple[str, ...] = (
    "is_active_t",
    "log_w_t",
    "share_of_holder",
    "share_of_issuer",
    "netflow_delta_log_w_1w",
    "netflow_delta_log_w_4w",
    "edge_age_4w",
    "netflow_edge_vol_4w",
    "holder_netflow_mom_1w",
    "holder_netflow_mom_4w",
    "issuer_netflow_mom_1w",
    "issuer_netflow_mom_4w",
    "edge_basket_usd",
    "edge_basket_eth",
    "edge_basket_btc",
    "edge_basket_other",
)

_USD_STABLE_SYMS: frozenset[str] = frozenset({
    "USDC", "USDC.E", "USDBC", "USDT", "USDT0", "DAI", "SDAI", "USDS", "SUSDS",
    "USDE", "SUSDE", "FRAX", "SFRAX", "FRXUSD", "SFRXUSD", "PYUSD", "FDUSD",
    "GHO", "CRVUSD", "SCRVUSD", "LUSD", "TUSD", "BUSD", "USDP", "USR", "RLUSD",
    "USD0", "USD0++", "USDY", "BUIDL", "USTB", "OUSG", "TBILL", "SYRUPUSDC",
    "SYRUPUSDT", "FXUSD", "DOLA", "DEUSD", "SDEUSD", "LVLUSD", "USDA", "MIM",
    "USDD", "UST", "USTC", "FEI", "TRIBE_FEI", "AUSD", "BOLD", "GUSD", "USDB",
    "HONEY", "USDX", "SUSDX", "USDF", "USDG", "USDO", "CGUSD", "YUSD", "BYUSD",
})
_USD_STABLE_ISSUERS: frozenset[str] = frozenset({
    "118", "4133", "587", "4795", "3139", "4449", "5989", "376", "1413",
})

_ETH_BETA_SYMS: frozenset[str] = frozenset({
    "ETH", "WETH", "STETH", "WSTETH", "EETH", "WEETH", "WEETH.S", "RETH",
    "CBETH", "WBETH", "BETH", "EZETH", "RSETH", "METH", "CMETH", "SWETH",
    "RSWETH", "SFRXETH", "FRXETH", "OSETH", "ANKRETH", "PUFETH", "UNIETH",
    "ETHX", "PZETH", "STONE", "LSETH", "YNETH", "SUPEROETHB", "OETH", "WOETH",
})
_ETH_BETA_ISSUERS: frozenset[str] = frozenset({
    "182", "2626", "900", "3933", "3946", "3107", "2914", "3591", "3366", "4030", "3882", "6359",
})

_BTC_BETA_SYMS: frozenset[str] = frozenset({
    "BTC", "WBTC", "WBTC.E", "CBBTC", "TBTC", "FBTC", "LBTC", "SOLVBTC",
    "SOLVBTC.BBN", "BTCB", "BTC.B", "EBTC", "UNIBTC", "PUMPBTC", "KBTC", "SBTC", "RENBTC",
})
_BTC_BETA_ISSUERS: frozenset[str] = frozenset({
    "2", "5097", "5292", "4440", "5557",
})

_CANONICAL_SPOT_ALIASES: dict[str, str] = {
    "WETH": "ETH",
    "ETH": "ETH",
    "WBTC": "BTC",
    "BTC": "BTC",
    "BTCB": "BTC",
    "TBTC": "BTC",
    "CBBTC": "BTC",
    "WBNB": "BNB",
    "WSOL": "SOL",
    "WAVAX": "AVAX",
    "WMATIC": "MATIC",
    "POL": "MATIC",
    "WTRX": "TRX",
    "WFTM": "FTM",
}


def classify_numeraire_basket_v4(sym_or_id: str) -> str:
    raw = str(sym_or_id or "").split("(")[0].strip().upper()
    if raw.startswith("COINGECKO:"):
        raw = raw.split(":", 1)[1].strip()
    if not raw:
        return "OTHER_CRYPTO"
    if raw in _USD_STABLE_ISSUERS or raw in _USD_STABLE_SYMS:
        return "USD_STABLE"
    if raw in _ETH_BETA_ISSUERS or raw in _ETH_BETA_SYMS:
        return "ETH_BETA"
    if raw in _BTC_BETA_ISSUERS or raw in _BTC_BETA_SYMS:
        return "BTC_BETA"
    if not raw.isdigit():
        if "USD" in raw or "DAI" in raw or "FRAX" in raw:
            return "USD_STABLE"
        if raw.endswith("ETH") and len(raw) <= 10:
            return "ETH_BETA"
        if raw.endswith("BTC") and len(raw) <= 10:
            return "BTC_BETA"
    return "OTHER_CRYPTO"


def v4_contract_hash(categories: list[str] | tuple[str, ...], spot_sha256: str) -> str:
    payload = {
        "schema_version": 4,
        "horizons": list(V4_HORIZONS),
        "edge_retention_ratio": EDGE_RETENTION_RATIO,
        "node_stress_ratio": NODE_STRESS_RATIO,
        "min_protocol_tvl_usd": DEFAULT_MIN_PROTOCOL_TVL_USD,
        "max_protocols": DEFAULT_MAX_PROTOCOLS,
        "min_edge_usd": DEFAULT_MIN_EDGE_USD,
        "min_conduit_stock_usd": DEFAULT_MIN_CONDUIT_STOCK_USD,
        "core_node_features": list(V4_CORE_NODE_FEATURES),
        "edge_state_features": list(V4_EDGE_STATE_FEATURES),
        "categories": list(categories),
        "spot_oracle_sha256": spot_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HistoricalSpotPriceOracle:
    """Exact DeFiLlama historical spot price oracle for constant-price net-flow valuation."""

    def __init__(self, bundle_path: Path | str = DEFAULT_SPOT_PRICES_BUNDLE) -> None:
        self.bundle_path = Path(bundle_path)
        raw_bytes = self.bundle_path.read_bytes()
        self.sha256 = hashlib.sha256(raw_bytes).hexdigest()
        raw = json.loads(raw_bytes.decode("utf-8"))
        self.prices_by_date: dict[str, dict[str, float]] = {
            str(d): {str(k).upper(): float(v) for k, v in pmap.items()}
            for d, pmap in raw["prices_by_date"].items()
        }

    def price_multiplier(self, raw_sym: str, from_date: str, to_ref_date: str) -> float:
        """Return `P_bench(to_ref_date) / P_bench(from_date)` to revalue `from_date` USD into `to_ref_date` prices."""
        if from_date == to_ref_date:
            return 1.0
        clean = TokenMapperV3.clean_symbol(raw_sym)
        basket = classify_numeraire_basket_v4(clean)
        if basket == "USD_STABLE":
            return 1.0

        p_from = self.prices_by_date.get(from_date)
        p_to = self.prices_by_date.get(to_ref_date)
        if not p_from or not p_to:
            return 1.0

        if basket == "ETH_BETA":
            u0, u1 = p_from.get("ETH", 0.0), p_to.get("ETH", 0.0)
            return float(np.clip(u1 / u0, 0.08, 12.5)) if (u0 > 0.0 and u1 > 0.0) else 1.0

        if basket == "BTC_BETA":
            u0, u1 = p_from.get("BTC", 0.0), p_to.get("BTC", 0.0)
            return float(np.clip(u1 / u0, 0.08, 12.5)) if (u0 > 0.0 and u1 > 0.0) else 1.0

        canon = _CANONICAL_SPOT_ALIASES.get(clean, clean)
        u0, u1 = p_from.get(canon, 0.0), p_to.get(canon, 0.0)
        if u0 > 0.0 and u1 > 0.0:
            return float(np.clip(u1 / u0, 0.08, 12.5))

        # Broad crypto market beta proxy for unindexed altcoins: ETH spot return
        e0, e1 = p_from.get("ETH", 0.0), p_to.get("ETH", 0.0)
        return float(np.clip(e1 / e0, 0.08, 12.5)) if (e0 > 0.0 and e1 > 0.0) else 1.0


@dataclass
class NetFlowStockSnapshot:
    """Stock-exposure snapshot carrying per-token compositions for fast constant-price revaluation."""

    date: str
    node_ids: list[str]
    node_index: dict[str, int]
    tvl_usd: np.ndarray                                    # [N] observed USD TVL at `date`
    is_conduit: np.ndarray                                 # [N] bool
    categories: list[str]                                  # [N] str
    edge_src: np.ndarray                                   # [E] int64
    edge_dst: np.ndarray                                   # [E] int64
    edge_usd: np.ndarray                                   # [E] observed USD stock at `date`
    edge_map: dict[tuple[str, str], float]
    proto_holdings: dict[str, dict[str, float]]            # {pid: {CLEAN_SYM: usd_at_date}}
    edge_token_breakdown: dict[tuple[str, str], dict[str, float]]  # {(u, v): {CLEAN_SYM: usd_at_date}}
    node_basket_shares: np.ndarray                         # [N, 4] float32
    edge_basket_shares: np.ndarray                         # [E, 4] float32
    features: np.ndarray | None = None                     # [N, F] float32


def build_netflow_snapshot_from_holdings(
    date_str: str,
    protocol_holdings: dict[str, dict[str, float]],
    mapper: TokenMapperV3,
    *,
    explicit_sizes: dict[str, float] | None = None,
    min_protocol_tvl_usd: float = DEFAULT_MIN_PROTOCOL_TVL_USD,
    max_protocols: int = DEFAULT_MAX_PROTOCOLS,
    min_edge_usd: float = DEFAULT_MIN_EDGE_USD,
    min_conduit_stock_usd: float = DEFAULT_MIN_CONDUIT_STOCK_USD,
) -> NetFlowStockSnapshot:
    proto_tvl: dict[str, float] = {}
    cleaned_holdings: dict[str, dict[str, float]] = {}

    for raw_pid, comp in protocol_holdings.items():
        pid = str(raw_pid).strip()
        if not pid or not pid.isdigit() or not isinstance(comp, dict):
            continue
        tok_map: dict[str, float] = defaultdict(float)
        comp_sum = 0.0
        for raw_sym, raw_val in comp.items():
            try:
                val = float(raw_val or 0.0)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(val) or val <= 0.0:
                continue
            sym = mapper.clean_symbol(raw_sym)
            if sym:
                tok_map[sym] += val
                comp_sum += val
        tvl = max(comp_sum, float((explicit_sizes or {}).get(pid, 0.0) or 0.0))
        if tvl >= min_protocol_tvl_usd:
            proto_tvl[pid] = tvl
            cleaned_holdings[pid] = dict(tok_map)

    ranked_pids = [
        pid for pid, _ in sorted(proto_tvl.items(), key=lambda kv: (-kv[1], kv[0]))[:max_protocols]
    ]
    active_proto_set = set(ranked_pids)
    active_proto_tvl = {pid: proto_tvl[pid] for pid in ranked_pids}
    kept_holdings = {pid: cleaned_holdings[pid] for pid in ranked_pids if pid in cleaned_holdings}

    token_totals: dict[str, float] = defaultdict(float)
    for pid in ranked_pids:
        for sym, val in kept_holdings.get(pid, {}).items():
            token_totals[sym] += val

    raw_pair_usd: dict[tuple[str, str], float] = defaultdict(float)
    raw_pair_toks: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    conduit_inbound_usd: dict[str, float] = defaultdict(float)
    conduit_holders: dict[str, set[str]] = defaultdict(set)

    for pid in ranked_pids:
        for sym, val in kept_holdings.get(pid, {}).items():
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
                dst_id = target
            elif not target.isdigit():
                dst_id = target
                conduit_inbound_usd[dst_id] += val
                conduit_holders[dst_id].add(pid)
            else:
                dst_id = sym
                conduit_inbound_usd[dst_id] += val
                conduit_holders[dst_id].add(pid)
            raw_pair_usd[(pid, dst_id)] += val
            raw_pair_toks[(pid, dst_id)][sym] += val

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
    n_nodes = len(node_ids)

    tvl_arr = np.zeros(n_nodes, dtype=np.float64)
    conduit_arr = np.zeros(n_nodes, dtype=bool)
    cats: list[str] = []
    node_baskets = np.zeros((n_nodes, len(NUMERAIRE_BASKETS)), dtype=np.float32)

    for idx, nid in enumerate(node_ids):
        if nid in active_proto_set:
            tvl_arr[idx] = active_proto_tvl[nid]
            conduit_arr[idx] = False
            cats.append(mapper.protocol_category(nid))
            h_map = kept_holdings.get(nid, {})
            h_sum = sum(h_map.values())
            if h_sum > 0.0:
                for sym, v in h_map.items():
                    b_i = BASKET_TO_IDX[classify_numeraire_basket_v4(sym)]
                    node_baskets[idx, b_i] += float(v / h_sum)
            else:
                b_i = BASKET_TO_IDX[classify_numeraire_basket_v4(nid)]
                node_baskets[idx, b_i] = 1.0
        else:
            tvl_arr[idx] = 0.0
            conduit_arr[idx] = True
            cats.append("Unknown")
            b_i = BASKET_TO_IDX[classify_numeraire_basket_v4(nid)]
            node_baskets[idx, b_i] = 1.0

    edge_src_list: list[int] = []
    edge_dst_list: list[int] = []
    edge_usd_list: list[float] = []
    edge_map: dict[tuple[str, str], float] = {}
    edge_tok_map: dict[tuple[str, str], dict[str, float]] = {}
    edge_baskets_list: list[list[float]] = []

    for (u, v), w_usd in sorted(raw_pair_usd.items()):
        if w_usd < min_edge_usd or u not in node_index or v not in node_index or u == v:
            continue
        edge_src_list.append(node_index[u])
        edge_dst_list.append(node_index[v])
        edge_usd_list.append(float(w_usd))
        edge_map[(u, v)] = float(w_usd)
        t_bd = dict(raw_pair_toks[(u, v)])
        edge_tok_map[(u, v)] = t_bd
        eb = [0.0, 0.0, 0.0, 0.0]
        tot_t = sum(t_bd.values())
        if tot_t > 0.0:
            for sym, tv in t_bd.items():
                eb[BASKET_TO_IDX[classify_numeraire_basket_v4(sym)]] += float(tv / tot_t)
        else:
            eb[BASKET_TO_IDX[classify_numeraire_basket_v4(v)]] = 1.0
        edge_baskets_list.append(eb)

    edge_baskets_arr = (
        np.asarray(edge_baskets_list, dtype=np.float32)
        if edge_baskets_list
        else np.zeros((0, len(NUMERAIRE_BASKETS)), dtype=np.float32)
    )

    return NetFlowStockSnapshot(
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
        proto_holdings=kept_holdings,
        edge_token_breakdown=edge_tok_map,
        node_basket_shares=node_baskets,
        edge_basket_shares=edge_baskets_arr,
    )


def build_netflow_snapshot_from_raw(
    date_str: str,
    raw_snap: dict[str, Any],
    mapper: TokenMapperV3,
    **kwargs: Any,
) -> NetFlowStockSnapshot:
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
    return build_netflow_snapshot_from_holdings(
        date_str,
        holdings,
        mapper,
        explicit_sizes=sizes,
        **kwargs,
    )


def revalue_snapshot_at_ref_date(
    snap: NetFlowStockSnapshot,
    ref_date: str,
    oracle: HistoricalSpotPriceOracle | None,
) -> tuple[dict[str, float], dict[tuple[str, str], float], dict[str, float], dict[str, float]]:
    """Return `(tvl_map, edge_map, out_usd_map, in_usd_map)` of `snap` valued at `ref_date` spot prices."""
    if oracle is None or snap.date == ref_date:
        out_m: dict[str, float] = defaultdict(float)
        in_m: dict[str, float] = defaultdict(float)
        for (u, v), w in snap.edge_map.items():
            out_m[u] += w
            in_m[v] += w
        tvl_m = {
            nid: float(snap.tvl_usd[i])
            for i, nid in enumerate(snap.node_ids)
            if not snap.is_conduit[i]
        }
        return tvl_m, dict(snap.edge_map), dict(out_m), dict(in_m)

    mult_cache: dict[str, float] = {}

    def _mult(sym: str) -> float:
        m = mult_cache.get(sym)
        if m is None:
            m = oracle.price_multiplier(sym, snap.date, ref_date)
            mult_cache[sym] = m
        return m

    tvl_m: dict[str, float] = {}
    for i, nid in enumerate(snap.node_ids):
        if snap.is_conduit[i]:
            continue
        h_map = snap.proto_holdings.get(nid)
        if h_map:
            comp_raw = sum(h_map.values())
            comp_const = sum(val * _mult(sym) for sym, val in h_map.items())
            raw_tvl = float(snap.tvl_usd[i])
            if comp_raw > 0.0 and raw_tvl > comp_raw:
                tvl_m[nid] = comp_const * (raw_tvl / comp_raw)
            else:
                tvl_m[nid] = comp_const
        else:
            tvl_m[nid] = float(snap.tvl_usd[i])

    edge_m: dict[tuple[str, str], float] = {}
    out_m: dict[str, float] = defaultdict(float)
    in_m: dict[str, float] = defaultdict(float)
    for key, t_bd in snap.edge_token_breakdown.items():
        w_const = sum(val * _mult(sym) for sym, val in t_bd.items())
        if w_const > 0.0:
            edge_m[key] = float(w_const)
            out_m[key[0]] += float(w_const)
            in_m[key[1]] += float(w_const)

    return tvl_m, edge_m, dict(out_m), dict(in_m)


def enrich_netflow_snapshots(
    snapshots: list[NetFlowStockSnapshot],
    categories_vocab: list[str] | tuple[str, ...],
    oracle: HistoricalSpotPriceOracle | None = None,
) -> list[NetFlowStockSnapshot]:
    """Populate 91-dim (`28` core + `63` category) causal constant-price Net-Flow node features."""
    cat_to_idx = {c: i for i, c in enumerate(categories_vocab)}
    unk_idx = cat_to_idx.get("Unknown", 0)
    num_core = len(V4_CORE_NODE_FEATURES)
    num_cats = len(categories_vocab)
    feat_dim = num_core + num_cats

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

        # Revalue causal lookback window [t-8 .. t] into `snap.date` constant prices (strictly causal!)
        start_k = max(0, t - 8)
        hist_tvl_log: dict[int, dict[str, float]] = {}
        hist_out_log: dict[int, dict[str, float]] = {}
        hist_in_log: dict[int, dict[str, float]] = {}
        prev_const_edges: dict[tuple[str, str], float] | None = None

        for k in range(start_k, t + 1):
            tvl_k, edge_k, out_k, in_k = revalue_snapshot_at_ref_date(snapshots[k], snap.date, oracle)
            hist_tvl_log[k] = {nid: math.log1p(max(0.0, v)) for nid, v in tvl_k.items()}
            hist_out_log[k] = {nid: math.log1p(max(0.0, v)) for nid, v in out_k.items()}
            hist_in_log[k] = {nid: math.log1p(max(0.0, v)) for nid, v in in_k.items()}
            if k == t - 1:
                prev_const_edges = edge_k

        out_retained_usd: dict[str, float] = defaultdict(float)
        if prev_const_edges is not None:
            for (u, v), w_now in snap.edge_map.items():
                w_prev = prev_const_edges.get((u, v), 0.0)
                if w_prev > 0.0:
                    out_retained_usd[u] += min(w_now, w_prev)

        for i, nid in enumerate(snap.node_ids):
            lt_now = float(log_tvl[i])
            lo_now = float(log_out[i])
            li_now = float(log_in[i])

            lt_1 = hist_tvl_log[t - 1].get(nid, lt_now) if t >= 1 else lt_now
            lt_4 = hist_tvl_log[t - 4].get(nid, lt_1) if t >= 4 else lt_1
            lt_8 = hist_tvl_log[t - 8].get(nid, lt_4) if t >= 8 else lt_4

            lo_1 = hist_out_log[t - 1].get(nid, lo_now) if t >= 1 else lo_now
            lo_4 = hist_out_log[t - 4].get(nid, lo_1) if t >= 4 else lo_1
            li_1 = hist_in_log[t - 1].get(nid, li_now) if t >= 1 else li_now
            li_4 = hist_in_log[t - 4].get(nid, li_1) if t >= 4 else li_1

            hist_vals = [
                hist_tvl_log[k].get(nid)
                for k in range(max(0, t - 4), t + 1)
                if nid in hist_tvl_log[k]
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
            age_cnt = sum(1 for k in range(t - lookback_len + 1, t + 1) if nid in hist_tvl_log[k])
            age_8w = float(age_cnt / max(1, lookback_len))

            if prev_const_edges is not None and out_usd[i] > 0.0:
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
            feats[i, 24:28] = snap.node_basket_shares[i]

            cidx = cat_to_idx.get(snap.categories[i], unk_idx)
            feats[i, num_core + cidx] = 1.0

        snap.features = np.nan_to_num(feats, nan=0.0, posinf=3.0, neginf=-3.0)

    return snapshots


def build_netflow_edge_state_features(
    snapshots: list[NetFlowStockSnapshot],
    t_idx: int,
    pair_src: np.ndarray,
    pair_dst: np.ndarray,
    oracle: HistoricalSpotPriceOracle | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 16-dim constant-price Net-Flow edge-state matrix `[P, 16]` + baseline `log1p(w_t)` `[P]`."""
    snap = snapshots[t_idx]
    assert snap.features is not None, "Call enrich_netflow_snapshots before build_netflow_edge_state_features"
    n_nodes = len(snap.node_ids)
    n_pairs = len(pair_src)
    z = np.zeros((n_pairs, len(V4_EDGE_STATE_FEATURES)), dtype=np.float32)
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

    hist_maps: list[dict[tuple[str, str], float]] = []
    for k in (t_idx - 4, t_idx - 3, t_idx - 2, t_idx - 1, t_idx):
        if k < 0:
            hist_maps.append({})
        elif k == t_idx:
            hist_maps.append(snap.edge_map)
        else:
            _, em_k, _, _ = revalue_snapshot_at_ref_date(snapshots[k], snap.date, oracle)
            hist_maps.append(em_k)

    active_basket_map = {
        (snap.node_ids[int(snap.edge_src[i])], snap.node_ids[int(snap.edge_dst[i])]): snap.edge_basket_shares[i]
        for i in range(len(snap.edge_src))
    }

    node_feats = snap.features
    for idx in range(n_pairs):
        u_i = int(pair_src[idx])
        v_i = int(pair_dst[idx])
        u_id = snap.node_ids[u_i]
        v_id = snap.node_ids[v_i]
        key = (u_id, v_id)

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

        eb = active_basket_map.get(key)
        if eb is None:
            eb = snap.node_basket_shares[v_i]

        z[idx, 0] = is_act
        z[idx, 1] = float(lw_now / 25.0)
        z[idx, 2] = share_h
        z[idx, 3] = share_i
        z[idx, 4] = float(np.clip(lw_now - lw_1, -4.0, 4.0))
        z[idx, 5] = float(np.clip(lw_now - lw_4, -5.0, 5.0))
        z[idx, 6] = age_4w
        z[idx, 7] = float(np.clip(vol_4w, 0.0, 4.0))
        z[idx, 8] = float(node_feats[u_i, 13])   # holder_netflow_mom_1w
        z[idx, 9] = float(node_feats[u_i, 14])   # holder_netflow_mom_4w
        z[idx, 10] = float(node_feats[v_i, 20])  # issuer_netflow_mom_1w
        z[idx, 11] = float(node_feats[v_i, 21])  # issuer_netflow_mom_4w
        z[idx, 12:16] = eb

    return np.nan_to_num(z, nan=0.0, posinf=4.0, neginf=-4.0), base_log_w


# ===========================================================================
# PyTorch Model Architecture: DeXposureFMV4
# ===========================================================================


class MonotonicNetFlowEdgeHead(nn.Module):
    """Joint multi-horizon Net-Flow edge head predicting fractional retention `r_{e,h} in (0, 1)`."""

    def __init__(
        self,
        embed_dim: int,
        edge_feat_dim: int = len(V4_EDGE_STATE_FEATURES),
        hidden_dim: int = 256,
        num_horizons: int = len(V4_HORIZONS),
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
        self.exist_head = nn.Linear(hidden_dim, num_horizons)
        # Fractional cumulative net-outflow hazard increments -> p_retain_frac(h) = E[min_{s<=h} min(1, w_{t+s}/w_t)]
        self.hazard_head = nn.Linear(hidden_dim, num_horizons)
        nn.init.constant_(self.hazard_head.bias, -2.8)  # ~94% initial 1w net-flow retention prior
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


class MonotonicNetFlowNodeHead(nn.Module):
    """Joint multi-horizon protocol Net-Flow log-TVL change & scale-free >=10% run hazard head."""

    def __init__(
        self,
        embed_dim: int,
        node_mom_dim: int = 16,
        hidden_dim: int = 256,
        num_horizons: int = len(V4_HORIZONS),
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
        nn.init.constant_(self.stress_hazard_head.bias, -2.4)

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
            out_denom = torch.zeros(num_nodes, dtype=h.dtype, device=device)
            out_denom.index_add_(0, edge_src, w.to(h.dtype))
            w_out = (w.to(h.dtype) / out_denom[edge_src].clamp(min=1e-6)).unsqueeze(-1)
            asset_ctx.index_add_(0, edge_src, h[edge_dst] * w_out)

            in_denom = torch.zeros(num_nodes, dtype=h.dtype, device=device)
            in_denom.index_add_(0, edge_dst, w.to(h.dtype))
            w_in = (w.to(h.dtype) / in_denom[edge_dst].clamp(min=1e-6)).unsqueeze(-1)
            liab_ctx.index_add_(0, edge_dst, h[edge_src] * w_in)

        # Slice rolling net-flow momentum + numeraire basket shares `node_features[:, 12:28]` (16 dims)
        mom_slice = node_features[:, 12:28].to(dtype=h.dtype, device=device)
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


class DeXposureFMV4(nn.Module):
    """Unified multi-horizon Price-Beta-Stripped Net-Flow Foundation Model."""

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
        self.link_scorer = MonotonicNetFlowEdgeHead(
            embed_dim=embed_dim,
            edge_feat_dim=len(V4_EDGE_STATE_FEATURES),
            hidden_dim=hidden_dim,
            num_horizons=len(V4_HORIZONS),
            dropout=dropout,
        )
        self.node_head = MonotonicNetFlowNodeHead(
            embed_dim=embed_dim,
            node_mom_dim=16,
            hidden_dim=hidden_dim,
            num_horizons=len(V4_HORIZONS),
            dropout=dropout,
        )
        self.register_buffer("retain_hazard_scales", torch.ones(len(V4_HORIZONS), dtype=torch.float32))
        self.register_buffer("stress_hazard_scales", torch.ones(len(V4_HORIZONS), dtype=torch.float32))
        self.register_buffer("exist_logit_temps", torch.ones(len(V4_HORIZONS), dtype=torch.float32))
        self.register_buffer("exist_logit_biases", torch.zeros(len(V4_HORIZONS), dtype=torch.float32))

    def encode(
        self,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        features: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        return _encode_with_graphpfn(self.encoder, edge_src, edge_dst, features, device)

    def calibrated_retain_probs(self, cum_hazard: torch.Tensor) -> torch.Tensor:
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


class FMV4Predictor:
    """Calibrated multi-horizon Net-Flow predictor with single-pass encoder memoization."""

    model_family = "dexposure_fm_v4"

    def __init__(
        self,
        checkpoint_dir: Path | str = DEFAULT_V4_CKPT_DIR,
        device: str | None = None,
        pi_min: float = 0.0,
        mapping_bundle: Path | str = DEFAULT_MAPPING_BUNDLE,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pi_min = float(pi_min)
        self.mapper = TokenMapperV3(mapping_bundle)

        schema_path = self.checkpoint_dir / "feature_schema.json"
        unified_ckpt = self.checkpoint_dir / "dexposure-fm-v4.pt"
        h1_ckpt = self.checkpoint_dir / "dexposure-fm-h1.pt"
        self._ckpt_path = unified_ckpt if unified_ckpt.is_file() else h1_ckpt
        self._available = schema_path.is_file() and self._ckpt_path.is_file()

        if self._available:
            self._schema = json.loads(schema_path.read_text(encoding="utf-8"))
            self._categories = list(self._schema["categories"])
        else:
            self._schema = {}
            self._categories = list(self.mapper.categories)

        self._model: DeXposureFMV4 | None = None
        self._memo_cache: dict[str, dict[str, Any]] = {}

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
        return list(V4_HORIZONS) if self._available else []

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
            "schema_version": self._schema.get("schema_version", 4),
            "settlement_basis": "constant_price_net_flow_ex_eth_btc_beta",
            "pi_min": self.pi_min,
        }

    def _ensure_loaded(self) -> DeXposureFMV4:
        if self._model is not None:
            return self._model
        if not self._available:
            raise FileNotFoundError(f"DeXposure-FM v4 checkpoint missing in {self.checkpoint_dir}")
        from lib.graphpfn.model import GraphPFN

        payload = torch.load(self._ckpt_path, map_location=self.device, weights_only=False)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        encoder = GraphPFN(edge_head=False)
        hidden_dim = int(payload.get("hidden_dim", 256)) if isinstance(payload, dict) else 256
        model = DeXposureFMV4(
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

    def _graph_snapshot_to_netflow_stock(self, gs: GraphSnapshot) -> NetFlowStockSnapshot:
        """Convert a (pre-revalued constant-price lookback) `GraphSnapshot` into `NetFlowStockSnapshot`."""
        node_ids = sorted(gs.nodes.keys())
        node_index = {nid: i for i, nid in enumerate(node_ids)}
        n_nodes = len(node_ids)
        tvl_arr = np.zeros(n_nodes, dtype=np.float64)
        cond_arr = np.zeros(n_nodes, dtype=bool)
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

        node_baskets = np.zeros((n_nodes, len(NUMERAIRE_BASKETS)), dtype=np.float32)
        out_tot = np.zeros(n_nodes, dtype=np.float64)
        src_l, dst_l, w_l = [], [], []
        edge_map: dict[tuple[str, str], float] = {}
        edge_tok_map: dict[tuple[str, str], dict[str, float]] = {}
        edge_baskets_list: list[list[float]] = []

        for (u, v), w in sorted(edge_agg.items()):
            u_i = node_index[u]
            v_i = node_index[v]
            src_l.append(u_i)
            dst_l.append(v_i)
            w_l.append(w)
            edge_map[(u, v)] = w
            edge_tok_map[(u, v)] = {v: w}
            b_v = BASKET_TO_IDX[classify_numeraire_basket_v4(v)]
            eb = [0.0, 0.0, 0.0, 0.0]
            eb[b_v] = 1.0
            edge_baskets_list.append(eb)
            node_baskets[u_i, b_v] += float(w)
            out_tot[u_i] += float(w)

        for i, nid in enumerate(node_ids):
            if out_tot[i] > 0.0:
                node_baskets[i] /= float(out_tot[i])
            else:
                node_baskets[i, BASKET_TO_IDX[classify_numeraire_basket_v4(nid)]] = 1.0

        edge_baskets_arr = (
            np.asarray(edge_baskets_list, dtype=np.float32)
            if edge_baskets_list
            else np.zeros((0, len(NUMERAIRE_BASKETS)), dtype=np.float32)
        )
        return NetFlowStockSnapshot(
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
            proto_holdings={},
            edge_token_breakdown=edge_tok_map,
            node_basket_shares=node_baskets,
            edge_basket_shares=edge_baskets_arr,
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

        # Note: in live serving, `history_graphs` from `build_stock_history_at_timestamp` are
        # ALREADY revalued into `current_graph.date` constant prices, so `oracle=None` preserves them!
        stock_seq = [self._graph_snapshot_to_netflow_stock(g) for g in seq_gs]
        enrich_netflow_snapshots(stock_seq, self._categories, oracle=None)
        curr_stock = stock_seq[-1]
        t_idx = len(stock_seq) - 1

        edge_z, base_log_w = build_netflow_edge_state_features(
            stock_seq, t_idx, curr_stock.edge_src, curr_stock.edge_dst, oracle=None
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
                retain_probs = np.zeros((0, len(V4_HORIZONS)), dtype=np.float32)
                delta_log_w = np.zeros((0, len(V4_HORIZONS)), dtype=np.float32)

            node_out = model.node_head(h, feat_t, src_t, dst_t, bw_t)
            delta_log_tvl = node_out["delta_log_tvl"].cpu().numpy()
            stress_probs = model.calibrated_stress_probs(node_out["cum_stress_hazard"]).cpu().numpy()

        bundle = {
            "stock": curr_stock,
            "retain_probs": retain_probs,
            "delta_log_w": delta_log_w,
            "delta_log_tvl": delta_log_tvl,
            "stress_probs": stress_probs,
        }
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
        if horizon not in HORIZON_TO_IDX:
            raise ValueError(f"Unsupported FM v4 horizon {horizon}; expected {V4_HORIZONS}")
        h_idx = HORIZON_TO_IDX[horizon]
        joint = self._run_joint_forward(current_graph, history_graphs=history_graphs)
        curr_stock: NetFlowStockSnapshot = joint["stock"]
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
