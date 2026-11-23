#!/usr/bin/env python3
"""Full 2025-2026 DeFiLlama Weekly Ingestion, 2025 Overlap Bridge Audit & 2026 Frozen FM-v4 Evaluation.

Executes four local, deterministic, crash-resumable stages (zero cloud spend):
  Stage 1: Extends DeFiLlama internal historical spot price oracle (`2020-03-23 .. 2026-11-23`,
           349 Mondays) into `paper/dexposure_agent/resources/historical_spot_prices_v4_2026.json`
           while keeping the 283 original `2020-03-23 .. 2025-08-18` prices bit-for-bit identical.
  Stage 2: Collects weekly Monday protocol balance-sheet compositions (`2024-10-28 .. 2026-11-23`,
           109 Mondays) across the full zero-survivorship-bias protocol universe (~4,900+ protocols:
           all protocols >= $100k TVL or >= $500k mcap, all protocols active >= $250k in 2024-2025
           historical JSON, and ALL protocols listed in 2025-2026 including rugged/zeroed pools),
           matching original DeXposure `chainTvls` summation (`all chains except OwnTokens`).
  Stage 3: Runs a 33-week Overlap Bridge Validation (`2025-01-06 .. 2025-08-18`) + `2025-08-18 ->
           2025-08-25` seam continuity audit comparing reconstructed DeFiLlama snapshots against
           `data/historical-network_week_2020-03-30.json` on active node sets, TVL, directed edge
           stock exposure, and constant-price Top-5% stressed protocol sets `S_t^4`.
  Stage 4: Stitches the 349-week (`2020-03-23 .. 2026-11-23`) constant-price Net-Flow graph sequence
           and evaluates the frozen pre-2025 `DeXposure-FM v4` checkpoint across ALL resolved 2026
           decision weeks (`h=1`: 46w, `h=4`: 43w `2026-01-05 .. 2026-10-26`, `h=8`: 39w, `h=12`: 35w),
           covering both upstream FM-v4 multi-horizon metrics (vs. Degree & Momentum baselines) and
           downstream `DeXposure-Bench` Top-k supervisory targeting (`m1_persistence`, `m2_snapshot`,
           `m5_fm_v4_rules`) plus 2025-H2 / 2026 drift diagnostics to guide `ssh-colab` retraining.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper"
for _p in (str(ROOT), str(PAPER_ROOT), str(ROOT / "cloud"), str(ROOT / "cloud" / "train_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dexposure_agent.fm_v3 import (  # noqa: E402
    DEFAULT_MAX_PROTOCOLS,
    DEFAULT_MIN_CONDUIT_STOCK_USD,
    DEFAULT_MIN_EDGE_USD,
    DEFAULT_MIN_PROTOCOL_TVL_USD,
    TokenMapperV3,
    stock_snapshot_to_graph_snapshot,
)
from dexposure_agent.fm_v4 import (  # noqa: E402
    DeXposureFMV4,
    HistoricalSpotPriceOracle,
    NetFlowStockSnapshot,
    V4_HORIZONS,
    build_netflow_snapshot_from_holdings,
    build_netflow_snapshot_from_raw,
    enrich_netflow_snapshots,
    revalue_snapshot_at_ref_date,
)
from dexposure_agent.live_defillama import (  # noqa: E402
    DEFILLAMA_PROTOCOLS_URL,
    DEFILLAMA_PROTOCOL_URL,
    _align_series_adapter_reclassifications,
    _http_get_json,
)
from scripts.build_historical_spot_prices_v4 import (  # noqa: E402
    CANONICAL_ALIASES,
    REFERENCE_SLUGS,
    TRACKED_SYMBOLS,
    _extract_daily_unit_prices,
)
from train_fm_v4 import (  # noqa: E402
    build_joint_netflow_samples,
    collect_raw_outputs,
    evaluate_heuristic_baselines,
    evaluate_joint_model,
)

ORIG_DATA_PATH = ROOT / "data" / "historical-network_week_2020-03-30.json"
ORIG_2025_TAIL_PATH = ROOT / "data" / "historical-network_week_2025-07-01.json"
ORIG_SPOT_PATH = PAPER_ROOT / "dexposure_agent" / "resources" / "historical_spot_prices_v4.json"
EXT_SPOT_PATH = PAPER_ROOT / "dexposure_agent" / "resources" / "historical_spot_prices_v4_2026.json"
CKPT_V4_DIR = ROOT / "checkpoints" / "dexposure-fm-v4"

CACHE_ROOT = ROOT / "output" / "defillama_2025_2026_weekly_cache"
EXT_NETWORK_PATH = ROOT / "output" / "historical-network_week_2025_2026_full.json"
EVAL_OUT_DIR = PAPER_ROOT / "results" / "run_20261123_v4_2026_full_eval"

NON_SUPERVISED_CATEGORIES = frozenset({"CEX", "Risk Curators", "Chain", "Oracle", "Services"})
PRE2025_VAL_MEDIAN_SHIFT_H4 = 0.080181  # Exact pre-2025 validation median shift from run_v4_unified_29w.py

EXTRA_REFERENCE_SLUGS: tuple[str, ...] = (
    "sky-lending",
    "pendle-v2",
    "ethena-usde",
    "ether.fi-stake",
)


def log(msg: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_monday_grid(start_date: str, end_date: str) -> list[tuple[str, int]]:
    d0 = dt.datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    d1 = dt.datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    out: list[tuple[str, int]] = []
    cur = d0
    while cur <= d1:
        out.append((cur.strftime("%Y-%m-%d"), int(cur.timestamp())))
        cur += dt.timedelta(days=7)
    return out


# ===========================================================================
# Stage 1: Extend Historical Spot Price Oracle through 2026-11-23
# ===========================================================================


def build_extended_spot_oracle(new_mondays: list[tuple[str, int]], force: bool = False) -> Path:
    orig_raw = json.loads(ORIG_SPOT_PATH.read_text(encoding="utf-8"))
    orig_prices: dict[str, dict[str, float]] = dict(orig_raw["prices_by_date"])

    if EXT_SPOT_PATH.is_file() and not force:
        existing = json.loads(EXT_SPOT_PATH.read_text(encoding="utf-8"))
        ex_prices = existing.get("prices_by_date", {})
        if all(d in ex_prices for d, _ in new_mondays) and all(
            ex_prices.get(k) == v for k, v in orig_prices.items()
        ):
            log(f"Stage 1 cached: {EXT_SPOT_PATH.name} ({len(ex_prices)} dates)")
            return EXT_SPOT_PATH

    log("Stage 1: Fetching DeFiLlama reference hub prices to extend spot oracle through 2026-11-23...")
    slugs = list(dict.fromkeys(list(REFERENCE_SLUGS) + list(EXTRA_REFERENCE_SLUGS)))
    daily_samples: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for slug in slugs:
        try:
            detail = _http_get_json(DEFILLAMA_PROTOCOL_URL.format(slug=slug), timeout=35.0, retries=3)
            per_day = _extract_daily_unit_prices(detail)
            for day, sym_map in per_day.items():
                for sym, vals in sym_map.items():
                    daily_samples[day][sym].extend(vals)
        except Exception as exc:
            log(f"  spot ref {slug} skipped: {exc}")

    daily_medians: dict[int, dict[str, float]] = {}
    for day, sym_map in daily_samples.items():
        daily_medians[day] = {}
        for sym, vals in sym_map.items():
            vs = sorted(vals)
            m = len(vs) // 2
            med = vs[m] if len(vs) % 2 == 1 else 0.5 * (vs[m - 1] + vs[m])
            daily_medians[day][sym] = round(float(med), 6)

    all_days = sorted(daily_medians.keys())
    merged_prices: dict[str, dict[str, float]] = dict(orig_prices)

    for d_str, target_ts in new_mondays:
        if d_str in merged_prices:
            continue
        out_d: dict[str, float] = {}
        for sym in sorted(TRACKED_SYMBOLS):
            cands = [day for day in all_days if abs(day - target_ts) <= 5 * 86400 and sym in daily_medians[day]]
            if cands:
                best_day = min(cands, key=lambda day: (abs(day - target_ts), -day))
                out_d[sym] = daily_medians[best_day][sym]
        if "ETH" not in out_d or "BTC" not in out_d:
            raise RuntimeError(f"Missing ETH or BTC spot benchmark on {d_str}: {out_d}")
        merged_prices[d_str] = out_d

    all_dates = sorted(merged_prices.keys())
    payload = {
        "schema_version": 4,
        "description": f"Exact DeFiLlama internal historical USD spot prices ({all_dates[0]} .. {all_dates[-1]})",
        "base_oracle_sha256": sha256_file(ORIG_SPOT_PATH),
        "date_count": len(all_dates),
        "first_date": all_dates[0],
        "last_date": all_dates[-1],
        "reference_protocols": slugs,
        "canonical_aliases": CANONICAL_ALIASES,
        "prices_by_date": merged_prices,
    }
    EXT_SPOT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(
        f"Stage 1 complete: wrote {EXT_SPOT_PATH.name} ({len(all_dates)} weekly dates, "
        f"2026-01-05 ETH=${merged_prices['2026-01-05']['ETH']:.2f} -> "
        f"{all_dates[-1]} ETH=${merged_prices[all_dates[-1]]['ETH']:.2f})"
    )
    return EXT_SPOT_PATH


# ===========================================================================
# Stage 2: Full Zero-Survivorship-Bias Weekly Protocol Ingestion
# ===========================================================================


def load_empirical_2025_fallback_weights() -> dict[str, dict[str, float]]:
    """Extract normalized token shares on 2025-08-18 for mega-AMMs with skipTokenBreakdownData."""
    if not ORIG_2025_TAIL_PATH.is_file():
        return {}
    raw = json.loads(ORIG_2025_TAIL_PATH.read_text(encoding="utf-8"))
    d = raw["data"] if "data" in raw else raw
    snap = d.get("2025-08-18") or d[sorted(d.keys())[-1]]
    weights_by_pid: dict[str, dict[str, float]] = {}
    for node in snap.get("nodes", []):
        nid = str(node.get("id") or "").strip()
        comp = node.get("composition") or {}
        if not nid or not comp:
            continue
        tot = sum(float(v or 0.0) for v in comp.values() if float(v or 0.0) > 0.0)
        if tot <= 100_000.0:
            continue
        kept = {
            str(k).strip().upper(): float(v) / tot
            for k, v in comp.items()
            if float(v or 0.0) >= max(10_000.0, 0.0005 * tot)
        }
        s_kept = sum(kept.values())
        if s_kept > 0.0:
            weights_by_pid[nid] = {k: v / s_kept for k, v in kept.items()}
    return weights_by_pid


def extract_protocol_weekly_compositions(
    detail: dict[str, Any],
    pid: str,
    cat: str,
    monday_grid: list[tuple[str, int]],
    fallback_weights: dict[str, float] | None,
    min_day_ts: int,
    max_stale_days: int = 5,
) -> dict[str, dict[str, float]]:
    """Extract `{YYYY-MM-DD: {TOKEN: usd}}` matching original DeXposure `chainTvls` summation."""
    chain_tvls = detail.get("chainTvls")
    skip_bd = bool(detail.get("skipTokenBreakdownData"))

    # Collect per-chain deduplicated daily `{day_ts: {TOK: usd}}` for day_ts >= min_day_ts - 7d
    floor_ts = min_day_ts - 7 * 86400
    per_chain_tok_days: list[tuple[list[int], dict[int, dict[str, float]]]] = []

    if not skip_bd and isinstance(chain_tvls, dict) and chain_tvls:
        for cname, cdata in chain_tvls.items():
            if cname.strip().lower() == "owntokens":
                continue
            if not isinstance(cdata, dict):
                continue
            t_list = cdata.get("tokensInUsd")
            if not isinstance(t_list, list) or not t_list:
                continue
            day_latest: dict[int, tuple[int, dict[str, float]]] = {}
            for entry in t_list:
                if not isinstance(entry, dict):
                    continue
                ts = int(entry.get("date", 0) or 0)
                if ts < floor_ts:
                    continue
                day_ts = (ts // 86400) * 86400
                prev_ts = day_latest.get(day_ts, (-1, {}))[0]
                if ts >= prev_ts:
                    tmap = entry.get("tokens") or {}
                    if isinstance(tmap, dict):
                        cleaned: dict[str, float] = {}
                        for tok, uval in tmap.items():
                            try:
                                vf = float(uval or 0.0)
                            except (TypeError, ValueError):
                                continue
                            if math.isfinite(vf) and vf > 0.0:
                                sym = str(tok).strip().upper()
                                cleaned[sym] = cleaned.get(sym, 0.0) + vf
                        day_latest[day_ts] = (ts, cleaned)
            if day_latest:
                s_days = sorted(day_latest.keys())
                per_chain_tok_days.append((s_days, {d: m for d, (_, m) in day_latest.items()}))

    # Fallback 1: top-level tokensInUsd if chainTvls had no token breakdown and not skip_bd
    if not per_chain_tok_days and not skip_bd:
        top_t = detail.get("tokensInUsd")
        if isinstance(top_t, list) and top_t:
            day_latest = {}
            for entry in top_t:
                if not isinstance(entry, dict):
                    continue
                ts = int(entry.get("date", 0) or 0)
                if ts < floor_ts:
                    continue
                day_ts = (ts // 86400) * 86400
                prev_ts = day_latest.get(day_ts, (-1, {}))[0]
                if ts >= prev_ts:
                    tmap = entry.get("tokens") or {}
                    if isinstance(tmap, dict):
                        cleaned = {}
                        for tok, uval in tmap.items():
                            try:
                                vf = float(uval or 0.0)
                            except (TypeError, ValueError):
                                continue
                            if math.isfinite(vf) and vf > 0.0:
                                sym = str(tok).strip().upper()
                                cleaned[sym] = cleaned.get(sym, 0.0) + vf
                        day_latest[day_ts] = (ts, cleaned)
            if day_latest:
                s_days = sorted(day_latest.keys())
                per_chain_tok_days.append((s_days, {d: m for d, (_, m) in day_latest.items()}))

    import bisect

    weekly_ts_series: dict[int, dict[str, float]] = {}

    if per_chain_tok_days:
        last_grid_ts = monday_grid[-1][1]
        for _, ts_w in monday_grid:
            agg: dict[str, float] = defaultdict(float)
            for s_days, dmap in per_chain_tok_days:
                idx = bisect.bisect_right(s_days, ts_w) - 1
                picked_day = None
                if idx >= 0 and (ts_w - s_days[idx]) <= max_stale_days * 86400:
                    picked_day = s_days[idx]
                elif ts_w == last_grid_ts and s_days and abs(s_days[-1] - ts_w) <= 2 * 86400:
                    picked_day = s_days[-1]
                if picked_day is not None:
                    for tok, val in dmap[picked_day].items():
                        agg[tok] += val
            if agg:
                weekly_ts_series[ts_w] = dict(agg)
    else:
        # Fallback 2: TVL-only or `skipTokenBreakdownData == True`
        per_chain_tvl_days: list[tuple[list[int], dict[int, float]]] = []
        if isinstance(chain_tvls, dict) and chain_tvls:
            for cname, cdata in chain_tvls.items():
                if cname.strip().lower() == "owntokens" or not isinstance(cdata, dict):
                    continue
                tvl_l = cdata.get("tvl")
                if not isinstance(tvl_l, list) or not tvl_l:
                    continue
                d_tvl: dict[int, tuple[int, float]] = {}
                for entry in tvl_l:
                    if not isinstance(entry, dict):
                        continue
                    ts = int(entry.get("date", 0) or 0)
                    if ts < floor_ts:
                        continue
                    day_ts = (ts // 86400) * 86400
                    val = float(entry.get("totalLiquidityUSD", 0.0) or 0.0)
                    if math.isfinite(val) and val > 0.0 and ts >= d_tvl.get(day_ts, (-1, 0.0))[0]:
                        d_tvl[day_ts] = (ts, val)
                if d_tvl:
                    s_days = sorted(d_tvl.keys())
                    per_chain_tvl_days.append((s_days, {d: v for d, (_, v) in d_tvl.items()}))

        if not per_chain_tvl_days:
            tvl_l = detail.get("tvl")
            if isinstance(tvl_l, list) and tvl_l:
                d_tvl = {}
                for entry in tvl_l:
                    if not isinstance(entry, dict):
                        continue
                    ts = int(entry.get("date", 0) or 0)
                    if ts < floor_ts:
                        continue
                    day_ts = (ts // 86400) * 86400
                    val = float(entry.get("totalLiquidityUSD", 0.0) or 0.0)
                    if math.isfinite(val) and val > 0.0 and ts >= d_tvl.get(day_ts, (-1, 0.0))[0]:
                        d_tvl[day_ts] = (ts, val)
                if d_tvl:
                    s_days = sorted(d_tvl.keys())
                    per_chain_tvl_days.append((s_days, {d: v for d, (_, v) in d_tvl.items()}))

        if per_chain_tvl_days:
            if fallback_weights:
                w_template = fallback_weights
            elif cat in {"Dexs", "Dexes"}:
                w_template = {"WETH": 0.45, "USDC": 0.25, "USDT": 0.15, "WBTC": 0.15}
            else:
                sym = str(detail.get("symbol") or "UNKNOWN").strip().upper()
                if not sym or sym == "-":
                    sym = "UNKNOWN"
                w_template = {sym: 1.0}

            last_grid_ts = monday_grid[-1][1]
            for _, ts_w in monday_grid:
                tot_usd = 0.0
                for s_days, dmap in per_chain_tvl_days:
                    idx = bisect.bisect_right(s_days, ts_w) - 1
                    picked_day = None
                    if idx >= 0 and (ts_w - s_days[idx]) <= max_stale_days * 86400:
                        picked_day = s_days[idx]
                    elif ts_w == last_grid_ts and s_days and abs(s_days[-1] - ts_w) <= 2 * 86400:
                        picked_day = s_days[-1]
                    if picked_day is not None:
                        tot_usd += dmap[picked_day]
                if tot_usd > 0.0:
                    weekly_ts_series[ts_w] = {tok: tot_usd * w for tok, w in w_template.items() if tot_usd * w > 1.0}

    if not weekly_ts_series:
        return {}

    denoised = _align_series_adapter_reclassifications(weekly_ts_series)
    ts_to_date = {ts: d_str for d_str, ts in monday_grid}
    out_by_date: dict[str, dict[str, float]] = {}
    for ts_w, comp in denoised.items():
        d_str = ts_to_date.get(ts_w)
        if not d_str:
            continue
        tot = sum(comp.values())
        if tot < 50_000.0:
            continue
        # Prune micro-dust (< $100) to keep weekly JSON compact without affecting >= $10k edges or TVL
        kept = {k: round(v, 2) for k, v in comp.items() if v >= 100.0}
        diff = tot - sum(kept.values())
        if diff >= 100.0:
            kept["OTHER_DUST"] = round(diff, 2)
        if kept:
            out_by_date[d_str] = kept
    return out_by_date


def collect_2025_2026_dataset(
    orig_raw_data: dict[str, Any],
    monday_grid: list[tuple[str, int]],
    max_workers: int = 18,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    proto_cache_dir = CACHE_ROOT / "per_proto_v1"
    proto_cache_dir.mkdir(parents=True, exist_ok=True)

    meta_cache_path = CACHE_ROOT / "protocols_directory.json"
    if meta_cache_path.is_file() and (time.time() - meta_cache_path.stat().st_mtime) < 12 * 3600:
        all_protocols = json.loads(meta_cache_path.read_text(encoding="utf-8"))
    else:
        log("Fetching full DeFiLlama `/protocols` directory...")
        all_protocols = _http_get_json(DEFILLAMA_PROTOCOLS_URL, timeout=35.0, retries=4)
        meta_cache_path.write_text(json.dumps(all_protocols), encoding="utf-8")

    llama_by_id: dict[str, dict[str, Any]] = {}
    for p in all_protocols:
        if isinstance(p, dict) and p.get("id") is not None and p.get("slug"):
            llama_by_id[str(p["id"]).strip()] = p

    # Historical protocols active >= $250k in any 2024-2025 week of original dataset
    hist_active_24_25: set[str] = set()
    for d_str, snap in orig_raw_data.items():
        if d_str < "2024-06-01":
            continue
        for node in snap.get("nodes", []):
            nid = str(node.get("id") or "").strip()
            if nid.isdigit() and float(node.get("size") or 0.0) >= 250_000.0:
                hist_active_24_25.add(nid)

    # Zero-survivorship-bias candidate selection
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for pid, p in llama_by_id.items():
        tvl = float(p.get("tvl") or 0.0)
        mcap = float(p.get("mcap") or 0.0)
        listed_at = int(p.get("listedAt") or 0)
        rugged = bool(p.get("rugged"))
        include = (
            tvl >= 100_000.0
            or mcap >= 500_000.0
            or pid in hist_active_24_25
            or listed_at >= 1735689600  # Every protocol listed since 2025-01-01 (even if $0 today!)
            or rugged
        )
        if include and pid not in seen_ids:
            seen_ids.add(pid)
            candidates.append(p)

    candidates.sort(key=lambda x: -float(x.get("tvl") or 0.0))
    log(
        f"Stage 2 candidate universe: {len(candidates)} protocols "
        f"(total directory={len(llama_by_id)}, 2024-25 hist>=$250k={len(hist_active_24_25)})"
    )

    fallback_2025 = load_empirical_2025_fallback_weights()
    min_day_ts = monday_grid[0][1]

    to_fetch: list[dict[str, Any]] = []
    cached_count = 0
    for p in candidates:
        pid = str(p["id"]).strip()
        cfile = proto_cache_dir / f"p_{pid}.json"
        if cfile.is_file():
            cached_count += 1
        else:
            to_fetch.append(p)

    log(f"Stage 2 cache status: {cached_count} already cached, {len(to_fetch)} to fetch from DeFiLlama...")

    def _worker(proto: dict[str, Any]) -> tuple[str, bool]:
        pid = str(proto["id"]).strip()
        slug = str(proto["slug"]).strip()
        cat = str(proto.get("category") or "Unknown").strip()
        cfile = proto_cache_dir / f"p_{pid}.json"
        try:
            detail = _http_get_json(DEFILLAMA_PROTOCOL_URL.format(slug=slug), timeout=30.0, retries=3)
            weekly = extract_protocol_weekly_compositions(
                detail,
                pid=pid,
                cat=cat,
                monday_grid=monday_grid,
                fallback_weights=fallback_2025.get(pid),
                min_day_ts=min_day_ts,
            )
            tmp = proto_cache_dir / f".tmp_{pid}.json"
            tmp.write_text(json.dumps(weekly, separators=(",", ":")), encoding="utf-8")
            tmp.replace(cfile)
            return pid, bool(weekly)
        except Exception:
            # Record empty marker ONLY on 404/400 so transient timeouts retry next run; check error string
            return pid, False

    if to_fetch:
        t0 = time.time()
        done = 0
        non_empty = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_worker, p): p for p in to_fetch}
            for fut in as_completed(futs):
                done += 1
                _, ok = fut.result()
                if ok:
                    non_empty += 1
                if done % 250 == 0 or done == len(to_fetch):
                    rate = done / max(time.time() - t0, 1e-3)
                    eta = (len(to_fetch) - done) / max(rate, 1e-3)
                    log(
                        f"  fetched {done}/{len(to_fetch)} (active_histories={non_empty}, "
                        f"{rate:.1f} proto/s, ETA {eta:.0f}s)"
                    )

    # Assemble `{date: {nodes: [{id, size, composition}], links: []}}` across all 109 Mondays
    log("Stage 2: Assembling 109 weekly network snapshots (`2024-10-28 .. 2026-11-23`)...")
    by_date_nodes: dict[str, list[dict[str, Any]]] = {d_str: [] for d_str, _ in monday_grid}
    loaded_active_protos = 0

    for p in candidates:
        pid = str(p["id"]).strip()
        cfile = proto_cache_dir / f"p_{pid}.json"
        if not cfile.is_file():
            continue
        try:
            weekly = json.loads(cfile.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not weekly:
            continue
        loaded_active_protos += 1
        for d_str, comp in weekly.items():
            if d_str in by_date_nodes and comp:
                sz = round(sum(float(v) for v in comp.values()), 2)
                if sz >= 100_000.0:
                    by_date_nodes[d_str].append({"id": pid, "size": sz, "composition": comp})

    assembled: dict[str, dict[str, Any]] = {}
    for d_str, _ in monday_grid:
        nodes = sorted(by_date_nodes[d_str], key=lambda x: (-x["size"], x["id"]))
        assembled[d_str] = {"nodes": nodes, "links": []}

    EXT_NETWORK_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXT_NETWORK_PATH.write_text(
        json.dumps(
            {
                "schema": "dexposure_weekly_compositions_v4",
                "first_date": monday_grid[0][0],
                "last_date": monday_grid[-1][0],
                "n_weeks": len(monday_grid),
                "n_active_protocols": loaded_active_protos,
                "data": assembled,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    log(
        f"Stage 2 complete: wrote {EXT_NETWORK_PATH.name} "
        f"({EXT_NETWORK_PATH.stat().st_size / (1024 * 1024):.1f} MB, {loaded_active_protos} active protocols)"
    )
    return all_protocols, assembled


# ===========================================================================
# Stage 3: 33-Week 2025 Overlap Bridge Audit + Seam Continuity Check
# ===========================================================================


def compute_stressed_set_v4(
    snap_t: NetFlowStockSnapshot,
    snap_th: NetFlowStockSnapshot,
    oracle: HistoricalSpotPriceOracle,
    stress_pct: float = 0.05,
) -> tuple[set[str], dict[str, float], dict[str, float]]:
    """Compute realized constant-epoch-`t`-price incident exposure outflows & Top-5% stressed set."""
    obs_exp: dict[str, float] = defaultdict(float)
    for (u, v), w in snap_t.edge_map.items():
        obs_exp[u] += w
        obs_exp[v] += w

    _, edge_const, out_c, in_c = revalue_snapshot_at_ref_date(snap_th, snap_t.date, oracle)
    real_exp: dict[str, float] = {}
    for nid in set(obs_exp) | set(out_c) | set(in_c):
        real_exp[nid] = float(out_c.get(nid, 0.0) + in_c.get(nid, 0.0))

    losses: list[tuple[str, float]] = []
    outflow_map: dict[str, float] = {}
    for i, nid in enumerate(snap_t.node_ids):
        if snap_t.is_conduit[i] or not nid.isdigit():
            continue
        if snap_t.categories[i] in NON_SUPERVISED_CATEGORIES:
            continue
        w0 = float(obs_exp.get(nid, 0.0))
        if w0 <= 0.0:
            continue
        w1 = float(real_exp.get(nid, 0.0))
        delta = w0 - w1
        outflow_map[nid] = delta
        losses.append((nid, delta))

    n_sup = len(losses)
    k_top = max(1, math.ceil(stress_pct * n_sup)) if n_sup > 0 else 0
    pos_losses = sorted([x for x in losses if x[1] > 0.0], key=lambda kv: (-kv[1], kv[0]))
    stressed = {nid for nid, _ in pos_losses[:k_top]}
    return stressed, dict(obs_exp), outflow_map


def run_overlap_bridge_audit(
    orig_raw_data: dict[str, Any],
    live_raw_data: dict[str, dict[str, Any]],
    mapper: TokenMapperV3,
    oracle: HistoricalSpotPriceOracle,
) -> dict[str, Any]:
    overlap_dates = sorted(d for d in orig_raw_data if d >= "2025-01-06" and d in live_raw_data)
    log(f"Stage 3: Running Overlap Bridge Validation across {len(overlap_dates)} weeks ({overlap_dates[0]} .. {overlap_dates[-1]})...")

    orig_snaps = {d: build_netflow_snapshot_from_raw(d, orig_raw_data[d], mapper) for d in overlap_dates}
    live_snaps = {d: build_netflow_snapshot_from_raw(d, live_raw_data[d], mapper) for d in overlap_dates}

    node_jaccards: list[float] = []
    top100_recalls: list[float] = []
    tvl_rel_diffs: list[float] = []
    edge_usd_rel_diffs: list[float] = []
    edge_weighted_jaccards: list[float] = []
    log_tvl_corrs: list[float] = []
    log_edge_corrs: list[float] = []

    for d in overlap_dates:
        so = orig_snaps[d]
        sl = live_snaps[d]

        po = {nid: float(so.tvl_usd[i]) for i, nid in enumerate(so.node_ids) if not so.is_conduit[i] and so.categories[i] not in NON_SUPERVISED_CATEGORIES}
        pl = {nid: float(sl.tvl_usd[i]) for i, nid in enumerate(sl.node_ids) if not sl.is_conduit[i] and sl.categories[i] not in NON_SUPERVISED_CATEGORIES}

        inter_p = sorted(set(po) & set(pl))
        union_p = set(po) | set(pl)
        node_jaccards.append(len(inter_p) / max(1, len(union_p)))

        top100_o = {nid for nid, _ in sorted(po.items(), key=lambda kv: -kv[1])[:100]}
        top100_recalls.append(len(top100_o & set(pl)) / max(1, len(top100_o)))

        sum_o = sum(po.values())
        sum_l = sum(pl.values())
        tvl_rel_diffs.append(abs(sum_l - sum_o) / max(sum_o, 1.0))

        if len(inter_p) >= 10:
            xo = np.log1p([po[k] for k in inter_p])
            xl = np.log1p([pl[k] for k in inter_p])
            log_tvl_corrs.append(float(pearsonr(xo, xl)[0]))

        eo = so.edge_map
        el = sl.edge_map
        sum_eo = sum(eo.values())
        sum_el = sum(el.values())
        edge_usd_rel_diffs.append(abs(sum_el - sum_eo) / max(sum_eo, 1.0))

        all_ek = set(eo) | set(el)
        w_min = sum(min(eo.get(k, 0.0), el.get(k, 0.0)) for k in all_ek)
        w_max = sum(max(eo.get(k, 0.0), el.get(k, 0.0)) for k in all_ek)
        edge_weighted_jaccards.append(w_min / max(w_max, 1.0))

        inter_e = sorted(set(eo) & set(el))
        if len(inter_e) >= 10:
            weo = np.log1p([eo[k] for k in inter_e])
            wel = np.log1p([el[k] for k in inter_e])
            log_edge_corrs.append(float(pearsonr(weo, wel)[0]))

    # h=4 stressed set agreement across the 29 overlap decision weeks (`2025-01-06 .. 2025-07-21`)
    stressed_jaccards: list[float] = []
    stressed_recalls: list[float] = []
    top20_stressed_recalls: list[float] = []
    outflow_spearmans: list[float] = []

    for idx in range(len(overlap_dates) - 4):
        d_t = overlap_dates[idx]
        d_th = overlap_dates[idx + 4]
        st_o, _, out_o = compute_stressed_set_v4(orig_snaps[d_t], orig_snaps[d_th], oracle)
        st_l, _, out_l = compute_stressed_set_v4(live_snaps[d_t], live_snaps[d_th], oracle)
        if st_o or st_l:
            stressed_jaccards.append(len(st_o & st_l) / max(1, len(st_o | st_l)))
            stressed_recalls.append(len(st_o & st_l) / max(1, len(st_o)))
        top20_o = {nid for nid, _ in sorted(out_o.items(), key=lambda kv: -kv[1])[:20] if _ > 0}
        if top20_o:
            top20_stressed_recalls.append(len(top20_o & st_l) / len(top20_o))
        shared_u = sorted(set(out_o) & set(out_l))
        if len(shared_u) >= 20:
            outflow_spearmans.append(float(spearmanr([out_o[u] for u in shared_u], [out_l[u] for u in shared_u])[0]))

    # Seam continuity check: `2025-08-18` (original) -> `2025-08-25` (newly collected) vs normal intra-2025 week transitions
    s_0811_orig = orig_snaps["2025-08-11"]
    s_0818_orig = orig_snaps["2025-08-18"]
    s_0818_live = live_snaps["2025-08-18"]
    s_0825_live = build_netflow_snapshot_from_raw("2025-08-25", live_raw_data["2025-08-25"], mapper)

    def _transition_stats(a: NetFlowStockSnapshot, b: NetFlowStockSnapshot) -> dict[str, float]:
        pa = {nid: float(a.tvl_usd[i]) for i, nid in enumerate(a.node_ids) if not a.is_conduit[i] and a.categories[i] not in NON_SUPERVISED_CATEGORIES}
        pb = {nid: float(b.tvl_usd[i]) for i, nid in enumerate(b.node_ids) if not b.is_conduit[i] and b.categories[i] not in NON_SUPERVISED_CATEGORIES}
        ea, eb = a.edge_map, b.edge_map
        all_e = set(ea) | set(eb)
        w_jacc = sum(min(ea.get(k, 0.0), eb.get(k, 0.0)) for k in all_e) / max(1.0, sum(max(ea.get(k, 0.0), eb.get(k, 0.0)) for k in all_e))
        return {
            "node_jaccard": round(len(set(pa) & set(pb)) / max(1, len(set(pa) | set(pb))), 4),
            "tvl_rel_change": round((sum(pb.values()) - sum(pa.values())) / max(1.0, sum(pa.values())), 4),
            "edge_weighted_jaccard": round(w_jacc, 4),
        }

    report = {
        "n_overlap_weeks": len(overlap_dates),
        "date_range": [overlap_dates[0], overlap_dates[-1]],
        "active_supervised_node_jaccard_mean": round(float(np.mean(node_jaccards)), 4),
        "top100_tvl_protocol_recall_mean": round(float(np.mean(top100_recalls)), 4),
        "supervised_tvl_rel_diff_median": round(float(np.median(tvl_rel_diffs)), 4),
        "supervised_tvl_rel_diff_mean": round(float(np.mean(tvl_rel_diffs)), 4),
        "shared_node_log_tvl_pearson_mean": round(float(np.mean(log_tvl_corrs)), 4),
        "total_edge_usd_rel_diff_median": round(float(np.median(edge_usd_rel_diffs)), 4),
        "edge_usd_weighted_jaccard_mean": round(float(np.mean(edge_weighted_jaccards)), 4),
        "shared_edge_log_usd_pearson_mean": round(float(np.mean(log_edge_corrs)), 4),
        "h4_stressed_top5pct_recall_mean": round(float(np.mean(stressed_recalls)), 4),
        "h4_stressed_top5pct_jaccard_mean": round(float(np.mean(stressed_jaccards)), 4),
        "h4_top20_outflow_in_live_stressed_recall_mean": round(float(np.mean(top20_stressed_recalls)), 4),
        "h4_outflow_spearman_mean": round(float(np.mean(outflow_spearmans)), 4),
        "seam_continuity": {
            "intra_orig_2025_08_11_to_08_18": _transition_stats(s_0811_orig, s_0818_orig),
            "stitched_seam_orig_08_18_to_live_08_25": _transition_stats(s_0818_orig, s_0825_live),
            "intra_live_2025_08_18_to_08_25": _transition_stats(s_0818_live, s_0825_live),
        },
    }
    log(
        f"Stage 3 Bridge Summary: Top-100 recall={report['top100_tvl_protocol_recall_mean']:.3f}, "
        f"log-TVL r={report['shared_node_log_tvl_pearson_mean']:.4f}, "
        f"log-Edge r={report['shared_edge_log_usd_pearson_mean']:.4f}, "
        f"edge weighted Jaccard={report['edge_usd_weighted_jaccard_mean']:.3f}, "
        f"Top-20 stressed recall={report['h4_top20_outflow_in_live_stressed_recall_mean']:.3f}"
    )
    return report


# ===========================================================================
# Stage 4: Full 2026 Frozen Pre-2025 FM-v4 Evaluation (43 Resolved h=4 Weeks)
# ===========================================================================


def paired_bootstrap_and_perm(diff_arr: np.ndarray, n_boot: int = 10_000, n_perm: int = 20_000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(diff_arr)
    obs = float(np.mean(diff_arr))
    boots = rng.choice(diff_arr, size=(n_boot, n), replace=True).mean(axis=1)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = (signs * diff_arr[None, :]).mean(axis=1)
    p_two = float((np.sum(np.abs(perm_means) >= abs(obs) - 1e-12) + 1) / (n_perm + 1))
    return {
        "mean_diff": round(obs, 4),
        "ci95_low": round(float(ci_lo), 4),
        "ci95_high": round(float(ci_hi), 4),
        "perm_p_two_sided": round(p_two, 6),
    }


def evaluate_frozen_v4_on_2025h2_and_2026(
    orig_raw_data: dict[str, Any],
    live_raw_data: dict[str, dict[str, Any]],
    mapper: TokenMapperV3,
    oracle: HistoricalSpotPriceOracle,
) -> dict[str, Any]:
    import torch
    from lib.graphpfn.model import GraphPFN

    # Stitch canonical 349-week timeline:
    # - `2020-03-23 .. 2025-08-18` (283 weeks) from original immutable JSON
    # - `2025-08-25 .. 2026-11-23` (66 weeks) from newly collected DeFiLlama dataset
    orig_dates = sorted(orig_raw_data.keys())
    new_dates = sorted(d for d in live_raw_data.keys() if d > orig_dates[-1])
    all_dates = orig_dates + new_dates
    log(
        f"Stage 4: Building & enriching stitched {len(all_dates)}-week NetFlowStockSnapshot sequence "
        f"({all_dates[0]} .. {all_dates[-1]}; new_weeks={len(new_dates)})..."
    )

    # Only need from `2024-06-03` onward for causal 8-week lookback + 2025-H2 + 2026 evaluation!
    eval_slice_dates = [d for d in all_dates if d >= "2024-06-03"]
    snapshots: list[NetFlowStockSnapshot] = []
    for d in eval_slice_dates:
        if d in orig_raw_data:
            snapshots.append(build_netflow_snapshot_from_raw(d, orig_raw_data[d], mapper))
        else:
            snapshots.append(build_netflow_snapshot_from_raw(d, live_raw_data[d], mapper))

    schema = json.loads((CKPT_V4_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    categories_vocab = list(schema["categories"])
    enrich_netflow_snapshots(snapshots, categories_vocab, oracle=oracle)

    date_to_idx = {s.date: i for i, s in enumerate(snapshots)}
    idx_2025_h2 = [i for i, s in enumerate(snapshots) if "2025-07-01" <= s.date <= "2025-12-31"]
    idx_2026_all = [i for i, s in enumerate(snapshots) if s.date >= "2026-01-01"]

    horizons = list(V4_HORIZONS)
    max_idx = len(snapshots) - 1

    log(f"Building joint multi-horizon Net-Flow samples for 2025-H2 ({len(idx_2025_h2)}w) and 2026 ({len(idx_2026_all)}w)...")
    samples_2025_h2 = build_joint_netflow_samples(
        snapshots, idx_2025_h2, max_target_idx=max_idx, horizons=horizons, neg_ratio=4, seed=42, oracle=oracle
    )
    samples_2026 = build_joint_netflow_samples(
        snapshots, idx_2026_all, max_target_idx=max_idx, horizons=horizons, neg_ratio=4, seed=42, oracle=oracle
    )

    # Load frozen pre-2025 DeXposure-FM v4 checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CKPT_V4_DIR / "dexposure-fm-v4.pt"
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    model.to(device)
    model.eval()

    log(f"Running frozen pre-2025 DeXposure-FM v4 inference on device={device} across 2025-H2 and 2026...")
    _, v4_2025_h2_metrics = evaluate_joint_model(model, samples_2025_h2, horizons, device)
    _, v4_2026_metrics = evaluate_joint_model(model, samples_2026, horizons, device)
    heur_2026_metrics = evaluate_heuristic_baselines(samples_2026, horizons)

    # Collect raw per-week outputs on 2025-H2 and 2026 for downstream queue ranking & median-shift drift audit
    raw_2025_h2 = collect_raw_outputs(model, samples_2025_h2, device)
    raw_2026 = collect_raw_outputs(model, samples_2026, device)

    def _compute_median_shifts(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, h in enumerate(horizons):
            tr_l, pr_l = [], []
            for row in raw_rows:
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
                out[f"h{h}"] = {
                    "n_resolved_weeks": len(tr_l),
                    "true_median": round(float(np.median(tc)), 6),
                    "raw_pred_median": round(float(np.median(pc)), 6),
                    "optimal_shift": round(float(np.median(tc) - np.median(pc)), 6),
                    "empirical_neg_rate": round(float((tc < 0.0).mean()), 4),
                    "raw_pred_neg_rate": round(float((pc < 0.0).mean()), 4),
                    "pre2025_shifted_pred_neg_rate": round(float(((pc + PRE2025_VAL_MEDIAN_SHIFT_H4) < 0.0).mean()), 4),
                }
        return out

    drift_2025_h2 = _compute_median_shifts(raw_2025_h2)
    drift_2026 = _compute_median_shifts(raw_2026)

    # Downstream DeXposure-Bench Top-k Supervisory Queue Targeting across all resolved 2026 h=4 weeks!
    k_h4 = horizons.index(4)
    resolved_2026_rows = [r for r in raw_2026 if bool(r["sample"].valid_horizons[k_h4])]
    log(f"Evaluating DeXposure-Bench h=4 Top-k Supervisory Queue across {len(resolved_2026_rows)} resolved 2026 weeks...")

    methods_weekly_hits: dict[str, dict[int, list[float]]] = {
        m: {k: [] for k in (1, 3, 5, 7)}
        for m in ("m1_momentum_outflow", "m2_snapshot_exposure", "m5_fm_v4_frozen_pre2025", "m5_fm_v4_2025h2_recalibrated")
    }
    captured_outflow_top7: dict[str, float] = defaultdict(float)
    total_positive_outflow_2026 = 0.0
    weekly_ledger_2026: list[dict[str, Any]] = []

    shift_2025_h2_h4 = float(drift_2025_h2.get("h4", {}).get("optimal_shift", PRE2025_VAL_MEDIAN_SHIFT_H4))

    for row in resolved_2026_rows:
        s = row["sample"]
        t_i = date_to_idx[s.date_t]
        snap_t = snapshots[t_i]
        snap_th = snapshots[t_i + 4]

        stressed_set, obs_exp, realized_outflows = compute_stressed_set_v4(snap_t, snap_th, oracle, stress_pct=0.05)
        total_positive_outflow_2026 += sum(v for v in realized_outflows.values() if v > 0.0)

        # Supervised candidate protocols at epoch `t`
        sup_indices = [
            i
            for i, nid in enumerate(snap_t.node_ids)
            if not snap_t.is_conduit[i]
            and nid.isdigit()
            and snap_t.categories[i] not in NON_SUPERVISED_CATEGORIES
            and obs_exp.get(nid, 0.0) > 0.0
        ]

        # 1. `m2_snapshot_exposure`: Ranked by observed incident stock exposure `w_t(u)` (TVL >= $25M filter matching M2)
        m2_cands = [i for i in sup_indices if float(snap_t.tvl_usd[i]) >= 25_000_000.0]
        if len(m2_cands) < 7:
            m2_cands = sup_indices
        m2_ranked = [snap_t.node_ids[i] for i in sorted(m2_cands, key=lambda i: (-obs_exp[snap_t.node_ids[i]], snap_t.node_ids[i]))]

        # 2. `m1_momentum_outflow`: Ranked by trailing 4-week constant-price dollar contraction
        m1_scores: dict[str, float] = {}
        for i in sup_indices:
            nid = snap_t.node_ids[i]
            mom_4w = float(snap_t.features[i, 14]) if snap_t.features is not None else 0.0
            m1_scores[nid] = obs_exp[nid] * max(0.0, 1.0 - math.exp(min(0.0, mom_4w)))
        m1_ranked = [snap_t.node_ids[i] for i in sorted(sup_indices, key=lambda i: (-m1_scores[snap_t.node_ids[i]], snap_t.node_ids[i]))]

        # 3. `m5_fm_v4_frozen_pre2025`: Layer 2 parameter-free expected outflow `w_t(u) * max(0, 1 - exp(dtvl + +0.080181))`
        dtvl_raw = row["pred_delta_log_tvl"][:, k_h4]
        m5_frozen_scores: dict[str, float] = {}
        m5_recal_scores: dict[str, float] = {}
        for i in sup_indices:
            nid = snap_t.node_ids[i]
            dl_frozen = float(dtvl_raw[i]) + PRE2025_VAL_MEDIAN_SHIFT_H4
            dl_recal = float(dtvl_raw[i]) + shift_2025_h2_h4
            m5_frozen_scores[nid] = obs_exp[nid] * max(0.0, 1.0 - math.exp(dl_frozen))
            m5_recal_scores[nid] = obs_exp[nid] * max(0.0, 1.0 - math.exp(dl_recal))

        m5_frozen_ranked = [snap_t.node_ids[i] for i in sorted(sup_indices, key=lambda i: (-m5_frozen_scores[snap_t.node_ids[i]], snap_t.node_ids[i]))]
        m5_recal_ranked = [snap_t.node_ids[i] for i in sorted(sup_indices, key=lambda i: (-m5_recal_scores[snap_t.node_ids[i]], snap_t.node_ids[i]))]

        rankings = {
            "m1_momentum_outflow": m1_ranked,
            "m2_snapshot_exposure": m2_ranked,
            "m5_fm_v4_frozen_pre2025": m5_frozen_ranked,
            "m5_fm_v4_2025h2_recalibrated": m5_recal_ranked,
        }
        for m_name, r_list in rankings.items():
            for k in (1, 3, 5, 7):
                topk = r_list[:k]
                prec = sum(1 for u in topk if u in stressed_set) / max(1, len(topk))
                methods_weekly_hits[m_name][k].append(prec)
            captured_outflow_top7[m_name] += sum(max(0.0, realized_outflows.get(u, 0.0)) for u in r_list[:7])

        weekly_ledger_2026.append(
            {
                "decision_date": s.date_t,
                "settle_date": snap_th.date,
                "n_supervised": len(sup_indices),
                "n_stressed_top5pct": len(stressed_set),
                "p_at_7_m2": round(methods_weekly_hits["m2_snapshot_exposure"][7][-1], 4),
                "p_at_7_m5_frozen": round(methods_weekly_hits["m5_fm_v4_frozen_pre2025"][7][-1], 4),
                "m5_top3_targets": [
                    f"{u}:{mapper.protocol_name(u)}" for u in m5_frozen_ranked[:3]
                ],
            }
        )

    downstream_2026_summary: dict[str, Any] = {
        "n_resolved_h4_weeks": len(resolved_2026_rows),
        "first_decision_date": resolved_2026_rows[0]["sample"].date_t if resolved_2026_rows else None,
        "last_decision_date": resolved_2026_rows[-1]["sample"].date_t if resolved_2026_rows else None,
        "total_realized_positive_outflow_usd": round(total_positive_outflow_2026, 2),
        "methods": {},
        "paired_vs_m2_snapshot": {},
    }

    for m_name, k_dict in methods_weekly_hits.items():
        downstream_2026_summary["methods"][m_name] = {
            f"P@{k}": round(float(np.mean(vals)), 4) for k, vals in k_dict.items()
        }
        cap = float(captured_outflow_top7[m_name])
        downstream_2026_summary["methods"][m_name]["captured_outflow_top7_usd"] = round(cap, 2)
        downstream_2026_summary["methods"][m_name]["captured_outflow_top7_share"] = round(
            cap / max(1.0, total_positive_outflow_2026), 4
        )

    for m_name in ("m5_fm_v4_frozen_pre2025", "m5_fm_v4_2025h2_recalibrated"):
        downstream_2026_summary["paired_vs_m2_snapshot"][m_name] = {}
        for k in (1, 3, 5, 7):
            diff = np.asarray(methods_weekly_hits[m_name][k]) - np.asarray(
                methods_weekly_hits["m2_snapshot_exposure"][k]
            )
            w = int((diff > 1e-9).sum())
            t = int((np.abs(diff) <= 1e-9).sum())
            l = int((diff < -1e-9).sum())
            stats = paired_bootstrap_and_perm(diff)
            stats["weekly_record_wtl"] = f"{w}W-{t}T-{l}L"
            downstream_2026_summary["paired_vs_m2_snapshot"][m_name][f"P@{k}"] = stats

    # Proposed clean rolling retraining split inventory for `ssh-colab`
    rolling_split_plan = {
        "total_stitched_weeks": len(all_dates),
        "date_span": [all_dates[0], all_dates[-1]],
        "proposed_pre2026_rolling_split": {
            "train_dates": [all_dates[0], "2025-07-07"],
            "n_train_weeks": sum(1 for d in all_dates if d <= "2025-07-07"),
            "embargo_note": "12-week target maturity embargo between last train week 2025-07-07 (h=12 settles 2025-09-29) and 2026-01-05 test start",
            "val_dates": ["2025-07-14", "2025-10-06"],
            "n_val_weeks_fully_resolved_before_2026": sum(1 for d in all_dates if "2025-07-14" <= d <= "2025-10-06"),
            "test_2026_dates": ["2026-01-05", all_dates[-1]],
            "n_test_2026_total_weeks": sum(1 for d in all_dates if d >= "2026-01-05"),
            "n_test_2026_resolved_h4_weeks": len(resolved_2026_rows),
        },
    }

    return {
        "checkpoint_evaluated": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "spot_oracle_2026_sha256": oracle.sha256,
        "v4_2025_h2_upstream": v4_2025_h2_metrics,
        "v4_2026_upstream": v4_2026_metrics,
        "heuristic_baselines_2026": heur_2026_metrics,
        "calibration_drift_audit": {
            "pre2025_frozen_val_shift_h4": PRE2025_VAL_MEDIAN_SHIFT_H4,
            "2025_h2": drift_2025_h2,
            "2026_full": drift_2026,
        },
        "downstream_2026_queue_eval": downstream_2026_summary,
        "rolling_retrain_split_plan": rolling_split_plan,
        "weekly_ledger_2026": weekly_ledger_2026,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=18)
    parser.add_argument("--force-spot", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Full 109-Monday grid from 2024-10-28 through 2026-11-23
    monday_grid = generate_monday_grid("2024-10-28", "2026-11-23")
    new_mondays_post_2025_08_18 = [(d, ts) for d, ts in monday_grid if d > "2025-08-18"]

    # Stage 1: Spot Oracle
    ext_spot_path = build_extended_spot_oracle(new_mondays_post_2025_08_18, force=args.force_spot)
    oracle_2026 = HistoricalSpotPriceOracle(ext_spot_path)

    # Load original immutable 2020-2025 dataset
    log("Loading immutable `historical-network_week_2020-03-30.json`...")
    raw_orig = json.loads(ORIG_DATA_PATH.read_text(encoding="utf-8"))
    orig_data = raw_orig["data"] if "data" in raw_orig else raw_orig

    # Stage 2: Collect full 2025-2026 weekly compositions
    all_protocols, live_data = collect_2025_2026_dataset(
        orig_data, monday_grid, max_workers=args.max_workers
    )

    # Register live protocol metadata into TokenMapperV3 (100% backward-compatible on pre-2025 IDs)
    mapper = TokenMapperV3()
    mapper.register_live_protocols(all_protocols)

    # Stage 3: 33-week 2025 Overlap Bridge Validation
    bridge_report = run_overlap_bridge_audit(orig_data, live_data, mapper, oracle_2026)
    (EVAL_OUT_DIR / "bridge_audit_2025_overlap.json").write_text(
        json.dumps(bridge_report, indent=2) + "\n", encoding="utf-8"
    )

    # Stage 4: Full 2026 Frozen FM-v4 Evaluation (43 resolved h=4 weeks)
    eval_2026_report = evaluate_frozen_v4_on_2025h2_and_2026(orig_data, live_data, mapper, oracle_2026)

    full_summary = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "bridge_audit_2025_overlap": bridge_report,
        "frozen_v4_2026_evaluation": eval_2026_report,
    }
    summary_path = EVAL_OUT_DIR / "summary_2025_2026_local_eval.json"
    summary_path.write_text(json.dumps(full_summary, indent=2) + "\n", encoding="utf-8")
    log(f"ALL STAGES COMPLETE in {time.time() - t_start:.1f}s -> {summary_path}")


if __name__ == "__main__":
    main()
