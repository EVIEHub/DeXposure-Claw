"""Build the DeXposure-Claw dashboard assets from real DeFiLlama data and DeXposure-FM.

Every number written by this module is either read from DeFiLlama, produced by a
DeXposure-FM checkpoint, computed by a documented closed-form formula on those
two, or copied from a committed result file. Missing values are written as
``null`` and rendered as "—" by the dashboard; nothing is filled with defaults.

Outputs (in ``docs/design/dexposure-claw-case-workbench/assets/`` and mirrored to
``cloudflare/public/assets/``):

* ``live-history.json``   walk-forward ledger from ``live_history.generate_initial_history``
                          plus the 2022 Terra/FTX replay summaries.
* ``live-horizons.json``  M7 queue, M2 candidates, and gate state for h = 1, 4, 8 weeks.
* ``live-network-3d.json`` per-node metrics for h = 1, 4, 8 and a display backbone.
* ``live-contagion-graph.json`` the full exposure graph with FM edge survival
                          probabilities, used by the contagion simulator.

Definitions (all per protocol u, horizon h):

* W_u(t)       incident exposure, sum of observed edge weights touching u.
* p_e          LinkScorer existence probability of observed edge e at t+h.
* E[dW_u]      sum over edges e touching u of w_e * (1 - p_e). Same as fm_predictor.
* flowDropPct  100 * E[dW_u] / W_u(t).
* poolDropPct  100 * E[dW_u] / max(TVL_u + W_u(t), W_u(t), 1). Same as live_defillama.
* HHI          sum of squared shares of u's incident edge weights.
* isBaseHub    u is a token issuer node (non-numeric id) or TVL_u >= $1B.
* Predicted contagion graph keeps observed edges with p_e >= PI_MIN and uses
  FM-predicted TVL (node head) as the loss buffer. Observed graph keeps all
  edges with current TVL. Contagion is ``contagion.simulate_contagion_traced``.

Run:  .venv/bin/python -m dexposure_agent.dashboard_assets   (from ``paper/``)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import gc
import json
import logging
import math
import re
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from dexposure_agent.contagion import simulate_contagion_traced
from dexposure_agent.fm_predictor import FMPredictor
from dexposure_agent.fm_v4 import FMV4Predictor
from dexposure_agent.live_defillama import (
    BetaNeutralizedFMV3Predictor,
    LiveDeFiLlamaCollector,
    _incident_exposure_map,
    run_live_prediction_pipeline,
)
from dexposure_agent.live_history import (
    DEFAULT_CACHE_DIR,
    build_snapshot_at_timestamp,
    build_stock_history_at_timestamp,
    generate_initial_history,
    load_cached_protocol_quantity_series,
    load_cached_protocol_series,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
WORKBENCH = _REPO_ROOT / "docs" / "design" / "dexposure-claw-case-workbench"
ASSETS = WORKBENCH / "assets"
CF_ASSETS = WORKBENCH / "cloudflare" / "public" / "assets"
REPLAY_PATH = _REPO_ROOT / "outputs" / "pre2022_clean_replay_20261103.json"

HORIZONS = (1, 4, 8)
PI_MIN = 0.5            # same local default as serve._get_fm when DEXPOSURE_FM_PI_MIN is unset
BASE_HUB_TVL_USD = 1e9
SPILLOVER_SHOCK = 0.5
DISPLAY_NODES = 120
DISPLAY_EDGES = 600

TOKEN_ICON_URLS: dict[str, str] = {
    "USDC": "https://icons.llamao.fi/icons/pegged/usd-coin?w=64&h=64",
    "USDT": "https://icons.llamao.fi/icons/pegged/tether?w=64&h=64",
    "USDE": "https://icons.llamao.fi/icons/pegged/ethena-usde?w=64&h=64",
    "PYUSD": "https://icons.llamao.fi/icons/pegged/paypal-usd?w=64&h=64",
    "FDUSD": "https://icons.llamao.fi/icons/pegged/first-digital-usd?w=64&h=64",
    "DAI": "https://icons.llamao.fi/icons/pegged/dai?w=64&h=64",
    "WETH": "https://icons.llamao.fi/icons/chains/rsz_ethereum.jpg?w=64&h=64",
    "WBTC": "https://icons.llamao.fi/icons/protocols/wbtc?w=64&h=64",
    "BTCB": "https://icons.llamao.fi/icons/protocols/binance-bitcoin?w=64&h=64",
    "CBBTC": "https://icons.llamao.fi/icons/protocols/coinbase-wrapped-btc?w=64&h=64",
    "BNB": "https://icons.llamao.fi/icons/chains/rsz_binance.jpg?w=64&h=64",
    "SOL": "https://icons.llamao.fi/icons/chains/rsz_solana.jpg?w=64&h=64",
    "TON": "https://icons.llamao.fi/icons/chains/rsz_ton.jpg?w=64&h=64",
    "SUI": "https://icons.llamao.fi/icons/chains/rsz_sui.jpg?w=64&h=64",
    "ARB": "https://icons.llamao.fi/icons/chains/rsz_arbitrum.jpg?w=64&h=64",
    "OP": "https://icons.llamao.fi/icons/chains/rsz_optimism.jpg?w=64&h=64",
    "AVAX": "https://icons.llamao.fi/icons/chains/rsz_avalanche.jpg?w=64&h=64",
}


def _safe_icon_stem(nid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", nid)


def _write_fallback_monogram_webp(out: Path, label: str) -> bool:
    """Generate a clean 64x64 monochrome monogram .webp icon when remote logo is unavailable."""
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (64, 64), (243, 241, 235))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((2, 2, 61, 61), radius=14, fill=(29, 29, 31), outline=(210, 207, 198), width=2)
        txt = re.sub(r"[^A-Za-z0-9]", "", label.upper())[:4] or "DF"
        # Centered crisp geometric badge text
        x_pos = max(8, 32 - 4 * len(txt))
        draw.text((x_pos, 26), txt, fill=(250, 249, 245))
        img.save(out, format="WEBP", quality=90)
        return out.is_file() and out.stat().st_size > 100
    except Exception as exc:
        logger.warning("Fallback icon generation failed for %s: %s", label, exc)
        return False


def _cache_display_icons(
    display_ids: set[str],
    proto_logos: dict[str, str],
    names: dict[str, str] | None = None,
) -> dict[str, str]:
    icons_dir = ASSETS / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    def remote_for(nid: str) -> str | None:
        if nid in TOKEN_ICON_URLS:
            return TOKEN_ICON_URLS[nid]
        u = proto_logos.get(nid.strip("<>"))
        if u:
            return u + ("&w=64&h=64" if "?" in u else "?w=64&h=64")
        return None

    def fetch_one(nid: str) -> tuple[str, str | None]:
        out = icons_dir / f"{_safe_icon_stem(nid)}.webp"
        if out.is_file() and out.stat().st_size > 100:
            return nid, f"/assets/icons/{out.name}"
        url = remote_for(nid)
        if url:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = resp.read()
                    if len(data) >= 101:
                        out.write_bytes(data)
                        return nid, f"/assets/icons/{out.name}"
            except Exception as exc:
                logger.warning("Icon fetch fallback to monogram for %s: %s", nid, exc)
        lbl = (names or {}).get(nid) or nid
        if _write_fallback_monogram_webp(out, str(lbl)):
            return nid, f"/assets/icons/{out.name}"
        return nid, url

    local_map: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        for nid, resolved_url in ex.map(fetch_one, sorted(display_ids)):
            if resolved_url:
                local_map[nid] = resolved_url
    return local_map


def _r(x: float | None, nd: int = 2) -> float | None:
    if x is None or not math.isfinite(float(x)):
        return None
    return round(float(x), nd)


def _tvl(snap, nid: str) -> float:
    node = snap.nodes.get(nid)
    if node is None or not node.log_size or node.log_size <= 0:
        return 0.0
    return math.expm1(float(node.log_size))


def _aggregate(edges) -> dict[tuple[str, str], float]:
    agg: dict[tuple[str, str], float] = defaultdict(float)
    for e in edges:
        if e.weight > 0:
            agg[(str(e.source), str(e.target))] += float(e.weight)
    return dict(agg)


def _replay_summaries(collector: LiveDeFiLlamaCollector) -> dict[str, Any]:
    """Copy 2022 Terra/FTX clean-replay results verbatim (no recomputation)."""
    if not REPLAY_PATH.is_file():
        return {}
    raw = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for key in ("terra", "ftx"):
        rk = raw.get(key)
        if not rk:
            continue
        cands = {str(c["protocol"]): c for c in rk.get("candidate_set", [])}
        m7 = (rk.get("gated_llm_outputs") or [{}])[0].get("target_protocols", [])
        m6 = {str(x["protocol"]): x for x in (rk.get("raw_llm_outputs") or [{}])[0].get("target_protocols", [])}
        stressed = {str(x) for x in rk.get("truly_stressed", [])}
        actual = rk.get("actual_losses", {}) or {}
        tickets = []
        for i, t in enumerate(m7):
            pid = str(t["protocol"])
            c = cands.get(pid, {})
            raw_t = m6.get(pid)
            tickets.append({
                "rank": i + 1,
                "entityId": pid,
                "targetName": collector.mapper.protocol_name(pid),
                "category": c.get("category") or collector.mapper.protocol_category(pid),
                "exposureUsd": _r(c.get("weight")),
                "scenarioMaxLossUsd": _r(c.get("scenario_max_loss")),
                "rawAction": raw_t.get("action") if raw_t else None,
                "ticketAction": t.get("action"),
                "demoted": bool(raw_t and raw_t.get("action") != t.get("action")),
                "inStressed": pid in stressed,
                "actualLossUsd": _r(actual.get(pid)),
            })
        a7 = rk.get("assessment_m7", {}) or {}
        a6 = rk.get("assessment_m6", {}) or {}
        out[key] = {
            "caseKey": key,
            "date": rk.get("date"),
            "futureDate": rk.get("future_date"),
            "horizonWeeks": rk.get("horizon"),
            "nNodes": rk.get("n_nodes"),
            "nEdges": rk.get("n_edges"),
            "nAlerts": rk.get("n_alerts"),
            "meanConfidence": rk.get("mean_confidence"),
            "precisionM7": a7.get("precision"),
            "precisionM6": a6.get("precision"),
            "firM7": a7.get("false_intervention_rate"),
            "firM6": a6.get("false_intervention_rate"),
            "nStressed": len(stressed),
            "scenarios": [
                {
                    "scenarioId": s.get("scenario_id"),
                    "name": s.get("name"),
                    "totalLossUsd": _r(s.get("total_loss")),
                    "contagionLossUsd": _r(s.get("contagion_loss")),
                    "nAffected": s.get("n_affected"),
                }
                for s in rk.get("scenarios", [])
            ],
            "tickets": tickets,
        }
    return out


def build(cache_dir: Path = DEFAULT_CACHE_DIR, *, refresh_live: bool = False) -> dict[str, Any]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    today_tag = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    has_today = any(cache_dir.glob(f"proto_*_{today_tag}.json"))
    has_today_qty = any(cache_dir.glob(f"qty_*_{today_tag}.json"))
    has_any = any(cache_dir.glob("proto_*.json"))
    if refresh_live or not has_any or not has_today or not has_today_qty:
        logger.info(
            "Fetching fresh DeFiLlama protocol token & quantity series into %s "
            "(refresh_live=%s, has_today=%s, has_today_qty=%s)...",
            cache_dir,
            refresh_live,
            has_today,
            has_today_qty,
        )
        live_collector = LiveDeFiLlamaCollector(
            top_n_protocols=1250,
            min_protocol_tvl=1e6,
            max_workers=24,
            cache_dir=cache_dir,
        )
        live_collector.collect_live_snapshots(
            focus_ids=["2786", "182", "1599", "118"],
            min_node_size=0.0,
        )
        del live_collector
        gc.collect()

    # 1. Walk-forward ledger (repo generator, unchanged logic).
    history = generate_initial_history(
        horizon=4,
        cache_dir=cache_dir,
        output_path=ASSETS / "live-history.json",
        horizons_path=None,
    )
    gc.collect()

    collector = LiveDeFiLlamaCollector(top_n_protocols=1250, min_protocol_tvl=1e6, cache_dir=cache_dir)
    meta = cache_dir / "protocols_meta.json"
    proto_logos: dict[str, str] = {}
    if meta.is_file():
        meta_list = json.loads(meta.read_text(encoding="utf-8"))
        collector.register_protocols_metadata(meta_list)
        proto_logos = {
            str(x["id"]): str(x["logo"])
            for x in meta_list
            if isinstance(x, dict) and x.get("id") is not None and x.get("logo")
        }

    history["historicalCrisisReplays"] = _replay_summaries(collector)
    v4_rep_path = _REPO_ROOT / "checkpoints" / "dexposure-fm-v4" / "v4_holdout_report.json"
    v4_base_path = _REPO_ROOT / "checkpoints" / "dexposure-fm-v4" / "baseline_comparison.json"
    if v4_rep_path.is_file():
        v4_rep = json.loads(v4_rep_path.read_text(encoding="utf-8"))
        v4_base = (
            json.loads(v4_base_path.read_text(encoding="utf-8")).get("test", {})
            if v4_base_path.is_file()
            else {}
        )
        history["v4Holdout33Weeks"] = {
            "dateRange": "2025-01-06 .. 2025-08-18 (33 weeks)",
            "settlementBasis": "constant_price_net_flow_ex_eth_btc_beta",
            "horizons": v4_rep,
            "baselines": v4_base,
        }
    (ASSETS / "live-history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. Latest snapshot, DeXposure-FM v4 (or BetaNeutralized FM v3 fallback) forecasts for h = 1, 4, 8.
    series = load_cached_protocol_series(cache_dir)
    qty_series = load_cached_protocol_quantity_series(cache_dir)
    max_ts = max(max(s.keys()) for s in series.values() if s)
    fm_v4 = FMV4Predictor(pi_min=0.0)
    if fm_v4.available:
        fm: Any = fm_v4
        use_v3 = True
    else:
        fm_v3 = BetaNeutralizedFMV3Predictor(pi_min=0.0)
        use_v3 = fm_v3.available
        fm = fm_v3 if use_v3 else FMPredictor(pi_min=0.0)

    if use_v3:
        curr, prev, bundle, hist_graphs = build_stock_history_at_timestamp(
            series, collector, max_ts, protocol_qty_series=qty_series
        )
    else:
        curr, prev, bundle = build_snapshot_at_timestamp(series, collector, max_ts, min_node_size=0.0)
        hist_graphs = [prev, curr]
    del qty_series

    obs_w = _incident_exposure_map(curr)
    edges_obs = _aggregate(curr.edges)
    edges_prev = _aggregate(prev.edges)

    probs_by_h: dict[int, dict[tuple[str, str], float]] = {}
    stress_by_h: dict[int, dict[str, float]] = {}
    pred_by_h = {}
    res_by_h: dict[int, dict[str, Any]] = {}
    n_series = len(series)
    del series
    gc.collect()
    for h in HORIZONS:
        if not use_v3:
            fm._models.clear()
            fm._link_scorers.clear()
            fm._node_heads.clear()
        gc.collect()
        res_by_h[h] = run_live_prediction_pipeline(
            top_n_protocols=n_series,
            horizon=h,
            min_node_size=0.0,
            mc_samples_count=50,
            fm_predictor=fm,
            precollected=(curr, prev, bundle, collector),
            history_graphs=hist_graphs,
        )
        pred_by_h[h] = fm.last_pred_snap
        probs_by_h[h] = dict(fm.last_edge_probs)
        stress_by_h[h] = dict(getattr(fm, "last_node_stress_probs", {}) or {})
    if use_v3:
        fm.clear_cache()
    else:
        fm._models.clear()
        fm._link_scorers.clear()
        fm._node_heads.clear()
    gc.collect()

    # 3. live-horizons.json (straight from the pipeline).
    horizons_out = {}
    for h in HORIZONS:
        r = res_by_h[h]
        horizons_out[str(h)] = {
            k: r.get(k)
            for k in (
                "status", "snapshot_date", "forecast_date", "horizon_weeks", "mc_samples",
                "fm_elapsed_ms", "n_alerts", "mean_alert_confidence", "confidence_gate_passed",
                "workbench_case", "candidate_set_m2", "candidate_set_m6", "alpha_queue",
            )
        }
    (ASSETS / "live-horizons.json").write_text(json.dumps(horizons_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 4. Full contagion graph.
    node_ids = sorted({n for e in edges_obs for n in e} | set(curr.nodes.keys()))
    tvl_now = {n: _tvl(curr, n) for n in node_ids}
    pred_tvl = {h: {n: _tvl(pred_by_h[h], n) for n in node_ids} for h in HORIZONS}
    name_fn = collector.mapper_v3.protocol_name if use_v3 else collector.mapper.protocol_name
    cat_fn = collector.mapper_v3.protocol_category if use_v3 else collector.mapper.protocol_category
    live_cat_map = getattr(collector, "live_categories", {}) or {}
    names = {n: name_fn(n) for n in node_ids}
    cats = {n: (live_cat_map.get(n) or cat_fn(n)) for n in node_ids}
    edge_list = sorted(edges_obs.items(), key=lambda kv: -kv[1])
    is_frac_mode = bool(fm_v4.available)
    contagion_graph = {
        "snapshotDate": curr.date,
        "piMin": PI_MIN,
        "weightMode": "fractional_retention" if is_frac_mode else "threshold",
        "horizons": list(HORIZONS),
        "edgeSemantics": "source (creditor) is exposed to target (debtor)",
        "nodes": {
            n: {
                "name": names[n],
                "category": cats[n],
                "tvlUsd": _r(tvl_now[n]),
                "predTvlUsd": {str(h): _r(pred_tvl[h][n]) for h in HORIZONS},
            }
            for n in node_ids
        },
        # [source, target, weightUsd, p_h1, p_h4, p_h8]; p is fractional net-flow retention r_hat in FM v4.
        "edges": [
            [s, t, _r(w)] + [_r(probs_by_h[h].get((s, t)), 4) for h in HORIZONS]
            for (s, t), w in edge_list
        ],
    }
    (ASSETS / "live-contagion-graph.json").write_text(json.dumps(contagion_graph, ensure_ascii=False), encoding="utf-8")

    def contagion_snap(h: int | None) -> dict[str, Any]:
        if h is None:
            nodes = {n: {"tvlUsd": tvl_now[n]} for n in node_ids}
            edges = [{"source": s, "target": t, "weight": w} for (s, t), w in edge_list]
        elif is_frac_mode:
            nodes = {n: {"tvlUsd": pred_tvl[h][n]} for n in node_ids}
            edges = [
                {"source": s, "target": t, "weight": w * probs_by_h[h].get((s, t), 1.0)}
                for (s, t), w in edge_list
                if w * probs_by_h[h].get((s, t), 1.0) >= 1.0
            ]
        else:
            nodes = {n: {"tvlUsd": pred_tvl[h][n]} for n in node_ids}
            edges = [
                {"source": s, "target": t, "weight": w}
                for (s, t), w in edge_list
                if probs_by_h[h].get((s, t), 1.0) >= PI_MIN
            ]
        return {"nodes": nodes, "edges": edges}

    snap_obs = contagion_snap(None)

    # 5. Per-node metrics on the full graph.
    incident: dict[str, list[tuple[str, float, tuple[str, str]]]] = defaultdict(list)
    for (s, t), w in edges_obs.items():
        incident[s].append((t, w, (s, t)))
        incident[t].append((s, w, (s, t)))

    queue_by_h = {
        h: {str(q["entityId"]): q for q in res_by_h[h]["workbench_case"]["review"].get("queue", [])}
        for h in HORIZONS
    }
    alpha_by_h = {
        h: {str(q["entityId"]): q for q in res_by_h[h]["workbench_case"]["review"].get("alphaQueue", [])}
        for h in HORIZONS
    }
    ranked = [n for n in node_ids if obs_w.get(n, 0.0) > 0]
    ranked.sort(key=lambda n: -obs_w.get(n, 0.0))

    # Stratified 4-deck display set: top hubs/mid-caps + M7 & Alpha queue targets + top Tier-3 (<$100M TVL) pools
    all_queue_ids = [
        qid
        for h in HORIZONS
        for qid in list(queue_by_h[h].keys()) + list(alpha_by_h[h].keys())
        if qid in obs_w and obs_w[qid] > 0
    ]
    tier3_tail = [
        n
        for n in ranked
        if n.isdigit() and 0.0 < tvl_now.get(n, 0.0) < 1e8
    ][:20]
    stratified_display: list[str] = []
    seen_disp: set[str] = set()
    for nid in ranked[:96] + all_queue_ids + tier3_tail + ranked:
        if nid not in seen_disp:
            seen_disp.add(nid)
            stratified_display.append(nid)
        if len(stratified_display) >= DISPLAY_NODES:
            break

    cached_icons = _cache_display_icons(set(stratified_display), proto_logos, names)

    def _logo_for(nid: str) -> str | None:
        if nid in cached_icons:
            return cached_icons[nid]
        if nid in TOKEN_ICON_URLS:
            return TOKEN_ICON_URLS[nid]
        u = proto_logos.get(nid.strip("<>"))
        return (u + ("&w=64&h=64" if "?" in u else "?w=64&h=64")) if u else None

    net_out: dict[str, Any] = {"snapshotDate": curr.date, "piMin": PI_MIN, "baseHubTvlUsd": BASE_HUB_TVL_USD}
    for h in HORIZONS:
        probs = probs_by_h[h]
        snap_pred = contagion_snap(h)
        nodes_h = []
        for n in ranked:
            peers = incident.get(n, [])
            W = obs_w.get(n, 0.0)
            tot = sum(w for _, w, _ in peers)
            scored = [(w, probs.get(k)) for _, w, k in peers]
            exp_loss = sum(w * (1.0 - p) for w, p in scored if p is not None)
            n_unscored = sum(1 for _, p in scored if p is None)
            tvl = tvl_now[n]
            hhi = sum((w / tot) ** 2 for _, w, _ in peers) if tot > 0 else None
            top = max(peers, key=lambda x: x[1]) if peers else None
            proto = [p for p in peers if p[0].isdigit()]
            top_proto = max(proto, key=lambda x: x[1]) if proto else None
            q = queue_by_h[h].get(n)
            aq = alpha_by_h[h].get(n)
            nodes_h.append({
                "id": n,
                "name": names[n],
                "logo": _logo_for(n),
                "category": cats[n],
                "isTokenIssuer": not n.isdigit(),
                "isBaseHub": (not n.isdigit()) or tvl >= BASE_HUB_TVL_USD,
                "totalTvlUsd": _r(tvl),
                "predTvlUsd": _r(pred_tvl[h][n]),
                "exposureUsd": _r(W),
                "expectedLossUsd": _r(exp_loss),
                "flowDropPct": _r(100.0 * exp_loss / W) if W > 0 else None,
                "poolDropPct": _r(100.0 * exp_loss / max(tvl + W, W, 1.0)),
                "pStress": _r(stress_by_h[h].get(n), 4),
                "unscoredEdges": n_unscored,
                "counterpartyCount": len(peers),
                "protoCounterpartyCount": len(proto),
                "hhi": _r(hhi, 4),
                "topCounterpartyId": top[0] if top else None,
                "topCounterpartyName": names.get(top[0]) if top else None,
                "topCounterpartySharePct": _r(100.0 * top[1] / tot, 1) if top and tot > 0 else None,
                "topProtoCounterpartyName": names.get(top_proto[0]) if top_proto else None,
                "topProtoCounterpartySharePct": _r(100.0 * top_proto[1] / tot, 1) if top_proto and tot > 0 else None,
                "m7InQueue": q is not None,
                "m7Rank": q.get("rank") if q else None,
                "m7RawAction": q.get("rawAction") if q else None,
                "m7TicketAction": q.get("ticketAction") if q else None,
                "m7Demoted": bool(q.get("demoted")) if q else None,
                "m7GateNote": q.get("gateNote") if q else None,
                "alphaInQueue": aq is not None,
                "alphaRank": aq.get("rank") if aq else None,
                "alphaScore": _r(aq.get("score"), 4) if aq else None,
                "alphaTicketAction": aq.get("ticketAction") if aq else None,
            })

        # Spillover (50% shock) for protocols that can actually propagate (TVL > 0).
        for row in nodes_h:
            n = row["id"]
            if row["isTokenIssuer"] or (tvl_now[n] <= 0 and pred_tvl[h][n] <= 0):
                row.update({"spilloverPredUsd": None, "spilloverObsUsd": None, "spilloverPredDistressed": None})
                continue
            sp = simulate_contagion_traced(snap_pred, [n], shock_fraction=SPILLOVER_SHOCK)
            so = simulate_contagion_traced(snap_obs, [n], shock_fraction=SPILLOVER_SHOCK)
            row["spilloverPredUsd"] = _r(sp["contagion_loss"])
            row["spilloverObsUsd"] = _r(so["contagion_loss"])
            row["spilloverPredDistressed"] = max(0, sp["affected_count"] - len(sp["shocked_nodes"]))

        display_ids = list(stratified_display)
        dset = set(display_ids)
        severe_cut = 0.70 if is_frac_mode else PI_MIN
        disp_edges = [
            {
                "source": s,
                "target": t,
                "weightUsd": _r(w),
                "prevWeightUsd": _r(edges_prev.get((s, t))),
                "survivalProb": _r(probs.get((s, t)), 4),
                "failProb": _r(1.0 - probs[(s, t)], 4) if (s, t) in probs else None,
                "keptInPredGraph": probs.get((s, t), 1.0) >= severe_cut,
            }
            for (s, t), w in edge_list
            if s in dset and t in dset
        ][:DISPLAY_EDGES]

        severe_cnt = sum(1 for (s, t), _ in edge_list if probs.get((s, t), 1.0) < severe_cut)
        partial_cnt = sum(1 for (s, t), _ in edge_list if 0.70 <= probs.get((s, t), 1.0) < 0.90)
        r = res_by_h[h]
        net_out[f"h{h}"] = {
            "horizonWeeks": h,
            "forecastDate": r.get("forecast_date"),
            "fullNodes": len(node_ids),
            "fullEdges": len(edge_list),
            "predGraphEdges": len(snap_pred["edges"]),
            "droppedEdges": severe_cnt,
            "severeWithdrawEdges": severe_cnt,
            "partialWithdrawEdges": partial_cnt,
            "nodes": nodes_h,
            "displayNodeIds": display_ids,
            "displayEdges": disp_edges,
        }
    (ASSETS / "live-network-3d.json").write_text(json.dumps(net_out, ensure_ascii=False), encoding="utf-8")

    CF_ASSETS.mkdir(parents=True, exist_ok=True)
    for fn in ("live-history.json", "live-horizons.json", "live-network-3d.json", "live-contagion-graph.json", "contagion.js"):
        shutil.copy2(ASSETS / fn, CF_ASSETS / fn)
    if (ASSETS / "icons").is_dir():
        shutil.copytree(ASSETS / "icons", CF_ASSETS / "icons", dirs_exist_ok=True)
    updated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("Dashboard assets written for snapshot %s at %s", curr.date, updated_at)
    return {
        "snapshotDate": curr.date,
        "updatedAt": updated_at,
        "fullNodes": len(node_ids),
        "fullEdges": len(edge_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(args.cache_dir)


if __name__ == "__main__":
    main()
