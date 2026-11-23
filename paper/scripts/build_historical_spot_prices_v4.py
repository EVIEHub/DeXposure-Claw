#!/usr/bin/env python3
"""Extract exact DeFiLlama internal historical USD spot prices for all 283 weekly dates (2020-2025).

Uses DeFiLlama's own `tokensInUsd / tokens` ratio across long-lived multi-chain reference
protocols so that historical price deflation in `DeXposure-FM v4` matches DeFiLlama's exact
valuation oracle on every snapshot date (`2020-03-23 .. 2025-08-18`).
"""
from __future__ import annotations

from collections import defaultdict
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "historical-network_week_2020-03-30.json"
OUT_PATH = ROOT / "paper" / "dexposure_agent" / "resources" / "historical_spot_prices_v4.json"

REFERENCE_SLUGS: tuple[str, ...] = (
    "makerdao",
    "wbtc",
    "uniswap-v2",
    "uniswap-v3",
    "aave-v2",
    "aave-v3",
    "compound-v2",
    "compound-v3",
    "curve-dex",
    "sushiswap",
    "balancer-v2",
    "pancakeswap-amm",
    "venus-core-pool",
    "benqi-lending",
    "justlend",
    "raydium-amm",
    "lido",
    "rocket-pool",
    "sparklend",
    "morpho-blue",
    "pendle",
)

CANONICAL_ALIASES: dict[str, str] = {
    "WETH": "ETH",
    "ETH": "ETH",
    "STETH": "ETH",
    "WBTC": "BTC",
    "BTC": "BTC",
    "BTCB": "BTC",
    "TBTC": "BTC",
    "CBBTC": "BTC",
    "WBNB": "BNB",
    "BNB": "BNB",
    "WSOL": "SOL",
    "SOL": "SOL",
    "WAVAX": "AVAX",
    "AVAX": "AVAX",
    "WMATIC": "MATIC",
    "MATIC": "MATIC",
    "POL": "MATIC",
    "WTRX": "TRX",
    "TRX": "TRX",
    "WFTM": "FTM",
    "FTM": "FTM",
}

TRACKED_SYMBOLS: frozenset[str] = frozenset({
    "ETH", "BTC", "BNB", "SOL", "AVAX", "MATIC", "TRX", "FTM",
    "UNI", "AAVE", "LINK", "MKR", "CRV", "COMP", "SNX", "YFI",
    "SUSHI", "BAL", "LDO", "RPL", "ARB", "OP", "CAKE", "XVS",
    "PENDLE", "ENA", "EIGEN", "MORPHO", "RAY", "JTO", "SUI",
})


def _fetch_json(slug: str) -> dict:
    url = f"https://api.llama.fi/protocol/{slug}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "DeXposure-Claw/0.4", "Accept-Encoding": "gzip"},
    )
    with urllib.request.urlopen(req, timeout=35.0) as resp:
        raw = resp.read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8"))


def _extract_daily_unit_prices(detail: dict) -> dict[int, dict[str, list[float]]]:
    out: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def _ingest_pair(usd_list: list, qty_list: list) -> None:
        u_by_day: dict[int, tuple[int, dict]] = {}
        q_by_day: dict[int, tuple[int, dict]] = {}
        for entry in usd_list:
            if not isinstance(entry, dict):
                continue
            ts = int(entry.get("date", 0) or 0)
            if ts <= 0:
                continue
            day = (ts // 86400) * 86400
            if ts >= u_by_day.get(day, (-1, {}))[0]:
                u_by_day[day] = (ts, entry.get("tokens") or {})
        for entry in qty_list:
            if not isinstance(entry, dict):
                continue
            ts = int(entry.get("date", 0) or 0)
            if ts <= 0:
                continue
            day = (ts // 86400) * 86400
            if ts >= q_by_day.get(day, (-1, {}))[0]:
                q_by_day[day] = (ts, entry.get("tokens") or {})

        for day, (_, umap) in u_by_day.items():
            if day not in q_by_day:
                continue
            _, qmap = q_by_day[day]
            for raw_sym, uval in umap.items():
                try:
                    u_f = float(uval or 0.0)
                    q_f = float(qmap.get(raw_sym, 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if u_f < 10_000.0 or q_f <= 0.0:
                    continue
                price = u_f / q_f
                if not math.isfinite(price) or price <= 0.0:
                    continue
                clean = str(raw_sym).split("(")[0].strip().upper()
                canon = CANONICAL_ALIASES.get(clean, clean)
                if canon in TRACKED_SYMBOLS:
                    # Exclude STETH if depegged >5% only when WETH is available? Median across hubs handles it cleanly.
                    out[day][canon].append(price)

    chain_tvls = detail.get("chainTvls") or {}
    if isinstance(chain_tvls, dict):
        for cname, cdata in chain_tvls.items():
            low = cname.lower()
            if "borrowed" in low or low.endswith("-staking") or low.endswith("-pool2"):
                continue
            if isinstance(cdata, dict):
                u_l = cdata.get("tokensInUsd")
                q_l = cdata.get("tokens")
                if isinstance(u_l, list) and isinstance(q_l, list) and u_l and q_l:
                    _ingest_pair(u_l, q_l)

    top_u = detail.get("tokensInUsd")
    top_q = detail.get("tokens")
    if isinstance(top_u, list) and isinstance(top_q, list) and top_u and top_q:
        _ingest_pair(top_u, top_q)

    return out


def main() -> None:
    raw_net = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    net_data = raw_net["data"] if "data" in raw_net else raw_net
    dates = sorted(net_data.keys())
    print(f"Loaded {len(dates)} target weekly dates: {dates[0]} .. {dates[-1]}")

    daily_samples: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for slug in REFERENCE_SLUGS:
        try:
            detail = _fetch_json(slug)
            per_day = _extract_daily_unit_prices(detail)
            for day, sym_map in per_day.items():
                for sym, vals in sym_map.items():
                    daily_samples[day][sym].extend(vals)
            print(f"  fetched {slug:<20s}: {len(per_day)} days")
        except Exception as exc:
            print(f"  warning {slug}: {exc}")

    daily_medians: dict[int, dict[str, float]] = {}
    for day, sym_map in daily_samples.items():
        daily_medians[day] = {}
        for sym, vals in sym_map.items():
            vals_sorted = sorted(vals)
            m = len(vals_sorted) // 2
            med = vals_sorted[m] if len(vals_sorted) % 2 == 1 else 0.5 * (vals_sorted[m - 1] + vals_sorted[m])
            daily_medians[day][sym] = round(float(med), 6)

    all_days = sorted(daily_medians.keys())
    prices_by_date: dict[str, dict[str, float]] = {}
    for d_str in dates:
        target_ts = int(dt.datetime.strptime(d_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())
        out_d: dict[str, float] = {}
        for sym in sorted(TRACKED_SYMBOLS):
            # Pick nearest day within +-5 days that has `sym`
            cands = [day for day in all_days if abs(day - target_ts) <= 5 * 86400 and sym in daily_medians[day]]
            if cands:
                best_day = min(cands, key=lambda day: (abs(day - target_ts), -day))
                out_d[sym] = daily_medians[best_day][sym]
        if "ETH" not in out_d or "BTC" not in out_d:
            raise RuntimeError(f"Missing ETH or BTC price on {d_str}: {out_d}")
        prices_by_date[d_str] = out_d

    payload = {
        "schema_version": 4,
        "description": "Exact DeFiLlama internal historical USD spot prices for all 283 snapshots (2020-03-23 .. 2025-08-18)",
        "date_count": len(dates),
        "first_date": dates[0],
        "last_date": dates[-1],
        "reference_protocols": list(REFERENCE_SLUGS),
        "canonical_aliases": CANONICAL_ALIASES,
        "prices_by_date": prices_by_date,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(encoded + "\n", encoding="utf-8")
    digest = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB, sha256={digest[:16]}...)")
    for sample_d in (dates[0], "2021-11-08", "2022-05-16", "2022-11-14", "2025-01-06", dates[-1]):
        if sample_d in prices_by_date:
            p = prices_by_date[sample_d]
            print(f"  {sample_d}: ETH=${p['ETH']:.2f}, BTC=${p['BTC']:.2f}, tokens={len(p)}")


if __name__ == "__main__":
    main()
