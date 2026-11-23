#!/usr/bin/env python3
"""Build seamless 349-week (`2020-03-23 .. 2026-11-23`) enriched NetFlowStockSnapshot bundle.

Stitches:
  - `2020-03-23 .. 2024-10-21` (240 Mondays) from immutable `data/historical-network_week_2020-03-30.json`
  - `2024-10-28 .. 2026-11-23` (109 Mondays) from `output/historical-network_week_2025_2026_full.json`
so that the entire 2025 training tail, 2025-H2 validation split, and 2026 holdout share a 100%
uniform DeFiLlama protocol schema (`node_jaccard = 0.967`, zero RWA migration seam in 2025).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "paper")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dexposure_agent.fm_v3 import TokenMapperV3  # noqa: E402
from dexposure_agent.fm_v4 import (  # noqa: E402
    HistoricalSpotPriceOracle,
    build_netflow_snapshot_from_raw,
    enrich_netflow_snapshots,
)

ORIG_JSON = ROOT / "data" / "historical-network_week_2020-03-30.json"
LIVE_JSON = ROOT / "output" / "historical-network_week_2025_2026_full.json"
PROTO_DIR_JSON = ROOT / "output" / "defillama_2025_2026_weekly_cache" / "protocols_directory.json"
SPOT_2026_JSON = ROOT / "paper" / "dexposure_agent" / "resources" / "historical_spot_prices_v4_2026.json"
SCHEMA_JSON = ROOT / "checkpoints" / "dexposure-fm-v4" / "feature_schema.json"
OUT_BUNDLE = ROOT / "output" / "stitched_349w_v4_bundle.pkl.gz"
OUT_META = ROOT / "output" / "stitched_349w_v4_bundle.meta.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    oracle = HistoricalSpotPriceOracle(SPOT_2026_JSON)
    mapper = TokenMapperV3()
    all_protocols = json.loads(PROTO_DIR_JSON.read_text(encoding="utf-8"))
    mapper.register_live_protocols(all_protocols)

    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    categories = list(schema["categories"])

    raw_orig = json.loads(ORIG_JSON.read_text(encoding="utf-8"))
    orig_data = raw_orig["data"] if "data" in raw_orig else raw_orig
    live_data = json.loads(LIVE_JSON.read_text(encoding="utf-8"))["data"]

    seam_date = "2024-10-28"
    orig_dates = sorted(d for d in orig_data.keys() if d < seam_date)
    live_dates = sorted(d for d in live_data.keys() if d >= seam_date)
    all_dates = orig_dates + live_dates

    print(
        f"Building {len(all_dates)} snapshots: orig={len(orig_dates)} ({orig_dates[0]}..{orig_dates[-1]}), "
        f"live={len(live_dates)} ({live_dates[0]}..{live_dates[-1]})",
        flush=True,
    )

    snaps = []
    for d in orig_dates:
        snaps.append(build_netflow_snapshot_from_raw(d, orig_data[d], mapper))
    for d in live_dates:
        snaps.append(build_netflow_snapshot_from_raw(d, live_data[d], mapper))

    del raw_orig, orig_data, live_data
    enrich_netflow_snapshots(snaps, categories, oracle=oracle)
    print(f"Enriched {len(snaps)} snapshots in {time.time() - t0:.1f}s; writing {OUT_BUNDLE.name}...", flush=True)

    payload = {
        "schema_version": 4,
        "seam_date": seam_date,
        "dates": all_dates,
        "categories": categories,
        "spot_oracle_sha256": oracle.sha256,
        "snapshots": snaps,
    }
    OUT_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT_BUNDLE, "wb", compresslevel=3) as gz:
        pickle.dump(payload, gz, protocol=pickle.HIGHEST_PROTOCOL)

    bundle_sha = sha256_file(OUT_BUNDLE)
    size_mb = OUT_BUNDLE.stat().st_size / (1024 * 1024)
    meta = {
        "bundle_file": OUT_BUNDLE.name,
        "bundle_sha256": bundle_sha,
        "bundle_size_mb": round(size_mb, 2),
        "n_weeks": len(all_dates),
        "first_date": all_dates[0],
        "last_date": all_dates[-1],
        "seam_date": seam_date,
        "spot_oracle_sha256": oracle.sha256,
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Done in {time.time() - t0:.1f}s: {OUT_BUNDLE} ({size_mb:.2f} MB, sha256={bundle_sha[:16]}...)", flush=True)


if __name__ == "__main__":
    main()
