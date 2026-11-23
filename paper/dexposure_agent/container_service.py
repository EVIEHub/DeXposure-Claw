"""Cloudflare Container compute service for DeXposure-Claw live pipeline.

Runs strictly on-demand inside a Cloudflare Container (`DexposureEngine`),
executes real DeFiLlama collection + PyTorch DeXposure-FM (`h=1,4,8w`, `MC=50`)
inference + horizon-aligned Walk-Forward Backtest settlement, serves generated
artifacts over container loopback to the parent Durable Object (which persists
them to Cloudflare R2), and stops immediately afterward to keep compute cost near zero.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from dexposure_agent import dashboard_assets
from dexposure_agent.contagion import simulate_contagion_traced

LOG_BUFFER: list[str] = []


class _RingBufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        LOG_BUFFER.append(msg)
        if len(LOG_BUFFER) > 80:
            del LOG_BUFFER[: len(LOG_BUFFER) - 80]


def _setup_logging() -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    ring = _RingBufferHandler()
    ring.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(ring)


_setup_logging()
logger = logging.getLogger("dexposure_agent.container_service")

CACHE_DIR = Path(os.environ.get("DEXPOSURE_CACHE_DIR", "/var/tmp/dexposure_live_cache_v3"))
ASSETS_DIR = dashboard_assets.ASSETS

ARTIFACT_FILES = (
    "live-network-3d.json",
    "live-horizons.json",
    "live-history.json",
    "live-contagion-graph.json",
)

_LOCK = threading.Lock()
JOB_STATE: dict[str, Any] = {
    "state": "idle",
    "stage": "ready",
    "startedAt": None,
    "finishedAt": None,
    "elapsedSec": 0.0,
    "snapshotDate": None,
    "updatedAt": None,
    "fullNodes": None,
    "fullEdges": None,
    "iconFiles": [],
    "error": None,
}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_refresh_worker(force_fetch: bool) -> None:
    t0 = time.time()
    try:
        with _LOCK:
            JOB_STATE.update(
                {
                    "state": "running",
                    "stage": "fetching_defillama_and_running_fm",
                    "startedAt": _now_iso(),
                    "finishedAt": None,
                    "elapsedSec": 0.0,
                    "error": None,
                }
            )
        logger.info("Starting live DeFiLlama + DeXposure-FM refresh (force_fetch=%s)...", force_fetch)
        summary = dashboard_assets.build(CACHE_DIR, refresh_live=force_fetch)
        icons_dir = ASSETS_DIR / "icons"
        icon_files = (
            sorted(p.name for p in icons_dir.glob("*.webp") if p.is_file() and p.stat().st_size > 64)
            if icons_dir.is_dir()
            else []
        )
        elapsed = round(time.time() - t0, 2)
        with _LOCK:
            JOB_STATE.update(
                {
                    "state": "done",
                    "stage": "completed",
                    "finishedAt": _now_iso(),
                    "elapsedSec": elapsed,
                    "snapshotDate": summary.get("snapshotDate"),
                    "updatedAt": summary.get("updatedAt"),
                    "fullNodes": summary.get("fullNodes"),
                    "fullEdges": summary.get("fullEdges"),
                    "iconFiles": icon_files,
                    "error": None,
                }
            )
        logger.info(
            "Live refresh completed in %.2fs (snapshot=%s, nodes=%s, edges=%s)",
            elapsed,
            summary.get("snapshotDate"),
            summary.get("fullNodes"),
            summary.get("fullEdges"),
        )
    except Exception as exc:
        elapsed = round(time.time() - t0, 2)
        tb = traceback.format_exc()
        logger.error("Live refresh failed after %.2fs: %s\n%s", elapsed, exc, tb)
        with _LOCK:
            JOB_STATE.update(
                {
                    "state": "error",
                    "stage": "failed",
                    "finishedAt": _now_iso(),
                    "elapsedSec": elapsed,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


app = FastAPI(title="DeXposure-Claw Container Compute Engine", version="1.0.0")


@app.get("/ping")
@app.get("/health")
def health() -> dict[str, Any]:
    with _LOCK:
        st = dict(JOB_STATE)
    return {"status": "ok", "state": st["state"], "snapshotDate": st["snapshotDate"]}


@app.get("/refresh/status")
def refresh_status() -> dict[str, Any]:
    with _LOCK:
        st = dict(JOB_STATE)
    if st["state"] == "running" and st["startedAt"]:
        st["recentLogs"] = LOG_BUFFER[-12:]
    else:
        st["recentLogs"] = LOG_BUFFER[-8:]
    return st


@app.post("/refresh/start")
def refresh_start(force: bool = True) -> JSONResponse:
    with _LOCK:
        if JOB_STATE["state"] == "running":
            return JSONResponse({"accepted": False, "reason": "already_running", **JOB_STATE})
        JOB_STATE["state"] = "running"
        JOB_STATE["stage"] = "initializing"
        JOB_STATE["startedAt"] = _now_iso()
        JOB_STATE["finishedAt"] = None
        JOB_STATE["error"] = None
    t = threading.Thread(target=_run_refresh_worker, args=(force,), daemon=True)
    t.start()
    with _LOCK:
        return JSONResponse({"accepted": True, **JOB_STATE}, status_code=202)


@app.get("/artifacts/{filename}")
def get_artifact(filename: str) -> FileResponse:
    if filename not in ARTIFACT_FILES:
        raise HTTPException(status_code=404, detail=f"Unknown artifact: {filename}")
    target = ASSETS_DIR / filename
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact not yet built: {filename}")
    return FileResponse(target, media_type="application/json")


@app.get("/artifacts/icons/{icon_name}")
def get_icon_artifact(icon_name: str) -> FileResponse:
    if "/" in icon_name or ".." in icon_name or not icon_name.endswith(".webp"):
        raise HTTPException(status_code=400, detail="Invalid icon name")
    target = ASSETS_DIR / "icons" / icon_name
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Icon not found")
    return FileResponse(target, media_type="image/webp")


@app.post("/api/contagion-simulate")
def contagion_simulate(payload: dict[str, Any]) -> dict[str, Any]:
    graph_path = ASSETS_DIR / "live-contagion-graph.json"
    if not graph_path.is_file():
        raise HTTPException(status_code=503, detail="Contagion graph not available")
    g = json.loads(graph_path.read_text(encoding="utf-8"))
    horizon = str(payload.get("horizon", 4))
    h_idx = {"1": 3, "4": 4, "8": 5}.get(horizon, 4)
    shocked = [str(x) for x in (payload.get("shocked_nodes") or [])]
    frac = float(payload.get("shock_fraction", 0.5))
    mode = str(payload.get("shock_mode", "protocol"))
    pi_min = float(g.get("piMin", 0.5))

    obs_nodes = {k: {"tvlUsd": float(v.get("tvlUsd") or 0.0)} for k, v in g["nodes"].items()}
    pred_nodes = {
        k: {"tvlUsd": float((v.get("predTvlUsd") or {}).get(horizon) or 0.0)}
        for k, v in g["nodes"].items()
    }
    obs_edges = [{"source": e[0], "target": e[1], "weight": float(e[2])} for e in g["edges"]]
    if g.get("weightMode") == "fractional_retention":
        pred_edges = [
            {
                "source": e[0],
                "target": e[1],
                "weight": float(e[2]) * (1.0 if e[h_idx] is None else float(e[h_idx])),
            }
            for e in g["edges"]
            if float(e[2]) * (1.0 if e[h_idx] is None else float(e[h_idx])) >= 1.0
        ]
    else:
        pred_edges = [
            {"source": e[0], "target": e[1], "weight": float(e[2])}
            for e in g["edges"]
            if e[h_idx] is None or float(e[h_idx]) >= pi_min
        ]
    return {
        "status": "ok",
        "snapshot_date": g.get("snapshotDate"),
        "horizon": int(horizon),
        "shock_mode": mode,
        "shock_fraction": frac,
        "shocked_nodes": shocked,
        "observed": simulate_contagion_traced(
            {"nodes": obs_nodes, "edges": obs_edges}, shocked, shock_fraction=frac, shock_mode=mode
        ),
        "predicted": simulate_contagion_traced(
            {"nodes": pred_nodes, "edges": pred_edges}, shocked, shock_fraction=frac, shock_mode=mode
        ),
    }
