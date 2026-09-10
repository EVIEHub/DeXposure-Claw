"""FastAPI server wrapping DeXposure-FM inference on GPU.

This server exposes the FM model as an HTTP API so that the LLM agent
can run on a separate (non-GPU) machine and call FM predictions remotely.

Endpoints:
    GET  /health           - Server status and model availability
    POST /forecast         - Load snapshot by date, run FM predict, return predicted graph
    POST /predict-graph    - Accept a full GraphSnapshot JSON, run FM predict, return predicted graph
    POST /batch-forecast   - Forecast across multiple horizons

Run on GPU server:
    cd /root/graph-dexposure/paper
    DGLBACKEND=pytorch DGL_DISABLE_GRAPHBOLT=1 \
        uvicorn dexposure_agent.serve:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_DISABLE_GRAPHBOLT", "1")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

Horizon = Literal[1, 4, 8, 12]


class StrictAPIModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


class ForecastRequest(StrictAPIModel):
    date: str = Field(min_length=1)
    horizon: Horizon = 4

class BatchForecastRequest(StrictAPIModel):
    date: str = Field(min_length=1)
    horizons: list[Horizon] = Field(
        default_factory=lambda: [1, 4, 8], min_length=1
    )

    @field_validator("horizons")
    @classmethod
    def validate_unique_horizons(cls, values: list[Horizon]) -> list[Horizon]:
        if len(values) != len(set(values)):
            raise ValueError("horizons cannot contain duplicates")
        return values

class GraphNodeFeatures(StrictAPIModel):
    log_size: float = Field(ge=0.0)
    num_tokens: int = Field(ge=0)
    max_share: float = Field(ge=0.0, le=1.0)
    entropy: float = Field(ge=0.0)
    category: str = Field(min_length=1)
    prev_tvl_change: float = 0.0
    missing_features: list[
        Literal[
            "log_size",
            "num_tokens",
            "max_share",
            "entropy",
            "prev_tvl_change",
        ]
    ] = Field(default_factory=list)

    @field_validator("missing_features")
    @classmethod
    def validate_unique_missing_features(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("missing_features cannot contain duplicates")
        return values

class GraphEdge(StrictAPIModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    weight: float = Field(ge=0.0)

class GraphSnapshotPayload(StrictAPIModel):
    date: str = Field(min_length=1)
    nodes: dict[str, GraphNodeFeatures]
    edges: list[GraphEdge]
    source_node_records_total: int = Field(0, ge=0)
    source_node_records_invalid: int = Field(0, ge=0)
    source_edge_records_total: int = Field(0, ge=0)
    source_edge_records_invalid: int = Field(0, ge=0)

    @model_validator(mode="after")
    def validate_graph_contract(self):
        if self.source_node_records_invalid > self.source_node_records_total:
            raise ValueError(
                "source_node_records_invalid cannot exceed "
                "source_node_records_total"
            )
        if self.source_edge_records_invalid > self.source_edge_records_total:
            raise ValueError(
                "source_edge_records_invalid cannot exceed "
                "source_edge_records_total"
            )
        if any(not node_id.strip() for node_id in self.nodes):
            raise ValueError("graph node ids cannot be blank")
        unknown_endpoints = sorted(
            {
                endpoint
                for edge in self.edges
                for endpoint in (edge.source, edge.target)
                if endpoint not in self.nodes
            }
        )
        if unknown_endpoints:
            raise ValueError(
                "graph edges reference unknown node ids: "
                f"{unknown_endpoints[:10]}"
            )
        return self

class PredictGraphRequest(StrictAPIModel):
    graph: GraphSnapshotPayload
    horizon: Horizon = 4

class PredictGraphResponse(StrictAPIModel):
    date: str
    nodes: dict[str, GraphNodeFeatures]
    edges: list[GraphEdge]
    n_nodes: int
    n_edges: int
    horizon: Horizon
    elapsed_ms: float
    pi_min: float | None = None
    checkpoint_identity: dict[str, Any] = Field(default_factory=dict)
    source_node_records_total: int = Field(0, ge=0)
    source_node_records_invalid: int = Field(0, ge=0)
    source_edge_records_total: int = Field(0, ge=0)
    source_edge_records_invalid: int = Field(0, ge=0)


# ---------------------------------------------------------------------------
# Global state (lazy-loaded)
# ---------------------------------------------------------------------------

_fm_predictor = None
_snapshot_loader = None

def _get_fm():
    global _fm_predictor
    if _fm_predictor is None:
        from dexposure_agent.fm_predictor import FMPredictor
        raw_pi_min = os.environ.get("DEXPOSURE_FM_PI_MIN")
        authoritative = os.environ.get("DEXPOSURE_AUTHORITATIVE", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        if authoritative and raw_pi_min is None:
            raise RuntimeError(
                "DEXPOSURE_FM_PI_MIN is required when DEXPOSURE_AUTHORITATIVE=1"
            )
        data_path = os.environ.get("DEXPOSURE_DATA_PATH")
        metadata_path = os.environ.get("DEXPOSURE_META_PATH")
        if authoritative and (not data_path or not metadata_path):
            raise RuntimeError(
                "DEXPOSURE_DATA_PATH and DEXPOSURE_META_PATH are required "
                "when DEXPOSURE_AUTHORITATIVE=1"
            )
        if raw_pi_min is None:
            pi_min = 0.5
            logger.warning(
                "DEXPOSURE_FM_PI_MIN is unset; using local diagnostic default pi_min=0.5"
            )
        else:
            try:
                pi_min = float(raw_pi_min)
            except ValueError as exc:
                raise RuntimeError("DEXPOSURE_FM_PI_MIN must be a finite float in [0,1]") from exc
            if not 0.0 <= pi_min <= 1.0:
                raise RuntimeError("DEXPOSURE_FM_PI_MIN must be a finite float in [0,1]")
        _fm_predictor = FMPredictor(
            pi_min=pi_min,
            data_path=data_path,
            metadata_path=metadata_path,
            require_provenance=authoritative,
        )
        logger.info("FMPredictor initialized: available=%s", _fm_predictor.available)
    return _fm_predictor

def _get_loader():
    global _snapshot_loader
    if _snapshot_loader is None:
        from dexposure_agent.data_loader import SnapshotLoader
        data_dir = os.environ.get(
            "DEXPOSURE_DATA_DIR",
            str(Path(_PROJECT_ROOT) / "data"),
        )
        meta_path = os.environ.get("DEXPOSURE_META_PATH", "data/meta_df.csv")
        _snapshot_loader = SnapshotLoader(data_dir=data_dir, meta_path=meta_path)
        logger.info("SnapshotLoader initialized: %d dates", len(_snapshot_loader.dates))
    return _snapshot_loader

def _to_internal_graph(payload: GraphSnapshotPayload):
    """Convert API payload to internal GraphSnapshot."""
    from dexposure_agent.types import GraphSnapshot, NodeFeatures, Edge
    nodes = {
        nid: NodeFeatures(
            log_size=nf.log_size, num_tokens=nf.num_tokens,
            max_share=nf.max_share, entropy=nf.entropy, category=nf.category,
            prev_tvl_change=nf.prev_tvl_change,
            missing_features=nf.missing_features,
        )
        for nid, nf in payload.nodes.items()
    }
    edges = [
        Edge(source=e.source, target=e.target, weight=e.weight)
        for e in payload.edges
    ]
    return GraphSnapshot(
        date=payload.date,
        nodes=nodes,
        edges=edges,
        source_node_records_total=payload.source_node_records_total,
        source_node_records_invalid=payload.source_node_records_invalid,
        source_edge_records_total=payload.source_edge_records_total,
        source_edge_records_invalid=payload.source_edge_records_invalid,
    )

def _to_response(
    graph, horizon: int, elapsed_ms: float, fm: object | None = None
) -> PredictGraphResponse:
    """Convert internal GraphSnapshot to API response."""
    nodes = {
        nid: GraphNodeFeatures(
            log_size=nf.log_size, num_tokens=nf.num_tokens,
            max_share=nf.max_share, entropy=nf.entropy, category=nf.category,
            prev_tvl_change=nf.prev_tvl_change,
            missing_features=nf.missing_features,
        )
        for nid, nf in graph.nodes.items()
    }
    edges = [
        GraphEdge(source=e.source, target=e.target, weight=e.weight)
        for e in graph.edges
    ]
    return PredictGraphResponse(
        date=graph.date, nodes=nodes, edges=edges,
        n_nodes=len(nodes), n_edges=len(edges),
        horizon=horizon, elapsed_ms=round(elapsed_ms, 1),
        pi_min=getattr(fm, "pi_min", None),
        checkpoint_identity=getattr(fm, "checkpoint_identity", {}),
        source_node_records_total=graph.source_node_records_total,
        source_node_records_invalid=graph.source_node_records_invalid,
        source_edge_records_total=graph.source_edge_records_total,
        source_edge_records_invalid=graph.source_edge_records_invalid,
    )


def _require_available_horizons(fm: object, horizons: list[int]) -> None:
    """Reject valid API horizons that have no exact loaded checkpoint."""
    available = sorted(set(getattr(fm, "available_horizons", [])))
    unavailable = sorted(set(horizons) - set(available))
    if unavailable:
        raise HTTPException(
            422,
            "Forecast horizon checkpoint unavailable: "
            f"requested={unavailable}, available_horizons={available}",
        )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    fm_provider: Callable[[], object] | None = None,
    loader_provider: Callable[[], object] | None = None,
) -> FastAPI:
    """Create the API with injectable providers for deterministic tests."""
    fm_provider = fm_provider or _get_fm
    loader_provider = loader_provider or _get_loader
    app = FastAPI(title="DeXposure-FM API", version="0.2.0")

    @app.get("/health")
    async def health():
        fm = fm_provider()
        return {
            "status": "ok",
            "fm_available": fm.available,
            "available_horizons": getattr(fm, "available_horizons", []),
            "pi_min": getattr(fm, "pi_min", None),
            "checkpoint_identity": getattr(fm, "checkpoint_identity", {}),
            "timestamp": time.time(),
        }

    @app.post("/forecast", response_model=PredictGraphResponse)
    async def forecast(req: ForecastRequest):
        """Load snapshot by date from local data, run FM prediction."""
        fm = fm_provider()
        if not fm.available:
            raise HTTPException(503, "FM model not available")
        _require_available_horizons(fm, [req.horizon])

        loader = loader_provider()
        try:
            snap = loader.load_single(req.date)
        except (KeyError, IndexError) as e:
            raise HTTPException(404, f"Snapshot not found for date {req.date}: {e}")

        t0 = time.time()
        try:
            predicted = fm.predict(snap, req.horizon)
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        elapsed_ms = (time.time() - t0) * 1000

        logger.info(
            "forecast | date=%s h=%d | %d→%d nodes, %d→%d edges | %.0fms",
            req.date, req.horizon,
            len(snap.nodes), len(predicted.nodes),
            len(snap.edges), len(predicted.edges),
            elapsed_ms,
        )

        return _to_response(predicted, req.horizon, elapsed_ms, fm)

    @app.post("/predict-graph", response_model=PredictGraphResponse)
    async def predict_graph(req: PredictGraphRequest):
        """Accept a full GraphSnapshot, run FM prediction, return predicted graph."""
        fm = fm_provider()
        if not fm.available:
            raise HTTPException(503, "FM model not available")
        _require_available_horizons(fm, [req.horizon])

        snap = _to_internal_graph(req.graph)

        t0 = time.time()
        try:
            predicted = fm.predict(snap, req.horizon)
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        elapsed_ms = (time.time() - t0) * 1000

        logger.info(
            "predict-graph | date=%s h=%d | %d→%d nodes, %d→%d edges | %.0fms",
            req.graph.date, req.horizon,
            len(snap.nodes), len(predicted.nodes),
            len(snap.edges), len(predicted.edges),
            elapsed_ms,
        )

        return _to_response(predicted, req.horizon, elapsed_ms, fm)

    @app.post("/batch-forecast")
    async def batch_forecast(req: BatchForecastRequest):
        """Forecast across multiple horizons for a single date."""
        fm = fm_provider()
        if not fm.available:
            raise HTTPException(503, "FM model not available")
        _require_available_horizons(fm, list(req.horizons))

        loader = loader_provider()
        try:
            snap = loader.load_single(req.date)
        except (KeyError, IndexError) as e:
            raise HTTPException(404, f"Snapshot not found: {e}")

        results = []
        for h in req.horizons:
            t0 = time.time()
            try:
                predicted = fm.predict(snap, h)
            except (FileNotFoundError, RuntimeError) as exc:
                raise HTTPException(503, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            elapsed_ms = (time.time() - t0) * 1000
            results.append(_to_response(predicted, h, elapsed_ms, fm))

        return results

    @app.get("/dates")
    async def list_dates():
        """List available snapshot dates."""
        loader = loader_provider()
        return {"dates": loader.dates, "count": len(loader.dates)}

    @app.get("/snapshot")
    async def get_snapshot(date: str):
        """Return raw snapshot (no FM prediction) for a given date.

        Used by the LLM agent for ground truth comparison and
        m2_snapshot_llm (pure LLM) prompts that need the current network state.
        """
        loader = loader_provider()
        try:
            snap = loader.load_single(date)
        except (KeyError, IndexError) as e:
            raise HTTPException(404, f"Snapshot not found for date {date}: {e}")

        nodes = {
            nid: GraphNodeFeatures(
                log_size=nf.log_size, num_tokens=nf.num_tokens,
                max_share=nf.max_share, entropy=nf.entropy, category=nf.category,
                prev_tvl_change=nf.prev_tvl_change,
                missing_features=nf.missing_features,
            )
            for nid, nf in snap.nodes.items()
        }
        edges = [
            GraphEdge(source=e.source, target=e.target, weight=e.weight)
            for e in snap.edges
        ]
        return {
            "date": snap.date,
            "nodes": {nid: nf.model_dump() for nid, nf in nodes.items()},
            "edges": [e.model_dump() for e in edges],
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "source_node_records_total": snap.source_node_records_total,
            "source_node_records_invalid": snap.source_node_records_invalid,
            "source_edge_records_total": snap.source_edge_records_total,
            "source_edge_records_invalid": snap.source_edge_records_invalid,
        }

    return app


# Default app instance
app = create_app()
