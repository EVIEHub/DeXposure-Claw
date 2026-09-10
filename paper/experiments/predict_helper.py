"""Shared prediction helper for b1_forecast/b3_calibration/b4_stress/b5_decision/b6_robustness.

Routes method IDs through the canonical registry and fails closed when a
requested predictor is unavailable. This prevents unimplemented baselines from
silently producing persistence results.
"""
from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dexposure_agent.config import AgentConfig
from dexposure_agent.types import GraphSnapshot
from experiments.exceptions import PredictionUnavailable
from experiments.methods import require_implemented

logger = logging.getLogger(__name__)

_FM_PREDICTOR = None
_FM_PREDICTOR_KEY = None
_EVOLVEGCN_PREDICTOR = None


def _authoritative_enabled() -> bool:
    return os.environ.get("DEXPOSURE_AUTHORITATIVE", "0").lower() in {
        "1",
        "true",
        "yes",
    }


def _fm_settings(config: AgentConfig | None = None) -> tuple[
    AgentConfig, bool, str, str, str
]:
    """Resolve FM config and provenance bindings before touching the cache."""
    authoritative = _authoritative_enabled()
    cfg = config or AgentConfig()
    data_path = os.environ.get("DEXPOSURE_DATA_PATH", "")
    metadata_path = os.environ.get("DEXPOSURE_META_PATH", "")
    if authoritative:
        raw_pi_min = os.environ.get("DEXPOSURE_FM_PI_MIN")
        if raw_pi_min is None or not raw_pi_min.strip():
            raise PredictionUnavailable(
                "DEXPOSURE_FM_PI_MIN is required when DEXPOSURE_AUTHORITATIVE=1"
            )
        try:
            env_pi_min = float(raw_pi_min)
        except ValueError as exc:
            raise PredictionUnavailable(
                "DEXPOSURE_FM_PI_MIN must be a finite float in [0,1]"
            ) from exc
        if not math.isfinite(env_pi_min) or not 0.0 <= env_pi_min <= 1.0:
            raise PredictionUnavailable(
                "DEXPOSURE_FM_PI_MIN must be a finite float in [0,1]"
            )
        if config is None:
            cfg = AgentConfig(pi_min=env_pi_min)
        elif float(cfg.pi_min) != env_pi_min:
            raise PredictionUnavailable(
                "explicit AgentConfig.pi_min conflicts with DEXPOSURE_FM_PI_MIN"
            )
        if not data_path or not metadata_path:
            raise PredictionUnavailable(
                "DEXPOSURE_DATA_PATH and DEXPOSURE_META_PATH are required "
                "when DEXPOSURE_AUTHORITATIVE=1"
            )
    checkpoint_dir = os.environ.get("DEXPOSURE_FM_CKPT_DIR", "")
    return cfg, authoritative, checkpoint_dir, data_path, metadata_path


def _fm_cache_key(config: AgentConfig | None = None) -> tuple[float, str, bool, str, str]:
    """Identify every config/input that changes FM prediction semantics."""
    cfg, authoritative, checkpoint_dir, data_path, metadata_path = _fm_settings(config)
    return (
        float(cfg.pi_min),
        checkpoint_dir,
        authoritative,
        data_path,
        metadata_path,
    )


def _get_fm(config: AgentConfig | None = None):
    global _FM_PREDICTOR, _FM_PREDICTOR_KEY
    cfg, authoritative, checkpoint_dir, data_path, metadata_path = _fm_settings(config)
    requested_key = (
        float(cfg.pi_min),
        checkpoint_dir,
        authoritative,
        data_path,
        metadata_path,
    )
    if _FM_PREDICTOR is None or _FM_PREDICTOR_KEY != requested_key:
        try:
            from dexposure_agent.fm_predictor import FMPredictor
            predictor_kwargs = {"pi_min": cfg.pi_min}
            if checkpoint_dir:
                predictor_kwargs["checkpoint_dir"] = checkpoint_dir
            if authoritative:
                predictor_kwargs.update(
                    {
                        "data_path": data_path,
                        "metadata_path": metadata_path,
                        "require_provenance": True,
                    }
                )
            _FM_PREDICTOR = FMPredictor(**predictor_kwargs)
            _FM_PREDICTOR_KEY = requested_key
            if _FM_PREDICTOR.available:
                logger.info("FM predictor ready")
            else:
                logger.warning("FM predictor: no checkpoints found")
        except Exception as e:
            logger.warning("FM predictor init failed: %s", e)
            _FM_PREDICTOR = None
            _FM_PREDICTOR_KEY = None
            if authoritative:
                raise PredictionUnavailable(
                    f"authoritative FM predictor initialization failed: {e}"
                ) from e
    return _FM_PREDICTOR


def _get_evolvegcn():
    global _EVOLVEGCN_PREDICTOR
    if _EVOLVEGCN_PREDICTOR is None:
        try:
            from experiments.competitors.evolvegcn import EvolveGCNPredictor
            _EVOLVEGCN_PREDICTOR = EvolveGCNPredictor()
            if _EVOLVEGCN_PREDICTOR.available:
                logger.info("EvolveGCN predictor ready")
            else:
                logger.warning("EvolveGCN predictor: no checkpoints found")
        except Exception as e:
            logger.warning("EvolveGCN predictor init failed: %s", e)
            _EVOLVEGCN_PREDICTOR = None
    return _EVOLVEGCN_PREDICTOR


def predict_graph(
    method_id: str,
    current_snapshot: GraphSnapshot,
    horizon: int,
    config: AgentConfig | None = None,
) -> GraphSnapshot:
    """Generate predicted graph G_{t+h} for a supported method.

    Returns a NEW GraphSnapshot for FM/GNN methods, or the same object for persistence.
    Raises PredictionUnavailable instead of substituting another method.
    """
    spec = require_implemented(method_id)

    if spec.predictor == "persistence":
        return current_snapshot

    if spec.predictor == "fm":
        fm = _get_fm(config)
        if fm is not None and fm.available:
            return fm.predict(current_snapshot, horizon)
        raise PredictionUnavailable(
            f"{method_id} ({spec.label}) requires FM checkpoints, but FM is unavailable"
        )

    if spec.predictor == "evolvegcn":
        egcn = _get_evolvegcn()
        if egcn is not None and egcn.available:
            return egcn.predict(current_snapshot, horizon)
        raise PredictionUnavailable(
            f"{method_id} ({spec.label}) requires trained EvolveGCN checkpoints"
        )

    if spec.predictor == "current":
        raise PredictionUnavailable(
            f"{method_id} ({spec.label}) is not a graph-forecasting method"
        )

    raise PredictionUnavailable(
        f"{method_id} ({spec.label}) has unsupported predictor={spec.predictor!r}"
    )
