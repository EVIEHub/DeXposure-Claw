from __future__ import annotations

import pytest

from dexposure_agent.config import AgentConfig
from experiments.exceptions import PredictionUnavailable
from experiments import predict_helper


def test_fm_cache_identity_includes_pi_min(monkeypatch):
    created: list[float] = []

    class FakePredictor:
        available = False

        def __init__(self, *, pi_min: float):
            created.append(pi_min)

    monkeypatch.setattr(
        "dexposure_agent.fm_predictor.FMPredictor", FakePredictor
    )
    monkeypatch.setattr(predict_helper, "_FM_PREDICTOR", None)
    monkeypatch.setattr(predict_helper, "_FM_PREDICTOR_KEY", None)

    predict_helper._get_fm(AgentConfig(pi_min=0.2))
    predict_helper._get_fm(AgentConfig(pi_min=0.2))
    predict_helper._get_fm(AgentConfig(pi_min=0.8))

    assert created == [0.2, 0.8]


def test_authoritative_fm_binds_pi_min_and_provenance_paths(monkeypatch):
    created: list[dict] = []

    class FakePredictor:
        available = False

        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr("dexposure_agent.fm_predictor.FMPredictor", FakePredictor)
    monkeypatch.setattr(predict_helper, "_FM_PREDICTOR", None)
    monkeypatch.setattr(predict_helper, "_FM_PREDICTOR_KEY", None)
    monkeypatch.setenv("DEXPOSURE_AUTHORITATIVE", "1")
    monkeypatch.setenv("DEXPOSURE_FM_PI_MIN", "0.35")
    monkeypatch.setenv("DEXPOSURE_DATA_PATH", "/data/graph.json")
    monkeypatch.setenv("DEXPOSURE_META_PATH", "/data/meta.csv")

    predict_helper._get_fm()
    predict_helper._get_fm()

    assert created == [
        {
            "pi_min": 0.35,
            "data_path": "/data/graph.json",
            "metadata_path": "/data/meta.csv",
            "require_provenance": True,
        }
    ]


def test_authoritative_fm_rejects_missing_or_conflicting_binding(monkeypatch):
    monkeypatch.setenv("DEXPOSURE_AUTHORITATIVE", "1")
    monkeypatch.delenv("DEXPOSURE_FM_PI_MIN", raising=False)
    monkeypatch.delenv("DEXPOSURE_DATA_PATH", raising=False)
    monkeypatch.delenv("DEXPOSURE_META_PATH", raising=False)

    with pytest.raises(PredictionUnavailable, match="DEXPOSURE_FM_PI_MIN"):
        predict_helper._get_fm()

    monkeypatch.setenv("DEXPOSURE_FM_PI_MIN", "0.35")
    monkeypatch.setenv("DEXPOSURE_DATA_PATH", "/data/graph.json")
    monkeypatch.setenv("DEXPOSURE_META_PATH", "/data/meta.csv")
    with pytest.raises(PredictionUnavailable, match="conflicts"):
        predict_helper._get_fm(AgentConfig(pi_min=0.2))
