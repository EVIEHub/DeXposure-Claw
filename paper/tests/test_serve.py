from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from dexposure_agent.serve import create_app
import dexposure_agent.serve as serve_module
from dexposure_agent.types import GraphSnapshot


class FakeFM:
    available = True
    available_horizons = [1, 4, 8]
    pi_min = 0.7
    checkpoint_identity = {"checkpoint_dir": "test", "files_sha256": {}}

    def __init__(self):
        self.predict_calls = 0

    def predict(self, graph: GraphSnapshot, horizon: int) -> GraphSnapshot:
        self.predict_calls += 1
        return graph.model_copy(update={"date": graph.date})


class FakeLoader:
    def __init__(self, graph: GraphSnapshot):
        self.graph = graph
        self.dates = [graph.date]

    def load_single(self, date: str) -> GraphSnapshot:
        if date != self.graph.date:
            raise KeyError(date)
        return self.graph


class BrokenFM(FakeFM):
    def predict(self, graph: GraphSnapshot, horizon: int) -> GraphSnapshot:
        raise RuntimeError("checkpoint contract failed")


@pytest.fixture
def app(sample_graph):
    fm = FakeFM()
    loader = FakeLoader(sample_graph)
    return create_app(fm_provider=lambda: fm, loader_provider=lambda: loader)


@pytest.mark.asyncio
async def test_health_endpoint(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["fm_available"] is True
    assert response.json()["available_horizons"] == [1, 4, 8]
    assert response.json()["pi_min"] == 0.7
    assert response.json()["checkpoint_identity"]["checkpoint_dir"] == "test"


@pytest.mark.asyncio
async def test_forecast_endpoint_returns_predicted_graph(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/forecast", json={"date": "2025-01-01", "horizon": 4}
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["n_nodes"] == 3
    assert payload["n_edges"] == 3
    assert payload["horizon"] == 4
    assert payload["pi_min"] == 0.7
    assert payload["nodes"]["aave-v3"]["prev_tvl_change"] == 0.0


@pytest.mark.asyncio
async def test_predict_graph_uses_submitted_graph(app, sample_graph):
    request = {"graph": sample_graph.model_dump(), "horizon": 1}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/predict-graph", json=request)
    assert response.status_code == 200
    assert response.json()["date"] == sample_graph.date
    assert response.json()["horizon"] == 1


@pytest.mark.asyncio
async def test_batch_forecast_endpoint(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/batch-forecast",
            json={"date": "2025-01-01", "horizons": [1, 4]},
        )
    assert response.status_code == 200
    assert [row["horizon"] for row in response.json()] == [1, 4]


@pytest.mark.asyncio
async def test_batch_forecast_rejects_duplicate_horizons(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/batch-forecast",
            json={"date": "2025-01-01", "horizons": [4, 4]},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_unknown_date_is_404(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/forecast", json={"date": "2099-01-01", "horizon": 4}
        )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_invalid_horizon_is_rejected_before_inference(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/forecast", json={"date": "2025-01-01", "horizon": 3}
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_missing_exact_checkpoint_rejects_all_inference_routes(sample_graph):
    fm = FakeFM()
    unavailable_app = create_app(
        fm_provider=lambda: fm,
        loader_provider=lambda: FakeLoader(sample_graph),
    )
    graph = sample_graph.model_dump()
    async with AsyncClient(
        transport=ASGITransport(app=unavailable_app), base_url="http://test"
    ) as client:
        responses = [
            await client.post(
                "/forecast", json={"date": sample_graph.date, "horizon": 12}
            ),
            await client.post(
                "/predict-graph", json={"graph": graph, "horizon": 12}
            ),
            await client.post(
                "/batch-forecast",
                json={"date": sample_graph.date, "horizons": [1, 12]},
            ),
        ]

    assert [response.status_code for response in responses] == [422, 422, 422]
    assert all("available_horizons=[1, 4, 8]" in response.json()["detail"] for response in responses)
    assert fm.predict_calls == 0


@pytest.mark.asyncio
async def test_negative_edge_weight_is_rejected_before_inference(app, sample_graph):
    graph = sample_graph.model_dump()
    graph["edges"][0]["weight"] = -1.0
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/predict-graph", json={"graph": graph, "horizon": 4}
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_unknown_edge_endpoint_is_rejected_before_inference(app, sample_graph):
    graph = sample_graph.model_dump()
    graph["edges"][0]["target"] = "unknown-node"
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/predict-graph", json={"graph": graph, "horizon": 4}
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_runtime_model_failure_is_service_unavailable(sample_graph):
    broken_app = create_app(
        fm_provider=lambda: BrokenFM(),
        loader_provider=lambda: FakeLoader(sample_graph),
    )
    async with AsyncClient(
        transport=ASGITransport(app=broken_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/forecast", json={"date": sample_graph.date, "horizon": 4}
        )
    assert response.status_code == 503
    assert "checkpoint contract failed" in response.json()["detail"]


def test_authoritative_fm_requires_explicit_threshold(monkeypatch):
    monkeypatch.setenv("DEXPOSURE_AUTHORITATIVE", "1")
    monkeypatch.delenv("DEXPOSURE_FM_PI_MIN", raising=False)
    serve_module._fm_predictor = None
    with pytest.raises(RuntimeError, match="DEXPOSURE_FM_PI_MIN is required"):
        serve_module._get_fm()
    serve_module._fm_predictor = None


def test_authoritative_fm_requires_data_and_metadata_paths(monkeypatch):
    monkeypatch.setenv("DEXPOSURE_AUTHORITATIVE", "1")
    monkeypatch.setenv("DEXPOSURE_FM_PI_MIN", "0.5")
    monkeypatch.delenv("DEXPOSURE_DATA_PATH", raising=False)
    monkeypatch.delenv("DEXPOSURE_META_PATH", raising=False)
    serve_module._fm_predictor = None
    with pytest.raises(RuntimeError, match="DEXPOSURE_DATA_PATH and DEXPOSURE_META_PATH"):
        serve_module._get_fm()
    serve_module._fm_predictor = None
