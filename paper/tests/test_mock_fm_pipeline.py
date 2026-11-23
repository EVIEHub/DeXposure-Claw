from __future__ import annotations

import pytest

from dexposure_agent.agent_loop import run_epoch
from dexposure_agent.config import AgentConfig
from dexposure_agent.types import AgentOutput


def _legacy_forecast():
    return {
        "edge_probs": {
            "aave-v3|lido": 0.9,
            "lido|uniswap-v3": 0.8,
            "uniswap-v3|aave-v3": 0.7,
        },
        "edge_weights": {
            "aave-v3|lido": 7.5,
            "lido|uniswap-v3": 6.2,
            "uniswap-v3|aave-v3": 5.8,
        },
        "weight_stds": {
            "aave-v3|lido": 0.5,
            "lido|uniswap-v3": 0.8,
            "uniswap-v3|aave-v3": 1.2,
        },
        "node_ids": ["aave-v3", "lido", "uniswap-v3"],
    }


@pytest.mark.asyncio
async def test_legacy_distribution_contract_still_runs(sample_graph):
    async def forecast(graph, horizon, config):
        del graph, horizon, config
        return _legacy_forecast()

    output = await run_epoch(
        graph=sample_graph,
        baseline_history=[],
        config=AgentConfig(horizons=[1, 4], mc_samples=5),
        forecast_fn=forecast,
    )
    assert isinstance(output, AgentOutput)
    assert output.data_health.safe_mode is False
    assert len(output.scenario_summary.ranked_losses) == 5


@pytest.mark.asyncio
async def test_safe_mode_suppresses_interventions(sample_graph):
    empty = sample_graph.model_copy(update={"nodes": {}, "edges": []})

    async def forecast(graph, horizon, config):
        del graph, horizon, config
        return _legacy_forecast()

    output = await run_epoch(
        graph=empty,
        baseline_history=[],
        config=AgentConfig(horizons=[4], mc_samples=3),
        forecast_fn=forecast,
    )
    assert output.data_health.safe_mode is True
    assert output.suppressed is True
    assert all(
        ticket.action in {"Monitor", "Investigate"}
        for ticket in output.tickets
    )


@pytest.mark.asyncio
async def test_pipeline_serialization_round_trip(sample_graph):
    async def forecast(graph, horizon, config):
        del graph, horizon, config
        return _legacy_forecast()

    output = await run_epoch(
        graph=sample_graph,
        baseline_history=[],
        config=AgentConfig(horizons=[4], mc_samples=3),
        forecast_fn=forecast,
    )
    restored = AgentOutput.model_validate_json(output.model_dump_json())
    assert restored == output
