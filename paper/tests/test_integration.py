from __future__ import annotations

import pytest

from dexposure_agent.agent_loop import run_epoch
from dexposure_agent.config import AgentConfig
from dexposure_agent.types import AgentOutput


@pytest.mark.asyncio
async def test_full_pipeline_accepts_current_graph_contract(sample_graph):
    async def forecast(graph, horizon, config):
        del horizon, config
        return {
            "date": graph.date,
            "nodes": {
                key: value.model_dump() for key, value in graph.nodes.items()
            },
            "edges": [edge.model_dump() for edge in graph.edges],
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
            "horizon": 4,
            "elapsed_ms": 1.0,
        }

    output = await run_epoch(
        graph=sample_graph,
        baseline_history=[],
        config=AgentConfig(horizons=[4], mc_samples=3),
        forecast_fn=forecast,
    )
    assert isinstance(output, AgentOutput)
    assert output.epoch_date == sample_graph.date
    assert len(output.scenario_summary.ranked_losses) == 5


@pytest.mark.asyncio
async def test_pipeline_rejects_unknown_forecast_shape(sample_graph):
    async def malformed(graph, horizon, config):
        del graph, horizon, config
        return {"unexpected": True}

    with pytest.raises(ValueError, match="missing fields"):
        await run_epoch(
            graph=sample_graph,
            baseline_history=[],
            config=AgentConfig(horizons=[4]),
            forecast_fn=malformed,
        )
