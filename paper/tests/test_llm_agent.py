import pytest

from experiments.competitors.llm_agent import (
    LLMAgentConfig,
    _validate_response,
    run_llm_agent,
)
from experiments.competitors.evolvegcn import EvolveGCNPredictor


def test_standalone_llm_agent_fails_closed_without_api_key(monkeypatch):
    monkeypatch.delenv("DEXPOSURE_TEST_LLM_KEY", raising=False)
    config = LLMAgentConfig(api_key_env="DEXPOSURE_TEST_LLM_KEY")
    with pytest.raises(RuntimeError, match="not set"):
        run_llm_agent({"nodes": {}, "edges": []}, config=config)


def test_standalone_llm_agent_rejects_invalid_ticket_contract():
    with pytest.raises(ValueError, match="risk_score"):
        _validate_response(
            {
                "risk_level": "elevated",
                "target_protocols": [
                    {
                        "protocol": "aave-v3",
                        "risk_score": 1.2,
                        "action": "Recommend-Reduce",
                        "reason": "Observed exposure concentration.",
                    }
                ],
                "rationale": "One target requires review.",
            }
        )


def test_standalone_llm_agent_accepts_valid_ticket_contract():
    parsed = {
        "risk_level": "moderate",
        "target_protocols": [],
        "rationale": "No target crosses the evidence threshold.",
    }
    assert _validate_response(parsed) is parsed


def test_evolvegcn_refuses_nearest_horizon_substitution(sample_graph):
    predictor = EvolveGCNPredictor.__new__(EvolveGCNPredictor)
    predictor.models = {4: object()}
    predictor.meta = {}
    with pytest.raises(RuntimeError, match="No EvolveGCN checkpoint for h=8"):
        predictor.predict(sample_graph, horizon=8)
