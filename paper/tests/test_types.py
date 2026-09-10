import pytest
from pydantic import ValidationError
from dexposure_agent.types import (
    AgentOutput,
    Alert,
    DataHealthResult,
    Edge,
    GraphSnapshot,
    NodeFeatures,
    ScenarioSummary,
    Ticket,
)


def test_graph_snapshot_construction(sample_graph):
    assert len(sample_graph.nodes) > 0
    assert len(sample_graph.edges) > 0
    assert sample_graph.date is not None


def test_negative_and_non_finite_financial_values_are_rejected():
    with pytest.raises(ValidationError):
        Edge(source="a", target="b", weight=-1.0)
    with pytest.raises(ValidationError):
        NodeFeatures(
            log_size=float("nan"),
            num_tokens=1,
            max_share=0.5,
            entropy=0.1,
            category="Lending",
        )


@pytest.mark.parametrize(
    "missing_features",
    [["unknown_feature"], ["log_size", "log_size"]],
)
def test_missing_feature_contract_rejects_unknown_or_duplicate_names(
    missing_features,
):
    with pytest.raises(ValidationError):
        NodeFeatures(
            log_size=1.0,
            num_tokens=1,
            max_share=0.5,
            entropy=0.1,
            category="Lending",
            missing_features=missing_features,
        )


def test_invalid_source_edge_count_contract_is_rejected():
    with pytest.raises(ValidationError, match="cannot exceed"):
        GraphSnapshot(
            date="2025-01-01",
            nodes={},
            edges=[],
            source_edge_records_total=1,
            source_edge_records_invalid=2,
        )


def test_graph_edges_cannot_reference_unknown_nodes():
    with pytest.raises(ValidationError, match="unknown node ids"):
        GraphSnapshot(
            date="2025-01-01",
            nodes={
                "a": NodeFeatures(
                    log_size=1.0,
                    num_tokens=1,
                    max_share=1.0,
                    entropy=0.0,
                    category="Lending",
                )
            },
            edges=[Edge(source="a", target="b", weight=1.0)],
        )


def test_alert_construction():
    alert = Alert(
        horizon=4, metric_id="N1", metric_name="Systemic Importance Score",
        value=0.85, baseline_mean=0.5, baseline_std=0.1,
        z_score=3.5, confidence=0.8,
        attribution={"aave-v3": 0.35, "lido": 0.25},
    )
    assert alert.z_score == 3.5
    assert alert.confidence == 0.8


def test_ticket_construction():
    ticket = Ticket(
        action="Recommend-Reduce", severity="High",
        score=0.75, targets=["aave-v3"],
        triggering_alerts=["N1_h4", "N2_h4"],
        scenario_impact={"S1": 0.45},
        rationale="SIS spike on aave-v3 at h=4",
    )
    assert ticket.severity == "High"


@pytest.mark.parametrize(
    ("action", "severity", "score"),
    [
        ("Delete", "High", 0.5),
        ("Monitor", "Urgent", 0.5),
        ("Monitor", "Low", 1.1),
    ],
)
def test_invalid_ticket_contract_is_rejected(action, severity, score):
    with pytest.raises(ValidationError):
        Ticket(action=action, severity=severity, score=score)


def test_agent_output_json_roundtrip():
    output = AgentOutput(
        epoch_date="2025-03-01",
        data_health=DataHealthResult(score=0.85, safe_mode=False, checks={}),
        alerts=[],
        scenario_summary=ScenarioSummary(ranked_losses=[], worst_scenario="S1", worst_horizon=4),
        tickets=[],
    )
    j = output.model_dump_json()
    restored = AgentOutput.model_validate_json(j)
    assert restored.epoch_date == "2025-03-01"
