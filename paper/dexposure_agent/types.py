"""Pydantic models for every data structure in the agent pipeline.

Maps to Algorithm 1 in DeXposure-Agent paper:
  GraphSnapshot -> DataHealthResult -> Alert/MonitorResult -> ScenarioSummary -> Ticket/DecisionResult -> AgentOutput
"""
from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MissingFeature = Literal[
    "log_size", "num_tokens", "max_share", "entropy", "prev_tvl_change"
]
TicketAction = Literal["Monitor", "Investigate", "Recommend-Reduce", "Contingency"]
TicketSeverity = Literal["Low", "Medium", "High", "Critical"]


class FiniteModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)


class NodeFeatures(FiniteModel):
    log_size: float = Field(ge=0.0)
    num_tokens: int = Field(ge=0)
    max_share: float = Field(ge=0.0, le=1.0)
    entropy: float = Field(ge=0.0)
    category: str = Field(min_length=1)
    prev_tvl_change: float = Field(
        0.0,
        description="log1p(TVL_t) - log1p(TVL_t-1); zero for new nodes",
    )
    missing_features: list[MissingFeature] = Field(
        default_factory=list,
        description="Numeric feature names unavailable in the source payload",
    )

    @field_validator("missing_features")
    @classmethod
    def validate_unique_missing_features(
        cls, values: list[MissingFeature]
    ) -> list[MissingFeature]:
        if len(values) != len(set(values)):
            raise ValueError("missing_features cannot contain duplicates")
        return values

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("category cannot be blank")
        return value


class Edge(FiniteModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    weight: float = Field(ge=0.0)

    @field_validator("source", "target")
    @classmethod
    def validate_endpoint_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("edge endpoints cannot be blank")
        return value


class GraphSnapshot(FiniteModel):
    date: str
    nodes: dict[str, NodeFeatures]
    edges: list[Edge]
    source_node_records_total: int = Field(
        0,
        ge=0,
        description="Raw node records observed before identifier validation",
    )
    source_node_records_invalid: int = Field(
        0,
        ge=0,
        description="Raw node records skipped for invalid or duplicate identifiers",
    )
    source_edge_records_total: int = Field(
        0,
        ge=0,
        description="Raw link records observed before endpoint validation",
    )
    source_edge_records_invalid: int = Field(
        0,
        ge=0,
        description="Raw link records with missing or unknown endpoints",
    )

    @model_validator(mode="after")
    def validate_source_edge_counts(self):
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
        blank_node_ids = [node_id for node_id in self.nodes if not node_id.strip()]
        if blank_node_ids:
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


class DataHealthResult(FiniteModel):
    score: float = Field(ge=0.0, le=1.0, description="DH_t in [0,1]")
    safe_mode: bool = Field(description="True if DH_t < tau_data")
    checks: dict[str, float] = Field(default_factory=dict, description="Individual check scores")


class Alert(FiniteModel):
    horizon: int
    metric_id: str
    metric_name: str
    value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    confidence: float = Field(ge=0.0, le=1.0)
    attribution: dict[str, float] = Field(default_factory=dict, description="Top-K contributing nodes")


class AlertEvidence(FiniteModel):
    """Evidence bundle for an alert (Eq. 5 in paper)."""
    alert: Alert
    horizon: int
    attribution: dict[str, float] = Field(default_factory=dict)


class MonitorResult(FiniteModel):
    alerts: list[Alert] = Field(default_factory=list)
    metrics: dict[str, dict[int, float]] = Field(
        default_factory=dict, description="metric_id -> {horizon: value}"
    )


class ScenarioLoss(FiniteModel):
    scenario_id: str
    scenario_name: str
    horizon: int
    expected_loss: float
    cvar_loss: float
    distressed_count: int
    propagation_depth: int
    top_targets: list[str] = Field(default_factory=list)


class ScenarioSummary(FiniteModel):
    ranked_losses: list[ScenarioLoss] = Field(default_factory=list)
    worst_scenario: str = ""
    worst_horizon: int = 0


class Ticket(FiniteModel):
    action: TicketAction
    severity: TicketSeverity
    score: float = Field(ge=0.0, le=1.0)
    targets: list[str] = Field(default_factory=list)
    triggering_alerts: list[str] = Field(default_factory=list)
    scenario_impact: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


class DecisionResult(FiniteModel):
    tickets: list[Ticket] = Field(default_factory=list)
    suppressed: bool = Field(False, description="True if safe-mode suppressed interventions")


class AgentOutput(FiniteModel):
    epoch_date: str
    data_health: DataHealthResult
    alerts: list[Alert] = Field(default_factory=list)
    scenario_summary: ScenarioSummary
    tickets: list[Ticket] = Field(default_factory=list)
    suppressed: bool = Field(False, description="True if safe-mode suppressed interventions")
