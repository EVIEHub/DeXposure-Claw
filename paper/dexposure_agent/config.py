"""Agent configuration with all tunable hyperparameters.

Maps to EXPERIMENT_PLAN.txt Section 2.5.
"""
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentConfig(BaseModel):
    """All hyperparameters for the DeXposure-Agent pipeline."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # Data-health gate
    tau_data: float = Field(0.7, ge=0.0, le=1.0, description="Data-health safe-mode threshold")

    # Monitor
    z_threshold: float = Field(2.0, gt=0.0, description="Alert z-score threshold")
    rolling_window: int = Field(42, gt=0, description="Rolling baseline window (weeks)")

    # PredGraph builder
    pi_min: float = Field(0.2, ge=0.0, le=1.0, description="Edge existence probability threshold")

    # MC sampling
    mc_samples: int = Field(50, gt=0, description="Monte Carlo samples for uncertainty")

    # Scenario engine
    lambda_tail: float = Field(0.5, ge=0.0, le=1.0, description="CVaR tail weight")

    # Decision
    tau_conf: float = Field(0.6, ge=0.0, le=1.0, description="Confidence gate for interventions")
    top_k: int = Field(5, gt=0, description="Top-K attribution nodes/edges")

    # Horizons
    horizons: list[int] = Field(
        default_factory=lambda: [1, 4, 8],
        description="Forecast horizons in weeks",
    )

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, value: list[int]) -> list[int]:
        supported = {1, 4, 8, 12}
        if not value or len(value) != len(set(value)) or not set(value) <= supported:
            raise ValueError(
                "horizons must be a non-empty unique subset of [1, 4, 8, 12]"
            )
        return value

    # Ablation toggles
    skip_scenario: bool = Field(False, description="Skip scenario engine (ablation A3)")
    unconstrained_actions: bool = Field(False, description="All actions always feasible (ablation A8)")

    # Server
    api_base_url: str = Field("http://localhost:8000", description="GPU server API base URL")
