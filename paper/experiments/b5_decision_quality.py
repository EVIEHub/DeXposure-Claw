#!/usr/bin/env python3
"""b5_decision: Decision Quality (Section 3.6 of EXPERIMENT_PLAN)

Evaluates the quality of regulatory/supervisory action recommendations
produced by the agent layer on top of risk forecasts.

Metrics:
- Ticket Precision       -- fraction of opened investigation tickets that were justified
- Mean Scenario Loss     -- pre-action scenario severity, reported descriptively
- Target Stability       -- consistency of top-K flagged protocols across consecutive weeks
- Audit Completeness     -- fraction of truly high-risk protocols covered by audit tickets
- Suppression Rate       -- fraction of low-risk protocols correctly suppressed (not flagged)
- False Intervention Rate -- fraction of interventions triggered on stable protocols
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp_logger import ExpLogger
from dexposure_agent.config import AgentConfig
from dexposure_agent.data_health import compute_data_health
from dexposure_agent.data_loader import SnapshotLoader
from dexposure_agent.decision import generate_tickets
from dexposure_agent.monitor import run_monitor
from dexposure_agent.scenario import run_scenarios
from dexposure_agent.types import Edge, GraphSnapshot, ScenarioSummary


# TKDE plan: replace absolute threshold with percentile-based ground truth.
# Old design (STRESS_THRESHOLD = 0.20 or 0.50 absolute drop) produced a pool
# that varied wildly week-to-week (1.7k - 5.2k protocols), making
# precision/recall unstable and dominated by node churn rather than method
# quality. Percentile selects a fixed-size top-K most-deteriorating pool
# per week, so comparisons across weeks and methods are apples-to-apples.
STRESS_PERCENTILE = 0.05  # top 5% most-deteriorating protocols per week
DEFAULT_HORIZON = 4
# MC noise sigma for persistence-based MC samples (calibrated at runtime)
MC_NOISE_SIGMA_DEFAULT = 0.1

# Legacy threshold kept for llm_eval_b5.py compatibility; b5 now uses
# percentile-based detection above.
STRESS_THRESHOLD = 0.50

# The decision benchmark, LLM replay, and harness ablations all consume the
# same typed evidence contract.  Keep the canonical JSON/hash functions here
# so matched policy comparisons cannot silently drift between script-local
# implementations.
GROUND_TRUTH_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    """Render hash inputs deterministically without accepting non-finite values."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateSet:
    """Structured protocol-level evidence shared by all matched policies.

    Each row is ``(protocol, exposure, scenario_max_loss, scenario_sum_loss)``.
    The tuple is sorted and immutable before hashing, so policy code cannot
    change candidate identity by presentation order or by mutating a dict.
    """

    rows: tuple[tuple[str, float, float, float], ...] = ()
    candidate_set_sha256: str = field(init=False)
    candidate_membership_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        normalized: list[tuple[str, float, float, float]] = []
        for row in self.rows:
            if len(row) != 4:
                raise ValueError("CandidateSet rows require four fields")
            protocol = str(row[0]).strip()
            if not protocol:
                raise ValueError("CandidateSet protocol names must be non-empty")
            values = tuple(float(value) for value in row[1:])
            if not all(math.isfinite(value) for value in values):
                raise ValueError("CandidateSet values must be finite")
            normalized.append((protocol, *values))
        normalized.sort(key=lambda row: row[0])
        if len({row[0] for row in normalized}) != len(normalized):
            raise ValueError("CandidateSet protocol names must be unique")
        canonical_rows = tuple(normalized)
        object.__setattr__(self, "rows", canonical_rows)
        object.__setattr__(
            self,
            "candidate_set_sha256",
            _sha256_json(
                {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "rows": [list(row) for row in canonical_rows],
                }
            ),
        )
        object.__setattr__(
            self,
            "candidate_membership_sha256",
            _sha256_json(
                {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "protocols": [row[0] for row in canonical_rows],
                }
            ),
        )

    @classmethod
    def from_rows(cls, rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]) -> "CandidateSet":
        normalized: list[tuple[str, float, float, float]] = []
        for row in rows:
            normalized.append(
                (
                    str(row.get("protocol", "")),
                    float(row.get("exposure", row.get("weight", 0.0)) or 0.0),
                    float(row.get("scenario_max_loss", row.get("max_loss", 0.0)) or 0.0),
                    float(row.get("scenario_sum_loss", row.get("sum_loss", 0.0)) or 0.0),
                )
            )
        return cls(tuple(normalized))

    def as_dicts(self) -> list[dict[str, float | str]]:
        return [
            {
                "protocol": protocol,
                "exposure": exposure,
                "scenario_max_loss": max_loss,
                "scenario_sum_loss": sum_loss,
            }
            for protocol, exposure, max_loss, sum_loss in self.rows
        ]


def evidence_hash(
    system_prompt: str,
    user_prompt: str,
    candidate_set: CandidateSet | None = None,
) -> str:
    """Hash rendered prompts plus typed candidates when available.

    The optional candidate set preserves compatibility with historical prompt
    artifacts. New artifacts must include it, and then the structured hash is
    part of the evidence identity rather than being reconstructed from prose.
    """
    payload: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "system_prompt": str(system_prompt),
        "user_prompt": str(user_prompt),
    }
    if candidate_set is not None:
        payload["candidate_set_sha256"] = candidate_set.candidate_set_sha256
    return _sha256_json(payload)


@dataclass(frozen=True)
class EvidenceBundle:
    """Immutable evidence object passed to every matched policy arm."""

    date: str
    system_prompt: str
    user_prompt: str
    evidence_sha256: str = ""
    candidate_set: CandidateSet | None = None

    def __post_init__(self) -> None:
        expected = evidence_hash(
            self.system_prompt,
            self.user_prompt,
            self.candidate_set,
        )
        if self.evidence_sha256 and self.evidence_sha256 != expected:
            raise ValueError("EvidenceBundle evidence_sha256 does not match content")
        object.__setattr__(self, "evidence_sha256", expected)

    @property
    def candidate_set_sha256(self) -> str | None:
        return self.candidate_set.candidate_set_sha256 if self.candidate_set else None

    @property
    def candidate_membership_sha256(self) -> str | None:
        return (
            self.candidate_set.candidate_membership_sha256
            if self.candidate_set
            else None
        )


def _coerce_node_total_weights(snapshot: GraphSnapshot | Mapping[str, Any]) -> dict[str, float]:
    """Return total incident weight for every active node in a snapshot.

    ``GraphSnapshot`` is the benchmark representation. The mapping form keeps
    the same implementation usable by the local LLM evaluator, whose HTTP
    snapshots are JSON dictionaries. Isolated nodes remain in the map with
    zero weight, while edge-only IDs are included when they carry weight.
    """
    if isinstance(snapshot, GraphSnapshot):
        node_ids = {str(node_id) for node_id in snapshot.nodes}
        edges = snapshot.edges
        get_source = lambda edge: edge.source
        get_target = lambda edge: edge.target
        get_weight = lambda edge: edge.weight
    elif isinstance(snapshot, Mapping):
        raw_nodes = snapshot.get("nodes", {})
        node_ids = set(raw_nodes) if isinstance(raw_nodes, Mapping) else set()
        edges = snapshot.get("edges", [])
        get_source = lambda edge: edge.get("source")
        get_target = lambda edge: edge.get("target")
        get_weight = lambda edge: edge.get("weight", 0.0)
    else:
        raise TypeError("snapshot must be a GraphSnapshot or mapping")

    totals: dict[str, float] = {str(node_id): 0.0 for node_id in node_ids}
    for edge in edges:
        try:
            raw_source = get_source(edge)
            raw_target = get_target(edge)
            if raw_source is None or raw_target is None:
                continue
            source = str(raw_source).strip()
            target = str(raw_target).strip()
            weight = float(get_weight(edge))
        except (TypeError, ValueError, AttributeError):
            continue
        if (
            not source
            or not target
            or not node_ids
            or source not in node_ids
            or target not in node_ids
            or not math.isfinite(weight)
            or weight <= 0.0
        ):
            continue
        totals[source] = totals.get(source, 0.0) + weight
        totals[target] = totals.get(target, 0.0) + weight
    return totals


def positive_absolute_losses(
    snap_current: GraphSnapshot | Mapping[str, Any],
    snap_future: GraphSnapshot | Mapping[str, Any],
) -> dict[str, float]:
    """Return positive absolute incident-weight loss over active protocols."""
    current = _coerce_node_total_weights(snap_current)
    future = _coerce_node_total_weights(snap_future)
    losses: dict[str, float] = {}
    for protocol in sorted(current):
        current_weight = current[protocol]
        drop = current_weight - future.get(protocol, 0.0)
        if current_weight > 0.0 and drop > 0.0 and math.isfinite(drop):
            losses[protocol] = drop
    return losses


def detect_truly_stressed_protocols(
    snap_current: GraphSnapshot | Mapping[str, Any],
    snap_future: GraphSnapshot | Mapping[str, Any],
    pct: float = STRESS_PERCENTILE,
) -> set[str]:
    """Deterministic top-``pct`` positive absolute-loss active protocols.

    The ranking is by descending absolute loss with protocol-name tie-breaks.
    ``ceil`` keeps the selected pool at or above the requested percentile, and
    a non-empty positive-loss pool always contributes at least one protocol.
    """
    if isinstance(pct, bool) or not isinstance(pct, (int, float)):
        raise TypeError("pct must be a finite number")
    pct = float(pct)
    if not math.isfinite(pct) or not 0.0 < pct <= 1.0:
        raise ValueError("pct must be finite and within (0, 1]")
    current_weights = _coerce_node_total_weights(snap_current)
    active_count = sum(weight > 0.0 for weight in current_weights.values())
    losses = positive_absolute_losses(snap_current, snap_future)
    if not losses:
        return set()
    ordered = sorted(losses.items(), key=lambda item: (-item[1], item[0]))
    count = min(max(1, int(math.ceil(active_count * pct))), len(ordered))
    return {protocol for protocol, _ in ordered[:count]}


def ranked_unique_targets(targets: Any) -> list[dict[str, Any]]:
    """Normalize and rank candidate target rows deterministically."""
    if isinstance(targets, Mapping):
        targets = targets.get("target_protocols", [])
    if not isinstance(targets, (list, tuple)):
        return []

    def key(row: Mapping[str, Any]) -> tuple[float, str]:
        value = row.get("risk_score", 0.0)
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = float("-inf")
        if not math.isfinite(score):
            score = float("-inf")
        protocol = str(row.get("protocol", "")).strip()
        return (-score, protocol)

    ordered = sorted(
        (row for row in targets if isinstance(row, Mapping)),
        key=key,
    )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ordered:
        protocol = str(row.get("protocol", "")).strip()
        if not protocol or protocol in seen:
            continue
        seen.add(protocol)
        normalized = dict(row)
        normalized["protocol"] = protocol
        result.append(normalized)
    return result


def require_exact_k_targets(targets: Any, k: int) -> list[dict[str, Any]]:
    """Select exactly K unique ranked targets or fail closed."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("target budget K must be a positive integer")
    ranked = ranked_unique_targets(targets)
    if len(ranked) < k:
        raise ValueError(
            f"target budget K={k} requires {k} unique ranked targets, found {len(ranked)}"
        )
    return ranked[:k]


@dataclass
class DecisionResult:
    method: str
    prediction_horizon: int = DEFAULT_HORIZON
    label_horizon: int = DEFAULT_HORIZON
    pi_min: float | None = None
    # Primary TKDE-redesigned metrics
    ticket_precision: float = float("nan")        # |tickets ∩ stressed| / |tickets|
    target_recall_at_k: float = float("nan")      # |tickets ∩ stressed| / |stressed|
    ticket_f1_macro: float = float("nan")         # mean per-week target F1
    score_discrimination: float = float("nan")    # mean tgt-rank gap: stressed vs non-stressed protocols
    target_stability: float = float("nan")        # mean Jaccard week-to-week
    stress_pool_size_mean: float = float("nan")   # sanity check: |stressed| per week
    n_tickets_mean: float = float("nan")          # sanity check: |tickets| per week
    # Descriptive pre-action severity. This is not intervention efficacy.
    mean_scenario_loss: float = float("nan")
    audit_completeness: float = float("nan")
    suppression_rate: float = float("nan")
    false_intervention_rate: float | None = None
    tail_false_intervention_rate: float | None = None
    non_loss_false_intervention_rate: float | None = None
    n_false_interventions: int = 0
    n_tail_false_interventions: int = 0
    n_non_loss_false_interventions: int = 0
    n_intervention_targets: int = 0
    n_weeks_evaluated: Optional[int] = None

    def __str__(self) -> str:
        return (
            f"DecisionResult(method={self.method}, "
            f"prediction_h={self.prediction_horizon}, "
            f"label_h={self.label_horizon}, "
            f"prec={self.ticket_precision:.3f}, "
            f"recall@k={self.target_recall_at_k:.3f}, "
            f"score_discrim={self.score_discrimination:.3f}, "
            f"stab={self.target_stability:.3f}, "
            f"pool_size={self.stress_pool_size_mean:.1f}, "
            f"n_tickets={self.n_tickets_mean:.1f})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_mc_samples(
    graph: GraphSnapshot,
    n_samples: int,
    sigma: float = MC_NOISE_SIGMA_DEFAULT,
    rng: np.random.Generator | None = None,
) -> list[GraphSnapshot]:
    """Generate MC samples by adding Gaussian noise to edge weights."""
    if rng is None:
        rng = np.random.default_rng()
    samples: list[GraphSnapshot] = []
    for _ in range(n_samples):
        new_edges: list[Edge] = []
        for edge in graph.edges:
            noise = rng.normal(0.0, sigma * abs(edge.weight)) if edge.weight != 0.0 else 0.0
            new_weight = max(0.0, edge.weight + noise)
            new_edges.append(Edge(source=edge.source, target=edge.target, weight=new_weight))
        samples.append(graph.model_copy(update={"edges": new_edges}))
    return samples


def _compute_node_total_weight(graph: GraphSnapshot) -> dict[str, float]:
    """Compatibility wrapper around the canonical typed implementation."""
    return _coerce_node_total_weights(graph)


def _graph_payload_for_evidence(graph: GraphSnapshot, horizon: int) -> dict[str, Any]:
    """Render a GraphSnapshot in the typed shape used by the M6 prompt builder."""
    return {
        "date": graph.date,
        "horizon": horizon,
        "nodes": {
            node_id: {
                "log_size": node.log_size,
                "num_tokens": node.num_tokens,
                "max_share": node.max_share,
                "entropy": node.entropy,
                "category": node.category,
                "prev_tvl_change": node.prev_tvl_change,
                "missing_features": list(node.missing_features),
            }
            for node_id, node in graph.nodes.items()
        },
        "edges": [
            {"source": edge.source, "target": edge.target, "weight": edge.weight}
            for edge in graph.edges
        ],
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
    }


def _detect_truly_stressed_protocols(
    snap_t: GraphSnapshot,
    snap_future: GraphSnapshot,
    pct: float = STRESS_PERCENTILE,
) -> set[str]:
    """Compatibility wrapper for the shared stress-label implementation."""
    return detect_truly_stressed_protocols(snap_t, snap_future, pct=pct)


def _ticket_target_score_map(decision) -> dict[str, float]:
    """Aggregate a per-protocol score from the ticket decision.

    The Ticket model exposes `.score` (a float computed by the decision
    engine from alert confidence + scenario impact); when the same protocol
    is named by several tickets we take the max. Protocols not in any
    ticket map to 0.

    (Codex review caught the earlier version, which read a non-existent
    `confidence` attribute and silently returned 0 for every protocol,
    collapsing the score_discrimination metric to a constant zero.)
    """
    scores: dict[str, float] = {}
    for ticket in decision.tickets:
        s = float(getattr(ticket, "score", 0.0) or 0.0)
        for target in ticket.targets:
            scores[target] = max(scores.get(target, 0.0), s)
    return scores


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity; two empty weeks do not establish target stability."""
    if not set_a and not set_b:
        return float("nan")
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _build_scenario_summary(
    pred_graph: GraphSnapshot,
    mc_samples: list[GraphSnapshot],
    config: AgentConfig,
    horizon: int,
) -> ScenarioSummary:
    """Run or skip the scenario engine according to the agent config."""
    if config.skip_scenario:
        return ScenarioSummary(ranked_losses=[], worst_scenario="", worst_horizon=0)
    return run_scenarios(pred_graph, mc_samples, config, horizon=horizon)


def _bind_authoritative_pi_min(config_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Bind standalone b5 FM runs to the authoritative environment contract."""
    authoritative = os.environ.get("DEXPOSURE_AUTHORITATIVE", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if not authoritative:
        return config_kwargs
    raw_pi_min = os.environ.get("DEXPOSURE_FM_PI_MIN")
    if raw_pi_min is None or not raw_pi_min.strip():
        raise ValueError(
            "DEXPOSURE_FM_PI_MIN is required when DEXPOSURE_AUTHORITATIVE=1"
        )
    try:
        pi_min = float(raw_pi_min)
    except ValueError as exc:
        raise ValueError(
            "DEXPOSURE_FM_PI_MIN must be a finite float in [0,1]"
        ) from exc
    if not math.isfinite(pi_min) or not 0.0 <= pi_min <= 1.0:
        raise ValueError("DEXPOSURE_FM_PI_MIN must be a finite float in [0,1]")
    if "pi_min" in config_kwargs and float(config_kwargs["pi_min"]) != pi_min:
        raise ValueError(
            "explicit pi_min conflicts with DEXPOSURE_FM_PI_MIN in authoritative mode"
        )
    config_kwargs["pi_min"] = pi_min
    return config_kwargs


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_b5(
    method_id: str,
    data_dir: str = "data/",
    test_split: str = "2025-01~2025-08",
    horizon: int = DEFAULT_HORIZON,
    **kwargs,
) -> list[DecisionResult]:
    """Run a horizon-aligned b5_decision benchmark for a given method.

    Args:
        method_id: Decision method with rule-based tickets, typically m5_fm_rules or m1_persistence_rules.
        data_dir: Path to processed graph snapshots and ground-truth action labels.
        test_split: Date range string 'YYYY-MM~YYYY-MM'.
        horizon: Forecast and future-label horizon in weeks.
        **kwargs: Extra method-specific config.

    Returns:
        List containing one DecisionResult whose prediction and label horizons
        are both ``horizon``.
    """
    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or horizon not in {1, 4, 8, 12}
    ):
        raise ValueError("horizon must be one of [1, 4, 8, 12]")
    results_dir = kwargs.pop("results_dir", "results/")
    config_kwargs = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in AgentConfig.model_fields
    }
    if kwargs:
        raise TypeError(
            "b5_decision received unsupported configuration keys: "
            + ", ".join(sorted(kwargs))
        )
    benchmark_name = (
        "b5_decision"
        if horizon == DEFAULT_HORIZON
        else f"b5_decision_h{horizon}"
    )
    log = ExpLogger(benchmark_name, method=method_id, results_dir=results_dir)

    log.info(
        f"b5_decision | method={method_id} | test_split={test_split} | "
        f"prediction_horizon={horizon} | label_horizon={horizon}"
    )

    config = AgentConfig(**_bind_authoritative_pi_min(config_kwargs))
    rng = np.random.default_rng(seed=42)

    # --- Load test snapshots with baselines ---
    loader = SnapshotLoader(data_dir=data_dir)
    all_dates = loader.dates

    # Collect all (snap, baseline) pairs
    test_pairs: list[tuple[GraphSnapshot, list[dict[str, float]]]] = []
    for snap, baseline in loader.iter_test_with_baselines(test_split, baseline_window=config.rolling_window):
        test_pairs.append((snap, baseline))

    if len(test_pairs) < 2:
        log.warning("b5_decision: fewer than 2 test snapshots, returning NaN result")
        return [
            DecisionResult(
                method=method_id,
                prediction_horizon=horizon,
                label_horizon=horizon,
                pi_min=config.pi_min,
                n_weeks_evaluated=0,
            )
        ]

    log.info(f"b5_decision: loaded {len(test_pairs)} test snapshots with baselines")

    # Calibrate MC sigma from training data
    from experiments.b3_uncertainty_calibration import _calibrate_mc_sigma
    mc_sigma = _calibrate_mc_sigma(loader, test_split, horizon=horizon)
    log.info(f"b5_decision: calibrated MC sigma = {mc_sigma:.4f}")

    # Build date -> snapshot for future lookups
    all_snapshots = loader.load()
    date_to_snap: dict[str, GraphSnapshot] = {s.date: s for s in all_snapshots}

    # --- Per-week accumulators ---
    n_weeks = 0
    ticket_correct: list[bool] = []        # was each ticket target truly stressed?
    week_recall_at_k: list[float] = []     # |tickets ∩ stressed| / |stressed| per week
    week_ticket_f1: list[float] = []       # per-week harmonic mean of precision and recall
    week_score_gap: list[float] = []       # mean(score | stressed) - mean(score | non-stressed) per week
    target_sets: list[set[str]] = []       # target sets per week for stability
    safe_mode_flags: list[bool] = []       # safe_mode flag per week
    false_interventions: list[bool] = []   # tail false interventions
    non_loss_false_interventions: list[bool] = []
    scenario_losses: list[float] = []      # pre-action mean scenario loss
    stress_pool_sizes: list[int] = []      # sanity: |stressed| per week
    n_tickets_per_week: list[int] = []     # sanity: |tickets| per week
    weekly_records: list[dict] = []        # per-week dump for bootstrap / budget analyses
    m5_evidence_records: list[dict[str, Any]] = []

    for snap_t, baseline_history in log.progress(
        test_pairs, desc="Weekly test snapshots", total=len(test_pairs), unit="week"
    ):
        # --- Find future snapshot for ground truth ---
        t_idx = all_dates.index(snap_t.date) if snap_t.date in all_dates else -1
        if t_idx < 0:
            continue
        future_idx = t_idx + horizon
        if future_idx >= len(all_dates):
            log.info(f"b5_decision: no future snapshot for {snap_t.date}, skipping")
            continue
        future_date = all_dates[future_idx]
        if future_date not in date_to_snap:
            try:
                gt_future = loader.load_single(future_date)
            except KeyError:
                log.info(f"b5_decision: cannot load future snapshot {future_date}, skipping")
                continue
        else:
            gt_future = date_to_snap[future_date]

        # --- Step a: Data health ---
        previous_graph = date_to_snap.get(all_dates[t_idx - 1]) if t_idx > 0 else None
        data_health = compute_data_health(
            snap_t,
            config,
            previous_graph=previous_graph,
        )

        # --- Step b: Generate prediction -> metrics, alerts ---
        from experiments.predict_helper import predict_graph
        pred_graph = predict_graph(
            method_id,
            snap_t,
            horizon=horizon,
            config=config,
        )
        if method_id == "m5_fm_rules":
            # M5 and M6 must attest to the same forecast and typed candidate
            # membership. The sidecar is generated from this exact prediction,
            # not reconstructed by the verifier from a second forecast.
            from experiments.llm_eval_b5 import (
                build_candidate_set,
                forecast_sha256,
                run_scenarios as run_prompt_scenarios,
            )

            evidence_payload = _graph_payload_for_evidence(pred_graph, horizon)
            evidence_candidates = build_candidate_set(
                evidence_payload,
                run_prompt_scenarios(evidence_payload),
            )
            m5_evidence_records.append(
                {
                    "date": snap_t.date,
                    "horizon": horizon,
                    "forecast_sha256": forecast_sha256(evidence_payload, horizon),
                    "candidate_set_sha256": evidence_candidates.candidate_set_sha256,
                    "candidate_membership_sha256": (
                        evidence_candidates.candidate_membership_sha256
                    ),
                    "candidate_set": evidence_candidates.as_dicts(),
                }
            )
        mc_samples = _generate_mc_samples(
            pred_graph,
            config.mc_samples,
            sigma=mc_sigma,
            rng=rng,
        )
        monitor_result = run_monitor(
            predicted_graph=pred_graph,
            mc_samples=mc_samples,
            baseline_history=baseline_history,
            horizon=horizon,
            config=config,
            dh_score=data_health.score,
        )
        alerts = monitor_result.alerts

        # --- Step c: Scenario engine ---
        scenario_summary = _build_scenario_summary(pred_graph, mc_samples, config, horizon=horizon)

        # --- Step d: Generate tickets ---
        decision = generate_tickets(alerts, scenario_summary, data_health, config)

        # --- Step e: Collect ticket targets ---
        all_ticket_targets: set[str] = set()
        intervention_targets: set[str] = set()  # targets from Recommend-Reduce / Contingency
        for ticket in decision.tickets:
            all_ticket_targets.update(ticket.targets)
            if ticket.action in ("Recommend-Reduce", "Contingency"):
                intervention_targets.update(ticket.targets)

        target_sets.append(all_ticket_targets)
        safe_mode_flags.append(decision.suppressed)

        # --- Step f: Ground truth check (percentile-based) ---
        truly_stressed = _detect_truly_stressed_protocols(snap_t, gt_future, pct=STRESS_PERCENTILE)
        stress_pool_sizes.append(len(truly_stressed))
        n_tickets_per_week.append(len(decision.tickets))

        # Ticket Precision: fraction of ticket targets that are truly stressed
        for target in all_ticket_targets:
            ticket_correct.append(target in truly_stressed)

        # target_recall@k: of the truly-stressed pool, how many appear in the
        # week's ticket targets. With ticket_budget << pool the absolute number
        # is small, but discriminates between methods that pick stressed targets
        # vs. arbitrary ones.
        if truly_stressed:
            hits = len(truly_stressed & all_ticket_targets)
            recall_k = hits / len(truly_stressed)
            precision_k = hits / len(all_ticket_targets) if all_ticket_targets else 0.0
            week_recall_at_k.append(recall_k)
            denominator = precision_k + recall_k
            week_ticket_f1.append(
                2.0 * precision_k * recall_k / denominator if denominator > 0 else 0.0
            )

        # score_discrimination: does the ticket engine assign higher
        # confidence to truly-stressed targets than non-stressed ones?
        # Per-protocol score = max confidence across tickets that name it,
        # 0 if not flagged. Gap = mean(score | stressed) - mean(score | non).
        score_map = _ticket_target_score_map(decision)
        snap_t_weights = _compute_node_total_weight(snap_t)
        active_nodes = [n for n in snap_t.nodes.keys() if snap_t_weights.get(n, 0.0) > 0]
        if active_nodes and truly_stressed:
            stressed_scores = [score_map.get(n, 0.0) for n in truly_stressed if n in snap_t.nodes]
            non_stressed_scores = [score_map.get(n, 0.0) for n in active_nodes if n not in truly_stressed]
            if stressed_scores and non_stressed_scores:
                week_score_gap.append(
                    float(np.mean(stressed_scores)) - float(np.mean(non_stressed_scores))
                )

        # False Intervention Rate is split into tail misses and no-loss misses.
        positive_loss_protocols = set(positive_absolute_losses(snap_t, gt_future))
        for target in intervention_targets:
            false_interventions.append(target not in truly_stressed)
            non_loss_false_interventions.append(target not in positive_loss_protocols)

        # Pre-action scenario severity. No ticket action is applied to the graph,
        # so this object must not be interpreted as risk reduction.
        if scenario_summary.ranked_losses:
            mean_loss = float(np.mean([sl.expected_loss for sl in scenario_summary.ranked_losses]))
            scenario_losses.append(mean_loss)

        n_weeks += 1
        weekly_records.append({
            "date": snap_t.date,
            "future_date": future_date,
            "prediction_horizon": horizon,
            "label_horizon": horizon,
            "n_tickets": len(decision.tickets),
            "targets_scored": sorted(score_map.items(), key=lambda kv: -kv[1]),
            "truly_stressed": sorted(truly_stressed),
            "precision_hits": [t in truly_stressed for t in sorted(all_ticket_targets)],
            "recall_at_k": (
                len(truly_stressed & all_ticket_targets) / len(truly_stressed)
                if truly_stressed else None
            ),
            "false_interventions": [
                t not in truly_stressed for t in sorted(intervention_targets)
            ],
            "tail_false_interventions": [
                t not in truly_stressed for t in sorted(intervention_targets)
            ],
            "non_loss_false_interventions": [
                t not in positive_loss_protocols
                for t in sorted(intervention_targets)
            ],
            "positive_loss_protocols": sorted(positive_loss_protocols),
            "intervention_targets": sorted(intervention_targets),
            "mean_scenario_loss": (
                mean_loss if scenario_summary.ranked_losses else None
            ),
            "safe_mode": bool(decision.suppressed),
        })
        log.step(
            f"week {n_weeks}",
            date=snap_t.date, tickets=len(decision.tickets),
            targets=len(all_ticket_targets), truly_stressed=len(truly_stressed),
            safe_mode=decision.suppressed,
        )

    if n_weeks == 0:
        log.warning("b5_decision: no weeks evaluated")
        return [
            DecisionResult(
                method=method_id,
                prediction_horizon=horizon,
                label_horizon=horizon,
                pi_min=config.pi_min,
                n_weeks_evaluated=0,
            )
        ]

    # --- Aggregate metrics ---

    # Ticket Precision: |tickets ∩ stressed| / |tickets|, micro-averaged across weeks
    if ticket_correct:
        ticket_precision = float(np.mean(ticket_correct))
    else:
        ticket_precision = float("nan")

    # target_recall@k: mean across weeks (NaN weeks skipped)
    finite_recalls = [r for r in week_recall_at_k if not np.isnan(r)]
    target_recall_at_k = (
        float(np.mean(finite_recalls)) if finite_recalls else float("nan")
    )
    ticket_f1_macro = (
        float(np.mean(week_ticket_f1)) if week_ticket_f1 else float("nan")
    )

    # score_discrimination: mean across weeks of (stressed-score - non-stressed-score)
    score_discrimination = (
        float(np.mean(week_score_gap)) if week_score_gap else float("nan")
    )

    # Descriptive pre-action scenario loss (lower is less severe).
    if scenario_losses:
        mean_scenario_loss = float(np.mean(scenario_losses))
    else:
        mean_scenario_loss = float("nan")

    # Target Stability: average Jaccard between consecutive weeks
    stability_scores: list[float] = []
    for i in range(1, len(target_sets)):
        stability = _jaccard(target_sets[i], target_sets[i - 1])
        if np.isfinite(stability):
            stability_scores.append(stability)
    target_stability = (
        float(np.mean(stability_scores)) if stability_scores else float("nan")
    )

    # Legacy audit completeness (kept == recall@k for back-compat)
    audit_completeness = target_recall_at_k

    # Suppression / FIR (kept for back-compat; both typically 0)
    suppression_rate = float(np.mean(safe_mode_flags)) if safe_mode_flags else 0.0
    n_intervention_targets = len(false_interventions)
    n_false_interventions = int(sum(false_interventions))
    n_non_loss_false_interventions = int(sum(non_loss_false_interventions))
    false_intervention_rate = (
        n_false_interventions / n_intervention_targets
        if n_intervention_targets > 0
        else None
    )
    tail_false_intervention_rate = false_intervention_rate
    non_loss_false_intervention_rate = (
        n_non_loss_false_interventions / n_intervention_targets
        if n_intervention_targets > 0
        else None
    )

    # Sanity-check aggregates
    stress_pool_size_mean = float(np.mean(stress_pool_sizes)) if stress_pool_sizes else 0.0
    n_tickets_mean = float(np.mean(n_tickets_per_week)) if n_tickets_per_week else 0.0

    result = DecisionResult(
        method=method_id,
        prediction_horizon=horizon,
        label_horizon=horizon,
        pi_min=config.pi_min,
        ticket_precision=ticket_precision,
        target_recall_at_k=target_recall_at_k,
        ticket_f1_macro=ticket_f1_macro,
        score_discrimination=score_discrimination,
        target_stability=target_stability,
        stress_pool_size_mean=stress_pool_size_mean,
        n_tickets_mean=n_tickets_mean,
        mean_scenario_loss=mean_scenario_loss,
        audit_completeness=audit_completeness,
        suppression_rate=suppression_rate,
        false_intervention_rate=false_intervention_rate,
        tail_false_intervention_rate=tail_false_intervention_rate,
        non_loss_false_intervention_rate=non_loss_false_intervention_rate,
        n_false_interventions=n_false_interventions,
        n_tail_false_interventions=n_false_interventions,
        n_non_loss_false_interventions=n_non_loss_false_interventions,
        n_intervention_targets=n_intervention_targets,
        n_weeks_evaluated=n_weeks,
    )

    log.summary({
        "prediction_horizon": horizon,
        "label_horizon": horizon,
        "pi_min": config.pi_min,
        "n_weeks_evaluated": n_weeks,
        "ticket_precision": ticket_precision,
        "target_recall_at_k": target_recall_at_k,
        "ticket_f1_macro": ticket_f1_macro,
        "score_discrimination": score_discrimination,
        "target_stability": target_stability,
        "stress_pool_size_mean": stress_pool_size_mean,
        "n_tickets_mean": n_tickets_mean,
        "mean_scenario_loss": mean_scenario_loss,
        "audit_completeness": audit_completeness,
        "suppression_rate": suppression_rate,
        "false_intervention_rate": false_intervention_rate,
        "tail_false_intervention_rate": tail_false_intervention_rate,
        "non_loss_false_intervention_rate": non_loss_false_intervention_rate,
        "n_false_interventions": n_false_interventions,
        "n_tail_false_interventions": n_false_interventions,
        "n_non_loss_false_interventions": n_non_loss_false_interventions,
        "n_intervention_targets": n_intervention_targets,
    })

    log.save_results([{
        "method": method_id,
        "prediction_horizon": horizon,
        "label_horizon": horizon,
        "pi_min": config.pi_min,
        "ticket_precision": ticket_precision,
        "target_recall_at_k": target_recall_at_k,
        "ticket_f1_macro": ticket_f1_macro,
        "score_discrimination": score_discrimination,
        "target_stability": target_stability,
        "stress_pool_size_mean": stress_pool_size_mean,
        "n_tickets_mean": n_tickets_mean,
        "mean_scenario_loss": mean_scenario_loss,
        "audit_completeness": audit_completeness,
        "suppression_rate": suppression_rate,
        "false_intervention_rate": false_intervention_rate,
        "tail_false_intervention_rate": tail_false_intervention_rate,
        "non_loss_false_intervention_rate": non_loss_false_intervention_rate,
        "n_false_interventions": n_false_interventions,
        "n_tail_false_interventions": n_false_interventions,
        "n_non_loss_false_interventions": n_non_loss_false_interventions,
        "n_intervention_targets": n_intervention_targets,
        "n_weeks_evaluated": n_weeks,
    }])

    # Per-week dump alongside the aggregate file, for bootstrap CIs and
    # matched-budget analyses without re-running the pipeline.
    try:
        artifact_suffix = (
            f"__{method_id}.json"
            if horizon == DEFAULT_HORIZON
            else f"_h{horizon}__{method_id}.json"
        )
        weekly_path = Path(results_dir) / f"b5_weekly{artifact_suffix}"
        weekly_path.parent.mkdir(parents=True, exist_ok=True)
        with open(weekly_path, "w") as f:
            json.dump(
                {
                    "method": method_id,
                    "test_split": test_split,
                    "pi_min": config.pi_min,
                    "horizon": horizon,
                    "prediction_horizon": horizon,
                    "label_horizon": horizon,
                    "weeks": weekly_records,
                },
                f,
                indent=1,
            )
        log.info(
            f"b5_decision weekly dump: {weekly_path} ({len(weekly_records)} weeks)"
        )
        if method_id == "m5_fm_rules":
            evidence_path = Path(results_dir) / f"b5_evidence{artifact_suffix}"
            evidence_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "method": method_id,
                        "horizon": horizon,
                        "pi_min": config.pi_min,
                        "weeks": m5_evidence_records,
                    },
                    indent=1,
                )
                + "\n"
            )
            log.info(
                f"b5_decision evidence metadata: {evidence_path} "
                f"({len(m5_evidence_records)} weeks)"
            )
    except OSError as e:
        log.warning(f"b5_decision weekly dump failed: {e}")

    log.info(f"b5_decision complete: {result}")
    return [result]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="b5_decision: Decision Quality benchmark")
    parser.add_argument("--method", required=True,
                        help="Method ID (e.g. m5_fm_rules or m1_persistence_rules)")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--test-split", default="2025-01~2025-08",
                        help="Test split range YYYY-MM~YYYY-MM")
    parser.add_argument(
        "--horizon",
        type=int,
        choices=(1, 4, 8, 12),
        default=DEFAULT_HORIZON,
        help="Aligned forecast and future-label horizon in weeks",
    )
    args = parser.parse_args()

    results = run_b5(
        args.method,
        args.data_dir,
        args.test_split,
        horizon=args.horizon,
    )
    for r in results:
        print(r)
