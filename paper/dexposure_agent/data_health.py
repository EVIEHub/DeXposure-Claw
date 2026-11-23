"""DataHealth Gate for the DeXposure-Agent pipeline.

Implements Section 2.4 of the DeXposure-Agent paper. Averages the deterministic
checks available for a DeFi credit-exposure GraphSnapshot and returns a
composite score DH_t in [0,1] plus a SAFE_MODE flag.

Checks:
    timestamp     - date field is parseable and non-empty
    missingness   - fraction of explicitly available numeric features
    topology      - fraction of nodes incident to a valid edge
    discontinuity - aggregate similarity to the prior snapshot, when supplied
    edge_validity  - fraction of raw link records with known endpoints
    node_validity  - fraction of valid raw node records
"""
from __future__ import annotations

import logging
import math
from datetime import datetime

from dexposure_agent.config import AgentConfig
from dexposure_agent.types import DataHealthResult, GraphSnapshot

logger = logging.getLogger(__name__)

# Numeric node feature names checked for missingness
_NUMERIC_FEATURES = (
    "log_size",
    "num_tokens",
    "max_share",
    "entropy",
    "prev_tvl_change",
)


def _check_timestamp(graph: GraphSnapshot) -> float:
    """Return 1.0 only when the snapshot timestamp is ISO-parseable."""
    if not graph.date:
        return 0.0
    try:
        datetime.fromisoformat(graph.date)
        return 1.0
    except ValueError:
        logger.warning("DataHealth: unparseable date '%s'", graph.date)
        return 0.0


def _check_missingness(graph: GraphSnapshot) -> float:
    """Return fraction of explicitly available numeric source features.

    Zero is a valid observation for several features. Only names recorded in
    NodeFeatures.missing_features are counted as missing.
    """
    nodes = graph.nodes
    if not nodes:
        return 0.0

    total = len(nodes) * len(_NUMERIC_FEATURES)
    missing = sum(
        1
        for features in nodes.values()
        for attr in set(features.missing_features)
        if attr in _NUMERIC_FEATURES
    )
    return 1.0 - (missing / total)


def _check_topology(graph: GraphSnapshot) -> float:
    """Return the fraction of nodes incident to at least one valid edge."""
    node_count = len(graph.nodes)
    if node_count == 0:
        return 0.0
    incident: set[str] = set()
    for edge in graph.edges:
        if edge.source in graph.nodes and edge.target in graph.nodes:
            incident.add(edge.source)
            incident.add(edge.target)
    return len(incident) / node_count


def _check_discontinuity(
    graph: GraphSnapshot,
    previous_graph: GraphSnapshot | None,
) -> float | None:
    """Compare node count, edge count, and total exposure with the prior week.

    Each component is min(current, previous) / max(current, previous), so the
    score is one for an unchanged aggregate and approaches zero after a large
    discontinuity. When no prior graph is supplied the check is unavailable and
    is excluded from DH_t instead of being replaced by a perfect score.
    """
    if previous_graph is None:
        return None

    def similarity(current: float, previous: float) -> float:
        if current == 0.0 and previous == 0.0:
            return 1.0
        return min(current, previous) / max(current, previous)

    current_total = sum(edge.weight for edge in graph.edges)
    previous_total = sum(edge.weight for edge in previous_graph.edges)
    return sum(
        (
            similarity(float(len(graph.nodes)), float(len(previous_graph.nodes))),
            similarity(float(len(graph.edges)), float(len(previous_graph.edges))),
            similarity(
                math.log1p(max(current_total, 0.0)),
                math.log1p(max(previous_total, 0.0)),
            ),
        )
    ) / 3.0


def compute_data_health(
    graph: GraphSnapshot,
    config: AgentConfig,
    previous_graph: GraphSnapshot | None = None,
) -> DataHealthResult:
    """Compute the DataHealth gate score DH_t for a single GraphSnapshot.

    Args:
        graph:  The current DeFi credit-exposure graph snapshot.
        config: Agent hyperparameters; uses config.tau_data as the safe-mode
                threshold.

    Returns:
        DataHealthResult with score DH_t in [0,1], safe_mode flag, and a
        per-check breakdown dict.
    """
    checks: dict[str, float] = {
        "timestamp_validity": _check_timestamp(graph),
        "missingness": _check_missingness(graph),
        "topology": _check_topology(graph),
    }
    if graph.source_edge_records_total > 0:
        checks["edge_record_validity"] = 1.0 - (
            graph.source_edge_records_invalid / graph.source_edge_records_total
        )
    if graph.source_node_records_total > 0:
        checks["node_record_validity"] = 1.0 - (
            graph.source_node_records_invalid / graph.source_node_records_total
        )
    discontinuity = _check_discontinuity(graph, previous_graph)
    if discontinuity is not None:
        checks["discontinuity"] = discontinuity

    dh_t = sum(checks.values()) / len(checks)
    safe_mode = dh_t < config.tau_data

    logger.info(
        "DataHealth | date=%s nodes=%d edges=%d DH_t=%.4f safe_mode=%s checks=%s",
        graph.date,
        len(graph.nodes),
        len(graph.edges),
        dh_t,
        safe_mode,
        checks,
    )

    return DataHealthResult(score=dh_t, safe_mode=safe_mode, checks=checks)
