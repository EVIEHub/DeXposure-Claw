#!/usr/bin/env python3
"""LLM Decision Evaluation Pipeline -- Layer 2 + Layer 3 for b5_decision.

Runs LOCALLY (not on GPU server). Architecture:
  - FM predictions: fetched from GPU server via SSH tunnel (localhost:8000)
  - LLM decisions:  Anthropic Claude API called locally
  - Ground truth:   raw snapshots fetched from GPU /snapshot endpoint

Tests four methods:
  m1_persistence_rules  (loaded from existing b5_decision results)
  m5_fm_rules           (loaded from existing b5_decision results)
  m2_snapshot_llm       (text-only snapshot -> Claude)
  m6_fm_llm             (FM predictions + metrics + scenarios -> Claude)
  m7_fm_llm_gated
          replay-only post-hoc gate over the exact persisted M6 tickets

Layer 2 metrics (LLM reasoning quality):
  - Groundedness: fraction of cited values traceable to input data
  - Consistency:  Jaccard similarity across repeated runs (temperature=0)

Layer 3 metrics (end-to-end decision quality):
  - Ticket Precision:      flagged protocols that were truly stressed
  - Audit Completeness:    truly stressed protocols that were flagged
  - Target Stability:      Jaccard between consecutive weeks
  - Severity Correlation:  Spearman rho between recommended severity vs actual loss
  - Grounding Score:       reasons citing real numbers from data
  - Explanation Quality:   LLM-as-Judge score (1-5)

Prerequisites:
  1. SSH tunnel: ssh -f -N -L 8000:localhost:8000 gpu-server
  2. FM API running (verify: curl localhost:8000/health)
  3. ANTHROPIC_API_KEY set locally

Usage:
    python paper/experiments/llm_eval_b5.py
    python paper/experiments/llm_eval_b5.py --method m6_fm_llm --method m2_snapshot_llm
    python paper/experiments/llm_eval_b5.py --resume   # skip completed weeks
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np
from loguru import logger
from scipy.stats import spearmanr

_PAPER_ROOT = str(Path(__file__).resolve().parent.parent)
if _PAPER_ROOT not in sys.path:
    sys.path.insert(0, _PAPER_ROOT)

from experiments.b5_decision_quality import (
    CandidateSet,
    EvidenceBundle,
    STRESS_PERCENTILE as CANONICAL_STRESS_PERCENTILE,
    _coerce_node_total_weights,
    detect_truly_stressed_protocols,
    positive_absolute_losses,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FM_API_URL = os.environ.get("FM_API_URL", "http://localhost:8000")
DECISION_MODEL = os.environ.get("LLM_EVAL_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "claude-haiku-4-5")
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4096
# Ground truth uses the same percentile-based definition as b5_decision_quality.py
# (top STRESS_PERCENTILE most-deteriorating active protocols per week). Keeping
# the two evaluators on identical ground truth is required for m5_fm_rules vs
# m6_fm_llm comparisons to be meaningful. The legacy absolute STRESS_THRESHOLD
# (0.50) is retained as a numeric fallback only.
STRESS_PERCENTILE = CANONICAL_STRESS_PERCENTILE
STRESS_THRESHOLD = 0.50  # legacy; not used by detect_truly_stressed below.
STRESS_LOOKAHEAD = 4
CONSISTENCY_RUNS = 3
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# LLM API backend: OpenRouter (default) or Anthropic direct
LLM_API_URL = os.environ.get("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_API_KEY = os.environ.get("OPENROUTER_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))

# Cost per million tokens (OpenRouter listed prices, USD).
# Keys MUST be the exact slug OpenRouter exposes. Anthropic slugs use dots
# (claude-opus-4.8), not dashes. For non-Anthropic models always pass the
# full slug (provider/model). The call_llm helper auto-prefixes "anthropic/"
# only when no provider is set.
# Legacy dash-form aliases are kept so existing run logs and CLI flags keep
# working; they all point at the same row as the dotted slug.
MODEL_COSTS = {
    # Anthropic (dotted, real OpenRouter slugs)
    "claude-haiku-4.5": {"input": 1.0, "output": 5.0},
    "claude-sonnet-4.6": {"input": 3.0, "output": 15.0},
    "claude-opus-4.6": {"input": 5.0, "output": 25.0},
    "claude-opus-4.7": {"input": 5.0, "output": 25.0},
    "claude-opus-4.8": {"input": 5.0, "output": 25.0},
    # Legacy dash-form aliases (kept for backward compat with prior run logs)
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    # OpenAI (full OpenRouter slugs)
    "openai/gpt-5.4": {"input": 1.25, "output": 10.0},
    "openai/gpt-5.5": {"input": 2.0, "output": 15.0},
    "openai/gpt-5.5-pro": {"input": 3.0, "output": 20.0},
    # Google
    "google/gemini-2.5-pro": {"input": 1.25, "output": 5.0},
    "google/gemini-3.1-pro-preview": {"input": 1.5, "output": 8.0},
    # xAI
    "x-ai/grok-4.3": {"input": 3.0, "output": 15.0},
    "x-ai/grok-4.20": {"input": 3.0, "output": 15.0},
    # DeepSeek
    "deepseek/deepseek-v3.2": {"input": 0.27, "output": 1.10},
    "deepseek/deepseek-v4-pro": {"input": 0.55, "output": 2.20},
}


def _normalize_model_slug(model: str) -> str:
    """Convert internal dash-form Anthropic names into OpenRouter dot-form.

    Anthropic OpenRouter slugs use dots (claude-opus-4.8). Code, CLI flags,
    and historical logs sometimes use the dash form (claude-opus-4-8); this
    helper rewrites the trailing version segment so both styles reach the
    same model row.
    """
    if "/" in model:
        return model  # already provider-qualified
    import re as _re
    return _re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", model)
SEVERITY_ORDER = {"Monitor": 1, "Investigate": 2, "Recommend-Reduce": 3, "Contingency": 4}
GATED_LLM_METHODS = {"m7_fm_llm_gated"}
LLM_METHODS = {"m2_snapshot_llm", "m6_fm_llm", "m7_fm_llm_gated"}
SAMPLED_LLM_METHODS = {"m2_snapshot_llm", "m6_fm_llm"}
MAX_TARGET_BUDGET = 7
PRIMARY_TARGET_BUDGET = 5

# M7 applies these thresholds after the raw decision is persisted. Keeping
# them outside SYSTEM_PROMPT prevents the raw M6/M7 request from pre-applying
# the gate and preserves an exact raw-ticket control for replay.
ACTION_GATE_POLICY = {
    "Recommend-Reduce": {
        "risk_levels": {"elevated", "critical"},
        "min_score": 0.75,
        "description": (
            "Recommend-Reduce requires elevated or critical overall risk and "
            "protocol risk_score >= 0.75."
        ),
    },
    "Contingency": {
        "risk_levels": {"critical"},
        "min_score": 0.90,
        "description": (
            "Contingency requires critical overall risk and protocol "
            "risk_score >= 0.90."
        ),
    },
}


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json(value) -> str:
    """Serialize evidence deterministically before computing provenance hashes."""
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


# ---------------------------------------------------------------------------
# HTTP helpers (FM API via SSH tunnel)
# ---------------------------------------------------------------------------

def _http_get(path: str, retries: int = 3) -> dict:
    import urllib.request, urllib.error
    url = f"{FM_API_URL}{path}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"GET {path} attempt {attempt+1} failed: {e}, retry in {wait}s")
                time.sleep(wait)
            else:
                raise


def _http_post(path: str, data: dict, retries: int = 3) -> dict:
    import urllib.request, urllib.error
    url = f"{FM_API_URL}{path}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, method="POST")
            req.data = json.dumps(data).encode()
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"POST {path} attempt {attempt+1} failed: {e}, retry in {wait}s")
                time.sleep(wait)
            else:
                raise


def get_test_dates(test_split: str = "2025-01~2025-08") -> list[str]:
    data = _http_get("/dates")
    start, end = test_split.split("~")
    return [d for d in data["dates"] if d >= f"{start}-01" and d <= f"{end}-31"]


def get_snapshot(date: str) -> dict:
    return _http_get(f"/snapshot?date={date}")


def get_forecast(date: str, horizon: int) -> dict:
    return _http_post("/forecast", {"date": date, "horizon": horizon})


# ---------------------------------------------------------------------------
# Local metrics computation (no GPU deps)
# ---------------------------------------------------------------------------

def _gini(vals: list[float]) -> float:
    n = len(vals)
    if n == 0 or sum(vals) == 0:
        return 0.0
    s = sorted(vals)
    total = sum(s)
    cumsum = gs = 0.0
    for v in s:
        cumsum += v
        gs += cumsum
    return 1.0 + (1.0 / n) - (2.0 * gs) / (n * total)


def compute_metrics(data: dict) -> dict:
    nodes = data.get("nodes", {})
    edges = data.get("edges", [])
    n = len(nodes)
    if n == 0:
        return {}

    # Build forward adjacency (for out-weights) + reverse adjacency (for PageRank)
    out_weight: dict[str, float] = {nid: 0.0 for nid in nodes}
    adj_in: dict[str, list[tuple[str, float]]] = {nid: [] for nid in nodes}
    wdeg: dict[str, float] = {nid: 0.0 for nid in nodes}

    edge_agg: dict[tuple[str, str], float] = {}
    for e in edges:
        src, tgt, w = e["source"], e["target"], e["weight"]
        key = (src, tgt)
        edge_agg[key] = edge_agg.get(key, 0.0) + w
        if src in wdeg:
            wdeg[src] += w

    for (src, tgt), w in edge_agg.items():
        if src in out_weight:
            out_weight[src] += w
        if tgt in adj_in:
            adj_in[tgt].append((src, w))

    deg_vals = list(wdeg.values())
    total_wd = sum(deg_vals)

    # PageRank via reverse adjacency: O(V+E) per iteration instead of O(V^2)
    pr = {nid: 1.0 / n for nid in nodes}
    base = 0.15 / n
    for _ in range(20):
        new_pr = {}
        for node in nodes:
            rank = base
            for src, w in adj_in[node]:
                ow = out_weight[src]
                if ow > 0:
                    rank += 0.85 * pr[src] * w / ow
            new_pr[node] = rank
        pr = new_pr
    pr_vals = list(pr.values())

    return {
        "n_nodes": n,
        "n_edges": len(edges),
        "N1_max_pagerank": round(max(pr_vals), 6),
        "N2_hhi": round(sum((d / total_wd) ** 2 for d in deg_vals) if total_wd > 0 else 0, 6),
        "N3_density": round(len(edges) / (n * (n - 1)) if n > 1 else 0, 6),
        "N4_pagerank_gini": round(_gini(pr_vals), 6),
        "N5_degree_gini": round(_gini(deg_vals), 6),
    }


def forecast_sha256(data: dict, horizon: int) -> str:
    """Hash the exact graph forecast used to build M6 evidence."""
    payload = {
        "date": data.get("date"),
        "horizon": int(horizon),
        "nodes": data.get("nodes", {}),
        "edges": data.get("edges", []),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def run_scenarios(data: dict) -> list[dict]:
    SCENARIOS = {
        "S1": {"name": "Top protocol failure", "type": "top_node", "shock_pct": 1.0, "count": 1},
        "S2": {"name": "Bridge cluster failure", "type": "category",
               "categories": ["Bridge", "Cross Chain"], "shock_pct": 1.0},
        "S3": {"name": "Stablecoin de-peg", "type": "category",
               "categories": ["Algo-Stables", "Decentralized Stablecoin", "CDP"], "shock_pct": 0.5},
        "S4": {"name": "Lending sector shock", "type": "category",
               "categories": ["Lending", "Uncollateralized Lending", "RWA Lending", "NFT Lending"],
               "shock_pct": 0.3},
        "S5": {"name": "Correlated stress (top-10)", "type": "top_nodes", "shock_pct": 0.2, "count": 10},
    }

    nodes = data.get("nodes", {})
    edges = data.get("edges", [])
    nw: dict[str, float] = defaultdict(float)
    for e in edges:
        nw[e["source"]] += e["weight"]
        nw[e["target"]] += e["weight"]

    results = []
    for sid, spec in SCENARIOS.items():
        shock_type, shock_pct = spec["type"], spec["shock_pct"]
        if shock_type in ("top_node", "top_nodes"):
            cnt = spec.get("count", 1)
            shocked = {n for n, _ in sorted(nw.items(), key=lambda x: x[1], reverse=True)[:cnt]}
        elif shock_type == "category":
            cats_lower = {c.lower() for c in spec.get("categories", [])}
            shocked = {nid for nid, nf in nodes.items()
                       if nf.get("category", "").lower() in cats_lower}
        else:
            shocked = set()

        sum_orig = sum(nw.values())
        sw: dict[str, float] = defaultdict(float)
        for e in edges:
            w = e["weight"]
            if e["source"] in shocked or e["target"] in shocked:
                w *= (1 - shock_pct)
            if w > 0:
                sw[e["source"]] += w
                sw[e["target"]] += w
        sum_shock = sum(sw.values())
        loss = (sum_orig - sum_shock) / sum_orig if sum_orig > 0 else 0

        distressed = sum(1 for nid, ow in nw.items()
                         if ow > 0 and (ow - sw.get(nid, 0)) / ow > 0.5)
        affected = sum(1 for nid in nw if nw[nid] - sw.get(nid, 0) > 0)
        top_aff = sorted(((nid, nw[nid] - sw.get(nid, 0)) for nid in nw
                          if nw[nid] - sw.get(nid, 0) > 0),
                         key=lambda x: x[1], reverse=True)[:5]
        results.append({
            "scenario_id": sid, "scenario_name": spec["name"],
            "system_loss_pct": round(loss * 100, 2),
            "distressed_count": distressed, "affected_nodes": affected,
            "top_affected": [{"protocol": nid, "loss": round(d, 2),
                              "category": nodes.get(nid, {}).get("category", "?")}
                             for nid, d in top_aff],
        })
    return results


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def detect_truly_stressed(snap_current: dict, snap_future: dict,
                          pct: float = STRESS_PERCENTILE) -> set[str]:
    """Compatibility wrapper around the shared deterministic stress labels."""
    return detect_truly_stressed_protocols(snap_current, snap_future, pct=pct)


def compute_actual_loss(snap_current: dict, snap_future: dict) -> dict[str, float]:
    """Per-protocol weight-drop fraction for severity correlation."""
    wt = _coerce_node_total_weights(snap_current)
    wf = _coerce_node_total_weights(snap_future)
    losses = {}
    for nid, w in wt.items():
        if w > 0:
            losses[nid] = max(0.0, (w - wf.get(nid, 0.0)) / w)
    return losses


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ACTION_GATE_POLICY_TEXT = "\n".join(
    f"{action}: {rule['description']}"
    for action, rule in ACTION_GATE_POLICY.items()
)

SYSTEM_PROMPT = """You are a DeFi systemic risk analyst for a regulatory supervisory body.
Your job: given network data, identify protocols at elevated risk of distress
over the next {horizon} weeks and recommend supervisory actions.

You MUST respond with valid JSON only (no markdown, no explanations outside JSON).
Schema:
{{
  "risk_level": "low" | "moderate" | "elevated" | "critical",
  "target_protocols": [
    {{
      "protocol": "<name>",
      "risk_score": <float 0-1>,
      "action": "Monitor" | "Investigate" | "Recommend-Reduce" | "Contingency",
      "reason": "<specific explanation citing data values from the input>"
    }}
  ],
  "rationale": "<2-3 sentence overall assessment citing specific metrics and numbers>"
}}

Rules:
- Only flag protocols you genuinely believe are at elevated risk.
- Cite specific numbers from the provided data to justify each assessment.
- Use the exact protocol names as they appear in the data.
- Return exactly 7 unique target_protocols, including Monitor rows when fewer
  than 7 protocols warrant escalation.
- Order target_protocols by risk_score descending."""

C0_LLM_USER_TEMPLATE = """Current DeFi network analysis ({date}), forecast horizon = {horizon} weeks.

== FM MODEL PREDICTIONS (predicted graph G_{{t+{horizon}}}) ==
Nodes: {n_nodes} protocols, Edges: {n_edges} weighted exposure links

Top-10 protocols by predicted exposure weight:
{top_protocols}

== PREDICTED NETWORK RISK METRICS ==
{metrics}

== STRESS SCENARIO ANALYSIS (applied to predicted graph) ==
{scenarios}

Based on these FM model predictions, metrics, and stress analysis, identify
which protocols are at elevated risk over the next {horizon} weeks."""

C3_USER_TEMPLATE = """Current DeFi network state ({date}), forecast horizon = {horizon} weeks.

NOTE: You do NOT have access to any predictive model. You must reason from the
current network snapshot only. There are no forward-looking predictions available.

== CURRENT NETWORK SNAPSHOT ==
Protocols: {n_nodes} total
Edges: {n_edges} weighted exposure links
Total network weight: {total_weight:.2f}

Top-10 protocols by current exposure weight:
{top_protocols}

Category breakdown:
{category_summary}

== CURRENT NETWORK METRICS ==
{metrics}

Based on this current network state (without predictive model forecasts),
identify which protocols are at elevated risk over the next {horizon} weeks."""

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of DeFi risk analysis reports.
You will compare two risk assessments for the same week and rate the quality
of the SECOND one (Report B) on a scale of 1-5.

Respond with JSON only:
{{
  "quality_score": <int 1-5>,
  "reasoning": "<1-2 sentence justification>"
}}

Scoring guide:
  5: Excellent -- cites specific metrics, identifies correct risk factors, actionable
  4: Good -- mostly correct, cites some data, reasonable recommendations
  3: Adequate -- general assessment, few specific citations, partially correct
  2: Poor -- vague, misses key risks, few data references
  1: Very poor -- contradicts data, no grounding, hallucinated claims"""

JUDGE_USER_TEMPLATE = """Week: {date}, horizon: {horizon} weeks

== INPUT DATA SUMMARY ==
Network: {n_nodes} nodes, {n_edges} edges
Key metrics: {metrics_summary}

== GROUND TRUTH ==
{n_stressed} protocols are in the top-5% set ranked by absolute weight loss over the horizon (regulator-relevant systemic risk): {stressed_list}

== REPORT TO EVALUATE ==
Risk level: {risk_level}
Targets: {llm_targets}
Rationale: {rationale}

Rate the report's quality (1-5)."""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_top_protocols(
    data: dict,
    candidate_set: CandidateSet | None = None,
) -> str:
    nodes = data.get("nodes", {})
    if candidate_set is None:
        candidate_set = build_candidate_set(data)
    lines = []
    ranked_rows = sorted(candidate_set.rows, key=lambda row: (-row[1], row[0]))
    for i, (nid, w, scenario_max_loss, scenario_sum_loss) in enumerate(
        ranked_rows, 1
    ):
        cat = nodes.get(nid, {}).get("category", "?")
        lines.append(
            f"  {i}. {nid} ({cat}) -- weight: {w:.2f}; "
            f"scenario_max_loss: {scenario_max_loss:.2f}; "
            f"scenario_sum_loss: {scenario_sum_loss:.2f}"
        )
    return "\n".join(lines)


def _format_metrics(metrics: dict) -> str:
    rendered = [f"  {k}: {v}" for k, v in metrics.items() if k.startswith("N")]
    if not rendered:
        raise ValueError("forecast metrics are missing the required N1-N5 fields")
    return "\n".join(rendered)


def _format_scenarios(
    scenarios: list[dict],
    allowed_protocols: set[str] | None = None,
) -> str:
    lines = []
    for s in scenarios:
        lines.append(f"  {s['scenario_id']} ({s['scenario_name']}): "
                     f"loss={s['system_loss_pct']:.2f}%, "
                     f"distressed={s['distressed_count']}, "
                     f"affected={s['affected_nodes']}")
        for t in s.get("top_affected", [])[:3]:
            if (
                allowed_protocols is not None
                and t.get("protocol") not in allowed_protocols
            ):
                continue
            lines.append(f"    - {t['protocol']} ({t['category']}): loss={t['loss']:.2f}")
    return "\n".join(lines)


def _format_categories(data: dict) -> str:
    cats: dict[str, int] = defaultdict(int)
    for nid, nf in data.get("nodes", {}).items():
        cats[nf.get("category", "Unknown")] += 1
    top = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:10]
    return "\n".join(f"  {cat}: {cnt} protocols" for cat, cnt in top)


def build_candidate_set(
    data: dict,
    scenarios: list[dict] | None = None,
) -> CandidateSet:
    """Build typed protocol evidence before rendering any prompt prose."""
    raw_nodes = data.get("nodes", {})
    node_ids = {str(node_id).strip() for node_id in raw_nodes} if isinstance(raw_nodes, dict) else set()
    exposure: dict[str, float] = defaultdict(float)
    for edge in data.get("edges", []):
        try:
            raw_source = edge["source"]
            raw_target = edge["target"]
            if raw_source is None or raw_target is None:
                continue
            source = str(raw_source).strip()
            target = str(raw_target).strip()
            weight = float(edge["weight"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not source
            or not target
            or not node_ids
            or source not in node_ids
            or target not in node_ids
            or not math.isfinite(weight)
            or weight <= 0
        ):
            continue
        exposure[source] += weight
        exposure[target] += weight

    selected_protocols = {
        protocol
        for protocol, _ in sorted(
            exposure.items(), key=lambda item: (-item[1], item[0])
        )[:10]
    }
    scenario_max: dict[str, float] = defaultdict(float)
    scenario_sum: dict[str, float] = defaultdict(float)
    for scenario in scenarios or []:
        for target in scenario.get("top_affected", []):
            try:
                raw_protocol = target["protocol"]
                if raw_protocol is None:
                    continue
                protocol = str(raw_protocol).strip()
                loss = abs(float(target["loss"]))
            except (KeyError, TypeError, ValueError):
                continue
            if (
                protocol
                and protocol in selected_protocols
                and protocol in node_ids
                and math.isfinite(loss)
            ):
                scenario_max[protocol] = max(scenario_max[protocol], loss)
                scenario_sum[protocol] += loss

    protocols = sorted(selected_protocols)
    return CandidateSet(
        tuple(
            (
                protocol,
                exposure.get(protocol, 0.0),
                scenario_max.get(protocol, 0.0),
                scenario_sum.get(protocol, 0.0),
            )
            for protocol in protocols
        )
    )


def build_prompt(method: str, date: str, horizon: int,
                 forecast: dict | None, snapshot: dict | None,
                 metrics: dict, scenarios: list[dict] | None,
                 candidate_set: CandidateSet | None = None) -> tuple[str, str]:
    system = SYSTEM_PROMPT.format(horizon=horizon)

    if method in ("m6_fm_llm", "m7_fm_llm_gated"):
        assert forecast is not None
        candidate_set = candidate_set or build_candidate_set(forecast, scenarios)
        user = C0_LLM_USER_TEMPLATE.format(
            date=date, horizon=horizon,
            n_nodes=forecast.get("n_nodes", 0),
            n_edges=forecast.get("n_edges", 0),
            top_protocols=_format_top_protocols(forecast, candidate_set),
            metrics=_format_metrics(metrics),
            scenarios=_format_scenarios(
                scenarios or [],
                allowed_protocols={row[0] for row in candidate_set.rows},
            ),
        )
    elif method == "m2_snapshot_llm":
        assert snapshot is not None
        candidate_set = candidate_set or build_candidate_set(snapshot)
        edges = snapshot.get("edges", [])
        user = C3_USER_TEMPLATE.format(
            date=date, horizon=horizon,
            n_nodes=len(snapshot.get("nodes", {})),
            n_edges=len(edges),
            total_weight=sum(e["weight"] for e in edges),
            top_protocols=_format_top_protocols(snapshot, candidate_set),
            category_summary=_format_categories(snapshot),
            metrics=_format_metrics(metrics),
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return system, user


# ---------------------------------------------------------------------------
# LLM API caller (OpenRouter / OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------


def _is_reasoning_model(api_model: str) -> bool:
    """Return True for OpenRouter models that produce hidden reasoning tokens
    by default (Gemini 2.x/3.x, GPT-5.x). Anthropic models default to no
    extended thinking and are treated as non-reasoning here.
    """
    m = api_model.lower()
    if m.startswith("google/gemini-2.") or m.startswith("google/gemini-3."):
        return True
    if m.startswith("openai/gpt-5"):
        return True
    if m.startswith("x-ai/grok-4"):
        return True
    return False


def call_llm(system: str, user: str, model: str | None = None,
             reasoning_effort: str | None = None) -> dict:
    """Call LLM via OpenRouter (or any OpenAI-compatible endpoint).

    For models that default to chain-of-thought reasoning (Gemini 2.x/3.x,
    GPT-5.x, Grok-4) we attach a reasoning control so judge/decision
    comparisons stay apples-to-apples against Anthropic models, which have
    no extended thinking by default. Pass `reasoning_effort="minimal"`
    explicitly to force minimal thinking; the default ("auto") attaches
    "minimal" only for reasoning-by-default models.

    Returns parsed JSON + usage metadata.
    """
    import urllib.request, urllib.error

    model = model or DECISION_MODEL
    # Normalize dash-form (claude-opus-4-8) to OpenRouter dot-form (4.8)
    # before adding the provider prefix.
    model = _normalize_model_slug(model)
    # OpenRouter needs provider prefix
    api_model = f"anthropic/{model}" if "/" not in model else model

    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }

    # Apply reasoning control: explicit override, or "minimal" auto-attached
    # for reasoning-by-default models so the cross-family judge panel is run
    # at a comparable thinking budget.
    eff = reasoning_effort
    if eff is None and _is_reasoning_model(api_model):
        eff = "minimal"
    if eff is not None:
        payload["reasoning"] = {"effort": eff}

    import http.client

    t0 = time.time()
    attempt_records: list[dict] = []
    for attempt in range(5):
        attempt_started = time.time()
        try:
            req = urllib.request.Request(LLM_API_URL, method="POST")
            req.data = json.dumps(payload).encode()
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {LLM_API_KEY}")
            req.add_header("HTTP-Referer", "https://github.com/dexposure-agent")
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
            input_tokens, output_tokens, usage_known = response_usage(result)
            attempt_records.append(
                {
                    "attempt_index": attempt,
                    "status": "ok",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "usage_known": usage_known,
                    "latency_ms": round((time.time() - attempt_started) * 1000, 1),
                }
            )
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:500]
            logger.error(f"LLM API HTTP {e.code}: {body}")
            attempt_records.append(
                {
                    "attempt_index": attempt,
                    "status": "error",
                    "error": f"HTTP {e.code}: {body}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "usage_known": False,
                    "latency_ms": round((time.time() - attempt_started) * 1000, 1),
                }
            )
            if attempt < 4 and e.code in (429, 500, 502, 503):
                wait = 15 * (attempt + 1)
                logger.warning(f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM API HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.IncompleteRead, http.client.RemoteDisconnected) as e:
            attempt_records.append(
                {
                    "attempt_index": attempt,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "usage_known": False,
                    "latency_ms": round((time.time() - attempt_started) * 1000, 1),
                }
            )
            if attempt < 4:
                wait = 10 * (attempt + 1)
                logger.warning(f"LLM API attempt {attempt+1} failed: {type(e).__name__}: {e}, retry in {wait}s")
                time.sleep(wait)
            else:
                raise RuntimeError("LLM API call failed after 5 attempts") from e

    latency_ms = (time.time() - t0) * 1000

    raw = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    # Parse JSON (handle markdown wrapper)
    text = raw
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Judges occasionally mimic the template's escaped braces and emit
        # {{...}}; unescape and retry before falling back.
        try:
            parsed = json.loads(text.replace("{{", "{").replace("}}", "}"))
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        input_tokens, output_tokens, usage_known = response_usage(result)
        return {
            "status": "parse_error",
            "error": f"LLM returned invalid JSON: {text[:200]}",
            "raw_response": raw,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "usage_known": usage_known,
            "latency_ms": round(latency_ms, 1),
            "model": api_model,
            "attempts": attempt_records,
        }

    input_tokens, output_tokens, usage_known = response_usage(result)
    parsed["raw_response"] = raw
    parsed["input_tokens"] = input_tokens
    parsed["output_tokens"] = output_tokens
    parsed["usage_known"] = usage_known
    parsed["latency_ms"] = round(latency_ms, 1)
    parsed["model"] = api_model
    parsed["attempts"] = attempt_records
    return parsed


def response_usage(response: dict[str, object]) -> tuple[int, int, bool]:
    """Return provider token counts only when both fields are explicit integers."""
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, False
    values = (usage.get("prompt_tokens"), usage.get("completion_tokens"))
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        or not float(value).is_integer()
        for value in values
    ):
        return 0, 0, False
    return int(values[0]), int(values[1]), True


def _usage_cost(record: dict, rates: dict[str, float]) -> float | None:
    """Cost one attempted request, including rejected contract outputs."""
    if record.get("usage_known") is not True:
        return None
    return (
        int(record.get("input_tokens", 0) or 0) * rates["input"]
        + int(record.get("output_tokens", 0) or 0) * rates["output"]
    ) / 1_000_000


def summarize_attempt_usage(
    attempts: list[dict], rates: dict[str, float]
) -> dict[str, int | float | None]:
    """Aggregate attempted-request usage without pricing unknown calls as zero."""
    input_tokens = sum(int(row.get("input_tokens", 0) or 0) for row in attempts)
    output_tokens = sum(int(row.get("output_tokens", 0) or 0) for row in attempts)
    unknown_attempts = sum(
        row.get("usage_known") is not True for row in attempts
    )
    known_cost = (
        input_tokens * rates["input"] + output_tokens * rates["output"]
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "unknown_usage_attempts": unknown_attempts,
        "known_cost_usd": known_cost,
        "cost_usd": None if unknown_attempts else known_cost,
    }


def billable_request_attempts(record: dict) -> list[dict]:
    """Return provider request attempts without double-counting the final response."""
    attempts = record.get("attempts")
    if isinstance(attempts, list) and attempts and all(
        isinstance(attempt, dict) for attempt in attempts
    ):
        return attempts
    return [record]


def recorded_candidate_set(entry: dict) -> CandidateSet:
    """Validate and return the typed candidates bound to a persisted prompt."""
    bundle = entry.get("evidence_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("persisted decision row has no evidence_bundle")
    rows = bundle.get("candidate_set")
    if not isinstance(rows, list) or not rows:
        raise ValueError("persisted decision row has no CandidateSet")
    candidate_set = CandidateSet.from_rows(rows)
    if bundle.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
        raise ValueError("persisted CandidateSet hash mismatch")
    if (
        bundle.get("candidate_membership_sha256")
        != candidate_set.candidate_membership_sha256
    ):
        raise ValueError("persisted candidate membership hash mismatch")
    EvidenceBundle(
        date=str(entry.get("date", "")),
        system_prompt=str(entry.get("system_prompt", "")),
        user_prompt=str(entry.get("user_prompt", "")),
        evidence_sha256=str(bundle.get("evidence_sha256", "")),
        candidate_set=candidate_set,
    )
    return candidate_set


def validate_decision_output(
    output: dict,
    candidate_set: CandidateSet | None = None,
    *,
    allow_historical_count: bool = False,
) -> None:
    """Fail closed when a decision response drifts from the ticket schema."""
    if output.get("risk_level") not in {"low", "moderate", "elevated", "critical"}:
        raise ValueError("decision output has invalid risk_level")
    if not isinstance(output.get("rationale"), str) or not output["rationale"].strip():
        raise ValueError("decision output is missing rationale")
    targets = output.get("target_protocols")
    if not isinstance(targets, list):
        raise ValueError("decision output target_protocols must be a list")
    previous_score = float("inf")
    seen: set[str] = set()
    allowed_protocols = (
        {protocol for protocol, _, _, _ in candidate_set.rows}
        if candidate_set is not None
        else None
    )
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError("decision target must be an object")
        protocol = normalize_protocol_name(target.get("protocol", ""))
        score = target.get("risk_score")
        if not protocol or protocol in seen:
            raise ValueError("decision target protocol must be unique and non-empty")
        if allowed_protocols is not None and protocol not in allowed_protocols:
            raise ValueError(
                f"decision target protocol {protocol!r} is outside CandidateSet"
            )
        seen.add(protocol)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not np.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or float(score) > previous_score
        ):
            raise ValueError("decision risk_score must be finite, sorted, and in [0,1]")
        previous_score = float(score)
        if target.get("action") not in SEVERITY_ORDER:
            raise ValueError("decision target has invalid action")
        if not isinstance(target.get("reason"), str) or not target["reason"].strip():
            raise ValueError("decision target is missing reason")
    if not allow_historical_count and len(targets) != MAX_TARGET_BUDGET:
        raise ValueError(
            f"decision output must contain exactly {MAX_TARGET_BUDGET} unique targets"
        )


# ---------------------------------------------------------------------------
# Layer 2 metrics: LLM reasoning quality
# ---------------------------------------------------------------------------

def compute_grounding_score(llm_output: dict, input_data: str) -> float:
    """Fraction of target reasons that cite numeric data present in the input.

    A reason is "grounded" if it contains at least one number AND a metric
    keyword, and that number actually appears in the input prompt.
    """
    protos = llm_output.get("target_protocols", [])
    if not protos:
        return 0.0

    # Extract all numbers from input data for verification
    import re
    input_numbers = set()
    for match in re.findall(r'[\d]+\.[\d]+|[\d]+', input_data):
        input_numbers.add(match)

    metric_keywords = {"N1", "N2", "N3", "N4", "N5", "loss", "weight",
                       "pagerank", "gini", "hhi", "density", "%", "distressed",
                       "affected", "exposure"}

    grounded = 0
    for p in protos:
        reason = p.get("reason", "") + " " + llm_output.get("rationale", "")
        reason_numbers = set(re.findall(r'[\d]+\.[\d]+|[\d]+', reason))
        has_metric = any(kw.lower() in reason.lower() for kw in metric_keywords)
        has_traceable_number = bool(reason_numbers & input_numbers)
        if has_metric and has_traceable_number:
            grounded += 1
    return grounded / len(protos)


def compute_consistency(run_outputs: list[dict]) -> float:
    """Mean pairwise Jaccard of target sets across repeated runs."""
    if len(run_outputs) < 2:
        return 1.0
    tsets = [
        {
            normalize_protocol_name(p.get("protocol", ""))
            for p in o.get("target_protocols", [])
            if isinstance(p, dict) and normalize_protocol_name(p.get("protocol", ""))
        }
        for o in run_outputs
    ]
    jacs = [_jaccard(tsets[i], tsets[j])
            for i in range(len(tsets)) for j in range(i + 1, len(tsets))]
    return float(np.mean(jacs)) if jacs else 1.0


def apply_action_gate(llm_output: dict) -> dict:
    """Apply conservative intervention constraints to an LLM decision.

    This is deliberately a post-hoc safety gate, not a replacement for the
    deterministic rules engine. It lets us separate target-selection quality
    from the LLM's tendency to over-escalate intervention severity.
    """
    gated = json.loads(json.dumps(llm_output))
    risk_level = str(gated.get("risk_level", "")).lower()
    targets = gated.get("target_protocols", [])
    if not isinstance(targets, list):
        gated["target_protocols"] = []
        return gated

    for target in targets:
        if not isinstance(target, dict):
            continue
        action = target.get("action", "Monitor")
        try:
            risk_score = float(target.get("risk_score", 0.0))
        except (TypeError, ValueError):
            risk_score = 0.0

        rule = ACTION_GATE_POLICY.get(action)
        if rule and (
            risk_level not in rule["risk_levels"]
            or risk_score < rule["min_score"]
        ):
            if action == "Contingency":
                target["action"] = "Investigate"
                target["gate_note"] = (
                    "Demoted from Contingency: Contingency requires critical "
                    "overall risk and protocol risk_score >= 0.90."
                )
            else:
                target["action"] = "Investigate"
                target["gate_note"] = (
                    "Demoted from Recommend-Reduce: intervention requires "
                    "elevated/critical overall risk and protocol risk_score >= 0.75."
                )

    return gated


# ---------------------------------------------------------------------------
# Layer 3 metrics: end-to-end decision quality
# ---------------------------------------------------------------------------

def normalize_protocol_name(name: str) -> str:
    """Normalize protocol names before matching with ground truth.

    LLM outputs may include a trailing category suffix such as
    "AAVE (Lending)" while ground truth uses "AAVE". Remove this suffix so
    formatting differences do not bias precision/recall.
    """
    if not isinstance(name, str):
        return ""
    normalized = name.strip()
    if not normalized:
        return ""
    normalized = re.sub(r"\s*\([^)]*\)\s*$", "", normalized)
    return normalized.strip()


def assess_week(llm_output: dict, truly_stressed: set[str],
                actual_losses: dict[str, float],
                user_prompt: str,
                positive_loss_protocols: set[str] | None = None) -> dict:
    """Compute all Layer 2 + Layer 3 metrics for one week."""
    targets = {
        normalize_protocol_name(p.get("protocol", ""))
        for p in llm_output.get("target_protocols", [])
    }
    targets.discard("")
    truly_stressed_norm = {normalize_protocol_name(p) for p in truly_stressed}
    truly_stressed_norm.discard("")

    # Precision: fraction of flagged that are truly stressed
    precision = len(targets & truly_stressed_norm) / len(targets) if targets else 0.0
    # Recall / audit completeness
    completeness = (
        len(targets & truly_stressed_norm) / len(truly_stressed_norm)
        if truly_stressed_norm
        else 1.0
    )

    # Grounding score (Layer 2)
    grounding = compute_grounding_score(llm_output, user_prompt)

    # Severity correlation (Layer 3):
    # Does the LLM assign higher severity to protocols that actually lost more?
    severity_rho = None
    protos = llm_output.get("target_protocols", [])
    normalized_losses = {
        normalize_protocol_name(name): loss for name, loss in actual_losses.items()
    }
    normalized_losses.pop("", None)
    if len(protos) >= 3:
        sev_vals = []
        loss_vals = []
        for p in protos:
            name = normalize_protocol_name(p.get("protocol", ""))
            if not name:
                continue
            sev = SEVERITY_ORDER.get(p.get("action", "Monitor"), 1)
            loss = normalized_losses.get(name, 0.0)
            sev_vals.append(sev)
            loss_vals.append(loss)
        if len(set(sev_vals)) > 1:
            rho, _ = spearmanr(sev_vals, loss_vals)
            if np.isfinite(rho):
                severity_rho = float(rho)

    # False intervention rates are split between positive-loss protocols outside
    # the top-5% tail and protocols with no positive absolute loss at all.
    normalized_actual_losses = {
        normalize_protocol_name(name): float(loss or 0.0)
        for name, loss in actual_losses.items()
        if normalize_protocol_name(name)
    }
    positive_loss_norm = {
        normalize_protocol_name(name)
        for name in (positive_loss_protocols or set(actual_losses))
        if normalize_protocol_name(name)
        and normalized_actual_losses.get(normalize_protocol_name(name), 0.0) > 0.0
    }
    false_interventions = 0
    non_loss_false_interventions = 0
    intervention_targets = 0
    for p in protos:
        name = normalize_protocol_name(p.get("protocol", ""))
        if not name:
            continue
        if p.get("action") in ("Recommend-Reduce", "Contingency"):
            intervention_targets += 1
            if name not in truly_stressed_norm:
                false_interventions += 1
            if name not in positive_loss_norm:
                non_loss_false_interventions += 1
    fir = (
        false_interventions / intervention_targets
        if intervention_targets > 0
        else None
    )

    return {
        "targets": sorted(targets),
        "precision": precision,
        "completeness": completeness,
        "grounding_score": grounding,
        "severity_rho": severity_rho,
        "false_intervention_rate": fir,
        "tail_false_intervention_rate": fir,
        "non_loss_false_intervention_rate": (
            non_loss_false_interventions / intervention_targets
            if intervention_targets > 0
            else None
        ),
        "false_interventions": false_interventions,
        "tail_false_interventions": false_interventions,
        "non_loss_false_interventions": non_loss_false_interventions,
        "intervention_targets": intervention_targets,
    }


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


# ---------------------------------------------------------------------------
# LLM-as-Judge: explanation quality
# ---------------------------------------------------------------------------

def judge_explanation(date: str, horizon: int, metrics: dict,
                     truly_stressed: set[str], llm_output: dict) -> dict:
    """Rate the LLM's explanation quality using a cheaper judge model."""
    system = JUDGE_SYSTEM_PROMPT
    user = JUDGE_USER_TEMPLATE.format(
        date=date, horizon=horizon,
        n_nodes=metrics.get("n_nodes", 0),
        n_edges=metrics.get("n_edges", 0),
        metrics_summary=_format_metrics(metrics).replace("\n", ", "),
        n_stressed=len(truly_stressed),
        stressed_list=", ".join(sorted(truly_stressed)[:10]) or "none",
        risk_level=llm_output.get("risk_level", "?"),
        llm_targets=", ".join(p["protocol"] for p in llm_output.get("target_protocols", [])[:10]),
        rationale=llm_output.get("rationale", "N/A"),
    )

    result = call_llm(system, user, model=JUDGE_MODEL)
    score = result.get("quality_score", 0)
    usage = {
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "latency_ms": result.get("latency_ms", 0.0),
        "attempts": result.get("attempts", []),
    }
    if isinstance(score, (int, float)) and 1 <= score <= 5:
        return {
            "quality_score": int(score),
            "judge_output_valid": True,
            "reasoning": result.get("reasoning", ""),
            **usage,
        }
    return {
        "quality_score": None,
        "judge_output_valid": False,
        "reasoning": result.get("reasoning", ""),
        **usage,
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WeekResult:
    date: str
    method: str
    risk_level: str = ""
    targets: list[str] = field(default_factory=list)
    truly_stressed: list[str] = field(default_factory=list)
    precision: float = 0.0
    completeness: float = 0.0
    grounding_score: float = 0.0
    consistency: float = 1.0
    severity_rho: float | None = None
    false_intervention_rate: float | None = None
    tail_false_intervention_rate: float | None = None
    non_loss_false_intervention_rate: float | None = None
    false_interventions: int = 0
    tail_false_interventions: int = 0
    non_loss_false_interventions: int = 0
    intervention_targets: int = 0
    explanation_quality: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Checkpoint / resume support
# ---------------------------------------------------------------------------

def _checkpoint_path(run_dir: Path, method: str) -> Path:
    return run_dir / f"checkpoint_{method}.json"


def _save_checkpoint(run_dir: Path, method: str, completed_dates: list[str],
                     week_results: list[WeekResult], raw_entries: list[dict],
                     run_identity: dict):
    cp = {
        "method": method,
        "run_identity": run_identity,
        "completed_dates": completed_dates,
        "week_results": [asdict(wr) for wr in week_results],
        "raw_entries": raw_entries,
    }
    _checkpoint_path(run_dir, method).write_text(
        json.dumps(_json_safe(cp), indent=2, default=str, allow_nan=False)
    )


def _load_checkpoint(
    run_dir: Path, method: str, expected_identity: dict
) -> tuple[set[str], list[WeekResult], list[dict]]:
    cp_path = _checkpoint_path(run_dir, method)
    if not cp_path.exists():
        return set(), [], []
    cp = json.loads(cp_path.read_text())
    if cp.get("method") != method or cp.get("run_identity") != expected_identity:
        raise ValueError(f"Checkpoint identity mismatch: {cp_path}")
    wrs = [WeekResult(**wr) for wr in cp.get("week_results", [])]
    return set(cp.get("completed_dates", [])), wrs, cp.get("raw_entries", [])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    methods: list[str],
    test_split: str = "2025-01~2025-08",
    horizon: int = STRESS_LOOKAHEAD,
    consistency_runs: int = CONSISTENCY_RUNS,
    resume: bool = False,
    run_judge: bool = True,
    export_prompts_only: bool = False,
) -> dict[str, list[WeekResult]]:
    """Run LLM evaluation pipeline locally.

    Connects to FM API via SSH tunnel, calls Claude API for decisions.
    With export_prompts_only=True, no LLM is called: the assembled prompts
    plus ground truth are dumped to prompts_<method>.json for later replay
    (used for the crisis-window backtest, where evidence is generated on
    the GPU box and the LLM decision layer is replayed locally).
    """
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("LLM evaluation methods must be non-empty and unique")
    unknown_methods = sorted(set(methods) - LLM_METHODS)
    if unknown_methods:
        raise ValueError(f"Unknown LLM evaluation methods: {unknown_methods}")
    independent_m7 = sorted(set(methods) & GATED_LLM_METHODS)
    if independent_m7:
        raise ValueError(
            "m7_fm_llm_gated is replay-only and cannot be sampled by "
            "run_pipeline; sample m6_fm_llm, persist raw tickets, then apply "
            "the gate offline to that exact artifact"
        )
    if horizon not in {1, 4, 8, 12}:
        raise ValueError("LLM evaluation horizon must be one of 1, 4, 8, or 12")
    if not isinstance(consistency_runs, int) or isinstance(consistency_runs, bool) or consistency_runs <= 0:
        raise ValueError("consistency_runs must be a positive integer")

    # Verify API key (not needed when only exporting prompts)
    if not LLM_API_KEY and not export_prompts_only:
        logger.error("No LLM API key set. Export OPENROUTER_API_KEY or ANTHROPIC_API_KEY.")
        sys.exit(1)

    # Verify FM API
    try:
        health = _http_get("/health")
        logger.info(f"FM API: {health}")
        if not health.get("fm_available"):
            logger.error("FM model not available on server")
            sys.exit(1)
    except Exception as e:
        logger.error(f"FM API unreachable at {FM_API_URL}: {e}")
        logger.error("Set up SSH tunnel: ssh -f -N -L 8000:localhost:8000 gpu-server")
        sys.exit(1)

    # Setup run directory (resume reuses the latest existing llm_eval_* dir)
    if resume:
        existing = sorted(RESULTS_DIR.glob("llm_eval_*/"), key=lambda p: p.name, reverse=True)
        if existing and any((existing[0] / f"checkpoint_{m}.json").exists() for m in methods):
            run_dir = existing[0]
            logger.info(f"Resuming from existing run: {run_dir.name}")
        else:
            run_dir = RESULTS_DIR / f"llm_eval_{time.strftime('%Y%m%d_%H%M%S')}"
            run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = RESULTS_DIR / f"llm_eval_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "eval.log"
    log_id = logger.add(str(log_file), level="DEBUG",
                        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}")

    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Decision model: {DECISION_MODEL}")
    logger.info(f"Judge model: {JUDGE_MODEL}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Consistency runs: {consistency_runs}")

    template_sha256 = hashlib.sha256(
        "\n".join(
            [
                SYSTEM_PROMPT,
                C0_LLM_USER_TEMPLATE,
                C3_USER_TEMPLATE,
                JUDGE_SYSTEM_PROMPT,
                JUDGE_USER_TEMPLATE,
            ]
        ).encode("utf-8")
    ).hexdigest()
    run_identity = {
        "test_split": test_split,
        "horizon": horizon,
        "consistency_runs": consistency_runs,
        "decision_model": DECISION_MODEL,
        "judge_model": JUDGE_MODEL if run_judge else None,
        "run_judge": run_judge,
        "export_prompts_only": export_prompts_only,
        "llm_api_url": LLM_API_URL,
        "fm_api_url": FM_API_URL,
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "template_sha256": template_sha256,
    }

    # Get dates
    test_dates = get_test_dates(test_split)
    all_dates = _http_get("/dates")["dates"]
    logger.info(f"Test dates: {len(test_dates)} weeks ({test_dates[0]} to {test_dates[-1]})")

    # Results
    results: dict[str, list[WeekResult]] = {}
    raw_logs: dict[str, list[dict]] = {}
    total_cost = 0.0
    if not export_prompts_only and DECISION_MODEL not in MODEL_COSTS:
        raise ValueError(f"No explicit cost configuration for decision model {DECISION_MODEL}")
    if run_judge and JUDGE_MODEL not in MODEL_COSTS:
        raise ValueError(f"No explicit cost configuration for judge model {JUDGE_MODEL}")
    costs = MODEL_COSTS.get(DECISION_MODEL, {"input": 0.0, "output": 0.0})
    judge_costs = MODEL_COSTS.get(JUDGE_MODEL, {"input": 0.0, "output": 0.0})

    for method in methods:
        # Resume support
        done_dates, prev_results, prev_raw = set(), [], []
        if resume:
            done_dates, prev_results, prev_raw = _load_checkpoint(
                run_dir, method, {**run_identity, "method": method}
            )
            if done_dates:
                logger.info(f"{method}: resuming, {len(done_dates)} weeks already done")
        results[method] = list(prev_results)
        raw_logs[method] = list(prev_raw)

        for wi, date_str in enumerate(test_dates):
            if date_str in done_dates:
                continue

            # Find future date for ground truth
            if date_str not in all_dates:
                continue
            t_idx = all_dates.index(date_str)
            future_idx = t_idx + horizon
            if future_idx >= len(all_dates):
                logger.info(f"[{wi+1}/{len(test_dates)}] {date_str}: no future snapshot, skip")
                continue
            future_date = all_dates[future_idx]

            # Fetch data from GPU server
            snap_current = get_snapshot(date_str)
            snap_future = get_snapshot(future_date)
            truly_stressed = detect_truly_stressed(snap_current, snap_future)
            if not truly_stressed:
                raise ValueError(f"{date_str}: ground-truth stressed set is empty")
            actual_losses = compute_actual_loss(snap_current, snap_future)

            logger.info(f"[{wi+1}/{len(test_dates)}] {date_str} -> {future_date}: "
                        f"{len(truly_stressed)} truly stressed")

            # Collect data for prompt
            forecast = None
            scenarios = None
            if method in ("m6_fm_llm", "m7_fm_llm_gated"):
                forecast = get_forecast(date_str, horizon)
                metrics = compute_metrics(forecast)
                scenarios = run_scenarios(forecast)
                evidence_source = forecast
            elif method == "m2_snapshot_llm":
                metrics = compute_metrics(snap_current)
                evidence_source = snap_current

            candidate_set = build_candidate_set(evidence_source, scenarios)
            forecast_digest = (
                forecast_sha256(forecast, horizon)
                if forecast is not None
                else None
            )
            if len(candidate_set.rows) < MAX_TARGET_BUDGET:
                raise ValueError(
                    f"{date_str}: candidate set has {len(candidate_set.rows)} "
                    f"protocols; exactly {MAX_TARGET_BUDGET} are required"
                )
            system, user = build_prompt(
                method,
                date_str,
                horizon,
                forecast,
                snap_current,
                metrics,
                scenarios,
                candidate_set=candidate_set,
            )
            evidence_bundle = EvidenceBundle(
                date=date_str,
                system_prompt=system,
                user_prompt=user,
                candidate_set=candidate_set,
            )

            if export_prompts_only:
                raw_logs[method].append({
                    "date": date_str, "method": method, "horizon": horizon,
                    "future_date": future_date,
                    "forecast_sha256": forecast_digest,
                    "system_prompt": system, "user_prompt": user,
                    "evidence_bundle": {
                        "schema_version": 1,
                        "date": evidence_bundle.date,
                        "evidence_sha256": evidence_bundle.evidence_sha256,
                        "candidate_set_sha256": candidate_set.candidate_set_sha256,
                        "candidate_membership_sha256": (
                            candidate_set.candidate_membership_sha256
                        ),
                        "forecast_sha256": forecast_digest,
                        "candidate_set": candidate_set.as_dicts(),
                    },
                    "truly_stressed": sorted(truly_stressed),
                    "actual_losses": actual_losses,
                    "positive_loss_protocols": sorted(
                        positive_absolute_losses(snap_current, snap_future)
                    ),
                })
                with open(run_dir / f"prompts_{method}.json", "w") as f:
                    json.dump(raw_logs[method], f, indent=1)
                logger.info(f"[{wi+1}/{len(test_dates)}] {date_str}: prompt exported "
                            f"({len(truly_stressed)} stressed)")
                continue

            # Run LLM (multiple times for consistency).  ``run_outputs`` keeps
            # the exact raw tickets for every method; M7 receives a separate
            # post-hoc gated copy for assessment.
            raw_outputs = []
            evaluated_outputs = []
            llm_attempts: list[dict] = []
            unknown_usage_attempts = 0
            for ri in range(consistency_runs):
                out = None
                run_attempts: list[dict] = []
                for validation_attempt in range(3):
                    try:
                        candidate = call_llm(system, user)
                    except Exception as exc:
                        candidate = {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "usage_known": False,
                            "latency_ms": 0.0,
                        }
                    attempt_record = {
                        "consistency_run": ri,
                        "validation_attempt": validation_attempt,
                        "status": "ok",
                        "validation_error": None,
                        **candidate,
                    }
                    try:
                        validate_decision_output(candidate, candidate_set)
                        out = candidate
                        attempt_record["validation_status"] = "accepted"
                        run_attempts.append(attempt_record)
                        break
                    except (TypeError, ValueError) as exc:
                        attempt_record["status"] = "contract_error"
                        attempt_record["validation_status"] = "rejected"
                        attempt_record["validation_error"] = str(exc)
                        run_attempts.append(attempt_record)
                        if validation_attempt == 2:
                            raise
                        logger.warning(
                            "decision output failed contract on consistency run "
                            f"{ri + 1}, validation retry {validation_attempt + 1}: {exc}"
                        )
                assert out is not None
                raw_outputs.append(out)
                evaluated_outputs.append(
                    apply_action_gate(out) if method in GATED_LLM_METHODS else out
                )
                llm_attempts.extend(run_attempts)
                for attempt in run_attempts:
                    for request_attempt in billable_request_attempts(attempt):
                        attempt_cost = _usage_cost(request_attempt, costs)
                        if attempt_cost is None:
                            unknown_usage_attempts += 1
                        else:
                            total_cost += attempt_cost

            primary = evaluated_outputs[0]
            positive_loss_protocols = set(
                positive_absolute_losses(snap_current, snap_future)
            )
            ev = assess_week(
                primary,
                truly_stressed,
                actual_losses,
                user,
                positive_loss_protocols=positive_loss_protocols,
            )
            consistency = compute_consistency(evaluated_outputs)

            # LLM-as-Judge
            explain_quality = None
            judge_result = None
            if run_judge:
                judge_result = judge_explanation(
                    date_str, horizon, metrics, truly_stressed, primary)
                explain_quality = judge_result.get("quality_score", 0)
                j_attempts = judge_result.get("attempts")
                if not isinstance(j_attempts, list) or not j_attempts:
                    j_attempts = [judge_result]
                for attempt in j_attempts:
                    attempt_cost = _usage_cost(attempt, judge_costs)
                    if attempt_cost is None:
                        unknown_usage_attempts += 1
                    else:
                        total_cost += attempt_cost

            wr = WeekResult(
                date=date_str, method=method,
                risk_level=primary.get("risk_level", "unknown"),
                targets=ev["targets"],
                truly_stressed=sorted(truly_stressed),
                precision=ev["precision"],
                completeness=ev["completeness"],
                grounding_score=ev["grounding_score"],
                consistency=consistency,
                severity_rho=ev["severity_rho"],
                false_intervention_rate=ev["false_intervention_rate"],
                tail_false_intervention_rate=ev["tail_false_intervention_rate"],
                non_loss_false_intervention_rate=ev[
                    "non_loss_false_intervention_rate"
                ],
                false_interventions=ev["false_interventions"],
                tail_false_interventions=ev["tail_false_interventions"],
                non_loss_false_interventions=ev[
                    "non_loss_false_interventions"
                ],
                intervention_targets=ev["intervention_targets"],
                explanation_quality=explain_quality,
                input_tokens=primary.get("input_tokens", 0),
                output_tokens=primary.get("output_tokens", 0),
                latency_ms=primary.get("latency_ms", 0),
            )
            results[method].append(wr)

            raw_entry = {
                "date": date_str, "method": method,
                "system_prompt": system, "user_prompt": user,
                "llm_outputs": raw_outputs,
                "raw_llm_outputs": raw_outputs,
                "evaluated_llm_outputs": evaluated_outputs,
                "llm_attempts": llm_attempts,
                "unknown_usage_attempts": unknown_usage_attempts,
                "action_gate_applied": method in GATED_LLM_METHODS,
                "llm_outputs_role": "raw",
                "evidence_bundle": {
                    "schema_version": 1,
                    "date": evidence_bundle.date,
                    "evidence_sha256": evidence_bundle.evidence_sha256,
                    "candidate_set_sha256": candidate_set.candidate_set_sha256,
                    "candidate_membership_sha256": (
                        candidate_set.candidate_membership_sha256
                    ),
                    "forecast_sha256": forecast_digest,
                    "candidate_set": candidate_set.as_dicts(),
                },
                "forecast_sha256": forecast_digest,
                "truly_stressed": sorted(truly_stressed),
                # Persist losses so re-decide can recompute FIR without a
                # snapshot service (avoids the empty-actual_losses FIR bug).
                "actual_losses": actual_losses,
                "positive_loss_protocols": sorted(positive_loss_protocols),
                "assessment": ev, "consistency": consistency,
                "explanation_quality": explain_quality,
                "judge_result": judge_result,
            }
            raw_logs[method].append(raw_entry)

            fir_text = (
                f"{ev['false_intervention_rate']:.3f}"
                if ev["false_intervention_rate"] is not None
                else "N/A"
            )
            logger.info(f"  {method}: prec={ev['precision']:.3f} "
                        f"compl={ev['completeness']:.3f} "
                        f"ground={ev['grounding_score']:.3f} "
                        f"consist={consistency:.3f} "
                        f"sev_rho={ev['severity_rho'] if ev['severity_rho'] is not None else 'N/A'} "
                        f"FIR={fir_text} "
                        f"explain={explain_quality if explain_quality is not None else 'invalid'}/5 "
                        f"targets={len(ev['targets'])}")

            # Checkpoint after each week
            done_dates.add(date_str)
            _save_checkpoint(run_dir, method,
                             sorted(done_dates), results[method], raw_logs[method],
                             {**run_identity, "method": method})

    # ---------------------------------------------------------------------------
    # Aggregate summaries
    # ---------------------------------------------------------------------------

    summaries = {}
    for method, wrs in results.items():
        if not wrs:
            continue

        precs = [w.precision for w in wrs]
        compls = [w.completeness for w in wrs]
        grounds = [w.grounding_score for w in wrs]
        consists = [w.consistency for w in wrs]
        sev_rhos = [
            w.severity_rho
            for w in wrs
            if isinstance(w.severity_rho, (int, float))
            and np.isfinite(w.severity_rho)
        ]
        firs = [w.false_intervention_rate for w in wrs if w.false_intervention_rate is not None]
        explains = [
            w.explanation_quality
            for w in wrs
            if isinstance(w.explanation_quality, (int, float))
            and w.explanation_quality > 0
        ]
        intervention_targets = sum(w.intervention_targets for w in wrs)
        false_interventions = sum(w.false_interventions for w in wrs)
        tail_false_interventions = sum(w.tail_false_interventions for w in wrs)
        non_loss_false_interventions = sum(
            w.non_loss_false_interventions for w in wrs
        )
        decision_attempts = [
            request_attempt
            for entry in raw_logs.get(method, [])
            for attempt in entry.get("llm_attempts", [])
            for request_attempt in billable_request_attempts(attempt)
        ]
        judge_attempts = [
            attempt
            for entry in raw_logs.get(method, [])
            for attempt in (entry.get("judge_result") or {}).get("attempts", [])
        ]
        decision_usage = summarize_attempt_usage(decision_attempts, costs)
        judge_usage = summarize_attempt_usage(judge_attempts, judge_costs)
        decision_unknown_usage_attempts = int(
            decision_usage["unknown_usage_attempts"]
        )
        judge_unknown_usage_attempts = int(judge_usage["unknown_usage_attempts"])
        unknown_usage_attempts = (
            decision_unknown_usage_attempts + judge_unknown_usage_attempts
        )
        stabilities = [
            _jaccard(set(wrs[i - 1].targets), set(wrs[i].targets))
            for i in range(1, len(wrs))
            if wrs[i - 1].targets or wrs[i].targets
        ]

        summary = {
            "method": method,
            "model": DECISION_MODEL,
            "n_weeks": len(wrs),
            # Layer 3
            "ticket_precision": round(float(np.mean(precs)), 4),
            "audit_completeness": round(float(np.mean(compls)), 4),
            "ticket_f1_macro": round(
                float(np.mean([
                    2.0 * precision * completeness / (precision + completeness)
                    if precision + completeness > 0
                    else 0.0
                    for precision, completeness in zip(precs, compls)
                ])),
                4,
            ),
            "target_stability": round(float(np.mean(stabilities)), 4) if stabilities else None,
            "severity_correlation": round(float(np.mean(sev_rhos)), 4) if sev_rhos else None,
            "false_intervention_rate": (
                round(false_interventions / intervention_targets, 4)
                if intervention_targets
                else None
            ),
            "false_intervention_rate_micro": (
                round(false_interventions / intervention_targets, 4)
                if intervention_targets
                else None
            ),
            "false_intervention_rate_macro_defined": (
                round(float(np.mean(firs)), 4) if firs else None
            ),
            "n_fir_defined_weeks": len(firs),
            "intervention_targets": intervention_targets,
            "false_interventions": false_interventions,
            "tail_false_interventions": tail_false_interventions,
            "non_loss_false_interventions": non_loss_false_interventions,
            # Layer 2
            "grounding_score": round(float(np.mean(grounds)), 4),
            "consistency": round(float(np.mean(consists)), 4),
            "explanation_quality": round(float(np.mean(explains)), 2) if explains else None,
            # Cost
            "total_input_tokens": sum(w.input_tokens for w in wrs),
            "total_output_tokens": sum(w.output_tokens for w in wrs),
            "attempt_input_tokens": decision_usage["input_tokens"],
            "attempt_output_tokens": decision_usage["output_tokens"],
            "judge_attempt_input_tokens": judge_usage["input_tokens"],
            "judge_attempt_output_tokens": judge_usage["output_tokens"],
            "unknown_usage_attempts": unknown_usage_attempts,
            "decision_unknown_usage_attempts": decision_unknown_usage_attempts,
            "judge_unknown_usage_attempts": judge_unknown_usage_attempts,
            "attempt_cost_usd": decision_usage["cost_usd"],
            "known_attempt_cost_usd": decision_usage["known_cost_usd"],
            "judge_attempt_cost_usd": judge_usage["cost_usd"],
            "known_judge_attempt_cost_usd": judge_usage["known_cost_usd"],
            "decision_rate_usd_per_million_tokens": dict(costs),
            "judge_rate_usd_per_million_tokens": dict(judge_costs),
            "rate_card_evidence": (
                "repository MODEL_COSTS constant; external source and freeze date "
                "must be recorded before publication"
            ),
        }
        summaries[method] = summary

        logger.info(f"\n{'='*60}\n{method} SUMMARY ({len(wrs)} weeks):")
        for k, v in summary.items():
            if k == "method":
                continue
            logger.info(f"  {k}: {v}")

    # Save results
    for method, summary in summaries.items():
        with open(run_dir / f"summary_{method}.json", "w") as f:
            json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    for method, raw in raw_logs.items():
        with open(run_dir / f"raw_{method}.json", "w") as f:
            json.dump(_json_safe(raw), f, indent=2, default=str, allow_nan=False)

    # Combined comparison table
    comparison = {"run_id": run_dir.name, "methods": summaries}
    with open(run_dir / "comparison.json", "w") as f:
        json.dump(_json_safe(comparison), f, indent=2, allow_nan=False)

    # Print comparison table
    _print_comparison_table(summaries)

    logger.info(f"\nTotal estimated cost: ${total_cost:.2f}")
    logger.info(f"Results saved to {run_dir}/")

    logger.remove(log_id)
    return results


def _print_comparison_table(summaries: dict[str, dict]):
    """Print a formatted comparison table to stdout and logger."""
    methods = list(summaries.keys())
    if not methods:
        return

    header = f"{'Method':<12} {'Prec':>6} {'Recall':>7} {'Stabil':>7} {'Ground':>7} {'Consist':>8} {'SevRho':>7} {'FIR':>6} {'Explain':>8}"
    sep = "-" * len(header)
    lines = ["\n" + sep, header, sep]

    for m in methods:
        s = summaries[m]
        severity = s.get("severity_correlation")
        sev = f"{severity:.3f}" if severity is not None else "N/A"
        fir = s.get("false_intervention_rate")
        fir_text = f"{fir:.3f}" if fir is not None else "N/A"
        stability = s.get("target_stability")
        stability_text = f"{stability:.3f}" if stability is not None else "N/A"
        explanation = s.get("explanation_quality")
        explanation_text = (
            f"{explanation:.1f}/5" if explanation is not None else "N/A"
        )
        lines.append(
            f"{m:<12} "
            f"{s['ticket_precision']:>6.3f} "
            f"{s['audit_completeness']:>7.3f} "
            f"{stability_text:>7} "
            f"{s['grounding_score']:>7.3f} "
            f"{s['consistency']:>8.3f} "
            f"{sev:>7} "
            f"{fir_text:>6} "
            f"{explanation_text:>9}"
        )
    lines.append(sep)
    table_str = "\n".join(lines)
    print(table_str)
    logger.info(table_str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="LLM Decision Evaluation Pipeline (runs locally)")
    parser.add_argument("--method", action="append", default=[],
                        help="m6_fm_llm and/or m2_snapshot_llm (repeatable). "
                             "m7_fm_llm_gated is replay-only over raw M6 tickets.")
    parser.add_argument("--test-split", default="2025-01~2025-08")
    parser.add_argument("--horizon", type=int, default=STRESS_LOOKAHEAD)
    parser.add_argument("--consistency-runs", type=int, default=CONSISTENCY_RUNS)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--export-prompts-only", action="store_true",
                        help="Assemble and dump prompts + ground truth without "
                             "calling any LLM (for crisis-window replay)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-Judge evaluation")
    parser.add_argument("--model", default=None,
                        help=f"Decision model (default: {DECISION_MODEL})")
    parser.add_argument("--judge-model", default=None,
                        help=f"Judge model (default: {JUDGE_MODEL})")
    args = parser.parse_args()

    if not args.method:
        args.method = ["m6_fm_llm", "m2_snapshot_llm"]
    if args.model:
        DECISION_MODEL = args.model
    if args.judge_model:
        JUDGE_MODEL = args.judge_model

    run_pipeline(
        methods=args.method,
        test_split=args.test_split,
        horizon=args.horizon,
        consistency_runs=args.consistency_runs,
        resume=args.resume,
        run_judge=not args.no_judge,
        export_prompts_only=args.export_prompts_only,
    )
