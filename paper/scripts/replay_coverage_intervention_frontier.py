#!/usr/bin/env python3
"""Strict same-input replay for the coverage--intervention-risk claim.

The original m6_fm_llm and m7_fm_llm_gated runs regenerated the FM forecast
and called the decision LLM independently.  Their differences therefore mix
input drift, model sampling, and the action gate.  This replay removes those
confounds by using each saved m6 prompt as the single frozen evidence bundle.

Policies evaluated on every identical bundle:

* exposure_rules: rank the prompt's protocol candidates by predicted exposure.
* scenario_max_rules: rank by maximum absolute scenario loss, then exposure.
* scenario_sum_rules: rank by summed absolute scenario loss, then exposure.
* llm_raw: replay the cached primary LLM decision.
* llm_gated_replay: apply the deterministic action gate to that exact output.

All rule policies emit Investigate actions.  They test target selection, not
intervention calibration.  The LLM policies retain their emitted actions so
the same run yields coverage, intervention risk, and token-cost measurements.

Outputs are deterministic apart from the generated_at timestamp.  No API or
FM service is required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_ROOT = ROOT / "paper"
if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))
from experiments.b5_decision_quality import (  # noqa: E402
    CandidateSet,
    EvidenceBundle,
    evidence_hash as _canonical_evidence_hash,
    positive_absolute_losses,
    ranked_unique_targets as _canonical_ranked_unique_targets,
    require_exact_k_targets,
)
from experiments.llm_eval_b5 import (  # noqa: E402
    apply_action_gate as _canonical_action_gate,
    billable_request_attempts,
)
DEFAULT_OUTPUT_DIR = ROOT / "paper/results/strict_same_input_replay"
DEFAULT_BUDGETS = (1, 3, 5, 7)
INTERVENTION_ACTIONS = {"Recommend-Reduce", "Contingency"}
ALLOWED_RISK_LEVELS = {"low", "moderate", "elevated", "critical"}
ALLOWED_ACTIONS = {"Monitor", "Investigate", *INTERVENTION_ACTIONS}
_TOP_PROTOCOL_RE = re.compile(
    r"^\s*\d+\.\s+(.+?)\s+\([^)]*\)\s+--\s+weight:\s*"
    r"([0-9.eE+-]+)\s*$"
)
_SCENARIO_TARGET_RE = re.compile(
    r"^\s*-\s+(.+?)\s+\([^)]*\):\s+loss=([0-9.eE+-]+)\s*$"
)
_CATEGORY_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")


class EvaluationCase:
    """One frozen typed evidence bundle plus labels kept outside evidence."""

    def __init__(
        self,
        evidence: EvidenceBundle,
        cached_primary_output: dict,
        truly_stressed: frozenset[str],
        positive_loss_protocols: frozenset[str] | None = None,
    ) -> None:
        self.evidence = evidence
        self.cached_primary_output = cached_primary_output
        self.truly_stressed = truly_stressed
        self.positive_loss_protocols = positive_loss_protocols or truly_stressed



def normalize_protocol(name: object) -> str:
    if not isinstance(name, str):
        return ""
    return _CATEGORY_SUFFIX_RE.sub("", name.strip()).strip()


def evidence_hash(
    system_prompt: str,
    user_prompt: str,
    candidate_set: CandidateSet | None = None,
) -> str:
    return _canonical_evidence_hash(system_prompt, user_prompt, candidate_set)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_decision_output(
    output: object,
    *,
    candidate_set: CandidateSet | None = None,
    allow_historical_count: bool = False,
) -> None:
    """Reject malformed cached outputs before replaying experimental policies."""
    if not isinstance(output, dict):
        raise ValueError("cached LLM output must be an object")
    if output.get("risk_level") not in ALLOWED_RISK_LEVELS:
        raise ValueError("cached LLM output has an invalid risk_level")
    if not isinstance(output.get("rationale"), str) or not output["rationale"].strip():
        raise ValueError("cached LLM output has no rationale")
    targets = output.get("target_protocols")
    if not isinstance(targets, list):
        raise ValueError("cached LLM target_protocols must be a list")
    seen: set[str] = set()
    previous_score = float("inf")
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise ValueError(f"target_protocols[{index}] must be an object")
        protocol = normalize_protocol(target.get("protocol"))
        score = target.get("risk_score")
        if not protocol or protocol in seen:
            raise ValueError(f"target_protocols[{index}] has an invalid protocol")
        seen.add(protocol)
        if candidate_set is not None and protocol not in {
            candidate for candidate, _, _, _ in candidate_set.rows
        }:
            raise ValueError(
                f"target_protocols[{index}] is outside the CandidateSet"
            )
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not np.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or float(score) > previous_score
        ):
            raise ValueError(f"target_protocols[{index}] has an invalid risk_score")
        previous_score = float(score)
        if target.get("action") not in ALLOWED_ACTIONS:
            raise ValueError(f"target_protocols[{index}] has an invalid action")
        if not isinstance(target.get("reason"), str) or not target["reason"].strip():
            raise ValueError(f"target_protocols[{index}] has no reason")
    if not allow_historical_count and len(targets) != 7:
        raise ValueError("cached LLM output must contain exactly 7 unique targets")


def load_cases(
    raw_path: Path,
    *,
    allow_historical_count: bool = False,
) -> list[EvaluationCase]:
    raw = json.loads(raw_path.read_text())
    cases: list[EvaluationCase] = []
    seen_dates: set[str] = set()
    for entry in raw:
        source_method = entry.get("method")
        if source_method is not None and source_method != "m6_fm_llm":
            raise ValueError(
                "strict replay requires a raw m6_fm_llm artifact; "
                f"found method={source_method!r}"
            )
        date = str(entry["date"])
        if date in seen_dates:
            raise ValueError(f"duplicate date in raw artifact: {date}")
        seen_dates.add(date)
        system_prompt = str(entry["system_prompt"])
        user_prompt = str(entry["user_prompt"])
        outputs = entry.get("raw_llm_outputs", entry.get("llm_outputs", []))
        if not outputs:
            raise ValueError(f"{date}: missing cached LLM outputs")
        evidence_payload = entry.get("evidence_bundle")
        candidate_set = None
        evidence_sha256 = ""
        if isinstance(evidence_payload, dict):
            candidate_rows = evidence_payload.get("candidate_set", [])
            if not isinstance(candidate_rows, list):
                raise ValueError(f"{date}: typed candidate_set must be a list")
            candidate_set = CandidateSet.from_rows(candidate_rows)
            if evidence_payload.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
                raise ValueError(f"{date}: candidate_set hash mismatch")
            recorded_membership = evidence_payload.get("candidate_membership_sha256")
            if (
                recorded_membership is not None
                and recorded_membership != candidate_set.candidate_membership_sha256
            ):
                raise ValueError(f"{date}: candidate membership hash mismatch")
            evidence_sha256 = str(evidence_payload.get("evidence_sha256", ""))
        elif "candidate_set" in entry:
            candidate_set = CandidateSet.from_rows(entry["candidate_set"])
        if not candidate_set:
            # Legacy artifacts are accepted only as an explicitly marked
            # fallback. New artifacts use the typed candidate set above.
            exposure, scenario_losses = parse_protocol_evidence(user_prompt)
            candidate_set = CandidateSet(
                tuple(
                    (
                        protocol,
                        exposure.get(protocol, 0.0),
                        max(scenario_losses.get(protocol, [0.0])),
                        sum(scenario_losses.get(protocol, [])),
                    )
                    for protocol in sorted(set(exposure) | set(scenario_losses))
                )
            )
        expected_evidence_hash = evidence_hash(
            system_prompt, user_prompt, candidate_set
        )
        if evidence_sha256 and evidence_sha256 != expected_evidence_hash:
            raise ValueError(f"{date}: EvidenceBundle hash mismatch")
        validate_decision_output(
            outputs[0],
            candidate_set=candidate_set,
            allow_historical_count=allow_historical_count,
        )
        cases.append(
            EvaluationCase(
                evidence=EvidenceBundle(
                    date=date,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    evidence_sha256=expected_evidence_hash,
                    candidate_set=candidate_set,
                ),
                cached_primary_output=deepcopy(outputs[0]),
                truly_stressed=frozenset(
                    normalize_protocol(name)
                    for name in entry.get("truly_stressed", [])
                    if normalize_protocol(name)
                ),
                positive_loss_protocols=frozenset(
                    normalize_protocol(name)
                    for name in entry.get(
                        "positive_loss_protocols",
                        [
                            name
                            for name, loss in entry.get("actual_losses", {}).items()
                            if float(loss or 0.0) > 0.0
                        ],
                    )
                    if normalize_protocol(name)
                ),
            )
        )
        if not cases[-1].truly_stressed:
            raise ValueError(f"{date}: empty stressed target set")
    return cases


def bundles_from_cases(cases: list[EvaluationCase]) -> list[EvidenceBundle]:
    return [case.evidence for case in cases]


def parse_protocol_evidence(
    user_prompt: str,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Extract exactly the protocol-level values visible to the LLM."""
    exposure: dict[str, float] = {}
    scenario_losses: dict[str, list[float]] = {}
    for line in user_prompt.splitlines():
        match = _TOP_PROTOCOL_RE.match(line)
        if match:
            protocol = normalize_protocol(match.group(1))
            if protocol:
                exposure[protocol] = float(match.group(2))
            continue
        match = _SCENARIO_TARGET_RE.match(line)
        if match:
            protocol = normalize_protocol(match.group(1))
            if protocol:
                scenario_losses.setdefault(protocol, []).append(
                    abs(float(match.group(2)))
                )
    if not exposure and not scenario_losses:
        raise ValueError("prompt contains no parseable protocol-level evidence")
    return exposure, scenario_losses


def rule_targets(bundle: EvidenceBundle, policy: str) -> list[dict]:
    if bundle.candidate_set is not None:
        exposure = {
            protocol: exposure
            for protocol, exposure, _, _ in bundle.candidate_set.rows
        }
        scenario_losses = {
            protocol: [max_loss] if max_loss else []
            for protocol, _, max_loss, _ in bundle.candidate_set.rows
        }
        scenario_sums = {
            protocol: sum_loss
            for protocol, _, _, sum_loss in bundle.candidate_set.rows
        }
    else:
        exposure, scenario_losses = parse_protocol_evidence(bundle.user_prompt)
        scenario_sums = {
            protocol: sum(values)
            for protocol, values in scenario_losses.items()
        }
    candidates = set(exposure) | set(scenario_losses)
    if policy == "exposure_rules":
        ranked = sorted(candidates, key=lambda p: (-exposure.get(p, 0.0), p))
    elif policy == "scenario_max_rules":
        ranked = sorted(
            candidates,
            key=lambda p: (
                -max(scenario_losses.get(p) or [0.0]),
                -exposure.get(p, 0.0),
                p,
            ),
        )
    elif policy == "scenario_sum_rules":
        ranked = sorted(
            candidates,
            key=lambda p: (
                -scenario_sums.get(p, 0.0),
                -exposure.get(p, 0.0),
                p,
            ),
        )
    else:
        raise ValueError(f"unknown rule policy: {policy}")
    return [
        {
            "protocol": protocol,
            "risk_score": float(len(ranked) - index) / max(1, len(ranked)),
            "action": "Investigate",
        }
        for index, protocol in enumerate(ranked)
    ]


def cached_llm_output(cached_primary_output: dict) -> dict:
    return deepcopy(cached_primary_output)


def apply_action_gate(llm_output: dict) -> tuple[dict, int]:
    """Apply the exact current M7 gate to a deep copy of raw tickets."""
    raw_actions = [
        target.get("action")
        for target in llm_output.get("target_protocols", [])
        if isinstance(target, dict)
    ]
    gated = _canonical_action_gate(llm_output)
    gated_actions = [
        target.get("action")
        for target in gated.get("target_protocols", [])
        if isinstance(target, dict)
    ]
    demotions = sum(
        before in INTERVENTION_ACTIONS and after == "Investigate"
        for before, after in zip(raw_actions, gated_actions)
    )
    return gated, demotions


def ranked_unique_targets(output: dict) -> list[dict]:
    normalized_output = deepcopy(output)
    if isinstance(normalized_output, dict) and isinstance(
        normalized_output.get("target_protocols"), list
    ):
        normalized_output["target_protocols"] = [
            {
                **target,
                "protocol": normalize_protocol(target.get("protocol", "")),
            }
            for target in normalized_output["target_protocols"]
            if isinstance(target, dict)
        ]
    ranked = _canonical_ranked_unique_targets(normalized_output)
    for target in ranked:
        target["protocol"] = normalize_protocol(target.get("protocol", ""))
    return ranked


def policy_output(
    evidence: EvidenceBundle, cached_primary_output: dict, policy: str
) -> tuple[dict, int]:
    if policy in {"exposure_rules", "scenario_max_rules", "scenario_sum_rules"}:
        return {
            "risk_level": "moderate",
            "target_protocols": rule_targets(evidence, policy),
        }, 0
    if policy == "llm_raw":
        return cached_llm_output(cached_primary_output), 0
    if policy == "llm_gated_replay":
        return apply_action_gate(cached_llm_output(cached_primary_output))
    raise ValueError(f"unknown policy: {policy}")


def evaluate_policy(
    cases: list[EvaluationCase], policy: str, budget: int
) -> tuple[dict, np.ndarray]:
    weekly_coverage: list[float] = []
    weekly_precision: list[float] = []
    selected_count = 0
    hit_count = 0
    intervention_count = 0
    false_intervention_count = 0
    non_loss_false_intervention_count = 0
    intervention_weeks = 0
    active_week_firs: list[float] = []
    demotions = 0

    for case in cases:
        output, output_demotions = policy_output(
            case.evidence, case.cached_primary_output, policy
        )
        demotions += output_demotions
        selected = require_exact_k_targets(ranked_unique_targets(output), budget)
        stressed = case.truly_stressed
        hits = sum(target["protocol"] in stressed for target in selected)
        selected_count += len(selected)
        hit_count += hits
        if not stressed:
            raise ValueError(f"{case.evidence.date}: empty stressed target set")
        weekly_coverage.append(hits / len(stressed))
        weekly_precision.append(hits / len(selected) if selected else 0.0)

        interventions = [
            target
            for target in selected
            if target.get("action") in INTERVENTION_ACTIONS
        ]
        false_interventions = [
            target for target in interventions if target["protocol"] not in stressed
        ]
        positive_loss = case.positive_loss_protocols
        non_loss_false_interventions = [
            target
            for target in interventions
            if target["protocol"] not in positive_loss
        ]
        intervention_count += len(interventions)
        false_intervention_count += len(false_interventions)
        non_loss_false_intervention_count += len(non_loss_false_interventions)
        if interventions:
            intervention_weeks += 1
            active_week_firs.append(len(false_interventions) / len(interventions))

    n_weeks = len(cases)
    metrics = {
        "policy": policy,
        "budget": budget,
        "n_weeks": n_weeks,
        "coverage_at_k": float(np.mean(weekly_coverage)),
        "precision_macro_at_k": float(np.mean(weekly_precision)),
        "precision_micro_at_k": hit_count / selected_count if selected_count else None,
        "selected_targets": selected_count,
        "correct_targets": hit_count,
        "budget_utilization": selected_count / (n_weeks * budget),
        "intervention_targets": intervention_count,
        "intervention_targets_per_week": intervention_count / n_weeks,
        "intervention_weeks": intervention_weeks,
        "false_interventions": false_intervention_count,
        "non_loss_false_interventions": non_loss_false_intervention_count,
        "false_interventions_per_week": false_intervention_count / n_weeks,
        "fir_micro": (
            false_intervention_count / intervention_count
            if intervention_count
            else None
        ),
        "tail_fir_micro": (
            false_intervention_count / intervention_count
            if intervention_count
            else None
        ),
        "non_loss_fir_micro": (
            non_loss_false_intervention_count / intervention_count
            if intervention_count
            else None
        ),
        "fir_active_week_macro": (
            float(np.mean(active_week_firs)) if active_week_firs else None
        ),
        "gate_demotions": demotions,
    }
    return metrics, np.asarray(weekly_coverage, dtype=float)


def circular_block_bootstrap_delta(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
) -> dict:
    if len(values_a) != len(values_b):
        raise ValueError("paired bootstrap arrays must have equal length")
    delta = values_a - values_b
    n = len(delta)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    n_blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    for index in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        selected = np.concatenate([((start + offsets) % n) for start in starts])[:n]
        samples[index] = float(np.mean(delta[selected]))
    observed = float(np.mean(delta))
    null_samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        offset = int(rng.integers(0, n))
        rotated = np.roll(delta, -offset)
        signs = rng.choice((-1.0, 1.0), size=n_blocks)
        signed = rotated.copy()
        for block in range(n_blocks):
            start = block * block_length
            stop = min(n, start + block_length)
            signed[start:stop] *= signs[block]
        null_samples[index] = float(np.mean(signed))
    p_two_sided = (
        np.count_nonzero(np.abs(null_samples) >= abs(observed)) + 1
    ) / (n_bootstrap + 1)
    return {
        "mean_delta": observed,
        "ci_95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "p_two_sided": p_two_sided,
        "block_length": block_length,
        "n_bootstrap": n_bootstrap,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def token_cost(
    raw: list[dict], model: str, runs: int, rates: dict[str, float] | None = None
) -> dict:
    input_tokens = 0
    output_tokens = 0
    unknown_usage_attempts = 0
    for entry in raw:
        outputs = entry.get("raw_llm_outputs", entry.get("llm_outputs", []))
        attempts = entry.get("llm_attempts")
        if isinstance(attempts, list) and attempts:
            if all("consistency_run" in attempt for attempt in attempts):
                selected_attempts = [
                    attempt
                    for attempt in attempts
                    if int(attempt["consistency_run"]) < runs
                ]
            else:
                selected_attempts = attempts[: runs * 3]
            for attempt in selected_attempts:
                for request_attempt in billable_request_attempts(attempt):
                    if request_attempt.get("usage_known") is not True:
                        unknown_usage_attempts += 1
                        continue
                    input_tokens += int(request_attempt.get("input_tokens", 0) or 0)
                    output_tokens += int(request_attempt.get("output_tokens", 0) or 0)
            continue
        for output in outputs[:runs]:
            for request_attempt in billable_request_attempts(output):
                if request_attempt.get("usage_known") is not True:
                    unknown_usage_attempts += 1
                    continue
                input_tokens += int(request_attempt.get("input_tokens", 0) or 0)
                output_tokens += int(request_attempt.get("output_tokens", 0) or 0)
    known_cost = None
    if rates:
        known_cost = (
            input_tokens * rates["input"] + output_tokens * rates["output"]
        ) / 1_000_000
    return {
        "model": model,
        "runs_per_week": runs,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "known_cost_usd": known_cost,
        "cost_usd": None if unknown_usage_attempts else known_cost,
        "unknown_usage_attempts": unknown_usage_attempts,
        "rate_usd_per_million_tokens": rates,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_replay(
    raw_path: Path,
    summary_path: Path,
    output_dir: Path,
    budgets: tuple[int, ...],
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
    allow_historical_count: bool = False,
) -> dict:
    if tuple(sorted(budgets)) != DEFAULT_BUDGETS:
        raise ValueError(
            "coverage frontier requires exactly K={1,3,5,7}; K=5 is primary"
        )
    if block_length <= 0 or n_bootstrap <= 0:
        raise ValueError("block_length and n_bootstrap must be positive")
    if not raw_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("raw and summary artifacts must both exist")
    cases = load_cases(raw_path, allow_historical_count=allow_historical_count)
    if not cases:
        raise ValueError("raw artifact contains no evaluation entries")
    bundles = bundles_from_cases(cases)
    raw = json.loads(raw_path.read_text())
    summary_payload = json.loads(summary_path.read_text())
    if summary_payload.get("method") not in {None, "m6_fm_llm"}:
        raise ValueError(
            "strict replay requires the m6_fm_llm summary artifact; "
            f"found method={summary_payload.get('method')!r}"
        )
    model = str(summary_payload.get("model", "unknown"))
    recorded_rates = summary_payload.get("decision_rate_usd_per_million_tokens")
    if not (
        isinstance(recorded_rates, dict)
        and set(recorded_rates) == {"input", "output"}
        and all(
            isinstance(recorded_rates[key], (int, float))
            and not isinstance(recorded_rates[key], bool)
            and np.isfinite(float(recorded_rates[key]))
            and float(recorded_rates[key]) >= 0.0
            for key in ("input", "output")
        )
    ):
        recorded_rates = None

    policies = (
        "exposure_rules",
        "scenario_max_rules",
        "scenario_sum_rules",
        "llm_raw",
        "llm_gated_replay",
    )
    rows: list[dict] = []
    coverage: dict[tuple[str, int], np.ndarray] = {}
    for policy in policies:
        for budget in budgets:
            metrics, weekly_coverage = evaluate_policy(cases, policy, budget)
            rows.append(metrics)
            coverage[(policy, budget)] = weekly_coverage

    comparisons: list[dict] = []
    for baseline in ("exposure_rules", "scenario_max_rules", "scenario_sum_rules"):
        family: list[dict] = []
        for budget in budgets:
            test = circular_block_bootstrap_delta(
                coverage[("llm_raw", budget)],
                coverage[(baseline, budget)],
                block_length=block_length,
                n_bootstrap=n_bootstrap,
                seed=seed + budget,
            )
            family.append(
                {
                    "comparison": f"llm_raw_minus_{baseline}",
                    "budget": budget,
                    **test,
                }
            )
        comparisons.extend(family)

    adjusted = holm_adjust([test["p_two_sided"] for test in comparisons])
    for test, corrected in zip(comparisons, adjusted, strict=True):
        test["p_holm_across_all_rule_baselines_and_budgets"] = corrected

    source_hash = file_sha256(raw_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = {
        "schema_version": 1,
        "source_raw": str(raw_path.resolve()),
        "source_sha256": source_hash,
        "cases": [
            {
                "evidence": {
                    "date": case.evidence.date,
                    "evidence_sha256": case.evidence.evidence_sha256,
                    "candidate_set_sha256": case.evidence.candidate_set_sha256,
                    "candidate_membership_sha256": (
                        case.evidence.candidate_membership_sha256
                    ),
                    "candidate_set": (
                        case.evidence.candidate_set.as_dicts()
                        if case.evidence.candidate_set is not None
                        else None
                    ),
                    "system_prompt": case.evidence.system_prompt,
                    "user_prompt": case.evidence.user_prompt,
                },
                "evaluation_label": {
                    "truly_stressed": sorted(case.truly_stressed),
                },
            }
            for case in cases
        ],
    }
    (output_dir / "frozen_evidence.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False)
    )

    cost = {
        "shared_fm_cost": "excluded; identical frozen evidence for all policies",
        "rules_incremental_api_cost_usd": 0.0,
        "gate_incremental_api_cost_usd": 0.0,
        "llm_primary": token_cost(raw, model, 1, recorded_rates),
        "llm_three_run_evaluation": token_cost(raw, model, 3, recorded_rates),
        "price_source": (
            summary_payload.get("rate_card_evidence")
            if recorded_rates is not None
            else "missing from source summary; USD cost is not identified"
        ),
    }
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "n_weeks": len(cases),
            "budgets": list(budgets),
            "primary_budget": 5,
            "budget_unit": "exactly K unique protocol targets per week",
            "canonical_evidence_source": "saved m6_fm_llm prompt",
            "source_sha256": source_hash,
            "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "evidence_hashes_unique": len({bundle.evidence_sha256 for bundle in bundles}),
            "candidate_set_hashes_unique": len(
                {
                    bundle.candidate_set_sha256
                    for bundle in bundles
                    if bundle.candidate_set_sha256
                }
            ),
            "same_input_enforcement": (
                "all policies receive the same EvidenceBundle; future labels "
                "remain in the separate EvaluationCase"
            ),
            "bootstrap": {
                "type": "paired circular moving-block",
                "block_length": block_length,
                "n_bootstrap": n_bootstrap,
                "seed": seed,
            },
        },
        "rows": rows,
        "comparisons": comparisons,
        "cost": cost,
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    write_csv(output_dir / "frontier.csv", rows)
    return result


def print_summary(result: dict) -> None:
    rows = result["rows"]
    print(
        f"{'policy':22s} {'K':>2s} {'coverage':>10s} {'Pmicro':>8s} "
        f"{'used':>7s} {'int/w':>7s} {'false/w':>8s} {'FIR':>7s}"
    )
    for row in rows:
        fir = "n/a" if row["fir_micro"] is None else f"{row['fir_micro']:.3f}"
        print(
            f"{row['policy']:22s} {row['budget']:2d} "
            f"{row['coverage_at_k']:10.6f} "
            f"{row['precision_micro_at_k']:8.3f} "
            f"{row['budget_utilization']:7.3f} "
            f"{row['intervention_targets_per_week']:7.2f} "
            f"{row['false_interventions_per_week']:8.2f} {fir:>7s}"
        )
    print("\nLLM minus same-evidence rules, paired block bootstrap:")
    for test in result["comparisons"]:
        ci = test["ci_95"]
        print(
            f"  {test['comparison']:37s} K={test['budget']}: "
            f"delta={test['mean_delta']:+.6f} "
            f"CI=[{ci[0]:+.6f},{ci[1]:+.6f}] "
            "p_holm_all="
            f"{test['p_holm_across_all_rule_baselines_and_budgets']:.4f}"
        )
    primary = result["cost"]["llm_primary"]
    evaluation = result["cost"]["llm_three_run_evaluation"]
    primary_cost = "unknown" if primary["cost_usd"] is None else f"${primary['cost_usd']:.4f}"
    evaluation_cost = (
        "unknown"
        if evaluation["cost_usd"] is None
        else f"${evaluation['cost_usd']:.4f}"
    )
    print(
        f"\nLLM decision cost: primary={primary_cost}; "
        f"three-run evaluation={evaluation_cost}; "
        f"unknown usage attempts={evaluation['unknown_usage_attempts']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True,
                        help="Corrected frozen raw_m6_fm_llm.json artifact")
    parser.add_argument("--summary", type=Path, required=True,
                        help="Summary artifact generated by the same corrected run")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--budgets", default="1,3,5,7")
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--allow-historical-count",
        action="store_true",
        help="Diagnostic-only compatibility for pre-exact-7 cached outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    if not budgets or any(value <= 0 for value in budgets):
        raise ValueError("budgets must be positive comma-separated integers")
    result = run_replay(
        args.raw.resolve(),
        args.summary.resolve(),
        args.output_dir.resolve(),
        budgets,
        block_length=args.block_length,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        allow_historical_count=args.allow_historical_count,
    )
    print_summary(result)
    print(f"\nWrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
