#!/usr/bin/env python3
"""Evaluate exact-K LLM decisions on frozen DeXposure EvidenceBundles.

This runner never invokes the FM. It rewrites only the historical target-count
instruction, hashes the resulting prompt, calls one fixed decision model over
multiple requested seeds, and compares the outputs with deterministic rules
and the saved M5 (FM + Rules) weekly targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper"
if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))

from experiments.b5_decision_quality import (  # noqa: E402
    CandidateSet,
    EvidenceBundle,
    require_exact_k_targets,
)
from scripts.replay_coverage_intervention_frontier import (  # noqa: E402
    circular_block_bootstrap_delta,
    holm_adjust,
    rule_targets,
)
from scripts.run_harness_control_ablation import (  # noqa: E402
    api_model_slug,
    atomic_write_json,
    call_llm,
    satisfies_ticket_contract,
    sha256_text,
    target_rows,
    trace_status,
)


INTERVENTION_ACTIONS = {"Recommend-Reduce", "Contingency"}
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_SEEDS = (20260806, 20260807, 20260808)
DEFAULT_PROVIDER_POLICY = {
    "only": ["anthropic"],
    "allow_fallbacks": False,
}
_TARGET_COUNT_RE = re.compile(
    r"Return exactly\s+\d+\s+unique target_protocols, including Monitor rows "
    r"when fewer\s+than\s+\d+\s+protocols warrant escalation\."
)
_WRITE_LOCK = threading.Lock()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_target_budget(system_prompt: str, budget: int) -> str:
    """Set the exact target-count instruction and fail on prompt drift."""
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("target budget must be a positive integer")
    replacement = (
        f"Return exactly {budget} unique target_protocols, including Monitor "
        f"rows when fewer than {budget} protocols warrant escalation."
    )
    rewritten, count = _TARGET_COUNT_RE.subn(replacement, system_prompt)
    if count == 1:
        return rewritten
    if count > 1 or "Return exactly" in system_prompt:
        raise ValueError(
            "system prompt must contain exactly one target-count instruction"
        )
    marker = "- Order target_protocols by risk_score descending."
    if system_prompt.count(marker) != 1:
        raise ValueError(
            "system prompt has neither a target-count instruction nor the "
            "expected target-order marker"
        )
    return system_prompt.replace(marker, marker + "\n- " + replacement)


def _validated_candidate_set(evidence: dict[str, Any]) -> CandidateSet:
    rows = evidence.get("candidate_set")
    if not isinstance(rows, list) or not rows:
        raise ValueError("frozen EvidenceBundle has no typed CandidateSet")
    candidate_set = CandidateSet.from_rows(rows)
    if evidence.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
        raise ValueError("frozen CandidateSet hash mismatch")
    if (
        evidence.get("candidate_membership_sha256")
        != candidate_set.candidate_membership_sha256
    ):
        raise ValueError("frozen CandidateSet membership hash mismatch")
    return candidate_set


def load_cases(
    frozen_path: Path, *, target_budget: int, limit: int | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(frozen_path.read_text())
    cases_payload = payload.get("cases")
    if not isinstance(cases_payload, list) or not cases_payload:
        raise ValueError("frozen artifact has no cases")
    source_raw = Path(str(payload.get("source_raw", "")))
    if not source_raw.is_file():
        raise FileNotFoundError("frozen artifact source_raw is unavailable")
    if file_sha256(source_raw) != payload.get("source_sha256"):
        raise ValueError("frozen artifact source_raw hash mismatch")
    source_rows = json.loads(source_raw.read_text())
    source_by_date = {str(row["date"]): row for row in source_rows}

    cases: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    for case_payload in cases_payload[:limit]:
        evidence = case_payload.get("evidence")
        labels = case_payload.get("evaluation_label")
        if not isinstance(evidence, dict) or not isinstance(labels, dict):
            raise ValueError("frozen case must separate evidence and labels")
        date = str(evidence.get("date", ""))
        if not date or date in seen_dates:
            raise ValueError("frozen case date is empty or duplicated")
        seen_dates.add(date)
        candidate_set = _validated_candidate_set(evidence)
        original = EvidenceBundle(
            date=date,
            system_prompt=str(evidence.get("system_prompt", "")),
            user_prompt=str(evidence.get("user_prompt", "")),
            candidate_set=candidate_set,
        )
        if original.evidence_sha256 != evidence.get("evidence_sha256"):
            raise ValueError(f"{date}: frozen EvidenceBundle hash mismatch")
        system_prompt = rewrite_target_budget(
            original.system_prompt, target_budget
        )
        rendered = EvidenceBundle(
            date=date,
            system_prompt=system_prompt,
            user_prompt=original.user_prompt,
            candidate_set=candidate_set,
        )
        stressed = {
            str(value).strip()
            for value in labels.get("truly_stressed", [])
            if str(value).strip()
        }
        if not stressed:
            raise ValueError(f"{date}: evaluation label is empty")
        source = source_by_date.get(date, {})
        actual_losses = source.get("actual_losses", {})
        positive_loss = {
            str(name).strip()
            for name, loss in actual_losses.items()
            if str(name).strip() and float(loss or 0.0) > 0.0
        }
        cases.append(
            {
                "date": date,
                "source_evidence_sha256": original.evidence_sha256,
                "rendered_evidence_sha256": rendered.evidence_sha256,
                "candidate_set": candidate_set,
                "system_prompt": system_prompt,
                "user_prompt": original.user_prompt,
                "truly_stressed": stressed,
                "positive_loss_protocols": positive_loss,
            }
        )
    return cases, payload


def request_identity(case: dict[str, Any], config: dict[str, Any], seed: int) -> str:
    return sha256_text(
        json.dumps(
            {
                "rendered_evidence_sha256": case["rendered_evidence_sha256"],
                "request_config": config,
                "seed": seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def validate_record(
    record: dict[str, Any], case: dict[str, Any], target_budget: int
) -> bool:
    if record.get("status") != "ok":
        return False
    parsed = record.get("parsed")
    valid = satisfies_ticket_contract(
        parsed,
        case["candidate_set"],
        expected_count=target_budget,
    )
    if not valid:
        return False
    return len(require_exact_k_targets(target_rows(parsed), target_budget)) == target_budget


def run_calls(
    *,
    cases: list[dict[str, Any]],
    output_dir: Path,
    api_key: str,
    api_url: str,
    model: str,
    target_budget: int,
    temperature: float,
    seeds: tuple[int, ...],
    workers: int,
    timeout: int,
) -> list[dict[str, Any]]:
    raw_path = output_dir / "raw_requests.json"
    records = json.loads(raw_path.read_text()) if raw_path.exists() else []
    config = {
        "api_url": api_url,
        "model": api_model_slug(model),
        "target_budget": target_budget,
        "temperature": temperature,
        "max_output_tokens": 4096,
        "provider_policy": DEFAULT_PROVIDER_POLICY,
    }
    by_date = {case["date"]: case for case in cases}
    complete = {
        (str(row["date"]), int(row["run_index"]), str(row["request_identity_sha256"]))
        for row in records
        if str(row.get("date")) in by_date
        and validate_record(row, by_date[str(row["date"])], target_budget)
    }
    tasks: list[tuple[dict[str, Any], int, int, str]] = []
    for case in cases:
        for run_index, seed in enumerate(seeds):
            identity = request_identity(case, config, seed)
            if (case["date"], run_index, identity) not in complete:
                tasks.append((case, run_index, seed, identity))
    random.Random(seeds[0]).shuffle(tasks)

    def execute(task: tuple[dict[str, Any], int, int, str]) -> dict[str, Any]:
        case, run_index, seed, identity = task
        response = call_llm(
            api_url=api_url,
            api_key=api_key,
            model=model,
            system_prompt=case["system_prompt"],
            user_prompt=case["user_prompt"],
            temperature=temperature,
            timeout=timeout,
            seed=seed,
            provider=DEFAULT_PROVIDER_POLICY,
        )
        record = {
            "date": case["date"],
            "run_index": run_index,
            "requested_seed": seed,
            "source_evidence_sha256": case["source_evidence_sha256"],
            "rendered_evidence_sha256": case["rendered_evidence_sha256"],
            "candidate_set_sha256": case["candidate_set"].candidate_set_sha256,
            "candidate_membership_sha256": (
                case["candidate_set"].candidate_membership_sha256
            ),
            "system_prompt_sha256": sha256_text(case["system_prompt"]),
            "user_prompt_sha256": sha256_text(case["user_prompt"]),
            "request_config": config,
            "request_identity_sha256": identity,
            **response,
        }
        record["contract_valid"] = validate_record(
            record, case, target_budget
        )
        return record

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        for future in as_completed(futures):
            record = future.result()
            with _WRITE_LOCK:
                current = json.loads(raw_path.read_text()) if raw_path.exists() else []
                key = (record["date"], record["run_index"])
                current = [
                    row
                    for row in current
                    if (row.get("date"), row.get("run_index")) != key
                ]
                current.append(record)
                current.sort(key=lambda row: (row["date"], row["run_index"]))
                atomic_write_json(raw_path, current)
            print(
                f"{record['date']} run={record['run_index']} "
                f"status={record['status']} contract={record['contract_valid']} "
                f"latency_ms={record.get('latency_ms')}"
            )
    records = json.loads(raw_path.read_text())
    expected = {(case["date"], index) for case in cases for index in range(len(seeds))}
    selected = [
        row
        for row in records
        if (str(row.get("date")), int(row.get("run_index", -1))) in expected
    ]
    if len(selected) != len(expected):
        raise ValueError("date/run request grid is incomplete")
    failures = [
        (row["date"], row["run_index"], row.get("status"))
        for row in selected
        if not validate_record(row, by_date[str(row["date"])], target_budget)
    ]
    if failures:
        raise ValueError(f"contract-valid exact-K requests required; failures={failures}")
    return selected


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def evaluate_llm(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    target_budget: int,
    rates: dict[str, float],
) -> tuple[dict[str, Any], dict[str, float]]:
    case_by_date = {case["date"]: case for case in cases}
    coverage_by_date: dict[str, list[float]] = {case["date"]: [] for case in cases}
    target_sets_by_date: dict[str, list[set[str]]] = {
        case["date"]: [] for case in cases
    }
    selected_count = hits = interventions = false_interventions = 0
    non_loss_false = input_tokens = output_tokens = unknown_usage = 0
    protocol_trace = numeric_trace = traced_targets = 0
    latencies: list[float] = []
    reported_costs: list[float] = []
    providers: Counter[str] = Counter()
    for record in sorted(records, key=lambda row: (row["date"], row["run_index"])):
        case = case_by_date[str(record["date"])]
        selected = require_exact_k_targets(
            target_rows(record["parsed"]), target_budget
        )
        protocols = {target["protocol"] for target in selected}
        stressed = case["truly_stressed"]
        row_hits = len(protocols & stressed)
        selected_count += len(selected)
        hits += row_hits
        coverage_by_date[case["date"]].append(row_hits / len(stressed))
        target_sets_by_date[case["date"]].append(protocols)
        for target in selected:
            status = trace_status(target, case["user_prompt"])
            traced_targets += 1
            protocol_trace += int(status["protocol_visible"])
            numeric_trace += int(status["numeric_trace"])
            if target.get("action") in INTERVENTION_ACTIONS:
                interventions += 1
                false_interventions += int(target["protocol"] not in stressed)
                non_loss_false += int(
                    target["protocol"] not in case["positive_loss_protocols"]
                )
        attempts = record.get("attempts") or [record]
        for attempt in attempts:
            unknown_usage += int(attempt.get("usage_known") is not True)
            input_tokens += int(attempt.get("input_tokens", 0) or 0)
            output_tokens += int(attempt.get("output_tokens", 0) or 0)
        if record.get("latency_ms") is not None:
            latencies.append(float(record["latency_ms"]))
        providers[str(record.get("response_provider") or "unknown")] += 1
        usage = record.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("cost"), (int, float)):
            reported_costs.append(float(usage["cost"]))
    n = len(records)
    computed_cost = (
        input_tokens * rates["input"] + output_tokens * rates["output"]
    ) / 1_000_000
    consistency_values = [
        _jaccard(left, right)
        for sets in target_sets_by_date.values()
        for left, right in combinations(sets, 2)
    ]
    metrics = {
        "policy": "m6_fm_llm_exact_k",
        "target_budget": target_budget,
        "n_weeks": len(cases),
        "runs_per_week": n // len(cases),
        "n_week_runs": n,
        "contract_success_rate": 1.0,
        "coverage_at_k": float(
            np.mean([value for values in coverage_by_date.values() for value in values])
        ),
        "precision_micro_at_k": hits / selected_count,
        "selected_targets": selected_count,
        "correct_targets": hits,
        "intervention_targets": interventions,
        "intervention_targets_per_week_run": interventions / n,
        "false_interventions": false_interventions,
        "false_interventions_per_week_run": false_interventions / n,
        "fir_micro": (
            false_interventions / interventions if interventions else None
        ),
        "non_loss_fir_micro": (
            non_loss_false / interventions if interventions else None
        ),
        "mean_pairwise_target_jaccard": (
            float(np.mean(consistency_values)) if consistency_values else None
        ),
        "protocol_trace_rate": protocol_trace / traced_targets,
        "numeric_trace_rate": numeric_trace / traced_targets,
        "mean_latency_ms": float(np.mean(latencies)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "unknown_usage_attempts": unknown_usage,
        "known_computed_cost_usd": computed_cost,
        "computed_cost_usd": None if unknown_usage else computed_cost,
        "provider_reported_cost_usd": (
            sum(reported_costs) if len(reported_costs) == n else None
        ),
        "response_provider_counts": dict(providers),
        "unique_response_ids": len(
            {record.get("response_id") for record in records if record.get("response_id")}
        ),
    }
    weekly = {
        date: float(np.mean(values)) for date, values in coverage_by_date.items()
    }
    return metrics, weekly


def evaluate_rule(
    cases: list[dict[str, Any]], policy: str, target_budget: int
) -> tuple[dict[str, Any], dict[str, float]]:
    selected_count = hits = 0
    weekly: dict[str, float] = {}
    for case in cases:
        evidence = EvidenceBundle(
            date=case["date"],
            system_prompt=case["system_prompt"],
            user_prompt=case["user_prompt"],
            candidate_set=case["candidate_set"],
        )
        selected = require_exact_k_targets(
            rule_targets(evidence, policy), target_budget
        )
        row_hits = sum(
            target["protocol"] in case["truly_stressed"] for target in selected
        )
        selected_count += len(selected)
        hits += row_hits
        weekly[case["date"]] = row_hits / len(case["truly_stressed"])
    return {
        "policy": policy,
        "target_budget": target_budget,
        "n_weeks": len(cases),
        "coverage_at_k": float(np.mean(list(weekly.values()))),
        "precision_micro_at_k": hits / selected_count,
        "selected_targets": selected_count,
        "correct_targets": hits,
        "incremental_api_cost_usd": 0.0,
    }, weekly


def evaluate_m5(
    cases: list[dict[str, Any]], weekly_path: Path, target_budget: int
) -> tuple[dict[str, Any], dict[str, float]]:
    payload = json.loads(weekly_path.read_text())
    by_date = {str(row["date"]): row for row in payload.get("weeks", [])}
    selected_count = hits = interventions = false_count = 0
    weekly: dict[str, float] = {}
    for case in cases:
        row = by_date.get(case["date"])
        if row is None:
            raise ValueError(f"M5 artifact has no week {case['date']}")
        ranked = [
            {"protocol": str(pair[0]).strip(), "risk_score": float(pair[1])}
            for pair in row.get("targets_scored", [])
            if isinstance(pair, list) and len(pair) >= 2 and str(pair[0]).strip()
        ]
        selected = require_exact_k_targets(ranked, target_budget)
        protocols = {target["protocol"] for target in selected}
        row_hits = len(protocols & case["truly_stressed"])
        intervention_set = {
            str(value).strip()
            for value in row.get("intervention_targets", [])
            if str(value).strip()
        }
        selected_interventions = protocols & intervention_set
        selected_count += len(selected)
        hits += row_hits
        interventions += len(selected_interventions)
        false_count += len(selected_interventions - case["truly_stressed"])
        weekly[case["date"]] = row_hits / len(case["truly_stressed"])
    return {
        "policy": "m5_fm_rules",
        "target_budget": target_budget,
        "n_weeks": len(cases),
        "coverage_at_k": float(np.mean(list(weekly.values()))),
        "precision_micro_at_k": hits / selected_count,
        "selected_targets": selected_count,
        "correct_targets": hits,
        "intervention_targets": interventions,
        "false_interventions": false_count,
        "fir_micro": false_count / interventions if interventions else None,
        "incremental_api_cost_usd": 0.0,
    }, weekly


def validate_alignment(
    alignment_path: Path,
    cases: list[dict[str, Any]],
    frozen_payload: dict[str, Any],
) -> dict[str, Any]:
    alignment = json.loads(alignment_path.read_text())
    if alignment.get("raw_sha256") != frozen_payload.get("source_sha256"):
        raise ValueError("M5 evidence alignment raw hash mismatch")
    rows = alignment.get("rows", [])
    by_date = {str(row["date"]): row for row in rows}
    for case in cases:
        row = by_date.get(case["date"])
        if not row or row.get("evidence_match") is not True:
            raise ValueError(f"{case['date']}: M5/M6 evidence is not aligned")
    return {
        "alignment_sha256": file_sha256(alignment_path),
        "matched_weeks": len(cases),
        "checkpoint_dir": alignment.get("checkpoint_dir"),
        "horizon": alignment.get("horizon"),
    }


def analyze(
    *,
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    frozen_payload: dict[str, Any],
    m5_weekly: Path | None,
    alignment: Path | None,
    output_dir: Path,
    model: str,
    target_budget: int,
    temperature: float,
    seeds: tuple[int, ...],
    rates: dict[str, float],
    price_source: str,
) -> dict[str, Any]:
    if (m5_weekly is None) != (alignment is None):
        raise ValueError("M5 weekly results and evidence alignment must be supplied together")
    alignment_result = (
        validate_alignment(alignment, cases, frozen_payload)
        if alignment is not None
        else None
    )
    llm_metrics, llm_weekly = evaluate_llm(
        records, cases, target_budget=target_budget, rates=rates
    )
    per_seed_rows: list[dict[str, Any]] = []
    for run_index, seed in enumerate(seeds):
        seed_records = [
            record
            for record in records
            if int(record["run_index"]) == run_index
        ]
        seed_metrics, _ = evaluate_llm(
            seed_records, cases, target_budget=target_budget, rates=rates
        )
        seed_metrics["requested_seed"] = seed
        per_seed_rows.append(seed_metrics)
    rows = [llm_metrics]
    weekly_by_policy = {"m6_fm_llm_exact_k": llm_weekly}
    for policy in ("exposure_rules", "scenario_max_rules", "scenario_sum_rules"):
        metrics, weekly = evaluate_rule(cases, policy, target_budget)
        rows.append(metrics)
        weekly_by_policy[policy] = weekly
    if m5_weekly is not None:
        m5_metrics, m5_weekly_coverage = evaluate_m5(
            cases, m5_weekly, target_budget
        )
        rows.append(m5_metrics)
        weekly_by_policy["m5_fm_rules"] = m5_weekly_coverage

    dates = [case["date"] for case in cases]
    comparisons: list[dict[str, Any]] = []
    llm_values = np.asarray([llm_weekly[date] for date in dates])
    baselines = ["exposure_rules", "scenario_max_rules", "scenario_sum_rules"]
    if m5_weekly is not None:
        baselines.insert(0, "m5_fm_rules")
    for index, baseline in enumerate(baselines):
        baseline_values = np.asarray(
            [weekly_by_policy[baseline][date] for date in dates]
        )
        comparisons.append(
            {
                "comparison": f"m6_fm_llm_exact_k_minus_{baseline}",
                "paired_unit": "week after averaging repeated LLM calls",
                "target_budget": target_budget,
                **circular_block_bootstrap_delta(
                    llm_values,
                    baseline_values,
                    block_length=min(4, len(cases)),
                    n_bootstrap=20_000,
                    seed=seeds[0] + index,
                ),
            }
        )
    adjusted = holm_adjust([row["p_two_sided"] for row in comparisons])
    for row, p_holm in zip(comparisons, adjusted, strict=True):
        row["p_holm_across_rule_comparisons"] = p_holm
        row["n_rule_comparisons"] = len(comparisons)

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "n_weeks": len(cases),
            "runs_per_week": len(seeds),
            "target_budget": target_budget,
            "budget_unit": "exactly K unique protocol targets per week-run",
            "decision_model_requested": model,
            "provider_model_requested": api_model_slug(model),
            "provider_policy": DEFAULT_PROVIDER_POLICY,
            "temperature": temperature,
            "requested_seeds": list(seeds),
            "seed_evidence_boundary": (
                "OpenRouter accepts seed, but provider-side adherence is not "
                "independently observable; repeated calls are the experimental unit."
            ),
            "labels_excluded_from_requests": True,
            "frozen_source_sha256": frozen_payload.get("source_sha256"),
            "m5_evidence_alignment": alignment_result,
            "rate_usd_per_million_tokens": rates,
            "price_source": price_source,
        },
        "rows": rows,
        "per_seed_rows": per_seed_rows,
        "comparisons": comparisons,
    }
    atomic_write_json(output_dir / "results.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--m5-weekly", type=Path)
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target-budget", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS)
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--input-rate", type=float, default=3.0)
    parser.add_argument("--output-rate", type=float, default=15.0)
    parser.add_argument(
        "--price-source",
        default="https://openrouter.ai/anthropic/claude-sonnet-4.6/pricing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.timeout <= 0:
        raise ValueError("workers and timeout must be positive")
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("temperature must be within [0,2]")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique list")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases, frozen_payload = load_cases(
        args.frozen.resolve(),
        target_budget=args.target_budget,
        limit=args.limit,
    )
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_path": str(args.frozen.resolve()),
        "frozen_sha256": file_sha256(args.frozen.resolve()),
        "source_raw_sha256": frozen_payload.get("source_sha256"),
        "labels_excluded_from_requests": True,
        "target_budget": args.target_budget,
        "cases": [
            {
                "date": case["date"],
                "source_evidence_sha256": case["source_evidence_sha256"],
                "rendered_evidence_sha256": case["rendered_evidence_sha256"],
                "candidate_set_sha256": case["candidate_set"].candidate_set_sha256,
                "candidate_membership_sha256": (
                    case["candidate_set"].candidate_membership_sha256
                ),
                "system_prompt_sha256": sha256_text(case["system_prompt"]),
                "user_prompt_sha256": sha256_text(case["user_prompt"]),
            }
            for case in cases
        ],
    }
    atomic_write_json(output_dir / "request_manifest.json", manifest)
    if args.prepare_only:
        print(json.dumps({"prepared_cases": len(cases), "output_dir": str(output_dir)}, indent=2))
        return
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    records = run_calls(
        cases=cases,
        output_dir=output_dir,
        api_key=api_key,
        api_url=os.environ.get(
            "LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions"
        ),
        model=args.model,
        target_budget=args.target_budget,
        temperature=args.temperature,
        seeds=seeds,
        workers=args.workers,
        timeout=args.timeout,
    )
    result = analyze(
        records=records,
        cases=cases,
        frozen_payload=frozen_payload,
        m5_weekly=args.m5_weekly.resolve() if args.m5_weekly else None,
        alignment=args.alignment.resolve() if args.alignment else None,
        output_dir=output_dir,
        model=args.model,
        target_budget=args.target_budget,
        temperature=args.temperature,
        seeds=seeds,
        rates={"input": args.input_rate, "output": args.output_rate},
        price_source=args.price_source,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
