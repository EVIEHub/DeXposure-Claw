#!/usr/bin/env python3
"""Build a matched-budget frontier from deterministic M5 and replay artifacts.

The M5 weekly dump stores the rule engine's ranked targets and the subset that
received an intervention action.  The strict replay stores the same metrics
for frozen M6 evidence.  This script only combines generated artifacts; it
does not rerun FM or an LLM.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))

from scripts.replay_coverage_intervention_frontier import (  # noqa: E402
    circular_block_bootstrap_delta,
    evaluate_policy,
    holm_adjust,
    load_cases,
    require_exact_k_targets,
)


def _normalise(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _m5_rows(weekly_path: Path, budgets: tuple[int, ...]) -> list[dict]:
    payload = json.loads(weekly_path.read_text())
    weeks = payload.get("weeks")
    if not isinstance(weeks, list) or not weeks:
        raise ValueError("M5 weekly artifact has no weeks")
    rows: list[dict] = []
    for budget in budgets:
        weekly_coverage: list[float] = []
        weekly_precision: list[float] = []
        selected_count = hit_count = 0
        intervention_count = false_intervention_count = 0
        intervention_weeks = 0
        active_week_firs: list[float] = []
        for week in weeks:
            stressed = {
                _normalise(name)
                for name in week.get("truly_stressed", [])
                if _normalise(name)
            }
            ranked = []
            for pair in week.get("targets_scored", []):
                if not isinstance(pair, list) or not pair:
                    continue
                target = _normalise(pair[0])
                if target and target not in {item["protocol"] for item in ranked}:
                    ranked.append({"protocol": target, "risk_score": float(pair[1])})
            selected = require_exact_k_targets(ranked, budget)
            selected_protocols = {item["protocol"] for item in selected}
            hits = sum(protocol in stressed for protocol in selected_protocols)
            selected_count += len(selected)
            hit_count += hits
            weekly_coverage.append(hits / len(stressed))
            weekly_precision.append(hits / len(selected) if selected else 0.0)

            intervention_targets = {
                _normalise(name)
                for name in week.get("intervention_targets", [])
                if _normalise(name)
            }
            interventions = selected_protocols & intervention_targets
            false_interventions = interventions - stressed
            intervention_count += len(interventions)
            false_intervention_count += len(false_interventions)
            if interventions:
                intervention_weeks += 1
                active_week_firs.append(len(false_interventions) / len(interventions))

        n_weeks = len(weeks)
        rows.append(
            {
                "policy": "m5_fm_rules",
                "budget": budget,
                "n_weeks": n_weeks,
                "coverage_at_k": sum(weekly_coverage) / n_weeks,
                "precision_macro_at_k": sum(weekly_precision) / n_weeks,
                "precision_micro_at_k": hit_count / selected_count if selected_count else None,
                "selected_targets": selected_count,
                "correct_targets": hit_count,
                "budget_utilization": selected_count / (n_weeks * budget),
                "intervention_targets": intervention_count,
                "intervention_targets_per_week": intervention_count / n_weeks,
                "intervention_weeks": intervention_weeks,
                "false_interventions": false_intervention_count,
                "false_interventions_per_week": false_intervention_count / n_weeks,
                "fir_micro": (
                    false_intervention_count / intervention_count
                    if intervention_count
                    else None
                ),
                "fir_active_week_macro": (
                    sum(active_week_firs) / len(active_week_firs)
                    if active_week_firs
                    else None
                ),
                "gate_demotions": 0,
                "evidence_source": "deterministic FM + M5 rules weekly dump",
            }
        )
    return rows


def _m5_weekly_coverage(weekly_path: Path, budget: int) -> tuple[list[str], np.ndarray]:
    payload = json.loads(weekly_path.read_text())
    weeks = payload.get("weeks")
    if not isinstance(weeks, list) or not weeks:
        raise ValueError("M5 weekly artifact has no weeks")
    dates: list[str] = []
    coverage: list[float] = []
    for week in weeks:
        stressed = {
            _normalise(name)
            for name in week.get("truly_stressed", [])
            if _normalise(name)
        }
        ranked: list[dict[str, str]] = []
        for pair in week.get("targets_scored", []):
            if not isinstance(pair, list) or not pair:
                continue
            target = _normalise(pair[0])
            if target and target not in {row["protocol"] for row in ranked}:
                ranked.append({"protocol": target})
        selected = require_exact_k_targets(ranked, budget)
        if not stressed:
            raise ValueError(f"{week.get('date')}: empty stressed target set")
        dates.append(str(week["date"]))
        coverage.append(
            sum(target["protocol"] in stressed for target in selected) / len(stressed)
        )
    return dates, np.asarray(coverage, dtype=float)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m5-weekly", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--replay-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="1,3,5,7")
    parser.add_argument(
        "--allow-historical-count",
        action="store_true",
        help="Diagnostic-only compatibility for pre-exact-7 cached LLM outputs",
    )
    args = parser.parse_args()
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    if not budgets or any(value <= 0 for value in budgets):
        raise ValueError("budgets must be positive comma-separated integers")
    replay = json.loads(args.replay_results.read_text())
    replay_rows = [
        row
        for row in replay.get("rows", [])
        if row.get("policy") in {"llm_raw", "llm_gated_replay", "scenario_max_rules"}
        and row.get("budget") in budgets
    ]
    m5_rows = _m5_rows(args.m5_weekly, budgets)
    rows = m5_rows + replay_rows
    if not rows:
        raise ValueError("no frontier rows generated")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "paired_frontier.csv", rows)
    cases = load_cases(
        args.raw, allow_historical_count=args.allow_historical_count
    )
    raw_dates = [case.evidence.date for case in cases]
    comparisons: list[dict] = []
    for budget in budgets:
        m5_dates, m5_coverage = _m5_weekly_coverage(args.m5_weekly, budget)
        if m5_dates != raw_dates:
            raise ValueError("M5 weekly dates do not match the M6 raw artifact")
        _, llm_coverage = evaluate_policy(cases, "llm_raw", budget)
        comparisons.append(
            {
                "comparison": "llm_raw_minus_m5_fm_rules",
                "budget": budget,
                **circular_block_bootstrap_delta(
                    llm_coverage,
                    m5_coverage,
                    block_length=4,
                    n_bootstrap=20_000,
                    seed=20260806 + budget,
                ),
            }
        )
    adjusted = holm_adjust([item["p_two_sided"] for item in comparisons])
    for item, corrected in zip(comparisons, adjusted, strict=True):
        item["p_holm_across_m5_budgets"] = corrected

    output = {
        "schema_version": 1,
        "diagnostic_historical_count": args.allow_historical_count,
        "budgets": list(budgets),
        "m5_weekly": str(args.m5_weekly.resolve()),
        "raw_m6": str(args.raw.resolve()),
        "replay_results": str(args.replay_results.resolve()),
        "source_sha256": {
            "m5_weekly": hashlib.sha256(args.m5_weekly.read_bytes()).hexdigest(),
            "raw_m6": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
            "replay_results": hashlib.sha256(args.replay_results.read_bytes()).hexdigest(),
        },
        "rows": rows,
        "comparisons": comparisons,
        "cost": replay.get("cost", {}),
        "notes": [
            "Every reported M5 row contains exactly K unique targets per week; underfilled artifacts fail closed.",
            "llm_gated_replay reuses the exact cached primary M6 output and applies the deterministic action gate; it is not a new LLM sample.",
        ],
    }
    (args.output_dir / "paired_frontier.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"rows": len(rows), "comparisons": len(comparisons), "budgets": list(budgets)}, indent=2))
    print(f"Wrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
