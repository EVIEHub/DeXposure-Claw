"""Replay the canonical M7 action gate on frozen exact-K M6 tickets.

The replay reuses parsed responses byte-for-byte and never invokes the FM or
an LLM.  It validates the frozen EvidenceBundle hashes, the response contract,
and candidate membership before comparing intervention outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper"
if str(PAPER_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_ROOT))

from experiments.llm_eval_b5 import (
    ACTION_GATE_POLICY,
    apply_action_gate,
    normalize_protocol_name,
)
from scripts.run_frozen_llm_decision_eval import (
    INTERVENTION_ACTIONS,
    load_cases,
    validate_record,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(value)
    return value


def evaluate_replay(
    records: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_by_date = {case["date"]: case for case in cases}
    if len(case_by_date) != len(cases):
        raise ValueError("frozen case dates must be unique")
    if not records:
        raise ValueError("raw response artifact is empty")
    if len({row.get("response_id") for row in records}) != len(records):
        raise ValueError("response IDs must be present and unique")
    date_runs = {(row.get("date"), row.get("run_index")) for row in records}
    if len(date_runs) != len(records):
        raise ValueError("date/run_index pairs must be unique")

    by_date: dict[str, list[dict[str, Any]]] = {date: [] for date in case_by_date}
    for record in records:
        date = str(record.get("date", ""))
        case = case_by_date.get(date)
        if case is None:
            raise ValueError(f"raw response date {date!r} is outside frozen cases")
        if not validate_record(record, case, 5):
            raise ValueError(f"{date}: response is not contract-valid exact-K=5")
        if record.get("source_evidence_sha256") != case["source_evidence_sha256"]:
            raise ValueError(f"{date}: source EvidenceBundle hash mismatch")
        if record.get("rendered_evidence_sha256") != case["rendered_evidence_sha256"]:
            raise ValueError(f"{date}: rendered EvidenceBundle hash mismatch")
        if (
            record.get("candidate_set_sha256")
            != case["candidate_set"].candidate_set_sha256
        ):
            raise ValueError(f"{date}: CandidateSet hash mismatch")
        if (
            record.get("candidate_membership_sha256")
            != case["candidate_set"].candidate_membership_sha256
        ):
            raise ValueError(f"{date}: candidate membership hash mismatch")
        by_date[date].append(record)

    run_counts = {len(rows) for rows in by_date.values()}
    if len(run_counts) != 1 or 0 in run_counts:
        raise ValueError("every frozen week must have the same positive run count")
    runs_per_week = run_counts.pop()

    weekly: list[dict[str, Any]] = []
    totals = {
        "m6_fm_llm": {"interventions": 0, "false_interventions": 0},
        "m7_fm_llm_action_gate": {
            "interventions": 0,
            "false_interventions": 0,
            "demotions": 0,
        },
    }
    for date in [case["date"] for case in cases]:
        stressed = {
            normalize_protocol_name(value)
            for value in case_by_date[date]["truly_stressed"]
        }
        row = {
            "date": date,
            "m6_interventions": 0,
            "m6_false_interventions": 0,
            "m7_interventions": 0,
            "m7_false_interventions": 0,
            "gate_demotions": 0,
        }
        for record in sorted(by_date[date], key=lambda value: value["run_index"]):
            raw = record["parsed"]
            gated = apply_action_gate(raw)
            raw_targets = {
                normalize_protocol_name(target["protocol"]): target
                for target in raw["target_protocols"]
            }
            gated_targets = {
                normalize_protocol_name(target["protocol"]): target
                for target in gated["target_protocols"]
            }
            if raw_targets.keys() != gated_targets.keys():
                raise ValueError(f"{date}: action gate changed the target set")
            for protocol, raw_target in raw_targets.items():
                gated_target = gated_targets[protocol]
                if raw_target["action"] in INTERVENTION_ACTIONS:
                    row["m6_interventions"] += 1
                    row["m6_false_interventions"] += int(protocol not in stressed)
                if gated_target["action"] in INTERVENTION_ACTIONS:
                    row["m7_interventions"] += 1
                    row["m7_false_interventions"] += int(protocol not in stressed)
                row["gate_demotions"] += int(
                    raw_target["action"] in INTERVENTION_ACTIONS
                    and gated_target["action"] == "Investigate"
                )
        weekly.append(row)
        totals["m6_fm_llm"]["interventions"] += row["m6_interventions"]
        totals["m6_fm_llm"]["false_interventions"] += row["m6_false_interventions"]
        totals["m7_fm_llm_action_gate"]["interventions"] += row["m7_interventions"]
        totals["m7_fm_llm_action_gate"]["false_interventions"] += row[
            "m7_false_interventions"
        ]
        totals["m7_fm_llm_action_gate"]["demotions"] += row["gate_demotions"]

    n_week_runs = len(cases) * runs_per_week
    for metrics in totals.values():
        interventions = metrics["interventions"]
        metrics["interventions_per_week_run"] = interventions / n_week_runs
        metrics["false_interventions_per_week_run"] = (
            metrics["false_interventions"] / n_week_runs
        )
        metrics["fir_micro"] = (
            metrics["false_interventions"] / interventions if interventions else None
        )
    return totals, weekly


def bootstrap_deltas(
    weekly: list[dict[str, Any]],
    *,
    runs_per_week: int,
    block_length: int = 4,
    n_bootstrap: int = 20_000,
    seed: int = 20260809,
) -> dict[str, Any]:
    values = np.asarray(
        [
            [
                row["m6_interventions"],
                row["m6_false_interventions"],
                row["m7_interventions"],
                row["m7_false_interventions"],
            ]
            for row in weekly
        ],
        dtype=float,
    )
    n_weeks = len(values)
    block_length = min(block_length, n_weeks)
    n_blocks = math.ceil(n_weeks / block_length)
    offsets = np.arange(block_length)
    rng = np.random.default_rng(seed)

    def deltas(sample: np.ndarray) -> np.ndarray:
        raw_i, raw_f, gated_i, gated_f = sample.sum(axis=0)
        return np.asarray(
            [
                (gated_i - raw_i) / (n_weeks * runs_per_week),
                (gated_f - raw_f) / (n_weeks * runs_per_week),
                gated_f / gated_i - raw_f / raw_i,
            ]
        )

    observed = deltas(values)
    samples = np.empty((n_bootstrap, 3), dtype=float)
    for index in range(n_bootstrap):
        starts = rng.integers(0, n_weeks, size=n_blocks)
        selected = np.concatenate([((start + offsets) % n_weeks) for start in starts])[
            :n_weeks
        ]
        samples[index] = deltas(values[selected])
    names = (
        "interventions_per_week_run",
        "false_interventions_per_week_run",
        "fir_micro",
    )
    return {
        "comparison": "m7_fm_llm_action_gate_minus_m6_fm_llm",
        "paired_unit": "week with three reused raw tickets",
        "block_length": block_length,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "deltas": {
            name: {
                "point": float(observed[column]),
                "ci_95": [
                    float(np.quantile(samples[:, column], 0.025)),
                    float(np.quantile(samples[:, column], 0.975)),
                ],
            }
            for column, name in enumerate(names)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frozen = args.frozen.resolve()
    raw = args.raw.resolve()
    cases, frozen_payload = load_cases(frozen, target_budget=5)
    records = json.loads(raw.read_text())
    totals, weekly = evaluate_replay(records, cases)
    runs_per_week = len(records) // len(cases)
    result = {
        "schema_version": 1,
        "design": {
            "n_weeks": len(cases),
            "runs_per_week": runs_per_week,
            "target_budget": 5,
            "labels_excluded_from_requests": True,
            "replay_only": True,
            "llm_calls": 0,
            "fm_training_or_inference": False,
            "frozen_sha256": file_sha256(frozen),
            "raw_responses_sha256": file_sha256(raw),
            "source_raw_sha256": frozen_payload.get("source_sha256"),
            "action_gate_policy": json_safe(ACTION_GATE_POLICY),
        },
        "policies": totals,
        "paired_block_bootstrap": bootstrap_deltas(weekly, runs_per_week=runs_per_week),
        "weekly": weekly,
        "evidence_boundary": (
            "The gate changes action labels but not target membership. Coverage "
            "is therefore identical by construction. Rule baselines do not "
            "provide comparable action labels, so their FIR is not estimated."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
