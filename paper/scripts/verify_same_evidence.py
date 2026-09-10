#!/usr/bin/env python3
"""Verify that M5 and M6 were rendered from the same FM evidence.

The M6 raw artifact stores the rendered prompt and a typed EvidenceBundle.  The
M5 benchmark must provide a generated evidence-metadata sidecar with the same
forecast and CandidateSet hashes.  This script reruns the deterministic FM
forecast for the M6 dates and compares it with both recorded artifacts.  A
mismatch is a hard failure because M5/M6 comparisons then mix evidence drift
with policy differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))

from dexposure_agent.data_loader import SnapshotLoader  # noqa: E402
from dexposure_agent.fm_predictor import FMPredictor  # noqa: E402
from experiments.llm_eval_b5 import (  # noqa: E402
    build_candidate_set,
    build_prompt,
    compute_metrics,
    forecast_sha256,
    run_scenarios,
)
from scripts.replay_coverage_intervention_frontier import (  # noqa: E402
    CandidateSet,
    evidence_hash,
)


def graph_payload(graph) -> dict:
    nodes = {
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
    }
    edges = [
        {"source": edge.source, "target": edge.target, "weight": edge.weight}
        for edge in graph.edges
    ]
    return {
        "date": graph.date,
        "nodes": nodes,
        "edges": edges,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "horizon": 4,
    }


def forecast_hash(payload: dict) -> str:
    return forecast_sha256(payload, int(payload["horizon"]))


def _recorded_evidence(entry: dict) -> tuple[dict, CandidateSet]:
    evidence = entry.get("evidence_bundle")
    if not isinstance(evidence, dict):
        raise ValueError(f"{entry.get('date')}: missing recorded evidence_bundle")
    rows = evidence.get("candidate_set")
    if not isinstance(rows, list):
        raise ValueError(f"{entry.get('date')}: missing recorded CandidateSet")
    candidate_set = CandidateSet.from_rows(rows)
    if evidence.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
        raise ValueError(f"{entry.get('date')}: recorded CandidateSet hash mismatch")
    if evidence.get("candidate_membership_sha256") != candidate_set.candidate_membership_sha256:
        raise ValueError(
            f"{entry.get('date')}: recorded candidate membership hash mismatch"
        )
    expected_evidence = evidence_hash(
        str(entry.get("system_prompt", "")),
        str(entry.get("user_prompt", "")),
        candidate_set,
    )
    if evidence.get("evidence_sha256") != expected_evidence:
        raise ValueError(f"{entry.get('date')}: recorded EvidenceBundle hash mismatch")
    recorded_forecast = evidence.get("forecast_sha256")
    if recorded_forecast != entry.get("forecast_sha256"):
        raise ValueError(f"{entry.get('date')}: recorded forecast hash mismatch")
    if not isinstance(recorded_forecast, str) or len(recorded_forecast) != 64:
        raise ValueError(f"{entry.get('date')}: missing recorded forecast hash")
    return evidence, candidate_set


def _load_m5_evidence(path: Path, horizon: int) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    if payload.get("method") != "m5_fm_rules":
        raise ValueError("M5 evidence metadata must identify method m5_fm_rules")
    if int(payload.get("horizon", -1)) != horizon:
        raise ValueError("M5 evidence metadata horizon mismatch")
    weeks = payload.get("weeks")
    if not isinstance(weeks, list) or not weeks:
        raise ValueError("M5 evidence metadata has no weeks")
    result: dict[str, dict] = {}
    for row in weeks:
        date = str(row.get("date", ""))
        if not date or date in result:
            raise ValueError("M5 evidence metadata dates must be unique and non-empty")
        candidate_set = CandidateSet.from_rows(row.get("candidate_set", []))
        if row.get("candidate_set_sha256") != candidate_set.candidate_set_sha256:
            raise ValueError(f"{date}: M5 CandidateSet hash mismatch")
        if row.get("candidate_membership_sha256") != candidate_set.candidate_membership_sha256:
            raise ValueError(f"{date}: M5 candidate membership hash mismatch")
        if not isinstance(row.get("forecast_sha256"), str):
            raise ValueError(f"{date}: M5 forecast hash is missing")
        result[date] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument(
        "--m5-evidence",
        type=Path,
        required=True,
        help="Generated b5_evidence__m5_fm_rules.json metadata",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints" / "dexposure-fm-release",
    )
    parser.add_argument("--horizon", type=int, default=4)
    args = parser.parse_args()
    if args.horizon != 4:
        raise ValueError("M6 evidence verification currently requires horizon=4")

    raw = json.loads(args.raw.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("M6 raw artifact must be a non-empty list")
    m5_evidence = _load_m5_evidence(args.m5_evidence, args.horizon)
    loader = SnapshotLoader(data_dir=str(args.data_dir))
    predictor = FMPredictor(checkpoint_dir=str(args.checkpoint_dir), device="cpu")
    if not predictor.available:
        raise RuntimeError(f"FM checkpoints are unavailable: {args.checkpoint_dir}")

    rows = []
    for entry in raw:
        if entry.get("method") not in {None, "m6_fm_llm"}:
            raise ValueError(
                f"{entry.get('date')}: verifier requires raw m6_fm_llm entries"
            )
        evidence, recorded_candidate_set = _recorded_evidence(entry)
        date = entry["date"]
        if date not in m5_evidence:
            raise ValueError(f"{date}: missing M5 evidence metadata")
        m5_row = m5_evidence[date]
        observed = loader.load_single(date)
        forecast = predictor.predict(observed, horizon=args.horizon)
        payload = graph_payload(forecast)
        metrics = compute_metrics(payload)
        scenarios = run_scenarios(payload)
        rebuilt_candidate_set = build_candidate_set(payload, scenarios)
        rebuilt_forecast_hash = forecast_hash(payload)
        if rebuilt_forecast_hash != evidence["forecast_sha256"]:
            raise RuntimeError(f"{date}: reconstructed forecast hash does not match M6")
        if rebuilt_forecast_hash != m5_row["forecast_sha256"]:
            raise RuntimeError(f"{date}: M5/M6 forecast hash mismatch")
        if rebuilt_candidate_set.candidate_set_sha256 != recorded_candidate_set.candidate_set_sha256:
            raise RuntimeError(f"{date}: reconstructed CandidateSet does not match M6")
        if rebuilt_candidate_set.candidate_set_sha256 != m5_row["candidate_set_sha256"]:
            raise RuntimeError(f"{date}: M5/M6 CandidateSet hash mismatch")
        if rebuilt_candidate_set.candidate_membership_sha256 != m5_row["candidate_membership_sha256"]:
            raise RuntimeError(f"{date}: M5/M6 candidate membership mismatch")
        system, user = build_prompt(
            "m6_fm_llm",
            date,
            args.horizon,
            payload,
            None,
            metrics,
            scenarios,
            candidate_set=rebuilt_candidate_set,
        )
        recorded_hash = evidence["evidence_sha256"]
        rendered_hash = evidence_hash(system, user, recorded_candidate_set)
        rows.append(
            {
                "date": date,
                "recorded_evidence_sha256": recorded_hash,
                "expected_evidence_sha256": recorded_hash,
                "rendered_evidence_sha256": rendered_hash,
                "evidence_match": recorded_hash == rendered_hash,
                "forecast_sha256": rebuilt_forecast_hash,
                "candidate_set_sha256": recorded_candidate_set.candidate_set_sha256,
                "candidate_membership_sha256": recorded_candidate_set.candidate_membership_sha256,
                "n_nodes": payload["n_nodes"],
                "n_edges": payload["n_edges"],
            }
        )

    if not rows or not all(row["evidence_match"] for row in rows):
        mismatches = [row["date"] for row in rows if not row["evidence_match"]]
        raise RuntimeError(f"EvidenceBundle mismatch for dates: {mismatches}")
    if set(m5_evidence) != {row["date"] for row in rows}:
        raise RuntimeError("M5 evidence metadata dates do not match the M6 raw artifact")

    output = {
        "schema_version": 1,
        "raw_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "horizon": args.horizon,
        "n_weeks": len(rows),
        "evidence_matches": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("n_weeks", "evidence_matches", "raw_sha256")}, indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
