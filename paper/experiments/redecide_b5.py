#!/usr/bin/env python3
"""Re-decide existing FM-conditioned LLM prompts with new decision LLMs.

Background
----------
Track A (rejudge_b5.py) swaps only the judge model while keeping the original
Claude Opus 4.7 decisions. Track B answers a different question: would a
weaker but tier-matched-to-judge decision model (Claude Sonnet 4.6) produce
materially different supervisory tickets given the SAME FM forecast inputs?
And do non-Anthropic decision models (GPT-5, Gemini 2.5 Pro) reproduce the
same Pareto operating point?

This script reuses the saved system_prompt + user_prompt from the original
run, so the FM forecast is held identical across decision models. Only the
decision LLM changes. After re-deciding, the script also re-judges with the
configured judge so each new (decision, judge) cell gets a complete row.

Like rejudge_b5.py, this runs purely against the OpenRouter API -- no FM
service, no GPU server.

Usage
-----
    export OPENROUTER_API_KEY=...
    python paper/experiments/redecide_b5.py \\
        --raw-dir paper/results/run_20260515_pipeline_full/llm_eval \\
        --decision claude-sonnet-4-6 \\
        --decision google/gemini-2.5-pro \\
        --judge claude-opus-4-8 \\
        --method m6_fm_llm

Output
------
    paper/results/redecide_<timestamp>/
        decide_<method>__<decision>.json     per-week decisions
        rejudge_<method>__<decision>__<judge>.json   per-week judge scores
        summary.json                           aggregated mean per cell
        redecide.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_eval_b5 as eval_mod  # noqa: E402
import rejudge_b5 as judge_mod  # noqa: E402
from snapshot_losses import load_actual_losses  # noqa: E402


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
GATED_LLM_METHODS = {"m7_fm_llm_gated"}


def _refresh_key():
    eval_mod.LLM_API_KEY = os.environ.get(
        "OPENROUTER_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "")
    )
    if not eval_mod.LLM_API_KEY:
        logger.error("OPENROUTER_API_KEY not set")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Decide-then-judge per method
# ---------------------------------------------------------------------------

def redecide_and_judge(
    raw_path: Path,
    method: str,
    decision_model: str,
    judge_model: str,
    consistency_runs: int,
    horizon: int,
    out_dir: Path,
    resume: bool,
) -> dict:
    if method == "m7_fm_llm_gated":
        raise ValueError(
            "m7_fm_llm_gated is replay-only; redecide m6_fm_llm once and "
            "apply the gate offline to those exact tickets"
        )
    if consistency_runs <= 0:
        raise ValueError("consistency_runs must be positive")
    if horizon not in {1, 4, 8, 12}:
        raise ValueError("horizon must be one of 1, 4, 8, or 12")
    if decision_model not in eval_mod.MODEL_COSTS:
        raise ValueError(f"No explicit cost configuration for decision model {decision_model}")
    if judge_model not in eval_mod.MODEL_COSTS:
        raise ValueError(f"No explicit cost configuration for judge model {judge_model}")
    raw_bytes = raw_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Raw decision artifact must contain a non-empty list: {raw_path}")
    dates = [entry.get("date") for entry in raw if isinstance(entry, dict)]
    if len(dates) != len(raw) or any(not isinstance(date, str) or not date for date in dates):
        raise ValueError(f"Every raw entry requires a non-empty date: {raw_path}")
    if len(dates) != len(set(dates)):
        raise ValueError(f"Raw decision dates must be unique: {raw_path}")
    required_fields = {"system_prompt", "user_prompt", "truly_stressed"}
    candidate_sets: dict[str, object] = {}
    for index, entry in enumerate(raw):
        missing = required_fields - set(entry)
        if missing:
            raise ValueError(f"Raw entry {index} is missing fields: {sorted(missing)}")
        if not isinstance(entry["truly_stressed"], list) or not entry["truly_stressed"]:
            raise ValueError(f"Raw entry {index} requires a non-empty stressed set")
        candidate_sets[entry["date"]] = eval_mod.recorded_candidate_set(entry)

    # Real per-protocol losses, keyed by week. Earlier versions passed
    # actual_losses={} here, which made assess_week's "stable" set empty and
    # forced FIR to 0.0 for every re-decided model (the redecide FIR bug).
    # Prefer losses persisted in the raw export; otherwise derive them from
    # local snapshot data (no FM service / GPU needed).
    losses_map: dict[str, dict] = {}
    missing_dates = []
    for entry in raw:
        if isinstance(entry.get("actual_losses"), dict) and entry["actual_losses"]:
            losses_map[entry["date"]] = entry["actual_losses"]
        else:
            missing_dates.append(entry["date"])
    if missing_dates:
        logger.info(f"  deriving actual_losses from snapshots for {len(missing_dates)} weeks")
        losses_map.update(
            load_actual_losses(missing_dates, horizon=horizon)
        )
    unresolved_losses = sorted(date for date in dates if not losses_map.get(date))
    if unresolved_losses:
        raise ValueError(
            "Missing per-protocol actual-loss evidence for dates: "
            + ", ".join(unresolved_losses)
        )

    safe_dec = decision_model.replace("/", "_")
    safe_judge = judge_model.replace("/", "_")
    decide_path = out_dir / f"decide_{method}__{safe_dec}.json"
    judge_path = out_dir / f"rejudge_{method}__{safe_dec}__{safe_judge}.json"

    decide_done: dict[str, dict] = {}
    judge_done: dict[str, dict] = {}
    if resume:
        if decide_path.exists():
            with decide_path.open() as handle:
                existing_decisions = json.load(handle)
            if existing_decisions.get("summary", {}).get("source_raw_sha256") != raw_sha256:
                raise ValueError(f"Resume source hash mismatch: {decide_path}")
            for e in existing_decisions.get("entries", []):
                if e.get("method") != method or e.get("decision_model") != decision_model:
                    raise ValueError(f"Resume decision identity mismatch: {decide_path}")
                decide_done[e["date"]] = e
        if judge_path.exists():
            with judge_path.open() as handle:
                existing_judges = json.load(handle)
            if existing_judges.get("summary", {}).get("source_raw_sha256") != raw_sha256:
                raise ValueError(f"Resume source hash mismatch: {judge_path}")
            for e in existing_judges.get("entries", []):
                if (
                    e.get("method") != method
                    or e.get("decision_model") != decision_model
                    or e.get("judge_model") != judge_model
                ):
                    raise ValueError(f"Resume judge identity mismatch: {judge_path}")
                judge_done[e["date"]] = e
        if decide_done or judge_done:
            logger.info(f"  resume: decide={len(decide_done)} judge={len(judge_done)}")

    decide_entries: list[dict] = list(decide_done.values())
    judge_entries: list[dict] = list(judge_done.values())
    dec_costs = eval_mod.MODEL_COSTS[decision_model]
    jud_costs = eval_mod.MODEL_COSTS[judge_model]
    decision_usage = eval_mod.summarize_attempt_usage(
        [
            request_attempt
            for entry in decide_entries
            for output in entry.get("llm_outputs", [])
            for request_attempt in eval_mod.billable_request_attempts(output)
        ],
        dec_costs,
    )
    judge_usage = eval_mod.summarize_attempt_usage(
        [
            request_attempt
            for entry in judge_entries
            for request_attempt in eval_mod.billable_request_attempts(entry)
        ],
        jud_costs,
    )
    total_dec_cost = float(decision_usage["known_cost_usd"])
    total_jud_cost = float(judge_usage["known_cost_usd"])
    unknown_dec_usage = int(decision_usage["unknown_usage_attempts"])
    unknown_jud_usage = int(judge_usage["unknown_usage_attempts"])

    for wi, entry in enumerate(raw):
        date = entry["date"]
        system_prompt = entry["system_prompt"]
        user_prompt = entry["user_prompt"]
        truly_stressed = entry["truly_stressed"]

        # --------- Decision ---------
        if date not in decide_done:
            _refresh_key()
            run_outputs = []
            for ri in range(consistency_runs):
                out = eval_mod.call_llm(system_prompt, user_prompt, model=decision_model)
                eval_mod.validate_decision_output(out, candidate_sets[date])
                if method in GATED_LLM_METHODS:
                    out = eval_mod.apply_action_gate(out)
                run_outputs.append(out)
                for request_attempt in eval_mod.billable_request_attempts(out):
                    attempt_cost = eval_mod._usage_cost(request_attempt, dec_costs)
                    if attempt_cost is None:
                        unknown_dec_usage += 1
                    else:
                        total_dec_cost += attempt_cost

            primary = run_outputs[0]
            actual_losses = losses_map.get(date, {})
            ev = eval_mod.assess_week(primary, set(truly_stressed), actual_losses, user_prompt)
            consistency = eval_mod.compute_consistency(run_outputs)

            decide_rec = {
                "date": date,
                "method": method,
                "decision_model": decision_model,
                "llm_outputs": run_outputs,
                "assessment": ev,
                "consistency": consistency,
                "truly_stressed_count": len(truly_stressed),
            }
            decide_entries.append(decide_rec)
            decide_done[date] = decide_rec
            self_assessment = ev
            self_primary = primary
        else:
            decide_rec = decide_done[date]
            self_primary = decide_rec["llm_outputs"][0]
            self_assessment = decide_rec["assessment"]
        eval_mod.validate_decision_output(self_primary, candidate_sets[date])
        for metric in ("precision", "completeness"):
            value = self_assessment.get(metric)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"Invalid cached {metric} for {date}")

        # --------- Judge ---------
        if date not in judge_done:
            meta = judge_mod.parse_prompt_meta(user_prompt)
            j_sys, j_user = judge_mod.build_judge_prompt(
                date=date, horizon=horizon, meta=meta,
                truly_stressed=truly_stressed,
                llm_output=self_primary,
            )
            _refresh_key()
            j_result = eval_mod.call_llm(j_sys, j_user, model=judge_model)
            score = j_result.get("quality_score", 0)
            if not isinstance(score, (int, float)) or not (1 <= score <= 5):
                score = None
            for request_attempt in eval_mod.billable_request_attempts(j_result):
                judge_attempt_cost = eval_mod._usage_cost(request_attempt, jud_costs)
                if judge_attempt_cost is None:
                    unknown_jud_usage += 1
                else:
                    total_jud_cost += judge_attempt_cost

            judge_rec = {
                "date": date,
                "method": method,
                "decision_model": decision_model,
                "judge_model": judge_model,
                "quality_score": int(score) if score is not None else None,
                "judge_output_valid": score is not None,
                "reasoning": j_result.get("reasoning", ""),
                "input_tokens": j_result.get("input_tokens", 0),
                "output_tokens": j_result.get("output_tokens", 0),
                "usage_known": j_result.get("usage_known") is True,
                "attempts": j_result.get("attempts", []),
                "precision": self_assessment.get("precision"),
                "completeness": self_assessment.get("completeness"),
                "false_intervention_rate": self_assessment.get("false_intervention_rate"),
                "false_interventions": self_assessment.get("false_interventions", 0),
                "intervention_targets": self_assessment.get("intervention_targets", 0),
                "n_targets": len(self_primary.get("target_protocols", [])),
            }
            judge_entries.append(judge_rec)
            judge_done[date] = judge_rec

        # Checkpoint after each week
        with open(decide_path, "w") as f:
            json.dump({
                "summary": {
                    "method": method,
                    "decision_model": decision_model,
                    "source_raw_sha256": raw_sha256,
                    "n_weeks": len(decide_entries),
                    "total_decision_cost_usd": (
                        None if unknown_dec_usage else round(total_dec_cost, 4)
                    ),
                    "known_decision_cost_usd": round(total_dec_cost, 4),
                    "unknown_usage_attempts": unknown_dec_usage,
                },
                "entries": decide_entries,
            }, f, indent=2)
        scores = [
            e["quality_score"]
            for e in judge_entries
            if isinstance(e.get("quality_score"), (int, float))
        ]
        with open(judge_path, "w") as f:
            json.dump({
                "summary": {
                    "method": method,
                    "decision_model": decision_model,
                    "judge_model": judge_model,
                    "source_raw_sha256": raw_sha256,
                    "n_weeks": len(judge_entries),
                    "mean_quality": round(sum(scores) / len(scores), 3) if scores else None,
                    "total_judge_cost_usd": (
                        None if unknown_jud_usage else round(total_jud_cost, 4)
                    ),
                    "known_judge_cost_usd": round(total_jud_cost, 4),
                    "unknown_usage_attempts": unknown_jud_usage,
                },
                "entries": judge_entries,
            }, f, indent=2)

        if (wi + 1) % 5 == 0 or wi == len(raw) - 1:
            fir = self_assessment.get("false_intervention_rate")
            fir_text = f"{fir:.3f}" if fir is not None else "N/A"
            logger.info(
                f"  [{wi+1}/{len(raw)}] {date}: dec={decision_model[:20]} jud={judge_model[:20]} "
                f"score={judge_done[date]['quality_score']} "
                f"prec={self_assessment['precision']:.3f} "
                f"FIR={fir_text}"
            )

    intervention_targets = sum(e.get("intervention_targets", 0) for e in judge_entries)
    false_interventions = sum(e.get("false_interventions", 0) for e in judge_entries)
    valid_scores = [
        e["quality_score"]
        for e in judge_entries
        if isinstance(e.get("quality_score"), (int, float))
    ]
    precisions = [e.get("precision") for e in judge_entries]
    completenesses = [e.get("completeness") for e in judge_entries]
    if any(not isinstance(value, (int, float)) for value in precisions):
        raise ValueError("Re-decide artifact contains undefined precision")
    if any(not isinstance(value, (int, float)) for value in completenesses):
        raise ValueError("Re-decide artifact contains undefined completeness")
    return {
        "method": method,
        "decision_model": decision_model,
        "judge_model": judge_model,
        "n_weeks": len(judge_entries),
        "n_valid_judge_scores": len(valid_scores),
        "mean_quality": round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else None,
        "mean_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
        "mean_completeness": round(sum(completenesses) / len(completenesses), 4) if completenesses else None,
        "mean_f1": (
            round(
                sum(
                    2.0 * precision * completeness / (precision + completeness)
                    if precision + completeness > 0
                    else 0.0
                    for precision, completeness in zip(precisions, completenesses)
                )
                / len(precisions),
                4,
            )
            if precisions
            else None
        ),
        "mean_fir": (
            round(false_interventions / intervention_targets, 4)
            if intervention_targets
            else None
        ),
        "false_interventions": false_interventions,
        "intervention_targets": intervention_targets,
        "decision_cost_usd": (
            None if unknown_dec_usage else round(total_dec_cost, 4)
        ),
        "known_decision_cost_usd": round(total_dec_cost, 4),
        "decision_unknown_usage_attempts": unknown_dec_usage,
        "judge_cost_usd": (
            None if unknown_jud_usage else round(total_jud_cost, 4)
        ),
        "known_judge_cost_usd": round(total_jud_cost, 4),
        "judge_unknown_usage_attempts": unknown_jud_usage,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Re-decide saved FM prompts with new decision LLMs")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--method", action="append", default=[],
                    help="m6_fm_llm / m2_snapshot_llm (repeatable); "
                         "m7_fm_llm_gated is replay-only")
    ap.add_argument("--decision", action="append", default=[], dest="decisions",
                    help="Decision model (repeatable). Default: Sonnet 4.6 + Gemini 2.5 Pro")
    ap.add_argument("--judge", default="claude-opus-4.8",
                    help="Judge model used for ALL decision models")
    ap.add_argument("--consistency-runs", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.consistency_runs <= 0:
        ap.error("--consistency-runs must be positive")
    if args.horizon not in {1, 4, 8, 12}:
        ap.error("--horizon must be one of 1, 4, 8, or 12")

    if not args.method:
        args.method = ["m6_fm_llm"]
    if not args.decisions:
        # OpenRouter slugs (Anthropic uses dots, not dashes)
        args.decisions = [
            "claude-sonnet-4.6",      # tier-matched-to-judge primary
            "google/gemini-2.5-pro",  # cross-family decision
        ]

    raw_dir = Path(args.raw_dir).resolve()
    if not raw_dir.is_dir():
        logger.error(f"raw-dir does not exist: {raw_dir}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else (
        RESULTS_DIR / f"redecide_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "redecide.log"
    logger.add(str(log_file), level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}")

    logger.info(f"raw_dir:   {raw_dir}")
    logger.info(f"out_dir:   {out_dir}")
    logger.info(f"methods:   {args.method}")
    logger.info(f"decisions: {args.decisions}")
    logger.info(f"judge:     {args.judge}")

    grid: dict[tuple[str, str, str], dict] = {}
    failures: list[dict] = []
    for method in args.method:
        raw_path = raw_dir / f"raw_{method}.json"
        if not raw_path.exists():
            logger.error(f"missing required artifact {raw_path}")
            failures.append({"method": method, "error": f"missing {raw_path}"})
            continue
        for dec in args.decisions:
            logger.info(f"--- {method} | decision={dec} | judge={args.judge} ---")
            try:
                summary = redecide_and_judge(
                    raw_path=raw_path,
                    method=method,
                    decision_model=dec,
                    judge_model=args.judge,
                    consistency_runs=args.consistency_runs,
                    horizon=args.horizon,
                    out_dir=out_dir,
                    resume=args.resume,
                )
            except SystemExit:
                raise
            except Exception as e:
                logger.exception(f"failed: {method} x {dec}: {e}")
                summary = {"method": method, "decision_model": dec,
                           "judge_model": args.judge, "error": str(e)}
                failures.append({"method": method, "decision": dec, "error": str(e)})
            grid[(method, dec, args.judge)] = summary

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"status": "failed" if failures else "done", "failures": failures, "cells": [{
            "method": k[0], "decision": k[1], "judge": k[2], **v
        } for k, v in grid.items()]}, f, indent=2)

    # Pretty-print
    print("\n=== Re-decide summary ===")
    hdr = f"{'method':<22} {'decision':<26} {'judge':<22} {'F1?':>8} {'FIR':>7} {'judge':>7}"
    print(hdr); print("-" * len(hdr))
    for (method, dec, jud), s in grid.items():
        f1 = s.get("mean_f1")
        f1_text = f"{f1:>8.4f}" if f1 is not None else f"{'N/A':>8}"
        fir = s.get("mean_fir")
        fir_text = f"{fir:>7.3f}" if fir is not None else f"{'N/A':>7}"
        quality = s.get("mean_quality")
        quality_text = f"{quality:>7.2f}" if quality is not None else f"{'N/A':>7}"
        print(f"{method:<22} {dec[:24]:<26} {jud[:20]:<22} {f1_text} "
              f"{fir_text} {quality_text}")
    logger.info(f"All done. Output: {out_dir}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
