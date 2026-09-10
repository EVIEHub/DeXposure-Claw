#!/usr/bin/env python3
"""Recompute the re-decide False Intervention Rate (FIR) with correct losses.

``redecide_b5.py`` originally passed ``actual_losses={}`` to
``assess_week``, which made the "stable protocol" set empty and forced FIR
to 0.0 for every re-decided decision model (Sonnet, Gemini). This script
rebuilds the real per-week losses from snapshot data
(``snapshot_losses.load_actual_losses``) and recomputes FIR directly from
the CACHED decision outputs -- no LLM calls, no judge calls, no cost.

Correctness gate: before trusting the Sonnet/Gemini numbers, it recomputes
Opus's FIR from the main run and asserts it reproduces the published
0.4483 / 0.4368. If that self-check fails the script aborts without
touching any artifact.

It then rewrites, in place (after backing up to ``*.orig_buggy``):
  - redecide_<ts>/summary.json            (cells[*].mean_fir)
  - redecide_<ts>/decide_*.json           (entries[*].assessment.false_intervention_rate)
  - redecide_<ts>/rejudge_*.json          (entries[*].false_intervention_rate)
and writes ``corrected_fir.json`` summarising old vs new.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from snapshot_losses import load_actual_losses  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
MAIN_RUN = ROOT / "paper/results/run_20260515_pipeline_full/llm_eval"
REDECIDE = ROOT / "paper/results/redecide_20260603_023030"
HORIZON = 4

# Published Opus values that the self-check must reproduce.
OPUS_EXPECTED = {"m6_fm_llm": 0.4483, "m7_fm_llm_gated": 0.4368}
CELLS = [
    ("m6_fm_llm", "claude-sonnet-4.6"),
    ("m6_fm_llm", "google_gemini-2.5-pro"),
    ("m7_fm_llm_gated", "claude-sonnet-4.6"),
    ("m7_fm_llm_gated", "google_gemini-2.5-pro"),
]


def normalize(name) -> str:
    if not isinstance(name, str):
        return ""
    n = re.sub(r"\s*\([^)]*\)\s*$", "", name.strip())
    return n.strip()


def week_fir(protos, loss_keys_norm: set[str], truly_stressed) -> tuple[int, int]:
    """(false_interventions, intervention_targets) -- faithful to
    assess_week lines 834-846."""
    ts = {normalize(p) for p in truly_stressed}
    ts.discard("")
    stable = loss_keys_norm - ts
    false = inter = 0
    for p in protos:
        name = normalize(p.get("protocol", ""))
        if not name:
            continue
        if p.get("action") in ("Recommend-Reduce", "Contingency"):
            inter += 1
            if name in stable:
                false += 1
    return false, inter


def main():
    test_dates = [e["date"] for e in json.load(
        open(REDECIDE / "decide_m6_fm_llm__claude-sonnet-4.6.json"))["entries"]]

    print(f"computing real actual_losses for {len(test_dates)} weeks ...")
    losses = load_actual_losses(test_dates, horizon=HORIZON)
    loss_keys_norm = {
        d: {normalize(k) for k in lk} - {""} for d, lk in losses.items()
    }
    empty = [d for d in test_dates if not losses[d]]
    if empty:
        print(f"  WARN no losses for: {empty}")

    # truly_stressed per (method, date) from the main run (what redecide used)
    truly = {}
    for m in ["m6_fm_llm", "m7_fm_llm_gated"]:
        for e in json.load(open(MAIN_RUN / f"raw_{m}.json")):
            truly[(m, e["date"])] = e["truly_stressed"]

    # ---- self-check on Opus (main run) ----
    print("\nself-check (Opus, main run):")
    for m, expect in OPUS_EXPECTED.items():
        raw = json.load(open(MAIN_RUN / f"raw_{m}.json"))
        firs = []
        for e in raw:
            f, i = week_fir(e["llm_outputs"][0].get("target_protocols", []), loss_keys_norm[e["date"]], e["truly_stressed"])
            firs.append(f / i if i > 0 else 0.0)
        got = sum(firs) / len(firs)
        ok = abs(got - expect) < 5e-4
        print(f"  {m}: recomputed={got:.4f} expected={expect} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print("ABORT: self-check failed, no artifacts modified.")
            sys.exit(1)

    # ---- recompute redecide cells & rewrite artifacts ----
    print("\nrecomputing re-decide cells:")
    report = []
    new_fir_by_cell = {}
    for method, dec in CELLS:
        decide_path = REDECIDE / f"decide_{method}__{dec}.json"
        dj = json.load(open(decide_path))
        firs = []
        for e in dj["entries"]:
            d = e["date"]
            f, i = week_fir(e["llm_outputs"][0].get("target_protocols", []), loss_keys_norm[d], truly[(method, d)])
            fir = f / i if i > 0 else 0.0
            firs.append(fir)
            e["assessment"]["false_intervention_rate"] = fir  # rewrite per-week
        mean_fir = round(sum(firs) / len(firs), 4)
        new_fir_by_cell[(method, dec)] = mean_fir

        # backup + write decide file
        bak = decide_path.with_suffix(".json.orig_buggy")
        if not bak.exists():
            bak.write_text(decide_path.read_text())
        decide_path.write_text(json.dumps(dj, indent=2))

        # rewrite matching rejudge per-week FIR (judge model fixed to opus 4.8)
        rj_path = REDECIDE / f"rejudge_{method}__{dec}__claude-opus-4.8.json"
        if rj_path.exists():
            rj = json.load(open(rj_path))
            by_date = {e["date"]: fir for e, fir in zip(dj["entries"], firs)}
            for e in rj.get("entries", []):
                if e["date"] in by_date:
                    e["false_intervention_rate"] = by_date[e["date"]]
            bak = rj_path.with_suffix(".json.orig_buggy")
            if not bak.exists():
                bak.write_text(rj_path.read_text())
            rj_path.write_text(json.dumps(rj, indent=2))

        print(f"  {method:<16} {dec:<24} 0.0000 -> {mean_fir:.4f}")
        report.append({"method": method, "decision": dec,
                       "fir_buggy": 0.0, "fir_corrected": mean_fir})

    # ---- rewrite summary.json cells ----
    sum_path = REDECIDE / "summary.json"
    sj = json.load(open(sum_path))
    for c in sj.get("cells", []):
        key = (c["method"], c["decision"].replace("/", "_"))
        if key in new_fir_by_cell:
            c["mean_fir"] = new_fir_by_cell[key]
    bak = sum_path.with_suffix(".json.orig_buggy")
    if not bak.exists():
        bak.write_text(sum_path.read_text())
    sum_path.write_text(json.dumps(sj, indent=2))

    (REDECIDE / "corrected_fir.json").write_text(json.dumps({
        "note": "FIR recomputed with real actual_losses; see recompute_redecide_fir.py. "
                "Opus self-check reproduced 0.4483/0.4368. F1/precision/judge unaffected.",
        "horizon": HORIZON,
        "cells": report,
    }, indent=2))
    print(f"\nwrote corrected summary + corrected_fir.json under {REDECIDE.name}/")
    print("originals backed up to *.orig_buggy")


if __name__ == "__main__":
    main()
