from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_paired_frontier.py"
SPEC = importlib.util.spec_from_file_location("paired_frontier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_m5_frontier_fails_closed_on_underfilled_week(tmp_path):
    path = tmp_path / "m5_weekly.json"
    path.write_text(
        json.dumps(
            {
                "weeks": [
                    {
                        "date": "2025-01-01",
                        "truly_stressed": ["Alpha"],
                        "targets_scored": [["Alpha", 0.9]],
                        "intervention_targets": [],
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="requires 3 unique"):
        MODULE._m5_rows(path, (1, 3))
    with pytest.raises(ValueError, match="requires 3 unique"):
        MODULE._m5_weekly_coverage(path, 3)


def test_csv_writer_accepts_policy_specific_metric_fields(tmp_path):
    path = tmp_path / "frontier.csv"
    MODULE._write_csv(
        path,
        [
            {"policy": "m5_fm_rules", "fir_micro": 0.5},
            {"policy": "llm_raw", "non_loss_fir_micro": 0.25},
        ],
    )
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["fir_micro"] == "0.5"
    assert rows[1]["non_loss_fir_micro"] == "0.25"


def test_m5_weekly_coverage_accepts_ranked_protocol_pairs(tmp_path):
    path = tmp_path / "m5_weekly.json"
    path.write_text(
        json.dumps(
            {
                "weeks": [
                    {
                        "date": "2025-01-01",
                        "truly_stressed": ["Alpha"],
                        "targets_scored": [["Alpha", 0.9], ["Beta", 0.8]],
                    }
                ]
            }
        )
    )
    dates, coverage = MODULE._m5_weekly_coverage(path, 1)
    assert dates == ["2025-01-01"]
    assert coverage.tolist() == [1.0]
