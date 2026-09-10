from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_frozen_llm_decision_eval.py"
SPEC = importlib.util.spec_from_file_location("frozen_llm_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


SYSTEM = """Role.\nRules:\n- Return exactly 7 unique target_protocols, including Monitor rows when fewer\n  than 7 protocols warrant escalation.\n- Sort them."""


def test_rewrite_target_budget_changes_only_count_instruction():
    rewritten = MODULE.rewrite_target_budget(SYSTEM, 5)
    assert "exactly 5 unique target_protocols" in rewritten
    assert "fewer than 5 protocols" in rewritten
    assert "exactly 7" not in rewritten
    assert rewritten.startswith("Role.\nRules:")


def test_rewrite_target_budget_fails_when_contract_is_missing():
    with pytest.raises(ValueError, match="neither"):
        MODULE.rewrite_target_budget("No target contract.", 5)


def test_rewrite_target_budget_adds_missing_count_after_order_rule():
    prompt = "Rules:\n- Order target_protocols by risk_score descending."
    rewritten = MODULE.rewrite_target_budget(prompt, 5)
    assert rewritten == (
        prompt
        + "\n- Return exactly 5 unique target_protocols, including Monitor rows "
        "when fewer than 5 protocols warrant escalation."
    )


def test_jaccard_is_defined_for_empty_and_nonempty_sets():
    assert MODULE._jaccard(set(), set()) == 1.0
    assert MODULE._jaccard({"A", "B"}, {"B", "C"}) == pytest.approx(1 / 3)


def test_analyze_requires_both_m5_artifacts_or_neither(tmp_path):
    with pytest.raises(ValueError, match="supplied together"):
        MODULE.analyze(
            records=[],
            cases=[],
            frozen_payload={},
            m5_weekly=tmp_path / "weekly.json",
            alignment=None,
            output_dir=tmp_path,
            model="test",
            target_budget=5,
            temperature=0.0,
            seeds=(1,),
            rates={"input": 0.0, "output": 0.0},
            price_source="test",
        )
