from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from experiments.llm_eval_b5 import build_candidate_set, build_prompt


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_harness_control_ablation.py"
)
SPEC = importlib.util.spec_from_file_location("harness_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


SYSTEM_PROMPT = """Role and task.

You MUST respond with valid JSON only (no markdown).
Schema:
{
  "risk_level": "low" | "elevated",
  "target_protocols": []
}

Rules:
- Cite numbers.
- Keep action constraints.
"""

USER_PROMPT = """FM evidence.
  1. Alpha (Lending) -- weight: 100.00

== STRESS SCENARIO ANALYSIS (applied to predicted graph) ==
  S1: loss=20.00%
    - Beta (Bridge): loss=70.00

Based on these FM model predictions, metrics, and stress analysis, identify targets."""


class HarnessControlAblationTests(unittest.TestCase):
    def test_scenario_ablation_removes_only_scenario_block(self):
        rendered = MODULE.remove_scenario_evidence(USER_PROMPT)
        self.assertIn("FM evidence", rendered)
        self.assertNotIn("STRESS SCENARIO", rendered)
        self.assertNotIn("Beta", rendered)
        self.assertIn("identify targets", rendered)

    def test_generated_prompt_scenario_ablation_keeps_candidates(self):
        data = {
            "nodes": {f"P{i}": {"category": "Lending"} for i in range(10)},
            "edges": [
                {"source": f"P{i}", "target": f"P{i}", "weight": 100 - i}
                for i in range(10)
            ],
        }
        scenarios = [
            {
                "scenario_id": "S1",
                "scenario_name": "shock",
                "system_loss_pct": 20.0,
                "distressed_count": 1,
                "affected_nodes": 1,
                "top_affected": [
                    {"protocol": "P0", "category": "Lending", "loss": 20.0}
                ],
            }
        ]
        candidate_set = build_candidate_set(data, scenarios)
        system, user = build_prompt(
            "m6_fm_llm",
            "2025-01-01",
            4,
            {**data, "n_nodes": 10, "n_edges": 10},
            None,
            {"N1_max_pagerank": 0.1},
            scenarios,
            candidate_set=candidate_set,
        )
        rendered = MODULE.remove_scenario_evidence(user)
        self.assertIn("P0", rendered)
        self.assertNotIn("scenario_max_loss", rendered)
        self.assertNotIn("scenario_sum_loss", rendered)
        self.assertNotIn("STRESS SCENARIO", rendered)
        self.assertIn("Based on these FM model predictions and metrics", rendered)

    def test_output_schema_ablation_preserves_rules(self):
        rendered = MODULE.remove_output_schema(SYSTEM_PROMPT)
        self.assertIn("Role and task", rendered)
        self.assertIn("Rules:", rendered)
        self.assertIn("Keep action constraints", rendered)
        self.assertNotIn("MUST respond", rendered)
        self.assertNotIn('"target_protocols"', rendered)

    def test_rendered_hashes_are_distinct(self):
        hashes = {
            MODULE.evidence_hash(*MODULE.render_condition(SYSTEM_PROMPT, USER_PROMPT, condition))
            for condition in (
                "full_harness",
                "without_scenario_evidence",
                "without_output_schema",
            )
        }
        self.assertEqual(len(hashes), 3)

    def test_json_parser_accepts_fence_and_rejects_prose(self):
        self.assertEqual(
            MODULE.parse_json_response('```json\n{"risk_level":"low"}\n```'),
            {"risk_level": "low"},
        )
        self.assertIsNone(MODULE.parse_json_response("Alpha looks risky."))

    def test_ticket_contract_checks_nested_field_names(self):
        valid = {
            "risk_level": "elevated",
            "rationale": "Alpha has weight 100.00.",
            "target_protocols": [
                {
                    "protocol": f"P{i}",
                    "risk_score": 0.9 - i * 0.1,
                    "action": "Investigate",
                    "reason": "weight 100.00",
                }
                for i in range(7)
            ],
        }
        drifted = {
            "overall_risk": "elevated",
            "rationale": "Alpha has weight 100.00.",
            "target_protocols": [
                {
                    "name": f"P{i}",
                    "risk_score": 0.9 - i * 0.1,
                    "recommended_action": "Investigate",
                    "reason": "weight 100.00",
                }
                for i in range(7)
            ],
        }
        self.assertTrue(MODULE.satisfies_ticket_contract(valid))
        self.assertTrue(
            MODULE.satisfies_ticket_contract(
                {**valid, "target_protocols": valid["target_protocols"][:5]},
                expected_count=5,
            )
        )
        self.assertFalse(MODULE.satisfies_ticket_contract(drifted))

    def test_numeric_trace_gate_demotes_only_untraceable_intervention(self):
        output = {
            "risk_level": "elevated",
            "target_protocols": [
                {
                    "protocol": "Alpha",
                    "risk_score": 0.9,
                    "action": "Recommend-Reduce",
                    "reason": "weight is 100.00",
                },
                {
                    "protocol": "Beta",
                    "risk_score": 0.8,
                    "action": "Recommend-Reduce",
                    "reason": "unsupported value 999.00",
                },
            ],
        }
        gated, demotions = MODULE.numeric_trace_gate(output, USER_PROMPT)
        self.assertEqual(demotions, 1)
        self.assertEqual(
            gated["target_protocols"][0]["action"], "Recommend-Reduce"
        )
        self.assertEqual(gated["target_protocols"][1]["action"], "Investigate")

    def test_exact_k_selection_fails_closed_on_underfilled_output(self):
        with self.assertRaisesRegex(ValueError, "requires 3 unique"):
            MODULE.require_exact_k_targets(
                {"target_protocols": [{"protocol": "Alpha", "risk_score": 0.9}]},
                3,
            )

    def test_ablation_reports_invalid_output_without_smaller_budget(self):
        valid = {
            "risk_level": "elevated",
            "rationale": "Seven ranked candidates are visible.",
            "target_protocols": [
                {
                    "protocol": f"P{i}",
                    "risk_score": 0.9 - i * 0.1,
                    "action": "Investigate",
                    "reason": "weight 100.00",
                }
                for i in range(7)
            ],
        }
        invalid = {
            "risk_level": "elevated",
            "rationale": "Only one candidate was emitted.",
            "target_protocols": [
                {
                    "protocol": "P0",
                    "risk_score": 0.9,
                    "action": "Recommend-Reduce",
                    "reason": "weight 100.00",
                }
            ],
        }
        records = [
            {
                "status": "ok",
                "date": "2025-01-01",
                "run_index": 0,
                "parsed": valid,
                "attempts": [{"usage_known": True, "input_tokens": 10, "output_tokens": 5}],
            },
            {
                "status": "ok",
                "date": "2025-01-02",
                "run_index": 0,
                "parsed": invalid,
                "attempts": [{"usage_known": True, "input_tokens": 10, "output_tokens": 5}],
            },
        ]
        source = {
            date: {
                "truly_stressed": ["P0"],
                "positive_loss_protocols": ["P0"],
            }
            for date in ("2025-01-01", "2025-01-02")
        }
        rows, coverage = MODULE.evaluate_outputs(
            condition="without_output_schema",
            outputs=records,
            source_by_date=source,
            prompt_by_date={date: "P0 weight 100.00" for date in source},
            budgets=(1, 3, 5, 7),
        )
        primary = next(row for row in rows if row["budget"] == 5)
        self.assertEqual(primary["contract_success_rate"], 0.5)
        self.assertEqual(primary["ticket_contract_success_rate"], 0.5)
        self.assertEqual(primary["n_valid_exact_k"], 1)
        self.assertEqual(primary["valid_exact_k_rate"], 0.5)
        self.assertEqual(primary["coverage_at_k"], 0.5)
        self.assertEqual(len(coverage[5]), 2)
        self.assertEqual(coverage[5][1], 0.0)
        self.assertIsNone(primary["conditional_fir_micro"])


if __name__ == "__main__":
    unittest.main()
