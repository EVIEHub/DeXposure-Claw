from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/replay_coverage_intervention_frontier.py"
)
SPEC = importlib.util.spec_from_file_location("coverage_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


PROMPT = """Top-10 protocols by predicted exposure weight:
  1. Alpha (Lending) -- weight: 100.00
  2. Beta (Bridge) -- weight: 80.00

== STRESS SCENARIO ANALYSIS (applied to predicted graph) ==
  S1 (Top protocol failure): loss=20.00%, distressed=2, affected=2
    - Beta (Bridge): loss=70.00
    - Alpha (Lending): loss=20.00
"""


class CoverageInterventionReplayTests(unittest.TestCase):
    def case(self):
        output = {
            "risk_level": "elevated",
            "target_protocols": [
                {
                    "protocol": "Alpha",
                    "risk_score": 0.80,
                    "action": "Recommend-Reduce",
                },
                {
                    "protocol": "Beta",
                    "risk_score": 0.70,
                    "action": "Recommend-Reduce",
                },
            ],
        }
        return MODULE.EvaluationCase(
            evidence=MODULE.EvidenceBundle(
                date="2025-01-01",
                system_prompt="system",
                user_prompt=PROMPT,
                evidence_sha256=MODULE.evidence_hash("system", PROMPT),
            ),
            cached_primary_output=output,
            truly_stressed=frozenset({"Beta"}),
        )

    def test_evidence_hash_is_stable_and_prompt_sensitive(self):
        first = MODULE.evidence_hash("system", PROMPT)
        self.assertEqual(first, MODULE.evidence_hash("system", PROMPT))
        self.assertNotEqual(first, MODULE.evidence_hash("system", PROMPT + "x"))

    def test_typed_candidate_set_hash_is_order_invariant(self):
        first = MODULE.CandidateSet.from_rows(
            [
                {"protocol": "Beta", "exposure": 80, "scenario_max_loss": 70},
                {"protocol": "Alpha", "exposure": 100, "scenario_max_loss": 20},
            ]
        )
        second = MODULE.CandidateSet.from_rows(
            [
                {"protocol": "Alpha", "exposure": 100, "scenario_max_loss": 20},
                {"protocol": "Beta", "exposure": 80, "scenario_max_loss": 70},
            ]
        )
        self.assertEqual(first.candidate_set_sha256, second.candidate_set_sha256)
        self.assertNotEqual(
            MODULE.evidence_hash("system", PROMPT, first),
            MODULE.evidence_hash("system", PROMPT, MODULE.CandidateSet.from_rows(
                [{"protocol": "Alpha", "exposure": 101}]
            )),
        )

    def test_rule_policies_use_only_visible_protocol_evidence(self):
        case = self.case()
        exposure = MODULE.rule_targets(case.evidence, "exposure_rules")
        scenario = MODULE.rule_targets(case.evidence, "scenario_max_rules")
        self.assertEqual(exposure[0]["protocol"], "Alpha")
        self.assertEqual(scenario[0]["protocol"], "Beta")
        self.assertTrue(all(row["action"] == "Investigate" for row in scenario))

    def test_scenario_rule_treats_missing_loss_as_zero(self):
        candidates = MODULE.CandidateSet.from_rows(
            [
                {"protocol": "Alpha", "exposure": 100},
                {"protocol": "Beta", "exposure": 80, "scenario_max_loss": 70},
            ]
        )
        evidence = MODULE.EvidenceBundle(
            date="2025-01-01",
            system_prompt="system",
            user_prompt=PROMPT,
            evidence_sha256=MODULE.evidence_hash("system", PROMPT, candidates),
            candidate_set=candidates,
        )
        ranked = MODULE.rule_targets(evidence, "scenario_max_rules")
        self.assertEqual([row["protocol"] for row in ranked], ["Beta", "Alpha"])

    def test_gate_replays_on_a_copy_of_the_exact_output(self):
        case = self.case()
        original = MODULE.cached_llm_output(case.cached_primary_output)
        gated, demotions = MODULE.apply_action_gate(original)
        self.assertEqual(demotions, 1)
        self.assertEqual(
            gated["target_protocols"][1]["action"], "Investigate"
        )
        self.assertEqual(
            case.cached_primary_output["target_protocols"][1]["action"],
            "Recommend-Reduce",
        )

    def test_fir_is_undefined_when_policy_never_intervenes(self):
        metrics, _ = MODULE.evaluate_policy(
            [self.case()], "scenario_max_rules", budget=1
        )
        self.assertIsNone(metrics["fir_micro"])
        self.assertEqual(metrics["coverage_at_k"], 1.0)

    def test_llm_metrics_count_false_interventions(self):
        metrics, _ = MODULE.evaluate_policy([self.case()], "llm_raw", budget=2)
        self.assertEqual(metrics["intervention_targets"], 2)
        self.assertEqual(metrics["false_interventions"], 1)
        self.assertEqual(metrics["fir_micro"], 0.5)

    def test_exact_k_selection_fails_closed_on_underfilled_output(self):
        with self.assertRaisesRegex(ValueError, "requires 3 unique"):
            MODULE.evaluate_policy([self.case()], "llm_raw", budget=3)

    def test_cached_output_contract_rejects_unsorted_scores(self):
        output = {
            "risk_level": "elevated",
            "rationale": "Two candidates are ranked.",
            "target_protocols": [
                {
                    "protocol": "Alpha",
                    "risk_score": 0.5,
                    "action": "Investigate",
                    "reason": "Visible exposure 100.00.",
                },
                {
                    "protocol": "Beta",
                    "risk_score": 0.7,
                    "action": "Investigate",
                    "reason": "Visible scenario loss 70.00.",
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "risk_score"):
            MODULE.validate_decision_output(output)

    def test_cost_requires_frozen_rates_and_explicit_usage(self):
        raw = [
            {
                "raw_llm_outputs": [
                    {
                        "usage_known": True,
                        "input_tokens": 100,
                        "output_tokens": 20,
                    }
                ]
            }
        ]
        missing_rate = MODULE.token_cost(raw, "model", 1)
        self.assertIsNone(missing_rate["cost_usd"])
        priced = MODULE.token_cost(
            raw, "model", 1, {"input": 5.0, "output": 25.0}
        )
        self.assertEqual(priced["cost_usd"], 0.001)

        raw[0]["raw_llm_outputs"][0].pop("usage_known")
        unknown = MODULE.token_cost(
            raw, "model", 1, {"input": 5.0, "output": 25.0}
        )
        self.assertIsNone(unknown["cost_usd"])
        self.assertEqual(unknown["unknown_usage_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
