from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/replay_unified_action_gate.py"
SPEC = importlib.util.spec_from_file_location("unified_gate_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class UnifiedActionGateReplayTests(unittest.TestCase):
    def test_gate_changes_actions_without_changing_targets(self):
        raw = {
            "risk_level": "elevated",
            "rationale": "Visible evidence.",
            "target_protocols": [
                {
                    "protocol": "Alpha",
                    "risk_score": 0.80,
                    "action": "Recommend-Reduce",
                    "reason": "Visible evidence 1.0.",
                },
                {
                    "protocol": "Beta",
                    "risk_score": 0.70,
                    "action": "Recommend-Reduce",
                    "reason": "Visible evidence 2.0.",
                },
            ],
        }
        gated = MODULE.apply_action_gate(raw)
        self.assertEqual(
            [row["protocol"] for row in gated["target_protocols"]],
            ["Alpha", "Beta"],
        )
        self.assertEqual(gated["target_protocols"][0]["action"], "Recommend-Reduce")
        self.assertEqual(gated["target_protocols"][1]["action"], "Investigate")
        self.assertEqual(raw["target_protocols"][1]["action"], "Recommend-Reduce")


if __name__ == "__main__":
    unittest.main()
