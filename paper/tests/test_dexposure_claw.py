"""Smoke tests for the DeXposure Claw portable agent extension."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CLAW_ROOT = REPO_ROOT / "claw"
SRC_ROOT = CLAW_ROOT / "src"


class DeXposureClawTests(unittest.TestCase):
    def setUp(self) -> None:
        if str(SRC_ROOT) not in sys.path:
            sys.path.insert(0, str(SRC_ROOT))

    def test_canonical_pack_layout_exists(self):
        required = [
            "pyproject.toml",
            "package.json",
            "bin/dexposure-claw.js",
            "pack/skills/run-full-suite/SKILL.md",
            "pack/commands/dexposure-benchmark.md",
            "adapters/claude-code/plugin.json.j2",
            "adapters/claude-code/mcp.json.j2",
            "src/dexposure_claw/cli.py",
            "src/dexposure_claw/mcp_server.py",
        ]

        for relative_path in required:
            self.assertTrue(
                (CLAW_ROOT / relative_path).exists(),
                f"missing {relative_path}",
            )

    def test_build_claude_code_plugin_renders_dist_from_canonical_pack(self):
        from dexposure_claw.installer import build_claude_code_plugin

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = pathlib.Path(tmp) / "dexposure-claw"
            result = build_claude_code_plugin(output_dir=output_dir)

            self.assertEqual(result["target"], "claude-code")
            manifest_path = output_dir / ".claude-plugin" / "plugin.json"
            mcp_path = output_dir / ".mcp.json"
            skill_path = output_dir / "skills" / "run-full-suite" / "SKILL.md"
            command_path = output_dir / "commands" / "dexposure-benchmark.md"

            self.assertTrue(manifest_path.exists())
            self.assertTrue(mcp_path.exists())
            self.assertTrue(skill_path.exists())
            self.assertTrue(command_path.exists())

            manifest = json.loads(manifest_path.read_text())
            mcp = json.loads(mcp_path.read_text())
            self.assertEqual(manifest["name"], "dexposure-claw")
            self.assertIn("dexposure", mcp["mcpServers"])
            self.assertEqual(mcp["mcpServers"]["dexposure"]["command"], "node")
            self.assertTrue((output_dir / "bin" / "dexposure-claw.js").exists())
            self.assertTrue((output_dir / "src" / "dexposure_claw" / "cli.py").exists())
            self.assertIn("dexposure-claw", skill_path.read_text().lower())

    def test_platform_config_snippets_are_generated_without_mutating_home(self):
        from dexposure_claw.installer import render_install_snippet

        hermes = render_install_snippet("hermes")
        codex = render_install_snippet("codex")
        mcp_only = render_install_snippet("mcp")

        self.assertIn("mcp_servers:", hermes)
        self.assertIn("dexposure-claw", hermes)
        self.assertIn("[mcp_servers.dexposure]", codex)
        self.assertIn("dexposure-claw", codex)
        self.assertIn('"mcpServers"', mcp_only)
        self.assertIn('"dexposure"', mcp_only)

    def test_node_wrapper_launches_claw_cli(self):
        package = json.loads((CLAW_ROOT / "package.json").read_text())

        self.assertEqual(package["name"], "@dexposure/claw")
        self.assertEqual(package["bin"]["dexposure-claw"], "bin/dexposure-claw.js")
        self.assertEqual(package["type"], "commonjs")
        self.assertEqual(package["publishConfig"]["access"], "public")

        result = subprocess.run(
            ["node", str(CLAW_ROOT / "bin" / "dexposure-claw.js"), "health"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["package"], "dexposure-claw")

    def test_mcp_exposes_live_predict_and_interval_builder(self):
        from dexposure_claw.mcp_server import TOOLS

        tool_names = {t["name"] for t in TOOLS}
        self.assertIn("dexposure_live_predict", tool_names)

        paper_dir = REPO_ROOT / "paper"
        if str(paper_dir) not in sys.path:
            sys.path.insert(0, str(paper_dir))
        from dexposure_agent.live_defillama import (
            DeXposureTokenMapper,
            build_dexposure_network_interval,
        )

        mapper = DeXposureTokenMapper()
        t1 = {"182": {"WETH": 100.0}, "1599": {"LDO": 50.0}}
        t2 = {"182": {"WETH": 120.0}, "1599": {"LDO": 30.0}}
        net = build_dexposure_network_interval(t1, t2, mapper)
        self.assertIn("nodes", net)
        self.assertIn("links", net)
        self.assertTrue(any(l["size"] > 0 for l in net["links"]))

    def test_audit_constraints_and_http_dashboard_endpoints(self):
        paper_dir = REPO_ROOT / "paper"
        if str(paper_dir) not in sys.path:
            sys.path.insert(0, str(paper_dir))

        # 1. Verify clean replay JSON has null mean_confidence when n_alerts == 0
        replay_path = REPO_ROOT / "outputs" / "pre2022_clean_replay_20261103.json"
        replay_data = json.loads(replay_path.read_text())
        self.assertEqual(replay_data["terra"]["n_alerts"], 0)
        self.assertIsNone(replay_data["terra"]["mean_confidence"])
        self.assertEqual(replay_data["ftx"]["n_alerts"], 2)
        self.assertAlmostEqual(
            replay_data["ftx"]["mean_confidence"],
            sum(a["confidence"] for a in replay_data["ftx"]["alerts"]) / len(replay_data["ftx"]["alerts"]),
            places=6,
        )

        # 2. Verify m7 apply_action_gate never replaces target protocols
        from experiments.llm_eval_b5 import apply_action_gate

        raw_m6 = {
            "risk_level": "elevated",
            "target_protocols": [
                {"protocol": "118", "action": "Contingency", "risk_score": 0.92},
                {"protocol": "1599", "action": "Recommend-Reduce", "risk_score": 0.80},
            ],
        }
        gated_m7 = apply_action_gate(raw_m6)
        raw_ids = {t["protocol"] for t in raw_m6["target_protocols"]}
        gated_ids = {t["protocol"] for t in gated_m7["target_protocols"]}
        self.assertEqual(raw_ids, gated_ids)
        by_id = {t["protocol"]: t for t in gated_m7["target_protocols"]}
        self.assertEqual(by_id["118"]["action"], "Investigate")
        self.assertEqual(by_id["1599"]["action"], "Recommend-Reduce")

        # 3. Verify HTTP Dashboard and /api/cases endpoints
        from fastapi.testclient import TestClient
        from dexposure_agent.serve import app

        client = TestClient(app)
        resp_root = client.get("/")
        self.assertEqual(resp_root.status_code, 200)
        self.assertIn("DeFiLlama", resp_root.text)
        self.assertIn('data-case="live"', resp_root.text)

        resp_cases = client.get("/api/cases")
        self.assertEqual(resp_cases.status_code, 200)
        self.assertEqual(resp_cases.json()["status"], "ok")

        resp_sim = client.post(
            "/api/contagion-simulate",
            json={"shocked_nodes": ["182"], "shock_fraction": 0.5, "horizon": 4},
        )
        self.assertEqual(resp_sim.status_code, 200)
        sim_json = resp_sim.json()
        self.assertEqual(sim_json["status"], "ok")
        self.assertGreater(sim_json["observed"]["direct_loss"], 0.0)
        self.assertGreater(sim_json["observed"]["contagion_loss"], 0.0)
        self.assertTrue(sim_json["predicted"]["per_node"])
        self.assertEqual(
            client.post("/api/contagion-simulate", json={"shocked_nodes": ["nope"], "horizon": 4}).status_code,
            422,
        )

    def test_2025_2026_token_mapping_resolution(self):
        paper_dir = REPO_ROOT / "paper"
        if str(paper_dir) not in sys.path:
            sys.path.insert(0, str(paper_dir))
        from dexposure_agent.live_defillama import (
            DeXposureTokenMapper,
            _align_series_adapter_reclassifications,
        )

        mapper = DeXposureTokenMapper()
        # 1. Check TVL-descending symbol deduplication & core token hubs
        self.assertEqual(mapper.resolve_token_issuer("SYRUP"), "587")
        self.assertEqual(mapper.resolve_token_issuer("SPK"), "2929")
        self.assertEqual(mapper.resolve_token_issuer("USDC"), "USDC")
        self.assertEqual(mapper.resolve_token_issuer("USDE"), "USDE")
        self.assertEqual(mapper.resolve_token_issuer("WBTC"), "WBTC")

        # 2. Check live registration sorts by descending TVL so $0-TVL protocols never overwrite active ones
        mapper.register_live_protocols(
            [
                {"id": "999999", "name": "Dead Protocol", "symbol": "SYRUP", "tvl": 0.0},
                {"id": "587", "name": "Maple", "symbol": "SYRUP", "tvl": 2_480_000_000.0},
            ]
        )
        self.assertEqual(mapper.resolve_token_issuer("SYRUP"), "587")

        # 3. Check protocol name resolution for 2025-2026 protocols
        self.assertIn("Lombard", mapper.protocol_name("5097"))
        self.assertIn("Lorenzo", mapper.protocol_name("5292"))
        self.assertIn("Sentora", mapper.protocol_name("6807"))
        self.assertIn("Jupiter Lend", mapper.protocol_name("6600"))

        # 4. Check DeFiLlama adapter token-reclassification denoising
        raw_series = {
            1000: {"USDC": 2_000_000_000.0},
            2000: {"BTC": 1_800_000_000.0, "XRP": 190_000_000.0},  # 0.5% net change, 199.5% gross turnover
        }
        aligned = _align_series_adapter_reclassifications(raw_series)
        self.assertAlmostEqual(sum(aligned[2000].values()), 1_990_000_000.0, places=2)
        self.assertAlmostEqual(aligned[2000].get("USDC", 0.0), 1_990_000_000.0, places=2)


if __name__ == "__main__":
    unittest.main()
