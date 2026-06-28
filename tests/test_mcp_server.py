from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dexposure_claw import mcp_server


class McpServerTest(unittest.TestCase):
    def test_initialize_reports_server_info(self) -> None:
        response = mcp_server._handle(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )

        self.assertIsNotNone(response)
        result = response["result"]
        self.assertEqual(result["serverInfo"]["name"], "dexposure-claw")
        self.assertEqual(result["capabilities"], {"tools": {}})

    def test_tools_list_exposes_current_tool_names(self) -> None:
        response = mcp_server._handle(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        )

        self.assertIsNotNone(response)
        names = [tool["name"] for tool in response["result"]["tools"]]
        self.assertEqual(
            names,
            [
                "dexposure_health",
                "dexposure_install_snippet",
                "dexposure_list_benchmarks",
            ],
        )

    def test_benchmark_catalog_contains_six_ids(self) -> None:
        result = mcp_server._call_tool("dexposure_list_benchmarks", {})
        benchmarks = result["content"][0]["text"]

        for benchmark_id in (
            "b1_forecast",
            "b2_warning",
            "b3_calibration",
            "b4_stress",
            "b5_decision",
            "b6_robustness",
        ):
            self.assertIn(benchmark_id, benchmarks)


if __name__ == "__main__":
    unittest.main()
