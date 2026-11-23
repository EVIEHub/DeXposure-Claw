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
                "dexposure_live_predict",
            ],
        )


if __name__ == "__main__":
    unittest.main()
