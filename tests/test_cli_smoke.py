from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    env["DEXPOSURE_CLAW_ROOT"] = str(ROOT)
    return env


class CliSmokeTest(unittest.TestCase):
    def test_python_cli_health_reports_package(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dexposure_claw.cli", "health"],
            cwd=ROOT,
            env=runtime_env(),
            text=True,
            capture_output=True,
            check=True,
        )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["package"], "dexposure-claw")

    def test_node_shim_health_reports_package(self) -> None:
        result = subprocess.run(
            ["node", "bin/dexposure-claw.js", "health"],
            cwd=ROOT,
            env=runtime_env(),
            text=True,
            capture_output=True,
            check=True,
        )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["package"], "dexposure-claw")


if __name__ == "__main__":
    unittest.main()
