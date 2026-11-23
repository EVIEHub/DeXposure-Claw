from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PYTHONPATH = str(ROOT / "src")


class CliSmokeTest(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = ENV_PYTHONPATH
        env["DEXPOSURE_CLAW_ROOT"] = str(ROOT)
        return subprocess.run(
            [sys.executable, "-m", "dexposure_claw.cli", *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

    def test_health_reports_ok_status(self) -> None:
        proc = self.run_cli("health")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["package"], "dexposure-claw")

    def test_claude_code_adapter_builds_dist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dexposure-claw"
            proc = self.run_cli("build", "claude-code", "--output", str(out))
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["target"], "claude-code")
            self.assertTrue((out / ".claude-plugin" / "plugin.json").exists())


import os

if __name__ == "__main__":
    unittest.main()
