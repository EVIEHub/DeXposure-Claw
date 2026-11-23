#!/usr/bin/env node
const fs = require("node:fs");
const { spawnSync } = require("node:child_process");
const path = require("node:path");

const args = process.argv.slice(2);
const packageRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(packageRoot, "..");
const localSrc = path.join(packageRoot, "src");
const paperSrc = path.join(repoRoot, "paper");
const extraPaths = [localSrc];
if (fs.existsSync(paperSrc)) {
  extraPaths.push(paperSrc);
}
const pythonPath = process.env.PYTHONPATH
  ? `${extraPaths.join(path.delimiter)}${path.delimiter}${process.env.PYTHONPATH}`
  : extraPaths.join(path.delimiter);
const env = {
  ...process.env,
  PYTHONPATH: pythonPath,
  DEXPOSURE_CLAW_ROOT: process.env.DEXPOSURE_CLAW_ROOT || packageRoot
};
const venvPython = path.join(repoRoot, ".venv", "bin", "python");
const candidates = [
  process.env.PYTHON,
  fs.existsSync(venvPython) ? venvPython : null,
  "python3",
  "python"
].filter(Boolean);

let result = null;
for (const python of candidates) {
  result = spawnSync(
    python,
    ["-m", "dexposure_claw.cli", ...args],
    { stdio: "inherit", env }
  );
  if (result.error && result.error.code === "ENOENT") {
    continue;
  }
  process.exit(result.status === null ? 1 : result.status);
}

console.error("dexposure-claw: Python 3 was not found. Install Python and retry.");
process.exit(127);
