# DeXposure-Claw

DeXposure Claw is a lightweight agent-runtime extension for the DeXposure
DeFi risk-monitoring project. It provides a `dexposure-claw` CLI and a
dependency-free stdio MCP server for Claude Code, OpenAI Codex, Hermes, and
other MCP-compatible runtimes.

Claw is a research and monitoring extension. It does not execute trades, sign
transactions, custody assets, or provide investment advice.

Detailed report: [https://arxiv.org/pdf/2606.19501](https://arxiv.org/pdf/2606.19501)

## Project State

This repository is an alpha runtime package. The current release target is
agent-runtime setup, MCP connectivity, and DeXposure-Bench catalog discovery.
It is not yet a complete benchmark runner.

## Current Capabilities

The current package focuses on runtime setup and benchmark discovery:

| Capability | Status |
| --- | --- |
| CLI health check | Available |
| Claude Code adapter build | Available |
| MCP stdio server | Available |
| Install snippets for Hermes, Codex, and generic MCP clients | Available |
| DeXposure-Bench catalog listing | Available |
| Direct benchmark execution through MCP | Not implemented |
| Direct report generation through MCP | Not implemented |

## Quick Start

### Node.js / npm

For agent-runtime users, npm is the most convenient entrypoint. The npm package
ships a Node.js binary that starts the Claw Python runtime under the hood, so it
requires Node.js 18+ and Python 3.10+.

Install from GitHub before the package is published to npm:

```bash
npm install -g git+ssh://git@github.com/EVIEHub/DeXposure-Claw.git
dexposure-claw health
```

Run without a global install:

```bash
npm exec --yes --package git+ssh://git@github.com/EVIEHub/DeXposure-Claw.git -- dexposure-claw health
npm exec --yes --package git+ssh://git@github.com/EVIEHub/DeXposure-Claw.git -- dexposure-claw mcp
```

From the repository root:

```bash
npm exec --package . -- dexposure-claw health
npm exec --package . -- dexposure-claw mcp
```

After publishing:

```bash
npx -y @dexposure/claw health
npx -y @dexposure/claw mcp
```

To install globally from a checkout:

```bash
npm install -g .
dexposure-claw health
```

### Python

From the repository root:

```bash
pipx install .
dexposure-claw install
```

For editable local development:

```bash
python -m pip install -e .
dexposure-claw health
```

## MCP Tools

Start the MCP server with:

```bash
dexposure-claw mcp
```

The server currently exposes:

| Tool | Purpose |
| --- | --- |
| `dexposure_health` | Check that the DeXposure Claw MCP server is reachable. |
| `dexposure_install_snippet` | Return install config for Hermes, Codex, or generic MCP clients. |
| `dexposure_list_benchmarks` | List the six DeXposure-Bench benchmark IDs and readable names. |

## Claude Code

For local development:

```bash
dexposure-claw build claude-code
claude --plugin-dir dist/claude-code/dexposure-claw
```

## OpenAI Codex

Codex consumes DeXposure Claw through MCP:

```bash
codex mcp add dexposure -- dexposure-claw mcp
codex mcp get dexposure
```

With the npm package, clients can use:

```bash
codex mcp add dexposure -- npx -y @dexposure/claw mcp
```

Before npm publishing, clients with GitHub SSH access can use:

```bash
codex mcp add dexposure -- npm exec --yes --package git+ssh://git@github.com/EVIEHub/DeXposure-Claw.git -- dexposure-claw mcp
```

## Package Checks

Before publishing or opening a package-related pull request, run:

```bash
python -m compileall src/dexposure_claw
python -m unittest discover -s tests
npm run pack:check
```

The smoke-test boundary is intentionally small: it verifies package syntax,
CLI health output, MCP initialization, MCP `tools/list`, and npm packaging. It
does not verify DeXposure-Bench execution because benchmark execution is not
implemented in this package yet.



## Citation

If you use this code in your research, please cite:

```
@techreport{shu2026dexposure,
  title={DeXposure-Claw: An Agentic System for DeFi Risk Supervision},
  author={Shu, Aijie and Chen, Bowei and Wu, Wenbin and Chen, Cathy Yi-Hsuan and He, Fengxiang},
  year={2026},
  institution={arXiv. org}
}

