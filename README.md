# DeXposure-Claw

DeXposure-Claw is an agentic DeFi systemic-risk monitoring, graph forecasting, and counterfactual contagion workbench built on inter-protocol credit exposure networks.

> Research and supervisory monitoring software only. DeXposure-Claw does not execute trades, sign transactions, custody assets, or provide investment advice.

## Repository Structure

- `claw/`: Lightweight `dexposure-claw` CLI, stdio MCP server (`dexposure_health`, `dexposure_install_snippet`, `dexposure_list_benchmarks`, `dexposure_live_predict`), and runtime adapters for Claude Code, OpenAI Codex, and Hermes.
- `paper/dexposure_agent/`: Core risk-monitoring pipeline, live DeFiLlama exposure-network ingestion (`live_defillama.py`, `live_history.py`), DeXposure-FM v3/v4 inference (`fm_v3.py`, `fm_v4.py`), DebtRank-style contagion stress testing (`contagion.py`), confidence-gated supervisory decision logic (`decision.py`), and FastAPI services (`serve.py`, `container_service.py`).
- `paper/experiments/` & `paper/scripts/`: Benchmark suite (`B1`--`B6`), baseline implementations, walk-forward evaluation, and data-processing scripts.
- `paper/tests/`: Unit and integration tests for the CLI, MCP server, contagion simulator, token mapper, forecasting interfaces, and API endpoints.
- `lib/`: Graph foundation model backbones and temporal graph utilities (`graphpfn`, `limix`, `tfm`).
- `cloud/train_src/`: Model training, sequential rolling evaluation, and hyperparameter tuning code (`train_fm_v3.py`, `train_fm_v4.py`, `tune_fm.py`).
- `data/mapping/`: Protocol and token-issuer resolution tables used by live ingestion and historical replay.
- `checkpoints/`: Feature schemas, calibration summaries, and horizon configurations for DeXposure-FM (binary `.pt`/`.ckpt` weights are excluded from Git and loaded locally or from Hugging Face).
- `docs/design/dexposure-claw-case-workbench/`: Interactive browser workbench and Cloudflare Worker proxy for live exposure networks, historical crisis replay, and contagion stress simulation.

## Quick Start

### 1. Lightweight CLI & MCP Server (Zero Heavy Dependencies)

Requires Python 3.10+ (and Node.js 18+ if invoking via `npm`/`node`):

```bash
# Python entrypoint
PYTHONPATH=claw/src python3 -m dexposure_claw.cli health
PYTHONPATH=claw/src python3 -m dexposure_claw.cli mcp

# Node.js wrapper
node claw/bin/dexposure-claw.js health
npm --prefix claw test
```

Build the Claude Code plugin bundle or print MCP config snippets:

```bash
PYTHONPATH=claw/src python3 -m dexposure_claw.cli build claude-code
PYTHONPATH=claw/src python3 -m dexposure_claw.cli snippet codex
```

### 2. Full Forecasting, Live Pipeline & Workbench Server

Install Python dependencies (Python 3.12 recommended; see `pyproject.toml`):

```bash
uv sync
```

Run live DeFiLlama exposure ingestion and forecast-grounded analysis:

```bash
PYTHONPATH=paper:claw/src uv run python -m dexposure_claw.cli live-predict --top-n 30 --horizon 4
```

Launch the FastAPI workbench and contagion simulation server:

```bash
PYTHONPATH=paper:claw/src uv run python -m dexposure_claw.cli serve --host 127.0.0.1 --port 8000
```

### 3. Running Tests

```bash
# Lightweight CLI & MCP smoke tests
python3 -m unittest discover -s claw/tests

# Agent, mapper, contagion, and API unit tests
PYTHONPATH=paper:claw/src uv run pytest -q \
  paper/tests/test_config.py \
  paper/tests/test_types.py \
  paper/tests/test_data_health.py \
  paper/tests/test_monitor.py \
  paper/tests/test_pred_graph.py \
  paper/tests/test_contagion_zero_tvl.py \
  paper/tests/test_dexposure_claw.py
```

## License

Apache-2.0. See `LICENSE` and `NOTICE` (and component licenses under `lib/`) for details.
