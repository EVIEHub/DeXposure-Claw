# DeXposure-Claw — FinNLP 2026 workshop code

Code snapshot for **DeXposure-Claw: An Agent System for DeFi Risk Supervision**, FinNLP 2026 submission 134.

## Version and provenance

- Source commit: `22ccb4014d7a969f75844581d2666b6eb08f918d` (August 10, 2026).
- Release tag: `finnlp-2026-code-snapshot-v1`.
- Citable code snapshot: https://github.com/EVIEHub/DeXposure-Claw/tree/finnlp-2026-code-snapshot-v1
- The existing default branch is independent of this historical export.
- Paper record: https://openreview.net/forum?id=rgYYIpRPHL
- This is a code-only export of that local commit. It has **not** been verified as byte-identical to the anonymous repository used during review, or as the exact code that produced every reported result.
- `SOURCE_MANIFEST.json` records the source Git blob and published SHA-256 for each exported file. Two personal absolute paths in a standalone loader example were replaced with relative data paths; the manifest records this change.

## Contents

- `paper/dexposure_agent/`: exposure-graph processing, forecasting interface, monitoring, stress scenarios, ticket decisions, and API.
- `paper/experiments/` and `paper/scripts/`: benchmark and evaluation code.
- `paper/tests/`: tests from the source snapshot.
- `claw/`: CLI, MCP interface, and runtime adapters.
- `lib/`: model and graph utilities, including vendored components.
- `cloud/train_src/`: model experiment code. Cloud provisioning scripts are not included.
- `pyproject.toml` and `uv.lock`: dependency definitions from the source snapshot.

Datasets, checkpoints, recorded results, manuscript files, private notes, credentials, and the original Git history are not included. Reproducing paper numbers requires the corresponding data, weights, inference settings, and result provenance in addition to this code. Historical scripts may reference files excluded from this export.

## Checks

The lightweight CLI can be checked without the GPU environment:

```sh
PYTHONPATH=claw/src python3 -m dexposure_claw.cli health
python3 verify_snapshot.py
```

The original research environment specifies Python 3.12.9 and CUDA-specific dependencies; see `pyproject.toml`. With the research dependencies available:

```sh
PYTHONPATH=paper:claw/src python -m pytest -q paper/tests/test_config.py paper/tests/test_data_health.py paper/tests/test_types.py paper/tests/test_monitor.py paper/tests/test_pred_graph.py paper/tests/test_dexposure_claw.py
```

Before publication, these selected tests passed (56 tests), and the CLI health check returned `ok`. This does not validate paper results or a GPU rerun. The broader historical integrity suite was also attempted with the system Python; it reported six errors due to missing dependencies and excluded manuscript/result files. The full suite is not claimed to pass in this code-only export.

## Evidence boundary

The paper evaluates offline agreement with an author-designed future exposure-decline proxy. The snapshot does not establish independently validated supervisory decisions or safe live deployment. Retained historical ticket traces do not validate every proposed data-health and confidence-gate field.

## License

See `LICENSE`, `NOTICE`, and the licenses included with vendored components. Original notices are preserved.
