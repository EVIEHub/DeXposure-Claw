from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cloud" / "train_src"))

from dexposure_fm.network_statistics import (  # noqa: E402
    compute_all_network_statistics,
    degree_assortativity,
    sector_connectivity_matrix,
)


def test_empty_graph_assortativity_is_undefined():
    edge_index = np.empty((2, 0), dtype=np.int64)
    assert np.isnan(degree_assortativity(edge_index, num_nodes=2))


def test_network_statistics_rejects_out_of_range_edges():
    with pytest.raises(ValueError, match="out-of-range"):
        compute_all_network_statistics(
            np.array([[0], [2]], dtype=np.int64), num_nodes=2
        )


def test_sector_matrix_rejects_missing_weights():
    with pytest.raises(ValueError, match="edge_weights length"):
        sector_connectivity_matrix(
            np.array([[0], [1]], dtype=np.int64),
            np.array([], dtype=float),
            ["A", "B"],
            ["A", "B"],
        )


def test_scenario_events_do_not_snap_to_an_unrelated_split():
    source = (ROOT / "cloud" / "train_src" / "run_full_experiment.py").read_text()
    assert "find_nearest_date" not in source
    assert "if event.event_date not in label_map" in source


def test_per_horizon_encoder_reset_requires_an_exact_state_match():
    source = (ROOT / "cloud" / "train_src" / "run_full_experiment.py").read_text()
    assert "encoder.load_state_dict(base_encoder_state, strict=True)" in source


def test_cloud_manifest_binds_primary_to_main_2025_h4():
    import json

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    primary = manifest["bindings"][manifest["primary"]["binding"]]
    assert primary["command"] == "bash cloud/main2025_run.sh"
    assert primary["horizons"] == [4]
    assert primary["holdout_start"] == "2025-01-01"
    assert primary["val_weeks"] == 24
    assert primary["epochs"] == 20
    assert primary["seed"] == 42
    assert primary.get("requires_pi_min") is not True


def test_cloud_source_digest_covers_executable_imports_and_sync_policy():
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    source = manifest["source"]
    assert ".skyignore" in source["paths"]
    assert "paper/experiments/**/*.py" in source["paths"]
    assert "paper/scripts/**/*.py" in source["paths"]
    expanded = {
        path.relative_to(ROOT).as_posix()
        for path in preflight._expand_source_paths(ROOT, source["paths"])
    }
    for relative in (
        "paper/experiments/exp_logger.py",
        "paper/experiments/methods.py",
        "paper/experiments/exceptions.py",
        "paper/scripts/build_paired_frontier.py",
        ".skyignore",
    ):
        assert relative in expanded

    skyignore = (ROOT / ".skyignore").read_text().splitlines()
    for generated in ("logs/", "output/", "paper/results/", "results/"):
        assert generated in skyignore

    for path in (ROOT / "cloud").glob("*"):
        if path.suffix not in {".sh", ".yaml"}:
            continue
        assert "uv sync --frozen || uv sync" not in path.read_text()


def test_fm_service_bindings_require_bounded_pi_min(monkeypatch):
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    for name in ("pre2022_crisis_h4", "pre2022_rq1", "llm_pilot"):
        assert manifest["bindings"][name]["requires_pi_min"] is True
    monkeypatch.setenv("PREFLIGHT_PI_MIN", "0.25")
    _, binding = preflight._validate_binding(
        manifest, "bash cloud/pre2022_run.sh"
    )
    assert binding["requires_pi_min"] is True
    monkeypatch.setenv("PREFLIGHT_PI_MIN", "nan")
    with pytest.raises(preflight.PreflightError, match="finite float"):
        preflight._validate_binding(manifest, "bash cloud/pre2022_run.sh")


def test_cloud_preflight_rejects_direct_all_horizon_command():
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    with pytest.raises(preflight.PreflightError, match="allowlisted"):
        preflight._binding_for_command(
            manifest,
            "python cloud/train_src/run_full_experiment.py --horizons 1,4,8,12",
        )


def test_cloud_preflight_disallows_independent_m7_sampling(monkeypatch):
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    bad = dict(manifest["bindings"]["llm_pilot"])
    bad["command"] = "bash cloud/llm_pilot_run.sh --method m7_fm_llm_gated"
    bad["enabled"] = True
    manifest["bindings"]["bad_m7"] = bad
    monkeypatch.setenv("PREFLIGHT_PI_MIN", "0.5")
    with pytest.raises(preflight.PreflightError, match="independent M7"):
        preflight._validate_binding(
            manifest,
            "bash cloud/llm_pilot_run.sh --method m7_fm_llm_gated",
        )


def test_cloud_scripts_keep_autodown_and_required_upload():
    train = (ROOT / "cloud" / "train.sh").read_text()
    task = (ROOT / "cloud" / "train.yaml").read_text()
    assert "--down" in train
    assert "cloud/preflight.py" in train
    assert "--phase host" in task
    assert "hf upload losdwind/graph-dexposure-ckpt" in task
    assert "HF_TOKEN is required" in task


@pytest.mark.parametrize(
    "entrypoint",
    ("bootstrap_and_run.sh", "launch_verified_gpu.sh"),
)
def test_legacy_gpu_entrypoints_are_retired_fail_closed_shims(entrypoint):
    source = (ROOT / "cloud" / entrypoint).read_text()
    assert "cloud/train.sh" in source
    assert "retired" in source
    assert "sky launch" not in source
    assert "sky exec" not in source
    assert "train_nosync.yaml" not in source


def test_training_bindings_require_only_raw_data_and_base_checkpoint():
    import json

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    assert "pre2022_smoke_2week" not in manifest["bindings"]
    assert "pre2022_smoke_2week" not in manifest["outputs"]
    assert set(manifest["inputs"]) == {"local"}
    for binding in manifest["bindings"].values():
        assert binding["required_inputs"] == ["local"]
    task = (ROOT / "cloud" / "train.yaml").read_text()
    assert "dexposure-fm-release" not in task
    assert not (ROOT / "cloud" / "train_nosync.yaml").exists()


def test_rq1_binding_requires_explicit_opt_in_with_source_attestation(monkeypatch, tmp_path):
    import hashlib
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    digest = preflight.source_digest(ROOT, manifest)
    attestation = tmp_path / "source-digest.txt"
    attestation.write_text(digest + "\n")
    monkeypatch.setenv("PREFLIGHT_RQ1", "1")
    monkeypatch.setenv("PREFLIGHT_PI_MIN", "0.5")
    monkeypatch.setenv("PREFLIGHT_SOURCE_ATTESTATION", str(attestation))
    monkeypatch.setenv(
        "PREFLIGHT_SOURCE_ATTESTATION_SHA256",
        hashlib.sha256(attestation.read_bytes()).hexdigest(),
    )
    preflight._validate_source_state(ROOT, manifest, "local")
    name, binding = preflight._validate_binding(
        manifest, "bash cloud/pre2022_run.sh --rq1-all-horizons"
    )
    assert name == "pre2022_rq1"
    assert binding["horizons"] == [1, 4, 8, 12]

    attestation.write_text("0" * 64 + "\n")
    monkeypatch.setenv(
        "PREFLIGHT_SOURCE_ATTESTATION_SHA256",
        hashlib.sha256(attestation.read_bytes()).hexdigest(),
    )
    with pytest.raises(preflight.PreflightError, match="does not bind"):
        preflight._validate_source_state(ROOT, manifest, "local")


def test_legacy_llm_pilot_is_blocked_before_cloud_launch():
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    assert manifest["bindings"]["llm_pilot"]["enabled"] is False
    with pytest.raises(preflight.PreflightError, match="disabled"):
        preflight._validate_binding(manifest, "bash cloud/llm_pilot_run.sh")


def test_cloud_output_contracts_name_exact_json_artifacts_and_minimum_counts():
    import json

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    crisis_json = {
        "logs/crisis_backtest/b1_crisis.json",
        "logs/crisis_backtest/b5_crisis.json",
        "logs/crisis_backtest/prompts_m6_fm_llm_2022-04__2022-07.json",
        "logs/crisis_backtest/prompts_m6_fm_llm_2022-10__2023-01.json",
        "logs/crisis_backtest/prompts_m6_fm_llm_2023-01__2023-05.json",
    }
    training_json = {
        "checkpoints/pre2022_train/finetuned/feature_schema.json",
        "checkpoints/pre2022_train/finetuned/experiment_results.json",
        "checkpoints/pre2022_train/finetuned/data_quality.json",
        "checkpoints/pre2022_train/finetuned/metrics.json",
        "checkpoints/pre2022_train/finetuned/all_results.json",
    }
    for name in ("pre2022_crisis_h4", "pre2022_rq1"):
        output_paths = {item["path"] for item in manifest["outputs"][name]}
        contract = manifest["bindings"][name]["json_contract"]
        assert set(contract["required_files"]) == crisis_json | training_json
        assert crisis_json | training_json <= output_paths
        assert contract["minimum_count"] >= len(crisis_json | training_json)

    main = manifest["bindings"]["main2025_h4"]["json_contract"]
    main_paths = {item["path"] for item in manifest["outputs"]["main2025_h4"]}
    assert set(main["required_files"]) <= main_paths
    assert "checkpoints/main2025_train/finetuned/best_model_h4.pt" in main_paths
    assert "checkpoints/main2025/dexposure-fm-h4.pt" in main_paths
    assert main["minimum_count"] >= len(main["required_files"])

    main_script = (ROOT / "cloud" / "main2025_run.sh").read_text()
    crisis_script = (ROOT / "cloud" / "pre2022_run.sh").read_text()
    assert 'src="checkpoints/main2025_train/finetuned/best_model_h4.pt"' in main_script
    assert 'src="checkpoints/pre2022_train/finetuned/best_model_h${h}.pt"' in crisis_script
    assert "find checkpoints/main2025_train" not in main_script
    assert "find checkpoints/pre2022_train" not in crisis_script
    assert "rm -rf checkpoints/pre2022_train" not in crisis_script
    assert "rm -f checkpoints/graphpfn-v1.ckpt" not in crisis_script
    assert "paper/results/llm_eval_*" not in crisis_script


def test_crisis_output_contract_requires_complete_aligned_horizon_grid():
    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    horizons = [1, 4, 8, 12]
    rows = [
        {
            "crisis": crisis,
            "method": method,
            "prediction_horizon": horizon,
            "label_horizon": horizon,
        }
        for crisis in ("terra_luna", "ftx", "svb")
        for method in ("m5_fm_rules", "m1_persistence_rules")
        for horizon in horizons
    ]
    preflight._verify_crisis_horizon_rows(rows, horizons)

    mismatched = [dict(row) for row in rows]
    mismatched[0]["label_horizon"] = 4
    with pytest.raises(preflight.PreflightError, match="horizon mismatch"):
        preflight._verify_crisis_horizon_rows(mismatched, horizons)

    with pytest.raises(preflight.PreflightError, match="grid is incomplete"):
        preflight._verify_crisis_horizon_rows(rows[:-1], horizons)


def test_cloud_json_verifier_fails_when_required_artifact_is_absent(tmp_path, monkeypatch):
    import json

    sys.path.insert(0, str(ROOT / "cloud"))
    import preflight

    manifest = json.loads((ROOT / "cloud" / "preflight_manifest.json").read_text())
    monkeypatch.setattr(
        preflight,
        "_import_verifiers",
        lambda *_args: (lambda *_values: "hash", lambda *_values: None, lambda value: (value, 0)),
    )
    with pytest.raises(preflight.PreflightError, match="required JSON artifact is missing"):
        preflight._verify_json_artifacts(tmp_path, "pre2022_crisis_h4", manifest)


def test_cloud_json_verifier_binds_raw_outputs_and_frozen_evidence_to_candidates(tmp_path):
    sys.path.insert(0, str(ROOT / "cloud"))
    sys.path.insert(0, str(ROOT / "paper"))
    import preflight
    from experiments.b5_decision_quality import CandidateSet, evidence_hash

    candidates = CandidateSet.from_rows(
        [
            {
                "protocol": f"P{index}",
                "exposure": 10 - index,
                "scenario_max_loss": index,
                "scenario_sum_loss": index,
            }
            for index in range(10)
        ]
    )
    system_prompt = "system"
    user_prompt = "user"
    bundle = {
        "evidence_sha256": evidence_hash(system_prompt, user_prompt, candidates),
        "candidate_set_sha256": candidates.candidate_set_sha256,
        "candidate_membership_sha256": candidates.candidate_membership_sha256,
        "candidate_set": candidates.as_dicts(),
    }
    output = {
        "risk_level": "elevated",
        "rationale": "Seven ranked candidates.",
        "target_protocols": [
            {
                "protocol": f"P{index}",
                "risk_score": 1.0 - index / 10,
                "action": "Investigate",
                "reason": "Visible candidate evidence.",
            }
            for index in range(7)
        ],
    }
    logs = tmp_path / "logs"
    logs.mkdir()
    raw_path = logs / "raw_m6_fm_llm.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "method": "m6_fm_llm",
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "evidence_bundle": bundle,
                    "raw_llm_outputs": [output],
                }
            ]
        )
    )
    frozen_path = logs / "frozen_evidence.json"
    frozen_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "evidence": {
                            **bundle,
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                        }
                    }
                ]
            }
        )
    )
    manifest = {
        "bindings": {
            "test": {
                "json_contract": {
                    "required_files": [
                        "logs/raw_m6_fm_llm.json",
                        "logs/frozen_evidence.json",
                    ],
                    "minimum_count": 2,
                }
            }
        },
        "verifiers": {
            "evidence": "module:evidence_hash",
            "candidate": "module:validate_decision_output",
            "gate": "module:apply_action_gate",
        },
    }
    summary = preflight._verify_json_artifacts(tmp_path, "test", manifest)
    assert summary["candidates"] == 1
    assert summary["gates"] == 1

    tampered = json.loads(raw_path.read_text())
    tampered[0]["evidence_bundle"]["candidate_membership_sha256"] = "0" * 64
    raw_path.write_text(json.dumps(tampered))
    with pytest.raises(preflight.PreflightError, match="candidate-membership hash mismatch"):
        preflight._verify_json_artifacts(tmp_path, "test", manifest)
