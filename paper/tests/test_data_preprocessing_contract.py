from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from dexposure_agent.data_loader import (
    build_feature_matrix,
    preprocessing_contract_hash,
    raw_to_graph_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def training_module():
    path = ROOT / "cloud" / "train_src" / "run_full_experiment.py"
    spec = importlib.util.spec_from_file_location("run_full_experiment_contract", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_and_inference_contract_hashes_match(training_module):
    assert training_module.preprocessing_contract_hash() == preprocessing_contract_hash()


def test_training_and_inference_features_match_after_edge_coalescing(training_module):
    categories = ["Lending", "Unknown"]
    category_to_idx = {name: i for i, name in enumerate(categories)}
    metadata = {"a": "Lending"}
    previous = {
        "nodes": [
            {"id": "a", "size": 10.0},
            {"id": "b", "size": 20.0},
        ],
        "links": [{"source": "a", "target": "b", "size": 1.0}],
    }
    current = {
        "nodes": [
            {"id": "a", "size": 20.0},
            {"id": "b", "size": 20.0},
            {"id": None, "size": 3.0},
        ],
        "links": [
            {"source": "a", "target": "b", "size": 1.0},
            {"source": "a", "target": "b", "size": 2.0},
        ],
    }
    train = training_module.enrich_snapshots_with_historical_features(
        [
            training_module.build_snapshot(
                "2025-01-01", previous, metadata, category_to_idx, categories
            ),
            training_module.build_snapshot(
                "2025-01-08", current, metadata, category_to_idx, categories
            ),
        ]
    )[1]
    inference = raw_to_graph_snapshot(
        "2025-01-08", current, metadata, previous_sizes={"a": 10.0, "b": 20.0}
    )

    assert train["source_node_records_invalid"] == 1
    assert inference.source_node_records_invalid == 1
    assert train["edge_src"].tolist() == [0]
    assert train["edge_dst"].tolist() == [1]
    assert train["edge_weight"].tolist() == pytest.approx([3.0])
    assert np.array_equal(train["features"], build_feature_matrix(inference, categories))


def test_new_checkpoint_provenance_contains_required_identity(tmp_path, training_module):
    data_path = tmp_path / "data.json"
    metadata_path = tmp_path / "meta.csv"
    data_path.write_text("{}")
    metadata_path.write_text("id,category\na,Lending\n")
    schema = training_module.build_feature_schema(str(metadata_path), ["Lending", "Unknown"])
    provenance = training_module.build_checkpoint_provenance(
        data_path=str(data_path),
        metadata_path=str(metadata_path),
        split={"train": ["2025-01-01"], "val": ["2025-01-08"], "test": ["2025-01-15"]},
        seed=42,
        epochs=3,
        completed_epochs=2,
        horizon=1,
        feature_schema=schema,
        best_validation_metric=0.75,
        best_validation_metric_name="auprc",
    )
    assert {
        "data_sha256",
        "metadata_sha256",
        "source",
        "split",
        "seed",
        "epochs",
        "horizon",
        "preprocessing_contract_hash",
        "best_validation_metric",
    } <= provenance.keys()
    assert provenance["preprocessing_contract_hash"] == schema["preprocessing_contract_hash"]
    assert provenance["source"]["dirty_patch_sha256"]
    assert provenance["source"]["source_digest_sha256"] is None
    assert provenance["source"]["source_digest_diagnostic"] is True


def test_checkpoint_provenance_records_manifest_bound_source_digest(
    tmp_path, training_module, monkeypatch
):
    data_path = tmp_path / "data.json"
    metadata_path = tmp_path / "meta.csv"
    data_path.write_text("{}")
    metadata_path.write_text("id,category\na,Lending\n")
    monkeypatch.setenv("PREFLIGHT_SOURCE_SHA256", "A" * 64)
    schema = training_module.build_feature_schema(str(metadata_path), ["Lending", "Unknown"])
    provenance = training_module.build_checkpoint_provenance(
        data_path=str(data_path),
        metadata_path=str(metadata_path),
        split={"train": ["2025-01-01"], "val": [], "test": []},
        seed=42,
        epochs=1,
        completed_epochs=1,
        horizon=1,
        feature_schema=schema,
        best_validation_metric=0.5,
        best_validation_metric_name="auprc",
    )
    assert provenance["source"]["preflight_source_sha256"] == "a" * 64
    assert provenance["source"]["source_digest_sha256"] == "a" * 64
    assert provenance["source"]["source_digest_kind"] == "preflight_manifest"
    assert provenance["source"]["source_digest_diagnostic"] is False


def test_invalid_preflight_source_digest_fails_closed(training_module, monkeypatch, tmp_path):
    monkeypatch.setenv("PREFLIGHT_SOURCE_SHA256", "not-a-digest")
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "meta.csv").write_text("id,category\na,Lending\n")
    with pytest.raises(ValueError, match="PREFLIGHT_SOURCE_SHA256"):
        training_module.build_checkpoint_provenance(
            data_path=str(tmp_path / "data.json"),
            metadata_path=str(tmp_path / "meta.csv"),
            split={"train": [], "val": [], "test": []},
            seed=42,
            epochs=1,
            completed_epochs=0,
            horizon=1,
            feature_schema={"preprocessing_contract_hash": "x"},
            best_validation_metric=0.5,
            best_validation_metric_name="auprc",
        )
