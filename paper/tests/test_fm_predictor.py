from __future__ import annotations

import json
import hashlib
import math

import pytest
import torch

from dexposure_agent.fm_predictor import (
    FMPredictor,
    _NodeHeadWrapper,
    _extract_encoder_state_dict,
    _prepare_for_inference,
)
from dexposure_agent.data_loader import preprocessing_contract_hash
from dexposure_agent.types import Edge, GraphSnapshot, NodeFeatures


def _schema(categories: list[str]) -> dict:
    return {
        "schema_version": 1,
        "metadata_sha256": "test",
        "feature_order": [
            "log_size",
            "num_tokens",
            "max_share",
            "entropy",
            "category_one_hot",
            "prev_tvl_change",
            "in_degree_norm",
            "out_degree_norm",
        ],
        "categories": categories,
        "edge_weight_space": "log1p",
        "node_target_space": "log1p_delta",
    }


@pytest.fixture
def predictor(tmp_path):
    (tmp_path / "feature_schema.json").write_text(
        json.dumps(_schema(["Bridge", "Lending", "Unknown"]))
    )
    return FMPredictor(checkpoint_dir=str(tmp_path), device="cpu")


def test_feature_builder_matches_training_order(predictor):
    graph = GraphSnapshot(
        date="2025-01-01",
        nodes={
            "a": NodeFeatures(
                log_size=1.0,
                num_tokens=2,
                max_share=0.7,
                entropy=0.5,
                category="Lending",
                prev_tvl_change=0.25,
            ),
            "b": NodeFeatures(
                log_size=2.0,
                num_tokens=1,
                max_share=1.0,
                entropy=0.0,
                category="not-in-schema",
                prev_tvl_change=-0.1,
            ),
        },
        edges=[Edge(source="a", target="b", weight=10.0)],
    )
    features = predictor._build_features(graph, ["a", "b"])
    assert features.shape == (2, 10)
    assert features[0].tolist() == pytest.approx(
        [1.0, 2.0, 0.7, 0.5, 0.0, 1.0, 0.0, 0.25, 0.0, 1.0]
    )
    assert features[1, 4:7].tolist() == [0.0, 0.0, 1.0]
    assert features[1, 8:].tolist() == [1.0, 0.0]


def test_weight_decoder_round_trip_and_residual(predictor):
    assert predictor._decode_weight(99.0, 0.0) == pytest.approx(99.0)
    assert predictor._decode_weight(99.0, math.log(2.0)) == pytest.approx(199.0)


def test_candidate_pairs_are_deterministic(predictor):
    first = predictor._get_candidate_pairs([0, 1], [1, 2], n=4, max_pairs=8)
    second = predictor._get_candidate_pairs([0, 1], [1, 2], n=4, max_pairs=8)
    assert first == second


def test_h12_requires_a_dedicated_checkpoint(predictor):
    with pytest.raises(FileNotFoundError, match="dexposure-fm-h12.pt"):
        predictor._load_model(12)


def test_missing_checkpoint_directory_does_not_fall_back_to_persistence(
    tmp_path,
):
    predictor = FMPredictor(checkpoint_dir=str(tmp_path / "missing"), device="cpu")
    graph = GraphSnapshot(date="2025-01-01", nodes={}, edges=[])
    with pytest.raises(RuntimeError, match="No usable FM checkpoints"):
        predictor.predict(graph, horizon=4)


def test_schema_without_checkpoint_is_not_reported_available(predictor):
    assert predictor.available is False
    assert predictor.available_horizons == []


def test_checkpoint_directory_is_not_reported_as_available_horizon(tmp_path):
    (tmp_path / "feature_schema.json").write_text(
        json.dumps(_schema(["Lending", "Unknown"]))
    )
    (tmp_path / "dexposure-fm-h12.pt").mkdir()
    predictor = FMPredictor(checkpoint_dir=str(tmp_path), device="cpu")

    assert predictor.available is False
    assert 12 not in predictor.available_horizons


def test_new_schema_rejects_incompatible_preprocessing_contract(tmp_path):
    schema = _schema(["Lending", "Unknown"])
    schema["schema_version"] = 2
    schema["preprocessing_contract_hash"] = "wrong"
    (tmp_path / "feature_schema.json").write_text(json.dumps(schema))
    with pytest.raises(ValueError, match="preprocessing contract hash"):
        FMPredictor(checkpoint_dir=str(tmp_path), device="cpu")


def test_new_schema_accepts_current_preprocessing_contract(tmp_path):
    schema = _schema(["Lending", "Unknown"])
    schema["schema_version"] = 2
    schema["preprocessing_contract_hash"] = preprocessing_contract_hash()
    (tmp_path / "feature_schema.json").write_text(json.dumps(schema))
    predictor = FMPredictor(checkpoint_dir=str(tmp_path), device="cpu")
    assert predictor.checkpoint_identity["legacy_schema"] is False


def test_authoritative_predictor_requires_bound_data_and_metadata(tmp_path):
    with pytest.raises(ValueError, match="both data_path and metadata_path"):
        FMPredictor(
            checkpoint_dir=str(tmp_path),
            device="cpu",
            require_provenance=True,
        )


def test_authoritative_predictor_rejects_legacy_checkpoint(tmp_path):
    (tmp_path / "feature_schema.json").write_text(
        json.dumps(_schema(["Bridge", "Lending", "Unknown"]))
    )
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "meta.csv").write_text("id,category\na,Lending\n")
    torch.save({"model": {}}, tmp_path / "dexposure-fm-h1.pt")
    predictor = FMPredictor(
        checkpoint_dir=str(tmp_path),
        device="cpu",
        data_path=str(tmp_path / "data.json"),
        metadata_path=str(tmp_path / "meta.csv"),
        require_provenance=True,
    )
    with pytest.raises(ValueError, match="lacks required provenance"):
        predictor._load_model(1)


def _authoritative_provenance_fixture(tmp_path):
    schema = _schema(["Bridge", "Lending", "Unknown"])
    schema["schema_version"] = 2
    schema["preprocessing_contract_hash"] = preprocessing_contract_hash()
    data_path = tmp_path / "data.json"
    metadata_path = tmp_path / "meta.csv"
    data_path.write_text("{}")
    metadata_path.write_text("id,category\na,Lending\n")
    schema["metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    (tmp_path / "feature_schema.json").write_text(json.dumps(schema))
    predictor = FMPredictor(
        checkpoint_dir=str(tmp_path),
        device="cpu",
        data_path=str(data_path),
        metadata_path=str(metadata_path),
        require_provenance=True,
    )
    provenance = {
        "provenance_version": 1,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "metadata_sha256": schema["metadata_sha256"],
        "source": {
            "git_commit": "commit",
            "dirty_patch_sha256": "b" * 64,
            "source_file_sha256": "c" * 64,
            "source_digest_sha256": "a" * 64,
        },
        "split": {"train": [], "val": [], "test": []},
        "seed": 42,
        "epochs": 1,
        "completed_epochs": 1,
        "horizon": 1,
        "preprocessing_contract_hash": schema["preprocessing_contract_hash"],
        "best_validation_metric": 0.5,
        "best_validation_metric_name": "auprc",
    }
    return predictor, provenance


def test_authoritative_provenance_rejects_source_digest_mismatch(tmp_path, monkeypatch):
    predictor, provenance = _authoritative_provenance_fixture(tmp_path)
    monkeypatch.setenv("PREFLIGHT_SOURCE_SHA256", "b" * 64)
    with pytest.raises(ValueError, match="does not match"):
        predictor._validate_provenance(provenance, horizon=1)


def test_authoritative_provenance_accepts_current_source_digest(tmp_path, monkeypatch):
    predictor, provenance = _authoritative_provenance_fixture(tmp_path)
    monkeypatch.setenv("PREFLIGHT_SOURCE_SHA256", "a" * 64)
    predictor._validate_provenance(provenance, horizon=1)


def test_authoritative_provenance_rejects_incomplete_training(tmp_path, monkeypatch):
    predictor, provenance = _authoritative_provenance_fixture(tmp_path)
    provenance["completed_epochs"] = 2
    monkeypatch.setenv("PREFLIGHT_SOURCE_SHA256", "a" * 64)
    with pytest.raises(ValueError, match="epoch counts"):
        predictor._validate_provenance(provenance, horizon=1)


@pytest.mark.parametrize("pi_min", [-0.1, 1.1, float("nan")])
def test_invalid_probability_threshold_is_rejected(tmp_path, pi_min):
    with pytest.raises(ValueError, match="pi_min"):
        FMPredictor(checkpoint_dir=str(tmp_path), device="cpu", pi_min=pi_min)


def test_node_head_uses_training_softmax_edge_aggregation():
    linear = torch.nn.Linear(3, 1, bias=False)
    linear.weight.data.fill_(1.0)
    wrapper = _NodeHeadWrapper(torch.nn.Sequential(linear))
    h = torch.tensor([[1.0], [2.0]])
    edge_src = torch.tensor([0, 1])
    edge_dst = torch.tensor([1, 0])
    edge_weight = torch.tensor([0.0, math.log(3.0)])
    actual = wrapper(h, edge_src, edge_dst, edge_weight)
    normalized = torch.softmax(edge_weight, dim=0)
    expected = torch.tensor(
        [
            1.0 + 2.0 * normalized[1] + 2.0 * normalized[0],
            2.0 + 1.0 * normalized[0] + 1.0 * normalized[1],
        ]
    )
    assert torch.allclose(actual, expected)


def test_encoder_checkpoint_mapping_preserves_graphpfn_tfm_namespace():
    tensor = torch.ones(1)
    mapped = _extract_encoder_state_dict(
        {
            "encoder.tfm.encoder_x.weight": tensor,
            "link_scorer.exist_head.weight": tensor,
        }
    )
    assert mapped == {"tfm.encoder_x.weight": tensor}


def test_encoder_checkpoint_mapping_rejects_missing_encoder():
    with pytest.raises(ValueError, match="valid GraphPFN encoder"):
        _extract_encoder_state_dict({"link_scorer.weight": torch.ones(1)})


def test_inference_module_disables_dropout_and_gradients():
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    prepared = _prepare_for_inference(module, torch.device("cpu"))
    assert prepared.training is False
    assert all(parameter.requires_grad is False for parameter in prepared.parameters())
    x = torch.ones(2, 2)
    assert torch.equal(prepared(x), prepared(x))
