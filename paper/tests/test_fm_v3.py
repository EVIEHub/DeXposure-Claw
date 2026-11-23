from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn as nn

import pytest
from cloud.train_src import train_fm_v3 as v3_train
from dexposure_agent.fm_v3 import (
    DeXposureFMV3,
    TokenMapperV3,
    V3_HORIZONS,
    build_stock_snapshot_from_holdings,
    enrich_stock_snapshots,
)


class _DummyEncoder(nn.Module):
    """Lightweight deterministic graph-free stub encoder for fast unit testing."""

    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.tfm = type("TFM", (), {"embed_dim": embed_dim})()
        self.proj = nn.Linear(87, embed_dim)


def test_token_mapper_v3_eliminates_dirty_symbol_collisions() -> None:
    mapper = TokenMapperV3()
    proto_tvls = {
        "182": 35_000_000_000.0,
        "118": 7_000_000_000.0,
        "4133": 10_000_000_000.0,
        "2626": 11_000_000_000.0,
        "2196": 2_900_000.0,  # Uniswap V1
    }
    tok_totals = {"UNI": 300_000_000.0, "WSTETH": 18_000_000_000.0}

    assert mapper.resolve_token_issuer("UNI", protocol_tvls=proto_tvls, token_totals=tok_totals) == "UNI"
    assert mapper.resolve_token_issuer("AAVE", protocol_tvls=proto_tvls, token_totals=tok_totals) == "AAVE"
    assert mapper.resolve_token_issuer("BTC", protocol_tvls=proto_tvls, token_totals=tok_totals) == "WBTC"
    assert mapper.resolve_token_issuer("ETH", protocol_tvls=proto_tvls, token_totals=tok_totals) == "WETH"
    assert mapper.resolve_token_issuer("SHIB", protocol_tvls=proto_tvls, token_totals=tok_totals) == "SHIB"
    assert mapper.resolve_token_issuer("WSTETH", protocol_tvls=proto_tvls, token_totals=tok_totals) == "182"
    assert mapper.resolve_token_issuer("USDE", protocol_tvls=proto_tvls, token_totals=tok_totals) == "USDE"
    assert mapper.resolve_token_issuer("SUSDE", protocol_tvls=proto_tvls, token_totals=tok_totals) == "4133"
    assert mapper.resolve_token_issuer("WEETH", protocol_tvls=proto_tvls, token_totals=tok_totals) == "2626"
    assert mapper.resolve_token_issuer("DAI", protocol_tvls=proto_tvls, token_totals=tok_totals) == "118"


def test_joint_samples_and_monotonic_calibration() -> None:
    mapper = TokenMapperV3()
    snaps = []
    for w in range(16):
        decay = 0.85 ** w
        holdings = {
            "1599": {"WETH": 5e8, "USDC": 4e8, "WSTETH": 3e8 * decay, "USDE": 2e8},
            "4025": {"USDC": 2e8, "WSTETH": 1.5e8, "WEETH": 1.2e8 * decay},
            "182": {"WETH": 1.2e9},
            "4133": {"USDT": 6e8, "WETH": 4e8 * decay},
            "2626": {"WETH": 8e8},
        }
        snaps.append(build_stock_snapshot_from_holdings(f"2025-01-{w+1:02d}", holdings, mapper))
    enrich_stock_snapshots(snaps, mapper.categories)

    samples = v3_train.build_joint_samples(
        snaps, [0, 1, 2, 3], max_target_idx=15, horizons=list(V3_HORIZONS), neg_ratio=2, seed=42
    )
    assert len(samples) == 4
    for s in samples:
        assert bool(s.valid_horizons.all())
        # Ground-truth cumulative survival must be monotonically non-increasing across horizons
        assert ((s.y_retain_active[:, :-1] - s.y_retain_active[:, 1:]) >= 0.0).all()
        assert ((s.y_node_stress[:, 1:] - s.y_node_stress[:, :-1]) >= 0.0).all()

    enc = _DummyEncoder(embed_dim=32)
    model = DeXposureFMV3(encoder=enc, embed_dim=32, hidden_dim=64)
    # Patch encode for lightweight CPU unit test without invoking 12-layer transformer
    model.encode = lambda src, dst, feat, dev: model.encoder.proj(feat.to(dev))  # type: ignore[method-assign]

    calib = v3_train.fit_monotonic_calibration_on_val(model, samples, torch.device("cpu"))
    assert calib["retain_nll_after"] <= calib["retain_nll_before"] + 1e-5
    t1, v3m = v3_train.evaluate_joint_model(model, samples, list(V3_HORIZONS), torch.device("cpu"))
    for h in V3_HORIZONS:
        assert v3m[f"h{h}"]["monotonicity_violation_rate"] == 0.0
        assert f"h{h}" in t1


def test_fm_v3_preflight_binding_validates() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "cloud" / "preflight_manifest.json"
    if not manifest_path.exists():
        pytest.skip("cloud preflight manifest not included in code export")
    from cloud import preflight
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "main2025_fm_v3" in manifest["bindings"]

    digest = preflight.source_digest(root, manifest)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
        tf.write(digest + "\n")
        att_path = Path(tf.name)
    try:
        att_sha = hashlib.sha256(att_path.read_bytes()).hexdigest()
        old_p = preflight.os.environ.get("PREFLIGHT_SOURCE_ATTESTATION")
        old_s = preflight.os.environ.get("PREFLIGHT_SOURCE_ATTESTATION_SHA256")
        preflight.os.environ["PREFLIGHT_SOURCE_ATTESTATION"] = str(att_path)
        preflight.os.environ["PREFLIGHT_SOURCE_ATTESTATION_SHA256"] = att_sha
        res = preflight.run_preflight(
            root=root,
            manifest_path=manifest_path,
            phase="local",
            train_cmd="bash cloud/fm_v3_run.sh",
        )
        assert res["binding"] == "main2025_fm_v3"
        assert res["horizons"] == [1, 4, 8, 12]
    finally:
        att_path.unlink(missing_ok=True)
        if old_p is None:
            preflight.os.environ.pop("PREFLIGHT_SOURCE_ATTESTATION", None)
        else:
            preflight.os.environ["PREFLIGHT_SOURCE_ATTESTATION"] = old_p
        if old_s is None:
            preflight.os.environ.pop("PREFLIGHT_SOURCE_ATTESTATION_SHA256", None)
        else:
            preflight.os.environ["PREFLIGHT_SOURCE_ATTESTATION_SHA256"] = old_s
