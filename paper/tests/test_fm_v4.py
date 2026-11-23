from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn as nn

import pytest
from cloud.train_src import train_fm_v4 as v4_train
from dexposure_agent.fm_v4 import (
    DeXposureFMV4,
    HistoricalSpotPriceOracle,
    TokenMapperV3,
    V4_HORIZONS,
    build_netflow_snapshot_from_holdings,
    enrich_netflow_snapshots,
    revalue_snapshot_at_ref_date,
)


class _DummyV4Encoder(nn.Module):
    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.tfm = type("TFM", (), {"embed_dim": embed_dim})()
        self.proj = nn.Linear(91, embed_dim)


def test_spot_price_oracle_covers_all_283_dates_and_strips_price_beta() -> None:
    oracle = HistoricalSpotPriceOracle()
    assert len(oracle.prices_by_date) == 283
    for d, pmap in oracle.prices_by_date.items():
        assert pmap.get("ETH", 0.0) > 0.0, d
        assert pmap.get("BTC", 0.0) > 0.0, d

    # Stablecoins keep 1.0 multiplier across crash weeks (preserving real depegs/redemptions)
    assert oracle.price_multiplier("USDC", "2021-11-08", "2022-11-14") == 1.0
    assert oracle.price_multiplier("UST", "2022-05-09", "2022-05-16") == 1.0

    # ETH-beta assets scale by exact ETH spot ratio
    m_eth = oracle.price_multiplier("WSTETH", "2021-11-08", "2022-11-14")
    expected_eth = oracle.prices_by_date["2022-11-14"]["ETH"] / oracle.prices_by_date["2021-11-08"]["ETH"]
    assert abs(m_eth - expected_eth) < 1e-6


def test_netflow_samples_strip_price_crash_while_catching_real_run_and_keep_monotonicity() -> None:
    mapper = TokenMapperV3()
    oracle = HistoricalSpotPriceOracle()
    dates = sorted(oracle.prices_by_date.keys())[:18]

    snaps = []
    for idx, dt in enumerate(dates):
        eth_p = oracle.prices_by_date[dt]["ETH"]
        eth_p0 = oracle.prices_by_date[dates[0]]["ETH"]
        price_ratio = eth_p / eth_p0
        real_run_decay = 0.85 ** idx  # 15%/week real redemption on MakerDAO/USDe, 0% on Lido WETH
        holdings = {
            "182": {"WETH": 1.0e9 * price_ratio},  # Pure ETH price movement, ZERO physical redemption!
            "1599": {
                "WETH": 5.0e8 * price_ratio,       # Zero WETH redemption
                "USDC": 4.0e8,
                "DAI": 3.0e8 * real_run_decay,     # Real 15%/wk DAI redemption
            },
            "118": {"USDC": 6.0e8 * real_run_decay, "WETH": 2.0e8 * price_ratio},
            "4133": {"USDT": 5.0e8 * real_run_decay},
        }
        snaps.append(build_netflow_snapshot_from_holdings(dt, holdings, mapper))

    enrich_netflow_snapshots(snaps, mapper.categories, oracle=oracle)

    # Verify Lido ("182") has 0.0% constant-price Net-Flow change despite ETH price swings!
    tvl_const, _, _, _ = revalue_snapshot_at_ref_date(snaps[4], snaps[0].date, oracle)
    assert abs(tvl_const["182"] - snaps[0].tvl_usd[snaps[0].node_index["182"]]) < 1e-2

    samples = v4_train.build_joint_netflow_samples(
        snaps, [0, 1, 2, 3], max_target_idx=17, horizons=list(V4_HORIZONS), neg_ratio=2, seed=42, oracle=oracle
    )
    assert len(samples) == 4
    s0 = samples[0]
    lido_idx = s0.node_ids.index("182")
    maker_idx = s0.node_ids.index("118")
    # Lido has 0% real redemption => y_node_stress == 0 across all horizons!
    assert (s0.y_node_stress[lido_idx] == 0.0).all()
    # MakerDAO has 15%/wk real redemption => y_node_stress == 1 across all horizons!
    assert (s0.y_node_stress[maker_idx] == 1.0).all()

    for s in samples:
        assert bool(s.valid_horizons.all())
        assert ((s.y_retain_frac[:, :-1] - s.y_retain_frac[:, 1:]) >= -1e-6).all()
        assert ((s.y_node_stress[:, 1:] - s.y_node_stress[:, :-1]) >= -1e-6).all()

    enc = _DummyV4Encoder(embed_dim=32)
    model = DeXposureFMV4(encoder=enc, embed_dim=32, hidden_dim=64)
    model.encode = lambda src, dst, feat, dev: model.encoder.proj(feat.to(dev))  # type: ignore[method-assign]

    calib = v4_train.fit_monotonic_calibration_on_val(model, samples, torch.device("cpu"))
    assert calib["retain_nll_after"] <= calib["retain_nll_before"] + 1e-5
    t1, v4m = v4_train.evaluate_joint_model(model, samples, list(V4_HORIZONS), torch.device("cpu"))
    for h in V4_HORIZONS:
        assert v4m[f"h{h}"]["monotonicity_violation_rate"] == 0.0
        assert f"h{h}" in t1


def test_fm_v4_preflight_binding_validates() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / "cloud" / "preflight_manifest.json"
    if not manifest_path.exists():
        pytest.skip("cloud preflight manifest not included in code export")
    from cloud import preflight
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "main2025_fm_v4" in manifest["bindings"]

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
            train_cmd="bash cloud/fm_v4_run.sh",
        )
        assert res["binding"] == "main2025_fm_v4"
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
