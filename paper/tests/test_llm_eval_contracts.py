from __future__ import annotations

from experiments.b5_decision_quality import CandidateSet, evidence_hash
from experiments.llm_eval_b5 import (
    ACTION_GATE_POLICY,
    SYSTEM_PROMPT,
    apply_action_gate,
    assess_week,
    billable_request_attempts,
    build_candidate_set,
    build_prompt,
    forecast_sha256,
    response_usage,
    recorded_candidate_set,
    run_pipeline,
    summarize_attempt_usage,
)


def test_forecast_hash_is_canonical_and_rejects_nonfinite_drift():
    left = {
        "date": "2025-01-01",
        "nodes": {"P0": {"score": float("nan"), "label": "Alpha"}},
        "edges": [{"source": "P0", "target": "P0", "weight": 10.0}],
    }
    right = {
        "edges": [{"weight": 10.0, "target": "P0", "source": "P0"}],
        "nodes": {"P0": {"label": "Alpha", "score": None}},
        "date": "2025-01-01",
    }
    assert forecast_sha256(left, 4) == forecast_sha256(right, 4)
    assert forecast_sha256(left, 4) != forecast_sha256(right, 8)


def test_raw_prompt_does_not_embed_m7_thresholds():
    assert "Recommend-Reduce requires" not in SYSTEM_PROMPT
    assert "Contingency requires" not in SYSTEM_PROMPT
    assert ACTION_GATE_POLICY["Recommend-Reduce"]["min_score"] == 0.75


def test_m7_gate_is_posthoc_copy_and_preserves_raw_ticket():
    raw = {
        "risk_level": "moderate",
        "rationale": "Evidence is available.",
        "target_protocols": [
            {
                "protocol": "Alpha",
                "risk_score": 0.80,
                "action": "Recommend-Reduce",
                "reason": "Visible exposure 100.",
            }
        ],
    }
    gated = apply_action_gate(raw)
    assert raw["target_protocols"][0]["action"] == "Recommend-Reduce"
    assert gated["target_protocols"][0]["action"] == "Investigate"
    assert gated["target_protocols"][0]["risk_score"] < 0.75


def test_m7_gate_recalibrates_demoted_score_and_preserves_target_set():
    raw = {
        "risk_level": "elevated",
        "rationale": "Elevated risk.",
        "target_protocols": [
            {
                "protocol": "P1",
                "risk_score": 0.95,
                "action": "Contingency",
                "reason": "weight 500000000.00",
            },
            {
                "protocol": "P2",
                "risk_score": 0.85,
                "action": "Recommend-Reduce",
                "reason": "expected_exposure_loss 300000000.00",
            },
            {
                "protocol": "P3",
                "risk_score": 0.50,
                "action": "Monitor",
                "reason": "weight 100000000.00",
            },
        ],
    }
    gated = apply_action_gate(raw)
    # P2 (surviving Recommend-Reduce) should now rank #1 ahead of demoted P1
    assert gated["target_protocols"][0]["protocol"] == "P2"
    assert gated["target_protocols"][0]["action"] == "Recommend-Reduce"
    assert gated["target_protocols"][1]["protocol"] == "P1"
    assert gated["target_protocols"][1]["action"] == "Investigate"
    assert gated["target_protocols"][2]["protocol"] == "P3"
    assert {t["protocol"] for t in gated["target_protocols"]} == {"P1", "P2", "P3"}


def test_fir_fields_are_separate_and_null_without_interventions():
    output = {
        "risk_level": "moderate",
        "rationale": "No intervention is required.",
        "target_protocols": [
            {
                "protocol": "Alpha",
                "risk_score": 0.50,
                "action": "Investigate",
                "reason": "Visible exposure 100.",
            }
        ],
    }
    metrics = assess_week(
        output,
        {"Alpha"},
        {"Alpha": 0.1},
        "exposure=100",
        positive_loss_protocols={"Alpha"},
    )
    assert metrics["tail_false_intervention_rate"] is None
    assert metrics["non_loss_false_intervention_rate"] is None


def test_candidate_set_is_top10_graph_protocols_and_prompt_visible():
    data = {
        "nodes": {f"P{i}": {"category": "Lending"} for i in range(10)},
        "edges": [
            {"source": f"P{i}", "target": f"P{i}", "weight": 100 - i}
            for i in range(10)
        ],
    }
    scenarios = [
        {
            "scenario_id": "S1",
            "scenario_name": "shock",
            "system_loss_pct": 10.0,
            "distressed_count": 1,
            "affected_nodes": 1,
            "top_affected": [
                {"protocol": "P0", "category": "Lending", "loss": 20.0},
                {"protocol": "SCENARIO_ONLY", "category": "Bridge", "loss": 99.0},
            ],
        }
    ]
    candidate_set = build_candidate_set(data, scenarios)
    assert len(candidate_set.rows) == 10
    system, user = build_prompt(
        "m6_fm_llm",
        "2025-01-01",
        4,
        {**data, "n_nodes": 10, "n_edges": 10},
        None,
        {"N1_max_pagerank": 0.1},
        scenarios,
        candidate_set=candidate_set,
    )
    assert all(protocol in user for protocol, *_ in candidate_set.rows)
    assert "SCENARIO_ONLY" not in user
    assert candidate_set.candidate_set_sha256 == build_candidate_set(data, scenarios).candidate_set_sha256


def test_m7_is_replay_only_and_cannot_be_sampled():
    try:
        run_pipeline(["m7_fm_llm_gated"])
    except ValueError as exc:
        assert "replay-only" in str(exc)
    else:
        raise AssertionError("run_pipeline must reject independent M7 sampling")


def test_unknown_attempt_usage_does_not_become_zero_cost():
    usage = summarize_attempt_usage(
        [
            {
                "usage_known": True,
                "input_tokens": 100,
                "output_tokens": 20,
            },
            {
                "usage_known": False,
                "input_tokens": 0,
                "output_tokens": 0,
            },
        ],
        {"input": 5.0, "output": 25.0},
    )
    assert usage["known_cost_usd"] == 0.001
    assert usage["unknown_usage_attempts"] == 1
    assert usage["cost_usd"] is None


def test_provider_usage_is_known_only_with_both_explicit_token_counts():
    assert response_usage({}) == (0, 0, False)
    assert response_usage({"usage": {"prompt_tokens": 10}}) == (0, 0, False)
    assert response_usage(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    ) == (10, 2, True)


def test_nested_retry_attempts_are_billed_without_double_counting_response():
    record = {
        "usage_known": True,
        "input_tokens": 10,
        "output_tokens": 2,
        "attempts": [
            {"usage_known": False, "input_tokens": 0, "output_tokens": 0},
            {"usage_known": True, "input_tokens": 10, "output_tokens": 2},
        ],
    }
    attempts = billable_request_attempts(record)
    assert len(attempts) == 2
    usage = summarize_attempt_usage(attempts, {"input": 5.0, "output": 25.0})
    assert usage["input_tokens"] == 10
    assert usage["unknown_usage_attempts"] == 1
    assert usage["cost_usd"] is None


def test_persisted_prompt_requires_matching_typed_candidate_hashes():
    candidates = CandidateSet.from_rows(
        [{"protocol": "P0", "exposure": 10.0}]
    )
    entry = {
        "date": "2025-01-01",
        "system_prompt": "system",
        "user_prompt": "user",
        "evidence_bundle": {
            "candidate_set": candidates.as_dicts(),
            "candidate_set_sha256": candidates.candidate_set_sha256,
            "candidate_membership_sha256": candidates.candidate_membership_sha256,
            "evidence_sha256": evidence_hash("system", "user", candidates),
        },
    }
    assert recorded_candidate_set(entry) == candidates
    entry["evidence_bundle"]["candidate_membership_sha256"] = "0" * 64
    try:
        recorded_candidate_set(entry)
    except ValueError as exc:
        assert "membership" in str(exc)
    else:
        raise AssertionError("tampered CandidateSet membership must be rejected")


def test_fm_forecast_candidate_set_prioritizes_node_drawdown_over_static_hubs():
    nodes = {
        "WETH": {"category": "Unknown", "log_size": 0.0, "prev_tvl_change": 0.0},
    }
    edges = [
        {"source": "WETH", "target": "WETH", "weight": 5e9},
    ]
    for i in range(12):
        pid = f"P{i}"
        # P0..P9 have predicted drawdown; P10..P11 are growing despite larger static weight
        delta = -0.20 + i * 0.01 if i < 10 else 0.15
        w = 2e7 + i * 1e6 if i < 10 else 5e8
        nodes[pid] = {
            "category": "Lending",
            "log_size": 18.0,
            "prev_tvl_change": delta,
        }
        edges.append({"source": pid, "target": "WETH", "weight": w})

    forecast = {
        "date": "2025-06-30",
        "horizon": 4,
        "nodes": nodes,
        "edges": edges,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
    }
    scenarios = [
        {
            "scenario_id": "S1",
            "scenario_name": "shock",
            "system_loss_pct": 10.0,
            "distressed_count": 1,
            "affected_nodes": 1,
            "top_affected": [
                {"protocol": "P0", "category": "Lending", "loss": 1e7},
            ],
        }
    ]
    candidates = build_candidate_set(forecast, scenarios)
    selected = {row[0] for row in candidates.rows}
    assert selected == {f"P{i}" for i in range(10)}
    assert "WETH" not in selected
    assert "P10" not in selected
    _, user = build_prompt(
        "m6_fm_llm",
        "2025-06-30",
        4,
        forecast,
        None,
        {"N1_max_pagerank": 0.1},
        scenarios,
        candidate_set=candidates,
    )
    assert "expected_exposure_loss:" in user
    assert "predicted_tvl_drop_pct:" in user
    assert "P0 (Lending)" in user
