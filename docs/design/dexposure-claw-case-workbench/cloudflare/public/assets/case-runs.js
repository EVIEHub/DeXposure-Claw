// 2026 Live DeFiLlama Case Runs (Strictly recent 2026 walk-forward telemetry)
window.CASE_RUNS = Object.freeze({
  "live": {
    "name": "Live DeFiLlama (2026-11-05)",
    "slug": "live-defillama",
    "eventWindow": "2026-10-29..2026-11-05",
    "isLive": true,
    "drafterMode": "deterministic_candidate_ranker",
    "provenance": {
      "dataSource": "DeFiLlama Public REST API (api.llama.fi, top 1022 protocols)",
      "snapshotDate": "2026-11-05",
      "forecastDate": "2026-12-03",
      "horizonWeeks": 4,
      "fmCheckpoint": "checkpoints/dexposure-fm-release/dexposure-fm-h4.pt (3042.1 ms)",
      "backboneCheckpoint": "checkpoints/LimiX-16M.ckpt",
      "mcSamples": 50,
      "mcSigma": 2.0,
      "drafterModel": "deterministic-candidate-ranker (M=50)"
    },
    "review": {
      "date": "2026-11-05",
      "week": "live",
      "forecast": "2026-12-03",
      "nodes": 1029,
      "edges": 2966,
      "scenario": "S1",
      "loss": 15.09,
      "s3Loss": 4.76,
      "riskLevel": "elevated",
      "rationale": "Forecast-grounded candidate queue (m6): ranked by parameter-free expected exposure drawdown E[Delta W_u] and S1-S5 stress loss (worst scenario=15.09%).",
      "model": "deterministic-candidate-ranker (M=50)",
      "recordedCall": "1/1",
      "consistency": 1.0,
      "contractValid": true,
      "status": "ok",
      "dataHealth": 0.970225,
      "safeMode": false,
      "nAlerts": 3,
      "alerts": [
        {
          "horizon": 4,
          "metric_id": "N3",
          "metric_name": "Graph Density",
          "value": 0.002803900882198349,
          "baseline_mean": 0.004596279868256363,
          "baseline_std": 6.617432965403975e-06,
          "z_score": 270.8571428571463,
          "confidence": 0.9702253352878358,
          "attribution": {}
        },
        {
          "horizon": 4,
          "metric_id": "N4",
          "metric_name": "PageRank Concentration (Gini)",
          "value": 0.49951166659589863,
          "baseline_mean": 0.6801490064146689,
          "baseline_std": 0.06499860647439054,
          "z_score": 2.7790955778404483,
          "confidence": 0.9172552196470751,
          "attribution": {}
        },
        {
          "horizon": 4,
          "metric_id": "N5",
          "metric_name": "Gini Weighted Degree",
          "value": 0.9846648441408029,
          "baseline_mean": 0.9701172837939771,
          "baseline_std": 0.004846428996700569,
          "z_score": 3.001707103669471,
          "confidence": 0.9679637938110578,
          "attribution": {}
        }
      ],
      "meanConfidence": 0.951815,
      "confidenceGatePass": true,
      "demotedIds": [
        "182"
      ],
      "hits": 0,
      "precision": 0.0,
      "scenarios": [
        {
          "scenario_id": "S1",
          "scenario_name": "Top protocol failure",
          "system_loss_pct": 15.09,
          "distressed_count": 27,
          "affected_nodes": 144,
          "top_affected": [
            {
              "protocol": "182",
              "name": "Lido",
              "category": "Liquid Staking",
              "loss": 597871700.32
            },
            {
              "protocol": "2929",
              "name": "Spark",
              "category": "Lending",
              "loss": 324654622.14
            },
            {
              "protocol": "1599",
              "name": "AAVE V3",
              "category": "Lending",
              "loss": 68215557.21
            },
            {
              "protocol": "3107",
              "name": "EigenLayer",
              "category": "Restaking",
              "loss": 46138081.88
            },
            {
              "protocol": "4025",
              "name": "Morpho Blue",
              "category": "Lending",
              "loss": 42800586.87
            }
          ]
        },
        {
          "scenario_id": "S2",
          "scenario_name": "Bridge cluster failure",
          "system_loss_pct": 3.93,
          "distressed_count": 36,
          "affected_nodes": 149,
          "top_affected": [
            {
              "protocol": "4867",
              "name": "LayerZero",
              "category": "Bridge",
              "loss": 128323310.97
            },
            {
              "protocol": "USDT",
              "name": "USDT",
              "category": "Unknown",
              "loss": 102291024.15
            },
            {
              "protocol": "USDC",
              "name": "USDC",
              "category": "Unknown",
              "loss": 18185679.85
            },
            {
              "protocol": "6042",
              "name": "Lighter Bridge",
              "category": "Bridge",
              "loss": 16529285.3
            },
            {
              "protocol": "USDE",
              "name": "USDE",
              "category": "Unknown",
              "loss": 11475215.23
            }
          ]
        },
        {
          "scenario_id": "S3",
          "scenario_name": "Stablecoin de-peg",
          "system_loss_pct": 4.76,
          "distressed_count": 0,
          "affected_nodes": 192,
          "top_affected": [
            {
              "protocol": "118",
              "name": "MakerDAO",
              "category": "CDP",
              "loss": 185995067.33
            },
            {
              "protocol": "USDC",
              "name": "USDC",
              "category": "Unknown",
              "loss": 165161275.59
            },
            {
              "protocol": "182",
              "name": "Lido",
              "category": "Liquid Staking",
              "loss": 16083294.52
            },
            {
              "protocol": "2929",
              "name": "Spark",
              "category": "Lending",
              "loss": 1989847.28
            },
            {
              "protocol": "3",
              "name": "Curve DEX",
              "category": "Dexes",
              "loss": 1898525.46
            }
          ]
        },
        {
          "scenario_id": "S4",
          "scenario_name": "Lending sector shock",
          "system_loss_pct": 11.38,
          "distressed_count": 0,
          "affected_nodes": 220,
          "top_affected": [
            {
              "protocol": "1599",
              "name": "AAVE V3",
              "category": "Lending",
              "loss": 182623657.92
            },
            {
              "protocol": "182",
              "name": "Lido",
              "category": "Liquid Staking",
              "loss": 141571667.92
            },
            {
              "protocol": "2929",
              "name": "Spark",
              "category": "Lending",
              "loss": 103750917.5
            },
            {
              "protocol": "WETH",
              "name": "WETH",
              "category": "Unknown",
              "loss": 75572390.38
            },
            {
              "protocol": "USDC",
              "name": "USDC",
              "category": "Unknown",
              "loss": 68451632.66
            }
          ]
        },
        {
          "scenario_id": "S5",
          "scenario_name": "Correlated stress (top-10)",
          "system_loss_pct": 10.64,
          "distressed_count": 0,
          "affected_nodes": 232,
          "top_affected": [
            {
              "protocol": "USDC",
              "name": "USDC",
              "category": "Unknown",
              "loss": 162328039.5
            },
            {
              "protocol": "1599",
              "name": "AAVE V3",
              "category": "Lending",
              "loss": 121749105.28
            },
            {
              "protocol": "182",
              "name": "Lido",
              "category": "Liquid Staking",
              "loss": 119574340.06
            },
            {
              "protocol": "118",
              "name": "MakerDAO",
              "category": "CDP",
              "loss": 74398026.93
            },
            {
              "protocol": "2929",
              "name": "Spark",
              "category": "Lending",
              "loss": 69167278.33
            }
          ]
        }
      ],
      "candidates": [
        {
          "protocol": "182",
          "name": "Lido",
          "category": "Liquid Staking",
          "exposure": 1363782853.41,
          "expected_exposure_loss": 597844556.99,
          "predicted_tvl_drop_pct": -43.84,
          "scenario_max_loss": 597871700.32,
          "scenario_sum_loss": 878635697.22
        },
        {
          "protocol": "118",
          "name": "MakerDAO",
          "category": "CDP",
          "exposure": 442180324.5,
          "expected_exposure_loss": 114715486.76,
          "predicted_tvl_drop_pct": -25.94,
          "scenario_max_loss": 185995067.33,
          "scenario_sum_loss": 293103723.22
        },
        {
          "protocol": "2929",
          "name": "Spark",
          "category": "Lending",
          "exposure": 361970824.81,
          "expected_exposure_loss": 102425244.92,
          "predicted_tvl_drop_pct": -28.3,
          "scenario_max_loss": 324654622.14,
          "scenario_sum_loss": 499870060.92
        },
        {
          "protocol": "587",
          "name": "Maple",
          "category": "Uncollateralized Lending",
          "exposure": 146534661.59,
          "expected_exposure_loss": 67433661.36,
          "predicted_tvl_drop_pct": -46.02,
          "scenario_max_loss": 35000711.44,
          "scenario_sum_loss": 58366171.56
        },
        {
          "protocol": "1599",
          "name": "AAVE V3",
          "category": "Lending",
          "exposure": 608750030.02,
          "expected_exposure_loss": 67071034.04,
          "predicted_tvl_drop_pct": -11.02,
          "scenario_max_loss": 182623657.92,
          "scenario_sum_loss": 372786829.23
        },
        {
          "protocol": "711",
          "name": "Polymarket",
          "category": "Prediction Market",
          "exposure": 235631082.41,
          "expected_exposure_loss": 50537799.4,
          "predicted_tvl_drop_pct": -21.45,
          "scenario_max_loss": 47126216.48,
          "scenario_sum_loss": 47126216.48
        },
        {
          "protocol": "3107",
          "name": "EigenLayer",
          "category": "Restaking",
          "exposure": 208273067.23,
          "expected_exposure_loss": 49883326.36,
          "predicted_tvl_drop_pct": -23.95,
          "scenario_max_loss": 46138081.88,
          "scenario_sum_loss": 56116567.13
        },
        {
          "protocol": "7895",
          "name": "Armitage by Wintermute",
          "category": "Risk Curators",
          "exposure": 105744855.27,
          "expected_exposure_loss": 44455492.19,
          "predicted_tvl_drop_pct": -42.04,
          "scenario_max_loss": 21148971.05,
          "scenario_sum_loss": 21148971.05
        },
        {
          "protocol": "3780",
          "name": "Base Bridge",
          "category": "Chain",
          "exposure": 116904260.41,
          "expected_exposure_loss": 37294052.59,
          "predicted_tvl_drop_pct": -31.9,
          "scenario_max_loss": 23372338.72,
          "scenario_sum_loss": 25897864.18
        },
        {
          "protocol": "7594",
          "name": "Aave V4",
          "category": "Lending",
          "exposure": 87905314.53,
          "expected_exposure_loss": 34964677.9,
          "predicted_tvl_drop_pct": -39.78,
          "scenario_max_loss": 10489879.61,
          "scenario_sum_loss": 17483132.68
        }
      ],
      "queue": [
        {
          "position": 1,
          "score": 0.705,
          "rawScore": 0.94,
          "targetName": "Lido",
          "protocolId": "182",
          "entityId": "182",
          "sourceClass": "Liquid Staking",
          "exposure": 1363782853.41,
          "expectedLoss": 597844556.99,
          "dropPct": -43.84,
          "scenarioMaxLoss": 597871700.32,
          "scenarioSumLoss": 878635697.22,
          "ticketAction": "Investigate",
          "rawAction": "Contingency",
          "demoted": true,
          "gateNote": "Demoted from Contingency: Contingency requires critical overall risk and protocol risk_score >= 0.90.",
          "reason": "预计 4 周敞口回撤 USD 597,844,557 (-43.8%)，压力测试最大单情景损失 USD 597,871,700。",
          "inStressed": false,
          "actualLossPct": 0.0
        },
        {
          "position": 2,
          "score": 0.65,
          "rawScore": 0.65,
          "targetName": "Spark",
          "protocolId": "2929",
          "entityId": "2929",
          "sourceClass": "Lending",
          "exposure": 361970824.81,
          "expectedLoss": 102425244.92,
          "dropPct": -28.3,
          "scenarioMaxLoss": 324654622.14,
          "scenarioSumLoss": 499870060.92,
          "ticketAction": "Investigate",
          "rawAction": "Investigate",
          "demoted": false,
          "gateNote": "PASS",
          "reason": "预计 4 周敞口回撤 USD 102,425,245 (-28.3%)，压力测试最大单情景损失 USD 324,654,622。",
          "inStressed": false,
          "actualLossPct": 0.0
        },
        {
          "position": 3,
          "score": 0.63,
          "rawScore": 0.63,
          "targetName": "MakerDAO",
          "protocolId": "118",
          "entityId": "118",
          "sourceClass": "CDP",
          "exposure": 442180324.5,
          "expectedLoss": 114715486.76,
          "dropPct": -25.94,
          "scenarioMaxLoss": 185995067.33,
          "scenarioSumLoss": 293103723.22,
          "ticketAction": "Investigate",
          "rawAction": "Investigate",
          "demoted": false,
          "gateNote": "PASS",
          "reason": "预计 4 周敞口回撤 USD 114,715,487 (-25.9%)，压力测试最大单情景损失 USD 185,995,067。",
          "inStressed": false,
          "actualLossPct": 0.0
        },
        {
          "position": 4,
          "score": 0.57,
          "rawScore": 0.57,
          "targetName": "Maple",
          "protocolId": "587",
          "entityId": "587",
          "sourceClass": "Uncollateralized Lending",
          "exposure": 146534661.59,
          "expectedLoss": 67433661.36,
          "dropPct": -46.02,
          "scenarioMaxLoss": 35000711.44,
          "scenarioSumLoss": 58366171.56,
          "ticketAction": "Investigate",
          "rawAction": "Investigate",
          "demoted": false,
          "gateNote": "PASS",
          "reason": "预计 4 周敞口回撤 USD 67,433,661 (-46.0%)，压力测试最大单情景损失 USD 35,000,711。",
          "inStressed": false,
          "actualLossPct": 0.0
        },
        {
          "position": 5,
          "score": 0.55,
          "rawScore": 0.55,
          "targetName": "AAVE V3",
          "protocolId": "1599",
          "entityId": "1599",
          "sourceClass": "Lending",
          "exposure": 608750030.02,
          "expectedLoss": 67071034.04,
          "dropPct": -11.02,
          "scenarioMaxLoss": 182623657.92,
          "scenarioSumLoss": 372786829.23,
          "ticketAction": "Investigate",
          "rawAction": "Investigate",
          "demoted": false,
          "gateNote": "PASS",
          "reason": "预计 4 周敞口回撤 USD 67,071,034 (-11.0%)，压力测试最大单情景损失 USD 182,623,658。",
          "inStressed": false,
          "actualLossPct": 0.0
        },
        {
          "position": 6,
          "score": 0.51,
          "rawScore": 0.51,
          "targetName": "Polymarket",
          "protocolId": "711",
          "entityId": "711",
          "sourceClass": "Prediction Market",
          "exposure": 235631082.41,
          "expectedLoss": 50537799.4,
          "dropPct": -21.45,
          "scenarioMaxLoss": 47126216.48,
          "scenarioSumLoss": 47126216.48,
          "ticketAction": "Investigate",
          "rawAction": "Investigate",
          "demoted": false,
          "gateNote": "PASS",
          "reason": "预计 4 周敞口回撤 USD 50,537,799 (-21.4%)，压力测试最大单情景损失 USD 47,126,216。",
          "inStressed": false,
          "actualLossPct": 0.0
        },
        {
          "position": 7,
          "score": 0.5,
          "rawScore": 0.5,
          "targetName": "EigenLayer",
          "protocolId": "3107",
          "entityId": "3107",
          "sourceClass": "Restaking",
          "exposure": 208273067.23,
          "expectedLoss": 49883326.36,
          "dropPct": -23.95,
          "scenarioMaxLoss": 46138081.88,
          "scenarioSumLoss": 56116567.13,
          "ticketAction": "Investigate",
          "rawAction": "Investigate",
          "demoted": false,
          "gateNote": "PASS",
          "reason": "预计 4 周敞口回撤 USD 49,883,326 (-23.9%)，压力测试最大单情景损失 USD 46,138,082。",
          "inStressed": false,
          "actualLossPct": 0.0
        }
      ]
    }
  },
  "recent2026": {
    "name": "Live DeFiLlama (2026-11-05)",
    "slug": "live-defillama",
    "eventWindow": "2026-10-29..2026-11-05",
    "isLive": true,
    "drafterMode": "deterministic_candidate_ranker",
    "provenance": {
      "dataSource": "DeFiLlama Public REST API (api.llama.fi, top 1022 protocols)",
      "snapshotDate": "2026-11-05",
      "forecastDate": "2026-12-03",
      "horizonWeeks": 4,
      "fmCheckpoint": "checkpoints/dexposure-fm-release/dexposure-fm-h4.pt (3042.1 ms)",
      "backboneCheckpoint": "checkpoints/LimiX-16M.ckpt",
      "mcSamples": 50,
      "mcSigma": 2.0,
      "drafterModel": "deterministic-candidate-ranker (M=50)"
    },
    "review": {
      "date": "2026-11-05",
      "week": "live",
      "forecast": "2026-12-03",
      "nodes": 1029,
      "edges": 2966,
      "scenario": "S1",
      "loss": 15.09,
      "s3Loss": 4.76,
      "riskLevel": "elevated",
      "rationale": "Forecast-grounded candidate queue (m6): ranked by parameter-free expected exposure drawdown E[Delta W_u] and S1-S5 stress loss (worst scenario=15.09%).",
      "model": "deterministic-candidate-ranker (M=50)",
      "recordedCall": "1/1",
      "consistency": 1.0,
      "contractValid": true,
      "status": "ok",
      "dataHealth": 0.970225,
      "safeMode": false,
      "nAlerts": 3,
      "alerts": [
        {
          "horizon": 4,
          "metric_id": "N3",
          "metric_name": "Graph Density",
          "value": 0.002803900882198349,
          "baseline_mean": 0.004596279868256363,
          "baseline_std": 6.617432965403975e-06,
          "z_score": 270.8571428571463,
          "confidence": 0.9702253352878358,
          "attribution": {}
        },
        {
          "horizon": 4,
          "metric_id": "N4",
          "metric_name": "PageRank Concentration (Gini)",
          "value": 0.49951166659589863,
          "baseline_mean": 0.6801490064146689,
          "baseline_std": 0.06499860647439054,
          "z_score": 2.7790955778404483,
          "confidence": 0.9172552196470751,
          "attribution": {}
        },
        {
          "horizon": 4,
          "metric_id": "N5",
          "metric_name": "Gini Weighted Degree",
          "value": 0.9846648441408029,
          "baseline_mean": 0.9701172837939771,
          "baseline_std": 0.004846428996700569,
          "z_score": 3.001707103669471,
          "confidence": 0.9679637938110578,
          "attribution": {}
        }
      ],
      "meanConfidence": 0.951815,
      "confidenceGatePass": true,
      "demotedIds": [
        "182"
      ],
      "hits": 0,
      "precision": 0.0,
      "scenarios": [
        {
          "scenario_id": "S1",
          "scenario_name": "Top protocol failure",
          "system_loss_pct": 15.09,
          "distressed_count": 27,
          "affected_nodes": 144,
          "top_affected": [
            {
              "protocol": "182",
              "name": "Lido",
              "category": "Liquid Staking",
              "loss": 597871700.32
            },
            {
              "protocol": "2929",
              "name": "Spark",
              "category": "Lending",
              "loss": 324654622.14
            },
            {
              "protocol": "1599",
              "name": "AAVE V3",
              "category": "Lending",
              "loss": 68215557.21
            },
            {
              "protocol": "3107",
              "name": "EigenLayer",
              "category": "Restaking",
              "loss": 46138081.88
            },
            {
              "protocol": "4025",
              "name": "Morpho Blue",
              "category": "Lending",
              "loss": 42800586.87
            }
          ]
        },
        {
          "scenario_id": "S2",
          "scenario_name": "Bridge cluster failure",
          "system_loss_pct": 3.93,
          "distressed_count": 36,
          "affected_nodes": 149,
          "top_affected": [
            {
              "protocol": "4867",
              "name": "LayerZero",
              "category": "Bridge",
              "loss": 128323310.97
            },
            {
              "protocol": "USDT",
              "name": "USDT",
              "category": "Unknown",
              "loss": 102291024.15
            },
            {
              "protocol": "USDC",
              "name": "USDC",
              "category": "Unknown",
              "loss": 18185679.85
            },
            {
              "protocol": "6042",
              "name": "Lighter Bridge",
              "category": "Bridge",
              "loss": 16529285.3
            },
            {
              "protocol": "USDE",
              "name": "USDE",
              "category": "Unknown",
              "loss": 11475215.23
            }
          ]
        },
        {
          "scenario_id": "S3",
          "scenario_name": "Stablecoin de-peg",
          "system_loss_pct": 4.76,
          "distressed_count": 0,
          "affected_nodes": 192,
          "top_affected": [
            {
              "protocol": "118",
              "name": "MakerDAO",
              "category": "CDP",
              "loss": 185995067.33
            },
            {
              "protocol": "USDC",
              "name": "USDC",
              "category": "Unknown",
              "loss": 165161275.59
            },
            {
              "protocol": "182",
              "name": "Lido",
              "category": "Liquid Staking",
              "loss": 16083294.52
            },
            {
              "protocol": "2929",
              "name": "Spark",
              "category": "Lending",
              "loss": 1989847.28
            },
            {
              "protocol": "3",
              "name": "Curve DEX",
              "category": "Dexes",
              "loss": 1898525.46
            }
          ]
        },
        {
          "scenario_id": "S4",
          "scenario_name": "Lending sector shock",
          "system_loss_pct": 11.38,
          "distressed_count": 0,
          "affected_nodes": 220,
          "top_affected": [
            {
              "protocol": "1599",
              "name": "AAVE V3",
              "category": "Lending",
              "loss": 182623657.92
            },
            {
              "protocol": "182",
              "name": "Lido",
              "category": "Liquid Staking",
              "loss": 141571667.92
            },
            {
              "protocol": "2929",
              "name": "Spark",
              "category": "Lending",
              "loss": 103750917.5
            },
            {
              "protocol": "WETH",
              "name": "WETH",
              "category": "Unknown",
              "loss": 75572390.38
            },
            {
              "protocol": "USDC",
              "name": "USDC",
              "category": "Unknown",
              "loss": 68451632.66
            }
          ]
        },
        {
          "scenario_id": "S5",
          "scenario_name": "Correlated stress (top-10)",
          "system_loss_pct": 10.64,
          "distressed_count": 0,
          "affected_nodes": 232,
          "top_affected": [
            {
              "protocol": "USDC",
              "name": "USDC",
              "category": "Unknown",
              "loss": 162328039.5
            },
            {
              "protocol": "1599",
              "name": "AAVE V3",
              "category": "Lending",
              "loss": 121749105.28
            },
            {
              "protocol": "182",
              "name": "Lido",
              "category": "Liquid Staking",
              "loss": 119574340.06
            },
            {
              "protocol": "118",
              "name": "MakerDAO",
              "category": "CDP",
              "loss": 74398026.93
            },
            {
              "protocol": "2929",
              "name": "Spark",
              "category": "Lending",
              "loss": 69167278.33
            }
          ]
        }
      ],
      "candidates": [
        {
          "protocol": "182",
          "name": "Lido",
          "category": "Liquid Staking",
          "exposure": 1363782853.41,
          "expected_exposure_loss": 597844556.99,
          "predicted_tvl_drop_pct": -43.84,
          "scenario_max_loss": 597871700.32,
          "scenario_sum_loss": 878635697.22
        },
        {
          "protocol": "118",
          "name": "MakerDAO",
          "category": "CDP",
          "exposure": 442180324.5,
          "expected_exposure_loss": 114715486.76,
          "predicted_tvl_drop_pct": -25.94,
          "scenario_max_loss": 185995067.33,
          "scenario_sum_loss": 293103723.22
        },
        {
          "protocol": "2929",
          "name": "Spark",
          "category": "Lending",
          "exposure": 361970824.81,
          "expected_exposure_loss": 102425244.92,
          "predicted_tvl_drop_pct": -28.3,
          "scenario_max_loss": 324654622.14,
          "scenario_sum_loss": 499870060.92
        },
        {
          "protocol": "587",
          "name": "Maple",
          "category": "Uncollateralized Lending",
          "exposure": 146534661.59,
          "expected_exposure_loss": 67433661.36,
          "predicted_tvl_drop_pct": -46.02,
          "scenario_max_loss": 35000711.44,
          "scenario_sum_loss": 58366171.56
        },
        {
          "protocol": "1599",
          "name": "AAVE V3",
          "category": "Lending",
          "exposure": 608750030.02,
          "expected_exposure_loss": 67071034.04,
          "predicted_tvl_drop_pct": -11.02,
          "scenario_max_loss": 182623657.92,
          "scenario_sum_loss": 372786829.23
        },
        {
          "protocol": "711",
          "name": "Polymarket",
          "category": "Prediction Market",
          "exposure": 235631082.41,
          "expected_exposure_loss": 50537799.4,
          "predicted_tvl_drop_pct": -21.45,
          "scenario_max_loss": 47126216.48,
          "scenario_sum_loss": 47126216.48
        },
        {
          "protocol": "3107",
          "name": "EigenLayer",
          "category": "Restaking",
          "exposure": 208273067.23,
          "expected_exposure_loss": 49883326.36,
          "predicted_tvl_drop_pct": -23.95,
          "scenario_max_loss": 46138081.88,
          "scenario_sum_loss": 56116567.13
        },
        {
          "protocol": "7895",
          "name": "Armitage by Wintermute",
          "category": "Risk Curators",
          "exposure": 105744855.27,
          "expected_exposure_loss": 44455492.19,
          "predicted_tvl_drop_pct": -42.04,
          "scenario_max_loss": 21148971.05,
          "scenario_sum_loss": 21148971.05
        },
        {
          "protocol": "3780",
          "name": "Base Bridge",
          "category": "Chain",
          "exposure": 116904260.41,
          "expected_exposure_loss": 37294052.59,
          "predicted_tvl_drop_pct": -31.9,
          "scenario_max_loss": 23372338.72,
          "scenario_sum_loss": 25897864.18
        },
        {
          "protocol": "7594",
          "name": "Aave V4",
          "category": "Lending",
          "exposure": 87905314.53,
          "expected_exposure_loss": 34964677.9,
          "predicted_tvl_drop_pct": -39.78,
          "scenario_max_loss": 10489879.61,
          "scenario_sum_loss": 17483132.68
        }
      ],
      "queue": [
        {
          "position": 1,
          "score": 0.705,
          "rawScore": 0.94,
          "targetName": "Lido",
          "protocolId": "182",
          "entityId": "182",
          "sourceClass": "Liquid Staking",
          "exposure": 1363782853.41,
          "expectedLoss": 597844556.99,
          "dropPct": -43.84,
          "scenarioMaxLoss": 597871700.32,
          "scenarioSumLoss": 878635697.22,
          "ticketAction": "Investigate",
          "rawAction": "Contingency",
          "demoted": true,
          "gateNote": "Demoted from Contingency: Contingency requires critical overall risk and protocol risk_score >= 0.90.",
          "reason": "预计 4 周敞口回撤 USD 597,844,557 (-43.8%)，压力测试最大单情景损失 USD 597,871,700。",
          "inStressed": false,
          "actualLossPct": 0.0
        },
        {
          "position": 2,
          "score": 0.65,
          "rawScore": 0.65,
          "targetName": "Spark",
          "protocolId": "2929",
          "entityId": "2929",
          "sourceClass": "Lending",
          "exposure": 361970824.81,
          "expectedLoss": 102425244.92,
          "dropPct": -28.3,
          "scenarioMaxLoss": 324654622.14,
          "scenarioSumLoss": 499870060.92,
          "ticketAction": "Investigate",
          "rawAction": "Investigate",
          "demoted": false,
          "gateNote": "PASS",
          "reason": "预计 4 周敞口回撤 USD 102,425,245 (-28.3%)，压力测试最大单情景损失 USD 324,654,622。",
          "inStressed": false,
          "actualLossPct": 0.0
        },
        {
          "position": 3,
          "score": 0.63,
          "rawScore": 0.63,
          "targetName": "MakerDAO",
          "protocolId": "118",
          "entityId": "118",
          "sourceClass": "CDP",
          "exposure": 442180324.5,
          "expectedLoss": 114715486.76,
          "dropPct": -25.94,
          "scenarioMaxLoss": 185995067.33,
          "scenarioSumLoss": 293103723.22,
          "ticketAction": "Investigate",
          "rawAction": "Investigate",
          "demoted": false,
          "gateNote": "PASS",
          "reason": "预计 4 周敞口回撤 USD 114,715,487 (-25.9%)，压力测试最大单情景损失 USD 185,995,067。",
          "inStressed": false,
          "actualLossPct": 0.0
        },
        {
          "position": 4,
          "score": 0.57,
          "rawScore": 0.57,
          "targetName": "Maple",
          "protocolId": "587",
          "entityId": "587",
          "sourceClass": "Uncollateralized Lending",
          "exposure": 146534661.59,
          "expectedLoss": 67433661.36,
          "dropPct": -46.02,
          "scenarioMaxLoss": 35000711.44,
          "scenarioSumLoss": 58366171.56,
          "ticketAction": "Investigate",
          "rawAction": "Investigate",
          "demoted": false,
          "gateNote": "PASS",
          "reason": "预计 4 周敞口回撤 USD 67,433,661 (-46.0%)，压力测试最大单情景损失 USD 35,000,711。",
          "inStressed": false,
          "actualLossPct": 0.0
        },
        {
          "position": 5,
          "score": 0.55,
          "rawScore": 0.55,
          "targetName": "AAVE V3",
          "protocolId": "1599",
          "entityId": "1599",
          "sourceClass": "Lending",
          "exposure": 608750030.02,
          "expectedLoss": 67071034.04,
          "dropPct": -11.02,
          "scenarioMaxLoss": 182623657.92,
          "scenarioSumLoss": 372786829.23,
          "ticketAction": "Investigate",
          "rawAction": "Investigate",
          "demoted": false,
          "gateNote": "PASS",
          "reason": "预计 4 周敞口回撤 USD 67,071,034 (-11.0%)，压力测试最大单情景损失 USD 182,623,658。",
          "inStressed": false,
          "actualLossPct": 0.0
        },
        {
          "position": 6,
          "score": 0.51,
          "rawScore": 0.51,
          "targetName": "Polymarket",
          "protocolId": "711",
          "entityId": "711",
          "sourceClass": "Prediction Market",
          "exposure": 235631082.41,
          "expectedLoss": 50537799.4,
          "dropPct": -21.45,
          "scenarioMaxLoss": 47126216.48,
          "scenarioSumLoss": 47126216.48,
          "ticketAction": "Investigate",
          "rawAction": "Investigate",
          "demoted": false,
          "gateNote": "PASS",
          "reason": "预计 4 周敞口回撤 USD 50,537,799 (-21.4%)，压力测试最大单情景损失 USD 47,126,216。",
          "inStressed": false,
          "actualLossPct": 0.0
        },
        {
          "position": 7,
          "score": 0.5,
          "rawScore": 0.5,
          "targetName": "EigenLayer",
          "protocolId": "3107",
          "entityId": "3107",
          "sourceClass": "Restaking",
          "exposure": 208273067.23,
          "expectedLoss": 49883326.36,
          "dropPct": -23.95,
          "scenarioMaxLoss": 46138081.88,
          "scenarioSumLoss": 56116567.13,
          "ticketAction": "Investigate",
          "rawAction": "Investigate",
          "demoted": false,
          "gateNote": "PASS",
          "reason": "预计 4 周敞口回撤 USD 49,883,326 (-23.9%)，压力测试最大单情景损失 USD 46,138,082。",
          "inStressed": false,
          "actualLossPct": 0.0
        }
      ]
    }
  }
});
