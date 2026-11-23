"""Dashboard contagion and asset-consistency tests.

Run: .venv/bin/pytest paper/tests/test_dashboard_contagion.py
"""
import importlib.util
import json
import random
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
if str(PAPER) not in sys.path:
    sys.path.insert(0, str(PAPER))

from dexposure_agent.contagion import simulate_contagion_traced  # noqa: E402

WB = ROOT / "docs" / "design" / "dexposure-claw-case-workbench"
ASSETS = WB / "assets"


def _canonical():
    spec = importlib.util.spec_from_file_location(
        "mp_tools", ROOT / "cloud" / "train_src" / "dexposure_fm" / "macroprudential_tools.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.simulate_contagion


def _random_layered(seed: int, layers: int = 5, width: int = 8, m: int = 80):
    """Layered graph: creditors in layer k are exposed only to debtors in layer k+1.

    Shocks hit the last layer, so nodes active in the same round never are
    exposed to each other and the canonical set-iteration order cannot matter.
    """
    rng = random.Random(seed)
    nodes = {f"L{k}n{i}": {"tvlUsd": rng.choice([0.0, rng.uniform(1, 1000)])} for k in range(layers) for i in range(width)}
    edges = []
    for _ in range(m):
        k = rng.randrange(layers - 1)
        edges.append({"source": f"L{k}n{rng.randrange(width)}", "target": f"L{k+1}n{rng.randrange(width)}", "weight": rng.uniform(0.1, 50)})
    return {"nodes": nodes, "edges": edges}


def _load_graph_snap(predicted_h=None):
    g = json.loads((ASSETS / "live-contagion-graph.json").read_text())
    hs = [int(h) for h in g["horizons"]]
    if predicted_h is None:
        nodes = {n: {"tvlUsd": v["tvlUsd"] or 0.0} for n, v in g["nodes"].items()}
        edges = [{"source": e[0], "target": e[1], "weight": e[2]} for e in g["edges"]]
    else:
        i = 3 + hs.index(predicted_h)
        nodes = {n: {"tvlUsd": (v["predTvlUsd"] or {}).get(str(predicted_h)) or 0.0} for n, v in g["nodes"].items()}
        if g.get("weightMode") == "fractional_retention":
            edges = [
                {"source": e[0], "target": e[1], "weight": float(e[2]) * (1.0 if e[i] is None else float(e[i]))}
                for e in g["edges"]
                if float(e[2]) * (1.0 if e[i] is None else float(e[i])) >= 1.0
            ]
        else:
            edges = [
                {"source": e[0], "target": e[1], "weight": e[2]}
                for e in g["edges"]
                if e[i] is None or e[i] >= g["piMin"]
            ]
    return g, {"nodes": nodes, "edges": edges}


class TracedParityTests(unittest.TestCase):
    def test_matches_canonical_when_order_cannot_matter(self):
        canonical = _canonical()
        for seed in range(30):
            snap = _random_layered(seed)
            shocked = [k for k, v in snap["nodes"].items() if k.startswith("L4") and v["tvlUsd"] > 0][:3]
            for frac in (0.2, 0.5, 1.0):
                a = canonical(snap, shocked, shock_fraction=frac)
                b = simulate_contagion_traced(snap, shocked, shock_fraction=frac)
                self.assertAlmostEqual(a["total_loss"], b["total_loss"], places=6)
                self.assertEqual(sorted(a["distressed_nodes"]), b["distressed_nodes"])
                self.assertAlmostEqual(
                    b["direct_loss"] + sum(r["lossUsd"] for r in b["per_node"]), b["total_loss"], places=6
                )

    def test_traced_is_deterministic_on_dashboard_graph(self):
        # The canonical tool iterates a Python set, so on cyclic graphs its result
        # depends on PYTHONHASHSEED. The traced port sorts and must be repeatable.
        g, snap = _load_graph_snap(None)
        a = simulate_contagion_traced(snap, ["182"], shock_fraction=0.5)
        b = simulate_contagion_traced(snap, ["182"], shock_fraction=0.5)
        self.assertEqual(a["total_loss"], b["total_loss"])
        self.assertEqual(a["distressed_nodes"], b["distressed_nodes"])

    def test_js_port_matches_python(self):
        g, snap = _load_graph_snap(4)
        prot = [n for n, v in snap["nodes"].items() if n.isdigit() and v["tvlUsd"] > 0][:10]
        script = """
const fs = require('fs');
const { simulateContagionTraced } = require(process.argv[1]);
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = input.shocks.map(s => simulateContagionTraced(input.snap, [s], { shockFraction: 0.5 }));
process.stdout.write(JSON.stringify(out.map(r => [r.total_loss, r.distressed_nodes])));
"""
        res = subprocess.run(
            ["node", "-e", script, str(ASSETS / "contagion.js")],
            input=json.dumps({"snap": snap, "shocks": prot}),
            capture_output=True, text=True, check=True,
        )
        js = json.loads(res.stdout)
        for n, (tot, dist) in zip(prot, js):
            py = simulate_contagion_traced(snap, [n], shock_fraction=0.5)
            self.assertAlmostEqual(py["total_loss"], tot, delta=1e-6 * max(1.0, tot))
            self.assertEqual(py["distressed_nodes"], dist)

    def test_depeg_mode_python_js_parity_and_order_invariance(self):
        _, snap = _load_graph_snap(None)
        cases = [["USDC"], ["USDT"], ["USDC", "USDT"], ["USDT", "USDC"], ["USDE"]]
        script = """
const fs = require('fs');
const { simulateContagionTraced } = require(process.argv[1]);
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = input.cases.map(c => simulateContagionTraced(input.snap, c, { shockFraction: 0.2, shockMode: 'depeg' }));
process.stdout.write(JSON.stringify(out));
"""
        res = subprocess.run(
            ["node", "-e", script, str(ASSETS / "contagion.js")],
            input=json.dumps({"snap": snap, "cases": cases}),
            capture_output=True, text=True, check=True,
        )
        js_out = json.loads(res.stdout)
        py_out = [simulate_contagion_traced(snap, c, shock_fraction=0.2, shock_mode="depeg") for c in cases]
        for py, js in zip(py_out, js_out):
            self.assertGreater(py["direct_loss"], 0.0)
            self.assertAlmostEqual(py["direct_loss"] + py["contagion_loss"], py["total_loss"], places=6)
            self.assertAlmostEqual(py["total_loss"], js["total_loss"], delta=1e-6 * max(1.0, py["total_loss"]))
            self.assertAlmostEqual(py["direct_loss"], js["direct_loss"], delta=1e-6 * max(1.0, py["direct_loss"]))
            self.assertEqual(py["distressed_nodes"], js["distressed_nodes"])
            self.assertEqual([r["via"] for r in py["per_node"][:15]], [r["via"] for r in js["per_node"][:15]])
        # Order invariance: ["USDC", "USDT"] vs ["USDT", "USDC"]
        self.assertAlmostEqual(py_out[2]["total_loss"], py_out[3]["total_loss"], places=6)
        self.assertEqual(py_out[2]["distressed_nodes"], py_out[3]["distressed_nodes"])
        self.assertEqual([r["via"] for r in py_out[2]["per_node"]], [r["via"] for r in py_out[3]["per_node"]])


class AssetConsistencyTests(unittest.TestCase):
    def test_contagion_propagates_on_real_graph(self):
        g, snap = _load_graph_snap(None)
        prot = [n for n, v in g["nodes"].items() if n.isdigit() and (v["tvlUsd"] or 0) > 0]
        best = max(simulate_contagion_traced(snap, [n], shock_fraction=0.5)["contagion_loss"] for n in prot[:200])
        self.assertGreater(best, 0.0, "no protocol shock propagates; edge weights are probably not being read")

    def test_network_nodes_have_no_filled_defaults(self):
        net = json.loads((ASSETS / "live-network-3d.json").read_text())
        for h in ("h1", "h4", "h8"):
            nodes = net[h]["nodes"]
            self.assertGreater(len(nodes), 50)
            # The old builder injected these placeholders; they must never come back.
            self.assertFalse(any(n.get("topCounterpartyName") == "WETH / USDC" for n in nodes))
            by_id = {n["id"]: n for n in nodes}
            for did in net[h]["displayNodeIds"]:
                logo = by_id[did].get("logo")
                self.assertTrue(logo and logo.startswith("/assets/icons/"), f"missing local icon for {did}: {logo}")
                icon_file = WB / logo.lstrip("/")
                self.assertTrue(icon_file.is_file() and icon_file.stat().st_size > 100, f"missing icon file {icon_file}")
            for n in nodes:
                if n["hhi"] is not None:
                    self.assertTrue(0.0 < n["hhi"] <= 1.0 + 1e-9)
                if n["flowDropPct"] is not None:
                    self.assertTrue(0.0 <= n["flowDropPct"] <= 100.0 + 1e-9)
            for e in net[h]["displayEdges"]:
                self.assertIn("weightUsd", e)

    def test_queue_drop_matches_network_drop(self):
        """Network poolDropPct must agree with the pipeline's M7 queue dropPct (same FM formula)."""
        net = json.loads((ASSETS / "live-network-3d.json").read_text())
        hz = json.loads((ASSETS / "live-horizons.json").read_text())
        for h in ("1", "4", "8"):
            nodes = {n["id"]: n for n in net[f"h{h}"]["nodes"]}
            for q in hz[h]["workbench_case"]["review"]["queue"]:
                n = nodes.get(str(q["entityId"]))
                if n is None or q.get("dropPct") is None:
                    continue
                self.assertAlmostEqual(abs(float(q["dropPct"])), n["poolDropPct"], delta=0.5,
                                       msg=f"h={h} {q.get('targetName')}")

    def test_html_reads_existing_keys_and_has_no_hardcoded_results(self):
        html = (WB / "index.html").read_text()
        for stale in ("28/35", "80.0%", "42.6%", "30.6%", "$179.4B", "0.433", "historicalReplays",
                      "benchmark29w", "realizedChangePct", "predWeight", "h_scale"):
            self.assertNotIn(stale, html, f"stale/hardcoded token in index.html: {stale}")
        hist = json.loads((ASSETS / "live-history.json").read_text())
        for key in ("liveWalkForwardSummary", "paperBacktest29Weeks", "historicalCrisisReplays", "snapshots"):
            self.assertIn(key, hist)
            self.assertIn(key, html)
        ranking = next(s for s in hist["snapshots"] if s["backtestResolved"])["rankings"][0]
        for key in re.findall(r"r\.(actualChangePct|backtestHit|inStressed|realizedLossUsd|expectedLoss)\b", html):
            self.assertIn(key, ranking)
        cf = (WB / "cloudflare" / "public" / "index.html").read_text()
        self.assertEqual(html, cf, "cloudflare/public/index.html out of sync")


class DashboardSmokeTests(unittest.TestCase):
    def test_page_renders_without_undefined_or_nan(self):
        res = subprocess.run(
            ["node", str(Path(__file__).with_name("dashboard_smoke.js")), str(WB)],
            capture_output=True, text=True, check=True,
        )
        out = json.loads(res.stdout)
        self.assertEqual(out["bad"], [], out["bad"])
        wf = json.loads((ASSETS / "live-history.json").read_text())["liveWalkForwardSummary"]
        self.assertEqual(
            out["kpis"]["kWf@4"],
            f"{wf['m7StressedHits']}/{wf['m7StressedTotal']} vs {wf['m2StressedHits']}/{wf['m2StressedTotal']}",
        )
        self.assertTrue(
            "无告警触发" in out["kpis"]["kEdgesSub@4"] or "平均置信度" in out["kpis"]["kEdgesSub@4"],
            out["kpis"]["kEdgesSub@4"],
        )
        self.assertNotIn(out["kpis"]["depegObs@4"], ("—", "$0"))

    def test_api_contagion_simulate_supports_depeg_mode(self):
        from fastapi.testclient import TestClient
        from dexposure_agent.serve import create_app

        client = TestClient(create_app())
        r = client.post(
            "/api/contagion-simulate",
            json={"shocked_nodes": ["USDC", "USDT"], "shock_fraction": 0.2, "horizon": 4, "shock_mode": "depeg"},
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["shock_mode"], "depeg")
        self.assertGreater(body["observed"]["direct_loss"], 1e8)
        self.assertGreater(body["observed"]["contagion_loss"], 0.0)
        self.assertGreater(body["predicted"]["total_loss"], 1e8)

        ico = client.get("/assets/icons/182.webp")
        self.assertEqual(ico.status_code, 200)
        self.assertGreater(len(ico.content), 500)

        for route in ("/", "/user", "/pro", "/compliance"):
            resp = client.get(route)
            self.assertEqual(resp.status_code, 200, route)
            self.assertIn("风险速览", resp.text)
            self.assertIn("专业工作台", resp.text)


class OptionANetFlowTests(unittest.TestCase):
    def test_constant_price_holdings_strips_eth_btc_beta_and_keeps_redemptions_and_depegs(self):
        from dexposure_agent.live_defillama import compute_constant_price_holdings

        # t_0: ETH = $2,000, WBTC = $80,000, USDC = $1.00, USDE = $1.00
        usd_then = {
            "182": {"WETH": 20_000_000.0},          # 10,000 WETH
            "2": {"WBTC": 80_000_000.0},            # 1,000 WBTC
            "1599": {"WETH": 10_000_000.0, "USDC": 10_000_000.0},
            "4133": {"USDE": 50_000_000.0},         # 50M USDe
        }
        qty_then = {
            "182": {"WETH": 10_000.0},
            "2": {"WBTC": 1_000.0},
            "1599": {"WETH": 5_000.0, "USDC": 10_000_000.0},
            "4133": {"USDE": 50_000_000.0},
        }
        # t_1: ETH crashes -20% ($1,600), BTC crashes -10% ($72,000):
        #   - 182 (Lido): 0 WETH redeemed (still 10,000 WETH -> $16M raw USD, MUST be $20M constant-price!)
        #   - 2 (WBTC): 10% physical redemption (900 WBTC -> $64.8M raw USD, MUST be $72M = -10% constant-price!)
        #   - 1599 (Aave): 20% WETH redeemed (4,000 WETH -> $6.4M raw) + 0% USDC redeemed ($10M) -> $18M constant-price
        #   - 4133 (USDe depeg): 50M USDe depegs to $0.80 ($40M raw USD) -> MUST remain $40M (-20% depeg loss!)
        usd_real = {
            "182": {"WETH": 16_000_000.0},
            "2": {"WBTC": 64_800_000.0},
            "1599": {"WETH": 6_400_000.0, "USDC": 10_000_000.0},
            "4133": {"USDE": 40_000_000.0},
        }
        qty_real = {
            "182": {"WETH": 10_000.0},
            "2": {"WBTC": 900.0},
            "1599": {"WETH": 4_000.0, "USDC": 10_000_000.0},
            "4133": {"USDE": 50_000_000.0},
        }

        const_h, meta = compute_constant_price_holdings(usd_then, usd_real, qty_then, qty_real)
        self.assertAlmostEqual(meta["ethReturnPct"], -20.0, places=2)
        self.assertAlmostEqual(meta["btcReturnPct"], -10.0, places=2)
        # Zero redemption on Lido => 0% constant-price change despite -20% ETH drop!
        self.assertAlmostEqual(const_h["182"]["WETH"], 20_000_000.0, places=2)
        # 10% BTC redemption => -10% constant-price ($72M vs $80M)
        self.assertAlmostEqual(const_h["2"]["WBTC"], 72_000_000.0, places=2)
        # Aave: 4,000 WETH @ $2,000 ($8M) + $10M USDC = $18M
        self.assertAlmostEqual(const_h["1599"]["WETH"] + const_h["1599"]["USDC"], 18_000_000.0, places=2)
        # Stablecoin depeg preserved in full ($40M vs $50M)
        self.assertAlmostEqual(const_h["4133"]["USDE"], 40_000_000.0, places=2)

    def test_live_history_uses_constant_price_net_flow_settlement(self):
        hist = json.loads((ASSETS / "live-history.json").read_text())
        hz = json.loads((ASSETS / "live-horizons.json").read_text())
        self.assertEqual(hist.get("settlementBasis"), "constant_price_net_flow_ex_eth_btc_beta")
        self.assertIn("alphaWalkForwardSummary", hist)
        awf = hist["alphaWalkForwardSummary"]
        self.assertEqual(awf["alphaTotal"], 35)
        self.assertGreater(awf["alphaPrecisionPct"], awf["alphaUniverseMeanBaseRatePct"])
        for h in ("1", "4", "8"):
            aq = hz[h]["workbench_case"]["review"].get("alphaQueue", [])
            self.assertEqual(len(aq), 7)
            for item in aq:
                self.assertGreaterEqual(item["totalTvlUsd"], 15_000_000.0)
                self.assertLessEqual(item["totalTvlUsd"], 5_000_000_000.0)
        for s in hist["snapshots"]:
            if s["backtestResolved"]:
                self.assertIn("benchmarkReturns", s)
                self.assertIsNotNone(s["benchmarkReturns"])
                self.assertEqual(s["alphaTotal"], 7)
                self.assertIsNotNone(s["alphaBaseRatePct"])
                for r in s["rankings"]:
                    if r["inSupervisoryQueue"]:
                        self.assertIn("rawUsdChangePct", r)
                        self.assertIsNotNone(r["actualChangePct"])


if __name__ == "__main__":
    unittest.main()
