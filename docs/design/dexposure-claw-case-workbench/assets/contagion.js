/*
 * Browser/Node port of paper/dexposure_agent/contagion.py::simulate_contagion_traced,
 * itself a traced copy of the canonical DebtRank-style simulate_contagion in
 * cloud/train_src/dexposure_fm/macroprudential_tools.py, plus an explicit
 * shockMode="depeg" option for reserve-token / stablecoin haircut propagation.
 *
 * Edge (source -> target): source (creditor) is exposed to target (debtor).
 * Received losses are capped at max(0, TVL). Parity with the Python version is
 * checked by paper/tests/test_dashboard_contagion.py.
 */
(function (root) {
  function simulateContagionTraced(snap, shockedNodes, opts) {
    opts = opts || {};
    const shockFraction = opts.shockFraction === undefined ? 1.0 : Number(opts.shockFraction);
    const distressThreshold = opts.distressThreshold === undefined ? 0.1 : Number(opts.distressThreshold);
    const maxRounds = opts.maxRounds === undefined ? 10 : Number(opts.maxRounds);
    const shockMode = opts.shockMode === undefined ? "protocol" : String(opts.shockMode);
    if (shockMode !== "protocol" && shockMode !== "depeg") {
      throw new Error("shockMode must be 'protocol' or 'depeg', got " + shockMode);
    }

    const nodes = snap.nodes || {};
    const edges = snap.edges || [];
    const nodeIds = Object.keys(nodes);
    const tvl = {};
    const losses = {};
    for (const n of nodeIds) {
      tvl[n] = Number((nodes[n] || {}).tvlUsd || 0) || 0;
      losses[n] = 0;
    }
    const firstRound = {};
    const firstVia = {};

    const exposuresIn = {};
    for (const e of edges) {
      if (!e) continue;
      const src = e.source, dst = e.target;
      const w = Number(e.weight || 0) || 0;
      if (!src || !dst || w <= 0) continue;
      const d = String(dst), s = String(src);
      if (!exposuresIn[d]) exposuresIn[d] = {};
      exposuresIn[d][s] = (exposuresIn[d][s] || 0) + w;
    }

    const distressed = new Set();
    let active = new Set();
    const shocked = [];
    const seenShocked = new Set();
    let exposedPoolUsd = null;
    let directLoss = 0;
    let affectedHistory;
    let rnd = 0;
    let remRounds;

    if (shockMode === "protocol") {
      for (const raw of shockedNodes) {
        const node = String(raw);
        if (seenShocked.has(node) || !(node in tvl)) continue;
        const t = tvl[node];
        if (t <= 0) continue;
        seenShocked.add(node);
        losses[node] = shockFraction * t;
        distressed.add(node);
        active.add(node);
        shocked.push(node);
        firstRound[node] = 0;
      }
      affectedHistory = [distressed.size];
      remRounds = maxRounds;
    } else {
      for (const raw of shockedNodes) {
        const node = String(raw);
        if (seenShocked.has(node) || !(node in tvl)) continue;
        const creds = exposuresIn[node];
        let totIn = 0;
        if (creds) {
          for (const k in creds) totIn += creds[k];
        }
        if (totIn <= 0) continue;
        seenShocked.add(node);
        shocked.push(node);
      }
      const shockedSetInit = new Set(shocked);
      exposedPoolUsd = 0;
      for (const t of shocked) {
        const creds = exposuresIn[t] || {};
        for (const c in creds) {
          if (!shockedSetInit.has(c) && (tvl[c] || 0) > 0) {
            exposedPoolUsd += creds[c];
          }
        }
      }
      affectedHistory = [0];
      if (shocked.length > 0) {
        rnd = 1;
        const sortedCreds = nodeIds.slice().sort();
        for (const creditor of sortedCreds) {
          if (shockedSetInit.has(creditor)) continue;
          const hits = [];
          for (const t of shocked) {
            const w = (exposuresIn[t] || {})[creditor] || 0;
            if (w > 0) hits.push([t, w]);
          }
          if (hits.length === 0) continue;
          let sumW = 0;
          for (const hw of hits) sumW += hw[1];
          hits.sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
          const bestT = hits[0][0];
          const before = losses[creditor];
          const tvlC = tvl[creditor] || 0;
          losses[creditor] = Math.min(losses[creditor] + shockFraction * sumW, Math.max(0, tvlC));
          if (losses[creditor] > before) {
            firstRound[creditor] = 1;
            firstVia[creditor] = bestT;
          }
          if (tvlC > 0 && !distressed.has(creditor) && losses[creditor] > distressThreshold * tvlC) {
            distressed.add(creditor);
            active.add(creditor);
          }
        }
        let anyR1Loss = false;
        for (const n of nodeIds) {
          if (losses[n] > 0) { anyR1Loss = true; break; }
        }
        if (anyR1Loss) affectedHistory.push(distressed.size);
      }
      for (const n of nodeIds) directLoss += losses[n];
      remRounds = Math.max(0, maxRounds - 1);
    }

    const shockedSet = new Set(shocked);
    for (let it = 0; it < remRounds; it++) {
      if (active.size === 0) break;
      rnd += 1;
      const newActive = new Set();
      const ordered = Array.from(active).sort();
      for (const debtor of ordered) {
        const creditors = exposuresIn[debtor];
        if (!creditors) continue;
        let totalExposure = 0;
        for (const k in creditors) totalExposure += creditors[k];
        if (totalExposure <= 0) continue;
        const debtorLoss = losses[debtor] || 0;
        if (debtorLoss <= 0) continue;
        const credKeys = Object.keys(creditors).sort();
        for (const creditor of credKeys) {
          const exposure = creditors[creditor];
          if (exposure <= 0 || !(creditor in tvl)) continue;
          if (shockMode === "depeg" && shockedSet.has(creditor)) continue;
          const propagated = debtorLoss * (exposure / totalExposure);
          const before = losses[creditor];
          losses[creditor] += propagated;
          const tvlC = tvl[creditor] || 0;
          losses[creditor] = Math.min(losses[creditor], Math.max(0, tvlC));
          if (losses[creditor] > before && firstRound[creditor] === undefined) {
            firstRound[creditor] = rnd;
            firstVia[creditor] = debtor;
          }
          if (tvlC > 0 && !distressed.has(creditor) && losses[creditor] > distressThreshold * tvlC) {
            distressed.add(creditor);
            newActive.add(creditor);
          }
        }
      }
      if (newActive.size === 0) break;
      active = newActive;
      affectedHistory.push(distressed.size);
    }

    let totalTvl = 0, totalLoss = 0;
    for (const n of nodeIds) { totalTvl += tvl[n]; totalLoss += losses[n]; }
    if (shockMode === "protocol") {
      directLoss = 0;
      for (const s of shocked) directLoss += losses[s];
    }
    const perNode = [];
    let maxLossRound = 0;
    for (const n of nodeIds) {
      if (shockedSet.has(n) || !(losses[n] > 0)) continue;
      const r = firstRound[n] === undefined ? null : firstRound[n];
      if (r !== null && r > maxLossRound) maxLossRound = r;
      perNode.push({
        id: n,
        lossUsd: losses[n],
        tvlUsd: tvl[n],
        lossPctOfTvl: tvl[n] > 0 ? (100 * losses[n]) / tvl[n] : 0,
        round: r,
        via: firstVia[n] === undefined ? null : firstVia[n],
        distressed: distressed.has(n),
      });
    }
    perNode.sort((a, b) => b.lossUsd - a.lossUsd);
    return {
      shock_mode: shockMode,
      shocked_nodes: shocked,
      shock_fraction: shockFraction,
      exposed_pool_usd: exposedPoolUsd,
      total_loss: totalLoss,
      direct_loss: directLoss,
      contagion_loss: totalLoss - directLoss,
      total_loss_pct: totalTvl > 0 ? (100 * totalLoss) / totalTvl : 0,
      affected_count: distressed.size,
      distressed_nodes: Array.from(distressed).sort(),
      propagation_rounds: affectedHistory.length - 1,
      max_loss_round: maxLossRound,
      affected_history: affectedHistory,
      per_node: perNode,
    };
  }

  root.simulateContagionTraced = simulateContagionTraced;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { simulateContagionTraced };
  }
})(typeof window !== "undefined" ? window : globalThis);
