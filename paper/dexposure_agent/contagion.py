"""DebtRank-style contagion with per-node tracing for the DeXposure-Claw dashboard.

This is a line-by-line port of ``simulate_contagion`` in
``cloud/train_src/dexposure_fm/macroprudential_tools.py`` (the implementation
used for the paper's B4 stress tests). It produces identical totals and
distressed sets, and additionally records, for each node, its final loss, the
propagation round in which it first received loss, and the debtor through
which that first loss arrived. ``paper/tests/test_dashboard_contagion.py``
checks parity against the canonical implementation, and the browser port in
``docs/design/dexposure-claw-case-workbench/assets/contagion.js``.

Edge semantics follow the canonical tool: edge (source -> target) means
``source`` (creditor) is exposed to ``target`` (debtor). Received losses are
capped at max(0, TVL).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def simulate_contagion_traced(
    snap: Mapping[str, Any],
    shocked_nodes: Sequence[str],
    *,
    shock_fraction: float = 1.0,
    distress_threshold: float = 0.1,
    max_rounds: int = 10,
    shock_mode: str = "protocol",
) -> dict[str, Any]:
    if shock_mode not in ("protocol", "depeg"):
        raise ValueError(f"shock_mode must be 'protocol' or 'depeg', got {shock_mode!r}")

    nodes = snap.get("nodes", {}) or {}
    edges = snap.get("edges", []) or []

    tvl = {n: float((data or {}).get("tvlUsd", 0.0) or 0.0) for n, data in nodes.items()}
    losses = {n: 0.0 for n in nodes}
    first_round: dict[str, int] = {}
    first_via: dict[str, str] = {}

    exposures_in: dict[str, dict[str, float]] = {}
    for e in edges:
        src = (e or {}).get("source")
        dst = (e or {}).get("target")
        w = float((e or {}).get("weight", 0.0) or 0.0)
        if not src or not dst or w <= 0:
            continue
        dst = str(dst)
        src = str(src)
        exposures_in.setdefault(dst, {})
        exposures_in[dst][src] = exposures_in[dst].get(src, 0.0) + w

    distressed: set[str] = set()
    active: set[str] = set()
    shocked: list[str] = []
    exposed_pool_usd: float | None = None

    if shock_mode == "protocol":
        seen_s: set[str] = set()
        for node in shocked_nodes:
            node = str(node)
            if node in seen_s or node not in tvl:
                continue
            tvl_n = float(tvl[node])
            if tvl_n <= 0:
                continue
            seen_s.add(node)
            losses[node] = float(shock_fraction) * tvl_n
            distressed.add(node)
            active.add(node)
            shocked.append(node)
            first_round[node] = 0
        shocked_set = set(shocked)
        direct_loss = float(sum(losses[s] for s in shocked))
        affected_history = [len(distressed)]
        rnd = 0
        rem_rounds = int(max_rounds)
    else:
        # Stablecoin / reserve-asset depeg mode:
        # Round 1 applies mark-to-market write-down w_{c->t} * shock_fraction directly
        # to protocols holding the depegged asset(s) (capped at TVL[c]). Holders whose
        # loss exceeds distress_threshold * TVL[c] propagate via canonical DebtRank in R2+.
        seen_s = set()
        for node in shocked_nodes:
            node = str(node)
            if node in seen_s or node not in tvl:
                continue
            if sum(exposures_in.get(node, {}).values()) <= 0:
                continue
            seen_s.add(node)
            shocked.append(node)
        shocked_set = set(shocked)
        exposed_pool_usd = float(
            sum(
                w
                for t in shocked
                for c, w in exposures_in.get(t, {}).items()
                if c not in shocked_set and tvl.get(c, 0.0) > 0
            )
        )
        affected_history = [0]
        rnd = 0
        if shocked:
            rnd = 1
            for creditor in sorted(tvl.keys()):
                if creditor in shocked_set:
                    continue
                hits = [
                    (t, exposures_in.get(t, {}).get(creditor, 0.0))
                    for t in shocked
                ]
                hits = [(t, float(w)) for t, w in hits if w > 0]
                if not hits:
                    continue
                raw_haircut = float(shock_fraction) * sum(w for _, w in hits)
                best_t = min(hits, key=lambda tw: (-tw[1], tw[0]))[0]
                before = losses[creditor]
                tvl_c = float(tvl.get(creditor, 0.0))
                losses[creditor] = min(losses[creditor] + raw_haircut, max(0.0, tvl_c))
                if losses[creditor] > before:
                    first_round[creditor] = 1
                    first_via[creditor] = best_t
                if tvl_c > 0 and creditor not in distressed and losses[creditor] > float(distress_threshold) * tvl_c:
                    distressed.add(creditor)
                    active.add(creditor)
            if any(losses[n] > 0 for n in tvl):
                affected_history.append(len(distressed))
        direct_loss = float(sum(losses.values()))
        rem_rounds = max(0, int(max_rounds) - 1)

    for _ in range(rem_rounds):
        if not active:
            break
        rnd += 1
        new_active: set[str] = set()
        # The canonical tool iterates a Python set (hash order). We sort for
        # determinism. Both give the same result unless two debtors active in
        # the same round are exposed to each other, in which case the
        # canonical result itself depends on hash order.
        for debtor in sorted(active):
            creditors = exposures_in.get(debtor, {})
            if not creditors:
                continue
            total_exposure = float(sum(creditors.values()))
            if total_exposure <= 0:
                continue
            debtor_loss = float(losses.get(debtor, 0.0))
            if debtor_loss <= 0:
                continue
            for creditor, exposure in sorted(creditors.items()):
                exposure = float(exposure)
                if exposure <= 0 or creditor not in tvl:
                    continue
                if shock_mode == "depeg" and creditor in shocked_set:
                    continue
                propagated = debtor_loss * (exposure / total_exposure)
                before = losses[creditor]
                losses[creditor] += propagated
                tvl_c = float(tvl.get(creditor, 0.0))
                losses[creditor] = min(losses[creditor], max(0.0, tvl_c))
                if losses[creditor] > before and creditor not in first_round:
                    first_round[creditor] = rnd
                    first_via[creditor] = debtor
                if tvl_c > 0:
                    if creditor not in distressed and losses[creditor] > float(distress_threshold) * tvl_c:
                        distressed.add(creditor)
                        new_active.add(creditor)
        if not new_active:
            break
        active = new_active
        affected_history.append(len(distressed))

    total_tvl = float(sum(tvl.values())) if tvl else 0.0
    total_loss = float(sum(losses.values())) if losses else 0.0
    if shock_mode == "protocol":
        direct_loss = float(sum(losses[s] for s in shocked))
    per_node = [
        {
            "id": nid,
            "lossUsd": losses[nid],
            "tvlUsd": tvl[nid],
            "lossPctOfTvl": (100.0 * losses[nid] / tvl[nid]) if tvl[nid] > 0 else 0.0,
            "round": first_round.get(nid),
            "via": first_via.get(nid),
            "distressed": nid in distressed,
        }
        for nid in nodes
        if nid not in shocked_set and losses[nid] > 0
    ]
    per_node.sort(key=lambda r: -r["lossUsd"])
    max_loss_round = max((int(r["round"]) for r in per_node if r["round"] is not None), default=0)
    return {
        "shock_mode": shock_mode,
        "shocked_nodes": shocked,
        "shock_fraction": float(shock_fraction),
        "exposed_pool_usd": exposed_pool_usd,
        "total_loss": total_loss,
        "direct_loss": direct_loss,
        "contagion_loss": total_loss - direct_loss,
        "total_loss_pct": 100.0 * total_loss / total_tvl if total_tvl > 0 else 0.0,
        "affected_count": len(distressed),
        "distressed_nodes": sorted(distressed),
        "propagation_rounds": int(len(affected_history) - 1),
        "max_loss_round": max_loss_round,
        "affected_history": affected_history,
        "per_node": per_node,
    }
