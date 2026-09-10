#!/usr/bin/env python3
"""Faithful, memory-safe per-protocol actual_losses from local snapshots.

This is the single source of truth for the `actual_losses` that
``llm_eval_b5.assess_week`` needs to compute the False Intervention Rate
(FIR) and severity correlation.

Historically ``redecide_b5.py`` passed ``actual_losses={}`` (see its inline
note "unavailable at this stage"), which silently collapsed FIR to 0.0 for
every re-decided model -- the bug this module exists to remove. Both the
main pipeline and the re-decide path can now derive identical losses from
the same snapshot data without an FM service or GPU.

It replicates the processing chain exactly:
  data_loader.raw_to_graph_snapshot(min_node_size=0)  ->  edges
  llm_eval_b5.compute_actual_loss                      ->  per-node loss

Snapshots are streamed with ijson so the 1.1 GB history file is never fully
materialised; only the requested dates are kept.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = _ROOT / "data"
DATA_FILE_GLOB = "historical-network_week_*.json"


def _node_weights(snapshot: dict) -> dict[str, float]:
    """Total edge weight per node, replicating raw_to_graph_snapshot
    (min_node_size=0: every node id is included; an edge counts only when
    both endpoints are included and its size is > 0)."""
    node_set = {
        str(n.get("id", "")) for n in snapshot.get("nodes", []) if str(n.get("id", ""))
    }
    wt: dict[str, float] = defaultdict(float)
    for lk in snapshot.get("links", []):
        src = str(lk.get("source", ""))
        tgt = str(lk.get("target", ""))
        w = float(lk.get("size", 0.0))
        if src in node_set and tgt in node_set and w > 0:
            wt[src] += w
            wt[tgt] += w
    return wt


def _actual_loss(wt_cur: dict[str, float], wt_fut: dict[str, float]) -> dict[str, float]:
    """Per-protocol weight-drop fraction; mirrors
    llm_eval_b5.compute_actual_loss exactly."""
    losses: dict[str, float] = {}
    for nid, w in wt_cur.items():
        if w > 0:
            losses[nid] = max(0.0, (w - wt_fut.get(nid, 0.0)) / w)
    return losses


def all_dates(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[str]:
    """Global sorted snapshot dates across all data files."""
    import ijson

    data_dir = Path(data_dir)
    keys: set[str] = set()
    for f in sorted(data_dir.glob(DATA_FILE_GLOB)):
        with open(f, "rb") as fh:
            for prefix, event, value in ijson.parse(fh):
                if prefix == "data" and event == "map_key":
                    keys.add(value)
    return sorted(keys)


def load_actual_losses(
    current_dates: list[str],
    horizon: int = 4,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> dict[str, dict[str, float]]:
    """Return {current_date: {protocol: loss_fraction}} for each requested
    current date, using the future snapshot at current_index + horizon in
    the global date ordering (matching llm_eval_b5's future_idx logic).

    Dates whose future snapshot is unavailable are returned with an empty
    loss dict (caller can treat that week as having no ground-truth losses).
    """
    import ijson

    data_dir = Path(data_dir)
    ordered = all_dates(data_dir)
    idx = {d: i for i, d in enumerate(ordered)}

    # Map each current date to its future date; collect snapshots we must read.
    fut_of: dict[str, str | None] = {}
    needed: set[str] = set()
    for d in current_dates:
        i = idx.get(d, -1)
        f = ordered[i + horizon] if 0 <= i and i + horizon < len(ordered) else None
        fut_of[d] = f
        needed.add(d)
        if f is not None:
            needed.add(f)

    # Stream both files once, keep node weights only for needed dates.
    weights: dict[str, dict[str, float]] = {}
    remaining = set(needed)
    for f in sorted(data_dir.glob(DATA_FILE_GLOB)):
        if not remaining:
            break
        with open(f, "rb") as fh:
            for date, snap in ijson.kvitems(fh, "data"):
                if date in remaining:
                    weights[date] = _node_weights(snap)
                    remaining.discard(date)
                    if not remaining:
                        break

    out: dict[str, dict[str, float]] = {}
    for d in current_dates:
        f = fut_of[d]
        if f is None or d not in weights or f not in weights:
            out[d] = {}
        else:
            out[d] = _actual_loss(weights[d], weights[f])
    return out


if __name__ == "__main__":
    import sys

    dates = sys.argv[1:] or ["2025-01-06", "2025-07-21"]
    losses = load_actual_losses(dates)
    for d in dates:
        ld = losses[d]
        print(f"{d}: {len(ld)} protocols with losses; "
              f"max={max(ld.values()) if ld else 0:.3f}")
