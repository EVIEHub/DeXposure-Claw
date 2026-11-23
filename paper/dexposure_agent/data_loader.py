"""Data loading pipeline for DeXposure-Agent experiments.

Bridges the raw JSON weekly snapshots (historical-network_week_*.json)
and protocol metadata (meta_df.csv) into the agent pipeline's
GraphSnapshot Pydantic objects.

Also provides helpers for:
- Date-range filtering (parse 'YYYY-MM~YYYY-MM' split strings)
- Baseline metric history (for rolling-window z-score comparisons)

Usage:
    loader = SnapshotLoader("data/", meta_path="data/meta_df.csv")
    snapshots = loader.load(date_range="2025-01~2025-08")
    baseline  = loader.build_baseline(before="2025-01", window=26)
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dexposure_agent.monitor import compute_metrics
from dexposure_agent.types import Edge, GraphSnapshot, NodeFeatures

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = "data/"
DEFAULT_META_PATH = "data/meta_df.csv"
DATA_FILE_GLOB = "historical-network_week_*.json"

EPS = 1e-12

# This is the single ordered feature contract consumed by the FM encoder.
# Keep the order explicit because changing a feature's position changes the
# meaning of every released checkpoint.
FEATURE_ORDER = (
    "log_size",
    "num_tokens",
    "max_share",
    "entropy",
    "category_one_hot",
    "prev_tvl_change",
    "in_degree_norm",
    "out_degree_norm",
)

PREPROCESSING_CONTRACT = {
    "contract_version": 2,
    "node_id_policy": "stringify_non_null_then_strip; count_invalid_and_skip_null_or_blank",
    "duplicate_node_policy": "count_invalid_and_keep_first_record",
    "metadata_id_policy": "stringify_non_null_then_strip; reject_null_or_blank_and_duplicates",
    "missing_node_size_policy": "use_zero_and_mark_log_size_missing",
    "edge_policy": {
        "endpoint_policy": "count_null_or_blank_or_unknown_as_invalid",
        "weight_policy": "count_missing; reject_non_finite_or_negative",
        "parallel_edges": "sum_by_directed_source_target",
        "output_order": "lexicographic_source_target",
        "zero_weight": "exclude_after_validation",
    },
    "feature_order": list(FEATURE_ORDER),
    "historical_feature": "log1p(current_positive)-log1p(previous_positive),_else_zero",
    "degree_features": "count_coalesced_directed_edges_then_divide_by_per_snapshot_max_or_one",
}


def _canonical_json(value: Any) -> str:
    """Serialize a contract payload without incidental formatting variance."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def preprocessing_contract_hash() -> str:
    """Return the stable hash identifying the input-to-feature contract."""
    import hashlib

    return hashlib.sha256(_canonical_json(PREPROCESSING_CONTRACT).encode()).hexdigest()


def normalize_node_id(value: Any, *, field: str = "node id") -> str:
    """Canonicalize an identifier or fail closed at the trust boundary."""
    if value is None:
        raise ValueError(f"{field} cannot be null")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"{field} cannot be NaN")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} cannot be blank")
    return normalized


def aggregate_parallel_edges(edges: list[Edge]) -> list[Edge]:
    """Coalesce directed parallel edges in deterministic endpoint order."""
    weights: dict[tuple[str, str], float] = {}
    for edge in edges:
        source = normalize_node_id(edge.source, field="edge source")
        target = normalize_node_id(edge.target, field="edge target")
        weight = float(edge.weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"Invalid edge weight {weight} for {source!r}->{target!r}"
            )
        if weight > 0.0:
            pair = (source, target)
            weights[pair] = weights.get(pair, 0.0) + weight
    return [
        Edge(source=source, target=target, weight=weights[(source, target)])
        for source, target in sorted(weights)
    ]


def build_feature_matrix(
    graph: GraphSnapshot,
    categories: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Build the exact ordered node feature matrix used by FM training.

    This helper is shared by inference and parity tests. Training's raw
    loader emits the same fields before converting them to a NumPy matrix.
    """
    categories = tuple(categories)
    if not categories or len(categories) != len(set(categories)):
        raise ValueError("categories must be a non-empty unique vocabulary")
    if "Unknown" not in categories:
        raise ValueError("categories must contain 'Unknown'")

    node_ids = list(graph.nodes)
    node_to_idx = {node_id: index for index, node_id in enumerate(node_ids)}
    coalesced_edges = aggregate_parallel_edges(graph.edges)
    in_degree = [0.0] * len(node_ids)
    out_degree = [0.0] * len(node_ids)
    for edge in coalesced_edges:
        if edge.source not in node_to_idx or edge.target not in node_to_idx:
            raise ValueError(
                f"graph edge references unknown node id: {edge.source!r}->{edge.target!r}"
            )
        out_degree[node_to_idx[edge.source]] += 1.0
        in_degree[node_to_idx[edge.target]] += 1.0
    max_in = max(max(in_degree, default=0.0), 1.0)
    max_out = max(max(out_degree, default=0.0), 1.0)

    rows: list[list[float]] = []
    category_to_idx = {category: index for index, category in enumerate(categories)}
    for index, node_id in enumerate(node_ids):
        normalized_id = normalize_node_id(node_id)
        if normalized_id != node_id:
            raise ValueError(f"graph node id is not canonical: {node_id!r}")
        node = graph.nodes[node_id]
        category = node.category if node.category in category_to_idx else "Unknown"
        category_one_hot = [0.0] * len(categories)
        category_one_hot[category_to_idx[category]] = 1.0
        rows.append(
            [
                float(node.log_size),
                float(node.num_tokens),
                float(node.max_share),
                float(node.entropy),
                *category_one_hot,
                float(node.prev_tvl_change),
                in_degree[index] / max_in,
                out_degree[index] / max_out,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


# ── Date helpers ────────────────────────────────────────────────────────


def parse_date_range(range_str: str) -> tuple[datetime, datetime]:
    """Parse 'YYYY-MM~YYYY-MM' into (start_date, end_date) datetimes.

    The start is the first day of start_month; the end is the last day
    of end_month (approximated as first day of next month - 1 day).

    >>> parse_date_range("2025-01~2025-08")
    (datetime(2025, 1, 1), datetime(2025, 8, 31, ...))
    """
    parts = range_str.split("~")
    if len(parts) != 2:
        raise ValueError(f"Expected 'YYYY-MM~YYYY-MM', got: {range_str!r}")

    start = datetime.strptime(parts[0].strip(), "%Y-%m")
    end_month = datetime.strptime(parts[1].strip(), "%Y-%m")
    # End of month: go to first of next month, subtract 1 day
    if end_month.month == 12:
        end = datetime(end_month.year + 1, 1, 1)
    else:
        end = datetime(end_month.year, end_month.month + 1, 1)
    from datetime import timedelta
    end = end - timedelta(days=1)

    return start, end


def _parse_snapshot_date(date_str: str) -> datetime:
    """Parse a snapshot date string (YYYY-MM-DD) into a datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def _date_in_range(
    date_str: str, start: datetime | None, end: datetime | None
) -> bool:
    """Check if a date string falls within [start, end] inclusive."""
    if start is None and end is None:
        return True
    dt = _parse_snapshot_date(date_str)
    if start is not None and dt < start:
        return False
    if end is not None and dt > end:
        return False
    return True


# ── Node feature extraction ─────────────────────────────────────────────


def _extract_node_features(
    node: dict[str, Any],
    meta_category: dict[str, str],
    previous_sizes: dict[str, float] | None = None,
) -> NodeFeatures:
    """Convert a raw JSON node dict into a NodeFeatures Pydantic object.

    Features:
      - log_size:  log(1 + TVL)
      - num_tokens: number of composition tokens
      - max_share:  max token share in composition
      - entropy:    Shannon entropy of composition shares
      - category:   from meta_df.csv (default 'Unknown')
    """
    node_id = normalize_node_id(node.get("id"))
    size_present = node.get("size") is not None
    composition_present = node.get("composition") is not None
    size = float(node.get("size", 0.0) or 0.0)
    if not math.isfinite(size):
        raise ValueError(f"Node {node_id!r} has non-finite size: {size}")
    comp = node.get("composition", {}) or {}
    if not isinstance(comp, dict):
        raise ValueError(f"Node {node_id!r} composition must be an object")
    missing_features: list[str] = []
    if not size_present:
        missing_features.append("log_size")
    elif size < 0.0:
        # Negative TVL is outside the feature domain. Keep the node to match
        # training topology, clamp its log-size input to zero, and make the
        # invalid observation visible to the data-health gate.
        missing_features.append("log_size")
    if not composition_present:
        missing_features.extend(("num_tokens", "max_share", "entropy"))

    log_size = math.log1p(max(size, 0.0))
    num_tokens = len(comp)

    if size > 0 and comp:
        values = np.array(list(comp.values()), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Node {node_id!r} has non-finite composition values")
        if (values < 0.0).any():
            missing_features.extend(("max_share", "entropy"))
        values = np.maximum(values, 0.0)
        total = values.sum() + EPS
        shares = values / total
        max_share = float(shares.max())
        entropy = max(0.0, float(-(shares * np.log(shares + EPS)).sum()))
    else:
        max_share = 0.0
        entropy = 0.0

    category = meta_category.get(node_id, "Unknown")
    previous_size = (previous_sizes or {}).get(node_id, 0.0)
    prev_tvl_change = 0.0
    if size > 0.0 and previous_size > 0.0:
        prev_tvl_change = math.log1p(size) - math.log1p(previous_size)

    return NodeFeatures(
        log_size=log_size,
        num_tokens=num_tokens,
        max_share=max_share,
        entropy=entropy,
        category=category,
        prev_tvl_change=prev_tvl_change,
        missing_features=missing_features,
    )


# ── Snapshot conversion ──────────────────────────────────────────────────


def raw_to_graph_snapshot(
    date: str,
    snapshot: dict[str, Any],
    meta_category: dict[str, str],
    min_node_size: float = 0.0,
    previous_sizes: dict[str, float] | None = None,
) -> GraphSnapshot:
    """Convert one raw JSON snapshot into a GraphSnapshot.

    Args:
        date: Snapshot date string (YYYY-MM-DD).
        snapshot: Raw dict with 'nodes' list and 'links' list.
        meta_category: node_id -> category string from metadata.
        min_node_size: Skip nodes with TVL below this (default 0 = keep all).

    Returns:
        GraphSnapshot with NodeFeatures and Edges.
    """
    raw_nodes = snapshot.get("nodes", [])
    raw_links = snapshot.get("links", [])
    if not isinstance(raw_nodes, list) or not isinstance(raw_links, list):
        raise ValueError(f"Snapshot {date} requires list-valued nodes and links")

    # Pass 1: build node dict, applying size filter
    nodes: dict[str, NodeFeatures] = {}
    seen_node_ids: set[str] = set()
    invalid_node_records = 0
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            invalid_node_records += 1
            continue
        try:
            node_id = normalize_node_id(raw_node.get("id"))
        except ValueError:
            invalid_node_records += 1
            continue
        if node_id in seen_node_ids:
            invalid_node_records += 1
            continue
        seen_node_ids.add(node_id)
        raw_size = raw_node.get("size")
        size = float(raw_size) if raw_size is not None else 0.0
        if not math.isfinite(size):
            raise ValueError(f"Node {node_id!r} has non-finite size in snapshot {date}")
        if max(size, 0.0) < min_node_size:
            continue
        nodes[node_id] = _extract_node_features(
            raw_node,
            meta_category,
            previous_sizes=previous_sizes,
        )

    node_set = set(nodes.keys())
    raw_node_set = seen_node_ids

    # Pass 2: aggregate parallel raw exposure records into a deterministic
    # directed weighted graph. Missing or unknown endpoints remain observable
    # through GraphSnapshot source-quality counters.
    edge_weights: dict[tuple[str, str], float] = {}
    invalid_edge_records = 0
    for raw_link in raw_links:
        if not isinstance(raw_link, dict):
            raise ValueError(f"Snapshot {date} contains a non-object link record")
        raw_src = raw_link.get("source")
        raw_tgt = raw_link.get("target")
        if raw_src is None or raw_tgt is None:
            invalid_edge_records += 1
            continue
        try:
            src = normalize_node_id(raw_src, field="edge source")
            tgt = normalize_node_id(raw_tgt, field="edge target")
        except ValueError:
            invalid_edge_records += 1
            continue
        raw_weight = raw_link.get("size")
        if raw_weight is None:
            invalid_edge_records += 1
            continue
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"Invalid edge weight {weight} for {src!r}->{tgt!r} in snapshot {date}"
            )
        if src not in raw_node_set or tgt not in raw_node_set:
            invalid_edge_records += 1
            continue
        if src in node_set and tgt in node_set and weight > 0:
            pair = (src, tgt)
            edge_weights[pair] = edge_weights.get(pair, 0.0) + weight

    edges = [
        Edge(source=source, target=target, weight=edge_weights[(source, target)])
        for source, target in sorted(edge_weights)
    ]

    return GraphSnapshot(
        date=date,
        nodes=nodes,
        edges=edges,
        source_node_records_total=len(raw_nodes),
        source_node_records_invalid=invalid_node_records,
        source_edge_records_total=len(raw_links),
        source_edge_records_invalid=invalid_edge_records,
    )


# ── Main loader ──────────────────────────────────────────────────────────


class SnapshotLoader:
    """Loads and filters weekly graph snapshots for the agent pipeline.

    Typical usage:
        loader = SnapshotLoader("data/")
        test_snaps = loader.load(date_range="2025-01~2025-08")
        baseline   = loader.build_baseline(before="2025-01", window=26)
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        meta_path: str = DEFAULT_META_PATH,
        min_node_size: float = 0.0,
        duplicate_policy: str = "latest_file_wins",
    ):
        self.data_dir = Path(data_dir)
        # Resolve meta_path: if default and not found, try inside data_dir
        meta = Path(meta_path)
        if not meta.exists() and not meta.is_absolute():
            meta_in_data = self.data_dir / meta.name
            if meta_in_data.exists():
                meta = meta_in_data
        self.meta_path = meta
        if (
            not isinstance(min_node_size, (int, float))
            or isinstance(min_node_size, bool)
            or not math.isfinite(float(min_node_size))
            or min_node_size < 0.0
        ):
            raise ValueError("min_node_size must be finite and non-negative")
        self.min_node_size = min_node_size
        if duplicate_policy not in {"latest_file_wins", "error"}:
            raise ValueError(
                "duplicate_policy must be 'latest_file_wins' or 'error'"
            )
        self.duplicate_policy = duplicate_policy
        self.duplicate_conflicts: list[dict[str, str]] = []

        # Lazy-loaded caches
        self._meta_category: dict[str, str] | None = None
        self._raw_data: dict[str, dict] | None = None
        self._sorted_dates: list[str] | None = None

    # ── Metadata ─────────────────────────────────────────────────────

    def _load_metadata(self) -> dict[str, str]:
        """Load meta_df.csv -> {node_id: category}."""
        if self._meta_category is not None:
            return self._meta_category

        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        df = pd.read_csv(self.meta_path)
        required = {"id", "category"}
        missing_columns = required - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Metadata missing columns {sorted(missing_columns)}: {self.meta_path}"
            )
        if df["id"].isna().any():
            raise ValueError(f"Metadata contains missing ids: {self.meta_path}")
        df["id"] = df["id"].astype("string").str.strip()
        if (df["id"] == "").any():
            raise ValueError(f"Metadata contains empty ids: {self.meta_path}")
        if df["id"].duplicated().any():
            duplicates = sorted(df.loc[df["id"].duplicated(), "id"].unique())
            raise ValueError(f"Duplicate metadata ids: {duplicates[:10]}")
        df["category"] = (
            df["category"]
            .fillna("Unknown")
            .astype("string")
            .str.strip()
            .replace("", "Unknown")
        )
        self._meta_category = df.set_index("id")["category"].to_dict()
        logger.info("Loaded metadata: %d protocols from %s", len(self._meta_category), self.meta_path)
        return self._meta_category

    # ── Raw JSON loading ─────────────────────────────────────────────

    def _load_raw_data(self) -> dict[str, dict]:
        """Load all JSON files in data_dir, merge into {date: snapshot}.

        Tries ijson for streaming large files; falls back to json.load().
        """
        if self._raw_data is not None:
            return self._raw_data

        json_files = sorted(self.data_dir.glob(DATA_FILE_GLOB))
        if not json_files:
            raise FileNotFoundError(
                f"No {DATA_FILE_GLOB} files found in {self.data_dir}"
            )

        merged: dict[str, dict] = {}
        source_by_date: dict[str, Path] = {}
        for path in json_files:
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info("Loading %s (%.1f MB)", path.name, size_mb)
            incoming = self._read_json_file(path)
            if not isinstance(incoming, dict):
                raise ValueError(f"Snapshot file must contain a date map: {path}")
            for date, snapshot in incoming.items():
                if not isinstance(date, str):
                    raise ValueError(f"Snapshot date keys must be strings: {path}")
                try:
                    _parse_snapshot_date(date)
                except ValueError as exc:
                    raise ValueError(f"Invalid snapshot date {date!r} in {path}") from exc
                if not isinstance(snapshot, dict):
                    raise ValueError(f"Snapshot {date} must be an object: {path}")
            duplicates = sorted(set(merged) & set(incoming))
            for date in duplicates:
                if merged[date] != incoming[date]:
                    conflict = {
                        "date": date,
                        "previous_source": str(source_by_date[date]),
                        "selected_source": str(path),
                    }
                    self.duplicate_conflicts.append(conflict)
                    if self.duplicate_policy == "error":
                        raise ValueError(
                            "Conflicting snapshot date "
                            f"{date} in {source_by_date[date]} and {path}"
                        )
                    logger.warning(
                        "Conflicting snapshot %s; selecting %s over %s "
                        "under duplicate_policy=%s",
                        date,
                        path.name,
                        source_by_date[date].name,
                        self.duplicate_policy,
                    )
                else:
                    logger.info(
                        "Identical duplicate snapshot %s in %s and %s",
                        date,
                        source_by_date[date].name,
                        path.name,
                    )
            merged.update(incoming)
            for date in incoming:
                source_by_date[date] = path

        self._raw_data = merged
        self._sorted_dates = sorted(merged.keys())
        logger.info(
            "Loaded %d weekly snapshots: %s ~ %s",
            len(self._sorted_dates),
            self._sorted_dates[0] if self._sorted_dates else "?",
            self._sorted_dates[-1] if self._sorted_dates else "?",
        )
        return self._raw_data

    @staticmethod
    def _size_map(snapshot: dict[str, Any] | None) -> dict[str, float]:
        if not snapshot:
            return {}
        result: dict[str, float] = {}
        for node in snapshot.get("nodes", []):
            if not isinstance(node, dict):
                continue
            try:
                node_id = normalize_node_id(node.get("id"))
            except ValueError:
                continue
            size = float(node.get("size", 0.0) or 0.0)
            if not math.isfinite(size):
                raise ValueError(f"Node {node.get('id')!r} has non-finite previous size")
            if node_id not in result:
                result[node_id] = size
        return result

    def _previous_sizes(self, date: str, raw: dict[str, dict]) -> dict[str, float]:
        index = self.dates.index(date)
        if index == 0:
            return {}
        return self._size_map(raw[self.dates[index - 1]])

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, dict]:
        """Read a single JSON file, returning {date: snapshot_dict}."""
        with path.open("rb") as f:
            try:
                import ijson
                return dict(ijson.kvitems(f, "data"))
            except ImportError:
                f.seek(0)
                payload = json.load(f)
                return payload.get("data", payload)
            except Exception:
                f.seek(0)
                payload = json.load(f)
                return payload.get("data", payload)

    # ── Public API ───────────────────────────────────────────────────

    @property
    def dates(self) -> list[str]:
        """All available snapshot dates, sorted chronologically."""
        self._load_raw_data()
        assert self._sorted_dates is not None
        return self._sorted_dates

    def load(
        self,
        date_range: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> list[GraphSnapshot]:
        """Load snapshots filtered by date range.

        Args:
            date_range: 'YYYY-MM~YYYY-MM' string (convenience).
            start: Explicit start date 'YYYY-MM-DD' (overrides date_range).
            end: Explicit end date 'YYYY-MM-DD' (overrides date_range).

        Returns:
            List of GraphSnapshot sorted by date, oldest first.
        """
        raw = self._load_raw_data()
        meta = self._load_metadata()

        # Resolve date bounds
        dt_start: datetime | None = None
        dt_end: datetime | None = None

        if date_range:
            dt_start, dt_end = parse_date_range(date_range)
        if start:
            dt_start = _parse_snapshot_date(start)
        if end:
            dt_end = _parse_snapshot_date(end)

        # Filter and convert
        snapshots: list[GraphSnapshot] = []
        for date_str in self.dates:
            if not _date_in_range(date_str, dt_start, dt_end):
                continue
            snap = raw_to_graph_snapshot(
                date_str,
                raw[date_str],
                meta,
                self.min_node_size,
                previous_sizes=self._previous_sizes(date_str, raw),
            )
            snapshots.append(snap)

        logger.info(
            "Loaded %d snapshots for range %s",
            len(snapshots),
            date_range or f"{start}~{end}",
        )
        return snapshots

    def load_single(self, date: str) -> GraphSnapshot:
        """Load a single snapshot by exact date string."""
        raw = self._load_raw_data()
        meta = self._load_metadata()

        if date not in raw:
            available = [d for d in self.dates if d.startswith(date[:7])]
            hint = f" (closest: {available})" if available else ""
            raise KeyError(f"No snapshot for date {date!r}{hint}")

        return raw_to_graph_snapshot(
            date,
            raw[date],
            meta,
            self.min_node_size,
            previous_sizes=self._previous_sizes(date, raw),
        )

    def build_baseline(
        self,
        before: str,
        window: int = 26,
    ) -> list[dict[str, float]]:
        """Build a baseline metric history for rolling-window z-score comparison.

        Loads `window` snapshots ending just before `before` date, computes
        Phi metrics (N1..N5) on each, returns list of metric dicts.

        Args:
            before: Date string 'YYYY-MM-DD' or 'YYYY-MM'. Snapshots strictly
                    before this date are used.
            window: Number of past snapshots to include (default 26 = ~6 months).

        Returns:
            List of {metric_id: value} dicts, oldest first.
        """
        raw = self._load_raw_data()
        meta = self._load_metadata()

        # Find snapshots before the cutoff
        if len(before) == 7:  # YYYY-MM
            cutoff = datetime.strptime(before, "%Y-%m")
        else:
            cutoff = _parse_snapshot_date(before)

        eligible_dates = [
            d for d in self.dates if _parse_snapshot_date(d) < cutoff
        ]
        selected = eligible_dates[-window:]  # last `window` dates

        if not selected:
            logger.warning("No snapshots found before %s for baseline", before)
            return []

        logger.info(
            "Building baseline: %d snapshots before %s (%s ~ %s)",
            len(selected), before, selected[0], selected[-1],
        )

        history: list[dict[str, float]] = []
        for date_str in selected:
            snap = raw_to_graph_snapshot(
                date_str,
                raw[date_str],
                meta,
                self.min_node_size,
                previous_sizes=self._previous_sizes(date_str, raw),
            )
            metrics = compute_metrics(snap)
            history.append(metrics)

        return history

    def iter_test_with_baselines(
        self,
        test_range: str,
        baseline_window: int = 26,
    ):
        """Iterate over test snapshots, yielding (snapshot, baseline_history) pairs.

        For each test snapshot at date t, the baseline is the preceding
        `baseline_window` snapshots' metric dicts. This is the main entry point
        for benchmark runners (b1_forecast..b6_robustness).

        Yields:
            (GraphSnapshot, list[dict[str, float]]) tuples.
        """
        raw = self._load_raw_data()
        meta = self._load_metadata()
        dt_start, dt_end = parse_date_range(test_range)

        # Pre-compute all metrics for the full timeline (cached)
        all_metrics: dict[str, dict[str, float]] = {}

        for date_str in self.dates:
            dt = _parse_snapshot_date(date_str)
            # We need metrics for dates before the test range (for baseline)
            # and within the test range (for ground truth comparison)
            if dt <= dt_end:
                snap = raw_to_graph_snapshot(
                    date_str,
                    raw[date_str],
                    meta,
                    self.min_node_size,
                    previous_sizes=self._previous_sizes(date_str, raw),
                )
                all_metrics[date_str] = compute_metrics(snap)

        # Yield test snapshots with their baselines
        for date_str in self.dates:
            if not _date_in_range(date_str, dt_start, dt_end):
                continue

            snap = raw_to_graph_snapshot(
                date_str,
                raw[date_str],
                meta,
                self.min_node_size,
                previous_sizes=self._previous_sizes(date_str, raw),
            )

            # Collect baseline: metrics from previous dates
            prior_dates = [
                d for d in self.dates
                if _parse_snapshot_date(d) < _parse_snapshot_date(date_str)
            ]
            baseline_dates = prior_dates[-baseline_window:]
            baseline = [all_metrics[d] for d in baseline_dates if d in all_metrics]

            yield snap, baseline
