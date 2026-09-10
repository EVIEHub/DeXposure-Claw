"""DeXposure-FM model predictor for the agent pipeline.

Loads fine-tuned GraphPFN checkpoints and generates graph predictions
(edge existence probabilities + edge weight predictions) for use in
b1_forecast..b6_robustness benchmarks.

Checkpoints:
    dexposure-fm-h1.pt   -> horizon 1 week
    dexposure-fm-h4.pt   -> horizon 4 weeks
    dexposure-fm-h8-h12.pt -> horizon 8 weeks (legacy filename)
    dexposure-fm-h12.pt -> horizon 12 weeks

Each checkpoint contains a full state_dict with keys:
    encoder.tfm.*       -> GraphPFN transformer encoder
    link_scorer.*       -> edge existence + weight prediction heads
    node_head.*         -> node TVL change prediction head
    scenario_head.*     -> scenario classification head
"""
from __future__ import annotations

import logging
import json
import hashlib
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Add project root for lib imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGL_DISABLE_GRAPHBOLT", "1")

from dexposure_agent.data_loader import (
    FEATURE_ORDER,
    aggregate_parallel_edges,
    build_feature_matrix,
    preprocessing_contract_hash,
    normalize_node_id,
)
from dexposure_agent.types import Edge, GraphSnapshot, NodeFeatures

logger = logging.getLogger(__name__)

# Checkpoint file mapping
CHECKPOINT_MAP = {
    1: "dexposure-fm-h1.pt",
    4: "dexposure-fm-h4.pt",
    8: "dexposure-fm-h8-h12.pt",
    12: "dexposure-fm-h12.pt",
}

# Default checkpoint directory (override with DEXPOSURE_FM_CKPT_DIR, e.g. to
# point the whole pipeline at the pre-2022 retrained checkpoints)
DEFAULT_CKPT_DIR = os.environ.get(
    "DEXPOSURE_FM_CKPT_DIR",
    os.path.join(_PROJECT_ROOT, "checkpoints", "dexposure-fm-release"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256_hex(value: Any) -> bool:
    """Return whether ``value`` is a 64-character hexadecimal SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == hashlib.sha256().digest_size * 2
        and all(char in "0123456789abcdef" for char in value.lower())
    )


class FMPredictor:
    """Loads DeXposure-FM and generates graph predictions.

    Usage:
        predictor = FMPredictor(checkpoint_dir="checkpoints/dexposure-fm-release/")
        predicted_graph = predictor.predict(current_graph, horizon=4)
    """

    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CKPT_DIR,
        device: str | None = None,
        pi_min: float = 0.5,
        inference_seed: int = 42,
        data_path: str | None = None,
        metadata_path: str | None = None,
        require_provenance: bool = False,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pi_min = pi_min
        if (
            isinstance(inference_seed, bool)
            or not isinstance(inference_seed, int)
            or inference_seed < 0
        ):
            raise ValueError("inference_seed must be a non-negative integer")
        self.inference_seed = inference_seed
        self.data_path = Path(data_path) if data_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        if require_provenance and (self.data_path is None or self.metadata_path is None):
            raise ValueError(
                "require_provenance needs both data_path and metadata_path"
            )
        self.require_provenance = bool(require_provenance)
        if (
            not isinstance(pi_min, (int, float))
            or isinstance(pi_min, bool)
            or not math.isfinite(float(pi_min))
            or not 0.0 <= float(pi_min) <= 1.0
        ):
            raise ValueError("pi_min must be finite and within [0, 1]")

        checkpoint_files = [
            self.checkpoint_dir / filename for filename in CHECKPOINT_MAP.values()
        ]
        has_checkpoint = any(path.is_file() for path in checkpoint_files)
        schema_path = self.checkpoint_dir / "feature_schema.json"
        if schema_path.is_file() or has_checkpoint:
            self._feature_schema = self._load_feature_schema()
            self._categories = tuple(self._feature_schema["categories"])
            self._category_to_idx = {
                category: index for index, category in enumerate(self._categories)
            }
        else:
            self._feature_schema = {}
            self._categories = ()
            self._category_to_idx = {}
        self._legacy_schema = "preprocessing_contract_hash" not in self._feature_schema
        self._available = has_checkpoint

        # Lazy-loaded models per horizon
        self._models: dict[int, nn.Module] = {}
        self._link_scorers: dict[int, nn.Module] = {}
        self._node_heads: dict[int, nn.Module] = {}

        if not self._available:
            logger.warning("No usable FM checkpoints found in %s", self.checkpoint_dir)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def available_horizons(self) -> list[int]:
        return sorted(
            horizon
            for horizon, filename in CHECKPOINT_MAP.items()
            if (self.checkpoint_dir / filename).is_file()
        )

    @property
    def checkpoint_identity(self) -> dict[str, Any]:
        """Return observable checkpoint/schema identity for health and audits."""
        files: dict[str, str] = {}
        for horizon, filename in CHECKPOINT_MAP.items():
            path = self.checkpoint_dir / filename
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                files[str(horizon)] = digest.hexdigest()
        schema_path = self.checkpoint_dir / "feature_schema.json"
        schema_sha = None
        if schema_path.is_file():
            schema_sha = hashlib.sha256(schema_path.read_bytes()).hexdigest()
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "feature_schema_sha256": schema_sha,
            "preprocessing_contract_hash": self._feature_schema.get(
                "preprocessing_contract_hash"
            ),
            "legacy_schema": self._legacy_schema,
            "files_sha256": files,
            "pi_min": float(self.pi_min),
        }

    def _get_horizon_key(self, horizon: int) -> int:
        """Return the exact checkpoint key; never substitute another horizon."""
        if horizon not in CHECKPOINT_MAP:
            raise ValueError(
                f"Unsupported FM horizon {horizon}; expected one of "
                f"{sorted(CHECKPOINT_MAP)}"
            )
        return horizon

    def _load_feature_schema(self) -> dict[str, Any]:
        schema_path = self.checkpoint_dir / "feature_schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(
                "FM inference requires the training feature schema: "
                f"{schema_path}"
            )
        schema = json.loads(schema_path.read_text())
        categories = schema.get("categories")
        if (
            not isinstance(categories, list)
            or not categories
            or len(categories) != len(set(categories))
            or "Unknown" not in categories
        ):
            raise ValueError(f"Invalid category vocabulary in {schema_path}")
        if schema.get("edge_weight_space") != "log1p":
            raise ValueError(
                f"Unsupported edge-weight space in {schema_path}: "
                f"{schema.get('edge_weight_space')!r}"
            )
        expected_order = list(FEATURE_ORDER)
        if schema.get("feature_order") != expected_order:
            raise ValueError(
                f"Unsupported feature order in {schema_path}: "
                f"{schema.get('feature_order')!r}"
            )
        if schema.get("node_target_space") != "log1p_delta":
            raise ValueError(
                f"Unsupported node-target space in {schema_path}: "
                f"{schema.get('node_target_space')!r}"
            )
        schema_version = int(schema.get("schema_version", 1))
        if schema_version >= 2:
            expected_hash = preprocessing_contract_hash()
            if schema.get("preprocessing_contract_hash") != expected_hash:
                raise ValueError(
                    "Unsupported preprocessing contract hash in "
                    f"{schema_path}: {schema.get('preprocessing_contract_hash')!r}"
                )
        return schema

    def _load_model(self, horizon: int):
        """Load encoder + link_scorer + node_head for a given horizon."""
        h_key = self._get_horizon_key(horizon)
        if h_key in self._models:
            return

        ckpt_name = CHECKPOINT_MAP.get(h_key)
        if not ckpt_name:
            raise ValueError(f"No checkpoint for horizon {horizon}")

        ckpt_path = self.checkpoint_dir / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        logger.info("Loading FM checkpoint: %s (horizon=%d)", ckpt_name, horizon)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or not isinstance(ckpt.get("model"), dict):
            raise ValueError(f"Checkpoint has no model state: {ckpt_path}")
        embedded_schema = ckpt.get("feature_schema")
        if embedded_schema is not None and embedded_schema != self._feature_schema:
            raise ValueError(
                f"Checkpoint schema does not match {self.checkpoint_dir / 'feature_schema.json'}"
            )
        embedded_horizon = ckpt.get("horizon")
        if embedded_horizon is not None and int(embedded_horizon) != horizon:
            raise ValueError(
                f"Checkpoint horizon {embedded_horizon} does not match requested {horizon}"
            )
        provenance = ckpt.get("provenance")
        if provenance is None:
            # Existing release artifacts predate provenance. Keep them usable
            # for diagnostics, but never mistake them for an auditable run.
            if self.require_provenance:
                raise ValueError(
                    f"Checkpoint lacks required provenance: {ckpt_path}"
                )
            logger.warning("Loading legacy FM checkpoint without provenance: %s", ckpt_path)
        else:
            if embedded_schema is None:
                raise ValueError("New checkpoint is missing its embedded feature schema")
            self._validate_provenance(provenance, horizon)
        sd = ckpt["model"]

        # Import model classes
        from lib.graphpfn.model import GraphPFN

        # 1. Load encoder
        encoder = GraphPFN(edge_head=False)
        encoder_params = _extract_encoder_state_dict(sd)
        # GraphPFN owns the transformer under ``self.tfm``. Removing
        # ``encoder.tfm.`` would also remove the target model's ``tfm.``
        # namespace; strict=False then silently leaves the encoder at its
        # base initialization. Strip only the outer predictor namespace and
        # require every released weight to match.
        encoder.load_state_dict(encoder_params, strict=True)
        encoder = _prepare_for_inference(encoder, self.device)

        # 2. Determine embed_dim and hidden_dim from checkpoint weights
        # link_scorer.exist_head.0.weight shape is [hidden_dim, 4*embed_dim]
        exist_w = sd.get("link_scorer.exist_head.0.weight")
        if exist_w is not None:
            hidden_dim = exist_w.shape[0]
            embed_dim = exist_w.shape[1] // 4
        else:
            embed_dim = 192
            hidden_dim = 256

        # 3. Build and load LinkScorer
        link_scorer = _LinkScorer(embed_dim, hidden_dim)
        ls_params = {
            k.replace("link_scorer.", ""): v
            for k, v in sd.items()
            if k.startswith("link_scorer.")
        }
        link_scorer.load_state_dict(ls_params, strict=True)
        link_scorer = _prepare_for_inference(link_scorer, self.device)

        # 4. Build NodeHead dynamically from checkpoint structure
        nh_params = {
            k.replace("node_head.", ""): v
            for k, v in sd.items()
            if k.startswith("node_head.")
        }
        node_head = _build_node_head_from_params(nh_params)
        node_head = _prepare_for_inference(node_head, self.device)

        self._models[h_key] = encoder
        self._link_scorers[h_key] = link_scorer
        self._node_heads[h_key] = node_head

        n_params = sum(p.numel() for p in encoder.parameters())
        logger.info(
            "FM h=%d loaded: encoder=%d params, embed_dim=%d, device=%s",
            horizon, n_params, embed_dim, self.device,
        )

    def _validate_provenance(self, provenance: Any, horizon: int) -> None:
        """Reject incompatible new checkpoints before model construction."""
        if not isinstance(provenance, dict):
            raise ValueError("Checkpoint provenance must be an object")
        required = {
            "provenance_version",
            "data_sha256",
            "metadata_sha256",
            "source",
            "split",
            "seed",
            "epochs",
            "completed_epochs",
            "horizon",
            "preprocessing_contract_hash",
            "best_validation_metric",
            "best_validation_metric_name",
        }
        missing = sorted(required - set(provenance))
        if missing:
            raise ValueError(f"Checkpoint provenance missing fields: {missing}")
        if int(provenance["horizon"]) != horizon:
            raise ValueError(
                f"Checkpoint provenance horizon {provenance['horizon']} does not match requested {horizon}"
            )
        if provenance["provenance_version"] != 1:
            raise ValueError("Checkpoint provenance version is unsupported")
        for key in ("data_sha256", "metadata_sha256"):
            if not _is_sha256_hex(provenance[key]):
                raise ValueError(f"Checkpoint provenance {key} is invalid")
        expected_contract = self._feature_schema.get("preprocessing_contract_hash")
        if expected_contract is None or provenance["preprocessing_contract_hash"] != expected_contract:
            raise ValueError("Checkpoint provenance preprocessing contract is incompatible")
        if provenance["metadata_sha256"] != self._feature_schema.get("metadata_sha256"):
            raise ValueError("Checkpoint provenance metadata hash is incompatible")
        source = provenance["source"]
        source_required = {"git_commit", "dirty_patch_sha256", "source_file_sha256"}
        if (
            not isinstance(source, dict)
            or source_required - set(source)
            or not source.get("git_commit")
        ):
            raise ValueError("Checkpoint provenance source identity is incomplete")
        if any(
            not _is_sha256_hex(source.get(key))
            for key in ("dirty_patch_sha256", "source_file_sha256")
        ):
            raise ValueError("Checkpoint provenance source file identity is invalid")
        source_digest_values = [
            source.get("source_digest_sha256"),
            source.get("preflight_source_sha256"),
        ]
        present_source_digests = [
            value for value in source_digest_values if value is not None
        ]
        if any(not _is_sha256_hex(value) for value in present_source_digests):
            raise ValueError("Checkpoint provenance source digest is invalid")
        normalized_source_digests = {
            value.lower() for value in present_source_digests
        }
        if len(normalized_source_digests) > 1:
            raise ValueError("Checkpoint provenance source digests disagree")
        checkpoint_source_digest = (
            present_source_digests[0] if present_source_digests else None
        )
        if self.require_provenance:
            current_source_digest = os.environ.get(
                "PREFLIGHT_SOURCE_SHA256", ""
            ).strip()
            if not _is_sha256_hex(current_source_digest):
                raise ValueError(
                    "Authoritative inference requires current "
                    "PREFLIGHT_SOURCE_SHA256 as a 64-hex SHA-256 digest"
                )
            if not _is_sha256_hex(checkpoint_source_digest):
                raise ValueError(
                    "Authoritative checkpoint provenance requires a 64-hex "
                    "source digest"
                )
            if checkpoint_source_digest.lower() != current_source_digest.lower():
                raise ValueError(
                    "Checkpoint provenance source digest does not match current "
                    "PREFLIGHT_SOURCE_SHA256"
                )
        split = provenance["split"]
        if not isinstance(split, dict) or not split:
            raise ValueError("Checkpoint provenance split must be a non-empty object")
        if (
            not isinstance(provenance["seed"], int)
            or isinstance(provenance["seed"], bool)
            or provenance["seed"] < 0
        ):
            raise ValueError("Checkpoint provenance seed must be non-negative")
        epochs = provenance["epochs"]
        completed_epochs = provenance["completed_epochs"]
        if (
            not isinstance(epochs, int)
            or isinstance(epochs, bool)
            or epochs <= 0
            or not isinstance(completed_epochs, int)
            or isinstance(completed_epochs, bool)
            or not 1 <= completed_epochs <= epochs
        ):
            raise ValueError("Checkpoint provenance epoch counts are invalid")
        if not math.isfinite(float(provenance["best_validation_metric"])):
            raise ValueError("Checkpoint provenance validation metric must be finite")
        if not str(provenance["best_validation_metric_name"]).strip():
            raise ValueError("Checkpoint provenance validation metric name is empty")
        if self.data_path is not None:
            if not self.data_path.is_file():
                raise FileNotFoundError(f"Inference data file not found: {self.data_path}")
            if _sha256_file(self.data_path) != provenance["data_sha256"]:
                raise ValueError("Checkpoint provenance data hash is incompatible")
        if self.metadata_path is not None:
            if not self.metadata_path.is_file():
                raise FileNotFoundError(
                    f"Inference metadata file not found: {self.metadata_path}"
                )
            if _sha256_file(self.metadata_path) != provenance["metadata_sha256"]:
                raise ValueError("Checkpoint provenance metadata hash is incompatible")

    def predict(
        self,
        graph: GraphSnapshot,
        horizon: int,
    ) -> GraphSnapshot:
        """Generate a predicted GraphSnapshot using the FM model.

        Args:
            graph: Current observation GraphSnapshot.
            horizon: Forecast horizon in weeks.

        Returns:
            Predicted GraphSnapshot with FM-generated edges and weights.
        """
        if not self._available:
            raise RuntimeError(
                f"No usable FM checkpoints are available in {self.checkpoint_dir}"
            )

        h_key = self._get_horizon_key(horizon)
        self._load_model(horizon)

        encoder = self._models[h_key]
        link_scorer = self._link_scorers[h_key]
        node_head = self._node_heads[h_key]

        # Convert GraphSnapshot to tensor format
        node_ids = list(graph.nodes.keys())
        for node_id in node_ids:
            canonical_id = normalize_node_id(node_id)
            if canonical_id != node_id:
                raise ValueError(f"Graph node id is not canonical: {node_id!r}")
        graph_edges = aggregate_parallel_edges(graph.edges)
        n = len(node_ids)
        if n < 2:
            raise ValueError("FM inference requires at least two nodes")

        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Build node features tensor [N, F]
        features = self._build_features(graph, node_ids)

        # Build edge index for DGL graph
        src_list, dst_list, weight_list = [], [], []
        for edge in graph_edges:
            if edge.source in node_to_idx and edge.target in node_to_idx:
                src_list.append(node_to_idx[edge.source])
                dst_list.append(node_to_idx[edge.target])
                weight_list.append(math.log1p(max(edge.weight, 0.0)))

        if not src_list:
            raise ValueError("FM inference requires at least one valid edge")

        import dgl
        g = dgl.graph(
            (torch.tensor(src_list), torch.tensor(dst_list)),
            num_nodes=n,
        )

        # Encode with GraphPFN
        # LimiX adds a random feature positional embedding during every
        # forward pass. Fork and seed the RNG so the same checkpoint and graph
        # produce the same forecast across API processes and policy runs while
        # preserving the caller's surrounding random state.
        rng_devices = (
            [self.device.index]
            if self.device.type == "cuda" and self.device.index is not None
            else []
        )
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(self.inference_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.inference_seed)

            with torch.no_grad():
                h = self._encode(encoder, g, features)

                # Generate candidate pairs (existing edges + 2-hop neighbors)
                src_idx, dst_idx = self._get_candidate_pairs(
                    src_list, dst_list, n, max_pairs=min(n * n, 50000)
                )

                src_t = torch.tensor(src_idx, dtype=torch.long, device=self.device)
                dst_t = torch.tensor(dst_idx, dtype=torch.long, device=self.device)

                logits, w_pred = link_scorer(h, src_t, dst_t)
                probs = torch.sigmoid(logits).cpu().numpy()
                weights = w_pred.cpu().numpy()

                edge_src_t = torch.tensor(
                    src_list, dtype=torch.long, device=self.device
                )
                edge_dst_t = torch.tensor(
                    dst_list, dtype=torch.long, device=self.device
                )
                edge_weight_t = torch.tensor(
                    weight_list, dtype=torch.float32, device=self.device
                )
                node_deltas = (
                    node_head(h, edge_src_t, edge_dst_t, edge_weight_t)
                    .cpu()
                    .numpy()
                )

        # Build predicted graph: FM-driven strategy
        # Follows DeXposure-FM paper Eq.(9): ŵ_{τ+h} = w̃_τ + r̂_{pq,τ,h}
        # Use existence probability to decide IF an edge exists (binary),
        # and the weight residual to decide HOW MUCH it weighs (regression).
        # These are separate prediction heads trained with separate losses.
        edges: list[Edge] = []
        # Index FM predictions by (src_idx, dst_idx)
        fm_pred: dict[tuple[int, int], tuple[float, float]] = {}
        for i in range(len(src_idx)):
            fm_pred[(src_idx[i], dst_idx[i])] = (float(probs[i]), float(weights[i]))

        # Current edge lookup
        current_edges: dict[tuple[str, str], float] = {}
        for edge in graph_edges:
            current_edges[(edge.source, edge.target)] = edge.weight

        n_removed = 0
        for edge in graph_edges:
            s_id, t_id = edge.source, edge.target
            if s_id not in node_to_idx or t_id not in node_to_idx:
                edges.append(edge)
                continue

            si, ti = node_to_idx[s_id], node_to_idx[t_id]
            if (si, ti) in fm_pred:
                prob, w_residual = fm_pred[(si, ti)]
                if prob < self.pi_min:
                    n_removed += 1
                    continue
                new_w = self._decode_weight(edge.weight, w_residual)
                if new_w > 0:
                    edges.append(Edge(source=s_id, target=t_id, weight=new_w))
            else:
                # No FM prediction for this edge — keep as-is
                edges.append(edge)

        # Step 2: Add NEW edges with high FM confidence
        n_new = 0
        for (si, ti), (prob, w_residual) in fm_pred.items():
            s_id = node_ids[si]
            t_id = node_ids[ti]
            if (s_id, t_id) not in current_edges and prob >= self.pi_min:
                w = self._decode_weight(0.0, w_residual)
                if w <= 0.0:
                    continue
                edges.append(Edge(source=s_id, target=t_id, weight=w))
                n_new += 1

        # The node head predicts log-TVL change. Keep the full observed node
        # universe so forecast evaluation is not conditioned on predicted edges.
        nodes: dict[str, NodeFeatures] = {}
        for index, nid in enumerate(node_ids):
            current = graph.nodes[nid]
            delta = float(node_deltas[index])
            if not math.isfinite(delta):
                raise ValueError(f"Non-finite node prediction for {nid}: {delta}")
            nodes[nid] = current.model_copy(
                update={"log_size": max(0.0, current.log_size + delta)}
            )

        logger.info(
            "FM predict h=%d: %d nodes, %d edges (kept=%d, removed=%d, new=%d, candidates=%d, pi_min=%.2f)",
            horizon, len(nodes), len(edges), len(edges) - n_new, n_removed, n_new,
            len(src_idx), self.pi_min,
        )

        return GraphSnapshot(
            date=graph.date,
            nodes=nodes,
            edges=edges,
            source_node_records_total=graph.source_node_records_total,
            source_node_records_invalid=graph.source_node_records_invalid,
            source_edge_records_total=graph.source_edge_records_total,
            source_edge_records_invalid=graph.source_edge_records_invalid,
        )

    @staticmethod
    def _decode_weight(baseline_weight: float, residual: float) -> float:
        """Reconstruct a USD weight from a residual predicted in log1p space."""
        if baseline_weight < 0.0 or not math.isfinite(baseline_weight):
            raise ValueError(f"Invalid baseline edge weight: {baseline_weight}")
        if not math.isfinite(residual):
            raise ValueError(f"Non-finite edge-weight residual: {residual}")
        predicted_log_weight = max(0.0, math.log1p(baseline_weight) + residual)
        try:
            return math.expm1(predicted_log_weight)
        except OverflowError as exc:
            raise ValueError(
                f"Predicted log edge weight overflows: {predicted_log_weight}"
            ) from exc

    def _build_features(
        self, graph: GraphSnapshot, node_ids: list[str]
    ) -> torch.Tensor:
        """Build feature tensor [N, F] from GraphSnapshot node features."""
        matrix = build_feature_matrix(graph, self._categories)
        if matrix.shape[0] != len(node_ids):
            raise ValueError("Feature matrix node count does not match graph")
        return torch.from_numpy(matrix)

    def _encode(
        self, encoder, graph, features: torch.Tensor
    ) -> torch.Tensor:
        """Encode graph using GraphPFN with proper setup."""
        from lib.graphpfn.model import GraphPFNLayerWrapper
        from lib.limix.model.layer import MultiheadAttention as MHA

        num_nodes = graph.num_nodes()
        device = self.device

        train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        train_mask[-1] = False  # At least 1 test node required

        n_train = int(train_mask.sum().item())
        y_train = torch.zeros(n_train, device=device)

        features = features.to(device)
        graph = graph.to(device)

        tfm_features = torch.cat([features[train_mask], features[~train_mask]], dim=0)

        for module in encoder.modules():
            if isinstance(module, GraphPFNLayerWrapper):
                module.train_mask = train_mask
                module.graph = graph
            if isinstance(module, MHA):
                module.batched = False

        # Use appropriate SDPA backend
        if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
            if device.type == "cpu":
                backends = [torch.nn.attention.SDPBackend.MATH]
            else:
                backends = [
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                ]
            with torch.nn.attention.sdpa_kernel(backends):
                out = encoder.tfm.forward(
                    x=tfm_features.unsqueeze(0),
                    y=y_train.unsqueeze(0),
                    eval_pos=n_train,
                    task_type="reg",
                    checkpointing=False,
                )
        else:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            ):
                out = encoder.tfm.forward(
                    x=tfm_features.unsqueeze(0),
                    y=y_train.unsqueeze(0),
                    eval_pos=n_train,
                    task_type="reg",
                    checkpointing=False,
                )

        inv_order = torch.argsort((~train_mask).float(), stable=True)
        order = torch.argsort(inv_order, stable=True)

        if "encoder_out_full" in out:
            feature_embeds = out["encoder_out_full"][:, :, :-1, :]
            node_embeds = feature_embeds.mean(dim=2).squeeze(0)[order]
        else:
            node_embeds = out["encoder_embed"].squeeze(0)[order]

        return node_embeds

    def _get_candidate_pairs(
        self,
        src_list: list[int],
        dst_list: list[int],
        n: int,
        max_pairs: int = 50000,
    ) -> tuple[list[int], list[int]]:
        """Get candidate node pairs for prediction.

        Includes all existing edges + 2-hop neighbors + random negatives.
        """
        pairs = set()
        # Include all existing edges
        for s, d in zip(src_list, dst_list):
            pairs.add((s, d))

        # Add 2-hop neighbors
        adj: dict[int, set[int]] = {}
        for s, d in zip(src_list, dst_list):
            adj.setdefault(s, set()).add(d)
            adj.setdefault(d, set()).add(s)

        for node in list(adj.keys()):
            for neighbor in list(adj.get(node, [])):
                for hop2 in adj.get(neighbor, []):
                    if hop2 != node:
                        pairs.add((node, hop2))
                        if len(pairs) >= max_pairs:
                            break
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break

        # If still under budget, add random samples
        rng = np.random.default_rng(42)
        attempts = 0
        while len(pairs) < min(max_pairs, n * (n - 1)) and attempts < max_pairs * 2:
            s = int(rng.integers(0, n))
            d = int(rng.integers(0, n))
            if s != d:
                pairs.add((s, d))
            attempts += 1

        pair_list = sorted(pairs)
        return [p[0] for p in pair_list], [p[1] for p in pair_list]


def _build_node_head_from_params(params: dict) -> nn.Module:
    """Build a NodeHead nn.Sequential that matches the checkpoint param keys.

    h1 checkpoint uses: net.0 -> ReLU -> Dropout -> net.3 -> ReLU -> net.5
    h4/h8 checkpoints use: net.0 -> ReLU -> net.2
    We detect which variant by checking which keys exist.
    """
    # Find all linear layer indices
    weight_keys = sorted(k for k in params if k.endswith(".weight"))
    layers = []
    for wk in weight_keys:
        idx = wk.split(".")[1]  # e.g. "0", "2", "3", "5"
        w = params[wk]
        b = params.get(wk.replace(".weight", ".bias"))
        linear = nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data = w
        if b is not None:
            linear.bias.data = b
        # Add ReLU + Dropout between layers (except after last)
        if layers:
            layers.append(nn.ReLU())
            if int(idx) - int(weight_keys[len(layers)//2 - 1].split(".")[1]) > 2:
                layers.append(nn.Dropout(0.1))
        layers.append(linear)

    module = nn.Sequential(*layers)
    # Wrap in a module that handles the inflow/outflow aggregation
    return _NodeHeadWrapper(module)


def _extract_encoder_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Map ``GraphPFNLinkPredictor.encoder`` keys back to ``GraphPFN`` keys."""
    encoder_params = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_params or not all(key.startswith("tfm.") for key in encoder_params):
        raise ValueError("Checkpoint does not contain a valid GraphPFN encoder state")
    return encoder_params


def _prepare_for_inference(module: nn.Module, device: torch.device) -> nn.Module:
    """Move, freeze, and disable stochastic training behavior."""
    module = module.to(device)
    module.requires_grad_(False)
    module.eval()
    return module


# ---- Model component replicas (must match checkpoint structure) ----


class _LinkScorer(nn.Module):
    """Replica of LinkScorer from run_full_experiment.py."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = 4 * embed_dim
        self.exist_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.weight_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
        h_u, h_v = h[src], h[dst]
        z = torch.cat([h_u, h_v, h_u * h_v, (h_u - h_v).abs()], dim=-1)
        return self.exist_head(z).squeeze(-1), self.weight_head(z).squeeze(-1)


class _NodeHeadWrapper(nn.Module):
    """Wraps a sequential net with inflow/outflow neighbor aggregation."""

    def __init__(self, net: nn.Sequential):
        super().__init__()
        self.net = net

    def forward(self, h, edge_src=None, edge_dst=None, edge_weight=None):
        # Check if the net expects 3*embed_dim (neighbor-aware) or just embed_dim
        first_layer = self.net[0]
        expects_neighbors = first_layer.in_features > h.shape[-1]

        if expects_neighbors and edge_src is not None and edge_dst is not None and edge_weight is not None:
            # Match the training NodeHead exactly. The weights stored in the
            # graph are log1p values and are normalized globally before the
            # inflow/outflow aggregation.
            w = torch.softmax(edge_weight.float(), dim=0)
            inflow = torch.zeros_like(h)
            inflow.index_add_(0, edge_dst, h[edge_src] * w.unsqueeze(-1))
            outflow = torch.zeros_like(h)
            outflow.index_add_(0, edge_src, h[edge_dst] * w.unsqueeze(-1))
            x = torch.cat([h, inflow, outflow], dim=-1)
        elif expects_neighbors:
            x = torch.cat([h, torch.zeros_like(h), torch.zeros_like(h)], dim=-1)
        else:
            x = h

        return self.net(x).squeeze(-1)
