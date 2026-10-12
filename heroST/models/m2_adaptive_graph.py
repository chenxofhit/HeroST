"""Structure-aware graph construction and message passing.

Edge weights combine spatial proximity, morphology similarity, and destination
boundary/complexity penalties. A monotone destination gate further limits
incoming messages at high-boundaryness spots. Softplus can keep combination
weights non-negative when enabled.
"""
from __future__ import annotations

from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

BoundaryPenaltyMode = Literal["both", "dst"]
GateSide = Literal["dst", "src", "sym"]


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus so a constrained weight initializes to `y` exactly:
    softplus(_inv_softplus(y)) == y for y > 0."""
    return torch.log(torch.expm1(y))


class AdaptiveEdgeWeighter(nn.Module):
    """Combine spatial proximity, morphology similarity, and a boundary/
    complexity penalty into a single learnable edge weight:

        edge_score = w_s * spatial_proximity + w_m * morphology_similarity
                     - w_b * boundary_penalty - w_c * complexity_penalty
        edge_weight = softmax_j(edge_score) over each node's neighbors j

    When `nonneg_weights` is set, every `w_*` is reparameterized as
    `softplus(raw_*)` so it can never train negative -- a negative
    `w_boundary`/`w_complexity` would turn the intended penalty into a reward.
    When `boundary_penalty_mode="dst"`, the penalty uses only the destination
    endpoint's boundaryness; because the edge softmax is grouped by source,
    the source-side half of the symmetric ("both") penalty is a per-group
    constant and cancels, so "dst" is required for the penalty to actually
    change relative out-edge weights.
    """

    def __init__(
        self,
        spatial_scale: float = 3.0,
        nonneg_weights: bool = False,
        boundary_penalty_mode: BoundaryPenaltyMode = "both",
    ) -> None:
        super().__init__()
        self.spatial_scale = spatial_scale
        self.nonneg_weights = nonneg_weights
        self.boundary_penalty_mode = boundary_penalty_mode
        if nonneg_weights:
            # Store raw pre-softplus values; effective weight = softplus(raw).
            # Initialized so effective weights reproduce the default 1/1/1/0.5 blend.
            self._raw_spatial = nn.Parameter(_inv_softplus(torch.tensor(1.0)))
            self._raw_morph = nn.Parameter(_inv_softplus(torch.tensor(1.0)))
            self._raw_boundary = nn.Parameter(_inv_softplus(torch.tensor(1.0)))
            self._raw_complexity = nn.Parameter(_inv_softplus(torch.tensor(0.5)))
        else:
            self.w_spatial = nn.Parameter(torch.tensor(1.0))
            self.w_morph = nn.Parameter(torch.tensor(1.0))
            self.w_boundary = nn.Parameter(torch.tensor(1.0))
            self.w_complexity = nn.Parameter(torch.tensor(0.5))

    def effective_weights(self) -> Dict[str, torch.Tensor]:
        if self.nonneg_weights:
            return {
                "w_spatial": F.softplus(self._raw_spatial),
                "w_morph": F.softplus(self._raw_morph),
                "w_boundary": F.softplus(self._raw_boundary),
                "w_complexity": F.softplus(self._raw_complexity),
            }
        return {
            "w_spatial": self.w_spatial,
            "w_morph": self.w_morph,
            "w_boundary": self.w_boundary,
            "w_complexity": self.w_complexity,
        }

    def weight_report(self) -> Dict[str, float]:
        """Learned scalar weights as plain floats, for logging every seed's
        final values (reviewer request: report all learned M2 weights)."""
        return {k: float(v.detach()) for k, v in self.effective_weights().items()}

    def forward(
        self,
        edge_index: torch.Tensor,  # (2, E)
        edge_dist: torch.Tensor,  # (E,)
        morphology_embedding: torch.Tensor,  # (N, D)
        boundaryness: torch.Tensor,  # (N,)
        complexity: torch.Tensor,  # (N,)
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        w = self.effective_weights()
        spatial_proximity = torch.exp(-edge_dist / self.spatial_scale)
        morph_sim = F.cosine_similarity(morphology_embedding[src], morphology_embedding[dst], dim=-1)
        if self.boundary_penalty_mode == "dst":
            boundary_penalty = boundaryness[dst]
            complexity_penalty = complexity[dst]
        else:
            boundary_penalty = 0.5 * (boundaryness[src] + boundaryness[dst])
            complexity_penalty = 0.5 * (complexity[src] + complexity[dst])

        edge_score = (
            w["w_spatial"] * spatial_proximity
            + w["w_morph"] * morph_sim
            - w["w_boundary"] * boundary_penalty
            - w["w_complexity"] * complexity_penalty
        )

        n = morphology_embedding.shape[0]
        edge_weight = _segment_softmax(edge_score, src, n)
        return edge_weight


def _segment_softmax(scores: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of `scores` grouped by `index` (i.e. per source node's outgoing
    edge set), implemented without external scatter deps."""
    max_per_node = torch.full((n,), float("-inf"), device=scores.device, dtype=scores.dtype)
    max_per_node = max_per_node.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
    shifted = scores - max_per_node[index]
    exp_scores = shifted.exp()
    sum_per_node = torch.zeros(n, device=scores.device, dtype=scores.dtype)
    sum_per_node.scatter_add_(0, index, exp_scores)
    return exp_scores / (sum_per_node[index] + 1e-8)


class AdaptiveMessagePassingLayer(nn.Module):
    """One layer of gated message passing.

    Nodes in homogeneous regions aggregate more from neighbours;
    boundary/transition nodes suppress aggregation and keep more of their
    own signal. With `monotone_gate=True`, gate strength is monotone
    non-increasing in boundaryness. With `use_node_gate=False`, gating is
    disabled (strength == 1).
    """

    def __init__(
        self,
        dim: int,
        monotone_gate: bool = False,
        use_node_gate: bool = True,
        gate_side: GateSide = "dst",
    ) -> None:
        super().__init__()
        if gate_side not in ("dst", "src", "sym"):
            raise ValueError(f"Unknown gate_side: {gate_side!r}")
        self.message_proj = nn.Linear(dim, dim)
        self.update_proj = nn.Linear(dim * 2, dim)
        self.use_node_gate = use_node_gate
        self.monotone_gate = monotone_gate
        self.gate_side = gate_side
        if use_node_gate:
            if monotone_gate:
                # strength = sigmoid(-(softplus(slope) * b + bias)); slope >= 0
                # so strength is monotone non-increasing in b. Init slope ~= 1.
                self._gate_raw_slope = nn.Parameter(_inv_softplus(torch.tensor(1.0)))
                self._gate_bias = nn.Parameter(torch.tensor(0.0))
            else:
                self.gate_head = nn.Sequential(nn.Linear(1, 8), nn.ReLU(inplace=True), nn.Linear(8, 1))

    def propagation_strength(self, boundaryness: torch.Tensor) -> torch.Tensor:
        if not self.use_node_gate:
            return torch.ones_like(boundaryness)
        if self.monotone_gate:
            slope = F.softplus(self._gate_raw_slope)
            return torch.sigmoid(-(slope * boundaryness + self._gate_bias))
        return torch.sigmoid(-self.gate_head(boundaryness.unsqueeze(-1)).squeeze(-1))

    def forward(
        self,
        node_features: torch.Tensor,  # (N, D)
        edge_index: torch.Tensor,  # (2, E)
        edge_weight: torch.Tensor,  # (E,)
        boundaryness: torch.Tensor,  # (N,) used as the per-node propagation gate
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n, d = node_features.shape
        messages = self.message_proj(node_features)[src] * edge_weight.unsqueeze(-1)
        if self.gate_side != "dst":
            if self.gate_side == "src":
                edge_gate = self.propagation_strength(boundaryness)[src]
            else:
                edge_gate = self.propagation_strength(0.5 * (boundaryness[src] + boundaryness[dst]))
            messages = messages * edge_gate.unsqueeze(-1)
        aggregated = torch.zeros(n, d, device=node_features.device, dtype=node_features.dtype)
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand(-1, d), messages)

        if self.gate_side == "dst":
            # High boundaryness lowers incoming message mass at the destination.
            gated_aggregated = aggregated * self.propagation_strength(boundaryness).unsqueeze(-1)
        else:
            gated_aggregated = aggregated

        updated = self.update_proj(torch.cat([node_features, gated_aggregated], dim=-1))
        return F.relu(updated) + node_features  # residual


class StructureAwareGraphPropagator(nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int = 2,
        spatial_scale: float = 3.0,
        nonneg_weights: bool = False,
        monotone_gate: bool = False,
        boundary_penalty_mode: BoundaryPenaltyMode = "both",
        use_edge_weighter: bool = True,
        use_node_gate: bool = True,
        gate_side: GateSide = "dst",
    ) -> None:
        super().__init__()
        self.use_edge_weighter = use_edge_weighter
        self.edge_weighter = AdaptiveEdgeWeighter(
            spatial_scale=spatial_scale,
            nonneg_weights=nonneg_weights,
            boundary_penalty_mode=boundary_penalty_mode,
        )
        self.layers = nn.ModuleList(
            [
                AdaptiveMessagePassingLayer(
                    dim, monotone_gate=monotone_gate, use_node_gate=use_node_gate, gate_side=gate_side
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
        morphology_embedding: torch.Tensor,
        boundaryness: torch.Tensor,
        complexity: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_edge_weighter:
            edge_weight = self.edge_weighter(
                edge_index, edge_dist, morphology_embedding, boundaryness, complexity
            )
        else:
            # Uniform within each source group: segment-softmax of zeros == 1/deg.
            n = morphology_embedding.shape[0]
            zeros = torch.zeros(edge_index.shape[1], device=node_features.device, dtype=node_features.dtype)
            edge_weight = _segment_softmax(zeros, edge_index[0], n)
        h = node_features
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight, boundaryness)
        return h

    def weight_report(self) -> Dict[str, float]:
        return self.edge_weighter.weight_report()


class FixedUniformGraphPropagator(nn.Module):
    """Uniform spatial k-NN propagator without boundary-aware gating.

    Useful as a control that applies the same mixing strength on every edge.
    """

    def __init__(self, dim: int, n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
        morphology_embedding: torch.Tensor,
        boundaryness: torch.Tensor,
        complexity: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n, d = node_features.shape
        h = node_features
        for layer in self.layers:
            proj = layer(h)
            aggregated = torch.zeros(n, d, device=h.device, dtype=h.dtype)
            deg = torch.zeros(n, device=h.device, dtype=h.dtype)
            aggregated.scatter_add_(0, dst.unsqueeze(-1).expand(-1, d), proj[src])
            deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            aggregated = aggregated / (deg.unsqueeze(-1) + 1e-8)
            h = F.relu(0.5 * h + 0.5 * aggregated)
        return h
