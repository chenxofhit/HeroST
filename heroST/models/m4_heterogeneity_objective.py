"""M4 -- local heterogeneity losses used by the five-term HERO-ST objective.

1. `boundary_hard_negative_loss`: spatially adjacent pairs that also have high
   morphological boundaryness and large true expression differences are pushed
   apart; homogeneous neighbours are pulled together.
2. `local_difference_preservation_loss`: predicted expression jumps on graph
   edges are matched to measured jumps, with boundary edges up-weighted.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _pair_boundaryness(boundaryness: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return 0.5 * (boundaryness[src] + boundaryness[dst])


def boundary_hard_negative_loss(
    embedding: torch.Tensor,
    expression_true: torch.Tensor,
    edge_index: torch.Tensor,
    boundaryness: torch.Tensor,
    margin: float = 1.0,
    boundary_quantile: float = 0.7,
    expr_diff_quantile: float = 0.7,
) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    pair_bnd = _pair_boundaryness(boundaryness, src, dst)
    expr_diff = (expression_true[src] - expression_true[dst]).norm(dim=-1)

    bnd_thresh = torch.quantile(pair_bnd, boundary_quantile)
    diff_thresh = torch.quantile(expr_diff, expr_diff_quantile)
    is_hard_negative = (pair_bnd >= bnd_thresh) & (expr_diff >= diff_thresh)
    is_positive = (pair_bnd < bnd_thresh) & (expr_diff < diff_thresh)

    emb_dist = (embedding[src] - embedding[dst]).norm(dim=-1)
    pos_loss = emb_dist[is_positive].pow(2).mean() if is_positive.any() else emb_dist.new_zeros(())
    if is_hard_negative.any():
        neg_loss = F.relu(margin - emb_dist[is_hard_negative]).pow(2).mean()
    else:
        neg_loss = emb_dist.new_zeros(())
    return pos_loss + neg_loss


def local_difference_preservation_loss(
    mu: torch.Tensor,
    expression_true: torch.Tensor,
    edge_index: torch.Tensor,
    boundaryness: torch.Tensor,
    boundary_upweight: float = 3.0,
) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    pred_diff = mu[src] - mu[dst]
    true_diff = expression_true[src] - expression_true[dst]
    per_edge_loss = (pred_diff - true_diff).pow(2).mean(dim=-1)
    pair_bnd = _pair_boundaryness(boundaryness, src, dst)
    weight = 1.0 + boundary_upweight * pair_bnd
    return (weight * per_edge_loss).sum() / (weight.sum() + 1e-8)


def m4_core_losses(
    embedding: torch.Tensor,
    mu: torch.Tensor,
    expression_true: torch.Tensor,
    edge_index: torch.Tensor,
    boundaryness: torch.Tensor,
    margin: float = 1.0,
    boundary_upweight: float = 3.0,
    w_contrastive: float = 1.0,
    w_local_diff: float = 1.0,
) -> dict[str, torch.Tensor]:
    contrastive = boundary_hard_negative_loss(
        embedding, expression_true, edge_index, boundaryness, margin=margin
    )
    local_diff = local_difference_preservation_loss(
        mu, expression_true, edge_index, boundaryness, boundary_upweight=boundary_upweight
    )
    return {
        "boundary_contrastive": contrastive,
        "local_diff_preservation": local_diff,
        "total": w_contrastive * contrastive + w_local_diff * local_diff,
    }
