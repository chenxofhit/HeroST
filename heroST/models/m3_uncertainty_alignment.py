"""M3 -- image-only prediction with a training-time expression teacher.

At inference only H&E is available, so true expression never enters the
image-only path. During training, a privileged expression encoder supplies
teacher means and embeddings for consistency and InfoNCE alignment. The
manuscript five-term objective uses an always-on teacher gate (every spot
weighted equally).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class M3Output:
    mu: torch.Tensor
    log_var: torch.Tensor
    mu_teacher: Optional[torch.Tensor] = None
    h_img: Optional[torch.Tensor] = None
    h_expr: Optional[torch.Tensor] = None


class ExpressionTeacherEncoder(nn.Module):
    def __init__(self, n_genes: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.net(expression)


class UncertaintyAwareAligner(nn.Module):
    def __init__(self, embed_dim: int, n_genes: int, n_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.expr_encoder = ExpressionTeacherEncoder(n_genes, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.teacher_fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, embed_dim)
        )
        self.predict_mu = nn.Linear(embed_dim, n_genes)
        self.predict_log_var = nn.Linear(embed_dim, n_genes)

    def _predict(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.predict_mu(h)
        log_var = self.predict_log_var(h).clamp(min=-8.0, max=8.0)
        return mu, log_var

    def forward(
        self,
        h_img: torch.Tensor,
        expression_true: Optional[torch.Tensor] = None,
    ) -> M3Output:
        mu, log_var = self._predict(h_img)
        if expression_true is None:
            return M3Output(mu=mu, log_var=log_var, h_img=h_img)

        h_expr = self.expr_encoder(expression_true)
        q, k, v = h_img.unsqueeze(0), h_expr.unsqueeze(0), h_expr.unsqueeze(0)
        teacher_context, _ = self.cross_attn(q, k, v)
        teacher_context = teacher_context.squeeze(0)
        h_teacher = self.teacher_fuse(torch.cat([h_img, teacher_context], dim=-1))
        mu_teacher, _ = self._predict(h_teacher)
        return M3Output(
            mu=mu,
            log_var=log_var,
            mu_teacher=mu_teacher,
            h_img=h_img,
            h_expr=h_expr,
        )


def heteroscedastic_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (0.5 * torch.exp(-log_var) * (mu - target).pow(2) + 0.5 * log_var).mean()


def infonce_contrastive_loss(
    h_img: torch.Tensor, h_expr: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    img_n = F.normalize(h_img, dim=-1)
    expr_n = F.normalize(h_expr, dim=-1)
    logits = img_n @ expr_n.t() / temperature
    targets = torch.arange(h_img.shape[0], device=h_img.device)
    return F.cross_entropy(logits, targets)


def teacher_consistency_loss(mu: torch.Tensor, mu_teacher: torch.Tensor) -> torch.Tensor:
    return (mu - mu_teacher.detach()).pow(2).mean()


def m3_losses(
    out: M3Output,
    target: torch.Tensor,
    contrastive_temperature: float = 0.1,
    w_consistency: float = 1.0,
    w_contrastive: float = 0.5,
) -> dict[str, torch.Tensor]:
    losses = {"heteroscedastic": heteroscedastic_nll(out.mu, out.log_var, target)}
    if out.mu_teacher is None or out.h_img is None or out.h_expr is None:
        losses["total"] = losses["heteroscedastic"]
        return losses

    losses["consistency"] = teacher_consistency_loss(out.mu, out.mu_teacher)
    losses["contrastive"] = infonce_contrastive_loss(
        out.h_img, out.h_expr, temperature=contrastive_temperature
    )
    losses["total"] = (
        losses["heteroscedastic"]
        + w_consistency * losses["consistency"]
        + w_contrastive * losses["contrastive"]
    )
    return losses
