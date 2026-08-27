"""Core HERO-ST model: structural encoding, selective graph propagation,
and expression-guided training with an image-only inference path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from heroST.models.m1_structural_encoder import StructuralPriorEncoder, M1Output
from heroST.models.m2_adaptive_graph import StructureAwareGraphPropagator, FixedUniformGraphPropagator
from heroST.models.m3_uncertainty_alignment import UncertaintyAwareAligner, M3Output

GraphVariant = Literal["adaptive", "fixed"]


@dataclass
class HeroSTOutput:
    m1: M1Output
    graph_embedding: torch.Tensor
    m3: M3Output


@dataclass
class HeroSTConfig:
    n_genes: int
    embed_dim: int = 64
    backbone_name: str = "tiny_cnn"
    n_message_passing_layers: int = 2
    spatial_scale: float = 3.0
    graph_variant: GraphVariant = "adaptive"
    m2_nonneg_weights: bool = True
    m2_monotone_gate: bool = True
    m2_boundary_penalty_mode: str = "dst"
    m2_use_edge_weighter: bool = True
    m2_use_node_gate: bool = True
    n_cross_attn_heads: int = 4


class HeroSTModel(nn.Module):
    def __init__(self, cfg: HeroSTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.m1 = StructuralPriorEncoder(
            embed_dim=cfg.embed_dim,
            backbone_name=cfg.backbone_name,
        )

        if cfg.graph_variant == "adaptive":
            self.m2 = StructureAwareGraphPropagator(
                dim=cfg.embed_dim,
                n_layers=cfg.n_message_passing_layers,
                spatial_scale=cfg.spatial_scale,
                nonneg_weights=cfg.m2_nonneg_weights,
                monotone_gate=cfg.m2_monotone_gate,
                boundary_penalty_mode=cfg.m2_boundary_penalty_mode,
                use_edge_weighter=cfg.m2_use_edge_weighter,
                use_node_gate=cfg.m2_use_node_gate,
            )
        elif cfg.graph_variant == "fixed":
            self.m2 = FixedUniformGraphPropagator(dim=cfg.embed_dim, n_layers=cfg.n_message_passing_layers)
        else:
            raise ValueError(f"Unknown graph_variant: {cfg.graph_variant}")

        self.m3 = UncertaintyAwareAligner(
            embed_dim=cfg.embed_dim, n_genes=cfg.n_genes, n_heads=cfg.n_cross_attn_heads
        )

    def set_feature_cache(self, cache_dir) -> None:
        self.m1.set_feature_cache(cache_dir)

    def forward(
        self,
        patches: torch.Tensor,
        coords: torch.Tensor,
        spatial_edge_index: torch.Tensor,
        spatial_edge_dist: torch.Tensor,
        expression_true: Optional[torch.Tensor] = None,
        cache_key: Optional[str] = None,
    ) -> HeroSTOutput:
        m1_out = self.m1(patches, cache_key=cache_key)
        graph_embedding = self.m2(
            m1_out.morphology_embedding,
            spatial_edge_index,
            spatial_edge_dist,
            m1_out.morphology_embedding,
            m1_out.boundaryness,
            m1_out.complexity,
        )
        m3_out = self.m3(graph_embedding, expression_true=expression_true)
        return HeroSTOutput(m1=m1_out, graph_embedding=graph_embedding, m3=m3_out)
