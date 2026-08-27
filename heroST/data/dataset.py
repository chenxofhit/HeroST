"""Per-slide graph dataset wrapper around SyntheticSlide (or a real HEST-1k
cohort via heroST.data.hest.load_hest_cohort, which returns the same
SyntheticSlide-shaped objects -- see that module for what is/isn't real for
non-synthetic slides).

Each dataset item is one whole slide, represented as a graph: spatial k-NN
edges (the "fixed" graph M2's structure-aware reweighting operates on top
of), spot patches, expression, and the oracle structural labels used only
for sanity-checking M1 (never fed to the model at train/eval time).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from heroST.data.synthetic import SyntheticSlide


@dataclass
class SlideGraphBatch:
    slide_id: str
    coords: torch.Tensor  # (N, 2)
    patches: torch.Tensor  # (N, 3, H, W)
    expression: torch.Tensor  # (N, G) normalized, model target
    expression_raw: torch.Tensor  # (N, G) un-normalized, for interpretable metrics
    domain_id: torch.Tensor  # (N,)
    infiltration_mask: torch.Tensor  # (N,) bool
    oracle_boundaryness: torch.Tensor  # (N,)
    oracle_zone: List[str]
    marker_gene_idx: torch.Tensor
    immune_gene_idx: torch.Tensor
    spatial_edge_index: torch.Tensor  # (2, E) kNN edges from raw coordinates only
    spatial_edge_dist: torch.Tensor  # (E,)

    def to(self, device: torch.device, *, keep_patches_on_cpu: bool = False) -> "SlideGraphBatch":
        """Move tensors to ``device``.

        For very large slides (e.g. Xenium OOD with 10k–20k spots), keeping
        patches on CPU avoids a multi-GB VRAM spike; frozen backbones then
        stream patch chunks to GPU internally.
        """
        patches = self.patches if keep_patches_on_cpu else self.patches.to(device)
        return SlideGraphBatch(
            slide_id=self.slide_id,
            coords=self.coords.to(device),
            patches=patches,
            expression=self.expression.to(device),
            expression_raw=self.expression_raw.to(device),
            domain_id=self.domain_id.to(device),
            infiltration_mask=self.infiltration_mask.to(device),
            oracle_boundaryness=self.oracle_boundaryness.to(device),
            oracle_zone=self.oracle_zone,
            marker_gene_idx=self.marker_gene_idx.to(device),
            immune_gene_idx=self.immune_gene_idx.to(device),
            spatial_edge_index=self.spatial_edge_index.to(device),
            spatial_edge_dist=self.spatial_edge_dist.to(device),
        )


def build_spatial_knn_graph(coords: np.ndarray, k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """k-NN graph from raw spot coordinates only (no morphology or boundary
    penalty). Used as the coordinate graph skeleton before adaptive weighting.
    """
    tree = cKDTree(coords)
    dists, idx = tree.query(coords, k=k + 1)  # includes self at k=0
    src = np.repeat(np.arange(coords.shape[0]), k)
    dst = idx[:, 1:].reshape(-1)
    edge_dist = dists[:, 1:].reshape(-1)
    edge_index = np.stack([src, dst], axis=0)
    return edge_index, edge_dist


class ExpressionNormalizer:
    """log1p + per-gene z-score, fit on train slides only.

    Fitting on train only and freezing before touching val/test is required
    to avoid leakage.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.fitted_ = False

    def fit(self, slides: Sequence[SyntheticSlide]) -> "ExpressionNormalizer":
        all_log = np.concatenate([np.log1p(s.expression) for s in slides], axis=0)
        self.mean_ = all_log.mean(axis=0)
        self.std_ = all_log.std(axis=0) + 1e-6
        self.fitted_ = True
        return self

    def transform(self, expression: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("ExpressionNormalizer must be fit on the train split before use.")
        log_expr = np.log1p(expression)
        return (log_expr - self.mean_) / self.std_

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        log_expr = normalized * self.std_ + self.mean_
        return np.expm1(log_expr)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean_, "std": self.std_}


class SlideGraphDataset(Dataset):
    """One dataset item == one whole slide (variable spot count per slide).

    Use `batch_size=1` with the identity collate function `collate_single_slide`
    when wrapping this in a DataLoader, since graph models here operate on a
    whole slide's spot graph at once rather than on i.i.d. spot batches.
    """

    def __init__(
        self,
        slides: Sequence[SyntheticSlide],
        normalizer: ExpressionNormalizer,
        knn_k: int = 6,
    ) -> None:
        self.slides = list(slides)
        self.normalizer = normalizer
        self.knn_k = knn_k

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> SlideGraphBatch:
        slide = self.slides[idx]
        edge_index, edge_dist = build_spatial_knn_graph(slide.coords, k=self.knn_k)
        normalized_expr = self.normalizer.transform(slide.expression)
        return SlideGraphBatch(
            slide_id=slide.slide_id,
            coords=torch.from_numpy(slide.coords).float(),
            patches=torch.from_numpy(slide.patches).float(),
            expression=torch.from_numpy(normalized_expr).float(),
            expression_raw=torch.from_numpy(slide.expression).float(),
            domain_id=torch.from_numpy(slide.domain_id).long(),
            infiltration_mask=torch.from_numpy(slide.infiltration_mask).bool(),
            oracle_boundaryness=torch.from_numpy(slide.oracle_boundaryness).float(),
            oracle_zone=list(slide.oracle_zone),
            marker_gene_idx=torch.from_numpy(slide.marker_gene_idx).long(),
            immune_gene_idx=torch.from_numpy(slide.immune_gene_idx).long(),
            spatial_edge_index=torch.from_numpy(edge_index).long(),
            spatial_edge_dist=torch.from_numpy(edge_dist).float(),
        )


def collate_single_slide(batch: List[SlideGraphBatch]) -> SlideGraphBatch:
    assert len(batch) == 1, "SlideGraphDataset must be used with batch_size=1 (whole-slide graphs)."
    return batch[0]
