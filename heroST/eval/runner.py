"""Zone-stratified evaluation shared by train.py and evaluate.py.

Zones are assigned from the model's predicted boundaryness, never from an
oracle domain map.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from heroST.data.dataset import SlideGraphBatch, SlideGraphDataset, collate_single_slide
from heroST.eval.metrics import (
    boundary_contrast_preservation,
    global_metrics,
    local_variance_ratio,
    oversmoothing_check,
    zone_stratified_pcc,
)
from heroST.eval.zones import assign_operational_zones, compute_zone_thresholds
from heroST.models.hero_st import HeroSTModel


@dataclass
class ZoneThresholds:
    tau_low: float
    tau_high: float


def compute_frozen_zone_thresholds(
    model: HeroSTModel, reference_dataset: SlideGraphDataset, quantiles: tuple[float, float], device: torch.device
) -> ZoneThresholds:
    model.eval()
    all_boundaryness = []
    loader = DataLoader(reference_dataset, batch_size=1, collate_fn=collate_single_slide)
    with torch.no_grad():
        for batch in loader:
            keep_cpu = int(batch.patches.shape[0]) >= 1024
            batch = batch.to(device, keep_patches_on_cpu=keep_cpu)
            out = model(batch.patches, batch.coords, batch.spatial_edge_index, batch.spatial_edge_dist)
            all_boundaryness.append(out.m1.boundaryness.cpu().numpy())
            if device.type == "cuda":
                torch.cuda.empty_cache()
    boundaryness = np.concatenate(all_boundaryness)
    tau_low, tau_high = compute_zone_thresholds(boundaryness, quantiles=quantiles)
    return ZoneThresholds(tau_low=tau_low, tau_high=tau_high)


def evaluate_single_slide(
    model: HeroSTModel,
    batch: SlideGraphBatch,
    thresholds: ZoneThresholds,
    device: torch.device,
    use_oracle_infiltration_as_proxy: bool = True,
) -> Dict[str, object]:
    model.eval()
    batch = batch.to(device)
    with torch.no_grad():
        out = model(batch.patches, batch.coords, batch.spatial_edge_index, batch.spatial_edge_dist)

    mu = out.m3.mu.cpu().numpy()
    target = batch.expression.cpu().numpy()
    boundaryness = out.m1.boundaryness.cpu().numpy()
    edge_index = batch.spatial_edge_index.cpu().numpy()
    n_spots = int(mu.shape[0])

    infiltration = batch.infiltration_mask.cpu().numpy() if use_oracle_infiltration_as_proxy else None
    zone = assign_operational_zones(boundaryness, thresholds.tau_low, thresholds.tau_high, infiltration)

    src, dst = edge_index[0], edge_index[1]
    edge_zone_mask = (zone[src] == "boundary") | (zone[dst] == "boundary")

    return {
        "slide_id": batch.slide_id,
        "n_spots": n_spots,
        "zone_counts": {z: int((zone == z).sum()) for z in sorted(set(zone.tolist()))},
        "zone_stratified_pcc": zone_stratified_pcc(mu, target, zone),
        "local_variance_ratio": local_variance_ratio(mu, target, edge_index),
        "boundary_contrast_preservation": boundary_contrast_preservation(mu, target, edge_index, edge_zone_mask),
        "oversmoothing": oversmoothing_check(mu, target, edge_index),
        "global_metrics": global_metrics(mu, target),
    }


def _mean_ignore_nan(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def aggregate_slide_results(per_slide: List[Dict[str, object]]) -> Dict[str, object]:
    def _avg_nested(key: str) -> Dict[str, float]:
        all_zones = set()
        for r in per_slide:
            all_zones |= set(r[key].keys())  # type: ignore[union-attr]
        return {z: _mean_ignore_nan([r[key].get(z, float("nan")) for r in per_slide]) for z in all_zones}  # type: ignore[union-attr]

    return {
        "n_slides": len(per_slide),
        "zone_stratified_pcc": _avg_nested("zone_stratified_pcc"),
        "local_variance_ratio": _mean_ignore_nan([r["local_variance_ratio"] for r in per_slide]),  # type: ignore[arg-type]
        "boundary_contrast_preservation": _mean_ignore_nan(
            [r["boundary_contrast_preservation"] for r in per_slide]  # type: ignore[arg-type]
        ),
        "oversmoothing_delta": _mean_ignore_nan([r["oversmoothing"]["delta"] for r in per_slide]),  # type: ignore[index]
        "oversmoothing_flagged_frac": float(
            np.mean([1.0 if r["oversmoothing"]["oversmoothing_flagged"] else 0.0 for r in per_slide])  # type: ignore[index]
        ),
        "global_mean_pcc": _mean_ignore_nan([r["global_metrics"]["mean_pcc"] for r in per_slide]),  # type: ignore[index]
        "global_mse": _mean_ignore_nan([r["global_metrics"]["mse"] for r in per_slide]),  # type: ignore[index]
    }


def evaluate_dataset(
    model: HeroSTModel,
    dataset: SlideGraphDataset,
    thresholds: ZoneThresholds,
    device: torch.device,
) -> Dict[str, object]:
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_single_slide)
    per_slide = [evaluate_single_slide(model, batch, thresholds, device) for batch in loader]
    return {"per_slide": per_slide, "aggregate": aggregate_slide_results(per_slide)}
