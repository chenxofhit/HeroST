"""Operational zone stratification from predicted boundaryness.

Quantile thresholds on model boundaryness define homogeneous, transition,
and boundary spots for zone-stratified evaluation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from heroST.data.synthetic import ZONE_BOUNDARY, ZONE_HOMOGENEOUS, ZONE_INFILTRATION, ZONE_TRANSITION


def compute_zone_thresholds(
    boundaryness: np.ndarray, quantiles: tuple[float, float] = (0.5, 0.85)
) -> tuple[float, float]:
    """Frozen-before-results quantile thresholds. Compute ONCE (e.g. on a
    held-out reference/validation set) and reuse the returned (tau_low,
    tau_high) everywhere downstream -- do not recompute per evaluation call
    on the test set, or thresholds become a hidden extra degree of freedom.
    """
    tau_low, tau_high = np.quantile(boundaryness, quantiles)
    return float(tau_low), float(tau_high)


def assign_operational_zones(
    boundaryness: np.ndarray,
    tau_low: float,
    tau_high: float,
    infiltration_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    boundaryness = np.asarray(boundaryness)
    zone = np.where(
        boundaryness >= tau_high,
        ZONE_BOUNDARY,
        np.where(boundaryness >= tau_low, ZONE_TRANSITION, ZONE_HOMOGENEOUS),
    )
    if infiltration_mask is not None:
        zone = np.where(np.asarray(infiltration_mask, dtype=bool), ZONE_INFILTRATION, zone)
    return zone
