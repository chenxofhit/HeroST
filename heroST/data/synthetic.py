"""Synthetic slide generator for offline smoke tests.

Produces small multi-domain cohorts with patches, coordinates, and expression
so the full train/eval path can run without HEST or network access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

ZONE_HOMOGENEOUS = "homogeneous"
ZONE_BOUNDARY = "boundary"
ZONE_TRANSITION = "transition"
ZONE_INFILTRATION = "infiltration"
ZONE_NAMES = [ZONE_HOMOGENEOUS, ZONE_TRANSITION, ZONE_BOUNDARY, ZONE_INFILTRATION]


@dataclass
class SyntheticSlide:
    slide_id: str
    coords: np.ndarray  # (N, 2) float, grid xy position of each spot
    domain_id: np.ndarray  # (N,) int, true tissue domain
    infiltration_mask: np.ndarray  # (N,) bool
    patches: np.ndarray  # (N, C, H, W) float32, synthetic H&E patch proxy
    expression: np.ndarray  # (N, G) float32, non-negative "raw" expression
    marker_gene_idx: np.ndarray  # (n_marker_genes,) indices of domain-marker genes
    immune_gene_idx: np.ndarray  # (n_immune_genes,) indices of infiltration-marker genes
    oracle_boundaryness: np.ndarray  # (N,) float in [0, 1], ground-truth
    oracle_zone: np.ndarray  # (N,) str, one of ZONE_NAMES
    covariates: dict = field(default_factory=dict)  # platform / batch metadata stubs

    @property
    def n_spots(self) -> int:
        return self.coords.shape[0]

    @property
    def n_genes(self) -> int:
        return self.expression.shape[1]


def _make_domain_map(
    grid_size: int, n_domains: int, rng: np.random.Generator
) -> np.ndarray:
    """Voronoi-like partition of a grid into contiguous domains."""
    seeds = rng.integers(0, grid_size, size=(n_domains, 2))
    yy, xx = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    coords = np.stack([yy, xx], axis=-1).reshape(-1, 2)  # (H*W, 2)
    dists = np.linalg.norm(coords[:, None, :] - seeds[None, :, :], axis=-1)  # (H*W, n_domains)
    domain_map = np.argmin(dists, axis=1).reshape(grid_size, grid_size)
    return domain_map


def _boundaryness_from_domain_map(domain_map: np.ndarray, tau: float) -> np.ndarray:
    """exp(-distance_to_nearest_different_domain / tau), in [0, 1]."""
    n_domains = domain_map.max() + 1
    grid_size = domain_map.shape[0]
    dist_to_boundary = np.full(domain_map.shape, np.inf, dtype=np.float64)
    for d in range(n_domains):
        mask = domain_map == d
        if not mask.any():
            continue
        # distance (in pixels) from each pixel of domain d to the nearest
        # pixel *outside* domain d.
        dist = distance_transform_edt(mask)
        dist_to_boundary = np.where(mask, np.minimum(dist_to_boundary, dist), dist_to_boundary)
    dist_to_boundary = np.nan_to_num(dist_to_boundary, posinf=grid_size)
    return np.exp(-dist_to_boundary / tau)


def _make_infiltration_mask(
    grid_size: int, n_patches: int, patch_radius: float, rng: np.random.Generator
) -> np.ndarray:
    if n_patches <= 0:
        return np.zeros((grid_size, grid_size), dtype=bool)
    centers = rng.uniform(0, grid_size, size=(n_patches, 2))
    yy, xx = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    for cy, cx in centers:
        r = patch_radius * rng.uniform(0.6, 1.4)
        mask |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
    return mask


def _smooth_noise_field(grid_size: int, n_genes: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(size=(grid_size, grid_size, n_genes))
    smoothed = gaussian_filter(white, sigma=(sigma, sigma, 0))
    # normalize to unit std per gene so the noise amplitude is controllable
    # independent of sigma via `noise_amplitude` in the caller.
    std = smoothed.std(axis=(0, 1), keepdims=True) + 1e-8
    return smoothed / std


def _make_patches(
    domain_map: np.ndarray,
    infiltration_mask: np.ndarray,
    n_domains: int,
    patch_size: int,
    boundary_blend_width: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthetic per-spot H&E patch proxy: (N, 3, patch_size, patch_size).

    Each domain gets a distinct sinusoidal texture (frequency, orientation,
    color mean). Textures are blended across a `boundary_blend_width`-pixel
    band using a softened (blurred) one-hot domain map, so the *image* signal
    is a fuzzy hint of structure while the *expression* signal (generated
    separately) stays a sharp step -- see module docstring.
    """
    grid_size = domain_map.shape[0]
    n_pixels = patch_size
    freqs = rng.uniform(0.15, 0.6, size=n_domains)
    orientations = rng.uniform(0, np.pi, size=n_domains)
    color_means = rng.uniform(0.3, 0.8, size=(n_domains, 3))

    # Soft (blurred) per-domain occupancy maps for smooth image blending.
    one_hot = np.zeros((n_domains, grid_size, grid_size), dtype=np.float64)
    for d in range(n_domains):
        one_hot[d] = domain_map == d
    soft_occupancy = gaussian_filter(one_hot, sigma=(0, boundary_blend_width, boundary_blend_width))
    soft_occupancy /= soft_occupancy.sum(axis=0, keepdims=True) + 1e-8

    yy, xx = np.meshgrid(np.arange(n_pixels), np.arange(n_pixels), indexing="ij")

    patches = np.zeros((grid_size, grid_size, 3, n_pixels, n_pixels), dtype=np.float32)
    for gy in range(grid_size):
        for gx in range(grid_size):
            weights = soft_occupancy[:, gy, gx]  # (n_domains,)
            texture = np.zeros((n_pixels, n_pixels), dtype=np.float64)
            color = np.zeros(3, dtype=np.float64)
            for d in range(n_domains):
                if weights[d] < 1e-4:
                    continue
                phase = rng.uniform(0, 2 * np.pi)
                proj = yy * np.cos(orientations[d]) + xx * np.sin(orientations[d])
                texture += weights[d] * np.sin(2 * np.pi * freqs[d] * proj / n_pixels + phase)
                color += weights[d] * color_means[d]
            spot_patch = color[:, None, None] + 0.15 * texture[None, :, :]
            if infiltration_mask[gy, gx]:
                stipple = rng.normal(scale=0.35, size=(n_pixels, n_pixels))
                stipple = gaussian_filter(stipple, sigma=0.6)
                spot_patch = spot_patch + 0.25 * stipple[None, :, :]
            spot_patch += rng.normal(scale=0.03, size=spot_patch.shape)
            patches[gy, gx] = np.clip(spot_patch, 0.0, 1.0).astype(np.float32)
    return patches.reshape(grid_size * grid_size, 3, n_pixels, n_pixels)


def generate_synthetic_slide(
    slide_id: str,
    grid_size: int = 32,
    n_domains: int = 5,
    n_genes: int = 60,
    n_marker_genes: int = 20,
    n_immune_genes: int = 8,
    patch_size: int = 32,
    boundary_tau: float = 1.5,
    boundary_blend_width: float = 1.8,
    n_infiltration_patches: int = 3,
    infiltration_patch_radius: float = 2.5,
    smooth_noise_sigma: float = 1.2,
    smooth_noise_amplitude: float = 0.15,
    measurement_noise_scale: float = 0.08,
    domain_fold_change: float = 2.2,
    immune_fold_change: float = 2.5,
    zone_quantiles: tuple[float, float] = (0.5, 0.85),
    seed: int = 0,
    covariates: Optional[dict] = None,
) -> SyntheticSlide:
    rng = np.random.default_rng(seed)

    domain_map = _make_domain_map(grid_size, n_domains, rng)
    infiltration_mask_2d = _make_infiltration_mask(
        grid_size, n_infiltration_patches, infiltration_patch_radius, rng
    )
    oracle_boundaryness_2d = _boundaryness_from_domain_map(domain_map, tau=boundary_tau)

    # --- expression: sharp domain step + smooth noise + measurement noise ---
    n_marker_genes = min(n_marker_genes, n_genes)
    base_level = rng.gamma(shape=4.0, scale=1.0, size=n_genes)
    domain_profiles = np.tile(base_level, (n_domains, 1))
    marker_idx = rng.choice(n_genes, size=n_marker_genes, replace=False)
    for d in range(n_domains):
        fold = rng.uniform(1.0 / domain_fold_change, domain_fold_change, size=n_marker_genes)
        domain_profiles[d, marker_idx] = base_level[marker_idx] * fold

    remaining = np.setdiff1d(np.arange(n_genes), marker_idx)
    immune_idx = rng.choice(remaining, size=min(n_immune_genes, len(remaining)), replace=False)
    immune_boost = np.ones(n_genes)
    immune_boost[immune_idx] = immune_fold_change

    domain_flat = domain_map.reshape(-1)
    infiltration_flat = infiltration_mask_2d.reshape(-1)
    mean_expr = domain_profiles[domain_flat]  # (N, G)
    mean_expr = np.where(infiltration_flat[:, None], mean_expr * immune_boost[None, :], mean_expr)

    smooth_noise = _smooth_noise_field(grid_size, n_genes, smooth_noise_sigma, rng).reshape(-1, n_genes)
    measurement_noise = rng.normal(scale=measurement_noise_scale, size=(grid_size * grid_size, n_genes))

    log_expr = np.log1p(mean_expr) + smooth_noise_amplitude * smooth_noise + measurement_noise
    expression = np.clip(np.expm1(log_expr), a_min=0.0, a_max=None).astype(np.float32)

    patches = _make_patches(
        domain_map, infiltration_mask_2d, n_domains, patch_size, boundary_blend_width, rng
    )

    oracle_boundaryness_flat = oracle_boundaryness_2d.reshape(-1)
    tau_low, tau_high = np.quantile(oracle_boundaryness_flat, zone_quantiles)
    oracle_zone = np.where(
        infiltration_flat,
        ZONE_INFILTRATION,
        np.where(
            oracle_boundaryness_flat >= tau_high,
            ZONE_BOUNDARY,
            np.where(oracle_boundaryness_flat >= tau_low, ZONE_TRANSITION, ZONE_HOMOGENEOUS),
        ),
    )

    yy, xx = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    coords = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=-1).astype(np.float32)

    return SyntheticSlide(
        slide_id=slide_id,
        coords=coords,
        domain_id=domain_flat.astype(np.int64),
        infiltration_mask=infiltration_flat,
        patches=patches,
        expression=expression,
        marker_gene_idx=marker_idx,
        immune_gene_idx=immune_idx,
        oracle_boundaryness=oracle_boundaryness_flat.astype(np.float32),
        oracle_zone=oracle_zone,
        covariates=covariates or {},
    )


def generate_synthetic_cohort(
    n_slides: int = 12,
    seed: int = 0,
    **slide_kwargs,
) -> List[SyntheticSlide]:
    """Generate a cohort of independent synthetic slides (one per 'patient').

    Each slide gets its own RNG seed derived from `seed`, and a synthetic
    platform/batch covariate so slide-level split and (in future batch-audit
    extensions) confound checks have something non-trivial to bite on.
    """
    platforms = ["platform_A", "platform_B"]
    slides = []
    for i in range(n_slides):
        cov = {
            "platform": platforms[i % len(platforms)],
            "donor": f"donor_{i:03d}",
        }
        slide = generate_synthetic_slide(
            slide_id=f"synthetic_slide_{i:03d}",
            seed=seed * 10_000 + i,
            covariates=cov,
            **slide_kwargs,
        )
        slides.append(slide)
    return slides
