"""HEST-1k loader for real histology-spatial transcriptomics cohorts.

Builds slide objects with H&E patches, coordinates, and expression matrices
compatible with the training pipeline. Gene selection and normalization are
fit on training slides only.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from heroST.data.synthetic import SyntheticSlide

HEST_REPO_ID = "MahmoodLab/hest"
ZONE_UNKNOWN = "unknown"

# Andersson et al. 2021 HER2-positive breast cancer cohort (HEST-1k ids).
# Eight patients (A-H), 3-6 serial sections each.
HER2_BREAST_SAMPLE_IDS: List[str] = [f"SPA{n}" for n in range(119, 155)]

# Canonical pan-immune / lineage marker genes, used only to build a real
# (non-oracle) infiltration proxy for zone auditing -- never fed to the model.
IMMUNE_MARKER_GENES: List[str] = [
    "PTPRC", "CD3D", "CD3E", "CD3G", "CD2", "CD8A", "CD8B", "CD4",
    "CD19", "MS4A1", "CD79A", "CD79B", "CD68", "CD14", "ITGAM", "LYZ",
    "NKG7", "GNLY", "FCGR3A", "IL2RG",
]


def normalize_library_size(expression: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Per-spot total-count ("library size") normalization, applied to raw
    HEST-1k UMI counts BEFORE `heroST.data.dataset.ExpressionNormalizer`'s
    log1p+z-score.

    Real ST spots have widely varying sequencing depth for purely technical
    reasons (observed range on one HER2 section: per-spot total counts
    377-8962, a 24x spread, CV~0.45) that is orthogonal to the biological
    signal H&E morphology could ever predict. Without this step, that
    depth noise inflates neighbor-to-neighbor expression differences, which
    then reads as spurious local "heterogeneity" the model is faulted for
    failing to reproduce -- i.e. it artificially worsens exactly the
    over-smoothing check these checks target
    built around, specific to real (never synthetic) data.

    Uses only each spot's own total count and a fixed constant `target_sum`
    (not fit from data), so this is leakage-safe across any split -- unlike
    `ExpressionNormalizer`'s mean/std, there is no fold-dependent statistic
    here to accidentally fit on val/test.
    """
    totals = expression.sum(axis=1, keepdims=True)
    safe_totals = np.where(totals > 0, totals, 1.0)
    return (expression / safe_totals * target_sum).astype(expression.dtype)


def resolve_hf_token(token: Optional[str] = None) -> Optional[str]:
    if token:
        return token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    default_path = Path.home() / ".cache" / "huggingface" / "token"
    if default_path.exists():
        return default_path.read_text().strip()
    return None


def ensure_hest_files(sample_id: str, token: Optional[str] = None, max_retries: int = 3) -> Dict[str, Path]:
    """Download (or reuse the local Hugging Face cache for) one HEST-1k
    sample's metadata / pre-extracted spot patches / expression AnnData.

    Requires network access and a token that has accepted MahmoodLab/hest's
    gated-access terms on huggingface.co. Raises the underlying
    huggingface_hub exception after `max_retries` attempts.

    Optional offline mirror: if ``HEROST_HEST_LOCAL_ROOT`` (or repo
    ``data/hest_local``) contains ``metadata/{id}.json``, ``patches/{id}.h5``,
    and ``st/{id}.h5ad``, those paths are returned without hitting the Hub.
    """
    rels = (
        ("metadata", "metadata/{id}.json"),
        ("patches", "patches/{id}.h5"),
        ("st", "st/{id}.h5ad"),
    )
    local_root = os.environ.get("HEROST_HEST_LOCAL_ROOT")
    candidates = []
    if local_root:
        candidates.append(Path(local_root))
    candidates.append(Path(__file__).resolve().parents[2] / "data" / "hest_local")
    for root in candidates:
        local_paths = {key: root / tmpl.format(id=sample_id) for key, tmpl in rels}
        if all(p.exists() and p.stat().st_size > 0 for p in local_paths.values()):
            return local_paths

    from huggingface_hub import hf_hub_download

    tok = resolve_hf_token(token)
    paths: Dict[str, Path] = {}
    for key, rel_tmpl in rels:
        rel = rel_tmpl.format(id=sample_id)
        last_err: Optional[Exception] = None
        for _ in range(max_retries):
            try:
                paths[key] = Path(hf_hub_download(HEST_REPO_ID, rel, repo_type="dataset", token=tok))
                last_err = None
                break
            except Exception as e:  # noqa: BLE001 - surface after retries exhausted
                last_err = e
        if last_err is not None:
            raise last_err
    return paths


def _extract_spot_coords(adata) -> np.ndarray:
    """Return (N, 2) float32 spot coordinates for kNN graph construction.

    Prefer Visium-style ``array_row``/``array_col`` when present; fall back to
    pixel columns, then ``obsm['spatial']`` (needed for some Xenium HEST
    entries such as TENX97/TENX98 that omit array indices).
    """
    obs = adata.obs
    if {"array_row", "array_col"}.issubset(obs.columns):
        return obs[["array_row", "array_col"]].to_numpy().astype(np.float32)
    if {"pxl_row_in_fullres", "pxl_col_in_fullres"}.issubset(obs.columns):
        return obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy().astype(np.float32)
    if hasattr(adata, "obsm") and "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError(f"obsm['spatial'] has unexpected shape {xy.shape}")
        return xy[:, :2].copy()
    raise KeyError(
        "No usable spot coordinates: need array_row/array_col, "
        "pxl_row_in_fullres/pxl_col_in_fullres, or obsm['spatial']"
    )


def _resize_patches(img_uint8: np.ndarray, size: int) -> np.ndarray:
    """(N, H, W, 3) uint8 in [0, 255] -> (N, 3, size, size) float32 in [0, 1].

    Uses area-interpolation resize (torch is already a core dependency here,
    so this avoids adding PIL/opencv just for a downsample)."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(img_uint8).float().permute(0, 3, 1, 2) / 255.0  # (N, 3, H, W)
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode="area")
    return t.numpy().astype(np.float32)


def _infiltration_proxy(expression: np.ndarray, gene_names: Sequence[str], quantile: float = 0.85) -> np.ndarray:
    """Weak, real (non-oracle) infiltration proxy: composite z-scored
    log-expression of canonical immune marker genes, thresholded at the
    `quantile`-th percentile within the slide. NOT a model input -- used
    only downstream for zone-enrichment auditing,
    exactly like `SyntheticSlide.infiltration_mask` is for synthetic data."""
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    idx = [name_to_idx[g] for g in IMMUNE_MARKER_GENES if g in name_to_idx]
    n = expression.shape[0]
    if not idx:
        return np.zeros(n, dtype=bool)
    log_expr = np.log1p(expression[:, idx])
    std = log_expr.std(axis=0) + 1e-6
    z = (log_expr - log_expr.mean(axis=0)) / std
    score = z.mean(axis=1)
    threshold = np.quantile(score, quantile)
    return score > threshold


def load_hest_slide(
    sample_id: str,
    token: Optional[str] = None,
    patch_resize: int = 64,
) -> Tuple[SyntheticSlide, List[str]]:
    """Load a single HEST-1k sample into a `SyntheticSlide`. Returns
    (slide, gene_names) since the gene panel size/order can differ slightly
    between samples (whole-transcriptome captures vary a bit in `nb_genes`);
    cohort-level alignment happens in `load_hest_cohort`."""
    import anndata as ad
    import h5py

    paths = ensure_hest_files(sample_id, token=token)
    meta = json.loads(paths["metadata"].read_text())
    adata = ad.read_h5ad(paths["st"])

    with h5py.File(paths["patches"], "r") as hf:
        raw_barcodes = hf["barcode"][:]
        patch_barcodes = [
            (b[0].decode() if isinstance(b[0], bytes) else str(b[0])) for b in raw_barcodes
        ]
        img = hf["img"][:]  # (N, H, W, 3) uint8

    barcode_to_patch_idx = {b: i for i, b in enumerate(patch_barcodes)}
    obs_names = list(adata.obs_names)
    keep = [i for i, b in enumerate(obs_names) if b in barcode_to_patch_idx]
    if len(keep) < len(obs_names):
        adata = adata[keep].copy()
        obs_names = list(adata.obs_names)
    if len(obs_names) == 0:
        raise ValueError(f"{sample_id}: no spots with both expression and an extracted patch.")

    patch_idx = np.array([barcode_to_patch_idx[b] for b in obs_names])
    patches = _resize_patches(img[patch_idx], patch_resize)

    X = adata.X
    expression = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
    expression = normalize_library_size(expression)
    gene_names = [str(g) for g in adata.var_names]

    coords = _extract_spot_coords(adata)

    n = expression.shape[0]
    infiltration_mask = _infiltration_proxy(expression, gene_names)
    immune_idx = np.array(
        [gene_names.index(g) for g in IMMUNE_MARKER_GENES if g in gene_names], dtype=np.int64
    )

    covariates = {
        "platform": meta.get("st_technology", "unknown"),
        "patient": meta.get("patient", sample_id),
        "organ": meta.get("organ", "unknown"),
        "disease_state": meta.get("disease_state", "unknown"),
        "preservation_method": meta.get("preservation_method", "unknown"),
        "hest_id": sample_id,
    }

    slide = SyntheticSlide(
        slide_id=sample_id,
        coords=coords,
        domain_id=np.zeros(n, dtype=np.int64),  # no ground-truth domain map for real tissue
        infiltration_mask=infiltration_mask,
        patches=patches,
        expression=expression,
        marker_gene_idx=np.array([], dtype=np.int64),  # set post-HVG-selection, see apply_gene_subset
        immune_gene_idx=immune_idx,
        oracle_boundaryness=np.zeros(n, dtype=np.float32),  # not available for real data
        oracle_zone=np.array([ZONE_UNKNOWN] * n),
        covariates=covariates,
    )
    return slide, gene_names


def load_hest_cohort(
    sample_ids: Sequence[str],
    token: Optional[str] = None,
    patch_resize: int = 64,
) -> Tuple[List[SyntheticSlide], List[str]]:
    """Load a cohort of HEST-1k samples, aligned to their common gene panel
    (sorted intersection of gene symbols across all requested samples)."""
    loaded = [load_hest_slide(sid, token=token, patch_resize=patch_resize) for sid in sample_ids]
    gene_sets = [set(genes) for _, genes in loaded]
    common_genes = sorted(set.intersection(*gene_sets))
    if not common_genes:
        raise ValueError("Requested HEST samples share no common genes; check sample_ids.")

    slides = []
    for slide, genes in loaded:
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        col_idx = np.array([gene_to_idx[g] for g in common_genes])
        immune_idx = np.array(
            [common_genes.index(g) for g in IMMUNE_MARKER_GENES if g in common_genes], dtype=np.int64
        )
        slides.append(
            dataclasses.replace(slide, expression=slide.expression[:, col_idx], immune_gene_idx=immune_idx)
        )
    return slides, common_genes


def select_hvg_indices(train_slides: Sequence[SyntheticSlide], n_top_genes: int) -> np.ndarray:
    """Highly-variable-gene selection fit on train slides only.

    Ranks genes by log1p-expression variance pooled across all train spots
    and returns the top-`n_top_genes` column indices (into the cohort's
    common gene list), sorted ascending so downstream column order stays
    deterministic.
    """
    n_top_genes = min(n_top_genes, train_slides[0].n_genes)
    pooled_log = np.concatenate([np.log1p(s.expression) for s in train_slides], axis=0)
    variance = pooled_log.var(axis=0)
    top_idx = np.argsort(-variance)[:n_top_genes]
    return np.sort(top_idx)


def apply_gene_subset(
    slides: Sequence[SyntheticSlide],
    gene_idx: np.ndarray,
    gene_names: Sequence[str],
    n_marker_genes: Optional[int] = None,
) -> Tuple[List[SyntheticSlide], List[str]]:
    """Subset every slide's expression matrix to the SAME `gene_idx` columns
    (fit once on train via `select_hvg_indices`, applied identically to
    train/val/test -- no fold sees a gene selection derived from itself).

    `marker_gene_idx` here is a coarse "top-variance-within-the-subset"
    proxy (there is no oracle domain-marker labeling for real tissue); it is
    informational only -- nothing in heroST.train/heroST.eval reads it."""
    gene_idx = np.asarray(gene_idx)
    subset_gene_names = [gene_names[i] for i in gene_idx]
    name_to_idx = {g: i for i, g in enumerate(subset_gene_names)}

    out = []
    for s in slides:
        new_expr = s.expression[:, gene_idx]
        immune_idx = np.array(
            [name_to_idx[g] for g in IMMUNE_MARKER_GENES if g in name_to_idx], dtype=np.int64
        )
        if n_marker_genes:
            k = min(n_marker_genes, new_expr.shape[1])
            marker_idx = np.sort(np.argsort(-np.log1p(new_expr).var(axis=0))[:k])
        else:
            marker_idx = np.array([], dtype=np.int64)
        out.append(
            dataclasses.replace(
                s, expression=new_expr, immune_gene_idx=immune_idx, marker_gene_idx=marker_idx
            )
        )
    return out, subset_gene_names
