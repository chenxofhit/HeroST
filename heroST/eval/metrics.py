"""Evaluation metrics for HERO-ST zone-stratified accuracy and over-smoothing.

Pure numpy/scipy (no torch) so metrics can be unit-tested independently of
training. Primary quantities: zone-stratified PCC, local variance / boundary
contrast, Moran's-I over-smoothing checks, and global compatibility metrics.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    r, _ = pearsonr(x, y)
    return float(r)


def per_gene_pcc(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pearson correlation per gene, across spots. pred/target: (N, G)."""
    n_genes = pred.shape[1]
    return np.array([_safe_pearson(pred[:, g], target[:, g]) for g in range(n_genes)])


def zone_stratified_pcc(
    pred: np.ndarray, target: np.ndarray, zone_labels: Sequence[str]
) -> Dict[str, float]:
    """Mean per-gene PCC computed separately within each operational zone."""
    zone_labels = np.asarray(zone_labels)
    results: Dict[str, float] = {}
    for zone in sorted(set(zone_labels.tolist())):
        mask = zone_labels == zone
        if mask.sum() < 2:
            results[zone] = float("nan")
            continue
        gene_pccs = per_gene_pcc(pred[mask], target[mask])
        results[zone] = float(np.nanmean(gene_pccs))
    return results


def local_variance_ratio(
    pred: np.ndarray, target: np.ndarray, edge_index: np.ndarray
) -> float:
    """Ratio of local (neighbor-pair) variance in predictions vs ground truth.

    Ratio << 1 is an over-smoothing signature: the model reproduces less
    local variation than actually exists.
    """
    src, dst = edge_index[0], edge_index[1]
    pred_local_var = np.mean(np.sum((pred[src] - pred[dst]) ** 2, axis=-1))
    true_local_var = np.mean(np.sum((target[src] - target[dst]) ** 2, axis=-1))
    if true_local_var < 1e-10:
        return float("nan")
    return float(pred_local_var / true_local_var)


def boundary_contrast_preservation(
    pred: np.ndarray, target: np.ndarray, edge_index: np.ndarray, edge_zone_mask: np.ndarray
) -> float:
    """Among edges flagged as boundary edges (`edge_zone_mask`),
    correlation between true and predicted pairwise differences. High value
    = the model preserves *which* boundaries are sharp and by how much, not
    just their existence."""
    src, dst = edge_index[0], edge_index[1]
    pred_diff = np.linalg.norm(pred[src] - pred[dst], axis=-1)
    true_diff = np.linalg.norm(target[src] - target[dst], axis=-1)
    mask = np.asarray(edge_zone_mask, dtype=bool)
    if mask.sum() < 2:
        return float("nan")
    return _safe_pearson(pred_diff[mask], true_diff[mask])


def morans_i(values: np.ndarray, edge_index: np.ndarray) -> float:
    """Global Moran's I spatial autocorrelation, using the (possibly
    asymmetric k-NN) spatial graph as an unweighted adjacency. `values` is a
    single spatial signal (N,); for multi-gene arrays, average across genes
    (see `morans_i_multigene`)."""
    src, dst = edge_index[0], edge_index[1]
    n = values.shape[0]
    x = values - values.mean()
    s0 = len(src)
    if s0 == 0 or np.sum(x**2) < 1e-12:
        return float("nan")
    numerator = n * np.sum(x[src] * x[dst])
    denominator = s0 * np.sum(x**2)
    return float(numerator / denominator)


def morans_i_multigene(values: np.ndarray, edge_index: np.ndarray) -> float:
    """Mean Moran's I across genes. values: (N, G)."""
    n_genes = values.shape[1]
    scores = [morans_i(values[:, g], edge_index) for g in range(n_genes)]
    if all(np.isnan(s) for s in scores):
        return float("nan")
    return float(np.nanmean(scores))


def oversmoothing_check(
    pred: np.ndarray,
    target: np.ndarray,
    edge_index: np.ndarray,
    inflation_threshold: float = 0.08,
    variance_collapse_ratio_threshold: float = 0.05,
) -> Dict[str, float | bool]:
    """Flag over-smoothing via Moran's I inflation or variance collapse.

    Predicted Moran's I must not exceed ground truth by more than
    `inflation_threshold`. Moran's I is undefined for a perfectly constant
    prediction, so `variance_collapse_ratio_threshold` additionally flags
    cases where predicted spatial variance is a tiny fraction of truth.
    """
    i_true = morans_i_multigene(target, edge_index)
    i_pred = morans_i_multigene(pred, edge_index)
    delta = i_pred - i_true

    # Per-gene spatial (across-spot) variance, averaged over genes. This is
    # deliberately NOT np.var(pred) on the flattened array: flattening would
    # mix across-gene differences (which are large and irrelevant here) with
    # across-spot differences (the only thing "spatially collapsed" means).
    pred_var = float(np.mean(np.var(pred, axis=0)))
    true_var = float(np.mean(np.var(target, axis=0)))
    variance_ratio = pred_var / true_var if true_var > 1e-12 else float("nan")
    variance_collapsed = bool(
        true_var > 1e-12 and variance_ratio < variance_collapse_ratio_threshold
    )
    moran_inflated = bool(np.isfinite(delta) and delta > inflation_threshold)

    return {
        "moran_i_true": i_true,
        "moran_i_pred": i_pred,
        "delta": delta,
        "variance_ratio": variance_ratio,
        "variance_collapsed": variance_collapsed,
        "oversmoothing_flagged": bool(moran_inflated or variance_collapsed),
    }



def slide_shuffle_seed(slide_id: str) -> int:
    """Deterministic RNG seed for shuffle-null ECE (hash(slide_id)+42)."""
    import hashlib

    digest = hashlib.md5(str(slide_id).encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + 42) % (2**31)




def global_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Global compatibility metrics (mean PCC/Spearman, MAE, MSE).

    Reported for comparability with prior work; zone-stratified and
    heterogeneity metrics remain the primary spatial checks.
    """
    gene_pccs = per_gene_pcc(pred, target)
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    gene_spearman = []
    for gene in range(pred.shape[1]):
        if np.std(pred[:, gene]) < 1e-10 or np.std(target[:, gene]) < 1e-10:
            gene_spearman.append(float("nan"))
        else:
            value, _ = spearmanr(pred[:, gene], target[:, gene])
            gene_spearman.append(float(value))
    return {
        "mean_pcc": float(np.nanmean(gene_pccs)),
        "median_pcc": float(np.nanmedian(gene_pccs)),
        "mean_spearman": float(np.nanmean(gene_spearman)),
        "mae": mae,
        "mse": mse,
    }


def batch_confound_probe_auc_proxy(
    grouped_values: Dict[str, np.ndarray],
) -> float:
    """Cheap between-group separability proxy (not a full classifier).

    Normalized between-group variance over total variance (ANOVA-like effect
    size) for a negative-control check of whether features separate batch or
    platform labels. High value => features
    strongly separate by group => batch/platform signal is leaking through,
    a flag to run the full negative control before trusting downstream
    biological claims.
    """
    all_values = np.concatenate(list(grouped_values.values()))
    grand_mean = all_values.mean()
    between = sum(len(v) * (v.mean() - grand_mean) ** 2 for v in grouped_values.values())
    total = np.sum((all_values - grand_mean) ** 2)
    if total < 1e-12:
        return float("nan")
    return float(between / total)
