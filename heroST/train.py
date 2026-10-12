"""Training entry point: cohort (synthetic or real) -> slide/patient-level
split -> HeroSTModel training -> frozen zone thresholds -> full
zone-stratified evaluation -> manifest + report written to the output
directory.

Usage:
    python -m heroST.train --config configs/smoke_test.yaml --output outputs/smoke_test
    python -m heroST.train --config configs/her2_breast_real_lite5_v1.yaml --output outputs/her2_real

`data.source` in the config selects the cohort:
- "synthetic" (default): heroST.data.synthetic's generator.
- "hest": a real HEST-1k cohort via heroST.data.hest (see that module and
  configs/her2_breast_real_lite5_v1.yaml). Both paths produce `SyntheticSlide`-shaped
  objects, so nothing else in this file branches on the source.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from heroST.data.dataset import ExpressionNormalizer, SlideGraphDataset, collate_single_slide
from heroST.data.splits import slide_level_split, split_report
from heroST.data.synthetic import SyntheticSlide, generate_synthetic_cohort
from heroST.eval.runner import compute_frozen_zone_thresholds, evaluate_dataset
from heroST.models.hero_st import HeroSTConfig, HeroSTModel
from heroST.models.m3_uncertainty_alignment import m3_losses
from heroST.models.m4_heterogeneity_objective import m4_core_losses
from heroST.utils.config import config_hash, load_config
from heroST.utils.manifest import build_manifest, write_manifest
from heroST.utils.seed import set_global_seed


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_synthetic_splits(
    data_cfg: Dict, seed: int
) -> Tuple[List[SyntheticSlide], List[SyntheticSlide], List[SyntheticSlide], Dict[str, object]]:
    slides: List[SyntheticSlide] = generate_synthetic_cohort(
        n_slides=data_cfg["n_slides"],
        grid_size=data_cfg["grid_size"],
        n_domains=data_cfg["n_domains"],
        n_genes=data_cfg["n_genes"],
        n_marker_genes=data_cfg["n_marker_genes"],
        n_immune_genes=data_cfg["n_immune_genes"],
        patch_size=data_cfg["patch_size"],
        seed=seed,
    )
    ids = [s.slide_id for s in slides]
    train_ids, val_ids, test_ids = slide_level_split(
        ids, val_frac=data_cfg["val_frac"], test_frac=data_cfg["test_frac"], seed=seed
    )
    report = split_report(train_ids, val_ids, test_ids)
    by_id = {s.slide_id: s for s in slides}
    return [by_id[i] for i in train_ids], [by_id[i] for i in val_ids], [by_id[i] for i in test_ids], report


def _build_hest_splits(
    data_cfg: Dict, seed: int
) -> Tuple[List[SyntheticSlide], List[SyntheticSlide], List[SyntheticSlide], Dict[str, object]]:
    from heroST.data.hest import HER2_BREAST_SAMPLE_IDS, apply_gene_subset, load_hest_cohort, select_hvg_indices
    from heroST.data.splits import patient_level_split

    sample_ids = data_cfg.get("sample_ids") or HER2_BREAST_SAMPLE_IDS
    slides, gene_names = load_hest_cohort(sample_ids, patch_resize=data_cfg.get("patch_resize", 64))

    patient_of = {s.slide_id: str(s.covariates.get("patient", s.slide_id)) for s in slides}
    ids = [s.slide_id for s in slides]
    train_ids, val_ids, test_ids = patient_level_split(
        ids, patient_of, val_frac=data_cfg["val_frac"], test_frac=data_cfg["test_frac"], seed=seed
    )
    report = split_report(train_ids, val_ids, test_ids)
    report["patient_of"] = patient_of

    by_id = {s.slide_id: s for s in slides}
    train_slides = [by_id[i] for i in train_ids]
    val_slides = [by_id[i] for i in val_ids]
    test_slides = [by_id[i] for i in test_ids]

    # HVG selection fit on TRAIN slides only, then applied identically to
    # every fold.
    hvg_idx = select_hvg_indices(train_slides, data_cfg["n_top_genes"])
    n_marker_genes = data_cfg.get("n_marker_genes")
    train_slides, hvg_names = apply_gene_subset(train_slides, hvg_idx, gene_names, n_marker_genes)
    val_slides, _ = apply_gene_subset(val_slides, hvg_idx, gene_names, n_marker_genes)
    test_slides, _ = apply_gene_subset(test_slides, hvg_idx, gene_names, n_marker_genes)
    report["hvg_gene_names"] = list(hvg_names)
    return train_slides, val_slides, test_slides, report


def build_dataset_splits(
    cfg: Dict, seed: int
) -> Tuple[SlideGraphDataset, SlideGraphDataset, SlideGraphDataset, Dict[str, object]]:
    data_cfg = cfg["data"]
    source = data_cfg.get("source", "synthetic")
    if source == "synthetic":
        train_slides, val_slides, test_slides, report = _build_synthetic_splits(data_cfg, seed)
    elif source == "hest":
        train_slides, val_slides, test_slides, report = _build_hest_splits(data_cfg, seed)
    else:
        raise ValueError(f"Unknown data.source: {source!r} (expected 'synthetic' or 'hest')")

    normalizer = ExpressionNormalizer().fit(train_slides)  # train-only fit
    knn_k = data_cfg["knn_k"]
    train_ds = SlideGraphDataset(train_slides, normalizer, knn_k=knn_k)
    val_ds = SlideGraphDataset(val_slides, normalizer, knn_k=knn_k)
    test_ds = SlideGraphDataset(test_slides, normalizer, knn_k=knn_k)
    return train_ds, val_ds, test_ds, report


def build_model(cfg: Dict, n_genes: int) -> HeroSTModel:
    m = cfg["model"]
    model_cfg = HeroSTConfig(
        n_genes=n_genes,
        embed_dim=m["embed_dim"],
        backbone_name=m.get("backbone_name", "tiny_cnn"),
        n_message_passing_layers=m["n_message_passing_layers"],
        spatial_scale=m["spatial_scale"],
        graph_variant=m["graph_variant"],
        n_cross_attn_heads=m["n_cross_attn_heads"],
        m2_nonneg_weights=m.get("m2_nonneg_weights", True),
        m2_monotone_gate=m.get("m2_monotone_gate", True),
        m2_boundary_penalty_mode=m.get("m2_boundary_penalty_mode", "dst"),
        m2_use_edge_weighter=m.get("m2_use_edge_weighter", True),
        m2_use_node_gate=m.get("m2_use_node_gate", True),
        m2_gate_side=m.get("m2_gate_side", "dst"),
    )
    model = HeroSTModel(model_cfg)
    cache_dir = os.environ.get("HEROST_FEATURE_CACHE")
    if cache_dir:
        model.set_feature_cache(cache_dir)
    return model


def train_one_epoch(
    model: HeroSTModel, dataset: SlideGraphDataset, optimizer: torch.optim.Optimizer, cfg: Dict, device: torch.device
) -> Dict[str, float]:
    model.train()
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single_slide)
    loss_cfg = cfg["loss"]
    totals: Dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        # Large Visium/Xenium slides: keep patches on CPU; frozen backbone streams chunks.
        keep_cpu = int(batch.patches.shape[0]) >= 1024
        batch = batch.to(device, keep_patches_on_cpu=keep_cpu)
        optimizer.zero_grad()

        out = model(
            batch.patches, batch.coords, batch.spatial_edge_index, batch.spatial_edge_dist,
            expression_true=batch.expression,
            cache_key=batch.slide_id,
        )
        # Five manuscript losses: heteroscedastic + consistency + InfoNCE +
        # hard-negative contrast + local-difference preservation.
        m3_out = m3_losses(
            out.m3,
            batch.expression,
            w_consistency=loss_cfg["w_m3_consistency"],
            w_contrastive=loss_cfg["w_m3_contrastive"],
            w_teacher_supervision=loss_cfg.get("w_m3_teacher_supervision", 0.0),
        )
        m4_out = m4_core_losses(
            out.graph_embedding,
            out.m3.mu,
            batch.expression,
            batch.spatial_edge_index,
            out.m1.boundaryness,
            margin=loss_cfg["contrastive_margin"],
            boundary_upweight=loss_cfg["boundary_upweight"],
            w_contrastive=loss_cfg["w_m4_contrastive"],
            w_local_diff=loss_cfg["w_m4_local_diff"],
        )
        total_loss = m3_out["total"] + m4_out["total"]
        total_loss.backward()
        optimizer.step()

        logged = {
            "heteroscedastic": m3_out["heteroscedastic"].item(),
            "consistency": m3_out["consistency"].item(),
            "infonce": m3_out["contrastive"].item(),
            "hard_negative": m4_out["boundary_contrastive"].item(),
            "local_diff": m4_out["local_diff_preservation"].item(),
            "total": total_loss.item(),
        }
        for name, value in logged.items():
            totals[name] = totals.get(name, 0.0) + value
        n_batches += 1
        if device.type == "cuda" and keep_cpu:
            torch.cuda.empty_cache()

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def run(config_path: str, output_dir: str) -> Dict[str, object]:
    cfg = load_config(config_path)
    cfg_hash = config_hash(cfg)
    seed = cfg["seed"]
    set_global_seed(seed)
    device = resolve_device(cfg["train"]["device"])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, split_rep = build_dataset_splits(cfg, seed)
    n_genes = train_ds.slides[0].n_genes  # derived from the built split, not cfg (real cohorts subset genes via HVG selection)
    model = build_model(cfg, n_genes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"]
    )

    verbose = os.environ.get("HEROST_VERBOSE_TRAIN") == "1"
    epoch_losses = []
    for epoch in range(cfg["train"]["n_epochs"]):
        losses = train_one_epoch(model, train_ds, optimizer, cfg, device)
        epoch_losses.append(losses)
        if verbose:
            print(f"[epoch {epoch + 1}/{cfg['train']['n_epochs']}] " + " ".join(f"{k}={v:.4f}" for k, v in losses.items()), flush=True)

    thresholds = compute_frozen_zone_thresholds(
        model, train_ds, quantiles=tuple(cfg["eval"]["zone_quantiles"]), device=device
    )

    val_eval = evaluate_dataset(model, val_ds, thresholds, device)
    test_eval = evaluate_dataset(model, test_ds, thresholds, device)

    manifest = build_manifest(
        config_path=config_path,
        config_hash=cfg_hash,
        seed=seed,
        extra={"device": str(device), "split_report": split_rep, "n_epochs_run": len(epoch_losses)},
    )
    write_manifest(out_dir / "manifest.json", manifest)

    report = {
        "config_hash": cfg_hash,
        "zone_thresholds": {"tau_low": thresholds.tau_low, "tau_high": thresholds.tau_high},
        "epoch_losses": epoch_losses,
        "val": val_eval["aggregate"],
        "test": test_eval["aggregate"],
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    with (out_dir / "test_per_slide.json").open("w", encoding="utf-8") as f:
        json.dump(test_eval["per_slide"], f, indent=2, default=float)

    torch.save(model.state_dict(), out_dir / "model.pt")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HeroSTModel on a synthetic cohort (smoke-test regime).")
    parser.add_argument("--config", required=True, help="Path to a YAML config (see configs/smoke_test.yaml).")
    parser.add_argument("--output", required=True, help="Output directory for manifest/report/checkpoint.")
    args = parser.parse_args()
    report = run(args.config, args.output)
    print(json.dumps(report["test"], indent=2, default=float))


if __name__ == "__main__":
    main()
