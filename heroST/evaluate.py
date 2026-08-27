"""Standalone evaluation entry point.

Load a checkpoint from heroST.train and re-run zone-stratified evaluation,
optionally on a later held-out cohort without retraining.

Usage:
    python -m heroST.evaluate --config configs/smoke_test.yaml \\
        --checkpoint outputs/smoke_test/model.pt --output outputs/smoke_test_reeval

Zone thresholds are recomputed on the train split of the cohort defined by
the config (same convention as train.py). A new cohort therefore gets its
own reference threshold pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from heroST.train import build_dataset_splits, build_model, resolve_device
from heroST.eval.runner import compute_frozen_zone_thresholds, evaluate_dataset
from heroST.utils.config import config_hash, load_config
from heroST.utils.manifest import build_manifest, write_manifest


def run(config_path: str, checkpoint_path: str, output_dir: str) -> dict:
    cfg = load_config(config_path)
    cfg_hash = config_hash(cfg)
    seed = cfg["seed"]
    device = resolve_device(cfg["train"]["device"])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, split_rep = build_dataset_splits(cfg, seed)
    n_genes = train_ds.slides[0].n_genes  # derived from the built split, not cfg (real cohorts subset genes via HVG selection)
    model = build_model(cfg, n_genes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    thresholds = compute_frozen_zone_thresholds(
        model, train_ds, quantiles=tuple(cfg["eval"]["zone_quantiles"]), device=device
    )
    test_eval = evaluate_dataset(model, test_ds, thresholds, device)

    manifest = build_manifest(
        config_path=config_path,
        config_hash=cfg_hash,
        seed=seed,
        extra={"device": str(device), "checkpoint": str(checkpoint_path), "split_report": split_rep},
    )
    write_manifest(out_dir / "eval_manifest.json", manifest)

    report = {
        "config_hash": cfg_hash,
        "checkpoint": str(checkpoint_path),
        "zone_thresholds": {"tau_low": thresholds.tau_low, "tau_high": thresholds.tau_high},
        "test": test_eval["aggregate"],
    }
    with (out_dir / "eval_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved HeroSTModel checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.config, args.checkpoint, args.output)
    print(json.dumps(report["test"], indent=2, default=float))


if __name__ == "__main__":
    main()
