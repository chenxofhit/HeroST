"""Slide- or patient-level train/validation/test splits.

Spots from the same slide are never split across folds. When patient IDs are
available, all slides from one patient stay in one fold.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def slide_level_split(
    slide_ids: Sequence[str],
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[List[str], List[str], List[str]]:
    """Shuffle-split at slide granularity. No spot ever crosses folds because
    the unit being split is the whole slide, not individual spots."""
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")
    rng = np.random.default_rng(seed)
    ids = list(slide_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * test_frac))) if n >= 3 else 0
    n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]
    return train_ids, val_ids, test_ids


def patient_level_split(
    slide_ids: Sequence[str],
    patient_of: Dict[str, str],
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
) -> Tuple[List[str], List[str], List[str]]:
    """Shuffle-split at PATIENT granularity, then expand back to slide ids.

    Required whenever a cohort has multiple slides per patient/donor (e.g.
    serial sections). Splitting slide ids directly (as `slide_level_split`
    does) would let two sections from the same patient land in different
    folds and leak information across folds.
    """
    patients = sorted({patient_of[sid] for sid in slide_ids})
    train_p, val_p, test_p = slide_level_split(patients, val_frac=val_frac, test_frac=test_frac, seed=seed)
    train_p_set, val_p_set, test_p_set = set(train_p), set(val_p), set(test_p)

    train_ids = [sid for sid in slide_ids if patient_of[sid] in train_p_set]
    val_ids = [sid for sid in slide_ids if patient_of[sid] in val_p_set]
    test_ids = [sid for sid in slide_ids if patient_of[sid] in test_p_set]
    return train_ids, val_ids, test_ids


def assert_no_leakage(
    train_ids: Sequence[str], val_ids: Sequence[str], test_ids: Sequence[str]
) -> None:
    """Raises AssertionError if any slide ID appears in more than one split.

    Call this in every training entry point immediately after computing
    splits, before any model sees data.
    """
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    overlaps = {
        "train_val": train_set & val_set,
        "train_test": train_set & test_set,
        "val_test": val_set & test_set,
    }
    leaking = {k: v for k, v in overlaps.items() if v}
    if leaking:
        raise AssertionError(f"Slide-level split leakage detected: {leaking}")


def split_report(
    train_ids: Sequence[str], val_ids: Sequence[str], test_ids: Sequence[str]
) -> Dict[str, object]:
    """Human-readable summary to log into a run manifest / gate report."""
    assert_no_leakage(train_ids, val_ids, test_ids)
    return {
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "test_ids": list(test_ids),
        "leakage_check": "PASS",
    }
