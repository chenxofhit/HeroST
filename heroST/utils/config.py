"""YAML config loading and stable hashing for experiment manifests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    """Stable sha256 hash of a config dict, independent of key order."""
    canonical = json.dumps(_sort_recursive(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sort_recursive(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_recursive(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_sort_recursive(v) for v in obj]
    return obj


def file_hash(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a flat dict of dotted-key overrides (e.g. {'model.w_unc': 0.5}).

    Any override applied to a run should be recorded in the run manifest
    (see heroST.utils.manifest).
    """
    cfg = copy.deepcopy(cfg)
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return cfg
