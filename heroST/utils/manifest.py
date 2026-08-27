"""Run manifest writer: config hash, seed, and environment metadata."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def build_manifest(
    *,
    config_path: str,
    config_hash: str,
    seed: int,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "config_hash": config_hash,
        "seed": seed,
        "git_commit": _git_commit(repo_root),
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            manifest["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        manifest["torch_version"] = None
        manifest["cuda_available"] = False
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path: str | Path, manifest: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
