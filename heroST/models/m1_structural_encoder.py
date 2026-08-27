"""H&E structural prior encoder.

Frozen pathology foundation features (or a tiny CNN in smoke tests) are
projected to a morphology embedding, with lightweight heads for local
boundaryness and structural complexity used to regulate graph propagation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class M1Output:
    morphology_embedding: torch.Tensor  # (N, D)
    boundaryness: torch.Tensor  # (N,) in (0, 1)
    complexity: torch.Tensor  # (N,) in (0, 1)


class _MultiScaleConvBackbone(nn.Module):
    """Tiny multi-scale CNN standing in for a frozen pathology foundation
    model ()."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        self.scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, 16, kernel_size=k, padding=k // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                for k in (3, 5, 9)
            ]
        )
        self.proj = nn.Linear(32 * len(self.scale_convs), embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        feats = [conv(patches).flatten(1) for conv in self.scale_convs]
        feats = torch.cat(feats, dim=1)
        return self.proj(feats)


class _FrozenFoundationModelBackbone(nn.Module):
    """Wraps a frozen HF `transformers` pathology encoder (default:
    owkin/phikon-v2, a DINOv2 ViT-L/16) behind the same `(N, 3, H, W) float
    in [0, 1] -> (N, embed_dim)` contract as `_MultiScaleConvBackbone`, so
    M1's heads (and everything downstream in M2-M4) never know which
    backbone is in use.

    The foundation model itself is frozen and run under `torch.no_grad()`;
    only the projection head from its native hidden size down to `embed_dim`
    is trainable. Input is resized to the model's expected resolution
    (bicubic) and ImageNet-normalized, so upstream data loading can keep
    feeding raw [0, 1] patches at whatever resolution they were extracted at
    (HEST-1k patches are natively 224x224, matching Phikon-v2).
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hf_model_name: str = "owkin/phikon-v2",
        image_size: int = 224,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.hf_model_name = hf_model_name
        self.image_size = image_size
        self.encoder = AutoModel.from_pretrained(hf_model_name)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden_size, embed_dim)
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        # Optional on-disk cache of the *frozen* CLS features. The backbone is
        # frozen and inputs are fixed per spot (no augmentation), so the CLS
        # feature for a given (slide, shape, image_size) is constant across
        # epochs / seeds / model variants. Caching it is numerically exact and
        # skips the expensive ViT-L forward; only the trainable `proj` runs on
        # every call. Disabled unless `set_feature_cache` is called.
        self._feature_cache_dir: Optional[Path] = None
        self._input_transform: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None
        self._input_transform_tag = "clean"

    def set_feature_cache(self, cache_dir: Optional[str | Path]) -> None:
        if cache_dir is None:
            self._feature_cache_dir = None
            return
        self._feature_cache_dir = Path(cache_dir)
        self._feature_cache_dir.mkdir(parents=True, exist_ok=True)

    def set_input_transform(
        self,
        transform: Optional[Callable[[torch.Tensor, int], torch.Tensor]],
        *,
        tag: str = "clean",
    ) -> None:
        """Install a chunk-wise inference transform for robustness audits.

        ``transform`` receives a device-resident image chunk and its first
        spot index. The tag becomes part of the frozen-feature cache key, so
        clean and perturbed CLS features cannot collide. Training and ordinary
        inference retain the exact previous behaviour because the default is
        ``None`` with tag ``clean``.
        """
        if not tag or "::" in tag:
            raise ValueError("input-transform tag must be non-empty and cannot contain '::'")
        self._input_transform = transform
        self._input_transform_tag = tag

    def train(self, mode: bool = True) -> "_FrozenFoundationModelBackbone":
        super().train(mode)
        self.encoder.eval()  # frozen backbone always stays in eval mode, even during model.train()
        return self

    def _encode_cls(self, patches: torch.Tensor, chunk_size: int) -> torch.Tensor:
        device = self.proj.weight.device
        cls_chunks = []
        with torch.no_grad():
            for start in range(0, patches.shape[0], chunk_size):
                x = patches[start : start + chunk_size].to(device, non_blocking=True)
                if self._input_transform is not None:
                    x = self._input_transform(x, start)
                    if x.shape != patches[start : start + chunk_size].shape:
                        raise RuntimeError("input transform changed the patch tensor shape")
                    x = x.clamp(0.0, 1.0)
                if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
                    x = F.interpolate(
                        x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False
                    )
                x = (x - self._mean) / self._std
                out = self.encoder(x)
                cls_chunks.append(out.last_hidden_state[:, 0])
        return torch.cat(cls_chunks, dim=0)

    def forward(
        self, patches: torch.Tensor, chunk_size: int = 64, cache_key: Optional[str] = None
    ) -> torch.Tensor:
        # Stream patches in chunks so large Xenium slides (10k–20k spots) fit
        # in VRAM; patches may live on CPU and are moved per chunk.
        device = self.proj.weight.device
        n = patches.shape[0]
        if n == 0:
            return patches.new_zeros((0, self.proj.out_features)).to(device)

        cache_path: Optional[Path] = None
        if self._feature_cache_dir is not None and cache_key is not None:
            tag = (
                f"{cache_key}::{tuple(patches.shape)}::{self.image_size}::"
                f"{self._input_transform_tag}"
            )
            fname = hashlib.sha1(tag.encode()).hexdigest() + ".pt"
            cache_path = self._feature_cache_dir / fname

        cls_feat: Optional[torch.Tensor] = None
        if cache_path is not None and cache_path.exists():
            cls_feat = torch.load(cache_path, map_location=device)
            if cls_feat.shape[0] != n:  # stale / collision guard
                cls_feat = None
        if cls_feat is None:
            cls_feat = self._encode_cls(patches, chunk_size)
            if cache_path is not None:
                torch.save(cls_feat.detach().cpu(), cache_path)
        return self.proj(cls_feat)


def build_m1_backbone(backbone_name: str, embed_dim: int, in_channels: int = 3) -> nn.Module:
    if backbone_name == "tiny_cnn":
        return _MultiScaleConvBackbone(in_channels, embed_dim)
    if backbone_name == "phikon_v2":
        return _FrozenFoundationModelBackbone(embed_dim, hf_model_name="owkin/phikon-v2", image_size=224)
    raise ValueError(
        f"Unknown backbone_name: {backbone_name!r} (expected 'tiny_cnn' or 'phikon_v2')"
    )


class StructuralPriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 64,
        backbone: nn.Module | None = None,
        backbone_name: str = "tiny_cnn",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone or build_m1_backbone(backbone_name, embed_dim, in_channels)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.morphology_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.boundary_head = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))
        self.complexity_head = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))

    def set_feature_cache(self, cache_dir) -> None:
        setter = getattr(self.backbone, "set_feature_cache", None)
        if callable(setter):
            setter(cache_dir)

    def forward(self, patches: torch.Tensor, cache_key: Optional[str] = None) -> M1Output:
        if cache_key is not None and hasattr(self.backbone, "_feature_cache_dir"):
            feats = self.backbone(patches, cache_key=cache_key)
        else:
            feats = self.backbone(patches)
        embedding = self.morphology_head(feats)
        boundaryness = torch.sigmoid(self.boundary_head(feats)).squeeze(-1)
        complexity = torch.sigmoid(self.complexity_head(feats)).squeeze(-1)
        return M1Output(morphology_embedding=embedding, boundaryness=boundaryness, complexity=complexity)
