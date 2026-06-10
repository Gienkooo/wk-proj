from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class Sample:
    image_path: Optional[str]
    shape_id: int
    color_id: int
    material_id: int
    x: float
    y: float
    z: float
    rotation: Tuple[float, float, float]  # (rx, ry, rz) in radians
    scale: float
    scene_id: int
    canonical_group_id: int


@dataclass
class Batch:
    images: torch.Tensor
    shape_id: torch.Tensor
    color_id: torch.Tensor
    material_id: torch.Tensor
    pos: torch.Tensor          # (B, 3) — x, y, z
    rotation: torch.Tensor     # (B, 3) — angles in radians
    scale: torch.Tensor
    scene_id: torch.Tensor
    canonical_group_id: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            images=self.images.to(device),
            shape_id=self.shape_id.to(device),
            color_id=self.color_id.to(device),
            material_id=self.material_id.to(device),
            pos=self.pos.to(device),
            rotation=self.rotation.to(device),
            scale=self.scale.to(device),
            scene_id=self.scene_id.to(device),
            canonical_group_id=self.canonical_group_id.to(device),
        )
