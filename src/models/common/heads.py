from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from src.models.common.mlp import MLP


class MultiTaskHeads(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_shapes: int,
        num_colors: int = 0,
        num_materials: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.shape_head = MLP(input_dim, [input_dim], num_shapes, dropout=dropout)
        self.position_head = MLP(input_dim, [input_dim], 3, dropout=dropout)   # x, y, z
        self.rotation_head = MLP(input_dim, [input_dim], 6, dropout=dropout)   # sin/cos × 3 axes
        self.scale_head = MLP(input_dim, [input_dim], 1, dropout=dropout)
        self.color_head = (
            MLP(input_dim, [input_dim], num_colors, dropout=dropout)
            if num_colors > 0
            else None
        )
        self.material_head = (
            MLP(input_dim, [input_dim], num_materials, dropout=dropout)
            if num_materials > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {
            "shape": self.shape_head(x),
            "position": self.position_head(x),       # (B, 3)
        }
        
        # Normalize rotation outputs to the unit circle to prevent magnitude collapse
        rot_raw = self.rotation_head(x)          # (B, 6)
        B = rot_raw.shape[0]
        # Reshape to (B, 3, 2) to normalize each axis independently
        rot_pairs = rot_raw.view(B, 3, 2)
        rot_out = rot_pairs / (rot_pairs.norm(dim=-1, keepdim=True) + 1e-6)
        outputs["rotation"] = rot_out.view(B, 6) # Flatten back to (B, 6)
        
        outputs["scale"] = self.scale_head(x)
        if self.color_head is not None:
            outputs["color"] = self.color_head(x)
        if self.material_head is not None:
            outputs["material"] = self.material_head(x)
        return outputs
