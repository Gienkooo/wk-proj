from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from src.data.schema import Batch
from src.losses.rotation import rotation_mse
from src.utils.config import LossWeights


class MultiTaskLoss(nn.Module):
    def __init__(self, weights: LossWeights) -> None:
        super().__init__()
        self.weights = weights
        self.ce = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Batch,
        traj_loss: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses: Dict[str, torch.Tensor] = {}
        losses["shape"] = self.ce(outputs["shape"], batch.shape_id)
        losses["position"] = self.smooth_l1(outputs["position"], batch.pos)
        losses["rotation"] = rotation_mse(outputs["rotation"], batch.rotation)
        losses["scale"] = self.smooth_l1(outputs["scale"], batch.scale.unsqueeze(-1))
        if "color" in outputs:
            losses["color"] = self.ce(outputs["color"], batch.color_id)
        if "material" in outputs:
            losses["material"] = self.ce(outputs["material"], batch.material_id)

        total = (
            self.weights.shape * losses["shape"]
            + self.weights.pos * losses["position"]
            + self.weights.rot * losses["rotation"]
            + self.weights.scale * losses["scale"]
        )
        if "color" in losses:
            total = total + self.weights.color * losses["color"]
        if "material" in losses:
            total = total + self.weights.material * losses["material"]
        if traj_loss is not None:
            total = total + self.weights.traj * traj_loss

        metrics = {name: value.item() for name, value in losses.items()}
        metrics["total"] = total.item()
        return total, metrics
