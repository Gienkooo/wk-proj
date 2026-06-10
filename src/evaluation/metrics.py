from __future__ import annotations

from typing import Dict

import torch

from src.data.schema import Batch
from src.losses.rotation import angular_error


def compute_metrics(outputs: Dict[str, torch.Tensor], batch: Batch) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    pred_shape = outputs["shape"].argmax(dim=-1)
    metrics["shape_acc"] = (pred_shape == batch.shape_id).float().mean().item()
    metrics["pos_mae"] = torch.mean(torch.abs(outputs["position"] - batch.pos)).item()
    metrics["rot_mae"] = angular_error(outputs["rotation"], batch.rotation).mean().item()
    metrics["scale_mae"] = torch.mean(torch.abs(outputs["scale"].squeeze(-1) - batch.scale)).item()
    return metrics
