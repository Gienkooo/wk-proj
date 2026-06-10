from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.losses.consistency import step_consistency_loss
from src.losses.multitask import MultiTaskLoss
from src.utils.amp import autocast_context


def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    amp: bool,
) -> Dict[str, float]:
    model.eval()

    all_outputs: Dict[str, list] = {}
    all_targets_list: list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with autocast_context(device, enabled=amp):
                try:
                    outputs = model(batch.images, return_embeddings=True)
                except TypeError:
                    outputs = model(batch.images)
                outputs = outputs[0] if isinstance(outputs, tuple) else outputs

            # Accumulate predictions
            for key, val in outputs.items():
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append(val.float())  # float() in case of AMP fp16

            # Accumulate targets (keep the whole batch object, but free images)
            batch.images = torch.empty(0, device=device)
            all_targets_list.append(batch)

    # Concatenate all predictions
    cat_outputs = {key: torch.cat(vals, dim=0) for key, vals in all_outputs.items()}

    # Concatenate all target batches into one
    # This assumes your batch object supports attribute-wise cat
    # Adjust field names to match your actual SceneBatch dataclass
    cat_batch = _cat_batches(all_targets_list)

    _, metrics = loss_fn(cat_outputs, cat_batch, traj_loss=None)
    return metrics


def _cat_batches(batches):
    """Concatenate a list of batch objects field by field."""
    # Replace with your actual batch dataclass fields
    from dataclasses import fields
    cls = type(batches[0])
    cat_fields = {}
    for f in fields(batches[0]):
        vals = [getattr(b, f.name) for b in batches]
        if isinstance(vals[0], torch.Tensor):
            cat_fields[f.name] = torch.cat(vals, dim=0)
        else:
            cat_fields[f.name] = vals[0]  # non-tensor fields: just use first
    return cls(**cat_fields)
