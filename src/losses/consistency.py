from __future__ import annotations

from typing import List

import torch


def step_consistency_loss(embeddings: List[torch.Tensor]) -> torch.Tensor:
    if len(embeddings) < 2:
        return torch.tensor(0.0, device=embeddings[0].device)
    diffs = []
    for prev, curr in zip(embeddings[:-1], embeddings[1:]):
        diffs.append(torch.mean((curr - prev) ** 2))
    return torch.stack(diffs).mean()
