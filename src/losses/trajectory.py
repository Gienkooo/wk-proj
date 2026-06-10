from __future__ import annotations

from typing import List

import torch


def distance_to_final(embeddings: List[torch.Tensor]) -> torch.Tensor:
    if len(embeddings) < 2:
        return torch.zeros(1, device=embeddings[0].device)
    final = embeddings[-1]
    distances = []
    for step in embeddings[:-1]:
        distances.append(torch.norm(final - step, dim=-1).mean())
    return torch.stack(distances)
