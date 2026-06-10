from __future__ import annotations

from typing import List

import torch

from src.models.recurrent_cvt.cell import RecurrentCvTCell


def unroll_cell(
    cell: RecurrentCvTCell,
    tokens: torch.Tensor,
    steps: int,
    state: torch.Tensor | None = None,
) -> List[torch.Tensor]:
    embeddings: List[torch.Tensor] = []
    current = state
    for _ in range(steps):
        current = cell(tokens, current)
        embeddings.append(current)
    return embeddings
