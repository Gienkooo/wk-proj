from __future__ import annotations

import torch
from torch import nn


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, num_tokens: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embedding
