from __future__ import annotations

import torch
from torch import nn
from src.models.cvt.block import TransformerBlock

class RecurrentCvTCell(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        depth: int = 1,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
            
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        
        # It takes concatenated (prev_state + tokens) -> 2 * embed_dim channels
        self.state_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)

    def forward(self, tokens: torch.Tensor, prev_state: torch.Tensor | None) -> torch.Tensor:
        # Both tokens and prev_state are now (B, C, H, W)
        if prev_state is None:
            x = tokens
        else:
            # Concatenate along the channel dimension (dim=1)
            # Resulting shape: (B, 2 * C, H, W)
            x = torch.cat([prev_state, tokens], dim=1)
            # Project back to (B, C, H, W)
            x = self.state_proj(x)
            
        x = self.blocks(x)
        return x  # Return the updated 2D grid
