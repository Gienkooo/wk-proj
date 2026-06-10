from __future__ import annotations

import torch
from torch import nn

from src.models.cvt.attention import MultiheadSelfAttention
from src.models.common.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim,
            [int(embed_dim * mlp_ratio)],
            embed_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Flatten temporarily for LayerNorm, then reshape back to 2D for Attention
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm1 = self.norm1(x_flat).transpose(1, 2).reshape(B, C, H, W)
        x = x + self.attn(x_norm1)
        
        # Flatten temporarily for LayerNorm and MLP, then reshape back
        xflat = x.flatten(2).transpose(1, 2)      # (B, HW, C)
        xnorm2 = self.norm2(xflat)                # LN over C
        xmlp = self.mlp(xnorm2)                   # (B, HW, C)
        x = x + xmlp.transpose(1, 2).reshape(B, C, H, W)
        
        return x
