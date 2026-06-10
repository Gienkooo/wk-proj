from __future__ import annotations

import torch
from torch import nn


class ConvTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return the 2D grid (B, C, H, W) instead of flattening
        return self.proj(x)
