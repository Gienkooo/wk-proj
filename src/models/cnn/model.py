from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from src.models.common.heads import MultiTaskHeads


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out


class CNNBackbone(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, depth: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        channels = base_channels
        blocks = []
        for stage in range(depth):
            out_channels = channels * 2 if stage == 1 else channels
            blocks.append(ResidualBlock(channels, out_channels, stride=2 if stage > 0 else 1))
            blocks.append(ResidualBlock(out_channels, out_channels))
            channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return x


class CNNModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_shapes: int,
        num_colors: int,
        num_materials: int,
        embed_dim: int,
        depth: int,
        base_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = CNNBackbone(in_channels, base_channels=base_channels, depth=depth)
        self.proj = nn.Linear(self.backbone.out_dim, embed_dim)
        self.heads = MultiTaskHeads(
            embed_dim,
            num_shapes=num_shapes,
            num_colors=num_colors,
            num_materials=num_materials,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.proj(self.backbone(x))
        return self.heads(latent)
