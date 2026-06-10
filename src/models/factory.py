from __future__ import annotations

from typing import Any

from torch import nn

from src.models.cnn.model import CNNModel
from src.models.cvt.model import CvTModel
from src.models.recurrent_cvt.model import RecurrentCvTModel
from src.utils.config import DataConfig, ModelConfig


def build_model(config: ModelConfig, data: DataConfig) -> nn.Module:
    name = config.name.lower()
    if name == "cnn":
        return CNNModel(
            in_channels=data.channels,
            num_shapes=data.num_shapes,
            num_colors=data.num_colors,
            num_materials=data.num_materials,
            embed_dim=config.embed_dim,
            depth=config.depth,
            base_channels=config.cnn_base_channels,
            dropout=config.dropout,
        )
    if name == "cvt":
        return CvTModel(
            in_channels=data.channels,
            num_shapes=data.num_shapes,
            num_colors=data.num_colors,
            num_materials=data.num_materials,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            tokenizer_kernel=config.tokenizer_kernel,
            tokenizer_stride=config.tokenizer_stride,
            use_cls_token=config.use_cls_token,
        )
    if name == "recurrent_cvt":
        return RecurrentCvTModel(
            in_channels=data.channels,
            num_shapes=data.num_shapes,
            num_colors=data.num_colors,
            num_materials=data.num_materials,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            tokenizer_kernel=config.tokenizer_kernel,
            tokenizer_stride=config.tokenizer_stride,
            recurrent_steps=config.recurrent_steps,
        )
    raise ValueError(f"Unknown model name: {config.name}")
