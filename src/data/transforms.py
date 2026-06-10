from __future__ import annotations

from typing import Callable

import torch


def rotation_to_sin_cos(theta: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)


def build_image_transform(image_size: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def _transform(image: torch.Tensor) -> torch.Tensor:
        if image.shape[-1] != image_size or image.shape[-2] != image_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return image

    return _transform
