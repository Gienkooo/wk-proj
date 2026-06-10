from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from src.data.schema import Sample


def generate_synthetic_metadata(
    output_path: str | Path,
    num_samples: int,
    num_shapes: int,
    num_colors: int,
    num_materials: int,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    samples: List[Sample] = []
    for idx in range(num_samples):
        samples.append(
            Sample(
                image_path=None,
                shape_id=int(rng.integers(0, num_shapes)),
                color_id=int(rng.integers(0, num_colors)),
                material_id=int(rng.integers(0, num_materials)),
                x=float(rng.uniform(-1.0, 1.0)),
                y=float(rng.uniform(-1.0, 1.0)),
                z=float(rng.uniform(-1.0, 1.0)),
                rotation=(
                    float(rng.uniform(0.0, 2.0 * np.pi)),
                    float(rng.uniform(0.0, 2.0 * np.pi)),
                    float(rng.uniform(0.0, 2.0 * np.pi)),
                ),
                scale=float(rng.uniform(0.7, 1.3)),
                scene_id=idx,
                canonical_group_id=idx,
            )
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            # Convert rotation tuple to list for JSON serialization
            d = sample.__dict__.copy()
            d["rotation"] = list(d["rotation"])
            handle.write(json.dumps(d) + "\n")
