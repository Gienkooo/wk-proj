from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def make_split_indices(
    num_samples: int,
    train_split: float,
    val_split: float,
    seed: int,
) -> Dict[str, List[int]]:
    if train_split + val_split >= 1.0:
        raise ValueError("train_split + val_split must be < 1.0")
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    train_end = int(num_samples * train_split)
    val_end = train_end + int(num_samples * val_split)
    return {
        "train": indices[:train_end].tolist(),
        "val": indices[train_end:val_end].tolist(),
        "test": indices[val_end:].tolist(),
    }


def save_splits(path: str | Path, splits: Dict[str, List[int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)


def load_splits(path: str | Path) -> Dict[str, List[int]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
