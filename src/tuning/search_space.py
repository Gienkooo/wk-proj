from __future__ import annotations

from typing import Dict


def cnn_search_space() -> Dict[str, object]:
    return {
        "lr": (1e-4, 3e-3),
        "weight_decay": (1e-6, 1e-2),
        "dropout": (0.0, 0.3),
        "embed_dim": [64, 96, 128, 192],
    }


def recurrent_search_space() -> Dict[str, object]:
    return {
        "recurrent_steps": [2, 4, 6, 8],
        "embed_dim": [64, 96, 128],
        "lr": (1e-4, 3e-3),
        "weight_decay": (1e-6, 1e-2),
    }
