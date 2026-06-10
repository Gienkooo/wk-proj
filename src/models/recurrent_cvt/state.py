from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class RecurrentState:
    embeddings: List[torch.Tensor]
