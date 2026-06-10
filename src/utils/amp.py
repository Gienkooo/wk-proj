from __future__ import annotations

from contextlib import contextmanager
import inspect
from typing import Generator

import torch


try:
    from torch.amp import autocast as _autocast
    from torch.amp import GradScaler as _GradScaler
    _AMP_KIND = "torch"
except ImportError:  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import autocast as _autocast
    from torch.cuda.amp import GradScaler as _GradScaler
    _AMP_KIND = "cuda"


def get_grad_scaler(device: torch.device, enabled: bool) -> _GradScaler:
    if not enabled:
        return _GradScaler(enabled=False)
    if _AMP_KIND == "torch":
        sig = inspect.signature(_GradScaler)
        if "device_type" in sig.parameters:
            return _GradScaler(device_type=device.type, enabled=True)
    return _GradScaler(enabled=True)


@contextmanager
def autocast_context(device: torch.device, enabled: bool) -> Generator[None, None, None]:
    if _AMP_KIND == "torch":
        with _autocast(device_type=device.type, enabled=enabled):
            yield
    else:
        with _autocast(enabled=enabled):
            yield
