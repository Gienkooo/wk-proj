from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.engine.validate import validate
from src.losses.consistency import step_consistency_loss
from src.losses.multitask import MultiTaskLoss
from src.utils.amp import autocast_context, get_grad_scaler
from src.utils.logging import get_logger


logger = get_logger(__name__)


def _forward_model(model: nn.Module, images: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], list[torch.Tensor] | None]:
    try:
        outputs = model(images, return_embeddings=True)
    except TypeError:
        outputs = model(images)
    if isinstance(outputs, tuple):
        return outputs[0], outputs[1]
    return outputs, None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    log_interval: int,
) -> Dict[str, float]:
    model.train()
    scaler = get_grad_scaler(device, amp)
    running: Dict[str, float] = {}

    for step, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, enabled=amp):
            outputs, embeddings = _forward_model(model, batch.images)
            traj_loss = step_consistency_loss(embeddings) if embeddings else None
            total_loss, metrics = loss_fn(outputs, batch, traj_loss)

        scaler.scale(total_loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        for key, value in metrics.items():
            running[key] = running.get(key, 0.0) + value

        if step % log_interval == 0:
            logger.info("train step %d loss %.4f", step, metrics["total"])

    return {key: value / max(len(loader), 1) for key, value in running.items()}


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    loss_fn: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
    log_interval: int,
    epochs: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    mlflow_per_epoch: bool = False,
    save_dir: str | None = None,
    patience: int = 10,
) -> List[Dict[str, float]]:
    """Train the model and return per-epoch metrics history.

    Returns:
        A list of dicts, one per epoch, containing ``train_*`` and ``val_*``
        metric keys.
    """
    history: List[Dict[str, float]] = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        logger.info("epoch %d/%d", epoch, epochs)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            amp,
            log_interval,
        )
        epoch_metrics: Dict[str, float] = {f"train_{k}": v for k, v in train_metrics.items()}
        if val_loader is not None:
            val_metrics = validate(model, val_loader, loss_fn, device, amp)
            epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            logger.info(
                "epoch %d — train_loss: %.4f  val_loss: %.4f",
                epoch,
                train_metrics.get("total", 0.0),
                val_metrics.get("total", 0.0),
            )

        history.append(epoch_metrics)

        if mlflow_per_epoch:
            from src.tracking.mlflow_utils import log_metrics
            log_metrics(epoch_metrics, step=epoch)

        if scheduler is not None:
            scheduler.step()

        if val_loader is not None:
            val_loss = epoch_metrics.get("val_total", float('inf'))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                if save_dir:
                    from src.engine.checkpoint import save_checkpoint
                    from pathlib import Path
                    save_checkpoint(Path(save_dir) / "best.pt", model, optimizer, epoch)
                    logger.info("Saved new best checkpoint with val_loss: %.4f", val_loss)
            else:
                epochs_without_improvement += 1
                logger.info("No improvement for %d epochs.", epochs_without_improvement)

            if epochs_without_improvement >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    return history
