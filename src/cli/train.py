from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import torch

from src.data.dataset import (
    build_dataloaders,
    count_classes,
    load_scene_json_samples_for_config,
)
from src.engine.checkpoint import save_checkpoint
from src.engine.train import train_loop
from src.engine.validate import validate
from src.evaluation.metrics import compute_metrics
from src.losses.multitask import MultiTaskLoss
from src.models.factory import build_model
from src.tracking.artifacts import save_config_artifact
from src.tracking.mlflow_utils import log_artifact, log_metrics, log_params, start_run
from src.utils.config import load_experiment_config, load_yaml
from src.utils.device import get_device
from src.utils.logging import get_logger
from src.utils.seed import set_seed


logger = get_logger(__name__)


def _resolve_data_config(config):
    """Auto-detect class counts for scene_json datasets."""
    data = config.data
    if data.name.lower() == "scene_json":
        samples = load_scene_json_samples_for_config(data)
        counts = count_classes(samples)
        logger.info(
            "auto-detected classes: %d shapes, %d colors, %d materials from %d samples",
            counts["num_shapes"],
            counts["num_colors"],
            counts["num_materials"],
            len(samples),
        )
        data = replace(
            data,
            num_shapes=counts["num_shapes"],
            num_colors=counts["num_colors"],
            num_materials=counts["num_materials"],
        )
        config = replace(config, data=data)
    return config


def _evaluate_split(
    model: torch.nn.Module,
    loader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    amp: bool,
    split_name: str,
) -> Dict[str, float]:
    """Run evaluation on a split and return metrics."""
    model.eval()
    all_metrics: Dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            try:
                outputs = model(batch.images, return_embeddings=True)
            except TypeError:
                outputs = model(batch.images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            metrics = compute_metrics(outputs, batch)
            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + v
            count += 1
    avg = {f"{split_name}_{k}": v / max(count, 1) for k, v in all_metrics.items()}
    return avg


def _save_results(
    save_dir: Path,
    history: List[Dict[str, float]],
    eval_results: Dict[str, float],
    config_dict: dict,
) -> Path:
    """Save training history and eval results as JSON."""
    results = {
        "config": {
            "model": config_dict.get("model", {}).get("name", ""),
            "epochs": config_dict.get("train", {}).get("epochs", 0),
            "lr": config_dict.get("train", {}).get("lr", 0),
            "data": config_dict.get("data", {}).get("name", ""),
        },
        "history": history,
        "evaluation": eval_results,
    }
    results_path = save_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-head models")
    parser.add_argument("--config", required=True, help="Path to experiment config")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    raw_config = load_yaml(args.config)
    config = load_experiment_config(args.config)

    if args.dry_run and config.data.name != "synthetic":
        config = replace(config, data=replace(config.data, name="synthetic"))

    config = _resolve_data_config(config)

    set_seed(config.train.seed)
    device = get_device(config.train.device)
    amp = config.train.amp and device.type == "cuda"

    loaders = build_dataloaders(config.data, config.train.seed)
    model = build_model(config.model, config.data).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model %s initialized with %s total parameters (%s trainable).",
        config.model.name,
        f"{total_params:,}",
        f"{trainable_params:,}"
    )
    loss_fn = MultiTaskLoss(config.train.loss_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.epochs, eta_min=config.train.lr * 0.01
    )

    save_dir = Path(config.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mlflow_enabled = config.tracking.mlflow_enabled

    def _run_training():
        return train_loop(
            model,
            loaders["train"],
            loaders.get("val"),
            loss_fn,
            optimizer,
            device,
            amp,
            config.train.log_interval,
            config.train.epochs,
            scheduler=scheduler,
            mlflow_per_epoch=mlflow_enabled,
            save_dir=str(save_dir),
            patience=30,
        )

    if mlflow_enabled:
        with start_run(
            config.tracking.experiment_name,
            run_name=config.tracking.run_name,
            tracking_uri=config.tracking.tracking_uri,
        ):
            log_params(
                {
                    "model": config.model.name,
                    "seed": config.train.seed,
                    "batch_size": config.data.batch_size,
                    "lr": config.train.lr,
                    "weight_decay": config.train.weight_decay,
                    "embed_dim": config.model.embed_dim,
                    "depth": config.model.depth,
                    "num_heads": config.model.num_heads,
                    "tokenizer_kernel": config.model.tokenizer_kernel,
                    "tokenizer_stride": config.model.tokenizer_stride,
                    "recurrent_steps": config.model.recurrent_steps,
                    "num_shapes": config.data.num_shapes,
                    "num_colors": config.data.num_colors,
                    "num_materials": config.data.num_materials,
                    "epochs": config.train.epochs,
                    "data_name": config.data.name,
                }
            )
            save_config_artifact(raw_config, config.train.save_dir)
            history = _run_training()

            # Evaluate on test set
            eval_results = {}
            if "test" in loaders:
                eval_results = _evaluate_split(
                    model, loaders["test"], loss_fn, device, amp, "test"
                )
                log_metrics(eval_results)

            # Save checkpoint and results
            checkpoint_path = save_dir / "last.pt"
            save_checkpoint(checkpoint_path, model, optimizer, config.train.epochs)
            log_artifact(str(checkpoint_path))

            results_path = _save_results(save_dir, history, eval_results, raw_config)
            log_artifact(str(results_path))
    else:
        history = _run_training()
        eval_results = {}
        if "test" in loaders:
            eval_results = _evaluate_split(
                model, loaders["test"], loss_fn, device, amp, "test"
            )

        checkpoint_path = save_dir / "last.pt"
        save_checkpoint(checkpoint_path, model, optimizer, config.train.epochs)
        _save_results(save_dir, history, eval_results, raw_config)

    # Print final results summary
    logger.info("saved checkpoint to %s", save_dir / "last.pt")
    logger.info("saved results to %s", save_dir / "results.json")
    if history:
        last = history[-1]
        logger.info("--- final epoch metrics ---")
        for k, v in sorted(last.items()):
            logger.info("  %s: %.4f", k, v)
    if eval_results:
        logger.info("--- test evaluation ---")
        for k, v in sorted(eval_results.items()):
            logger.info("  %s: %.4f", k, v)


if __name__ == "__main__":
    main()
