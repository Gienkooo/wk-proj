from __future__ import annotations

from typing import Callable, Dict

import optuna


def build_objective(train_fn: Callable[[Dict[str, object]], float]) -> Callable[[optuna.Trial], float]:
    def _objective(trial: optuna.Trial) -> float:
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "embed_dim": trial.suggest_categorical("embed_dim", [64, 96, 128, 192]),
        }
        return train_fn(params)

    return _objective
