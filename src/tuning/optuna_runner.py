from __future__ import annotations

from typing import Callable

import optuna


def run_study(objective: Callable[[optuna.Trial], float], n_trials: int = 20) -> optuna.Study:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study
