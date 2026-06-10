from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional


try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None


@contextmanager
def start_run(
    experiment_name: str,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> Iterator[None]:
    if mlflow is None:
        yield
        return

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.enable_system_metrics_logging()
    with mlflow.start_run(run_name=run_name):
        yield


def log_params(params: Dict[str, object]) -> None:
    if mlflow is None:
        return
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    if mlflow is None:
        return
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str) -> None:
    if mlflow is None:
        return
    mlflow.log_artifact(path)
