from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from src.tracking.mlflow_utils import log_artifact


def save_config_artifact(config: Dict[str, object], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    log_artifact(str(path))
