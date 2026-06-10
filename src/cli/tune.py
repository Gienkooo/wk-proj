from __future__ import annotations

import argparse

from src.utils.logging import get_logger


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning entry point")
    parser.add_argument("--config", required=True, help="Path to tuning config")
    args = parser.parse_args()
    logger.info("tuning stub ready; implement optuna")


if __name__ == "__main__":
    main()
