#!/usr/bin/env bash
# Run the FINAL FULL CAPACITY suite (embed_dim=128) on the 50k dataset
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "========================================="
echo " 50k FINAL Model Experiment (128 dim)"
echo " MLflow tracking: ON"
echo "========================================="
echo ""

# 2. Train Standard CvT
echo "[2/3] Training Standard CvT..."
uv run python -m src.cli.train --config configs/experiment/exp08_cvt_final.yaml

# 3. Train Recurrent CvT
echo "[3/3] Training Recurrent CvT..."
uv run python -m src.cli.train --config configs/experiment/exp09_recurrent_final.yaml

# 1. Train CNN
echo "[1/3] Training CNN..."
uv run python -m src.cli.train --config configs/experiment/exp07_cnn_final.yaml

echo "ALL TRAINING FINISHED! View metrics in MLflow."
