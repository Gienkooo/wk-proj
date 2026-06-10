# Recurrent CvT — Single-Object Multi-Head Prediction

Can a recurrent CvT backbone learn object-centric representations for single-object scenes that support multi-attribute prediction and exhibit convergent latent trajectories over recurrent steps?

## Quick Overview

| Component | Details |
|---|---|
| **Task** | Multi-head geometric understanding: shape class, 3D position (x,y,z), 3D rotation (rx,ry,rz), scale |
| **Models** | CNN baseline → Standard CvT → Recurrent CvT (iterative refinement) |
| **Data** | 50,000 synthetic CLEVR-style scenes with Rotation OOD splits |
| **Experiments** | In-distribution accuracy, Rotation Extrapolation (270°-360°), and Recurrent Depth Ablation (1, 2, 4, 8 steps) |
| **Tracking** | MLflow, local checkpoints |
| **Hardware** | NVIDIA RTX 5070 Ti 16GB, mixed precision (AMP) |

---

## Setup

### 1. Install dependencies

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e .
```

### 2. Verify CUDA

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

---

## Dataset Format

The project uses **scene JSON** metadata files paired with PNG images:

```
pilot_dataset/
├── 00000_cube_gray_rubber_metadata.json
├── 00000_cube_gray_rubber.png
├── 00001_cube_gray_rubber_metadata.json
├── 00001_cube_gray_rubber.png
└── ...
```

Each `*_metadata.json` has this structure:
```json
{
    "scene_id": "00017_cube_red_metal",
    "object_rotation_unit": "degrees",
    "resolution": [240, 240],
    "objects": [
        {
            "id": "obj_1",
            "shape": "cube",
            "color": "red",
            "material": "Metal",
            "size": 0.325,
            "position": [1.3648, -2.2237, 1.3297],
            "rotation": [298.54, 243.24, 158.06]
        }
    ]
}
```

**Key fields:**
- `position`: 3D coordinates `[x, y, z]`
- `rotation`: 3D Euler angles in the unit specified by `object_rotation_unit` (converted to radians internally)
- `size`: object scale factor
- `shape`/`color`/`material`: categorical labels (auto-mapped to integer IDs)

---

## Running the Pilot Experiment

The pilot dataset (240 samples) is small enough to train in seconds and validates the full pipeline before committing to large-scale training.

### Option A: Run all 3 experiments at once

```bash
bash scripts/run_pilot_experiment.sh
```

This runs CNN → CvT → Recurrent CvT sequentially and saves checkpoints.

### Option B: Run experiments individually

```bash
# Step 1: Generate train/val/test splits (70/15/15)
python -m src.cli.make_dataset --config configs/data/pilot_single.yaml

# Step 2: Train CNN baseline
python -m src.cli.train --config configs/experiment/exp01_cnn_pilot.yaml

# Step 3: Train CvT baseline
python -m src.cli.train --config configs/experiment/exp02_cvt_pilot.yaml

# Step 4: Train Recurrent CvT (T=4 unroll steps)
python -m src.cli.train --config configs/experiment/exp03_recurrent_pilot.yaml
```

### Option C: Quick synthetic smoke test (no data needed)

```bash
python -m src.cli.train --config configs/experiment/exp01_cnn.yaml --dry-run
```

---

## Configuration Reference

Experiment configs live in `configs/experiment/` and compose three sections:

### `data` section

```yaml
data:
  name: scene_json          # "scene_json" | "synthetic" | "metadata"
  image_size: 64             # images resized to this (square)
  channels: 3
  batch_size: 16
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  root: pilot_dataset        # directory containing *_metadata.json + *.png
  split_path: data/splits/pilot_single.json
  metadata_glob: "*_metadata.json"
  image_ext: ".png"
  rotation_unit_override: null   # set to "degrees" or "radians" to force
  enforce_single_object: true    # assert exactly 1 object per scene
```

> **Note:** `num_shapes`, `num_colors`, `num_materials` are **auto-detected** from the data when using `scene_json` mode. Set them to `0` in the config.

### `model` section

```yaml
model:
  name: cnn                  # "cnn" | "cvt" | "recurrent_cvt"
  embed_dim: 128             # latent embedding dimension
  depth: 2                   # number of transformer/residual stages
  num_heads: 4               # attention heads (CvT/recurrent only)
  mlp_ratio: 4.0
  dropout: 0.1
  tokenizer_kernel: 7        # conv tokenizer kernel (CvT only)
  tokenizer_stride: 4        # conv tokenizer stride (CvT only)
  recurrent_steps: 4         # T unroll steps (recurrent_cvt only)
  use_cls_token: true
```

### `train` section

```yaml
train:
  epochs: 10                 # increase for real training (50-100+)
  lr: 0.001
  weight_decay: 0.0001
  seed: 42
  device: cuda               # "cuda" or "cpu"
  amp: true                  # mixed precision (faster on GPU)
  log_interval: 5            # log every N steps
  loss_weights:
    shape: 1.0               # cross-entropy for shape classification
    pos: 1.0                 # SmoothL1 for (x, y, z) regression
    rot: 1.0                 # MSE on sin/cos(rx, ry, rz) pairs
    scale: 1.0               # SmoothL1 for scale regression
    traj: 0.0                # trajectory consistency (recurrent only, try 0.1)
  save_dir: outputs/checkpoints
```

### `tracking` section (optional)

```yaml
tracking:
  mlflow_enabled: false       # set true to log to MLflow
  tracking_uri: null          # e.g., "http://localhost:5000"
  experiment_name: single_object_cnn
  run_name: exp01_cnn_pilot
```

---

## Model Output Heads

All models predict **full 3D attributes** from a shared latent embedding:

| Head | Output dim | Target | Loss |
|---|---|---|---|
| `shape` | num_shapes (3) | shape class | CrossEntropy |
| `position` | 3 | x, y, z | SmoothL1 |
| `rotation` | 6 | sin/cos for rx, ry, rz | MSE |
| `scale` | 1 | size | SmoothL1 |
| `color` | num_colors (8) | color class | CrossEntropy |
| `material` | num_materials (2) | material class | CrossEntropy |

Rotation is encoded as 6 values `[sin(rx), cos(rx), sin(ry), cos(ry), sin(rz), cos(rz)]` to handle the cyclical nature of angles.

---

## Evaluation

```bash
python -m src.cli.evaluate \
  --config configs/experiment/exp01_cnn_pilot.yaml \
  --checkpoint outputs/checkpoints/last.pt \
  --split test
```

---

## Preparing Your Own Dataset

1. Generate scene images and metadata JSONs following the format above.
2. Place all `*_metadata.json` + `*.png` pairs in a directory (e.g., `my_dataset/`).
3. Create or copy a config:
   ```yaml
   data:
     name: scene_json
     root: my_dataset
     image_size: 64          # or 96, 128 for higher resolution
     batch_size: 32
     # ... rest of config
   ```
4. Run:
   ```bash
   python -m src.cli.make_dataset --config configs/data/my_data.yaml
   python -m src.cli.train --config configs/experiment/my_experiment.yaml
   ```

---

## Experiment Matrix (from PLAN.md)

| Exp | Model | Config | Purpose |
|---|---|---|---|
| E1 | CNN | `exp01_cnn_pilot.yaml` | Baseline |
| E2 | CvT | `exp02_cvt_pilot.yaml` | Transformer baseline |
| E3 | Recurrent CvT T=4 | `exp03_recurrent_pilot.yaml` | Recurrent with trajectories |

After pilot validation, scale up with larger datasets and increase `epochs` to 50–100+.

---

## Project Structure

```
├── configs/
│   ├── data/              # dataset configs
│   ├── experiment/        # full experiment configs (data+model+train)
│   ├── model/             # model-only configs
│   └── train/             # training-only configs
├── pilot_dataset/         # 240 CLEVR-like scene samples
├── src/
│   ├── cli/               # entry points: train, evaluate, make_dataset
│   ├── data/              # schema, dataset, scene_json loader, splits
│   ├── engine/            # training loop, validation, checkpointing
│   ├── losses/            # multitask, rotation, consistency, trajectory
│   ├── models/            # cnn/, cvt/, recurrent_cvt/, factory
│   ├── evaluation/        # metrics, plots, trajectory analysis
│   ├── tracking/          # MLflow utilities
│   └── utils/             # config, seed, device, logging, amp
├── scripts/               # automation scripts
├── outputs/               # checkpoints, figures, tables
└── notebooks/             # analysis notebooks
```
