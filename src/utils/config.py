from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class LossWeights:
    shape: float = 1.0
    color: float = 1.0
    material: float = 1.0
    pos: float = 1.0
    rot: float = 1.0
    scale: float = 1.0
    traj: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LossWeights":
        return cls(
            shape=float(data.get("shape", cls.shape)),
            color=float(data.get("color", cls.color)),
            material=float(data.get("material", cls.material)),
            pos=float(data.get("pos", cls.pos)),
            rot=float(data.get("rot", cls.rot)),
            scale=float(data.get("scale", cls.scale)),
            traj=float(data.get("traj", cls.traj)),
        )


@dataclass(frozen=True)
class DataConfig:
    name: str
    image_size: int
    channels: int
    num_samples: int
    num_shapes: int
    num_colors: int
    num_materials: int
    batch_size: int
    num_workers: int
    train_split: float
    val_split: float
    test_split: float
    root: Optional[str] = None
    metadata_path: Optional[str] = None
    split_path: Optional[str] = None
    pin_memory: bool = True
    # scene_json-specific fields
    metadata_glob: str = "*_metadata.json"
    image_ext: str = ".png"
    rotation_unit_override: Optional[str] = None
    enforce_single_object: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        return cls(
            name=str(data["name"]),
            image_size=int(data["image_size"]),
            channels=int(data.get("channels", 3)),
            num_samples=int(data.get("num_samples", 0)),
            num_shapes=int(data.get("num_shapes", 0)),
            num_colors=int(data.get("num_colors", 0)),
            num_materials=int(data.get("num_materials", 0)),
            batch_size=int(data["batch_size"]),
            num_workers=int(data.get("num_workers", 0)),
            train_split=float(data.get("train_split", 0.8)),
            val_split=float(data.get("val_split", 0.1)),
            test_split=float(data.get("test_split", 0.1)),
            root=data.get("root"),
            metadata_path=data.get("metadata_path"),
            split_path=data.get("split_path"),
            pin_memory=bool(data.get("pin_memory", True)),
            metadata_glob=str(data.get("metadata_glob", "*_metadata.json")),
            image_ext=str(data.get("image_ext", ".png")),
            rotation_unit_override=data.get("rotation_unit_override"),
            enforce_single_object=bool(data.get("enforce_single_object", True)),
        )


@dataclass(frozen=True)
class ModelConfig:
    name: str
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    tokenizer_kernel: int
    tokenizer_stride: int
    recurrent_steps: int
    use_cls_token: bool = True
    cnn_base_channels: int = 32

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(
            name=str(data["name"]),
            embed_dim=int(data.get("embed_dim", 128)),
            depth=int(data.get("depth", 2)),
            num_heads=int(data.get("num_heads", 4)),
            mlp_ratio=float(data.get("mlp_ratio", 4.0)),
            dropout=float(data.get("dropout", 0.1)),
            tokenizer_kernel=int(data.get("tokenizer_kernel", 7)),
            tokenizer_stride=int(data.get("tokenizer_stride", 4)),
            recurrent_steps=int(data.get("recurrent_steps", 1)),
            use_cls_token=bool(data.get("use_cls_token", True)),
            cnn_base_channels=int(data.get("cnn_base_channels", 32)),
        )


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    seed: int
    device: str
    amp: bool
    log_interval: int
    loss_weights: LossWeights
    save_dir: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        return cls(
            epochs=int(data.get("epochs", 1)),
            lr=float(data.get("lr", 1e-3)),
            weight_decay=float(data.get("weight_decay", 1e-4)),
            seed=int(data.get("seed", 42)),
            device=str(data.get("device", "cuda")),
            amp=bool(data.get("amp", True)),
            log_interval=int(data.get("log_interval", 10)),
            loss_weights=LossWeights.from_dict(data.get("loss_weights", {})),
            save_dir=str(data.get("save_dir", "outputs/checkpoints")),
        )


@dataclass(frozen=True)
class TrackingConfig:
    mlflow_enabled: bool
    tracking_uri: Optional[str]
    experiment_name: str
    run_name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackingConfig":
        return cls(
            mlflow_enabled=bool(data.get("mlflow_enabled", False)),
            tracking_uri=data.get("tracking_uri"),
            experiment_name=str(data.get("experiment_name", "default")),
            run_name=data.get("run_name"),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    tracking: TrackingConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            data=DataConfig.from_dict(data["data"]),
            model=ModelConfig.from_dict(data["model"]),
            train=TrainConfig.from_dict(data["train"]),
            tracking=TrackingConfig.from_dict(data.get("tracking", {})),
        )


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    data = load_yaml(path)
    return ExperimentConfig.from_dict(data)
