from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.schema import Batch, Sample
from src.data.splits import load_splits, make_split_indices, save_splits
from src.data.transforms import build_image_transform
from src.utils.config import DataConfig


class SyntheticSingleObjectDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        image_size: int,
        channels: int,
        num_shapes: int,
        num_colors: int,
        num_materials: int,
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.rand(
            num_samples,
            channels,
            image_size,
            image_size,
            generator=generator,
        )
        shape_classes = max(num_shapes, 1)
        color_classes = max(num_colors, 1)
        material_classes = max(num_materials, 1)
        self.shape_id = torch.randint(0, shape_classes, (num_samples,), generator=generator)
        self.color_id = torch.randint(0, color_classes, (num_samples,), generator=generator)
        self.material_id = torch.randint(
            0, material_classes, (num_samples,), generator=generator
        )
        self.x = torch.rand(num_samples, generator=generator) * 2.0 - 1.0
        self.y = torch.rand(num_samples, generator=generator) * 2.0 - 1.0
        self.z = torch.rand(num_samples, generator=generator) * 2.0 - 1.0
        self.rotation = torch.rand(num_samples, 3, generator=generator) * 2.0 * np.pi
        self.scale = 0.7 + torch.rand(num_samples, generator=generator) * 0.6
        self.scene_id = torch.arange(num_samples)
        self.canonical_group_id = torch.arange(num_samples)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image": self.images[idx],
            "shape_id": self.shape_id[idx],
            "color_id": self.color_id[idx],
            "material_id": self.material_id[idx],
            "x": self.x[idx],
            "y": self.y[idx],
            "z": self.z[idx],
            "rotation": self.rotation[idx],          # (3,)
            "scale": self.scale[idx],
            "scene_id": self.scene_id[idx],
            "canonical_group_id": self.canonical_group_id[idx],
        }


class MetadataSingleObjectDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        image_size: int,
        channels: int,
        root: Optional[str] = None,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.channels = channels
        self.root = Path(root) if root else None
        self.transform = build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        if sample.image_path is None:
            raise ValueError("Sample is missing image_path")
        image_path = Path(sample.image_path)
        if self.root is not None:
            image_path = self.root / image_path

        if image_path.suffix in {".pt", ".pth"}:
            image = torch.load(image_path)
        else:
            pil_image = Image.open(image_path).convert("RGB")
            image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image.shape[0] != self.channels:
            raise ValueError("Unexpected channel count")
        image = self.transform(image)

        return {
            "image": image,
            "shape_id": torch.tensor(sample.shape_id, dtype=torch.long),
            "color_id": torch.tensor(sample.color_id, dtype=torch.long),
            "material_id": torch.tensor(sample.material_id, dtype=torch.long),
            "x": torch.tensor(sample.x, dtype=torch.float32),
            "y": torch.tensor(sample.y, dtype=torch.float32),
            "z": torch.tensor(sample.z, dtype=torch.float32),
            "rotation": torch.tensor(sample.rotation, dtype=torch.float32),  # (3,)
            "scale": torch.tensor(sample.scale, dtype=torch.float32),
            "scene_id": torch.tensor(sample.scene_id, dtype=torch.long),
            "canonical_group_id": torch.tensor(sample.canonical_group_id, dtype=torch.long),
        }


def collate_batch(items: Iterable[Dict[str, torch.Tensor]]) -> Batch:
    batch_items = list(items)
    images = torch.stack([item["image"] for item in batch_items])
    shape_id = torch.stack([item["shape_id"] for item in batch_items])
    color_id = torch.stack([item["color_id"] for item in batch_items])
    material_id = torch.stack([item["material_id"] for item in batch_items])
    pos = torch.stack(
        [torch.stack([item["x"], item["y"], item["z"]]) for item in batch_items]
    )  # (B, 3)
    rotation = torch.stack([item["rotation"] for item in batch_items])  # (B, 3)
    scale = torch.stack([item["scale"] for item in batch_items])
    scene_id = torch.stack([item["scene_id"] for item in batch_items])
    canonical_group_id = torch.stack([item["canonical_group_id"] for item in batch_items])

    return Batch(
        images=images,
        shape_id=shape_id,
        color_id=color_id,
        material_id=material_id,
        pos=pos,
        rotation=rotation,
        scale=scale,
        scene_id=scene_id,
        canonical_group_id=canonical_group_id,
    )


def load_metadata(path: str | Path) -> List[Sample]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return _load_metadata_jsonl(path)
    if path.suffix == ".csv":
        return _load_metadata_csv(path)
    raise ValueError(f"Unsupported metadata format: {path.suffix}")


def _parse_rotation(value) -> Tuple[float, float, float]:
    """Parse rotation from JSON/CSV into a 3-tuple of radians."""
    if isinstance(value, (list, tuple)):
        r = [float(v) for v in value]
        while len(r) < 3:
            r.append(0.0)
        return (r[0], r[1], r[2])
    return (float(value), 0.0, 0.0)


def _load_metadata_jsonl(path: Path) -> List[Sample]:
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            data["z"] = float(data.get("z", 0.0))
            data["rotation"] = _parse_rotation(data.get("rotation", (0.0, 0.0, 0.0)))
            samples.append(Sample(**data))
    return samples


def _load_metadata_csv(path: Path) -> List[Sample]:
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rotation = (
                float(row.get("rotation_x", row.get("rotation", 0.0))),
                float(row.get("rotation_y", 0.0)),
                float(row.get("rotation_z", 0.0)),
            )
            samples.append(
                Sample(
                    image_path=row.get("image_path"),
                    shape_id=int(row["shape_id"]),
                    color_id=int(row.get("color_id", 0)),
                    material_id=int(row.get("material_id", 0)),
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row.get("z", 0.0)),
                    rotation=rotation,
                    scale=float(row["scale"]),
                    scene_id=int(row.get("scene_id", 0)),
                    canonical_group_id=int(row.get("canonical_group_id", 0)),
                )
            )
    return samples


def select_split(
    samples: List[Sample],
    split: str,
    split_path: Optional[str],
    train_split: float,
    val_split: float,
    seed: int,
) -> List[Sample]:
    split_indices: Dict[str, List[int]]
    if split_path and Path(split_path).exists():
        split_indices = load_splits(split_path)
    else:
        split_indices = make_split_indices(len(samples), train_split, val_split, seed)
        if split_path:
            save_splits(split_path, split_indices)

    indices = split_indices.get(split)
    if indices is None:
        raise ValueError(f"Unknown split: {split}")
    return [samples[idx] for idx in indices]


def count_classes(samples: List[Sample]) -> Dict[str, int]:
    """Count unique class values from a list of samples."""
    shapes = {s.shape_id for s in samples}
    colors = {s.color_id for s in samples}
    materials = {s.material_id for s in samples}
    return {
        "num_shapes": max(len(shapes), max(shapes, default=-1) + 1),
        "num_colors": max(len(colors), max(colors, default=-1) + 1),
        "num_materials": max(len(materials), max(materials, default=-1) + 1),
    }


def load_scene_json_samples_for_config(config: DataConfig) -> List[Sample]:
    """Load all scene_json samples using DataConfig parameters."""
    from src.data.scene_json import load_scene_json_samples

    if not config.root:
        raise ValueError("root is required for scene_json dataset")
    return load_scene_json_samples(
        metadata_dir=config.root,
        metadata_glob=config.metadata_glob,
        image_ext=config.image_ext,
        rotation_unit_override=config.rotation_unit_override,
        enforce_single_object=config.enforce_single_object,
    )


def build_dataset(config: DataConfig, split: str, seed: int) -> Dataset:
    name = config.name.lower()
    if name == "synthetic":
        return SyntheticSingleObjectDataset(
            num_samples=config.num_samples,
            image_size=config.image_size,
            channels=config.channels,
            num_shapes=config.num_shapes,
            num_colors=max(config.num_colors, 1),
            num_materials=max(config.num_materials, 1),
            seed=seed,
        )
    if name == "metadata":
        if not config.metadata_path:
            raise ValueError("metadata_path is required for metadata dataset")
        samples = load_metadata(config.metadata_path)
        subset = select_split(
            samples,
            split=split,
            split_path=config.split_path,
            train_split=config.train_split,
            val_split=config.val_split,
            seed=seed,
        )
        return MetadataSingleObjectDataset(
            subset,
            image_size=config.image_size,
            channels=config.channels,
            root=config.root,
        )
    if name == "scene_json":
        samples = load_scene_json_samples_for_config(config)
        subset = select_split(
            samples,
            split=split,
            split_path=config.split_path,
            train_split=config.train_split,
            val_split=config.val_split,
            seed=seed,
        )
        return MetadataSingleObjectDataset(
            subset,
            image_size=config.image_size,
            channels=config.channels,
            root=None,  # scene_json already returns absolute paths
        )
    raise ValueError(f"Unknown dataset name: {config.name}")


def build_dataloaders(
    config: DataConfig,
    seed: int,
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = build_dataset(config, split, seed)
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_batch,
        )
    return loaders
