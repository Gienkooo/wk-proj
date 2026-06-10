from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.data.schema import Sample


@dataclass(frozen=True)
class RawSceneRecord:
    image_path: str
    scene_id: str
    shape: str
    color: str
    material: str
    size: float
    position: Sequence[float]
    rotation: Sequence[float]
    rotation_unit: str


def load_scene_json_samples(
    metadata_dir: str | Path,
    metadata_glob: str = "*_metadata.json",
    image_ext: str = ".png",
    rotation_unit_override: Optional[str] = None,
    enforce_single_object: bool = True,
) -> List[Sample]:
    """Load scene JSON metadata files and return a list of Samples with full 3D data.

    Each metadata JSON is expected to have an ``objects`` list with at least one
    entry containing ``shape``, ``color``, ``material``, ``size``, ``position``
    (3-element list) and ``rotation`` (3-element list).
    """
    metadata_dir = Path(metadata_dir)
    records = _load_scene_records(
        metadata_dir,
        metadata_glob,
        image_ext,
        rotation_unit_override,
        enforce_single_object,
    )
    if not records:
        raise ValueError(f"No metadata files found in {metadata_dir}")

    shape_map = _build_label_map(record.shape for record in records)
    color_map = _build_label_map(record.color for record in records)
    material_map = _build_label_map(record.material for record in records)

    num_colors = max(len(color_map), 1)
    num_materials = max(len(material_map), 1)

    samples: List[Sample] = []
    for idx, record in enumerate(records):
        shape_id = shape_map[_normalize_label(record.shape)]
        color_id = color_map[_normalize_label(record.color)]
        material_id = material_map[_normalize_label(record.material)]

        pos = record.position
        x = float(pos[0]) if len(pos) > 0 else 0.0
        y = float(pos[1]) if len(pos) > 1 else 0.0
        z = float(pos[2]) if len(pos) > 2 else 0.0

        is_degrees = record.rotation_unit.lower() == "degrees"
        raw_rot = record.rotation
        rotation = (
            math.radians(float(raw_rot[0])) if is_degrees else float(raw_rot[0]) if len(raw_rot) > 0 else 0.0,
            math.radians(float(raw_rot[1])) if is_degrees else float(raw_rot[1]) if len(raw_rot) > 1 else 0.0,
            math.radians(float(raw_rot[2])) if is_degrees else float(raw_rot[2]) if len(raw_rot) > 2 else 0.0,
        )

        scene_id = _scene_id_to_int(record.scene_id, idx)
        canonical_group_id = (
            shape_id * (num_colors * num_materials) + color_id * num_materials + material_id
        )

        samples.append(
            Sample(
                image_path=record.image_path,
                shape_id=shape_id,
                color_id=color_id,
                material_id=material_id,
                x=x,
                y=y,
                z=z,
                rotation=rotation,
                scale=float(record.size),
                scene_id=scene_id,
                canonical_group_id=canonical_group_id,
            )
        )

    return samples


def _load_scene_records(
    metadata_dir: Path,
    metadata_glob: str,
    image_ext: str,
    rotation_unit_override: Optional[str],
    enforce_single_object: bool,
) -> List[RawSceneRecord]:
    records: List[RawSceneRecord] = []
    for meta_path in sorted(metadata_dir.glob(metadata_glob)):
        with meta_path.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                from src.utils.logging import get_logger
                get_logger(__name__).warning("Skipping invalid JSON file: %s", meta_path)
                continue
        objects = data.get("objects", [])
        if not objects:
            continue
        if enforce_single_object and len(objects) != 1:
            raise ValueError(f"Expected single object in {meta_path}")
        obj = objects[0]

        image_path = _resolve_image_path(meta_path, image_ext)
        rotation_unit = (
            rotation_unit_override
            or data.get("object_rotation_unit")
            or data.get("rotation_unit")
            or "degrees"
        )
        records.append(
            RawSceneRecord(
                image_path=str(image_path),
                scene_id=str(data.get("scene_id", meta_path.stem)),
                shape=str(obj.get("shape", "unknown")),
                color=str(obj.get("color", "unknown")),
                material=str(obj.get("material", "unknown")),
                size=float(obj.get("size", 1.0)),
                position=obj.get("position", [0.0, 0.0, 0.0]),
                rotation=obj.get("rotation", [0.0, 0.0, 0.0]),
                rotation_unit=str(rotation_unit),
            )
        )
    return records


def _resolve_image_path(meta_path: Path, image_ext: str) -> Path:
    if meta_path.name.endswith("_metadata.json"):
        image_name = meta_path.name.replace("_metadata.json", image_ext)
        return meta_path.with_name(image_name)
    return meta_path.with_suffix(image_ext)


def _build_label_map(values: Iterable[str]) -> Dict[str, int]:
    normalized = sorted({_normalize_label(value) for value in values})
    return {value: idx for idx, value in enumerate(normalized)}


def _normalize_label(value: str) -> str:
    return value.strip().lower()


def _scene_id_to_int(scene_id: str, fallback: int) -> int:
    match = re.match(r"^(\d+)", scene_id)
    if match:
        return int(match.group(1))
    return fallback
