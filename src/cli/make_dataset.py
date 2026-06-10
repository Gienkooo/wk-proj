from __future__ import annotations

import argparse
from src.data.dataset import count_classes, load_scene_json_samples_for_config
from src.data.generate import generate_synthetic_metadata
from src.data.splits import make_split_indices, save_splits
from src.utils.config import DataConfig, load_yaml
from src.utils.logging import get_logger


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset metadata or splits")
    parser.add_argument("--config", required=True, help="Path to data config")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    data = DataConfig.from_dict(raw["data"] if "data" in raw else raw)

    if data.name == "synthetic":
        output_path = data.metadata_path or "data/processed/synthetic_metadata.jsonl"
        generate_synthetic_metadata(
            output_path,
            num_samples=data.num_samples,
            num_shapes=max(data.num_shapes, 1),
            num_colors=max(data.num_colors, 1),
            num_materials=max(data.num_materials, 1),
            seed=0,
        )
        logger.info("wrote synthetic metadata to %s", output_path)
        num_samples = data.num_samples

    elif data.name == "scene_json":
        samples = load_scene_json_samples_for_config(data)
        counts = count_classes(samples)
        num_samples = len(samples)
        logger.info(
            "loaded %d scene_json samples: %d shapes, %d colors, %d materials",
            num_samples,
            counts["num_shapes"],
            counts["num_colors"],
            counts["num_materials"],
        )
    else:
        num_samples = data.num_samples

    split_path = data.split_path or "data/splits/default_splits.json"
    splits = make_split_indices(num_samples, data.train_split, data.val_split, seed=0)
    save_splits(split_path, splits)
    logger.info("wrote splits to %s", split_path)


if __name__ == "__main__":
    main()
