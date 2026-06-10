import json
from pathlib import Path
import numpy as np
from src.data.splits import save_splits
from src.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    root = Path("dataset_50k")
    split_path = Path("data/splits/compositional_50k.json")
    metrics_path = Path("data/splits/compositional_metrics.json")
    
    # 1. Identify the rules
    # Color-Shape Combos
    ood_combos = {
        ("l", "blue"),
        ("cut_cone", "red"),
        ("macaroni", "yellow"),
        ("cut_cube", "cyan")
    }
    
    # Size bounds
    min_size = 0.35
    max_size = 0.95
    
    # Positional bounds
    border_limit = 2.2

    # 2. Iterate through all metadata
    metadata_files = sorted(root.glob("*_metadata.json"))
    if not metadata_files:
        logger.error(f"No metadata files found in {root}")
        return

    test_indices = []
    standard_indices = []
    
    metrics = {
        "rule_combo": 0,
        "rule_size": 0,
        "rule_position": 0,
        "total_test": 0,
        "total_train": 0,
        "total_val": 0,
        "total_samples": len(metadata_files)
    }

    logger.info(f"Scanning {len(metadata_files)} files for compositional splits...")

    for idx, meta_path in enumerate(metadata_files):
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
            
        is_test = False
        
        # We only look at the first object per the enforce_single_object rules
        if not data.get("objects"):
            standard_indices.append(idx)
            continue
            
        obj = data["objects"][0]
        
        shape = obj.get("shape", "").lower()
        color = obj.get("color", "").lower()
        size = float(obj.get("size", 0.0))
        pos = obj.get("position", [0.0, 0.0, 0.0])
        
        # Rule 1: Combos
        if (shape, color) in ood_combos:
            is_test = True
            metrics["rule_combo"] += 1
            
        # Rule 2: Extreme Size
        if size < min_size or size > max_size:
            is_test = True
            metrics["rule_size"] += 1
            
        # Rule 3: Extreme Border
        if abs(pos[0]) > border_limit or abs(pos[1]) > border_limit:
            is_test = True
            metrics["rule_position"] += 1
            
        if is_test:
            test_indices.append(idx)
        else:
            standard_indices.append(idx)

    # 3. Create Splits
    # Shuffle standard indices for train/val split
    np.random.seed(42)
    standard_indices = np.array(standard_indices)
    np.random.shuffle(standard_indices)
    
    # 90% train, 10% val of the REMAINING standard images
    val_split_ratio = 0.1
    val_count = int(len(standard_indices) * val_split_ratio)
    
    val_indices = standard_indices[:val_count].tolist()
    train_indices = standard_indices[val_count:].tolist()
    
    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    
    metrics["total_test"] = len(test_indices)
    metrics["total_val"] = len(val_indices)
    metrics["total_train"] = len(train_indices)

    # 4. Save outputs
    save_splits(split_path, splits)
    
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Compositional split generation complete!")
    logger.info(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
