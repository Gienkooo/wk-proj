import json
from pathlib import Path
import numpy as np
from src.data.splits import save_splits
from src.utils.logging import get_logger
import math

logger = get_logger(__name__)

def main():
    root = Path("dataset_50k")
    split_path = Path("data/splits/rotation_ood_50k.json")
    
    # 2. Iterate through all metadata
    metadata_files = sorted(root.glob("*_metadata.json"))
    if not metadata_files:
        logger.error(f"No metadata files found in {root}")
        return

    test_indices = []
    standard_indices = []
    
    # The quadrant to withhold: 270 to 360 degrees (in radians: 1.5*pi to 2.0*pi)
    min_ood_angle = 1.5 * math.pi
    max_ood_angle = 2.0 * math.pi

    logger.info(f"Scanning {len(metadata_files)} files for Rotation OOD split...")

    for idx, meta_path in enumerate(metadata_files):
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
            
        obj = data["objects"][0]
        
        # Rotation is often represented as Euler angles, let's grab the Z rotation
        # Depending on how rotation is stored in metadata (usually a float or list of 3)
        rot = obj.get("rotation", 0.0)
        if isinstance(rot, list):
            rot_z = rot[-1] # Usually Z is the last element
        else:
            rot_z = float(rot)
            
        # normalize to 0-2pi just in case
        rot_z = rot_z % (2 * math.pi)
        
        # If it falls in the forbidden quadrant, it's a test sample
        if min_ood_angle <= rot_z <= max_ood_angle:
            test_indices.append(idx)
        else:
            standard_indices.append(idx)

    # 3. Create Splits
    np.random.seed(42)
    standard_arr = np.array(standard_indices)
    np.random.shuffle(standard_arr)
    
    # 80/20 split of the standard indices
    val_size = int(len(standard_arr) * 0.2)
    val_idx = standard_arr[:val_size].tolist()
    train_idx = standard_arr[val_size:].tolist()
    
    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_indices,
        "test_by_rule": {
            "rotation_quadrant": test_indices
        }
    }
    
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w") as f:
        json.dump(splits, f, indent=2)
        
    logger.info(f"Saved splits to {split_path}")
    logger.info(f"  Train: {len(train_idx)}")
    logger.info(f"  Val:   {len(val_idx)}")
    logger.info(f"  Test:  {len(test_indices)} (All fall in 270-360 degree quadrant)")

if __name__ == "__main__":
    main()
