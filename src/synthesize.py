"""Synthetic multi-celebrity detection dataset (Milestone 2).

Builds synthetic group images for YOLOv8 by pasting 2+ CelebA face crops
onto shared backgrounds. Source faces come from ``data/celeba/<identity>/``
(the original 178 x 218 aligned CelebA photos) and are drawn only from the
matching Milestone 1 split in ``data/splits.csv``, so no face appears in
more than one of train / val / test.

Output layout (``data/detection/``)
-----------------------------------
    data.yaml                 - YOLO config: split folders and class names
    images/{train,val,test}/  - synthetic images
    labels/{train,val,test}/  - one .txt per image, one row per face:
                                class_id x_center y_center width height
                                (all normalised to 0-1)

Pipeline
--------
1. ``setup_folders``  - create the YOLO folder structure and ``data.yaml``.
2. (to be added)      - trim CelebA photos to the face for tight boxes.
3. (to be added)      - load and resize background photos.
4. (to be added)      - compose one synthetic image from several faces.
5. (to be added)      - write the YOLO label file for that image.
6. (to be added)      - generate the full dataset for every split.

Paths, class names and the split are imported from ``src/preprocessing.py``
so Milestone 1 and Milestone 2 always agree.

Run from the repo root:
    python -m src.synthesize
"""

import yaml

from src.preprocessing import PROJECT_ROOT, class_names_from, load_splits

# --- Settings: paths, class names and splits (shared with Milestone 1) ---
DET_DIR = PROJECT_ROOT / "data" / "detection"
CLASS_NAMES = class_names_from(load_splits())  # ["id_2336", "id_2970", "id_4422", "id_7007"]
SPLITS = ("train", "val", "test")


# --- Part A: create YOLO folders and data.yaml ---
def setup_folders():
    """Create the YOLO folder structure and data.yaml."""
    for kind in ("images", "labels"):
        for split in SPLITS:
            (DET_DIR / kind / split).mkdir(parents=True, exist_ok=True)

    config = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    with open(DET_DIR / "data.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Created YOLO folders and data.yaml in {DET_DIR}")


# --- Run setup when called from the terminal ---
if __name__ == "__main__":
    setup_folders()