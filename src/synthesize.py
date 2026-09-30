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
2. ``load_face``      - trim a CelebA photo to ``FACE_BOX`` so each pasted
                        face gets a tight bounding box.
3. (to be added)      - load and resize background photos.
4. (to be added)      - compose one synthetic image from several faces.
5. (to be added)      - write the YOLO label file for that image.
6. (to be added)      - generate the full dataset for every split.

Paths, class names and the split are imported from ``src/preprocessing.py``
so Milestone 1 and Milestone 2 always agree.

Run from the repo root:
    python -m src.synthesize
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml
from PIL import Image

from src.preprocessing import IMAGE_DIR, PROJECT_ROOT, class_names_from, load_splits

# --- Settings: paths, class names and splits (shared with Milestone 1) ---
DET_DIR = PROJECT_ROOT / "data" / "detection"
RESULTS_DIR = PROJECT_ROOT / "results" / "milestone2"
CLASS_NAMES = class_names_from(load_splits())  # ["id_2336", "id_2970", "id_4422", "id_7007"]
SPLITS = ("train", "val", "test")

# --- Face trim region (left, top, right, bottom) in the 178x218 CelebA photo ---
FACE_BOX = (38, 60, 140, 185)


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


# --- Part B: trim CelebA photos to the face ---
def load_face(path):
    """Open one CelebA photo and trim it to FACE_BOX."""
    img = Image.open(IMAGE_DIR / path).convert("RGB")
    return img.crop(FACE_BOX)

# --- Check: draw FACE_BOX on sample photos to confirm it fits the faces ---
def preview_face_box(n_per_class=4):
    """Save a grid of CelebA photos with FACE_BOX drawn on each."""
    splits = load_splits()
    left, top, right, bottom = FACE_BOX
    fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class,
                             figsize=(n_per_class * 2, len(CLASS_NAMES) * 2.4))

    for row, (label, group) in enumerate(splits.groupby("label")):
        for col, path in enumerate(group["path"].sample(n_per_class, random_state=0)):
            ax = axes[row, col]
            ax.imshow(Image.open(IMAGE_DIR / path))
            ax.add_patch(patches.Rectangle((left, top), right - left, bottom - top,
                                           fill=False, edgecolor="lime", linewidth=2))
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[label])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "face_box_check.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")


# --- Run setup when called from the terminal ---
if __name__ == "__main__":
    setup_folders()