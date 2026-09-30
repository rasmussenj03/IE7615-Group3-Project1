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
3. ``compose_image``  - paste 2-4 different celebrities onto a background
                        from the same split, with random size, brightness and
                        position (at head height, no overlapping faces), using
                        a soft oval blend.
4. (to be added)      - write the YOLO label file for that image.
5. (to be added)      - generate the full dataset for every split.

Paths, class names and the split are imported from ``src/preprocessing.py``
so Milestone 1 and Milestone 2 always agree.

Run from the repo root:
    python -m src.synthesize
"""

import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from src.preprocessing import IMAGE_DIR, PROJECT_ROOT, class_names_from, load_splits

# --- Settings: paths, class names and splits (shared with Milestone 1) ---
DET_DIR = PROJECT_ROOT / "data" / "detection"
BG_DIR = PROJECT_ROOT / "data" / "backgrounds"
RESULTS_DIR = PROJECT_ROOT / "results" / "milestone2"
FACES = load_splits()                  # path, identity, label, split for all 99 photos
CLASS_NAMES = class_names_from(FACES)  # ["id_2336", "id_2970", "id_4422", "id_7007"]
SPLITS = ("train", "val", "test")

# --- Face trim region (left, top, right, bottom) in the 178x218 CelebA photo ---
FACE_BOX = (38, 60, 140, 185)

# --- Composition settings ---
IMG_SIZE = 640            # backgrounds are 640 x 640
FACES_PER_IMAGE = (2, 4)  # min / max different celebrities per image
FACE_HEIGHT = (90, 220)   # pasted face height range in pixels (scale variation)
BRIGHTNESS = (0.7, 1.3)   # brightness factor range (lighting variation)
HEAD_BAND = (0.2, 0.8)    # faces stay within this vertical band (fraction of height)
MAX_TRIES = 50            # attempts to find a non-overlapping spot for a face


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
    left, top, right, bottom = FACE_BOX
    fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class,
                             figsize=(n_per_class * 2, len(CLASS_NAMES) * 2.4))

    for row, (label, group) in enumerate(FACES.groupby("label")):
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


# --- Part D: compose one synthetic image ---
def oval_mask(size):
    """Oval paste mask with a soft edge: keeps the face, fades out the photo's corners."""
    w, h = size
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).ellipse([0, 0, w - 1, h - 1], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(max(2, w // 25)))


def overlaps(box, boxes):
    """True if box (x1, y1, x2, y2) overlaps any box in boxes."""
    x1, y1, x2, y2 = box
    return any(x1 < b[2] and b[0] < x2 and y1 < b[3] and b[1] < y2 for b in boxes)


def compose_image(split, rng):
    """Paste 2-4 different celebrities onto one background from the same split.

    Returns the image and a list of (label, x1, y1, x2, y2) pixel boxes.
    """
    pool = FACES[FACES["split"] == split]
    bg_path = rng.choice(sorted((BG_DIR / split).glob("bg_*.jpg")))
    image = Image.open(bg_path).convert("RGB")

    n_faces = rng.randint(*FACES_PER_IMAGE)
    labels = rng.sample(sorted(pool["label"].unique().tolist()), n_faces)
    boxes = []
    for label in labels:
        # random photo of this celebrity, random brightness and starting size
        path = rng.choice(pool.loc[pool["label"] == label, "path"].tolist())
        face = load_face(path)
        face = ImageEnhance.Brightness(face).enhance(rng.uniform(*BRIGHTNESS))
        h = rng.randint(*FACE_HEIGHT)

        # random position at head height that doesn't overlap an earlier face;
        # if the image is crowded, shrink the face 15% every 10 failed tries
        top, bottom = int(HEAD_BAND[0] * IMG_SIZE), int(HEAD_BAND[1] * IMG_SIZE)
        for attempt in range(MAX_TRIES):
            if attempt % 10 == 0:
                if attempt:
                    h = round(h * 0.85)
                w = round(h * face.width / face.height)
                patch = face.resize((w, h), Image.LANCZOS)
            x, y = rng.randint(0, IMG_SIZE - w), rng.randint(top, bottom - h)
            box = (x, y, x + w, y + h)
            if not overlaps(box, [b[1:] for b in boxes]):
                image.paste(patch, (x, y), oval_mask((w, h)))
                boxes.append((label, *box))
                break
    return image, boxes


# --- Check: draw boxes on a few composed images ---
def preview_composites(split="train", n=4, seed=0):
    """Save a row of composed images with each face's box and name drawn on."""
    rng = random.Random(seed)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.2, 3.4))

    for ax in axes:
        image, boxes = compose_image(split, rng)
        ax.imshow(image)
        for label, x1, y1, x2, y2 in boxes:
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor="lime", linewidth=1.5))
            text_y = y1 - 4 if y1 > 15 else y1 + 14  # keep names inside the image
            ax.text(x1, text_y, CLASS_NAMES[label], color="lime", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"composite_check_{split}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {out_path}")


# --- Run setup when called from the terminal ---
if __name__ == "__main__":
    setup_folders()