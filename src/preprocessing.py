"""Preprocessing pipeline for the CelebA celebrity subset (Milestone 1).

The raw subset lives in ``data/celeba/<celeba_identity_id>/<celeba_file>.jpg``.
Images are CelebA *aligned & cropped* faces (178 x 218 px, RGB JPEG).

Pipeline
--------
1. ``scan_images``        - index every image and its identity label.
2. ``validate_images``    - check files open, are RGB, record size, and find
                            exact duplicates (MD5) that could leak across splits.
3. ``make_splits``        - per-identity (stratified) train / val / test split
                            with a fixed seed; saved to ``data/splits.csv``.
4. ``get_transforms``     - resize + normalisation (all splits) and data
                            augmentation (train split only).
5. ``get_dataloaders``    - PyTorch ``DataLoader`` objects for each split.
                            Images are cached in memory, which keeps epochs
                            fast even when the repo is read from Google Drive.


"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "data" / "celeba"
SPLITS_CSV = PROJECT_ROOT / "data" / "splits.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2

# Every model sees the same 224 x 224 input so results are comparable.
# ImageNet statistics are used for normalisation because some of the
# architectures are ImageNet-pretrained; the custom CNN uses the same values
# for consistency (the subset's channel means match ImageNet to within 0.03;
# see the preprocessing notebook).
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Indexing, validation and splitting
# --------------------------------------------------------------------------- #
def scan_images(image_dir: Path | str = IMAGE_DIR) -> pd.DataFrame:
    """Return one row per image: relative path, CelebA identity and class label.

    Each sub-folder of ``image_dir`` is one identity. Labels are assigned in
    sorted identity order.
    """
    image_dir = Path(image_dir)
    identities = sorted(p.name for p in image_dir.iterdir() if p.is_dir())
    if not identities:
        raise FileNotFoundError(f"No identity folders found in {image_dir}")

    rows = []
    for label, identity in enumerate(identities):
        for f in sorted((image_dir / identity).iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({
                    "path": f.relative_to(image_dir).as_posix(),
                    "identity": identity,
                    "label": label,
                })
    return pd.DataFrame(rows)


def class_names_from(df: pd.DataFrame) -> list[str]:
    """Class names ordered by label index"""
    ids = df.drop_duplicates("label").sort_values("label")["identity"]
    return [f"id_{i}" for i in ids]


def validate_images(df: pd.DataFrame, image_dir: Path | str = IMAGE_DIR) -> pd.DataFrame:
    """Open every image and add ``width``, ``height``, ``mode``, ``md5``, ``ok`` columns."""
    image_dir = Path(image_dir)
    out = df.copy()
    widths, heights, modes, hashes, oks = [], [], [], [], []
    for rel in out["path"]:
        fp = image_dir / rel
        try:
            with Image.open(fp) as im:
                im.verify()
            with Image.open(fp) as im:
                widths.append(im.width)
                heights.append(im.height)
                modes.append(im.mode)
            hashes.append(hashlib.md5(fp.read_bytes()).hexdigest())
            oks.append(True)
        except Exception:  # corrupt / unreadable file
            widths.append(None)
            heights.append(None)
            modes.append(None)
            hashes.append(None)
            oks.append(False)
    out["width"], out["height"], out["mode"] = widths, heights, modes
    out["md5"], out["ok"] = hashes, oks
    return out


def make_splits(
    df: pd.DataFrame,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = SEED,
) -> pd.DataFrame:
    """Add a ``split`` column using a per-identity random split.

    Each identity is shuffled independently with a fixed seed and cut into
    train / val / test, so every identity is represented in every split with
    (almost) the same proportions.
    """
    rng = random.Random(seed)
    parts = []
    for _, group in df.groupby("label", sort=True):
        idx = list(group.index)
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, round(n * test_fraction))
        n_val = max(1, round(n * val_fraction))
        split = (["test"] * n_test) + (["val"] * n_val) + (["train"] * (n - n_test - n_val))
        parts.append(pd.Series(split, index=idx))
    out = df.copy()
    out["split"] = pd.concat(parts)
    return out


def split_summary(splits: pd.DataFrame) -> pd.DataFrame:
    """Image counts per identity x split (with totals)."""
    table = pd.crosstab(splits["identity"], splits["split"], margins=True, margins_name="total")
    return table[["train", "val", "test", "total"]]


def save_splits(splits: pd.DataFrame, path: Path | str = SPLITS_CSV) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits[["path", "identity", "label", "split"]].to_csv(path, index=False)
    return path


def load_splits(path: Path | str = SPLITS_CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"identity": str})


def subsample_train(splits: pd.DataFrame, per_class: int, seed: int = SEED) -> pd.DataFrame:
    """Keep only ``per_class`` training images per identity (val/test unchanged).

    Used by the data-efficiency study to measure how much training data each
    architecture needs.
    """
    train = splits[splits["split"] == "train"]
    kept = pd.concat(
        g.sample(n=min(per_class, len(g)), random_state=seed) for _, g in train.groupby("label")
    )
    return pd.concat([kept, splits[splits["split"] != "train"]])


def compute_channel_stats(splits: pd.DataFrame, image_dir: Path | str = IMAGE_DIR):
    """Per-channel mean / std of the *training* images (pixel values in [0, 1])."""
    image_dir = Path(image_dir)
    pixels = []
    for rel in splits.loc[splits["split"] == "train", "path"]:
        with Image.open(image_dir / rel) as im:
            pixels.append(np.asarray(im.convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255.0)
    pixels = np.concatenate(pixels)
    return pixels.mean(axis=0), pixels.std(axis=0)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def get_transforms(train: bool, image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Image transforms.

    Eval (val / test): resize the short side to ``image_size`` (178x218 ->
    224x274) and centre-crop to ``image_size`` x ``image_size``. The aligned
    faces are centred, so this keeps the whole face and trims background.

    Train: same resize, then a random crop covering 70-100% of the image,
    horizontal flip, small rotation and colour jitter. These mimic the pose,
    framing and lighting variation between CelebA photos without changing
    identity.
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation (for plotting)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)


# --------------------------------------------------------------------------- #
# Dataset / DataLoaders
# --------------------------------------------------------------------------- #
class CelebSubsetDataset(Dataset):
    """Images for one split. All images are decoded once and kept in memory."""

    def __init__(self, frame: pd.DataFrame, image_dir: Path | str = IMAGE_DIR, transform=None):
        image_dir = Path(image_dir)
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.transform = transform
        self.images = []
        for rel in self.paths:
            with Image.open(image_dir / rel) as im:
                self.images.append(im.convert("RGB"))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]


def get_dataloaders(
    splits: pd.DataFrame | None = None,
    image_dir: Path | str = IMAGE_DIR,
    batch_size: int = 16,
    image_size: int = IMAGE_SIZE,
    num_workers: int = 0,
    seed: int = SEED,
) -> tuple[dict[str, DataLoader], list[str]]:
    """Build ``{'train', 'val', 'test'}`` DataLoaders and return them with class names.

    ``splits`` defaults to the saved ``data/splits.csv`` so every notebook uses
    the identical split. Only the training loader shuffles / augments.
    """
    if splits is None:
        splits = load_splits()
    generator = torch.Generator().manual_seed(seed)
    loaders = {}
    for name in ("train", "val", "test"):
        is_train = name == "train"
        ds = CelebSubsetDataset(
            splits[splits["split"] == name],
            image_dir=image_dir,
            transform=get_transforms(train=is_train, image_size=image_size),
        )
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            generator=generator if is_train else None,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders, class_names_from(splits)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", default=str(IMAGE_DIR))
    parser.add_argument("--out", default=str(SPLITS_CSV))
    parser.add_argument("--val", type=float, default=VAL_FRACTION)
    parser.add_argument("--test", type=float, default=TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    df = validate_images(scan_images(args.image_dir), args.image_dir)
    bad = df[~df["ok"]]
    if len(bad):
        print(f"Dropping {len(bad)} unreadable image(s):\n{bad['path'].tolist()}")
    df = df[df["ok"]]
    dups = df[df.duplicated("md5", keep=False)]
    if len(dups):
        print(f"Warning: {len(dups)} exact-duplicate images found; keeping first copy.")
        df = df.drop_duplicates("md5")

    splits = make_splits(df.reset_index(drop=True), args.val, args.test, args.seed)
    out = save_splits(splits, args.out)
    print(split_summary(splits))
    print(f"Saved splits to {out}")


if __name__ == "__main__":
    main()
