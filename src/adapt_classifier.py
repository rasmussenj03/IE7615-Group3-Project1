"""Adapt the Milestone 1 classifier to detector face crops (Milestone 4).

The Milestone 1 EfficientNet-B0 (CNN_2) was trained on whole CelebA photos. In the pipeline it sees
the faces the detector boxes in the synthetic scenes, which look different (pasted, rescaled,
augmented). This module continues its training on those face crops.

Pipeline
--------
1. ``collect_crops`` - cut every labelled face out of a detection split (same leak-free split).
2. ``train``         - fine-tune the Milestone 1 model on the training crops with the Milestone 1
                       settings and save the epoch with the best validation accuracy.

Run from the repo root:
    python -m src.adapt_classifier
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from src.detection_eval import read_gt, split_images
from src.pipeline import CLASSIFIER, classify, load_classifier
from src.preprocessing import PROJECT_ROOT, get_transforms

# --- Settings (Milestone 1 protocol) ---
BASE = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_ft.pt"  # Milestone 1 CNN_2 model
HISTORY = PROJECT_ROOT / "results" / "milestone4" / "classifier_adaptation.csv"
EPOCHS, BATCH_SIZE, SEED = 5, 32, 0
BACKBONE_LR, HEAD_LR, WEIGHT_DECAY, LABEL_SMOOTHING = 1e-4, 1e-3, 1e-4, 0.1


# --- Face crops ---
def collect_crops(split):
    """All labelled faces of a detection split: PIL crops and a table with label and image name."""
    crops, rows = [], []
    for path in split_images(split):
        image = Image.open(path).convert("RGB")
        classes, boxes, _ = read_gt(path)
        for c, b in zip(classes, boxes):
            crops.append(image.crop(tuple(b)))
            rows.append(dict(image=path.name, label=int(c), augmented="_aug_" in path.name,
                             x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
    return crops, pd.DataFrame(rows)


def accuracy(model, crops, table, device="cpu"):
    probs = np.concatenate([classify(model, crops[i:i + 64], device) for i in range(0, len(crops), 64)])
    return (probs.argmax(1) == table["label"]).mean(), probs


# --- Training ---
def train(device="cpu"):
    torch.manual_seed(SEED)
    rng = np.random.RandomState(SEED)
    train_crops, train_table = collect_crops("train")
    val_crops, val_table = collect_crops("val")
    model = load_classifier(BASE, device)
    head = [p for n, p in model.named_parameters() if n.startswith("classifier.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    optimizer = torch.optim.AdamW([{"params": body, "lr": BACKBONE_LR}, {"params": head, "lr": HEAD_LR}],
                                  weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    tf = get_transforms(train=True)
    labels = train_table["label"].to_numpy()
    class_names = torch.load(BASE, map_location="cpu", weights_only=False)["class_names"]

    history, best = [], 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order, losses = rng.permutation(len(labels)), []
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            x = torch.stack([tf(train_crops[j]) for j in idx]).to(device)
            y = torch.as_tensor(labels[idx]).to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        val_acc, _ = accuracy(model, val_crops, val_table, device)
        history.append(dict(epoch=epoch, train_loss=np.mean(losses), val_accuracy=val_acc))
        print(f"epoch {epoch}: train loss {np.mean(losses):.3f}, val accuracy {val_acc:.3f}")
        if val_acc > best:  # keep the best validation epoch
            best = val_acc
            torch.save({"model_name": "efficientnet_b0_crops", "state_dict": model.state_dict(),
                        "class_names": class_names, "epoch": epoch, "val_accuracy": val_acc}, CLASSIFIER)

    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(HISTORY, index=False)
    print(f"Best validation accuracy {best:.3f}, saved {CLASSIFIER}")
    return pd.DataFrame(history)


if __name__ == "__main__":
    train("cuda" if torch.cuda.is_available() else "cpu")
