"""Integrated celebrity detection and identification pipeline (Milestone 4).

Pipeline
--------
1. ``detect``   - the Milestone 3 YOLOv8n finds the faces (class-agnostic NMS, one box per face).
2. ``classify`` - the Milestone 1 classifier, adapted to face crops (``src/adapt_classifier.py``),
                  names each face.
3. ``predict``  - returns both names per face. The final name comes from the model selected on
                  the validation split (``results/milestone4/pipeline_settings.json``).

Run from the repo root:
    python -m src.pipeline --image path/to/image.jpg
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import models as tvm
from ultralytics import YOLO

from src.detection_eval import CLASS_COLORS, CLASS_NAMES, IMAGE_EXTENSIONS, _font, _label, read_gt
from src.preprocessing import PROJECT_ROOT, get_transforms

# --- Settings: models and defaults (overwritten by pipeline_settings.json) ---
DETECTOR = PROJECT_ROOT / "results" / "milestone3" / "sweep" / "best_model.pt"
CLASSIFIER = PROJECT_ROOT / "checkpoints" / "efficientnet_b0_crops.pt"
SETTINGS = PROJECT_ROOT / "results" / "milestone4" / "pipeline_settings.json"
DEFAULTS = {"det_conf": 0.7, "nms_iou": 0.7, "imgsz": 640, "identity": "detector"}


# --- Classifier ---
def load_classifier(path, device="cpu"):
    """EfficientNet-B0 with the Milestone 1 head (CNN_2), loaded from a checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = tvm.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, len(ckpt["class_names"])))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def classify(model, crops, device="cpu"):
    """Class probabilities for a list of PIL face crops (Milestone 1 test transform)."""
    if not crops:
        return np.zeros((0, len(CLASS_NAMES)))
    tf = get_transforms(train=False)
    batch = torch.stack([tf(c) for c in crops]).to(device)
    return torch.softmax(model(batch), dim=1).cpu().numpy()


# --- Pipeline ---
class CelebrityPipeline:
    def __init__(self, detector=DETECTOR, classifier=CLASSIFIER, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.settings = dict(DEFAULTS)
        if SETTINGS.is_file():
            self.settings.update(json.loads(SETTINGS.read_text()))
        self.detector = YOLO(str(detector))
        self.classifier = load_classifier(classifier, self.device)

    def detect(self, image, conf=None):
        """Face boxes (x1, y1, x2, y2), detector confidences and detector class ids."""
        s = self.settings
        r = self.detector.predict(image, conf=conf or s["det_conf"], iou=s["nms_iou"], imgsz=s["imgsz"],
                                  agnostic_nms=True, device=self.device, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)

    def predict(self, image, conf=None):
        """One dict per face: box, final name and confidence, both models' answers, and whether they agree."""
        image = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image
        boxes, det_conf, det_cls = self.detect(image, conf)
        probs = classify(self.classifier, [image.crop(tuple(b)) for b in boxes], self.device)
        faces = []
        for box, dc, dp, p in zip(boxes, det_cls, det_conf, probs):
            cc = int(p.argmax())
            use_detector = self.settings["identity"] == "detector"
            faces.append(dict(box=[round(float(v)) for v in box],
                              name=CLASS_NAMES[dc if use_detector else cc],
                              confidence=round(float(dp if use_detector else p[cc]), 3),
                              detector=CLASS_NAMES[dc], classifier=CLASS_NAMES[cc],
                              agree=bool(dc == cc)))
        return faces

    def annotate(self, image, faces):
        """Draw each face in its identity colour; '?' marks faces where the two models disagree."""
        image = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.copy()
        draw = ImageDraw.Draw(image)
        font = _font(max(13, image.width // 45))
        for f in faces:
            color = CLASS_COLORS[CLASS_NAMES.index(f["name"])]
            draw.rectangle(f["box"], outline=color, width=3)
            text = f"{f['name']} {f['confidence']:.2f}" + ("" if f["agree"] else " ?")
            _label(draw, f["box"][:2], text, color, font)
        return image


# --- Evaluation: cache a split in the format of detection_eval.predict_split ---
def predict_split(pipe, image_paths, conf_floor=0.05):
    """Boxes, both models' classes and the ground truth for every image (low threshold, filtered later)."""
    records = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        boxes, scores, det_cls = pipe.detect(image, conf_floor)
        probs = classify(pipe.classifier, [image.crop(tuple(b)) for b in boxes], pipe.device)
        gt_classes, gt_boxes, size = read_gt(path)
        records.append(dict(image=Path(path).name, path=str(path), size=size,
                            gt_classes=gt_classes, gt_boxes=gt_boxes, boxes=boxes, scores=scores,
                            detector=det_cls, classifier=probs.argmax(1).astype(int), classes=det_cls))
    return records


def use_identity(records, identity):
    """Copy of the records where ``classes`` holds the detector's or the classifier's answer."""
    return [dict(r, classes=r[identity]) for r in records]


# --- Command line ---
def main():
    parser = argparse.ArgumentParser(description="Detect and identify the celebrities in an image.")
    parser.add_argument("--image", required=True, help="image file or folder of images")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "milestone4" / "predictions"))
    args = parser.parse_args()

    src = Path(args.image)
    paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS) if src.is_dir() else [src]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = CelebrityPipeline()
    for path in paths:
        faces = pipe.predict(path)
        pipe.annotate(path, faces).save(out / f"{path.stem}_pred.jpg")
        print(path.name, [(f["name"], f["confidence"]) + (() if f["agree"] else ("models disagree",)) for f in faces])
    print("Annotated images saved to", out)


if __name__ == "__main__":
    main()
