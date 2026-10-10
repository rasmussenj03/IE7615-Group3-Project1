"""Detection evaluation helpers for Milestone 3 (hyperparameter sweep and gallery).

The matching protocol is the one used in
``Milestone3_YOLOv8_Training_Evaluation.ipynb``, so precision, recall and IoU
from the sweep and the gallery are directly comparable with the main report:

* a prediction is a true positive when it overlaps an unmatched ground-truth
  face **of the same identity** with IoU >= ``MATCH_IOU`` (0.5);
* predictions are matched in descending confidence order, each to the
  highest-IoU candidate, and every box can be matched only once;
* wrong identities and duplicates are false positives, unmatched faces are
  false negatives.

On top of the TP / FP / FN counts, ``analyze_image`` sorts every error into a
reason (wrong identity, duplicate, poor localization, background, missed face),
which the gallery uses for selecting and captioning examples.

Prediction caching
------------------
``predict_split`` runs the model once per NMS setting with a very low
confidence floor and keeps every surviving box. Greedy NMS only lets a box be
suppressed by a higher-scoring box, so filtering these cached boxes by a
confidence threshold afterwards gives exactly the boxes the model would return
when called with that threshold. The confidence sweep therefore needs no
further forward passes.
"""

from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "detection"
CLASS_NAMES = ["id_2336", "id_2970", "id_4422", "id_7007"]
MATCH_IOU = 0.50
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Okabe-Ito colours (colour-blind safe), one per identity, used in every figure.
CLASS_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]
ERROR_COLOR = "#FF2A2A"


# ---------------------------------------------------------------- dataset ---
def write_runtime_yaml(out_dir):
    """Write a copy of data.yaml with absolute paths and return its path.

    ``data/detection/data.yaml`` uses relative folders without a ``path:`` key,
    which Ultralytics would resolve against its own datasets directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"path": str(DATA_DIR),
              **{s: str(DATA_DIR / "images" / s) for s in ("train", "val", "test")},
              "names": dict(enumerate(CLASS_NAMES))}
    path = out_dir / "dataset_absolute.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def split_images(split):
    folder = DATA_DIR / "images" / split
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_gt(image_path):
    """Ground truth of one image as (classes, xyxy pixel boxes, (w, h))."""
    image_path = Path(image_path)
    label = image_path.parent.parent.parent / "labels" / image_path.parent.name / (image_path.stem + ".txt")
    with Image.open(image_path) as im:
        w, h = im.size
    rows = [line.split() for line in label.read_text().strip().splitlines() if line.strip()]
    values = np.array(rows, dtype=float).reshape(-1, 5)
    centers, sizes = values[:, 1:3] * [w, h], values[:, 3:5] * [w, h]
    boxes = np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1)
    return values[:, 0].astype(int), boxes, (w, h)


# --------------------------------------------------------------- matching ---
def box_iou(a, b):
    """Pairwise IoU between xyxy boxes, shape (len(a), len(b))."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(hi - lo, 0, None).prod(axis=2)
    area_a = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(axis=1)
    area_b = np.clip(b[:, 2:] - b[:, :2], 0, None).prod(axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def match_image(gt_boxes, gt_classes, pred_boxes, pred_classes, scores, match_iou=MATCH_IOU):
    """Greedy one-to-one, class-aware matching. Returns [(pred_idx, gt_idx, iou)]."""
    used, matches = set(), []
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return matches
    ious = box_iou(pred_boxes, gt_boxes)
    for pi in np.argsort(-np.asarray(scores), kind="stable"):
        candidates = [gi for gi, c in enumerate(gt_classes) if c == pred_classes[pi] and gi not in used]
        if not candidates:
            continue
        gi = candidates[int(np.argmax(ious[pi, candidates]))]
        if ious[pi, gi] >= match_iou:
            used.add(gi)
            matches.append((int(pi), int(gi), float(ious[pi, gi])))
    return matches


def analyze_image(rec, conf, match_iou=MATCH_IOU):
    """Label every prediction and every ground-truth face of one image.

    Prediction outcomes: ``TP``, ``wrong_identity`` (overlaps a face of another
    identity with IoU >= match_iou), ``duplicate`` (second box on an already
    found face), ``poor_localization`` (right identity, 0.1 <= IoU < match_iou),
    ``background`` (no face nearby).
    Ground-truth outcomes: ``found``, ``confused`` (only boxed with the wrong
    identity), ``poorly_localized``, ``missed`` (no box nearby at all).
    """
    keep = rec["scores"] >= conf
    boxes, classes, scores = rec["boxes"][keep], rec["classes"][keep], rec["scores"][keep]
    gt_cls, gt_boxes = rec["gt_classes"], rec["gt_boxes"]
    matches = match_image(gt_boxes, gt_cls, boxes, classes, scores, match_iou)
    ious = box_iou(boxes, gt_boxes) if len(boxes) and len(gt_boxes) else np.zeros((len(boxes), len(gt_boxes)))
    pred_out = ["background"] * len(boxes)
    pred_gt = [None] * len(boxes)
    gt_out = ["missed"] * len(gt_boxes)
    gt_iou = [0.0] * len(gt_boxes)
    for pi, gi, v in matches:
        pred_out[pi], pred_gt[pi], gt_out[gi], gt_iou[gi] = "TP", gi, "found", v
    for pi in range(len(boxes)):
        if pred_out[pi] == "TP" or ious.shape[1] == 0:
            continue
        gi = int(np.argmax(ious[pi]))
        best = ious[pi, gi]
        pred_gt[pi] = gi
        if best >= match_iou and gt_cls[gi] != classes[pi]:
            pred_out[pi] = "wrong_identity"
        elif best >= match_iou:
            pred_out[pi] = "duplicate"
        elif best >= 0.1 and gt_cls[gi] == classes[pi]:
            pred_out[pi] = "poor_localization"
        elif best >= 0.1:
            pred_out[pi] = "wrong_identity" if best >= 0.3 else "background"
        else:
            pred_gt[pi] = None
    for gi in range(len(gt_boxes)):
        if gt_out[gi] == "found":
            continue
        near = [pi for pi in range(len(boxes)) if ious[pi, gi] >= 0.1]
        if any(pred_out[pi] == "wrong_identity" and pred_gt[pi] == gi for pi in near):
            gt_out[gi] = "confused"
        elif near:
            gt_out[gi] = "poorly_localized"
        if near:
            gt_iou[gi] = float(ious[near, gi].max())
    return dict(boxes=boxes, classes=classes, scores=scores, pred_outcome=pred_out, pred_gt=pred_gt,
                gt_outcome=gt_out, gt_iou=gt_iou, matches=matches)


# ------------------------------------------------------------- prediction ---
def predict_split(model, image_paths, nms_iou=0.7, agnostic_nms=False, imgsz=640, device=None,
                  conf_floor=0.001, max_det=100, batch=8):
    """Run the model once and cache predictions + ground truth for every image."""
    records = []
    results = model.predict(source=[str(p) for p in image_paths], stream=True, imgsz=imgsz,
                            conf=conf_floor, iou=nms_iou, agnostic_nms=agnostic_nms, max_det=max_det,
                            device=device, batch=batch, verbose=False, save=False)
    for path, result in zip(image_paths, results):
        assert Path(result.path).name == Path(path).name
        gt_classes, gt_boxes, size = read_gt(path)
        records.append(dict(image=Path(path).name, path=str(path), size=size,
                            gt_classes=gt_classes, gt_boxes=gt_boxes,
                            boxes=result.boxes.xyxy.cpu().numpy(),
                            classes=result.boxes.cls.cpu().numpy().astype(int),
                            scores=result.boxes.conf.cpu().numpy()))
    return records


def evaluate(records, conf, match_iou=MATCH_IOU):
    """Pooled and per-class TP / FP / FN, precision, recall, F1 and matched IoU."""
    nc = len(CLASS_NAMES)
    tp, fp, fn = np.zeros(nc, int), np.zeros(nc, int), np.zeros(nc, int)
    ious = [[] for _ in range(nc)]
    errors = dict(wrong_identity=0, duplicate=0, poor_localization=0, background=0, missed=0)
    for rec in records:
        a = analyze_image(rec, conf, match_iou)
        for c, out in zip(a["classes"], a["pred_outcome"]):
            if out == "TP":
                tp[c] += 1
            else:
                fp[c] += 1
                errors[out] += 1
        for c, out in zip(rec["gt_classes"], a["gt_outcome"]):
            if out != "found":
                fn[c] += 1
                errors["missed"] += out == "missed"
        for pi, gi, v in a["matches"]:
            ious[rec["gt_classes"][gi]].append(v)

    def summary(tp_, fp_, fn_, iou_list):
        p = tp_ / max(tp_ + fp_, 1)
        r = tp_ / max(tp_ + fn_, 1)
        return dict(TP=int(tp_), FP=int(fp_), FN=int(fn_), precision=p, recall=r,
                    F1=2 * p * r / max(p + r, 1e-12),
                    mean_matched_IoU=float(np.mean(iou_list)) if iou_list else np.nan)

    overall = summary(tp.sum(), fp.sum(), fn.sum(), [v for c in ious for v in c]) | errors
    per_class = {CLASS_NAMES[c]: summary(tp[c], fp[c], fn[c], ious[c]) for c in range(nc)}
    return overall, per_class


# ---------------------------------------------------------------- drawing ---
def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("arialbd.ttf", size)
        except OSError:
            return ImageFont.load_default(size=size)


def _label(draw, xy, text, fill, font, below=False):
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    w, h = r - l + 8, b - t + 6
    width, height = draw.im.size
    x = min(max(x, 0), width - w)  # keep the label inside the image
    y = min(max(y if below else y - h, 0), height - h)
    draw.rectangle([x, y, x + w, y + h], fill=fill)
    draw.text((x + 4, y + 3 - t), text, fill="black" if fill != ERROR_COLOR else "white", font=font)


def _dashed_rect(draw, box, fill, width=2, dash=8):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]:
        length = max(abs(bx - ax), abs(by - ay))
        steps = max(int(length // dash), 1)
        for s in range(0, steps, 2):
            t0, t1 = s / steps, min((s + 1) / steps, 1)
            draw.line([ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1],
                      fill=fill, width=width)


def render(rec, analysis, out_width=640):
    """Draw predictions (solid, identity colour, label + confidence) and ground truth.

    Correct faces show only the prediction. Errors are marked in red: a wrong or
    extra prediction gets an "x" and the true identity, a face the model did not
    find gets a dashed red box labelled MISSED (or the true identity when it was
    boxed under another name).
    """
    im = Image.open(rec["path"]).convert("RGB")
    scale = out_width / im.width
    im = im.resize((out_width, round(im.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    font, small = _font(15), _font(13)
    gt_boxes = rec["gt_boxes"] * scale
    for gi, out in enumerate(analysis["gt_outcome"]):
        box = gt_boxes[gi]
        if out == "found":
            continue
        _dashed_rect(draw, box, ERROR_COLOR, width=3)
        name = CLASS_NAMES[rec["gt_classes"][gi]]
        tag = {"missed": f"MISSED {name}", "confused": f"true: {name}",
               "poorly_localized": f"loose box: {name}"}[out]
        _label(draw, (box[0], box[3] + 2), tag, ERROR_COLOR, small, below=True)
    for box, c, s, out in zip(analysis["boxes"] * scale, analysis["classes"], analysis["scores"],
                              analysis["pred_outcome"]):
        color = CLASS_COLORS[c]
        draw.rectangle(list(box), outline=color, width=3)
        text = f"{CLASS_NAMES[c]} {s:.2f}"
        if out != "TP":
            text += {"wrong_identity": " x", "duplicate": " x dup", "poor_localization": " x loc",
                     "background": " x bg"}[out]
            draw.rectangle(list(box), outline=ERROR_COLOR, width=1)
        _label(draw, (box[0], box[1]), text, color, font)
    return im
