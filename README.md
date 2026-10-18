# IE7615-Group3-Project1

Project 1 repository for Group 3 of IE7615: a discriminative computer-vision pipeline that **detects and identifies
celebrities** in an image, built on a four-identity subset of
[CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).

| Milestone | Goal | Where |
|---|---|---|
| 1 | Classify single face crops: preprocessing pipeline, custom CNN vs. pretrained models | [`milestone1/`](milestone1) |
| 2 | Build a synthetic multi-celebrity detection dataset with YOLO labels, augmentation and a leak-free split | [`milestone2/`](milestone2) |
| 3 | Fine-tune YOLOv8n on that dataset, evaluate it, sweep hyperparameters, gallery and live demo | [`milestone3/`](milestone3) |
| 4 | Integrate detector and classifier into one end-to-end pipeline, final evaluation, gallery and demo | [`milestone4/`](milestone4), [`src/pipeline.py`](src/pipeline.py) |

## Repository layout

```
requirements.txt                   Python packages (see Setup)
data/
  celeba/<identity>/*.jpg          99 CelebA aligned & cropped faces (178 x 218), the raw input
  splits.csv                       per-identity train / val / test split of those photos (used by every milestone)
  backgrounds/{train,val,test}/    30 room photos for the synthetic images, sources and licenses in SOURCES.md
  detection/                       Milestone 2 YOLO dataset: images/, labels/, data.yaml
src/
  preprocessing.py                 Milestone 1 data indexing, split, transforms and DataLoaders (paths shared by all)
  synthesize.py                    Milestone 2 face trimming, image composition and YOLO label writing
  detection_eval.py                Milestone 3 matching protocol, metrics, error analysis and drawing
  pipeline.py                      Milestone 4 integrated pipeline (detector + classifier), also a command-line tool
  adapt_classifier.py              Milestone 4 fine-tuning of the Milestone 1 classifier on face crops
milestone1/                        Preprocessing_Pipeline, CNN_1, Transfer_Learning_1, Transfer_Learning_2, CNN_2
milestone2/Data_Preparation.ipynb  composition + labels, augmentation, verification
milestone3/                        YOLOv8 training/evaluation, hyperparameter sweep, detection gallery, live demo
milestone4/
  Integrated_Pipeline.ipynb        classifier adaptation, settings chosen on val, test results
  Final_Gallery.ipynb              final results gallery
  Live_Demo.ipynb                  live demo of the integrated pipeline
checkpoints/
  efficientnet_b0_ft.pt            Milestone 1 classifier (CNN_2), 16 MB
  efficientnet_b0_crops.pt         the same model fine-tuned on face crops (Milestone 4), used by the pipeline, 16 MB
results/
  milestone1/ milestone2/          logs, metrics and figures of each milestone
  milestone3/sweep/best_model.pt   YOLOv8n detector selected by the sweep (Milestone 3), used by the pipeline, 6 MB
  milestone4/                      pipeline_settings.json, result tables (CSV) and gallery/
```

## Setup

**Python.** 3.9 or newer (tested on 3.9 and 3.10). Everything runs on a CPU; an NVIDIA GPU is used automatically

```bash
git clone https://github.com/rasmussenj03/IE7615-Group3-Project1.git
cd IE7615-Group3-Project1
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip && pip install -r requirements.txt
```

## Reproducing the reported numbers

Run the notebooks from top to bottom in this order. Each step reads only what the previous steps saved, and every
committed result can be regenerated.

| Step | Notebook / command | Produces | Approx. time (CPU) |
|---|---|---|---|
| 1 | `milestone1/Preprocessing_Pipeline.ipynb` | `data/splits.csv` | < 1 min |
| 2 | `milestone1/CNN_1`, `Transfer_Learning_1`, `Transfer_Learning_2`, `CNN_2` | `results/milestone1/*`, `checkpoints/*.pt` | GPU recommended |
| 3 | `milestone2/Data_Preparation.ipynb` | `data/detection/`, `results/milestone2/` | few min |
| 4 | `milestone3/Milestone3_YOLOv8_Training_Evaluation.ipynb` | `results/milestone3/<run>/` | hours on CPU, GPU recommended |
| 5 | `milestone3/Hyperparameter_Sweep.ipynb` | `results/milestone3/sweep/` incl. `best_model.pt` | hours on CPU, GPU recommended |
| 6 | `milestone3/Detection_Gallery.ipynb` | `results/milestone3/gallery/` | few min |
| 7 | `milestone4/Integrated_Pipeline.ipynb` | `checkpoints/efficientnet_b0_crops.pt`, `results/milestone4/` | ~5 min (+ 5-15 min to train the classifier) |
| 8 | `milestone4/Final_Gallery.ipynb` | `results/milestone4/gallery/` | ~2 min |

To check the final results only, steps 8 and 9 are enough: they use the committed checkpoints.

---

## Milestone 1: Single-face classification

### Celebrity subset

CelebA publishes identities as anonymous integer IDs, so classes are named `id_<CelebA ID>`. The four identities have
24-25 images each, close to the most CelebA provides per identity, so the classes are balanced.

| Class | Images | Train | Val | Test |
|---|---|---|---|---|
| `id_2336` | 25 | 15 | 5 | 5 |
| `id_2970` | 25 | 15 | 5 | 5 |
| `id_4422` | 25 | 15 | 5 | 5 |
| `id_7007` | 24 | 14 | 5 | 5 |
| **Total** | **99** | **59** | **20** | **20** |

### Preprocessing (`Preprocessing_Pipeline.ipynb`, `src/preprocessing.py`)

- **Split:** per-identity 60 / 20 / 20 split with a fixed seed (42), saved to `data/splits.csv` and reused by every
  notebook, including Milestone 2. Images are checked for corruption and exact (MD5) duplicates across splits.
- **Resizing and normalisation:** short side resized to 224 px, centre crop to 224 x 224, ImageNet mean/std.
- **Augmentation (train only):** random resized crop (70-100% of the image), horizontal flip, rotation up to ±10°,
  colour jitter (brightness, contrast, saturation 0.3, hue 0.05).

### Results

Held-out test split (20 images, 5 per identity), mean ± sd over 3 seeds, as reported in each notebook.

| Model | Notebook | Test accuracy | Macro-F1 | Parameters |
|---|---|---|---|---|
| Custom CNN, from scratch | `CNN_1` | 38.3% ± 7.6 | 0.33 | 0.98 M |
| ResNet-18, fine-tuned (20 epochs, best validation epoch) | `Transfer_Learning_1` | **83.3% ± 2.9** | **0.83** | 11.2 M |
| ResNet-18, fine-tuned (shared protocol) | `Transfer_Learning_2` | 78.3% ± 5.8 | 0.77 | 11.2 M |
| EfficientNet-B0, fine-tuned | `CNN_2` | 80.0% ± 8.7 | 0.79 | 4.0 M |

- ImageNet pretraining is what makes 14-15 training images per identity enough: every pretrained model is about twice
  as accurate as the custom CNN, which under-fits and varies strongly between seeds.
- The three pretrained models are within about one standard deviation of each other; with 20 test images, one image is
  5 percentage points.
- `id_2970` and `id_7007` are the hardest identities. The most common error is `id_2970` predicted as `id_4422`, the
  deliberately chosen look-alike pair (similar age and hair colour).
- Both ResNet-18 notebooks fine-tune all layers (backbone LR 1e-4, head LR 1e-3, AdamW). `Transfer_Learning_2` uses the
  protocol shared with `CNN_1` and `CNN_2` (cosine LR schedule, label smoothing 0.1, early stopping), so those three
  are directly comparable.

Logs and figures of `CNN_1` and `Transfer_Learning_1` are in `results/milestone1/` (`custom_cnn/`, `resnet18/`); the
results of `Transfer_Learning_2` and `CNN_2` are shown in the saved notebook outputs. Of the trained weights only
`checkpoints/efficientnet_b0_ft.pt` (`CNN_2`) is committed, because Milestone 4 builds on it; the others are
recreated by the notebooks.

---

## Milestone 2: Detection dataset construction

CelebA only provides single-face photos, so we create synthetic "group photos" by pasting face crops of the four
celebrities onto photos of empty rooms, label every face with a YOLO bounding box, add augmented copies, and verify the
result. The whole pipeline is in [`milestone2/Data_Preparation.ipynb`](milestone2/Data_Preparation.ipynb).

### Dataset

**Location:** `data/detection/` (YOLO config: `data/detection/data.yaml`), about 67 MB, committed to the repo.

```
data/detection/
  data.yaml                      train/val/test folders and class names
  images/{train,val,test}/       <split>_<n>.jpg (original), <split>_aug_<n>.jpg (augmented copy)
  labels/{train,val,test}/       one .txt per image: class_id x_center y_center width height (normalised 0-1)
```

| | Train | Val | Test | Total |
|---|---|---|---|---|
| Original images (640 x 640) | 300 | 100 | 100 | 500 |
| Augmented images (450 x 450) | 300 | 100 | 100 | 500 |
| **Images** | **600** | **200** | **200** | **1,000** |
| Faces `id_2336` | 416 | 131 | 142 | 689 |
| Faces `id_2970` | 411 | 145 | 142 | 698 |
| Faces `id_4422` | 414 | 140 | 129 | 683 |
| Faces `id_7007` | 415 | 148 | 130 | 693 |
| **Face boxes** | **1,656** | **564** | **543** | **2,763** |

Class IDs match Milestone 1: 0 = `id_2336`, 1 = `id_2970`, 2 = `id_4422`, 3 = `id_7007`.

### Split (60 / 20 / 20)

The split follows the Milestone 1 split of the source photos, so the detector is tested on photos the Milestone 1
classifier never trained on, and every identity keeps 5 test photos for meaningful per-class results.

- **Faces:** each synthetic image draws its faces only from the CelebA photos of its own split (59 / 20 / 20 photos).
- **Backgrounds:** 30 room photos split 18 / 6 / 6, so validation and test images use rooms never seen in training.
- **Augmented copies** stay in the split of their original, so no scene appears in two splits.

Each split has 300 / 100 / 100 composed scenes; with one augmented copy each, the image counts are 600 / 200 / 200.

### Task 1: Composition and labels

- **Face trimming.** Each CelebA photo is trimmed to the face region (`FACE_BOX`, 102 x 125 px), so the pasted patch is
  the face and its bounding box is tight.
- **Backgrounds.** 30 openly licensed photos of empty rooms (classrooms, offices, conference rooms, libraries, living
  rooms) from [Openverse](https://openverse.org), all CC0 1.0, resized to 640 x 640. None contains a
  person, because an unlabelled face would teach the detector that faces are background. Close-ups, ceilings and unusual
  angles were excluded. Sources and licenses: [`data/backgrounds/SOURCES.md`](data/backgrounds/SOURCES.md).
- **Composition.** Each image combines 2-4 different celebrities on one background. Each face gets a random height of
  90-220 px (scale) and a brightness factor of 0.7-1.3 (lighting), is placed at head height within the middle 60% of the
  image without overlapping another face, and is pasted with a soft oval blend so it does not look like a rectangular patch.
- **Labels.** One label file per image with one line per face, written from the exact paste position of each face.

### Task 2: Augmentation

One augmented copy of every image, made with [Albumentations](https://albumentations.ai), which transforms the YOLO
boxes together with the image. The pipeline is seeded (`seed=137`).

| Transform | Parameters | Why |
|---|---|---|
| `RandomCrop` | 450 x 450 px, always | larger, off-centre and partly visible faces, as in a differently framed photo |
| `HorizontalFlip` | p = 0.5 | faces are roughly left-right symmetric, so a mirrored face is the same person |
| `CoarseDropout` | 3-8 black holes of 5-15% of the image size, p = 0.5 | occlusion (hands, microphones, other people) |
| `ToGray` | p = 0.2 | the detector should not rely on colour alone |
| `RandomBrightnessContrast` | ±20%, p = 0.2 | lighting differences between rooms and cameras |
| `Affine` | scale 0.8-1.2, translate ±10%, rotate ±15°, shear ±5°, p = 0.5 | distance to the camera and head tilt; small angles keep axis-aligned boxes valid |
| `min_visibility` | 0.3 | a box is dropped when less than 30% of it remains inside the crop |

Augmented images are 450 x 450 and originals 640 x 640; YOLOv8 resizes both to its input size.

### Task 3: Verification

Task 3 audits the saved files without modifying them; its results are shown in the notebook output.

| Check | Result |
|---|---|
| Image / label pairs, label format (5 columns, class 0-3, values in 0-1, positive size, box inside the image) | 1,000 pairs, 2,763 boxes, 0 errors |
| Source photos shared between splits (file path and SHA-256 of the cropped face pixels) | 0 for train/val, train/test and val/test |
| Every identity in every split (source photos and saved labels) | yes |
| Visual review of annotated originals and augmented copies | boxes on the intended faces, class IDs consistent |

**Known properties of the augmented copies.** Faces can be cut by the crop border (kept as long as 30% is visible) or
partly covered by dropout holes; rotated faces get a slightly loose box, because Albumentations replaces a rotated box
with the upright box that encloses it; and some augmented images keep only one face after cropping. Validation and test
contain augmented copies as well, so Milestone 3 metrics can be reported for all images and separately for originals
(`<split>_<n>`) and augmented copies (`<split>_aug_<n>`).

### Reproducing the dataset

Run `milestone2/Data_Preparation.ipynb` from top to bottom. Task 1 (Part F) deletes and rewrites all of
`data/detection/`, including the augmented copies, so Task 2 must run after it; both are seeded and reproduce the same
labels. JPEG pixels can differ very slightly between library versions. Task 3 only reads the dataset and can be run on
its own.

---

## Milestone 3: YOLOv8 Transfer Learning and Evaluation

### Task 1: Fine-tune YOLOv8 Model

We used Ultralytics' COCO-pretrained YOLOv8 model, and fine-tuned it to work with our saved dataset from Milestone 2. A training budget of 150 epochs was used, with early stopping at 30 epochs. Before testing, the batch size and thresholds were set. The batch size was set to 16, confidence was set to 0.25, class-aware NMS IoU was set at 0.70, and TP matching IoU was set at 0.50. Note that the NMS threshold used is different from the evaluation matching threshold.

### Task 2: Detection Metrics on Held-Out Test Split

For this task, we produced loss curves, per-class and overall AP/mAP, precision, recall and bounding-box IoU, plus CSV reports and checkpoints.

To get the specific detection metrics for these fine-tuned models, held-out data was fed into the model for the test split, after the training protocol and checkpoint were finished. The AP integrates the precision–recall curve, using a low prediction confidence floor of 0.001 to retain the curve, not the fixed reporting threshold. Per-class values are **AP**; their average across the four identities is **mAP**. The test split is not used for checkpoint selection or threshold tuning.

Ultralytics also returns precision and recall at its internally selected curve operating point. Those are saved separately for reference. The main combined report below uses predeclared confidence 0.25 instead, making its precision/recall and matched IoU directly interpretable together. 

| Test split (200 images, 543 faces) | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Mean matched IoU |
|---|---|---|---|---|---|
| All identities | 0.983 | 0.941 | 0.944 | 0.958 | 0.964 |

Per-identity AP@0.5 ranges from 0.974 (`id_7007`) to 0.990 (`id_4422`). The loss curves and per-class tables are
in the saved outputs of `Milestone3_YOLOv8_Training_Evaluation.ipynb`; running it again writes them to
`results/milestone3/<run>/`.

### Task 3: Hyperparameter Sweep

The hyperparameter sweep covers five of the hyperparameters named in the assignment, in two parts:

| Part | Hyperparameters | Needs retraining | Selected on |
|---|---|---|---|
| A. Training sweep | learning rate, image size, augmentation intensity | yes, 1 run per configuration | validation mAP@0.5:0.95 |
| B. Inference sweep | confidence threshold, NMS IoU threshold (class-aware and class-agnostic NMS) | no, the selected model is re-run on cached predictions | validation F1 |

For the training sweep, each run had a different parameter changed, for a total of 7 runs. Additionally, each training was over the course of 30 epochs, with an early stop at 10 epochs. This is noticeably less than the epochs used in Task 1, but as the cosine schedule is stretched, the learning rate still decays fully. Ultimately, the configuration with the highest **validation**, mAP@0.5:0.95, was selected.

The results of the training sweep can be found in the `results/milestone3/sweep/` folder, with the parameters for the best model of the sweep stored in `best_model.pt`. This is also what was used for Tasks 4 and 5.

For the inference sweep, the model parameters stayed the same, but which boxes were ultimately kept were explored. Two different thresholds were explored: confidence and NMS IoU, in addition to a comparison of class-aware or class-agnostic NMS. 

On the test split the selected model reaches precision 0.965, recall 0.919 and mean matched IoU 0.971 at that operating point (confidence 0.80, NMS IoU 0.70, class-aware NMS).

### Task 4: Detection Visualization Gallery

For the detection visualization gallery, the predictions of the fine-tuned YOLOv8 model are rendered, for the held-out **test** split.

* **Model and thresholds:** the checkpoint and operating point selected on the validation split by the hyperparameter sweep (`results/milestone3/sweep/best_model.pt` and `operating_point.json`, Task 3).
* **Error analysis:** every test prediction and every ground-truth face is labelled with the Task 2 matching protocol (same identity, IoU >= 0.5) and an error reason, see `src/detection_eval.py`:

| Outcome | Meaning |
|---|---|
| correct | box on the face with the right identity (IoU >= 0.5) |
| wrong identity | box on a face (IoU >= 0.5, or >= 0.3 if the identity is wrong) but with another person's name |
| duplicate | second box on a face that was already found |
| poor localization | right identity, but the box overlaps the face with IoU < 0.5 |
| background | box on no face at all |
| missed face | no box anywhere near a labelled face |

The gallery outputs can be found in the `results/milestone3/gallery/` folder. There is one annotated JPEG per example, with a `README.md` serving as the gallery with captions.

### Task 5: Live Demo

The live demo runs the fine-tuned YOLOv8n celebrity detector on held-out test images or on an image you provide.

* **Model and thresholds:** the checkpoint and operating point selected on the validation split by the
  hyperparameter sweep (`results/milestone3/sweep/best_model.pt` and `operating_point.json`, Task 3).
* The notebook loads the model, then opens a Gradio app: pick a test image, upload or drag and drop an image, or take a webcam photo, and change the confidence threshold with a slider. Run the last cell to stop the app.

Each face is boxed in its identity color and labelled with the predicted identity and confidence.

### Reproducing the reported metrics

To reproduce the reported metrics, please run Task 1 and 2's `milestone3/Milestone3_YOLOv8_Training_Evaluation.ipynb` from top to bottom. You should see the detection metrics under **7. Compare identities and save a concise run report**. The trained YOLOv8 weights can be reproduced with this script as well. Then, please run `milestone3/Hyperparameter_Sweep.ipynb` to reproduce the hyperparameter sweep of Task 3, and `milestone3/Detection_Gallery.ipynb` to see the visualized detection gallery of Task 4. Lastly, to run the demo, please refer to the live demo located at `milestone3/Live_Demo.ipynb`. This should be run from top to bottom, excluding the last cell. The second-to-last cell runs the app locally, and here you will be able to see the model identify faces, from the image of your choice, in real-time. Once you have finished using the app, please run the very last cell, to close it. 

The weights of the model selected by the sweep are in `results/milestone3/sweep/best_model.pt`.

---

## Milestone 4: Integrated pipeline

### The pipeline (`src/pipeline.py`)

1. **Detect:** the Milestone 3 YOLOv8n (`best_model.pt`) finds the faces, one box per face (class-agnostic NMS).
2. **Classify:** each box is cut out and named by the Milestone 1 EfficientNet-B0 (`CNN_2`), fine-tuned on face crops.
3. **Combine:** both models name every face. The final name comes from the model chosen on the validation split;
   faces where the two models disagree are marked.

Run it on any image with `python -m src.pipeline --image <file or folder>`, or use the demo
(`milestone4/Live_Demo.ipynb`).

### Part A: Classifier on face crops

The Milestone 1 classifier was trained on whole CelebA photos. On the faces the detector boxes in the synthetic
scenes (pasted, rescaled, augmented) it is much less accurate, so `src/adapt_classifier.py` fine-tunes it on the
face crops of the detection **training** split for 5 epochs, with the Milestone 1 settings (AdamW, backbone LR 1e-4,
head LR 1e-3, label smoothing 0.1, same augmentation), keeping the best validation epoch.

| Labelled face crops | Before | After |
|---|---|---|
| Validation (564 faces) | 65.6% | 96.1% |
| Test (543 faces) | 61.1% | 86.4% |

The largest remaining confusion is `id_7007` predicted as `id_4422`, the same look-alike error as the detector.

### Part B: Settings chosen on the validation split

Detector thresholds from 0.25 to 0.90 and both identity sources (detector or classifier) were scored with the
Milestone 3 matching (same identity, IoU >= 0.5). Best F1: **detector confidence 0.70, final name from the detector**
(F1 0.969, against 0.953 when the classifier names the faces). Saved in `results/milestone4/pipeline_settings.json`.

### Part C: Test results

Held-out test split, 200 images, 543 labelled faces.

| System | Precision | Recall | F1 | Wrong-name boxes | Missed faces |
|---|---|---|---|---|---|
| Milestone 3 detector only (conf 0.80) | 0.965 | 0.919 | 0.942 | 16 | 28 |
| **Pipeline, detector names faces (selected)** | 0.946 | 0.930 | 0.938 | 21 | 17 |
| Pipeline, classifier names faces | 0.865 | 0.851 | 0.858 | 64 | 17 |

| Test faces | Share | Final name correct |
|---|---|---|
| Both models agree | 88.2% | 97.8% |
| Models disagree | 11.8% | 82.3% |

- The detector names faces better than the separate classifier, so validation chose it. The pipeline keeps the
  detector's F1 and finds more faces (17 missed instead of 28).
- The classifier works as a second opinion: when both models agree the name is right 98% of the time, and the faces
  where they disagree are flagged in the demo.

Tables: `results/milestone4/` (`classifier_on_crops.csv`, `val_sweep.csv`, `test_results.csv`, `test_agreement.csv`).

### Final results gallery

[`results/milestone4/gallery/README.md`](results/milestone4/gallery/README.md), built by `Final_Gallery.ipynb`:
24 correctly classified single faces, 8 misclassified single faces with a short analysis each, and 12 multi-face
test images with the pipeline's boxes and names.

### Live demo

`milestone4/Live_Demo.ipynb` works like the Milestone 3 demo, with the integrated pipeline: pick a test image,
upload or drag and drop an image, or take a webcam photo, and change the detector confidence with a slider. The
table shows each face's final name, both models' answers and whether they agree. Run the last cell to stop the app.

### Limitations

- Four identities only, and a closed set: every detected face is given one of the four names, including people who
  are none of them.
- All detection data is synthetic (faces pasted on empty rooms). Real group photos have different lighting, poses,
  occlusion and image statistics; performance on them is not measured.
- Each identity has only 24-25 source photos (5 test photos each), so test faces are re-used across scenes and
  small differences between systems are within noise.

---

## Data sources

- CelebA: Z. Liu, P. Luo, X. Wang, X. Tang, "Deep Learning Face Attributes in the Wild", ICCV 2015, via
  [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).
- Room backgrounds: Openverse, CC0 1.0; see [`data/backgrounds/SOURCES.md`](data/backgrounds/SOURCES.md).
