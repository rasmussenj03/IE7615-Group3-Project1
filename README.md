# IE7615-Group3-Project1

Project 1 repository for Group 3 of IE7615: a discriminative computer-vision pipeline for celebrity identification and
detection, built on a four-identity subset of [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).

| Milestone | Goal | Where |
|---|---|---|
| 1 | Classify single face crops: preprocessing pipeline, custom CNN vs. pretrained models | [`milestone1/`](milestone1) |
| 2 | Build a synthetic multi-celebrity detection dataset with YOLO labels, augmentation and a leak-free split | [`milestone2/Data_Preparation.ipynb`](milestone2/Data_Preparation.ipynb) |
| 3 | Fine-tune YOLOv8 on the Milestone 2 dataset (next) | `data/detection/data.yaml` |

## Repository layout

```
data/
  celeba/<identity>/*.jpg          99 CelebA aligned & cropped faces (178 x 218), the raw input
  splits.csv                       per-identity train / val / test split of those photos (used by both milestones)
  backgrounds/{train,val,test}/    30 room photos for the synthetic images, sources and licenses in SOURCES.md
  detection/                       Milestone 2 YOLO dataset: images/, labels/, data.yaml
milestone1/
  Preprocessing_Pipeline.ipynb     subset, validation, split, transforms, DataLoaders
  CNN_1.ipynb                      architecture 1: custom CNN from scratch
  Transfer_Learning_1.ipynb        architecture 2: ResNet-18 fine-tuned (transfer-learning baseline)
  Transfer_Learning_2.ipynb        ResNet-18 fine-tuned with the training protocol shared with CNN_1 / CNN_2
  CNN_2.ipynb                      architecture 3: EfficientNet-B0 fine-tuned
milestone2/
  Data_Preparation.ipynb           Task 1 composition + labels, Task 2 augmentation, Task 3 verification
src/
  preprocessing.py                 Milestone 1 data indexing, split, transforms and DataLoaders (paths shared with M2)
  synthesize.py                    Milestone 2 face trimming, image composition and YOLO label writing
results/
  milestone1/                      training logs, metrics and figures of the classifiers
  milestone2/                      figures of the dataset construction (verification reports are written to verification/)
```

## Setup

**Google Colab (recommended).** Put the repo in Google Drive at `MyDrive/IE7615-Group3-Project1`, or let the first cell
of any notebook clone it (this needs a GitHub personal access token saved as a Colab secret named `GITHUB_TOKEN`). Open
a notebook, select a GPU runtime for the Milestone 1 training notebooks (CPU is enough for Milestone 2) and run all
cells. The first cell mounts Drive and changes into the repo.

**Local.**

```bash
pip install -r requirements.txt
jupyter lab
```

Every notebook starts with the same setup cell, which finds the repo root, so notebooks can be run from any folder.

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

Per-notebook logs and figures are in `results/milestone1/` (`custom_cnn/`, `resnet18/`; `CNN_2` and
`Transfer_Learning_2` write theirs when run). Trained weights are not committed (`*.pt` is git-ignored) and are
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

Task 3 audits the saved files without modifying them and writes its reports to `results/milestone2/verification/`.

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

## Data sources

- CelebA: Z. Liu, P. Luo, X. Wang, X. Tang, "Deep Learning Face Attributes in the Wild", ICCV 2015, via
  [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).
- Room backgrounds: Openverse, CC0 1.0; see [`data/backgrounds/SOURCES.md`](data/backgrounds/SOURCES.md).
