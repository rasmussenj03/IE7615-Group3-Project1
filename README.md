# IE7615-Group3-Project1
Project #1 repository for Group 3 of IE7615.

This repository homes three different image-recognition models: two custom CNN models - one based on ImageNet EfficientNet-B0, and one based on ResNet-18, as well as a Transfer-Learning model. Training each of these models is the same set of images, pulled from [Kaggle's CelebA dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset), which are pre-processed in [Preprocessing_Pipeline.ipynb](Preprocessing_Pipeline.ipynb). 

To use this repository, please download it and run the notebooks in Jupyter Notebook. The images used can be modified as needed in the pre-processing file, and the models can be re-run to re-train them, and get updated results.


## Milestone 2 — Detection Dataset Construction

Milestone 2 builds the dataset for the detection half of the project. Since CelebA only provides single-face photos, we create synthetic "group photos" by pasting face crops of our four celebrities onto photos of empty rooms. Each face gets a YOLO-format bounding box, and the finished dataset is used to fine-tune YOLOv8 in Milestone 3.

**Faces.** Each CelebA photo is trimmed to the face region before pasting, so every bounding box fits the face tightly. Faces are drawn only from the matching Milestone 1 split, so no face appears in more than one of train, validation, and test.

**Backgrounds.** The backgrounds are 30 openly licensed photos of empty rooms (classrooms, offices, conference rooms, libraries, living rooms) from [Openverse](https://openverse.org), limited to CC0, public-domain, and CC BY licenses.
- No background contains a person. A face in the background without a bounding box would teach the detector that faces are background.
- Close-ups, ceilings, and unusual angles were excluded, so each background looks like a plausible setting for a group photo.
- Photos are resized to 640 × 640 and split 18 / 6 / 6 (train / val / test), so test images use rooms the model never saw in training.
- Sources and licenses are listed in `data/backgrounds/SOURCES.md`.

**Composition.** Each synthetic image combines 2–4 different celebrities on one background, with the background and every face drawn from the same split. To make the images resemble real group photos:
- Each face is resized to a random height between 90 and 220 pixels, and its brightness varies between 0.7× and 1.3×.
- Faces are placed at head height, within the middle 60% of the image, and never overlap.
- Faces are pasted with a soft oval blend, so they sit naturally in the room instead of appearing as rectangular patches.
- A fixed random seed makes the dataset fully reproducible.

