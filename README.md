# GC in HRI: Multimodal Hand Gesture Recognition for Robot Control

The project studies hand gesture recognition for robot control using two input types:
- RGB video clips
- MediaPipe hand landmarks

The official analysis in the report compares four experiment tracks:
- `resnet10_3d`: RGB-only baseline
- `mediapipe_logreg`: landmark-only baseline
- `mediapipe_random_forest`: landmark-only baseline
- `resnet10_landmark_fusion`: proposed multimodal fusion model

Other models in the repository are extra experiments and are not part of the core report comparison.

## Project Overview

We use a five-class subset of the IPN Hand dataset and map each gesture to a robot command:

| Gesture label | Robot command |
| --- | --- |
| `G01` | `stop` |
| `G02` | `move_back` |
| `G05` | `move_left` |
| `G06` | `move_right` |
| `G07` | `move_forward` |

The final dataset used in this project contains:
- `1000` clips total
- `200` source videos
- `700 / 150 / 150` train/validation/test clips

For the report configuration:
- RGB clips are uniformly sampled to `16` frames
- RGB frames are resized to `64x64`
- landmark clips are stored as `16 x 63` arrays
- the fusion model uses late fusion between an RGB branch and a landmark branch

## Data Links

Original dataset link used for preprocessing:
- <https://drive.google.com/file/d/1GDZ9gIqat7ROGJkiy379xwdJax0lJo66/view?usp=share_link>

Precomputed landmark folder:
- <https://drive.google.com/drive/folders/1fcSs42j1rsvJyMZ_mj4QK2K3IKlPbm6J?usp=share_link>

Because the dataset artifacts are large, they are referenced externally rather than committed to GitHub.

## Environment Setup

Create and activate a Python environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- `torch` and `torchvision` may need platform-specific install commands on some systems.
- The live demo also requires webcam access.
- Some MediaPipe installations use the Tasks API and require `checkpoints/hand_landmarker.task`.

## Expected Repository Layout

After setup, the important project paths are:

```text
GC_HRI/
├── baseline_models/
├── dataset/
├── project_models/
├── scripts/
├── checkpoints/
├── data_clean/                  # generated or restored locally
├── raw data/                    # downloaded IPN subset, local only
├── runs/                        # training outputs
└── live_robot_demo.py
```

The main entry points are:
- preprocessing: `dataset/clean_ipn_dataset.py`, `dataset/generate_splits.py`, `scripts/extract_mediapipe_landmarks.py`
- training: `baseline_models/train_baseline.py`, `baseline_models/train_mediapipe_baseline.py`
- demo: `live_robot_demo.py`
- model definitions: `baseline_models/`, `project_models/`

## Data Setup

### Option 1: Full Pipeline from Raw Data

Use this path if you want to reproduce the full analysis starting from the raw IPN subset.

1. Download the raw dataset from the Google Drive file above.
2. Extract or copy it into the repository under `raw data/`.
3. Make sure the annotation files live under `raw data/annotations/`.
4. Make sure the frame archives or extracted frame folders live under `raw data/frame/`.

The preprocessing scripts expect this layout:

```text
raw data/
├── annotations/
│   ├── Annot_List.txt
│   ├── classIdx.txt
│   └── metadata.csv
└── frame/
    ├── ... extracted video frame folders ...
    └── ... or .tgz archives containing frames ...
```

### Option 2: Fast Path with Precomputed Landmarks

Use this path if you want a faster reproduction path for the landmark baselines and the fusion model.

1. Download the landmark folder from the Google Drive folder above.
2. Place the `.npy` files under `data_clean/landmarks/`.
3. Make sure `data_clean/annotations.csv`, `data_clean/splits.csv`, and `data_clean/label_map.csv` also exist locally.

If you only have the landmark folder and not the metadata CSV files, run the raw-data preprocessing steps first to generate the CSV files, then copy the downloaded landmarks into `data_clean/landmarks/`.

## Full Pipeline Commands

Run these commands from the repository root.

### 1. Clean the raw IPN subset into clip folders

```bash
python3 -m dataset.clean_ipn_dataset
```

This generates `data_clean/` and writes:
- `data_clean/annotations.csv`
- `data_clean/label_map.csv`
- `data_clean/cleaning_summary.csv`
- `data_clean/cleaning_report.md`

### 2. Generate reproducible train/val/test splits

```bash
python3 -m dataset.generate_splits
```

This creates:
- `data_clean/splits.csv`
- `data_clean/split_report.md`

The split is video-level and uses seed `20260331`.

### 3. Extract MediaPipe landmarks

```bash
python3 scripts/extract_mediapipe_landmarks.py \
  --data-root data_clean \
  --annotations data_clean/annotations.csv \
  --output-root data_clean/landmarks \
  --seq-len 16
```

This writes:
- per-clip landmark arrays to `data_clean/landmarks/*.npy`
- `data_clean/landmarks/landmarks_manifest.csv`

## Official Training Commands

All commands below match the report configuration by explicitly using `16` frames and `64x64` RGB resolution.

### RGB-only baseline: ResNet10-3D

```bash
python3 -m baseline_models.train_baseline \
  --model resnet10_3d \
  --data-root data_clean \
  --annotations data_clean/annotations.csv \
  --splits data_clean/splits.csv \
  --label-map data_clean/label_map.csv \
  --seq-len 16 \
  --image-size 64 \
  --output-root runs
```

### Landmark-only baseline: Logistic Regression

```bash
python3 -m baseline_models.train_mediapipe_baseline \
  --classifier logreg \
  --annotations data_clean/annotations.csv \
  --splits data_clean/splits.csv \
  --label-map data_clean/label_map.csv \
  --landmarks-root data_clean/landmarks \
  --seq-len 16 \
  --output-root runs
```

### Landmark-only baseline: Random Forest

```bash
python3 -m baseline_models.train_mediapipe_baseline \
  --classifier random_forest \
  --annotations data_clean/annotations.csv \
  --splits data_clean/splits.csv \
  --label-map data_clean/label_map.csv \
  --landmarks-root data_clean/landmarks \
  --seq-len 16 \
  --output-root runs
```

### Proposed model: ResNet10-3D + Landmark Fusion

```bash
python3 -m baseline_models.train_baseline \
  --model resnet10_landmark_fusion \
  --data-root data_clean \
  --annotations data_clean/annotations.csv \
  --splits data_clean/splits.csv \
  --label-map data_clean/label_map.csv \
  --landmarks-root data_clean/landmarks \
  --seq-len 16 \
  --image-size 64 \
  --output-root runs
```

## Fast Path Commands

If `data_clean/annotations.csv`, `data_clean/splits.csv`, and `data_clean/label_map.csv` already exist and you have downloaded `data_clean/landmarks/`, you can skip landmark extraction and run:

```bash
python3 -m baseline_models.train_mediapipe_baseline --classifier logreg --landmarks-root data_clean/landmarks
python3 -m baseline_models.train_mediapipe_baseline --classifier random_forest --landmarks-root data_clean/landmarks
python3 -m baseline_models.train_baseline --model resnet10_landmark_fusion --landmarks-root data_clean/landmarks --seq-len 16 --image-size 64
```

## Output Artifacts

Training runs are written under `runs/`.

Typical outputs include:
- `best_model.pt` or `model.joblib`
- `config.json`
- `history.json`
- `metrics.json`
- `label_map.json`
- `val_confusion_matrix.csv`
- `test_confusion_matrix.csv`

The live demo can also use:
- `checkpoints/best_model.pt`
- `checkpoints/hand_landmarker.task`

## Live Robot Demo

The live demo performs webcam inference with the fusion model.

Example:

```bash
python3 live_robot_demo.py \
  --checkpoint checkpoints/best_model.pt \
  --label-map data_clean/label_map.csv \
  --seq-len 16 \
  --image-size 64
```

If your MediaPipe installation requires the task asset explicitly, add:

```bash
python3 live_robot_demo.py \
  --checkpoint checkpoints/best_model.pt \
  --label-map data_clean/label_map.csv \
  --hand-landmarker-task checkpoints/hand_landmarker.task \
  --seq-len 16 \
  --image-size 64
```

Press `q` to quit the webcam window.

## Reproducibility Notes

- This repository is intended to reproduce the project workflow and comparison setup, not guarantee identical metrics across every machine.
- Landmark-only experiments still depend on the shared environment because the repository packages import PyTorch-backed modules.
- For the report-faithful setup, prefer the explicit commands in this README rather than relying on training-script defaults.
