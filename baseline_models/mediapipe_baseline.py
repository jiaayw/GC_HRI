#!/usr/bin/env python3
"""MediaPipe landmark-only baseline utilities for Bee-Wo."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_label_maps(label_map_path: Path) -> tuple[dict[str, int], dict[int, str], dict[str, str]]:
    rows = read_csv(label_map_path)
    rows.sort(key=lambda row: int(row["class_id"]))
    label_to_index = {row["gesture_label"]: index for index, row in enumerate(rows)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    label_to_command = {row["gesture_label"]: row["command"] for row in rows}
    return label_to_index, index_to_label, label_to_command


def load_split_features(
    split: str,
    annotations: dict[str, dict[str, str]],
    split_rows: list[dict[str, str]],
    label_to_index: dict[str, int],
    landmarks_root: Path,
    seq_len: int,
    allow_missing_landmarks: bool,
    max_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    clip_ids: list[str] = []

    for row in split_rows:
        if row["split"] != split:
            continue
        clip_id = row["clip_id"]
        annotation = annotations[clip_id]
        landmark_path = landmarks_root / f"{clip_id}.npy"
        if landmark_path.exists():
            landmarks = np.load(landmark_path)
        elif allow_missing_landmarks:
            landmarks = np.zeros((seq_len, 63), dtype=np.float32)
        else:
            raise FileNotFoundError(f"Missing landmark file for clip_id={clip_id}: {landmark_path}")

        if landmarks.shape != (seq_len, 63):
            raise ValueError(
                f"Unexpected landmark shape for {clip_id}: {landmarks.shape}, expected {(seq_len, 63)}"
            )

        features.append(landmarks.reshape(-1).astype(np.float32))
        labels.append(label_to_index[annotation["gesture_label"]])
        clip_ids.append(clip_id)

        if max_samples and len(features) >= max_samples:
            break

    return np.stack(features), np.asarray(labels, dtype=np.int64), clip_ids


def build_classifier(name: str) -> Pipeline | RandomForestClassifier:
    if name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        solver="lbfgs",
                        n_jobs=None,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=20260331,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported classifier: {name}")
