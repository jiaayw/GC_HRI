#!/usr/bin/env python3
"""Dataset utilities for Bee-Wo gesture classification."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BeeWoSample:
    clip_id: str
    clip_path: Path
    video_id: str
    gesture_label: str
    label_index: int
    split: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_label_maps(label_map_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    rows = _read_csv(label_map_path)
    rows.sort(key=lambda row: int(row["class_id"]))
    label_to_index = {
        row["gesture_label"]: index for index, row in enumerate(rows)
    }
    index_to_label = {index: label for label, index in label_to_index.items()}
    return label_to_index, index_to_label


def sample_frame_paths(frame_paths: list[Path], seq_len: int) -> list[Path]:
    if not frame_paths:
        raise ValueError("Cannot sample an empty clip.")
    positions = np.linspace(0, len(frame_paths) - 1, num=seq_len)
    indices = np.clip(np.rint(positions).astype(int), 0, len(frame_paths) - 1)
    return [frame_paths[index] for index in indices.tolist()]


class BeeWoClipDataset(Dataset):
    """Loads RGB gesture clips and samples them to a fixed sequence length."""

    def __init__(
        self,
        data_root: Path,
        annotations_path: Path,
        splits_path: Path,
        label_map_path: Path,
        split: str,
        seq_len: int = 16,
        image_size: int = 96,
        use_landmarks: bool = False,
        landmarks_root: Path | None = None,
        allow_missing_landmarks: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.annotations_path = Path(annotations_path)
        self.splits_path = Path(splits_path)
        self.label_map_path = Path(label_map_path)
        self.split = split
        self.seq_len = seq_len
        self.image_size = image_size
        self.use_landmarks = use_landmarks
        self.landmarks_root = Path(landmarks_root) if landmarks_root is not None else self.data_root / "landmarks"
        self.allow_missing_landmarks = allow_missing_landmarks

        self.label_to_index, self.index_to_label = build_label_maps(self.label_map_path)
        annotations = {row["clip_id"]: row for row in _read_csv(self.annotations_path)}
        split_rows = _read_csv(self.splits_path)

        self.samples: list[BeeWoSample] = []
        for split_row in split_rows:
            if split_row["split"] != split:
                continue
            annotation = annotations[split_row["clip_id"]]
            self.samples.append(
                BeeWoSample(
                    clip_id=annotation["clip_id"],
                    clip_path=self.data_root / annotation["clip_path"],
                    video_id=annotation["video_id"],
                    gesture_label=annotation["gesture_label"],
                    label_index=self.label_to_index[annotation["gesture_label"]],
                    split=split,
                )
            )

        if not self.samples:
            raise ValueError(f"No samples found for split '{split}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame(self, frame_path: Path) -> torch.Tensor:
        image = Image.open(frame_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        frame_paths = sorted(sample.clip_path.glob("*.jpg"))
        sampled_frame_paths = sample_frame_paths(frame_paths, self.seq_len)
        frames = torch.stack([self._load_frame(frame_path) for frame_path in sampled_frame_paths])
        item = {
            "frames": frames,
            "label": torch.tensor(sample.label_index, dtype=torch.long),
            "clip_id": sample.clip_id,
            "video_id": sample.video_id,
            "gesture_label": sample.gesture_label,
        }
        if self.use_landmarks:
            landmark_path = self.landmarks_root / f"{sample.clip_id}.npy"
            if landmark_path.exists():
                landmarks = np.load(landmark_path)
            elif self.allow_missing_landmarks:
                landmarks = np.zeros((self.seq_len, 63), dtype=np.float32)
            else:
                raise FileNotFoundError(
                    f"Missing landmark file for clip_id={sample.clip_id}: {landmark_path}"
                )

            if landmarks.shape != (self.seq_len, 63):
                raise ValueError(
                    f"Unexpected landmark shape for {sample.clip_id}: {landmarks.shape}, "
                    f"expected {(self.seq_len, 63)}"
                )
            item["landmarks"] = torch.from_numpy(landmarks.astype(np.float32))
        return item
