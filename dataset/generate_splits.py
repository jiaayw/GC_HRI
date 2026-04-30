#!/usr/bin/env python3
"""Generate a reproducible train/val/test split for the cleaned Bee-Wo dataset."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CLEAN_ROOT = PROJECT_ROOT / "data_clean"
ANNOTATIONS_FILE = DATA_CLEAN_ROOT / "annotations.csv"
SPLITS_FILE = DATA_CLEAN_ROOT / "splits.csv"
REPORT_FILE = DATA_CLEAN_ROOT / "split_report.md"

SEED = 20260331
TRAIN_VIDEOS = 140
VAL_VIDEOS = 30
TEST_VIDEOS = 30
VALID_SPLITS = ("train", "val", "test")


def read_annotations() -> list[dict[str, str]]:
    with ANNOTATIONS_FILE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assign_video_splits(video_ids: list[str]) -> dict[str, str]:
    if len(video_ids) != TRAIN_VIDEOS + VAL_VIDEOS + TEST_VIDEOS:
        raise ValueError(
            f"Expected 200 videos, found {len(video_ids)}."
        )

    shuffled = list(video_ids)
    random.Random(SEED).shuffle(shuffled)

    assignments: dict[str, str] = {}
    for video_id in shuffled[:TRAIN_VIDEOS]:
        assignments[video_id] = "train"
    for video_id in shuffled[TRAIN_VIDEOS : TRAIN_VIDEOS + VAL_VIDEOS]:
        assignments[video_id] = "val"
    for video_id in shuffled[TRAIN_VIDEOS + VAL_VIDEOS :]:
        assignments[video_id] = "test"
    return assignments


def build_split_rows(
    annotations: list[dict[str, str]], video_assignments: dict[str, str]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in annotations:
        split = video_assignments[row["video_id"]]
        rows.append(
            {
                "clip_id": row["clip_id"],
                "video_id": row["video_id"],
                "gesture_label": row["gesture_label"],
                "split": split,
            }
        )
    rows.sort(key=lambda item: item["clip_id"])
    return rows


def write_splits(rows: list[dict[str, str]]) -> None:
    with SPLITS_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["clip_id", "video_id", "gesture_label", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)


def verify(rows: list[dict[str, str]]) -> dict[str, object]:
    clip_ids = [row["clip_id"] for row in rows]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("Duplicate clip_id values found in splits.csv.")

    split_by_video: dict[str, set[str]] = defaultdict(set)
    split_clip_counts: Counter[str] = Counter()
    split_video_ids: dict[str, set[str]] = defaultdict(set)
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        split = row["split"]
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split value: {split}")
        split_by_video[row["video_id"]].add(split)
        split_clip_counts[split] += 1
        split_video_ids[split].add(row["video_id"])
        class_counts[split][row["gesture_label"]] += 1

    leaking_videos = {
        video_id: sorted(splits)
        for video_id, splits in split_by_video.items()
        if len(splits) > 1
    }
    if leaking_videos:
        raise ValueError(f"Video leakage detected: {leaking_videos}")

    expected_video_counts = {"train": 140, "val": 30, "test": 30}
    expected_clip_counts = {"train": 700, "val": 150, "test": 150}

    for split, expected in expected_video_counts.items():
        actual = len(split_video_ids[split])
        if actual != expected:
            raise ValueError(
                f"Unexpected {split} video count: expected {expected}, found {actual}"
            )

    for split, expected in expected_clip_counts.items():
        actual = split_clip_counts[split]
        if actual != expected:
            raise ValueError(
                f"Unexpected {split} clip count: expected {expected}, found {actual}"
            )

    for split, counts in class_counts.items():
        values = set(counts.values())
        if len(values) != 1:
            raise ValueError(f"Class imbalance detected in {split}: {dict(counts)}")

    return {
        "split_clip_counts": dict(split_clip_counts),
        "split_video_counts": {
            split: len(split_video_ids[split]) for split in VALID_SPLITS
        },
        "class_counts": {split: dict(class_counts[split]) for split in VALID_SPLITS},
    }


def write_report(summary: dict[str, object]) -> None:
    split_video_counts = summary["split_video_counts"]
    split_clip_counts = summary["split_clip_counts"]
    class_counts = summary["class_counts"]

    lines = [
        "# Bee-Wo Split Report",
        "",
        f"- Seed: {SEED}",
        "- Split rule: video-level random split",
        "",
        "## Split Sizes",
        "",
        "| Split | Videos | Clips |",
        "| --- | ---: | ---: |",
    ]
    for split in VALID_SPLITS:
        lines.append(
            f"| {split} | {split_video_counts[split]} | {split_clip_counts[split]} |"
        )

    lines.extend(
        [
            "",
            "## Class Counts",
            "",
            "| Split | G01 | G02 | G05 | G06 | G07 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in VALID_SPLITS:
        counts = class_counts[split]
        lines.append(
            f"| {split} | {counts['G01']} | {counts['G02']} | {counts['G05']} | "
            f"{counts['G06']} | {counts['G07']} |"
        )

    lines.extend(
        [
            "",
            "## Integrity",
            "",
            "- No clip duplicates in `splits.csv`.",
            "- No `video_id` appears in more than one split.",
            "- Class counts are balanced within each split.",
        ]
    )

    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    annotations = read_annotations()
    if len(annotations) != 1000:
        raise ValueError(f"Expected 1000 annotation rows, found {len(annotations)}.")

    video_ids = sorted({row["video_id"] for row in annotations})
    assignments = assign_video_splits(video_ids)
    split_rows = build_split_rows(annotations, assignments)
    write_splits(split_rows)
    summary = verify(split_rows)
    write_report(summary)

    print("Created:", SPLITS_FILE)
    print("Created:", REPORT_FILE)
    print("Seed:", SEED)
    print("Video counts:", summary["split_video_counts"])
    print("Clip counts:", summary["split_clip_counts"])


if __name__ == "__main__":
    main()
