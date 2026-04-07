#!/usr/bin/env python3
"""Quick inspection utility for the Bee-Wo training dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bee_wo_dataset import BeeWoClipDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one Bee-Wo dataset split.")
    parser.add_argument("--data-root", type=Path, default=Path("data_clean"))
    parser.add_argument("--annotations", type=Path, default=Path("data_clean/annotations.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data_clean/splits.csv"))
    parser.add_argument("--label-map", type=Path, default=Path("data_clean/label_map.csv"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = BeeWoClipDataset(
        data_root=args.data_root,
        annotations_path=args.annotations,
        splits_path=args.splits,
        label_map_path=args.label_map,
        split=args.split,
        seq_len=args.seq_len,
        image_size=args.image_size,
    )
    sample = dataset[args.index]
    print(
        json.dumps(
            {
                "split": args.split,
                "dataset_size": len(dataset),
                "clip_id": sample["clip_id"],
                "video_id": sample["video_id"],
                "gesture_label": sample["gesture_label"],
                "frames_shape": list(sample["frames"].shape),
                "label_index": int(sample["label"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
