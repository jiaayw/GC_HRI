#!/usr/bin/env python3
"""Train a MediaPipe-landmarks-only baseline for Bee-Wo."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from baseline_models import (
    build_mediapipe_classifier,
    build_mediapipe_label_maps,
    load_mediapipe_split_features,
)
from baseline_models.mediapipe_baseline import read_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MediaPipe baseline on landmark features.")
    parser.add_argument("--annotations", type=Path, default=Path("data_clean/annotations.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data_clean/splits.csv"))
    parser.add_argument("--label-map", type=Path, default=Path("data_clean/label_map.csv"))
    parser.add_argument("--landmarks-root", type=Path, default=Path("data_clean/landmarks"))
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--classifier", choices=("logreg", "random_forest"), default="logreg")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--allow-missing-landmarks", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    return parser.parse_args()


def compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro"),
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
    }


def save_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def save_confusion_csv(path: Path, matrix: list[list[int]], index_to_label: dict[int, str]) -> None:
    labels = [index_to_label[index] for index in sorted(index_to_label)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row)


def main() -> None:
    args = parse_args()
    label_to_index, index_to_label, label_to_command = build_mediapipe_label_maps(args.label_map)
    annotations = {row["clip_id"]: row for row in read_csv(args.annotations)}
    split_rows = read_csv(args.splits)

    x_train, y_train, train_clip_ids = load_mediapipe_split_features(
        split="train",
        annotations=annotations,
        split_rows=split_rows,
        label_to_index=label_to_index,
        landmarks_root=args.landmarks_root,
        seq_len=args.seq_len,
        allow_missing_landmarks=args.allow_missing_landmarks,
        max_samples=args.max_train_samples,
    )
    x_val, y_val, _ = load_mediapipe_split_features(
        split="val",
        annotations=annotations,
        split_rows=split_rows,
        label_to_index=label_to_index,
        landmarks_root=args.landmarks_root,
        seq_len=args.seq_len,
        allow_missing_landmarks=args.allow_missing_landmarks,
    )
    x_test, y_test, _ = load_mediapipe_split_features(
        split="test",
        annotations=annotations,
        split_rows=split_rows,
        label_to_index=label_to_index,
        landmarks_root=args.landmarks_root,
        seq_len=args.seq_len,
        allow_missing_landmarks=args.allow_missing_landmarks,
    )

    model = build_mediapipe_classifier(args.classifier)
    model.fit(x_train, y_train)

    val_predictions = model.predict(x_val)
    test_predictions = model.predict(x_test)
    train_predictions = model.predict(x_train)

    train_metrics = compute_metrics(y_train, train_predictions)
    val_metrics = compute_metrics(y_val, val_predictions)
    test_metrics = compute_metrics(y_test, test_predictions)

    run_name = f"mediapipe_{args.classifier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / "model.joblib")
    save_json(
        output_dir / "config.json",
        {
            "model": "mediapipe_baseline",
            "classifier": args.classifier,
            "seq_len": args.seq_len,
            "landmarks_root": str(args.landmarks_root),
            "allow_missing_landmarks": args.allow_missing_landmarks,
            "train_size": len(y_train),
            "val_size": len(y_val),
            "test_size": len(y_test),
            "feature_dim": int(x_train.shape[1]),
        },
    )
    save_json(
        output_dir / "metrics.json",
        {
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
        },
    )
    save_json(
        output_dir / "label_map.json",
        {
            "label_to_index": label_to_index,
            "index_to_label": index_to_label,
            "label_to_command": label_to_command,
        },
    )
    save_confusion_csv(output_dir / "val_confusion_matrix.csv", val_metrics["confusion_matrix"], index_to_label)
    save_confusion_csv(output_dir / "test_confusion_matrix.csv", test_metrics["confusion_matrix"], index_to_label)

    print("Saved MediaPipe baseline artifacts to", output_dir)
    print(
        json.dumps(
            {
                "model": "mediapipe_baseline",
                "classifier": args.classifier,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
