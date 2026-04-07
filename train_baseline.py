#!/usr/bin/env python3
"""Train Bee-Wo baseline gesture classification models."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from baseline_models import BeeWoMobileNetV2_3D, BeeWoResNet10_3D
from bee_wo_dataset import BeeWoClipDataset, build_label_maps
from project_models import (
    BeeWoCNNLSTM,
    BeeWoR2Plus1D_18,
    BeeWoResNet10LandmarkFusion,
    BeeWoResNet10TemporalAttention,
)


MODEL_CHOICES = (
    "simple_cnn",
    "resnet10_3d",
    "r2plus1d_18",
    "mobilenetv2_3d",
    "cnn_lstm",
    "resnet10_temporal_attention",
    "resnet10_landmark_fusion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Bee-Wo baseline models.")
    parser.add_argument("--data-root", type=Path, default=Path("data_clean"))
    parser.add_argument("--annotations", type=Path, default=Path("data_clean/annotations.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data_clean/splits.csv"))
    parser.add_argument("--label-map", type=Path, default=Path("data_clean/label_map.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--model", type=str, choices=MODEL_CHOICES, default="simple_cnn")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260331)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--pretrained-video-weights", action="store_true")
    parser.add_argument("--landmarks-root", type=Path, default=Path("data_clean/landmarks"))
    parser.add_argument("--allow-missing-landmarks", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FrameEncoderTemporalAvg(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, height, width = frames.shape
        encoded = self.encoder(frames.view(batch_size * seq_len, channels, height, width))
        encoded = encoded.view(batch_size, seq_len, 64, 1, 1).mean(dim=1)
        return self.classifier(encoded)


def build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    if args.model == "simple_cnn":
        return FrameEncoderTemporalAvg(num_classes=num_classes)
    if args.model == "cnn_lstm":
        return BeeWoCNNLSTM(num_classes=num_classes)
    if args.model == "resnet10_landmark_fusion":
        return BeeWoResNet10LandmarkFusion(
            num_classes=num_classes,
            seq_len=args.seq_len,
        )
    if args.model == "resnet10_temporal_attention":
        return BeeWoResNet10TemporalAttention(
            num_classes=num_classes,
            sample_size=args.image_size,
            sample_duration=args.seq_len,
        )
    if args.model == "resnet10_3d":
        return BeeWoResNet10_3D(
            num_classes=num_classes,
            sample_size=args.image_size,
            sample_duration=args.seq_len,
        )
    if args.model == "r2plus1d_18":
        return BeeWoR2Plus1D_18(
            num_classes=num_classes,
            pretrained=args.pretrained_video_weights,
        )
    if args.model == "mobilenetv2_3d":
        return BeeWoMobileNetV2_3D(
            num_classes=num_classes,
            sample_size=args.image_size,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def prepare_frames(frames: torch.Tensor, model_name: str) -> torch.Tensor:
    if model_name in {"simple_cnn", "cnn_lstm"}:
        return frames
    return frames.permute(0, 2, 1, 3, 4).contiguous()


def build_training_components(
    args: argparse.Namespace,
    model: nn.Module,
) -> tuple[nn.Module, torch.optim.Optimizer, CosineAnnealingLR | None, float]:
    if args.model == "resnet10_temporal_attention":
        label_smoothing = args.label_smoothing
        learning_rate = args.learning_rate if args.learning_rate != 1e-3 else 5e-4
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler: CosineAnnealingLR | None = None
        return criterion, optimizer, scheduler, label_smoothing
    if args.model == "resnet10_landmark_fusion":
        label_smoothing = args.label_smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        return criterion, optimizer, None, label_smoothing
    if args.model == "r2plus1d_18":
        label_smoothing = args.label_smoothing
        learning_rate = args.learning_rate if args.learning_rate != 1e-3 else 3e-4
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler: CosineAnnealingLR | None = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        return criterion, optimizer, scheduler, label_smoothing

    label_smoothing = args.label_smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return criterion, optimizer, None, label_smoothing


def build_loader(args: argparse.Namespace, split: str, shuffle: bool) -> DataLoader:
    dataset = BeeWoClipDataset(
        data_root=args.data_root,
        annotations_path=args.annotations,
        splits_path=args.splits,
        label_map_path=args.label_map,
        split=split,
        seq_len=args.seq_len,
        image_size=args.image_size,
        use_landmarks=args.model == "resnet10_landmark_fusion",
        landmarks_root=args.landmarks_root,
        allow_missing_landmarks=args.allow_missing_landmarks,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    model_name: str,
    max_batches: int = 0,
) -> tuple[float, list[int], list[int]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break

        frames = prepare_frames(batch["frames"], model_name).to(device)
        labels = batch["label"].to(device)
        landmarks = batch.get("landmarks")
        if landmarks is not None:
            landmarks = landmarks.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            if model_name == "resnet10_landmark_fusion":
                logits = model(frames, landmarks)
            else:
                logits = model(frames)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = frames.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        all_targets.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    avg_loss = total_loss / max(total_examples, 1)
    return avg_loss, all_targets, all_predictions


def compute_metrics(targets: list[int], predictions: list[int]) -> dict[str, object]:
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
    set_seed(args.seed)
    device = resolve_device(args.device)

    label_to_index, index_to_label = build_label_maps(args.label_map)
    train_loader = build_loader(args, split="train", shuffle=True)
    val_loader = build_loader(args, split="val", shuffle=False)
    test_loader = build_loader(args, split="test", shuffle=False)

    run_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, num_classes=len(label_to_index)).to(device)
    criterion, optimizer, scheduler, label_smoothing = build_training_components(args, model)

    best_val_f1 = -1.0
    best_state_path = output_dir / "best_model.pt"
    history: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_targets, train_predictions = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            model_name=args.model,
            max_batches=args.max_train_batches,
        )
        val_loss, val_targets, val_predictions = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            model_name=args.model,
            max_batches=args.max_eval_batches,
        )

        train_metrics = compute_metrics(train_targets, train_predictions)
        val_metrics = compute_metrics(val_targets, val_predictions)
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result))

        if float(val_metrics["macro_f1"]) > best_val_f1:
            best_val_f1 = float(val_metrics["macro_f1"])
            torch.save(model.state_dict(), best_state_path)

        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(torch.load(best_state_path, map_location=device))
    val_loss, val_targets, val_predictions = run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        model_name=args.model,
        max_batches=args.max_eval_batches,
    )
    test_loss, test_targets, test_predictions = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        model_name=args.model,
        max_batches=args.max_eval_batches,
    )

    val_metrics = compute_metrics(val_targets, val_predictions)
    test_metrics = compute_metrics(test_targets, test_predictions)
    save_json(
        output_dir / "config.json",
        {
            "model": args.model,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "label_smoothing": label_smoothing,
            "landmarks_root": str(args.landmarks_root),
            "allow_missing_landmarks": args.allow_missing_landmarks,
            "image_size": args.image_size,
            "seed": args.seed,
            "device": str(device),
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "test_size": len(test_loader.dataset),
            "max_train_batches": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
            "optimizer": type(optimizer).__name__,
            "scheduler": type(scheduler).__name__ if scheduler is not None else None,
        },
    )
    save_json(output_dir / "history.json", {"epochs": history})
    save_json(
        output_dir / "metrics.json",
        {
            "val_loss": val_loss,
            "test_loss": test_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
        },
    )
    save_confusion_csv(
        output_dir / "val_confusion_matrix.csv",
        val_metrics["confusion_matrix"],
        index_to_label,
    )
    save_confusion_csv(
        output_dir / "test_confusion_matrix.csv",
        test_metrics["confusion_matrix"],
        index_to_label,
    )

    print("Saved run artifacts to", output_dir)
    print(
        json.dumps(
            {
                "model": args.model,
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
