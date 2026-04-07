#!/usr/bin/env python3
"""Run live webcam inference and print mapped robot commands."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from project_models import BeeWoResNet10LandmarkFusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bee-Wo webcam inference with a pretrained fusion model."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, default=Path("data_clean/label_map.csv"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hand-landmarker-task", type=Path, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--vote-size", type=int, default=5)
    parser.add_argument("--min-vote-count", type=int, default=4)
    parser.add_argument("--min-detected-frames", type=int, default=12)
    parser.add_argument("--print-cooldown", type=float, default=1.0)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_label_maps(label_map_path: Path) -> tuple[dict[str, int], dict[int, str], dict[str, str]]:
    rows: list[dict[str, str]]
    with label_map_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["class_id"]))
    label_to_index = {row["gesture_label"]: index for index, row in enumerate(rows)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    label_to_command = {row["gesture_label"]: row["command"] for row in rows}
    return label_to_index, index_to_label, label_to_command


def resolve_hand_landmarker_task(task_path: Path | None) -> Path | None:
    candidates: list[Path] = []
    if task_path is not None:
        candidates.append(task_path)
    repo_root = Path(__file__).resolve().parent
    candidates.extend(
        [
            repo_root / "hand_landmarker.task",
            repo_root / "checkpoints" / "hand_landmarker.task",
            repo_root / "models" / "hand_landmarker.task",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_mediapipe_hands(task_path: Path | None):
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "MediaPipe is required for live landmark inference. Install it with `pip install mediapipe`."
        ) from exc

    if hasattr(mp, "solutions"):
        return {
            "kind": "solutions",
            "HandsClass": mp.solutions.hands.Hands,
            "drawing_utils": mp.solutions.drawing_utils,
            "hand_connections": mp.solutions.hands.HAND_CONNECTIONS,
        }

    try:
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore
        from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore
        from mediapipe.tasks.python.vision import drawing_utils as task_drawing_utils  # type: ignore
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Installed MediaPipe package does not expose a supported Hands API."
        ) from exc

    resolved_task_path = resolve_hand_landmarker_task(task_path)
    if resolved_task_path is None:
        raise SystemExit(
            "This MediaPipe install uses the Tasks API and requires a `hand_landmarker.task` model file. "
            "Place it at `./hand_landmarker.task`, `./checkpoints/hand_landmarker.task`, "
            "or pass `--hand-landmarker-task /absolute/path/to/hand_landmarker.task`."
        )

    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(resolved_task_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return {
        "kind": "tasks",
        "landmarker": mp_vision.HandLandmarker.create_from_options(options),
        "drawing_utils": task_drawing_utils,
        "hand_connections": mp_vision.HandLandmarksConnections.HAND_CONNECTIONS,
        "Image": Image,
        "ImageFormat": ImageFormat,
        "task_path": resolved_task_path,
    }


def preprocess_frame(frame_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = frame_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def extract_landmarks(hand_result, backend_kind: str) -> tuple[np.ndarray, bool]:
    if backend_kind == "solutions":
        landmarks = hand_result.multi_hand_landmarks
    else:
        landmarks = hand_result.hand_landmarks

    if not landmarks:
        return np.zeros(63, dtype=np.float32), False

    coords: list[float] = []
    if backend_kind == "solutions":
        iterable = landmarks[0].landmark
    else:
        iterable = landmarks[0]
    for landmark in iterable:
        coords.extend([landmark.x, landmark.y, landmark.z])
    return np.asarray(coords, dtype=np.float32), True


def build_model(checkpoint_path: Path, device: torch.device, seq_len: int) -> BeeWoResNet10LandmarkFusion:
    model = BeeWoResNet10LandmarkFusion(num_classes=5, seq_len=seq_len)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    _, index_to_label, label_to_command = load_label_maps(args.label_map)
    mp_backend = load_mediapipe_hands(args.hand_landmarker_task)
    drawing_utils = mp_backend["drawing_utils"]
    hand_connections = mp_backend["hand_connections"]

    model = build_model(args.checkpoint, device, args.seq_len)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam at index {args.camera_index}.")

    frame_buffer: deque[torch.Tensor] = deque(maxlen=args.seq_len)
    landmark_buffer: deque[np.ndarray] = deque(maxlen=args.seq_len)
    detection_buffer: deque[bool] = deque(maxlen=args.seq_len)
    prediction_buffer: deque[str] = deque(maxlen=max(1, args.vote_size))
    last_printed_command = ""
    last_print_time = 0.0
    current_label = "warming_up"
    current_command = "collecting frames"
    current_confidence = 0.0
    raw_label = "warming_up"
    raw_confidence = 0.0

    print("Webcam demo started. Press 'q' to quit.")
    print(f"Using checkpoint: {args.checkpoint}")
    print(f"Using device: {device}")
    if mp_backend["kind"] == "tasks":
        print(f"Using MediaPipe task model: {mp_backend['task_path']}")

    if mp_backend["kind"] == "solutions":
        hands_context = mp_backend["HandsClass"](
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    else:
        hands_context = mp_backend["landmarker"]

    with hands_context as hands:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Failed to read from webcam.")
                break

            display_frame = frame_bgr.copy()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if mp_backend["kind"] == "solutions":
                hand_result = hands.process(frame_rgb)
            else:
                mp_image = mp_backend["Image"](image_format=mp_backend["ImageFormat"].SRGB, data=frame_rgb)
                hand_result = hands.detect(mp_image)
            landmarks, detected = extract_landmarks(hand_result, mp_backend["kind"])
            if detected:
                if mp_backend["kind"] == "solutions":
                    drawn_landmarks = hand_result.multi_hand_landmarks[0]
                else:
                    drawn_landmarks = hand_result.hand_landmarks[0]
                drawing_utils.draw_landmarks(
                    display_frame,
                    drawn_landmarks,
                    hand_connections,
                )

            frame_buffer.append(preprocess_frame(frame_bgr, args.image_size))
            landmark_buffer.append(landmarks)
            detection_buffer.append(detected)

            if (
                len(frame_buffer) == args.seq_len
                and len(landmark_buffer) == args.seq_len
                and len(detection_buffer) == args.seq_len
            ):
                frames = torch.stack(list(frame_buffer)).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
                landmark_tensor = torch.from_numpy(np.stack(list(landmark_buffer))).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(frames, landmark_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    confidence, pred_index = torch.max(probs, dim=0)

                predicted_label = index_to_label[int(pred_index.item())]
                raw_label = predicted_label
                raw_confidence = float(confidence.item())
                prediction_buffer.append(predicted_label)
                voted_label, voted_count = Counter(prediction_buffer).most_common(1)[0]
                voted_command = label_to_command[voted_label]
                detected_frames = sum(detection_buffer)

                current_confidence = raw_confidence
                is_approved = (
                    raw_confidence >= args.confidence_threshold
                    and voted_count >= args.min_vote_count
                    and detected_frames >= args.min_detected_frames
                )

                if is_approved:
                    current_label = voted_label
                    current_command = voted_command
                else:
                    current_label = "none"
                    current_command = "idle"

                now = time.time()
                if (
                    is_approved
                    and (voted_command != last_printed_command or now - last_print_time >= args.print_cooldown)
                ):
                    print(
                        f"gesture={voted_label} command={voted_command} "
                        f"confidence={current_confidence:.3f} "
                        f"vote={voted_count}/{len(prediction_buffer)} "
                        f"detected_frames={detected_frames}/{args.seq_len}"
                    )
                    last_printed_command = voted_command
                    last_print_time = now

            status_lines = [
                f"gesture: {current_label}",
                f"command: {current_command}",
                f"approved conf: {current_confidence:.2f}",
                f"raw: {raw_label} ({raw_confidence:.2f})",
                f"detected: {sum(detection_buffer)}/{args.seq_len}",
                f"frames: {len(frame_buffer)}/{args.seq_len}",
            ]
            for idx, text in enumerate(status_lines):
                cv2.putText(
                    display_frame,
                    text,
                    (10, 30 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Bee-Wo Live Robot Command Demo", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
