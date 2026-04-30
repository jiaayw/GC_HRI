#!/usr/bin/env python3
"""Extract MediaPipe hand landmarks for Bee-Wo clip folders."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.bee_wo_dataset import sample_frame_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks for Bee-Wo clips.")
    parser.add_argument("--data-root", type=Path, default=Path("data_clean"))
    parser.add_argument("--annotations", type=Path, default=Path("data_clean/annotations.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data_clean/landmarks"))
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def read_annotations(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "MediaPipe is not installed. Install it first, for example:\n"
            "  pip install mediapipe\n"
        ) from exc
    return mp


def extract_landmarks_for_frame(mp_hands, frame_path: Path) -> tuple[np.ndarray, bool]:
    from PIL import Image

    image = Image.open(frame_path).convert("RGB")
    image_array = np.asarray(image)
    result = mp_hands.process(image_array)
    if not result.multi_hand_landmarks:
        return np.zeros(63, dtype=np.float32), False

    hand_landmarks = result.multi_hand_landmarks[0]
    coords = []
    for landmark in hand_landmarks.landmark:
        coords.extend([landmark.x, landmark.y, landmark.z])
    return np.asarray(coords, dtype=np.float32), True


def main() -> None:
    args = parse_args()
    mp = load_mediapipe()
    annotations = read_annotations(args.annotations)
    args.output_root.mkdir(parents=True, exist_ok=True)
    total_clips = min(len(annotations), args.max_clips) if args.max_clips else len(annotations)

    summary_rows: list[dict[str, object]] = []
    processed = 0
    skipped = 0
    failed = 0
    saved = 0
    detected_frames_total = 0
    sampled_frames_total = 0

    print(
        f"Starting landmark extraction for {total_clips} clips "
        f"(seq_len={args.seq_len}, output={args.output_root})"
    )

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        for row in annotations:
            clip_id = row["clip_id"]
            clip_path = args.data_root / row["clip_path"]
            frame_paths = sorted(clip_path.glob("*.jpg"))
            if not frame_paths:
                skipped += 1
                print(f"[warn] No JPG frames found for clip {clip_id} at {clip_path}")
                if args.max_clips and (processed + skipped + failed) >= args.max_clips:
                    break
                continue

            sampled_frame_paths = sample_frame_paths(frame_paths, args.seq_len)
            if not sampled_frame_paths:
                skipped += 1
                print(f"[warn] No sampled frames available for clip {clip_id}")
                if args.max_clips and (processed + skipped + failed) >= args.max_clips:
                    break
                continue

            clip_landmarks = []
            detected_frames = 0
            try:
                for frame_path in sampled_frame_paths:
                    coords, detected = extract_landmarks_for_frame(hands, frame_path)
                    clip_landmarks.append(coords)
                    detected_frames += int(detected)

                landmark_array = np.stack(clip_landmarks).astype(np.float32)
                np.save(args.output_root / f"{clip_id}.npy", landmark_array)
                summary_rows.append(
                    {
                        "clip_id": clip_id,
                        "landmark_path": str((args.output_root / f"{clip_id}.npy").relative_to(args.data_root)),
                        "detected_frames": detected_frames,
                        "seq_len": args.seq_len,
                    }
                )
            except Exception as exc:
                failed += 1
                print(f"[warn] Failed to process clip {clip_id}: {exc}")
                if args.max_clips and (processed + skipped + failed) >= args.max_clips:
                    break
                continue

            processed += 1
            saved += 1
            detected_frames_total += detected_frames
            sampled_frames_total += len(sampled_frame_paths)

            if (
                args.log_every > 0
                and (processed % args.log_every == 0 or processed == total_clips)
            ):
                percent = 100.0 * processed / max(total_clips, 1)
                detection_rate = 100.0 * detected_frames_total / max(sampled_frames_total, 1)
                print(
                    f"[progress] {processed}/{total_clips} clips ({percent:.1f}%) "
                    f"saved={saved} skipped={skipped} failed={failed} "
                    f"detection_rate={detection_rate:.1f}% last_clip={clip_id}"
                )

            if args.max_clips and (processed + skipped + failed) >= args.max_clips:
                break

    summary_path = args.output_root / "landmarks_manifest.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["clip_id", "landmark_path", "detected_frames", "seq_len"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    overall_detection_rate = 100.0 * detected_frames_total / max(sampled_frames_total, 1)
    print("Saved landmark files to", args.output_root)
    print("Saved manifest to", summary_path)
    print("Processed clips:", processed)
    print("Saved clips:", saved)
    print("Skipped clips:", skipped)
    print("Failed clips:", failed)
    print(f"Overall detected-frame rate: {overall_detection_rate:.1f}%")


if __name__ == "__main__":
    main()
