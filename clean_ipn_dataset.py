#!/usr/bin/env python3
"""Clean the IPN Hand dataset into clip-level gesture folders.

This script:
1. Reads the combined annotation list.
2. Keeps only the Bee-Wo target gestures.
3. Reuses or extracts raw frame archives as needed.
4. Copies all available JPGs inside each annotation range into one clip folder.
5. Writes dataset metadata files under data_clean/.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_ROOT / "raw data"
ANNOTATIONS_ROOT = RAW_ROOT / "annotations"
FRAME_ROOT = RAW_ROOT / "frame"
OUTPUT_ROOT = PROJECT_ROOT / "data_clean"
DATASET_ROOT = OUTPUT_ROOT / "all"

ANNOTATION_FILE = ANNOTATIONS_ROOT / "Annot_List.txt"
CLASS_INDEX_FILE = ANNOTATIONS_ROOT / "classIdx.txt"
VIDEO_METADATA_FILE = ANNOTATIONS_ROOT / "metadata.csv"

LABEL_TO_COMMAND = {
    "G01": "stop",
    "G02": "move_back",
    "G05": "move_left",
    "G06": "move_right",
    "G07": "move_forward",
}
TARGET_LABELS = set(LABEL_TO_COMMAND)

FRAME_PATTERN = re.compile(r"_(\d+)\.jpg$", re.IGNORECASE)


@dataclass(frozen=True)
class ClipAnnotation:
    video_id: str
    gesture_label: str
    class_id: str
    start_frame: int
    end_frame: int
    annotated_num_frames: int

    @property
    def clip_id(self) -> str:
        return (
            f"{self.video_id}__{self.gesture_label}"
            f"__s{self.start_frame:06d}_e{self.end_frame:06d}"
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_annotations(path: Path) -> list[ClipAnnotation]:
    clips: list[ClipAnnotation] = []
    for row in read_csv_rows(path):
        if row["label"] not in TARGET_LABELS:
            continue
        clips.append(
            ClipAnnotation(
                video_id=row["video"],
                gesture_label=row["label"],
                class_id=row["id"],
                start_frame=int(row["t_start"]),
                end_frame=int(row["t_end"]),
                annotated_num_frames=int(row["frames"]),
            )
        )
    return clips


def load_class_ids(path: Path) -> dict[str, str]:
    return {row["label"]: row["id"] for row in read_csv_rows(path)}


def load_video_metadata(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path)
    metadata: dict[str, dict[str, str]] = {}
    for row in rows:
        metadata[row["Video Name"]] = {
            "video_total_frames": row["Frames"],
            "sex": row["Sex"],
            "hand": row["Hand"],
            "background": row["Background"],
            "illumination": row["Illumination"],
            "people_in_scene": row["People in Scene"],
            "background_motion": row["Background Motion"],
            "source_set": row["Set"],
        }
    return metadata


def discover_video_dirs(root: Path) -> dict[str, Path]:
    video_dirs: dict[str, Path] = {}
    for current_root, dirnames, filenames in os.walk(root):
        if not filenames:
            continue
        jpgs = [name for name in filenames if name.lower().endswith(".jpg")]
        if not jpgs:
            continue
        current_path = Path(current_root)
        video_id = current_path.name
        if video_id not in video_dirs:
            video_dirs[video_id] = current_path
    return video_dirs


def find_missing_video_ids(
    clips: list[ClipAnnotation], video_dirs: dict[str, Path]
) -> set[str]:
    return {clip.video_id for clip in clips if clip.video_id not in video_dirs}


def extract_needed_archives(root: Path, missing_video_ids: set[str]) -> set[Path]:
    extracted_archives: set[Path] = set()
    if not missing_video_ids:
        return extracted_archives

    tgz_files = sorted(root.rglob("*.tgz"))
    for tgz_path in tgz_files:
        if not missing_video_ids:
            break
        archive_matches = False
        with tarfile.open(tgz_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if len(parts) >= 2 and parts[-2] == "frames":
                    candidate = parts[-1]
                elif len(parts) >= 2 and parts[-1].lower().endswith(".jpg"):
                    candidate = parts[-2]
                else:
                    continue
                if candidate in missing_video_ids:
                    archive_matches = True
                    break
            if archive_matches:
                archive.extractall(path=tgz_path.parent)
                extracted_archives.add(tgz_path)
        if archive_matches:
            video_dirs = discover_video_dirs(root)
            missing_video_ids = {
                video_id for video_id in missing_video_ids if video_id not in video_dirs
            }
    return extracted_archives


def extract_frame_number(path: Path) -> int | None:
    match = FRAME_PATTERN.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def list_frames_in_range(
    video_dir: Path, start_frame: int, end_frame: int
) -> list[tuple[int, Path]]:
    frame_paths: list[tuple[int, Path]] = []
    for image_path in sorted(video_dir.glob("*.jpg")):
        frame_number = extract_frame_number(image_path)
        if frame_number is None:
            continue
        if start_frame <= frame_number <= end_frame:
            frame_paths.append((frame_number, image_path))
    return frame_paths


def ensure_clean_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    DATASET_ROOT.mkdir(exist_ok=True)
    for label in sorted(TARGET_LABELS):
        (DATASET_ROOT / label).mkdir(parents=True, exist_ok=True)


def copy_clip_frames(
    clip: ClipAnnotation, frame_paths: list[tuple[int, Path]]
) -> tuple[Path, int, bool]:
    clip_dir = DATASET_ROOT / clip.gesture_label / clip.clip_id
    expected_names = [path.name for _, path in frame_paths]

    already_complete = False
    if clip_dir.exists():
        existing_names = sorted(
            path.name for path in clip_dir.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"
        )
        if existing_names == expected_names:
            already_complete = True
            return clip_dir, len(existing_names), already_complete
        shutil.rmtree(clip_dir)

    clip_dir.mkdir(parents=True, exist_ok=True)
    for _, source_path in frame_paths:
        destination_path = clip_dir / source_path.name
        shutil.copy2(source_path, destination_path)
    return clip_dir, len(frame_paths), already_complete


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_annotation_row(
    clip: ClipAnnotation,
    clip_dir: Path,
    copied_frame_count: int,
    copied_frame_numbers: list[int],
    video_metadata: dict[str, dict[str, str]],
) -> dict[str, object]:
    metadata = video_metadata.get(clip.video_id, {})
    return {
        "clip_id": clip.clip_id,
        "clip_path": clip_dir.relative_to(OUTPUT_ROOT).as_posix(),
        "video_id": clip.video_id,
        "gesture_label": clip.gesture_label,
        "class_id": clip.class_id,
        "command": LABEL_TO_COMMAND[clip.gesture_label],
        "start_frame": clip.start_frame,
        "end_frame": clip.end_frame,
        "annotated_num_frames": clip.annotated_num_frames,
        "num_frames": copied_frame_count,
        "first_available_frame": copied_frame_numbers[0] if copied_frame_numbers else "",
        "last_available_frame": copied_frame_numbers[-1] if copied_frame_numbers else "",
        "video_total_frames": metadata.get("video_total_frames", ""),
        "sex": metadata.get("sex", ""),
        "hand": metadata.get("hand", ""),
        "background": metadata.get("background", ""),
        "illumination": metadata.get("illumination", ""),
        "people_in_scene": metadata.get("people_in_scene", ""),
        "background_motion": metadata.get("background_motion", ""),
        "source_set": metadata.get("source_set", ""),
    }


def build_summary_rows(annotations: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped_counts: dict[str, list[int]] = defaultdict(list)
    for row in annotations:
        grouped_counts[str(row["gesture_label"])].append(int(row["num_frames"]))

    summary_rows: list[dict[str, object]] = []
    for label in sorted(grouped_counts):
        frame_counts = grouped_counts[label]
        summary_rows.append(
            {
                "gesture_label": label,
                "command": LABEL_TO_COMMAND[label],
                "clip_count": len(frame_counts),
                "min_num_frames": min(frame_counts),
                "mean_num_frames": round(mean(frame_counts), 2),
                "max_num_frames": max(frame_counts),
            }
        )
    return summary_rows


def write_summary_report(
    path: Path,
    summary_rows: list[dict[str, object]],
    total_annotations: int,
    completed_clips: int,
    skipped_clips: int,
    failed_clips: list[dict[str, object]],
    extracted_archives: set[Path],
) -> None:
    lines = [
        "# Bee-Wo IPN Cleaning Report",
        "",
        f"- Total target annotations: {total_annotations}",
        f"- Completed clip folders: {completed_clips}",
        f"- Reused existing clip folders: {skipped_clips}",
        f"- Failed clips: {len(failed_clips)}",
        f"- Archives extracted this run: {len(extracted_archives)}",
        "",
        "## Class Summary",
        "",
        "| Label | Command | Clips | Min Frames | Mean Frames | Max Frames |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {gesture_label} | {command} | {clip_count} | {min_num_frames} | "
            "{mean_num_frames} | {max_num_frames} |".format(**row)
        )

    if failed_clips:
        lines.extend(
            [
                "",
                "## Failures",
                "",
                "| clip_id | video_id | gesture_label | reason |",
                "| --- | --- | --- | --- |",
            ]
        )
        for failure in failed_clips:
            lines.append(
                f"| {failure['clip_id']} | {failure['video_id']} | "
                f"{failure['gesture_label']} | {failure['reason']} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_clean_output_dirs()

    clips = load_annotations(ANNOTATION_FILE)
    class_ids = load_class_ids(CLASS_INDEX_FILE)
    video_metadata = load_video_metadata(VIDEO_METADATA_FILE)

    video_dirs = discover_video_dirs(FRAME_ROOT)
    missing_video_ids = find_missing_video_ids(clips, video_dirs)
    extracted_archives = extract_needed_archives(FRAME_ROOT, missing_video_ids)
    if extracted_archives:
        video_dirs = discover_video_dirs(FRAME_ROOT)

    annotation_rows: list[dict[str, object]] = []
    failed_clips: list[dict[str, object]] = []
    completed_clips = 0
    skipped_clips = 0

    for clip in clips:
        if clip.class_id != class_ids.get(clip.gesture_label):
            failed_clips.append(
                {
                    "clip_id": clip.clip_id,
                    "video_id": clip.video_id,
                    "gesture_label": clip.gesture_label,
                    "reason": "class_id mismatch with classIdx.txt",
                }
            )
            continue

        video_dir = video_dirs.get(clip.video_id)
        if video_dir is None:
            failed_clips.append(
                {
                    "clip_id": clip.clip_id,
                    "video_id": clip.video_id,
                    "gesture_label": clip.gesture_label,
                    "reason": "video frames not found",
                }
            )
            continue

        frame_paths = list_frames_in_range(video_dir, clip.start_frame, clip.end_frame)
        if not frame_paths:
            failed_clips.append(
                {
                    "clip_id": clip.clip_id,
                    "video_id": clip.video_id,
                    "gesture_label": clip.gesture_label,
                    "reason": "no JPG frames inside annotated range",
                }
            )
            continue

        clip_dir, copied_frame_count, already_complete = copy_clip_frames(clip, frame_paths)
        if already_complete:
            skipped_clips += 1
        else:
            completed_clips += 1

        annotation_rows.append(
            build_annotation_row(
                clip=clip,
                clip_dir=clip_dir,
                copied_frame_count=copied_frame_count,
                copied_frame_numbers=[frame_number for frame_number, _ in frame_paths],
                video_metadata=video_metadata,
            )
        )

    annotation_rows.sort(key=lambda row: str(row["clip_id"]))
    failed_clips.sort(key=lambda row: str(row["clip_id"]))

    annotations_fieldnames = [
        "clip_id",
        "clip_path",
        "video_id",
        "gesture_label",
        "class_id",
        "command",
        "start_frame",
        "end_frame",
        "annotated_num_frames",
        "num_frames",
        "first_available_frame",
        "last_available_frame",
        "video_total_frames",
        "sex",
        "hand",
        "background",
        "illumination",
        "people_in_scene",
        "background_motion",
        "source_set",
    ]
    write_csv(OUTPUT_ROOT / "annotations.csv", annotations_fieldnames, annotation_rows)

    label_map_rows = [
        {
            "gesture_label": label,
            "class_id": class_ids[label],
            "command": LABEL_TO_COMMAND[label],
        }
        for label in sorted(TARGET_LABELS)
    ]
    write_csv(
        OUTPUT_ROOT / "label_map.csv",
        ["gesture_label", "class_id", "command"],
        label_map_rows,
    )

    summary_rows = build_summary_rows(annotation_rows)
    write_csv(
        OUTPUT_ROOT / "cleaning_summary.csv",
        ["gesture_label", "command", "clip_count", "min_num_frames", "mean_num_frames", "max_num_frames"],
        summary_rows,
    )

    write_csv(
        OUTPUT_ROOT / "cleaning_failures.csv",
        ["clip_id", "video_id", "gesture_label", "reason"],
        failed_clips,
    )

    write_summary_report(
        OUTPUT_ROOT / "cleaning_report.md",
        summary_rows=summary_rows,
        total_annotations=len(clips),
        completed_clips=completed_clips,
        skipped_clips=skipped_clips,
        failed_clips=failed_clips,
        extracted_archives=extracted_archives,
    )

    counts = Counter(row["gesture_label"] for row in annotation_rows)
    print("Cleaned clips:", len(annotation_rows))
    print("Counts by class:", dict(sorted(counts.items())))
    print("Failures:", len(failed_clips))
    print("Output:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
