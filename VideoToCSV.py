#!/usr/bin/env python3
"""Three-dancer video -> tracked per-person CSV -> enhanced OpenSim TRC.

The full-frame Pose Landmarker locates and tracks people. A separate
single-person Pose Landmarker analyses each tracked crop, preserving stable
person IDs while reducing multi-person pose-estimation error.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from native_paths import model_path as native_model_path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
DEFAULT_POSE_MODEL = SCRIPT_DIR / "pose_landmarker_full.task"

POSE_NAMES = (
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR",
    "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER",
    "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST",
    "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP",
    "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
    "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
)

RAW_FIELDS = (
    "frame", "time", "person", "landmark", "landmark_name",
    "x", "y", "z", "visibility", "presence", "coordinate_space",
    "image_x", "image_y", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "track_score", "track_predicted",
)
PROCESSED_FIELDS = RAW_FIELDS + ("quality",)
MARKER_FIELDS = ("frame", "time", "person", "marker", "x", "y", "z", "units")


def probability(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("必須介於 0 到 1")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必須是大於 0 的整數")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必須是大於 0 的有限數字")
    return number


def resolve_dataset(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (DATASET_DIR / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="單人影片：追蹤、逐人裁切 Pose、清理、marker 擴增及逐人 TRC",
    )
    parser.add_argument("input", type=Path, help="dataset 內的 MP4 或絕對路徑")
    parser.add_argument(
        "--fit-model", type=Path, required=True,
        help="含 MarkerSet 的 scaled OpenSim .osim 模型",
    )
    parser.add_argument("--output-dir", type=Path, help="預設 dataset/<影片名>_dance")
    parser.add_argument("--model", type=Path, help="MediaPipe pose_landmarker .task")
    parser.add_argument("--num-people", type=positive_int, default=1)
    parser.add_argument(
        "--target-shirt-color", choices=("red",),
        help="只追蹤指定衣服顏色的人；目前支援 red",
    )
    parser.add_argument("--crop-padding", type=float, default=0.22)
    parser.add_argument("--max-track-gap", type=int, default=12)
    parser.add_argument("--min-detection-confidence", type=probability, default=0.45)
    parser.add_argument("--min-presence-confidence", type=probability, default=0.45)
    parser.add_argument("--min-tracking-confidence", type=probability, default=0.45)
    parser.add_argument("--min-landmark-confidence", type=probability, default=0.50)
    parser.add_argument("--max-interpolation-gap", type=int, default=6)
    parser.add_argument("--outlier-window", type=positive_int, default=1)
    parser.add_argument("--outlier-threshold-m", type=positive_float, default=0.20)
    parser.add_argument("--smoothing-alpha", type=probability, default=0.35)
    parser.add_argument("--max-marker-speed-mps", type=positive_float, default=8.0)
    parser.add_argument("--max-marker-acceleration-mps2", type=positive_float, default=120.0)
    parser.add_argument(
        "--no-ground-correction", action="store_true",
        help="不要依最低支撐腳修正全身 Y；預設會修正以減少蹲下時浮空",
    )
    parser.add_argument(
        "--allow-inverted-poses", action="store_true",
        help="允許肩膀低於髖、腳踝高於髖的影像姿勢（倒立舞蹈才使用）",
    )
    parser.add_argument(
        "--min-upright-image-span", type=positive_float, default=0.015,
        help="肩膀/腳踝與髖部的最小影像垂直間距（預設：0.015）",
    )
    parser.add_argument(
        "--keep-long-gaps", action="store_true",
        help="TRC 保留長掉點空白；預設補最近有效值以確保可播放",
    )
    parser.add_argument("--calibration-frames", type=positive_int, default=1)
    parser.add_argument("--stage-width-m", type=positive_float, default=8.0)
    parser.add_argument("--stage-depth-m", type=positive_float, default=6.0)
    parser.add_argument(
        "--enable-global-formation", action="store_true",
        help="加入由影像估計的舞台 X/Z 隊形位移（預設關閉）",
    )
    parser.add_argument(
        "--stage-calibration", type=Path,
        help="JSON：image_points(normalized) 與 stage_points_m(X,Z)，至少四點",
    )
    parser.add_argument("--no-center-formation", action="store_true")
    parser.add_argument("--mirror-horizontal", action="store_true")
    parser.add_argument("--progress-every", type=positive_int, default=50)
    parser.add_argument(
        "--from-raw", action="store_true",
        help="略過影片分析，從 output-dir 既有 person##_raw.csv 重跑後處理",
    )
    return parser


@dataclass
class Detection:
    bbox: np.ndarray
    center: np.ndarray
    score: float
    appearance: np.ndarray
    target_color_score: float = 0.0


@dataclass
class Track:
    person: int
    bbox: np.ndarray | None = None
    center: np.ndarray | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    appearance: np.ndarray | None = None
    misses: int = 0
    score: float = 0.0

    @property
    def active(self) -> bool:
        return self.bbox is not None

    def predicted_center(self) -> np.ndarray:
        return np.asarray([0.5, 0.5]) if self.center is None else self.center + self.velocity

    def update(self, detection: Detection) -> None:
        if self.center is None:
            self.center = detection.center.copy()
            self.bbox = detection.bbox.copy()
        else:
            old = self.center.copy()
            self.center = 0.65 * detection.center + 0.35 * self.predicted_center()
            self.velocity = 0.65 * (self.center - old) + 0.35 * self.velocity
            self.bbox = 0.65 * detection.bbox + 0.35 * self.bbox
        self.appearance = (
            detection.appearance.copy() if self.appearance is None
            else 0.20 * detection.appearance + 0.80 * self.appearance
        )
        self.misses = 0
        self.score = detection.score

    def miss(self) -> None:
        if self.center is not None and self.bbox is not None:
            self.center = self.predicted_center()
            self.bbox[:2] += self.velocity
        self.velocity *= 0.8
        self.misses += 1
        self.score *= 0.8

    def reset(self) -> None:
        self.bbox = None
        self.center = None
        self.velocity[:] = 0
        self.appearance = None
        self.misses = 0
        self.score = 0.0


def clamp_bbox(bbox: np.ndarray, padding: float = 0.0) -> np.ndarray:
    x, y, w, h = (float(v) for v in bbox)
    x -= w * padding
    y -= h * padding
    w *= 1 + 2 * padding
    h *= 1 + 2 * padding
    x2, y2 = min(1.0, x + w), min(1.0, y + h)
    x, y = max(0.0, x), max(0.0, y)
    return np.asarray([x, y, max(1e-4, x2 - x), max(1e-4, y2 - y)])


def appearance_histogram(frame: np.ndarray, bbox: np.ndarray, cv2: Any) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, w, h = bbox
    x1, y1 = int(x * width), int(y * height)
    x2 = max(x1 + 1, min(width, int((x + w) * width)))
    y2 = max(y1 + 1, min(height, int((y + h) * height)))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(32, dtype=float)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).ravel()
    total = float(hist.sum())
    return hist / total if total > 0 else np.zeros(32, dtype=float)


def red_shirt_score(frame: np.ndarray, landmarks: Any, cv2: Any) -> float:
    """Return the red-pixel ratio inside the shoulder/hip torso polygon."""
    height, width = frame.shape[:2]
    polygon = np.asarray([
        [landmarks[index].x * width, landmarks[index].y * height]
        for index in (11, 12, 24, 23)
    ], dtype=np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    pixel_count = int(np.count_nonzero(mask))
    if pixel_count == 0:
        return 0.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = (
        (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168))
         & (hsv[:, :, 1] >= 70)
         & (hsv[:, :, 2] >= 35))
    )
    return float(np.count_nonzero(red & (mask > 0)) / pixel_count)


def detections_from_result(result: Any, frame: np.ndarray, cv2: Any) -> list[Detection]:
    detections: list[Detection] = []
    for landmarks in result.pose_landmarks:
        usable = [
            point for point in landmarks
            if float(getattr(point, "visibility", 1)) >= 0.25
            and float(getattr(point, "presence", 1)) >= 0.25
        ]
        if len(usable) < 8:
            continue
        xs = np.asarray([point.x for point in usable])
        ys = np.asarray([point.y for point in usable])
        x1, x2 = float(np.percentile(xs, 2)), float(np.percentile(xs, 98))
        y1, y2 = float(np.percentile(ys, 2)), float(np.percentile(ys, 98))
        bbox = clamp_bbox(np.asarray([x1, y1, max(.03, x2 - x1), max(.06, y2 - y1)]))
        hips = [landmarks[index] for index in (23, 24)]
        center = np.asarray([np.mean([p.x for p in hips]), np.mean([p.y for p in hips])])
        score = float(np.mean([
            min(float(getattr(p, "visibility", 1)), float(getattr(p, "presence", 1)))
            for p in usable
        ]))
        detections.append(Detection(
            bbox, center, score, appearance_histogram(frame, bbox, cv2),
            red_shirt_score(frame, landmarks, cv2),
        ))
    return detections


def track_cost(track: Track, detection: Detection) -> float:
    distance = float(np.linalg.norm(track.predicted_center() - detection.center))
    size_cost = 0.0 if track.bbox is None else abs(
        math.log(max(detection.bbox[3], 1e-4) / max(track.bbox[3], 1e-4))
    )
    appearance_cost = 0.0 if track.appearance is None else (
        0.5 * float(np.abs(track.appearance - detection.appearance).sum())
    )
    return distance + 0.12 * size_cost + 0.20 * appearance_cost


def assign_tracks(tracks: list[Track], detections: list[Detection]) -> dict[int, int]:
    """Small-N global assignment; appearance helps preserve IDs during crossings."""
    active = [index for index, track in enumerate(tracks) if track.active]
    if not active or not detections:
        return {}
    count = min(len(active), len(detections))
    best: tuple[float, dict[int, int]] | None = None
    for subset in itertools.combinations(active, count):
        for order in itertools.permutations(range(len(detections)), count):
            mapping = dict(zip(subset, order))
            costs = [track_cost(tracks[t], detections[d]) for t, d in mapping.items()]
            if any(cost > 0.75 for cost in costs):
                continue
            total = sum(costs) + 0.25 * (len(active) - count)
            if best is None or total < best[0]:
                best = total, mapping
    return {} if best is None else best[1]


def update_tracks(
    tracks: list[Track], detections: list[Detection], max_misses: int | None = None,
) -> None:
    if max_misses is not None:
        for track in tracks:
            if track.misses > max_misses:
                track.reset()
    mapping = assign_tracks(tracks, detections)
    used = set(mapping.values())
    for index, track in enumerate(tracks):
        track.update(detections[mapping[index]]) if index in mapping else (track.miss() if track.active else None)
    remaining = sorted(
        (index for index in range(len(detections)) if index not in used),
        key=lambda index: detections[index].center[0],
    )
    empty = [index for index, track in enumerate(tracks) if not track.active]
    for track_index, detection_index in zip(empty, remaining):
        tracks[track_index].update(detections[detection_index])


def pose_options(mp: Any, model_path: Path, count: int, args: argparse.Namespace) -> Any:
    return mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=count,
        min_pose_detection_confidence=args.min_detection_confidence,
        min_pose_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        output_segmentation_masks=False,
    )


def crop_pixels(frame: np.ndarray, bbox: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a square pixel ROI and its normalized full-frame rectangle."""
    height, width = frame.shape[:2]
    x, y, w, h = (float(value) for value in bbox)
    center_x = (x + w / 2) * width
    center_y = (y + h / 2) * height
    side = max(w * width, h * height) * (1 + 2 * padding)
    side = max(2, min(int(round(side)), width, height))
    x1 = int(round(center_x - side / 2))
    y1 = int(round(center_y - side / 2))
    x1 = min(max(0, x1), width - side)
    y1 = min(max(0, y1), height - side)
    x2, y2 = x1 + side, y1 + side
    square_bbox = np.asarray([x1 / width, y1 / height, side / width, side / height])
    return np.ascontiguousarray(frame[y1:y2, x1:x2]), square_bbox


def crop_pose_rows(
    result: Any, frame_index: int, time_seconds: float, person: int,
    bbox: np.ndarray, track: Track,
) -> Iterable[dict[str, Any]]:
    if not result.pose_world_landmarks or not result.pose_landmarks:
        return
    world, normalized = result.pose_world_landmarks[0], result.pose_landmarks[0]
    bx, by, bw, bh = (float(value) for value in bbox)
    for index, point in enumerate(world):
        meta = normalized[index]
        yield {
            "frame": frame_index, "time": f"{time_seconds:.9f}", "person": person,
            "landmark": index, "landmark_name": POSE_NAMES[index],
            "x": f"{point.x:.9f}", "y": f"{point.y:.9f}", "z": f"{point.z:.9f}",
            "visibility": f"{float(getattr(meta, 'visibility', 1)):.6f}",
            "presence": f"{float(getattr(meta, 'presence', 1)):.6f}",
            "coordinate_space": "world",
            "image_x": f"{bx + float(meta.x) * bw:.9f}",
            "image_y": f"{by + float(meta.y) * bh:.9f}",
            "bbox_x": f"{bx:.9f}", "bbox_y": f"{by:.9f}",
            "bbox_w": f"{bw:.9f}", "bbox_h": f"{bh:.9f}",
            "track_score": f"{track.score:.6f}",
            "track_predicted": int(track.misses > 0),
        }


def extract_raw_csvs(
    video_path: Path, output_dir: Path, model_path: Path, args: argparse.Namespace,
) -> tuple[list[Path], float, int]:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("需要 OpenCV、MediaPipe 與 NumPy：pip install opencv-python mediapipe numpy") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV 無法開啟影片：{video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise RuntimeError("影片 FPS 無效")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"person{person:02d}_raw.csv" for person in range(1, args.num_people + 1)]
    files = [path.open("w", newline="", encoding="utf-8-sig") for path in paths]
    writers = [csv.DictWriter(handle, fieldnames=RAW_FIELDS) for handle in files]
    for writer in writers:
        writer.writeheader()
    tracks = [Track(person=index + 1) for index in range(args.num_people)]
    detection_count = max(args.num_people, 2) if args.target_shirt_color else args.num_people
    detector = mp.tasks.vision.PoseLandmarker.create_from_options(
        pose_options(mp, model_path, detection_count, args)
    )
    crop_landmarkers = [
        mp.tasks.vision.PoseLandmarker.create_from_options(pose_options(mp, model_path, 1, args))
        for _ in tracks
    ]
    frame_index = 0
    try:
        with detector:
            for landmarker in crop_landmarkers:
                landmarker.__enter__()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_ms = round(frame_index * 1000 / fps)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                full_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                full_result = detector.detect_for_video(full_image, timestamp_ms)
                detections = detections_from_result(full_result, frame, cv2)
                if args.target_shirt_color == "red":
                    red_detections = [
                        detection for detection in detections
                        if detection.target_color_score >= 0.05
                    ]
                    detections = (
                        [max(red_detections, key=lambda detection: detection.target_color_score)]
                        if red_detections else []
                    )
                update_tracks(tracks, detections, max_misses=args.max_track_gap)
                for index, track in enumerate(tracks):
                    if not track.active or track.misses > args.max_track_gap:
                        continue
                    crop, padded_bbox = crop_pixels(frame, track.bbox, args.crop_padding)
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop_rgb))
                    result = crop_landmarkers[index].detect_for_video(image, timestamp_ms)
                    writers[index].writerows(crop_pose_rows(
                        result, frame_index, frame_index / fps, track.person, padded_bbox, track
                    ))
                frame_index += 1
                if frame_index % args.progress_every == 0:
                    total = total_frames if total_frames > 0 else "?"
                    print(f"\r影片分析 {frame_index}/{total} 幀", end="", file=sys.stderr, flush=True)
    finally:
        capture.release()
        for landmarker in crop_landmarkers:
            try:
                landmarker.__exit__(None, None, None)
            except Exception:
                pass
        for handle in files:
            handle.close()
    if frame_index >= args.progress_every:
        print(file=sys.stderr)
    return paths, fps, frame_index


@dataclass
class PoseSeries:
    person: int
    frames: np.ndarray
    times: np.ndarray
    points: np.ndarray
    confidence: np.ndarray
    image_points: np.ndarray
    bboxes: np.ndarray
    track_scores: np.ndarray
    predicted: np.ndarray
    quality: np.ndarray


def read_raw_series(path: Path, total_frames: int, fps: float) -> PoseSeries:
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8-sig")))
    if not rows:
        raise ValueError(f"沒有偵測到人物資料：{path}")
    person = int(rows[0]["person"])
    frame_count = max(total_frames, max(int(row["frame"]) for row in rows) + 1)
    points = np.full((frame_count, 33, 3), np.nan)
    confidence = np.zeros((frame_count, 33), dtype=float)
    image_points = np.full((frame_count, 33, 2), np.nan)
    bboxes = np.full((frame_count, 4), np.nan)
    track_scores = np.zeros(frame_count, dtype=float)
    predicted = np.ones(frame_count, dtype=bool)
    for row in rows:
        frame, landmark = int(row["frame"]), int(row["landmark"])
        points[frame, landmark] = [float(row[axis]) for axis in "xyz"]
        visibility = float(row.get("visibility") or 0)
        presence = float(row.get("presence") or 0)
        confidence[frame, landmark] = min(visibility, presence)
        image_points[frame, landmark] = [float(row["image_x"]), float(row["image_y"])]
        bboxes[frame] = [float(row[f"bbox_{axis}"]) for axis in ("x", "y", "w", "h")]
        track_scores[frame] = float(row.get("track_score") or 0)
        predicted[frame] = bool(int(row.get("track_predicted") or 0))
    return PoseSeries(
        person=person,
        frames=np.arange(frame_count),
        times=np.arange(frame_count, dtype=float) / fps,
        points=points,
        confidence=confidence,
        image_points=image_points,
        bboxes=bboxes,
        track_scores=track_scores,
        predicted=predicted,
        quality=np.full((frame_count, 33), "raw", dtype=object),
    )


def fill_short_gaps(values: np.ndarray, max_gap: int, quality: np.ndarray | None = None) -> np.ndarray:
    """Linearly fill only NaN runs no longer than max_gap (first axis is time)."""
    result = values.copy()
    valid = np.all(np.isfinite(result), axis=tuple(range(1, result.ndim)))
    index = 0
    while index < len(valid):
        if valid[index]:
            index += 1
            continue
        start = index
        while index < len(valid) and not valid[index]:
            index += 1
        end = index
        if start > 0 and end < len(valid) and end - start <= max_gap:
            for frame in range(start, end):
                fraction = (frame - start + 1) / (end - start + 1)
                result[frame] = (1 - fraction) * result[start - 1] + fraction * result[end]
                if quality is not None:
                    quality[frame] = "interpolated"
    return result


def nearest_fill(values: np.ndarray, quality: np.ndarray | None = None) -> np.ndarray:
    result = values.copy()
    valid_indices = np.flatnonzero(np.all(np.isfinite(result), axis=tuple(range(1, result.ndim))))
    if not len(valid_indices):
        return result
    for frame in range(len(result)):
        if np.all(np.isfinite(result[frame])):
            continue
        nearest = valid_indices[np.argmin(np.abs(valid_indices - frame))]
        result[frame] = result[nearest]
        if quality is not None:
            quality[frame] = "long_gap_filled"
    return result


def fill_remaining_gaps(values: np.ndarray, quality: np.ndarray | None = None) -> np.ndarray:
    """Fill long gaps linearly between valid endpoints and hold only at edges."""
    result = values.copy()
    axes = tuple(range(1, result.ndim))
    valid_indices = np.flatnonzero(np.all(np.isfinite(result), axis=axes))
    if not len(valid_indices):
        return result
    first, last = int(valid_indices[0]), int(valid_indices[-1])
    for frame in range(0, first):
        result[frame] = result[first]
        if quality is not None:
            quality[frame] = "long_gap_filled"
    for frame in range(last + 1, len(result)):
        result[frame] = result[last]
        if quality is not None:
            quality[frame] = "long_gap_filled"
    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        left, right = int(left), int(right)
        if right == left + 1:
            continue
        for frame in range(left + 1, right):
            fraction = (frame - left) / (right - left)
            result[frame] = (1 - fraction) * result[left] + fraction * result[right]
            if quality is not None:
                quality[frame] = (
                    "invalid_pose_filled"
                    if str(quality[frame]) == "invalid_pose"
                    else "long_gap_filled"
                )
    return result


def anatomically_invalid_frames(series: PoseSeries, min_span: float) -> np.ndarray:
    """Find upside-down/hallucinated poses using full-image landmark ordering."""
    image = series.image_points
    required = (11, 12, 23, 24, 27, 28)
    observed = np.all(np.isfinite(image[:, required]), axis=(1, 2))
    # Missing pairs remain NaN and are ignored by the `observed` mask below.
    shoulder_y = np.mean(image[:, (11, 12), 1], axis=1)
    hip_y = np.mean(image[:, (23, 24), 1], axis=1)
    ankle_y = np.mean(image[:, (27, 28), 1], axis=1)
    upright = (shoulder_y < hip_y - min_span) & (ankle_y > hip_y + min_span)
    return observed & ~upright


def bidirectional_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Zero-phase-like EMA: smooth forward/backward and average to reduce lag."""
    if alpha <= 0 or len(values) < 2:
        return values.copy()
    forward = values.copy()
    for index in range(1, len(forward)):
        if np.all(np.isfinite(forward[index - 1])) and np.all(np.isfinite(forward[index])):
            forward[index] = alpha * forward[index] + (1 - alpha) * forward[index - 1]
    backward = values.copy()
    for index in range(len(backward) - 2, -1, -1):
        if np.all(np.isfinite(backward[index + 1])) and np.all(np.isfinite(backward[index])):
            backward[index] = alpha * backward[index] + (1 - alpha) * backward[index + 1]
    return np.where(np.isfinite(forward) & np.isfinite(backward), (forward + backward) / 2, values)


def preprocess_series(series: PoseSeries, args: argparse.Namespace) -> PoseSeries:
    points = series.points.copy()
    image_points = series.image_points.copy()
    quality = np.full(points.shape[:2], "measured", dtype=object)
    invalid_pose = (
        np.zeros(len(series.frames), dtype=bool)
        if args.allow_inverted_poses
        else anatomically_invalid_frames(series, args.min_upright_image_span)
    )
    points[invalid_pose] = np.nan
    image_points[invalid_pose] = np.nan
    quality[invalid_pose] = "invalid_pose"
    low = (series.confidence < args.min_landmark_confidence) | ~np.all(np.isfinite(points), axis=2)
    points[low] = np.nan
    quality[low & ~invalid_pose[:, None]] = "low_confidence"

    # Hampel-style spatial test, independently for every landmark trajectory.
    window = args.outlier_window
    for landmark in range(33):
        trajectory = points[:, landmark]
        for frame in range(len(trajectory)):
            if not np.all(np.isfinite(trajectory[frame])):
                continue
            left, right = max(0, frame - window), min(len(trajectory), frame + window + 1)
            neighborhood = trajectory[left:right]
            neighborhood = neighborhood[np.all(np.isfinite(neighborhood), axis=1)]
            if len(neighborhood) < 3:
                continue
            median = np.median(neighborhood, axis=0)
            distances = np.linalg.norm(neighborhood - median, axis=1)
            mad = float(np.median(np.abs(distances - np.median(distances))))
            limit = max(args.outlier_threshold_m, 3.5 * 1.4826 * mad)
            if float(np.linalg.norm(trajectory[frame] - median)) > limit:
                trajectory[frame] = np.nan
                quality[frame, landmark] = "outlier"

        trajectory = fill_short_gaps(trajectory, args.max_interpolation_gap, quality[:, landmark])
        if not args.keep_long_gaps:
            trajectory = fill_remaining_gaps(trajectory, quality[:, landmark])
        points[:, landmark] = bidirectional_ema(trajectory, args.smoothing_alpha)

    # Image trajectories drive formation; smooth them but never mix landmark identities.
    for landmark in range(33):
        image_points[:, landmark] = bidirectional_ema(
            fill_remaining_gaps(
                fill_short_gaps(image_points[:, landmark], args.max_interpolation_gap)
            ),
            args.smoothing_alpha,
        )
    bboxes = bidirectional_ema(
        nearest_fill(fill_short_gaps(series.bboxes, args.max_interpolation_gap)),
        args.smoothing_alpha,
    )
    return PoseSeries(
        series.person, series.frames, series.times, points, series.confidence,
        image_points, bboxes, series.track_scores, series.predicted, quality,
    )


def write_processed_csv(path: Path, series: PoseSeries) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROCESSED_FIELDS)
        writer.writeheader()
        for frame in range(len(series.frames)):
            for landmark in range(33):
                point = series.points[frame, landmark]
                if not np.all(np.isfinite(point)):
                    continue
                image = series.image_points[frame, landmark]
                bbox = series.bboxes[frame]
                writer.writerow({
                    "frame": frame, "time": f"{series.times[frame]:.9f}",
                    "person": series.person, "landmark": landmark,
                    "landmark_name": POSE_NAMES[landmark],
                    "x": f"{point[0]:.9f}", "y": f"{point[1]:.9f}", "z": f"{point[2]:.9f}",
                    "visibility": f"{series.confidence[frame, landmark]:.6f}",
                    "presence": f"{series.confidence[frame, landmark]:.6f}",
                    "coordinate_space": "world",
                    "image_x": f"{image[0]:.9f}", "image_y": f"{image[1]:.9f}",
                    "bbox_x": f"{bbox[0]:.9f}", "bbox_y": f"{bbox[1]:.9f}",
                    "bbox_w": f"{bbox[2]:.9f}", "bbox_h": f"{bbox[3]:.9f}",
                    "track_score": f"{series.track_scores[frame]:.6f}",
                    "track_predicted": int(series.predicted[frame]),
                    "quality": series.quality[frame, landmark],
                })


def import_converter() -> Any:
    converter_dir = PROJECT_DIR / "csv_to_trc"
    if str(converter_dir) not in sys.path:
        sys.path.insert(0, str(converter_dir))
    import CSVToTRC as converter
    return converter


def series_to_csv_data(series: PoseSeries, converter: Any) -> Any:
    keys = [converter.MarkerKey(0, index, POSE_NAMES[index], "pose") for index in range(33)]
    points: dict[int, dict[Any, tuple[float, float, float]]] = {}
    for frame in range(len(series.frames)):
        frame_points = {}
        for index, key in enumerate(keys):
            value = series.points[frame, index]
            if np.all(np.isfinite(value)):
                frame_points[key] = tuple(float(axis) for axis in value)
        if frame_points:
            points[frame] = frame_points
    return converter.CsvData(
        points=points,
        times={frame: float(time) for frame, time in enumerate(series.times)},
        handedness={}, kind="pose", coordinate_space="world",
    )


def normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        raise ValueError("無法從重合的點建立身體局部座標系")
    return vector / length


def body_basis(origin: np.ndarray, distal: np.ndarray, lateral_hint: np.ndarray) -> np.ndarray:
    longitudinal = normalize(distal - origin)
    lateral = lateral_hint - float(np.dot(lateral_hint, longitudinal)) * longitudinal
    if float(np.linalg.norm(lateral)) < 1e-8:
        fallback = np.asarray([0.0, 0.0, 1.0])
        lateral = fallback - float(np.dot(fallback, longitudinal)) * longitudinal
    lateral = normalize(lateral)
    forward = normalize(np.cross(lateral, longitudinal))
    lateral = normalize(np.cross(longitudinal, forward))
    return np.stack([longitudinal, lateral, forward], axis=0)


def midpoint(points: dict[int, np.ndarray], left: int, right: int) -> np.ndarray:
    return (points[left] + points[right]) / 2


def skeleton_frames(points: dict[int, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pelvis = midpoint(points, 23, 24)
    shoulders = midpoint(points, 11, 12)
    lateral = points[24] - points[23]
    frames: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "torso": (pelvis, body_basis(pelvis, shoulders, lateral)),
        "pelvis": (pelvis, body_basis(pelvis, shoulders, lateral)),
        "head": (shoulders, body_basis(shoulders, points[0], points[12] - points[11])),
    }
    for side, hip, knee, ankle, heel, toe in (
        ("L", 23, 25, 27, 29, 31),
        ("R", 24, 26, 28, 30, 32),
    ):
        frames[f"{side}_thigh"] = (points[hip], body_basis(points[hip], points[knee], lateral))
        frames[f"{side}_shank"] = (points[knee], body_basis(points[knee], points[ankle], lateral))
        foot_direction = points[toe] - points[heel]
        frames[f"{side}_foot"] = (points[ankle], body_basis(points[ankle], points[ankle] + foot_direction, lateral))
    for side, shoulder, elbow, wrist, hand in (
        ("L", 11, 13, 15, 19),
        ("R", 12, 14, 16, 20),
    ):
        if not all(index in points for index in (shoulder, elbow, wrist, hand)):
            continue
        frames[f"{side}_upper_arm"] = (
            points[shoulder], body_basis(points[shoulder], points[elbow], lateral)
        )
        frames[f"{side}_forearm"] = (
            points[elbow], body_basis(points[elbow], points[wrist], lateral)
        )
        pinky = 17 if side == "L" else 18
        thumb = 21 if side == "L" else 22
        if pinky in points and thumb in points:
            hand_center = (points[hand] + points[pinky] + points[thumb]) / 3
            palm_lateral = points[thumb] - points[pinky]
            if float(np.dot(palm_lateral, lateral)) < 0:
                palm_lateral = -palm_lateral
        else:
            hand_center = points[hand]
            palm_lateral = lateral
        frames[f"{side}_hand"] = (
            points[wrist], body_basis(points[wrist], hand_center, palm_lateral)
        )
    return frames


def marker_segment(name: str) -> str | None:
    if name in {
        "Sternum", "L.Acromium", "R.Acromium", "C7", "T10", "Clavicle",
        "LSJC", "RSJC",
    }:
        return "torso"
    if name == "Top.Head":
        return "head"
    if name in {"L.ASIS", "R.ASIS", "V.Sacral", "LHJC", "RHJC"}:
        return "pelvis"
    for side in ("L", "R"):
        upper_arm = {
            f"{side}.Bicep", f"{side}UA1", f"{side}UA2", f"{side}UA3",
            f"{side}UAmedial", f"{side}.Elbow", f"{side}MEL", f"{side}EJC",
        }
        forearm = {
            f"{side}FAsuperior", f"{side}FAradius", f"{side}FAulna",
            f"{side}.Wrist.Med", f"{side}.Wrist.Lat", f"{side}WJC",
        }
        hand = {f"{side}HNknuckle", f"{side}HNulna", f"{side}HNradius"}
        if name in upper_arm:
            return f"{side}_upper_arm"
        if name in forearm:
            return f"{side}_forearm"
        if name in hand:
            return f"{side}_hand"
        if name == f"{side}KJC":
            return f"{side}_thigh"
        if name == f"{side}AJC":
            return f"{side}_shank"
        if name.startswith(f"{side}.Thigh") or name.startswith(f"{side}.Knee"):
            return f"{side}_thigh"
        if name.startswith(f"{side}.Shank") or name.startswith(f"{side}.Ankle"):
            return f"{side}_shank"
        if name.startswith(f"{side}.Heel") or name.startswith(f"{side}.Midfoot") or name.startswith(f"{side}.Toe"):
            return f"{side}_foot"
    return None


def model_landmark_points(marker_positions: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    def require(name: str) -> np.ndarray:
        if name not in marker_positions:
            raise ValueError(f"模型缺少必要 marker：{name}")
        return marker_positions[name]

    shoulders = (require("L.Acromium") + require("R.Acromium")) / 2
    if "Top.Head" in marker_positions:
        head_reference = marker_positions["Top.Head"]
    elif "C7" in marker_positions and "Sternum" in marker_positions:
        # Hamner has no head marker. Build a virtual model reference from its
        # upper torso; MediaPipe NOSE supplies the moving reference per frame.
        torso_up = normalize(marker_positions["C7"] - marker_positions["Sternum"])
        shoulder_width = float(np.linalg.norm(require("R.Acromium") - require("L.Acromium")))
        head_reference = marker_positions["C7"] + torso_up * (0.75 * shoulder_width)
    else:
        torso_up = normalize(shoulders - require("Sternum"))
        shoulder_width = float(np.linalg.norm(require("R.Acromium") - require("L.Acromium")))
        head_reference = shoulders + torso_up * (0.75 * shoulder_width)

    result = {
        0: head_reference,
        11: require("L.Acromium"), 12: require("R.Acromium"),
        23: require("L.ASIS"), 24: require("R.ASIS"),
        25: (require("L.Knee.Lat") + require("L.Knee.Med")) / 2,
        26: (require("R.Knee.Lat") + require("R.Knee.Med")) / 2,
        27: (require("L.Ankle.Lat") + require("L.Ankle.Med")) / 2,
        28: (require("R.Ankle.Lat") + require("R.Ankle.Med")) / 2,
        29: require("L.Heel"), 30: require("R.Heel"),
        31: require("L.Toe.Tip"), 32: require("R.Toe.Tip"),
    }
    upper_names = {
        13: "LEJC", 14: "REJC", 15: "LWJC", 16: "RWJC",
        19: "LHNknuckle", 20: "RHNknuckle",
    }
    if all(name in marker_positions for name in upper_names.values()):
        result.update({index: marker_positions[name] for index, name in upper_names.items()})
    return result


def load_model_markers(model_path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    try:
        import opensim as osim
    except ImportError as exc:
        raise RuntimeError("模型貼合需要 OpenSim Python 套件") from exc
    model = osim.Model(str(native_model_path(model_path)))
    state = model.initSystem()
    marker_set = model.getMarkerSet()
    names: list[str] = []
    positions: dict[str, np.ndarray] = {}
    for index in range(marker_set.getSize()):
        marker = marker_set.get(index)
        name = str(marker.getName())
        location = marker.getLocationInGround(state)
        names.append(name)
        positions[name] = np.asarray([location.get(0), location.get(1), location.get(2)], dtype=float)
    if not names:
        raise ValueError("OpenSim 模型 MarkerSet 是空的")
    return names, positions


def fitted_pose_array(data: Any, frame_count: int) -> np.ndarray:
    result = np.full((frame_count, 33, 3), np.nan)
    for frame, frame_points in data.points.items():
        for key, point in frame_points.items():
            if key.kind == "pose" and 0 <= key.landmark < 33:
                result[frame, key.landmark] = point
    return result


def fit_pose_to_model(
    series: PoseSeries, model_path: Path, calibration_frames: int,
) -> tuple[np.ndarray, Any]:
    converter = import_converter()
    complete = series.points.copy()
    for landmark in range(33):
        complete[:, landmark] = nearest_fill(complete[:, landmark])
    if not np.all(np.isfinite(complete)):
        missing = [POSE_NAMES[index] for index in range(33) if not np.any(np.isfinite(complete[:, index]))]
        raise ValueError(f"person{series.person:02d} 完全缺少 landmarks：{missing}")
    fit_series = PoseSeries(
        series.person, series.frames, series.times, complete, series.confidence,
        series.image_points, series.bboxes, series.track_scores, series.predicted,
        series.quality,
    )
    data = series_to_csv_data(fit_series, converter)
    fitted, summary = converter.auto_fit_pose_to_model(
        data, model_path=model_path, person=0,
        calibration_frames=calibration_frames,
        anchor_each_frame=False, marker_map_path=None,
    )
    return fitted_pose_array(fitted, len(series.frames)), summary


def create_marker_templates(
    marker_order: list[str], marker_positions: dict[str, np.ndarray],
) -> tuple[list[str], dict[str, tuple[str, np.ndarray]]]:
    model_points = model_landmark_points(marker_positions)
    model_frames = skeleton_frames(model_points)
    templates: dict[str, tuple[str, np.ndarray]] = {}
    supported: list[str] = []
    for name in marker_order:
        segment = marker_segment(name)
        if segment is None:
            continue
        if segment not in model_frames:
            continue
        origin, basis = model_frames[segment]
        local = (marker_positions[name] - origin) @ basis.T
        templates[name] = segment, local
        supported.append(name)
    if len(supported) < 12:
        raise ValueError(f"模型可推估的 markers 太少（{len(supported)}）")
    return supported, templates


def augment_markers(
    fitted_pose: np.ndarray, marker_names: list[str],
    templates: dict[str, tuple[str, np.ndarray]],
    validity_pose: np.ndarray | None = None,
) -> np.ndarray:
    output = np.full((len(fitted_pose), len(marker_names), 3), np.nan)
    needed = (0, 11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
    optional = (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
    previous_bases: dict[str, np.ndarray] = {}
    for frame in range(len(fitted_pose)):
        if not all(np.all(np.isfinite(fitted_pose[frame, index])) for index in needed):
            continue
        points = {
            index: fitted_pose[frame, index]
            for index in needed + optional
            if np.all(np.isfinite(fitted_pose[frame, index]))
        }
        try:
            frames = skeleton_frames(points)
        except ValueError:
            continue
        # The cross-product frame around a long bone has a 180-degree sign
        # ambiguity. Keep lateral/forward axes continuous so synthetic marker
        # clusters cannot flip around the bone between adjacent frames.
        continuous_frames = {}
        for segment, (origin, basis) in frames.items():
            previous = previous_bases.get(segment)
            if previous is not None and float(np.dot(basis[1], previous[1])) < 0:
                basis = basis.copy()
                basis[1:] *= -1
            continuous_frames[segment] = origin, basis
            previous_bases[segment] = basis
        frames = continuous_frames
        for marker_index, name in enumerate(marker_names):
            segment, local = templates[name]
            if validity_pose is not None:
                requirements = {
                    "torso": (11, 12, 23, 24), "pelvis": (11, 12, 23, 24),
                    "head": (0, 11, 12),
                    "L_upper_arm": (11, 13), "R_upper_arm": (12, 14),
                    "L_forearm": (13, 15), "R_forearm": (14, 16),
                    "L_hand": (15, 19), "R_hand": (16, 20),
                    "L_thigh": (23, 25), "R_thigh": (24, 26),
                    "L_shank": (25, 27), "R_shank": (26, 28),
                    "L_foot": (27, 29, 31), "R_foot": (28, 30, 32),
                }[segment]
                if not all(np.all(np.isfinite(validity_pose[frame, index])) for index in requirements):
                    continue
            origin, basis = frames[segment]
            output[frame, marker_index] = origin + local @ basis
    return output


def filter_marker_spikes(
    markers: np.ndarray, fps: float, *, max_speed: float, max_acceleration: float,
) -> tuple[np.ndarray, int]:
    """Remove isolated impossible marker velocity/acceleration spikes."""
    result = markers.copy()
    rejected = 0
    for _ in range(2):
        changed = False
        for marker in range(result.shape[1]):
            trajectory = result[:, marker]
            for frame in range(1, len(trajectory) - 1):
                if not np.all(np.isfinite(trajectory[frame - 1:frame + 2])):
                    continue
                before = float(np.linalg.norm(trajectory[frame] - trajectory[frame - 1]) * fps)
                after = float(np.linalg.norm(trajectory[frame + 1] - trajectory[frame]) * fps)
                acceleration = float(
                    np.linalg.norm(trajectory[frame + 1] - 2 * trajectory[frame] + trajectory[frame - 1])
                    * fps * fps
                )
                neighbor_speed = float(
                    np.linalg.norm(trajectory[frame + 1] - trajectory[frame - 1]) * fps / 2
                )
                isolated_spike = (
                    min(before, after) > max_speed
                    and acceleration > max_acceleration
                    and neighbor_speed < max(before, after) * 0.65
                )
                if isolated_spike:
                    trajectory[frame] = np.nan
                    rejected += 1
                    changed = True
            result[:, marker] = fill_remaining_gaps(trajectory)
        if not changed:
            break
    return result, rejected


def apply_ground_correction(
    markers: np.ndarray, marker_names: list[str], marker_positions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Translate all markers in Y so the lowest support-foot marker stays grounded."""
    foot_names = [
        name for name in marker_names
        if any(token in name for token in ("Heel", "Toe.Tip", "Midfoot.Lat", "Toe.Lat", "Toe.Med"))
    ]
    indices = [marker_names.index(name) for name in foot_names]
    if not indices:
        return markers.copy(), np.zeros(len(markers))
    ground_y = min(float(marker_positions[name][1]) for name in foot_names)
    observed_floor = np.nanmin(markers[:, indices, 1], axis=1)
    correction = ground_y - observed_floor
    result = markers.copy()
    result[:, :, 1] += correction[:, None]
    return result, correction


MODEL_ALIGNMENT_MARKERS = (
    "Sternum", "C7", "L.Acromium", "R.Acromium",
    "L.ASIS", "R.ASIS", "V.Sacral",
)


def align_markers_to_model_reference(
    markers: np.ndarray,
    marker_names: list[str],
    marker_positions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Translate a trial so its first-frame torso/pelvis centroid matches the model."""
    available = [
        name for name in MODEL_ALIGNMENT_MARKERS
        if name in marker_names and name in marker_positions
    ]
    if not available:
        return markers.copy(), np.zeros(3)
    indices = [marker_names.index(name) for name in available]
    observed = markers[0, indices]
    valid = np.all(np.isfinite(observed), axis=1)
    if not np.any(valid):
        return markers.copy(), np.zeros(3)
    targets = np.stack([marker_positions[name] for name in available], axis=0)[valid]
    offset = targets.mean(axis=0) - observed[valid].mean(axis=0)
    return markers + offset[None, None, :], offset


def load_stage_homography(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    image = np.asarray(payload["image_points"], dtype=float)
    stage = np.asarray(payload["stage_points_m"], dtype=float)
    if image.shape != stage.shape or image.ndim != 2 or image.shape[1] != 2 or len(image) < 4:
        raise ValueError("stage calibration 需至少四組 image_points 與 stage_points_m")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("stage calibration 需要 OpenCV") from exc
    matrix, _ = cv2.findHomography(image, stage)
    if matrix is None:
        raise ValueError("舞台標定點無法計算 homography")
    return matrix


def apply_homography(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = matrix @ np.asarray([point[0], point[1], 1.0])
    return homogeneous[:2] / homogeneous[2]


def formation_trajectory(
    series: PoseSeries, args: argparse.Namespace, homography: np.ndarray | None,
) -> np.ndarray:
    result = np.full((len(series.frames), 3), np.nan)
    for frame in range(len(series.frames)):
        image = series.image_points[frame]
        hip = np.nanmean(image[[23, 24]], axis=0)
        feet = image[[27, 28, 29, 30, 31, 32]]
        ground_y = float(np.nanmax(feet[:, 1])) if np.any(np.isfinite(feet[:, 1])) else np.nan
        if not np.all(np.isfinite(hip)) or not math.isfinite(ground_y):
            bbox = series.bboxes[frame]
            if not np.all(np.isfinite(bbox)):
                continue
            hip = np.asarray([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] * 0.55])
            ground_y = bbox[1] + bbox[3]
        if homography is not None:
            stage_xz = apply_homography(np.asarray([hip[0], ground_y]), homography)
            depth, horizontal = float(stage_xz[0]), float(stage_xz[1])
        else:
            # OpenSim convention used here: X=stage depth, Y=up, Z=stage horizontal.
            depth = (0.85 - ground_y) * args.stage_depth_m
            horizontal = (hip[0] - 0.5) * args.stage_width_m
        if args.mirror_horizontal:
            horizontal *= -1
        result[frame] = [depth, 0.0, horizontal]
    result = nearest_fill(fill_short_gaps(result, args.max_interpolation_gap))
    return bidirectional_ema(result, args.smoothing_alpha)


def center_formations(trajectories: list[np.ndarray]) -> None:
    if not trajectories:
        return
    frame_count = min(len(item) for item in trajectories)
    reference = None
    for frame in range(frame_count):
        values = np.stack([item[frame] for item in trajectories])
        if np.all(np.isfinite(values)):
            reference = values.mean(axis=0)
            break
    if reference is None:
        return
    for trajectory in trajectories:
        trajectory -= reference


def format_number(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def write_marker_csv(
    path: Path, person: int, times: np.ndarray,
    marker_names: list[str], markers: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=MARKER_FIELDS)
        writer.writeheader()
        for frame in range(len(markers)):
            for index, name in enumerate(marker_names):
                point = markers[frame, index]
                if not np.all(np.isfinite(point)):
                    continue
                writer.writerow({
                    "frame": frame + 1, "time": format_number(times[frame] - times[0]),
                    "person": person, "marker": name,
                    "x": format_number(point[0]), "y": format_number(point[1]),
                    "z": format_number(point[2]), "units": "m",
                })


def write_trc(
    path: Path, fps: float, times: np.ndarray,
    marker_names: list[str], markers: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["PathFileType", "4", "(X/Y/Z)", path.name])
        writer.writerow([
            "DataRate", "CameraRate", "NumFrames", "NumMarkers", "Units",
            "OrigDataRate", "OrigDataStartFrame", "OrigNumFrames",
        ])
        writer.writerow([
            format_number(fps), format_number(fps), len(markers), len(marker_names),
            "m", format_number(fps), 1, len(markers),
        ])
        marker_header: list[str] = ["Frame#", "Time"]
        coordinate_header: list[str] = ["", ""]
        for index, name in enumerate(marker_names, start=1):
            marker_header.extend([name, "", ""])
            coordinate_header.extend([f"X{index}", f"Y{index}", f"Z{index}"])
        writer.writerow(marker_header)
        writer.writerow(coordinate_header)
        writer.writerow([])
        for frame in range(len(markers)):
            row: list[str | int] = [frame + 1, format_number(times[frame] - times[0])]
            for point in markers[frame]:
                row.extend(format_number(axis) for axis in point)
            writer.writerow(row)


def probe_video(path: Path) -> tuple[float, int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("需要 OpenCV 讀取影片資訊") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV 無法開啟影片：{path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0:
        raise RuntimeError("影片 FPS 無效")
    return fps, frames


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_path = resolve_dataset(args.input)
    model_path = args.fit_model.expanduser().resolve()
    pose_model = (args.model.expanduser().resolve() if args.model else DEFAULT_POSE_MODEL)
    output_dir = (
        resolve_dataset(args.output_dir) if args.output_dir
        else DATASET_DIR / f"{video_path.stem}_dance"
    )
    try:
        if not video_path.is_file() or video_path.suffix.lower() != ".mp4":
            raise FileNotFoundError(f"找不到 MP4：{video_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"找不到 OpenSim 模型：{model_path}")
        if not pose_model.is_file():
            raise FileNotFoundError(f"找不到 MediaPipe 模型：{pose_model}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.from_raw:
            fps, total_frames = probe_video(video_path)
            raw_paths = [output_dir / f"person{person:02d}_raw.csv" for person in range(1, args.num_people + 1)]
            missing = [str(path) for path in raw_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"缺少 raw CSV：{missing}")
        else:
            raw_paths, fps, total_frames = extract_raw_csvs(
                video_path, output_dir, pose_model, args
            )

        processed: list[PoseSeries] = []
        for raw_path in raw_paths:
            series = preprocess_series(read_raw_series(raw_path, total_frames, fps), args)
            write_processed_csv(output_dir / f"person{series.person:02d}_clean.csv", series)
            processed.append(series)

        marker_order, marker_positions = load_model_markers(model_path)
        marker_names, templates = create_marker_templates(marker_order, marker_positions)
        homography = (
            load_stage_homography(args.stage_calibration)
            if args.enable_global_formation else None
        )
        if args.enable_global_formation:
            formations = [
                formation_trajectory(series, args, homography)
                for series in processed
            ]
            if not args.no_center_formation:
                center_formations(formations)
        else:
            formations = [
                np.zeros((len(series.frames), 3), dtype=float)
                for series in processed
            ]

        summaries = []
        for series, formation in zip(processed, formations):
            fitted_pose, fit_summary = fit_pose_to_model(
                series, model_path, args.calibration_frames
            )
            markers = augment_markers(
                fitted_pose, marker_names, templates,
                validity_pose=series.points if args.keep_long_gaps else None,
            )
            markers, rejected_spikes = filter_marker_spikes(
                markers, fps,
                max_speed=args.max_marker_speed_mps,
                max_acceleration=args.max_marker_acceleration_mps2,
            )
            ground_shift = np.zeros(len(markers))
            if not args.no_ground_correction:
                markers, ground_shift = apply_ground_correction(
                    markers, marker_names, marker_positions
                )
            markers, model_alignment_offset = align_markers_to_model_reference(
                markers, marker_names, marker_positions
            )
            markers += formation[:, None, :]
            marker_csv = output_dir / f"person{series.person:02d}_markers.csv"
            trc_path = output_dir / f"person{series.person:02d}.trc"
            write_marker_csv(marker_csv, series.person, series.times, marker_names, markers)
            write_trc(trc_path, fps, series.times, marker_names, markers)
            summaries.append({
                "person": series.person,
                "raw_csv": str(raw_paths[series.person - 1]),
                "clean_csv": str(output_dir / f"person{series.person:02d}_clean.csv"),
                "marker_csv": str(marker_csv), "trc": str(trc_path),
                "model_markers": marker_names,
                "fit_scale": fit_summary.scale_factor,
                "fit_rms_m": fit_summary.calibration_rms,
                "rejected_marker_spikes": rejected_spikes,
                "ground_correction_y_min_m": float(np.min(ground_shift)),
                "ground_correction_y_max_m": float(np.max(ground_shift)),
                "model_alignment_offset_m": [
                    float(value) for value in model_alignment_offset
                ],
            })

        manifest = {
            "video": str(video_path), "fps": fps, "frames": total_frames,
            "people": args.num_people, "units": "m",
            "opensim_axes": {"X": "stage depth", "Y": "up", "Z": "stage horizontal"},
            "formation_method": (
                "disabled" if not args.enable_global_formation
                else ("homography" if homography is not None else "linear monocular approximation")
            ),
            "formation_centered": (
                args.enable_global_formation and not args.no_center_formation
            ),
            "outputs": summaries,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError, KeyError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1

    print(f"完成：{args.num_people} 人舞蹈 CSV/TRC 已輸出至 {output_dir}")
    for summary in summaries:
        print(f"person{summary['person']:02d}: {summary['trc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



