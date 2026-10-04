#!/usr/bin/env python3
"""Convert MediaPipe hand or whole-body CSV data to an OpenSim TRC file.

Optional model fitting:
- Hand mode retargets all 20 hand segments.
- Pose mode recognizes direct MediaPipe marker names and common gait model
  aliases, including medial/lateral knee and ankle marker midpoints.
- Uses stable calibration frames to estimate global similarity and available
  model segment lengths/directions.
- Preserves wrist or pelvis translation by default.
- Interpolates missing samples to keep marker trajectories continuous.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from native_paths import model_path as native_model_path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "dataset"

REQUIRED_COLUMNS = {
    "frame",
    "time",
    "landmark",
    "landmark_name",
    "x",
    "y",
    "z",
}

PALM_FIT_LANDMARKS = (0, 5, 9, 17)
HAND_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
ALL_HAND_LANDMARKS = tuple(range(21))

POSE_LANDMARK_NAMES = (
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
POSE_FIT_PRIORITY = (11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
POSE_CHAINS = (
    (11, 13, 15, 19),
    (12, 14, 16, 20),
    (23, 25, 27, 31),
    (24, 26, 28, 32),
    (27, 29),
    (28, 30),
)

# Each inner tuple is one possible model-marker group. Multiple names in a
# group are averaged to approximate the MediaPipe joint center.
POSE_MARKER_ALIASES = {
    0: (("NOSE",), ("Top.Head",)),
    11: (("LEFT_SHOULDER",), ("L.Acromium",), ("LACR",)),
    12: (("RIGHT_SHOULDER",), ("R.Acromium",), ("RACR",)),
    13: (("LEFT_ELBOW",), ("LEJC",), ("L.Elbow", "LMEL")),
    14: (("RIGHT_ELBOW",), ("REJC",), ("R.Elbow", "RMEL")),
    15: (("LEFT_WRIST",), ("LWJC",), ("L.Wrist.Med", "L.Wrist.Lat")),
    16: (("RIGHT_WRIST",), ("RWJC",), ("R.Wrist.Med", "R.Wrist.Lat")),
    19: (("LEFT_INDEX",), ("LHNknuckle",)),
    20: (("RIGHT_INDEX",), ("RHNknuckle",)),
    23: (("LEFT_HIP",), ("L.ASIS",), ("LASI",)),
    24: (("RIGHT_HIP",), ("R.ASIS",), ("RASI",)),
    25: (("LEFT_KNEE",), ("L.Knee.Lat", "L.Knee.Med"), ("LKNE",)),
    26: (("RIGHT_KNEE",), ("R.Knee.Lat", "R.Knee.Med"), ("RKNE",)),
    27: (("LEFT_ANKLE",), ("L.Ankle.Lat", "L.Ankle.Med"), ("LANK",)),
    28: (("RIGHT_ANKLE",), ("R.Ankle.Lat", "R.Ankle.Med"), ("RANK",)),
    29: (("LEFT_HEEL",), ("L.Heel",), ("LHEE",)),
    30: (("RIGHT_HEEL",), ("R.Heel",), ("RHEE",)),
    31: (("LEFT_FOOT_INDEX",), ("L.Toe.Tip",), ("LTOE",)),
    32: (("RIGHT_FOOT_INDEX",), ("R.Toe.Tip",), ("RTOE",)),
}


@dataclass(frozen=True, order=True)
class MarkerKey:
    hand: int
    landmark: int
    landmark_name: str
    kind: str = "hand"

    @property
    def track(self) -> int:
        return self.hand


@dataclass
class CsvData:
    points: dict[int, dict[MarkerKey, tuple[float, float, float]]]
    times: dict[int, float]
    handedness: dict[int, Counter[str]]
    kind: str = "hand"
    coordinate_space: str = "normalized"


@dataclass(frozen=True)
class FitSummary:
    model_path: Path
    marker_names: tuple[str, ...]
    scale_factor: float
    calibration_rms: float
    anchored_each_frame: bool
    retargeted_segments: int
    kind: str = "hand"
    matched_markers: int = 21


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必須是大於 0 的有限數字")
    return number


def finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("必須是有限數字")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必須是大於或等於 0 的整數")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必須是大於 0 的整數")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="將 MediaPipe 手部或全身 landmarks CSV 自動貼合 OpenSim 模型並輸出 TRC。",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="dataset 內的 CSV 檔名，也可使用絕對路徑",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="輸出 TRC 檔名；相對路徑會寫入 dataset，預設為輸入 CSV 的同檔名 .trc",
    )
    parser.add_argument(
        "--data-rate",
        type=positive_float,
        help="取樣率（Hz）；預設從 frame 與 time 自動推算",
    )
    parser.add_argument(
        "--units",
        default="m",
        help="寫入 TRC 標頭的單位（預設：m）",
    )
    parser.add_argument(
        "--scale",
        type=finite_float,
        default=1.0,
        help="在自動對齊前套用的 x、y、z 整體倍率（預設：1）",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="反轉 y 軸方向，將影像向下為正改成向上為正",
    )
    parser.add_argument("--flip-z", action="store_true", help="反轉 z 軸方向")
    parser.add_argument(
        "--fit-model",
        type=Path,
        help=(
            "讀取含同名 markers 的 .osim 模型，自動計算每支影片的固定旋轉、"
            "縮放與位置"
        ),
    )
    parser.add_argument(
        "--fit-index",
        "--fit-hand",
        "--fit-person",
        dest="fit_index",
        type=nonnegative_int,
        default=0,
        help="要自動對齊的 hand/person 編號（預設：0）",
    )
    parser.add_argument(
        "--pose-marker-map",
        type=Path,
        help=(
            "選用的 JSON 對應檔：MediaPipe landmark 名稱對到模型 marker "
            "名稱或名稱陣列；只用於全身模式"
        ),
    )
    parser.add_argument(
        "--calibration-frames",
        type=positive_int,
        help="使用開頭多少個完整偵測幀估計對齊（預設：手部 30、全身 1）",
    )
    root_group = parser.add_mutually_exclusive_group()
    root_group.add_argument(
        "--keep-root-motion",
        "--keep-wrist-motion",
        dest="keep_root_motion",
        action="store_true",
        help="保留手腕或骨盆根節點的平移（預設）",
    )
    root_group.add_argument(
        "--anchor-root-each-frame",
        "--anchor-wrist-each-frame",
        dest="keep_root_motion",
        action="store_false",
        help="每幀固定手腕或骨盆根節點；只適合刻意排除全域平移的分析",
    )
    parser.set_defaults(keep_root_motion=True)
    parser.add_argument(
        "--no-fill-missing-frames",
        action="store_true",
        help="只輸出 CSV 中有偵測資料的幀，不補上缺失幀",
    )
    return parser


def resolve_dataset_path(path: Path) -> Path:
    """Resolve relative user paths inside the project's dataset directory."""
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (DATASET_DIR / expanded).resolve()


def resolve_auxiliary_path(path: Path) -> Path:
    """Resolve a model path from common project locations."""
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()

    candidates = [
        Path.cwd() / expanded,
        SCRIPT_DIR / expanded,
        DATASET_DIR / expanded,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def parse_int(value: str, field: str, row_number: int) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"第 {row_number} 列的 {field} 不是整數：{value!r}") from exc


def parse_number(value: str, field: str, row_number: int) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"第 {row_number} 列的 {field} 不是數字：{value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"第 {row_number} 列的 {field} 必須是有限數字")
    return number


def read_csv_data(path: Path) -> CsvData:
    points: dict[int, dict[MarkerKey, tuple[float, float, float]]] = defaultdict(dict)
    times: dict[int, float] = {}
    handedness: dict[int, Counter[str]] = defaultdict(Counter)
    coordinate_spaces: set[str] = set()

    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV 沒有標題列")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV 缺少必要欄位：{', '.join(sorted(missing))}")
        fieldnames = set(reader.fieldnames)
        if "hand" in fieldnames:
            kind = "hand"
            track_field = "hand"
        elif "person" in fieldnames:
            kind = "pose"
            track_field = "person"
        else:
            raise ValueError("CSV 必須包含 hand（手部）或 person（全身）欄位")

        for row_number, row in enumerate(reader, start=2):
            frame = parse_int(row["frame"], "frame", row_number)
            hand = parse_int(row[track_field], track_field, row_number)
            landmark = parse_int(row["landmark"], "landmark", row_number)
            time = parse_number(row["time"], "time", row_number)
            xyz = tuple(
                parse_number(row[axis], axis, row_number) for axis in ("x", "y", "z")
            )

            if frame < 0 or hand < 0 or landmark < 0:
                raise ValueError(f"第 {row_number} 列的 frame、hand、landmark 不可為負數")

            existing_time = times.get(frame)
            if existing_time is not None and not math.isclose(
                existing_time, time, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(f"第 {row_number} 列：同一 frame 出現不同 time")
            times[frame] = time

            name = row["landmark_name"].strip() or f"LANDMARK_{landmark}"
            key = MarkerKey(hand, landmark, name, kind)
            if key in points[frame]:
                raise ValueError(f"第 {row_number} 列：同一 frame 的 marker 重複")
            points[frame][key] = xyz  # type: ignore[assignment]

            if kind == "hand":
                side = row.get("handedness", "").strip()
                if side:
                    handedness[hand][side] += 1
            coordinate_space = row.get("coordinate_space", "").strip().lower()
            if coordinate_space:
                coordinate_spaces.add(coordinate_space)

    if not points:
        raise ValueError("CSV 沒有任何 landmark 資料")
    if len(coordinate_spaces) > 1:
        raise ValueError("同一 CSV 不可混用多種 coordinate_space")
    default_space = "normalized" if kind == "hand" else "unknown"
    coordinate_space = next(iter(coordinate_spaces), default_space)
    return CsvData(
        dict(points),
        times,
        dict(handedness),
        kind,
        coordinate_space,
    )


def infer_data_rate(times: dict[int, float]) -> float:
    ordered = sorted(times.items())
    candidates = []
    for (frame_a, time_a), (frame_b, time_b) in zip(ordered, ordered[1:]):
        frame_delta = frame_b - frame_a
        time_delta = time_b - time_a
        if frame_delta > 0 and time_delta > 0:
            candidates.append(frame_delta / time_delta)
    if not candidates:
        raise ValueError("資料不足以推算取樣率，請使用 --data-rate 指定 Hz")
    return statistics.median(candidates)


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    return token.strip("_") or "Unknown"


def marker_names(
    keys: Iterable[MarkerKey], side_counts: dict[int, Counter[str]]
) -> dict[MarkerKey, str]:
    keys = list(keys)
    result: dict[MarkerKey, str] = {}
    used: set[str] = set()
    pose_tracks = {key.hand for key in keys if key.kind == "pose"}
    for key in keys:
        if key.kind == "pose":
            base = safe_token(key.landmark_name)
            if len(pose_tracks) > 1 or key.hand != 0:
                base = f"P{key.hand}_{base}"
        else:
            side = side_counts.get(key.hand, Counter()).most_common(1)
            side_name = safe_token(side[0][0]) if side else "Unknown"
            base = f"H{key.hand}_{side_name}_{safe_token(key.landmark_name)}"
        name = base
        if name in used:
            name = f"{base}_{key.landmark}"
        counter = 2
        while name in used:
            name = f"{base}_{key.landmark}_{counter}"
            counter += 1
        used.add(name)
        result[key] = name
    return result


def find_landmark_key(keys: Iterable[MarkerKey], hand: int, landmark: int) -> MarkerKey:
    matches = [key for key in keys if key.hand == hand and key.landmark == landmark]
    if not matches:
        raise ValueError(f"hand={hand} 缺少 landmark={landmark}")
    if len(matches) > 1:
        names = ", ".join(key.landmark_name for key in matches)
        raise ValueError(f"hand={hand} 的 landmark={landmark} 有多個名稱：{names}")
    return matches[0]


def apply_basic_transform(
    data: CsvData, *, scale: float, flip_y: bool, flip_z: bool
) -> CsvData:
    transformed: dict[int, dict[MarkerKey, tuple[float, float, float]]] = {}
    for frame, frame_points in data.points.items():
        transformed_frame: dict[MarkerKey, tuple[float, float, float]] = {}
        for key, point in frame_points.items():
            x, y, z = point
            y = -y if flip_y else y
            z = -z if flip_z else z
            transformed_frame[key] = (x * scale, y * scale, z * scale)
        transformed[frame] = transformed_frame
    return CsvData(
        transformed,
        dict(data.times),
        dict(data.handedness),
        data.kind,
        data.coordinate_space,
    )


def vec3_to_tuple(value: object) -> tuple[float, float, float]:
    try:
        getter = getattr(value, "get")
        return tuple(float(getter(i)) for i in range(3))  # type: ignore[return-value]
    except (AttributeError, TypeError):
        return tuple(float(value[i]) for i in range(3))  # type: ignore[index, return-value]


def rotation_between_vectors(
    source: object, target: object, np: object
) -> object:
    """Return a row-vector rotation matrix that maps source onto target."""
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm < 1e-12 or target_norm < 1e-12:
        raise ValueError("模型或校正資料含有長度接近 0 的骨段")

    a = source / source_norm
    b = target / target_norm
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))

    if cosine > 1.0 - 1e-10:
        return np.eye(3)

    if cosine < -1.0 + 1e-10:
        basis = np.zeros(3)
        basis[int(np.argmin(np.abs(a)))] = 1.0
        axis = np.cross(a, basis)
        axis = axis / np.linalg.norm(axis)
        column_rotation = 2.0 * np.outer(axis, axis) - np.eye(3)
        return column_rotation.T

    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    skew = np.asarray(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    column_rotation = (
        np.eye(3) + skew + (skew @ skew) * ((1.0 - cosine) / (sine * sine))
    )
    return column_rotation.T


def retarget_hand_pose(
    aligned: dict[MarkerKey, object],
    *,
    wrist_key: MarkerKey,
    segment_specs: list[tuple[MarkerKey, MarkerKey, float, object]],
    wrist_position: object,
    np: object,
) -> dict[MarkerKey, object]:
    """Apply model bone lengths and calibration direction offsets to one pose."""
    result = dict(aligned)
    result[wrist_key] = np.asarray(wrist_position, dtype=float)

    for parent_key, child_key, model_length, correction in segment_specs:
        if (
            parent_key not in aligned
            or child_key not in aligned
            or parent_key not in result
        ):
            continue
        observed_segment = np.asarray(aligned[child_key], dtype=float) - np.asarray(
            aligned[parent_key], dtype=float
        )
        observed_length = float(np.linalg.norm(observed_segment))
        if observed_length < 1e-12:
            continue
        direction = (observed_segment / observed_length) @ correction
        result[child_key] = (
            np.asarray(result[parent_key], dtype=float) + model_length * direction
        )
    return result


def auto_fit_hand_to_model(
    data: CsvData,
    *,
    model_path: Path,
    hand: int,
    calibration_frames: int,
    anchor_each_frame: bool,
) -> tuple[CsvData, FitSummary]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError("使用 --fit-model 需要 NumPy：pip install numpy") from exc

    try:
        import opensim as osim
    except ImportError as exc:
        raise ValueError(
            "使用 --fit-model 需要 OpenSim Python 套件，請在已安裝 opensim 的環境執行"
        ) from exc

    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 OpenSim 模型：{model_path}")

    keys = sorted({key for frame_points in data.points.values() for key in frame_points})
    available_hands = sorted({key.hand for key in keys})
    if hand not in available_hands:
        raise ValueError(f"CSV 沒有 hand={hand}；可用 hand：{available_hands}")

    hand_keys = {
        landmark: find_landmark_key(keys, hand, landmark)
        for landmark in ALL_HAND_LANDMARKS
    }
    palm_keys = [hand_keys[landmark] for landmark in PALM_FIT_LANDMARKS]
    names = marker_names(keys, data.handedness)
    hand_names = tuple(names[hand_keys[index]] for index in ALL_HAND_LANDMARKS)

    model = osim.Model(str(native_model_path(model_path)))
    state = model.initSystem()
    marker_set = model.getMarkerSet()

    target_by_key = {}
    for index, name in enumerate(hand_names):
        try:
            marker = marker_set.get(name)
        except Exception as exc:
            raise ValueError(f"模型中找不到同名 marker：{name}") from exc
        target_by_key[hand_keys[index]] = np.asarray(
            vec3_to_tuple(marker.getLocationInGround(state)), dtype=float
        )

    samples = []
    for frame in sorted(data.points):
        frame_points = data.points[frame]
        if all(key in frame_points for key in hand_keys.values()):
            samples.append(
                {
                    key: np.asarray(frame_points[key], dtype=float)
                    for key in hand_keys.values()
                }
            )
            if len(samples) >= calibration_frames:
                break

    if len(samples) < 3:
        raise ValueError(
            "開頭找不到至少 3 個包含全部 21 個 hand landmarks 的完整幀"
        )

    source_by_key = {
        key: np.median(np.stack([sample[key] for sample in samples], axis=0), axis=0)
        for key in hand_keys.values()
    }
    source = np.asarray([source_by_key[key] for key in palm_keys], dtype=float)
    target = np.asarray([target_by_key[key] for key in palm_keys], dtype=float)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center

    source_variance = float(np.sum(source_centered**2))
    if source_variance < 1e-12:
        raise ValueError("校正 landmarks 太靠近，無法估計縮放與旋轉")

    # Orthogonal Procrustes. Reflection is allowed because image and OpenSim
    # coordinate systems may have different handedness.
    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt  # Row-vector convention: p_aligned = p @ rotation.
    fit_scale = float(np.sum(singular_values) / source_variance)

    # Anchor the calibration wrist exactly on the model wrist.
    wrist_key = hand_keys[0]
    target_wrist = target_by_key[wrist_key]
    translation = target_wrist - fit_scale * (source_by_key[wrist_key] @ rotation)

    aligned_reference = {
        key: fit_scale * (point @ rotation) + translation
        for key, point in source_by_key.items()
    }

    segment_specs = []
    for chain in HAND_CHAINS:
        for parent_index, child_index in zip(chain, chain[1:]):
            parent_key = hand_keys[parent_index]
            child_key = hand_keys[child_index]
            source_segment = (
                aligned_reference[child_key] - aligned_reference[parent_key]
            )
            target_segment = target_by_key[child_key] - target_by_key[parent_key]
            model_length = float(np.linalg.norm(target_segment))
            correction = rotation_between_vectors(source_segment, target_segment, np)
            segment_specs.append(
                (parent_key, child_key, model_length, correction)
            )

    fitted_reference_by_key = retarget_hand_pose(
        aligned_reference,
        wrist_key=wrist_key,
        segment_specs=segment_specs,
        wrist_position=target_wrist,
        np=np,
    )
    fitted_reference = np.asarray(
        [fitted_reference_by_key[hand_keys[index]] for index in ALL_HAND_LANDMARKS]
    )
    target_reference = np.asarray(
        [target_by_key[hand_keys[index]] for index in ALL_HAND_LANDMARKS]
    )
    calibration_rms = float(
        np.sqrt(np.mean(np.sum((fitted_reference - target_reference) ** 2, axis=1)))
    )

    transformed: dict[int, dict[MarkerKey, tuple[float, float, float]]] = {}

    for frame, frame_points in data.points.items():
        transformed_frame = dict(frame_points)
        aligned_hand = {}
        for key, point in frame_points.items():
            if key.hand != hand:
                continue
            vector = np.asarray(point, dtype=float)
            aligned_hand[key] = fit_scale * (vector @ rotation) + translation

        if wrist_key in aligned_hand:
            wrist_position = (
                target_wrist if anchor_each_frame else aligned_hand[wrist_key]
            )
            retargeted = retarget_hand_pose(
                aligned_hand,
                wrist_key=wrist_key,
                segment_specs=segment_specs,
                wrist_position=wrist_position,
                np=np,
            )
            for key, point in retargeted.items():
                transformed_frame[key] = tuple(float(value) for value in point)

        transformed[frame] = transformed_frame

    summary = FitSummary(
        model_path=model_path,
        marker_names=hand_names,
        scale_factor=fit_scale,
        calibration_rms=calibration_rms,
        anchored_each_frame=anchor_each_frame,
        retargeted_segments=len(segment_specs),
        kind="hand",
        matched_markers=len(hand_names),
    )
    return (
        CsvData(
            transformed,
            dict(data.times),
            dict(data.handedness),
            data.kind,
            data.coordinate_space,
        ),
        summary,
    )


def load_pose_marker_map(path: Path | None) -> dict[int, tuple[str, ...]]:
    if path is None:
        return {}
    path = resolve_auxiliary_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 pose marker 對應檔：{path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取 pose marker 對應 JSON：{exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("pose marker 對應 JSON 必須是物件")

    name_to_index = {name: index for index, name in enumerate(POSE_LANDMARK_NAMES)}
    result = {}
    for key, value in raw.items():
        try:
            index = int(key) if str(key).isdigit() else name_to_index[str(key)]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"未知的 MediaPipe pose landmark：{key}") from exc
        if isinstance(value, str):
            names = (value,)
        elif isinstance(value, list) and value and all(
            isinstance(item, str) for item in value
        ):
            names = tuple(value)
        else:
            raise ValueError(
                f"{key} 的對應必須是模型 marker 名稱或非空名稱陣列"
            )
        result[index] = names
    return result


def pose_target_candidates(
    index: int, custom_map: dict[int, tuple[str, ...]]
) -> tuple[tuple[str, ...], ...]:
    if index in custom_map:
        return (custom_map[index],)
    direct = ((POSE_LANDMARK_NAMES[index],),)
    aliases = POSE_MARKER_ALIASES.get(index, ())
    return direct + tuple(group for group in aliases if group != direct[0])


def find_pose_model_targets(
    marker_set: object,
    state: object,
    *,
    custom_map: dict[int, tuple[str, ...]],
    np: object,
) -> tuple[dict[int, object], dict[int, tuple[str, ...]]]:
    targets = {}
    matched_groups = {}
    for index in range(len(POSE_LANDMARK_NAMES)):
        for group in pose_target_candidates(index, custom_map):
            locations = []
            try:
                for marker_name in group:
                    marker = marker_set.get(marker_name)
                    locations.append(
                        np.asarray(
                            vec3_to_tuple(marker.getLocationInGround(state)),
                            dtype=float,
                        )
                    )
            except Exception:
                continue
            targets[index] = np.mean(np.stack(locations, axis=0), axis=0)
            matched_groups[index] = group
            break
    return targets, matched_groups


def retarget_pose_skeleton(
    aligned: dict[MarkerKey, object],
    *,
    root_offsets: dict[MarkerKey, object],
    segment_specs: list[tuple[MarkerKey, MarkerKey, float, object]],
    np: object,
) -> dict[MarkerKey, object]:
    result = dict(aligned)
    for root_key, offset in root_offsets.items():
        if root_key in result:
            result[root_key] = np.asarray(result[root_key], dtype=float) + offset

    for parent_key, child_key, model_length, correction in segment_specs:
        if (
            parent_key not in aligned
            or child_key not in aligned
            or parent_key not in result
        ):
            continue
        observed = np.asarray(aligned[child_key], dtype=float) - np.asarray(
            aligned[parent_key], dtype=float
        )
        observed_length = float(np.linalg.norm(observed))
        if observed_length < 1e-12:
            continue
        direction = (observed / observed_length) @ correction
        result[child_key] = (
            np.asarray(result[parent_key], dtype=float) + model_length * direction
        )
    return result


def retarget_pose_hand_width(
    retargeted: dict[MarkerKey, object],
    aligned: dict[MarkerKey, object],
    pose_keys: dict[int, MarkerKey],
    *,
    np: object,
) -> None:
    """Keep MediaPipe palm-width points attached to the retargeted wrist."""
    for wrist_index, index_index, auxiliary_indices in (
        (15, 19, (17, 21)),
        (16, 20, (18, 22)),
    ):
        wrist_key = pose_keys[wrist_index]
        index_key = pose_keys[index_index]
        if not all(key in aligned for key in (wrist_key, index_key)):
            continue
        source_length = float(
            np.linalg.norm(
                np.asarray(aligned[index_key]) - np.asarray(aligned[wrist_key])
            )
        )
        target_length = float(
            np.linalg.norm(
                np.asarray(retargeted[index_key]) - np.asarray(retargeted[wrist_key])
            )
        )
        scale = target_length / source_length if source_length > 1e-12 else 1.0
        for auxiliary_index in auxiliary_indices:
            auxiliary_key = pose_keys[auxiliary_index]
            if auxiliary_key in aligned:
                retargeted[auxiliary_key] = np.asarray(retargeted[wrist_key]) + scale * (
                    np.asarray(aligned[auxiliary_key]) - np.asarray(aligned[wrist_key])
                )


def auto_fit_pose_to_model(
    data: CsvData,
    *,
    model_path: Path,
    person: int,
    calibration_frames: int,
    anchor_each_frame: bool,
    marker_map_path: Path | None,
) -> tuple[CsvData, FitSummary]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError("使用 --fit-model 需要 NumPy：pip install numpy") from exc

    try:
        import opensim as osim
    except ImportError as exc:
        raise ValueError(
            "使用 --fit-model 需要 OpenSim Python 套件，請在已安裝 opensim 的環境執行"
        ) from exc

    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 OpenSim 模型：{model_path}")

    keys = sorted({key for frame_points in data.points.values() for key in frame_points})
    pose_keys = {
        key.landmark: key
        for key in keys
        if key.kind == "pose" and key.hand == person
    }
    missing_landmarks = [
        index for index in range(33) if index not in pose_keys
    ]
    if missing_landmarks:
        raise ValueError(
            f"person={person} 缺少 pose landmarks：{missing_landmarks}"
        )

    model = osim.Model(str(native_model_path(model_path)))
    state = model.initSystem()
    marker_set = model.getMarkerSet()
    custom_map = load_pose_marker_map(marker_map_path)
    targets, matched_groups = find_pose_model_targets(
        marker_set,
        state,
        custom_map=custom_map,
        np=np,
    )
    torso_indices = [index for index in (11, 12, 23, 24) if index in targets]
    fit_indices = (
        torso_indices
        if len(torso_indices) == 4
        else [index for index in POSE_FIT_PRIORITY if index in targets]
    )
    if len(fit_indices) < 4:
        fit_indices = sorted(targets)
    if len(fit_indices) < 4:
        found = [
            f"{POSE_LANDMARK_NAMES[index]}={' + '.join(group)}"
            for index, group in sorted(matched_groups.items())
        ]
        raise ValueError(
            "模型中找不到至少 4 個可對應的全身 markers。"
            "請使用含 MarkerSet 的 scaled model，或用 --pose-marker-map 指定。"
            f"目前找到：{found or '無'}"
        )

    samples = []
    all_person_keys = list(pose_keys.values())
    for frame in sorted(data.points):
        frame_points = data.points[frame]
        if all(key in frame_points for key in all_person_keys):
            samples.append(
                {
                    key: np.asarray(frame_points[key], dtype=float)
                    for key in all_person_keys
                }
            )
            if len(samples) >= calibration_frames:
                break
    if not samples:
        raise ValueError("找不到包含完整 33 點的全身偵測幀")

    source_by_key = {
        key: np.median(np.stack([sample[key] for sample in samples], axis=0), axis=0)
        for key in all_person_keys
    }
    source = np.asarray(
        [source_by_key[pose_keys[index]] for index in fit_indices], dtype=float
    )
    target = np.asarray([targets[index] for index in fit_indices], dtype=float)

    # Shoulder and hip centers are almost coplanar, so fitting only those four
    # points cannot distinguish front from back. Add a virtual anterior point
    # derived from the body's left and up axes to make the orientation explicit.
    # MediaPipe's landmark ordering and the OpenSim model then share a stable
    # right-handed frame: anterior = left x up.
    if all(index in targets for index in (11, 12, 23, 24)):
        def anterior_reference(points: dict[int, object]) -> object:
            left = (np.asarray(points[11]) + np.asarray(points[23])) / 2
            right = (np.asarray(points[12]) + np.asarray(points[24])) / 2
            shoulders = (np.asarray(points[11]) + np.asarray(points[12])) / 2
            hips = (np.asarray(points[23]) + np.asarray(points[24])) / 2
            lateral = left - right
            upward = shoulders - hips
            anterior = np.cross(lateral, upward)
            anterior_norm = float(np.linalg.norm(anterior))
            span = (float(np.linalg.norm(lateral)) + float(np.linalg.norm(upward))) / 2
            if anterior_norm < 1e-12 or span < 1e-12:
                raise ValueError("軀幹 landmarks 無法判斷身體前後方向")
            center = (shoulders + hips) / 2
            return center + anterior * (span / anterior_norm)

        source_torso = {
            index: source_by_key[pose_keys[index]] for index in (11, 12, 23, 24)
        }
        target_torso = {index: targets[index] for index in (11, 12, 23, 24)}
        source = np.vstack([source, anterior_reference(source_torso)])
        target = np.vstack([target, anterior_reference(target_torso)])
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    source_variance = float(np.sum(source_centered**2))
    if source_variance < 1e-12:
        raise ValueError("全身校正 landmarks 太靠近，無法估計縮放與旋轉")

    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if float(np.linalg.det(rotation)) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    fit_scale = float(np.sum(singular_values) / source_variance)
    translation = target_center - fit_scale * (source_center @ rotation)
    aligned_reference = {
        key: fit_scale * (point @ rotation) + translation
        for key, point in source_by_key.items()
    }

    root_indices = (11, 12, 23, 24)
    root_offsets = {
        pose_keys[index]: targets[index] - aligned_reference[pose_keys[index]]
        for index in root_indices
        if index in targets
    }
    segment_specs = []
    for chain in POSE_CHAINS:
        for parent_index, child_index in zip(chain, chain[1:]):
            if parent_index not in targets or child_index not in targets:
                continue
            parent_key = pose_keys[parent_index]
            child_key = pose_keys[child_index]
            target_segment = targets[child_index] - targets[parent_index]
            model_length = float(np.linalg.norm(target_segment))
            # The torso similarity transform has already put the video in the
            # OpenSim coordinate frame. Retain every observed segment direction
            # and change only its length. Independent minimal rotations around
            # nearly vertical limb segments have an unconstrained twist; they
            # can reverse sagittal motion or make heel/toe directions disagree.
            correction = np.eye(3)
            segment_specs.append(
                (parent_key, child_key, model_length, correction)
            )

    fitted_reference = retarget_pose_skeleton(
        aligned_reference,
        root_offsets=root_offsets,
        segment_specs=segment_specs,
        np=np,
    )
    retarget_pose_hand_width(
        fitted_reference, aligned_reference, pose_keys, np=np
    )
    calibration_errors = [
        np.asarray(fitted_reference[pose_keys[index]]) - targets[index]
        for index in targets
    ]
    calibration_rms = float(
        np.sqrt(
            np.mean(
                [float(np.dot(error, error)) for error in calibration_errors]
            )
        )
    )

    pelvis_indices = (23, 24)
    target_pelvis = (
        np.mean(np.stack([targets[index] for index in pelvis_indices]), axis=0)
        if all(index in targets for index in pelvis_indices)
        else target_center
    )
    transformed = {}
    for frame, frame_points in data.points.items():
        transformed_frame = dict(frame_points)
        aligned_person = {}
        for key, point in frame_points.items():
            if key.kind != "pose" or key.hand != person:
                continue
            vector = np.asarray(point, dtype=float)
            aligned_person[key] = fit_scale * (vector @ rotation) + translation

        retargeted = retarget_pose_skeleton(
            aligned_person,
            root_offsets=root_offsets,
            segment_specs=segment_specs,
            np=np,
        )
        retarget_pose_hand_width(retargeted, aligned_person, pose_keys, np=np)
        if anchor_each_frame and all(
            pose_keys[index] in retargeted for index in pelvis_indices
        ):
            pelvis = np.mean(
                np.stack(
                    [retargeted[pose_keys[index]] for index in pelvis_indices]
                ),
                axis=0,
            )
            offset = target_pelvis - pelvis
            retargeted = {
                key: np.asarray(point, dtype=float) + offset
                for key, point in retargeted.items()
            }
        for key, point in retargeted.items():
            transformed_frame[key] = tuple(float(value) for value in point)
        transformed[frame] = transformed_frame

    descriptions = tuple(
        f"{POSE_LANDMARK_NAMES[index]}={' + '.join(group)}"
        for index, group in sorted(matched_groups.items())
    )
    summary = FitSummary(
        model_path=model_path,
        marker_names=descriptions,
        scale_factor=fit_scale,
        calibration_rms=calibration_rms,
        anchored_each_frame=anchor_each_frame,
        retargeted_segments=len(segment_specs),
        kind="pose",
        matched_markers=len(targets),
    )
    return (
        CsvData(
            transformed,
            dict(data.times),
            dict(data.handedness),
            data.kind,
            data.coordinate_space,
        ),
        summary,
    )


def auto_fit_to_model(
    data: CsvData,
    *,
    model_path: Path,
    hand: int = 0,
    calibration_frames: int,
    anchor_each_frame: bool,
    marker_map_path: Path | None = None,
) -> tuple[CsvData, FitSummary]:
    if data.kind == "pose":
        return auto_fit_pose_to_model(
            data,
            model_path=model_path,
            person=hand,
            calibration_frames=calibration_frames,
            anchor_each_frame=anchor_each_frame,
            marker_map_path=marker_map_path,
        )
    return auto_fit_hand_to_model(
        data,
        model_path=model_path,
        hand=hand,
        calibration_frames=calibration_frames,
        anchor_each_frame=anchor_each_frame,
    )


def format_number(value: float) -> str:
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def frame_sequence(data: CsvData, fill_missing: bool) -> list[int]:
    existing = sorted(data.points)
    if not fill_missing:
        return existing
    return list(range(existing[0], existing[-1] + 1))


def interpolate_missing_points(
    data: CsvData, keys: list[MarkerKey], frames: list[int]
) -> dict[int, dict[MarkerKey, tuple[float, float, float]]]:
    """Fill missing marker samples so OpenSim receives continuous trajectories."""
    filled = {frame: dict(data.points.get(frame, {})) for frame in frames}

    for key in keys:
        observations = [
            (frame, data.points[frame][key])
            for frame in sorted(data.points)
            if key in data.points[frame]
        ]
        if not observations:
            continue

        observation_index = 0
        for frame in frames:
            if key in filled[frame]:
                while (
                    observation_index + 1 < len(observations)
                    and observations[observation_index + 1][0] <= frame
                ):
                    observation_index += 1
                continue

            while (
                observation_index + 1 < len(observations)
                and observations[observation_index + 1][0] < frame
            ):
                observation_index += 1

            previous = observations[observation_index]
            if frame <= observations[0][0]:
                filled[frame][key] = observations[0][1]
            elif observation_index + 1 >= len(observations):
                filled[frame][key] = observations[-1][1]
            else:
                following = observations[observation_index + 1]
                span = following[0] - previous[0]
                fraction = (frame - previous[0]) / span
                filled[frame][key] = tuple(
                    previous[1][axis]
                    + fraction * (following[1][axis] - previous[1][axis])
                    for axis in range(3)
                )
    return filled


def write_trc(
    output_path: Path,
    data: CsvData,
    *,
    data_rate: float,
    units: str,
    fill_missing: bool,
) -> tuple[int, int]:
    keys = sorted({key for frame_points in data.points.values() for key in frame_points})
    names = marker_names(keys, data.handedness)
    frames = frame_sequence(data, fill_missing)
    output_points = (
        interpolate_missing_points(data, keys, frames)
        if fill_missing
        else {frame: data.points.get(frame, {}) for frame in frames}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    marker_header = ["Frame#", "Time"]
    coordinate_header = ["", ""]
    for index, key in enumerate(keys, start=1):
        marker_header.extend([names[key], "", ""])
        coordinate_header.extend([f"X{index}", f"Y{index}", f"Z{index}"])

    with output_path.open("w", newline="", encoding="utf-8") as trc_file:
        writer = csv.writer(trc_file, delimiter="\t", lineterminator="\n")
        writer.writerow(["PathFileType", "4", "(X/Y/Z)", output_path.name])
        writer.writerow(
            [
                "DataRate",
                "CameraRate",
                "NumFrames",
                "NumMarkers",
                "Units",
                "OrigDataRate",
                "OrigDataStartFrame",
                "OrigNumFrames",
            ]
        )
        writer.writerow(
            [
                format_number(data_rate),
                format_number(data_rate),
                len(frames),
                len(keys),
                units,
                format_number(data_rate),
                1,
                len(frames),
            ]
        )
        writer.writerow(marker_header)
        writer.writerow(coordinate_header)
        writer.writerow([])

        first_frame = frames[0]
        first_time = data.times.get(first_frame, first_frame / data_rate)
        for output_frame, frame in enumerate(frames, start=1):
            time = data.times.get(
                frame, first_time + (frame - first_frame) / data_rate
            )
            row: list[str | int] = [
                output_frame,
                format_number(time - first_time),
            ]
            frame_points = output_points[frame]
            for key in keys:
                point = frame_points.get(key)
                if point is None:
                    row.extend(["", "", ""])
                    continue
                row.extend(format_number(axis) for axis in point)
            writer.writerow(row)

    return len(frames), len(keys)


def convert(
    args: argparse.Namespace,
) -> tuple[Path, int, int, float, FitSummary | None]:
    input_path = resolve_dataset_path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到輸入 CSV：{input_path}")
    if input_path.suffix.lower() != ".csv":
        raise ValueError("輸入檔案必須是 .csv")

    output_path = (
        resolve_dataset_path(args.output)
        if args.output
        else input_path.with_suffix(".trc")
    )
    if output_path == input_path:
        raise ValueError("輸出路徑不可與輸入 CSV 相同")

    data = read_csv_data(input_path)
    data = apply_basic_transform(
        data,
        scale=args.scale,
        flip_y=args.flip_y,
        flip_z=args.flip_z,
    )

    fit_summary = None
    if args.fit_model is not None:
        if args.units.lower() != "m":
            raise ValueError("使用 --fit-model 時，--units 必須是 m")
        calibration_frames = (
            args.calibration_frames
            if args.calibration_frames is not None
            else (1 if data.kind == "pose" else 30)
        )
        model_path = resolve_auxiliary_path(args.fit_model)
        data, fit_summary = auto_fit_to_model(
            data,
            model_path=model_path,
            hand=args.fit_index,
            calibration_frames=calibration_frames,
            anchor_each_frame=not args.keep_root_motion,
            marker_map_path=args.pose_marker_map,
        )

    data_rate = args.data_rate if args.data_rate is not None else infer_data_rate(data.times)
    frames, markers = write_trc(
        output_path,
        data,
        data_rate=data_rate,
        units=args.units,
        fill_missing=not args.no_fill_missing_frames,
    )
    return output_path, frames, markers, data_rate, fit_summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_path, frames, markers, data_rate, fit_summary = convert(args)
    except (OSError, ValueError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 1

    print(f"完成：{frames} 幀、{markers} 個 markers、{data_rate:.6g} Hz")
    if fit_summary is not None:
        root_name = "骨盆" if fit_summary.kind == "pose" else "手腕"
        anchor_mode = (
            f"每幀固定{root_name}"
            if fit_summary.anchored_each_frame
            else f"保留{root_name}平移"
        )
        mode_name = "全身" if fit_summary.kind == "pose" else "手部"
        print(f"模型自動對齊：{fit_summary.model_path}")
        print(f"對齊模式：{mode_name}")
        print(f"自動縮放倍率：{fit_summary.scale_factor:.6g}")
        print(f"校正 marker RMS：{fit_summary.calibration_rms:.6g} m")
        print(f"模型對應 markers：{fit_summary.matched_markers}")
        print(f"逐段骨長校正：{fit_summary.retargeted_segments} 段")
        print(f"根節點模式：{anchor_mode}")
    print(f"TRC：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

