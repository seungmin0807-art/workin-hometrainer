from __future__ import annotations

import argparse
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from tqdm import tqdm


from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

TARGET_FPS = 10.0
MODEL_NAME = "MediaPipe Pose"
EXERCISE_NAME = "Barbell Back Squat"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH = Path(".models/pose_landmarker_full.task")

POSE_LANDMARK = vision.PoseLandmark
LANDMARK_INDEX = {landmark.name.lower(): landmark.value for landmark in POSE_LANDMARK}
CONNECTIONS = sorted(
    (int(connection.start), int(connection.end))
    for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS
)

LEFT_CORE = [
    POSE_LANDMARK.LEFT_SHOULDER.value,
    POSE_LANDMARK.LEFT_HIP.value,
    POSE_LANDMARK.LEFT_KNEE.value,
    POSE_LANDMARK.LEFT_ANKLE.value,
    POSE_LANDMARK.LEFT_FOOT_INDEX.value,
]
RIGHT_CORE = [
    POSE_LANDMARK.RIGHT_SHOULDER.value,
    POSE_LANDMARK.RIGHT_HIP.value,
    POSE_LANDMARK.RIGHT_KNEE.value,
    POSE_LANDMARK.RIGHT_ANKLE.value,
    POSE_LANDMARK.RIGHT_FOOT_INDEX.value,
]

LEFT_JOINT_SET = {
    "shoulder": POSE_LANDMARK.LEFT_SHOULDER.value,
    "hip": POSE_LANDMARK.LEFT_HIP.value,
    "knee": POSE_LANDMARK.LEFT_KNEE.value,
    "ankle": POSE_LANDMARK.LEFT_ANKLE.value,
    "foot": POSE_LANDMARK.LEFT_FOOT_INDEX.value,
}
RIGHT_JOINT_SET = {
    "shoulder": POSE_LANDMARK.RIGHT_SHOULDER.value,
    "hip": POSE_LANDMARK.RIGHT_HIP.value,
    "knee": POSE_LANDMARK.RIGHT_KNEE.value,
    "ankle": POSE_LANDMARK.RIGHT_ANKLE.value,
    "foot": POSE_LANDMARK.RIGHT_FOOT_INDEX.value,
}

ISSUE_SPECS = {
    "torso_lean": {
        "label": "상체 전경 과다",
        "threshold": 8.0,
        "scale": 18.0,
        "joints": ["shoulder", "hip", "knee"],
    },
    "depth": {
        "label": "깊이 부족",
        "threshold": 0.08,
        "scale": 0.18,
        "joints": ["hip", "knee", "ankle"],
    },
    "knee_flexion": {
        "label": "무릎 굴곡 부족",
        "threshold": 8.0,
        "scale": 18.0,
        "joints": ["hip", "knee", "ankle"],
    },
    "shin_lean": {
        "label": "정강이 전진 부족",
        "threshold": 6.0,
        "scale": 14.0,
        "joints": ["knee", "ankle", "foot"],
    },
}


@dataclass
class FrameResult:
    frame_idx: int
    sample_idx: int
    time_sec: float
    landmarks2d: list[list[float]] | None
    world_landmarks: list[list[float]] | None
    visibility: list[float] | None
    side_visibility: dict[str, float]
    metrics_by_side: dict[str, dict[str, float | None]]
    pose_detected: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze squat videos and generate prototype assets.")
    parser.add_argument("--correct", type=Path, required=True, help="Reference video path.")
    parser.add_argument("--wrong", type=Path, required=True, help="User video path.")
    parser.add_argument("--output-dir", type=Path, default=Path("app"), help="Static app output root.")
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS, help="Sampling FPS.")
    return parser.parse_args()


def ensure_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose landmarker model to {model_path} ...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def average_visibility(visibility: list[float] | None, indices: list[int]) -> float:
    if visibility is None:
        return 0.0
    values = [visibility[i] for i in indices if i < len(visibility)]
    return float(sum(values) / len(values)) if values else 0.0


def safe_point(points: list[list[float]] | None, index: int) -> np.ndarray | None:
    if points is None or index >= len(points):
        return None
    return np.asarray(points[index][:3], dtype=np.float64)


def midpoint(points: list[list[float]] | None, left_idx: int, right_idx: int) -> np.ndarray | None:
    left = safe_point(points, left_idx)
    right = safe_point(points, right_idx)
    if left is None or right is None:
        return None
    return (left + right) / 2.0


def angle_at(points: list[list[float]] | None, a_idx: int, b_idx: int, c_idx: int) -> float | None:
    a = safe_point(points, a_idx)
    b = safe_point(points, b_idx)
    c = safe_point(points, c_idx)
    if a is None or b is None or c is None:
        return None
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 1e-6:
        return None
    cos_value = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_value)))


def tilt_from_vertical(points: list[list[float]] | None, lower_idx: int, upper_idx: int) -> float | None:
    lower = safe_point(points, lower_idx)
    upper = safe_point(points, upper_idx)
    if lower is None or upper is None:
        return None
    vector = upper[:2] - lower[:2]
    denom = np.linalg.norm(vector)
    if denom <= 1e-6:
        return None
    up = np.array([0.0, -1.0])
    cos_value = float(np.clip(np.dot(vector / denom, up), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_value)))


def round_nested(obj: Any, digits: int = 4) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, digits)
    if isinstance(obj, list):
        return [round_nested(item, digits) for item in obj]
    if isinstance(obj, dict):
        return {key: round_nested(value, digits) for key, value in obj.items()}
    return obj


def to_landmark_list(landmarks: Any) -> tuple[list[list[float]], list[float]]:
    coords: list[list[float]] = []
    visibility: list[float] = []
    for landmark in landmarks.landmark:
        vis = float(getattr(landmark, "visibility", 0.0))
        coords.append([float(landmark.x), float(landmark.y), float(landmark.z), vis])
        visibility.append(vis)
    return coords, visibility


def compute_body_scale(landmarks2d: list[list[float]] | None) -> float | None:
    if landmarks2d is None:
        return None
    shoulder_mid = midpoint(
        landmarks2d,
        POSE_LANDMARK.LEFT_SHOULDER.value,
        POSE_LANDMARK.RIGHT_SHOULDER.value,
    )
    ankle_mid = midpoint(
        landmarks2d,
        POSE_LANDMARK.LEFT_ANKLE.value,
        POSE_LANDMARK.RIGHT_ANKLE.value,
    )
    if shoulder_mid is None or ankle_mid is None:
        return None
    value = float(np.linalg.norm(ankle_mid[:2] - shoulder_mid[:2]))
    return value if value > 1e-6 else None


def compute_side_metrics(landmarks2d: list[list[float]] | None, side: str) -> dict[str, float | None]:
    joints = LEFT_JOINT_SET if side == "left" else RIGHT_JOINT_SET
    shoulder_mid = midpoint(
        landmarks2d,
        POSE_LANDMARK.LEFT_SHOULDER.value,
        POSE_LANDMARK.RIGHT_SHOULDER.value,
    )
    hip_mid = midpoint(
        landmarks2d,
        POSE_LANDMARK.LEFT_HIP.value,
        POSE_LANDMARK.RIGHT_HIP.value,
    )
    knee_mid = midpoint(
        landmarks2d,
        POSE_LANDMARK.LEFT_KNEE.value,
        POSE_LANDMARK.RIGHT_KNEE.value,
    )
    ankle_mid = midpoint(
        landmarks2d,
        POSE_LANDMARK.LEFT_ANKLE.value,
        POSE_LANDMARK.RIGHT_ANKLE.value,
    )

    body_scale = compute_body_scale(landmarks2d)
    depth_score = None
    hip_height_norm = None
    if hip_mid is not None and knee_mid is not None and body_scale:
        depth_score = float((knee_mid[1] - hip_mid[1]) / body_scale)
    if shoulder_mid is not None and hip_mid is not None and ankle_mid is not None:
        denom = max(float(ankle_mid[1] - shoulder_mid[1]), 1e-6)
        hip_height_norm = float((hip_mid[1] - shoulder_mid[1]) / denom)

    return {
        "knee_angle": angle_at(landmarks2d, joints["hip"], joints["knee"], joints["ankle"]),
        "hip_angle": angle_at(landmarks2d, joints["shoulder"], joints["hip"], joints["knee"]),
        "ankle_angle": angle_at(landmarks2d, joints["knee"], joints["ankle"], joints["foot"]),
        "torso_lean_deg": tilt_from_vertical(landmarks2d, joints["hip"], joints["shoulder"]),
        "shin_lean_deg": tilt_from_vertical(landmarks2d, joints["ankle"], joints["knee"]),
        "depth_score": depth_score,
        "hip_height_norm": hip_height_norm,
    }


def detect_pose_sequence(video_path: Path, role: str, target_fps: float) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_interval = max(int(round(fps / target_fps)), 1) if fps else 3

    frames: list[FrameResult] = []
    detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(MODEL_PATH.resolve()))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
    )

    sampled_index = 0
    progress = tqdm(total=frame_count, desc=f"{role} pose", unit="f")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((frame_idx / fps) * 1000.0) if fps else sampled_index * int(1000 / max(target_fps, 1))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(image, timestamp_ms)

            if result.pose_landmarks and result.pose_world_landmarks:
                landmarks2d = [[float(lm.x), float(lm.y), float(lm.z), float(lm.visibility)] for lm in result.pose_landmarks[0]]
                world_landmarks = [[float(lm.x), float(lm.y), float(lm.z), float(lm.visibility)] for lm in result.pose_world_landmarks[0]]
                visibility = [float(lm.visibility) for lm in result.pose_landmarks[0]]
                frame_result = FrameResult(
                    frame_idx=frame_idx,
                    sample_idx=sampled_index,
                    time_sec=frame_idx / fps if fps else sampled_index / target_fps,
                    landmarks2d=landmarks2d,
                    world_landmarks=world_landmarks,
                    visibility=visibility,
                    side_visibility={
                        "left": average_visibility(visibility, LEFT_CORE),
                        "right": average_visibility(visibility, RIGHT_CORE),
                    },
                    metrics_by_side={
                        "left": compute_side_metrics(landmarks2d, "left"),
                        "right": compute_side_metrics(landmarks2d, "right"),
                    },
                    pose_detected=True,
                )
            else:
                frame_result = FrameResult(
                    frame_idx=frame_idx,
                    sample_idx=sampled_index,
                    time_sec=frame_idx / fps if fps else sampled_index / target_fps,
                    landmarks2d=None,
                    world_landmarks=None,
                    visibility=None,
                    side_visibility={"left": 0.0, "right": 0.0},
                    metrics_by_side={"left": {}, "right": {}},
                    pose_detected=False,
                )
            frames.append(frame_result)
            sampled_index += 1

        frame_idx += 1
        progress.update(1)

    progress.close()
    detector.close()
    cap.release()

    detected_frames = [frame for frame in frames if frame.pose_detected]
    left_vis = np.mean([frame.side_visibility["left"] for frame in detected_frames]) if detected_frames else 0.0
    right_vis = np.mean([frame.side_visibility["right"] for frame in detected_frames]) if detected_frames else 0.0
    primary_side = "left" if left_vis >= right_vis else "right"

    return {
        "role": role,
        "path": str(video_path.resolve()),
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_sec": frame_count / fps if fps else 0.0,
        "sample_interval": sample_interval,
        "sampled_fps": fps / sample_interval if fps else target_fps,
        "primary_side": primary_side,
        "frames": frames,
    }


def numeric_series(sequence: dict[str, Any], side: str, metric_names: list[str]) -> np.ndarray:
    frame_count = len(sequence["frames"])
    matrix = np.full((frame_count, len(metric_names)), np.nan, dtype=np.float64)
    for row, frame in enumerate(sequence["frames"]):
        metrics = frame.metrics_by_side.get(side, {})
        for col, name in enumerate(metric_names):
            value = metrics.get(name)
            if value is not None:
                matrix[row, col] = float(value)
    return matrix


def fill_series(values: np.ndarray) -> np.ndarray:
    result = values.astype(np.float64).copy()
    for col in range(result.shape[1]):
        column = result[:, col]
        valid = np.isfinite(column)
        if not np.any(valid):
            result[:, col] = 0.0
            continue
        valid_indices = np.where(valid)[0]
        result[:, col] = np.interp(np.arange(len(column)), valid_indices, column[valid])
        if len(column) >= 7:
            window = min(len(column) if len(column) % 2 == 1 else len(column) - 1, 21)
            if window >= 7:
                result[:, col] = savgol_filter(result[:, col], window_length=window, polyorder=2, mode="interp")
    return result


def standardize_pair(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    merged = np.vstack([reference, target])
    mean = merged.mean(axis=0)
    std = merged.std(axis=0)
    std[std < 1e-6] = 1.0
    return (reference - mean) / std, (target - mean) / std


def dtw_align(reference: np.ndarray, target: np.ndarray) -> list[int]:
    n, m = len(reference), len(target)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local = float(np.linalg.norm(reference[i - 1] - target[j - 1]))
            options = [cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]]
            best = int(np.argmin(options))
            cost[i, j] = local + options[best]
            move[i, j] = best

    i, j = n, m
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(move[i, j])
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    path.reverse()

    mapping: dict[int, list[int]] = {}
    for ref_idx, target_idx in path:
        mapping.setdefault(target_idx, []).append(ref_idx)

    mapped = []
    previous = 0
    for target_idx in range(m):
        candidates = mapping.get(target_idx)
        if not candidates:
            mapped.append(previous)
            continue
        current = int(round(float(np.median(candidates))))
        current = max(previous, current)
        mapped.append(current)
        previous = current
    return mapped


def detect_reps(sequence: dict[str, Any], side: str) -> dict[str, Any]:
    metrics = fill_series(numeric_series(sequence, side, ["hip_height_norm"]))[:, 0]
    if len(metrics) < 3:
        return {"count": 0, "bottom_indices": [], "phase": ["steady"] * len(metrics), "rep_index": [1] * len(metrics)}

    bottoms, _ = find_peaks(metrics, distance=max(int(sequence["sampled_fps"] * 1.1), 4), prominence=0.03)
    derivative = np.diff(metrics, prepend=metrics[0])
    phase = []
    rep_index = []
    completed = 0
    next_bottom_idx = 0
    bottom_list = bottoms.tolist()
    for idx in range(len(metrics)):
        while next_bottom_idx < len(bottom_list) and idx >= bottom_list[next_bottom_idx]:
            completed += 1
            next_bottom_idx += 1
        rep_index.append(max(completed + 1, 1))
        motion = derivative[idx]
        if motion > 0.002:
            phase.append("descent")
        elif motion < -0.002:
            phase.append("ascent")
        else:
            phase.append("steady")
    return {
        "count": len(bottom_list),
        "bottom_indices": bottom_list,
        "phase": phase,
        "rep_index": rep_index,
    }


def build_issue_segments(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments = []
    active: dict[str, dict[str, Any]] = {}

    def close_segment(issue_id: str) -> None:
        segment = active.pop(issue_id, None)
        if segment:
            segments.append(segment)

    for frame in frames:
        active_ids = {issue["id"] for issue in frame["issues"]}
        current_time = frame["time_sec"]

        for issue in frame["issues"]:
            existing = active.get(issue["id"])
            if existing is None:
                active[issue["id"]] = {
                    "id": issue["id"],
                    "label": issue["label"],
                    "start_time": current_time,
                    "end_time": current_time,
                    "peak_severity": issue["severity"],
                    "peak_delta": issue["delta"],
                }
            else:
                existing["end_time"] = current_time
                existing["peak_severity"] = max(existing["peak_severity"], issue["severity"])
                existing["peak_delta"] = max(existing["peak_delta"], issue["delta"])

        for issue_id in list(active):
            if issue_id not in active_ids:
                close_segment(issue_id)

    for issue_id in list(active):
        close_segment(issue_id)

    return segments


def compare_sequences(correct: dict[str, Any], wrong: dict[str, Any], side: str) -> dict[str, Any]:
    metric_names = ["hip_height_norm", "knee_angle", "hip_angle", "torso_lean_deg", "shin_lean_deg", "depth_score"]
    correct_matrix = fill_series(numeric_series(correct, side, metric_names))
    wrong_matrix = fill_series(numeric_series(wrong, side, metric_names))
    correct_scaled, wrong_scaled = standardize_pair(correct_matrix, wrong_matrix)
    mapped_reference = dtw_align(correct_scaled, wrong_scaled)

    rep_info = detect_reps(wrong, side)

    frame_payloads: list[dict[str, Any]] = []
    issue_buckets: dict[str, list[dict[str, Any]]] = {key: [] for key in ISSUE_SPECS}
    score_values = []

    for idx, wrong_frame in enumerate(wrong["frames"]):
        ref_idx = mapped_reference[idx]
        correct_frame = correct["frames"][ref_idx]
        wrong_metrics = wrong_frame.metrics_by_side.get(side, {})
        correct_metrics = correct_frame.metrics_by_side.get(side, {})

        delta_torso = None
        if wrong_metrics.get("torso_lean_deg") is not None and correct_metrics.get("torso_lean_deg") is not None:
            delta_torso = float(wrong_metrics["torso_lean_deg"] - correct_metrics["torso_lean_deg"])

        delta_depth = None
        if wrong_metrics.get("depth_score") is not None and correct_metrics.get("depth_score") is not None:
            delta_depth = float(correct_metrics["depth_score"] - wrong_metrics["depth_score"])

        delta_knee = None
        if wrong_metrics.get("knee_angle") is not None and correct_metrics.get("knee_angle") is not None:
            delta_knee = float(wrong_metrics["knee_angle"] - correct_metrics["knee_angle"])

        delta_shin = None
        if wrong_metrics.get("shin_lean_deg") is not None and correct_metrics.get("shin_lean_deg") is not None:
            delta_shin = float(correct_metrics["shin_lean_deg"] - wrong_metrics["shin_lean_deg"])

        issue_values = {
            "torso_lean": delta_torso,
            "depth": delta_depth,
            "knee_flexion": delta_knee,
            "shin_lean": delta_shin,
        }

        active_issues: list[dict[str, Any]] = []
        highlighted_joint_names: set[str] = set()
        messages = []
        for issue_id, delta in issue_values.items():
            if delta is None:
                continue
            spec = ISSUE_SPECS[issue_id]
            if delta <= spec["threshold"]:
                continue
            severity = float(min(1.0, max(0.0, delta / spec["scale"])))
            active_issues.append(
                {
                    "id": issue_id,
                    "label": spec["label"],
                    "delta": delta,
                    "severity": severity,
                }
            )
            issue_buckets[issue_id].append(
                {
                    "frame": idx,
                    "time_sec": wrong_frame.time_sec,
                    "delta": delta,
                    "severity": severity,
                }
            )
            highlighted_joint_names.update(spec["joints"])

        active_issues.sort(key=lambda item: item["severity"], reverse=True)

        if active_issues:
            for issue in active_issues[:2]:
                if issue["id"] == "torso_lean":
                    messages.append(f"상체를 기준보다 {issue['delta']:.1f}도 덜 숙여보세요.")
                elif issue["id"] == "depth":
                    messages.append("고관절을 조금 더 아래로 보내 깊이를 맞춰보세요.")
                elif issue["id"] == "knee_flexion":
                    messages.append(f"무릎 굴곡이 기준보다 {issue['delta']:.1f}도 부족합니다.")
                elif issue["id"] == "shin_lean":
                    messages.append("정강이를 조금 더 앞으로 보내 발목을 써보세요.")
        else:
            messages.append("기준 자세와 큰 차이 없이 잘 따라가고 있어요.")

        score_penalty = 0.0
        if delta_torso is not None:
            score_penalty += max(delta_torso - ISSUE_SPECS["torso_lean"]["threshold"], 0.0) * 1.4
        if delta_depth is not None:
            score_penalty += max(delta_depth - ISSUE_SPECS["depth"]["threshold"], 0.0) * 220.0
        if delta_knee is not None:
            score_penalty += max(delta_knee - ISSUE_SPECS["knee_flexion"]["threshold"], 0.0) * 1.0
        if delta_shin is not None:
            score_penalty += max(delta_shin - ISSUE_SPECS["shin_lean"]["threshold"], 0.0) * 1.2
        score = float(max(45.0, min(99.0, 100.0 - score_penalty)))
        score_values.append(score)

        frame_payloads.append(
            {
                "sample_idx": idx,
                "frame_idx": wrong_frame.frame_idx,
                "time_sec": wrong_frame.time_sec,
                "reference_sample_idx": ref_idx,
                "score": score,
                "pose_detected": wrong_frame.pose_detected,
                "rep_index": rep_info["rep_index"][idx] if idx < len(rep_info["rep_index"]) else 1,
                "phase": rep_info["phase"][idx] if idx < len(rep_info["phase"]) else "steady",
                "issues": round_nested(active_issues),
                "coach_text": " ".join(messages[:2]),
                "wrong": {
                    "landmarks2d": round_nested(wrong_frame.landmarks2d),
                    "world_landmarks": round_nested(wrong_frame.world_landmarks),
                    "metrics": round_nested(wrong_metrics),
                },
                "reference": {
                    "landmarks2d": round_nested(correct_frame.landmarks2d),
                    "world_landmarks": round_nested(correct_frame.world_landmarks),
                    "metrics": round_nested(correct_metrics),
                },
                "highlighted_joint_names": sorted(highlighted_joint_names),
            }
        )

    issue_summary = []
    for issue_id, items in issue_buckets.items():
        if not items:
            continue
        avg_delta = float(np.mean([item["delta"] for item in items]))
        peak_delta = float(np.max([item["delta"] for item in items]))
        avg_severity = float(np.mean([item["severity"] for item in items]))
        issue_summary.append(
            {
                "id": issue_id,
                "label": ISSUE_SPECS[issue_id]["label"],
                "frame_hits": len(items),
                "avg_delta": avg_delta,
                "peak_delta": peak_delta,
                "avg_severity": avg_severity,
            }
        )
    issue_summary.sort(key=lambda item: (item["avg_severity"], item["frame_hits"]), reverse=True)

    segments = build_issue_segments(frame_payloads)

    return {
        "frames": frame_payloads,
        "issue_summary": round_nested(issue_summary),
        "issue_segments": round_nested(segments),
        "rep_info": rep_info,
        "average_score": round(float(np.mean(score_values)), 2) if score_values else 0.0,
    }


def joint_indices_from_names(side: str, names: set[str]) -> set[int]:
    mapping = LEFT_JOINT_SET if side == "left" else RIGHT_JOINT_SET
    return {mapping[name] for name in names if name in mapping}


def transform_reference_landmarks(
    ref_landmarks: list[list[float]] | None,
    wrong_landmarks: list[list[float]] | None,
) -> list[list[float]] | None:
    if ref_landmarks is None or wrong_landmarks is None:
        return None

    ref_hip = midpoint(ref_landmarks, POSE_LANDMARK.LEFT_HIP.value, POSE_LANDMARK.RIGHT_HIP.value)
    wrong_hip = midpoint(wrong_landmarks, POSE_LANDMARK.LEFT_HIP.value, POSE_LANDMARK.RIGHT_HIP.value)
    ref_shoulder = midpoint(ref_landmarks, POSE_LANDMARK.LEFT_SHOULDER.value, POSE_LANDMARK.RIGHT_SHOULDER.value)
    wrong_shoulder = midpoint(wrong_landmarks, POSE_LANDMARK.LEFT_SHOULDER.value, POSE_LANDMARK.RIGHT_SHOULDER.value)
    if ref_hip is None or wrong_hip is None or ref_shoulder is None or wrong_shoulder is None:
        return None

    ref_vec = ref_shoulder[:2] - ref_hip[:2]
    wrong_vec = wrong_shoulder[:2] - wrong_hip[:2]
    ref_scale = max(float(np.linalg.norm(ref_vec)), 1e-6)
    wrong_scale = max(float(np.linalg.norm(wrong_vec)), 1e-6)
    scale = wrong_scale / ref_scale

    ref_angle = math.atan2(ref_vec[1], ref_vec[0])
    wrong_angle = math.atan2(wrong_vec[1], wrong_vec[0])
    rotation = wrong_angle - ref_angle
    rotation_matrix = np.array(
        [
            [math.cos(rotation), -math.sin(rotation)],
            [math.sin(rotation), math.cos(rotation)],
        ],
        dtype=np.float64,
    )

    transformed: list[list[float]] = []
    for point in ref_landmarks:
        vec = np.asarray(point[:2], dtype=np.float64) - ref_hip[:2]
        rotated = (rotation_matrix @ vec) * scale
        new_xy = wrong_hip[:2] + rotated
        transformed.append([float(new_xy[0]), float(new_xy[1]), float(point[2]), float(point[3])])
    return transformed


def draw_pose_overlay(
    frame: np.ndarray,
    actual_landmarks: list[list[float]] | None,
    reference_landmarks: list[list[float]] | None,
    highlighted_indices: set[int],
) -> np.ndarray:
    height, width = frame.shape[:2]
    output = frame.copy()

    if reference_landmarks is not None:
        ghost_layer = output.copy()
        for a_idx, b_idx in CONNECTIONS:
            a = safe_point(reference_landmarks, a_idx)
            b = safe_point(reference_landmarks, b_idx)
            if a is None or b is None:
                continue
            cv2.line(
                ghost_layer,
                (int(a[0] * width), int(a[1] * height)),
                (int(b[0] * width), int(b[1] * height)),
                (255, 208, 79),
                5,
                cv2.LINE_AA,
            )
        output = cv2.addWeighted(ghost_layer, 0.3, output, 0.7, 0.0)

    if actual_landmarks is not None:
        for a_idx, b_idx in CONNECTIONS:
            a = safe_point(actual_landmarks, a_idx)
            b = safe_point(actual_landmarks, b_idx)
            if a is None or b is None:
                continue
            color = (82, 223, 255)
            if a_idx in highlighted_indices or b_idx in highlighted_indices:
                color = (54, 67, 244)
            cv2.line(
                output,
                (int(a[0] * width), int(a[1] * height)),
                (int(b[0] * width), int(b[1] * height)),
                color,
                4,
                cv2.LINE_AA,
            )

        for idx, point in enumerate(actual_landmarks):
            x = int(point[0] * width)
            y = int(point[1] * height)
            radius = 7 if idx in highlighted_indices else 5
            color = (54, 67, 244) if idx in highlighted_indices else (255, 247, 236)
            cv2.circle(output, (x, y), radius, color, -1, cv2.LINE_AA)

    return output


def draw_hud(frame: np.ndarray, frame_data: dict[str, Any]) -> np.ndarray:
    output = frame.copy()
    width = output.shape[1]
    badge_height = 170
    cv2.rectangle(output, (24, 24), (width - 24, 24 + badge_height), (14, 17, 20), -1)
    cv2.rectangle(output, (24, 24), (width - 24, 24 + badge_height), (34, 42, 50), 2)

    issue_names = {
        "torso_lean": "Torso lean",
        "depth": "Depth gap",
        "knee_flexion": "Knee flexion",
        "shin_lean": "Shin travel",
    }

    score = frame_data["score"]
    score_text = f"POSE SCORE {score:.0f}"
    status = "ON TRACK" if score >= 85 else "NEEDS FIX"
    issue_labels = ", ".join(issue_names.get(issue["id"], issue["id"]) for issue in frame_data["issues"][:2]) or "Aligned"
    if frame_data["issues"]:
        top_issue = frame_data["issues"][0]
        if top_issue["id"] == "torso_lean":
            summary = f"Reduce torso lean by {top_issue['delta']:.1f} deg."
        elif top_issue["id"] == "depth":
            summary = "Sit slightly deeper to match the target path."
        elif top_issue["id"] == "knee_flexion":
            summary = f"Add about {top_issue['delta']:.1f} deg of knee flexion."
        else:
            summary = "Let the knees travel a bit more forward."
    else:
        summary = "Tracking close to the reference squat."
    rep_text = f"Rep {frame_data['rep_index']} - {frame_data['phase']}"

    cv2.putText(output, score_text, (48, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (82, 223, 255), 3, cv2.LINE_AA)
    cv2.putText(output, status, (48, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 208, 79), 2, cv2.LINE_AA)
    cv2.putText(output, rep_text, (48, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(output, issue_labels[:56], (370, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 247, 236), 2, cv2.LINE_AA)
    cv2.putText(output, summary[:74], (370, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (197, 208, 222), 2, cv2.LINE_AA)
    cv2.putText(output, MODEL_NAME, (370, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (148, 166, 187), 1, cv2.LINE_AA)
    return output


def render_overlay_video(
    wrong_video_path: Path,
    comparison: dict[str, Any],
    output_path: Path,
    side: str,
    sampled_fps: float,
) -> None:
    cap = cv2.VideoCapture(str(wrong_video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open wrong video for overlay rendering: {wrong_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    sample_interval = max(int(round(fps / sampled_fps)), 1) if fps else 3
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        sampled_fps,
        (width, height),
    )

    frames = comparison["frames"]
    sample_idx = 0
    frame_idx = 0
    progress = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), desc="overlay", unit="f")

    while True:
        ok, frame = cap.read()
        if not ok or sample_idx >= len(frames):
            break

        if frame_idx % sample_interval == 0:
            frame_data = frames[sample_idx]
            ref_landmarks = transform_reference_landmarks(
                frame_data["reference"]["landmarks2d"],
                frame_data["wrong"]["landmarks2d"],
            )
            highlighted = joint_indices_from_names(side, set(frame_data["highlighted_joint_names"]))
            composed = draw_pose_overlay(
                frame,
                frame_data["wrong"]["landmarks2d"],
                ref_landmarks,
                highlighted,
            )
            composed = draw_hud(composed, frame_data)
            writer.write(composed)
            sample_idx += 1

        frame_idx += 1
        progress.update(1)

    progress.close()
    writer.release()
    cap.release()


def build_overview(comparison: dict[str, Any], wrong: dict[str, Any]) -> dict[str, Any]:
    issues = comparison["issue_summary"]
    top_issue = issues[0]["label"] if issues else "안정적"
    rep_count = comparison["rep_info"]["count"]
    return {
        "exercise": EXERCISE_NAME,
        "model": MODEL_NAME,
        "target_fps": TARGET_FPS,
        "average_score": comparison["average_score"],
        "rep_count": rep_count,
        "primary_side": wrong["primary_side"],
        "headline": f"{EXERCISE_NAME} single-camera correction prototype",
        "summary": f"MediaPipe Pose로 {wrong['primary_side']} 측 랜드마크를 기준 삼아 wrong 영상을 correct 기준 궤적에 맞췄고, 가장 큰 이탈은 '{top_issue}'로 요약됐습니다.",
        "top_findings": issues[:3],
    }


def serialize_sequence(sequence: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": sequence["role"],
        "path": sequence["path"],
        "width": sequence["width"],
        "height": sequence["height"],
        "fps": round(sequence["fps"], 4),
        "sampled_fps": round(sequence["sampled_fps"], 4),
        "frame_count": sequence["frame_count"],
        "sample_count": len(sequence["frames"]),
        "duration_sec": round(sequence["duration_sec"], 4),
        "primary_side": sequence["primary_side"],
    }


def ensure_directories(output_dir: Path) -> tuple[Path, Path]:
    data_dir = output_dir / "data"
    media_dir = output_dir / "media"
    data_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, media_dir


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    data_dir, media_dir = ensure_directories(output_dir)

    correct = detect_pose_sequence(args.correct.resolve(), "correct", args.target_fps)
    wrong = detect_pose_sequence(args.wrong.resolve(), "wrong", args.target_fps)
    side = wrong["primary_side"]
    comparison = compare_sequences(correct, wrong, side)

    overlay_path = media_dir / "wrong_overlay.mp4"
    render_overlay_video(args.wrong.resolve(), comparison, overlay_path, side, wrong["sampled_fps"])

    payload = {
        "overview": build_overview(comparison, wrong),
        "input_videos": {
            "correct": serialize_sequence(correct),
            "wrong": serialize_sequence(wrong),
        },
        "landmark_index": LANDMARK_INDEX,
        "connections": CONNECTIONS,
        "issue_segments": comparison["issue_segments"],
        "frames": comparison["frames"],
        "media": {
            "overlay_video": "media/wrong_overlay.mp4",
        },
    }

    analysis_path = data_dir / "analysis.json"
    analysis_path.write_text(
        json.dumps(round_nested(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Analysis written to {analysis_path}")
    print(f"Overlay video written to {overlay_path}")


if __name__ == "__main__":
    main()
