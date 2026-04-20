#!/usr/bin/env python3
from __future__ import annotations

import base64
import heapq
import io
import math
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import requests
from PIL import Image

_GUI_IMPORT_ERROR: Exception | None = None
try:
    import tkinter as tk
    from PIL import ImageTk
    from tkinter import messagebox, scrolledtext, ttk
except Exception as exc:  # pragma: no cover
    _GUI_IMPORT_ERROR = exc
    tk = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    scrolledtext = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _extra in (_REPO_ROOT / "place_trajectory", _REPO_ROOT / "push"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import place_trajectory_planning_service as place_service
import placedof_planner
import push_trajectory_planner as push_planner

DEFAULT_LEFT_IMAGE = _THIS_DIR / "data" / "left.png"
DEFAULT_RIGHT_IMAGE = _THIS_DIR / "data" / "right.png"
DEFAULT_TOPIC_INPUTS = _THIS_DIR / "data"  / "raw_topic_inputs.txt"
FIXED_S2M2_URL = str(place_service.DEFAULT_S2M2_API)
FIXED_SAM3_URL = str(place_service.DEFAULT_SAM3_URL)

DEFAULT_MAX_RENDER_POINTS = 500000
DEFAULT_MAX_COLLISION_POINTS = 40000
DEFAULT_MAX_COMPLETION_POINTS = 180000
DEFAULT_MAX_OBJECT_DISPLAY_POINTS = 5000
DEFAULT_MAX_TABLE_PLANE_DISPLAY_POINTS = 12000
DEFAULT_MOVE_STEP_M = 0.02
DEFAULT_ROT_STEP_DEG = 5.0
DEFAULT_OBJECT_PLANE_CLEARANCE_M = 0.004
DEFAULT_OBJECT_MIN_HEIGHT_M = 0.015
DEFAULT_OBJECT_MIN_CROSS_SECTION_M = 0.006
DEFAULT_OBJECT_MAX_HEIGHT_M = 0.40
DEFAULT_PLANE_RANSAC_THR_M = 0.012
DEFAULT_COLLISION_MARGIN_M = 0.003
DEFAULT_PLANE_CONTACT_EPS_M = 0.0015
DEFAULT_TABLE_PLANE_EXCLUDE_M = 0.010
DEFAULT_SLENDER_OBJECT_ASPECT_RATIO = 3.0
DEFAULT_SLENDER_OBJECT_LENGTH_RECOVER_RATIO = 0.75
DEFAULT_OBJECT_HEIGHT_QUANTILE = 0.96
DEFAULT_SLENDER_OBJECT_HEIGHT_QUANTILE = 0.90
DEFAULT_SLENDER_OBJECT_THICKNESS_CAP_RATIO = 1.35
DEFAULT_PLACE_CAMERA_PIXEL_MARGIN_PX = 12
DEFAULT_PLACE_VISIBILITY_LIFT_STEP_M = 0.015
DEFAULT_PLACE_VISIBILITY_MAX_LIFT_M = 0.24
DEFAULT_PLACE_VISIBILITY_OPENING_STEP_M = 0.015
DEFAULT_PLACE_VISIBILITY_EXTRA_OPENING_M = 0.12
DEFAULT_PLACE_OPENING_EXPOSURE_MIN_M = 0.015
DEFAULT_PLACE_OPENING_EXPOSURE_RATIO = 0.90
DEFAULT_PLACE_OPENING_EXPOSURE_SLENDER_RATIO = 0.25
DEFAULT_PLACE_OPENING_SHIFT_MAX_M = 0.30
DEFAULT_PLACE_TRAJECTORY_VOXEL_M = 0.03
DEFAULT_PLACE_TRAJECTORY_MAX_ASTAR_ITERS = 160000
DEFAULT_PLACE_TRAJECTORY_BOUNDS_MARGIN_M = 0.18
DEFAULT_PLACE_TRAJECTORY_Z_LIFT_MARGIN_M = 0.12
DEFAULT_PLACE_TRAJECTORY_ROT_STEP_DEG = 12.0
DEFAULT_PLACE_TRAJECTORY_MAX_ROTATION_SAMPLES = 5
DEFAULT_PLACE_TRAJECTORY_ROTATION_WEIGHT_M = 0.10
DEFAULT_PLACE_TRAJECTORY_TRANSITION_STEP_M = 0.02
DEFAULT_PLACE_TRAJECTORY_SMOOTH_PASSES = 3
DEFAULT_PLACE_TRAJECTORY_SMOOTH_MAX_LIFT_M = 0.18
DEFAULT_PLACE_TRAJECTORY_SMOOTH_LIFT_WEIGHT = 0.35
DEFAULT_PLACE_TRAJECTORY_MAX_RENDER_FRAMES = 18
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_RENDER_BG_RGB = (14, 20, 28)
MAX_IMAGE_DISPLAY_WIDTH = 560
MAX_IMAGE_DISPLAY_HEIGHT = 420
DEFAULT_MAX_COORD_ABS = 1e6
XYZ_AXIS_COLORS_BGR = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
)
TABLE_PLANE_OVERLAY_RGB = np.array([0.86, 0.92, 1.0], dtype=np.float32)


def _empty_points() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


def _empty_colors() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


def _empty_pixels() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.int32)


def _finite_points(points: np.ndarray, max_abs: float = DEFAULT_MAX_COORD_ABS) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return _empty_points()
    keep = _finite_point_mask(pts, max_abs=max_abs)
    if not np.any(keep):
        return _empty_points()
    return pts[keep].astype(np.float32, copy=False)


def _finite_point_mask(points: np.ndarray, max_abs: float = DEFAULT_MAX_COORD_ABS) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.zeros((0,), dtype=bool)
    keep = np.all(np.isfinite(pts), axis=1)
    keep &= np.max(np.abs(pts), axis=1) <= float(max_abs)
    return keep.astype(bool, copy=False)


def _is_finite_rotation(rot: np.ndarray) -> bool:
    arr = np.asarray(rot, dtype=np.float64)
    return bool(arr.shape == (3, 3) and np.all(np.isfinite(arr)))


def _is_finite_vector(vec: np.ndarray, length: int) -> bool:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    return bool(arr.shape[0] >= int(length) and np.all(np.isfinite(arr[:length])))


def _image_to_jpeg_b64(image: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=int(quality), optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_sam3_generic(
    image: Image.Image,
    sam3_url: str,
    timeout_s: float,
    text_prompt: str = "",
    box_prompts: list[list[int]] | None = None,
    box_labels: list[bool] | None = None,
    point_prompts: list[list[int]] | None = None,
    point_labels: list[int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"image": _image_to_jpeg_b64(image)}
    if text_prompt:
        payload["text_prompt"] = str(text_prompt)
    if box_prompts:
        payload["box_prompts"] = [[int(v) for v in box] for box in box_prompts]
        payload["box_labels"] = [bool(v) for v in (box_labels or [True] * len(box_prompts))]
    if point_prompts:
        payload["point_prompts"] = [[int(v) for v in point] for point in point_prompts]
        payload["point_labels"] = [int(v) for v in (point_labels or [1] * len(point_prompts))]

    out: dict[str, Any] = {
        "ok": False,
        "latency_ms": 0,
        "num_detections": 0,
        "error": "",
        "mask": None,
        "mask_area_px": 0,
    }

    t0 = time.perf_counter()
    try:
        response = requests.post(
            f"{sam3_url.rstrip('/')}/segment",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=float(timeout_s),
        )
    except requests.RequestException as exc:
        out["latency_ms"] = int(round((time.perf_counter() - t0) * 1000.0))
        out["error"] = f"sam3_request_error: {exc}"
        return out

    out["latency_ms"] = int(round((time.perf_counter() - t0) * 1000.0))
    if response.status_code >= 400:
        out["error"] = f"sam3_http_{response.status_code}: {response.text}"
        return out

    try:
        data = response.json()
    except ValueError:
        out["error"] = "sam3_invalid_json"
        return out

    if not data.get("success", False):
        out["error"] = str(data.get("error", "sam3_failed"))
        return out

    detections = data.get("detections") or []
    out["num_detections"] = int(len(detections))
    width, height = image.size
    mask = np.zeros((height, width), dtype=bool)
    for det in detections:
        if not isinstance(det, dict):
            continue
        mask_b64 = det.get("mask")
        if not isinstance(mask_b64, str) or not mask_b64:
            continue
        try:
            raw = base64.b64decode(mask_b64)
            m_img = Image.open(io.BytesIO(raw)).convert("L")
            if m_img.size != (width, height):
                m_img = m_img.resize((width, height), Image.Resampling.NEAREST)
            mask = np.logical_or(mask, np.asarray(m_img, dtype=np.uint8) > 127)
        except Exception:
            continue

    if mask.any():
        out["ok"] = True
        out["mask"] = mask
        out["mask_area_px"] = int(mask.sum())
    else:
        out["error"] = "sam3_empty_mask"
    return out


def _bbox_to_mask(bbox: tuple[int, int, int, int], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    mask = np.zeros((height, width), dtype=bool)
    mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not np.any(m):
        return None
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    ratio: float = 0.18,
    min_pad_px: int = 18,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad_x = max(int(round((x2 - x1 + 1) * float(ratio))), int(min_pad_px))
    pad_y = max(int(round((y2 - y1 + 1) * float(ratio))), int(min_pad_px))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(int(width) - 1, x2 + pad_x),
        min(int(height) - 1, y2 + pad_y),
    )


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _subsample_indices(length: int, max_points: int, seed: int = 0) -> np.ndarray:
    if max_points <= 0 or length <= int(max_points):
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(length, size=int(max_points), replace=False))


def _subsample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return _empty_points()
    idx = _subsample_indices(pts.shape[0], int(max_points), seed=seed)
    return pts[idx].astype(np.float32, copy=False)


def _overlay_mask(image: Image.Image, mask: np.ndarray, color_rgb: tuple[int, int, int], alpha: int) -> Image.Image:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not np.any(m):
        return image
    base = image.convert("RGBA")
    overlay = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = int(color_rgb[0])
    overlay[..., 1] = int(color_rgb[1])
    overlay[..., 2] = int(color_rgb[2])
    overlay[..., 3] = np.where(m, int(alpha), 0).astype(np.uint8)
    comp = Image.alpha_composite(base, Image.fromarray(overlay))
    return comp.convert("RGB")


def _safe_quantile(values: np.ndarray, q: float, default: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, float(q)))


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _orthonormalize_axes(axes: np.ndarray) -> np.ndarray:
    arr = np.asarray(axes, dtype=np.float64).reshape(3, 3)
    x_axis = arr[:, 0]
    y_axis = arr[:, 1]
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-8)
    y_axis = y_axis - x_axis * float(np.dot(x_axis, y_axis))
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-8)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-8)
    out = np.stack([x_axis, y_axis, z_axis], axis=1)
    if float(np.linalg.det(out)) < 0.0:
        out[:, 1] *= -1.0
        out[:, 2] = np.cross(out[:, 0], out[:, 1])
        out[:, 2] = out[:, 2] / max(float(np.linalg.norm(out[:, 2])), 1e-8)
    return out.astype(np.float32, copy=False)


def _rpy_to_matrix(rpy_rad: np.ndarray) -> np.ndarray:
    roll = float(rpy_rad[0])
    pitch = float(rpy_rad[1])
    yaw = float(rpy_rad[2])
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (rz @ ry @ rx).astype(np.float64, copy=False)


def _rotation_geodesic_angle(axes_a: np.ndarray, axes_b: np.ndarray) -> float:
    rot_a = np.asarray(axes_a, dtype=np.float64).reshape(3, 3)
    rot_b = np.asarray(axes_b, dtype=np.float64).reshape(3, 3)
    rel = rot_a.T @ rot_b
    cos_theta = 0.5 * (float(np.trace(rel)) - 1.0)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(math.acos(cos_theta))


def _rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(unit))
    if not np.isfinite(norm) or norm <= 1e-8:
        return np.eye(3, dtype=np.float64)
    x, y, z = (unit / norm).tolist()
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    ident = np.eye(3, dtype=np.float64)
    s = math.sin(float(angle_rad))
    c = math.cos(float(angle_rad))
    return ident + s * skew + (1.0 - c) * (skew @ skew)


def _interpolate_axes_shortest(axes_start: np.ndarray, axes_end: np.ndarray, t: float) -> np.ndarray:
    tau = float(np.clip(t, 0.0, 1.0))
    if tau <= 1e-8:
        return _orthonormalize_axes(axes_start)
    if tau >= 1.0 - 1e-8:
        return _orthonormalize_axes(axes_end)
    rot_start = _orthonormalize_axes(axes_start).astype(np.float64, copy=False)
    rot_end = _orthonormalize_axes(axes_end).astype(np.float64, copy=False)
    rel = rot_start.T @ rot_end
    cos_theta = float(np.clip(0.5 * (float(np.trace(rel)) - 1.0), -1.0, 1.0))
    angle = float(math.acos(cos_theta))
    if angle <= 1e-8:
        return rot_start.astype(np.float32, copy=False)
    denom = 2.0 * math.sin(angle)
    if abs(denom) > 1e-6:
        axis_local = np.array(
            [
                rel[2, 1] - rel[1, 2],
                rel[0, 2] - rel[2, 0],
                rel[1, 0] - rel[0, 1],
            ],
            dtype=np.float64,
        ) / denom
    else:
        # Near 180 deg, the skew-symmetric part collapses toward zero and a naive
        # axis extraction causes the interpolation to stay at rot_start until tau=1.
        diag = np.clip(0.5 * (np.diag(rel) + 1.0), 0.0, None)
        axis_local = np.sqrt(diag).astype(np.float64, copy=False)
        if float(np.linalg.norm(axis_local)) <= 1e-6:
            axis_local = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            if axis_local[0] >= axis_local[1] and axis_local[0] >= axis_local[2] and axis_local[0] > 1e-6:
                axis_local[1] = math.copysign(axis_local[1], rel[0, 1] + rel[1, 0])
                axis_local[2] = math.copysign(axis_local[2], rel[0, 2] + rel[2, 0])
            elif axis_local[1] >= axis_local[2] and axis_local[1] > 1e-6:
                axis_local[0] = math.copysign(axis_local[0], rel[0, 1] + rel[1, 0])
                axis_local[2] = math.copysign(axis_local[2], rel[1, 2] + rel[2, 1])
            else:
                axis_local[0] = math.copysign(axis_local[0], rel[0, 2] + rel[2, 0])
                axis_local[1] = math.copysign(axis_local[1], rel[1, 2] + rel[2, 1])
    norm = float(np.linalg.norm(axis_local))
    if not np.isfinite(norm) or norm <= 1e-8:
        axis_local = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis_local = axis_local / norm
    interp = rot_start @ _rotation_about_axis(axis_local, tau * angle)
    return _orthonormalize_axes(interp)


def _box_corners_world(center: np.ndarray, rpy_rad: np.ndarray, size_xyz: np.ndarray) -> np.ndarray:
    ctr = np.asarray(center, dtype=np.float64).reshape(3)
    size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
    half = 0.5 * size
    rot = _rpy_to_matrix(rpy_rad)
    local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    return (local @ rot.T + ctr.reshape((1, 3))).astype(np.float32, copy=False)


def _box_corners_axes(center: np.ndarray, axes_world: np.ndarray, size_xyz: np.ndarray) -> np.ndarray:
    ctr = np.asarray(center, dtype=np.float64).reshape(3)
    size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
    axes = np.asarray(axes_world, dtype=np.float64).reshape(3, 3)
    half = 0.5 * size
    local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    return (local @ axes.T + ctr.reshape((1, 3))).astype(np.float32, copy=False)


def _cylinder_wire_points_axes(
    center: np.ndarray,
    axes_world: np.ndarray,
    radius: float,
    height: float,
    segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    ctr = np.asarray(center, dtype=np.float64).reshape(3)
    axes = np.asarray(axes_world, dtype=np.float64).reshape(3, 3)
    ex = axes[:, 0]
    ey = axes[:, 1]
    ez = axes[:, 2]
    radius = max(float(radius), 1e-4)
    half_h = 0.5 * max(float(height), 1e-4)
    thetas = np.linspace(0.0, 2.0 * math.pi, max(8, int(segments)), endpoint=False, dtype=np.float64)
    circle_xy = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    top = (
        ctr.reshape((1, 3))
        + (circle_xy[:, 0:1] * radius) * ex.reshape((1, 3))
        + (circle_xy[:, 1:2] * radius) * ey.reshape((1, 3))
        + half_h * ez.reshape((1, 3))
    )
    bottom = (
        ctr.reshape((1, 3))
        + (circle_xy[:, 0:1] * radius) * ex.reshape((1, 3))
        + (circle_xy[:, 1:2] * radius) * ey.reshape((1, 3))
        - half_h * ez.reshape((1, 3))
    )
    return top.astype(np.float32, copy=False), bottom.astype(np.float32, copy=False)


def _opening_face_axis_spec(face: str | None) -> tuple[int, float] | None:
    if face is None:
        return None
    face_key = str(face).strip().lower()
    if len(face_key) != 2 or face_key[0] not in {"+", "-"} or face_key[1] not in {"x", "y", "z"}:
        return None
    axis_idx = {"x": 0, "y": 1, "z": 2}[face_key[1]]
    sign = 1.0 if face_key[0] == "+" else -1.0
    return axis_idx, sign


def _sample_target_collision_points(profile: placedof_planner.TargetProfile) -> np.ndarray:
    center = np.asarray(profile.center_base, dtype=np.float64).reshape(3)
    axes = np.asarray(profile.rotation_base, dtype=np.float64).reshape(3, 3)
    size = np.asarray(profile.size_xyz, dtype=np.float64).reshape(3)
    half = 0.5 * size
    open_spec = _opening_face_axis_spec(profile.opening_face)
    face_samples: list[np.ndarray] = []
    sample_step = 0.012

    if profile.primitive_shape == "cylinder" and profile.primitive_radius is not None:
        radius = max(float(profile.primitive_radius), 1e-4)
        height = max(float(size[2]), 1e-4)
        side_segments = max(24, int(math.ceil(2.0 * math.pi * radius / sample_step)))
        side_layers = max(6, int(math.ceil(height / sample_step)) + 1)
        thetas = np.linspace(0.0, 2.0 * math.pi, side_segments, endpoint=False, dtype=np.float64)
        heights = np.linspace(-0.5 * height, 0.5 * height, side_layers, dtype=np.float64)
        ex = axes[:, 0]
        ey = axes[:, 1]
        ez = axes[:, 2]
        shell = []
        for h in heights.tolist():
            for theta in thetas.tolist():
                shell.append(
                    center
                    + radius * math.cos(theta) * ex
                    + radius * math.sin(theta) * ey
                    + h * ez
                )
        face_samples.append(np.asarray(shell, dtype=np.float32))
        if open_spec is None or open_spec[0] != 2 or open_spec[1] < 0.0:
            radial_layers = max(3, int(math.ceil(radius / sample_step)) + 1)
            bottom_pts = []
            for radial in np.linspace(0.0, radius, radial_layers, dtype=np.float64).tolist():
                ring_segments = max(12, int(math.ceil(2.0 * math.pi * max(radial, sample_step) / sample_step)))
                for theta in np.linspace(0.0, 2.0 * math.pi, ring_segments, endpoint=False, dtype=np.float64).tolist():
                    bottom_pts.append(
                        center
                        + radial * math.cos(theta) * ex
                        + radial * math.sin(theta) * ey
                        - 0.5 * height * ez
                    )
            if bottom_pts:
                face_samples.append(np.asarray(bottom_pts, dtype=np.float32))
    else:
        for axis_idx in range(3):
            other_axes = [idx for idx in range(3) if idx != axis_idx]
            spans = [max(2, int(math.ceil(size[idx] / sample_step)) + 1) for idx in other_axes]
            grid_u = np.linspace(-half[other_axes[0]], half[other_axes[0]], spans[0], dtype=np.float64)
            grid_v = np.linspace(-half[other_axes[1]], half[other_axes[1]], spans[1], dtype=np.float64)
            for sign in (-1.0, 1.0):
                if open_spec is not None and axis_idx == open_spec[0] and sign == open_spec[1]:
                    continue
                pts = []
                for u in grid_u.tolist():
                    for v in grid_v.tolist():
                        local = np.zeros((3,), dtype=np.float64)
                        local[axis_idx] = sign * half[axis_idx]
                        local[other_axes[0]] = u
                        local[other_axes[1]] = v
                        pts.append(center + axes @ local)
                if pts:
                    face_samples.append(np.asarray(pts, dtype=np.float32))

    if not face_samples:
        return _empty_points()
    points = _finite_points(np.vstack(face_samples).astype(np.float32, copy=False))
    if points.shape[0] > 6000:
        points = _subsample_points(points, 6000, seed=71)
    return points.astype(np.float32, copy=False)


def _fitted_target_face_annotations(profile: placedof_planner.TargetProfile) -> list[dict[str, Any]]:
    ctr = np.asarray(profile.center_base, dtype=np.float64).reshape(3)
    axes = np.asarray(profile.rotation_base, dtype=np.float64).reshape(3, 3)
    size = np.asarray(profile.size_xyz, dtype=np.float64).reshape(3)
    half = 0.5 * size
    annotations: list[dict[str, Any]] = []

    def _add(text: str, point: np.ndarray, state: str) -> None:
        annotations.append(
            {
                "text": str(text),
                "point": np.asarray(point, dtype=np.float32).reshape(3),
                "state": str(state),
            }
        )

    if profile.primitive_shape == "plane" or profile.kind == "support_surface":
        top = ctr + axes[:, 2] * max(half[2], 0.003)
        _add("+Z SUPPORT", top, "support")
        return annotations

    open_faces: set[str] = set()
    if profile.kind == "top_open_box":
        open_faces.add("+z")
    if profile.kind == "front_open_cabinet" and profile.opening_face is not None:
        open_faces.add(str(profile.opening_face).strip().lower())

    if profile.primitive_shape == "cylinder":
        top_state = "open" if "+z" in open_faces else "closed"
        top_label = "+Z OPEN" if top_state == "open" else "+Z CLOSED"
        _add(top_label, ctr + axes[:, 2] * half[2], top_state)
        _add("-Z CLOSED", ctr - axes[:, 2] * half[2], "closed")
        radius = float(profile.primitive_radius or (0.5 * min(size[0], size[1])))
        side_dirs = (
            ("+X CLOSED", axes[:, 0]),
            ("-X CLOSED", -axes[:, 0]),
            ("+Y CLOSED", axes[:, 1]),
            ("-Y CLOSED", -axes[:, 1]),
        )
        for text, direction in side_dirs:
            _add(text, ctr + direction * radius, "closed")
        return annotations

    face_defs = (
        ("+X", 0, 1.0, axes[:, 0]),
        ("-X", 0, -1.0, -axes[:, 0]),
        ("+Y", 1, 1.0, axes[:, 1]),
        ("-Y", 1, -1.0, -axes[:, 1]),
        ("+Z", 2, 1.0, axes[:, 2]),
        ("-Z", 2, -1.0, -axes[:, 2]),
    )
    for face_name, axis_idx, sign, direction in face_defs:
        face_key = face_name.lower()
        state = "open" if face_key in open_faces else "closed"
        label = f"{face_name} {'OPEN' if state == 'open' else 'CLOSED'}"
        point = ctr + direction * half[axis_idx]
        if axis_idx == 2:
            point = point + direction * max(0.002, 0.08 * half[2])
        _add(label, point, state)
    return annotations


def _obb_local_points(points_base: np.ndarray, center: np.ndarray, rpy_rad: np.ndarray) -> np.ndarray:
    pts = _finite_points(points_base).astype(np.float64, copy=False)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    ctr = np.asarray(center, dtype=np.float64).reshape(1, 3)
    rot = _rpy_to_matrix(rpy_rad)
    if not np.all(np.isfinite(ctr)) or not _is_finite_rotation(rot):
        return np.zeros((0, 3), dtype=np.float64)
    return (pts - ctr) @ rot


def _plane_margin_z(points_base: np.ndarray, normal_base: np.ndarray, d_base: float) -> np.ndarray:
    pts = _finite_points(points_base).astype(np.float64, copy=False)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    n = np.asarray(normal_base, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(n)) or not np.isfinite(float(d_base)):
        raise RuntimeError("Invalid plane model for base-Z margin.")
    nz = float(n[2])
    if abs(nz) <= 1e-6:
        raise RuntimeError("Invalid plane normal for base-Z margin.")
    return (pts @ n + float(d_base)) / nz


def _max_plane_height_under_footprint(
    center_xy: np.ndarray,
    yaw_rad: float,
    size_xy: np.ndarray,
    normal_base: np.ndarray,
    d_base: float,
) -> float:
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    sx = 0.5 * float(size_xy[0])
    sy = 0.5 * float(size_xy[1])
    rot2 = push_planner._rot2(float(yaw_rad))
    footprint_local = np.array(
        [
            [-sx, -sy],
            [sx, -sy],
            [sx, sy],
            [-sx, sy],
        ],
        dtype=np.float64,
    )
    heights: list[float] = []
    for local_xy in footprint_local:
        xy = center + rot2 @ local_xy
        z = push_planner._surface_z_from_plane_at_xy(normal_base, float(d_base), xy)
        if z is not None and np.isfinite(z):
            heights.append(float(z))
    if not heights:
        z0 = push_planner._surface_z_from_plane_at_xy(normal_base, float(d_base), center)
        if z0 is None or not np.isfinite(z0):
            raise RuntimeError("Failed to evaluate table plane under footprint.")
        return float(z0)
    return float(max(heights))


def _build_placeholder_view(width: int, height: int, message: str) -> Image.Image:
    img = np.zeros((max(32, height), max(32, width), 3), dtype=np.uint8)
    img[..., 0] = DEFAULT_RENDER_BG_RGB[2]
    img[..., 1] = DEFAULT_RENDER_BG_RGB[1]
    img[..., 2] = DEFAULT_RENDER_BG_RGB[0]
    cv2.putText(
        img,
        str(message),
        (24, max(48, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (215, 225, 235),
        2,
        cv2.LINE_AA,
    )
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


@dataclass
class CameraBundle:
    k1: np.ndarray
    baseline: float
    rot_cb: np.ndarray
    shift_cb: np.ndarray


@dataclass
class PlaneModel:
    normal_base: np.ndarray
    d_base: float
    support_points_base: np.ndarray
    source: str


@dataclass
class TableCompletionCache:
    mask: np.ndarray
    plane_points_base: np.ndarray
    plane_colors: np.ndarray
    plane_pixels_yx: np.ndarray
    plane_model: PlaneModel
    prompt_used: str
    debug: dict[str, Any]


@dataclass
class ObjectState:
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    mask_area_px: int
    points_base: np.ndarray
    fit_points_base: np.ndarray
    plane_model: PlaneModel
    center_base: np.ndarray
    rpy_rad: np.ndarray
    size_xyz: np.ndarray
    init_center_base: np.ndarray
    init_rpy_rad: np.ndarray
    debug: dict[str, Any]


@dataclass
class ViewState:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    zoom: float = 1.0


@dataclass
class GraspState:
    point_obj: np.ndarray
    axes_obj: np.ndarray
    init_point_obj: np.ndarray
    init_axes_obj: np.ndarray
    debug: dict[str, Any]


@dataclass
class PlaceTargetState:
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    mask_area_px: int
    points_base: np.ndarray
    fit_points_base: np.ndarray
    target_profile: placedof_planner.TargetProfile
    preferred_center_base: np.ndarray
    debug: dict[str, Any]


@dataclass
class PlacePlanState:
    object_center_base: np.ndarray
    object_axes_base: np.ndarray
    object_rpy_rad: np.ndarray
    gripper_origin_base: np.ndarray
    gripper_axes_base: np.ndarray
    object_center_trajectory_base: np.ndarray
    object_axes_trajectory_base: np.ndarray
    gripper_origin_trajectory_base: np.ndarray
    gripper_axes_trajectory_base: np.ndarray
    rule_id: str
    target_kind: str
    score: float
    matched_rules: list[str]
    score_breakdown: dict[str, float]
    debug: dict[str, Any]


class PlaceDofInteractiveDemo:
    def __init__(self) -> None:
        if _GUI_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Tk GUI is unavailable in this Python environment. "
                "Use a Python build with tkinter support."
            ) from _GUI_IMPORT_ERROR
        self.root = tk.Tk()
        self.root.title("PlaceDOF PointCloud Interactive Demo")
        self.root.geometry("1760x1000")

        self.left_image_path_var = tk.StringVar(value=str(DEFAULT_LEFT_IMAGE))
        self.right_image_path_var = tk.StringVar(value=str(DEFAULT_RIGHT_IMAGE))
        self.topic_inputs_path_var = tk.StringVar(value=str(DEFAULT_TOPIC_INPUTS))
        self.timeout_var = tk.StringVar(value=f"{DEFAULT_TIMEOUT_S:.1f}")
        self.max_points_var = tk.StringVar(value=str(DEFAULT_MAX_RENDER_POINTS))
        self.move_step_var = tk.StringVar(value=f"{DEFAULT_MOVE_STEP_M:.3f}")
        self.rot_step_var = tk.StringVar(value=f"{DEFAULT_ROT_STEP_DEG:.1f}")
        self.place_target_kind_var = tk.StringVar(value=placedof_planner.DEFAULT_TARGET_KIND)
        self.place_target_fit_overlay_label_var = tk.StringVar(value="Hide Fitted Target")
        self.status_var = tk.StringVar(value="Load inputs and run S2M2.")
        self.enable_table_completion_var = tk.BooleanVar(value=False)
        self.free_pose_var = tk.BooleanVar(value=True)
        self.show_place_target_fit_var = tk.BooleanVar(value=True)
        self.show_place_target_axes_var = tk.BooleanVar(value=False)
        self.show_object_points_var = tk.BooleanVar(value=False)
        self.show_object_fit_points_var = tk.BooleanVar(value=False)
        self.show_pick_obb_axes_var = tk.BooleanVar(value=False)
        self.show_grasp_axes_var = tk.BooleanVar(value=True)
        self.show_place_obb_axes_var = tk.BooleanVar(value=False)
        self.show_place_grasp_axes_var = tk.BooleanVar(value=True)

        self.camera_bundle: CameraBundle | None = None
        self.left_image_full: Image.Image | None = None
        self.right_image_full: Image.Image | None = None
        self.left_image_np: np.ndarray | None = None

        self.raw_dense_map_cam: np.ndarray | None = None
        self.raw_valid_mask: np.ndarray | None = None
        self.raw_points_base: np.ndarray = _empty_points()
        self.raw_colors: np.ndarray = _empty_colors()
        self.raw_pixels_yx: np.ndarray = _empty_pixels()

        self.table_cache: TableCompletionCache | None = None
        self.object_state: ObjectState | None = None
        self.grasp_state: GraspState | None = None
        self.place_target_state: PlaceTargetState | None = None
        self.place_plan_state: PlacePlanState | None = None
        self.scene_points_base: np.ndarray = _empty_points()
        self.scene_colors: np.ndarray = _empty_colors()
        self.collision_points_base: np.ndarray = _empty_points()

        self.selected_bbox: tuple[int, int, int, int] | None = None
        self.segment_scene_bbox: tuple[int, int, int, int] | None = None
        self.preview_bbox: tuple[int, int, int, int] | None = None
        self._bbox_drag_start: tuple[int, int] | None = None
        self.awaiting_grasp_click = False

        self.left_display_image: Image.Image | None = None
        self.left_display_photo: ImageTk.PhotoImage | None = None
        self.left_canvas_image_id: int | None = None
        self.left_display_scale_x = 1.0
        self.left_display_scale_y = 1.0
        self.left_bbox_item_id: int | None = None
        self.left_preview_item_id: int | None = None

        self.viewer_state = ViewState()
        self.viewer_drag_last: tuple[int, int] | None = None
        self.viewer_photo: ImageTk.PhotoImage | None = None
        self.viewer_image_id: int | None = None
        self.plan_window: tk.Toplevel | None = None
        self.info_text: Any | None = None

        self.busy = False
        self._build_ui()
        self._bind_hotkeys()
        self._load_inputs_from_paths(show_success=False)

    def _build_ui(self) -> None:
        menu_bar = tk.Menu(self.root)
        window_menu = tk.Menu(menu_bar, tearoff=False)
        window_menu.add_command(label="Show Pick / Place Planning", command=self._show_plan_window)
        menu_bar.add_cascade(label="Window", menu=window_menu)
        self.root.config(menu=menu_bar)

        viewer_panel = ttk.LabelFrame(self.root, text="PointCloud")
        viewer_panel.pack(fill="both", expand=True, padx=8, pady=8)
        self.viewer_canvas = tk.Canvas(viewer_panel, bg="#0e141c", highlightthickness=0)
        self.viewer_canvas.pack(fill="both", expand=True)
        self.viewer_canvas.bind("<Configure>", lambda _evt: self._render_viewer())
        self.viewer_canvas.bind("<ButtonPress-1>", self._on_viewer_press)
        self.viewer_canvas.bind("<B1-Motion>", self._on_viewer_drag)
        self.viewer_canvas.bind("<ButtonRelease-1>", self._on_viewer_release)
        self.viewer_canvas.bind("<MouseWheel>", self._on_viewer_wheel)
        self.viewer_canvas.bind("<Button-4>", self._on_viewer_wheel)
        self.viewer_canvas.bind("<Button-5>", self._on_viewer_wheel)
        self._build_plan_window()

    def _add_labeled_entry(self, parent: ttk.Widget, label: str, var: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(row, text=label, width=12).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=(6, 0))

    def _add_button_grid(
        self,
        parent: ttk.Widget,
        buttons: list[tuple[str, Callable[[], None]]],
        columns: int,
        pady: tuple[int, int] = (0, 6),
    ) -> None:
        grid = ttk.Frame(parent)
        grid.pack(fill="x", padx=8, pady=pady)
        col_count = max(1, int(columns))
        for col in range(col_count):
            grid.columnconfigure(col, weight=1, uniform="button-grid")
        for idx, (text, command) in enumerate(buttons):
            row = idx // col_count
            col = idx % col_count
            ttk.Button(grid, text=text, command=command).grid(
                row=row,
                column=col,
                sticky="ew",
                padx=2,
                pady=2,
            )

    def _build_plan_window(self) -> None:
        self.plan_window = tk.Toplevel(self.root)
        self.plan_window.title("Pick / Place Planning")
        self.plan_window.geometry("1360x980")
        self.plan_window.protocol("WM_DELETE_WINDOW", self._hide_plan_window)

        container = ttk.Frame(self.plan_window, padding=8)
        container.pack(fill="both", expand=True)

        layout = ttk.Panedwindow(container, orient=tk.HORIZONTAL)
        layout.pack(fill="both", expand=True)

        controls = ttk.Frame(layout, padding=4)
        layout.add(controls, weight=3)

        info_wrap = ttk.Frame(layout, padding=4)
        layout.add(info_wrap, weight=2)

        source_panel = ttk.LabelFrame(controls, text="Inputs")
        source_panel.pack(fill="x")

        self._add_labeled_entry(source_panel, "Left Image", self.left_image_path_var)
        self._add_labeled_entry(source_panel, "Right Image", self.right_image_path_var)
        self._add_labeled_entry(source_panel, "Topic Inputs", self.topic_inputs_path_var)

        row = ttk.Frame(source_panel)
        row.pack(fill="x", padx=8, pady=(2, 8))
        ttk.Label(row, text="Timeout(s)").pack(side="left")
        timeout_entry = ttk.Entry(row, textvariable=self.timeout_var, width=8)
        timeout_entry.pack(side="left", padx=(6, 10))
        ttk.Label(row, text="Max Points").pack(side="left")
        max_points_entry = ttk.Entry(row, textvariable=self.max_points_var, width=9)
        max_points_entry.pack(side="left", padx=(6, 10))
        max_points_entry.bind("<Return>", lambda _evt: self._render_viewer())
        ttk.Checkbutton(
            row,
            text="Table Completion",
            variable=self.enable_table_completion_var,
            command=self._on_toggle_table_completion,
        ).pack(side="left")

        btn_row = ttk.Frame(source_panel)
        btn_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(btn_row, text="Load Inputs", command=self._load_inputs_from_paths).pack(side="left")
        ttk.Button(btn_row, text="Run S2M2", command=self._start_s2m2).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Segment Scene", command=self._apply_segment_scene_from_bbox).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Segment Object", command=self._start_segment_object).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Clear Object", command=self._clear_object).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Clear BBox", command=self._clear_bbox).pack(side="left", padx=(8, 0))

        object_overlay_row = ttk.Frame(source_panel)
        object_overlay_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Checkbutton(
            object_overlay_row,
            text="Show Object Cloud",
            variable=self.show_object_points_var,
            command=self._render_viewer,
        ).pack(side="left")
        ttk.Checkbutton(
            object_overlay_row,
            text="Show Object Fit Cloud",
            variable=self.show_object_fit_points_var,
            command=self._render_viewer,
        ).pack(side="left", padx=(10, 0))

        image_panel = ttk.LabelFrame(controls, text="Left Image (drag a bbox)")
        image_panel.pack(fill="both", expand=False, pady=(8, 8))
        self.left_canvas = tk.Canvas(image_panel, width=MAX_IMAGE_DISPLAY_WIDTH, height=MAX_IMAGE_DISPLAY_HEIGHT, bg="black")
        self.left_canvas.pack(fill="both", expand=True)
        self.left_canvas.bind("<ButtonPress-1>", self._on_left_canvas_press)
        self.left_canvas.bind("<B1-Motion>", self._on_left_canvas_drag)
        self.left_canvas.bind("<ButtonRelease-1>", self._on_left_canvas_release)

        pose_panel = ttk.LabelFrame(controls, text="Box Pose Controls")
        pose_panel.pack(fill="x")

        pose_row = ttk.Frame(pose_panel)
        pose_row.pack(fill="x", padx=8, pady=(8, 6))
        ttk.Label(pose_row, text="Move Step(m)").pack(side="left")
        ttk.Entry(pose_row, textvariable=self.move_step_var, width=8).pack(side="left", padx=(6, 12))
        ttk.Label(pose_row, text="Rot Step(deg)").pack(side="left")
        ttk.Entry(pose_row, textvariable=self.rot_step_var, width=8).pack(side="left", padx=(6, 12))
        ttk.Checkbutton(
            pose_row,
            text="Free Pose",
            variable=self.free_pose_var,
        ).pack(side="left")
        ttk.Button(pose_row, text="Release / Drop", command=self._release_box_pose).pack(side="left", padx=(10, 0))
        ttk.Button(pose_row, text="Reset Box", command=self._reset_box_pose).pack(side="left", padx=(8, 0))

        move_row = ttk.Frame(pose_panel)
        move_row.pack(fill="x", padx=8, pady=(0, 4))
        for text, dx, dy, dz in (
            ("X-", -1, 0, 0),
            ("X+", 1, 0, 0),
            ("Y-", 0, -1, 0),
            ("Y+", 0, 1, 0),
            ("Z-", 0, 0, -1),
            ("Z+", 0, 0, 1),
        ):
            ttk.Button(
                move_row,
                text=text,
                command=lambda ddx=dx, ddy=dy, ddz=dz: self._nudge_box(np.array([ddx, ddy, ddz], dtype=np.float32), None),
            ).pack(side="left", padx=(0, 4))

        rot_row = ttk.Frame(pose_panel)
        rot_row.pack(fill="x", padx=8, pady=(0, 8))
        for text, dr, dp, dyaw in (
            ("Roll-", -1, 0, 0),
            ("Roll+", 1, 0, 0),
            ("Pitch-", 0, -1, 0),
            ("Pitch+", 0, 1, 0),
            ("Yaw-", 0, 0, -1),
            ("Yaw+", 0, 0, 1),
        ):
            ttk.Button(
                rot_row,
                text=text,
                command=lambda ddr=dr, ddp=dp, ddyaw=dyaw: self._nudge_box(None, np.array([ddr, ddp, ddyaw], dtype=np.float32)),
            ).pack(side="left", padx=(0, 4))

        status = ttk.Label(controls, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", pady=(8, 0))

        plan_panel = ttk.LabelFrame(info_wrap, text="Pick / Place Planning")
        plan_panel.pack(fill="x", pady=(0, 8))

        self._add_button_grid(
            plan_panel,
            [
                ("Init Grasp", self._init_default_grasp),
                ("Pick Grasp Point", self._begin_pick_grasp_point),
                ("Reset Grasp", self._reset_grasp_pose),
                ("Segment Place Target", self._start_segment_place_target),
                (self.place_target_fit_overlay_label_var.get(), self._toggle_place_target_fit_overlay),
                ("Plan Place", self._start_plan_place),
                ("Clear Place", self._clear_place_target),
            ],
            columns=2,
            pady=(8, 6),
        )
        plan_buttons = plan_panel.winfo_children()[-1]
        if isinstance(plan_buttons, ttk.Frame):
            button_widgets = [child for child in plan_buttons.winfo_children() if isinstance(child, ttk.Button)]
            if len(button_widgets) >= 5:
                button_widgets[4].configure(textvariable=self.place_target_fit_overlay_label_var)

        target_kind_row = ttk.Frame(plan_panel)
        target_kind_row.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(target_kind_row, text="Target Kind", width=12).pack(side="left")
        ttk.Combobox(
            target_kind_row,
            textvariable=self.place_target_kind_var,
            values=list(placedof_planner.SUPPORTED_TARGET_KINDS),
            width=16,
            state="readonly",
        ).pack(side="left", fill="x", expand=True, padx=(6, 0))

        frame_overlay_row = ttk.Frame(plan_panel)
        frame_overlay_row.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Checkbutton(
            frame_overlay_row,
            text="Show Pick OBB Axes",
            variable=self.show_pick_obb_axes_var,
            command=self._render_viewer,
        ).pack(anchor="w")
        ttk.Checkbutton(
            frame_overlay_row,
            text="Show Grasp Axes",
            variable=self.show_grasp_axes_var,
            command=self._render_viewer,
        ).pack(anchor="w", pady=(2, 0))
        ttk.Checkbutton(
            frame_overlay_row,
            text="Show Place Target Axes",
            variable=self.show_place_target_axes_var,
            command=self._render_viewer,
        ).pack(anchor="w", pady=(2, 0))
        ttk.Checkbutton(
            frame_overlay_row,
            text="Show Place OBB Axes",
            variable=self.show_place_obb_axes_var,
            command=self._render_viewer,
        ).pack(anchor="w", pady=(2, 0))
        ttk.Checkbutton(
            frame_overlay_row,
            text="Show Place Grasp Axes",
            variable=self.show_place_grasp_axes_var,
            command=self._render_viewer,
        ).pack(anchor="w", pady=(2, 0))

        grasp_move_label = ttk.Label(plan_panel, text="Grasp Translate")
        grasp_move_label.pack(anchor="w", padx=8)
        self._add_button_grid(
            plan_panel,
            [
                ("GX-", lambda: self._nudge_grasp(np.array([-1, 0, 0], dtype=np.float32), None)),
                ("GX+", lambda: self._nudge_grasp(np.array([1, 0, 0], dtype=np.float32), None)),
                ("GY-", lambda: self._nudge_grasp(np.array([0, -1, 0], dtype=np.float32), None)),
                ("GY+", lambda: self._nudge_grasp(np.array([0, 1, 0], dtype=np.float32), None)),
                ("GZ-", lambda: self._nudge_grasp(np.array([0, 0, -1], dtype=np.float32), None)),
                ("GZ+", lambda: self._nudge_grasp(np.array([0, 0, 1], dtype=np.float32), None)),
            ],
            columns=3,
            pady=(0, 6),
        )

        grasp_rot_label = ttk.Label(plan_panel, text="Grasp Rotate")
        grasp_rot_label.pack(anchor="w", padx=8)
        self._add_button_grid(
            plan_panel,
            [
                ("GRoll-", lambda: self._nudge_grasp(None, np.array([-1, 0, 0], dtype=np.float32))),
                ("GRoll+", lambda: self._nudge_grasp(None, np.array([1, 0, 0], dtype=np.float32))),
                ("GPitch-", lambda: self._nudge_grasp(None, np.array([0, -1, 0], dtype=np.float32))),
                ("GPitch+", lambda: self._nudge_grasp(None, np.array([0, 1, 0], dtype=np.float32))),
                ("GYaw-", lambda: self._nudge_grasp(None, np.array([0, 0, -1], dtype=np.float32))),
                ("GYaw+", lambda: self._nudge_grasp(None, np.array([0, 0, 1], dtype=np.float32))),
            ],
            columns=3,
            pady=(0, 8),
        )

        info_panel = ttk.LabelFrame(info_wrap, text="Info")
        info_panel.pack(fill="both", expand=True)
        self.info_text = scrolledtext.ScrolledText(info_panel, height=24, wrap="word")
        self.info_text.pack(fill="both", expand=True)
        self.info_text.config(state="disabled")

    def _show_plan_window(self) -> None:
        if self.plan_window is None or not self.plan_window.winfo_exists():
            self._build_plan_window()
        assert self.plan_window is not None
        self.plan_window.deiconify()
        self.plan_window.lift()
        try:
            self.plan_window.focus_force()
        except Exception:
            pass

    def _hide_plan_window(self) -> None:
        if self.plan_window is None or not self.plan_window.winfo_exists():
            return
        self.plan_window.withdraw()

    def _bind_hotkeys(self) -> None:
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.after_idle(self._focus_main_view)

    def _focus_main_view(self) -> None:
        for widget_name in ("viewer_canvas", "left_canvas", "root"):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                widget.focus_set()
                return
            except Exception:
                continue

    def _parse_timeout_s(self) -> float:
        try:
            value = float(self.timeout_var.get().strip())
            if np.isfinite(value) and value > 0.0:
                return float(value)
        except Exception:
            pass
        return float(DEFAULT_TIMEOUT_S)

    def _parse_max_points(self) -> int:
        try:
            value = int(float(self.max_points_var.get().strip()))
            return max(0, int(value))
        except Exception:
            return int(DEFAULT_MAX_RENDER_POINTS)

    def _parse_move_step(self) -> float:
        try:
            value = float(self.move_step_var.get().strip())
            if np.isfinite(value) and value > 0.0:
                return float(value)
        except Exception:
            pass
        return float(DEFAULT_MOVE_STEP_M)

    def _parse_rot_step_rad(self) -> float:
        try:
            value = float(self.rot_step_var.get().strip())
            if np.isfinite(value) and value > 0.0:
                return math.radians(float(value))
        except Exception:
            pass
        return math.radians(float(DEFAULT_ROT_STEP_DEG))

    def _clear_place_plan(self) -> None:
        self.place_plan_state = None

    def _clear_place_target_state_only(self) -> None:
        self.place_target_state = None
        self._clear_place_plan()

    def _current_object_axes_base(self) -> np.ndarray | None:
        if self.object_state is None:
            return None
        return _rpy_to_matrix(self.object_state.rpy_rad).astype(np.float64, copy=False)

    def _compose_grasp_pose_from_object_pose(
        self,
        object_center_base: np.ndarray,
        object_axes_base: np.ndarray,
        grasp_origin_local: np.ndarray | None = None,
        grasp_axes_local: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.grasp_state is None and (grasp_origin_local is None or grasp_axes_local is None):
            raise RuntimeError("Missing grasp state for pose composition.")
        center = np.asarray(object_center_base, dtype=np.float64).reshape(3)
        axes = _orthonormalize_axes(object_axes_base).astype(np.float64, copy=False)
        origin_local = (
            np.asarray(self.grasp_state.point_obj, dtype=np.float64).reshape(3)
            if grasp_origin_local is None
            else np.asarray(grasp_origin_local, dtype=np.float64).reshape(3)
        )
        axes_local = (
            np.asarray(self.grasp_state.axes_obj, dtype=np.float64).reshape(3, 3)
            if grasp_axes_local is None
            else _orthonormalize_axes(grasp_axes_local).astype(np.float64, copy=False)
        )
        grasp_origin = (
            center.reshape((1, 3))
            + origin_local.reshape((1, 3)) @ axes.T
        ).reshape(3)
        grasp_axes = (axes @ axes_local).astype(np.float32, copy=False)
        return grasp_origin.astype(np.float32, copy=False), grasp_axes

    def _derive_grasp_pose_local_from_object_pose(
        self,
        object_center_base: np.ndarray,
        object_axes_base: np.ndarray,
        grasp_origin_base: np.ndarray,
        grasp_axes_base: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        center = np.asarray(object_center_base, dtype=np.float64).reshape(3)
        object_axes = _orthonormalize_axes(object_axes_base).astype(np.float64, copy=False)
        grasp_origin = np.asarray(grasp_origin_base, dtype=np.float64).reshape(3)
        grasp_axes = _orthonormalize_axes(grasp_axes_base).astype(np.float64, copy=False)
        origin_local = ((grasp_origin - center).reshape(1, 3) @ object_axes).reshape(3)
        axes_local = (object_axes.T @ grasp_axes).astype(np.float32, copy=False)
        return origin_local.astype(np.float32, copy=False), axes_local

    def _project_base_points_to_image(
        self,
        points_base: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts = _finite_points(points_base).astype(np.float64, copy=False)
        if pts.shape[0] == 0 or self.camera_bundle is None or self.left_image_full is None:
            return (
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=bool),
            )
        rot_cb = np.asarray(self.camera_bundle.rot_cb, dtype=np.float64).reshape(3, 3)
        shift_cb = np.asarray(self.camera_bundle.shift_cb, dtype=np.float64).reshape(3)
        k1 = np.asarray(self.camera_bundle.k1, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(rot_cb)) or not np.all(np.isfinite(shift_cb)) or not np.all(np.isfinite(k1)):
            return (
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=bool),
            )
        pts_cam = (pts - shift_cb.reshape((1, 3))) @ rot_cb
        depth = pts_cam[:, 2]
        fx = float(k1[0, 0])
        fy = float(k1[1, 1])
        cx = float(k1[0, 2])
        cy = float(k1[1, 2])
        if abs(fx) <= 1e-8 or abs(fy) <= 1e-8:
            return (
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=bool),
            )
        valid_depth = np.isfinite(depth) & (depth > float(place_service.DEFAULT_MIN_CAM_DEPTH))
        u = np.full((pts.shape[0],), np.nan, dtype=np.float64)
        v = np.full((pts.shape[0],), np.nan, dtype=np.float64)
        u[valid_depth] = fx * pts_cam[valid_depth, 0] / depth[valid_depth] + cx
        v[valid_depth] = fy * pts_cam[valid_depth, 1] / depth[valid_depth] + cy
        return u, v, valid_depth.astype(bool, copy=False)

    def _evaluate_points_camera_visibility(
        self,
        points_base: np.ndarray,
        pixel_margin_px: int = DEFAULT_PLACE_CAMERA_PIXEL_MARGIN_PX,
    ) -> dict[str, Any]:
        if self.camera_bundle is None or self.left_image_full is None:
            return {"available": False}
        u, v, valid_depth = self._project_base_points_to_image(points_base)
        width, height = self.left_image_full.size
        margin = max(0, int(pixel_margin_px))
        in_frame = (
            valid_depth
            & np.isfinite(u)
            & np.isfinite(v)
            & (u >= float(margin))
            & (u <= float(width - 1 - margin))
            & (v >= float(margin))
            & (v <= float(height - 1 - margin))
        )
        return {
            "available": True,
            "valid_depth_count": int(np.count_nonzero(valid_depth)),
            "in_frame_count": int(np.count_nonzero(in_frame)),
            "all_valid_depth": bool(np.all(valid_depth)) if valid_depth.size > 0 else False,
            "all_in_frame": bool(np.all(in_frame)) if in_frame.size > 0 else False,
            "u_range": [
                float(np.min(u[np.isfinite(u)])) if np.any(np.isfinite(u)) else None,
                float(np.max(u[np.isfinite(u)])) if np.any(np.isfinite(u)) else None,
            ],
            "v_range": [
                float(np.min(v[np.isfinite(v)])) if np.any(np.isfinite(v)) else None,
                float(np.max(v[np.isfinite(v)])) if np.any(np.isfinite(v)) else None,
            ],
        }

    def _evaluate_box_camera_visibility(
        self,
        center_base: np.ndarray,
        axes_base: np.ndarray,
        size_xyz: np.ndarray,
        pixel_margin_px: int = DEFAULT_PLACE_CAMERA_PIXEL_MARGIN_PX,
    ) -> dict[str, Any]:
        corners = _box_corners_axes(center_base, axes_base, size_xyz).astype(np.float64, copy=False)
        return self._evaluate_points_camera_visibility(corners, pixel_margin_px=pixel_margin_px)

    def _evaluate_grasp_camera_visibility(
        self,
        grasp_origin_base: np.ndarray,
        pixel_margin_px: int = DEFAULT_PLACE_CAMERA_PIXEL_MARGIN_PX,
    ) -> dict[str, Any]:
        point = np.asarray(grasp_origin_base, dtype=np.float32).reshape(1, 3)
        return self._evaluate_points_camera_visibility(point, pixel_margin_px=pixel_margin_px)

    def _evaluate_place_camera_visibility(
        self,
        center_base: np.ndarray,
        axes_base: np.ndarray,
        size_xyz: np.ndarray,
        grasp_origin_base: np.ndarray,
        pixel_margin_px: int = DEFAULT_PLACE_CAMERA_PIXEL_MARGIN_PX,
    ) -> dict[str, Any]:
        object_vis = self._evaluate_box_camera_visibility(
            center_base,
            axes_base,
            size_xyz,
            pixel_margin_px=pixel_margin_px,
        )
        grasp_vis = self._evaluate_grasp_camera_visibility(
            grasp_origin_base,
            pixel_margin_px=pixel_margin_px,
        )
        if not bool(object_vis.get("available", False)) or not bool(grasp_vis.get("available", False)):
            return {"available": False}
        return {
            "available": True,
            "all_in_frame": bool(object_vis.get("all_in_frame", False) and grasp_vis.get("all_in_frame", False)),
            "object": dict(object_vis),
            "grasp": dict(grasp_vis),
        }

    def _compute_place_opening_visibility_shift(
        self,
        center_base: np.ndarray,
        axes_base: np.ndarray,
        size_xyz: np.ndarray,
        profile: placedof_planner.TargetProfile,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if profile.opening_face is None or profile.opening_normal_base is None:
            return np.zeros((3,), dtype=np.float32), {"applied": False, "reason": "no_opening_face"}
        face = str(profile.opening_face).strip().lower()
        if len(face) != 2 or face[0] not in {"+", "-"} or face[1] not in {"x", "y", "z"}:
            return np.zeros((3,), dtype=np.float32), {"applied": False, "reason": f"unsupported_opening_face:{face}"}
        axis_idx = {"x": 0, "y": 1, "z": 2}[face[1]]
        sign = 1.0 if face[0] == "+" else -1.0
        corners = _box_corners_axes(center_base, axes_base, size_xyz).astype(np.float64, copy=False)
        local = profile.world_to_local(corners).astype(np.float64, copy=False)
        opening_coord = sign * local[:, axis_idx]
        opening_plane = 0.5 * float(profile.size_xyz[axis_idx])
        object_extent = float(np.max(opening_coord) - np.min(opening_coord))
        dims = np.sort(np.asarray(size_xyz, dtype=np.float64).reshape(3))
        slender_ratio = float(dims[2] / max(dims[1], 1e-4))
        exposure_ratio = (
            float(DEFAULT_PLACE_OPENING_EXPOSURE_SLENDER_RATIO)
            if slender_ratio >= float(DEFAULT_SLENDER_OBJECT_ASPECT_RATIO)
            else float(DEFAULT_PLACE_OPENING_EXPOSURE_RATIO)
        )
        required_exposure = float(
            np.clip(
                max(DEFAULT_PLACE_OPENING_EXPOSURE_MIN_M, exposure_ratio * object_extent),
                0.0,
                min(float(DEFAULT_PLACE_OPENING_SHIFT_MAX_M), max(object_extent, DEFAULT_PLACE_OPENING_EXPOSURE_MIN_M)),
            )
        )
        current_exposure = float(np.max(opening_coord) - opening_plane)
        shift_amount = float(np.clip(required_exposure - current_exposure, 0.0, DEFAULT_PLACE_OPENING_SHIFT_MAX_M))
        if not np.isfinite(shift_amount) or shift_amount <= 1e-6:
            return np.zeros((3,), dtype=np.float32), {
                "applied": False,
                "opening_face": face,
                "slender_ratio": float(slender_ratio),
                "required_exposure_m": float(required_exposure),
                "current_exposure_m": float(current_exposure),
            }
        delta = (
            np.asarray(profile.opening_normal_base, dtype=np.float64).reshape(3)
            * float(shift_amount)
        ).astype(np.float32, copy=False)
        return delta, {
            "applied": True,
            "opening_face": face,
            "slender_ratio": float(slender_ratio),
            "required_exposure_m": float(required_exposure),
            "current_exposure_m": float(current_exposure),
            "shift_amount_m": float(shift_amount),
            "delta_base": [float(v) for v in delta.tolist()],
        }

    def _post_adjust_place_plan_visibility(
        self,
        object_center_base: np.ndarray,
        object_axes_base: np.ndarray,
        object_rpy_rad: np.ndarray,
        gripper_origin_base: np.ndarray,
        rule_id: str,
        profile: placedof_planner.TargetProfile,
        collision_points_base: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self.object_state is None:
            return (
                np.asarray(object_center_base, dtype=np.float32),
                np.asarray(gripper_origin_base, dtype=np.float32),
                {"applied": False, "reason": "no_object_state"},
            )
        center = np.asarray(object_center_base, dtype=np.float32).reshape(3).copy()
        axes = np.asarray(object_axes_base, dtype=np.float32).reshape(3, 3)
        gripper = np.asarray(gripper_origin_base, dtype=np.float32).reshape(3).copy()
        size = np.asarray(self.object_state.size_xyz, dtype=np.float32).reshape(3)
        plane_model = self._current_plane_model()
        collision_points = (
            self.collision_points_base.astype(np.float32, copy=False)
            if collision_points_base is None
            else _finite_points(collision_points_base).astype(np.float32, copy=False)
        )
        debug: dict[str, Any] = {
            "rule_id": str(rule_id),
            "before": self._evaluate_place_camera_visibility(center, axes, size, gripper),
            "collision_scene_points": int(collision_points.shape[0]),
        }
        container_rule = str(rule_id) in {"top_open_box", "front_open_cabinet"}
        opening_delta = np.zeros((3,), dtype=np.float32)
        opening_debug: dict[str, Any] = {"applied": False, "reason": "non_container_rule"}
        opening_normal = None
        if container_rule:
            opening_delta, opening_debug = self._compute_place_opening_visibility_shift(center, axes, size, profile)
            if profile.opening_normal_base is not None:
                opening_normal = np.asarray(profile.opening_normal_base, dtype=np.float32).reshape(3)
        debug["opening_adjustment"] = dict(opening_debug)

        def _trial_collision(trial_center: np.ndarray) -> bool:
            if self._has_box_point_collision_axes_in_points(trial_center, axes, size, collision_points):
                return True
            return bool(self._has_box_plane_collision(trial_center, object_rpy_rad, size, plane_model))

        opening_step = float(DEFAULT_PLACE_VISIBILITY_OPENING_STEP_M)
        opening_extra_max = float(DEFAULT_PLACE_VISIBILITY_EXTRA_OPENING_M if container_rule else 0.0)
        lift_step = float(DEFAULT_PLACE_VISIBILITY_LIFT_STEP_M if container_rule else 0.0)
        lift_max = float(DEFAULT_PLACE_VISIBILITY_MAX_LIFT_M if container_rule else 0.0)
        opening_steps = max(0, int(math.ceil(opening_extra_max / max(opening_step, 1e-6))))
        lift_steps = max(0, int(math.ceil(lift_max / max(lift_step, 1e-6))))

        best_center = center.copy()
        best_gripper = gripper.copy()
        best_visibility = debug["before"]
        best_key = (
            int(bool(best_visibility.get("all_in_frame", False))),
            int(best_visibility.get("object", {}).get("in_frame_count", 0)) + int(best_visibility.get("grasp", {}).get("in_frame_count", 0)),
            int(best_visibility.get("object", {}).get("valid_depth_count", 0)) + int(best_visibility.get("grasp", {}).get("valid_depth_count", 0)),
            0.0,
            0.0,
        )
        best_open_extra = 0.0
        best_lift = 0.0

        for open_idx in range(opening_steps + 1):
            extra_open = float(open_idx * opening_step)
            extra_open_delta = np.zeros((3,), dtype=np.float32)
            if opening_normal is not None and extra_open > 1e-6:
                extra_open_delta = (opening_normal.astype(np.float64) * extra_open).astype(np.float32, copy=False)
            for lift_idx in range(lift_steps + 1):
                lift = float(lift_idx * lift_step)
                lift_delta = np.array([0.0, 0.0, lift], dtype=np.float32)
                total_delta = (
                    opening_delta.astype(np.float64)
                    + extra_open_delta.astype(np.float64)
                    + lift_delta.astype(np.float64)
                ).astype(np.float32, copy=False)
                trial_center = (center.astype(np.float64) + total_delta.astype(np.float64)).astype(np.float32, copy=False)
                if float(np.linalg.norm(total_delta.astype(np.float64))) > 1e-6 and _trial_collision(trial_center):
                    continue
                trial_gripper = (gripper.astype(np.float64) + total_delta.astype(np.float64)).astype(np.float32, copy=False)
                trial_visibility = self._evaluate_place_camera_visibility(trial_center, axes, size, trial_gripper)
                trial_key = (
                    int(bool(trial_visibility.get("all_in_frame", False))),
                    int(trial_visibility.get("object", {}).get("in_frame_count", 0))
                    + int(trial_visibility.get("grasp", {}).get("in_frame_count", 0)),
                    int(trial_visibility.get("object", {}).get("valid_depth_count", 0))
                    + int(trial_visibility.get("grasp", {}).get("valid_depth_count", 0)),
                    -float(np.linalg.norm(total_delta.astype(np.float64))),
                    -float(lift),
                )
                if trial_key > best_key:
                    best_key = trial_key
                    best_center = trial_center.copy()
                    best_gripper = trial_gripper.copy()
                    best_visibility = dict(trial_visibility)
                    best_open_extra = float(extra_open)
                    best_lift = float(lift)
                if bool(trial_visibility.get("all_in_frame", False)):
                    break
            if bool(best_visibility.get("all_in_frame", False)):
                break

        total_delta = (best_center.astype(np.float64) - center.astype(np.float64)).astype(np.float32, copy=False)
        debug["search"] = {
            "opening_steps": int(opening_steps + 1),
            "lift_steps": int(lift_steps + 1),
        }
        debug["after_opening"] = self._evaluate_box_camera_visibility(
            (center.astype(np.float64) + opening_delta.astype(np.float64)).astype(np.float32, copy=False),
            axes,
            size,
        )
        debug["lift_adjustment"] = {
            "applied": bool(best_lift > 1e-6),
            "delta_base": [0.0, 0.0, float(best_lift)],
            "after_lift": dict(best_visibility),
        }
        if opening_normal is not None:
            total_opening_delta = (
                opening_delta.astype(np.float64)
                + opening_normal.astype(np.float64) * float(best_open_extra)
            ).astype(np.float32, copy=False)
            debug["opening_adjustment"]["selected_total_delta_base"] = [float(v) for v in total_opening_delta.tolist()]
            debug["opening_adjustment"]["selected_extra_opening_m"] = float(best_open_extra)
        final_visibility = self._evaluate_place_camera_visibility(best_center, axes, size, best_gripper)
        debug["final"] = final_visibility
        debug["applied"] = bool(float(np.linalg.norm(total_delta.astype(np.float64))) > 1e-6)
        debug["delta_base"] = [float(v) for v in total_delta.tolist()]
        if debug["applied"]:
            collision = bool(self._has_box_point_collision_axes_in_points(best_center, axes, size, collision_points))
            debug["post_adjust_collision"] = collision
        return best_center.astype(np.float32, copy=False), best_gripper.astype(np.float32, copy=False), debug

    def _make_default_grasp_state(self, obj: ObjectState) -> GraspState:
        obj_rot = _rpy_to_matrix(np.asarray(obj.rpy_rad, dtype=np.float32))
        object_box = placedof_planner.OrientedBox(
            center_base=np.asarray(obj.center_base, dtype=np.float32),
            rotation_base=np.asarray(obj_rot, dtype=np.float32),
            size_xyz=np.asarray(obj.size_xyz, dtype=np.float32),
        )
        pose_world = placedof_planner.compute_default_pick_pose(object_box)
        pose_origin_local = (
            (np.asarray(pose_world.origin_base, dtype=np.float64).reshape((1, 3))
             - np.asarray(obj.center_base, dtype=np.float64).reshape((1, 3)))
            @ np.asarray(obj_rot, dtype=np.float64)
        ).reshape(3)
        pose_axes_local = (
            np.asarray(obj_rot, dtype=np.float64).T
            @ np.asarray(pose_world.rotation_base, dtype=np.float64)
        )
        return GraspState(
            point_obj=np.asarray(pose_origin_local, dtype=np.float32),
            axes_obj=np.asarray(pose_axes_local, dtype=np.float32),
            init_point_obj=np.asarray(pose_origin_local, dtype=np.float32),
            init_axes_obj=np.asarray(pose_axes_local, dtype=np.float32),
            debug={"source": "default_vertical_long_axis_grasp"},
        )

    def _init_default_grasp(self) -> None:
        if self.object_state is None:
            self.status_var.set("Segment object first.")
            return
        self.grasp_state = self._make_default_grasp_state(self.object_state)
        self.awaiting_grasp_click = False
        self._clear_place_plan()
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set("Initialized default grasp pose: +Z down, +X along the OBB long edge.")
        self._focus_main_view()

    def _begin_pick_grasp_point(self) -> None:
        if self.object_state is None:
            self.status_var.set("Segment object before picking a grasp point.")
            return
        if self.grasp_state is None:
            self.grasp_state = self._make_default_grasp_state(self.object_state)
        self.awaiting_grasp_click = True
        self.status_var.set("Click a point on the left image to set the grasp point.")
        self._focus_main_view()

    def _set_grasp_point_from_pixel(self, x_img: int, y_img: int) -> bool:
        if self.object_state is None or self.grasp_state is None or self.raw_pixels_yx.shape[0] == 0:
            return False
        keep = self.object_state.mask[self.raw_pixels_yx[:, 0], self.raw_pixels_yx[:, 1]]
        if not np.any(keep):
            return False
        obj_pixels = self.raw_pixels_yx[keep].astype(np.int32, copy=False)
        obj_points_base = self.raw_points_base[keep].astype(np.float32, copy=False)
        finite_keep = _finite_point_mask(obj_points_base)
        obj_pixels = obj_pixels[finite_keep]
        obj_points_base = obj_points_base[finite_keep].astype(np.float32, copy=False)
        if obj_points_base.shape[0] == 0:
            return False
        diff = obj_pixels.astype(np.float64) - np.array([float(y_img), float(x_img)], dtype=np.float64).reshape((1, 2))
        dist_sq = np.sum(diff * diff, axis=1)
        pick_idx = int(np.argmin(dist_sq))
        world_point = obj_points_base[pick_idx : pick_idx + 1]
        local = _obb_local_points(world_point, self.object_state.center_base, self.object_state.rpy_rad)
        if local.shape[0] == 0:
            return False
        local_point = np.asarray(local[0], dtype=np.float32)
        half = 0.5 * np.asarray(self.object_state.size_xyz, dtype=np.float32).reshape(3)
        local_point[0] = float(np.clip(local_point[0], -half[0], half[0]))
        local_point[1] = float(np.clip(local_point[1], -half[1], half[1]))
        local_point[2] = float(half[2])
        self.grasp_state.point_obj = local_point.astype(np.float32, copy=False)
        self.grasp_state.debug["source"] = "image_click"
        self.grasp_state.debug["pixel"] = [int(x_img), int(y_img)]
        self.awaiting_grasp_click = False
        self._clear_place_plan()
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set(f"Grasp point set from image click at ({int(x_img)}, {int(y_img)}).")
        self._focus_main_view()
        return True

    def _reset_grasp_pose(self) -> None:
        if self.grasp_state is None:
            return
        self.grasp_state.point_obj = self.grasp_state.init_point_obj.astype(np.float32, copy=True)
        self.grasp_state.axes_obj = self.grasp_state.init_axes_obj.astype(np.float32, copy=True)
        self.awaiting_grasp_click = False
        self._clear_place_plan()
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set("Grasp pose reset.")
        self._focus_main_view()

    def _nudge_grasp(self, delta_xyz_dir: np.ndarray | None, delta_rpy_dir: np.ndarray | None) -> None:
        if self.object_state is None:
            self.status_var.set("No object selected.")
            return
        if self.grasp_state is None:
            self.grasp_state = self._make_default_grasp_state(self.object_state)
        if delta_xyz_dir is not None:
            self.grasp_state.point_obj = (
                self.grasp_state.point_obj
                + np.asarray(delta_xyz_dir, dtype=np.float32).reshape(3) * float(self._parse_move_step())
            ).astype(np.float32, copy=False)
        if delta_rpy_dir is not None:
            delta = np.asarray(delta_rpy_dir, dtype=np.float64).reshape(3) * float(self._parse_rot_step_rad())
            delta_rot = _rpy_to_matrix(delta.astype(np.float32))
            self.grasp_state.axes_obj = _orthonormalize_axes(
                np.asarray(self.grasp_state.axes_obj, dtype=np.float64) @ delta_rot
            )
        half = 0.5 * np.asarray(self.object_state.size_xyz, dtype=np.float32).reshape(3)
        self.grasp_state.point_obj = np.clip(
            self.grasp_state.point_obj,
            -half - 0.06,
            half + 0.06,
        ).astype(np.float32, copy=False)
        self._clear_place_plan()
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set("Adjusted grasp pose in object coordinates.")
        self._focus_main_view()

    def _extract_target_points_base(
        self,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self.raw_points_base.shape[0] == 0 or self.left_image_full is None:
            raise RuntimeError("Raw scene point cloud is unavailable.")
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape != (self.left_image_full.height, self.left_image_full.width):
            raise RuntimeError("Target mask shape mismatch.")
        raw_keep = mask_bool[self.raw_pixels_yx[:, 0], self.raw_pixels_yx[:, 1]]
        used_mode = "mask"
        if int(np.count_nonzero(raw_keep)) < 24:
            raw_keep = self._bbox_pixel_keep_mask(self.raw_pixels_yx, bbox)
            used_mode = "bbox_fallback"
        raw_points = self.raw_points_base[raw_keep].astype(np.float32, copy=False)
        raw_points = _finite_points(raw_points)
        extra_points = _empty_points()
        if self.table_cache is not None:
            flat_like = False
            if raw_points.shape[0] >= 16:
                z_span = float(np.ptp(raw_points[:, 2].astype(np.float64, copy=False)))
                flat_like = z_span <= 0.050
            if used_mode != "mask" or flat_like:
                table_keep = mask_bool[self.table_cache.plane_pixels_yx[:, 0], self.table_cache.plane_pixels_yx[:, 1]]
                if int(np.count_nonzero(table_keep)) < 24:
                    table_keep = self._bbox_pixel_keep_mask(self.table_cache.plane_pixels_yx, bbox)
                extra_points = _finite_points(self.table_cache.plane_points_base[table_keep])
        merged = raw_points
        if extra_points.shape[0] > 0:
            merged = np.vstack([merged, extra_points]).astype(np.float32, copy=False) if merged.shape[0] > 0 else extra_points.copy()
        merged = _finite_points(merged)
        if merged.shape[0] < 24:
            raise RuntimeError("Too few target points inside the selected place target.")
        preferred_center = np.median(merged.astype(np.float64, copy=False), axis=0).astype(np.float32)
        debug = {
            "point_extract_mode": str(used_mode),
            "raw_points": int(raw_points.shape[0]),
            "extra_table_points": int(extra_points.shape[0]),
            "merged_points": int(merged.shape[0]),
        }
        return merged.astype(np.float32, copy=False), preferred_center, debug

    def _fit_place_target_profile(
        self,
        points_base: np.ndarray,
        kind_hint: str,
    ) -> tuple[np.ndarray, placedof_planner.TargetProfile, dict[str, Any]]:
        pts = _finite_points(points_base)
        if pts.shape[0] < 24:
            raise RuntimeError("Too few target points for primitive fitting.")

        q_lo_xy, q_hi_xy = np.quantile(pts[:, :2].astype(np.float64, copy=False), [0.10, 0.90], axis=0)
        robust_xy_spread = np.maximum(q_hi_xy - q_lo_xy, 1e-4)
        fit_component_voxel_size = float(np.clip(0.07 * float(np.min(robust_xy_spread)), 0.0030, 0.0100))
        fit_points_filtered, fit_component_debug = push_planner._keep_largest_xy_component(
            pts,
            voxel_size=fit_component_voxel_size,
        )
        min_fit_keep_count = max(24, int(round(0.35 * float(pts.shape[0]))))
        fit_cluster_used = bool(fit_points_filtered.shape[0] >= min_fit_keep_count)
        fit_points = (
            fit_points_filtered.astype(np.float32, copy=False)
            if fit_cluster_used
            else pts.astype(np.float32, copy=False)
        )
        camera_origin_base = None
        if self.camera_bundle is not None:
            camera_origin = np.asarray(self.camera_bundle.shift_cb, dtype=np.float32).reshape(-1)
            if camera_origin.shape[0] >= 3 and np.all(np.isfinite(camera_origin[:3])):
                camera_origin_base = camera_origin[:3].astype(np.float32, copy=False)
        profile = placedof_planner.fit_target_profile(
            fit_points,
            kind_hint=kind_hint,
            camera_origin_base=camera_origin_base,
        )
        debug = {
            "fit_cluster_used": bool(fit_cluster_used),
            "fit_cluster_component_count": int(fit_component_debug.get("component_count", 0)),
            "fit_cluster_largest_component_point_count": int(
                fit_component_debug.get("largest_component_point_count", 0)
            ),
            "fit_cluster_voxel_size_m": float(
                fit_component_debug.get("voxel_size_m", fit_component_voxel_size)
            ),
            "fit_points": int(fit_points.shape[0]),
            "fit_primitive_shape": str(profile.primitive_shape),
            "fit_primitive_radius": None if profile.primitive_radius is None else float(profile.primitive_radius),
            "fit_target_kind_auto": str(profile.kind),
        }
        return fit_points.astype(np.float32, copy=False), profile, debug

    def _build_place_collision_points(self) -> np.ndarray:
        if self.raw_points_base.shape[0] == 0:
            return _empty_points()
        keep = np.ones((self.raw_points_base.shape[0],), dtype=bool)
        if self.object_state is not None:
            keep &= ~self.object_state.mask[self.raw_pixels_yx[:, 0], self.raw_pixels_yx[:, 1]]
        if self.place_target_state is not None:
            keep &= ~self.place_target_state.mask[self.raw_pixels_yx[:, 0], self.raw_pixels_yx[:, 1]]
        scene_points_full = _finite_points(self.raw_points_base[keep])
        if self.enable_table_completion_var.get() and self.table_cache is not None:
            table_keep = np.ones((self.table_cache.plane_points_base.shape[0],), dtype=bool)
            if self.place_target_state is not None:
                table_keep &= ~self.place_target_state.mask[
                    self.table_cache.plane_pixels_yx[:, 0],
                    self.table_cache.plane_pixels_yx[:, 1],
                ]
            table_points = _finite_points(self.table_cache.plane_points_base[table_keep])
            if table_points.shape[0] > 0:
                if scene_points_full.shape[0] == 0:
                    scene_points_full = table_points.copy()
                else:
                    scene_points_full = np.vstack([scene_points_full, table_points]).astype(np.float32, copy=False)

        plane_model = self._current_plane_model()
        collision_points = scene_points_full
        if (
            plane_model is not None
            and collision_points.shape[0] > 0
            and _is_finite_vector(plane_model.normal_base, 3)
            and np.isfinite(float(plane_model.d_base))
        ):
            plane_dist = np.abs(
                collision_points.astype(np.float64, copy=False) @ plane_model.normal_base.astype(np.float64, copy=False)
                + float(plane_model.d_base)
            )
            collision_points = collision_points[plane_dist > float(DEFAULT_TABLE_PLANE_EXCLUDE_M)]
        if collision_points.shape[0] > DEFAULT_MAX_COLLISION_POINTS:
            collision_points = _subsample_points(collision_points, DEFAULT_MAX_COLLISION_POINTS, seed=17)
        return collision_points.astype(np.float32, copy=False)

    def _start_segment_place_target(self) -> None:
        if self.raw_dense_map_cam is None or self.left_image_full is None:
            messagebox.showerror("Missing scene", "Run S2M2 before segmenting a place target.")
            return
        if self.selected_bbox is None:
            messagebox.showerror("Missing bbox", "Draw a bbox on the left image first.")
            return
        bbox = tuple(int(v) for v in self.selected_bbox)

        def _worker() -> PlaceTargetState:
            assert self.left_image_full is not None
            sam3_res = _call_sam3_generic(
                self.left_image_full,
                sam3_url=FIXED_SAM3_URL,
                timeout_s=self._parse_timeout_s(),
                box_prompts=[[int(v) for v in bbox]],
                box_labels=[True],
            )
            use_mask = sam3_res.get("ok", False) and int(sam3_res.get("mask_area_px", 0)) >= 200
            if use_mask:
                target_mask = np.asarray(sam3_res["mask"], dtype=bool)
            else:
                target_mask = _bbox_to_mask(bbox, self.left_image_full.width, self.left_image_full.height)
            points_base, _preferred_center_raw, extract_debug = self._extract_target_points_base(target_mask, bbox)
            fit_points_base, target_profile, fit_debug = self._fit_place_target_profile(
                points_base,
                kind_hint=placedof_planner.DEFAULT_TARGET_KIND,
            )
            debug = {
                "sam3_used": bool(use_mask),
                "sam3_latency_ms": int(sam3_res.get("latency_ms", 0)) if isinstance(sam3_res, dict) else 0,
                "sam3_error": str(sam3_res.get("error", "")) if isinstance(sam3_res, dict) else "",
            }
            debug.update(extract_debug)
            debug.update(fit_debug)
            return PlaceTargetState(
                bbox=bbox,
                mask=np.asarray(target_mask, dtype=bool),
                mask_area_px=int(np.count_nonzero(target_mask)),
                points_base=points_base.astype(np.float32, copy=False),
                fit_points_base=fit_points_base.astype(np.float32, copy=False),
                target_profile=target_profile,
                preferred_center_base=np.asarray(target_profile.center_base, dtype=np.float32),
                debug=debug,
            )

        self._run_background("Segmenting place target", _worker, self._on_place_target_ready)
        self._focus_main_view()

    def _on_place_target_ready(self, target: PlaceTargetState) -> None:
        self.place_target_state = target
        self._clear_place_plan()
        self._render_left_image()
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set(
            f"Place target ready. primitive={target.target_profile.primitive_shape}, "
            f"mask_px={target.mask_area_px}, fit_points={target.fit_points_base.shape[0]}"
        )
        self._focus_main_view()

    def _clear_place_target(self) -> None:
        self.place_target_state = None
        self._clear_place_plan()
        self._render_left_image()
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set("Place target and placement plan cleared.")
        self._focus_main_view()

    def _toggle_place_target_fit_overlay(self) -> None:
        show = not bool(self.show_place_target_fit_var.get())
        self.show_place_target_fit_var.set(show)
        self.place_target_fit_overlay_label_var.set("Hide Fitted Target" if show else "Show Fitted Target")
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set(
            "Fitted place target overlay shown in viewer."
            if show
            else "Fitted place target overlay hidden in viewer."
        )
        self._focus_main_view()

    def _start_plan_place(self) -> None:
        if self.object_state is None:
            messagebox.showerror("Missing object", "Segment the object first.")
            return
        if self.grasp_state is None:
            messagebox.showerror("Missing grasp", "Initialize or pick a grasp pose first.")
            return
        if self.place_target_state is None:
            messagebox.showerror("Missing place target", "Segment a place target first.")
            return

        def _worker() -> PlacePlanState:
            assert self.object_state is not None
            assert self.grasp_state is not None
            assert self.place_target_state is not None
            current_object_axes = _orthonormalize_axes(_rpy_to_matrix(self.object_state.rpy_rad))
            current_grasp_origin_base, current_grasp_axes_base = self._compose_grasp_pose_from_object_pose(
                np.asarray(self.object_state.center_base, dtype=np.float32),
                current_object_axes,
            )
            camera_origin_base = None
            camera_visibility_constraint = None
            if self.camera_bundle is not None:
                camera_origin = np.asarray(self.camera_bundle.shift_cb, dtype=np.float32).reshape(-1)
                if camera_origin.shape[0] >= 3 and np.all(np.isfinite(camera_origin[:3])):
                    camera_origin_base = camera_origin[:3].astype(np.float32, copy=False)
                if self.left_image_full is not None:
                    camera_visibility_constraint = placedof_planner.CameraVisibilityConstraint(
                        k1=np.asarray(self.camera_bundle.k1, dtype=np.float32),
                        rot_cb=np.asarray(self.camera_bundle.rot_cb, dtype=np.float32),
                        shift_cb=np.asarray(self.camera_bundle.shift_cb, dtype=np.float32),
                        image_size_wh=(int(self.left_image_full.size[0]), int(self.left_image_full.size[1])),
                        pixel_margin_px=int(DEFAULT_PLACE_CAMERA_PIXEL_MARGIN_PX),
                        require_full_object_in_frame=True,
                        require_grasp_origin_in_frame=True,
                    )
            target_profile = placedof_planner.fit_target_profile(
                self.place_target_state.fit_points_base.astype(np.float32, copy=False),
                kind_hint=self.place_target_kind_var.get().strip(),
                camera_origin_base=camera_origin_base,
            )
            table_points = (
                self.table_cache.plane_points_base.astype(np.float32, copy=False)
                if self.table_cache is not None
                else self.raw_points_base.astype(np.float32, copy=False)
            )
            scene_points = self._build_place_collision_points()
            target_collision_points = _sample_target_collision_points(target_profile)
            trajectory_collision_points = (
                scene_points
                if target_collision_points.shape[0] == 0
                else np.vstack([scene_points, target_collision_points]).astype(np.float32, copy=False)
            )
            if trajectory_collision_points.shape[0] > DEFAULT_MAX_COLLISION_POINTS:
                trajectory_collision_points = _subsample_points(
                    trajectory_collision_points,
                    DEFAULT_MAX_COLLISION_POINTS,
                    seed=73,
                ).astype(np.float32, copy=False)
            req = placedof_planner.PlannerRequest(
                obb=placedof_planner.OBBSpec(size_xyz=np.asarray(self.object_state.size_xyz, dtype=np.float32)),
                grasp_pose_local=placedof_planner.RelativePose(
                    origin_local=np.asarray(self.grasp_state.point_obj, dtype=np.float32),
                    axes_local=np.asarray(self.grasp_state.axes_obj, dtype=np.float32),
                ),
                table_points_base=table_points,
                target_points_base=self.place_target_state.fit_points_base.astype(np.float32, copy=False),
                scene_points_base=scene_points,
                preferred_center_base=np.asarray(target_profile.center_base, dtype=np.float32),
                target_kind_hint=self.place_target_kind_var.get().strip(),
                target_profile_override=target_profile,
                start_object_center_base=np.asarray(self.object_state.center_base, dtype=np.float32),
                start_gripper_axes_base=np.asarray(current_grasp_axes_base, dtype=np.float32),
                camera_visibility_constraint=camera_visibility_constraint,
                allow_best_effort=False,
                debug_hints={
                    "target_bbox": list(self.place_target_state.bbox),
                    "target_mask_area_px": int(self.place_target_state.mask_area_px),
                    "target_kind_hint": self.place_target_kind_var.get().strip(),
                    "target_primitive_shape": str(target_profile.primitive_shape),
                },
            )
            plan = placedof_planner.plan_placement(req)
            if not plan.success or plan.best is None:
                invalid_reason_counts = dict(plan.debug.get("invalid_reason_counts", {}))
                top_reasons = ", ".join(
                    f"{reason} x{count}"
                    for reason, count in list(invalid_reason_counts.items())[:4]
                )
                if not top_reasons:
                    top_reasons = "no valid candidate after primitive fitting / orientation search"
                raise RuntimeError(f"Planner failed to find a feasible placement candidate: {top_reasons}")
            best = plan.best
            best_effort_selected = bool(plan.debug.get("best_effort_selected", False))
            adjusted_object_center, adjusted_gripper_origin, visibility_adjust = self._post_adjust_place_plan_visibility(
                np.asarray(best.object_center_base, dtype=np.float32),
                np.asarray(best.object_axes_base, dtype=np.float32),
                np.asarray(best.object_rpy_rad, dtype=np.float32),
                np.asarray(best.gripper_origin_base, dtype=np.float32),
                str(best.rule_id),
                target_profile,
                collision_points_base=trajectory_collision_points,
            )
            final_visibility = visibility_adjust.get("final", {})
            if (
                isinstance(final_visibility, dict)
                and bool(final_visibility.get("available", False))
                and not bool(final_visibility.get("all_in_frame", False))
                and not best_effort_selected
            ):
                raise RuntimeError(
                    "Planner found a candidate, but the final place grasp / OBB could not be kept fully visible in the camera view."
                )
            adjusted_gripper_origin = np.asarray(adjusted_gripper_origin, dtype=np.float32).reshape(3)
            adjusted_gripper_axes = _orthonormalize_axes(np.asarray(best.gripper_axes_base, dtype=np.float32))
            trajectory = self._plan_place_grasp_trajectory(
                start_center_base=np.asarray(self.object_state.center_base, dtype=np.float32),
                start_axes_base=current_object_axes,
                start_gripper_origin_base=np.asarray(current_grasp_origin_base, dtype=np.float32),
                start_gripper_axes_base=np.asarray(current_grasp_axes_base, dtype=np.float32),
                goal_center_base=np.asarray(adjusted_object_center, dtype=np.float32),
                goal_axes_base=np.asarray(best.object_axes_base, dtype=np.float32),
                goal_gripper_origin_base=np.asarray(adjusted_gripper_origin, dtype=np.float32),
                goal_gripper_axes_base=np.asarray(adjusted_gripper_axes, dtype=np.float32),
                size_xyz=np.asarray(self.object_state.size_xyz, dtype=np.float32),
                collision_points_base=trajectory_collision_points,
                target_profile=target_profile,
                placement_rule_id=str(best.rule_id),
            )
            return PlacePlanState(
                object_center_base=np.asarray(adjusted_object_center, dtype=np.float32),
                object_axes_base=np.asarray(best.object_axes_base, dtype=np.float32),
                object_rpy_rad=np.asarray(best.object_rpy_rad, dtype=np.float32),
                gripper_origin_base=np.asarray(adjusted_gripper_origin, dtype=np.float32),
                gripper_axes_base=np.asarray(adjusted_gripper_axes, dtype=np.float32),
                object_center_trajectory_base=np.asarray(trajectory["object_center_trajectory_base"], dtype=np.float32),
                object_axes_trajectory_base=np.asarray(trajectory["object_axes_trajectory_base"], dtype=np.float32),
                gripper_origin_trajectory_base=np.asarray(trajectory["gripper_origin_trajectory_base"], dtype=np.float32),
                gripper_axes_trajectory_base=np.asarray(trajectory["gripper_axes_trajectory_base"], dtype=np.float32),
                rule_id=str(best.rule_id),
                target_kind=str(best.target_kind),
                score=float(best.score),
                matched_rules=[str(v) for v in plan.matched_rules],
                score_breakdown={str(key): float(val) for key, val in best.score_breakdown.items()},
                debug={
                    "planner": dict(plan.debug),
                    "candidate": dict(best.debug),
                    "target_analysis": {
                        "kind": str(plan.target_analysis.kind),
                        "face_coverage": dict(plan.target_analysis.face_coverage),
                        **dict(plan.target_analysis.debug),
                    },
                    "visibility_adjustment": dict(visibility_adjust),
                    "trajectory": dict(trajectory.get("debug", {})),
                    "trajectory_collision_scene_points": int(trajectory_collision_points.shape[0]),
                    "trajectory_target_collision_points": int(target_collision_points.shape[0]),
                    "start_object_center_base": [float(v) for v in np.asarray(self.object_state.center_base, dtype=np.float32).tolist()],
                    "start_grasp_origin_base": [float(v) for v in np.asarray(current_grasp_origin_base, dtype=np.float32).tolist()],
                },
            )

        self._run_background("Planning place pose", _worker, self._on_place_plan_ready)
        self._focus_main_view()

    def _on_place_plan_ready(self, plan_state: PlacePlanState) -> None:
        self.place_plan_state = plan_state
        self._render_viewer()
        self._update_info_panel()
        visibility_adjustment = plan_state.debug.get("visibility_adjustment", {})
        trajectory_debug = plan_state.debug.get("trajectory", {})
        adjusted = bool(visibility_adjustment.get("applied", False)) if isinstance(visibility_adjustment, dict) else False
        suffix = " visibility-adjusted" if adjusted else ""
        if isinstance(trajectory_debug, dict) and int(trajectory_debug.get("path_points", 0)) > 0:
            suffix += f" traj={int(trajectory_debug.get('path_points', 0))}pts"
        self.status_var.set(
            f"Place plan ready. rule={plan_state.rule_id}, target={plan_state.target_kind}, score={plan_state.score:.3f}{suffix}"
        )
        self._focus_main_view()

    def _load_inputs_from_paths(self, show_success: bool = True) -> None:
        try:
            left_path = Path(self.left_image_path_var.get()).expanduser()
            right_path = Path(self.right_image_path_var.get()).expanduser()
            topic_path = Path(self.topic_inputs_path_var.get()).expanduser()
            if not left_path.exists():
                raise FileNotFoundError(f"Left image not found: {left_path}")
            if not right_path.exists():
                raise FileNotFoundError(f"Right image not found: {right_path}")
            if not topic_path.exists():
                raise FileNotFoundError(f"Topic inputs not found: {topic_path}")

            left = Image.open(left_path).convert("RGB")
            right = Image.open(right_path).convert("RGB")
            topic_text = topic_path.read_text(encoding="utf-8")
            k1, baseline, rot_cb, shift_cb = place_service._parse_topic_inputs(topic_text)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            self.status_var.set("Failed to load inputs.")
            return

        self.left_image_full = left
        self.right_image_full = right
        self.left_image_np = np.asarray(left.convert("RGB"), dtype=np.uint8)
        self.camera_bundle = CameraBundle(
            k1=np.asarray(k1, dtype=np.float32),
            baseline=float(baseline),
            rot_cb=np.asarray(rot_cb, dtype=np.float32),
            shift_cb=np.asarray(shift_cb, dtype=np.float32),
        )

        self.raw_dense_map_cam = None
        self.raw_valid_mask = None
        self.raw_points_base = _empty_points()
        self.raw_colors = _empty_colors()
        self.raw_pixels_yx = _empty_pixels()
        self.table_cache = None
        self.object_state = None
        self.grasp_state = None
        self.place_target_state = None
        self.place_plan_state = None
        self.selected_bbox = None
        self.segment_scene_bbox = None
        self.preview_bbox = None
        self.awaiting_grasp_click = False
        self.viewer_state = ViewState()
        self.scene_points_base = _empty_points()
        self.scene_colors = _empty_colors()
        self.collision_points_base = _empty_points()

        self._render_left_image()
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set("Inputs loaded. Run S2M2 next." if show_success else "Ready.")
        self._focus_main_view()

    def _run_background(
        self,
        label: str,
        worker: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        if self.busy:
            self.status_var.set("Another task is still running.")
            return
        self.busy = True
        self.status_var.set(f"{label}...")

        def _target() -> None:
            try:
                result = worker()
            except Exception as exc:
                tb = traceback.format_exc()
                self.root.after(
                    0,
                    lambda label=label, exc=exc, tb=tb: self._finish_background_error(label, exc, tb),
                )
                return
            self.root.after(
                0,
                lambda result=result, on_success=on_success: self._finish_background_success(result, on_success),
            )

        threading.Thread(target=_target, daemon=True).start()

    def _finish_background_success(self, result: Any, on_success: Callable[[Any], None]) -> None:
        self.busy = False
        on_success(result)

    def _finish_background_error(self, label: str, exc: Exception, tb: str) -> None:
        self.busy = False
        self.status_var.set(f"{label} failed: {exc}")
        messagebox.showerror("Task failed", f"{exc}\n\n{tb}")

    def _start_s2m2(self) -> None:
        if self.left_image_full is None or self.right_image_full is None or self.camera_bundle is None:
            messagebox.showerror("Missing inputs", "Load images and topic inputs first.")
            return

        def _worker() -> dict[str, Any]:
            assert self.left_image_full is not None
            assert self.right_image_full is not None
            assert self.camera_bundle is not None
            arr, latency_ms = place_service._call_s2m2_api(
                self.left_image_full,
                self.right_image_full,
                self.camera_bundle.k1,
                self.camera_bundle.baseline,
                FIXED_S2M2_URL,
                self._parse_timeout_s(),
            )
            dense_map, _, _ = place_service._build_dense_from_cloud_or_depth(
                arr,
                self.left_image_full.size,
                self.camera_bundle.k1,
            )
            return {"dense_map_cam": dense_map.astype(np.float32), "latency_ms": int(latency_ms)}

        self._run_background("Running S2M2", _worker, self._on_s2m2_ready)
        self._focus_main_view()

    def _on_s2m2_ready(self, result: dict[str, Any]) -> None:
        dense_map = np.asarray(result["dense_map_cam"], dtype=np.float32)
        if self.camera_bundle is None or self.left_image_np is None:
            return

        valid = place_service._valid_cam_xyz_mask_map(dense_map)
        ys, xs = np.nonzero(valid)
        pts_cam = dense_map[ys, xs].astype(np.float32, copy=False)
        pts_base = push_planner._transform_points_cam_to_base(
            pts_cam,
            self.camera_bundle.rot_cb,
            self.camera_bundle.shift_cb,
        ).astype(np.float32, copy=False)
        if pts_base.shape[0] != pts_cam.shape[0]:
            raise RuntimeError("Camera-to-base transform changed point count unexpectedly.")
        finite_keep = np.all(np.isfinite(pts_base), axis=1) & (np.max(np.abs(pts_base), axis=1) <= float(DEFAULT_MAX_COORD_ABS))
        if not np.any(finite_keep):
            raise RuntimeError("Camera-to-base transform produced no finite base-frame points.")
        if int(np.count_nonzero(finite_keep)) != int(pts_base.shape[0]):
            pts_cam = pts_cam[finite_keep]
            ys = ys[finite_keep]
            xs = xs[finite_keep]
            pts_base = pts_base[finite_keep].astype(np.float32, copy=False)
        colors = (self.left_image_np[ys, xs].astype(np.float32) / 255.0).astype(np.float32, copy=False)

        self.raw_dense_map_cam = dense_map
        self.raw_valid_mask = valid
        self.raw_points_base = pts_base
        self.raw_colors = colors
        self.raw_pixels_yx = np.stack([ys, xs], axis=1).astype(np.int32, copy=False)

        self.table_cache = None
        self.object_state = None
        self.grasp_state = None
        self.place_target_state = None
        self.place_plan_state = None
        self.awaiting_grasp_click = False
        self.scene_points_base = _empty_points()
        self.scene_colors = _empty_colors()
        self.collision_points_base = _empty_points()

        if self.enable_table_completion_var.get():
            self._start_table_completion()
            return

        self._refresh_scene_cache()
        self._render_left_image()
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set(
            f"S2M2 ready. valid_points={self.raw_points_base.shape[0]}, latency_ms={int(result.get('latency_ms', 0))}"
        )

    def _on_toggle_table_completion(self) -> None:
        if self.enable_table_completion_var.get():
            if self.raw_dense_map_cam is None:
                self.status_var.set("Run S2M2 before enabling table completion.")
                self.enable_table_completion_var.set(False)
                return
            if self.table_cache is not None:
                self._refresh_scene_cache()
                self._render_left_image()
                self._render_viewer()
                self._update_info_panel()
                self.status_var.set("Table completion enabled.")
                self._focus_main_view()
                return
            self._start_table_completion()
            return

        self._refresh_scene_cache()
        self._render_left_image()
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set("Table completion disabled.")
        self._focus_main_view()

    def _start_table_completion(self) -> None:
        if self.raw_dense_map_cam is None or self.left_image_full is None or self.camera_bundle is None or self.left_image_np is None:
            messagebox.showerror("Missing scene", "Run S2M2 before table completion.")
            return

        def _worker() -> TableCompletionCache:
            assert self.left_image_full is not None
            assert self.raw_dense_map_cam is not None
            assert self.camera_bundle is not None
            assert self.left_image_np is not None

            timeout_s = self._parse_timeout_s()
            best_res: dict[str, Any] | None = None
            best_prompt = ""
            last_error = ""
            for prompt in ("table", "desk", "tabletop", "桌面"):
                res = _call_sam3_generic(
                    self.left_image_full,
                    sam3_url=FIXED_SAM3_URL,
                    timeout_s=timeout_s,
                    text_prompt=prompt,
                )
                if res.get("ok", False) and int(res.get("mask_area_px", 0)) > 2000:
                    best_res = res
                    best_prompt = prompt
                    break
                last_error = str(res.get("error", "sam3_empty_mask"))
            if best_res is None:
                raise RuntimeError(f"Table SAM3 failed: {last_error}")

            table_mask = np.asarray(best_res["mask"], dtype=bool)
            if table_mask.shape != self.raw_dense_map_cam.shape[:2]:
                raise RuntimeError("Table mask shape does not match dense_map_cam.")

            fit_mask = np.logical_and(table_mask, place_service._valid_cam_xyz_mask_map(self.raw_dense_map_cam))
            if int(np.count_nonzero(fit_mask)) < 80:
                raise RuntimeError("Too few valid table points for plane fit.")
            fit_points_cam = self.raw_dense_map_cam[fit_mask].astype(np.float32, copy=False)

            fitted = place_service._fit_plane_ransac(
                fit_points_cam,
                threshold=float(DEFAULT_PLANE_RANSAC_THR_M),
                max_iters=int(place_service.DEFAULT_TABLE_RANSAC_ITERS),
                min_inliers=max(80, int(round(0.18 * float(fit_points_cam.shape[0])))),
            )
            if fitted is None:
                raise RuntimeError("RANSAC failed to fit the table plane in camera frame.")

            plane_normal_cam, plane_d_cam, plane_inlier_mask = fitted
            support_points_cam = fit_points_cam[np.asarray(plane_inlier_mask, dtype=bool)]
            if support_points_cam.shape[0] < 80:
                raise RuntimeError("Table plane inlier support is too small.")

            ys, xs = np.nonzero(table_mask)
            sample_budget = max(1, int(DEFAULT_MAX_COMPLETION_POINTS))
            if ys.size > sample_budget:
                step = int(math.ceil(math.sqrt(float(ys.size) / float(sample_budget))))
                keep = ((ys - int(ys.min())) % step == 0) & ((xs - int(xs.min())) % step == 0)
                ys = ys[keep]
                xs = xs[keep]
            if ys.size == 0:
                raise RuntimeError("Table completion produced no candidate pixels.")

            fx = float(self.camera_bundle.k1[0, 0])
            fy = float(self.camera_bundle.k1[1, 1])
            cx = float(self.camera_bundle.k1[0, 2])
            cy = float(self.camera_bundle.k1[1, 2])
            if abs(fx) <= 1e-6 or abs(fy) <= 1e-6:
                raise RuntimeError("Invalid camera intrinsics.")

            dirs = np.stack(
                [
                    (xs.astype(np.float64) - cx) / fx,
                    (ys.astype(np.float64) - cy) / fy,
                    np.ones_like(xs, dtype=np.float64),
                ],
                axis=1,
            )
            normal64 = np.asarray(plane_normal_cam, dtype=np.float64).reshape(3)
            denom = dirs @ normal64
            valid = np.isfinite(denom) & (np.abs(denom) > 1e-6)
            t = np.zeros_like(denom, dtype=np.float64)
            t[valid] = -float(plane_d_cam) / denom[valid]
            valid &= np.isfinite(t) & (t > 1e-4)
            if not np.any(valid):
                raise RuntimeError("Table completion rays do not intersect the fitted plane.")

            dirs = dirs[valid]
            ys = ys[valid]
            xs = xs[valid]
            t = t[valid]
            points_cam = (dirs * t[:, None]).astype(np.float32, copy=False)
            z_ref = support_points_cam[:, 2].astype(np.float64, copy=False)
            z_lo = float(np.quantile(z_ref, 0.02) - 0.10)
            z_hi = float(np.quantile(z_ref, 0.98) + 0.10)
            keep = np.isfinite(points_cam[:, 2]) & (points_cam[:, 2] >= z_lo) & (points_cam[:, 2] <= z_hi)
            if not np.any(keep):
                raise RuntimeError("Table completion points are outside the valid plane depth band.")
            points_cam = points_cam[keep]
            ys = ys[keep]
            xs = xs[keep]

            support_points_base = push_planner._transform_points_cam_to_base(
                support_points_cam,
                self.camera_bundle.rot_cb,
                self.camera_bundle.shift_cb,
            ).astype(np.float32, copy=False)
            plane_points_base = push_planner._transform_points_cam_to_base(
                points_cam,
                self.camera_bundle.rot_cb,
                self.camera_bundle.shift_cb,
            ).astype(np.float32, copy=False)
            if support_points_base.shape[0] == 0 or plane_points_base.shape[0] == 0:
                raise RuntimeError("Camera-to-base transform failed for table completion points.")

            plane_normal_base, plane_d_base, plane_inlier_mask_base = place_service._fit_table_plane_in_base(
                support_points_base,
                voxel_size=float(place_service.DEFAULT_VOXEL_SIZE),
            )
            support_points_base_inliers = support_points_base[np.asarray(plane_inlier_mask_base, dtype=bool)]
            plane_colors = (self.left_image_np[ys, xs].astype(np.float32) / 255.0).astype(np.float32, copy=False)

            return TableCompletionCache(
                mask=table_mask,
                plane_points_base=plane_points_base.astype(np.float32, copy=False),
                plane_colors=plane_colors.astype(np.float32, copy=False),
                plane_pixels_yx=np.stack([ys, xs], axis=1).astype(np.int32, copy=False),
                plane_model=PlaneModel(
                    normal_base=np.asarray(plane_normal_base, dtype=np.float32),
                    d_base=float(plane_d_base),
                    support_points_base=support_points_base_inliers.astype(np.float32, copy=False),
                    source="sam3_text_table",
                ),
                prompt_used=str(best_prompt),
                debug={
                    "mask_area_px": int(table_mask.sum()),
                    "fit_points": int(fit_points_cam.shape[0]),
                    "inlier_points_cam": int(support_points_cam.shape[0]),
                    "completion_points": int(plane_points_base.shape[0]),
                    "sam3_latency_ms": int(best_res.get("latency_ms", 0)),
                },
            )

        self._run_background("Building table completion", _worker, self._on_table_completion_ready)
        self._focus_main_view()

    def _on_table_completion_ready(self, cache: TableCompletionCache) -> None:
        self.table_cache = cache
        if self.object_state is not None:
            self.object_state.plane_model = cache.plane_model
        self._refresh_scene_cache()
        self._render_left_image()
        if self.object_state is not None:
            self._release_box_pose()
        else:
            self._render_viewer()
            self._update_info_panel()
        self.status_var.set(
            f"Table completion ready. prompt={cache.prompt_used}, completion_points={cache.plane_points_base.shape[0]}"
        )
        self._focus_main_view()

    def _fit_local_table_plane(self, bbox: tuple[int, int, int, int], object_mask: np.ndarray) -> PlaneModel:
        if self.raw_points_base.shape[0] == 0 or self.left_image_full is None:
            raise RuntimeError("Missing raw base cloud for local table fit.")
        img_w, img_h = self.left_image_full.size
        region_bbox = _expand_bbox(bbox, img_w, img_h)
        region_mask = _bbox_to_mask(region_bbox, img_w, img_h)
        keep = region_mask[self.raw_pixels_yx[:, 0], self.raw_pixels_yx[:, 1]]
        keep &= ~object_mask[self.raw_pixels_yx[:, 0], self.raw_pixels_yx[:, 1]]
        points_base = self.raw_points_base[keep]
        points_base = _finite_points(points_base)
        if points_base.shape[0] < 80:
            raise RuntimeError("Not enough local support points outside the object mask to fit the table plane.")

        fitted = place_service._fit_plane_ransac(
            points_base,
            threshold=float(DEFAULT_PLANE_RANSAC_THR_M),
            max_iters=int(place_service.DEFAULT_TABLE_RANSAC_ITERS),
            min_inliers=max(80, int(round(0.18 * float(points_base.shape[0])))),
        )
        if fitted is None:
            raise RuntimeError("Local table plane RANSAC failed.")
        normal, d, inlier_mask = fitted
        support_points = points_base[np.asarray(inlier_mask, dtype=bool)]
        if support_points.shape[0] < 80:
            raise RuntimeError("Local table plane support is too small.")
        return PlaneModel(
            normal_base=np.asarray(normal, dtype=np.float32),
            d_base=float(d),
            support_points_base=support_points.astype(np.float32, copy=False),
            source="local_bbox_support",
        )

    def _fit_object_box(self, bbox: tuple[int, int, int, int], object_mask: np.ndarray, plane_model: PlaneModel) -> ObjectState:
        if self.raw_points_base.shape[0] == 0:
            raise RuntimeError("Missing raw base cloud.")

        target_keep = object_mask[self.raw_pixels_yx[:, 0], self.raw_pixels_yx[:, 1]]
        target_points_base = _finite_points(self.raw_points_base[target_keep])
        if target_points_base.shape[0] < 24:
            raise RuntimeError("Segmented object cloud is too small.")

        target_margin = _plane_margin_z(target_points_base, plane_model.normal_base, plane_model.d_base)
        target_keep_above = target_margin > -0.003
        if int(np.count_nonzero(target_keep_above)) >= 24:
            target_points_base = target_points_base[target_keep_above]
        rect_target = push_planner._fit_rectangle_pose_base(target_points_base.astype(np.float32, copy=False))

        fit_mask, fit_mask_erode_px = push_planner._erode_mask_for_fit(object_mask)
        fit_keep = fit_mask[self.raw_pixels_yx[:, 0], self.raw_pixels_yx[:, 1]]
        fit_points_base_pre = _finite_points(self.raw_points_base[fit_keep])
        fit_mask_selection_reason = "eroded_mask" if fit_mask_erode_px > 0 else "full_mask"
        if fit_points_base_pre.shape[0] < 24:
            fit_points_base_pre = target_points_base.copy()
            fit_mask_erode_px = 0
            fit_mask_selection_reason = "full_mask_fallback"

        rect_fit_seed = push_planner._fit_rectangle_pose_base(fit_points_base_pre.astype(np.float32, copy=False))
        slender_mask_truncated = (
            fit_mask_erode_px > 0
            and float(rect_target.get("aspect_ratio", 0.0)) >= float(DEFAULT_SLENDER_OBJECT_ASPECT_RATIO)
            and float(rect_fit_seed.get("length", 0.0))
            < float(DEFAULT_SLENDER_OBJECT_LENGTH_RECOVER_RATIO) * max(float(rect_target.get("length", 0.0)), 1e-6)
        )
        if slender_mask_truncated:
            fit_points_base_pre = target_points_base.copy()
            fit_mask_erode_px = 0
            fit_mask_selection_reason = "full_mask_recovered_from_erode"
            rect_fit_seed = rect_target

        fit_points_before_clearance = fit_points_base_pre.astype(np.float32, copy=False)
        fit_plane_clearance_used = False
        fit_plane_clearance_reason = "skipped"
        fit_points_after_clearance_candidate = fit_points_before_clearance

        fit_margin = _plane_margin_z(fit_points_before_clearance, plane_model.normal_base, plane_model.d_base)
        fit_keep_above = fit_margin > float(DEFAULT_OBJECT_PLANE_CLEARANCE_M)
        if int(np.count_nonzero(fit_keep_above)) >= 24:
            fit_points_after_clearance_candidate = fit_points_before_clearance[fit_keep_above].astype(np.float32, copy=False)
            rect_after_clearance = push_planner._fit_rectangle_pose_base(fit_points_after_clearance_candidate)
            slender_clearance_truncated = (
                float(rect_fit_seed.get("aspect_ratio", 0.0)) >= float(DEFAULT_SLENDER_OBJECT_ASPECT_RATIO)
                and float(rect_after_clearance.get("length", 0.0))
                < float(DEFAULT_SLENDER_OBJECT_LENGTH_RECOVER_RATIO) * max(float(rect_fit_seed.get("length", 0.0)), 1e-6)
            )
            if slender_clearance_truncated:
                fit_points_base_pre = fit_points_before_clearance.copy()
                fit_plane_clearance_reason = "recovered_from_clearance"
            else:
                fit_points_base_pre = fit_points_after_clearance_candidate
                fit_plane_clearance_used = True
                fit_plane_clearance_reason = "applied"
        else:
            fit_points_base_pre = fit_points_before_clearance.copy()
            fit_plane_clearance_reason = "insufficient_clearance_points"

        q_lo_xy, q_hi_xy = np.quantile(fit_points_base_pre[:, :2].astype(np.float64, copy=False), [0.10, 0.90], axis=0)
        robust_xy_spread = np.maximum(q_hi_xy - q_lo_xy, 1e-4)
        fit_component_voxel_size = float(np.clip(0.07 * float(np.min(robust_xy_spread)), 0.0025, 0.0060))
        fit_points_base_filtered, fit_component_debug = push_planner._keep_largest_xy_component(
            fit_points_base_pre,
            voxel_size=fit_component_voxel_size,
        )
        min_fit_keep_count = max(24, int(round(0.35 * float(fit_points_base_pre.shape[0]))))
        fit_cluster_used = bool(fit_points_base_filtered.shape[0] >= min_fit_keep_count)
        fit_points_base = (
            fit_points_base_filtered.astype(np.float32, copy=False)
            if fit_cluster_used
            else fit_points_base_pre.astype(np.float32, copy=False)
        )

        rect_pre = push_planner._fit_rectangle_pose_base(fit_points_base_pre.astype(np.float32, copy=False))
        rect_filtered: dict[str, Any] | None = None
        fit_selection_reason = "largest_component" if fit_cluster_used else "pre_component_union"
        if fit_cluster_used:
            rect_filtered = push_planner._fit_rectangle_pose_base(fit_points_base_filtered.astype(np.float32, copy=False))
            pre_length = float(rect_pre.get("length", 0.0))
            filtered_length = float(rect_filtered.get("length", 0.0))
            pre_aspect_ratio = float(rect_pre.get("aspect_ratio", 0.0))
            component_count = int(fit_component_debug.get("component_count", 0))
            slender_cluster_truncated = (
                component_count >= 2
                and pre_aspect_ratio >= float(DEFAULT_SLENDER_OBJECT_ASPECT_RATIO)
                and filtered_length < float(DEFAULT_SLENDER_OBJECT_LENGTH_RECOVER_RATIO) * max(pre_length, 1e-6)
            )
            if slender_cluster_truncated:
                fit_points_base = fit_points_base_pre.astype(np.float32, copy=False)
                fit_cluster_used = False
                fit_selection_reason = "slender_union_recovered"
        rect = rect_pre if fit_selection_reason != "largest_component" else rect_filtered
        if rect is None:
            rect = rect_pre
        center_xy = np.asarray(rect["center_base"], dtype=np.float32).reshape(3)[:2]
        yaw = float(rect["yaw"])
        rect_aspect_ratio = float(rect.get("aspect_ratio", 0.0))
        is_slender_object = bool(rect_aspect_ratio >= float(DEFAULT_SLENDER_OBJECT_ASPECT_RATIO))
        min_cross_section = float(DEFAULT_OBJECT_MIN_CROSS_SECTION_M if is_slender_object else 0.02)
        min_height = float(DEFAULT_OBJECT_MIN_CROSS_SECTION_M if is_slender_object else DEFAULT_OBJECT_MIN_HEIGHT_M)
        length = float(max(0.02, rect["length"]))
        width = float(max(min_cross_section, rect["width"]))

        support_z_max = _max_plane_height_under_footprint(
            center_xy=center_xy,
            yaw_rad=yaw,
            size_xy=np.array([length, width], dtype=np.float32),
            normal_base=plane_model.normal_base,
            d_base=plane_model.d_base,
        )
        height_source_points = fit_points_base.astype(np.float32, copy=False)
        if height_source_points.shape[0] < 24:
            height_source_points = target_points_base.astype(np.float32, copy=False)
        local_heights = height_source_points[:, 2].astype(np.float64, copy=False) - float(support_z_max)
        local_heights = local_heights[np.isfinite(local_heights)]
        height_points_above = local_heights[local_heights > -0.002]
        height_quantile = float(
            DEFAULT_SLENDER_OBJECT_HEIGHT_QUANTILE if is_slender_object else DEFAULT_OBJECT_HEIGHT_QUANTILE
        )
        raw_height = float(
            _safe_quantile(
                height_points_above,
                height_quantile,
                _safe_quantile(target_points_base[:, 2], 0.98, support_z_max + min_height) - float(support_z_max),
            )
        )
        if is_slender_object and np.isfinite(raw_height):
            raw_height = float(min(raw_height, float(width) * float(DEFAULT_SLENDER_OBJECT_THICKNESS_CAP_RATIO)))
        if not np.isfinite(raw_height) or raw_height < min_height:
            fallback_height = float(
                _safe_quantile(target_points_base[:, 2], 0.98, 0.0) - _safe_quantile(target_points_base[:, 2], 0.02, 0.0)
            )
            if is_slender_object and np.isfinite(fallback_height):
                fallback_height = float(min(fallback_height, float(width) * float(DEFAULT_SLENDER_OBJECT_THICKNESS_CAP_RATIO)))
            raw_height = max(float(min_height), float(fallback_height))
        height = float(np.clip(raw_height, min_height, DEFAULT_OBJECT_MAX_HEIGHT_M))

        center_base = np.array([float(center_xy[0]), float(center_xy[1]), float(support_z_max + 0.5 * height)], dtype=np.float32)
        rpy_rad = np.array([0.0, 0.0, float(yaw)], dtype=np.float32)
        size_xyz = np.array([float(length), float(width), float(height)], dtype=np.float32)

        return ObjectState(
            bbox=tuple(int(v) for v in bbox),
            mask=np.asarray(object_mask, dtype=bool),
            mask_area_px=int(np.count_nonzero(object_mask)),
            points_base=target_points_base.astype(np.float32, copy=False),
            fit_points_base=fit_points_base.astype(np.float32, copy=False),
            plane_model=plane_model,
            center_base=center_base.astype(np.float32, copy=True),
            rpy_rad=rpy_rad.astype(np.float32, copy=True),
            size_xyz=size_xyz.astype(np.float32, copy=True),
            init_center_base=center_base.astype(np.float32, copy=True),
            init_rpy_rad=rpy_rad.astype(np.float32, copy=True),
            debug={
                "fit_mask_erode_px": int(fit_mask_erode_px),
                "fit_mask_selection_reason": str(fit_mask_selection_reason),
                "fit_cluster_used": bool(fit_cluster_used),
                "fit_selection_reason": str(fit_selection_reason),
                "fit_plane_clearance_used": bool(fit_plane_clearance_used),
                "fit_plane_clearance_reason": str(fit_plane_clearance_reason),
                "fit_cluster_component_count": int(fit_component_debug.get("component_count", 0)),
                "fit_cluster_largest_component_point_count": int(
                    fit_component_debug.get("largest_component_point_count", 0)
                ),
                "fit_cluster_voxel_size_m": float(
                    fit_component_debug.get("voxel_size_m", fit_component_voxel_size)
                ),
                "raw_object_points": int(target_points_base.shape[0]),
                "fit_points_before_clearance": int(fit_points_before_clearance.shape[0]),
                "fit_points_after_clearance_candidate": int(fit_points_after_clearance_candidate.shape[0]),
                "fit_points": int(fit_points_base.shape[0]),
                "fit_rect_target_length": float(rect_target.get("length", 0.0)),
                "fit_rect_target_width": float(rect_target.get("width", 0.0)),
                "fit_rect_pre_length": float(rect_pre.get("length", 0.0)),
                "fit_rect_pre_width": float(rect_pre.get("width", 0.0)),
                "fit_rect_filtered_length": float(
                    rect_filtered.get("length", 0.0) if isinstance(rect_filtered, dict) else 0.0
                ),
                "fit_rect_filtered_width": float(
                    rect_filtered.get("width", 0.0) if isinstance(rect_filtered, dict) else 0.0
                ),
                "rect_aspect_ratio": float(rect.get("aspect_ratio", 0.0)),
                "object_is_slender": bool(is_slender_object),
                "height_quantile": float(height_quantile),
                "min_cross_section": float(min_cross_section),
                "min_height": float(min_height),
                "height_points_above_count": int(height_points_above.shape[0]),
                "support_z_max": float(support_z_max),
            },
        )

    def _start_segment_object(self) -> None:
        if self.raw_dense_map_cam is None or self.left_image_full is None:
            messagebox.showerror("Missing scene", "Run S2M2 before object segmentation.")
            return
        if self.selected_bbox is None:
            messagebox.showerror("Missing bbox", "Draw a bbox on the left image first.")
            return

        bbox = tuple(int(v) for v in self.selected_bbox)

        def _worker() -> ObjectState:
            assert self.left_image_full is not None
            sam3_res = _call_sam3_generic(
                self.left_image_full,
                sam3_url=FIXED_SAM3_URL,
                timeout_s=self._parse_timeout_s(),
                box_prompts=[[int(v) for v in bbox]],
                box_labels=[True],
            )
            if not sam3_res.get("ok", False):
                raise RuntimeError(f"Object SAM3 failed: {sam3_res.get('error', 'unknown')}")
            object_mask = np.asarray(sam3_res["mask"], dtype=bool)
            if object_mask.shape != self.raw_dense_map_cam.shape[:2]:
                raise RuntimeError("Object mask shape mismatch.")
            plane_model = self.table_cache.plane_model if self.table_cache is not None else self._fit_local_table_plane(bbox, object_mask)
            obj = self._fit_object_box(bbox, object_mask, plane_model)
            obj.debug["sam3_latency_ms"] = int(sam3_res.get("latency_ms", 0))
            return obj

        self._run_background("Segmenting object", _worker, self._on_object_ready)
        self._focus_main_view()

    def _on_object_ready(self, obj: ObjectState) -> None:
        self.object_state = obj
        self.grasp_state = self._make_default_grasp_state(obj)
        self._clear_place_plan()
        self.awaiting_grasp_click = False
        self._refresh_scene_cache()
        self._release_box_pose()
        self._render_left_image()
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set(
            f"Object ready. mask_px={obj.mask_area_px}, fit_points={obj.fit_points_base.shape[0]}"
        )
        self._focus_main_view()

    def _current_plane_model(self) -> PlaneModel | None:
        if self.object_state is not None:
            return self.object_state.plane_model
        if self.table_cache is not None:
            return self.table_cache.plane_model
        return None

    def _bbox_pixel_keep_mask(self, pixels_yx: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
        pixels = np.asarray(pixels_yx, dtype=np.int32)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.shape[0] == 0 or bbox is None:
            return np.zeros((pixels.shape[0] if pixels.ndim == 2 else 0,), dtype=bool)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        ys = pixels[:, 0]
        xs = pixels[:, 1]
        return (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)

    def _refresh_scene_cache(self) -> None:
        if self.raw_points_base.shape[0] == 0:
            self.scene_points_base = _empty_points()
            self.scene_colors = _empty_colors()
            self.collision_points_base = _empty_points()
            return

        keep = np.ones((self.raw_points_base.shape[0],), dtype=bool)
        if self.object_state is not None:
            keep &= ~self.object_state.mask[self.raw_pixels_yx[:, 0], self.raw_pixels_yx[:, 1]]

        display_keep = keep.copy()
        if self.segment_scene_bbox is not None:
            display_keep &= self._bbox_pixel_keep_mask(self.raw_pixels_yx, self.segment_scene_bbox)

        scene_points_full_raw = self.raw_points_base[keep].astype(np.float32, copy=False)
        scene_colors_full_raw = self.raw_colors[keep].astype(np.float32, copy=False)
        keep_full_finite = _finite_point_mask(scene_points_full_raw)
        scene_points_full = scene_points_full_raw[keep_full_finite].astype(np.float32, copy=False)
        scene_colors_full = scene_colors_full_raw[keep_full_finite].astype(np.float32, copy=False)

        scene_points_raw = self.raw_points_base[display_keep].astype(np.float32, copy=False)
        scene_colors_raw = self.raw_colors[display_keep].astype(np.float32, copy=False)
        keep_display_finite = _finite_point_mask(scene_points_raw)
        scene_points = scene_points_raw[keep_display_finite].astype(np.float32, copy=False)
        scene_colors = scene_colors_raw[keep_display_finite].astype(np.float32, copy=False)
        if self.enable_table_completion_var.get() and self.table_cache is not None:
            table_points_full_raw = self.table_cache.plane_points_base.astype(np.float32, copy=False)
            table_colors_full_raw = self.table_cache.plane_colors.astype(np.float32, copy=False)
            keep_table_full = _finite_point_mask(table_points_full_raw)
            table_points_full = table_points_full_raw[keep_table_full].astype(np.float32, copy=False)
            table_colors_full = table_colors_full_raw[keep_table_full].astype(np.float32, copy=False)
            if scene_points_full.shape[0] == 0:
                scene_points_full = table_points_full.copy()
                scene_colors_full = table_colors_full.copy()
            else:
                scene_points_full = np.vstack([scene_points_full, table_points_full]).astype(np.float32, copy=False)
                scene_colors_full = np.vstack([scene_colors_full, table_colors_full]).astype(np.float32, copy=False)

            table_display_keep = np.ones((table_points_full.shape[0],), dtype=bool)
            if self.segment_scene_bbox is not None:
                table_display_keep &= self._bbox_pixel_keep_mask(self.table_cache.plane_pixels_yx, self.segment_scene_bbox)
            table_points_raw = table_points_full[table_display_keep].astype(np.float32, copy=False)
            table_colors_raw = table_colors_full[table_display_keep].astype(np.float32, copy=False)
            keep_table_display = _finite_point_mask(table_points_raw)
            table_points = table_points_raw[keep_table_display].astype(np.float32, copy=False)
            table_colors = table_colors_raw[keep_table_display].astype(np.float32, copy=False)
            if scene_points.shape[0] == 0:
                scene_points = table_points.copy()
                scene_colors = table_colors.copy()
            elif table_points.shape[0] > 0:
                scene_points = np.vstack([scene_points, table_points]).astype(np.float32, copy=False)
                scene_colors = np.vstack([scene_colors, table_colors]).astype(np.float32, copy=False)

        plane_model = self._current_plane_model()
        collision_points = scene_points_full
        if (
            plane_model is not None
            and collision_points.shape[0] > 0
            and _is_finite_vector(plane_model.normal_base, 3)
            and np.isfinite(float(plane_model.d_base))
        ):
            plane_dist = np.abs(
                collision_points.astype(np.float64, copy=False) @ plane_model.normal_base.astype(np.float64, copy=False)
                + float(plane_model.d_base)
            )
            keep_collision = plane_dist > float(DEFAULT_TABLE_PLANE_EXCLUDE_M)
            collision_points = collision_points[keep_collision]

        if collision_points.shape[0] > DEFAULT_MAX_COLLISION_POINTS:
            collision_points = _subsample_points(collision_points, DEFAULT_MAX_COLLISION_POINTS, seed=11)

        self.scene_points_base = scene_points.astype(np.float32, copy=False)
        self.scene_colors = scene_colors.astype(np.float32, copy=False)
        self.collision_points_base = collision_points.astype(np.float32, copy=False)

    def _apply_segment_scene_from_bbox(self) -> None:
        if self.raw_points_base.shape[0] == 0:
            self.status_var.set("Run S2M2 before segmenting the scene.")
            return
        self.segment_scene_bbox = (
            tuple(int(v) for v in self.selected_bbox) if self.selected_bbox is not None else None
        )
        self._refresh_scene_cache()
        self._render_viewer()
        self._update_info_panel()
        if self.segment_scene_bbox is None:
            self.status_var.set(
                f"Scene reset to full point cloud. render_points={self.scene_points_base.shape[0]}"
            )
        else:
            self.status_var.set(
                f"Scene segmented to bbox {self.segment_scene_bbox}. render_points={self.scene_points_base.shape[0]}"
            )
        self._focus_main_view()

    def _clear_bbox(self) -> None:
        self.selected_bbox = None
        self.preview_bbox = None
        self._bbox_drag_start = None
        self.awaiting_grasp_click = False
        self._redraw_left_bboxes()
        self._update_info_panel()
        self.status_var.set("BBox cleared.")
        self._focus_main_view()

    def _clear_object(self) -> None:
        self.object_state = None
        self.grasp_state = None
        self.place_target_state = None
        self.place_plan_state = None
        self.selected_bbox = None
        self.segment_scene_bbox = None
        self.preview_bbox = None
        self.awaiting_grasp_click = False
        self._refresh_scene_cache()
        self._render_left_image()
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set("Object cleared.")
        self._focus_main_view()

    def _reset_box_pose(self) -> None:
        if self.object_state is None:
            return
        self.object_state.center_base = self.object_state.init_center_base.astype(np.float32, copy=True)
        self.object_state.rpy_rad = self.object_state.init_rpy_rad.astype(np.float32, copy=True)
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set("Box reset to fitted pose.")
        self._focus_main_view()

    def _nudge_box(self, delta_xyz_dir: np.ndarray | None, delta_rpy_dir: np.ndarray | None) -> None:
        if self.object_state is None:
            self.status_var.set("No object selected.")
            return
        if delta_xyz_dir is not None:
            self.object_state.center_base = (
                self.object_state.center_base
                + np.asarray(delta_xyz_dir, dtype=np.float32).reshape(3) * float(self._parse_move_step())
            ).astype(np.float32, copy=False)
        if delta_rpy_dir is not None:
            self.object_state.rpy_rad = (
                self.object_state.rpy_rad
                + np.asarray(delta_rpy_dir, dtype=np.float32).reshape(3) * float(self._parse_rot_step_rad())
            ).astype(np.float32, copy=False)
            self.object_state.rpy_rad[2] = float(_wrap_to_pi(self.object_state.rpy_rad[2]))
        if not self.free_pose_var.get():
            self._release_box_pose()
            return
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set("Adjusted free pose. Press Space or 'Release / Drop' to settle.")
        self._focus_main_view()

    def _has_box_point_collision_axes_in_points(
        self,
        center: np.ndarray,
        axes_base: np.ndarray,
        size_xyz: np.ndarray,
        collision_points_base: np.ndarray,
    ) -> bool:
        if collision_points_base.shape[0] == 0:
            return False
        pts = _finite_points(collision_points_base).astype(np.float64, copy=False)
        ctr = np.asarray(center, dtype=np.float64).reshape(1, 3)
        axes = _orthonormalize_axes(axes_base).astype(np.float64, copy=False)
        local = (pts - ctr) @ axes
        half = 0.5 * np.asarray(size_xyz, dtype=np.float64).reshape(1, 3)
        inside = np.all(np.abs(local) <= (half + float(DEFAULT_COLLISION_MARGIN_M)), axis=1)
        return bool(np.any(inside))

    def _has_box_point_collision_axes(self, center: np.ndarray, axes_base: np.ndarray, size_xyz: np.ndarray) -> bool:
        return bool(self._has_box_point_collision_axes_in_points(center, axes_base, size_xyz, self.collision_points_base))

    def _has_box_point_collision(self, center: np.ndarray, rpy_rad: np.ndarray, size_xyz: np.ndarray) -> bool:
        return bool(self._has_box_point_collision_axes(center, _rpy_to_matrix(rpy_rad), size_xyz))

    def _has_box_plane_collision_axes(self, center: np.ndarray, axes_base: np.ndarray, size_xyz: np.ndarray, plane_model: PlaneModel | None) -> bool:
        if plane_model is None:
            return False
        corners = _box_corners_axes(center, axes_base, size_xyz)
        margins = _plane_margin_z(corners, plane_model.normal_base, plane_model.d_base)
        return bool(np.any(margins < -float(DEFAULT_PLANE_CONTACT_EPS_M)))

    def _has_box_plane_collision(self, center: np.ndarray, rpy_rad: np.ndarray, size_xyz: np.ndarray, plane_model: PlaneModel | None) -> bool:
        return bool(self._has_box_plane_collision_axes(center, _rpy_to_matrix(rpy_rad), size_xyz, plane_model))

    def _resolve_box_penetration(
        self,
        center: np.ndarray,
        rpy_rad: np.ndarray,
        size_xyz: np.ndarray,
        plane_model: PlaneModel | None,
    ) -> np.ndarray:
        ctr = np.asarray(center, dtype=np.float32).reshape(3).copy()
        size = np.asarray(size_xyz, dtype=np.float32).reshape(3)
        half = 0.5 * size.astype(np.float64, copy=False)
        rot = _rpy_to_matrix(rpy_rad)

        for _ in range(12):
            candidates: list[tuple[float, np.ndarray]] = []

            if self.collision_points_base.shape[0] > 0:
                local = _obb_local_points(self.collision_points_base, ctr, rpy_rad)
                inside = np.all(np.abs(local) <= (half.reshape((1, 3)) + float(DEFAULT_COLLISION_MARGIN_M)), axis=1)
                inside_pts = local[inside]
                if inside_pts.shape[0] > 0:
                    for axis in range(3):
                        axis_world = rot[:, axis].astype(np.float64, copy=False)
                        t_pos = float(np.max(inside_pts[:, axis] + half[axis] + float(DEFAULT_COLLISION_MARGIN_M)))
                        t_neg = float(np.max(half[axis] - inside_pts[:, axis] + float(DEFAULT_COLLISION_MARGIN_M)))
                        if np.isfinite(t_pos) and t_pos > 1e-6:
                            candidates.append((abs(t_pos), axis_world * t_pos))
                        if np.isfinite(t_neg) and t_neg > 1e-6:
                            candidates.append((abs(t_neg), -axis_world * t_neg))

            if plane_model is not None:
                corners = _box_corners_world(ctr, rpy_rad, size)
                margins = _plane_margin_z(corners, plane_model.normal_base, plane_model.d_base)
                min_margin = float(np.min(margins)) if margins.size > 0 else 0.0
                if np.isfinite(min_margin) and min_margin < float(DEFAULT_PLANE_CONTACT_EPS_M):
                    dz = float(DEFAULT_PLANE_CONTACT_EPS_M - min_margin)
                    if dz > 1e-6:
                        candidates.append((dz, np.array([0.0, 0.0, dz], dtype=np.float64)))

            if not candidates:
                break

            candidates.sort(key=lambda item: item[0])
            best_delta = np.asarray(candidates[0][1], dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(best_delta)) or float(np.linalg.norm(best_delta)) <= 1e-6:
                break
            ctr = (ctr.astype(np.float64) + best_delta).astype(np.float32, copy=False)

        return ctr.astype(np.float32, copy=False)

    def _drop_box_along_gravity(
        self,
        center: np.ndarray,
        rpy_rad: np.ndarray,
        size_xyz: np.ndarray,
        plane_model: PlaneModel | None,
    ) -> np.ndarray:
        ctr = np.asarray(center, dtype=np.float32).reshape(3).copy()
        plane_limit = 0.40
        if plane_model is not None:
            corners = _box_corners_world(ctr, rpy_rad, size_xyz)
            margins = _plane_margin_z(corners, plane_model.normal_base, plane_model.d_base)
            if margins.size > 0:
                plane_limit = max(0.0, float(np.min(margins)))
        if plane_limit <= 1e-6:
            return ctr

        step = float(np.clip(plane_limit / 24.0, 0.002, 0.015))
        free_drop = 0.0
        current = step
        while current < plane_limit:
            test_center = ctr.copy()
            test_center[2] -= current
            if self._has_box_point_collision(test_center, rpy_rad, size_xyz):
                break
            if self._has_box_plane_collision(test_center, rpy_rad, size_xyz, plane_model):
                break
            free_drop = current
            current += step

        if current >= plane_limit:
            test_center = ctr.copy()
            test_center[2] -= plane_limit
            if not self._has_box_point_collision(test_center, rpy_rad, size_xyz) and not self._has_box_plane_collision(
                test_center, rpy_rad, size_xyz, plane_model
            ):
                ctr[2] -= plane_limit
                return ctr
            current = plane_limit

        low = float(free_drop)
        high = float(min(current, plane_limit))
        for _ in range(18):
            mid = 0.5 * (low + high)
            test_center = ctr.copy()
            test_center[2] -= mid
            if self._has_box_point_collision(test_center, rpy_rad, size_xyz) or self._has_box_plane_collision(
                test_center, rpy_rad, size_xyz, plane_model
            ):
                high = mid
            else:
                low = mid
        ctr[2] -= float(low)
        return ctr

    def _plan_place_grasp_trajectory(
        self,
        start_center_base: np.ndarray,
        start_axes_base: np.ndarray,
        start_gripper_origin_base: np.ndarray,
        start_gripper_axes_base: np.ndarray,
        goal_center_base: np.ndarray,
        goal_axes_base: np.ndarray,
        goal_gripper_origin_base: np.ndarray,
        goal_gripper_axes_base: np.ndarray,
        size_xyz: np.ndarray,
        collision_points_base: np.ndarray | None = None,
        target_profile: placedof_planner.TargetProfile | None = None,
        placement_rule_id: str | None = None,
    ) -> dict[str, Any]:
        plane_model = self._current_plane_model()
        start_center = np.asarray(start_center_base, dtype=np.float32).reshape(3)
        goal_center = np.asarray(goal_center_base, dtype=np.float32).reshape(3)
        start_axes = _orthonormalize_axes(start_axes_base)
        goal_axes = _orthonormalize_axes(goal_axes_base)
        start_gripper_origin = np.asarray(start_gripper_origin_base, dtype=np.float32).reshape(3)
        goal_gripper_origin = np.asarray(goal_gripper_origin_base, dtype=np.float32).reshape(3)
        start_gripper_axes = _orthonormalize_axes(start_gripper_axes_base)
        goal_gripper_axes = _orthonormalize_axes(goal_gripper_axes_base)
        start_grasp_origin_local, start_grasp_axes_local = self._derive_grasp_pose_local_from_object_pose(
            start_center,
            start_axes,
            start_gripper_origin,
            start_gripper_axes,
        )
        goal_grasp_origin_local, goal_grasp_axes_local = self._derive_grasp_pose_local_from_object_pose(
            goal_center,
            goal_axes,
            goal_gripper_origin,
            goal_gripper_axes,
        )
        size = np.asarray(size_xyz, dtype=np.float32).reshape(3)
        total_rotation_rad = _rotation_geodesic_angle(start_axes, goal_axes)
        grasp_local_rotation_delta_rad = _rotation_geodesic_angle(start_grasp_axes_local, goal_grasp_axes_local)
        gripper_rotation_total_rad = _rotation_geodesic_angle(start_gripper_axes, goal_gripper_axes)
        voxel = float(max(DEFAULT_PLACE_TRAJECTORY_VOXEL_M, 1e-4))
        transition_step_m = float(DEFAULT_PLACE_TRAJECTORY_TRANSITION_STEP_M)
        rotation_weight = float(DEFAULT_PLACE_TRAJECTORY_ROTATION_WEIGHT_M)
        xy_margin = float(DEFAULT_PLACE_TRAJECTORY_BOUNDS_MARGIN_M + 0.5 * np.linalg.norm(size[:2]))
        z_margin = float(DEFAULT_PLACE_TRAJECTORY_Z_LIFT_MARGIN_M + 0.5 * size[2] + 0.03)
        smooth_passes = max(0, int(DEFAULT_PLACE_TRAJECTORY_SMOOTH_PASSES))
        smooth_max_lift_cells = max(0, int(math.ceil(float(DEFAULT_PLACE_TRAJECTORY_SMOOTH_MAX_LIFT_M) / voxel)))
        smooth_lift_weight = float(max(0.0, DEFAULT_PLACE_TRAJECTORY_SMOOTH_LIFT_WEIGHT))

        collision_points = (
            self.collision_points_base.astype(np.float32, copy=False)
            if collision_points_base is None
            else _finite_points(collision_points_base).astype(np.float32, copy=False)
        )
        near_pts = collision_points.astype(np.float64, copy=False)
        if near_pts.shape[0] > 0:
            keep = (
                (near_pts[:, 0] >= float(min(start_center[0], goal_center[0]) - xy_margin))
                & (near_pts[:, 0] <= float(max(start_center[0], goal_center[0]) + xy_margin))
                & (near_pts[:, 1] >= float(min(start_center[1], goal_center[1]) - xy_margin))
                & (near_pts[:, 1] <= float(max(start_center[1], goal_center[1]) + xy_margin))
                & (near_pts[:, 2] >= float(min(start_center[2], goal_center[2]) - z_margin))
                & (near_pts[:, 2] <= float(max(start_center[2], goal_center[2]) + z_margin + size[2]))
            )
            if np.any(keep):
                near_pts = near_pts[keep]
            if near_pts.shape[0] > 12000:
                near_pts = _subsample_points(near_pts.astype(np.float32, copy=False), 12000, seed=59).astype(np.float64, copy=False)

        def _point_collision(center: np.ndarray, axes: np.ndarray) -> bool:
            if near_pts.shape[0] == 0:
                return False
            local = (near_pts - np.asarray(center, dtype=np.float64).reshape((1, 3))) @ np.asarray(axes, dtype=np.float64).reshape(3, 3)
            half = 0.5 * np.asarray(size, dtype=np.float64).reshape(1, 3)
            inside = np.all(np.abs(local) <= (half + float(DEFAULT_COLLISION_MARGIN_M)), axis=1)
            return bool(np.any(inside))

        def _pose_collision(center: np.ndarray, axes: np.ndarray) -> bool:
            if _point_collision(center, axes):
                return True
            return bool(self._has_box_plane_collision_axes(center, axes, size, plane_model))

        if _pose_collision(start_center, start_axes):
            raise RuntimeError("Pick pose OBB is not collision-free for trajectory planning.")
        if _pose_collision(goal_center, goal_axes):
            raise RuntimeError("Final place OBB is not collision-free for trajectory planning.")

        obstacle_top_z = float(max(start_center[2], goal_center[2]))
        if near_pts.shape[0] > 0:
            obstacle_top_z = max(obstacle_top_z, float(np.max(near_pts[:, 2])))

        lower = np.array(
            [
                float(min(start_center[0], goal_center[0]) - xy_margin),
                float(min(start_center[1], goal_center[1]) - xy_margin),
                float(min(start_center[2], goal_center[2]) - max(z_margin, 0.5 * float(size[2]) + 0.04)),
            ],
            dtype=np.float64,
        )
        upper = np.array(
            [
                float(max(start_center[0], goal_center[0]) + xy_margin),
                float(max(start_center[1], goal_center[1]) + xy_margin),
                float(max(max(start_center[2], goal_center[2]), obstacle_top_z + 0.5 * float(size[2]) + 0.04) + z_margin),
            ],
            dtype=np.float64,
        )
        span = np.maximum(upper - lower, voxel)
        dims_arr = np.maximum(3, np.ceil(span / voxel).astype(np.int32) + 1)
        dims_tuple = (int(dims_arr[0]), int(dims_arr[1]), int(dims_arr[2]))

        def _idx_in_bounds(idx: tuple[int, int, int]) -> bool:
            return 0 <= idx[0] < dims_tuple[0] and 0 <= idx[1] < dims_tuple[1] and 0 <= idx[2] < dims_tuple[2]

        def _point_to_idx(point: np.ndarray) -> tuple[int, int, int]:
            arr = np.asarray(point, dtype=np.float64).reshape(3)
            idx = np.floor((arr - lower) / voxel).astype(np.int32)
            idx = np.clip(idx, 0, np.asarray(dims_tuple, dtype=np.int32) - 1)
            return int(idx[0]), int(idx[1]), int(idx[2])

        def _idx_to_point(idx: tuple[int, int, int]) -> np.ndarray:
            return (lower + (np.asarray(idx, dtype=np.float64) + 0.5) * voxel).astype(np.float32)

        occupancy_raw = np.zeros(dims_tuple, dtype=bool)
        if near_pts.shape[0] > 0:
            obs_idx = np.floor((near_pts - lower.reshape((1, 3))) / voxel).astype(np.int32)
            keep = (
                (obs_idx[:, 0] >= 0)
                & (obs_idx[:, 0] < dims_tuple[0])
                & (obs_idx[:, 1] >= 0)
                & (obs_idx[:, 1] < dims_tuple[1])
                & (obs_idx[:, 2] >= 0)
                & (obs_idx[:, 2] < dims_tuple[2])
            )
            if np.any(keep):
                obs_idx = obs_idx[keep]
                occupancy_raw[obs_idx[:, 0], obs_idx[:, 1], obs_idx[:, 2]] = True

        # For slender objects like pencils, inflating obstacles by the full object length seals the workspace.
        # Use a thinner occupancy proxy based on the cross-section while keeping exact OBB checks in the validity test.
        size_sorted = np.sort(np.asarray(size, dtype=np.float64).reshape(3))
        is_slender_search = bool(
            size_sorted[2] / max(float(size_sorted[1]), 1e-6) >= float(DEFAULT_SLENDER_OBJECT_ASPECT_RATIO)
        )
        if is_slender_search:
            fast_collision_radius = float(0.5 * np.linalg.norm(size_sorted[:2]) + DEFAULT_COLLISION_MARGIN_M)
        else:
            fast_collision_radius = float(0.5 * np.max(size) + DEFAULT_COLLISION_MARGIN_M)
        inflate_cells = max(0, int(math.ceil(fast_collision_radius / voxel)))
        occupancy_inflated = place_service._inflate_occupancy(occupancy_raw, inflate_cells)

        start_idx = _point_to_idx(start_center)
        goal_idx = _point_to_idx(goal_center)
        override_points: dict[tuple[int, int, int], np.ndarray] = {
            start_idx: start_center.astype(np.float32, copy=False),
            goal_idx: goal_center.astype(np.float32, copy=False),
        }

        rot_sample_count = max(
            2,
            int(math.ceil(total_rotation_rad / max(math.radians(float(DEFAULT_PLACE_TRAJECTORY_ROT_STEP_DEG)), 1e-4))) + 1,
        )
        rot_sample_count = min(int(DEFAULT_PLACE_TRAJECTORY_MAX_ROTATION_SAMPLES), rot_sample_count)
        progress_window = 0.0
        if total_rotation_rad > 1e-5:
            progress_window = float(
                np.clip(
                    math.radians(float(DEFAULT_PLACE_TRAJECTORY_ROT_STEP_DEG)) / max(total_rotation_rad, 1e-6),
                    0.08,
                    0.22,
                )
            )

        exact_valid_cache: dict[tuple[tuple[int, int, int], str], bool] = {}

        def _estimated_progress_from_point(center_point: np.ndarray) -> float:
            point = np.asarray(center_point, dtype=np.float64).reshape(3)
            dist_start = float(np.linalg.norm(point - start_center.astype(np.float64)))
            dist_goal = float(np.linalg.norm(goal_center.astype(np.float64) - point))
            denom = dist_start + dist_goal
            if denom <= 1e-8:
                return 0.0
            return float(np.clip(dist_start / denom, 0.0, 1.0))

        def _rotation_progress_samples(center_point: np.ndarray, mode: str) -> np.ndarray:
            if total_rotation_rad <= 1e-5:
                if mode == "goal":
                    return np.array([1.0], dtype=np.float64)
                return np.array([0.0], dtype=np.float64)
            if mode == "start":
                return np.array([0.0], dtype=np.float64)
            if mode == "goal":
                return np.array([1.0], dtype=np.float64)
            center = np.asarray(center_point, dtype=np.float64).reshape(3)
            est = _estimated_progress_from_point(center)
            values = [est]
            if mode == "search":
                values.extend(
                    [
                        max(0.0, est - progress_window),
                        min(1.0, est + progress_window),
                    ]
                )
            unique = sorted({round(float(v), 4) for v in values})
            return np.asarray(unique, dtype=np.float64)

        def _exact_pose_valid_for_point(center_point: np.ndarray, mode: str) -> bool:
            center_arr = np.asarray(center_point, dtype=np.float32).reshape(3)
            for rot_t in _rotation_progress_samples(center_arr, mode):
                axes = _interpolate_axes_shortest(start_axes, goal_axes, float(rot_t))
                if _pose_collision(center_arr, axes):
                    return False
            return True

        def _is_valid_idx_with_mode(idx: tuple[int, int, int], mode: str) -> bool:
            if not _idx_in_bounds(idx):
                return False
            if idx not in override_points and bool(occupancy_inflated[idx]):
                return False
            cache_key = (idx, str(mode))
            cached = exact_valid_cache.get(cache_key)
            if cached is not None:
                return bool(cached)
            point = override_points.get(idx, _idx_to_point(idx))
            valid = bool(_exact_pose_valid_for_point(point, mode))
            exact_valid_cache[cache_key] = valid
            return valid

        def _search_valid_voxel(
            seed_idx: tuple[int, int, int],
            mode: str,
            error_message: str,
        ) -> tuple[tuple[int, int, int], np.ndarray]:
            if _is_valid_idx_with_mode(seed_idx, mode):
                return seed_idx, override_points.get(seed_idx, _idx_to_point(seed_idx)).astype(np.float32, copy=False)

            search_radius = max(6, int(math.ceil(0.12 / voxel)))
            up_cells = max(2, int(math.ceil((float(DEFAULT_PLACE_TRAJECTORY_Z_LIFT_MARGIN_M) + 0.06) / voxel)))
            for dz in range(1, up_cells + 1):
                nz = min(dims_tuple[2] - 1, seed_idx[2] + dz)
                idx = (seed_idx[0], seed_idx[1], nz)
                if _is_valid_idx_with_mode(idx, mode):
                    return idx, _idx_to_point(idx).astype(np.float32, copy=False)

            from collections import deque

            visited = {seed_idx}
            queue = deque([(seed_idx, 0)])
            neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            while queue:
                cur, dist = queue.popleft()
                if dist >= search_radius:
                    continue
                x, y, z = cur
                for dx, dy, dz in neighbors:
                    nxt = (x + dx, y + dy, z + dz)
                    if nxt in visited or not _idx_in_bounds(nxt):
                        continue
                    visited.add(nxt)
                    if _is_valid_idx_with_mode(nxt, mode):
                        return nxt, _idx_to_point(nxt).astype(np.float32, copy=False)
                    queue.append((nxt, dist + 1))
            raise RuntimeError(error_message)

        planning_start_idx, planning_start_point = _search_valid_voxel(
            start_idx,
            "start",
            "Failed to resolve a collision-free launch voxel near the pick pose.",
        )
        planning_goal_idx, planning_goal_point = _search_valid_voxel(
            goal_idx,
            "goal",
            "Failed to resolve a collision-free goal voxel near the final place pose.",
        )

        def _is_valid_idx(idx: tuple[int, int, int]) -> bool:
            if idx == planning_start_idx:
                return _is_valid_idx_with_mode(idx, "start")
            if idx == planning_goal_idx:
                return _is_valid_idx_with_mode(idx, "goal")
            return _is_valid_idx_with_mode(idx, "search")

        direct_idx_path = place_service._trace_voxel_segment_indices(planning_start_idx, planning_goal_idx, _is_valid_idx)
        astar_stats: dict[str, Any] = {}
        if direct_idx_path is None:
            direct_idx_path, astar_stats = place_service._astar_voxel_path(
                is_valid_fn=_is_valid_idx,
                dims=dims_tuple,
                start_idx=planning_start_idx,
                goal_idx=planning_goal_idx,
                max_iters=int(DEFAULT_PLACE_TRAJECTORY_MAX_ASTAR_ITERS),
            )

        base_search_debug: dict[str, Any] = {
            "search_mode": "direct" if direct_idx_path is not None and astar_stats == {} else ("astar" if direct_idx_path is not None else "manual"),
            "astar": dict(astar_stats),
            "grid_dims": [int(dims_tuple[0]), int(dims_tuple[1]), int(dims_tuple[2])],
            "voxel_size_m": float(voxel),
            "inflate_cells": int(inflate_cells),
            "is_slender_search": bool(is_slender_search),
            "fast_collision_radius_m": float(fast_collision_radius),
            "inflated_occupied_cells": int(np.count_nonzero(occupancy_inflated)),
            "raw_occupied_cells": int(np.count_nonzero(occupancy_raw)),
            "near_collision_points": int(near_pts.shape[0]),
            "collision_scene_points": int(collision_points.shape[0]),
            "rotation_node_samples": int(rot_sample_count),
            "requested_start_idx": [int(start_idx[0]), int(start_idx[1]), int(start_idx[2])],
            "resolved_start_idx": [int(planning_start_idx[0]), int(planning_start_idx[1]), int(planning_start_idx[2])],
            "requested_goal_idx": [int(goal_idx[0]), int(goal_idx[1]), int(goal_idx[2])],
            "resolved_goal_idx": [int(planning_goal_idx[0]), int(planning_goal_idx[1]), int(planning_goal_idx[2])],
        }

        path_variants: list[tuple[str, list[tuple[int, int, int]], dict[str, Any]]] = []
        if direct_idx_path is not None:
            path_variants.append(("raw", list(direct_idx_path), {"enabled": False}))
        if direct_idx_path is not None and smooth_passes > 0 and len(direct_idx_path) >= 3:
            protected_positions = {0, len(direct_idx_path) - 1}
            smoothed_idx_path, protected_positions, shortcut_stats = place_service._shortcut_path_between_protected_points(
                idx_path=list(direct_idx_path),
                protected_positions=protected_positions,
                is_valid_fn=_is_valid_idx,
                num_passes=smooth_passes,
            )
            lift_stats = {
                "enabled": False,
                "passes": 0,
                "changed_points": 0,
                "total_lift_cells": 0,
                "max_lift_cells_applied": 0,
            }
            if smooth_max_lift_cells > 0 and len(smoothed_idx_path) >= 3:
                smoothed_idx_path, lift_stats = place_service._lift_path_for_smoothing(
                    idx_path=smoothed_idx_path,
                    protected_positions=protected_positions,
                    is_valid_fn=_is_valid_idx,
                    idx_to_point=_idx_to_point,
                    max_lift_cells=smooth_max_lift_cells,
                    lift_weight=smooth_lift_weight,
                    num_passes=smooth_passes,
                )
            if bool(shortcut_stats.get("enabled", False) or lift_stats.get("enabled", False)) or list(smoothed_idx_path) != list(direct_idx_path):
                path_variants.insert(
                    0,
                    (
                        "smoothed",
                        list(smoothed_idx_path),
                        {
                            "enabled": bool(shortcut_stats.get("enabled", False) or lift_stats.get("enabled", False)),
                            "shortcut": shortcut_stats,
                            "lift": lift_stats,
                        },
                    ),
                )

        def _compress_idx_path(idx_path: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
            simplified_idx_path = list(idx_path)
            if len(simplified_idx_path) <= 2:
                return simplified_idx_path
            compressed: list[tuple[int, int, int]] = [simplified_idx_path[0]]
            cursor = 0
            while cursor < len(simplified_idx_path) - 1:
                best_next = cursor + 1
                for candidate in range(len(simplified_idx_path) - 1, cursor + 1, -1):
                    traced = place_service._trace_voxel_segment_indices(
                        simplified_idx_path[cursor],
                        simplified_idx_path[candidate],
                        _is_valid_idx,
                    )
                    if traced is not None:
                        best_next = candidate
                        break
                compressed.append(simplified_idx_path[best_next])
                cursor = best_next
            return compressed

        def _build_path_points(
            idx_path: list[tuple[int, int, int]],
            compress_path: bool,
        ) -> list[np.ndarray]:
            if not idx_path:
                return []
            simplified_idx_path = _compress_idx_path(idx_path) if compress_path else list(idx_path)
            points: list[np.ndarray] = [start_center.astype(np.float32, copy=False)]
            if (
                planning_start_idx != start_idx
                and float(np.linalg.norm(planning_start_point.astype(np.float64) - points[-1].astype(np.float64))) > 1e-6
            ):
                points.append(planning_start_point.astype(np.float32, copy=False))
            if len(simplified_idx_path) > 2:
                points.extend(_idx_to_point(idx) for idx in simplified_idx_path[1:-1])
            if (
                planning_goal_idx != goal_idx
                and float(np.linalg.norm(planning_goal_point.astype(np.float64) - points[-1].astype(np.float64))) > 1e-6
            ):
                points.append(planning_goal_point.astype(np.float32, copy=False))
            if float(np.linalg.norm(goal_center.astype(np.float64) - points[-1].astype(np.float64))) > 1e-6:
                points.append(goal_center.astype(np.float32, copy=False))
            if len(points) == 1:
                points.append(goal_center.astype(np.float32, copy=False))
            return [np.asarray(p, dtype=np.float32).reshape(3) for p in points]

        def _build_lift_translate_waypoints() -> list[tuple[str, list[np.ndarray], dict[str, Any]]]:
            lift_floor = max(
                float(max(start_center[2], goal_center[2]) + 0.08),
                float(obstacle_top_z + 0.5 * float(size[2]) + 0.03),
            )
            lift_z = float(min(float(upper[2] - 0.5 * voxel), lift_floor))
            if not np.isfinite(lift_z) or lift_z <= float(max(start_center[2], goal_center[2]) + 0.01):
                return []

            def _append_if_far(out: list[np.ndarray], point: np.ndarray) -> None:
                point_arr = np.asarray(point, dtype=np.float32).reshape(3)
                if not out:
                    out.append(point_arr)
                    return
                if float(np.linalg.norm(point_arr.astype(np.float64) - out[-1].astype(np.float64))) > 1e-6:
                    out.append(point_arr)

            variants: list[tuple[str, list[np.ndarray], dict[str, Any]]] = []
            for variant_label, anchor_start, anchor_goal in (
                ("lift-direct", start_center, goal_center),
                ("lift-resolved", planning_start_point, planning_goal_point),
            ):
                waypoints: list[np.ndarray] = []
                _append_if_far(waypoints, start_center)
                lifted_start = np.asarray([float(anchor_start[0]), float(anchor_start[1]), lift_z], dtype=np.float32)
                lifted_goal = np.asarray([float(anchor_goal[0]), float(anchor_goal[1]), lift_z], dtype=np.float32)
                _append_if_far(waypoints, lifted_start)
                _append_if_far(waypoints, lifted_goal)
                _append_if_far(waypoints, goal_center)
                if len(waypoints) >= 2:
                    variants.append(
                        (
                            variant_label,
                            waypoints,
                            {"enabled": True, "strategy": "lift_translate", "lift_z_m": float(lift_z)},
                        )
                    )
            return variants

        def _build_container_entry_waypoints() -> list[tuple[str, list[np.ndarray], dict[str, Any]]]:
            if target_profile is None or str(placement_rule_id or "").strip() not in {"top_open_box", "front_open_cabinet"}:
                return []
            open_spec = _opening_face_axis_spec(target_profile.opening_face)
            if open_spec is None or target_profile.opening_normal_base is None:
                return []
            opening_normal = np.asarray(target_profile.opening_normal_base, dtype=np.float64).reshape(3)
            opening_norm = float(np.linalg.norm(opening_normal))
            if not np.isfinite(opening_norm) or opening_norm <= 1e-8:
                opening_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                opening_normal = opening_normal / opening_norm
            axis_idx, sign = open_spec
            goal_corners_world = _box_corners_axes(goal_center, goal_axes, size).astype(np.float64, copy=False)
            goal_corners_local = target_profile.world_to_local(goal_corners_world).astype(np.float64, copy=False)
            opening_plane = 0.5 * float(target_profile.size_xyz[axis_idx])
            signed_coords = sign * goal_corners_local[:, axis_idx]
            insertion_clearance = float(max(0.018, 3.0 * DEFAULT_COLLISION_MARGIN_M))
            entry_shift_m = float(max(0.0, opening_plane + insertion_clearance - float(np.min(signed_coords))))
            if not np.isfinite(entry_shift_m):
                return []
            entry_center = (
                goal_center.astype(np.float64)
                + opening_normal.reshape(3) * float(entry_shift_m)
            ).astype(np.float32, copy=False)
            safe_z = float(
                max(
                    start_center[2],
                    goal_center[2],
                    entry_center[2],
                    obstacle_top_z + 0.5 * float(size[2]) + 0.05,
                )
            )

            def _append_if_far(out: list[np.ndarray], point: np.ndarray) -> None:
                point_arr = np.asarray(point, dtype=np.float32).reshape(3)
                if not out or float(np.linalg.norm(point_arr.astype(np.float64) - out[-1].astype(np.float64))) > 1e-6:
                    out.append(point_arr)

            waypoints: list[np.ndarray] = []
            _append_if_far(waypoints, start_center)
            _append_if_far(waypoints, np.asarray([float(start_center[0]), float(start_center[1]), safe_z], dtype=np.float32))
            _append_if_far(waypoints, np.asarray([float(entry_center[0]), float(entry_center[1]), safe_z], dtype=np.float32))
            _append_if_far(waypoints, entry_center)
            _append_if_far(waypoints, goal_center)
            if len(waypoints) < 2:
                return []
            return [
                (
                    "container-entry",
                    waypoints,
                    {
                        "enabled": True,
                        "strategy": "container_entry",
                        "entry_shift_m": float(entry_shift_m),
                        "safe_z_m": float(safe_z),
                        "finish_rotation_before_last_segment": True,
                        "cost_bias_m": -0.10,
                    },
                )
            ]

        def _rotation_schedules(
            path_length: float,
            rotate_start_dist: float,
            rotate_end_dist: float,
            prefix_lengths: list[float],
            waypoints: list[np.ndarray],
        ) -> list[tuple[str, float, float, float]]:
            if total_rotation_rad <= 1e-5 or path_length <= 1e-8:
                return [("fixed", 0.0, float(path_length), 0.0)]
            start_dist = float(np.clip(rotate_start_dist, 0.0, path_length))
            end_dist = float(np.clip(rotate_end_dist, start_dist + 1e-6, path_length))
            base_span = max(end_dist - start_dist, 1e-6)
            schedules: list[tuple[str, float, float, float]] = []
            seen: set[tuple[int, int]] = set()

            def _add_schedule(label: str, start_m: float, end_m: float, penalty: float) -> None:
                s = float(np.clip(start_m, 0.0, path_length))
                e = float(np.clip(end_m, s + 1e-6, path_length))
                key = (int(round(s * 1000.0)), int(round(e * 1000.0)))
                if key in seen:
                    return
                seen.add(key)
                schedules.append((label, s, e, float(penalty)))

            _add_schedule("uniform", start_dist, end_dist, 0.0)
            require_progressive_rotation = bool(
                gripper_rotation_total_rad >= math.radians(75.0)
                or grasp_local_rotation_delta_rad >= math.radians(60.0)
            )
            if require_progressive_rotation:
                middle_span = max(0.12, 0.62 * base_span)
                middle_center = 0.5 * (start_dist + end_dist)
                _add_schedule("middle", middle_center - 0.5 * middle_span, middle_center + 0.5 * middle_span, 0.006)
                early_span = max(0.12, 0.56 * base_span)
                _add_schedule("early", start_dist, min(end_dist, start_dist + early_span), 0.010)
            else:
                late_span = max(0.08, 0.40 * base_span)
                _add_schedule("late", max(start_dist, end_dist - late_span), end_dist, 0.008)
                very_late_span = max(0.06, 0.24 * base_span)
                _add_schedule("very-late", max(start_dist, end_dist - very_late_span), end_dist, 0.012)
                early_span = max(0.08, 0.36 * base_span)
                _add_schedule("early", start_dist, min(end_dist, start_dist + early_span), 0.016)
                middle_span = max(0.08, 0.45 * base_span)
                middle_center = 0.5 * (start_dist + end_dist)
                _add_schedule("middle", middle_center - 0.5 * middle_span, middle_center + 0.5 * middle_span, 0.020)
            if len(waypoints) >= 3 and not require_progressive_rotation:
                peak_idx = int(np.argmax([float(point[2]) for point in waypoints]))
                if 0 < peak_idx < len(waypoints) - 1:
                    peak_dist = float(prefix_lengths[peak_idx])
                    peak_span = max(0.08, 0.35 * base_span)
                    _add_schedule("peak-z", peak_dist - 0.5 * peak_span, peak_dist + 0.5 * peak_span, 0.010)
            return schedules

        def _densify_candidate(
            variant_label: str,
            idx_path: list[tuple[int, int, int]],
            waypoints: list[np.ndarray],
            waypoint_mode: str,
            smooth_stats: dict[str, Any],
        ) -> dict[str, Any] | None:
            if len(waypoints) < 2:
                return None
            seg_lengths = [
                float(np.linalg.norm(waypoints[i + 1].astype(np.float64) - waypoints[i].astype(np.float64)))
                for i in range(len(waypoints) - 1)
            ]
            path_length = float(sum(seg_lengths))
            if path_length <= 1e-8:
                return None
            prefix_lengths = [0.0]
            for seg_len in seg_lengths:
                prefix_lengths.append(prefix_lengths[-1] + seg_len)
            rotate_start_dist = 0.0
            rotate_end_dist = float(path_length)
            if planning_start_idx != start_idx and len(waypoints) >= 3:
                rotate_start_dist = float(prefix_lengths[1])
            if planning_goal_idx != goal_idx and len(waypoints) >= 3:
                rotate_end_dist = float(prefix_lengths[max(1, len(waypoints) - 2)])
            if bool(smooth_stats.get("finish_rotation_before_last_segment", False)) and len(waypoints) >= 3:
                rotate_end_dist = min(rotate_end_dist, float(prefix_lengths[-2]))
            rotate_end_dist = float(np.clip(rotate_end_dist, rotate_start_dist + 1e-6, path_length))
            best_schedule_candidate: dict[str, Any] | None = None
            best_schedule_cost = float("inf")
            for schedule_label, schedule_start_m, schedule_end_m, schedule_penalty in _rotation_schedules(
                path_length,
                rotate_start_dist,
                rotate_end_dist,
                prefix_lengths,
                waypoints,
            ):
                rotate_span = max(schedule_end_m - schedule_start_m, 1e-8)
                object_centers: list[np.ndarray] = []
                object_axes_traj: list[np.ndarray] = []
                gripper_origins: list[np.ndarray] = []
                gripper_axes_traj: list[np.ndarray] = []
                schedule_valid = True
                rot_step_rad = max(math.radians(float(DEFAULT_PLACE_TRAJECTORY_ROT_STEP_DEG)), 1e-4)

                def _scheduled_rotation_progress(traveled_m: float) -> float:
                    if traveled_m <= schedule_start_m + 1e-8:
                        return 0.0
                    if traveled_m >= schedule_end_m - 1e-8:
                        return 1.0
                    return float(np.clip((traveled_m - schedule_start_m) / rotate_span, 0.0, 1.0))

                for seg_idx, seg_len in enumerate(seg_lengths):
                    seg_start_dist = float(prefix_lengths[seg_idx])
                    seg_end_dist = float(prefix_lengths[seg_idx + 1])
                    rot_t0 = _scheduled_rotation_progress(seg_start_dist)
                    rot_t1 = _scheduled_rotation_progress(seg_end_dist)
                    seg_object_axes_0 = _interpolate_axes_shortest(start_axes, goal_axes, rot_t0)
                    seg_object_axes_1 = _interpolate_axes_shortest(start_axes, goal_axes, rot_t1)
                    seg_grasp_local_axes_0 = _interpolate_axes_shortest(start_grasp_axes_local, goal_grasp_axes_local, rot_t0)
                    seg_grasp_local_axes_1 = _interpolate_axes_shortest(start_grasp_axes_local, goal_grasp_axes_local, rot_t1)
                    seg_gripper_axes_0 = _orthonormalize_axes(seg_object_axes_0 @ seg_grasp_local_axes_0)
                    seg_gripper_axes_1 = _orthonormalize_axes(seg_object_axes_1 @ seg_grasp_local_axes_1)
                    segment_rotation_rad = max(
                        _rotation_geodesic_angle(seg_object_axes_0, seg_object_axes_1),
                        _rotation_geodesic_angle(seg_grasp_local_axes_0, seg_grasp_local_axes_1),
                        _rotation_geodesic_angle(seg_gripper_axes_0, seg_gripper_axes_1),
                    )
                    step_count = max(
                        2,
                        int(math.ceil(seg_len / max(transition_step_m, 1e-4))) + 1,
                        int(math.ceil(segment_rotation_rad / rot_step_rad)) + 1,
                    )
                    p0 = waypoints[seg_idx].astype(np.float64)
                    p1 = waypoints[seg_idx + 1].astype(np.float64)
                    for sample_idx, alpha in enumerate(np.linspace(0.0, 1.0, step_count, dtype=np.float64)):
                        if seg_idx > 0 and sample_idx == 0:
                            continue
                        center = (p0 * (1.0 - float(alpha)) + p1 * float(alpha)).astype(np.float32, copy=False)
                        traveled = float(prefix_lengths[seg_idx] + float(alpha) * seg_len)
                        if traveled <= schedule_start_m + 1e-8:
                            rot_t = 0.0
                        elif traveled >= schedule_end_m - 1e-8:
                            rot_t = 1.0
                        else:
                            rot_t = float(np.clip((traveled - schedule_start_m) / rotate_span, 0.0, 1.0))
                        axes = _interpolate_axes_shortest(start_axes, goal_axes, rot_t)
                        if _pose_collision(center, axes):
                            schedule_valid = False
                            break
                        grasp_origin_local = (
                            (1.0 - float(rot_t)) * start_grasp_origin_local.astype(np.float64)
                            + float(rot_t) * goal_grasp_origin_local.astype(np.float64)
                        ).astype(np.float32, copy=False)
                        grasp_axes_local = _interpolate_axes_shortest(start_grasp_axes_local, goal_grasp_axes_local, rot_t)
                        grasp_origin, grasp_axes = self._compose_grasp_pose_from_object_pose(
                            center,
                            axes,
                            grasp_origin_local=grasp_origin_local,
                            grasp_axes_local=grasp_axes_local,
                        )
                        object_centers.append(center.astype(np.float32, copy=False))
                        object_axes_traj.append(np.asarray(axes, dtype=np.float32))
                        gripper_origins.append(np.asarray(grasp_origin, dtype=np.float32))
                        gripper_axes_traj.append(np.asarray(grasp_axes, dtype=np.float32))
                    if not schedule_valid:
                        break

                if not schedule_valid:
                    continue

                cost = float(path_length + rotation_weight * total_rotation_rad + schedule_penalty + float(smooth_stats.get("cost_bias_m", 0.0)))
                candidate = {
                    "object_center_trajectory_base": np.asarray(object_centers, dtype=np.float32),
                    "object_axes_trajectory_base": np.asarray(object_axes_traj, dtype=np.float32),
                    "gripper_origin_trajectory_base": np.asarray(gripper_origins, dtype=np.float32),
                    "gripper_axes_trajectory_base": np.asarray(gripper_axes_traj, dtype=np.float32),
                    "debug": {
                        "mode": f"{base_search_debug['search_mode']}-{variant_label}",
                        "search": base_search_debug,
                        "path_smoothing": smooth_stats,
                        "idx_path_points": int(len(idx_path)),
                        "waypoint_mode": str(waypoint_mode),
                        "waypoints": int(len(waypoints)),
                        "trajectory_points": int(len(object_centers)),
                        "object_path_length_m": float(path_length),
                        "path_cost_m": float(cost),
                        "rotation_total_deg": float(math.degrees(total_rotation_rad)),
                        "grasp_local_rotation_total_deg": float(math.degrees(grasp_local_rotation_delta_rad)),
                        "gripper_rotation_total_deg": float(math.degrees(gripper_rotation_total_rad)),
                        "exact_valid_cache_size": int(len(exact_valid_cache)),
                        "rotation_schedule": str(schedule_label),
                        "rotation_schedule_start_m": float(schedule_start_m),
                        "rotation_schedule_end_m": float(schedule_end_m),
                        "start_gripper_x_base_z": float(start_gripper_axes[2, 0]),
                        "goal_gripper_x_base_z": float(goal_gripper_axes[2, 0]),
                    },
                }
                if cost < best_schedule_cost:
                    best_schedule_cost = cost
                    best_schedule_candidate = candidate

            return best_schedule_candidate

        waypoint_variants: list[tuple[str, list[tuple[int, int, int]], list[np.ndarray], str, dict[str, Any]]] = []
        waypoint_signatures: set[tuple[tuple[int, int, int], ...]] = set()

        def _register_waypoint_variant(
            variant_label: str,
            idx_path: list[tuple[int, int, int]],
            waypoints: list[np.ndarray],
            waypoint_mode: str,
            smooth_stats: dict[str, Any],
        ) -> None:
            if len(waypoints) < 2:
                return
            signature = tuple(
                tuple(int(round(float(coord) * 1000.0)) for coord in np.asarray(point, dtype=np.float64).reshape(3))
                for point in waypoints
            )
            if signature in waypoint_signatures:
                return
            waypoint_signatures.add(signature)
            waypoint_variants.append((variant_label, list(idx_path), waypoints, waypoint_mode, smooth_stats))

        for variant_label, idx_path, smooth_stats in path_variants:
            _register_waypoint_variant(
                variant_label,
                idx_path,
                _build_path_points(idx_path, compress_path=True),
                "compressed",
                smooth_stats,
            )
            _register_waypoint_variant(
                variant_label,
                idx_path,
                _build_path_points(idx_path, compress_path=False),
                "full",
                smooth_stats,
            )
        for variant_label, waypoints, variant_debug in _build_lift_translate_waypoints():
            _register_waypoint_variant(
                variant_label,
                [],
                [np.asarray(point, dtype=np.float32).reshape(3) for point in waypoints],
                "manual-lift",
                variant_debug,
            )
        for variant_label, waypoints, variant_debug in _build_container_entry_waypoints():
            _register_waypoint_variant(
                variant_label,
                [],
                [np.asarray(point, dtype=np.float32).reshape(3) for point in waypoints],
                "container-entry",
                variant_debug,
            )

        best_candidate: dict[str, Any] | None = None
        best_cost = float("inf")
        for variant_label, idx_path, waypoints, waypoint_mode, smooth_stats in waypoint_variants:
            candidate = _densify_candidate(variant_label, idx_path, waypoints, waypoint_mode, smooth_stats)
            if candidate is None:
                continue
            cost = float(candidate["debug"].get("path_cost_m", float("inf")))
            if cost < best_cost:
                best_cost = cost
                best_candidate = candidate

        if best_candidate is None:
            if direct_idx_path is None:
                raise RuntimeError(
                    f"Failed to find a collision-free pick-to-place grasp trajectory (termination={astar_stats.get('termination')})."
                )
            raise RuntimeError("Failed to densify a collision-free pick-to-place grasp trajectory.")
        return best_candidate

    def _release_box_pose(self) -> None:
        if self.object_state is None:
            return
        plane_model = self.object_state.plane_model
        center = self.object_state.center_base.astype(np.float32, copy=True)
        center = self._resolve_box_penetration(center, self.object_state.rpy_rad, self.object_state.size_xyz, plane_model)
        center = self._drop_box_along_gravity(center, self.object_state.rpy_rad, self.object_state.size_xyz, plane_model)
        center = self._resolve_box_penetration(center, self.object_state.rpy_rad, self.object_state.size_xyz, plane_model)
        self.object_state.center_base = center.astype(np.float32, copy=False)
        self._render_viewer()
        self._update_info_panel()
        self.status_var.set("Box released and settled against the table / point cloud.")
        self._focus_main_view()

    def _render_left_image(self) -> None:
        if self.left_image_full is None:
            return
        image = self.left_image_full.copy()
        if self.table_cache is not None and self.enable_table_completion_var.get():
            image = _overlay_mask(image, self.table_cache.mask, (64, 160, 255), 70)
        if self.object_state is not None:
            image = _overlay_mask(image, self.object_state.mask, (110, 235, 110), 90)
        if self.place_target_state is not None:
            image = _overlay_mask(image, self.place_target_state.mask, (255, 176, 64), 80)

        display = image.copy()
        if display.width > MAX_IMAGE_DISPLAY_WIDTH or display.height > MAX_IMAGE_DISPLAY_HEIGHT:
            display.thumbnail((MAX_IMAGE_DISPLAY_WIDTH, MAX_IMAGE_DISPLAY_HEIGHT), Image.Resampling.LANCZOS)

        self.left_display_image = display
        self.left_display_scale_x = float(display.width) / float(image.width) if image.width > 0 else 1.0
        self.left_display_scale_y = float(display.height) / float(image.height) if image.height > 0 else 1.0

        self.left_canvas.config(width=display.width, height=display.height)
        self.left_display_photo = ImageTk.PhotoImage(display)
        if self.left_canvas_image_id is None:
            self.left_canvas_image_id = self.left_canvas.create_image(0, 0, anchor="nw", image=self.left_display_photo)
        else:
            self.left_canvas.itemconfig(self.left_canvas_image_id, image=self.left_display_photo)
        self._redraw_left_bboxes()

    def _to_display_bbox(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return (
            int(round(float(bbox[0]) * self.left_display_scale_x)),
            int(round(float(bbox[1]) * self.left_display_scale_y)),
            int(round(float(bbox[2]) * self.left_display_scale_x)),
            int(round(float(bbox[3]) * self.left_display_scale_y)),
        )

    def _redraw_left_bboxes(self) -> None:
        for item_id in (self.left_bbox_item_id, self.left_preview_item_id):
            if item_id is not None:
                self.left_canvas.delete(item_id)
        self.left_bbox_item_id = None
        self.left_preview_item_id = None

        if self.selected_bbox is not None:
            x1, y1, x2, y2 = self._to_display_bbox(self.selected_bbox)
            self.left_bbox_item_id = self.left_canvas.create_rectangle(
                x1, y1, x2, y2, outline="#ff4d4d", width=2
            )
        if self.preview_bbox is not None:
            x1, y1, x2, y2 = self._to_display_bbox(self.preview_bbox)
            self.left_preview_item_id = self.left_canvas.create_rectangle(
                x1, y1, x2, y2, outline="#ffd166", width=2, dash=(4, 2)
            )

    def _display_to_image_xy(self, x_disp: float, y_disp: float) -> tuple[int, int]:
        if self.left_image_full is None or self.left_display_image is None:
            return 0, 0
        x = int(round(float(x_disp) / max(self.left_display_scale_x, 1e-6)))
        y = int(round(float(y_disp) / max(self.left_display_scale_y, 1e-6)))
        x = max(0, min(x, self.left_image_full.width - 1))
        y = max(0, min(y, self.left_image_full.height - 1))
        return x, y

    def _on_left_canvas_press(self, event: tk.Event) -> None:
        if self.left_image_full is None or self.busy:
            return
        if self.awaiting_grasp_click:
            x_img, y_img = self._display_to_image_xy(event.x, event.y)
            if self._set_grasp_point_from_pixel(x_img, y_img):
                return
            self.awaiting_grasp_click = False
            self.status_var.set("Failed to set grasp point from the clicked image location.")
            self._focus_main_view()
            return
        self._bbox_drag_start = self._display_to_image_xy(event.x, event.y)
        self.preview_bbox = None

    def _on_left_canvas_drag(self, event: tk.Event) -> None:
        if self._bbox_drag_start is None or self.left_image_full is None:
            return
        x0, y0 = self._bbox_drag_start
        x1, y1 = self._display_to_image_xy(event.x, event.y)
        self.preview_bbox = _clip_bbox((x0, y0, x1, y1), self.left_image_full.width, self.left_image_full.height)
        self._redraw_left_bboxes()

    def _on_left_canvas_release(self, event: tk.Event) -> None:
        if self._bbox_drag_start is None or self.left_image_full is None:
            return
        x0, y0 = self._bbox_drag_start
        x1, y1 = self._display_to_image_xy(event.x, event.y)
        bbox = _clip_bbox((x0, y0, x1, y1), self.left_image_full.width, self.left_image_full.height)
        self._bbox_drag_start = None
        self.preview_bbox = None
        if (bbox[2] - bbox[0] + 1) < 4 or (bbox[3] - bbox[1] + 1) < 4:
            self.selected_bbox = None
            self._redraw_left_bboxes()
            self.status_var.set("BBox too small.")
            self._update_info_panel()
            return
        self.selected_bbox = bbox
        self._redraw_left_bboxes()
        self._update_info_panel()
        self.status_var.set(f"BBox set to {bbox}.")

    def _on_viewer_press(self, event: tk.Event) -> None:
        self.viewer_drag_last = (int(event.x), int(event.y))

    def _on_viewer_drag(self, event: tk.Event) -> None:
        if self.viewer_drag_last is None:
            return
        last_x, last_y = self.viewer_drag_last
        dx = float(event.x - last_x)
        dy = float(event.y - last_y)
        self.viewer_drag_last = (int(event.x), int(event.y))
        self.viewer_state.yaw_deg = float(self.viewer_state.yaw_deg + 0.35 * dx)
        self.viewer_state.pitch_deg = float(np.clip(self.viewer_state.pitch_deg - 0.35 * dy, -89.0, 89.0))
        self._render_viewer()

    def _on_viewer_release(self, _event: tk.Event) -> None:
        self.viewer_drag_last = None

    def _on_viewer_wheel(self, event: tk.Event) -> None:
        delta = 0.0
        if hasattr(event, "delta") and event.delta:
            delta = float(event.delta)
        elif getattr(event, "num", None) == 4:
            delta = 120.0
        elif getattr(event, "num", None) == 5:
            delta = -120.0
        if delta == 0.0:
            return
        scale = 1.10 if delta > 0.0 else 1.0 / 1.10
        self.viewer_state.zoom = float(np.clip(self.viewer_state.zoom * scale, 0.35, 6.0))
        self._render_viewer()

    def _build_scene_render(self, width: int, height: int) -> Image.Image:
        canvas_w = max(64, int(width))
        canvas_h = max(64, int(height))
        if self.scene_points_base.shape[0] == 0 and self.object_state is None and self.place_plan_state is None:
            return _build_placeholder_view(canvas_w, canvas_h, "Run S2M2 to load the scene.")

        scene_points = self.scene_points_base
        scene_colors = self.scene_colors
        max_points = self._parse_max_points()
        if scene_points.shape[0] > 0 and max_points > 0:
            idx = _subsample_indices(scene_points.shape[0], max_points, seed=3)
            scene_points = scene_points[idx]
            scene_colors = scene_colors[idx]

        extents_list: list[np.ndarray] = []
        if scene_points.shape[0] > 0:
            extents_list.append(scene_points)
        if self.place_target_state is not None and self.place_target_state.points_base.shape[0] > 0:
            extents_list.append(self.place_target_state.points_base.astype(np.float32, copy=False))
            if bool(self.show_place_target_fit_var.get()):
                profile = self.place_target_state.target_profile
                if profile.primitive_shape == "cylinder" and profile.primitive_radius is not None:
                    cyl_top, cyl_bottom = _cylinder_wire_points_axes(
                        profile.center_base,
                        profile.rotation_base,
                        float(profile.primitive_radius),
                        float(profile.size_xyz[2]),
                    )
                    extents_list.append(cyl_top)
                    extents_list.append(cyl_bottom)
                else:
                    extents_list.append(
                        _box_corners_axes(profile.center_base, profile.rotation_base, profile.size_xyz)
                    )
        if self.object_state is not None:
            extents_list.append(_box_corners_world(self.object_state.center_base, self.object_state.rpy_rad, self.object_state.size_xyz))
            if bool(self.show_object_points_var.get()) and self.object_state.points_base.shape[0] > 0:
                extents_list.append(self.object_state.points_base.astype(np.float32, copy=False))
            if bool(self.show_object_fit_points_var.get()) and self.object_state.fit_points_base.shape[0] > 0:
                extents_list.append(self.object_state.fit_points_base.astype(np.float32, copy=False))
        if self.grasp_state is not None and self.object_state is not None:
            obj_rot = _rpy_to_matrix(self.object_state.rpy_rad).astype(np.float64, copy=False)
            grasp_origin = (np.asarray(self.grasp_state.point_obj, dtype=np.float64).reshape((1, 3)) @ obj_rot.T
                            + np.asarray(self.object_state.center_base, dtype=np.float64).reshape((1, 3)))
            extents_list.append(grasp_origin.astype(np.float32, copy=False))
        if self.place_plan_state is not None:
            extents_list.append(_box_corners_axes(self.place_plan_state.object_center_base, self.place_plan_state.object_axes_base, self.object_state.size_xyz if self.object_state is not None else np.ones((3,), dtype=np.float32)))
            extents_list.append(np.asarray(self.place_plan_state.gripper_origin_base, dtype=np.float32).reshape((1, 3)))
            if self.place_plan_state.object_center_trajectory_base.shape[0] > 0:
                extents_list.append(self.place_plan_state.object_center_trajectory_base.astype(np.float32, copy=False))
            if self.place_plan_state.gripper_origin_trajectory_base.shape[0] > 0:
                extents_list.append(self.place_plan_state.gripper_origin_trajectory_base.astype(np.float32, copy=False))
        if not extents_list:
            return _build_placeholder_view(canvas_w, canvas_h, "No renderable points.")

        world_all = np.vstack(extents_list).astype(np.float32, copy=False)
        target = np.median(world_all, axis=0).astype(np.float64, copy=False)

        if self.camera_bundle is not None:
            rot_cb = np.asarray(self.camera_bundle.rot_cb, dtype=np.float64)
            if rot_cb.shape == (3, 3) and np.all(np.isfinite(rot_cb)):
                base_view_rot = np.stack(
                    [
                        rot_cb[:, 0],
                        rot_cb[:, 2],
                        -rot_cb[:, 1],
                    ],
                    axis=0,
                ).astype(np.float64, copy=False)
            else:
                base_view_rot = np.eye(3, dtype=np.float64)
        else:
            base_view_rot = np.eye(3, dtype=np.float64)

        yaw = math.radians(float(self.viewer_state.yaw_deg))
        pitch = math.radians(float(self.viewer_state.pitch_deg))
        rz = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(pitch), -math.sin(pitch)],
                [0.0, math.sin(pitch), math.cos(pitch)],
            ],
            dtype=np.float64,
        )
        view_rot = (rx @ rz) @ base_view_rot

        def _project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            centered = np.asarray(points, dtype=np.float64) - target.reshape((1, 3))
            view = centered @ view_rot.T
            sx = view[:, 0]
            sy = -view[:, 2]
            depth = view[:, 1]
            return sx, sy, depth

        proj_all_x, proj_all_y, _ = _project(world_all)
        span_x = float(np.ptp(proj_all_x)) if proj_all_x.size > 0 else 1.0
        span_y = float(np.ptp(proj_all_y)) if proj_all_y.size > 0 else 1.0
        span_x = max(span_x, 1e-3)
        span_y = max(span_y, 1e-3)
        fit_scale = 0.90 * min((canvas_w - 32) / span_x, (canvas_h - 32) / span_y)
        scale = float(max(1e-3, fit_scale * float(self.viewer_state.zoom)))

        img_bgr = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        img_bgr[..., 0] = int(DEFAULT_RENDER_BG_RGB[2])
        img_bgr[..., 1] = int(DEFAULT_RENDER_BG_RGB[1])
        img_bgr[..., 2] = int(DEFAULT_RENDER_BG_RGB[0])

        def _draw_point_cloud(points: np.ndarray, colors_rgb: np.ndarray) -> None:
            pts = np.asarray(points, dtype=np.float32)
            cols = np.asarray(colors_rgb, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
                return
            if cols.ndim != 2 or cols.shape[0] != pts.shape[0]:
                cols = np.tile(np.array([[0.8, 0.8, 0.8]], dtype=np.float32), (pts.shape[0], 1))
            sx, sy, depth = _project(pts)
            x_pix = np.rint(canvas_w * 0.5 + sx * scale).astype(np.int32)
            y_pix = np.rint(canvas_h * 0.5 + sy * scale).astype(np.int32)
            colors_bgr = np.clip(cols[:, ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)
            order = np.argsort(depth)
            for idx in order:
                x = int(x_pix[idx])
                y = int(y_pix[idx])
                if x < 0 or x >= canvas_w or y < 0 or y >= canvas_h:
                    continue
                img_bgr[y, x] = colors_bgr[idx]

        def _draw_box(center: np.ndarray, axes_world: np.ndarray, size_xyz: np.ndarray, edge_bgr: tuple[int, int, int]) -> None:
            corners = _box_corners_axes(center, axes_world, size_xyz)
            sx, sy, _ = _project(corners)
            x_pix = np.rint(canvas_w * 0.5 + sx * scale).astype(np.int32)
            y_pix = np.rint(canvas_h * 0.5 + sy * scale).astype(np.int32)
            corners_px = np.stack([x_pix, y_pix], axis=1)
            edges = (
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            )
            for a, b in edges:
                cv2.line(
                    img_bgr,
                    tuple(int(v) for v in corners_px[a]),
                    tuple(int(v) for v in corners_px[b]),
                    edge_bgr,
                    2,
                    cv2.LINE_AA,
                )

        def _draw_cylinder(center: np.ndarray, axes_world: np.ndarray, radius: float, height_xyz: float, edge_bgr: tuple[int, int, int]) -> None:
            top, bottom = _cylinder_wire_points_axes(center, axes_world, radius, height_xyz)
            sx_top, sy_top, _ = _project(top)
            sx_bottom, sy_bottom, _ = _project(bottom)
            top_px = np.stack(
                [
                    np.rint(canvas_w * 0.5 + sx_top * scale).astype(np.int32),
                    np.rint(canvas_h * 0.5 + sy_top * scale).astype(np.int32),
                ],
                axis=1,
            )
            bottom_px = np.stack(
                [
                    np.rint(canvas_w * 0.5 + sx_bottom * scale).astype(np.int32),
                    np.rint(canvas_h * 0.5 + sy_bottom * scale).astype(np.int32),
                ],
                axis=1,
            )
            count = int(top_px.shape[0])
            for idx in range(count):
                j = (idx + 1) % count
                cv2.line(img_bgr, tuple(int(v) for v in top_px[idx]), tuple(int(v) for v in top_px[j]), edge_bgr, 2, cv2.LINE_AA)
                cv2.line(img_bgr, tuple(int(v) for v in bottom_px[idx]), tuple(int(v) for v in bottom_px[j]), edge_bgr, 2, cv2.LINE_AA)
            for idx in range(0, count, max(1, count // 6)):
                cv2.line(
                    img_bgr,
                    tuple(int(v) for v in top_px[idx]),
                    tuple(int(v) for v in bottom_px[idx]),
                    edge_bgr,
                    2,
                    cv2.LINE_AA,
                )

        def _draw_frame(origin: np.ndarray, axes_world: np.ndarray, axis_len: float, axis_colors_bgr: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> None:
            o = np.asarray(origin, dtype=np.float64).reshape(1, 3)
            axes = np.asarray(axes_world, dtype=np.float64).reshape(3, 3)
            pts = np.vstack(
                [
                    o,
                    o + axes[:, 0].reshape((1, 3)) * float(axis_len),
                    o + axes[:, 1].reshape((1, 3)) * float(axis_len),
                    o + axes[:, 2].reshape((1, 3)) * float(axis_len),
                ]
            ).astype(np.float32)
            ax_x, ax_y, _ = _project(pts)
            axis_px = np.stack(
                [
                    np.rint(canvas_w * 0.5 + ax_x * scale).astype(np.int32),
                    np.rint(canvas_h * 0.5 + ax_y * scale).astype(np.int32),
                ],
                axis=1,
            )
            origin_px = tuple(int(v) for v in axis_px[0])
            cv2.line(img_bgr, origin_px, tuple(int(v) for v in axis_px[1]), axis_colors_bgr[0], 2, cv2.LINE_AA)
            cv2.line(img_bgr, origin_px, tuple(int(v) for v in axis_px[2]), axis_colors_bgr[1], 2, cv2.LINE_AA)
            cv2.line(img_bgr, origin_px, tuple(int(v) for v in axis_px[3]), axis_colors_bgr[2], 2, cv2.LINE_AA)

        def _draw_polyline(points_world: np.ndarray, color_bgr: tuple[int, int, int], thickness: int = 1) -> None:
            pts = np.asarray(points_world, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
                return
            sx, sy, _ = _project(pts.astype(np.float32, copy=False))
            px = np.stack(
                [
                    np.rint(canvas_w * 0.5 + sx * scale).astype(np.int32),
                    np.rint(canvas_h * 0.5 + sy * scale).astype(np.int32),
                ],
                axis=1,
            )
            for idx in range(px.shape[0] - 1):
                cv2.line(
                    img_bgr,
                    tuple(int(v) for v in px[idx]),
                    tuple(int(v) for v in px[idx + 1]),
                    color_bgr,
                    max(1, int(thickness)),
                    cv2.LINE_AA,
                )

        def _draw_text_label(point_world: np.ndarray, text: str, state: str) -> None:
            point = np.asarray(point_world, dtype=np.float64).reshape(1, 3)
            sx, sy, _ = _project(point)
            x = int(round(canvas_w * 0.5 + float(sx[0]) * scale))
            y = int(round(canvas_h * 0.5 + float(sy[0]) * scale))
            if x < -80 or x >= canvas_w + 80 or y < -30 or y >= canvas_h + 30:
                return
            state_key = str(state).strip().lower()
            color_bgr = {
                "open": (64, 220, 96),
                "closed": (96, 118, 255),
                "support": (255, 220, 96),
            }.get(state_key, (180, 180, 180))
            anchor = (x + 7, y - 5)
            cv2.circle(img_bgr, (x, y), 4, color_bgr, -1, cv2.LINE_AA)
            cv2.putText(img_bgr, str(text), anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (16, 20, 28), 3, cv2.LINE_AA)
            cv2.putText(img_bgr, str(text), anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1, cv2.LINE_AA)

        if scene_points.shape[0] > 0:
            _draw_point_cloud(scene_points, scene_colors)

        if self.enable_table_completion_var.get() and self.table_cache is not None:
            table_points = self.table_cache.plane_points_base.astype(np.float32, copy=False)
            if self.segment_scene_bbox is not None and table_points.shape[0] > 0:
                table_keep = self._bbox_pixel_keep_mask(self.table_cache.plane_pixels_yx, self.segment_scene_bbox)
                table_points = table_points[table_keep].astype(np.float32, copy=False)
            table_points = _finite_points(table_points)
            if table_points.shape[0] > DEFAULT_MAX_TABLE_PLANE_DISPLAY_POINTS:
                table_points = _subsample_points(table_points, DEFAULT_MAX_TABLE_PLANE_DISPLAY_POINTS, seed=23)
            if table_points.shape[0] > 0:
                table_colors = np.tile(TABLE_PLANE_OVERLAY_RGB.reshape((1, 3)), (table_points.shape[0], 1))
                _draw_point_cloud(table_points, table_colors.astype(np.float32, copy=False))

        if self.place_target_state is not None and self.place_target_state.points_base.shape[0] > 0:
            target_points = self.place_target_state.points_base.astype(np.float32, copy=False)
            if target_points.shape[0] > 6000:
                target_points = _subsample_points(target_points, 6000, seed=29)
            target_colors = np.tile(np.array([[1.0, 0.72, 0.30]], dtype=np.float32), (target_points.shape[0], 1))
            _draw_point_cloud(target_points, target_colors)
            if bool(self.show_place_target_fit_var.get()):
                profile = self.place_target_state.target_profile
                if profile.primitive_shape == "cylinder" and profile.primitive_radius is not None:
                    _draw_cylinder(
                        profile.center_base,
                        profile.rotation_base,
                        float(profile.primitive_radius),
                        float(profile.size_xyz[2]),
                        (48, 196, 255),
                    )
                else:
                    _draw_box(profile.center_base, profile.rotation_base, profile.size_xyz, (48, 196, 255))
                for entry in _fitted_target_face_annotations(profile):
                    _draw_text_label(entry["point"], entry["text"], entry["state"])
            if bool(self.show_place_target_axes_var.get()):
                profile = self.place_target_state.target_profile
                _draw_frame(
                    profile.center_base,
                    profile.rotation_base,
                    max(0.05, 0.35 * float(max(profile.size_xyz[0], profile.size_xyz[1], profile.size_xyz[2]))),
                    XYZ_AXIS_COLORS_BGR,
                )

        if self.object_state is not None:
            if bool(self.show_object_points_var.get()) and self.object_state.points_base.shape[0] > 0:
                object_points = self.object_state.points_base.astype(np.float32, copy=False)
                if object_points.shape[0] > DEFAULT_MAX_OBJECT_DISPLAY_POINTS:
                    object_points = _subsample_points(object_points, DEFAULT_MAX_OBJECT_DISPLAY_POINTS, seed=37)
                object_colors = np.tile(np.array([[1.0, 0.42, 0.68]], dtype=np.float32), (object_points.shape[0], 1))
                _draw_point_cloud(object_points, object_colors)
            if bool(self.show_object_fit_points_var.get()) and self.object_state.fit_points_base.shape[0] > 0:
                fit_points = self.object_state.fit_points_base.astype(np.float32, copy=False)
                if fit_points.shape[0] > DEFAULT_MAX_OBJECT_DISPLAY_POINTS:
                    fit_points = _subsample_points(fit_points, DEFAULT_MAX_OBJECT_DISPLAY_POINTS, seed=41)
                fit_colors = np.tile(np.array([[0.24, 0.92, 1.0]], dtype=np.float32), (fit_points.shape[0], 1))
                _draw_point_cloud(fit_points, fit_colors)
            rot = _rpy_to_matrix(self.object_state.rpy_rad)
            ctr = np.asarray(self.object_state.center_base, dtype=np.float64).reshape(1, 3)
            _draw_box(self.object_state.center_base, rot, self.object_state.size_xyz, (66, 107, 255))
            axis_len = 0.5 * float(max(self.object_state.size_xyz[0], self.object_state.size_xyz[1], self.object_state.size_xyz[2]))
            if bool(self.show_pick_obb_axes_var.get()):
                _draw_frame(
                    ctr[0],
                    rot,
                    axis_len,
                    XYZ_AXIS_COLORS_BGR,
                )
            if self.grasp_state is not None:
                grasp_origin = (np.asarray(self.grasp_state.point_obj, dtype=np.float64).reshape((1, 3)) @ rot.T + ctr).reshape(3)
                grasp_axes = (rot @ np.asarray(self.grasp_state.axes_obj, dtype=np.float64)).astype(np.float64, copy=False)
                if bool(self.show_grasp_axes_var.get()):
                    _draw_frame(
                        grasp_origin,
                        grasp_axes,
                        max(0.05, 0.42 * axis_len),
                        XYZ_AXIS_COLORS_BGR,
                    )

            center = self.object_state.center_base
            rpy_deg = np.degrees(self.object_state.rpy_rad.astype(np.float64, copy=False))
            text = (
                f"box xyz=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m  "
                f"rpy=({rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}) deg  "
                f"size=({self.object_state.size_xyz[0]:.3f}, {self.object_state.size_xyz[1]:.3f}, {self.object_state.size_xyz[2]:.3f}) m"
            )
            cv2.putText(img_bgr, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 235, 240), 1, cv2.LINE_AA)

        if self.place_plan_state is not None and self.object_state is not None:
            if self.place_plan_state.gripper_origin_trajectory_base.shape[0] >= 2:
                _draw_polyline(
                    self.place_plan_state.gripper_origin_trajectory_base,
                    (120, 214, 255),
                    thickness=1,
                )
            _draw_box(
                self.place_plan_state.object_center_base,
                self.place_plan_state.object_axes_base,
                self.object_state.size_xyz,
                (255, 225, 64),
            )
            axis_len = 0.5 * float(max(self.object_state.size_xyz[0], self.object_state.size_xyz[1], self.object_state.size_xyz[2]))
            if bool(self.show_place_obb_axes_var.get()):
                _draw_frame(
                    self.place_plan_state.object_center_base,
                    self.place_plan_state.object_axes_base,
                    axis_len,
                    XYZ_AXIS_COLORS_BGR,
                )
            if bool(self.show_place_grasp_axes_var.get()):
                traj_origins = self.place_plan_state.gripper_origin_trajectory_base.astype(np.float32, copy=False)
                traj_axes = self.place_plan_state.gripper_axes_trajectory_base.astype(np.float32, copy=False)
                if traj_origins.shape[0] > 0 and traj_axes.shape[0] == traj_origins.shape[0]:
                    draw_idx = _subsample_indices(
                        traj_origins.shape[0],
                        DEFAULT_PLACE_TRAJECTORY_MAX_RENDER_FRAMES,
                        seed=53,
                    )
                    for idx in draw_idx.tolist():
                        _draw_frame(
                            traj_origins[idx],
                            traj_axes[idx],
                            max(0.035, 0.28 * axis_len),
                            XYZ_AXIS_COLORS_BGR,
                        )
                _draw_frame(
                    self.place_plan_state.gripper_origin_base,
                    self.place_plan_state.gripper_axes_base,
                    max(0.05, 0.42 * axis_len),
                    XYZ_AXIS_COLORS_BGR,
                )
            plan_text = (
                f"plan rule={self.place_plan_state.rule_id} target={self.place_plan_state.target_kind} score={self.place_plan_state.score:.3f}"
            )
            cv2.putText(img_bgr, plan_text, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 232, 168), 1, cv2.LINE_AA)

        cv2.putText(img_bgr, "Axes: X=Red  Y=Green  Z=Blue", (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 220, 230), 1, cv2.LINE_AA)
        footer = "Viewer drag: orbit | Wheel: zoom | Keys: WASDQE move, UIOJKL rotate, Space release"
        cv2.putText(img_bgr, footer, (14, canvas_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 182, 196), 1, cv2.LINE_AA)

        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    def _render_viewer(self) -> None:
        width = max(400, int(self.viewer_canvas.winfo_width() or 400))
        height = max(320, int(self.viewer_canvas.winfo_height() or 320))
        image = self._build_scene_render(width, height)
        self.viewer_photo = ImageTk.PhotoImage(image)
        if self.viewer_image_id is None:
            self.viewer_image_id = self.viewer_canvas.create_image(0, 0, anchor="nw", image=self.viewer_photo)
        else:
            self.viewer_canvas.itemconfig(self.viewer_image_id, image=self.viewer_photo)

    def _set_info_text(self, text: str) -> None:
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", text)
        self.info_text.config(state="disabled")

    def _update_info_panel(self) -> None:
        lines: list[str] = []
        lines.append(f"left_image: {self.left_image_path_var.get().strip() or '-'}")
        lines.append(f"right_image: {self.right_image_path_var.get().strip() or '-'}")
        lines.append(f"topic_inputs: {self.topic_inputs_path_var.get().strip() or '-'}")
        if self.camera_bundle is not None:
            lines.append(
                "camera: "
                f"fx={float(self.camera_bundle.k1[0,0]):.3f}, fy={float(self.camera_bundle.k1[1,1]):.3f}, "
                f"cx={float(self.camera_bundle.k1[0,2]):.3f}, cy={float(self.camera_bundle.k1[1,2]):.3f}, "
                f"baseline={float(self.camera_bundle.baseline):.5f} m"
            )
        else:
            lines.append("camera: -")

        lines.append(f"s2m2_ready: {bool(self.raw_dense_map_cam is not None)}")
        lines.append(f"raw_scene_points_base: {int(self.raw_points_base.shape[0])}")
        lines.append(f"render_scene_points: {int(self.scene_points_base.shape[0])}")
        lines.append(f"collision_scene_points: {int(self.collision_points_base.shape[0])}")

        if self.selected_bbox is not None:
            lines.append(f"bbox: {list(self.selected_bbox)}")
        else:
            lines.append("bbox: -")
        if self.segment_scene_bbox is not None and self.raw_pixels_yx.shape[0] > 0:
            bbox_points = int(np.count_nonzero(self._bbox_pixel_keep_mask(self.raw_pixels_yx, self.segment_scene_bbox)))
            lines.append(
                f"segment_scene: active_bbox={list(self.segment_scene_bbox)}, bbox_scene_points={bbox_points}"
            )
        else:
            lines.append("segment_scene: active_bbox=-, bbox_scene_points=-")

        if self.table_cache is not None:
            lines.append(
                "table_completion: "
                f"enabled={bool(self.enable_table_completion_var.get())}, "
                f"prompt={self.table_cache.prompt_used}, "
                f"mask_px={int(self.table_cache.debug.get('mask_area_px', 0))}, "
                f"completion_points={int(self.table_cache.plane_points_base.shape[0])}"
            )
            lines.append(
                "table_plane_base: "
                f"normal={[round(float(v), 6) for v in self.table_cache.plane_model.normal_base.reshape(-1)[:3]]}, "
                f"d={float(self.table_cache.plane_model.d_base):.6f}, "
                f"source={self.table_cache.plane_model.source}"
            )
        else:
            lines.append(f"table_completion: enabled={bool(self.enable_table_completion_var.get())}, cache=empty")

        if self.object_state is not None:
            rpy_deg = np.degrees(self.object_state.rpy_rad.astype(np.float64, copy=False))
            lines.append(
                "object: "
                f"mask_px={int(self.object_state.mask_area_px)}, "
                f"points={int(self.object_state.points_base.shape[0])}, "
                f"fit_points={int(self.object_state.fit_points_base.shape[0])}"
            )
            lines.append(
                "box_pose_base: "
                f"center={[round(float(v), 4) for v in self.object_state.center_base.tolist()]}, "
                f"rpy_deg={[round(float(v), 2) for v in rpy_deg.tolist()]}"
            )
            lines.append(
                "box_size_xyz: "
                f"{[round(float(v), 4) for v in self.object_state.size_xyz.tolist()]}"
            )
            lines.append(
                "object_plane_base: "
                f"normal={[round(float(v), 6) for v in self.object_state.plane_model.normal_base.reshape(-1)[:3]]}, "
                f"d={float(self.object_state.plane_model.d_base):.6f}, "
                f"source={self.object_state.plane_model.source}"
            )
            for key in sorted(self.object_state.debug.keys()):
                lines.append(f"object_debug.{key}: {self.object_state.debug[key]}")
            lines.append(f"object_overlay.raw_visible: {bool(self.show_object_points_var.get())}")
            lines.append(f"object_overlay.fit_visible: {bool(self.show_object_fit_points_var.get())}")
            lines.append(f"object_overlay.pick_obb_axes_visible: {bool(self.show_pick_obb_axes_var.get())}")
            lines.append(f"object_overlay.grasp_axes_visible: {bool(self.show_grasp_axes_var.get())}")
        else:
            lines.append("object: -")

        if self.grasp_state is not None:
            lines.append(
                "grasp_local: "
                f"point_obj={[round(float(v), 4) for v in self.grasp_state.point_obj.tolist()]}"
            )
            axes = np.asarray(self.grasp_state.axes_obj, dtype=np.float64)
            lines.append(
                "grasp_axes_obj: "
                f"x={[round(float(v), 4) for v in axes[:, 0].tolist()]}, "
                f"y={[round(float(v), 4) for v in axes[:, 1].tolist()]}, "
                f"z={[round(float(v), 4) for v in axes[:, 2].tolist()]}"
            )
            for key in sorted(self.grasp_state.debug.keys()):
                lines.append(f"grasp_debug.{key}: {self.grasp_state.debug[key]}")
        else:
            lines.append("grasp_local: -")

        if self.place_target_state is not None:
            profile = self.place_target_state.target_profile
            lines.append(
                "place_target: "
                f"bbox={list(self.place_target_state.bbox)}, "
                f"mask_px={int(self.place_target_state.mask_area_px)}, "
                f"points={int(self.place_target_state.points_base.shape[0])}, "
                f"fit_points={int(self.place_target_state.fit_points_base.shape[0])}"
            )
            lines.append(f"place_target_kind_hint: {self.place_target_kind_var.get().strip() or placedof_planner.DEFAULT_TARGET_KIND}")
            lines.append(
                "place_target_fit: "
                f"primitive={profile.primitive_shape}, "
                f"auto_kind={profile.kind}, "
                f"opening_face={profile.opening_face or '-'}"
            )
            face_visibility = profile.debug.get("face_visibility", {})
            if isinstance(face_visibility, dict) and face_visibility:
                vis_text = ", ".join(
                    f"{str(face).upper()}={float(score):.2f}"
                    for face, score in sorted(face_visibility.items())
                )
                lines.append(f"place_target_face_visibility: {vis_text}")
            face_state_text = ", ".join(entry["text"] for entry in _fitted_target_face_annotations(profile))
            if face_state_text:
                lines.append(f"place_target_face_states: {face_state_text}")
            lines.append(f"place_target_fit_overlay_visible: {bool(self.show_place_target_fit_var.get())}")
            lines.append(f"place_target_axes_visible: {bool(self.show_place_target_axes_var.get())}")
            lines.append(
                "place_target_center_base: "
                f"{[round(float(v), 4) for v in self.place_target_state.preferred_center_base.tolist()]}"
            )
            lines.append(
                "place_target_size_xyz: "
                f"{[round(float(v), 4) for v in profile.size_xyz.tolist()]}"
            )
            if profile.primitive_radius is not None:
                lines.append(f"place_target_radius: {float(profile.primitive_radius):.4f}")
            for key in sorted(self.place_target_state.debug.keys()):
                lines.append(f"place_target_debug.{key}: {self.place_target_state.debug[key]}")
        else:
            lines.append("place_target: -")

        if self.place_plan_state is not None:
            lines.append(
                "place_plan: "
                f"rule={self.place_plan_state.rule_id}, "
                f"target_kind={self.place_plan_state.target_kind}, "
                f"score={self.place_plan_state.score:.4f}, "
                f"matched_rules={self.place_plan_state.matched_rules}"
            )
            lines.append(
                "place_box_pose_base: "
                f"center={[round(float(v), 4) for v in self.place_plan_state.object_center_base.tolist()]}, "
                f"rpy_deg={[round(float(v), 2) for v in np.degrees(self.place_plan_state.object_rpy_rad.astype(np.float64)).tolist()]}"
            )
            lines.append(
                "place_gripper_origin_base: "
                f"{[round(float(v), 4) for v in self.place_plan_state.gripper_origin_base.tolist()]}"
            )
            lines.append(f"place_plan_overlay.obb_axes_visible: {bool(self.show_place_obb_axes_var.get())}")
            lines.append(f"place_plan_overlay.grasp_axes_visible: {bool(self.show_place_grasp_axes_var.get())}")
            target_analysis = self.place_plan_state.debug.get("target_analysis", {})
            if isinstance(target_analysis, dict) and target_analysis:
                lines.append(
                    "place_plan_target_analysis: "
                    f"primitive={target_analysis.get('primitive_shape', '-')}, "
                    f"opening_face={target_analysis.get('opening_face', '-')}, "
                    f"top_open_score={float(target_analysis.get('top_open_score', 0.0)):.4f}, "
                    f"front_open_score={float(target_analysis.get('front_open_score', 0.0)):.4f}"
                )
            planner_debug = self.place_plan_state.debug.get("planner", {})
            if isinstance(planner_debug, dict):
                invalid_reason_counts = planner_debug.get("invalid_reason_counts", {})
                if isinstance(invalid_reason_counts, dict) and invalid_reason_counts:
                    lines.append(f"place_plan_invalid_reasons: {invalid_reason_counts}")
            visibility_adjustment = self.place_plan_state.debug.get("visibility_adjustment", {})
            if isinstance(visibility_adjustment, dict) and visibility_adjustment:
                lines.append(
                    "place_plan_visibility_adjustment: "
                    f"applied={bool(visibility_adjustment.get('applied', False))}, "
                    f"delta_base={visibility_adjustment.get('delta_base', [0.0, 0.0, 0.0])}"
                )
                opening_adjustment = visibility_adjustment.get("opening_adjustment", {})
                if isinstance(opening_adjustment, dict) and opening_adjustment:
                    lines.append(f"place_plan_opening_adjustment: {opening_adjustment}")
                lift_adjustment = visibility_adjustment.get("lift_adjustment", {})
                if isinstance(lift_adjustment, dict) and lift_adjustment:
                    lines.append(f"place_plan_lift_adjustment: {lift_adjustment}")
                final_visibility = visibility_adjustment.get("final", {})
                if isinstance(final_visibility, dict) and final_visibility:
                    lines.append(f"place_plan_camera_visibility_final: {final_visibility}")
            trajectory_debug = self.place_plan_state.debug.get("trajectory", {})
            if isinstance(trajectory_debug, dict) and trajectory_debug:
                lines.append(
                    "place_plan_trajectory: "
                    f"mode={trajectory_debug.get('mode', '-')}, "
                    f"points={int(trajectory_debug.get('path_points', 0))}, "
                    f"path_length_m={float(trajectory_debug.get('object_path_length_m', 0.0)):.4f}, "
                    f"rotation_total_deg={float(trajectory_debug.get('rotation_total_deg', 0.0)):.2f}"
                )
                lines.append(f"place_plan_trajectory_debug: {trajectory_debug}")
            for key in sorted(self.place_plan_state.score_breakdown.keys()):
                lines.append(f"place_plan_score.{key}: {self.place_plan_state.score_breakdown[key]:.6f}")
        else:
            lines.append("place_plan: -")

        lines.append("")
        lines.append("hotkeys:")
        lines.append("  w/s -> +Y/-Y, a/d -> -X/+X, q/e -> +Z/-Z")
        lines.append("  u/o -> -roll/+roll, i/k -> +pitch/-pitch, j/l -> -yaw/+yaw")
        lines.append("  space -> release/drop, r -> reset fitted box")
        lines.append("  p -> show planning window")
        lines.append("  Pick Grasp Point button: next left-image click sets grasp point")
        lines.append("  Free Pose ON: nudge without dropping; Free Pose OFF: each nudge settles immediately")
        self._set_info_text("\n".join(lines))

    def _focus_on_text_input(self) -> bool:
        widget = self.root.focus_get()
        if widget is None:
            return False
        try:
            cls = str(widget.winfo_class())
        except Exception:
            return False
        return cls in {"Entry", "TEntry", "Text"}

    def _on_key_press(self, event: tk.Event) -> None:
        if self._focus_on_text_input():
            return
        key = str(event.keysym).lower()
        if key == "space":
            self._release_box_pose()
            return
        if key == "r":
            self._reset_box_pose()
            return
        if key == "p":
            self._show_plan_window()
            return
        move_map = {
            "a": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "d": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "s": np.array([0.0, -1.0, 0.0], dtype=np.float32),
            "w": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "e": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "q": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        rot_map = {
            "u": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "o": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "k": np.array([0.0, -1.0, 0.0], dtype=np.float32),
            "i": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "j": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "l": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        if key in move_map:
            self._nudge_box(move_map[key], None)
            return
        if key in rot_map:
            self._nudge_box(None, rot_map[key])

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    try:
        app = PlaceDofInteractiveDemo()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    app.run()


if __name__ == "__main__":
    main()
