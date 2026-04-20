#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import socket
import socketserver
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image, ImageColor, ImageDraw

import placedof_planner
from placedof_protocol import (
    PICK_CONTEXT_LEN,
    flatten_pick_context,
    parse_pick_context,
    recv_exact,
)

QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode",
).rstrip("/")
QWEN_CHAT_URL = os.getenv("QWEN_CHAT_URL", f"{QWEN_BASE_URL}/v1/chat/completions")
QWEN_AUTH_TOKEN = os.getenv(
    "QWEN_AUTH_TOKEN",
    "sk-51bbc617c8a346379498e3f4874c3cbe",
)
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "qwen3.5-397b-a17b")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = int(os.getenv("PLACEDOF_SOCKET_PORT", "5688"))
DEFAULT_TIMEOUT_S = float(os.getenv("PLACEDOF_SOCKET_TIMEOUT_S", "180.0"))
DEFAULT_S2M2_API = os.getenv("PLACEDOF_S2M2_API_URL", os.getenv("S2M2_API_URL", "http://192.168.20.225:5060/api/process")).strip()
DEFAULT_SAM3_URL = os.getenv("PLACEDOF_SAM3_URL", os.getenv("SAM3_SERVER_URL", "http://192.168.20.40:5081")).rstrip("/")
DEFAULT_LOG_DIR = os.getenv("PLACEDOF_SOCKET_LOG_DIR", "placedof_socket_logs")
DEFAULT_MIN_CAM_DEPTH = 1e-4
DEFAULT_MAX_COORD_ABS = 1e6
DEFAULT_PLANE_RANSAC_THR_M = 0.012
DEFAULT_TABLE_RANSAC_ITERS = 320
DEFAULT_TABLE_RANSAC_MIN_POINTS = 80
DEFAULT_OBJECT_PLANE_CLEARANCE_M = 0.004
DEFAULT_OBJECT_MIN_HEIGHT_M = 0.015
DEFAULT_OBJECT_MIN_CROSS_SECTION_M = 0.006
DEFAULT_OBJECT_MAX_HEIGHT_M = 0.40
DEFAULT_SLENDER_OBJECT_ASPECT_RATIO = 3.0
DEFAULT_SLENDER_OBJECT_LENGTH_RECOVER_RATIO = 0.75
DEFAULT_OBJECT_HEIGHT_QUANTILE = 0.96
DEFAULT_SLENDER_OBJECT_HEIGHT_QUANTILE = 0.90
DEFAULT_SLENDER_OBJECT_THICKNESS_CAP_RATIO = 1.35
DEFAULT_COLLISION_MARGIN_M = 0.003
DEFAULT_PLANE_CONTACT_EPS_M = 0.0015
DEFAULT_TABLE_PLANE_EXCLUDE_M = 0.010
DEFAULT_PLACE_CAMERA_PIXEL_MARGIN_PX = 12
DEFAULT_PLACE_VISIBILITY_LIFT_STEP_M = 0.015
DEFAULT_PLACE_VISIBILITY_MAX_LIFT_M = 0.24
DEFAULT_PLACE_VISIBILITY_OPENING_STEP_M = 0.015
DEFAULT_PLACE_VISIBILITY_EXTRA_OPENING_M = 0.12
DEFAULT_PLACE_OPENING_EXPOSURE_MIN_M = 0.015
DEFAULT_PLACE_OPENING_EXPOSURE_RATIO = 0.90
DEFAULT_PLACE_OPENING_EXPOSURE_SLENDER_RATIO = 0.25
DEFAULT_PLACE_OPENING_SHIFT_MAX_M = 0.30

DEFAULT_GRIPPER_OPEN_QUAT = [0.0, 0.9239, 0.0, 0.3827]
PREDEFINED_QUATS = {
    "vertical": DEFAULT_GRIPPER_OPEN_QUAT,
    "horizontal": [0.0, 0.7, 0.0, 0.7],
    "pencil_quat": [0.0, 1.0, 0.0, 0.0],
    "left": [0.448, 0.779, -0.372, 0.252],
    "right": [-0.385, 0.772, 0.517, 0.2848],
}
DRAW_COLORS = ("red", "blue", "yellow", "green", "orange", "cyan")
BOX_EDGE_RGB = (255, 184, 64)
PICK_BOX_RGB = (255, 96, 72)
PICK_FIT_RGB = (64, 220, 255)
PLACE_BOX_RGB = (255, 236, 80)
PLACE_TARGET_RGB = (96, 196, 255)
MASK_PICK_RGB = (255, 82, 110)
MASK_PLACE_RGB = (255, 188, 82)
AXIS_COLORS_RGB = ((255, 0, 0), (0, 255, 0), (0, 96, 255))

DEFUALT_SYSTEM_PROMPT = (
    "You are a professional and very intelligent mechanical arm robot. "
    "You need to complete a task according to the image and the task description."
    "If you can answer the question, give your response in JSON format and strictly follow the <OUTPUT FORMAT>."
    "If you cannot find the object or answer the question, just return a single word '<FAILED>'."
)
GROUNDING_OUTPUT_TEMPLATE = '{"bbox": [x1, y1, x2, y2]}'


@dataclass
class ServiceConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    timeout_s: float = DEFAULT_TIMEOUT_S
    s2m2_api_url: str = DEFAULT_S2M2_API
    sam3_url: str = DEFAULT_SAM3_URL
    log_dir: str = DEFAULT_LOG_DIR


@dataclass
class PlaneModel:
    normal_base: np.ndarray
    d_base: float


def _elapsed_ms(t0: float) -> int:
    return int(round((time.perf_counter() - t0) * 1000.0))


def recv_message(sock: socket.socket) -> str:
    header = recv_exact(sock, 4)
    msg_len = int.from_bytes(header, byteorder="big", signed=False)
    if msg_len <= 0:
        raise ValueError("Invalid message length")
    payload = recv_exact(sock, msg_len)
    return payload.decode("utf-8")


def send_message(sock: socket.socket, obj: dict[str, Any]) -> None:
    data = json.dumps(_to_jsonable(obj), ensure_ascii=False).encode("utf-8")
    header = len(data).to_bytes(4, byteorder="big", signed=False)
    sock.sendall(header + data)


def _finite_point_mask(points: np.ndarray, max_abs: float = DEFAULT_MAX_COORD_ABS) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.zeros((0,), dtype=bool)
    keep = np.all(np.isfinite(pts), axis=1)
    keep &= np.max(np.abs(pts), axis=1) <= float(max_abs)
    return keep.astype(bool, copy=False)


def _finite_points(points: np.ndarray, max_abs: float = DEFAULT_MAX_COORD_ABS) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32)
    keep = _finite_point_mask(pts, max_abs=max_abs)
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.float32)
    return pts[keep].astype(np.float32, copy=False)


def _subsample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    pts = _finite_points(points)
    limit = max(0, int(max_points))
    if limit <= 0 or pts.shape[0] <= limit:
        return pts.astype(np.float32, copy=False)
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(pts.shape[0], size=limit, replace=False))
    return pts[idx].astype(np.float32, copy=False)


def _valid_cam_xyz_points(
    points: np.ndarray,
    min_depth: float = DEFAULT_MIN_CAM_DEPTH,
    max_abs: float = DEFAULT_MAX_COORD_ABS,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32)
    keep = np.all(np.isfinite(pts), axis=1)
    keep &= np.max(np.abs(pts), axis=1) <= float(max_abs)
    keep &= pts[:, 2] > float(min_depth)
    return pts[keep].astype(np.float32, copy=False)


def _valid_cam_xyz_mask_map(
    xyz_map: np.ndarray,
    min_depth: float = DEFAULT_MIN_CAM_DEPTH,
    max_abs: float = DEFAULT_MAX_COORD_ABS,
) -> np.ndarray:
    valid = np.all(np.isfinite(xyz_map), axis=2)
    valid &= np.max(np.abs(xyz_map), axis=2) <= float(max_abs)
    valid &= xyz_map[..., 2] > float(min_depth)
    return valid


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
                shell.append(center + radius * math.cos(theta) * ex + radius * math.sin(theta) * ey + h * ez)
        face_samples.append(np.asarray(shell, dtype=np.float32))
        if open_spec is None or open_spec[0] != 2 or open_spec[1] < 0.0:
            radial_layers = max(3, int(math.ceil(radius / sample_step)) + 1)
            bottom_pts = []
            for radial in np.linspace(0.0, radius, radial_layers, dtype=np.float64).tolist():
                ring_segments = max(12, int(math.ceil(2.0 * math.pi * max(radial, sample_step) / sample_step)))
                for theta in np.linspace(0.0, 2.0 * math.pi, ring_segments, endpoint=False, dtype=np.float64).tolist():
                    bottom_pts.append(center + radial * math.cos(theta) * ex + radial * math.sin(theta) * ey - 0.5 * height * ez)
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
        return np.zeros((0, 3), dtype=np.float32)
    points = _finite_points(np.vstack(face_samples).astype(np.float32, copy=False))
    if points.shape[0] > 6000:
        points = _subsample_points(points, 6000, seed=71)
    return points.astype(np.float32, copy=False)


def _normalize(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size > 0 and np.all(np.isfinite(arr)):
        norm = float(np.linalg.norm(arr))
        if norm > 1e-8:
            return (arr / norm).astype(np.float64, copy=False)
    if fallback is not None:
        fb = np.asarray(fallback, dtype=np.float64).reshape(-1)
        if fb.size > 0 and np.all(np.isfinite(fb)):
            norm = float(np.linalg.norm(fb))
            if norm > 1e-8:
                return (fb / norm).astype(np.float64, copy=False)
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _orthonormalize_axes(axes: np.ndarray) -> np.ndarray:
    arr = np.asarray(axes, dtype=np.float64).reshape(3, 3)
    x_axis = _normalize(arr[:, 0], fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    y_axis = arr[:, 1] - x_axis * float(np.dot(x_axis, arr[:, 1]))
    y_axis = _normalize(y_axis, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    z_axis = _normalize(np.cross(x_axis, y_axis), fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    y_axis = _normalize(np.cross(z_axis, x_axis), fallback=y_axis)
    out = np.stack([x_axis, y_axis, z_axis], axis=1)
    if float(np.linalg.det(out)) < 0.0:
        out[:, 1] *= -1.0
        out[:, 2] = _normalize(np.cross(out[:, 0], out[:, 1]), fallback=out[:, 2])
    return out.astype(np.float32, copy=False)


def _quat_to_matrix(quat: np.ndarray | list[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in np.asarray(quat, dtype=np.float64).reshape(4)]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def _safe_quantile(values: np.ndarray, q: float, default: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, float(q)))


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _rot2(yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _bbox_to_mask(bbox: tuple[int, int, int, int], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    mask = np.zeros((int(height), int(width)), dtype=bool)
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    mask[y1 : y2 + 1, x1 : x2 + 1] = True
    return mask


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


def _transform_points_cam_to_base(points: np.ndarray, rotation: np.ndarray, shift: np.ndarray) -> np.ndarray:
    pts = _valid_cam_xyz_points(points)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sh = np.asarray(shift, dtype=np.float64).reshape(3)
    out = np.asarray(pts, dtype=np.float64) @ rot.T + sh.reshape((1, 3))
    return _finite_points(out.astype(np.float32, copy=False))


def _transform_dense_map_cam_to_base(dense_map_cam: np.ndarray, rotation: np.ndarray, shift: np.ndarray) -> np.ndarray:
    dense = np.asarray(dense_map_cam, dtype=np.float32)
    out = np.full_like(dense, np.nan, dtype=np.float32)
    valid = _valid_cam_xyz_mask_map(dense)
    if not np.any(valid):
        return out
    pts_base = _transform_points_cam_to_base(dense[valid], rotation, shift)
    if pts_base.shape[0] == 0:
        return out
    out[valid] = pts_base[: np.count_nonzero(valid)]
    return out


def _parse_json_from_text(text: str) -> Any:
    candidate = (text or "").strip()
    if not candidate:
        return None
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", candidate, re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
    decoder = json.JSONDecoder()
    for idx, char in enumerate(candidate):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[idx:])
            return parsed
        except json.JSONDecodeError:
            continue
    return None


def _extract_message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = getattr(item, "text", "")
            if text:
                texts.append(str(text))
        return "\n".join(texts)
    if isinstance(content, str):
        return content
    return ""


def _image_to_jpeg_b64(image: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=int(quality), optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _decode_image_from_payload(payload: dict[str, Any], prefix: str) -> Image.Image:
    path_key = f"{prefix}_path"
    b64_key = f"{prefix}_b64"
    b64_value = payload.get(b64_key)
    if not isinstance(b64_value, str) and prefix in payload:
        b64_value = payload.get(prefix)
    if isinstance(b64_value, str) and b64_value.strip():
        b64_text = b64_value.strip()
        if b64_text.startswith("data:image") and "," in b64_text:
            b64_text = b64_text.split(",", 1)[1]
        raw = base64.b64decode(b64_text)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(payload.get(path_key), str) and str(payload[path_key]).strip():
        return Image.open(str(payload[path_key]).strip()).convert("RGB")
    raise ValueError(f"Missing {path_key} or {b64_key} in request.")


def _summarize_payload_blob(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    blob = value.strip()
    if not blob:
        return ""
    if blob.startswith("data:image") and "," in blob:
        header, body = blob.split(",", 1)
        return f"{header},<omitted {len(body)} chars>"
    return f"<omitted {len(blob)} chars>"


def _sanitize_payload_for_logging(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(payload)
    for key in ("image_left_b64", "image_right_b64", "image_left", "image_right"):
        if key in sanitized:
            sanitized[key] = _summarize_payload_blob(sanitized.get(key))
    return sanitized


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _compact_logged_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    role = str(result.get("role", ""))
    base = {
        "role": role,
        "label": str(result.get("label", "")),
        "bbox": [int(v) for v in (result.get("bbox") or [])],
        "text": [str(v) for v in (result.get("text") or [])],
        "data_len": int(len(result.get("data") or [])),
    }
    if role == "pick":
        pick_fit = result.get("pick_fit") or {}
        base["pick_fit"] = _to_jsonable(
            {
                "center_base": pick_fit.get("center_base"),
                "axes_base": pick_fit.get("axes_base"),
                "size_xyz": pick_fit.get("size_xyz"),
                "grasp_origin_base": pick_fit.get("grasp_origin_base"),
                "grasp_axes_base": pick_fit.get("grasp_axes_base"),
                "grasp_origin_local": pick_fit.get("grasp_origin_local"),
                "grasp_axes_local": pick_fit.get("grasp_axes_local"),
                "debug": pick_fit.get("debug", {}),
            }
        )
        base["pick_xquat"] = _to_jsonable(result.get("pick_xquat"))
    elif role == "place":
        target_fit = result.get("target_fit") or {}
        place_plan = result.get("place_plan") or {}
        base["target_fit"] = _to_jsonable(
            {
                "center_base": target_fit.get("center_base"),
                "axes_base": target_fit.get("axes_base"),
                "size_xyz": target_fit.get("size_xyz"),
                "primitive_shape": target_fit.get("primitive_shape"),
                "primitive_radius": target_fit.get("primitive_radius"),
                "kind": target_fit.get("kind"),
                "opening_face": target_fit.get("opening_face"),
                "debug": target_fit.get("debug", {}),
            }
        )
        base["place_plan"] = _to_jsonable(
            {
                "object_center_base": place_plan.get("object_center_base"),
                "object_axes_base": place_plan.get("object_axes_base"),
                "object_size_xyz": place_plan.get("object_size_xyz"),
                "gripper_origin_base": place_plan.get("gripper_origin_base"),
                "gripper_axes_base": place_plan.get("gripper_axes_base"),
                "rule_id": place_plan.get("rule_id"),
                "target_kind": place_plan.get("target_kind"),
                "score": place_plan.get("score"),
                "score_breakdown": place_plan.get("score_breakdown", {}),
                "visibility_adjustment": place_plan.get("visibility_adjustment", {}),
            }
        )
    return base


def _load_topic_inputs_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("topic_inputs_text"), str) and payload["topic_inputs_text"].strip():
        return str(payload["topic_inputs_text"])
    if isinstance(payload.get("topic_inputs_path"), str) and payload["topic_inputs_path"].strip():
        with open(str(payload["topic_inputs_path"]).strip(), "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError("Missing topic_inputs_text/topic_inputs_path with raw_topic_inputs format.")


def _parse_topic_inputs(topic_text: str) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    lines = topic_text.splitlines()
    p_values: list[float] = []
    p_started = False
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        if raw.lower() == "p:" or raw.lower().startswith("p:"):
            p_started = True
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
            if nums:
                p_values.extend(float(x) for x in nums)
            continue
        if p_started:
            if raw.startswith("-"):
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
                if nums:
                    p_values.append(float(nums[0]))
                if len(p_values) >= 12:
                    break
            else:
                if len(p_values) >= 12:
                    break
                nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
                p_values.extend(float(x) for x in nums)
                if len(p_values) >= 12:
                    break
    if len(p_values) < 12:
        all_nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", topic_text)
        if len(all_nums) >= 12:
            p_values = [float(x) for x in all_nums[:12]]
    if len(p_values) < 12:
        raise ValueError("Failed to parse camera projection matrix p[12] from topic inputs.")
    fx = float(p_values[0])
    fy = float(p_values[5])
    cx = float(p_values[2])
    cy = float(p_values[6])
    tx = float(p_values[3])
    baseline = abs(tx) / max(abs(fx), 1e-9)
    if baseline > 1.0:
        baseline *= 0.001
    if baseline <= 1e-6:
        baseline = 0.08
    k1 = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    rot = np.eye(3, dtype=np.float32)
    shift = np.zeros((3,), dtype=np.float32)
    marker = "T_cam_to_base"
    idx = topic_text.find(marker)
    if idx >= 0:
        parsed = _parse_json_from_text(topic_text[idx:])
        if isinstance(parsed, dict):
            if "T_cam_to_base" in parsed and isinstance(parsed["T_cam_to_base"], dict):
                parsed = parsed["T_cam_to_base"]
            rot_raw = np.asarray(parsed.get("rotation"), dtype=np.float32)
            shift_raw = np.asarray(parsed.get("shift"), dtype=np.float32)
            if rot_raw.shape == (3, 3) and shift_raw.shape == (3,) and np.all(np.isfinite(rot_raw)):
                rot = rot_raw
                shift = shift_raw
    return k1, float(baseline), rot, shift


def _rgb_to_nv12(image: Image.Image) -> tuple[bytes, int, int]:
    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if height % 2 != 0 or width % 2 != 0:
        height -= height % 2
        width -= width % 2
        rgb = rgb[:height, :width]
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
    y = yuv[:height, :]
    u = yuv[height : height + height // 4, :].reshape((height // 2, width // 2))
    v = yuv[height + height // 4 :, :].reshape((height // 2, width // 2))
    uv = np.empty((height // 2, width), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    nv12 = np.vstack((y, uv))
    return nv12.tobytes(), width, height


def _load_pfm(data: bytes) -> np.ndarray:
    buffer = io.BytesIO(data)
    header = buffer.readline().decode("ascii", errors="ignore").strip()
    if header not in ("PF", "Pf"):
        raise ValueError("Invalid PFM header.")
    dims = buffer.readline().decode("ascii", errors="ignore").strip()
    while dims.startswith("#"):
        dims = buffer.readline().decode("ascii", errors="ignore").strip()
    width, height = map(int, dims.split())
    scale = float(buffer.readline().decode("ascii", errors="ignore").strip())
    endian = "<" if scale < 0 else ">"
    arr = np.frombuffer(buffer.read(), dtype=endian + "f")
    shape = (height, width, 3) if header == "PF" else (height, width)
    arr = np.reshape(arr, shape)
    arr = np.flipud(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def _depth_image_to_array(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def _load_depth_from_bytes(data: bytes) -> np.ndarray:
    if data.startswith(b"\x93NUMPY"):
        arr = np.load(io.BytesIO(data))
        if isinstance(arr, np.lib.npyio.NpzFile):
            if "depth" in arr:
                arr = arr["depth"]
            else:
                arr = arr[arr.files[0]]
        return np.array(arr, dtype=np.float32)
    if data.startswith(b"PF") or data.startswith(b"Pf"):
        return _load_pfm(data)
    return _depth_image_to_array(Image.open(io.BytesIO(data)))


def _decode_depth_payload(payload: Any) -> np.ndarray:
    if isinstance(payload, list):
        return np.array(payload, dtype=np.float32)
    if isinstance(payload, dict):
        for key in ("data", "depth", "value", "base64", "b64"):
            if key in payload:
                return _decode_depth_payload(payload[key])
        raise ValueError(f"Unsupported depth payload dict keys: {list(payload.keys())}")
    if isinstance(payload, str):
        return _load_depth_from_bytes(base64.b64decode(payload))
    raise ValueError(f"Unsupported depth payload type: {type(payload)}")


def _parse_depth_response(response: requests.Response) -> np.ndarray:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = response.json()
        for key in ("depth", "disp", "disparity"):
            if key in payload:
                return _decode_depth_payload(payload[key])
        if "data" in payload:
            return _decode_depth_payload(payload["data"])
        raise ValueError(f"Unsupported JSON payload keys: {list(payload.keys())}")
    return _load_depth_from_bytes(response.content)


def _resize_pointcloud_map(pointcloud: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    if pointcloud.shape[0] == target_h and pointcloud.shape[1] == target_w:
        return pointcloud
    return cv2.resize(pointcloud, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def _resize_depth_map(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    if depth.shape[0] == target_h and depth.shape[1] == target_w:
        return depth
    return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def _build_dense_from_cloud_or_depth(
    cloud_or_depth: np.ndarray,
    image_size: tuple[int, int],
    k1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    img_w, img_h = image_size
    data = np.asarray(cloud_or_depth, dtype=np.float32)
    dense_map = np.full((img_h, img_w, 3), np.nan, dtype=np.float32)
    depth_map = np.full((img_h, img_w), np.nan, dtype=np.float32)
    if data.ndim == 3 and data.shape[2] == 3:
        if data.shape[0] != img_h or data.shape[1] != img_w:
            data = _resize_pointcloud_map(data, (img_w, img_h))
        valid = _valid_cam_xyz_mask_map(data)
        dense_map[valid] = data[valid]
        depth_map[valid] = data[..., 2][valid]
        return dense_map, depth_map
    if data.ndim == 2 and data.shape[1] == 3:
        points = _valid_cam_xyz_points(data.astype(np.float32))
        if points.size == 0:
            return dense_map, depth_map
        fx = float(k1[0, 0])
        fy = float(k1[1, 1])
        cx = float(k1[0, 2])
        cy = float(k1[1, 2])
        u_f = points[:, 0].astype(np.float64) * fx / points[:, 2].astype(np.float64) + cx
        v_f = points[:, 1].astype(np.float64) * fy / points[:, 2].astype(np.float64) + cy
        finite_uv = np.isfinite(u_f) & np.isfinite(v_f)
        points = points[finite_uv]
        u_f = u_f[finite_uv]
        v_f = v_f[finite_uv]
        pu = np.rint(u_f).astype(np.int32)
        pv = np.rint(v_f).astype(np.int32)
        inside = (pu >= 0) & (pu < img_w) & (pv >= 0) & (pv < img_h)
        points = points[inside]
        pu = pu[inside]
        pv = pv[inside]
        dense_map[pv, pu] = points
        depth_map[pv, pu] = points[:, 2]
        return dense_map, depth_map
    depth = np.asarray(data, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Unsupported S2M2 output shape: {depth.shape}")
    if depth.shape[0] != img_h or depth.shape[1] != img_w:
        depth = _resize_depth_map(depth, (img_w, img_h))
    fx = float(k1[0, 0])
    fy = float(k1[1, 1])
    cx = float(k1[0, 2])
    cy = float(k1[1, 2])
    u = np.arange(img_w, dtype=np.float32)[None, :]
    v = np.arange(img_h, dtype=np.float32)[:, None]
    valid = np.isfinite(depth) & (depth > DEFAULT_MIN_CAM_DEPTH)
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    dense_map[..., 0][valid] = x[valid]
    dense_map[..., 1][valid] = y[valid]
    dense_map[..., 2][valid] = depth[valid]
    depth_map[valid] = depth[valid]
    return dense_map, depth_map


def _call_s2m2_api(
    left_img: Image.Image,
    right_img: Image.Image,
    k1: np.ndarray,
    baseline: float,
    api_url: str,
    timeout_s: float,
) -> tuple[np.ndarray, int]:
    k_flat = " ".join(str(v) for v in k1.reshape(-1))
    left_nv12, width, height = _rgb_to_nv12(left_img)
    right_nv12, right_w, right_h = _rgb_to_nv12(right_img)
    if right_w != width or right_h != height:
        raise ValueError("Left/right NV12 sizes do not match.")
    data = {"K": k_flat, "baseline": str(float(baseline)), "width": str(int(width))}
    files = {
        "left_file": ("left.nv12", left_nv12, "application/octet-stream"),
        "right_file": ("right.nv12", right_nv12, "application/octet-stream"),
    }
    t0 = time.perf_counter()
    response = requests.post(str(api_url).strip(), data=data, files=files, timeout=float(timeout_s))
    latency_ms = _elapsed_ms(t0)
    if response.status_code != 200:
        raise RuntimeError(f"S2M2 API failed ({response.status_code}): {response.text}")
    return _parse_depth_response(response), latency_ms


def parse_prompt(prompt: str) -> tuple[str, str | None, bool, Any]:
    assert isinstance(prompt, str), f"prompt item must be a string, got {type(prompt)}"
    if prompt.startswith("&"):
        need_tracking = True
        prompt = prompt[1:]
    else:
        need_tracking = False
    pattern = r"^(.+?)(?::([^@]+))?(?:@(.+))?$"
    match = re.match(pattern, prompt.strip())
    if not match:
        raise ValueError(f"prompt string format mismatch: {prompt}")
    label = str(match.group(1) or "").strip()
    orientation = match.group(2)
    mask = match.group(3)
    if mask is not None:
        mask = eval(mask)
    return label, orientation, need_tracking, mask


def _build_prompt_from_parts(label: Any, gesture: Any = None, mask: Any = None) -> str:
    label_text = str(label or "").strip()
    if not label_text:
        raise ValueError("target label is empty")
    prompt = label_text
    gesture_text = str(gesture or "").strip()
    if gesture_text:
        prompt += f":{gesture_text}"
    if mask is not None:
        prompt += f"@{list(mask)}"
    return prompt


def _resolve_shift_prop(data: list[float]) -> tuple[list[float], float]:
    vals = [float(v) for v in data]
    if not vals:
        return [0.0, 0.0, 0.0], 0.5
    if len(vals) < 3:
        raise ValueError(f"Target data must be []/[dx,dy,dz]/[dx,dy,dz,shift], got len={len(vals)}")
    offset = vals[:3]
    shift_prop = float(vals[3]) if len(vals) >= 4 else 0.5
    return offset, shift_prop


def _split_shared_data(text_list: list[str], data: list[float]) -> list[list[float]]:
    vals = [float(v) for v in data]
    if not vals:
        return [[] for _ in text_list]
    if len(vals) in (3, 4):
        return [list(vals) for _ in text_list]
    if len(vals) == len(text_list) * 3:
        return [vals[i * 3 : (i + 1) * 3] for i in range(len(text_list))]
    if len(vals) == len(text_list) * 4:
        return [vals[i * 4 : (i + 1) * 4] for i in range(len(text_list))]
    raise ValueError(
        "text/data compatibility mode expects data len in {0,3,4,len(text)*3,len(text)*4}, "
        f"got len(text)={len(text_list)}, len(data)={len(vals)}"
    )


def _normalize_target_dict(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    target_id = str(raw.get("target_id") or raw.get("role") or f"target_{idx}")
    role = str(raw.get("role") or target_id)
    prompt = raw.get("prompt")
    if prompt is None:
        prompt = raw.get("text")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
    if prompt is None:
        prompt = _build_prompt_from_parts(raw.get("label"), raw.get("gesture"), raw.get("mask"))
    data = raw.get("data")
    if data is None:
        offset = raw.get("offset") or raw.get("offsets") or []
        shift_prop = raw.get("shift_prop")
        data = list(offset)
        if shift_prop is not None:
            data = list(data)[:3] + [float(shift_prop)]
    if data is None:
        data = []
    offset, shift_prop = _resolve_shift_prop(list(data))
    return {
        "order": int(raw.get("order", idx)),
        "target_id": target_id,
        "role": role,
        "prompt": str(prompt).strip(),
        "offset": [float(v) for v in offset],
        "shift_prop": float(shift_prop),
    }


def _normalize_targets(payload: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    if isinstance(payload.get("targets"), list) and payload["targets"]:
        for idx, raw in enumerate(payload["targets"]):
            if not isinstance(raw, dict):
                raise ValueError("targets[] must be objects")
            targets.append(_normalize_target_dict(raw, idx))
        return targets
    have_pick = any(key in payload for key in ("pick_prompt", "pick_label", "pick_text"))
    have_place = any(key in payload for key in ("place_prompt", "place_label", "place_text"))
    if have_pick or have_place:
        if have_pick:
            prompt = payload.get("pick_prompt") or payload.get("pick_text")
            if prompt is None:
                prompt = _build_prompt_from_parts(
                    payload.get("pick_label"),
                    payload.get("pick_gesture"),
                    payload.get("pick_mask"),
                )
            data = payload.get("pick_data") or []
            targets.append(_normalize_target_dict({"role": "pick", "target_id": "pick", "prompt": prompt, "data": data}, len(targets)))
        if have_place:
            prompt = payload.get("place_prompt") or payload.get("place_text")
            if prompt is None:
                prompt = _build_prompt_from_parts(
                    payload.get("place_label"),
                    payload.get("place_gesture"),
                    payload.get("place_mask"),
                )
            data = payload.get("place_data") or []
            targets.append(_normalize_target_dict({"role": "place", "target_id": "place", "prompt": prompt, "data": data}, len(targets)))
        return targets
    text_list = payload.get("text")
    if isinstance(text_list, str):
        text_list = [text_list]
    if isinstance(text_list, list) and text_list:
        prompt_texts = [str(v).strip() for v in text_list]
        grouped_data = _split_shared_data(prompt_texts, list(payload.get("data") or []))
        for idx, (prompt, data) in enumerate(zip(prompt_texts, grouped_data, strict=False)):
            targets.append(_normalize_target_dict({"target_id": f"target_{idx}", "role": f"target_{idx}", "prompt": prompt, "data": data}, idx))
        return targets
    raise ValueError("Request must provide targets[] or pick/place fields or text/data.")


def _apply_visual_mask_to_qwen_image(rgb_img: Image.Image, visual_mask: Any) -> Image.Image:
    if visual_mask is None:
        return rgb_img
    reasoning_rgb_img = Image.new(rgb_img.mode, rgb_img.size, 0)
    crop = rgb_img.crop(visual_mask)
    reasoning_rgb_img.paste(crop, visual_mask[:2])
    return reasoning_rgb_img


class QwenBase:
    def __init__(self, model: str | None = None):
        self.model = model or QWEN_MODEL_ID

    @staticmethod
    def encode_image(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def remote_call(self, query: str, imgs: list[Image.Image] | None = None) -> str:
        image_parts = []
        for img in imgs or []:
            image_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(img)}"}})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DEFUALT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        *image_parts,
                        {"type": "text", "text": "<OUTPUT TEMPLATE>" + GROUNDING_OUTPUT_TEMPLATE + "</OUTPUT TEMPLATE>"},
                        {"type": "text", "text": query},
                    ],
                },
            ],
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "enable_thinking": False,
        }
        headers = {"Authorization": f"Bearer {QWEN_AUTH_TOKEN}", "Content-Type": "application/json"}
        response = requests.post(QWEN_CHAT_URL, headers=headers, json=payload, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(f"Qwen service returned {response.status_code}: {response.text}")
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        return _extract_message_content(choices[0].get("message", {}))

    @staticmethod
    def parse_json(json_output: str) -> str:
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i + 1 :]).split("```")[0]
                break
        match = re.search(r"\{[\s\S]*\}", json_output)
        if not match:
            raise ValueError("Cannot find JSON object in Qwen output")
        return match.group()

    @staticmethod
    def extract_bbox(bbox: list[float], img_width: int, img_height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        return (
            int(x1 / 1000 * img_width),
            int(y1 / 1000 * img_height),
            int(x2 / 1000 * img_width),
            int(y2 / 1000 * img_height),
        )


class QwenGrounding(QwenBase):
    def get_response(self, query: str, imgs: list[Image.Image] | None = None) -> dict[str, Any]:
        assert imgs is not None and len(imgs) == 1, "QwenGrounding only supports one image"
        msg_text = self.remote_call(query, imgs)
        try:
            if msg_text == "<FAILED>":
                return {"bbox": None, "response": msg_text, "status": f"cannot find the object: {query}"}
            msg_json = self.parse_json(msg_text)
            msg_dict = json.loads(msg_json.strip())
            bbox = None
            for key, value in msg_dict.items():
                if "bbox" in key or "bbox_2d" in key:
                    bbox = self.extract_bbox(value, imgs[0].width, imgs[0].height)
                    break
            return {"bbox": bbox, "response": msg_text, "status": "success"}
        except Exception as exc:
            return {"bbox": None, "response": msg_text, "status": f"response error: {exc}"}


def _call_sam3_mask(image: Image.Image, bbox: tuple[int, int, int, int], sam3_url: str, timeout_s: float) -> dict[str, Any]:
    payload = {"image": QwenBase.encode_image(image), "box_prompt": [int(v) for v in bbox]}
    out: dict[str, Any] = {"ok": False, "latency_ms": 0, "num_detections": 0, "error": "", "mask": None, "mask_area_px": 0}
    t0 = time.perf_counter()
    try:
        response = requests.post(
            f"{sam3_url.rstrip('/')}/segment",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=float(timeout_s),
        )
    except requests.RequestException as exc:
        out["latency_ms"] = _elapsed_ms(t0)
        out["error"] = f"sam3_request_error: {exc}"
        return out
    out["latency_ms"] = _elapsed_ms(t0)
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
    if len(detections) <= 0 or not isinstance(detections[0], dict):
        out["error"] = "sam3_empty_detections"
        return out
    mask_b64 = detections[0].get("mask")
    if not isinstance(mask_b64, str) or not mask_b64:
        out["error"] = "sam3_empty_mask"
        return out
    try:
        raw = base64.b64decode(mask_b64)
        mask_img = Image.open(io.BytesIO(raw)).convert("L")
        width, height = image.size
        if mask_img.size != (width, height):
            mask_img = mask_img.resize((width, height), Image.Resampling.NEAREST)
        mask = np.array(mask_img, dtype=np.uint8) > 0
    except Exception as exc:
        out["error"] = f"sam3_mask_decode_error: {exc}"
        return out
    if not np.any(mask):
        out["error"] = "sam3_empty_mask"
        return out
    out["ok"] = True
    out["mask"] = mask
    out["mask_area_px"] = int(mask.sum())
    return out


def _bbox_center_with_shift(bbox: list[int], shift_prop: float) -> list[int]:
    return [
        int((bbox[0] + bbox[2]) // 2),
        int(round(bbox[1] * (1.0 - float(shift_prop)) + bbox[3] * float(shift_prop))),
    ]


def compute_2d_eye_quat(mask_bool: np.ndarray) -> dict[str, Any] | None:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    points_base = np.stack([xs, ys], dtype=np.float32).T
    center = points_base.mean(axis=0)
    pts_centered = points_base - center
    _, s, vt = np.linalg.svd(pts_centered, full_matrices=False)
    ss = s**2
    r_square = ss[0] / ss.sum()
    x_axis = np.append(vt[0], 0.0)
    if x_axis[1] < 0:
        x_axis = -x_axis
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    eye_quat = _quat_from_matrix(np.array([x_axis, y_axis, z_axis]).T)
    dx, dy = x_axis[0], x_axis[1]
    delta_angle = math.degrees(-math.atan2(dx, dy))
    return {
        "center_base": center.tolist(),
        "eye_angles": delta_angle,
        "eye_quat": eye_quat.tolist(),
        "long_axis": x_axis.tolist(),
        "r_square": float(r_square),
    }


def _eye_quat_to_base_quat(eye_quat: np.ndarray, rotation_cam_to_base: np.ndarray) -> np.ndarray:
    eye_matrix = _quat_to_matrix(np.asarray(eye_quat, dtype=np.float64))
    base_matrix = np.asarray(rotation_cam_to_base, dtype=np.float64) @ eye_matrix
    return _quat_from_matrix(base_matrix).astype(np.float32)


def _transform_point_cam_to_base(point: np.ndarray, rotation: np.ndarray, shift: np.ndarray) -> np.ndarray:
    pt = np.asarray(point, dtype=np.float32).reshape(3)
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sh = np.asarray(shift, dtype=np.float64).reshape(3)
    return (rot @ pt.astype(np.float64) + sh).astype(np.float32)


def _pick_nearest_valid_pixel(valid_mask: np.ndarray, center_pixel: list[int]) -> tuple[int, int] | None:
    img_h, img_w = valid_mask.shape[:2]
    cx = int(max(0, min(img_w - 1, center_pixel[0])))
    cy = int(max(0, min(img_h - 1, center_pixel[1])))
    ys, xs = np.where(valid_mask)
    if xs.size <= 0:
        return None
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    dist = np.linalg.norm(coords - np.array([[cx, cy]], dtype=np.float32), axis=1)
    idx = int(np.argmin(dist))
    return int(coords[idx, 0]), int(coords[idx, 1])


def _fit_rectangle_pose_base(points_base: np.ndarray) -> dict[str, Any]:
    pts = _finite_points(points_base)
    if pts.shape[0] < 4:
        raise RuntimeError("Need at least 4 valid base-frame target points to fit a rectangle.")
    xy = pts[:, :2].astype(np.float64, copy=False)
    z_ref = float(np.median(pts[:, 2].astype(np.float64, copy=False)))
    center_seed = np.median(xy, axis=0)
    centered = xy - center_seed.reshape((1, 2))
    cov = centered.T @ centered / float(max(1, xy.shape[0]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    main_axis = eigvecs[:, order[0]]
    yaw = float(math.atan2(float(main_axis[1]), float(main_axis[0])) % math.pi)

    def _project_with_yaw(yaw_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        ex = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float64)
        ey = np.array([-math.sin(yaw_rad), math.cos(yaw_rad)], dtype=np.float64)
        proj_x = xy @ ex
        proj_y = xy @ ey
        x0, x1 = np.quantile(proj_x, [0.04, 0.96])
        y0, y1 = np.quantile(proj_y, [0.04, 0.96])
        length = float(x1 - x0)
        width = float(y1 - y0)
        center_xy = ex * (0.5 * (float(x1) + float(x0))) + ey * (0.5 * (float(y1) + float(y0)))
        return ex, ey, center_xy.astype(np.float64), length, width

    ex, ey, center_xy, length, width = _project_with_yaw(yaw)
    if width > length:
        yaw = float((yaw + 0.5 * math.pi) % math.pi)
        ex, ey, center_xy, length, width = _project_with_yaw(yaw)
    aspect_ratio = float(length / max(width, 1e-6))
    center_base = np.array([float(center_xy[0]), float(center_xy[1]), z_ref], dtype=np.float32)
    return {
        "center_base": center_base,
        "yaw": float(yaw),
        "length": float(length),
        "width": float(width),
        "aspect_ratio": aspect_ratio,
        "principal_axis_base_xy": [float(ex[0]), float(ex[1])],
        "secondary_axis_base_xy": [float(ey[0]), float(ey[1])],
    }


def _erode_mask_for_fit(mask: np.ndarray) -> tuple[np.ndarray, int]:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not np.any(m):
        return m.copy(), 0
    ys, xs = np.nonzero(m)
    bbox_h = int(np.max(ys) - np.min(ys) + 1)
    bbox_w = int(np.max(xs) - np.min(xs) + 1)
    erode_px = int(np.clip(round(0.015 * float(min(bbox_h, bbox_w))), 1, 5))
    if erode_px <= 0:
        return m.copy(), 0
    kernel_size = int(2 * erode_px + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(m.astype(np.uint8), kernel, iterations=1) > 0
    if int(np.count_nonzero(eroded)) < 24:
        return m.copy(), 0
    return eroded, erode_px


def _keep_largest_xy_component(points_base: np.ndarray, voxel_size: float) -> tuple[np.ndarray, dict[str, Any]]:
    pts = _finite_points(points_base)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), {"component_count": 0, "largest_component_point_count": 0, "voxel_size_m": float(voxel_size), "used": False}
    xy = pts[:, :2].astype(np.float64, copy=False)
    origin = np.min(xy, axis=0)
    voxel = max(1e-4, float(voxel_size))
    coords = np.floor((xy - origin.reshape((1, 2))) / voxel).astype(np.int32)
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    voxel_point_counts = np.bincount(inverse, minlength=unique_coords.shape[0]).astype(np.int32)
    coord_to_idx = {(int(c[0]), int(c[1])): int(idx) for idx, c in enumerate(unique_coords)}
    visited = np.zeros((unique_coords.shape[0],), dtype=bool)
    components: list[tuple[list[int], int]] = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for start_idx in range(unique_coords.shape[0]):
        if visited[start_idx]:
            continue
        stack = [int(start_idx)]
        visited[start_idx] = True
        voxel_indices: list[int] = []
        point_count = 0
        while stack:
            cur = stack.pop()
            voxel_indices.append(cur)
            point_count += int(voxel_point_counts[cur])
            cx, cy = unique_coords[cur]
            for dx, dy in neighbors:
                nxt = coord_to_idx.get((int(cx + dx), int(cy + dy)))
                if nxt is None or visited[nxt]:
                    continue
                visited[nxt] = True
                stack.append(int(nxt))
        components.append((voxel_indices, point_count))
    best_voxels, best_count = max(components, key=lambda item: item[1])
    keep_voxel_mask = np.zeros((unique_coords.shape[0],), dtype=bool)
    keep_voxel_mask[np.asarray(best_voxels, dtype=np.int32)] = True
    keep_point_mask = keep_voxel_mask[inverse]
    filtered = pts[keep_point_mask]
    return filtered.astype(np.float32, copy=False), {
        "component_count": int(len(components)),
        "largest_component_point_count": int(best_count),
        "voxel_size_m": float(voxel),
        "used": True,
    }


def _surface_z_from_plane_at_xy(plane_normal: np.ndarray, plane_d: float, xy: np.ndarray) -> float | None:
    normal = np.asarray(plane_normal, dtype=np.float64).reshape(-1)
    q = np.asarray(xy, dtype=np.float64).reshape(-1)
    if normal.shape[0] != 3 or q.shape[0] != 2 or abs(float(normal[2])) <= 1e-6:
        return None
    z = -(float(normal[0]) * float(q[0]) + float(normal[1]) * float(q[1]) + float(plane_d)) / float(normal[2])
    if not np.isfinite(z):
        return None
    return float(z)


def _plane_margin_z(points_base: np.ndarray, normal_base: np.ndarray, d_base: float) -> np.ndarray:
    pts = _finite_points(points_base).astype(np.float64, copy=False)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    n = np.asarray(normal_base, dtype=np.float64).reshape(3)
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
    rot2 = _rot2(float(yaw_rad))
    footprint_local = np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]], dtype=np.float64)
    heights: list[float] = []
    for local_xy in footprint_local:
        xy = center + rot2 @ local_xy
        z = _surface_z_from_plane_at_xy(normal_base, float(d_base), xy)
        if z is not None and np.isfinite(z):
            heights.append(float(z))
    if not heights:
        z0 = _surface_z_from_plane_at_xy(normal_base, float(d_base), center)
        if z0 is None or not np.isfinite(z0):
            raise RuntimeError("Failed to evaluate table plane under footprint.")
        return float(z0)
    return float(max(heights))


def _fit_plane_ransac(points: np.ndarray, threshold: float, max_iters: int, min_inliers: int) -> tuple[np.ndarray, float, np.ndarray] | None:
    pts = _finite_points(points).astype(np.float64, copy=False)
    if pts.shape[0] < 3:
        return None
    min_inliers = int(max(3, min(min_inliers, pts.shape[0])))
    rng = np.random.default_rng(0)
    best_mask: np.ndarray | None = None
    best_count = 0
    best_error = float("inf")
    for _ in range(int(max_iters)):
        sample_idx = rng.choice(pts.shape[0], 3, replace=False)
        p0, p1, p2 = pts[sample_idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm < 1e-8:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p0))
        dist = np.abs(pts @ normal + d)
        mask = dist <= float(threshold)
        count = int(np.count_nonzero(mask))
        if count < min_inliers:
            continue
        mean_error = float(np.mean(dist[mask]))
        if count > best_count or (count == best_count and mean_error < best_error):
            best_mask = mask
            best_count = count
            best_error = mean_error
    if best_mask is None or best_count < min_inliers:
        return None
    inliers = pts[best_mask]
    centroid = np.mean(inliers, axis=0)
    centered = inliers - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float64, copy=False)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
    d = -float(np.dot(normal, centroid))
    if normal[2] < 0.0:
        normal = -normal
        d = -d
    refined_dist = np.abs(pts @ normal + d)
    refined_mask = refined_dist <= float(threshold * 1.25)
    if int(np.count_nonzero(refined_mask)) < min_inliers:
        return None
    return normal.astype(np.float32), float(d), refined_mask


def _fit_local_plane_from_bbox(dense_map_base: np.ndarray, bbox: tuple[int, int, int, int], object_mask: np.ndarray) -> PlaneModel:
    img_h, img_w = dense_map_base.shape[:2]
    region_bbox = _expand_bbox(bbox, img_w, img_h)
    region_mask = _bbox_to_mask(region_bbox, img_w, img_h)
    valid = _finite_point_mask(dense_map_base.reshape(-1, 3)).reshape(img_h, img_w)
    keep = region_mask & (~np.asarray(object_mask, dtype=bool)) & valid
    points_base = _finite_points(dense_map_base[keep])
    if points_base.shape[0] < DEFAULT_TABLE_RANSAC_MIN_POINTS:
        raise RuntimeError("Not enough local support points outside the object mask to fit the table plane.")
    fitted = _fit_plane_ransac(
        points_base,
        threshold=float(DEFAULT_PLANE_RANSAC_THR_M),
        max_iters=int(DEFAULT_TABLE_RANSAC_ITERS),
        min_inliers=max(DEFAULT_TABLE_RANSAC_MIN_POINTS, int(round(0.18 * float(points_base.shape[0])))),
    )
    if fitted is None:
        raise RuntimeError("Local table plane RANSAC failed.")
    normal, d, _inlier_mask = fitted
    return PlaneModel(normal_base=np.asarray(normal, dtype=np.float32), d_base=float(d))


def _fit_pick_object_box(dense_map_base: np.ndarray, bbox: tuple[int, int, int, int], object_mask: np.ndarray, plane_model: PlaneModel) -> dict[str, Any]:
    valid = _finite_point_mask(dense_map_base.reshape(-1, 3)).reshape(dense_map_base.shape[:2])
    target_points_base = _finite_points(dense_map_base[np.asarray(object_mask, dtype=bool) & valid])
    if target_points_base.shape[0] < 24:
        raise RuntimeError("Segmented object cloud is too small.")

    target_margin = _plane_margin_z(target_points_base, plane_model.normal_base, plane_model.d_base)
    target_keep_above = target_margin > -0.003
    if int(np.count_nonzero(target_keep_above)) >= 24:
        target_points_base = target_points_base[target_keep_above]
    rect_target = _fit_rectangle_pose_base(target_points_base)

    fit_mask, fit_mask_erode_px = _erode_mask_for_fit(object_mask)
    fit_points_base_pre = _finite_points(dense_map_base[fit_mask & valid])
    fit_mask_selection_reason = "eroded_mask" if fit_mask_erode_px > 0 else "full_mask"
    if fit_points_base_pre.shape[0] < 24:
        fit_points_base_pre = target_points_base.copy()
        fit_mask_selection_reason = "full_mask_fallback"
        fit_mask_erode_px = 0

    rect_fit_seed = _fit_rectangle_pose_base(fit_points_base_pre)
    slender_mask_truncated = (
        fit_mask_erode_px > 0
        and float(rect_target.get("aspect_ratio", 0.0)) >= float(DEFAULT_SLENDER_OBJECT_ASPECT_RATIO)
        and float(rect_fit_seed.get("length", 0.0)) < float(DEFAULT_SLENDER_OBJECT_LENGTH_RECOVER_RATIO) * max(float(rect_target.get("length", 0.0)), 1e-6)
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
        rect_after_clearance = _fit_rectangle_pose_base(fit_points_after_clearance_candidate)
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
    fit_points_base_filtered, fit_component_debug = _keep_largest_xy_component(fit_points_base_pre, fit_component_voxel_size)
    min_fit_keep_count = max(24, int(round(0.35 * float(fit_points_base_pre.shape[0]))))
    fit_cluster_used = bool(fit_points_base_filtered.shape[0] >= min_fit_keep_count)
    fit_points_base = fit_points_base_filtered if fit_cluster_used else fit_points_base_pre

    rect_pre = _fit_rectangle_pose_base(fit_points_base_pre)
    rect_filtered: dict[str, Any] | None = None
    fit_selection_reason = "largest_component" if fit_cluster_used else "pre_component_union"
    if fit_cluster_used:
        rect_filtered = _fit_rectangle_pose_base(fit_points_base_filtered)
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
            fit_points_base = fit_points_base_pre
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
    support_z_max = _max_plane_height_under_footprint(center_xy, yaw, np.array([length, width], dtype=np.float32), plane_model.normal_base, plane_model.d_base)
    height_source_points = fit_points_base if fit_points_base.shape[0] >= 24 else target_points_base
    local_heights = height_source_points[:, 2].astype(np.float64, copy=False) - float(support_z_max)
    local_heights = local_heights[np.isfinite(local_heights)]
    height_points_above = local_heights[local_heights > -0.002]
    height_quantile = float(DEFAULT_SLENDER_OBJECT_HEIGHT_QUANTILE if is_slender_object else DEFAULT_OBJECT_HEIGHT_QUANTILE)
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
        fallback_height = float(_safe_quantile(target_points_base[:, 2], 0.98, 0.0) - _safe_quantile(target_points_base[:, 2], 0.02, 0.0))
        if is_slender_object and np.isfinite(fallback_height):
            fallback_height = float(min(fallback_height, float(width) * float(DEFAULT_SLENDER_OBJECT_THICKNESS_CAP_RATIO)))
        raw_height = max(float(min_height), float(fallback_height))
    height = float(np.clip(raw_height, min_height, DEFAULT_OBJECT_MAX_HEIGHT_M))

    center_base = np.array([float(center_xy[0]), float(center_xy[1]), float(support_z_max + 0.5 * height)], dtype=np.float32)
    axes_base = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    size_xyz = np.array([float(length), float(width), float(height)], dtype=np.float32)
    default_local = placedof_planner.make_default_grasp_pose_local(placedof_planner.OBBSpec(size_xyz=size_xyz))
    return {
        "bbox": [int(v) for v in bbox],
        "mask": np.asarray(object_mask, dtype=bool),
        "mask_area_px": int(np.count_nonzero(object_mask)),
        "points_base": target_points_base.astype(np.float32, copy=False),
        "fit_points_base": fit_points_base.astype(np.float32, copy=False),
        "plane_model": plane_model,
        "center_base": center_base.astype(np.float32, copy=True),
        "axes_base": _orthonormalize_axes(axes_base),
        "size_xyz": size_xyz.astype(np.float32, copy=True),
        "grasp_origin_local": np.asarray(default_local.origin_local, dtype=np.float32),
        "grasp_axes_local": np.asarray(default_local.axes_local, dtype=np.float32),
        "debug": {
            "fit_mask_erode_px": int(fit_mask_erode_px),
            "fit_mask_selection_reason": str(fit_mask_selection_reason),
            "fit_cluster_used": bool(fit_cluster_used),
            "fit_selection_reason": str(fit_selection_reason),
            "fit_plane_clearance_used": bool(fit_plane_clearance_used),
            "fit_plane_clearance_reason": str(fit_plane_clearance_reason),
            "fit_cluster_component_count": int(fit_component_debug.get("component_count", 0)),
            "fit_cluster_largest_component_point_count": int(fit_component_debug.get("largest_component_point_count", 0)),
            "fit_cluster_voxel_size_m": float(fit_component_debug.get("voxel_size_m", fit_component_voxel_size)),
            "raw_object_points": int(target_points_base.shape[0]),
            "fit_points": int(fit_points_base.shape[0]),
            "rect_aspect_ratio": float(rect.get("aspect_ratio", 0.0)),
            "object_is_slender": bool(is_slender_object),
            "support_z_max": float(support_z_max),
        },
    }


def _finalize_pick_pose(
    stage_result: dict[str, Any],
    dense_map_cam: np.ndarray,
    rot_cb: np.ndarray,
    shift_cb: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(stage_result["mask"], dtype=bool)
    bbox = [int(v) for v in stage_result["bbox"]]
    center_pixel = [int(v) for v in stage_result["center_pixel"]]
    pruned_mask = cv2.erode(mask.astype(np.uint8), np.ones((10, 10), dtype=np.uint8), iterations=1) > 0
    if not np.any(pruned_mask):
        pruned_mask = mask.copy()
    dense_valid = _valid_cam_xyz_mask_map(dense_map_cam)
    valid_mask = pruned_mask & dense_valid
    if not np.any(valid_mask):
        valid_mask = mask & dense_valid
    if not np.any(valid_mask):
        bbox_mask = _bbox_to_mask(tuple(bbox), dense_map_cam.shape[1], dense_map_cam.shape[0])
        valid_mask = bbox_mask & dense_valid
    nearest_pixel = _pick_nearest_valid_pixel(valid_mask, center_pixel)
    if nearest_pixel is None:
        raise RuntimeError(f"No valid S2M2 point for [{stage_result['label']}]")
    ys, xs = np.where(valid_mask)
    cam_points = dense_map_cam[ys, xs].astype(np.float32)
    cam_points = cam_points[np.all(np.isfinite(cam_points), axis=1)]
    cam_point = dense_map_cam[nearest_pixel[1], nearest_pixel[0]].astype(np.float32) if cam_points.shape[0] <= 0 else np.mean(cam_points, axis=0).astype(np.float32)
    base3d = _transform_point_cam_to_base(cam_point, rot_cb, shift_cb)
    orientation = stage_result["orientation"]
    label = stage_result["label"]
    if orientation == "dof":
        quat = stage_result["dof_quat"].astype(np.float32)
    elif orientation in PREDEFINED_QUATS:
        quat = np.asarray(PREDEFINED_QUATS[orientation], dtype=np.float32)
    else:
        quat = np.asarray(PREDEFINED_QUATS["vertical"], dtype=np.float32)
    if "笔" in label or "pen" in label:
        quat = np.asarray(PREDEFINED_QUATS["pencil_quat"], dtype=np.float32)
    offset = np.asarray(stage_result["offset"], dtype=np.float32)
    out_xquat = np.concatenate([base3d + offset, quat]).astype(np.float32)
    return {
        "xquat": out_xquat,
        "bbox": bbox,
        "pixel": [int(nearest_pixel[0]), int(nearest_pixel[1])],
        "base3d": [float(v) for v in base3d.tolist()],
        "quat": [float(v) for v in quat.tolist()],
        "offset": [float(v) for v in offset.tolist()],
        "dof_r_square": float(stage_result["dof_r_square"]),
    }


def _evaluate_points_camera_visibility(
    points_base: np.ndarray,
    k1: np.ndarray,
    rot_cb: np.ndarray,
    shift_cb: np.ndarray,
    image_size: tuple[int, int],
    pixel_margin_px: int = DEFAULT_PLACE_CAMERA_PIXEL_MARGIN_PX,
) -> dict[str, Any]:
    pts = _finite_points(points_base).astype(np.float64, copy=False)
    if pts.shape[0] == 0:
        return {"available": False}
    pts_cam = (pts - np.asarray(shift_cb, dtype=np.float64).reshape((1, 3))) @ np.asarray(rot_cb, dtype=np.float64).reshape(3, 3)
    depth = pts_cam[:, 2]
    fx = float(k1[0, 0])
    fy = float(k1[1, 1])
    cx = float(k1[0, 2])
    cy = float(k1[1, 2])
    valid_depth = np.isfinite(depth) & (depth > DEFAULT_MIN_CAM_DEPTH)
    u = np.full((pts_cam.shape[0],), np.nan, dtype=np.float64)
    v = np.full((pts_cam.shape[0],), np.nan, dtype=np.float64)
    u[valid_depth] = fx * pts_cam[valid_depth, 0] / depth[valid_depth] + cx
    v[valid_depth] = fy * pts_cam[valid_depth, 1] / depth[valid_depth] + cy
    width, height = image_size
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
    }


def _evaluate_box_camera_visibility(
    center_base: np.ndarray,
    axes_base: np.ndarray,
    size_xyz: np.ndarray,
    k1: np.ndarray,
    rot_cb: np.ndarray,
    shift_cb: np.ndarray,
    image_size: tuple[int, int],
    pixel_margin_px: int = DEFAULT_PLACE_CAMERA_PIXEL_MARGIN_PX,
) -> dict[str, Any]:
    corners = placedof_planner._box_corners_world(center_base, axes_base, size_xyz).astype(np.float64, copy=False)
    return _evaluate_points_camera_visibility(corners, k1, rot_cb, shift_cb, image_size, pixel_margin_px=pixel_margin_px)


def _evaluate_grasp_camera_visibility(
    gripper_origin_base: np.ndarray,
    k1: np.ndarray,
    rot_cb: np.ndarray,
    shift_cb: np.ndarray,
    image_size: tuple[int, int],
    pixel_margin_px: int = DEFAULT_PLACE_CAMERA_PIXEL_MARGIN_PX,
) -> dict[str, Any]:
    point = np.asarray(gripper_origin_base, dtype=np.float32).reshape(1, 3)
    return _evaluate_points_camera_visibility(point, k1, rot_cb, shift_cb, image_size, pixel_margin_px=pixel_margin_px)


def _evaluate_place_camera_visibility(
    center_base: np.ndarray,
    axes_base: np.ndarray,
    size_xyz: np.ndarray,
    gripper_origin_base: np.ndarray,
    k1: np.ndarray,
    rot_cb: np.ndarray,
    shift_cb: np.ndarray,
    image_size: tuple[int, int],
    pixel_margin_px: int = DEFAULT_PLACE_CAMERA_PIXEL_MARGIN_PX,
) -> dict[str, Any]:
    object_vis = _evaluate_box_camera_visibility(
        center_base,
        axes_base,
        size_xyz,
        k1,
        rot_cb,
        shift_cb,
        image_size,
        pixel_margin_px=pixel_margin_px,
    )
    grasp_vis = _evaluate_grasp_camera_visibility(
        gripper_origin_base,
        k1,
        rot_cb,
        shift_cb,
        image_size,
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
    center_base: np.ndarray,
    axes_base: np.ndarray,
    size_xyz: np.ndarray,
    profile: placedof_planner.TargetProfile,
) -> tuple[np.ndarray, dict[str, Any]]:
    if profile.opening_face is None or profile.opening_normal_base is None:
        return np.zeros((3,), dtype=np.float32), {"applied": False, "reason": "no_opening_face"}
    face = str(profile.opening_face).strip().lower()
    axis_idx = {"x": 0, "y": 1, "z": 2}[face[1]]
    sign = 1.0 if face[0] == "+" else -1.0
    corners = placedof_planner._box_corners_world(center_base, axes_base, size_xyz).astype(np.float64, copy=False)
    local = profile.world_to_local(corners).astype(np.float64, copy=False)
    opening_coord = sign * local[:, axis_idx]
    opening_plane = 0.5 * float(profile.size_xyz[axis_idx])
    object_extent = float(np.max(opening_coord) - np.min(opening_coord))
    dims = np.sort(np.asarray(size_xyz, dtype=np.float64).reshape(3))
    slender_ratio = float(dims[2] / max(dims[1], 1e-4))
    exposure_ratio = float(DEFAULT_PLACE_OPENING_EXPOSURE_SLENDER_RATIO if slender_ratio >= DEFAULT_SLENDER_OBJECT_ASPECT_RATIO else DEFAULT_PLACE_OPENING_EXPOSURE_RATIO)
    required_exposure = float(np.clip(max(DEFAULT_PLACE_OPENING_EXPOSURE_MIN_M, exposure_ratio * object_extent), 0.0, min(DEFAULT_PLACE_OPENING_SHIFT_MAX_M, max(object_extent, DEFAULT_PLACE_OPENING_EXPOSURE_MIN_M))))
    current_exposure = float(np.max(opening_coord) - opening_plane)
    shift_amount = float(np.clip(required_exposure - current_exposure, 0.0, DEFAULT_PLACE_OPENING_SHIFT_MAX_M))
    if not np.isfinite(shift_amount) or shift_amount <= 1e-6:
        return np.zeros((3,), dtype=np.float32), {"applied": False, "opening_face": face}
    delta = (np.asarray(profile.opening_normal_base, dtype=np.float64).reshape(3) * float(shift_amount)).astype(np.float32, copy=False)
    return delta, {"applied": True, "opening_face": face, "shift_amount_m": float(shift_amount), "delta_base": [float(v) for v in delta.tolist()]}


def _has_box_point_collision(center: np.ndarray, axes_base: np.ndarray, size_xyz: np.ndarray, collision_points_base: np.ndarray) -> bool:
    if collision_points_base.shape[0] == 0:
        return False
    local = (np.asarray(collision_points_base, dtype=np.float64) - np.asarray(center, dtype=np.float64).reshape((1, 3))) @ np.asarray(axes_base, dtype=np.float64)
    half = 0.5 * np.asarray(size_xyz, dtype=np.float64).reshape(1, 3)
    inside = np.all(np.abs(local) <= (half + float(DEFAULT_COLLISION_MARGIN_M)), axis=1)
    return bool(np.any(inside))


def _has_box_plane_collision(center: np.ndarray, axes_base: np.ndarray, size_xyz: np.ndarray, plane_model: PlaneModel | None) -> bool:
    if plane_model is None:
        return False
    corners = placedof_planner._box_corners_world(center, axes_base, size_xyz)
    margins = _plane_margin_z(corners, plane_model.normal_base, plane_model.d_base)
    return bool(np.any(margins < -float(DEFAULT_PLANE_CONTACT_EPS_M)))


def _post_adjust_place_plan_visibility(
    object_center_base: np.ndarray,
    object_axes_base: np.ndarray,
    object_size_xyz: np.ndarray,
    gripper_origin_base: np.ndarray,
    rule_id: str,
    profile: placedof_planner.TargetProfile,
    collision_points_base: np.ndarray,
    plane_model: PlaneModel | None,
    k1: np.ndarray,
    rot_cb: np.ndarray,
    shift_cb: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    center = np.asarray(object_center_base, dtype=np.float32).reshape(3).copy()
    axes = np.asarray(object_axes_base, dtype=np.float32).reshape(3, 3)
    size = np.asarray(object_size_xyz, dtype=np.float32).reshape(3)
    gripper = np.asarray(gripper_origin_base, dtype=np.float32).reshape(3).copy()
    debug: dict[str, Any] = {"rule_id": str(rule_id), "before": _evaluate_place_camera_visibility(center, axes, size, gripper, k1, rot_cb, shift_cb, image_size)}
    container_rule = str(rule_id) in {"top_open_box", "front_open_cabinet"}
    opening_delta = np.zeros((3,), dtype=np.float32)
    opening_debug: dict[str, Any] = {"applied": False, "reason": "non_container_rule"}
    opening_normal = None
    if container_rule:
        opening_delta, opening_debug = _compute_place_opening_visibility_shift(center, axes, size, profile)
        if profile.opening_normal_base is not None:
            opening_normal = np.asarray(profile.opening_normal_base, dtype=np.float32).reshape(3)
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

    def _trial_collision(trial_center: np.ndarray) -> bool:
        if _has_box_point_collision(trial_center, axes, size, collision_points_base):
            return True
        return _has_box_plane_collision(trial_center, axes, size, plane_model)

    for open_idx in range(opening_steps + 1):
        extra_open = float(open_idx * opening_step)
        extra_open_delta = np.zeros((3,), dtype=np.float32)
        if opening_normal is not None and extra_open > 1e-6:
            extra_open_delta = (opening_normal.astype(np.float64) * extra_open).astype(np.float32, copy=False)
        for lift_idx in range(lift_steps + 1):
            lift = float(lift_idx * lift_step)
            lift_delta = np.array([0.0, 0.0, lift], dtype=np.float32)
            total_delta = (opening_delta.astype(np.float64) + extra_open_delta.astype(np.float64) + lift_delta.astype(np.float64)).astype(np.float32, copy=False)
            trial_center = (center.astype(np.float64) + total_delta.astype(np.float64)).astype(np.float32, copy=False)
            if float(np.linalg.norm(total_delta.astype(np.float64))) > 1e-6 and _trial_collision(trial_center):
                continue
            trial_gripper = (gripper.astype(np.float64) + total_delta.astype(np.float64)).astype(np.float32, copy=False)
            trial_visibility = _evaluate_place_camera_visibility(trial_center, axes, size, trial_gripper, k1, rot_cb, shift_cb, image_size)
            trial_key = (
                int(bool(trial_visibility.get("all_in_frame", False))),
                int(trial_visibility.get("object", {}).get("in_frame_count", 0)) + int(trial_visibility.get("grasp", {}).get("in_frame_count", 0)),
                int(trial_visibility.get("object", {}).get("valid_depth_count", 0)) + int(trial_visibility.get("grasp", {}).get("valid_depth_count", 0)),
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
    debug["opening_adjustment"] = dict(opening_debug)
    debug["search"] = {"opening_steps": int(opening_steps + 1), "lift_steps": int(lift_steps + 1)}
    if opening_normal is not None:
        total_opening_delta = (opening_delta.astype(np.float64) + opening_normal.astype(np.float64) * float(best_open_extra)).astype(np.float32, copy=False)
        debug["opening_adjustment"]["selected_total_delta_base"] = [float(v) for v in total_opening_delta.tolist()]
        debug["opening_adjustment"]["selected_extra_opening_m"] = float(best_open_extra)
    debug["lift_adjustment"] = {"applied": bool(best_lift > 1e-6), "delta_base": [0.0, 0.0, float(best_lift)]}
    debug["final"] = _evaluate_place_camera_visibility(best_center, axes, size, best_gripper, k1, rot_cb, shift_cb, image_size)
    debug["applied"] = bool(float(np.linalg.norm(total_delta.astype(np.float64))) > 1e-6)
    debug["delta_base"] = [float(v) for v in total_delta.tolist()]
    return best_center.astype(np.float32), best_gripper.astype(np.float32), debug


def _run_qwen_sam3_target(target: dict[str, Any], left: Image.Image, rotation_cam_to_base: np.ndarray, sam3_url: str, timeout_s: float) -> dict[str, Any]:
    label, orientation, need_tracking, visual_mask = parse_prompt(target["prompt"])
    reasoning_rgb = _apply_visual_mask_to_qwen_image(left, visual_mask)
    t_total0 = time.perf_counter()
    t_qwen0 = time.perf_counter()
    qwen_result = QwenGrounding().get_response(label, [reasoning_rgb])
    qwen_ms = _elapsed_ms(t_qwen0)
    if qwen_result["status"] != "success" or qwen_result["bbox"] is None:
        raise RuntimeError(f"Qwen grounding failed for [{label}]: {qwen_result['status']}")
    bbox = [int(v) for v in qwen_result["bbox"]]
    t_sam30 = time.perf_counter()
    sam3_info = _call_sam3_mask(left, tuple(bbox), sam3_url=sam3_url, timeout_s=timeout_s)
    sam3_ms = _elapsed_ms(t_sam30)
    if not sam3_info.get("ok", False):
        raise RuntimeError(f"SAM3 failed for [{label}]: {sam3_info.get('error', 'unknown')}")
    mask = np.asarray(sam3_info["mask"], dtype=bool)
    dof_res = compute_2d_eye_quat(mask)
    dof_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    dof_r_square = 0.0
    if dof_res is not None:
        dof_r_square = float(dof_res["r_square"])
        if dof_r_square >= 0.7:
            dof_quat = _eye_quat_to_base_quat(np.asarray(dof_res["eye_quat"], dtype=np.float32), rotation_cam_to_base)
    return {
        "order": target["order"],
        "target_id": target["target_id"],
        "role": target["role"],
        "prompt": target["prompt"],
        "label": label,
        "orientation": orientation,
        "need_tracking": bool(need_tracking),
        "visual_mask": list(visual_mask) if visual_mask is not None else None,
        "offset": [float(v) for v in target["offset"]],
        "shift_prop": float(target["shift_prop"]),
        "bbox": bbox,
        "center_pixel": _bbox_center_with_shift(bbox, target["shift_prop"]),
        "mask": mask,
        "dof_quat": dof_quat.astype(np.float32),
        "dof_r_square": float(dof_r_square),
        "qwen_status": str(qwen_result["status"]),
        "timings_ms": {"qwen": int(qwen_ms), "sam3": int(sam3_ms), "total": int(_elapsed_ms(t_total0))},
    }


def _fit_place_target_profile(points_base: np.ndarray, kind_hint: str, camera_origin_base: np.ndarray | None) -> tuple[np.ndarray, placedof_planner.TargetProfile, dict[str, Any]]:
    pts = _finite_points(points_base)
    if pts.shape[0] < 24:
        raise RuntimeError("Too few target points for primitive fitting.")
    q_lo_xy, q_hi_xy = np.quantile(pts[:, :2].astype(np.float64, copy=False), [0.10, 0.90], axis=0)
    robust_xy_spread = np.maximum(q_hi_xy - q_lo_xy, 1e-4)
    fit_component_voxel_size = float(np.clip(0.07 * float(np.min(robust_xy_spread)), 0.0030, 0.0100))
    fit_points_filtered, fit_component_debug = _keep_largest_xy_component(pts, fit_component_voxel_size)
    min_fit_keep_count = max(24, int(round(0.35 * float(pts.shape[0]))))
    fit_cluster_used = bool(fit_points_filtered.shape[0] >= min_fit_keep_count)
    fit_points = fit_points_filtered if fit_cluster_used else pts
    profile = placedof_planner.fit_target_profile(fit_points, kind_hint=kind_hint, camera_origin_base=camera_origin_base)
    debug = {
        "fit_cluster_used": bool(fit_cluster_used),
        "fit_cluster_component_count": int(fit_component_debug.get("component_count", 0)),
        "fit_cluster_largest_component_point_count": int(fit_component_debug.get("largest_component_point_count", 0)),
        "fit_cluster_voxel_size_m": float(fit_component_debug.get("voxel_size_m", fit_component_voxel_size)),
        "fit_points": int(fit_points.shape[0]),
        "fit_primitive_shape": str(profile.primitive_shape),
        "fit_primitive_radius": None if profile.primitive_radius is None else float(profile.primitive_radius),
        "fit_target_kind_auto": str(profile.kind),
    }
    return fit_points.astype(np.float32, copy=False), profile, debug


def _build_collision_points_base(
    dense_map_base: np.ndarray,
    exclude_masks: list[np.ndarray],
    plane_model: PlaneModel | None,
) -> np.ndarray:
    valid = _finite_point_mask(dense_map_base.reshape(-1, 3)).reshape(dense_map_base.shape[:2])
    keep = valid.copy()
    for mask in exclude_masks:
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape == keep.shape:
            keep &= ~mask_bool
    scene_points = _finite_points(dense_map_base[keep])
    if plane_model is not None and scene_points.shape[0] > 0:
        plane_dist = np.abs(scene_points.astype(np.float64, copy=False) @ np.asarray(plane_model.normal_base, dtype=np.float64) + float(plane_model.d_base))
        scene_points = scene_points[plane_dist > float(DEFAULT_TABLE_PLANE_EXCLUDE_M)]
    return scene_points.astype(np.float32, copy=False)


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


def _project_base_points_to_image(points_base: np.ndarray, k1: np.ndarray, rot_cb: np.ndarray, shift_cb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    pts_cam = (pts - np.asarray(shift_cb, dtype=np.float64).reshape((1, 3))) @ np.asarray(rot_cb, dtype=np.float64).reshape(3, 3)
    depth = pts_cam[:, 2]
    valid = np.isfinite(depth) & (depth > DEFAULT_MIN_CAM_DEPTH)
    pixels = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        fx = float(k1[0, 0])
        fy = float(k1[1, 1])
        cx = float(k1[0, 2])
        cy = float(k1[1, 2])
        pixels[valid, 0] = fx * pts_cam[valid, 0] / depth[valid] + cx
        pixels[valid, 1] = fy * pts_cam[valid, 1] / depth[valid] + cy
    finite = valid & np.all(np.isfinite(pixels), axis=1)
    return pixels.astype(np.float32, copy=False), finite.astype(bool, copy=False)


def _draw_projected_polyline(draw: ImageDraw.ImageDraw, pixels: np.ndarray, edges: list[tuple[int, int]], rgb: tuple[int, int, int], width: int = 2, valid_mask: np.ndarray | None = None) -> None:
    pix = np.asarray(pixels, dtype=np.float32).reshape(-1, 2)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1) if valid_mask is not None else np.ones((pix.shape[0],), dtype=bool)
    for i, j in edges:
        if i >= pix.shape[0] or j >= pix.shape[0]:
            continue
        if not (valid[i] and valid[j]):
            continue
        draw.line([(float(pix[i, 0]), float(pix[i, 1])), (float(pix[j, 0]), float(pix[j, 1]))], fill=rgb, width=int(width))


def _draw_projected_box(draw: ImageDraw.ImageDraw, center: np.ndarray, axes: np.ndarray, size_xyz: np.ndarray, k1: np.ndarray, rot_cb: np.ndarray, shift_cb: np.ndarray, rgb: tuple[int, int, int], width: int = 3) -> None:
    corners = _box_corners_axes(center, axes, size_xyz)
    pixels, valid = _project_base_points_to_image(corners, k1, rot_cb, shift_cb)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    _draw_projected_polyline(draw, pixels, edges, rgb, width=width, valid_mask=valid)


def _draw_projected_frame(draw: ImageDraw.ImageDraw, origin: np.ndarray, axes: np.ndarray, axis_len: float, k1: np.ndarray, rot_cb: np.ndarray, shift_cb: np.ndarray) -> None:
    origin3 = np.asarray(origin, dtype=np.float64).reshape(3)
    axes3 = np.asarray(axes, dtype=np.float64).reshape(3, 3)
    pts = np.vstack([
        origin3.reshape((1, 3)),
        origin3.reshape((1, 3)) + axes3[:, 0].reshape((1, 3)) * float(axis_len),
        origin3.reshape((1, 3)) + axes3[:, 1].reshape((1, 3)) * float(axis_len),
        origin3.reshape((1, 3)) + axes3[:, 2].reshape((1, 3)) * float(axis_len),
    ]).astype(np.float32)
    pixels, valid = _project_base_points_to_image(pts, k1, rot_cb, shift_cb)
    if pixels.shape[0] < 4 or not valid[0]:
        return
    for axis_idx in range(3):
        if not valid[axis_idx + 1]:
            continue
        color = AXIS_COLORS_RGB[axis_idx]
        draw.line(
            [(float(pixels[0, 0]), float(pixels[0, 1])), (float(pixels[axis_idx + 1, 0]), float(pixels[axis_idx + 1, 1]))],
            fill=color,
            width=3,
        )


def _draw_projected_cylinder(draw: ImageDraw.ImageDraw, center: np.ndarray, axes: np.ndarray, radius: float, height: float, k1: np.ndarray, rot_cb: np.ndarray, shift_cb: np.ndarray, rgb: tuple[int, int, int]) -> None:
    ctr = np.asarray(center, dtype=np.float64).reshape(3)
    basis = np.asarray(axes, dtype=np.float64).reshape(3, 3)
    ex = basis[:, 0]
    ey = basis[:, 1]
    ez = basis[:, 2]
    thetas = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False, dtype=np.float64)
    half_h = 0.5 * float(height)
    top = ctr.reshape((1, 3)) + radius * np.cos(thetas)[:, None] * ex.reshape((1, 3)) + radius * np.sin(thetas)[:, None] * ey.reshape((1, 3)) + half_h * ez.reshape((1, 3))
    bottom = ctr.reshape((1, 3)) + radius * np.cos(thetas)[:, None] * ex.reshape((1, 3)) + radius * np.sin(thetas)[:, None] * ey.reshape((1, 3)) - half_h * ez.reshape((1, 3))
    pix_top, valid_top = _project_base_points_to_image(top, k1, rot_cb, shift_cb)
    pix_bottom, valid_bottom = _project_base_points_to_image(bottom, k1, rot_cb, shift_cb)
    for seq_pix, seq_valid in ((pix_top, valid_top), (pix_bottom, valid_bottom)):
        for idx in range(seq_pix.shape[0]):
            j = (idx + 1) % seq_pix.shape[0]
            if seq_valid[idx] and seq_valid[j]:
                draw.line([(float(seq_pix[idx, 0]), float(seq_pix[idx, 1])), (float(seq_pix[j, 0]), float(seq_pix[j, 1]))], fill=rgb, width=2)
    for idx in range(0, pix_top.shape[0], 4):
        if valid_top[idx] and valid_bottom[idx]:
            draw.line([(float(pix_top[idx, 0]), float(pix_top[idx, 1])), (float(pix_bottom[idx, 0]), float(pix_bottom[idx, 1]))], fill=rgb, width=2)


def _overlay_mask(image: Image.Image, mask: np.ndarray, color_rgb: tuple[int, int, int], alpha: int) -> Image.Image:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not np.any(m):
        return image.convert("RGB")
    base = image.convert("RGBA")
    overlay = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = int(color_rgb[0])
    overlay[..., 1] = int(color_rgb[1])
    overlay[..., 2] = int(color_rgb[2])
    overlay[..., 3] = np.where(m, int(alpha), 0).astype(np.uint8)
    return Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA")).convert("RGB")


def _render_pick_overlay(left: Image.Image, pick_result: dict[str, Any], k1: np.ndarray, rot_cb: np.ndarray, shift_cb: np.ndarray) -> Image.Image:
    image = _overlay_mask(left, np.asarray(pick_result["mask"], dtype=bool), MASK_PICK_RGB, 70)
    draw = ImageDraw.Draw(image)
    bbox = [int(v) for v in pick_result["bbox"]]
    draw.rectangle(bbox, outline=PICK_BOX_RGB, width=3)
    obj_fit = pick_result.get("pick_fit")
    if isinstance(obj_fit, dict):
        _draw_projected_box(
            draw,
            np.asarray(obj_fit["center_base"], dtype=np.float32),
            np.asarray(obj_fit["axes_base"], dtype=np.float32),
            np.asarray(obj_fit["size_xyz"], dtype=np.float32),
            k1, rot_cb, shift_cb, PICK_FIT_RGB, width=3,
        )
        _draw_projected_frame(
            draw,
            np.asarray(obj_fit["grasp_origin_base"], dtype=np.float32),
            np.asarray(obj_fit["grasp_axes_base"], dtype=np.float32),
            max(0.05, 0.35 * float(np.max(np.asarray(obj_fit["size_xyz"], dtype=np.float32)))),
            k1, rot_cb, shift_cb,
        )
        draw.text((bbox[0] + 4, max(2, bbox[1] - 18)), f"pick:{pick_result.get('label', '')}", fill=ImageColor.getrgb("white"))
    return image


def _render_place_overlay(left: Image.Image, place_result: dict[str, Any], k1: np.ndarray, rot_cb: np.ndarray, shift_cb: np.ndarray) -> Image.Image:
    image = _overlay_mask(left, np.asarray(place_result["mask"], dtype=bool), MASK_PLACE_RGB, 70)
    draw = ImageDraw.Draw(image)
    bbox = [int(v) for v in place_result["bbox"]]
    draw.rectangle(bbox, outline=BOX_EDGE_RGB, width=3)
    target_fit = place_result.get("target_fit")
    if isinstance(target_fit, dict):
        primitive = str(target_fit.get("primitive_shape", "box"))
        if primitive == "cylinder" and target_fit.get("primitive_radius") is not None:
            _draw_projected_cylinder(
                draw,
                np.asarray(target_fit["center_base"], dtype=np.float32),
                np.asarray(target_fit["axes_base"], dtype=np.float32),
                float(target_fit["primitive_radius"]),
                float(target_fit["size_xyz"][2]),
                k1, rot_cb, shift_cb, PLACE_TARGET_RGB,
            )
        else:
            _draw_projected_box(
                draw,
                np.asarray(target_fit["center_base"], dtype=np.float32),
                np.asarray(target_fit["axes_base"], dtype=np.float32),
                np.asarray(target_fit["size_xyz"], dtype=np.float32),
                k1, rot_cb, shift_cb, PLACE_TARGET_RGB, width=3,
            )
    plan_fit = place_result.get("place_plan")
    if isinstance(plan_fit, dict):
        _draw_projected_box(
            draw,
            np.asarray(plan_fit["object_center_base"], dtype=np.float32),
            np.asarray(plan_fit["object_axes_base"], dtype=np.float32),
            np.asarray(plan_fit["object_size_xyz"], dtype=np.float32),
            k1, rot_cb, shift_cb, PLACE_BOX_RGB, width=3,
        )
        _draw_projected_frame(
            draw,
            np.asarray(plan_fit["gripper_origin_base"], dtype=np.float32),
            np.asarray(plan_fit["gripper_axes_base"], dtype=np.float32),
            max(0.05, 0.35 * float(np.max(np.asarray(plan_fit["object_size_xyz"], dtype=np.float32)))),
            k1, rot_cb, shift_cb,
        )
        draw.text((bbox[0] + 4, max(2, bbox[1] - 18)), f"place:{place_result.get('label', '')}", fill=ImageColor.getrgb("white"))
    return image


def _save_logs(log_root: str, request_id: str, request_payload: dict[str, Any], left: Image.Image, right: Image.Image, response_payload: dict[str, Any], k1: np.ndarray, rot_cb: np.ndarray, shift_cb: np.ndarray) -> str:
    file_stem = f"{request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    log_dir = os.path.join(log_root, file_stem)
    os.makedirs(log_dir, exist_ok=True)
    left.save(os.path.join(log_dir, "image_left.png"))
    right.save(os.path.join(log_dir, "image_right.png"))
    results = response_payload.get("results") or {}
    pick_result = results.get("pick")
    place_result = results.get("place")
    if isinstance(pick_result, dict):
        _render_pick_overlay(left, pick_result, k1, rot_cb, shift_cb).save(os.path.join(log_dir, "pick_overlay_left.png"))
    if isinstance(place_result, dict):
        _render_place_overlay(left, place_result, k1, rot_cb, shift_cb).save(os.path.join(log_dir, "place_overlay_left.png"))
    results = response_payload.get("results") or {}
    compact_results = {
        str(key): _compact_logged_result(val) for key, val in results.items() if isinstance(val, dict)
    }
    with open(os.path.join(log_dir, "request_log.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "request": _sanitize_payload_for_logging(request_payload),
                "response": {
                    "ok": bool(response_payload.get("ok", False)),
                    "error": str(response_payload.get("error", "")),
                    "request_id": str(response_payload.get("request_id", "")),
                    "text": response_payload.get("text", []),
                    "data_len": len(response_payload.get("data", [])),
                    "timings_ms": _to_jsonable(response_payload.get("timings_ms", {})),
                    "results": compact_results,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return log_dir


class PlaceDofSocketService:
    def __init__(self, config: ServiceConfig):
        self.config = config
        os.makedirs(self.config.log_dir, exist_ok=True)

    def _process_pick_result(
        self,
        stage_result: dict[str, Any],
        dense_map_cam: np.ndarray,
        dense_map_base: np.ndarray,
        rot_cb: np.ndarray,
        shift_cb: np.ndarray,
    ) -> dict[str, Any]:
        pick_pose = _finalize_pick_pose(stage_result, dense_map_cam, rot_cb, shift_cb)
        plane_model = _fit_local_plane_from_bbox(dense_map_base, tuple(stage_result["bbox"]), np.asarray(stage_result["mask"], dtype=bool))
        obj_fit = _fit_pick_object_box(dense_map_base, tuple(stage_result["bbox"]), np.asarray(stage_result["mask"], dtype=bool), plane_model)
        grasp_origin_local = np.asarray(obj_fit["grasp_origin_local"], dtype=np.float32)
        grasp_axes_local = np.asarray(obj_fit["grasp_axes_local"], dtype=np.float32)
        grasp_origin_base = (
            np.asarray(obj_fit["center_base"], dtype=np.float64).reshape(1, 3)
            + np.asarray(grasp_origin_local, dtype=np.float64).reshape(1, 3) @ np.asarray(obj_fit["axes_base"], dtype=np.float64).T
        ).reshape(3).astype(np.float32)
        grasp_axes_base = (np.asarray(obj_fit["axes_base"], dtype=np.float64) @ np.asarray(grasp_axes_local, dtype=np.float64)).astype(np.float32)
        flat = flatten_pick_context(
            pick_xquat=np.asarray(pick_pose["xquat"], dtype=np.float32),
            obb_center_base=np.asarray(obj_fit["center_base"], dtype=np.float32),
            obb_axes_base=np.asarray(obj_fit["axes_base"], dtype=np.float32),
            size_xyz=np.asarray(obj_fit["size_xyz"], dtype=np.float32),
            grasp_origin_local=grasp_origin_local,
            grasp_axes_local=grasp_axes_local,
        )
        return {
            "order": int(stage_result["order"]),
            "target_id": str(stage_result["target_id"]),
            "role": "pick",
            "prompt": str(stage_result["prompt"]),
            "text": [str(stage_result["label"])],
            "data": flat,
            "label": str(stage_result["label"]),
            "bbox": [int(v) for v in stage_result["bbox"]],
            "pixel": list(pick_pose["pixel"]),
            "mask": np.asarray(stage_result["mask"], dtype=bool),
            "pick_xquat": [float(v) for v in np.asarray(pick_pose["xquat"], dtype=np.float32).tolist()],
            "pick_fit": {
                "center_base": [float(v) for v in np.asarray(obj_fit["center_base"], dtype=np.float32).tolist()],
                "axes_base": np.asarray(obj_fit["axes_base"], dtype=np.float32).tolist(),
                "size_xyz": [float(v) for v in np.asarray(obj_fit["size_xyz"], dtype=np.float32).tolist()],
                "grasp_origin_base": [float(v) for v in grasp_origin_base.tolist()],
                "grasp_axes_base": np.asarray(grasp_axes_base, dtype=np.float32).tolist(),
                "grasp_origin_local": [float(v) for v in grasp_origin_local.tolist()],
                "grasp_axes_local": np.asarray(grasp_axes_local, dtype=np.float32).tolist(),
                "debug": dict(obj_fit["debug"]),
            },
            "timings_ms": dict(stage_result["timings_ms"]),
            "dof_r_square": float(stage_result["dof_r_square"]),
        }

    def _process_place_result(
        self,
        stage_result: dict[str, Any],
        dense_map_base: np.ndarray,
        dense_map_cam: np.ndarray,
        pick_context_flat: list[float] | np.ndarray,
        rot_cb: np.ndarray,
        shift_cb: np.ndarray,
        k1: np.ndarray,
        image_size: tuple[int, int],
        target_kind_hint: str,
        exclude_masks: list[np.ndarray],
    ) -> dict[str, Any]:
        pick_ctx = parse_pick_context(pick_context_flat)
        current_object_axes_base = _orthonormalize_axes(np.asarray(pick_ctx["obb_axes_base"], dtype=np.float32))
        current_grasp_axes_base = (
            current_object_axes_base.astype(np.float64) @ np.asarray(pick_ctx["grasp_axes_local"], dtype=np.float64).reshape(3, 3)
        ).astype(np.float32)
        mask = np.asarray(stage_result["mask"], dtype=bool)
        valid = _finite_point_mask(dense_map_base.reshape(-1, 3)).reshape(dense_map_base.shape[:2])
        target_points = _finite_points(dense_map_base[mask & valid])
        camera_origin_base = np.asarray(shift_cb, dtype=np.float32).reshape(3)
        fit_points, target_profile, target_debug = _fit_place_target_profile(target_points, target_kind_hint, camera_origin_base)
        plane_model = None
        try:
            plane_model = _fit_local_plane_from_bbox(dense_map_base, tuple(stage_result["bbox"]), mask)
        except Exception:
            plane_model = None
        collision_points = _build_collision_points_base(dense_map_base, [*exclude_masks, mask], plane_model)
        target_collision_points = _sample_target_collision_points(target_profile)
        visibility_collision_points = (
            np.asarray(collision_points, dtype=np.float32)
            if target_collision_points.shape[0] == 0
            else np.vstack([collision_points, target_collision_points]).astype(np.float32, copy=False)
        )
        camera_visibility_constraint = placedof_planner.CameraVisibilityConstraint(
            k1=np.asarray(k1, dtype=np.float32),
            rot_cb=np.asarray(rot_cb, dtype=np.float32),
            shift_cb=np.asarray(shift_cb, dtype=np.float32),
            image_size_wh=(int(image_size[0]), int(image_size[1])),
            pixel_margin_px=int(DEFAULT_PLACE_CAMERA_PIXEL_MARGIN_PX),
            require_full_object_in_frame=True,
            require_grasp_origin_in_frame=True,
        )
        request = placedof_planner.PlannerRequest(
            obb=placedof_planner.OBBSpec(size_xyz=np.asarray(pick_ctx["size_xyz"], dtype=np.float32)),
            grasp_pose_local=placedof_planner.RelativePose(
                origin_local=np.asarray(pick_ctx["grasp_origin_local"], dtype=np.float32),
                axes_local=np.asarray(pick_ctx["grasp_axes_local"], dtype=np.float32),
            ),
            table_points_base=np.zeros((0, 3), dtype=np.float32),
            target_points_base=np.asarray(fit_points, dtype=np.float32),
            scene_points_base=np.asarray(collision_points, dtype=np.float32),
            preferred_center_base=np.asarray(target_profile.center_base, dtype=np.float32),
            target_kind_hint=str(target_kind_hint or placedof_planner.DEFAULT_TARGET_KIND),
            target_profile_override=target_profile,
            start_object_center_base=np.asarray(pick_ctx["obb_center_base"], dtype=np.float32),
            start_gripper_axes_base=np.asarray(current_grasp_axes_base, dtype=np.float32),
            camera_visibility_constraint=camera_visibility_constraint,
            allow_best_effort=False,
            debug_hints={"target_bbox": [int(v) for v in stage_result["bbox"]], "target_primitive_shape": str(target_profile.primitive_shape)},
        )
        plan = placedof_planner.plan_placement(request)
        if not plan.success or plan.best is None:
            invalid_reason_counts = dict(plan.debug.get("invalid_reason_counts", {}))
            top_reasons = ", ".join(f"{reason} x{count}" for reason, count in list(invalid_reason_counts.items())[:4])
            if not top_reasons:
                top_reasons = "no valid candidate after primitive fitting / orientation search"
            raise RuntimeError(f"Planner failed to find a feasible placement candidate: {top_reasons}")
        best = plan.best
        best_effort_selected = bool(plan.debug.get("best_effort_selected", False))
        adjusted_center, adjusted_gripper, visibility_adjust = _post_adjust_place_plan_visibility(
            np.asarray(best.object_center_base, dtype=np.float32),
            np.asarray(best.object_axes_base, dtype=np.float32),
            np.asarray(pick_ctx["size_xyz"], dtype=np.float32),
            np.asarray(best.gripper_origin_base, dtype=np.float32),
            str(best.rule_id),
            target_profile,
            collision_points_base=visibility_collision_points,
            plane_model=plane_model,
            k1=k1,
            rot_cb=rot_cb,
            shift_cb=shift_cb,
            image_size=image_size,
        )
        final_visibility = visibility_adjust.get("final", {})
        if (
            isinstance(final_visibility, dict)
            and bool(final_visibility.get("available", False))
            and not bool(final_visibility.get("all_in_frame", False))
            and not best_effort_selected
        ):
            raise RuntimeError("Planner found a candidate, but the final place grasp / OBB could not be kept fully visible in the camera view.")
        place_quat = _quat_from_matrix(np.asarray(best.gripper_axes_base, dtype=np.float32))
        place_xquat = np.concatenate([np.asarray(adjusted_gripper, dtype=np.float32), place_quat.astype(np.float32)], axis=0).astype(np.float32)
        return {
            "order": int(stage_result["order"]),
            "target_id": str(stage_result["target_id"]),
            "role": "place",
            "prompt": str(stage_result["prompt"]),
            "text": [str(stage_result["label"])],
            "data": [float(v) for v in place_xquat.tolist()],
            "label": str(stage_result["label"]),
            "bbox": [int(v) for v in stage_result["bbox"]],
            "mask": mask,
            "target_fit": {
                "center_base": [float(v) for v in np.asarray(target_profile.center_base, dtype=np.float32).tolist()],
                "axes_base": np.asarray(target_profile.rotation_base, dtype=np.float32).tolist(),
                "size_xyz": [float(v) for v in np.asarray(target_profile.size_xyz, dtype=np.float32).tolist()],
                "primitive_shape": str(target_profile.primitive_shape),
                "primitive_radius": None if target_profile.primitive_radius is None else float(target_profile.primitive_radius),
                "kind": str(target_profile.kind),
                "opening_face": target_profile.opening_face,
                "debug": {
                    **dict(target_debug),
                    "visibility_collision_scene_points": int(visibility_collision_points.shape[0]),
                    "target_collision_points": int(target_collision_points.shape[0]),
                },
            },
            "place_plan": {
                "object_center_base": [float(v) for v in np.asarray(adjusted_center, dtype=np.float32).tolist()],
                "object_axes_base": np.asarray(best.object_axes_base, dtype=np.float32).tolist(),
                "object_size_xyz": [float(v) for v in np.asarray(pick_ctx["size_xyz"], dtype=np.float32).tolist()],
                "gripper_origin_base": [float(v) for v in np.asarray(adjusted_gripper, dtype=np.float32).tolist()],
                "gripper_axes_base": np.asarray(best.gripper_axes_base, dtype=np.float32).tolist(),
                "rule_id": str(best.rule_id),
                "target_kind": str(best.target_kind),
                "score": float(best.score),
                "score_breakdown": {str(k): float(v) for k, v in best.score_breakdown.items()},
                "planner_debug": dict(plan.debug),
                "visibility_adjustment": dict(visibility_adjust),
            },
            "timings_ms": dict(stage_result["timings_ms"]),
        }

    def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        t_total0 = time.perf_counter()
        request_id = str(payload.get("request_id") or f"req_{int(time.time() * 1000)}")
        left = _decode_image_from_payload(payload, "image_left")
        right = _decode_image_from_payload(payload, "image_right")
        if left.size != right.size:
            raise ValueError("image_left and image_right must have the same size")
        topic_text = _load_topic_inputs_text(payload)
        k1, baseline, rot_cb, shift_cb = _parse_topic_inputs(topic_text)
        arm_id = int(payload.get("arm_id", 0))
        timeout_s = float(payload.get("timeout_s", self.config.timeout_s))
        log_mode = str(payload.get("log_mode") or "compact").strip().lower()
        if log_mode not in {"full", "compact", "none"}:
            raise ValueError("log_mode must be one of: full, compact, none")
        targets = _normalize_targets(payload)
        if not targets:
            raise ValueError("No targets resolved from request")

        def run_s2m2_task() -> dict[str, Any]:
            t0 = time.perf_counter()
            arr, latency_ms = _call_s2m2_api(left_img=left, right_img=right, k1=k1, baseline=baseline, api_url=str(payload.get("s2m2_api_url") or self.config.s2m2_api_url).strip(), timeout_s=timeout_s)
            t_dense0 = time.perf_counter()
            dense_map_cam, depth_map = _build_dense_from_cloud_or_depth(arr, left.size, k1)
            dense_map_base = _transform_dense_map_cam_to_base(dense_map_cam, rot_cb, shift_cb)
            return {
                "raw_shape": list(np.asarray(arr).shape),
                "dense_map_cam": dense_map_cam,
                "dense_map_base": dense_map_base,
                "depth_map": depth_map,
                "latency_ms": int(latency_ms),
                "dense_build_ms": _elapsed_ms(t_dense0),
                "total_ms": _elapsed_ms(t0),
            }

        timings_ms: dict[str, int] = {"request_prepare": _elapsed_ms(t_total0)}
        t_parallel0 = time.perf_counter()
        stage_results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(2, len(targets) + 1)) as executor:
            fut_s2m2 = executor.submit(run_s2m2_task)
            fut_targets = [
                executor.submit(
                    _run_qwen_sam3_target,
                    target,
                    left,
                    rot_cb,
                    str(payload.get("sam3_url") or self.config.sam3_url).strip().rstrip("/"),
                    timeout_s,
                )
                for target in targets
            ]
            for fut in fut_targets:
                stage_results.append(fut.result())
            s2m2_res = fut_s2m2.result()
        timings_ms["parallel_wait"] = _elapsed_ms(t_parallel0)
        timings_ms["s2m2_call"] = int(s2m2_res["latency_ms"])
        timings_ms["s2m2_dense_build"] = int(s2m2_res["dense_build_ms"])
        timings_ms["s2m2_total"] = int(s2m2_res["total_ms"])

        dense_map_cam = np.asarray(s2m2_res["dense_map_cam"], dtype=np.float32)
        dense_map_base = np.asarray(s2m2_res["dense_map_base"], dtype=np.float32)
        stage_results.sort(key=lambda item: int(item["order"]))

        provided_pick_context = payload.get("pick_context")
        pick_context_flat: list[float] | None = None
        if isinstance(provided_pick_context, dict):
            pick_context_flat = flatten_pick_context(
                provided_pick_context["pick_xquat"],
                provided_pick_context["obb_center_base"],
                provided_pick_context["obb_axes_base"],
                provided_pick_context["size_xyz"],
                provided_pick_context["grasp_origin_local"],
                provided_pick_context["grasp_axes_local"],
            )
        elif isinstance(provided_pick_context, list):
            pick_context_flat = [float(v) for v in provided_pick_context]
        elif isinstance(payload.get("pick_context_flat"), list):
            pick_context_flat = [float(v) for v in list(payload["pick_context_flat"])]

        finalized: list[dict[str, Any]] = []
        pick_result: dict[str, Any] | None = None
        exclude_masks: list[np.ndarray] = []
        for stage in stage_results:
            if stage["role"] == "pick":
                pick_result = self._process_pick_result(stage, dense_map_cam, dense_map_base, rot_cb, shift_cb)
                pick_context_flat = list(pick_result["data"])
                finalized.append(pick_result)
                exclude_masks.append(np.asarray(stage["mask"], dtype=bool))
        for stage in stage_results:
            if stage["role"] == "place":
                if pick_context_flat is None:
                    raise RuntimeError("PlaceDOF requires pick context (pick OBB + grasp local pose), but no pick context was provided or computed.")
                place_result = self._process_place_result(
                    stage,
                    dense_map_base,
                    dense_map_cam,
                    pick_context_flat,
                    rot_cb,
                    shift_cb,
                    k1,
                    left.size,
                    str(payload.get("target_kind_hint") or placedof_planner.DEFAULT_TARGET_KIND),
                    exclude_masks=exclude_masks,
                )
                finalized.append(place_result)
        finalized.sort(key=lambda item: int(item["order"]))

        text: list[str] = []
        data: list[float] = []
        results_map: dict[str, Any] = {}
        qwen_pick = None
        qwen_place = None
        for item in finalized:
            text.extend(item["text"])
            data.extend(item["data"])
            results_map[item["target_id"]] = item
            if item["role"] == "pick":
                qwen_pick = {"text": list(item["text"]), "data": list(item["data"])}
            elif item["role"] == "place":
                qwen_place = {"text": list(item["text"]), "data": list(item["data"])}
            stage_timing = dict(item.get("timings_ms", {}))
            for key, value in stage_timing.items():
                timings_ms[f"{item['target_id']}_{key}"] = int(value)
        timings_ms["total"] = _elapsed_ms(t_total0)
        response = {
            "ok": True,
            "error": "",
            "request_id": request_id,
            "arm_id": int(arm_id),
            "text": text,
            "data": data,
            "results": results_map,
            "results_ordered": finalized,
            "qwen_pick": qwen_pick,
            "qwen_place": qwen_place,
            "timings_ms": timings_ms,
            "s2m2_shape": s2m2_res["raw_shape"],
        }
        if log_mode != "none":
            response["log_dir"] = _save_logs(self.config.log_dir, request_id, payload, left, right, response, k1, rot_cb, shift_cb)
        else:
            response["log_dir"] = ""
        return response


class PlaceDofTCPHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        service: PlaceDofSocketService = self.server.service  # type: ignore[attr-defined]
        try:
            raw = recv_message(self.request)
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("request payload must be a JSON object")
            result = service.process(payload)
        except Exception as exc:
            result = {
                "ok": False,
                "error": str(exc),
                "request_id": "",
                "text": [],
                "data": [],
                "results": {},
                "results_ordered": [],
                "qwen_pick": None,
                "qwen_place": None,
                "timings_ms": {},
                "log_dir": "",
            }
        send_message(self.request, result)


class PlaceDofTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], service: PlaceDofSocketService):
        self.service = service
        super().__init__(server_address, PlaceDofTCPHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="placedof socket service")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--s2m2-api-url", type=str, default=DEFAULT_S2M2_API)
    parser.add_argument("--sam3-url", type=str, default=DEFAULT_SAM3_URL)
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()
    config = ServiceConfig(
        host=str(args.host).strip(),
        port=int(args.port),
        timeout_s=float(args.timeout),
        s2m2_api_url=str(args.s2m2_api_url).strip(),
        sam3_url=str(args.sam3_url).strip().rstrip("/"),
        log_dir=str(args.log_dir).strip(),
    )
    service = PlaceDofSocketService(config)
    with PlaceDofTCPServer((config.host, config.port), service) as server:
        print(f"PlaceDOF socket service listening on {config.host}:{config.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
