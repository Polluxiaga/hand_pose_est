#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import logging
import math
import re
import secrets
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
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover
    _GUI_IMPORT_ERROR = exc
    tk = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]


THIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = THIS_DIR / ".viewer_cache"
# S2M2 深度/点云缓存：存在则直接加载，避免重复请求接口
S2M2_CACHE_SUFFIX = "_s2m2.npz"

# ---------------------------------------------------------------------------
# 远端服务与 SAM3D 行为（均为常量；修改后需重启本脚本生效）
# ---------------------------------------------------------------------------

# S2M2 深度 / 点云 HTTP 接口完整 URL（旧：环境变量 S2M2_API_URL）
DEFAULT_S2M2_API = "http://101.132.143.105:5060/api/process"

# SAM 分割服务基础 URL，无末尾斜杠（旧：SAM3_SERVER_URL）
DEFAULT_SAM3_URL = "http://101.132.143.105:5081".rstrip("/")

# SAM3D FastAPI 推理服务基础 URL（旧：SAM3D_API_URL）
DEFAULT_SAM3D_URL = "http://101.132.143.105:5083".rstrip("/")

# True：向 POST /infer_npz 上传 S2M2 的 dense_map 作为 pointmap，与 doc/API.md 一致。
# False：不传 pointmap，由服务端自行估计深度（如 MoGe）。（旧：SAM3D_SEND_POINTMAP）
SAM3D_SEND_POINTMAP = True

# True：Model Completion 前不调用 GET /health（旧服务无该路由或本地调试）。（旧：SAM3D_SKIP_HEALTH）
SAM3D_SKIP_HEALTH = False

# True：Model Completion 只上传 ROI（优先 selected_bbox，否则 mask 紧包围盒），图像 / mask / pointmap 同步裁剪。
# False：上传整幅左图。（旧：SAM3D_UPLOAD_CROP_TO_BBOX）
SAM3D_UPLOAD_CROP_TO_BBOX = False

# True：POST /infer_npz 传 ply_simplify=true，服务端按 API_PLY_* 精简导出 PLY；False 与 doc/openapi.json 默认一致。
SAM3D_INFER_PLY_SIMPLIFY = True
_SH_C0 = 0.28209479177387814
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_POINTS = 120000
DEFAULT_MIN_CAM_DEPTH = 1e-4
DEFAULT_MAX_COORD_ABS = 1e6
DEFAULT_RENDER_BG_RGB = (14, 20, 28)
# 右侧视图原点处 XYZ 轴长度（与 points_cam 坐标同单位；默认按米理解，即 10 cm）
VIEWER_AXIS_LENGTH = 0.1
MAX_IMAGE_DISPLAY_WIDTH = 620
MAX_IMAGE_DISPLAY_HEIGHT = 460
MIN_LEFT_IMAGE_PANE_HEIGHT = 220

_RESAMPLING = getattr(Image, "Resampling", Image)
LOG_PATH = THIS_DIR / "pointcloud_viewer.log"

LOGGER = logging.getLogger("pointcloud_viewer")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    LOGGER.addHandler(_sh)
    try:
        _fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        _fh.setFormatter(_fmt)
        LOGGER.addHandler(_fh)
    except Exception:
        # File logging is optional; still keep console logs.
        pass


def _fmt_latency_s(ms: int | float) -> str:
    """将毫秒转为秒字符串，固定三位小数（用于日志与界面）。"""
    return f"{max(0.0, float(ms)) / 1000.0:.3f}"


@dataclass
class CameraBundle:
    k1: np.ndarray
    baseline: float


@dataclass
class PointCloudFrame:
    sample_id: str
    dense_map_cam: np.ndarray
    valid_mask: np.ndarray
    points_cam: np.ndarray
    colors: np.ndarray
    pixels_yx: np.ndarray
    latency_ms: int
    # True 表示来自磁盘缓存（切换样本时自动加载或 Run S2M2 命中缓存文件）
    from_cache: bool = False


@dataclass
class ViewState:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    zoom: float = 1.0
    # 相对自动 median 的旋转中心偏移（相机坐标系，与 points_cam 同空间）
    pivot_ox: float = 0.0
    pivot_oy: float = 0.0
    pivot_oz: float = 0.0


def _parse_topic_inputs(topic_text: str) -> tuple[np.ndarray, float]:
    if not isinstance(topic_text, str) or not topic_text.strip():
        raise ValueError("camera_intrinsics.txt is empty.")

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
        if not p_started:
            continue
        if raw.startswith("-"):
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
            if nums:
                p_values.append(float(nums[0]))
        else:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
            p_values.extend(float(x) for x in nums)
        if len(p_values) >= 12:
            break

    if len(p_values) < 12:
        all_nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", topic_text)
        if len(all_nums) >= 12:
            p_values = [float(x) for x in all_nums[:12]]

    if len(p_values) < 12:
        raise ValueError("Failed to parse 12 projection-matrix values from camera_intrinsics.txt.")

    fx = float(p_values[0])
    fy = float(p_values[5])
    cx = float(p_values[2])
    cy = float(p_values[6])
    tx = float(p_values[3])
    if not np.isfinite(fx) or not np.isfinite(fy) or abs(fx) < 1e-9 or abs(fy) < 1e-9:
        raise ValueError("Invalid fx/fy in camera_intrinsics.txt.")

    baseline = abs(tx) / max(abs(fx), 1e-9)
    if baseline > 1.0:
        baseline *= 0.001
    if baseline <= 1e-6:
        raise ValueError("Baseline parsed as zero.")

    k1 = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return k1, float(baseline)


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
            arr = arr["depth"] if "depth" in arr else arr[arr.files[0]]
        return np.array(arr, dtype=np.float32)
    if data.startswith(b"PF") or data.startswith(b"Pf"):
        return _load_pfm(data)
    try:
        return _depth_image_to_array(Image.open(io.BytesIO(data)))
    except Exception as exc:
        raise ValueError("Unable to parse depth response bytes.") from exc


def _decode_depth_payload(payload: Any) -> np.ndarray:
    if isinstance(payload, list):
        return np.array(payload, dtype=np.float32)
    if isinstance(payload, dict):
        for key in ("data", "depth", "value", "base64", "b64"):
            if key in payload:
                return _decode_depth_payload(payload[key])
        raise ValueError(f"Unsupported depth payload dict keys: {list(payload.keys())}")
    if isinstance(payload, str):
        try:
            data = base64.b64decode(payload)
        except Exception as exc:
            raise ValueError("Failed to decode base64 depth payload.") from exc
        return _load_depth_from_bytes(data)
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
        raise ValueError("Left/right sizes do not match after NV12 conversion.")

    data = {
        "K": k_flat,
        "baseline": str(float(baseline)),
        "width": str(int(width)),
    }
    files = {
        "left_file": ("left.nv12", left_nv12, "application/octet-stream"),
        "right_file": ("right.nv12", right_nv12, "application/octet-stream"),
    }
    t0 = time.perf_counter()
    response = requests.post(str(api_url).strip(), data=data, files=files, timeout=float(timeout_s))
    latency_ms = int(round((time.perf_counter() - t0) * 1000.0))
    LOGGER.info("[HTTP] POST S2M2 latency_s=%s http=%d", _fmt_latency_s(latency_ms), response.status_code)
    if response.status_code != 200:
        raise RuntimeError(f"S2M2 API failed ({response.status_code}): {response.text}")
    return _parse_depth_response(response), latency_ms


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


def _finite_xyz_mask_map(xyz_map: np.ndarray, max_abs: float = DEFAULT_MAX_COORD_ABS) -> np.ndarray:
    finite = np.all(np.isfinite(xyz_map), axis=2)
    finite &= np.max(np.abs(xyz_map), axis=2) <= float(max_abs)
    return finite


def _valid_cam_xyz_mask_map(
    xyz_map: np.ndarray,
    min_depth: float = DEFAULT_MIN_CAM_DEPTH,
    max_abs: float = DEFAULT_MAX_COORD_ABS,
) -> np.ndarray:
    valid = _finite_xyz_mask_map(xyz_map, max_abs=max_abs)
    valid &= xyz_map[..., 2] > float(min_depth)
    return valid


def _build_dense_from_cloud_or_depth(
    cloud_or_depth: np.ndarray,
    image_size: tuple[int, int],
    k1: np.ndarray,
) -> np.ndarray:
    img_w, img_h = image_size
    data = np.asarray(cloud_or_depth, dtype=np.float32)

    dense_map = np.full((img_h, img_w, 3), np.nan, dtype=np.float32)

    if data.ndim == 3 and data.shape[2] == 3:
        if data.shape[0] != img_h or data.shape[1] != img_w:
            data = _resize_pointcloud_map(data, (img_w, img_h))
        valid = _valid_cam_xyz_mask_map(data)
        dense_map[valid] = data[valid]
        return dense_map

    if data.ndim == 2 and data.shape[1] == 3:
        points = np.asarray(data, dtype=np.float32)
        valid = np.all(np.isfinite(points), axis=1)
        valid &= np.max(np.abs(points), axis=1) <= float(DEFAULT_MAX_COORD_ABS)
        valid &= points[:, 2] > float(DEFAULT_MIN_CAM_DEPTH)
        points = points[valid]
        if points.shape[0] == 0:
            return dense_map
        fx = float(k1[0, 0])
        fy = float(k1[1, 1])
        cx = float(k1[0, 2])
        cy = float(k1[1, 2])
        u_f = points[:, 0].astype(np.float64) * fx / points[:, 2].astype(np.float64) + cx
        v_f = points[:, 1].astype(np.float64) * fy / points[:, 2].astype(np.float64) + cy
        finite_uv = np.isfinite(u_f) & np.isfinite(v_f)
        points = points[finite_uv]
        u = np.rint(u_f[finite_uv]).astype(np.int64)
        v = np.rint(v_f[finite_uv]).astype(np.int64)
        keep = (u >= 0) & (u < int(img_w)) & (v >= 0) & (v < int(img_h))
        points = points[keep]
        u = u[keep].astype(np.int32)
        v = v[keep].astype(np.int32)
        if points.shape[0] == 0:
            return dense_map
        linear = v * img_w + u
        order = np.argsort(points[:, 2])
        linear_sorted = linear[order]
        first_idx = np.unique(linear_sorted, return_index=True)[1]
        picked = order[first_idx]
        dense_map[v[picked], u[picked]] = points[picked]
        return dense_map

    depth = np.asarray(data, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise RuntimeError(f"Unsupported S2M2 output shape: {tuple(data.shape)}")

    depth = _resize_depth_map(depth, (img_w, img_h)).astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > float(DEFAULT_MIN_CAM_DEPTH))
    valid &= np.abs(depth) <= float(DEFAULT_MAX_COORD_ABS)
    if not np.any(valid):
        return dense_map

    fx = float(k1[0, 0])
    fy = float(k1[1, 1])
    cx = float(k1[0, 2])
    cy = float(k1[1, 2])
    grid_x, grid_y = np.meshgrid(np.arange(img_w, dtype=np.float64), np.arange(img_h, dtype=np.float64))
    depth64 = depth.astype(np.float64, copy=False)
    x = (grid_x - cx) / fx * depth64
    y = (grid_y - cy) / fy * depth64
    valid &= np.isfinite(x) & np.isfinite(y)
    valid &= (np.abs(x) <= float(DEFAULT_MAX_COORD_ABS)) & (np.abs(y) <= float(DEFAULT_MAX_COORD_ABS))
    dense_map[..., 0][valid] = x[valid].astype(np.float32)
    dense_map[..., 1][valid] = y[valid].astype(np.float32)
    dense_map[..., 2][valid] = depth[valid]
    return dense_map


def _cache_key_from_inputs(left_path: Path, right_path: Path, intrinsics_path: Path) -> str:
    payload = "||".join(
        [
            str(left_path.expanduser().resolve()),
            str(right_path.expanduser().resolve()),
            str(intrinsics_path.expanduser().resolve()),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _s2m2_cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{str(cache_key).strip()}{S2M2_CACHE_SUFFIX}"


def _load_s2m2_cache_file(path: Path) -> tuple[np.ndarray, int] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "dense_map_cam" not in data:
                return None
            dm = np.asarray(data["dense_map_cam"], dtype=np.float32)
            if dm.ndim != 3 or dm.shape[2] != 3:
                return None
            lat = int(np.asarray(data["latency_ms"]).reshape(())) if "latency_ms" in data else 0
            return dm, lat
    except Exception:
        LOGGER.warning("failed to read s2m2 cache: %s", path, exc_info=True)
        return None


def _save_s2m2_cache_file(path: Path, dense_map: np.ndarray, latency_ms: int) -> None:
    dm = np.asarray(dense_map, dtype=np.float32)
    if dm.ndim != 3 or dm.shape[2] != 3:
        raise ValueError(f"invalid dense_map for cache: {getattr(dm, 'shape', None)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = int(dm.shape[0]), int(dm.shape[1])
    np.savez_compressed(
        path,
        dense_map_cam=dm,
        latency_ms=np.int32(int(latency_ms)),
        img_w=np.int32(w),
        img_h=np.int32(h),
    )


def _pointcloud_frame_from_dense_map(
    sample_id: str,
    dense_map: np.ndarray,
    left_np: np.ndarray,
    latency_ms: int,
    *,
    from_cache: bool = False,
) -> PointCloudFrame:
    dm = np.asarray(dense_map, dtype=np.float32)
    if dm.shape[:2] != left_np.shape[:2]:
        raise ValueError(
            f"dense_map HW {dm.shape[:2]} does not match left image {left_np.shape[:2]}"
        )
    valid = _valid_cam_xyz_mask_map(dm)
    ys, xs = np.nonzero(valid)
    points_cam = dm[ys, xs].astype(np.float32, copy=False)
    finite = np.all(np.isfinite(points_cam), axis=1)
    finite &= np.max(np.abs(points_cam), axis=1) <= float(DEFAULT_MAX_COORD_ABS)
    points_cam = points_cam[finite].astype(np.float32, copy=False)
    ys = ys[finite]
    xs = xs[finite]
    colors = (left_np[ys, xs].astype(np.float32) / 255.0).astype(np.float32, copy=False)
    pixels_yx = np.stack([ys, xs], axis=1).astype(np.int32, copy=False)
    return PointCloudFrame(
        sample_id=str(sample_id),
        dense_map_cam=dm.astype(np.float32, copy=False),
        valid_mask=valid.astype(bool, copy=False),
        points_cam=points_cam,
        colors=colors,
        pixels_yx=pixels_yx,
        latency_ms=int(latency_ms),
        from_cache=from_cache,
    )


def _image_to_jpeg_b64(image: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=int(quality), optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_sam3_generic(
    image: Image.Image,
    sam3_url: str,
    timeout_s: float,
    box_prompts: list[list[int]] | None = None,
    box_labels: list[bool] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"image": _image_to_jpeg_b64(image)}
    if box_prompts:
        payload["box_prompts"] = [[int(v) for v in box] for box in box_prompts]
        payload["box_labels"] = [bool(v) for v in (box_labels or [True] * len(box_prompts))]

    out: dict[str, Any] = {
        "ok": False,
        "latency_ms": 0,
        "num_detections": 0,
        "error": "",
        "mask": None,
        "mask_area_px": 0,
    }
    t0 = time.perf_counter()
    sam3_segment_url = f"{sam3_url.rstrip('/')}/segment"
    try:
        response = requests.post(
            sam3_segment_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=float(timeout_s),
        )
    except requests.RequestException as exc:
        out["latency_ms"] = int(round((time.perf_counter() - t0) * 1000.0))
        LOGGER.info(
            "[HTTP] POST SAM3/segment latency_s=%s http=- err=%s",
            _fmt_latency_s(out["latency_ms"]),
            exc,
        )
        out["error"] = f"sam3_request_error: {exc}"
        return out

    out["latency_ms"] = int(round((time.perf_counter() - t0) * 1000.0))
    LOGGER.info("[HTTP] POST SAM3/segment latency_s=%s http=%d", _fmt_latency_s(out["latency_ms"]), response.status_code)
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
            mask_img = Image.open(io.BytesIO(raw)).convert("L")
            if mask_img.size != (width, height):
                mask_img = mask_img.resize((width, height), _RESAMPLING.NEAREST)
            mask |= np.asarray(mask_img, dtype=np.uint8) > 127
        except Exception:
            continue

    if mask.any():
        out["ok"] = True
        out["mask"] = mask
        out["mask_area_px"] = int(mask.sum())
    else:
        out["error"] = "sam3_empty_mask"
    return out


def _overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    alpha: int,
) -> Image.Image:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not np.any(m):
        return image
    base = image.convert("RGBA")
    overlay = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
    overlay[..., 0] = int(color_rgb[0])
    overlay[..., 1] = int(color_rgb[1])
    overlay[..., 2] = int(color_rgb[2])
    overlay[..., 3] = np.where(m, int(alpha), 0).astype(np.uint8)
    return Image.alpha_composite(base, Image.fromarray(overlay)).convert("RGB")


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
    if max_points <= 0 or length <= max_points:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(length, size=int(max_points), replace=False))


def _empty_points() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


def _empty_colors() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


def _tight_bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _sam3d_resolve_infer_roi_xyxy(
    width: int,
    height: int,
    selected_bbox: tuple[int, int, int, int] | None,
    mask_hw: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """返回含端点的 xyxy；无有效 ROI 时返回 None。"""
    if selected_bbox is not None:
        roi = _clip_bbox(tuple(int(v) for v in selected_bbox), width, height)
        if (roi[2] - roi[0] + 1) >= 2 and (roi[3] - roi[1] + 1) >= 2:
            return roi
    x1, y1, x2, y2 = _tight_bbox_from_mask(mask_hw)
    if (x2 - x1 + 1) < 2 or (y2 - y1 + 1) < 2:
        return None
    return x1, y1, x2, y2


def _object_outlier_removal_edge_depth_jump(
    dense_map_cam: np.ndarray,
    object_mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int] | None,
    *,
    canny_low: int = 40,
    canny_high: int = 120,
    depth_pct_lo: float = 2.0,
    depth_pct_hi: float = 98.0,
    min_depth: float = DEFAULT_MIN_CAM_DEPTH,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """在 bbox 内对深度 Z 做 Canny 得边缘，再从 object_mask 中去掉落在边缘上的像素。"""
    t_start = time.perf_counter()

    def _elapsed_ms() -> int:
        return int(round((time.perf_counter() - t_start) * 1000.0))

    h, w = int(object_mask.shape[0]), int(object_mask.shape[1])
    z = np.asarray(dense_map_cam[..., 2], dtype=np.float64)
    om = np.asarray(object_mask, dtype=bool)
    valid_z = np.isfinite(z) & (z > float(min_depth))
    m = om & valid_z
    if not np.any(m):
        return om.copy(), {
            "n_input": int(np.count_nonzero(om)),
            "n_removed": 0,
            "n_kept": int(np.count_nonzero(om)),
            "elapsed_ms": _elapsed_ms(),
        }

    if bbox_xyxy is not None:
        x1, y1, x2, y2 = _clip_bbox(bbox_xyxy, w, h)
    else:
        x1, y1, x2, y2 = _tight_bbox_from_mask(om)
    if x2 < x1 or y2 < y1:
        return om.copy(), {
            "n_input": int(np.count_nonzero(om)),
            "n_removed": 0,
            "n_kept": int(np.count_nonzero(om)),
            "elapsed_ms": _elapsed_ms(),
        }

    z_roi = z[y1 : y2 + 1, x1 : x2 + 1]
    valid_roi = valid_z[y1 : y2 + 1, x1 : x2 + 1]
    if not np.any(valid_roi):
        return om.copy(), {
            "n_input": int(np.count_nonzero(om)),
            "n_edge_removed": 0,
            "n_kept": int(np.count_nonzero(om)),
            "n_canny_total": 0,
            "canny_low": int(canny_low),
            "canny_high": int(canny_high),
            "bbox": (x1, y1, x2, y2),
            "mode": "depth_canny_strip_edges",
            "elapsed_ms": _elapsed_ms(),
        }

    zv = z_roi[valid_roi]
    lo, hi = np.percentile(zv, [float(depth_pct_lo), float(depth_pct_hi)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-9
    depth_u8_roi = np.zeros(z_roi.shape, dtype=np.uint8)
    depth_u8_roi[valid_roi] = np.clip((z_roi[valid_roi] - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)

    t1 = max(0, int(canny_low))
    t2 = max(t1 + 1, int(canny_high))
    edges_roi = cv2.Canny(depth_u8_roi, t1, t2)
    canny_roi = edges_roi > 0

    canny_mask = np.zeros((h, w), dtype=bool)
    canny_mask[y1 : y2 + 1, x1 : x2 + 1] = canny_roi

    n_in = int(np.count_nonzero(om))
    n_canny_roi = int(np.count_nonzero(canny_roi))
    n_edge_removed = int(np.count_nonzero(om & canny_mask))
    new_mask = om & (~canny_mask)
    n_kept = int(np.count_nonzero(new_mask))

    info: dict[str, float | int] = {
        "n_input": n_in,
        "n_edge_removed": n_edge_removed,
        "n_kept": n_kept,
        "n_canny_total": n_canny_roi,
        "canny_low": t1,
        "canny_high": t2,
        "bbox": (x1, y1, x2, y2),
        "mode": "depth_canny_strip_edges",
        "elapsed_ms": _elapsed_ms(),
    }
    return new_mask, info


def _dense_map_to_pointmap_npy_bytes(dense_hw3: np.ndarray) -> bytes:
    arr = np.asarray(dense_hw3, dtype=np.float32)
    # SAM3D 侧要求 points 为 (N,3) 或 (H,W,3)；若上游带 batch 维 [1,H,W,3] 会先被 unsqueeze 成 4D 而报错
    while arr.ndim > 3 and int(arr.shape[0]) == 1:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim != 3 or int(arr.shape[2]) != 3:
        raise ValueError(f"pointmap must be HxWx3, got {tuple(arr.shape)}")
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    buf.seek(0)
    return buf.getvalue()


def _sam3d_format_http_detail(response: requests.Response) -> str:
    """FastAPI 错误体多为 {\"detail\": ...}，detail 可能为 str 或校验错误列表。"""
    try:
        j = response.json()
    except ValueError:
        return (response.text or "").strip() or f"http_{response.status_code}"
    d = j.get("detail", j)
    if d is None:
        return (response.text or "").strip() or f"http_{response.status_code}"
    if isinstance(d, list):
        parts: list[str] = []
        for item in d:
            if isinstance(item, dict):
                loc = item.get("loc", ())
                msg = item.get("msg", item)
                parts.append(f"{loc}: {msg}")
            else:
                parts.append(str(item))
        return "; ".join(parts) if parts else str(j)
    return str(d)


def _sam3d_preflight_health(base_url: str, timeout_s: float = 8.0) -> None:
    """GET /health：确认 model_loaded（见 doc/API.md）。不存在该路由时仅打日志并继续。"""
    if SAM3D_SKIP_HEALTH:
        return
    url = f"{base_url.rstrip('/')}/health"
    t0 = time.perf_counter()
    try:
        r = requests.get(url, timeout=min(15.0, float(timeout_s)))
    except requests.RequestException as exc:
        ms = int(round((time.perf_counter() - t0) * 1000.0))
        LOGGER.info("[HTTP] GET SAM3D/health latency_s=%s http=- err=%s", _fmt_latency_s(ms), exc)
        return
    ms = int(round((time.perf_counter() - t0) * 1000.0))
    LOGGER.info("[HTTP] GET SAM3D/health latency_s=%s http=%d", _fmt_latency_s(ms), r.status_code)
    if r.status_code == 404:
        LOGGER.info("SAM3D GET /health 404 — 可能是旧版服务，跳过预检。")
        return
    if r.status_code != 200:
        raise RuntimeError(f"SAM3D /health failed: {_sam3d_format_http_detail(r)}")
    try:
        j = r.json()
    except ValueError:
        return
    if j.get("model_loaded") is False:
        raise RuntimeError(
            "SAM3D 模型未就绪（GET /health model_loaded=false），请等待服务加载完成后再试。"
        )


def _mask_bool_to_png_bytes(mask: np.ndarray) -> bytes:
    m = np.asarray(mask, dtype=bool)
    u8 = m.astype(np.uint8) * 255
    buf = io.BytesIO()
    Image.fromarray(u8, mode="L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _numpy_dtype_for_ply_scalar(ply_type: str, little_endian: bool) -> np.dtype:
    t = ply_type.lower()
    e = "<" if little_endian else ">"
    if t in ("char", "int8"):
        return np.dtype(np.int8)
    if t in ("uchar", "uint8"):
        return np.dtype(np.uint8)
    if t in ("short", "int16"):
        return np.dtype(f"{e}i2")
    if t in ("ushort", "uint16"):
        return np.dtype(f"{e}u2")
    if t in ("int", "int32"):
        return np.dtype(f"{e}i4")
    if t in ("uint", "uint32"):
        return np.dtype(f"{e}u4")
    if t in ("float", "float32"):
        return np.dtype(f"{e}f4")
    if t in ("double", "float64"):
        return np.dtype(f"{e}f8")
    raise ValueError(f"Unsupported PLY property type: {ply_type}")


def _parse_gaussian_splat_ply(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(data) < 64 or data[:3] != b"ply":
        raise ValueError("Not a PLY file (missing ply header).")

    if b"end_header\r\n" in data[:8192]:
        sep = b"end_header\r\n"
    else:
        sep = b"end_header\n"
    hi = data.find(sep)
    if hi < 0:
        raise ValueError("PLY missing end_header.")
    header = data[: hi + len(sep)].decode("ascii", errors="replace")
    body = data[hi + len(sep) :]

    fmt = "ascii"
    little_endian = True
    n_vertex = 0
    in_vertex = False
    props: list[tuple[str, str]] = []

    for raw in header.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
            little_endian = "little" in fmt
        elif parts[0] == "element" and len(parts) >= 3:
            if parts[1] == "vertex":
                in_vertex = True
                n_vertex = int(parts[2])
                props = []
            else:
                in_vertex = False
        elif parts[0] == "property" and in_vertex:
            if len(parts) >= 3 and parts[1] != "list":
                props.append((parts[1], parts[2]))
            elif len(parts) >= 2 and parts[1] == "list":
                in_vertex = False

    if n_vertex <= 0 or not props:
        raise ValueError("PLY has no vertex element or zero vertices.")

    if fmt == "ascii":
        text = body.decode("ascii", errors="replace").strip().split()
        floats = [float(x) for x in text]
        stride = len(props)
        need = n_vertex * stride
        if len(floats) < need:
            raise ValueError(f"ASCII PLY: expected {need} floats, got {len(floats)}")
        block = np.asarray(floats[:need], dtype=np.float64).reshape((n_vertex, stride))
        names = [p[1] for p in props]
        col_dict: dict[str, np.ndarray] = {names[i]: block[:, i].astype(np.float32) for i in range(stride)}
    else:
        dtype_fields = [(_numpy_dtype_for_ply_scalar(t, little_endian), n) for t, n in props]
        dt = np.dtype([(n, d) for d, n in dtype_fields])
        need_bytes = n_vertex * int(dt.itemsize)
        if len(body) < need_bytes:
            raise ValueError(f"PLY body too short: need {need_bytes} bytes, have {len(body)}")
        col_dict = np.frombuffer(body[:need_bytes], dtype=dt, count=n_vertex, offset=0)

    def _col(name: str) -> np.ndarray | None:
        if isinstance(col_dict, dict):
            v = col_dict.get(name)
            return None if v is None else np.asarray(v, dtype=np.float32)
        names_nd = col_dict.dtype.names
        if not names_nd or name not in names_nd:
            return None
        return np.asarray(col_dict[name], dtype=np.float32)

    cx = _col("x")
    cy = _col("y")
    cz = _col("z")
    if cx is None or cy is None or cz is None:
        raise ValueError("PLY must contain x, y, z vertex properties.")
    xyz = np.stack([cx, cy, cz], axis=1).astype(np.float32, copy=False)

    f0 = _col("f_dc_0")
    f1 = _col("f_dc_1")
    f2 = _col("f_dc_2")
    if f0 is not None and f1 is not None and f2 is not None:
        rgb = np.stack(
            [
                np.clip(0.5 + _SH_C0 * f0, 0.0, 1.0),
                np.clip(0.5 + _SH_C0 * f1, 0.0, 1.0),
                np.clip(0.5 + _SH_C0 * f2, 0.0, 1.0),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
    else:
        rgb = np.full((xyz.shape[0], 3), 0.85, dtype=np.float32)

    finite = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite]
    if xyz.shape[0] == 0:
        raise ValueError("PLY contains no finite vertex positions.")
    return xyz, rgb


def _call_sam3d_infer_npz(
    image: Image.Image,
    mask: np.ndarray,
    base_url: str,
    timeout_s: float,
    seed: int | None = None,
    pointmap_npy_bytes: bytes | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": False,
        "latency_ms": 0,
        "error": "",
        "ply_bytes": b"",
    }
    url = f"{base_url.rstrip('/')}/infer_npz"
    img_buf = io.BytesIO()
    image.convert("RGB").save(img_buf, format="PNG", optimize=True)
    img_buf.seek(0)
    mask_buf = io.BytesIO(_mask_bool_to_png_bytes(mask))
    files: dict[str, tuple[str, Any, str]] = {
        "image": ("image.png", img_buf, "image/png"),
        "mask": ("mask.png", mask_buf, "image/png"),
    }
    if pointmap_npy_bytes is not None and len(pointmap_npy_bytes) > 0:
        files["pointmap"] = ("pointmap.npy", io.BytesIO(pointmap_npy_bytes), "application/octet-stream")
    data: dict[str, str] = {"ply_simplify": "true" if SAM3D_INFER_PLY_SIMPLIFY else "false"}
    if seed is not None:
        data["seed"] = str(int(seed))
    t0 = time.perf_counter()
    try:
        response = requests.post(url, files=files, data=data or None, timeout=float(timeout_s))
    except requests.RequestException as exc:
        out["latency_ms"] = int(round((time.perf_counter() - t0) * 1000.0))
        LOGGER.info(
            "[HTTP] POST SAM3D/infer_npz latency_s=%s http=- err=%s",
            _fmt_latency_s(out["latency_ms"]),
            exc,
        )
        out["error"] = f"sam3d_request_error: {exc}"
        return out

    out["latency_ms"] = int(round((time.perf_counter() - t0) * 1000.0))
    LOGGER.info("[HTTP] POST SAM3D/infer_npz latency_s=%s http=%d", _fmt_latency_s(out["latency_ms"]), response.status_code)
    if response.status_code >= 400:
        out["error"] = f"sam3d_http_{response.status_code}: {_sam3d_format_http_detail(response)}"
        return out

    ctype = response.headers.get("Content-Type", "")
    if "json" in ctype.lower():
        try:
            err = response.json()
            out["error"] = str(err.get("detail", err))
        except Exception:
            out["error"] = "sam3d_unexpected_json_body"
        return out

    raw = response.content
    out["ply_bytes"] = raw
    if len(raw) < 64 or raw[:3] != b"ply":
        out["error"] = "sam3d_response_not_ply_infer_npz_expects_ply"
        return out
    out["ok"] = True
    return out


def _infer_npz_ply_bytes_to_gaussians(
    ply_bytes: bytes,
) -> tuple[np.ndarray, np.ndarray, None, None, None, dict[str, Any]]:
    """解析 POST /infer_npz 返回的未压缩 PLY（见 doc/openapi.json）。"""
    xyz, rgb = _parse_gaussian_splat_ply(ply_bytes)
    pose_meta: dict[str, Any] = {"ply_pose_baked_into_vertices": True}
    return xyz, rgb, None, None, None, pose_meta


class PointCloudViewerApp:
    def __init__(self) -> None:
        if _GUI_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Tk GUI is unavailable in this Python environment. Use a Python build with tkinter support."
            ) from _GUI_IMPORT_ERROR

        self.root = tk.Tk()
        self.root.title("PointCloud Viewer")
        self.root.geometry("1680x960")
        LOGGER.info("viewer start | log_file=%s", str(LOG_PATH))

        self.s2m2_url_var = tk.StringVar(value=DEFAULT_S2M2_API)
        self.sam3_url_var = tk.StringVar(value=DEFAULT_SAM3_URL)
        self.sam3d_url_var = tk.StringVar(value=DEFAULT_SAM3D_URL)
        self.timeout_var = tk.StringVar(value=f"{DEFAULT_TIMEOUT_S:.1f}")
        self.max_points_var = tk.StringVar(value=str(DEFAULT_MAX_POINTS))
        self.status_var = tk.StringVar(value="Load left/right images and camera intrinsics.")
        self.info_var = tk.StringVar(value="")
        self.left_path_var = tk.StringVar(value="")
        self.right_path_var = tk.StringVar(value="")
        self.intrinsics_path_var = tk.StringVar(value="")

        self.camera: CameraBundle | None = None
        self.selected_sample_id: str | None = None
        self.current_cache_key: str | None = None
        self.left_image_full: Image.Image | None = None
        self.right_image_full: Image.Image | None = None
        self.left_image_np: np.ndarray | None = None

        self.pointcloud: PointCloudFrame | None = None
        self.pointcloud_cache: dict[str, PointCloudFrame] = {}

        self.selected_bbox: tuple[int, int, int, int] | None = None
        self.scene_bbox: tuple[int, int, int, int] | None = None
        self.preview_bbox: tuple[int, int, int, int] | None = None
        self.object_mask: np.ndarray | None = None
        self._bbox_drag_start: tuple[int, int] | None = None

        self.sam3d_points: np.ndarray | None = None
        self.sam3d_colors: np.ndarray | None = None
        self.sam3d_infer_seed: int | None = None
        self.sam3d_latency_ms: int = 0
        self.sam3d_pose_R: np.ndarray | None = None
        self.sam3d_pose_t: np.ndarray | None = None
        self.sam3d_pose_s: np.ndarray | None = None
        self.sam3d_sent_pointmap: bool = False
        self.sam3d_ply_simplify: bool = False
        self.sam3d_response_kind: str = ""
        self.sam3d_infer_crop_roi: tuple[int, int, int, int] | None = None

        self.left_display_image: Image.Image | None = None
        self.left_display_photo: ImageTk.PhotoImage | None = None
        self.left_canvas_image_id: int | None = None
        self.left_display_scale_x = 1.0
        self.left_display_scale_y = 1.0
        self.left_bbox_item_id: int | None = None
        self.left_scene_bbox_item_id: int | None = None
        self.left_preview_item_id: int | None = None

        self.viewer_state = ViewState()
        self.viewer_drag_last: tuple[int, int] | None = None
        self.viewer_pivot_drag_last: tuple[int, int] | None = None
        self._viewer_last_view_rot: np.ndarray | None = None
        self._viewer_last_scale: float = 1.0
        self.viewer_photo: ImageTk.PhotoImage | None = None
        self.viewer_image_id: int | None = None

        self.busy = False
        self._build_ui()
        self._update_info()

    def _build_ui(self) -> None:
        root_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        root_pane.pack(fill="both", expand=True)

        left_panel = ttk.Frame(root_pane, padding=8)
        right_panel = ttk.Frame(root_pane, padding=8)
        root_pane.add(left_panel, weight=3)
        root_pane.add(right_panel, weight=5)

        left_split = ttk.Panedwindow(left_panel, orient=tk.VERTICAL)
        left_split.pack(fill="both", expand=True)

        scroll_host = ttk.Frame(left_split)
        ctrl_canvas = tk.Canvas(scroll_host, highlightthickness=0)
        ctrl_scroll = ttk.Scrollbar(scroll_host, orient="vertical", command=ctrl_canvas.yview)
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)
        ctrl_inner = ttk.Frame(ctrl_canvas, padding=0)
        _ctrl_win = ctrl_canvas.create_window((0, 0), window=ctrl_inner, anchor="nw")

        def _ctrl_inner_configure(_evt: Any = None) -> None:
            ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all"))

        def _ctrl_canvas_configure(evt: Any) -> None:
            try:
                ctrl_canvas.itemconfigure(_ctrl_win, width=int(evt.width))
            except tk.TclError:
                pass

        ctrl_inner.bind("<Configure>", _ctrl_inner_configure)
        ctrl_canvas.bind("<Configure>", _ctrl_canvas_configure)

        def _ctrl_wheel(evt: Any) -> str | None:
            if ctrl_canvas.winfo_height() <= 1:
                return None
            delta = getattr(evt, "delta", 0) or 0
            if delta == 0 and str(getattr(evt, "num", "")) == "4":
                delta = 120
            elif delta == 0 and str(getattr(evt, "num", "")) == "5":
                delta = -120
            ctrl_canvas.yview_scroll(int(-1 * (delta / 120)), "units")
            return "break"

        ctrl_canvas.bind("<Enter>", lambda _e: ctrl_canvas.focus_set())
        ctrl_canvas.bind("<MouseWheel>", _ctrl_wheel)
        ctrl_canvas.bind("<Button-4>", _ctrl_wheel)
        ctrl_canvas.bind("<Button-5>", _ctrl_wheel)
        ctrl_canvas.pack(side="left", fill="both", expand=True)
        ctrl_scroll.pack(side="right", fill="y")

        left_split.add(scroll_host, weight=2)
        img_pane = ttk.Frame(left_split)
        left_split.add(img_pane, weight=3)
        try:
            left_split.pane(img_pane, minsize=int(MIN_LEFT_IMAGE_PANE_HEIGHT))
        except tk.TclError:
            pass

        _left_ctrl = ctrl_inner

        input_panel = ttk.LabelFrame(_left_ctrl, text="Inputs")
        input_panel.pack(fill="x")
        self._add_path_selector(input_panel, "Left RGB", self.left_path_var, self._choose_left_image)
        self._add_path_selector(input_panel, "Right RGB", self.right_path_var, self._choose_right_image)
        self._add_path_selector(input_panel, "Intrinsics", self.intrinsics_path_var, self._choose_intrinsics)
        load_row = ttk.Frame(input_panel)
        load_row.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Button(load_row, text="Load Inputs", command=self._load_current_inputs).pack(side="left")
        ttk.Button(load_row, text="Clear Inputs", command=self._clear_loaded_inputs).pack(side="left", padx=(8, 0))

        params_panel = ttk.LabelFrame(_left_ctrl, text="Params")
        params_panel.pack(fill="x", pady=(8, 0))
        self._add_labeled_entry(params_panel, "S2M2 URL", self.s2m2_url_var)
        self._add_labeled_entry(params_panel, "SAM3 URL", self.sam3_url_var)
        self._add_labeled_entry(params_panel, "SAM3D URL", self.sam3d_url_var)
        self._add_labeled_entry(params_panel, "Timeout(s)", self.timeout_var)
        self._add_labeled_entry(params_panel, "Max Points", self.max_points_var, bind_return=self._render_viewer)

        action_panel = ttk.LabelFrame(_left_ctrl, text="Actions")
        action_panel.pack(fill="x", pady=(8, 0))
        row1 = ttk.Frame(action_panel)
        row1.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Button(row1, text="Run S2M2", command=self._start_s2m2).pack(side="left")
        ttk.Button(row1, text="Segment Scene", command=self._apply_scene_bbox).pack(side="left", padx=(8, 0))
        ttk.Button(row1, text="Segment Object", command=self._start_segment_object).pack(side="left", padx=(8, 0))

        row1b = ttk.Frame(action_panel)
        row1b.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Button(row1b, text="Edge Depth Filter", command=self._start_edge_depth_outlier_removal).pack(side="left")
        ttk.Button(row1b, text="Model Completion", command=self._start_sam3d).pack(side="left", padx=(8, 0))

        row2 = ttk.Frame(action_panel)
        row2.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(row2, text="Reset Scene", command=self._clear_scene_bbox).pack(side="left")
        ttk.Button(row2, text="Clear Object", command=self._clear_object_mask).pack(side="left", padx=(8, 0))
        ttk.Button(row2, text="Clear BBox", command=self._clear_bbox).pack(side="left", padx=(8, 0))
        ttk.Button(row2, text="Reset View", command=self._reset_view).pack(side="left", padx=(8, 0))

        info_panel = ttk.LabelFrame(_left_ctrl, text="Info")
        info_panel.pack(fill="x", pady=(8, 0))
        ttk.Label(info_panel, textvariable=self.info_var, justify="left").pack(fill="x", padx=8, pady=8)

        image_panel = ttk.LabelFrame(
            img_pane,
            text="Left Image (drag bbox)",
        )
        image_panel.pack(fill="both", expand=True)
        self.left_canvas = tk.Canvas(
            image_panel,
            width=MAX_IMAGE_DISPLAY_WIDTH,
            height=MAX_IMAGE_DISPLAY_HEIGHT,
            bg="black",
            highlightthickness=0,
        )
        self.left_canvas.pack(fill="both", expand=True)
        self.left_canvas.bind("<ButtonPress-1>", self._on_left_canvas_press)
        self.left_canvas.bind("<B1-Motion>", self._on_left_canvas_drag)
        self.left_canvas.bind("<ButtonRelease-1>", self._on_left_canvas_release)

        pc_panel = ttk.LabelFrame(right_panel, text="Point Cloud (LMB rotate · RMB move pivot · wheel zoom)")
        pc_panel.pack(fill="both", expand=True)
        self.viewer_canvas = tk.Canvas(pc_panel, bg="#0e141c", highlightthickness=0)
        self.viewer_canvas.pack(fill="both", expand=True)
        self.viewer_canvas.bind("<Configure>", lambda _evt: self._render_viewer())
        self.viewer_canvas.bind("<ButtonPress-1>", self._on_viewer_press)
        self.viewer_canvas.bind("<B1-Motion>", self._on_viewer_drag_rotate)
        self.viewer_canvas.bind("<ButtonRelease-1>", self._on_viewer_release_rotate)
        self.viewer_canvas.bind("<ButtonPress-3>", self._on_viewer_pivot_press)
        self.viewer_canvas.bind("<B3-Motion>", self._on_viewer_pivot_drag)
        self.viewer_canvas.bind("<ButtonRelease-3>", self._on_viewer_pivot_release)
        self.viewer_canvas.bind("<MouseWheel>", self._on_viewer_wheel)
        self.viewer_canvas.bind("<Button-4>", self._on_viewer_wheel)
        self.viewer_canvas.bind("<Button-5>", self._on_viewer_wheel)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", padx=8, pady=(0, 8))

    def _add_labeled_entry(
        self,
        parent: ttk.Widget,
        label: str,
        var: tk.StringVar,
        bind_return: Callable[[], None] | None = None,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(row, text=label, width=11).pack(side="left")
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
        if bind_return is not None:
            entry.bind("<Return>", lambda _evt: bind_return())

    def _add_path_selector(
        self,
        parent: ttk.Widget,
        label: str,
        var: tk.StringVar,
        command: Callable[[], None],
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(row, text=label, width=11).pack(side="left")
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Button(row, text="Browse", command=command).pack(side="left", padx=(6, 0))

    def _choose_left_image(self) -> None:
        if filedialog is None:
            return
        path = filedialog.askopenfilename(
            title="Select Left RGB Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")],
        )
        if path:
            self.left_path_var.set(str(path))

    def _choose_right_image(self) -> None:
        if filedialog is None:
            return
        path = filedialog.askopenfilename(
            title="Select Right RGB Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")],
        )
        if path:
            self.right_path_var.set(str(path))

    def _choose_intrinsics(self) -> None:
        if filedialog is None:
            return
        path = filedialog.askopenfilename(
            title="Select camera_intrinsics.txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if path:
            self.intrinsics_path_var.set(str(path))

    @staticmethod
    def _default_sample_id(left_path: Path, right_path: Path) -> str:
        left_stem = left_path.stem
        right_stem = right_path.stem
        suffixes = ("_origin_L", "_left", "_L")
        for suffix in suffixes:
            if left_stem.endswith(suffix):
                left_stem = left_stem[: -len(suffix)]
                break
        if left_stem:
            return left_stem
        return f"{left_path.stem}__{right_stem}"

    def _load_current_inputs(self) -> None:
        left_path_raw = self.left_path_var.get().strip()
        right_path_raw = self.right_path_var.get().strip()
        intrinsics_path_raw = self.intrinsics_path_var.get().strip()
        if not left_path_raw or not right_path_raw or not intrinsics_path_raw:
            self.status_var.set("Set left/right images and intrinsics first.")
            return
        left_path = Path(left_path_raw).expanduser()
        right_path = Path(right_path_raw).expanduser()
        intrinsics_path = Path(intrinsics_path_raw).expanduser()
        if not left_path.is_file():
            self.status_var.set(f"Left image not found: {left_path}")
            return
        if not right_path.is_file():
            self.status_var.set(f"Right image not found: {right_path}")
            return
        if not intrinsics_path.is_file():
            self.status_var.set(f"Intrinsics file not found: {intrinsics_path}")
            return
        try:
            topic_text = intrinsics_path.read_text(encoding="utf-8")
            k1, baseline = _parse_topic_inputs(topic_text)
            left_image = Image.open(left_path).convert("RGB")
            right_image = Image.open(right_path).convert("RGB")
        except Exception as exc:
            self.status_var.set(f"Failed to load inputs: {exc}")
            return
        sample_id = self._default_sample_id(left_path, right_path)
        self.camera = CameraBundle(k1=k1.astype(np.float32), baseline=float(baseline))
        self.current_cache_key = _cache_key_from_inputs(left_path, right_path, intrinsics_path)
        self.left_image_full = left_image
        self.right_image_full = right_image
        self.left_image_np = np.asarray(self.left_image_full, dtype=np.uint8)
        self.selected_sample_id = sample_id
        self.selected_bbox = None
        self.scene_bbox = None
        self.preview_bbox = None
        self.object_mask = None
        self._bbox_drag_start = None
        self.sam3d_points = None
        self.sam3d_colors = None
        self.sam3d_infer_seed = None
        self.sam3d_latency_ms = 0
        self.sam3d_pose_R = None
        self.sam3d_pose_t = None
        self.sam3d_pose_s = None
        self.sam3d_sent_pointmap = False
        self.sam3d_ply_simplify = False
        self.sam3d_response_kind = ""
        self.sam3d_infer_crop_roi = None
        self.viewer_state = ViewState()
        self.pointcloud = self.pointcloud_cache.get(self.current_cache_key or sample_id)
        if self.pointcloud is None:
            self._try_restore_s2m2_from_disk(sample_id, self.current_cache_key)
        self._render_left_image()
        self._render_viewer()
        self._update_info()
        if self.pointcloud is None:
            self.status_var.set(f"Loaded inputs for {sample_id}. Run S2M2 to generate the point cloud.")
        else:
            hint = "（磁盘缓存）" if self.pointcloud.from_cache else "（会话缓存）"
            self.status_var.set(
                f"Loaded inputs for {sample_id}. 点云已就绪{hint}: {self.pointcloud.points_cam.shape[0]} 点。"
            )

    def _clear_loaded_inputs(self) -> None:
        self.left_path_var.set("")
        self.right_path_var.set("")
        self.intrinsics_path_var.set("")
        self.camera = None
        self.current_cache_key = None
        self.selected_sample_id = None
        self.left_image_full = None
        self.right_image_full = None
        self.left_image_np = None
        self.pointcloud = None
        self.selected_bbox = None
        self.scene_bbox = None
        self.preview_bbox = None
        self.object_mask = None
        self.sam3d_points = None
        self.sam3d_colors = None
        self.sam3d_infer_seed = None
        self.sam3d_latency_ms = 0
        self.sam3d_pose_R = None
        self.sam3d_pose_t = None
        self.sam3d_pose_s = None
        self.sam3d_sent_pointmap = False
        self.sam3d_ply_simplify = False
        self.sam3d_response_kind = ""
        self.sam3d_infer_crop_roi = None
        self.viewer_state = ViewState()
        if self.left_canvas_image_id is not None:
            self.left_canvas.delete(self.left_canvas_image_id)
            self.left_canvas_image_id = None
        self.left_display_image = None
        self.left_display_photo = None
        self._redraw_left_bboxes()
        self._render_viewer()
        self._update_info()
        self.status_var.set("Inputs cleared.")

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
            return int(DEFAULT_MAX_POINTS)

    def _run_background(
        self,
        label: str,
        worker: Callable[[], Any],
        on_success: Callable[[Any], None],
        *,
        show_status: bool = True,
    ) -> None:
        if self.busy:
            self.status_var.set("A task is already running.")
            return

        self.busy = True
        if show_status:
            self.status_var.set(f"{label}...")
        LOGGER.info("task start: %s", label)

        def _target() -> None:
            try:
                result = worker()
            except Exception as exc:
                tb = traceback.format_exc()
                LOGGER.exception("task failed: %s", label)
                # Default args bind now: Py3.11+ clears `exc` after except; bare lambda would raise NameError.
                self.root.after(
                    0,
                    lambda lb=label, err=exc, trace=tb: self._finish_background_error(lb, err, trace),
                )
                return
            LOGGER.info("task success: %s", label)
            self.root.after(
                0,
                lambda cb=on_success, res=result: self._finish_background_success(cb, res),
            )

        threading.Thread(target=_target, daemon=True).start()

    def _finish_background_success(self, on_success: Callable[[Any], None], result: Any) -> None:
        self.busy = False
        on_success(result)

    def _finish_background_error(self, label: str, exc: Exception, tb: str) -> None:
        self.busy = False
        self.status_var.set(f"{label} failed: {exc}")
        LOGGER.error("task error shown: %s | %s", label, exc)
        if messagebox is not None:
            messagebox.showerror("Task failed", f"{exc}\n\n{tb}")

    def _make_s2m2_worker(self) -> Callable[[], PointCloudFrame] | None:
        """若当前输入不完整则返回 None。"""
        if (
            self.left_image_full is None
            or self.right_image_full is None
            or self.left_image_np is None
            or self.camera is None
            or self.current_cache_key is None
        ):
            return None
        sample_id = str(self.selected_sample_id or "")
        cache_key = str(self.current_cache_key)
        left = self.left_image_full.copy()
        right = self.right_image_full.copy()
        left_np = self.left_image_np.copy()
        k1 = self.camera.k1.copy()
        baseline = float(self.camera.baseline)
        api_url = self.s2m2_url_var.get().strip()
        timeout_s = self._parse_timeout_s()
        cache_path = _s2m2_cache_path(cache_key)

        def _worker() -> PointCloudFrame:
            if cache_path.is_file():
                cached = _load_s2m2_cache_file(cache_path)
                if cached is not None:
                    dm, lat_ms = cached
                    try:
                        LOGGER.info("s2m2 cache hit | sample=%s path=%s", sample_id, cache_path.name)
                        return _pointcloud_frame_from_dense_map(
                            sample_id, dm, left_np, lat_ms, from_cache=True
                        )
                    except ValueError as exc:
                        LOGGER.warning("s2m2 cache invalid, calling API: %s", exc)
            arr, latency_ms = _call_s2m2_api(left, right, k1, baseline, api_url, timeout_s)
            dense_map = _build_dense_from_cloud_or_depth(arr, left.size, k1)
            try:
                _save_s2m2_cache_file(cache_path, dense_map, latency_ms)
                LOGGER.info("s2m2 cache saved | sample=%s path=%s", sample_id, cache_path.name)
            except Exception as exc:
                LOGGER.warning("s2m2 cache save failed: %s", exc)
            return _pointcloud_frame_from_dense_map(
                sample_id, dense_map, left_np, int(latency_ms), from_cache=False
            )

        return _worker

    def _try_restore_s2m2_from_disk(self, sample_id: str, cache_key: str | None) -> None:
        if self.left_image_np is None or not cache_key:
            return
        path = _s2m2_cache_path(cache_key)
        if not path.is_file():
            return
        loaded = _load_s2m2_cache_file(path)
        if loaded is None:
            return
        dm, lat_ms = loaded
        try:
            frame = _pointcloud_frame_from_dense_map(
                sample_id, dm, self.left_image_np, lat_ms, from_cache=True
            )
        except ValueError as exc:
            LOGGER.warning("s2m2 disk restore skipped: %s", exc)
            return
        self.pointcloud_cache[str(cache_key)] = frame
        self.pointcloud = frame

    def _start_s2m2(self) -> None:
        w = self._make_s2m2_worker()
        if w is None:
            self.status_var.set("Load inputs first.")
            return
        cache_key = str(self.current_cache_key or "")
        label = "Loading S2M2 cache" if cache_key and _s2m2_cache_path(cache_key).is_file() else "Running S2M2"
        self._run_background(label, w, self._on_s2m2_ready)

    def _on_s2m2_ready(self, frame: PointCloudFrame) -> None:
        self.pointcloud = frame
        cache_key = str(self.current_cache_key or frame.sample_id)
        self.pointcloud_cache[cache_key] = frame
        self.scene_bbox = None
        self.object_mask = None
        self.sam3d_points = None
        self.sam3d_colors = None
        self.sam3d_infer_seed = None
        self.sam3d_latency_ms = 0
        self.sam3d_pose_R = None
        self.sam3d_pose_t = None
        self.sam3d_pose_s = None
        self.sam3d_sent_pointmap = False
        self.sam3d_ply_simplify = False
        self.sam3d_response_kind = ""
        self.sam3d_infer_crop_roi = None
        self._render_left_image()
        self._render_viewer()
        self._update_info()
        src = "cache" if frame.from_cache else "api"
        self.status_var.set(
            f"S2M2 ready ({src}) for {frame.sample_id}. valid_points={frame.points_cam.shape[0]}, "
            f"latency_s={_fmt_latency_s(frame.latency_ms)}"
        )
        LOGGER.info(
            "s2m2 ready | sample=%s points=%d latency_s=%s source=%s",
            frame.sample_id,
            int(frame.points_cam.shape[0]),
            _fmt_latency_s(frame.latency_ms),
            src,
        )

    def _bbox_pixel_keep_mask(
        self,
        pixels_yx: np.ndarray,
        bbox: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        pixels = np.asarray(pixels_yx, dtype=np.int32)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.shape[0] == 0 or bbox is None:
            return np.zeros((pixels.shape[0] if pixels.ndim == 2 else 0,), dtype=bool)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        ys = pixels[:, 0]
        xs = pixels[:, 1]
        return (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)

    def _visible_scene_keep_mask(self) -> np.ndarray:
        if self.pointcloud is None or self.pointcloud.points_cam.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        keep = np.ones((self.pointcloud.points_cam.shape[0],), dtype=bool)
        if self.scene_bbox is not None:
            keep &= self._bbox_pixel_keep_mask(self.pointcloud.pixels_yx, self.scene_bbox)
        return keep

    def _visible_object_keep_mask(self) -> np.ndarray:
        if self.pointcloud is None or self.object_mask is None or self.pointcloud.points_cam.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        keep = self.object_mask[self.pointcloud.pixels_yx[:, 0], self.pointcloud.pixels_yx[:, 1]]
        scene_keep = self._visible_scene_keep_mask()
        if scene_keep.shape[0] == keep.shape[0]:
            keep &= scene_keep
        return keep.astype(bool, copy=False)

    def _apply_scene_bbox(self) -> None:
        if self.pointcloud is None:
            self.status_var.set("Run S2M2 before segmenting the scene.")
            return
        if self.selected_bbox is None:
            self.status_var.set("Draw a bbox on the left image first.")
            return
        self.scene_bbox = tuple(int(v) for v in self.selected_bbox)
        self._render_left_image()
        self._render_viewer()
        self._update_info()
        visible_count = int(np.count_nonzero(self._visible_scene_keep_mask()))
        self.status_var.set(f"Scene segmented to bbox {self.scene_bbox}. visible_points={visible_count}")

    def _clear_scene_bbox(self) -> None:
        self.scene_bbox = None
        self._render_left_image()
        self._render_viewer()
        self._update_info()
        self.status_var.set("Scene filter cleared.")

    def _start_segment_object(self) -> None:
        if self.pointcloud is None or self.left_image_full is None:
            self.status_var.set("Run S2M2 before segmenting the object.")
            return
        if self.selected_bbox is None:
            self.status_var.set("Draw a bbox on the left image first.")
            return
        bbox = tuple(int(v) for v in self.selected_bbox)
        left = self.left_image_full.copy()
        sam3_url = self.sam3_url_var.get().strip()
        timeout_s = self._parse_timeout_s()

        def _worker() -> dict[str, Any]:
            res = _call_sam3_generic(
                image=left,
                sam3_url=sam3_url,
                timeout_s=timeout_s,
                box_prompts=[[int(v) for v in bbox]],
                box_labels=[True],
            )
            if not res.get("ok", False):
                raise RuntimeError(f"SAM3 failed: {res.get('error', 'unknown')}")
            return {
                "bbox": bbox,
                "mask": np.asarray(res["mask"], dtype=bool),
                "latency_ms": int(res.get("latency_ms", 0)),
                "mask_area_px": int(res.get("mask_area_px", 0)),
            }

        self._run_background("Running SAM3", _worker, self._on_segment_object_ready)

    def _on_segment_object_ready(self, result: dict[str, Any]) -> None:
        self.object_mask = np.asarray(result["mask"], dtype=bool)
        self._render_left_image()
        self._render_viewer()
        self._update_info()
        highlight_count = int(np.count_nonzero(self._visible_object_keep_mask()))
        self.status_var.set(
            f"Object mask ready. bbox={result['bbox']}, mask_px={result['mask_area_px']}, "
            f"latency_s={_fmt_latency_s(result['latency_ms'])}, highlighted_points={highlight_count}"
        )
        LOGGER.info(
            "object segmented | bbox=%s sam_s=%s mask_px=%d object_pts=%d",
            list(result["bbox"]),
            _fmt_latency_s(result["latency_ms"]),
            int(result["mask_area_px"]),
            int(highlight_count),
        )

    def _start_edge_depth_outlier_removal(self) -> None:
        if self.pointcloud is None or self.object_mask is None:
            self.status_var.set("Run Segment Object first.")
            return
        dense_copy = np.asarray(self.pointcloud.dense_map_cam, dtype=np.float32, copy=True)
        object_mask_copy = np.asarray(self.object_mask, dtype=bool, copy=True)
        h, w = int(object_mask_copy.shape[0]), int(object_mask_copy.shape[1])
        bbox_arg = self.selected_bbox
        if bbox_arg is not None:
            bbox_arg = _clip_bbox(tuple(int(v) for v in bbox_arg), w, h)

        def _worker() -> tuple[np.ndarray, dict[str, float | int]]:
            return _object_outlier_removal_edge_depth_jump(
                dense_copy,
                object_mask_copy,
                bbox_arg,
            )

        self._run_background(
            "Strip depth Canny edges",
            _worker,
            self._on_edge_depth_outlier_removal_ready,
            show_status=False,
        )

    def _on_edge_depth_outlier_removal_ready(self, payload: tuple[np.ndarray, dict[str, float | int]]) -> None:
        new_mask, info = payload
        self.object_mask = np.asarray(new_mask, dtype=bool, copy=False)
        self._render_left_image()
        self._render_viewer()
        self._update_info()
        highlight_count = int(np.count_nonzero(self._visible_object_keep_mask()))
        self.status_var.set("")
        LOGGER.info(
            "[Edge Depth Filter] removed=%s kept=%s canny_in_ROI=%s object_was=%s highlighted_points=%s Canny(%s,%s) time_s=%s bbox=%s",
            info.get("n_edge_removed", 0),
            info.get("n_kept", 0),
            info.get("n_canny_total", 0),
            info.get("n_input", 0),
            highlight_count,
            info.get("canny_low", 0),
            info.get("canny_high", 0),
            _fmt_latency_s(info.get("elapsed_ms", 0)),
            info.get("bbox", ()),
        )

    def _start_sam3d(self) -> None:
        if self.left_image_full is None:
            self.status_var.set("Load inputs first.")
            return
        left = self.left_image_full.copy()
        w, h = left.size
        sam3_url = self.sam3_url_var.get().strip()
        sam3d_url = self.sam3d_url_var.get().strip()
        if not sam3d_url:
            self.status_var.set(
                "Set SAM3D URL for model completion (FastAPI base, e.g. http://localhost:8000)."
            )
            return
        timeout_s = self._parse_timeout_s()
        infer_timeout = max(float(timeout_s), 300.0)

        object_mask_ref = self.object_mask
        selected_bbox_ref = self.selected_bbox
        pointcloud_ref = self.pointcloud

        def _worker() -> dict[str, Any]:
            mask: np.ndarray | None = None
            if object_mask_ref is not None:
                om = np.asarray(object_mask_ref, dtype=bool)
                if om.shape[0] == h and om.shape[1] == w and np.any(om):
                    mask = om
            if mask is None and selected_bbox_ref is not None:
                res = _call_sam3_generic(
                    image=left,
                    sam3_url=sam3_url,
                    timeout_s=timeout_s,
                    box_prompts=[[int(v) for v in selected_bbox_ref]],
                    box_labels=[True],
                )
                if not res.get("ok", False):
                    raise RuntimeError(f"SAM3 mask failed: {res.get('error', 'unknown')}")
                mask = np.asarray(res["mask"], dtype=bool)
            if mask is None or not np.any(mask):
                raise RuntimeError(
                    "Need a foreground mask: run Segment Object, or draw a bbox (SAM3 will build the mask)."
                )

            seed_u32 = secrets.randbits(32)
            infer_crop_roi: tuple[int, int, int, int] | None = None
            img_send = left
            mask_send = mask
            if SAM3D_UPLOAD_CROP_TO_BBOX:
                roi = _sam3d_resolve_infer_roi_xyxy(w, h, selected_bbox_ref, mask)
                if roi is not None:
                    x1, y1, x2, y2 = roi
                    infer_crop_roi = roi
                    img_send = left.crop((x1, y1, x2 + 1, y2 + 1))
                    mask_send = np.asarray(mask[y1 : y2 + 1, x1 : x2 + 1], dtype=bool, copy=True)

            pointmap_npy: bytes | None = None
            sent_pointmap = False
            if SAM3D_SEND_POINTMAP and pointcloud_ref is not None:
                dm = np.asarray(pointcloud_ref.dense_map_cam, dtype=np.float32)
                while dm.ndim > 3 and int(dm.shape[0]) == 1:
                    dm = np.squeeze(dm, axis=0)
                if dm.ndim == 3 and int(dm.shape[0]) == h and int(dm.shape[1]) == w and int(dm.shape[2]) == 3:
                    if infer_crop_roi is not None:
                        cx1, cy1, cx2, cy2 = infer_crop_roi
                        dm = dm[cy1 : cy2 + 1, cx1 : cx2 + 1, :].copy()
                    pointmap_npy = _dense_map_to_pointmap_npy_bytes(dm)
                    sent_pointmap = True

            _sam3d_preflight_health(sam3d_url, timeout_s=min(15.0, infer_timeout))
            api_res = _call_sam3d_infer_npz(
                image=img_send,
                mask=mask_send,
                base_url=sam3d_url,
                timeout_s=infer_timeout,
                seed=seed_u32,
                pointmap_npy_bytes=pointmap_npy,
            )
            if not api_res.get("ok", False):
                raise RuntimeError(f"Model completion failed: {api_res.get('error', 'unknown')}")
            xyz_t, rgb, R_p, t_p, s_p, pose_meta = _infer_npz_ply_bytes_to_gaussians(api_res["ply_bytes"])
            return {
                "xyz": xyz_t.astype(np.float32, copy=False),
                "rgb": rgb.astype(np.float32, copy=False),
                "latency_ms": int(api_res.get("latency_ms", 0)),
                "infer_seed": int(seed_u32),
                "pose_R": R_p,
                "pose_t": t_p,
                "pose_s": s_p,
                "pose_meta": pose_meta,
                "sent_pointmap": sent_pointmap,
                "ply_simplify": SAM3D_INFER_PLY_SIMPLIFY,
                "response_kind": "infer_npz",
                "infer_crop_roi": infer_crop_roi,
            }

        self._run_background("Model completion", _worker, self._on_sam3d_ready)

    def _on_sam3d_ready(self, result: dict[str, Any]) -> None:
        self.sam3d_points = np.asarray(result["xyz"], dtype=np.float32, copy=False)
        self.sam3d_colors = np.asarray(result["rgb"], dtype=np.float32, copy=False)
        self.sam3d_infer_seed = int(result["infer_seed"]) if "infer_seed" in result else None
        self.sam3d_latency_ms = int(result.get("latency_ms", 0))
        self.sam3d_pose_R = np.asarray(result["pose_R"], dtype=np.float32) if result.get("pose_R") is not None else None
        self.sam3d_pose_t = np.asarray(result["pose_t"], dtype=np.float32) if result.get("pose_t") is not None else None
        ps = result.get("pose_s")
        self.sam3d_pose_s = np.asarray(ps, dtype=np.float32).ravel() if ps is not None else None
        self.sam3d_sent_pointmap = bool(result.get("sent_pointmap", False))
        self.sam3d_ply_simplify = bool(result.get("ply_simplify", False))
        self.sam3d_response_kind = str(result.get("response_kind", "") or "")
        ic = result.get("infer_crop_roi")
        self.sam3d_infer_crop_roi = tuple(int(v) for v in ic) if ic is not None else None
        self._render_viewer()
        self._update_info()
        n = int(self.sam3d_points.shape[0]) if self.sam3d_points is not None else 0
        seed_part = f", seed={self.sam3d_infer_seed}" if self.sam3d_infer_seed is not None else ""
        pm_part = " +pointmap" if self.sam3d_sent_pointmap else " (MoGe depth)"
        simp_part = ", ply_simplify" if self.sam3d_ply_simplify else ""
        rk = self.sam3d_response_kind or "?"
        roi_part = ""
        if self.sam3d_infer_crop_roi is not None:
            a, b, c, d = self.sam3d_infer_crop_roi
            roi_part = f", ROI=({a},{b})-({c},{d}) {c - a + 1}×{d - b + 1}"
        self.status_var.set(
            f"Model completion [{rk}]{pm_part}{simp_part}: {n} pts, latency_s={_fmt_latency_s(self.sam3d_latency_ms)}{seed_part}{roi_part}"
        )

    def _sam3d_pose_info_lines(self) -> str:
        if self.sam3d_pose_t is None:
            return "sam3d_pose: (not loaded)"
        t = self.sam3d_pose_t
        s_txt = (
            np.array2string(np.asarray(self.sam3d_pose_s, dtype=np.float64), precision=4, suppress_small=True)
            if self.sam3d_pose_s is not None
            else "-"
        )
        r_txt = (
            np.array2string(np.asarray(self.sam3d_pose_R, dtype=np.float64), precision=3, suppress_small=True)
            if self.sam3d_pose_R is not None
            else "-"
        )
        t_txt = np.array2string(np.asarray(t, dtype=np.float64), precision=4, suppress_small=True)
        return f"sam3d_pose translation={t_txt} scale={s_txt}\nR={r_txt}"

    def _clear_object_mask(self) -> None:
        self.object_mask = None
        self.sam3d_points = None
        self.sam3d_colors = None
        self.sam3d_infer_seed = None
        self.sam3d_latency_ms = 0
        self.sam3d_pose_R = None
        self.sam3d_pose_t = None
        self.sam3d_pose_s = None
        self.sam3d_sent_pointmap = False
        self.sam3d_ply_simplify = False
        self.sam3d_response_kind = ""
        self.sam3d_infer_crop_roi = None
        self._render_left_image()
        self._render_viewer()
        self._update_info()
        self.status_var.set("Object mask and model completion cleared.")

    def _clear_bbox(self) -> None:
        self.selected_bbox = None
        self.preview_bbox = None
        self._bbox_drag_start = None
        self._render_left_image()
        self._update_info()
        self.status_var.set("BBox cleared.")

    def _reset_view(self) -> None:
        self.viewer_state = ViewState()
        self._render_viewer()
        self.status_var.set("Viewer pose and pivot reset.")

    def _render_left_image(self) -> None:
        if self.left_image_full is None:
            return
        image = self.left_image_full.copy()
        if self.object_mask is not None:
            image = _overlay_mask(image, self.object_mask, (96, 232, 120), 95)
        display = image.copy()
        if display.width > MAX_IMAGE_DISPLAY_WIDTH or display.height > MAX_IMAGE_DISPLAY_HEIGHT:
            display.thumbnail((MAX_IMAGE_DISPLAY_WIDTH, MAX_IMAGE_DISPLAY_HEIGHT), _RESAMPLING.LANCZOS)
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
        for item_id in (
            self.left_scene_bbox_item_id,
            self.left_bbox_item_id,
            self.left_preview_item_id,
        ):
            if item_id is not None:
                self.left_canvas.delete(item_id)
        self.left_scene_bbox_item_id = None
        self.left_bbox_item_id = None
        self.left_preview_item_id = None

        if self.scene_bbox is not None:
            x1, y1, x2, y2 = self._to_display_bbox(self.scene_bbox)
            self.left_scene_bbox_item_id = self.left_canvas.create_rectangle(
                x1, y1, x2, y2, outline="#4cc9f0", width=2, dash=(6, 3)
            )
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

    def _on_left_canvas_press(self, event: Any) -> None:
        if self.left_image_full is None or self.busy:
            return
        self._bbox_drag_start = self._display_to_image_xy(event.x, event.y)
        self.preview_bbox = None

    def _on_left_canvas_drag(self, event: Any) -> None:
        if self._bbox_drag_start is None or self.left_image_full is None:
            return
        x0, y0 = self._bbox_drag_start
        x1, y1 = self._display_to_image_xy(event.x, event.y)
        self.preview_bbox = _clip_bbox((x0, y0, x1, y1), self.left_image_full.width, self.left_image_full.height)
        self._redraw_left_bboxes()

    def _on_left_canvas_release(self, event: Any) -> None:
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
            self._update_info()
            self.status_var.set("BBox too small.")
            return
        self.selected_bbox = bbox
        self._redraw_left_bboxes()
        self._update_info()
        self.status_var.set(f"BBox set to {bbox}.")

    def _on_viewer_press(self, event: Any) -> None:
        self.viewer_drag_last = (int(event.x), int(event.y))

    def _on_viewer_drag_rotate(self, event: Any) -> None:
        if self.viewer_drag_last is None:
            return
        last_x, last_y = self.viewer_drag_last
        dx = float(event.x - last_x)
        dy = float(event.y - last_y)
        self.viewer_drag_last = (int(event.x), int(event.y))
        self.viewer_state.yaw_deg = float(self.viewer_state.yaw_deg - 0.35 * dx)
        self.viewer_state.pitch_deg = float(np.clip(self.viewer_state.pitch_deg + 0.35 * dy, -89.0, 89.0))
        self._render_viewer()

    def _on_viewer_release_rotate(self, _event: Any) -> None:
        self.viewer_drag_last = None

    def _on_viewer_pivot_press(self, event: Any) -> None:
        self.viewer_pivot_drag_last = (int(event.x), int(event.y))

    def _on_viewer_pivot_drag(self, event: Any) -> None:
        if self.viewer_pivot_drag_last is None or self._viewer_last_view_rot is None:
            return
        last_x, last_y = self.viewer_pivot_drag_last
        dx = float(event.x - last_x)
        dy = float(event.y - last_y)
        self.viewer_pivot_drag_last = (int(event.x), int(event.y))
        R = self._viewer_last_view_rot
        s = max(float(self._viewer_last_scale), 1e-6)
        # view = centered @ R.T，屏幕右/下 平移旋转中心：沿 R 的第 0/2 行方向修正 pivot（相对 median）
        delta = (-R[0, :] * dx + R[2, :] * dy) / s
        self.viewer_state.pivot_ox = float(self.viewer_state.pivot_ox + float(delta[0]))
        self.viewer_state.pivot_oy = float(self.viewer_state.pivot_oy + float(delta[1]))
        self.viewer_state.pivot_oz = float(self.viewer_state.pivot_oz + float(delta[2]))
        self._render_viewer()

    def _on_viewer_pivot_release(self, _event: Any) -> None:
        self.viewer_pivot_drag_last = None

    def _on_viewer_wheel(self, event: Any) -> None:
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

    def _build_placeholder_view(self, width: int, height: int, message: str) -> Image.Image:
        img = np.zeros((max(32, height), max(32, width), 3), dtype=np.uint8)
        img[..., 0] = int(DEFAULT_RENDER_BG_RGB[2])
        img[..., 1] = int(DEFAULT_RENDER_BG_RGB[1])
        img[..., 2] = int(DEFAULT_RENDER_BG_RGB[0])
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

    def _build_scene_render(self, width: int, height: int) -> Image.Image:
        canvas_w = max(64, int(width))
        canvas_h = max(64, int(height))

        sam3d_pts = _empty_points()
        sam3d_cols = _empty_colors()
        if self.sam3d_points is not None and self.sam3d_colors is not None:
            if self.sam3d_points.shape[0] > 0 and self.sam3d_colors.shape[0] == self.sam3d_points.shape[0]:
                sam3d_pts = self.sam3d_points.astype(np.float32, copy=False)
                sam3d_cols = self.sam3d_colors.astype(np.float32, copy=False)

        # 曾对 SAM3D 做 ROI 裁剪上传时：右侧只展示该次 model completion 的高斯（不叠整幅 S2M2 场景 / 物体高亮）
        roi_mc_only = self.sam3d_infer_crop_roi is not None
        if roi_mc_only:
            if sam3d_pts.shape[0] == 0:
                return self._build_placeholder_view(
                    canvas_w, canvas_h, "ROI Model Completion: no Gaussian points."
                )
            points = _empty_points()
            colors = _empty_colors()
            object_points = _empty_points()
        elif self.pointcloud is not None and self.pointcloud.points_cam.shape[0] > 0:
            scene_keep = self._visible_scene_keep_mask()
            points = self.pointcloud.points_cam[scene_keep].astype(np.float32, copy=False)
            colors = self.pointcloud.colors[scene_keep].astype(np.float32, copy=False)
            object_keep = self._visible_object_keep_mask()
            object_points = self.pointcloud.points_cam[object_keep].astype(np.float32, copy=False)
            if points.shape[0] == 0:
                if sam3d_pts.shape[0] == 0:
                    return self._build_placeholder_view(
                        canvas_w, canvas_h, "No points inside the current bbox."
                    )
                points = _empty_points()
                colors = _empty_colors()
                object_points = _empty_points()
        else:
            points = _empty_points()
            colors = _empty_colors()
            object_points = _empty_points()
            if sam3d_pts.shape[0] == 0:
                return self._build_placeholder_view(
                    canvas_w, canvas_h, "Run S2M2 or Model Completion to show a point cloud."
                )

        max_points = self._parse_max_points()
        bg_idx = _subsample_indices(points.shape[0], max_points, seed=3)
        points = points[bg_idx]
        colors = colors[bg_idx]

        max_object_points = max(5000, min(30000, max_points if max_points > 0 else 30000))
        if object_points.shape[0] > 0:
            obj_idx = _subsample_indices(object_points.shape[0], max_object_points, seed=7)
            object_points = object_points[obj_idx]

        max_sam3d = max(8000, min(100000, max_points * 4 if max_points > 0 else 50000))
        if sam3d_pts.shape[0] > max_sam3d:
            s_idx = _subsample_indices(sam3d_pts.shape[0], max_sam3d, seed=19)
            sam3d_pts = sam3d_pts[s_idx]
            sam3d_cols = sam3d_cols[s_idx]

        extents_list = [points]
        if sam3d_pts.shape[0] > 0:
            extents_list.append(sam3d_pts)
        if object_points.shape[0] > 0:
            extents_list.append(object_points)
        world_all = np.vstack(extents_list).astype(np.float32, copy=False)
        median = np.median(world_all, axis=0).astype(np.float64, copy=False)

        base_view_rot = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
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
        vs = self.viewer_state
        target = median + np.array([vs.pivot_ox, vs.pivot_oy, vs.pivot_oz], dtype=np.float64)

        def _project(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            centered = np.asarray(pts, dtype=np.float64) - target.reshape((1, 3))
            view = centered @ view_rot.T
            sx = view[:, 0]
            sy = -view[:, 2]
            depth = view[:, 1]
            return sx, sy, depth

        proj_all_x, proj_all_y, _ = _project(world_all)
        span_x = max(float(np.ptp(proj_all_x)), 1e-3)
        span_y = max(float(np.ptp(proj_all_y)), 1e-3)
        fit_scale = 0.90 * min((canvas_w - 32) / span_x, (canvas_h - 32) / span_y)
        scale = float(max(1e-3, fit_scale * float(self.viewer_state.zoom)))
        self._viewer_last_view_rot = view_rot.astype(np.float64, copy=True)
        self._viewer_last_scale = scale

        img_bgr = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        img_bgr[..., 0] = int(DEFAULT_RENDER_BG_RGB[2])
        img_bgr[..., 1] = int(DEFAULT_RENDER_BG_RGB[1])
        img_bgr[..., 2] = int(DEFAULT_RENDER_BG_RGB[0])

        def _draw_point_cloud(pts: np.ndarray, cols_rgb: np.ndarray, radius: int = 1) -> None:
            if pts.shape[0] == 0:
                return
            sx, sy, depth = _project(pts)
            x_pix = np.rint(canvas_w * 0.5 + sx * scale).astype(np.int32)
            y_pix = np.rint(canvas_h * 0.5 + sy * scale).astype(np.int32)
            valid = (x_pix >= 0) & (x_pix < canvas_w) & (y_pix >= 0) & (y_pix < canvas_h)
            if not np.any(valid):
                return
            x_pix = x_pix[valid]
            y_pix = y_pix[valid]
            depth = depth[valid]
            cols_bgr = np.clip(cols_rgb[valid][:, ::-1] * 255.0, 0.0, 255.0).astype(np.uint8)
            order = np.argsort(depth)
            x_pix = x_pix[order]
            y_pix = y_pix[order]
            cols_bgr = cols_bgr[order]
            if radius <= 1:
                img_bgr[y_pix, x_pix] = cols_bgr
                return
            for idx in range(x_pix.shape[0]):
                cv2.circle(
                    img_bgr,
                    (int(x_pix[idx]), int(y_pix[idx])),
                    int(radius),
                    tuple(int(v) for v in cols_bgr[idx]),
                    -1,
                    cv2.LINE_AA,
                )

        _draw_point_cloud(points, colors, radius=1)
        if sam3d_pts.shape[0] > 0:
            _draw_point_cloud(sam3d_pts, sam3d_cols, radius=2)
        if object_points.shape[0] > 0:
            object_colors = np.tile(np.array([[0.2, 0.95, 0.55]], dtype=np.float32), (object_points.shape[0], 1))
            _draw_point_cloud(object_points, object_colors, radius=2)

        # 相机坐标系原点处绘制 XYZ 轴（与 points_cam 一致：X 右、Y 下、Z 前；BGR：X 红、Y 绿、Z 蓝）
        axis_len = float(VIEWER_AXIS_LENGTH)
        o = np.zeros((1, 3), dtype=np.float64)
        ox = np.array([[axis_len, 0.0, 0.0]], dtype=np.float64)
        oy = np.array([[0.0, axis_len, 0.0]], dtype=np.float64)
        oz = np.array([[0.0, 0.0, axis_len]], dtype=np.float64)
        sx_o, sy_o, _ = _project(np.zeros((1, 3), dtype=np.float64))
        ox_pix = int(np.rint(canvas_w * 0.5 + float(sx_o[0]) * scale))
        oy_pix = int(np.rint(canvas_h * 0.5 + float(sy_o[0]) * scale))
        cv2.circle(img_bgr, (ox_pix, oy_pix), 4, (210, 210, 220), -1, cv2.LINE_AA)

        axis_specs: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int], str]] = [
            (o, ox, (0, 0, 255), "X"),
            (o, oy, (0, 255, 0), "Y"),
            (o, oz, (255, 0, 0), "Z"),
        ]
        for a, b, bgr, label in axis_specs:
            seg = np.vstack([a, b])
            sx_a, sy_a, _d = _project(seg)
            x_pix = np.rint(canvas_w * 0.5 + sx_a * scale).astype(np.int32)
            y_pix = np.rint(canvas_h * 0.5 + sy_a * scale).astype(np.int32)
            p0 = (int(x_pix[0]), int(y_pix[0]))
            p1 = (int(x_pix[1]), int(y_pix[1]))
            cv2.line(img_bgr, p0, p1, bgr, 2, cv2.LINE_AA)
            cv2.putText(
                img_bgr,
                label,
                (p1[0] + 4, p1[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bgr,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            img_bgr,
            f"yaw={self.viewer_state.yaw_deg:.1f}  pitch={self.viewer_state.pitch_deg:.1f}  zoom={self.viewer_state.zoom:.2f}",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 230, 240),
            1,
            cv2.LINE_AA,
        )
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    def _render_viewer(self) -> None:
        if not hasattr(self, "viewer_canvas"):
            return
        width = int(self.viewer_canvas.winfo_width())
        height = int(self.viewer_canvas.winfo_height())
        if width <= 1 or height <= 1:
            return
        display = self._build_scene_render(width, height)
        self.viewer_photo = ImageTk.PhotoImage(display)
        if self.viewer_image_id is None:
            self.viewer_image_id = self.viewer_canvas.create_image(0, 0, anchor="nw", image=self.viewer_photo)
        else:
            self.viewer_canvas.itemconfig(self.viewer_image_id, image=self.viewer_photo)

    def _update_info(self) -> None:
        sample = self.selected_sample_id or "-"
        if self.camera is None:
            intrinsics_text = "intrinsics: -"
            baseline_text = "baseline: -"
        else:
            intrinsics_text = (
                f"intrinsics: fx={self.camera.k1[0,0]:.3f}, fy={self.camera.k1[1,1]:.3f}, "
                f"cx={self.camera.k1[0,2]:.3f}, cy={self.camera.k1[1,2]:.3f}"
            )
            baseline_text = f"baseline: {self.camera.baseline:.5f} m"
        raw_points = int(self.pointcloud.points_cam.shape[0]) if self.pointcloud is not None else 0
        visible_points = int(np.count_nonzero(self._visible_scene_keep_mask())) if self.pointcloud is not None else 0
        object_points = int(np.count_nonzero(self._visible_object_keep_mask())) if self.pointcloud is not None else 0
        bbox_text = str(list(self.selected_bbox)) if self.selected_bbox is not None else "-"
        scene_bbox_text = str(list(self.scene_bbox)) if self.scene_bbox is not None else "-"
        mask_px = int(self.object_mask.sum()) if self.object_mask is not None else 0
        self.info_var.set(
            "\n".join(
                [
                    f"sample: {sample}",
                    intrinsics_text,
                    baseline_text,
                    f"raw_points: {raw_points}",
                    f"visible_scene_points: {visible_points}",
                    f"highlighted_object_points: {object_points}",
                    f"selected_bbox: {bbox_text}",
                    f"scene_bbox: {scene_bbox_text}",
                    f"object_mask_px: {mask_px}",
                    f"sam3d_infer_roi (xyxy): {list(self.sam3d_infer_crop_roi) if self.sam3d_infer_crop_roi is not None else '-'}",
                    f"model_completion: {int(self.sam3d_points.shape[0]) if self.sam3d_points is not None else 0} pts, "
                    f"kind={self.sam3d_response_kind or '-'}, "
                    f"sent_pointmap={'yes' if self.sam3d_sent_pointmap else 'no'}, "
                    f"ply_simplify={'yes' if self.sam3d_ply_simplify else 'no'}, "
                    f"latency_s={_fmt_latency_s(self.sam3d_latency_ms)}, seed={self.sam3d_infer_seed}",
                    self._sam3d_pose_info_lines(),
                ]
            )
        )

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = PointCloudViewerApp()
    app.run()


if __name__ == "__main__":
    main()
